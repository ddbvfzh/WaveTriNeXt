"""Fixed tri-region objective and sample-weighted FP32 training/evaluation.

All classification metrics use only logits_cls. Auxiliary outputs train the shared backbone.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .closed_set_metrics import classification_metrics


DIAGNOSTIC_KEYS = ("global_ce", "foreground_ce", "background_loss", "main_ce",
                   "fg_mass", "bg_mass", "am_mass", "pbg_true")


def compact_metrics(labels, predictions, classes):
    return {name: value for name, value in classification_metrics(labels, predictions, classes).items()
            if not name.startswith("confusion_matrix")}


def background_suppression_loss(logits, target, reduction="mean"):
    """Compute -log(1-p_y) without subtracting a rounded probability from one."""
    if logits.ndim != 2 or logits.shape[1] < 2 or target.ndim != 1 or len(logits) != len(target):
        raise ValueError("Expected background logits N,K (K>=2) and N class labels")
    other = logits.masked_fill(F.one_hot(target, num_classes=logits.shape[1]).bool(), float("-inf"))
    loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(other, dim=1)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be none, sum, or mean")


def loss_components(outputs, target, label_smoothing=0.1):
    """Return sample sums, preserving autograd for the complete objective."""
    for name in ("logits_cls", "logits_fg", "logits_bg", "region_maps"):
        if not torch.isfinite(outputs[name]).all():
            raise FloatingPointError("Non-finite tri-region output: " + name)
    global_ce = F.cross_entropy(outputs["logits_cls"], target,
                                label_smoothing=label_smoothing, reduction="sum")
    foreground_ce = F.cross_entropy(outputs["logits_fg"], target,
                                    label_smoothing=label_smoothing, reduction="sum")
    background_loss = background_suppression_loss(outputs["logits_bg"], target, reduction="sum")
    total_loss = global_ce + 0.1 * foreground_ce + 0.1 * background_loss
    with torch.no_grad():
        main_ce = F.cross_entropy(outputs["logits_cls"], target, reduction="sum")
        mass = outputs["region_maps"].mean(dim=(2, 3)).sum(dim=0)
        true_probability = outputs["logits_bg"].softmax(dim=1).gather(
            1, target[:, None]).sum()
    return {"total_loss": total_loss, "global_ce": global_ce,
            "foreground_ce": foreground_ce, "background_loss": background_loss,
            "main_ce": main_ce, "fg_mass": mass[0], "bg_mass": mass[1],
            "am_mass": mass[2], "pbg_true": true_probability}


def _add_diagnostics(totals, values):
    # One transfer for the eight scalar diagnostics per microbatch.
    scalars = torch.stack([values[name].detach() for name in DIAGNOSTIC_KEYS]).cpu().tolist()
    for name, value in zip(DIAGNOSTIC_KEYS, scalars):
        totals[name] += value


def train_epoch(model, loader, optimizer, scheduler, device, classes,
                effective_batch=32, progress=None):
    if type(effective_batch) is not int or effective_batch < 1:
        raise ValueError("effective_batch must be a positive integer")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count, window_samples = 0, 0
    total_loss = 0.0
    diagnostics = dict.fromkeys(DIAGNOSTIC_KEYS, 0.0)
    labels, predictions = [], []
    for batch_index, (images, target) in enumerate(loader):
        images, target = images.to(device), target.to(device)
        outputs = model(images, return_aux=True)
        values = loss_components(outputs, target, label_smoothing=0.1)
        loss = values["total_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite tri-region training loss")
        loss.backward()
        size = len(target)
        window_samples += size
        count += size
        total_loss += float(loss.detach().cpu())
        _add_diagnostics(diagnostics, values)
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(outputs["logits_cls"].detach().argmax(1).cpu().tolist())
        if window_samples >= effective_batch or batch_index + 1 == len(loader):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_samples)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            window_samples = 0
        if progress:
            progress(count, len(loader.dataset), total_loss / count, scheduler.step_index)
    if count == 0:
        raise ValueError("Cannot train on an empty loader")
    result = compact_metrics(labels, predictions, classes)
    result.update({name: value / count for name, value in diagnostics.items()})
    result["loss"] = total_loss / count
    return result


@torch.inference_mode()
def evaluate(model, loader, device, classes, return_branches=False):
    model.eval()
    labels, blocks = [], []
    branch_blocks = {"logits_cls": [], "logits_fg": [], "logits_bg": []} if return_branches else None
    count, total_objective = 0, 0.0
    diagnostics = dict.fromkeys(DIAGNOSTIC_KEYS, 0.0)
    for images, target in loader:
        outputs = model(images.to(device), return_aux=True)
        # Plain CE components are evaluation diagnostics. Model selection uses
        # ordinary main CE exclusively, never the auxiliary total objective.
        values = loss_components(outputs, target.to(device), label_smoothing=0.0)
        if not torch.isfinite(values["total_loss"]):
            raise FloatingPointError("Non-finite tri-region evaluation loss")
        total_objective += float(values["total_loss"].cpu())
        _add_diagnostics(diagnostics, values)
        blocks.append(outputs["logits_cls"].cpu().numpy())
        if branch_blocks is not None:
            for key, pieces in branch_blocks.items():
                pieces.append(outputs[key].cpu().numpy())
        labels.extend(target.cpu().tolist())
        count += len(target)
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    logits = np.concatenate(blocks, axis=0)
    result = compact_metrics(labels, logits.argmax(axis=1), classes)
    result.update({name: value / count for name, value in diagnostics.items()})
    result["loss"] = result["main_ce"]
    result["auxiliary_objective"] = total_objective / count
    result_tuple = result, np.asarray(labels, dtype=np.int64), logits
    if branch_blocks is not None:
        return (*result_tuple, {key: np.concatenate(pieces, axis=0)
                               for key, pieces in branch_blocks.items()})
    return result_tuple
