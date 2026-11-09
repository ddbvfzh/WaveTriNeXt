"""ConvNeXt-T with only the six early depthwise convolutions replaced by WTConv.

The numerical wavelet implementation is shared with the frozen A+a+b module.
Building A first and isolating WT initialization preserves both A's shared
initial tensors and its CPU RNG endpoint; no attention modules are constructed.
"""
from __future__ import annotations

import torch
from torch import nn

from .closed_set_baseline import build_model as _build_baseline
from .convnext_pmam_wtconv import (
    WT_CONFIG, WT_SOURCE_COMMIT, WTConv2d, ScaleModule, baseline_parameter_name,
)


MODEL_ID = "convnext_tiny_wtconv_s12"
ADDED_PARAMETERS = WT_ADDED_PARAMETERS = 346_464
PARAMETERS = {"trashnet": 28_171_206, "realwaste": 28_173_513}
WT_PATHS = tuple(f"features.{stage}.{block}.block.0" for stage in (1, 3) for block in range(3))


def build_model(num_classes, weight_path=None):
    """Load A, then replace six DW layers without consuming caller RNG."""
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    model = _build_baseline(num_classes, weight_path)
    with torch.random.fork_rng(devices=[]):
        for stage_index, channels in ((1, 96), (3, 192)):
            if len(model.features[stage_index]) != 3:
                raise ValueError("Expected three ConvNeXt blocks in each early stage")
            for block in model.features[stage_index]:
                original = block.block[0]
                if not (isinstance(original, nn.Conv2d) and original.in_channels == channels
                        and original.out_channels == channels and original.groups == channels
                        and original.kernel_size == (7, 7) and original.stride == (1, 1)
                        and original.bias is not None):
                    raise ValueError("Expected the original pretrained 7x7 depthwise convolution")
                replacement = WTConv2d(channels, channels, kernel_size=7, wt_levels=2)
                replacement.base_conv.load_state_dict(original.state_dict(), strict=True)
                block.block[0] = replacement
    return model.float()


def optimizer_for(model):
    """Partition backbone/wavelet/head and explicitly exempt all WT scales."""
    if hasattr(model, "a3") or hasattr(model, "a4"):
        raise ValueError("A+b must not contain attention modules")
    head_ids = {id(parameter) for parameter in model.classifier[2].parameters()}
    wavelet_ids, imported_base_ids = set(), set()
    wt_count = 0
    for _name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, WTConv2d):
            wt_count += 1
            base_ids = {id(parameter) for parameter in module.base_conv.parameters()}
            new_ids = {id(parameter) for parameter in module.parameters()} - base_ids
            if (wavelet_ids | imported_base_ids) & (new_ids | base_ids):
                raise ValueError("WTConv modules must not share parameters")
            wavelet_ids.update(new_ids)
            imported_base_ids.update(base_ids)
    if wt_count != 6:
        raise ValueError("A+b requires exactly six WTConv modules")
    if head_ids & (wavelet_ids | imported_base_ids):
        raise ValueError("Classifier and wavelet parameters must be independent")
    identities = [id(parameter) for _, parameter in model.named_parameters(remove_duplicate=False)]
    if len(identities) != len(set(identities)):
        raise ValueError("A+b parameters must not share identities")
    no_decay_ids = set()
    for module in model.modules():
        norm = isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm))
        for name, parameter in module.named_parameters(recurse=False):
            if norm or isinstance(module, ScaleModule) or name in ("bias", "layer_scale") or parameter.ndim <= 1:
                no_decay_ids.add(id(parameter))
    rates = {"backbone": 1e-4, "wavelet": 1e-3, "head": 1e-3}
    groups, assigned = {}, set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        assigned.add(identity)
        role = "head" if identity in head_ids else "wavelet" if identity in wavelet_ids else "backbone"
        decay = identity not in no_decay_ids
        key = (role, decay)
        if key not in groups:
            groups[key] = {"params": [], "param_names": [], "lr": rates[role],
                           "initial_lr": rates[role], "weight_decay": 0.05 if decay else 0.0,
                           "role": role}
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
    if assigned != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise ValueError("Optimizer groups do not cover exactly all trainable parameters")
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999), eps=1e-8, foreach=False)
