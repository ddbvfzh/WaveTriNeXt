"""Independent, fixed-protocol ConvNeXt-T closed-set baseline experiments."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import traceback
import urllib.request

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models, transforms

from .closed_set_metrics import classification_metrics, write_run_report, summarize_suite

ROOT = Path(__file__).resolve().parents[2]
WEIGHT_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
WEIGHT_PREFIX = "983f1562"
VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def capture_rng():
    state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
              "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        result["xpu"] = torch.xpu.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if "xpu" in state:
        torch.xpu.set_rng_state_all([item.cpu() for item in state["xpu"]])


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    if name == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("Requested Intel XPU is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    return torch.device(name)


def synchronize(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def fetch_weights(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / WEIGHT_URL.rsplit("/", 1)[-1]
    receipt = target.with_suffix(".json")
    if not target.exists():
        print("Downloading official ConvNeXt-T ImageNet-1K weights", flush=True)
        temporary = target.with_suffix(".part")
        with urllib.request.urlopen(WEIGHT_URL, timeout=60) as response, temporary.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
        if not digest(temporary).startswith(WEIGHT_PREFIX):
            raise ValueError("Official pretrained weight SHA256 prefix does not match")
        temporary.replace(target)
    actual = digest(target)
    if not actual.startswith(WEIGHT_PREFIX):
        raise ValueError("Cached ConvNeXt weights fail publisher hash-prefix verification")
    metadata = {"url": WEIGHT_URL, "sha256": actual, "publisher_hash_prefix": WEIGHT_PREFIX,
                "weights_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"}
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8"))["sha256"] != actual:
        raise ValueError("Cached weight receipt does not match the file")
    save_json(receipt, metadata)
    return target


def build_model(num_classes, weight_path=None):
    model = models.convnext_tiny(weights=None, stochastic_depth_prob=0.1)
    if weight_path is not None:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    model.classifier[2] = nn.Linear(768, num_classes)
    nn.init.trunc_normal_(model.classifier[2].weight, std=0.02)
    nn.init.zeros_(model.classifier[2].bias)
    return model


def optimizer_for(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        head = name.startswith("classifier.2.")
        decay = parameter.ndim > 1 and not name.endswith("bias") and "layer_scale" not in name
        key = (head, decay)
        if key not in groups:
            lr = 1e-3 if head else 1e-4
            groups[key] = {"params": [], "param_names": [], "lr": lr, "initial_lr": lr,
                           "weight_decay": 0.05 if decay else 0.0, "role": "head" if head else "backbone"}
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999), eps=1e-8, foreach=False)


class WarmupCosine:
    """Learning rate for each optimizer update, including exact warmup/decay endpoints."""
    def __init__(self, optimizer, total_steps, warmup_steps, minimum_ratio=0.01):
        if total_steps < 1 or not 0 <= warmup_steps < total_steps:
            raise ValueError("Require total_steps > warmup_steps >= 0")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.minimum_ratio = float(minimum_ratio)
        self.step_index = 0
        self.apply()

    def multiplier(self, step):
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        remaining = self.total_steps - self.warmup_steps
        progress = min(max((step - self.warmup_steps) / max(remaining - 1, 1), 0.0), 1.0)
        return self.minimum_ratio + (1 - self.minimum_ratio) * (1 + math.cos(math.pi * progress)) / 2

    def apply(self):
        factor = self.multiplier(self.step_index)
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor

    def step(self):
        self.step_index += 1
        self.apply()

    def state_dict(self):
        return {"total_steps": self.total_steps, "warmup_steps": self.warmup_steps,
                "minimum_ratio": self.minimum_ratio, "step_index": self.step_index}

    def load_state_dict(self, state):
        for name in ("total_steps", "warmup_steps", "minimum_ratio"):
            if state[name] != getattr(self, name):
                raise ValueError("Scheduler configuration mismatch")
        self.step_index = int(state["step_index"])
        self.apply()


def stable_seed(seed, epoch, identity):
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{identity}".encode()).digest()[:8], "little") % (2**63 - 1)


class ClosedSetImages(Dataset):
    def __init__(self, rows, training=False, seed=0, epoch=0):
        self.rows = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
        self.training, self.seed, self.epoch = training, seed, epoch
        self.transform = (transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 4 / 3),
                                         interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            if training else models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if self.training:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(stable_seed(self.seed, self.epoch, row["sample_id"]))
                    tensor = self.transform(image)
            else:
                tensor = self.transform(image)
        return tensor, int(row["label"])


def make_loader(rows, micro_batch, training, seed, epoch=0):
    return DataLoader(ClosedSetImages(rows, training, seed, epoch), batch_size=micro_batch,
                      shuffle=training, drop_last=False, num_workers=0,
                      generator=torch.Generator().manual_seed(stable_seed(seed, epoch, "loader")))


def train_epoch(model, loader, optimizer, scheduler, device, classes, effective_batch=32, progress=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, count, window_samples = 0.0, 0, 0
    labels, predictions = [], []
    for batch_index, (images, target) in enumerate(loader):
        images, target = images.to(device), target.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, target, label_smoothing=0.1, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        size = len(target)
        window_samples += size
        total_loss += float(loss.detach().cpu())
        count += size
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(1).cpu().tolist())
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
    result = classification_metrics(labels, predictions, classes)
    result["loss"] = total_loss / count
    return result


@torch.inference_mode()
def evaluate(model, loader, device, classes):
    model.eval()
    labels, blocks = [], []
    loss_sum = 0.0
    for images, target in loader:
        logits = model(images.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        loss_sum += float(F.cross_entropy(logits, target.to(device), reduction="sum").cpu())
        blocks.append(logits.cpu().numpy())
        labels.extend(target.tolist())
    logits = np.concatenate(blocks)
    result = classification_metrics(labels, logits.argmax(1), classes)
    result["loss"] = loss_sum / len(labels)
    return result, np.asarray(labels), logits


def is_better(candidate, previous):
    return previous is None or (candidate["macro_f1"], -candidate["loss"]) > (previous["macro_f1"], -previous["loss"])


def code_hashes():
    folder = Path(__file__).parent
    return {name: digest(folder / name) for name in
            ("closed_set_baseline.py", "closed_set_data.py", "closed_set_metrics.py", "data.py")}


def snapshot_sources(root):
    destination = Path(root) / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, expected in code_hashes().items():
        source = Path(__file__).parent / name
        target = destination / name
        if target.exists() and digest(target) != expected:
            raise ValueError("Frozen source snapshot differs: " + name)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        records[name] = expected
    save_json(destination / "sha256.json", records)


def run_config(dataset, seed, manifest, weights, device, epochs, micro_batch):
    from .closed_set_data import CLASSES
    return {"version": VERSION, "model": "convnext_tiny", "dataset": dataset, "seed": seed,
            "classes": list(CLASSES[dataset]), "epochs": epochs, "micro_batch_size": micro_batch,
            "effective_batch_size": 32, "dtype": "float32", "device": str(device),
            "manifest_sha256": digest(manifest), "weights_sha256": digest(weights),
            "weights_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "code_sha256": code_hashes(),
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "packages": {name: package_version(name) for name in ("numpy", "pandas", "Pillow", "scipy")},
            "optimizer": "AdamW", "backbone_lr": 1e-4, "head_lr": 1e-3, "betas": [0.9, 0.999],
            "eps": 1e-8, "weight_decay": 0.05, "no_decay": "bias, normalization, LayerScale",
            "warmup_epochs": min(5, max(0, epochs - 1)), "scheduler": "update-wise linear warmup + cosine",
            "minimum_lr_ratio": 0.01, "label_smoothing": 0.1, "clip_grad_norm": 1.0,
            "drop_path": 0.1, "input_size": 224, "eval_resize": 236,
            "augmentation": {"random_resized_crop_scale": [0.8, 1.0], "ratio": [0.75, 4 / 3], "horizontal_flip": 0.5},
            "selection": "max validation macro_f1; ties min ordinary cross_entropy; ties earliest epoch",
            "determinism": "seeded all RNGs; deterministic_algorithms warn_only; no bitwise cross-device guarantee"}


def validate_config(existing, requested):
    if existing != requested:
        differences = [key for key in sorted(set(existing) | set(requested)) if existing.get(key) != requested.get(key)]
        raise ValueError("Run configuration mismatch: " + ", ".join(differences))


def verify_complete(run, config):
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    validate_config(complete["config"], config)
    if complete["status"] != "completed" or complete["epochs_completed"] != config["epochs"]:
        raise ValueError("Run completion metadata is inconsistent")
    for filename, expected in complete["artifacts_sha256"].items():
        if digest(run / filename) != expected:
            raise ValueError("Completed run artifact changed: " + filename)
    return complete


def train_run(root, dataset, seed, manifest, weights, device, epochs=50, micro_batch=8, resume=False):
    import pandas as pd
    from .closed_set_data import CLASSES, load_manifest
    root = Path(root)
    run = root / "runs" / f"convnext_tiny__{dataset}__seed{seed}"
    config = run_config(dataset, seed, manifest, weights, device, epochs, micro_batch)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        validate_config(json.loads((run / "config.json").read_text(encoding="utf-8")), config)
    elif any(run.iterdir()):
        raise ValueError("Run directory is nonempty without a configuration")
    save_json(run / "config.json", config)
    if (run / "complete.json").exists():
        verify_complete(run, config)
        print(f"Verified completed run: {dataset}/seed{seed}", flush=True)
        return run
    if (run / "last.pt").exists() and not resume:
        raise ValueError("An unfinished run exists; use --resume")
    rows = load_manifest(manifest, dataset)
    fit, val = rows[rows.split == "train"], rows[rows.split == "val"]
    classes = list(CLASSES[dataset])
    seed_everything(seed)
    model = build_model(len(classes), weights).to(device)
    optimizer = optimizer_for(model)
    updates = math.ceil(len(fit) / 32)
    scheduler = WarmupCosine(optimizer, updates * epochs, updates * config["warmup_epochs"])
    history, best_metrics, best_state, best_epoch, first_epoch = [], None, None, None, 0
    total_seconds = 0.0
    if (run / "last.pt").exists():
        saved = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        validate_config(saved["config"], config)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, first_epoch = saved["history"], saved["epochs_completed"]
        best_metrics, best_state, best_epoch = saved["best_metrics"], saved["best_model"], saved["best_epoch"]
        total_seconds = saved["total_seconds"]
        restore_rng(saved["rng"])
        del saved
        if len(history) != first_epoch or scheduler.step_index != first_epoch * updates:
            raise ValueError("Resume history or optimizer-update count is inconsistent")
        print(f"Resuming {dataset}/seed{seed} after epoch {first_epoch}", flush=True)
    val_loader = make_loader(val, micro_batch, False, seed)
    last_progress = 0.0
    for epoch in range(first_epoch, epochs):
        start = time.monotonic()
        loader = make_loader(fit, micro_batch, True, seed, epoch)
        start_lrs = {group["role"]: group["lr"] for group in optimizer.param_groups}

        def progress(count, total, loss, step):
            nonlocal last_progress
            now = time.monotonic()
            if now - last_progress >= 20 or count == total:
                status = {"status": "training", "dataset": dataset, "seed": seed, "epoch": epoch + 1,
                          "epochs": epochs, "images": count, "training_images": total, "train_loss_so_far": loss,
                          "optimizer_updates": step, "epoch_elapsed_seconds": now - start, "pid": os.getpid()}
                save_json(run / "progress.json", status)
                save_json(root / "progress.json", status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
                last_progress = now

        training = train_epoch(model, loader, optimizer, scheduler, device, classes, progress=progress)
        validation, _, _ = evaluate(model, val_loader, device, classes)
        synchronize(device)
        seconds = time.monotonic() - start
        total_seconds += seconds
        record = {"epoch": epoch + 1, "train_loss": training["loss"], "train_accuracy": training["accuracy"],
                  "train_macro_f1": training["macro_f1"], "val_loss": validation["loss"],
                  "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
                  "learning_rate_backbone": start_lrs["backbone"], "learning_rate_head": start_lrs["head"], "seconds": seconds}
        history.append(record)
        if is_better(validation, best_metrics):
            best_metrics, best_epoch, best_state = validation, epoch + 1, cpu_tree(model.state_dict())
        checkpoint = {"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                      "scheduler": scheduler.state_dict(), "rng": capture_rng(), "config": config,
                      "epochs_completed": epoch + 1, "history": history, "total_seconds": total_seconds,
                      "best_model": best_state, "best_metrics": best_metrics, "best_epoch": best_epoch}
        atomic_torch_save(run / "last.pt", checkpoint)
        del checkpoint
        atomic_torch_save(run / "best.pt", {"model": best_state, "config": config, "epoch": best_epoch,
                                           "validation_metrics": best_metrics})
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(f"EPOCH {dataset} seed={seed} {epoch+1}/{epochs} val_acc={validation['accuracy']:.5f} "
              f"val_macro_f1={validation['macro_f1']:.5f} best_epoch={best_epoch} seconds={seconds:.1f}", flush=True)
    if len(history) != epochs or best_state is None:
        raise ValueError("Training did not complete its requested epoch budget")
    # last.pt is the authoritative transaction, including its selected best state.
    atomic_torch_save(run / "best.pt", {"model": best_state, "config": config, "epoch": best_epoch,
                                       "validation_metrics": best_metrics})
    pd.DataFrame(history).to_csv(run / "history.csv", index=False)
    model.load_state_dict(best_state, strict=True)
    test = rows[rows.split == "test"].reset_index(drop=True)
    save_json(root / "progress.json", {"status": "testing", "dataset": dataset, "seed": seed, "best_epoch": best_epoch})
    _, labels, logits = evaluate(model, make_loader(test, micro_batch, False, seed), device, classes)
    predictions = test[["sample_id", "class_name", "label", "group_id"]].copy()
    if not np.array_equal(predictions.label.to_numpy(), labels):
        raise ValueError("Test output order changed")
    predictions["prediction"] = logits.argmax(1)
    for index in range(len(classes)):
        predictions[f"logit_{index}"] = logits[:, index]
    predictions.to_csv(run / "test_predictions.csv", index=False)
    write_run_report(run, classes)
    artifacts = {name: digest(run / name) for name in ("best.pt", "history.csv", "test_predictions.csv", "test_metrics.json")}
    complete = {"status": "completed", "dataset": dataset, "seed": seed, "epochs_completed": len(history),
                "requested_epochs": epochs, "best_epoch": best_epoch, "best_validation_metrics": best_metrics,
                "total_training_seconds": total_seconds, "classes": classes, "config": config,
                "training_samples": len(fit), "validation_samples": len(val), "test_samples": len(test),
                "artifacts_sha256": artifacts, "test_metrics_sha256": artifacts["test_metrics.json"]}
    save_json(run / "complete.json", complete)
    save_json(run / "progress.json", {"status": "completed", "dataset": dataset, "seed": seed, "epochs_completed": epochs})
    print(f"COMPLETED {dataset}/seed{seed}", flush=True)
    del model, optimizer, best_state
    gc.collect()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    return run


def technical_check(root, manifests, weights, device, micro_batch):
    from .closed_set_data import CLASSES, load_manifest
    identity = {"manifests": {name: digest(path) for name, path in manifests.items()}, "weights_sha256": digest(weights),
                "code_sha256": code_hashes(), "device": str(device), "micro_batch_size": micro_batch,
                "torch": str(torch.__version__), "torchvision": str(torchvision.__version__)}
    receipt = Path(root) / "technical_check.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous.get("identity") == identity and previous.get("passed"):
            return previous
    checks = []
    for dataset, manifest in manifests.items():
        seed_everything(14)
        classes = list(CLASSES[dataset])
        rows = load_manifest(manifest, dataset)
        model = build_model(len(classes), weights)
        reference = torch.load(weights, map_location="cpu", weights_only=True)
        copied = [name for name in model.state_dict() if not name.startswith("classifier.2.")]
        if not all(torch.equal(model.state_dict()[name], reference[name]) for name in copied):
            raise AssertionError("Pretrained backbone tensors were not preserved")
        del reference
        model.to(device)
        optimizer = optimizer_for(model)
        before = model.features[0][0].weight.detach().cpu().clone()
        selected = rows[rows.split == "train"].head(min(31, max(micro_batch + 1, 2)))
        scheduler = WarmupCosine(optimizer, total_steps=2, warmup_steps=0)
        synchronize(device)
        start = time.monotonic()
        result = train_epoch(model, make_loader(selected, micro_batch, True, 14), optimizer, scheduler, device, classes)
        synchronize(device)
        seconds = time.monotonic() - start
        if torch.equal(before, model.features[0][0].weight.detach().cpu()):
            raise AssertionError("Backbone parameters did not update")
        if scheduler.step_index != 1:
            raise AssertionError("Partial accumulation window did not execute exactly one update")
        metrics, _, logits = evaluate(model, make_loader(selected.head(2), micro_batch, False, 14), device, classes)
        if logits.shape != (2, len(classes)):
            raise AssertionError("Incorrect classifier shape")
        entry = {"dataset": dataset, "classes": len(classes), "pretrained_tensors_verified": len(copied),
                 "train_images_only": len(selected), "test_images_used": 0, "loss": result["loss"],
                 "finite_gradients": True, "backbone_updated": True, "partial_window_update": True,
                 "forward_backward_seconds": seconds, "evaluation_loss": metrics["loss"]}
        checks.append(entry)
        print("TECHNICAL_CHECK " + json.dumps(entry), flush=True)
        del model, optimizer
        gc.collect()
        if device.type == "xpu":
            torch.xpu.empty_cache()
    result = {"passed": True, "identity": identity, "checks": checks,
              "interpretation": "Engineering verification only; formal runs reinitialize models and RNGs."}
    save_json(receipt, result)
    return result


@contextlib.contextmanager
def suite_lock(root):
    path = Path(root) / "suite.lock"
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def main(argv=None):
    from .closed_set_data import CLASSES, prepare_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(CLASSES), default=["trashnet", "realwaste"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[14, 42, 70])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "xpu", "cuda", "cpu"], default="xpu")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/convnext_t_baseline_v1")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "data/cache/weights")
    parser.add_argument("--trashnet-root", type=Path, default=ROOT / "data/raw/trashnet")
    parser.add_argument("--realwaste-root", type=Path, default=ROOT / "data/raw/realwaste")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.micro_batch_size < 1 or 32 % args.micro_batch_size:
        parser.error("epochs must be positive; micro batch must divide effective batch 32")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds) or len(set(args.datasets)) != len(args.datasets):
        parser.error("Dataset/seed selections must be unique; seeds must be nonnegative")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with suite_lock(root):
        try:
            if args.report_only:
                summarize_suite(root, args.datasets, args.seeds)
                return
            roots = {"trashnet": args.trashnet_root, "realwaste": args.realwaste_root}
            manifests = {dataset: prepare_dataset(dataset, roots[dataset], root / "data" / dataset,
                         split_seed=args.split_seed, near_distance=4) for dataset in args.datasets}
            save_json(root / "data_summary.json", {dataset: json.loads((path.parent / "audit.json").read_text(encoding="utf-8"))
                                                  for dataset, path in manifests.items()})
            if args.prepare_only:
                print("Dataset preparation complete", flush=True)
                return
            device = resolve_device(args.device)
            weights = fetch_weights(args.weights_dir)
            first_dataset = args.datasets[0]
            protocol = run_config(first_dataset, args.seeds[0], manifests[first_dataset], weights,
                                  device, args.epochs, args.micro_batch_size)
            for key in ("dataset", "seed", "classes", "manifest_sha256"):
                protocol.pop(key)
            protocol.update({"split_seed": args.split_seed, "split_ratios": [0.7, 0.15, 0.15],
                             "near_duplicate_phash_distance": 4, "task": "independent closed-set classification"})
            protocol_path = root / "protocol.json"
            if protocol_path.exists():
                validate_config(json.loads(protocol_path.read_text(encoding="utf-8")), protocol)
            elif not args.check_only:
                save_json(protocol_path, protocol)
            save_json(root / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
                      "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "device": str(device),
                      "device_name": torch.xpu.get_device_name(0) if device.type == "xpu" else torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                      "precision": "float32", "determinism": "seeded, deterministic_algorithms warn_only"})
            technical_check(root, manifests, weights, device, args.micro_batch_size)
            if args.check_only:
                return
            snapshot_sources(root)
            for dataset in args.datasets:
                for seed in args.seeds:
                    train_run(root, dataset, seed, manifests[dataset], weights, device, args.epochs, args.micro_batch_size, args.resume)
                    summarize_suite(root, args.datasets, args.seeds)
            save_json(root / "progress.json", {"status": "completed", "datasets": args.datasets, "seeds": args.seeds,
                      "completed_runs": len(args.datasets) * len(args.seeds), "epochs_per_run": args.epochs})
        except BaseException as error:
            save_json(root / "failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
