"""Release training transaction adapted from the frozen baseline train_run.

Model builder, optimizer and train/evaluation functions are selected explicitly.
The original epoch order, seeded augmentation, checkpointing, scheduler and
validation-based selection are preserved. Historical source remains unchanged.
"""
import gc, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from .closed_set_baseline import (
    save_json, validate_config, verify_complete, seed_everything, WarmupCosine,
    restore_rng, make_loader, synchronize, is_better, cpu_tree, capture_rng,
    atomic_torch_save, digest,
)
from .closed_set_metrics import write_run_report
from .portable import model_module, epoch_functions, portable_config

def train_run(root, alias, dataset, seed, manifest, weights, device, epochs=50, micro_batch=8, resume=False):
    import pandas as pd
    from .closed_set_data import CLASSES, load_manifest
    root = Path(root)
    implementation = model_module(alias)
    build_model, optimizer_for = implementation.build_model, implementation.optimizer_for
    train_epoch, evaluate = epoch_functions(alias)
    run = root / "runs" / f"{alias}__{dataset}__seed{seed}"
    config = portable_config(alias, dataset, seed, manifest, weights, device, epochs, micro_batch)
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
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != config["parameter_count"]:
        raise ValueError("Trainable parameter count differs from the paper configuration")
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

