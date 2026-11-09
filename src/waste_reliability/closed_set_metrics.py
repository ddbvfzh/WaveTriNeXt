"""Closed-set metrics and reports, independent of the five-class transfer study.

All machine-readable scores use the interval [0, 1]. Confusion matrix rows
are true classes and columns are predicted classes, in the supplied order.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


METRICS = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
HISTORY_COLUMNS = (
    "epoch", "train_loss", "train_accuracy", "train_macro_f1", "val_loss",
    "val_accuracy", "val_macro_f1", "learning_rate_backbone",
    "learning_rate_head", "seconds",
)


def _classes(classes: Sequence[str]) -> list[str]:
    result = list(classes)
    if not result or any(not isinstance(name, str) or not name for name in result):
        raise ValueError("classes must be a nonempty sequence of nonempty names")
    if len(set(result)) != len(result):
        raise ValueError("class names must be unique")
    return result


def _labels(values, name: str, count: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{name} must contain integer class indices")
    if np.any(labels < 0) or np.any(labels >= count):
        raise ValueError(f"{name} contains a class index outside [0, {count})")
    return labels.astype(np.int64, copy=False)


def classification_metrics(y_true, y_pred, classes) -> dict:
    """Compute full-dataset metrics; undefined class scores are explicitly zero.

    Macro averages include every supplied class, even if its support is zero.
    Empty input is rejected instead of producing a plausible-looking zero score.
    """
    names = _classes(classes)
    true = _labels(y_true, "y_true", len(names))
    predicted = _labels(y_pred, "y_pred", len(names))
    if true.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    count = len(names)
    matrix = np.bincount(true * count + predicted, minlength=count * count).reshape(count, count)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    correct = np.diag(matrix)
    precision = np.divide(correct, predicted_count, out=np.zeros(count), where=predicted_count != 0)
    recall = np.divide(correct, support, out=np.zeros(count), where=support != 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(count), where=denominator != 0)
    normalized = np.divide(matrix, support[:, None], out=np.zeros((count, count)), where=support[:, None] != 0)
    return {
        "classes": names,
        "n_samples": int(true.size),
        "accuracy": float(correct.sum() / true.size),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "per_class": [
            {"class_name": name, "precision": float(precision[index]),
             "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
            for index, name in enumerate(names)
        ],
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
    }


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict], columns: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    _write_text(path, buffer.getvalue())


def _read_csv(path: Path, required: Sequence[str]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing required columns in {path.name}: {list(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows


def _plot_confusion(matrix, classes: list[str], path: Path, normalized: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix)
    side = max(6.0, 0.65 * len(classes) + 2.0)
    fig, ax = plt.subplots(figsize=(side, side), layout="constrained")
    plotted = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    fig.colorbar(plotted, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel="Predicted class", ylabel="True class",
           title="Confusion matrix (row normalized)" if normalized else "Confusion matrix (counts)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = (1 if normalized else max(1, float(values.max()))) / 2
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = values[row, column]
            ax.text(column, row, f"{value:.2f}" if normalized else str(int(value)),
                    ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "black")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_history(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, key, title in zip(axes, ("loss", "accuracy", "macro_f1"), ("Loss", "Accuracy", "Macro-F1")):
        ax.plot(epochs, [row[f"train_{key}"] for row in history], label="Train", color="#0072B2")
        ax.plot(epochs, [row[f"val_{key}"] for row in history], label="Validation", color="#D55E00")
        ax.set(xlabel="Epoch", ylabel=title)
        ax.grid(alpha=0.2)
        if key != "loss":
            ax.set_ylim(0, 1)
        ax.legend()
    # Training uses smoothed CE, validation uses ordinary CE by design.
    axes[0].set_title("Train: smoothed CE; validation: CE", fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_run_report(run_dir, classes) -> dict:
    """Write test metrics, per-class tables, confusion matrices and learning curves.

    Expects predictions and history produced by a single run. This function never
    selects checkpoints or seeds and does not read image files.
    """
    run = Path(run_dir)
    names = _classes(classes)
    predictions = _read_csv(run / "test_predictions.csv", ("sample_id", "label", "prediction"))
    identifiers = [row["sample_id"] for row in predictions]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("test_predictions.csv must have unique nonempty sample_id values")
    try:
        true = [int(row["label"]) for row in predictions]
        predicted = [int(row["prediction"]) for row in predictions]
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction labels must be integer class indices") from exc
    metrics = classification_metrics(true, predicted, names)
    history_rows = _read_csv(run / "history.csv", HISTORY_COLUMNS)
    history = []
    for row in history_rows:
        try:
            epoch = int(row["epoch"])
            numeric = {key: float(row[key]) for key in HISTORY_COLUMNS if key != "epoch"}
        except (TypeError, ValueError) as exc:
            raise ValueError("History must contain numeric epochs and scores") from exc
        if epoch < 1 or (history and epoch <= history[-1]["epoch"]):
            raise ValueError("History epochs must be positive and strictly increasing")
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("History contains non-finite values")
        history.append({"epoch": epoch, **numeric})

    _write_text(run / "test_metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _write_csv(run / "class_report.csv", metrics["per_class"], ("class_name", "precision", "recall", "f1", "support"))
    for key, normalized in (("confusion_matrix", False), ("confusion_matrix_normalized", True)):
        rows = [{"true_class": name, **dict(zip(names, values))} for name, values in zip(names, metrics[key])]
        _write_csv(run / f"{key}.csv", rows, ["true_class", *names])
        _plot_confusion(metrics[key], names, run / f"{key}.png", normalized)
    _plot_history(history, run / "training_curves.png")
    return metrics


def _completion(run: Path) -> tuple[dict | None, dict | None, str, str]:
    completion_path = run / "complete.json"
    metrics_path = run / "test_metrics.json"
    if not completion_path.is_file():
        return None, None, "missing" if not run.exists() else "incomplete", "No complete.json"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if not isinstance(completion, dict):
            raise ValueError("complete.json must contain an object")
        if completion.get("status") != "completed":
            return completion, None, "incomplete", "Completion status is not completed"
        requested, finished = completion.get("requested_epochs"), completion.get("epochs_completed")
        if type(requested) is not int or requested < 1 or type(finished) is not int or finished != requested:
            return completion, None, "incomplete", "Completed epoch count does not match the requested budget"
        best_epoch = completion.get("best_epoch")
        if type(best_epoch) is not int or not 1 <= best_epoch <= finished:
            return completion, None, "invalid", "Invalid best_epoch"
        if not metrics_path.is_file():
            return completion, None, "invalid", "Missing test_metrics.json"
        metric_bytes = metrics_path.read_bytes()
        if hashlib.sha256(metric_bytes).hexdigest() != completion.get("test_metrics_sha256"):
            return completion, None, "hash_mismatch", "test_metrics.json hash differs from complete.json"
        metrics = json.loads(metric_bytes)
        if not isinstance(metrics, dict):
            raise ValueError("test_metrics.json must contain an object")
        for key in METRICS:
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                return completion, None, "invalid", f"Invalid score: {key}"
        return completion, metrics, "completed", ""
    except (OSError, ValueError, TypeError) as exc:
        return None, None, "invalid", str(exc)


def summarize_suite(root, datasets, seeds) -> dict:
    """Summarize verified completed runs separately by dataset (sample SD, ddof=1).

    Every requested seed is represented in per_seed.csv, including incomplete or
    invalid runs. A one-seed mean is allowed but its undefined SD is left blank.
    """
    output = Path(root)
    dataset_names = _classes(datasets)
    seed_values = list(seeds)
    if not seed_values or any(type(seed) is not int or seed < 0 for seed in seed_values) or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique nonnegative integers")
    if any(Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name for name in dataset_names):
        raise ValueError("datasets must be simple dataset identifiers")
    per_seed, summaries = [], []
    for dataset in dataset_names:
        eligible = []
        for seed in seed_values:
            run_name = f"convnext_tiny__{dataset}__seed{seed}"
            completion, metrics, status, reason = _completion(output / "runs" / run_name)
            row = {"dataset": dataset, "seed": seed, "status": status, "reason": reason,
                   "requested_epochs": (completion or {}).get("requested_epochs", ""),
                   "epochs_completed": (completion or {}).get("epochs_completed", ""),
                   "best_epoch": (completion or {}).get("best_epoch", ""),
                   **{key: metrics[key] if metrics else "" for key in METRICS}, "run_dir": f"runs/{run_name}"}
            per_seed.append(row)
            if metrics is not None:
                eligible.append(metrics)
        summary = {"dataset": dataset, "completed_seeds": len(eligible), "expected_seeds": len(seed_values),
                   "status": "completed" if len(eligible) == len(seed_values) else "partial"}
        for key in METRICS:
            values = [float(metrics[key]) for metrics in eligible]
            summary[f"{key}_mean"] = float(np.mean(values)) if values else None
            summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summaries.append(summary)

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_seed.csv", per_seed, ["dataset", "seed", "status", "reason", "requested_epochs",
               "epochs_completed", "best_epoch", *METRICS, "run_dir"])
    _write_csv(output / "summary.csv", summaries, ["dataset", "completed_seeds", "expected_seeds", "status",
               *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")]])

    def format_score(mean, std):
        if mean is None:
            return "—"
        if std is None:
            return f"{100 * mean:.2f} ± —"
        return f"{100 * mean:.2f} ± {100 * std:.2f}"

    lines = ["# ConvNeXt-T (A) closed-set baseline", "",
             "Each dataset is trained and evaluated independently. Accuracy and Macro-F1 are the primary metrics.",
             "CSV/JSON scores are fractions in [0, 1]. The table below uses percentages (mean ± sample SD, ddof=1).",
             "Only runs completing their requested epoch budget with verified test-metric hashes are included.",
             "Missing or invalid runs are excluded, never treated as zero. No best seed is selected.", "",
             "| Dataset | Completed seeds | Status | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in summaries:
        values = [format_score(row[f"{key}_mean"], row[f"{key}_std"]) for key in METRICS]
        lines.append("| " + " | ".join([row["dataset"].replace("|", "/"),
                     f"{row['completed_seeds']}/{row['expected_seeds']}", row["status"], *values]) + " |")
    lines.extend(["", "A single completed seed has no defined sample SD. Partial results are provisional.", "",
                  "| Dataset | Seed | Status | Reason |", "| --- | --- | --- | --- |"])
    for row in per_seed:
        lines.append("| " + " | ".join(str(row[key]).replace("|", "/").replace("\n", " ")
                     for key in ("dataset", "seed", "status", "reason")) + " |")
    lines.extend(["", "Per-seed scores: [per_seed.csv](per_seed.csv). Aggregate scores: [summary.csv](summary.csv).", ""])
    report_path = output / "report.md"
    _write_text(report_path, "\n".join(lines))
    return {"summary": summaries, "per_seed": per_seed, "report_path": str(report_path)}
