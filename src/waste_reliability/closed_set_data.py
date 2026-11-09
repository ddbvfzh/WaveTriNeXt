"""Independent six-/nine-class manifests for closed-set waste classification.

This module deliberately does not change the historical five-class protocol.
Preparation reads source images but never changes them; loading a manifest does
not decode images. Near-duplicate components are indivisible across partitions.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _BKTree, _image_hashes, _inventory, _normalise, _sha


CLASSES = {
    "trashnet": ("cardboard", "glass", "metal", "paper", "plastic", "trash"),
    "realwaste": ("cardboard", "food organics", "glass", "metal",
                  "miscellaneous trash", "paper", "plastic", "textile trash", "vegetation"),
}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
MANIFEST_COLUMNS = ("sample_id", "path", "dataset", "class_name", "label", "group_id",
                    "sha256", "pixel_sha256", "phash", "split")
EXCLUDED_COLUMNS = ("dataset", "path", "class_name", "sample_id", "sha256", "pixel_sha256", "reason")
DUPLICATE_COLUMNS = ("dataset", "sample_id_a", "sample_id_b", "path_a", "path_b",
                     "class_name_a", "class_name_b", "match_type", "hamming_distance",
                     "cross_class", "excluded_a", "excluded_b")
COUNT_COLUMNS = ("dataset", "class_name", "label", "split", "count", "group_count")
_OUTPUTS = ("manifest.csv", "duplicates.csv", "excluded.csv", "class_counts.csv")
_VERSION = 1


def _json_digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _check_dataset(dataset):
    if dataset not in CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset}; expected one of {tuple(CLASSES)}")


def _class_name(path, root, dataset):
    for part in reversed(path.relative_to(root).parts[:-1]):
        name = _normalise(part)
        if dataset == "realwaste" and name == "misc trash":
            name = "miscellaneous trash"
        if name in CLASSES[dataset]:
            return name
    return None


def _input_inventory(dataset, root):
    records, metadata = _inventory({dataset: root})
    # The historical helper excludes archive metadata before hashing it. Include
    # its bytes here so additions/removals/changes to excluded inputs are locked.
    ignored = []
    for item in metadata:
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Archive metadata escapes dataset directory: {path}")
        ignored.append({"dataset": dataset, "path": str(path),
                        "relative_path": path.relative_to(root).as_posix(), "sha256": _sha(path)})
    identity = [{"relative_path": row["relative_path"], "sha256": row["sha256"], "metadata": False}
                for row in records]
    identity += [{"relative_path": row["relative_path"], "sha256": row["sha256"], "metadata": True}
                 for row in ignored]
    identity.sort(key=lambda item: item["relative_path"])
    return records, ignored, _json_digest(identity)


def _verify_outputs(directory, audit):
    hashes = audit.get("output_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_OUTPUTS):
        raise ValueError("Locked audit lacks the expected output hashes")
    for name in _OUTPUTS:
        path = directory / name
        if not path.is_file() or _sha(path) != hashes[name]:
            raise ValueError(f"Locked output changed or is missing: {name}")
    expected = audit.get("audit_content_sha256")
    body = {key: value for key, value in audit.items() if key != "audit_content_sha256"}
    if expected != _json_digest(body):
        raise ValueError("Locked audit content has changed")


def _validate_frame(rows, dataset):
    missing = set(MANIFEST_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("Closed-set manifest is empty")
    for name in set(MANIFEST_COLUMNS) - {"label"}:
        if rows[name].isna().any() or rows[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Manifest column {name} contains empty values")
    if rows.sample_id.duplicated().any() or rows.path.duplicated().any():
        raise ValueError("Duplicate sample IDs or source paths")
    if set(rows.dataset) != {dataset}:
        raise ValueError("Manifest must contain only the requested dataset")
    if not rows.split.isin(SPLITS).all():
        raise ValueError("Closed-set splits must be train, val, and test")
    if (rows.groupby("group_id").split.nunique() > 1).any():
        raise ValueError("Duplicate groups cross partitions")
    for name in ("sha256", "pixel_sha256"):
        if not rows[name].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Invalid {name} values")
    if not rows.phash.astype(str).str.fullmatch(r"[0-9a-f]{16}").all():
        raise ValueError("Invalid phash values")
    labels = pd.to_numeric(rows.label, errors="raise")
    if labels.isna().any() or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Labels must be finite integers")
    expected = rows.class_name.map({name: index for index, name in enumerate(CLASSES[dataset])})
    if expected.isna().any() or not labels.eq(expected).all():
        raise ValueError("Class names and labels do not match the fixed class order")
    rows = rows.copy()
    rows["label"] = labels.astype(np.int64)
    for split in SPLITS:
        present = set(rows.loc[rows.split == split, "class_name"])
        missing = set(CLASSES[dataset]) - present
        if missing:
            raise ValueError(f"Every class must appear in {split}; missing {sorted(missing)}")
    # Defense in depth: a forged group ID must not allow exact content leakage.
    for name in ("sha256", "pixel_sha256"):
        if (rows.groupby(name).split.nunique() > 1).any():
            raise ValueError("Exact duplicate content crosses partitions")
    if (rows.groupby("pixel_sha256").label.nunique() > 1).any():
        raise ValueError("Conflicting exact-content labels remain in the manifest")
    return rows


def load_manifest(manifest_path, dataset):
    """Validate a closed-set manifest without opening source images.

    If an audit accompanies the manifest, all locked output hashes are checked.
    A standalone CSV remains usable for synthetic checks and explicit imports.
    """
    _check_dataset(dataset)
    path = Path(manifest_path)
    audit_path = path.parent / "audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        config = audit.get("config", {})
        if config.get("dataset") != dataset or config.get("classes") != list(CLASSES[dataset]):
            raise ValueError("Manifest audit dataset or class order does not match")
        if path.name != "manifest.csv":
            raise ValueError("A locked dataset must be loaded through manifest.csv")
        _verify_outputs(path.parent, audit)
    rows = pd.read_csv(path, dtype={name: str for name in MANIFEST_COLUMNS if name != "label"})
    return _validate_frame(rows, dataset)


def _assign_splits(groups, rows, classes, seed):
    """Stratify component count vectors, with explicit class-coverage repair.

    Counts target images, not merely group totals; cross-class near matches are
    represented by vectors. Failure to find valid coverage stops preparation.
    """
    keys = sorted(groups)
    vectors = np.zeros((len(keys), len(classes)), dtype=np.int64)
    for i, group in enumerate(keys):
        for index in groups[group]:
            vectors[i, rows[index]["label"]] += 1
    for label, name in enumerate(classes):
        if np.count_nonzero(vectors[:, label]) < len(SPLITS):
            raise ValueError(f"Need at least three independent duplicate groups for {name}")
    targets = np.asarray(SPLIT_RATIOS)[:, None] * vectors.sum(axis=0)[None, :]
    scale = np.maximum(targets, 1.0)

    def objective(counts):
        return float((((counts - targets) ** 2) / scale).sum())

    rng = np.random.default_rng(seed)
    best = None
    # Usually the first assignment suffices. Retry tie orders only if coverage
    # cannot be repaired, rather than silently excluding a class or splitting a group.
    for _ in range(16):
        ties = rng.random(len(keys))
        order = sorted(range(len(keys)), key=lambda index: (-int(vectors[index].sum()), ties[index]))
        counts = np.zeros_like(targets, dtype=np.int64)
        assigned = np.full(len(keys), -1, dtype=np.int64)
        for index in order:
            delta = (((counts + vectors[index] - targets) ** 2 - (counts - targets) ** 2) / scale).sum(axis=1)
            destination = int(np.argmin(delta))
            assigned[index] = destination
            counts[destination] += vectors[index]
        for _repair in range(3 * len(classes)):
            missing = np.argwhere(counts == 0)
            if not len(missing):
                break
            choices = []
            before_missing = len(missing)
            for destination, label in missing:
                for index in np.flatnonzero(vectors[:, label]):
                    source = assigned[index]
                    if source == destination:
                        continue
                    candidate = counts.copy()
                    candidate[source] -= vectors[index]
                    candidate[destination] += vectors[index]
                    if np.count_nonzero(candidate == 0) < before_missing:
                        choices.append((objective(candidate), int(index), int(destination)))
            if not choices:
                break
            _, index, destination = min(choices)
            counts[assigned[index]] -= vectors[index]
            counts[destination] += vectors[index]
            assigned[index] = destination
        if (counts == 0).any():
            continue
        for _refinement in range(4):
            improved = False
            for index in order:
                source = assigned[index]
                if np.any(counts[source] - vectors[index] <= 0):
                    continue
                old = objective(counts)
                choices = []
                for destination in range(len(SPLITS)):
                    if destination == source:
                        continue
                    candidate = counts.copy()
                    candidate[source] -= vectors[index]
                    candidate[destination] += vectors[index]
                    choices.append((objective(candidate), destination))
                score, destination = min(choices)
                if score < old - 1e-10:
                    counts[source] -= vectors[index]
                    counts[destination] += vectors[index]
                    assigned[index] = destination
                    improved = True
            if not improved:
                break
        best = assigned
        break
    if best is None:
        raise ValueError("Indivisible duplicate groups could not give every class a train/val/test partition")
    return {group: SPLITS[int(best[index])] for index, group in enumerate(keys)}


def _write_csv(path, records, columns):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def prepare_dataset(dataset, raw_root, output_dir, split_seed=42, near_distance=4):
    """Create or verify an immutable, independently split six-/nine-class dataset."""
    _check_dataset(dataset)
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    if type(near_distance) is not int or not 0 <= near_distance <= 64:
        raise ValueError("near_distance must be an integer between 0 and 64")
    root, output = Path(raw_root).resolve(), Path(output_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    if output == root or output.is_relative_to(root):
        raise ValueError("Prepared outputs must be outside the raw dataset directory")
    config = {"version": _VERSION, "dataset": dataset, "raw_root": str(root),
              "classes": list(CLASSES[dataset]), "split_seed": split_seed,
              "near_distance": near_distance, "split_names": list(SPLITS),
              "split_ratios": list(SPLIT_RATIOS), "exact_conflict_policy": "exclude_all_copies"}
    all_outputs = [output / name for name in (*_OUTPUTS, "audit.json")]
    if any(path.exists() for path in all_outputs):
        if not all(path.is_file() for path in all_outputs):
            raise ValueError("Incomplete locked output; use a new output directory")
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        if audit.get("config") != config:
            raise ValueError("Locked dataset configuration changed; use a new output directory")
        _verify_outputs(output, audit)
        _, _, fingerprint = _input_inventory(dataset, root)
        if audit.get("input_fingerprint") != fingerprint:
            raise ValueError("Dataset input bytes or paths changed after locking")
        load_manifest(output / "manifest.csv", dataset)
        return output / "manifest.csv"

    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".prepare.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError("Another preparation owns this output directory") from error
    try:
        handle.close()
        if any(path.exists() for path in all_outputs):
            raise ValueError("Output appeared during preparation; refusing to overwrite")
        inventory, metadata, fingerprint = _input_inventory(dataset, root)
        rows, excluded = [], []
        for item in metadata:
            excluded.append({**item, "reason": "archive_metadata"})
        for record in inventory:
            name = _class_name(Path(record["path"]), root, dataset)
            sample_id = hashlib.sha256(f"{dataset}/{record['relative_path']}".encode()).hexdigest()[:20]
            if name is None:
                excluded.append({**record, "sample_id": sample_id, "reason": "unmapped_directory"})
                continue
            pixel_hash, phash = _image_hashes(record["path"])
            rows.append({"sample_id": sample_id, "path": record["path"], "dataset": dataset,
                         "class_name": name, "label": CLASSES[dataset].index(name),
                         "sha256": record["sha256"], "pixel_sha256": pixel_hash,
                         "phash": f"{phash:016x}", "_phash": phash})
        exact_labels = defaultdict(set)
        for row in rows:
            exact_labels[row["pixel_sha256"]].add(row["label"])
        conflicts = {key for key, labels in exact_labels.items() if len(labels) > 1}
        blocked = {i for i, row in enumerate(rows) if row["pixel_sha256"] in conflicts}
        for index in sorted(blocked):
            excluded.append({**rows[index], "reason": "conflicting_exact_class_labels"})
        parents = list(range(len(rows)))

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        duplicates, tree = [], _BKTree()
        for index, row in enumerate(rows):
            for previous, distance in tree.query(row["_phash"], near_distance):
                other = rows[previous]
                if index not in blocked and previous not in blocked:
                    parents[find(index)] = find(previous)
                match = ("sha256" if row["sha256"] == other["sha256"] else
                         "pixel_sha256" if row["pixel_sha256"] == other["pixel_sha256"] else "phash")
                duplicates.append({"dataset": dataset, "sample_id_a": other["sample_id"],
                    "sample_id_b": row["sample_id"], "path_a": other["path"], "path_b": row["path"],
                    "class_name_a": other["class_name"], "class_name_b": row["class_name"],
                    "match_type": match, "hamming_distance": distance,
                    "cross_class": row["class_name"] != other["class_name"],
                    "excluded_a": previous in blocked, "excluded_b": index in blocked})
            tree.add(row["_phash"], index)
        components = defaultdict(list)
        for index in range(len(rows)):
            if index not in blocked:
                components[find(index)].append(index)
        groups = {}
        for indices in components.values():
            group = hashlib.sha256("\n".join(sorted(rows[i]["sample_id"] for i in indices)).encode()).hexdigest()[:20]
            groups[group] = indices
            for index in indices:
                rows[index]["group_id"] = group
        assignment = _assign_splits(groups, rows, CLASSES[dataset], split_seed)
        retained = []
        for index, row in enumerate(rows):
            if index not in blocked:
                row["split"] = assignment[row["group_id"]]
                retained.append({key: row[key] for key in MANIFEST_COLUMNS})
        frame = _validate_frame(pd.DataFrame(retained, columns=MANIFEST_COLUMNS), dataset)
        counts = []
        for label, name in enumerate(CLASSES[dataset]):
            for split in SPLITS:
                part = frame[(frame.label == label) & (frame.split == split)]
                counts.append({"dataset": dataset, "class_name": name, "label": label,
                               "split": split, "count": len(part), "group_count": part.group_id.nunique()})
        _write_csv(output / "manifest.csv", retained, MANIFEST_COLUMNS)
        _write_csv(output / "duplicates.csv", duplicates, DUPLICATE_COLUMNS)
        _write_csv(output / "excluded.csv", excluded, EXCLUDED_COLUMNS)
        _write_csv(output / "class_counts.csv", counts, COUNT_COLUMNS)
        audit = {"config": config, "input_fingerprint": fingerprint, "input_images": len(inventory),
                 "archive_metadata_count": len(metadata), "retained_images": len(retained),
                 "excluded_count": len(excluded), "excluded_by_reason": dict(Counter(r["reason"] for r in excluded)),
                 "conflicting_pixel_groups": len(conflicts), "group_count": len(groups),
                 "duplicate_pairs": len(duplicates), "counts": counts,
                 "actual_split_counts": {split: int((frame.split == split).sum()) for split in SPLITS},
                 "output_sha256": {name: _sha(output / name) for name in _OUTPUTS},
                 "source_code_sha256": _sha(__file__),
                 "notes": ["Ratios are approximate because duplicate groups cannot be split.",
                           "Perceptual matches retain their labels; all copies of exact pixel label conflicts are excluded.",
                           "Source files are never modified; archive metadata is excluded before image decoding.",
                           "Duplicate checks do not prove absence of same-object or pretraining overlap."]}
        audit["audit_content_sha256"] = _json_digest(audit)
        temporary = output / "audit.json.tmp"
        temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(output / "audit.json")
    finally:
        lock.unlink(missing_ok=True)
    return output / "manifest.csv"
