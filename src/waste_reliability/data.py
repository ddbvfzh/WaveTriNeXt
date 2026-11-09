"""Deterministic, duplicate-group-safe preparation of TrashNet and RealWaste."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

KNOWN_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic")
UNKNOWN_CLASSES = ("food organics", "textile trash", "vegetation")
MANIFEST_COLUMNS = ("sample_id", "path", "dataset", "class_name", "label", "group_id", "sha256", "phash", "split")
_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_VERSION = 3


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(name):
    return re.sub(r"\s+", " ", re.sub(r"^\d+[\s_.-]*", "", name.lower().replace("_", " ").replace("-", " "))).strip()


def _class_name(path, root):
    accepted = set(KNOWN_CLASSES + UNKNOWN_CLASSES + ("trash", "miscellaneous trash", "misc trash"))
    for part in reversed(path.relative_to(root).parts[:-1]):
        candidate = _normalise(part)
        if candidate in accepted:
            return candidate
    return None


def _inventory(roots):
    inventory = []
    ignored_metadata = []
    for dataset, root in roots.items():
        for path in sorted(root.rglob("*"), key=lambda p: p.as_posix().lower()):
            if path.is_file() and path.suffix.lower() in _EXTENSIONS:
                relative = path.relative_to(root)
                # Finder's AppleDouble files have image suffixes but are metadata,
                # not corrupt training images. Official TrashNet ZIP contains them.
                if "__MACOSX" in relative.parts or path.name.startswith("._"):
                    ignored_metadata.append({"dataset": dataset, "path": str(path.resolve())})
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(f"Image escapes dataset directory: {path}")
                inventory.append({"dataset": dataset, "relative_path": path.relative_to(root).as_posix(),
                                  "path": str(resolved), "sha256": _sha(path)})
        if not any(row["dataset"] == dataset for row in inventory):
            raise ValueError(f"No supported images found in {dataset}: {root}")
    return inventory, ignored_metadata


def _fingerprint(inventory):
    payload = [{k: row[k] for k in ("dataset", "relative_path", "sha256")} for row in inventory]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _image_hashes(path):
    try:
        with Image.open(path) as original:
            rgb = ImageOps.exif_transpose(original).convert("RGB")
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + b"\0" + rgb.tobytes()).hexdigest()
            grey = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    except Exception as error:
        raise ValueError(f"Cannot decode image {path}: {error}") from error
    coefficients = dctn(grey, type=2, norm="ortho")[:8, :8]
    bits = coefficients > np.median(coefficients)
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return pixel_hash, value


class _BKTree:
    """Hamming metric index; identical hashes share a node."""
    def __init__(self):
        self.root = None

    def query(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for index in node[1]:
                    yield index, distance
            pending.extend(child for edge, child in node[2].items() if distance - radius <= edge <= distance + radius)

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]


def _assign_splits(groups, rows, seed):
    """Greedy global grouping with a stratified squared-deficit objective."""
    keys = [(dataset, name) for dataset in ("trashnet", "realwaste") for name in KNOWN_CLASSES]
    positions = {key: index for index, key in enumerate(keys)}
    vectors = {}
    for group, indices in groups.items():
        vector = np.zeros(len(keys), dtype=float)
        for index in indices:
            row = rows[index]
            if row["label"] >= 0:
                vector[positions[(row["dataset"], row["class_name"])]] += 1
        if vector.sum():
            vectors[group] = vector
    totals = sum(vectors.values())
    targets = np.asarray([0.65, 0.15, 0.20])[:, None] * totals[None, :]
    scale = np.maximum(targets, 1)
    counts = np.zeros_like(targets)
    rng = np.random.default_rng(seed)
    ties = {group: rng.random() for group in sorted(vectors)}
    order = sorted(vectors, key=lambda group: (-vectors[group].sum(), ties[group]))
    assignment = {}
    for group in order:
        vector = vectors[group]
        delta = (((counts + vector - targets) ** 2 - (counts - targets) ** 2) / scale).sum(axis=1)
        destination = int(np.argmin(delta))
        counts[destination] += vector
        assignment[group] = destination
    for _ in range(3):
        improved = False
        for group in order:
            source = assignment[group]
            vector = vectors[group]
            remove_cost = (((counts[source] - vector - targets[source]) ** 2 - (counts[source] - targets[source]) ** 2) / scale[source]).sum()
            add_cost = (((counts + vector - targets) ** 2 - (counts - targets) ** 2) / scale).sum(axis=1)
            delta = remove_cost + add_cost
            delta[source] = 0
            destination = int(np.argmin(delta))
            if delta[destination] < -1e-9:
                counts[source] -= vector
                counts[destination] += vector
                assignment[group] = destination
                improved = True
        if not improved:
            break
    names = ("train", "calibration", "test")
    return {group: names[index] for group, index in assignment.items()}


def _write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(trashnet_root, realwaste_root, output_dir, seed=20261001, near_distance=4,
                 exact_conflict_policy="exclude"):
    """Return a locked manifest Path; changing input bytes/settings requires a new directory.

    Near duplicates are grouped conservatively. Exact pixel duplicates with conflicting
    known labels are excluded by default without relabeling any image. Mere perceptual
    matches retain the original labels and remain grouped for split safety.
    """
    if not isinstance(near_distance, int) or not 0 <= near_distance <= 64:
        raise ValueError("near_distance must be an integer from 0 to 64")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if exact_conflict_policy not in ("exclude", "retain"):
        raise ValueError("exact_conflict_policy must be 'exclude' or 'retain'")
    roots = {"trashnet": Path(trashnet_root).resolve(), "realwaste": Path(realwaste_root).resolve()}
    for root in roots.values():
        if not root.is_dir():
            raise ValueError(f"Dataset directory does not exist: {root}")
    if roots["trashnet"] == roots["realwaste"]:
        raise ValueError("TrashNet and RealWaste must have different roots")
    output_dir = Path(output_dir).resolve()
    manifest = output_dir / "manifest.csv"
    audit_path = output_dir / "audit.json"
    duplicates_path = output_dir / "duplicates.csv"
    config = {"version": _VERSION, "roots": {key: str(value) for key, value in roots.items()},
              "seed": seed, "near_distance": near_distance, "split_ratios": [0.65, 0.15, 0.20],
              "known_classes": list(KNOWN_CLASSES), "unknown_classes": list(UNKNOWN_CLASSES),
              "exact_conflict_policy": exact_conflict_policy,
              "exact_conflict_definition": "identical decoded RGB pixels with at least two different known class labels"}
    existing = any(path.exists() for path in (manifest, audit_path, duplicates_path))
    if existing:
        if not all(path.is_file() for path in (manifest, audit_path, duplicates_path)):
            raise ValueError("Incomplete locked output; use a new output directory")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("config") != config:
            raise ValueError("Existing locked manifest has different configuration; use a new output directory")
        if audit.get("output_sha256") != {"manifest.csv": _sha(manifest), "duplicates.csv": _sha(duplicates_path)}:
            raise ValueError("Locked manifest or duplicates.csv has changed")
    inventory, ignored_metadata = _inventory(roots)
    fingerprint = _fingerprint(inventory)
    if existing:
        if audit.get("input_fingerprint") != fingerprint:
            raise ValueError("Dataset input bytes or paths changed after manifest was locked")
        return manifest
    rows, excluded = [], []
    for record in inventory:
        dataset = record["dataset"]
        name = _class_name(Path(record["path"]), roots[dataset])
        reason = None
        if name not in KNOWN_CLASSES and not (dataset == "realwaste" and name in UNKNOWN_CLASSES):
            reason = "excluded_trash_category" if name in ("trash", "misc trash", "miscellaneous trash") else "unmapped_directory"
        if reason:
            excluded.append({"path": record["path"], "dataset": dataset, "class_name": name, "reason": reason})
            continue
        pixel_hash, phash = _image_hashes(record["path"])
        identity = f"{dataset}/{record['relative_path']}"
        rows.append({"sample_id": hashlib.sha256(identity.encode()).hexdigest()[:20], "path": record["path"],
                     "dataset": dataset, "class_name": name, "label": KNOWN_CLASSES.index(name) if name in KNOWN_CLASSES else -1,
                     "sha256": record["sha256"], "phash": f"{phash:016x}", "_phash": phash, "_rgb_sha256": pixel_hash})
    for dataset in roots:
        present = {row["class_name"] for row in rows if row["dataset"] == dataset and row["label"] >= 0}
        missing = set(KNOWN_CLASSES) - present
        if missing:
            raise ValueError(f"Missing known classes in {dataset}: {', '.join(sorted(missing))}")
    exact_labels = defaultdict(set)
    for row in rows:
        if row["label"] >= 0:
            exact_labels[row["_rgb_sha256"]].add(row["label"])
    ambiguous_hashes = {key for key, labels in exact_labels.items() if len(labels) > 1}
    ambiguous_indices = {index for index, row in enumerate(rows)
                         if exact_conflict_policy == "exclude" and row["_rgb_sha256"] in ambiguous_hashes}
    for index in sorted(ambiguous_indices):
        row = rows[index]
        excluded.append({"path": row["path"], "dataset": row["dataset"], "class_name": row["class_name"],
                         "label": row["label"], "sample_id": row["sample_id"], "sha256": row["sha256"],
                         "pixel_sha256": row["_rgb_sha256"], "reason": "conflicting_exact_known_labels",
                         "rationale": "Identical decoded RGB content has multiple known labels; exclude all copies without relabeling."})
    for dataset in roots:
        remaining = {row["class_name"] for index, row in enumerate(rows)
                     if index not in ambiguous_indices and row["dataset"] == dataset and row["label"] >= 0}
        if set(KNOWN_CLASSES) - remaining:
            raise ValueError(f"Missing known classes after exact conflict exclusions in {dataset}")
    parents = list(range(len(rows)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    tree = _BKTree()
    duplicates = []
    for index, row in enumerate(rows):
        for previous, distance in tree.query(row["_phash"], near_distance):
            other = rows[previous]
            parents[find(index)] = find(previous)
            match = "sha256" if row["sha256"] == other["sha256"] else "rgb_sha256" if row["_rgb_sha256"] == other["_rgb_sha256"] else "phash"
            cross_class = row["class_name"] != other["class_name"]
            duplicates.append({"sample_id_a": other["sample_id"], "sample_id_b": row["sample_id"],
                               "path_a": other["path"], "path_b": row["path"], "dataset_a": other["dataset"], "dataset_b": row["dataset"],
                               "class_name_a": other["class_name"], "class_name_b": row["class_name"],
                               "match_type": match, "hamming_distance": distance, "cross_class": cross_class,
                               "excluded_exact_conflict_a": previous in ambiguous_indices,
                               "excluded_exact_conflict_b": index in ambiguous_indices,
                               "review_required": cross_class or match == "phash"})
        tree.add(row["_phash"], index)
    components = defaultdict(list)
    for index in range(len(rows)):
        if index in ambiguous_indices:
            continue
        components[find(index)].append(index)
    groups = {}
    for indices in components.values():
        group = hashlib.sha256("\n".join(sorted(rows[index]["sample_id"] for index in indices)).encode()).hexdigest()[:20]
        groups[group] = indices
        for index in indices:
            rows[index]["group_id"] = group
    assignment = _assign_splits(groups, rows, seed)
    retained = []
    for index, row in enumerate(rows):
        if index in ambiguous_indices:
            continue
        if row["label"] == -1 and row["group_id"] in assignment:
            excluded.append({"path": row["path"], "dataset": row["dataset"], "class_name": row["class_name"],
                             "sample_id": row["sample_id"], "group_id": row["group_id"], "reason": "unknown_duplicate_of_known"})
            continue
        row["split"] = assignment[row["group_id"]] if row["label"] >= 0 else "unknown_test"
        retained.append(row)
    counts = Counter((row["dataset"], row["class_name"], row["split"]) for row in retained)
    warning_messages = []
    for dataset in roots:
        for name in KNOWN_CLASSES:
            for split in ("train", "calibration", "test"):
                if not counts[(dataset, name, split)]:
                    warning_messages.append(f"No {split} images for {dataset}/{name}; indivisible duplicate groups may prevent balance")
    audit = {"config": config, "input_fingerprint": fingerprint, "input_images": len(inventory),
             "ignored_archive_metadata_count": len(ignored_metadata), "ignored_archive_metadata": ignored_metadata,
             "retained_images": len(retained), "known_images": sum(row["label"] >= 0 for row in retained),
             "unknown_images": sum(row["label"] == -1 for row in retained),
             "group_count": len({row["group_id"] for row in retained}), "duplicate_pairs": len(duplicates),
             "conflicting_exact_content_groups": len(ambiguous_hashes),
             "excluded_exact_conflict_images": len(ambiguous_indices),
             "pending_review_pairs": sum(pair["review_required"] for pair in duplicates),
             "cross_class_pairs": sum(pair["cross_class"] for pair in duplicates),
             "match_counts": dict(Counter(pair["match_type"] for pair in duplicates)),
             "counts": [{"dataset": key[0], "class_name": key[1], "split": key[2], "count": value} for key, value in sorted(counts.items())],
             "excluded": excluded, "warnings": warning_messages,
             "notes": ["Near duplicate matches require review; labels are not automatically changed.",
                       "Exact conflicting content is excluded before split allocation under the default policy.",
                       "Connected components are indivisible, so split ratios are approximate.",
                       "No augmentation is generated before the split.",
                       "No duplicate detected by these rules does not prove absence of leakage."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".prepare.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError("Another preparation owns this output directory") from error
    try:
        handle.close()
        if any(path.exists() for path in (manifest, audit_path, duplicates_path)):
            raise ValueError("Output appeared during preparation; refusing to overwrite")
        _write_csv(manifest, retained, MANIFEST_COLUMNS)
        _write_csv(duplicates_path, duplicates, ("sample_id_a", "sample_id_b", "path_a", "path_b", "dataset_a", "dataset_b", "class_name_a", "class_name_b", "match_type", "hamming_distance", "cross_class", "excluded_exact_conflict_a", "excluded_exact_conflict_b", "review_required"))
        audit["output_sha256"] = {"manifest.csv": _sha(manifest), "duplicates.csv": _sha(duplicates_path)}
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        lock.unlink(missing_ok=True)
    return manifest
