#!/usr/bin/env python3
"""Verify portable fixed splits and original image bytes without ML dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def assignment_digest(rows: list[dict[str, str]]) -> str:
    content = [{k: v for k, v in row.items() if k != "path"}
               for row in sorted(rows, key=lambda row: row["sample_id"])]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def local_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or "\\" in relative or ":" in relative:
        raise ValueError(f"Expected a portable relative path: {relative}")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes repository: {relative}")
    return path


def verify(root: Path, image_hashes: bool = True) -> dict:
    inventory = json.loads((root / "datasets/dataset_inventory.json").read_text(encoding="utf-8"))
    result = {"status": "passed", "image_hashes_checked": image_hashes, "datasets": {}}
    for dataset, expected in inventory["datasets"].items():
        base = root / "datasets" / dataset
        manifest_path = local_path(root, expected["manifest"])
        image_index_path = local_path(root, expected["raw_image_inventory"])
        if sha256(manifest_path) != expected["manifest_sha256"]:
            raise ValueError(f"{dataset}: manifest checksum mismatch")
        if sha256(image_index_path) != expected["raw_image_inventory_sha256"]:
            raise ValueError(f"{dataset}: image inventory checksum mismatch")
        rows, images = read_csv(manifest_path), read_csv(image_index_path)
        if len(rows) != expected["retained_images"] or len(images) != expected["images"]:
            raise ValueError(f"{dataset}: image count mismatch")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError(f"{dataset}: duplicate sample IDs")
        if len({row["path"] for row in images}) != len(images):
            raise ValueError(f"{dataset}: duplicate image paths")
        if len({row["path"] for row in rows}) != len(rows):
            raise ValueError(f"{dataset}: duplicate manifest image paths")
        if assignment_digest(rows) != expected["portable_assignment_sha256"]:
            raise ValueError(f"{dataset}: fixed assignments changed")
        if dict(Counter(row["split"] for row in rows)) != expected["split_counts"]:
            raise ValueError(f"{dataset}: split counts changed")
        if dict(Counter(row["class_name"] for row in images)) != expected["raw_class_counts"]:
            raise ValueError(f"{dataset}: raw class counts changed")
        classes = expected["class_order"]
        indexed_images = {row["path"]: row for row in images}
        for row in rows:
            if row["dataset"] != dataset or row["split"] not in ("train", "val", "test"):
                raise ValueError(f"{dataset}: invalid dataset or split")
            if row["class_name"] not in classes or int(row["label"]) != classes.index(row["class_name"]):
                raise ValueError(f"{dataset}: class-label mismatch")
            indexed = indexed_images.get(row["path"])
            if indexed is None or indexed["sha256"] != row["sha256"]:
                raise ValueError(f"{dataset}: manifest image absent from original-byte inventory")
        for field in ("group_id", "sha256", "pixel_sha256"):
            partitions = defaultdict(set)
            for row in rows:
                partitions[row[field]].add(row["split"])
            if any(len(splits) > 1 for splits in partitions.values()):
                raise ValueError(f"{dataset}: {field} crosses fixed splits")
        for split in ("train", "val", "test"):
            split_rows = read_csv(base / "splits" / f"{split}.csv")
            if split_rows != [row for row in rows if row["split"] == split]:
                raise ValueError(f"{dataset}: {split}.csv differs from manifest")
        actual_images = {path.relative_to(root).as_posix() for path in (base / "images").rglob("*")
                         if path.is_file()}
        if actual_images != set(indexed_images):
            raise ValueError(f"{dataset}: missing or unexpected files in images/")
        total_bytes = 0
        for image in images:
            path = local_path(root, image["path"])
            if not path.is_file() or path.stat().st_size != int(image["bytes"]):
                raise ValueError(f"Missing or different-size image: {image['path']}")
            if image_hashes and sha256(path) != image["sha256"]:
                raise ValueError(f"Image checksum mismatch: {image['path']}")
            total_bytes += path.stat().st_size
        if total_bytes != expected["image_bytes"]:
            raise ValueError(f"{dataset}: total bytes mismatch")
        result["datasets"][dataset] = {"images": len(images), "retained_images": len(rows),
                                         "image_bytes": total_bytes, "split_counts": expected["split_counts"]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-image-hashes", action="store_true", help="Check structure and sizes only; skip image SHA256")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = verify(args.repo_root.resolve(), not args.skip_image_hashes)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
