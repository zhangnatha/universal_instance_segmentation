#!/usr/bin/env python3
"""复制已整理的 LabelMe 数据集，并为指定类别添加 PCA 角度标签。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from instance_segmentation.data.labelme import polygon_angle


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Prepared Detectron2 root containing train/ and val/")
    parser.add_argument("--output-dir", required=True, help="New dataset copy to create")
    parser.add_argument("--class-name", default="leg", help="Only this class receives PCA angle labels")
    args = parser.parse_args()

    source_root = Path(args.source_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if not all((source_root / split).is_dir() for split in ("train", "val")):
        parser.error("--source-dir must contain train/ and val/ directories")
    if output_root.exists():
        parser.error(f"output already exists; choose a new path: {output_root}")

    totals = {"images": 0, "annotations": 0, "angle_labels": 0}
    for split in ("train", "val"):
        source_split = source_root / split
        output_split = output_root / split
        output_split.mkdir(parents=True)
        split_counts = {"images": 0, "annotations": 0, "angle_labels": 0}
        for source in sorted(source_split.iterdir()):
            if source.suffix.lower() in IMAGE_SUFFIXES:
                (output_split / source.name).symlink_to(source.resolve())
                split_counts["images"] += 1
                continue
            if source.suffix.lower() != ".json":
                continue

            record = json.loads(source.read_text(encoding="utf-8"))
            for shape in record.get("shapes", []):
                if shape.get("label") != args.class_name:
                    continue
                points = shape.get("points", [])
                if len(points) < 3:
                    continue
                angle = polygon_angle(points)  # unoriented PCA axis in [0, 180)
                shape["angle_degrees"] = angle
                split_counts["angle_labels"] += 1
            (output_split / source.name).write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            split_counts["annotations"] += 1
        if split_counts["images"] != split_counts["annotations"]:
            raise RuntimeError(f"{split}: found {split_counts['images']} images and {split_counts['annotations']} JSON files")
        if split_counts["angle_labels"] == 0:
            raise RuntimeError(f"{split}: no polygon angle labels found for class {args.class_name!r}")
        for key, value in split_counts.items():
            totals[key] += value
        print(f"{split}: {split_counts}")

    manifest = {
        "source": str(source_root),
        "class_name": args.class_name,
        "angle_source": "polygon PCA major axis",
        "angle_period_degrees": 180,
        "counts": totals,
    }
    (output_root / "pca_angle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"created: {output_root}")


if __name__ == "__main__":
    main()
