#!/usr/bin/env python3
"""构建针对稀有类别的确定性真实重采样数据集。

与 ``rare_resample.py`` 不同，此脚本不执行任何视觉或多边形几何变换。
它直接克隆源数据集，并使用唯一文件名添加字节级完全一致的副本。源目录结构绝不会被修改。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
import sys

try:
    from .rare_resample import (
        CLASS_NAMES,
        DEFAULT_SEED,
        IMAGE_EXTS,
        choose_tail_hard_samples,
        file_hash_manifest,
        image_files,
        read_yolo,
        verify_pairs,
    )
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from instance_segmentation.data.rare_resample import (
        CLASS_NAMES,
        DEFAULT_SEED,
        IMAGE_EXTS,
        choose_tail_hard_samples,
        file_hash_manifest,
        image_files,
        read_yolo,
        verify_pairs,
    )



TAIL_HARD_LIMIT = 400


def clone_tree_as_hardlinks(source: Path, destination: Path) -> None:
    """在同一文件系统上通过硬链接克隆目录树，避免重复占用存储。"""
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination} (use --force to rebuild)")
    shutil.copytree(source, destination, copy_function=os.link)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算指定文件的 SHA-256 哈希值。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_pair(source_image: Path, source_label: Path, output_root: Path, stem_suffix: str) -> Tuple[Path, Path]:
    """复制图像与标签对至训练集并同步建立硬链接。"""
    image_name = f"{source_image.stem}{stem_suffix}{source_image.suffix}"
    label_name = f"{source_image.stem}{stem_suffix}.txt"
    output_image = output_root / "images" / "train" / image_name
    output_label = output_root / "labels" / "train" / label_name
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    # copy2 将副本制作为独立文件同时保留源字节。并行的 train/images 布局硬链接至该文件。
    shutil.copy2(source_image, output_image)
    shutil.copy2(source_label, output_label)
    train_image = output_root / "train" / "images" / image_name
    train_label = output_root / "train" / "labels" / label_name
    train_image.parent.mkdir(parents=True, exist_ok=True)
    train_label.parent.mkdir(parents=True, exist_ok=True)
    os.link(output_image, train_image)
    os.link(output_label, train_label)
    return output_image, output_label


def duplicate_hash_record(source_image: Path, duplicate_image: Path, source_label: Path, duplicate_label: Path) -> Dict[str, object]:
    """生成源文件与副本文件的哈希记录并校验字节一致性。"""
    return {
        "source_image_sha256": sha256_file(source_image),
        "duplicate_image_sha256": sha256_file(duplicate_image),
        "source_label_sha256": sha256_file(source_label),
        "duplicate_label_sha256": sha256_file(duplicate_label),
        "image_bytes_identical": source_image.read_bytes() == duplicate_image.read_bytes(),
        "label_bytes_identical": source_label.read_bytes() == duplicate_label.read_bytes(),
    }


def build(source_root: Path, output_root: Path, seed: int, tail_limit: int, force: bool) -> Dict[str, object]:
    """执行真实重采样数据集的确定性构建流程。"""
    if force and output_root.exists():
        shutil.rmtree(output_root)
    clone_tree_as_hardlinks(source_root, output_root)

    source_images = image_files(source_root / "images" / "train")
    milkcup_id = next((cid for cid, name in CLASS_NAMES.items() if name == "milkcup"), 1)
    milk_sources: List[Tuple[Path, Path]] = []
    for image_path in source_images:
        label_path = source_root / "labels" / "train" / f"{image_path.stem}.txt"
        rows = read_yolo(label_path)
        if any(class_id == milkcup_id for class_id, _ in rows):
            milk_sources.append((image_path, label_path))
    if not milk_sources:
        raise RuntimeError(f"No milkcup source images found in dataset: {source_root}")

    tail_sources = choose_tail_hard_samples(source_root, seed, tail_limit)
    if not tail_sources:
        raise RuntimeError(f"No tail-only samples found in dataset: {source_root}")

    records: List[Dict[str, object]] = []
    duplicate_checks: List[Dict[str, object]] = []
    for source_image, source_label in milk_sources:
        duplicate_image, duplicate_label = copy_pair(source_image, source_label, output_root, "__real_milk_v1")
        checks = duplicate_hash_record(source_image, duplicate_image, source_label, duplicate_label)
        if not checks["image_bytes_identical"] or not checks["label_bytes_identical"]:
            raise RuntimeError(f"Milk duplicate hash mismatch for {source_image.name}")
        records.append({
            "category": "milkcup",
            "seed": seed,
            "source_image": f"images/train/{source_image.name}",
            "source_label": f"labels/train/{source_label.name}",
            "duplicate_image": f"images/train/{duplicate_image.name}",
            "duplicate_label": f"labels/train/{duplicate_label.name}",
            "operation": "byte_copy_no_transform",
        })
        duplicate_checks.append({"category": "milkcup", "source_image": source_image.name, **checks})

    for item in tail_sources:
        source_image = source_root / "images" / "train" / str(item["image"])
        source_label = source_root / "labels" / "train" / str(item["label"])
        duplicate_image, duplicate_label = copy_pair(source_image, source_label, output_root, "__real_tail_v1")
        checks = duplicate_hash_record(source_image, duplicate_image, source_label, duplicate_label)
        if not checks["image_bytes_identical"] or not checks["label_bytes_identical"]:
            raise RuntimeError(f"Tail duplicate hash mismatch for {source_image.name}")
        records.append({
            "category": "tail_hard",
            "seed": seed,
            "source_image": f"images/train/{source_image.name}",
            "source_label": f"labels/train/{source_label.name}",
            "duplicate_image": f"images/train/{duplicate_image.name}",
            "duplicate_label": f"labels/train/{duplicate_label.name}",
            "difficulty": item["difficulty"],
            "tail_area_norm": item["tail_area_norm"],
            "tail_edge_norm": item["tail_edge_norm"],
            "operation": "byte_copy_no_transform",
        })
        duplicate_checks.append({"category": "tail_hard", "source_image": source_image.name, **checks})

    yaml_lines = [
        f"path: {output_root.name}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for cid in sorted(CLASS_NAMES):
        yaml_lines.append(f"  {cid}: {CLASS_NAMES[cid]}")
    yaml_text = "\n".join(yaml_lines) + "\n"

    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (output_root / "data.yml").write_text(yaml_text, encoding="utf-8")

    hash_checks: Dict[str, object] = {}
    for split in ("valid", "test", "images/val", "labels/val", "images/test", "labels/test"):
        source_hashes = file_hash_manifest(source_root, split)
        output_hashes = file_hash_manifest(output_root, split)
        hash_checks[f"{split.replace('/', '_')}_unchanged"] = source_hashes == output_hashes
        if source_hashes != output_hashes:
            raise RuntimeError(f"Source/output hashes differ for {split}")

    # 原始训练集文件原样克隆。校验完整的原始训练图像/标签映射以及所有新副本的哈希。
    for split in ("images/train", "labels/train"):
        source_hashes = file_hash_manifest(source_root, split)
        output_hashes = file_hash_manifest(output_root, split)
        original_output_hashes = {name: value for name, value in output_hashes.items() if "__real_" not in name}
        hash_checks[f"{split.replace('/', '_')}_originals_unchanged"] = source_hashes == original_output_hashes
        if source_hashes != original_output_hashes:
            raise RuntimeError(f"Original train hashes differ for {split}")

    duplicate_checks_ok = all(c["image_bytes_identical"] and c["label_bytes_identical"] for c in duplicate_checks)
    hash_checks["all_duplicate_bytes_identical"] = duplicate_checks_ok
    if not duplicate_checks_ok:
        raise RuntimeError("At least one duplicate does not match its source bytes")

    verification = {split: verify_pairs(output_root, split) for split in ("train", "valid", "test")}
    if any(v["missing_labels"] or v["orphan_labels"] or v["bad_rows"] for v in verification.values()):
        raise RuntimeError(f"Dataset validation failed: {verification}")

    manifest = {
        "dataset": output_root.name,
        "source_dataset": str(source_root),
        "seed": seed,
        "policy": {
            "milkcup_source_images": len(milk_sources),
            "milkcup_extra_exposures_each": 1,
            "tail_hard_source_images": len(tail_sources),
            "tail_extra_exposures_each": 1,
            "new_images": len(records),
            "operation": "byte-identical copy; no pixel or polygon transformation",
            "tail_selection": "top difficulty: small tail polygon (65%) + border proximity (35%), deterministic SHA-256 tie-break",
        },
        "records": records,
        "tail_hard_selection": tail_sources,
        "duplicate_hash_checks": duplicate_checks,
        "verification": {"splits": verification, "hash_checks": hash_checks},
    }
    (output_root / "real_resampling_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {"dataset": output_root.name, "source_dataset": str(source_root), "seed": seed, "new_images": len(records), "splits": verification, "hash_checks": hash_checks}
    (output_root / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """解析命令行参数并执行真实重采样数据集构建。"""
    parser = argparse.ArgumentParser(description="Build the deterministic real-resampling dataset for the rare classes.")
    parser.add_argument("--source", type=Path, default=Path("datasets/current_dataset"), help="Source dataset path")
    parser.add_argument("--output", type=Path, default=Path("datasets/retrain_real_v2"), help="Output dataset path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED + 1, help="Random seed for deterministic generation")
    parser.add_argument("--tail-limit", type=int, default=TAIL_HARD_LIMIT, help="Maximum number of tail-hard samples to choose")
    parser.add_argument("--force", action="store_true", help="Overwrite output dataset if it already exists")
    args = parser.parse_args(argv)
    manifest = build(args.source.resolve(), args.output.resolve(), args.seed, args.tail_limit, args.force)
    print(json.dumps({"output": str(args.output.resolve()), "seed": manifest["seed"], "policy": manifest["policy"], "verification": manifest["verification"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
