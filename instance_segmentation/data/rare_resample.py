#!/usr/bin/env python3
"""构建确定性的稀有类别数据增强重采样数据集。

源数据集绝不会被修改。现有文件在文件系统支持时将硬链接到新数据集中，
仅有新增强的训练图像/标签会被写入。该脚本特意采用 Pillow 和纯 Python 几何运算，
以确保其输出不依赖于 OpenCV 版本或随机全局状态。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageEnhance, ImageOps

try:
    from .classes import load_class_names
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from instance_segmentation.data.classes import load_class_names


def _get_default_class_names() -> Dict[int, str]:
    """加载配置的类别名称，若未配置则回退到默认类别映射。"""
    names = load_class_names()
    if names:
        return {i: name for i, name in enumerate(names)}
    return {0: "leg", 1: "milkcup", 2: "nipple", 3: "tail"}


CLASS_NAMES = _get_default_class_names()
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
DEFAULT_SEED = 20260916
MILKCUP_VERSIONS = 2
TAIL_HARD_LIMIT = 400


def _find_class_id(name: str, default: int) -> int:
    """根据类别名称查找对应的类别 ID。"""
    for cid, cname in CLASS_NAMES.items():
        if cname == name:
            return cid
    return default


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算指定文件的 SHA-256 校验和哈希值。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(seed: int, *parts: str) -> bytes:
    """结合随机种子与字符串参数生成确定性 SHA-256 摘要字节。"""
    value = f"{seed}|" + "|".join(parts)
    return hashlib.sha256(value.encode("utf-8")).digest()


def read_yolo(path: Path) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """读取 YOLO 分割格式的多边形标注文件。"""
    rows: List[Tuple[int, List[Tuple[float, float]]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"Malformed polygon at {path}:{line_number}")
        class_id = int(float(fields[0]))
        coords = [float(v) for v in fields[1:]]
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        rows.append((class_id, points))
    return rows


def write_yolo(path: Path, rows: Sequence[Tuple[int, Sequence[Tuple[float, float]]]]) -> None:
    """将多边形标注列表写入 YOLO 分割格式文本文件。"""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for class_id, points in rows:
            vals: List[str] = [str(class_id)]
            for x, y in points:
                vals.extend((f"{min(1.0, max(0.0, x)):.8f}", f"{min(1.0, max(0.0, y)):.8f}"))
            fh.write(" ".join(vals) + "\n")


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    """使用鞋带公式（Shoelace formula）计算归一化多边形的面积。"""
    if len(points) < 3:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        total += x0 * y1 - x1 * y0
    return abs(total) * 0.5


def image_files(directory: Path) -> List[Path]:
    """获取目录下所有支持格式的图像文件路径列表。"""
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def copy_tree_as_hardlinks(source: Path, destination: Path) -> None:
    """在同一文件系统上通过硬链接克隆目录树，避免重复占用存储。"""
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination} (use --force to rebuild)")
    shutil.copytree(source, destination, copy_function=os.link)


def link_file(source: Path, destination: Path) -> None:
    """创建硬链接，如果目标文件已存在则先删除后重新链接。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    os.link(source, destination)


def append_train_file(output_root: Path, image_name: str, image: Image.Image, label_text: str) -> None:
    """在 images/train 下写入新文件，并硬链接到对应的 RF-DETR 目录结构中。"""
    root_image = output_root / "images" / "train" / image_name
    root_label = output_root / "labels" / "train" / (Path(image_name).stem + ".txt")
    image.save(root_image)
    root_label.write_text(label_text, encoding="utf-8", newline="\n")
    link_file(root_image, output_root / "train" / "images" / image_name)
    link_file(root_label, output_root / "train" / "labels" / root_label.name)


def reflect_pad(image: Image.Image, pad: int) -> Image.Image:
    """对 Pillow 图像执行边缘反射填充（镜像边缘）。"""
    # Pillow 没有开箱即用的反射填充工具。此处特意避免依赖 Numpy，
    # 采用切片镜像实现以保持脚本轻量化。
    src = image.convert("RGB")
    w, h = src.size
    left = src.crop((0, 0, min(pad, w), h)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    right = src.crop((max(0, w - pad), 0, w, h)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    mid = ImageOps.expand(src, border=(pad, 0, pad, 0), fill=0)
    # 上述边缘切片已镜像，需放置到两端。
    padded = Image.new("RGB", (w + 2 * pad, h), 0)
    padded.paste(left, (0, 0))
    padded.paste(src, (pad, 0))
    padded.paste(right, (pad + w, 0))
    top = padded.crop((0, 0, padded.width, min(pad, h))).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    bottom = padded.crop((0, max(0, h - pad), padded.width, h)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    out = Image.new("RGB", (padded.width, h + 2 * pad), 0)
    out.paste(top, (0, 0))
    out.paste(padded, (0, pad))
    out.paste(bottom, (0, pad + h))
    return out


def affine_image_and_points(
    image: Image.Image,
    rows: Sequence[Tuple[int, Sequence[Tuple[float, float]]]],
    angle: float,
    scale: float,
) -> Tuple[Image.Image, List[Tuple[int, List[Tuple[float, float]]]]]:
    """围绕图像中心进行旋转与缩放仿射变换，并返回归一化的多边形顶点坐标。"""
    image = image.convert("RGB")
    w, h = image.size
    pad = max(16, int(math.ceil(max(w, h) * 0.12)))
    padded = reflect_pad(image, pad)
    pw, ph = padded.size
    cx, cy = pw / 2.0, ph / 2.0
    theta = math.radians(angle)
    c, s = math.cos(theta) * scale, math.sin(theta) * scale

    # 在图像坐标系中执行正向变换（遵循 Pillow rotate 规范）。
    a, b, d, e = c, -s, s, c
    tx, ty = cx - a * cx - b * cy, cy - d * cx - e * cy
    det = a * e - b * d
    ia, ib, id_, ie = e / det, -b / det, -d / det, a / det
    ic = -(ia * tx + ib * ty)
    if_ = -(id_ * tx + ie * ty)
    transformed = padded.transform(
        padded.size,
        Image.Transform.AFFINE,
        (ia, ib, ic, id_, ie, if_),
        resample=Image.Resampling.BICUBIC,
    )
    output = transformed.crop((pad, pad, pad + w, pad + h))

    out_rows: List[Tuple[int, List[Tuple[float, float]]]] = []
    for class_id, points in rows:
        out_points: List[Tuple[float, float]] = []
        for x_norm, y_norm in points:
            # 仿射变换在反射填充坐标系中进行，源点需加上相同的填充偏移量。
            x, y = x_norm * w + pad, y_norm * h + pad
            x, y = a * x + b * y + tx - pad, d * x + e * y + ty - pad
            out_points.append((min(1.0, max(0.0, x / w)), min(1.0, max(0.0, y / h))))
        out_rows.append((class_id, out_points))
    return output, out_rows


def horizontal_flip(image: Image.Image, rows: Sequence[Tuple[int, Sequence[Tuple[float, float]]]]) -> Tuple[Image.Image, List[Tuple[int, List[Tuple[float, float]]]]]:
    """对图像执行水平镜像翻转，并同步翻转多边形标注点的 x 坐标。"""
    output = ImageOps.mirror(image.convert("RGB"))
    out_rows = [(cid, [(1.0 - x, y) for x, y in points]) for cid, points in rows]
    return output, out_rows


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """使用 Pillow 的 ImageEnhance 调整图像亮度。"""
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(factor)


def choose_tail_hard_samples(source_root: Path, seed: int, limit: int) -> List[Dict[str, object]]:
    """按难度指标（小面积占比与边缘邻近度）筛选尾部难样本。"""
    candidates: List[Dict[str, object]] = []
    image_dir = source_root / "images" / "train"
    label_dir = source_root / "labels" / "train"
    tail_id = _find_class_id("tail", 3)
    milkcup_id = _find_class_id("milkcup", 1)

    for image_path in image_files(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        rows = read_yolo(label_path)
        classes = {cid for cid, _ in rows}
        if tail_id not in classes or milkcup_id in classes:
            continue
        with Image.open(image_path) as image:
            w, h = image.size
        tail_stats = []
        for cid, points in rows:
            if cid != tail_id:
                continue
            area = polygon_area(points)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            edge = min(min(xs), min(ys), 1.0 - max(xs), 1.0 - max(ys))
            tail_stats.append((area, max(0.0, edge)))
        if not tail_stats:
            continue
        min_area = min(item[0] for item in tail_stats)
        min_edge = min(item[1] for item in tail_stats)
        small_target = max(0.0, min(1.0, (0.06 - min_area) / 0.06))
        edge_target = max(0.0, min(1.0, (0.12 - min_edge) / 0.12))
        difficulty = 0.65 * small_target + 0.35 * edge_target
        tie = stable_digest(seed, image_path.name).hex()
        candidates.append(
            {
                "image": image_path.name,
                "label": label_path.name,
                "difficulty": difficulty,
                "tail_area_norm": min_area,
                "tail_edge_norm": min_edge,
                "tie": tie,
            }
        )
    candidates.sort(key=lambda item: (-float(item["difficulty"]), str(item["tie"])))
    return candidates[:limit]


def file_hash_manifest(root: Path, relative_dir: str) -> Dict[str, str]:
    """递归计算指定目录下所有文件的哈希清单字典。"""
    base = root / relative_dir
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(base.rglob("*")) if p.is_file()}


def verify_pairs(root: Path, split: str) -> Dict[str, object]:
    """校验指定子集中的图像与标签对应关系、类别有效性及坐标范围。"""
    if split == "train":
        image_dir, label_dir = root / "images" / "train", root / "labels" / "train"
    elif split == "valid":
        image_dir, label_dir = root / "valid" / "images", root / "valid" / "labels"
    else:
        image_dir, label_dir = root / "test" / "images", root / "test" / "labels"
    images = image_files(image_dir)
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    class_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    bad_rows: List[str] = []
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        try:
            rows = read_yolo(label_path)
            seen = set()
            for row_no, (cid, points) in enumerate(rows, 1):
                if cid not in CLASS_NAMES or len(points) < 3:
                    bad_rows.append(f"{label_path}:{row_no}:class_or_polygon")
                if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in points):
                    bad_rows.append(f"{label_path}:{row_no}:coordinate")
                class_counts[cid] += 1
                seen.add(cid)
            for cid in seen:
                image_counts[cid] += 1
        except Exception as exc:
            bad_rows.append(f"{label_path}:{exc}")
    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "bad_rows": bad_rows,
        "instance_counts": {CLASS_NAMES.get(k, str(k)): v for k, v in sorted(class_counts.items())},
        "image_counts": {CLASS_NAMES.get(k, str(k)): v for k, v in sorted(image_counts.items())},
    }


def build(source_root: Path, output_root: Path, seed: int, tail_limit: int, force: bool) -> Dict[str, object]:
    """执行稀有类别数据增强数据集的确定性构建流程。"""
    if force and output_root.exists():
        shutil.rmtree(output_root)
    copy_tree_as_hardlinks(source_root, output_root)

    train_images = image_files(source_root / "images" / "train")
    milkcup_id = _find_class_id("milkcup", 1)
    milk_sources: List[Tuple[Path, Path, List[Tuple[int, List[Tuple[float, float]]]]]] = []
    for image_path in train_images:
        label_path = source_root / "labels" / "train" / f"{image_path.stem}.txt"
        rows = read_yolo(label_path)
        if any(cid == milkcup_id for cid, _ in rows):
            milk_sources.append((image_path, label_path, rows))
    if not milk_sources:
        raise RuntimeError(f"No milkcup images found in dataset: {source_root}")

    tail_sources = choose_tail_hard_samples(source_root, seed, tail_limit)
    manifest_records: List[Dict[str, object]] = []

    for image_path, label_path, rows in milk_sources:
        with Image.open(image_path) as source_image:
            base = source_image.convert("RGB")
            for version in range(1, MILKCUP_VERSIONS + 1):
                token = stable_digest(seed, image_path.name, f"milkcup_v{version}")
                if version == 1:
                    augmented, new_rows = horizontal_flip(base, rows)
                    brightness = 0.94 + (token[0] / 255.0) * 0.12
                    augmented = adjust_brightness(augmented, brightness)
                    params = {"horizontal_flip": True, "angle_deg": 0.0, "scale": 1.0, "brightness": round(brightness, 6)}
                else:
                    angle = (-1.0 if token[0] & 1 else 1.0) * (5.0 + (token[1] / 255.0) * 5.0)
                    scale = 0.97 + (token[2] / 255.0) * 0.06
                    brightness = 0.95 + (token[3] / 255.0) * 0.10
                    augmented, new_rows = affine_image_and_points(base, rows, angle, scale)
                    augmented = adjust_brightness(augmented, brightness)
                    params = {"horizontal_flip": False, "angle_deg": round(angle, 6), "scale": round(scale, 6), "brightness": round(brightness, 6)}
                image_name = f"{image_path.stem}__rare_milk_v{version}{image_path.suffix}"
                label_text = ""
                for cid, points in new_rows:
                    vals = [str(cid)] + [f"{x:.8f}" for point in points for x in point]
                    label_text += " ".join(vals) + "\n"
                append_train_file(output_root, image_name, augmented, label_text)
                manifest_records.append(
                    {"output_image": f"images/train/{image_name}", "output_label": f"labels/train/{Path(image_name).stem}.txt", "source_image": f"images/train/{image_path.name}", "source_label": f"labels/train/{label_path.name}", "category": "milkcup", "version": version, "seed": seed, "params": params}
                )

    for item in tail_sources:
        image_path = source_root / "images" / "train" / str(item["image"])
        label_path = source_root / "labels" / "train" / str(item["label"])
        rows = read_yolo(label_path)
        with Image.open(image_path) as source_image:
            base = source_image.convert("RGB")
            token = stable_digest(seed, image_path.name, "tail_hard_v1")
            angle = (-1.0 if token[0] & 1 else 1.0) * (3.0 + (token[1] / 255.0) * 5.0)
            scale = 0.98 + (token[2] / 255.0) * 0.04
            brightness = 0.94 + (token[3] / 255.0) * 0.12
            augmented, new_rows = affine_image_and_points(base, rows, angle, scale)
            augmented = adjust_brightness(augmented, brightness)
            image_name = f"{image_path.stem}__rare_tail_v1{image_path.suffix}"
            label_text = ""
            for cid, points in new_rows:
                vals = [str(cid)] + [f"{x:.8f}" for point in points for x in point]
                label_text += " ".join(vals) + "\n"
            append_train_file(output_root, image_name, augmented, label_text)
            manifest_records.append(
                {"output_image": f"images/train/{image_name}", "output_label": f"labels/train/{Path(image_name).stem}.txt", "source_image": f"images/train/{image_path.name}", "source_label": f"labels/train/{label_path.name}", "category": "tail_hard", "version": 1, "seed": seed, "difficulty": item["difficulty"], "tail_area_norm": item["tail_area_norm"], "tail_edge_norm": item["tail_edge_norm"], "params": {"horizontal_flip": False, "angle_deg": round(angle, 6), "scale": round(scale, 6), "brightness": round(brightness, 6)}}
            )

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

    checks = {
        "source_valid_hashes": file_hash_manifest(source_root, "valid"),
        "output_valid_hashes": file_hash_manifest(output_root, "valid"),
        "source_test_hashes": file_hash_manifest(source_root, "test"),
        "output_test_hashes": file_hash_manifest(output_root, "test"),
        "source_images_val_hashes": file_hash_manifest(source_root, "images/val"),
        "output_images_val_hashes": file_hash_manifest(output_root, "images/val"),
        "source_images_test_hashes": file_hash_manifest(source_root, "images/test"),
        "output_images_test_hashes": file_hash_manifest(output_root, "images/test"),
    }
    checks["valid_unchanged"] = checks["source_valid_hashes"] == checks["output_valid_hashes"]
    checks["test_unchanged"] = checks["source_test_hashes"] == checks["output_test_hashes"]
    checks["images_val_unchanged"] = checks["source_images_val_hashes"] == checks["output_images_val_hashes"]
    checks["images_test_unchanged"] = checks["source_images_test_hashes"] == checks["output_images_test_hashes"]

    verification = {split: verify_pairs(output_root, split) for split in ("train", "valid", "test")}
    if not all(checks[k] for k in ("valid_unchanged", "test_unchanged", "images_val_unchanged", "images_test_unchanged")):
        raise RuntimeError("Validation/test byte hashes changed")
    if any(v["missing_labels"] or v["orphan_labels"] or v["bad_rows"] for v in verification.values()):
        raise RuntimeError(f"Dataset validation failed: {verification}")

    manifest = {
        "dataset": output_root.name,
        "source_dataset": str(source_root),
        "seed": seed,
        "policy": {"milkcup_source_images": len(milk_sources), "milkcup_versions_each": MILKCUP_VERSIONS, "tail_hard_source_images": len(tail_sources), "max_new_images": 1500, "tail_selection": "top difficulty: small tail polygon (65%) + border proximity (35%), deterministic SHA-256 tie-break"},
        "new_images": len(manifest_records),
        "records": manifest_records,
        "tail_hard_selection": tail_sources,
        "verification": {"splits": verification, "hash_checks": {k: v for k, v in checks.items() if k.endswith("unchanged")}},
    }
    (output_root / "rare_augmentation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {"dataset": output_root.name, "source_dataset": str(source_root), "seed": seed, "new_images": len(manifest_records), "splits": verification, "hash_checks": {k: v for k, v in checks.items() if k.endswith("unchanged")}}
    (output_root / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """解析命令行参数并执行稀有类别重采样数据集构建。"""
    parser = argparse.ArgumentParser(description="Build a deterministic rare-class augmentation dataset.")
    parser.add_argument("--source", type=Path, default=Path("datasets/current_dataset"), help="Source dataset path")
    parser.add_argument("--output", type=Path, default=Path("datasets/retrain_rare_v1"), help="Output dataset path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic generation")
    parser.add_argument("--tail-limit", type=int, default=TAIL_HARD_LIMIT, help="Maximum number of tail-hard samples to choose")
    parser.add_argument("--force", action="store_true", help="Overwrite output dataset if it already exists")
    args = parser.parse_args(argv)
    manifest = build(args.source.resolve(), args.output.resolve(), args.seed, args.tail_limit, args.force)
    print(json.dumps({"output": str(args.output.resolve()), "seed": manifest["seed"], "new_images": manifest["new_images"], "policy": manifest["policy"], "verification": manifest["verification"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
