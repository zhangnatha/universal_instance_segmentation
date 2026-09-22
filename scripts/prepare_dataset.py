#!/usr/bin/env python3
"""通用数据集准备工具
将原始 LabelMe 标注转换为针对 Detectron2、YOLO-seg 和 RF-DETR 的预划分数据集。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".BMP", ".PNG", ".JPG"}


def polygon_area(coords: List[float]) -> float:
    """使用鞋带公式计算多边形面积。"""
    x = coords[0::2]
    y = coords[1::2]
    if len(x) < 3:
        return 0.0
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Split and convert raw LabelMe dataset for Detectron2, YOLO-seg, and RF-DETR"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Path to source directory containing images and LabelMe JSON annotations",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root directory to store prepared datasets",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="List of class names in deterministic order (supports arbitrary number of classes, e.g. --classes cls1 cls2 ...)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--format",
        choices=["all", "detectron2", "yolo", "rfdetr"],
        default="all",
        help="Target backend format to export (default: all)",
    )
    parser.add_argument(
        "--resize",
        nargs="+",
        type=int,
        default=None,
        help="Optional target resolution (WIDTH HEIGHT or SIZE), e.g. --resize 640 640 or --resize 432",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split (default: 42)",
    )
    parser.add_argument(
        "--copy",
        "--copy-images",
        dest="copy",
        action="store_true",
        help="Copy images instead of creating symlinks (default: False, uses symlinks if possible when no resize)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from instance_segmentation.run_logging import setup_run_logging
    setup_run_logging("PREPARE_DATASET")

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    classes = list(args.classes)
    cat_to_id = {name: idx for idx, name in enumerate(classes)}

    if not source_dir.is_dir():
        print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    target_size = None
    if args.resize:
        if len(args.resize) == 1:
            target_size = (args.resize[0], args.resize[0])
        elif len(args.resize) == 2:
            target_size = (args.resize[0], args.resize[1])
        else:
            print("Error: --resize accepts 1 or 2 integers", file=sys.stderr)
            sys.exit(1)

    # 1. 发现样本
    json_files = sorted(source_dir.glob("*.json"))
    pairs: List[Tuple[Path, Path]] = []
    for jf in json_files:
        stem = jf.stem
        img_file = None
        for ext in IMG_EXTS:
            cand = source_dir / f"{stem}{ext}"
            if cand.is_file():
                img_file = cand
                break
        if img_file is not None:
            pairs.append((img_file, jf))

    if not pairs:
        print(f"Error: No valid image-JSON pairs found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} image-JSON pairs in {source_dir}.")
    print(f"Classes ({len(classes)}): {classes}")
    if target_size:
        print(f"Resize target: {target_size[0]}x{target_size[1]}")

    # 2. 打乱与划分（按基础词干分组以防止数据泄露）
    import re
    random.seed(args.seed)
    base_pairs: List[Tuple[Path, Path]] = []
    rep_pairs: List[Tuple[Path, Path]] = []
    for p in pairs:
        stem = p[0].stem
        if re.match(r"^rep\d+_", stem):
            rep_pairs.append(p)
        else:
            base_pairs.append(p)

    shuffled_base = list(base_pairs)
    random.shuffle(shuffled_base)
    val_count = max(1, int(len(shuffled_base) * args.val_ratio))
    val_pairs = shuffled_base[:val_count]
    train_base = shuffled_base[val_count:]

    # 对于副本对，仅当其基础父样本在 train_base 中时才分配到训练集
    train_base_stems = {p[0].stem for p in train_base}
    valid_reps = [p for p in rep_pairs if re.sub(r"^rep\d+_", "", p[0].stem) in train_base_stems]
    train_pairs = train_base + valid_reps

    print(f"Split completed: Train={len(train_pairs)} (Base={len(train_base)}, Replicas={len(valid_reps)}), Val={len(val_pairs)} (ratio={args.val_ratio:.2f})")

    splits = {"train": train_pairs, "val": val_pairs}
    export_d2 = args.format in ("all", "detectron2")
    export_yolo = args.format in ("all", "yolo")
    export_rfdetr = args.format in ("all", "rfdetr")

    # 目标目录
    d2_root = output_dir / "detectron2"
    yolo_root = output_dir / "yolo"
    rfdetr_root = output_dir / "rfdetr"

    for split_name, split_list in splits.items():
        print(f"\nProcessing {split_name} set ({len(split_list)} samples)...")

        # Detectron2 划分目录
        d2_split_dir = d2_root / split_name
        if export_d2:
            d2_split_dir.mkdir(parents=True, exist_ok=True)

        # YOLO 划分目录
        yolo_img_dir = yolo_root / "images" / split_name
        yolo_lbl_dir = yolo_root / "labels" / split_name
        if export_yolo:
            yolo_img_dir.mkdir(parents=True, exist_ok=True)
            yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

        # RF-DETR 划分目录（RF-DETR 标准使用 'valid' 表示验证集）
        rf_split_name = "valid" if split_name == "val" else split_name
        rf_split_dir = rfdetr_root / rf_split_name
        if export_rfdetr:
            rf_split_dir.mkdir(parents=True, exist_ok=True)

        rf_coco_images = []
        rf_coco_annotations = []
        rf_ann_id = 1

        for idx, (img_path, json_path) in enumerate(split_list, 1):
            with open(json_path, "r", encoding="utf-8") as f:
                labelme_data = json.load(f)

            orig_w = labelme_data.get("imageWidth")
            orig_h = labelme_data.get("imageHeight")

            # 如果需要调整大小或缺失尺寸信息，则加载图像
            img = None
            if target_size or not orig_w or not orig_h:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: unable to read {img_path}, skipping", file=sys.stderr)
                    continue
                orig_h, orig_w = img.shape[:2]

            curr_w = target_size[0] if target_size else orig_w
            curr_h = target_size[1] if target_size else orig_h
            scale_x = curr_w / float(orig_w)
            scale_y = curr_h / float(orig_h)

            out_img_name = f"{img_path.stem}.jpg" if target_size else img_path.name

            # 写入或链接图像的辅助函数
            def write_image(dst_path: Path):
                if target_size:
                    resized = cv2.resize(img, (curr_w, curr_h), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(str(dst_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                elif getattr(args, "copy", False) or getattr(args, "copy_images", False):
                    shutil.copy2(img_path, dst_path)
                else:
                    if dst_path.exists() or dst_path.is_symlink():
                        dst_path.unlink()
                    try:
                        dst_path.symlink_to(img_path)
                    except OSError:
                        shutil.copy2(img_path, dst_path)

            # 导出 Detectron2 格式（LabelMe 格式）
            if export_d2:
                write_image(d2_split_dir / out_img_name)
                # 如果缩放则更新标注多边形
                d2_json = dict(labelme_data)
                d2_json["imagePath"] = out_img_name
                d2_json["imageWidth"] = curr_w
                d2_json["imageHeight"] = curr_h
                if target_size:
                    new_shapes = []
                    for s in d2_json.get("shapes", []):
                        ns = dict(s)
                        ns["points"] = [[p[0] * scale_x, p[1] * scale_y] for p in s["points"]]
                        new_shapes.append(ns)
                    d2_json["shapes"] = new_shapes
                with open(d2_split_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as fp:
                    json.dump(d2_json, fp, indent=2, ensure_ascii=False)

            # 导出 YOLO 格式
            if export_yolo:
                write_image(yolo_img_dir / out_img_name)
                yolo_lines = []
                for s in labelme_data.get("shapes", []):
                    lbl = s.get("label")
                    if lbl not in cat_to_id:
                        continue
                    cid = cat_to_id[lbl]
                    pts = s.get("points", [])
                    if len(pts) < 3:
                        continue
                    coords = []
                    for p in pts:
                        norm_x = min(1.0, max(0.0, (p[0] * scale_x) / curr_w))
                        norm_y = min(1.0, max(0.0, (p[1] * scale_y) / curr_h))
                        coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                    yolo_lines.append(f"{cid} " + " ".join(coords))
                (yolo_lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines) + "\n")

            # 导出 RF-DETR 格式（COCO 格式）
            if export_rfdetr:
                write_image(rf_split_dir / out_img_name)
                rf_coco_images.append({
                    "id": idx,
                    "file_name": out_img_name,
                    "width": curr_w,
                    "height": curr_h,
                })
                for s in labelme_data.get("shapes", []):
                    lbl = s.get("label")
                    if lbl not in cat_to_id:
                        continue
                    cid = cat_to_id[lbl]
                    pts = s.get("points", [])
                    if len(pts) < 3:
                        continue
                    scaled_pts = [[p[0] * scale_x, p[1] * scale_y] for p in pts]
                    xs = [p[0] for p in scaled_pts]
                    ys = [p[1] for p in scaled_pts]
                    x0, x1 = max(0.0, min(xs)), min(float(curr_w), max(xs))
                    y0, y1 = max(0.0, min(ys)), min(float(curr_h), max(ys))
                    bw = max(1.0, x1 - x0)
                    bh = max(1.0, y1 - y0)
                    poly_flat = [coord for pt in scaled_pts for coord in pt]
                    area = polygon_area(poly_flat)
                    if area <= 0:
                        area = bw * bh

                    rf_coco_annotations.append({
                        "id": rf_ann_id,
                        "image_id": idx,
                        "category_id": cid,
                        "bbox": [round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)],
                        "area": round(area, 2),
                        "segmentation": [[round(c, 2) for c in poly_flat]],
                        "iscrowd": 0,
                    })
                    rf_ann_id += 1

        if export_rfdetr:
            rf_coco_data = {
                "images": rf_coco_images,
                "annotations": rf_coco_annotations,
                "categories": [{"id": idx, "name": name, "supercategory": "none"} for idx, name in enumerate(classes)],
            }
            with open(rf_split_dir / "_annotations.coco.json", "w", encoding="utf-8") as fp:
                json.dump(rf_coco_data, fp, indent=2)

    # 写入 YOLO data.yaml
    if export_yolo:
        yolo_yaml = {
            "path": str(yolo_root),
            "train": "images/train",
            "val": "images/val",
            "names": {idx: name for idx, name in enumerate(classes)},
        }
        import yaml
        with open(yolo_root / "data.yaml", "w", encoding="utf-8") as fp:
            yaml.dump(yolo_yaml, fp, sort_keys=False)

    print("\n" + "=" * 60)
    print("Dataset preparation completed successfully!")
    if export_d2:
        print(f"  Detectron2: {d2_root} (train: {len(train_pairs)}, val: {len(val_pairs)})")
    if export_yolo:
        print(f"  YOLO-seg:   {yolo_root} (data.yaml: {yolo_root / 'data.yaml'})")
    if export_rfdetr:
        print(f"  RF-DETR:    {rfdetr_root} (train & valid with _annotations.coco.json)")
    print("=" * 60)


if __name__ == "__main__":
    main()
