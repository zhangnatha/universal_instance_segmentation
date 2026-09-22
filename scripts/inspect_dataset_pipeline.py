#!/usr/bin/env python3
"""数据集处理与可视化质检流水线
读取原始 LabelMe 标注，应用少样本类别过采样与物理遮挡增强，
执行零泄露的训练集/验证集划分，生成带有掩膜/边界框的可视化叠加图像，
并构建交互式 HTML 画廊以便于人工核验。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instance_segmentation.run_logging import setup_run_logging
from scripts.create_balanced_dataset import (
    apply_brightness,
    apply_clahe,
    apply_contrast,
    apply_occlusion_aug,
    apply_random_erasing,
    draw_realistic_pipe,
    rotate_image_and_shapes,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".BMP", ".PNG", ".JPG"}

def hsv_to_rgb(hue: int, saturation: int = 100, value: int = 100) -> tuple[int, int, int]:
    """精确匹配 HSV 转 RGB 颜色转换逻辑。"""
    rgb_max = value * 2.55
    rgb_min = rgb_max * (100 - saturation) / 100.0
    sector = (hue % 360) // 60
    difference = (hue % 360) % 60
    adjustment = (rgb_max - rgb_min) * difference / 60.0

    if sector == 0:
        r, g, b = rgb_max, rgb_min + adjustment, rgb_min
    elif sector == 1:
        r, g, b = rgb_max - adjustment, rgb_max, rgb_min
    elif sector == 2:
        r, g, b = rgb_min, rgb_max, rgb_min + adjustment
    elif sector == 3:
        r, g, b = rgb_min, rgb_max - adjustment, rgb_max
    elif sector == 4:
        r, g, b = rgb_min + adjustment, rgb_min, rgb_max
    else:
        r, g, b = rgb_max, rgb_min, rgb_max - adjustment
    return int(r), int(g), int(b)


def get_color(class_id: int, num_classes: int = 1) -> tuple[int, int, int]:
    """根据类别 ID 与类别总数动态获取对应的显示颜色（OpenCV BGR 格式）。"""
    hue = int(360.0 / max(1, num_classes) * max(0, class_id)) % 360
    r, g, b = hsv_to_rgb(hue, 100, 100)
    return (b, g, r)


def render_labelme_overlay(
    img: np.ndarray,
    shapes: List[Dict[str, Any]],
    classes: List[str],
    alpha: float = 0.40,
) -> np.ndarray:
    """在图像上渲染半透明多边形掩膜叠加层、轮廓、边界框和类别标签。"""
    canvas = img.copy()
    overlay = img.copy()
    h, w = img.shape[:2]

    # 1. 填充多边形掩膜
    for s in shapes:
        lbl = s.get("label", "")
        cid = classes.index(lbl) if lbl in classes else 0
        c = get_color(cid, len(classes))
        pts = np.array(s.get("points", []), dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], c)

    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # 2. 绘制轮廓、边界框和文本标题
    for s in shapes:
        lbl = s.get("label", "")
        cid = classes.index(lbl) if lbl in classes else 0
        c = get_color(cid, len(classes))
        pts = np.array(s.get("points", []), dtype=np.int32)
        if len(pts) >= 3:
            cv2.drawContours(canvas, [pts], -1, c, 2, cv2.LINE_AA if hasattr(cv2, "LINE_AA") else 16)
            xmin, ymin = pts.min(axis=0)
            xmax, ymax = pts.max(axis=0)
            cv2.rectangle(canvas, (int(xmin), int(ymin)), (int(xmax), int(ymax)), c, 1)

            # 标签文本
            text = lbl
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            t_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            tx = max(0, int(xmin))
            ty = max(t_size[1] + 4, int(ymin) - 4)
            cv2.rectangle(
                canvas,
                (tx, ty - t_size[1] - 4),
                (tx + t_size[0] + 6, ty + 2),
                c,
                cv2.FILLED,
            )
            cv2.putText(
                canvas,
                text,
                (tx + 3, ty - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA if hasattr(cv2, "LINE_AA") else 16,
            )

    return canvas


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Dataset Processing & Visual Inspection Pipeline"
    )
    parser.add_argument("--source-dir", required=True, help="Raw LabelMe directory containing images and JSON files")
    parser.add_argument("--output-dir", required=True, help="Output destination folder for inspection package")
    parser.add_argument("--classes", nargs="+", required=True, help="Deterministic list of class names")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    parser.add_argument("--oversample-classes", nargs="+", default=[], help="Classes to oversample")
    parser.add_argument("--oversample-ratio", type=int, default=3, help="Oversample ratio (default: 3)")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--occlusion-aug", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable realistic occlusion augmentation")
        parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable augmentations on oversampled replicas (default: True)")
        parser.add_argument("--augment-rotation", action=argparse.BooleanOptionalAction, default=False, help="Enable random small-angle rotation on replicas")
        parser.add_argument("--augment-brightness", action=argparse.BooleanOptionalAction, default=False, help="Enable random brightness adjustment")
        parser.add_argument("--augment-contrast", action=argparse.BooleanOptionalAction, default=False, help="Enable random contrast adjustment")
        parser.add_argument("--augment-clahe", action=argparse.BooleanOptionalAction, default=False, help="Enable CLAHE local contrast enhancement")
        parser.add_argument("--augment-erasing", action=argparse.BooleanOptionalAction, default=False, help="Enable random patch erasing")
    else:
        parser.add_argument("--occlusion-aug", action="store_true", default=True, help="Enable realistic occlusion augmentation")
        parser.add_argument("--no-occlusion-aug", dest="occlusion_aug", action="store_false", help="Disable realistic occlusion augmentation")
        parser.add_argument("--augment", action="store_true", default=True, help="Enable augmentations on oversampled replicas")
        parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentations")
        parser.add_argument("--augment-rotation", action="store_true", default=False, help="Enable rotation")
        parser.add_argument("--no-augment-rotation", dest="augment_rotation", action="store_false", help="Disable rotation")
        parser.add_argument("--augment-brightness", action="store_true", default=False, help="Enable brightness adjustment")
        parser.add_argument("--no-augment-brightness", dest="augment_brightness", action="store_false", help="Disable brightness adjustment")
        parser.add_argument("--augment-contrast", action="store_true", default=False, help="Enable contrast adjustment")
        parser.add_argument("--no-augment-contrast", dest="augment_contrast", action="store_false", help="Disable contrast adjustment")
        parser.add_argument("--augment-clahe", action="store_true", default=False, help="Enable CLAHE")
        parser.add_argument("--no-augment-clahe", dest="augment_clahe", action="store_false", help="Disable CLAHE")
        parser.add_argument("--augment-erasing", action="store_true", default=False, help="Enable erasing")
        parser.add_argument("--no-augment-erasing", dest="augment_erasing", action="store_false", help="Disable erasing")

    parser.add_argument("--rotation-range", nargs=2, type=float, default=[-5.0, 5.0], help="Rotation angle range in degrees (default: -5.0 5.0)")
    parser.add_argument("--rotation-prob", type=float, default=0.5, help="Probability of applying rotation (default: 0.5)")
    parser.add_argument("--brightness-range", nargs=2, type=float, default=[0.85, 1.15], help="Brightness factor range (default: 0.85 1.15)")
    parser.add_argument("--brightness-prob", type=float, default=0.5, help="Probability of applying brightness adjustment (default: 0.5)")
    parser.add_argument("--contrast-range", nargs=2, type=float, default=[0.85, 1.15], help="Contrast factor range (default: 0.85 1.15)")
    parser.add_argument("--contrast-prob", type=float, default=0.5, help="Probability of applying contrast adjustment (default: 0.5)")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--clahe-prob", type=float, default=0.25, help="Probability of applying CLAHE (default: 0.25)")
    parser.add_argument("--erasing-prob", type=float, default=0.25, help="Probability of applying random erasing (default: 0.25)")
    parser.add_argument("--erasing-smin", type=float, default=0.01, help="Min area fraction for erasing (default: 0.01)")
    parser.add_argument("--erasing-smax", type=float, default=0.04, help="Max area fraction for erasing (default: 0.04)")

    parser.add_argument("--vis-samples", "--num-previews", dest="vis_samples", type=int, default=40, help="Max number of sample overlays per split (default: 40)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args(argv)
    args.num_previews = args.vis_samples
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    setup_run_logging("INSPECT_DATASET")

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_dir = Path(args.source_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    classes = list(args.classes)

    if not src_dir.is_dir():
        print(f"Error: Source directory {src_dir} not found!", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 输出目录结构
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    vis_dir = out_dir / "preview_vis"
    vis_aug_dir = vis_dir / "augmented_samples"
    vis_train_dir = vis_dir / "train_clean_samples"
    vis_val_dir = vis_dir / "val_samples"

    for d in [train_dir, val_dir, vis_aug_dir, vis_train_dir, vis_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 发现原始样本
    json_files = sorted(src_dir.glob("*.json"))
    raw_samples = []
    raw_class_counts = Counter()

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                d = json.load(f)
            labels = [s.get("label") for s in d.get("shapes", []) if s.get("label")]
            for l in labels:
                raw_class_counts[l] += 1
            img_p = None
            img_name = d.get("imagePath", "")
            if img_name and (src_dir / img_name).is_file():
                img_p = src_dir / img_name
            else:
                for ext in IMG_EXTS:
                    cand = src_dir / f"{jf.stem}{ext}"
                    if cand.is_file():
                        img_p = cand
                        break
            if img_p:
                raw_samples.append((jf, img_p, d, labels))
        except Exception:
            continue

    print(f"Discovered {len(raw_samples)} raw samples in {src_dir}")
    print(f"Raw class instance distribution: {dict(raw_class_counts)}")

    # 2. 训练集/验证集划分（零泄露，先划分原始样本）
    shuffled = list(raw_samples)
    random.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * args.val_ratio))
    val_samples = shuffled[:val_count]
    train_base_samples = shuffled[val_count:]

    print(f"Split completed: Train Base={len(train_base_samples)}, Val={len(val_samples)}")

    # 若未指定过采样类别，则自动选择少样本类别
    oversample_targets = set(args.oversample_classes)
    if not oversample_targets and raw_class_counts:
        median_cnt = sorted(raw_class_counts.values())[len(raw_class_counts) // 2]
        oversample_targets = {cls for cls, cnt in raw_class_counts.items() if cnt < median_cnt * 0.5}

    active_augs = []
    if args.occlusion_aug:
        active_augs.append("Occlusion(Pipe/Shadow)")
    if args.augment:
        if args.augment_rotation:
            active_augs.append(f"Rotation({args.rotation_range[0]}°~{args.rotation_range[1]}°, p={args.rotation_prob})")
        if args.augment_brightness:
            active_augs.append(f"Brightness({args.brightness_range}, p={args.brightness_prob})")
        if args.augment_contrast:
            active_augs.append(f"Contrast({args.contrast_range}, p={args.contrast_prob})")
        if args.augment_clahe:
            active_augs.append(f"CLAHE(clip={args.clahe_clip_limit}, p={args.clahe_prob})")
        if args.augment_erasing:
            active_augs.append(f"Erasing(p={args.erasing_prob})")
    aug_desc = ", ".join(active_augs) if active_augs else "None"
    print(f"Oversample target classes: {list(oversample_targets)} (Ratio={args.oversample_ratio}x, ActiveAugs=[{aug_desc}])")

    # 3. 处理验证集（100% 纯净拷贝）
    val_class_counts = Counter()
    for jf, img_p, d, labels in val_samples:
        dst_img = val_dir / img_p.name
        dst_json = val_dir / jf.name
        shutil.copy2(img_p, dst_img)
        with open(dst_json, "w", encoding="utf-8") as fp:
            json.dump(d, fp, indent=2, ensure_ascii=False)
        for l in labels:
            val_class_counts[l] += 1

    # 4. 处理训练集（基础拷贝 + 增强副本）
    train_class_counts = Counter()
    augmented_records = []  # 用于存储以便可视化的记录
    total_train_images = 0

    for jf, img_p, d, labels in train_base_samples:
        # 拷贝基础样本
        dst_img = train_dir / img_p.name
        dst_json = train_dir / jf.name
        shutil.copy2(img_p, dst_img)
        with open(dst_json, "w", encoding="utf-8") as fp:
            json.dump(d, fp, indent=2, ensure_ascii=False)
        total_train_images += 1
        for l in labels:
            train_class_counts[l] += 1

        # 检查过采样
        if any(l in oversample_targets for l in labels):
            raw_img = cv2.imread(str(img_p))
            if raw_img is None:
                continue

            for rep in range(1, max(1, args.oversample_ratio)):
                stem = f"rep{rep}_{jf.stem}"
                rep_img_path = train_dir / f"{stem}{img_p.suffix}"
                rep_json_path = train_dir / f"{stem}.json"

                aug_img = raw_img.copy()
                shapes = [dict(s) for s in d.get("shapes", [])]

                if args.augment:
                    # 1. 物理小角度旋转（同时转换图像和多边形顶点）
                    if args.augment_rotation and (args.rotation_prob >= 1.0 or random.random() < args.rotation_prob):
                        ang = random.uniform(args.rotation_range[0], args.rotation_range[1])
                        aug_img, shapes = rotate_image_and_shapes(aug_img, shapes, ang)

                    # 2. 亮度调整
                    if args.augment_brightness and (args.brightness_prob >= 1.0 or random.random() < args.brightness_prob):
                        aug_img = apply_brightness(aug_img, args.brightness_range[0], args.brightness_range[1])

                    # 3. 对比度调整
                    if args.augment_contrast and (args.contrast_prob >= 1.0 or random.random() < args.contrast_prob):
                        aug_img = apply_contrast(aug_img, args.contrast_range[0], args.contrast_range[1])

                    # 4. CLAHE 局部对比度增强
                    if args.augment_clahe and (args.clahe_prob >= 1.0 or random.random() < args.clahe_prob):
                        aug_img = apply_clahe(aug_img, args.clahe_clip_limit)

                    # 5. 随机矩形擦除
                    if args.augment_erasing and (args.erasing_prob >= 1.0 or random.random() < args.erasing_prob):
                        aug_img = apply_random_erasing(aug_img, args.erasing_smin, args.erasing_smax)

                # 6. 物理遮挡（金属管道或切块阴影）
                if args.occlusion_aug:
                    aug_img = apply_occlusion_aug(aug_img, shapes, oversample_targets, rep_idx=rep)

                cv2.imwrite(str(rep_img_path), aug_img)

                d_copy = dict(d)
                d_copy["imagePath"] = rep_img_path.name
                d_copy["shapes"] = shapes
                with open(rep_json_path, "w", encoding="utf-8") as fp:
                    json.dump(d_copy, fp, indent=2, ensure_ascii=False)

                total_train_images += 1
                for l in labels:
                    train_class_counts[l] += 1

                augmented_records.append((rep_img_path, d_copy, img_p.name, rep))

    print(f"Training set prepared: {total_train_images} images (including {len(augmented_records)} augmented replicas)")
    print(f"Enhanced Train class distribution: {dict(train_class_counts)}")
    print(f"Clean Val class distribution: {dict(val_class_counts)}")

    # 5. 生成可视化叠加图
    print("\nRendering visual inspection overlays...")

    # 5.1 渲染增强样本（全部或最多 num_previews 个）
    vis_aug_items = []
    for rep_img_path, d_data, base_name, rep_num in augmented_records[:args.num_previews * 2]:
        img = cv2.imread(str(rep_img_path))
        if img is not None:
            vis = render_labelme_overlay(img, d_data.get("shapes", []), classes)
            out_vis_path = vis_aug_dir / f"{rep_img_path.stem}_vis.jpg"
            cv2.imwrite(str(out_vis_path), vis)
            vis_aug_items.append({
                "vis_name": out_vis_path.name,
                "base_name": base_name,
                "rep_num": rep_num,
                "labels": [s.get("label") for s in d_data.get("shapes", [])],
            })

    # 5.2 渲染训练集纯净样本（最多 num_previews 个）
    vis_train_items = []
    for jf, img_p, d_data, labels in train_base_samples[:args.num_previews]:
        img = cv2.imread(str(img_p))
        if img is not None:
            vis = render_labelme_overlay(img, d_data.get("shapes", []), classes)
            out_vis_path = vis_train_dir / f"{img_p.stem}_clean_vis.jpg"
            cv2.imwrite(str(out_vis_path), vis)
            vis_train_items.append({
                "vis_name": out_vis_path.name,
                "labels": labels,
            })

    # 5.3 渲染验证集样本（最多 num_previews 个）
    vis_val_items = []
    for jf, img_p, d_data, labels in val_samples[:args.num_previews]:
        img = cv2.imread(str(img_p))
        if img is not None:
            vis = render_labelme_overlay(img, d_data.get("shapes", []), classes)
            out_vis_path = vis_val_dir / f"{img_p.stem}_val_vis.jpg"
            cv2.imwrite(str(out_vis_path), vis)
            vis_val_items.append({
                "vis_name": out_vis_path.name,
                "labels": labels,
            })

    # 6. 为 3 个后端导出标准格式到 exports/ 目录
    print("\nExporting standard training formats for Detectron2, YOLO, and RF-DETR...")
    exports_dir = out_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    # 为避免重新划分，使用已建立的划分直接导出
    export_backends(train_dir, val_dir, exports_dir, classes)

    # 7. 保存数据集统计摘要 JSON
    summary_data = {
        "source_dir": str(src_dir),
        "classes": classes,
        "raw_images_count": len(raw_samples),
        "raw_class_instances": dict(raw_class_counts),
        "train_images_count": total_train_images,
        "train_base_images": len(train_base_samples),
        "train_augmented_images": len(augmented_records),
        "train_class_instances": dict(train_class_counts),
        "val_images_count": len(val_samples),
        "val_class_instances": dict(val_class_counts),
        "oversample_classes": list(oversample_targets),
        "oversample_ratio": args.oversample_ratio,
        "occlusion_aug_enabled": args.occlusion_aug,
        "augmentations": {
            "augment_enabled": args.augment,
            "occlusion_aug": args.occlusion_aug,
            "rotation": {"enabled": args.augment_rotation, "range": args.rotation_range, "prob": args.rotation_prob},
            "brightness": {"enabled": args.augment_brightness, "range": args.brightness_range, "prob": args.brightness_prob},
            "contrast": {"enabled": args.augment_contrast, "range": args.contrast_range, "prob": args.contrast_prob},
            "clahe": {"enabled": args.augment_clahe, "clip_limit": args.clahe_clip_limit, "prob": args.clahe_prob},
            "erasing": {"enabled": args.augment_erasing, "smin": args.erasing_smin, "smax": args.erasing_smax, "prob": args.erasing_prob},
        },
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary_data, fp, indent=2, ensure_ascii=False)

    # 8. 生成交互式 HTML 画廊
    generate_html_gallery(
        out_dir / "index.html",
        summary_data,
        vis_aug_items,
        vis_train_items,
        vis_val_items,
    )

    print("\n" + "=" * 60)
    print(f"Inspection Dataset successfully generated at: {out_dir}")
    print(f"  ├── train/                  ({total_train_images} images with LabelMe JSONs)")
    print(f"  ├── val/                    ({len(val_samples)} 100% clean images with LabelMe JSONs)")
    print(f"  ├── preview_vis/            (Visual overlays with semi-transparent masks & boxes)")
    print(f"  │   ├── augmented_samples/  ({len(vis_aug_items)} simulated pipe/cutout rendered overlays)")
    print(f"  │   ├── train_clean_samples/({len(vis_train_items)} clean train overlays)")
    print(f"  │   └── val_samples/        ({len(vis_val_items)} validation overlays)")
    print(f"  ├── exports/                (Direct-to-train formats)")
    print(f"  │   ├── rfdetr/             (train & valid + _annotations.coco.json)")
    print(f"  │   ├── yolo/               (images/ + labels/ + data.yaml)")
    print(f"  │   └── detectron2/         (train/ + val/)")
    print(f"  ├── dataset_summary.json    (Comprehensive statistics)")
    print(f"  └── index.html              (Interactive Web Visual Gallery - Open in Browser!)")
    print("=" * 60)


def export_backends(train_dir: Path, val_dir: Path, exports_dir: Path, classes: List[str]):
    """从划分后的 LabelMe 目录导出适用于 Detectron2、YOLO 和 RF-DETR 的标准训练格式。"""
    cat_to_id = {c: i for i, c in enumerate(classes)}

    d2_root = exports_dir / "detectron2"
    yolo_root = exports_dir / "yolo"
    rf_root = exports_dir / "rfdetr"

    for d in [d2_root / "train", d2_root / "val", yolo_root / "images/train", yolo_root / "images/val",
             yolo_root / "labels/train", yolo_root / "labels/val", rf_root / "train", rf_root / "valid"]:
        d.mkdir(parents=True, exist_ok=True)

    # 导出 YOLO data.yaml
    yolo_yaml_path = yolo_root / "data.yaml"
    with open(yolo_yaml_path, "w", encoding="utf-8") as fp:
        fp.write(f"path: {yolo_root.resolve()}\n")
        fp.write("train: images/train\nval: images/val\n\nnames:\n")
        for idx, name in enumerate(classes):
            fp.write(f"  {idx}: {name}\n")

    for split, split_dir, rf_split_name in [("train", train_dir, "train"), ("val", val_dir, "valid")]:
        rf_split_dir = rf_root / rf_split_name
        yolo_img_dir = yolo_root / "images" / split
        yolo_lbl_dir = yolo_root / "labels" / split
        d2_split_dir = d2_root / split

        rf_coco_images = []
        rf_coco_annotations = []
        ann_id = 1

        json_files = sorted(split_dir.glob("*.json"))
        for img_id, jf in enumerate(json_files, 1):
            with open(jf, "r", encoding="utf-8") as fp:
                d = json.load(fp)

            img_p = None
            img_name = d.get("imagePath", "")
            if img_name and (split_dir / img_name).is_file():
                img_p = split_dir / img_name
            else:
                for ext in IMG_EXTS:
                    cand = split_dir / f"{jf.stem}{ext}"
                    if cand.is_file():
                        img_p = cand
                        break
            if not img_p:
                continue

            # 拷贝到 Detectron2
            if not (d2_split_dir / img_p.name).exists():
                (d2_split_dir / img_p.name).symlink_to(img_p)
            if not (d2_split_dir / jf.name).exists():
                shutil.copy2(jf, d2_split_dir / jf.name)

            # 拷贝到 RF-DETR
            if not (rf_split_dir / img_p.name).exists():
                (rf_split_dir / img_p.name).symlink_to(img_p)

            # 拷贝到 YOLO
            if not (yolo_img_dir / img_p.name).exists():
                (yolo_img_dir / img_p.name).symlink_to(img_p)

            w = d.get("imageWidth")
            h = d.get("imageHeight")
            if not w or not h:
                im = cv2.imread(str(img_p))
                if im is not None:
                    h, w = im.shape[:2]

            rf_coco_images.append({
                "id": img_id,
                "file_name": img_p.name,
                "width": int(w),
                "height": int(h),
            })

            yolo_lines = []
            for s in d.get("shapes", []):
                lbl = s.get("label")
                if lbl not in cat_to_id:
                    continue
                cid = cat_to_id[lbl]
                pts = s.get("points", [])
                if len(pts) < 3:
                    continue

                # COCO 标注
                pts_arr = np.array(pts, dtype=np.float32)
                xmin, ymin = pts_arr.min(axis=0)
                xmax, ymax = pts_arr.max(axis=0)
                bw, bh = float(xmax - xmin), float(ymax - ymin)
                flat_poly = [float(coord) for pt in pts for coord in pt]
                area = 0.5 * abs(sum(flat_poly[i] * flat_poly[i + 3] - flat_poly[i + 2] * flat_poly[i + 1] for i in range(0, len(flat_poly) - 2, 2)))

                rf_coco_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "segmentation": [flat_poly],
                    "area": float(area),
                    "bbox": [float(xmin), float(ymin), bw, bh],
                    "iscrowd": 0,
                })
                ann_id += 1

                # YOLO 标注行
                normalized = []
                for px, py in pts:
                    normalized.extend([f"{px / w:.6f}", f"{py / h:.6f}"])
                yolo_lines.append(f"{cid} " + " ".join(normalized))

            with open(yolo_lbl_dir / f"{img_p.stem}.txt", "w", encoding="utf-8") as fp:
                fp.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

        # 写入 COCO json
        coco_dict = {
            "images": rf_coco_images,
            "annotations": rf_coco_annotations,
            "categories": [{"id": cid, "name": c, "supercategory": "none"} for c, cid in cat_to_id.items()],
        }
        with open(rf_split_dir / "_annotations.coco.json", "w", encoding="utf-8") as fp:
            json.dump(coco_dict, fp)


def generate_html_gallery(
    html_path: Path,
    summary: Dict[str, Any],
    aug_items: List[Dict[str, Any]],
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
):
    """生成包含统计数据和样本可视化卡片的交互式 HTML 质检画廊。"""
    classes = summary['classes']
    raw_counts = summary['raw_class_instances']
    train_counts = summary['train_class_instances']
    val_counts = summary['val_class_instances']

    table_rows = []
    for c in classes:
        raw_c = raw_counts.get(c, 0)
        tr_c = train_counts.get(c, 0)
        val_c = val_counts.get(c, 0)
        diff = tr_c - int(raw_c * 0.9)
        diff_str = f"(+{diff} augmented)" if diff > 0 else ""
        row = f"<tr><td><strong>{html.escape(c)}</strong></td><td>{raw_c}</td><td><span class='badge-train'>{tr_c}</span> <small style='color:#00ff80;'>{diff_str}</small></td><td><span class='badge-val'>{val_c}</span></td></tr>"
        table_rows.append(row)

    aug_cards = []
    for item in aug_items:
        img_rel = f"preview_vis/augmented_samples/{item['vis_name']}"
        labels_html = " ".join([f"<span class='tag'>{html.escape(l)}</span>" for l in item['labels']])
        card = f"<div class='card'><div class='card-img-wrap'><img src='{img_rel}' loading='lazy' alt='{item['vis_name']}' onclick='openModal(this.src)' /></div><div class='card-info'><div class='card-title'>🛡️ Augmented Replica #{item['rep_num']} (from {html.escape(item['base_name'])})</div><div class='card-tags'>{labels_html}</div></div></div>"
        aug_cards.append(card)

    train_cards = []
    for item in train_items:
        img_rel = f"preview_vis/train_clean_samples/{item['vis_name']}"
        labels_html = " ".join([f"<span class='tag'>{html.escape(l)}</span>" for l in item['labels']])
        card = f"<div class='card'><div class='card-img-wrap'><img src='{img_rel}' loading='lazy' alt='{item['vis_name']}' onclick='openModal(this.src)' /></div><div class='card-info'><div class='card-title'>{html.escape(item['vis_name'])}</div><div class='card-tags'>{labels_html}</div></div></div>"
        train_cards.append(card)

    val_cards = []
    for item in val_items:
        img_rel = f"preview_vis/val_samples/{item['vis_name']}"
        labels_html = " ".join([f"<span class='tag tag-val'>{html.escape(l)}</span>" for l in item['labels']])
        card = f"<div class='card'><div class='card-img-wrap'><img src='{img_rel}' loading='lazy' alt='{item['vis_name']}' onclick='openModal(this.src)' /></div><div class='card-info'><div class='card-title'>{html.escape(item['vis_name'])}</div><div class='card-tags'>{labels_html}</div></div></div>"
        val_cards.append(card)

    val_pct = f"{summary['val_images_count'] / summary['raw_images_count'] * 100:.1f}" if summary['raw_images_count'] > 0 else "0"

    parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <title>Dataset Split & Augmentation Visual Inspection Dashboard</title>',
        '    <style>',
        '        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #121418; color: #e2e8f0; margin: 0; padding: 24px; }',
        '        .header { max-width: 1300px; margin: 0 auto 24px auto; border-bottom: 1px solid #334155; padding-bottom: 16px; }',
        '        h1 { margin: 0 0 8px 0; color: #ffffff; font-size: 26px; }',
        '        .subtitle { color: #94a3b8; font-size: 14px; }',
        '        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; max-width: 1300px; margin: 0 auto 24px auto; }',
        '        .stat-card { background: #1e2229; padding: 18px; border-radius: 10px; border: 1px solid #334155; }',
        '        .stat-val { font-size: 28px; font-weight: bold; color: #00d084; margin: 6px 0; }',
        '        .stat-lbl { color: #94a3b8; font-size: 13px; }',
        '        .table-wrap { max-width: 1300px; margin: 0 auto 32px auto; background: #1e2229; border-radius: 10px; border: 1px solid #334155; overflow: hidden; }',
        '        table { width: 100%; border-collapse: collapse; text-align: left; }',
        '        th, td { padding: 12px 18px; border-bottom: 1px solid #334155; font-size: 14px; }',
        '        th { background: #181b21; color: #94a3b8; font-weight: 600; }',
        '        .badge-train { background: rgba(0, 208, 132, 0.15); color: #00d084; padding: 3px 8px; border-radius: 4px; font-weight: bold; }',
        '        .badge-val { background: rgba(0, 180, 216, 0.15); color: #00b4d8; padding: 3px 8px; border-radius: 4px; font-weight: bold; }',
        '        .section-wrap { max-width: 1300px; margin: 0 auto 32px auto; }',
        '        .section-header { margin-bottom: 16px; }',
        '        .section-title { font-size: 20px; font-weight: bold; color: #ffffff; }',
        '        .section-desc { color: #94a3b8; font-size: 13px; margin-top: 4px; }',
        '        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }',
        '        .card { background: #1e2229; border: 1px solid #334155; border-radius: 8px; overflow: hidden; transition: transform 0.2s ease; }',
        '        .card:hover { transform: translateY(-3px); border-color: #00d084; }',
        '        .card-img-wrap { width: 100%; height: 200px; background: #000; overflow: hidden; cursor: pointer; }',
        '        .card-img-wrap img { width: 100%; height: 100%; object-fit: cover; }',
        '        .card-info { padding: 12px; }',
        '        .card-title { font-size: 13px; font-weight: 600; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }',
        '        .card-tags { display: flex; flex-wrap: wrap; gap: 4px; }',
        '        .tag { font-size: 11px; background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 3px; color: #94a3b8; }',
        '        .tag-val { color: #00b4d8; }',
        '        #modal { display: none; position: fixed; z-index: 9999; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.85); align-items: center; justify-content: center; cursor: zoom-out; }',
        '        #modal img { max-width: 92vw; max-height: 92vh; border-radius: 6px; }',
        '    </style>',
        '</head>',
        '<body>',
        '    <div class="header">',
        '        <h1>📊 Dataset Split & Augmentation Visual Inspection Dashboard</h1>',
        f'        <div class="subtitle">Source Path: {html.escape(summary["source_dir"])}</div>',
        '    </div>',
        '    <div class="stats-grid">',
        f'        <div class="stat-card"><div class="stat-lbl">Raw Annotated Images</div><div class="stat-val">{summary["raw_images_count"]}</div><div class="stat-lbl">Clean sample count before augmentation</div></div>',
        f'        <div class="stat-card"><div class="stat-lbl">Total Processed Train Images</div><div class="stat-val" style="color:#00d084;">{summary["train_images_count"]}</div><div class="stat-lbl">{summary["train_base_images"]} base + {summary["train_augmented_images"]} occlusion/shadow augmented replicas</div></div>',
        f'        <div class="stat-card"><div class="stat-lbl">Validation Images (100% Zero-Leakage)</div><div class="stat-val" style="color:#00b4d8;">{summary["val_images_count"]}</div><div class="stat-lbl">Ratio {val_pct}%, clean raw images without synthetic modifications</div></div>',
        f'        <div class="stat-card"><div class="stat-lbl">Oversampled Classes & Multiplier</div><div class="stat-val" style="color:#f59e0b; font-size:22px;">{", ".join(summary["oversample_classes"])} ({summary["oversample_ratio"]}x)</div><div class="stat-lbl">Horizontal metallic pipe and shadow occlusions applied</div></div>',
        '    </div>',
        '    <div class="table-wrap">',
        '        <table>',
        '            <thead><tr><th>Target Class</th><th>Raw Total Instances</th><th>Train Final Instances</th><th>Val Final Instances</th></tr></thead>',
        f'            <tbody>{"".join(table_rows)}</tbody>',
        '        </table>',
        '    </div>',
        '    <div class="section-wrap">',
        '        <div class="section-header"><div class="section-title">🛡️ Physical Occlusion & Augmentation Samples (Augmented Samples)</div><div class="section-desc">Programmatic horizontal metallic pipe occlusions and shadows with specular highlights. Mask overlays rendered to verify annotation integrity under strong occlusion.</div></div>',
        f'        <div class="grid">{"".join(aug_cards)}</div>',
        '    </div>',
        '    <div class="section-wrap">',
        '        <div class="section-header"><div class="section-title">🌲 Clean Train Samples (Train Clean Samples)</div><div class="section-desc">Clean training set standard samples with rendered annotation overlays.</div></div>',
        f'        <div class="grid">{"".join(train_cards)}</div>',
        '    </div>',
        '    <div class="section-wrap">',
        '        <div class="section-header"><div class="section-title">✅ Clean Validation Samples (Validation Clean Samples)</div><div class="section-desc">100% clean raw validation set, isolated to ensure unbiased evaluation.</div></div>',
        f'        <div class="grid">{"".join(val_cards)}</div>',
        '    </div>',
        '    <div id="modal" onclick="closeModal()"><img id="modal-img" src="" /></div>',
        '    <script>',
        '        function openModal(src) { document.getElementById("modal-img").src = src; document.getElementById("modal").style.display = "flex"; }',
        '        function closeModal() { document.getElementById("modal").style.display = "none"; }',
        '    </script>',
        '</body>',
        '</html>'
    ]

    with open(html_path, "w", encoding="utf-8") as fp:
        fp.write(chr(10).join(parts))


if __name__ == '__main__':
    main()
