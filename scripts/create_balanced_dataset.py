#!/usr/bin/env python3
"""通过对少样本类别进行过采样来创建均衡的训练数据集。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from instance_segmentation.run_logging import setup_run_logging

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".BMP", ".PNG", ".JPG"}


def draw_realistic_pipe(img: np.ndarray, y_center: int, thickness: int = 26) -> np.ndarray:
    """在图像上绘制逼真的金属水平管道遮挡。"""
    h, w = img.shape[:2]
    pipe_strip = np.zeros((thickness, w, 3), dtype=np.uint8)
    for i in range(thickness):
        rel = i / float(thickness)
        if rel < 0.3:
            val = int(110 + 80 * (rel / 0.3))
        elif rel < 0.5:
            val = int(190 + 50 * ((0.5 - rel) / 0.2))
        else:
            val = int(190 - 120 * ((rel - 0.5) / 0.5))
        pipe_strip[i, :] = (val, val, val)
    y1 = max(0, y_center - thickness // 2)
    y2 = min(h, y1 + thickness)
    strip_h = y2 - y1
    if strip_h > 0:
        alpha = random.uniform(0.92, 0.98)
        img[y1:y2, :] = cv2.addWeighted(pipe_strip[:strip_h, :], alpha, img[y1:y2, :], 1 - alpha, 0)
    return img


def apply_occlusion_aug(img: np.ndarray, shapes: list, target_classes: set, rep_idx: int = 1) -> np.ndarray:
    """针对特定少样本目标应用逼真的遮挡（管道或切块/阴影）。"""
    img_aug = img.copy()
    target_boxes = []
    for s in shapes:
        if s.get("label") in target_classes:
            pts = np.array(s.get("points", []))
            if len(pts) >= 3:
                xmin, ymin = pts.min(axis=0)
                xmax, ymax = pts.max(axis=0)
                target_boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
    if not target_boxes:
        return img_aug

    # 奇数次重复：模拟水平管道遮挡；偶数次重复：切块/阴影
    mode = "pipe" if (rep_idx % 2 == 1) else "cutout_shadow"

    if mode == "pipe":
        target_box = random.choice(target_boxes)
        xmin, ymin, xmax, ymax = target_box
        box_h = ymax - ymin
        y_center = int(ymin + random.uniform(0.15, 0.45) * box_h)
        thickness = int(random.uniform(20, 34))
        img_aug = draw_realistic_pipe(img_aug, y_center, thickness)
    else:
        for box in target_boxes:
            xmin, ymin, xmax, ymax = [int(v) for v in box]
            bw, bh = xmax - xmin, ymax - ymin
            cx1 = max(0, xmin + int(random.uniform(-0.1, 0.2) * bw))
            cy1 = max(0, ymin + int(random.uniform(0.1, 0.4) * bh))
            cx2 = min(img_aug.shape[1], cx1 + int(random.uniform(0.8, 1.2) * bw))
            cy2 = min(img_aug.shape[0], cy1 + int(random.uniform(0.2, 0.5) * bh))
            gray_val = random.randint(40, 90)
            img_aug[cy1:cy2, cx1:cx2] = gray_val
            sy1 = max(0, ymin)
            sy2 = min(img_aug.shape[0], ymax)
            sx1 = max(0, xmin - int(0.2 * bw))
            sx2 = min(img_aug.shape[1], xmax + int(0.2 * bw))
            factor = random.uniform(0.5, 0.75)
            img_aug[sy1:sy2, sx1:sx2] = (img_aug[sy1:sy2, sx1:sx2] * factor).astype(np.uint8)
    return img_aug


def rotate_image_and_shapes(
    img: np.ndarray,
    shapes: list[dict],
    angle: float,
) -> tuple[np.ndarray, list[dict]]:
    """围绕中心将图像及其 LabelMe 多边形标注旋转指定角度（度）。"""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    new_shapes = []
    for s in shapes:
        s_new = dict(s)
        pts = np.array(s.get("points", []), dtype=np.float32)
        if len(pts) > 0:
            pts_hom = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
            rot_pts = pts_hom @ matrix.T
            rot_pts[:, 0] = np.clip(rot_pts[:, 0], 0, w - 1)
            rot_pts[:, 1] = np.clip(rot_pts[:, 1], 0, h - 1)
            s_new["points"] = rot_pts.tolist()
        new_shapes.append(s_new)

    return rotated_img, new_shapes


def apply_brightness(img: np.ndarray, min_f: float = 0.85, max_f: float = 1.15) -> np.ndarray:
    """随机调整图像亮度。"""
    factor = random.uniform(min_f, max_f)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_contrast(img: np.ndarray, min_f: float = 0.85, max_f: float = 1.15) -> np.ndarray:
    """随机调整图像对比度。"""
    factor = random.uniform(min_f, max_f)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """在 LAB 颜色空间中应用 CLAHE 局部对比度增强。"""
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def apply_random_erasing(
    img: np.ndarray,
    s_min: float = 0.01,
    s_max: float = 0.04,
    r_min: float = 0.3,
    r_max: float = 3.3,
) -> np.ndarray:
    """随机擦除矩形区域以模拟局部遮挡。"""
    h, w = img.shape[:2]
    area = h * w
    target_area = float(random.uniform(s_min, s_max) * area)
    aspect_ratio = float(random.uniform(r_min, r_max))
    eh = int(round(np.sqrt(target_area * aspect_ratio)))
    ew = int(round(np.sqrt(target_area / aspect_ratio)))
    if 0 < eh < h and 0 < ew < w:
        y1 = int(random.randint(0, h - eh))
        x1 = int(random.randint(0, w - ew))
        img = img.copy()
        color = np.random.randint(30, 80, (3,), dtype=img.dtype)
        img[y1 : y1 + eh, x1 : x1 + ew] = color
    return img


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Create balanced training dataset via oversampling")
    parser.add_argument("--source", "--source-dir", dest="source", required=True, help="Source directory containing LabelMe JSON and images")
    parser.add_argument("--output", "--output-dir", dest="output", required=True, help="Destination directory for balanced dataset")
    parser.add_argument("--oversample-classes", nargs="+", default=[], help="Classes to oversample")
    parser.add_argument("--oversample-ratio", type=int, default=2, help="Multiplication factor for oversampled classes (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--occlusion-aug", action="store_true", help="Apply realistic pipe/cutout occlusion to oversampled minority images")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    setup_run_logging("BALANCE_DATASET")

    random.seed(args.seed)
    src_dir = Path(args.source).expanduser().resolve()
    dst_dir = Path(args.output).expanduser().resolve()

    if not src_dir.is_dir():
        print(f"Error: Source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(src_dir.glob("*.json"))
    samples = []
    class_counts = Counter()

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                d = json.load(f)
            labels = [s.get("label") for s in d.get("shapes", []) if s.get("label")]
            for l in labels:
                class_counts[l] += 1
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
                samples.append((jf, img_p, d, set(labels)))
        except Exception:
            continue

    print(f"Discovered {len(samples)} valid samples in {src_dir}.")
    print(f"Class distribution: {dict(class_counts)}")

    oversample_map = {}
    for item in args.oversample_classes:
        if ":" in item or "=" in item:
            sep = ":" if ":" in item else "="
            cls_name, r = item.split(sep, 1)
            oversample_map[cls_name.strip()] = int(r.strip())
        else:
            oversample_map[item.strip()] = args.oversample_ratio

    if not oversample_map and class_counts:
        median_cnt = sorted(class_counts.values())[len(class_counts) // 2]
        for cls_name, cnt in class_counts.items():
            if cnt < median_cnt * 0.5:
                oversample_map[cls_name] = args.oversample_ratio

    oversample_targets = set(oversample_map.keys())
    if oversample_map:
        print(f"Oversampling configuration: {oversample_map}")

    total_created = 0
    for jf, img_p, d, labels in samples:
        # 基础拷贝
        new_img = dst_dir / img_p.name
        new_json = dst_dir / jf.name
        if not new_img.exists():
            new_img.symlink_to(img_p)
        with open(new_json, "w", encoding="utf-8") as fp:
            json.dump(d, fp)
        total_created += 1

        # 检查过采样
        rep_ratio = 1
        for l in labels:
            if l in oversample_map:
                rep_ratio = max(rep_ratio, oversample_map[l])

        if rep_ratio > 1:
            for rep in range(1, rep_ratio):
                stem = f"rep{rep}_{jf.stem}"
                rep_img = dst_dir / f"{stem}{img_p.suffix}"
                rep_json = dst_dir / f"{stem}.json"
                if not rep_img.exists():
                    if args.occlusion_aug:
                        raw_img = cv2.imread(str(img_p))
                        if raw_img is not None:
                            aug_img = apply_occlusion_aug(raw_img, d.get("shapes", []), oversample_targets, rep_idx=rep)
                            if rep % 3 == 0:
                                aug_img = apply_brightness(aug_img, 0.88, 1.12)
                            elif rep % 3 == 1:
                                aug_img = apply_contrast(aug_img, 0.88, 1.12)
                            elif rep % 3 == 2:
                                aug_img = apply_clahe(aug_img, 1.5)
                            cv2.imwrite(str(rep_img), aug_img)
                        else:
                            rep_img.symlink_to(img_p)
                    else:
                        rep_img.symlink_to(img_p)
                d_copy = dict(d)
                d_copy["imagePath"] = rep_img.name
                with open(rep_json, "w", encoding="utf-8") as fp:
                    json.dump(d_copy, fp)
                total_created += 1

    print(f"Balanced dataset created with {total_created} samples in {dst_dir}")


if __name__ == "__main__":
    main()
