#!/usr/bin/env python3
"""
Roboflow 流程 - 模型推理与结果导出工具套件
对输入的图像或图像目录执行实例分割推理，
保存渲染的可视化结果图像（*.BMP / *.jpg）以及与
instance-segmentation-repro 格式兼容的结构化 JSON 记录（*.json）。
"""

from __future__ import annotations

import os
import sys
import time
import json
import gc
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch

BASE_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, BASE_DIR)

from .evaluator import ModelEvaluator
from ..data.classes import load_class_names

IMAGE_EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.webp', '.tiff', '.tif', '.BMP', '.PNG', '.JPG', '.JPEG'}


def parse_args(argv=None):
    """解析用于批量/单图模型推理及结果导出的命令行参数。"""
    parser = argparse.ArgumentParser(description="Run Roboflow Instance Segmentation Inference & Export JSON")
    parser.add_argument("--input", "-i", required=True, help="Input image file or image directory")
    parser.add_argument("--output", "-o", default="results/predictions", help="Output directory for predictions and visualizations")
    parser.add_argument("--weights", "-w", default=None, help="Model weights checkpoint path (default: saved_models/latest/best.pt)")
    parser.add_argument("--conf", type=float, default=0.50, help="Default confidence score threshold (default: 0.50)")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS mask IoU threshold for overlapping instances (default: 0.50, 1.0 to disable)")
    parser.add_argument("--mask-alpha", "--alpha", type=float, default=0.40, help="Mask overlay transparency alpha (0.0~1.0, default: 0.40)")
    parser.add_argument(
        "--class-thresholds", nargs="*", metavar="CLASS=VALUE",
        help="Optional per-class confidence thresholds, e.g. <label>=0.5"
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most the first N images")
    parser.add_argument("--no-json", action="store_true", help="Skip generating prediction JSON files")
    parser.add_argument("--no-image", action="store_true", help="Skip rendering visualization result images")
    parser.add_argument("--device", default="auto", help="Execution device: auto / cuda:0 / cpu")
    parser.add_argument("--benchmark", action="store_true", help="Profile and display modular latency breakdown (ms)")
    return parser.parse_args(argv)


def parse_class_thresholds(raw_list: Optional[List[str]], default_conf: float) -> Dict[str, float]:
    """解析自定义每类别置信度阈值映射字典。"""
    thresholds = {}
    if raw_list:
        for item in raw_list:
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    thresholds[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return thresholds


def format_duration(seconds: float) -> str:
    """将秒数耗时格式化为易读的 mm:ss 或 hh:mm:ss 字符串。"""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def main(argv=None):
    """主执行逻辑：加载模型、执行实例分割推理并导出可视化结果与预测 JSON。"""
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.weights or os.path.join(BASE_DIR, "saved_models", "latest", "best.pt")
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model checkpoint not found at: {weights_path}")
        sys.exit(1)

    if args.benchmark and input_path.is_file():
        from .legacy_infer_single import ModularInferencer, print_benchmark_report
        profiler = ModularInferencer(weights_path=weights_path, device=args.device)
        profiler.warmup(warmup_runs=2)
        runs = 5
        results = []
        for i in range(runs):
            save_now = (not args.no_image) and (i == runs - 1)
            res_item = profiler.infer_single_image(
                image_path=str(input_path),
                conf_threshold=args.conf,
                save_output=save_now,
                output_dir=str(output_dir)
            )
            results.append(res_item)
        print_benchmark_report(profiler, results, str(input_path), runs)
        return

    print("=" * 78)
    print(" 🚀 ROBOFLOW MODEL INFERENCE & ARTIFACT EXPORTER")
    print("=" * 78)
    print(f" • Input Source:       {input_path}")
    print(f" • Output Directory:   {output_dir}")
    print(f" • Model Checkpoint:   {os.path.relpath(weights_path, BASE_DIR) if os.path.isabs(weights_path) and str(weights_path).startswith(BASE_DIR) else weights_path}")
    print(f" • Global Conf Thresh: {args.conf * 100:.0f}%")

    class_thresh_map = parse_class_thresholds(args.class_thresholds, args.conf)
    if class_thresh_map:
        print(f" • Class Thresholds:   {class_thresh_map}")

    # 收集图像文件
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted([
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix in IMAGE_EXTENSIONS
        ])
    else:
        print(f"❌ Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if args.limit and args.limit > 0:
        image_files = image_files[:args.limit]

    total_images = len(image_files)
    if total_images == 0:
        print(f"❌ Error: No valid image files found in {input_path}")
        sys.exit(1)

    print(f" • Target Images:      {total_images} images to process")
    print("=" * 78 + "\n")

    evaluator = ModelEvaluator()
    model_obj = evaluator.load_model(weights_path)
    configured_names = load_class_names()
    if model_obj:
        print(f"✅ Loaded {model_obj['type'].upper()} model with classes: {', '.join(model_obj['classes'])}\n")
    else:
        print(f"⚠️  Loaded model directly from checkpoint: {weights_path}\n")

    t_start = time.time()
    total_detections = 0

    for idx, img_p in enumerate(image_files, 1):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        
        # 使用 torch inference_mode 执行零内存开销推理
        res = evaluator.predict_image_array(
            img_bgr,
            conf_threshold=0.10,  # 使用较低阈值以允许按类别过滤
            model_obj=model_obj,
            weights_path=weights_path,
            return_annotated=False
        )
        
        raw_preds = res.get("predictions", [])
        filtered_preds = []
        detection_records = []

        for p in raw_preds:
            cid = p.get("class_id", 0)
            cname = p.get("class") or (
                configured_names[cid] if isinstance(cid, int) and cid < len(configured_names) else f"class_{cid}"
            )
            score = float(p.get("confidence", 0.0))
            required_conf = class_thresh_map.get(cname, args.conf)

            if score < required_conf:
                continue

            filtered_preds.append(p)

        # 对过滤后的预测结果应用掩码 NMS
        if args.iou is not None and 0.0 < args.iou < 1.0:
            filtered_preds = evaluator.apply_mask_nms(filtered_preds, iou_threshold=args.iou)

        for p in filtered_preds:
            cid = p.get("class_id", 0)
            cname = p.get("class") or (
                configured_names[cid] if isinstance(cid, int) and cid < len(configured_names) else f"class_{cid}"
            )
            score = float(p.get("confidence", 0.0))
            cx, cy, bw, bh = p["x"], p["y"], p["width"], p["height"]
            x0 = float(cx - bw / 2.0)
            y0 = float(cy - bh / 2.0)
            x1 = float(cx + bw / 2.0)
            y1 = float(cy + bh / 2.0)

            # 格式化轮廓点以适配 instance-segmentation-repro 结构
            pts_data = p.get("points", [])
            if pts_data and len(pts_data) >= 3:
                contour = [[int(pt["x"]), int(pt["y"])] for pt in pts_data]
                contours_xy = [contour]
            else:
                contours_xy = [[[int(x0), int(y0)], [int(x1), int(y0)], [int(x1), int(y1)], [int(x0), int(y1)]]]

            detection_records.append({
                "class_id": int(cid),
                "class_name": str(cname),
                "score": float(score),
                "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                "contours_xy": contours_xy
            })

        total_detections += len(detection_records)

        # 1. 保存可视化图像
        target_img_path = output_dir / f"{img_p.stem}{img_p.suffix}"
        if not args.no_image:
            drawn = evaluator.draw_annotated_image(img_bgr, filtered_preds, conf_threshold=args.conf, mask_alpha=args.mask_alpha)
            cv2.imwrite(str(target_img_path), drawn)
            del drawn

        # 2. 保存预测 JSON
        if not args.no_json:
            target_json_path = output_dir / f"{img_p.stem}.json"
            json_payload = {
                "image": str(img_p),
                "detections": detection_records,
                "visualization": str(target_img_path)
            }
            with open(target_json_path, "w", encoding="utf-8") as jf:
                json.dump(json_payload, jf, indent=2, ensure_ascii=False)

        del img_bgr, res, raw_preds, filtered_preds

        # 内存释放与清理
        if idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 进度更新显示
        elapsed = time.time() - t_start
        speed = idx / max(0.001, elapsed)
        eta_sec = (total_images - idx) / max(0.001, speed)
        pct = (idx / total_images) * 100.0
        print(
            f"\r ⏳ [INFER] |{'#' * int(pct // 4)}{'-' * (25 - int(pct // 4))}| "
            f"{idx}/{total_images} ({pct:5.1f}%) "
            f"Speed: {speed:4.1f} img/s "
            f"Elapsed: {format_duration(elapsed)} "
            f"ETA: {format_duration(eta_sec)}",
            end="",
            flush=True
        )

    t_total = time.time() - t_start
    print("\n\n" + "=" * 78)
    print(" 🎉 INFERENCE & ARTIFACT EXPORT COMPLETED")
    print("=" * 78)
    print(f" • Total Processed Images: {total_images} images in {format_duration(t_total)} ({t_total/total_images*1000:.1f} ms/image)")
    print(f" • Total Detected Objects: {total_detections} instances")
    print(f" • Results Directory:      {output_dir}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
