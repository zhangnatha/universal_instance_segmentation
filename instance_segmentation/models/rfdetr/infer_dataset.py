#!/usr/bin/env python3
"""在图像目录上运行 RF-DETR 实例分割推理，并保存 JSON 预测结果和可视化标注图像。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfdetr import variants as rfdetr_variants
from instance_segmentation.models.thresholds import class_thresholds, bbox_iou
from instance_segmentation.data.classes import load_class_names

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

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


def mask_to_row_major_rle(mask: np.ndarray) -> list[int]:
    """将二值二维掩码编码为以背景开始的行优先游程编码（RLE）。"""
    flat = mask.reshape(-1).astype(bool)
    if flat.size == 0:
        return []
    diffs = np.diff(flat.view(np.int8))
    change_indices = np.where(diffs != 0)[0] + 1
    split_indices = np.concatenate(([0], change_indices, [flat.size]))
    runs = np.diff(split_indices).tolist()
    if flat[0]:
        runs = [0] + runs
    return runs


def render_visual_overlay(image_bgr: np.ndarray, detections: list[dict], alpha: float = 0.40, num_classes: int = 1) -> np.ndarray:
    """渲染半透明掩码、轮廓、边界框以及类别标签框。"""
    vis = image_bgr.copy()
    overlay = image_bgr.copy()

    for det in detections:
        mask = det.get("mask_np")
        color = get_color(det["class_id"], num_classes)
        if mask is not None and mask.any():
            overlay[mask] = color

    cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)

    for det in detections:
        cid = det["class_id"]
        cname = det["class_name"]
        score = det["score"]
        box = [int(round(v)) for v in det["bbox_xyxy"]]
        color = get_color(cid, num_classes)
        mask = det.get("mask_np")

        if mask is not None and mask.any():
            mask_u8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2, cv2.LINE_AA)

        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2, cv2.LINE_AA)

        label_text = f"{cname} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        tx = max(0, box[0])
        ty = max(th + 4, box[1] - 4)
        cv2.rectangle(vis, (tx, ty - th - 4), (tx + tw + 6, ty + baseline - 2), color, -1)
        cv2.putText(vis, label_text, (tx + 3, ty - 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return vis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析数据集推理命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--size", choices=("nano", "small", "medium", "large", "xlarge", "2xlarge"), default="small", help="RF-DETR model size (default: small)")
    parser.add_argument("--input", required=True, help="Input directory containing images")
    parser.add_argument("--output", required=True, help="Output directory to store JSON predictions and visual overlays")
    parser.add_argument("--classes", nargs="+", default=None, help="Class names in fixed order (supports arbitrary number of classes; defaults to classes.names or checkpoint)")
    parser.add_argument("--class-conf", nargs="+", default=None, help="Per-class confidence thresholds, e.g. class1=0.5 class2=0.4 or class1=0.5,class2=0.4")
    parser.add_argument("--class-iou", nargs="+", default=None, help="Per-class IoU thresholds, e.g. class1=0.5 class2=0.5")
    parser.add_argument("--score-threshold", "--conf", dest="score_threshold", type=float, default=0.40, help="Default confidence score threshold (default: 0.40)")
    parser.add_argument("--iou-threshold", "--iou", dest="iou_threshold", type=float, default=0.50, help="Default NMS IoU threshold (default: 0.50, 1.0 to disable)")
    parser.add_argument("--draw", action=argparse.BooleanOptionalAction, default=True, help="Draw and save visual overlay images (*_vis.jpg)")
    parser.add_argument("--device", default="cuda", help="Computation device (cuda/cpu)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None, help="Use float16 for inference (native mode defaults on; cpp mode defaults off)")
    parser.add_argument("--preprocess", choices=("native", "cpp"), default="native", help="native uses RF-DETR PIL/torch preprocessing; cpp matches the C++ OpenCV resize/normalize input")
    parser.add_argument("--resolution", type=int, default=432, help="Input resolution (default: 432)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行数据集推理主入口。"""
    args = parse_args(argv)
    from instance_segmentation.run_logging import setup_run_logging
    setup_run_logging("RFDETR_INFER_DATASET")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        images = [input_path] if input_path.suffix.lower() in IMAGE_SUFFIXES else []
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        images = []
    if not images:
        print(f"Error: No valid images found in {input_path}", file=sys.stderr)
        return 1

    classes = list(args.classes) if args.classes else load_class_names()
    if not classes:
        try:
            ckpt = torch.load(str(args.weights), map_location="cpu")
            state = ckpt.get("model", ckpt)
            if "class_embed.weight" in state:
                classes = [f"class_{i}" for i in range(state["class_embed.weight"].shape[0])]
            elif "class_embed.bias" in state:
                classes = [f"class_{i}" for i in range(state["class_embed.bias"].shape[0])]
        except Exception:
            pass
    if not classes:
        raise ValueError("Classes must be provided via --classes or configured in classes.names")
    conf_map = class_thresholds(classes, args.class_conf, args.score_threshold, name="class-conf")
    iou_map = class_thresholds(classes, args.class_iou, args.iou_threshold, name="class-iou")
    min_conf = min(conf_map.values())

    model_class = getattr(rfdetr_variants, f"RFDETRSeg{args.size.title().replace('Xlarge', 'XLarge').replace('2Xlarge', '2XLarge')}")
    print(f"Loading {model_class.__name__} model from {args.weights} (device={args.device}, num_classes={len(classes)}, resolution={args.resolution})...")
    model = model_class(pretrain_weights=str(args.weights), num_classes=len(classes), device=args.device)
    if hasattr(model, "model_config") and hasattr(model.model_config, "resolution"):
        model.model_config.resolution = args.resolution
    if hasattr(model, "model") and hasattr(model.model, "resolution"):
        model.model.resolution = args.resolution
    use_fp16 = args.fp16 if args.fp16 is not None else args.preprocess == "native"
    if args.preprocess == "cpp" and use_fp16:
        raise ValueError("--preprocess cpp requires FP32 parity; pass --no-fp16")
    if use_fp16 and args.device == "cuda":
        model.inference(dtype=torch.float16)
    elif args.preprocess == "cpp":
        device = torch.device(args.device)
        model.model.model.to(device).eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    total = len(images)
    print(f"Running inference on {total} images from {input_path}...")
    print(f"Confidence thresholds: {conf_map}")
    print(f"IoU thresholds: {iou_map}")
    print(f"Visual overlays: {'enabled' if args.draw else 'disabled'}")

    t0 = time.time()
    total_detections = 0

    for idx, img_path in enumerate(images, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: Failed to read {img_path}", file=sys.stderr)
            continue
        h, w = img_bgr.shape[:2]

        if args.preprocess == "cpp":
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
            image_float = resized.astype(np.float32) * (1.0 / 255.0)
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            stdev = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            normalized = (image_float.transpose(2, 0, 1) - mean) / stdev
            batch = torch.from_numpy(normalized.copy()).unsqueeze(0).to(args.device)
            target_sizes = torch.tensor([[h, w]], device=args.device)
            with torch.inference_mode():
                raw = model.model.model(batch)
                result = model.model.postprocess(raw, target_sizes=target_sizes, score_threshold=min_conf)[0]
            class_ids = result["labels"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            boxes = result["boxes"].cpu().numpy()
            masks = result["masks"][:, 0].cpu().numpy().astype(bool)
            dets = [
                SimpleNamespace(
                    class_id=int(cid), confidence=float(score),
                    xyxy=box, mask=mask,
                )
                for cid, score, box, mask in zip(class_ids, scores, boxes, masks)
            ]
        else:
            dets = model.predict(str(img_path), threshold=min_conf, shape=(args.resolution, args.resolution))

        candidates = []
        for i in range(len(dets)):
            if args.preprocess == "cpp":
                detection = dets[i]
                cid = detection.class_id
                conf = detection.confidence
                box = detection.xyxy
                mask_np = detection.mask
            else:
                cid = int(dets.class_id[i])
                conf = float(dets.confidence[i])
                box = dets.xyxy[i]
                mask_np = dets.mask[i].astype(bool) if hasattr(dets, "mask") and dets.mask is not None and len(dets.mask) > i else None
            if cid >= len(classes):
                continue
            cname = classes[cid]
            if conf < conf_map.get(cname, args.score_threshold):
                continue
            candidates.append({
                "class_id": cid,
                "class_name": cname,
                "score": conf,
                "bbox_xyxy": [float(v) for v in box],
                "mask_np": mask_np,
            })

        # 基于按类别 IoU 阈值的按类别贪心 NMS
        filtered_records = []
        for cname in classes:
            c_cands = [d for d in candidates if d["class_name"] == cname]
            c_cands.sort(key=lambda d: d["score"], reverse=True)
            selected = []
            th_iou = iou_map.get(cname, 1.0)
            for cand in c_cands:
                if th_iou < 1.0:
                    suppressed = False
                    for kept in selected:
                        if bbox_iou(tuple(cand["bbox_xyxy"]), tuple(kept["bbox_xyxy"])) > th_iou:
                            suppressed = True
                            break
                    if suppressed:
                        continue
                selected.append(cand)
            filtered_records.extend(selected)

        filtered_records.sort(key=lambda d: d["score"], reverse=True)
        total_detections += len(filtered_records)

        # JSON 数据结构组装
        clean_records = []
        for rec in filtered_records:
            item = {
                "class_id": rec["class_id"],
                "class_name": rec["class_name"],
                "score": round(rec["score"], 4),
                "bbox_xyxy": [round(v, 2) for v in rec["bbox_xyxy"]],
            }
            if rec["mask_np"] is not None:
                item["mask_rle"] = mask_to_row_major_rle(rec["mask_np"])
                item["mask_area"] = int(rec["mask_np"].sum())
            clean_records.append(item)

        payload = {
            "image": str(img_path),
            "imageWidth": w,
            "imageHeight": h,
            "preprocess": args.preprocess,
            "precision": "fp16" if use_fp16 and args.device == "cuda" else "fp32",
            "detections": clean_records,
        }

        out_json_path = output_dir / f"{img_path.stem}.json"
        with open(out_json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

        # 绘制可视化标注图
        if args.draw:
            vis_image = render_visual_overlay(img_bgr, filtered_records, num_classes=len(args.classes))
            vis_path = output_dir / f"{img_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_path), vis_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        det_summary = ", ".join(f"{d['class_name']}({d['score']:.2f})" for d in filtered_records)
        print(f"[{idx}/{total}] {img_path.name}: {len(filtered_records)} detections [{det_summary}]")

    elapsed = time.time() - t0
    fps = total / elapsed if elapsed > 0 else 0
    print(f"\nInference completed in {elapsed:.2f}s ({fps:.1f} FPS, total detections: {total_detections}). Output saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
