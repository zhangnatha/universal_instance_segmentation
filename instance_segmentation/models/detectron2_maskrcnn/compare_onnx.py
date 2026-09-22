#!/usr/bin/env python3
"""对比 PyTorch 与 C++ ONNX Runtime 的推理预测结果一致性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.modeling import build_model

from .config import add_model_arguments, build_cfg, experiment_from_args
from .run_logging import setup_run_logging
from .visualize import mask_angle


def fnv1a64(mask: np.ndarray) -> str:
    """计算掩码数组的 FNV-1a 64位哈希值。"""
    value = 1469598103934665603
    for byte in mask.astype(np.uint8, copy=False).ravel():
        value ^= int(byte)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Compare PyTorch and C++ ONNX inference results")
    parser.add_argument("--input", required=True, help="Path to input image file")
    parser.add_argument("--cpp-json", required=True, help="Path to JSON results produced by C++ inference")
    parser.add_argument("--class-thresholds", nargs="*", metavar="CLASS=VALUE", help="Confidence threshold per class")
    parser.add_argument("--angle-classes", nargs="*", default=[], help="Classes for which orientation angle is evaluated")
    parser.add_argument("--score-atol", type=float, default=1e-5, help="Absolute tolerance for detection scores")
    parser.add_argument("--box-atol", type=float, default=1e-3, help="Absolute tolerance for bounding box coordinates")
    parser.add_argument("--angle-atol", type=float, default=1e-3, help="Absolute tolerance for orientation angle")
    parser.add_argument("--mask-min-iou", type=float, default=0.9999, help="Minimum IoU required between masks")
    add_model_arguments(parser, weights_required=True)
    parser.set_defaults(device="cpu")
    return parser.parse_args()


def main():
    """比较主函数：加载 PyTorch 模型并与 C++ ONNX 输出对比。"""
    setup_run_logging("COMPARE_ONNX")
    args = parse_args()
    exp = experiment_from_args(args)
    cfg = build_cfg(exp, training=False)
    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    image = cv2.imread(str(Path(args.input).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Unable to read image: {args.input}")
    height, width = image.shape[:2]
    resize = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    model_image = (
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if cfg.INPUT.FORMAT == "RGB" else image
    )
    transformed = resize.get_transform(model_image).apply_image(model_image)
    tensor = torch.as_tensor(transformed.astype("float32").transpose(2, 0, 1))
    with torch.inference_mode():
        instances = model([{"image": tensor, "height": height, "width": width}])[0]["instances"].to("cpu")

    expected = []
    for index in range(len(instances)):
        class_id = int(instances.pred_classes[index])
        class_name = exp["classes"][class_id]
        score = float(instances.scores[index])
        if score < float(exp.get("class_thresholds", {}).get(class_name, 0.0)):
            continue
        mask = instances.pred_masks[index].numpy().astype(bool)
        expected.append({
            "class_id": class_id,
            "score": score,
            "bbox_xyxy": instances.pred_boxes.tensor[index].numpy(),
            "mask": mask,
            "mask_area": int(mask.sum()),
            "mask_fnv1a64": fnv1a64(mask),
            "angle_degrees": mask_angle(mask) if class_name in set(exp.get("angle_classes", [])) else None,
        })

    actual = json.loads(Path(args.cpp_json).read_text(encoding="utf-8"))["detections"]
    if len(expected) != len(actual):
        raise SystemExit(f"Detection count mismatch: PyTorch={len(expected)}, ONNX={len(actual)}")
    max_score = max_box = max_angle = 0.0
    max_mask_difference = 0
    min_mask_iou = 1.0
    exact_masks = 0
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left["class_id"] != right["class_id"]:
            raise SystemExit(f"Class mismatch for instance {index}")
        max_score = max(max_score, abs(left["score"] - right["score"]))
        max_box = max(max_box, float(np.max(np.abs(left["bbox_xyxy"] - right["bbox_xyxy"]))))
        values = []
        value = 0
        for run in right["mask_rle"]:
            values.extend([value] * int(run))
            value = 1 - value
        right_mask = np.asarray(values, dtype=bool).reshape(left["mask"].shape)
        difference = int(np.count_nonzero(left["mask"] != right_mask))
        intersection = int(np.logical_and(left["mask"], right_mask).sum())
        union = int(np.logical_or(left["mask"], right_mask).sum())
        mask_iou = intersection / union if union else 1.0
        max_mask_difference = max(max_mask_difference, difference)
        min_mask_iou = min(min_mask_iou, mask_iou)
        exact_masks += difference == 0
        if left["angle_degrees"] is not None:
            max_angle = max(max_angle, abs(left["angle_degrees"] - right["angle_degrees"]))
    if (max_score > args.score_atol or max_box > args.box_atol or
            max_angle > args.angle_atol or min_mask_iou < args.mask_min_iou):
        raise SystemExit(
            f"Tolerance exceeded: score={max_score:.9g}, box={max_box:.9g}, angle={max_angle:.9g}"
        )
    print(f"Consistency check passed: {len(expected)} instances")
    print(f"max score abs error: {max_score:.9g}")
    print(f"max box abs error: {max_box:.9g} px")
    print(f"max angle abs error: {max_angle:.9g} deg")
    print(f"masks bit-exact: {exact_masks}/{len(expected)}")
    print(f"max mask differing pixels: {max_mask_difference}")
    print(f"min mask IoU: {min_mask_iou:.9g}")


if __name__ == "__main__":
    main()
