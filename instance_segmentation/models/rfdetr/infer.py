"""RF-DETR 实例分割推理命令行入口。"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Mapping


def run(config: Mapping[str, Any]):
    """根据提供的配置字典执行 RF-DETR 实例分割推理。"""
    from instance_segmentation.run_logging import setup_run_logging
    setup_run_logging("RFDETR_INFER")
    from . import RFDETRBackend
    return RFDETRBackend().predict(config)


infer = run


def parse_args(argv=None):
    """解析推理命令行参数。"""
    parser = argparse.ArgumentParser(description="RF-DETR instance segmentation inference")
    parser.add_argument("--weights", required=True, help="Path to checkpoint (.pth / .pt)")
    parser.add_argument("--source", "--input", dest="source", required=True, help="Input image or directory")
    parser.add_argument("--size", default="small", choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"], help="Model size (default: small)")
    parser.add_argument("--classes", nargs="+", help="Class names")
    parser.add_argument("--class-conf", nargs="+", help="Per-class confidence thresholds, e.g. class1=0.80")
    parser.add_argument("--class-iou", nargs="+", help="Per-class IoU thresholds, e.g. class1=0.50")
    parser.add_argument("--score-threshold", "--conf", dest="score_threshold", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--iou-threshold", "--iou", dest="iou_threshold", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--draw", action=argparse.BooleanOptionalAction, default=True, help="Draw and save visual overlay images (*_vis.jpg)")
    parser.add_argument("--resolution", type=int, default=432, help="Input resolution (default: 432)")
    parser.add_argument("--device", default="cuda", help="Computation device (cuda/cpu)")
    parser.add_argument("--output", help="Optional output directory to store JSON predictions and visual overlays")
    return parser.parse_args(argv)


def main(argv=None):
    """推理主入口函数。"""
    args = parse_args(argv)
    if args.output:
        from .infer_dataset import main as dataset_infer_main
        dataset_args = [
            "--weights", str(args.weights),
            "--input", str(args.source),
            "--output", str(args.output),
            "--device", str(args.device),
            "--score-threshold", str(args.score_threshold),
            "--iou-threshold", str(args.iou_threshold),
            "--resolution", str(args.resolution),
        ]
        if not args.draw:
            dataset_args.append("--no-draw")
        if args.classes:
            dataset_args.extend(["--classes", *args.classes])
        if args.class_conf:
            dataset_args.extend(["--class-conf", *args.class_conf])
        if args.class_iou:
            dataset_args.extend(["--class-iou", *args.class_iou])
        return dataset_infer_main(dataset_args)
    cfg = vars(args)
    return run(cfg)


if __name__ == "__main__":
    main()

__all__ = ["run", "infer", "main"]
