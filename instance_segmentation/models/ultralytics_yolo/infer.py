"""Ultralytics YOLO 实例分割推理命令行入口。"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Mapping


def run(config: Mapping[str, Any]):
    """根据提供的配置字典执行 YOLO 实例分割推理。"""
    from instance_segmentation.run_logging import setup_run_logging
    setup_run_logging("YOLO_INFER")
    from . import UltralyticsBackend
    return UltralyticsBackend().predict(config)


infer = run


def parse_args(argv=None):
    """解析推理命令行参数。"""
    parser = argparse.ArgumentParser(description="Ultralytics YOLO instance segmentation inference")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--source", "--input", dest="source", required=True, help="Input image or directory")
    parser.add_argument("--output", "--output-dir", dest="output", default=None, help="Output directory to save predictions and visualizations")
    parser.add_argument("--classes", nargs="+", help="Class names")
    parser.add_argument("--class-conf", nargs="+", help="Per-class confidence thresholds, e.g. class1=0.80")
    parser.add_argument("--class-iou", nargs="+", help="Per-class IoU thresholds, e.g. class1=0.50")
    parser.add_argument("--conf", type=float, default=0.25, help="Global confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="Global IoU threshold")
    parser.add_argument("--device", default="cuda", help="Computation device (cuda/cpu)")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Save visualization results")
    parser.add_argument("--draw", action=argparse.BooleanOptionalAction, default=True, help="Render and save visual overlays (*_vis.jpg)")
    return parser.parse_args(argv)


def main(argv=None):
    """推理主入口函数。"""
    args = parse_args(argv)
    cfg = vars(args)
    return run(cfg)


if __name__ == "__main__":
    main()

__all__ = ["run", "infer", "main"]
