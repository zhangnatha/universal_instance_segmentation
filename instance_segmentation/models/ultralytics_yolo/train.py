"""Ultralytics YOLO 实例分割训练命令行入口。"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Mapping


def run(config: Mapping[str, Any]):
    """根据提供的配置字典执行 YOLO 实例分割训练。"""
    from instance_segmentation.run_logging import setup_run_logging
    setup_run_logging("YOLO_TRAIN")
    from . import UltralyticsBackend
    return UltralyticsBackend().train(config)


train = run


def parse_args(argv=None):
    """解析训练命令行参数。"""
    parser = argparse.ArgumentParser(description="Ultralytics YOLO instance segmentation training")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model-id", default="yolov11", choices=["yolov11", "yolo26", "roboflow-3.0"], help="YOLO family")
    parser.add_argument("--size", default="small", choices=["nano", "small", "medium", "large", "xlarge", "fast", "accurate"], help="Model size")
    parser.add_argument("--weights", help="Initial weights path (.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--device", default="0", help="CUDA device or cpu (default: 0)")
    parser.add_argument("--project", default="output", help="Project directory (default: output)")
    parser.add_argument("--name", default="yolo11s_seg", help="Experiment name (default: yolo11s_seg)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training")
    return parser.parse_args(argv)


def main(argv=None):
    """训练主入口函数。"""
    args = parse_args(argv)
    cfg = vars(args)
    return run(cfg)


if __name__ == "__main__":
    main()

__all__ = ["run", "train", "main"]
