"""Ultralytics YOLO 模型导出命令行入口。"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Mapping


def run(config: Mapping[str, Any]):
    """根据提供的配置字典导出 YOLO 实例分割模型。"""
    from instance_segmentation.run_logging import setup_run_logging
    setup_run_logging("YOLO_EXPORT")
    from . import UltralyticsBackend
    return UltralyticsBackend().export(config)


export = run


def parse_args(argv=None):
    """解析模型导出的命令行参数。"""
    parser = argparse.ArgumentParser(description="Ultralytics YOLO instance segmentation export")
    parser.add_argument("--weights", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="Export resolution (default: 640)")
    parser.add_argument("--format", default="onnx", help="Export format (default: onnx)")
    parser.add_argument("--dynamic", action="store_true", default=False, help="Use dynamic shape")
    parser.add_argument("--classes", nargs="+", help="Class names")
    return parser.parse_args(argv)


def main(argv=None):
    """模型导出主入口函数。"""
    args = parse_args(argv)
    cfg = vars(args)
    return run(cfg)


if __name__ == "__main__":
    main()

__all__ = ["run", "export", "main"]
