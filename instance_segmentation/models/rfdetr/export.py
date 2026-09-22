"""RF-DETR ONNX 导出命令行入口。"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Mapping


def run(config: Mapping[str, Any]):
    """根据提供的配置字典导出 RF-DETR ONNX 模型。"""
    from instance_segmentation.run_logging import setup_run_logging
    setup_run_logging("RFDETR_EXPORT")
    from . import RFDETRBackend
    return RFDETRBackend().export(config)


export = run


def parse_args(argv=None):
    """解析 ONNX 导出命令行参数。"""
    parser = argparse.ArgumentParser(description="RF-DETR ONNX export")
    parser.add_argument("--weights", required=True, help="Path to checkpoint (.pth / .pt)")
    parser.add_argument("--output-dir", default="output/onnx/rfdetr_small", help="Output directory")
    parser.add_argument("--size", default="small", choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"], help="Model size (default: small)")
    parser.add_argument("--resolution", type=int, default=432, help="Input resolution H and W (default: 432)")
    parser.add_argument("--height", type=int, default=None, help="Input height (defaults to resolution)")
    parser.add_argument("--width", type=int, default=None, help="Input width (defaults to resolution)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--dynamic-batch", action="store_true", default=False, help="Enable dynamic batch")
    parser.add_argument("--opset", "--opset-version", dest="opset_version", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False, help="Export in float16")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Verbose logging")
    return parser.parse_args(argv)


def main(argv=None):
    """ONNX 导出主入口函数。"""
    args = parse_args(argv)
    cfg = vars(args)
    if cfg.get("height") is None:
        cfg["height"] = cfg["resolution"]
    if cfg.get("width") is None:
        cfg["width"] = cfg["resolution"]
    return run(cfg)


if __name__ == "__main__":
    main()

__all__ = ["run", "export", "main"]
