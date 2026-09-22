"""RF-DETR 实例分割训练命令行入口。"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Mapping


def run(config: Mapping[str, Any]):
    """根据提供的配置字典执行 RF-DETR 实例分割训练。"""
    from instance_segmentation.run_logging import setup_run_logging
    setup_run_logging("RFDETR_TRAIN")
    from . import RFDETRBackend
    return RFDETRBackend().train(config)


train = run


def parse_args(argv=None):
    """解析训练命令行参数。"""
    parser = argparse.ArgumentParser(description="RF-DETR instance segmentation training")
    parser.add_argument("--data", required=True, help="Path to prepared dataset directory containing train and valid")
    parser.add_argument("--size", default="small", choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"], help="Model size (default: small)")
    parser.add_argument("--weights", "--pretrain-weights", dest="weights", help="Path to pretrained weights")
    parser.add_argument("--resolution", type=int, default=432, help="Input resolution (default: 432)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--output-dir", default="output/rfdetr_small", help="Output directory (default: output/rfdetr_small)")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (default: 1)")
    parser.add_argument("--accelerator", default="auto", help="Accelerator (auto/cuda/cpu)")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluation interval in epochs (default: 1)")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    return parser.parse_args(argv)


def main(argv=None):
    """训练主入口函数。"""
    args = parse_args(argv)
    cfg = vars(args)
    return run(cfg)


if __name__ == "__main__":
    main()

__all__ = ["run", "train", "main"]
