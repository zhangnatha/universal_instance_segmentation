#!/usr/bin/env python3
"""从训练集预测结果中挖掘假阳性切片作为难负样本，供后续训练轮次使用。

典型流程（所有预测 JSON 必须严格由 TRAIN_DIR 生成）:

  python infer.py --input TRAIN_DIR --output results/train_predictions ...
  python scripts/mine_hard_negatives.py --train-dir TRAIN_DIR \
      --prediction-dir results/train_predictions --output-dir output/hard_negatives
  instance-seg run detectron2 train --config '{"hard_negative_dir": "output/hard_negatives"}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from instance_segmentation.data.hard_negative import mine_hard_negative_crops


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Mine background crops from predictions made on the training set only"
    )
    parser.add_argument("--train-dir", required=True, help="Original LabelMe training directory")
    parser.add_argument("--prediction-dir", required=True, help="Predictions made on --train-dir")
    parser.add_argument("--output-dir", required=True, help="New LabelMe directory for empty crops")
    parser.add_argument("--class-name", default="all", help="False-positive class to mine, or 'all' for any class (default: all)")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=2.0, help="Crop size multiplier around a prediction")
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--max-per-image", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """主函数执行难负样本挖掘流程。"""
    args = parse_args(argv)
    stats = mine_hard_negative_crops(
        args.train_dir,
        args.prediction_dir,
        args.output_dir,
        class_name=args.class_name,
        score_threshold=args.score_threshold,
        margin=args.margin,
        min_size=args.min_size,
        max_per_image=args.max_per_image,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
