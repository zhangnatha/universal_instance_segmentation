#!/usr/bin/env python3
"""基于 LabelMe 格式验证集计算 Detectron2 Mask R-CNN 模型的 COCO AP 评估指标。"""

from __future__ import annotations

import argparse
import os
import sys

from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model

from .config import add_model_arguments, build_cfg, experiment_from_args
from instance_segmentation.data.labelme import register_labelme
from .infer import StrictDetectionCheckpointer, resolve_checkpoint_architecture
from .run_logging import setup_run_logging


def main(argv=None):
    """评估主函数：加载模型并运行 COCO 评估器。"""
    setup_run_logging("EVALUATE")
    parser = argparse.ArgumentParser(description="Compute COCO AP on a LabelMe validation set")
    parser.add_argument("--experiment", default="evaluation", help="Experiment name")
    parser.add_argument("--val-dir", required=True, help="LabelMe validation dataset directory")
    parser.add_argument("--output-dir", default="results", help="Directory where evaluation results are stored")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader worker processes")
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True,
        help="Show a terminal evaluation progress bar",
    )
    add_model_arguments(parser, weights_required=True)
    args = parser.parse_args(argv)
    # 评估必须使用与权重检查点相同的骨干网络实现和 RPN Anchor 配置。
    # 从状态字典解析缺省的参数标志，避免在模型部分加载的情况下执行评估。
    exp = resolve_checkpoint_architecture(experiment_from_args(args))
    cfg = build_cfg(exp, training=True)
    name = cfg.DATASETS.TEST[0]
    register_labelme(name, exp["val_dir"], exp["classes"], cfg.ANGLE_BINS, cfg.ANGLE_PERIOD)
    model = build_model(cfg)
    model.eval()
    StrictDetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    output = os.path.join(cfg.OUTPUT_DIR, "evaluation")
    evaluator = COCOEvaluator(name, output_dir=output)
    data_loader = build_detection_test_loader(cfg, name)
    if args.progress:
        from tqdm import tqdm

        data_loader = tqdm(
            data_loader,
            total=len(data_loader),
            desc="Evaluation",
            unit="image",
            dynamic_ncols=True,
            file=sys.__stderr__,
        )
    results = inference_on_dataset(model, data_loader, evaluator)
    print(results)


if __name__ == "__main__":
    main()
