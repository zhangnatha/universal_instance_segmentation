#!/usr/bin/env python3
"""分析 Detectron2 .pth 权重文件的网络结构与训练状态。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .run_logging import setup_run_logging


def main():
    """解析权重文件并输出网络架构与状态报告。"""
    setup_run_logging("ANALYZE_MODEL")
    parser = argparse.ArgumentParser(description="Analyze a Detectron2 .pth architecture and training state")
    parser.add_argument("weights", help="Path to Detectron2 checkpoint (.pth) file")
    parser.add_argument("--json", action="store_true", help="Output report in JSON format")
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)

    blocks = {}
    for stage in ("res2", "res3", "res4", "res5"):
        indices = []
        pattern = re.compile(rf"^backbone\.bottom_up\.{stage}\.(\d+)\.")
        for key in state:
            match = pattern.match(key)
            if match:
                indices.append(int(match.group(1)))
        blocks[stage] = max(indices) + 1 if indices else 0
    depth_map = {(3, 4, 6, 3): 50, (3, 4, 23, 3): 101, (3, 8, 36, 3): 152}
    cls_shape = tuple(state["roi_heads.box_predictor.cls_score.weight"].shape)
    mask_shape = tuple(state["roi_heads.mask_head.predictor.weight"].shape)
    angle_key = "roi_heads.angle_head.predictor.weight"
    report = {
        "file": str(Path(args.weights).expanduser()),
        "framework": "Detectron2",
        "architecture": f"Mask R-CNN R{depth_map.get(tuple(blocks.values()), '?')}-FPN",
        "resnet_blocks": blocks,
        "foreground_classes": cls_shape[0] - 1,
        "classifier_shape": cls_shape,
        "mask_predictor_shape": mask_shape,
        "angle_head": angle_key in state,
        "angle_bins": int(state[angle_key].shape[0]) if angle_key in state else None,
        "iteration": checkpoint.get("iteration"),
        "parameter_tensors": len(state),
        "has_optimizer": "optimizer" in checkpoint,
        "has_scheduler": "scheduler" in checkpoint,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
