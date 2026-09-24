#!/usr/bin/env python3
"""Detectron2 Swin-S Mask R-CNN PyTorch 检查点（.pth）到 ONNX 导出工具。

本脚本将训练好的 Swin-S Mask R-CNN PyTorch 模型（.pth）导出为 ONNX 模型，
以兼容 maskRCNN_swin 中的 C++ 推理引擎。

输入输出规范：
  输入：
    - image: FP32 [3, H, W]，BGR 颜色格式
  输出：
    - boxes: FP32 [N, 4]，(x1, y1, x2, y2) 坐标
    - scores: FP32 [N]，置信度分数
    - classes: INT64 [N]，类别索引
    - mask_probs: FP32 [N, 1, 28, 28]，Sigmoid 掩码概率
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence

import cv2
import numpy as np
import onnx
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

# 确保当前目录/仓库根目录在 sys.path 中以正确导入 swin.py
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

try:
    from instance_segmentation.models.detectron2_maskrcnn.backbones import swin  # noqa: F401
except ImportError:
    try:
        import swin  # noqa: F401 - registers build_swin_s_fpn_backbone
    except ImportError:
        parent_swin = SCRIPT_DIR.parent / "swin.py"
        if parent_swin.is_file():
            sys.path.insert(0, str(SCRIPT_DIR.parent))
            import swin  # noqa: F401
        else:
            raise


class OnnxModel(torch.nn.Module):
    """用于固定分辨率推理的 Detectron2 Mask R-CNN ONNX 封装类。"""

    def __init__(self, model: torch.nn.Module, height: int, width: int):
        super().__init__()
        self.model = model
        self.height = height
        self.width = width

    def forward(self, image: torch.Tensor):
        inputs = [{"image": image, "height": self.height, "width": self.width}]
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, None)
        instances = self.model.roi_heads._forward_box(features, proposals)[0]
        heads = self.model.roi_heads
        if hasattr(heads, "angle_head"):
            pooled = heads.box_pooler(
                [features[name] for name in heads.box_in_features], [instances.pred_boxes]
            )
            angle_logits = heads.angle_head(pooled)
            angle_probabilities = angle_logits.softmax(dim=1)
            angle_confidence, angle_bins = angle_probabilities.max(dim=1)
            instances.pred_angles = angle_bins.to(dtype=angle_logits.dtype) * (
                heads.angle_period / heads.angle_bins
            )
            instances.angle_scores = angle_confidence
        mask_features = heads.mask_pooler(
            [features[name] for name in heads.mask_in_features],
            [instances.pred_boxes],
        )
        mask_logits = heads.mask_head.layers(mask_features)
        indices = torch.arange(mask_logits.shape[0], device=instances.pred_classes.device)
        mask_probs = mask_logits[indices, instances.pred_classes][:, None].sigmoid()
        outputs = (
            instances.pred_boxes.tensor,
            instances.scores,
            instances.pred_classes,
            mask_probs,
        )
        if hasattr(heads, "angle_head"):
            angle_values = getattr(instances, "pred_angles", instances.scores)
            angle_scores = getattr(instances, "angle_scores", instances.scores)
            outputs += (
                angle_values,
                angle_scores,
                heads.angle_class_mask_meta.to(dtype=angle_values.dtype)
                + angle_values.sum() * 0.0,
            )
        return outputs


def load_classes_from_file(classes_path: Path) -> list[str]:
    """从 classes.names 文件加载类别名称列表。"""
    classes = []
    if classes_path.is_file():
        with open(classes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    classes.append(line)
    return classes


def auto_find_weights(search_dirs: Sequence[Path]) -> Path | None:
    """在候选目录中查找 .pth 权重文件。"""
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        # 优先查找指定名称
        preferred = [
            directory / "model_followed_20260906.pth",
            directory / "model_final.pth",
            directory / "model.pth",
        ]
        for candidate in preferred:
            if candidate.is_file():
                return candidate
        # 否则查找任意 .pth 文件
        pth_files = sorted(directory.glob("*.pth"))
        if pth_files:
            return pth_files[0]
    return None


def auto_find_sample(search_dirs: Sequence[Path]) -> Path | None:
    """查找用于尺寸跟踪的示例图像。"""
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            images = sorted(directory.glob(ext))
            if images:
                return images[0]
    return None


def parse_args(argv: list[str] | None = None):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Export Detectron2 Swin-S Mask R-CNN PyTorch checkpoint to ONNX",
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default=None,
        help="Path to PyTorch .pth checkpoint (default: auto-detected in model/)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output ONNX path (default: same directory as weights, e.g. model/<stem>.onnx)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="List of class names in order (default: read from classes.names)",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Sample image path for tracing input shape (default: auto-detected or synthesized)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Fixed model input height (default: 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Fixed model input width (default: 640)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Score threshold during tracing (default: 0.5)",
    )
    parser.add_argument("--angle-head", action=argparse.BooleanOptionalAction, default=False,
                        help="Export the trained angle head when the checkpoint includes it")
    parser.add_argument("--angle-classes", nargs="*", default=None,
                        help="Classes trained with angle labels; defaults to all classes")
    parser.add_argument("--angle-bins", type=int, default=72)
    parser.add_argument("--angle-period", type=float, default=360.0)
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX opset version (default: 16)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to load model on for export (default: cpu)",
    )
    parser.add_argument(
        "--sync-model-onnx",
        action="store_true",
        default=True,
        help="Also copy/update model/model.onnx for standard C++ binary default path (default: True)",
    )
    return parser.parse_args(argv)


def export_onnx(
    weights_path: Path,
    output_path: Path,
    classes: list[str],
    height: int = 480,
    width: int = 640,
    score_threshold: float = 0.5,
    sample_image_path: Path | None = None,
    opset_version: int = 16,
    device: str = "cpu",
    sync_model_onnx: bool = True,
    angle_head: bool = False,
    angle_classes: list[str] | None = None,
    angle_bins: int = 72,
    angle_period: float = 360.0,
) -> Path:
    """将 Detectron2 Swin-S Mask R-CNN 模型导出为 ONNX 格式。"""
    print(f"[EXPORT_ONNX] Loading weights: {weights_path}")
    print(f"[EXPORT_ONNX] Classes ({len(classes)}): {classes}")
    print(f"[EXPORT_ONNX] Target resolution: {width}x{height} (BGR format)")
    print(f"[EXPORT_ONNX] Target opset: {opset_version}")

    # 构建 Detectron2 配置
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.BACKBONE.NAME = "build_swin_s_fpn_backbone"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.ROI_HEADS.NAME = "AngleROIHeads" if angle_head else "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.INPUT.MIN_SIZE_TEST = height
    cfg.INPUT.MAX_SIZE_TEST = width
    cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]
    cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = device
    if angle_head:
        from instance_segmentation.models.detectron2_maskrcnn import angle_head as _angle_head  # noqa: F401
        angle_names = classes if angle_classes is None else angle_classes
        unknown = sorted(set(angle_names) - set(classes))
        if unknown:
            raise ValueError("Unknown angle classes: " + ", ".join(unknown))
        cfg.ANGLE_BINS = int(angle_bins)
        cfg.ANGLE_PERIOD = float(angle_period)
        cfg.ANGLE_LOSS_WEIGHT = 1.0
        cfg.ANGLE_LR_FACTOR = 0.01
        cfg.ANGLE_CLASS_IDS = tuple(classes.index(name) for name in angle_names)

    # 构建 PyTorch 模型并加载权重
    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    wrapper = OnnxModel(model, height, width).eval()

    # 准备假数据或示例图像
    if sample_image_path and sample_image_path.is_file():
        image = cv2.imread(str(sample_image_path), cv2.IMREAD_COLOR)
        if image is not None and image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        image = np.zeros((height, width, 3), dtype=np.uint8)

    tensor = torch.as_tensor(
        image.astype("float32").transpose(2, 0, 1),
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=".onnx-export-", suffix=".onnx",
        dir=str(output_path.parent), delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        print(f"[EXPORT_ONNX] Tracing and exporting graph...")
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (tensor,),
                str(temp_path),
                opset_version=opset_version,
                input_names=["image"],
                output_names=(
                    ["boxes", "scores", "classes", "mask_probs", "angles", "angle_scores", "angle_classes"]
                    if angle_head else ["boxes", "scores", "classes", "mask_probs"]
                ),
                dynamic_axes={
                    "boxes": {0: "detections"},
                    "scores": {0: "detections"},
                    "classes": {0: "detections"},
                    "mask_probs": {0: "detections"},
                    **({"angles": {0: "detections"}} if angle_head else {}),
                    **({"angle_scores": {0: "detections"}} if angle_head else {}),
                },
                do_constant_folding=True,
            )

        print("[EXPORT_ONNX] Checking ONNX model consistency...")
        onnx.checker.check_model(onnx.load(str(temp_path)))
        temp_path.replace(output_path)
        print(f"[EXPORT_ONNX] Successfully exported ONNX: {output_path}")
        print(f"[EXPORT_ONNX] File size: {output_path.stat().st_size / (1024 * 1024):.2f} MiB")

        if sync_model_onnx and output_path.name != "model.onnx":
            canonical_model = output_path.parent / "model.onnx"
            shutil.copy2(output_path, canonical_model)
            print(f"[EXPORT_ONNX] Synchronized canonical model: {canonical_model}")

        return output_path
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None):
    """导出主函数。"""
    args = parse_args(argv)

    repo_dir = SCRIPT_DIR
    candidate_roots = [
        Path.cwd(),
        SCRIPT_DIR.parents[3] if len(SCRIPT_DIR.parents) >= 4 else SCRIPT_DIR,
        SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parents) >= 3 else SCRIPT_DIR,
        SCRIPT_DIR.parent,
        SCRIPT_DIR,
    ]
    root_dir = candidate_roots[0]

    # 1. 解析权重路径
    if args.weights:
        weights_path = Path(args.weights).expanduser().resolve()
    else:
        candidate_dirs = [
            root_dir / "pretrained", root_dir / "model", root_dir,
            Path.cwd() / "pretrained", Path.cwd() / "model", Path.cwd(),
        ]
        weights_path = auto_find_weights(candidate_dirs)
        if not weights_path:
            raise SystemExit(
                "Error: No .pth weights file found. Specify via --weights <path/to/model.pth>"
            )

    if not weights_path.is_file():
        raise SystemExit(f"Error: Weights file does not exist: {weights_path}")

    # 2. 解析输出路径
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        # 存放在权重文件所在目录
        output_path = weights_path.parent / f"{weights_path.stem}.onnx"

    # 3. 解析类别列表
    if args.classes:
        classes = args.classes
    else:
        classes_candidates = [
            root_dir / "configs" / "classes.names",
            root_dir / "cpp" / "models" / "detectron2_maskrcnn" / "classes.names",
            root_dir / "classes.names",
            Path.cwd() / "configs" / "classes.names",
            Path.cwd() / "classes.names",
        ]
        classes_file = next((p for p in classes_candidates if p.is_file()), None)
        if classes_file and classes_file.is_file():
            classes = load_classes_from_file(classes_file)
        else:
            raise ValueError("Must specify --classes or provide a classes.names file")

    if not classes:
        raise ValueError("Must specify --classes or provide a classes.names file")

    # 4. 解析示例图像
    sample_path = None
    if args.sample:
        sample_path = Path(args.sample).expanduser().resolve()
    else:
        candidate_sample_dirs = [
            root_dir / "images",
            root_dir / "test",
            root_dir / "samples",
        ]
        sample_path = auto_find_sample(candidate_sample_dirs)

    export_onnx(
        weights_path=weights_path,
        output_path=output_path,
        classes=classes,
        height=args.height,
        width=args.width,
        score_threshold=args.score_threshold,
        sample_image_path=sample_path,
        opset_version=args.opset,
        device=args.device,
        sync_model_onnx=args.sync_model_onnx,
        angle_head=args.angle_head,
        angle_classes=args.angle_classes,
        angle_bins=args.angle_bins,
        angle_period=args.angle_period,
    )


if __name__ == "__main__":
    main()
