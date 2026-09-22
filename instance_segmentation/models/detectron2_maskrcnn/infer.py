#!/usr/bin/env python3
"""Detectron2 Mask R-CNN 模型推理预测入口，支持按类别后处理与多边形/方向结果导出。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from torchvision.ops import nms
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.modeling import build_model

from .config import add_model_arguments, build_cfg, experiment_from_args
from instance_segmentation.data.labelme import IMAGE_SUFFIXES
from .run_logging import setup_run_logging
from .visualize import draw_predictions, save_json


_ANGLE_METADATA_KEYS = {
    "roi_heads.angle_period_meta",
    "roi_heads.angle_class_mask_meta",
}


def _state_architecture_signature(state):
    """在不构建模型的情况下提取权重检查点中的架构配置信息。"""
    if not isinstance(state, dict):
        return {}
    keys = tuple(state)
    has_timm = any(key.startswith("backbone.bottom_up.model.") for key in keys)
    has_native = any(
        key.startswith("backbone.bottom_up.patch_embed.")
        or key.startswith("backbone.bottom_up.stem.")
        for key in keys
    )
    if has_timm and has_native:
        # 混合的 state_dict 不是该后端输出的合法架构；切勿从有歧义的检查点中任选实现。
        implementation = None
    elif has_timm:
        implementation = "timm"
    elif has_native:
        implementation = "native"
    else:
        implementation = None
    rpn_weight = state.get("proposal_generator.rpn_head.objectness_logits.weight")
    try:
        anchor_count = int(rpn_weight.shape[0])
    except (AttributeError, IndexError, TypeError):
        anchor_count = None
    angle_weight = state.get("roi_heads.angle_head.predictor.weight")
    try:
        angle_bins = int(angle_weight.shape[0])
    except (AttributeError, IndexError, TypeError):
        angle_bins = None
    period_meta = state.get("roi_heads.angle_period_meta")
    try:
        angle_period = float(period_meta.reshape(-1)[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        angle_period = None
    if angle_period is not None and not torch.isfinite(torch.tensor(angle_period)):
        angle_period = None
    return {
        "implementation": implementation,
        "anchor_count": anchor_count,
        "angle_head": angle_bins is not None,
        "angle_bins": angle_bins,
        "angle_period": angle_period,
    }


class StrictDetectionCheckpointer(DetectionCheckpointer):
    """快速失败报错，避免在头部结构不匹配时使用随机初始化权重静默运行。"""

    @staticmethod
    def _meaningful_unexpected_keys(keys):
        # 使用 PyTorch BatchNorm 训练的骨干网络会保存此标量计数器，而 Detectron2 的 FrozenBN
        # 刻意不含此 buffer。FrozenBN 仍加载权重/偏置/均值/方差，因此仅丢弃该计数器在推理上等价，
        # 且不应掩盖真正的结构不匹配。
        return [key for key in keys if not key.endswith(".num_batches_tracked")]

    def _load_model(self, checkpoint):
        checkpoint_state = checkpoint.get("model", {})
        checkpoint_signature = _state_architecture_signature(checkpoint_state)
        model_signature = _state_architecture_signature(self.model.state_dict())
        model_keys = set(self.model.state_dict())
        for key in list(checkpoint_state):
            if key.endswith(".num_batches_tracked") and key not in model_keys:
                checkpoint_state.pop(key)
        incompatible = super()._load_model(checkpoint)
        incorrect_shapes = getattr(incompatible, "incorrect_shapes", [])
        missing_keys = [
            key for key in getattr(incompatible, "missing_keys", [])
            if key not in _ANGLE_METADATA_KEYS
        ]
        unexpected_keys = self._meaningful_unexpected_keys(
            getattr(incompatible, "unexpected_keys", [])
        )
        unexpected_keys = [key for key in unexpected_keys if key not in _ANGLE_METADATA_KEYS]
        if incorrect_shapes or missing_keys or unexpected_keys:
            details = "; ".join(
                f"{name}: checkpoint={checkpoint_shape}, model={model_shape}"
                for name, checkpoint_shape, model_shape in incorrect_shapes
            )
            hints = []
            backbone_issue = any(
                name.startswith("backbone.") for name in missing_keys + unexpected_keys
            ) or any(name.startswith("backbone.") for name, _, _ in incorrect_shapes)
            checkpoint_impl = checkpoint_signature.get("implementation")
            model_impl = model_signature.get("implementation")
            if backbone_issue:
                if checkpoint_impl and model_impl and checkpoint_impl != model_impl:
                    hints.append(
                        f"The checkpoint backbone appears to use {checkpoint_impl}, "
                        f"but the current model uses {model_impl}; pass "
                        f"--backbone-impl {checkpoint_impl} (with the matching --backbone)."
                    )
                else:
                    hints.append(
                        "Pass the same --backbone and --backbone-impl values used for training "
                        "(native or timm)."
                    )
            checkpoint_anchors = checkpoint_signature.get("anchor_count")
            model_anchors = model_signature.get("anchor_count")
            rpn_issue = any(
                name.startswith("proposal_generator.rpn_head.")
                for name, _, _ in incorrect_shapes
            ) or any(
                name.startswith("proposal_generator.rpn_head.")
                for name in missing_keys + unexpected_keys
            )
            if rpn_issue:
                if checkpoint_anchors in (3, 6) and model_anchors in (3, 6):
                    expected = (
                        "--small-object-anchors" if checkpoint_anchors == 6
                        else "--no-small-object-anchors"
                    )
                    hints.append(
                        f"The checkpoint RPN has {checkpoint_anchors} anchors/location, "
                        f"but the current model has {model_anchors}; pass {expected}."
                    )
                else:
                    hints.append(
                        "Pass the same --small-object-anchors/--no-small-object-anchors "
                        "setting used for training."
                    )
            key_details = []
            if missing_keys:
                key_details.append(
                    "missing=" + ", ".join(missing_keys[:8])
                    + (f" ... ({len(missing_keys)} total)" if len(missing_keys) > 8 else "")
                )
            if unexpected_keys:
                key_details.append(
                    "unexpected=" + ", ".join(unexpected_keys[:8])
                    + (f" ... ({len(unexpected_keys)} total)" if len(unexpected_keys) > 8 else "")
                )
            if key_details:
                details = "; ".join(part for part in [details, *key_details] if part)
            raise RuntimeError(
                "Checkpoint/model architecture mismatch; inference was stopped "
                "before producing invalid predictions. " + details
                + ((" Hints: " + " ".join(hints)) if hints else "")
            )
        return incompatible


def parse_args(argv=None):
    """解析推理命令行参数。"""
    parser = argparse.ArgumentParser(description="Run instance segmentation inference")
    parser.add_argument("--input", required=True, help="Input image or image directory")
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory; default: results under the project root",
    )
    parser.add_argument(
        "--class-thresholds", "--class-conf", dest="class_thresholds", nargs="*", metavar="CLASS=VALUE",
        help="Per-class score thresholds; unspecified classes default to 0.5",
    )
    parser.add_argument(
        "--class-iou-thresholds", "--class-iou", dest="class_iou_thresholds", nargs="*", metavar="CLASS=VALUE",
        help="Per-class NMS IoU thresholds; unspecified classes default to 0.5",
    )
    parser.add_argument(
        "--class-max-detections", nargs="*", metavar="CLASS=INTEGER",
        help="Optional per-class output limit, e.g. class_name=1; disabled when unspecified",
    )
    parser.add_argument("--angle-classes", nargs="*", default=[], help="Classes that display direction arrows")
    parser.add_argument(
        "--angle-source",
        choices=["prediction", "mask"],
        default="mask",
        help="Direction source for arrows; mask principal axis matches the legacy renderer",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.35, help="Mask overlay opacity")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for images in directories")
    parser.add_argument("--limit", type=int, help="Process at most the first N images")
    parser.add_argument("--no-json", action="store_true", help="Do not save per-image detection JSON results")
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True,
        help="Show a terminal inference progress bar",
    )
    add_model_arguments(
        parser,
        weights_required=True,
        backbone_required=True,
        include_score_threshold=True,
    )
    return parser.parse_args(argv)


def image_paths(path: Path, recursive: bool):
    """获取输入路径下的所有图像文件。"""
    if path.is_file():
        return [path]
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def synchronize_model(model):
    """等待 CUDA 任务完成，确保计时包含实际执行时间。"""
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_device_request(device: str):
    """显式请求 CUDA 时进行前置检查，禁止静默回退到 CPU。"""
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but is unavailable. Check the NVIDIA driver, "
            "CUDA_VISIBLE_DEVICES, and container GPU mapping."
        )


def model_device_description(model, requested_device: str) -> str:
    """校验模型参数的实际设备并返回便于核验的设备描述。"""
    device = next(model.parameters()).device
    if requested_device == "cuda" and device.type != "cuda":
        raise RuntimeError(f"CUDA was requested, but the model is on {device}")
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)


def load_model(exp):
    """构建模型并加载 checkpoint，返回配置、模型和加载耗时（ms）。"""
    start = time.perf_counter()
    exp = resolve_checkpoint_architecture(exp)
    cfg = build_cfg(exp, training=False)
    model = build_model(cfg)
    model.eval()
    StrictDetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    synchronize_model(model)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return cfg, model, elapsed_ms


def _checkpoint_signature(path):
    """仅读取模型构建前所需的确定性网络架构特征。"""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"WARNING: checkpoint architecture auto-detection unavailable: {error}")
        return {}
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else {}
    if not isinstance(state, dict):
        return {}
    return _state_architecture_signature(state)


def resolve_checkpoint_architecture(exp):
    """从无歧义的检查点特征中解析缺省的架构参数标志。

    显式指定的标志绝不被覆盖。检测到的冲突会在构建模型前报错，
    而缺省的标志仅针对该后端输出的两种架构（3 个标准 Anchor 或 6 个小目标 Anchor）自动适配。
    """
    resolved = dict(exp)
    signature = _checkpoint_signature(resolved.get("weights", ""))
    detected_impl = signature.get("implementation")
    requested_impl = resolved.get("backbone_impl")
    if detected_impl and requested_impl is None:
        resolved["backbone_impl"] = detected_impl
        print(
            f"Checkpoint architecture: detected {detected_impl}; "
            "using it because --backbone-impl was omitted"
        )
    elif detected_impl and requested_impl and requested_impl != detected_impl:
        raise RuntimeError(
            f"Checkpoint architecture is {detected_impl}, but the requested model "
            f"uses {requested_impl}. Pass --backbone-impl {detected_impl} "
            "(or use a checkpoint trained with the requested implementation)."
        )

    anchor_count = signature.get("anchor_count")
    if anchor_count in (3, 6):
        detected_small = anchor_count == 6
        requested_small = resolved.get("small_object_anchors")
        if requested_small is None:
            resolved["small_object_anchors"] = detected_small
            flag = "--small-object-anchors" if detected_small else "--no-small-object-anchors"
            print(
                f"Checkpoint RPN: detected {anchor_count} anchors/location; "
                f"using {flag} because the flag was omitted"
            )
        elif bool(requested_small) != detected_small:
            expected = "--small-object-anchors" if detected_small else "--no-small-object-anchors"
            raise RuntimeError(
                f"Checkpoint RPN head has {anchor_count} anchors/location, but the "
                f"requested model uses the opposite anchor profile. Pass {expected}."
            )
    detected_angle = signature.get("angle_head")
    requested_angle = resolved.get("angle_head")
    if detected_angle and requested_angle is None:
        resolved["angle_head"] = True
        print("Checkpoint angle head: detected; enabling it because --angle-head was omitted")
    elif detected_angle and requested_angle is False:
        raise RuntimeError(
            "Checkpoint contains an angle head, but --no-angle-head was requested. "
            "Use --angle-head with the matching --angle-bins/--angle-period values."
        )
    elif not detected_angle and requested_angle is True:
        raise RuntimeError(
            "--angle-head was requested, but the checkpoint has no angle-head weights. "
            "Use a checkpoint trained with --angle-head or omit the option."
        )
    detected_bins = signature.get("angle_bins")
    requested_bins = resolved.get("angle_bins")
    if detected_bins is not None and requested_bins is None:
        resolved["angle_bins"] = detected_bins
        print(
            f"Checkpoint angle head: detected {detected_bins} bins; "
            "using it because --angle-bins was omitted"
        )
    elif detected_bins is not None and requested_bins is not None and int(requested_bins) != detected_bins:
        raise RuntimeError(
            f"Checkpoint angle head has {detected_bins} bins, but the requested model "
            f"uses {requested_bins}; pass --angle-bins {detected_bins}."
        )
    if detected_angle:
        detected_period = signature.get("angle_period")
        requested_period = resolved.get("angle_period")
        if detected_period is None and requested_period is None:
            raise RuntimeError(
                "Angle checkpoint has no persisted angle-period metadata. "
                "Pass the exact training value explicitly with --angle-period "
                "(legacy mask/PCA models normally use 180; directed annotation "
                "models may use 360)."
            )
        if detected_period is not None and requested_period is None:
            resolved["angle_period"] = detected_period
            print(
                f"Checkpoint angle period: detected {detected_period:g}; "
                "using it because --angle-period was omitted"
            )
        elif (
            detected_period is not None
            and requested_period is not None
            and abs(float(requested_period) - detected_period) > 1e-6
        ):
            raise RuntimeError(
                f"Checkpoint angle period is {detected_period:g}, but the requested "
                f"model uses {float(requested_period):g}; pass --angle-period "
                f"{detected_period:g}."
            )
    return resolved


def infer_instances(model, tensor, height: int, width: int):
    """只执行模型前向，返回实例结果和推理耗时（ms）。"""
    synchronize_model(model)
    start = time.perf_counter()
    with torch.inference_mode():
        instances = model([{"image": tensor, "height": height, "width": width}])[0]["instances"]
    synchronize_model(model)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return instances, elapsed_ms


def complete_thresholds(classes, configured, default=0.5):
    """补全所有类别的阈值字典并验证其合法性。"""
    configured = configured or {}
    unknown = set(configured) - set(classes)
    if unknown:
        raise ValueError("thresholds contain unknown classes: " + ", ".join(sorted(unknown)))
    result = {name: float(configured.get(name, default)) for name in classes}
    if any(not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError("thresholds must be between 0 and 1")
    return result


def apply_angle_output_policy(exp, cfg):
    """在检查点自动检测后保留所请求的方向输出类别策略。

    推理时 ``--angle-head`` 允许缺省，因此该策略必须在
    ``resolve_checkpoint_architecture`` / ``build_cfg`` 之后运行。
    """
    if not bool(getattr(cfg, "ANGLE_HEAD", False)) and exp.get("angle_classes"):
        print(
            "WARNING: --angle-classes was provided but the checkpoint has no angle head; "
            "no angle arrows or angle fields will be generated."
        )
        exp["angle_classes"] = []
    return exp


def apply_class_nms(instances, classes, iou_thresholds, score_thresholds=None):
    """按类别使用独立 IoU 阈值进行第二阶段 NMS。"""
    if len(instances) == 0:
        return instances
    keep = []
    for class_id, class_name in enumerate(classes):
        indices = torch.nonzero(instances.pred_classes == class_id, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        if score_thresholds:
            indices = indices[instances.scores[indices] >= float(score_thresholds[class_name])]
            if indices.numel() == 0:
                continue
        selected = nms(
            instances.pred_boxes.tensor[indices],
            instances.scores[indices],
            float(iou_thresholds[class_name]),
        )
        keep.append(indices[selected])
    if not keep:
        return instances[:0]
    keep = torch.cat(keep)
    keep = keep[instances.scores[keep].argsort(descending=True)]
    return instances[keep]


def apply_class_max_detections(instances, classes, limits):
    """按类别保留最高分的若干实例；未配置类别不受限制。"""
    if len(instances) == 0 or not limits:
        return instances
    keep = []
    for class_id, class_name in enumerate(classes):
        indices = torch.nonzero(instances.pred_classes == class_id, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        limit = limits.get(class_name)
        if limit is not None and indices.numel() > limit:
            order = instances.scores[indices].argsort(descending=True)
            indices = indices[order[:limit]]
        keep.append(indices)
    if not keep:
        return instances[:0]
    keep = torch.cat(keep)
    return instances[keep[instances.scores[keep].argsort(descending=True)]]


def main(argv=None):
    """推理预测主函数。"""
    args = parse_args(argv)
    setup_run_logging("INFER")
    # 在应用角度输出策略前先解析检查点。当省略 --angle-head 时，带角度的检查点必须保留 --angle-classes。
    exp = resolve_checkpoint_architecture(experiment_from_args(args))
    exp["class_thresholds"] = complete_thresholds(
        exp["classes"], exp.get("class_thresholds"), default=0.5
    )
    exp["class_iou_thresholds"] = complete_thresholds(
        exp["classes"], exp.get("class_iou_thresholds"), default=0.5
    )
    # Detectron2 首先使用全局阈值。较宽松的值保留后续按类别过滤和 NMS 所需的所有候选框。
    exp["score_threshold"] = min(exp["class_thresholds"].values())
    exp["nms_threshold"] = max(exp["class_iou_thresholds"].values())
    requested_device = exp.get("device", "auto")
    validate_device_request(requested_device)
    cfg, model, load_ms = load_model(exp)
    exp = apply_angle_output_policy(exp, cfg)
    actual_device = model_device_description(model, requested_device)
    print(f"Inference device: {actual_device}")
    print(f"Model loading time: {load_ms:.2f} ms")
    resize = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    input_root = Path(args.input).expanduser()
    output_root = Path(args.output).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_root}")
    paths = image_paths(input_root, args.recursive)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be greater than zero")
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"No images found: {input_root}")

    inference_times = []
    prediction_count = 0
    from tqdm import tqdm

    progress = tqdm(
        paths,
        desc="Inference",
        unit="image",
        dynamic_ncols=True,
        disable=not args.progress,
        file=sys.__stderr__,
    )
    for path in progress:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        height, width = image.shape[:2]
        # cv2.imread 统一返回 BGR。timm 模型配置为 RGB，而原生模型保持 BGR 以兼容检查点。
        model_image = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if cfg.INPUT.FORMAT == "RGB" else image
        )
        transformed = resize.get_transform(model_image).apply_image(model_image)
        tensor = torch.as_tensor(transformed.astype("float32").transpose(2, 0, 1))
        instances, inference_ms = infer_instances(model, tensor, height, width)
        instances = apply_class_nms(
            instances, exp["classes"], exp["class_iou_thresholds"], exp["class_thresholds"]
        )
        instances = apply_class_max_detections(
            instances, exp["classes"], exp.get("class_max_detections", {})
        )
        inference_times.append(inference_ms)
        progress.set_postfix(ms=f"{inference_ms:.1f}", refresh=False)
        if len(paths) == 1:
            print(f"{path.name} inference time: {inference_ms:.2f} ms")
        rendered, records = draw_predictions(
            image,
            instances,
            exp["classes"],
            exp.get("class_thresholds"),
            alpha=float(exp.get("overlay_alpha", 0.35)),
            angle_classes=exp.get("angle_classes"),
            angle_source=exp.get("angle_source", "mask"),
        )
        prediction_count += len(records)
        relative = path.name if input_root.is_file() else path.relative_to(input_root)
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), rendered)
        if not args.no_json:
            save_json(target.with_suffix(".json"), path, target, records, width=width, height=height)
    if len(inference_times) > 1:
        print(
            f"Total inference time: {sum(inference_times):.2f} ms; "
            f"average: {sum(inference_times) / len(inference_times):.2f} ms/image"
        )
    print(f"Saved {len(inference_times)} image result(s), {prediction_count} predicted instance(s)")
    if inference_times and prediction_count == 0:
        print(
            "WARNING: every image has zero predictions at the configured score thresholds. "
            "Verify that --backbone, --classes, --small-object-anchors and angle-head settings "
            "match training before assessing model quality."
        )


if __name__ == "__main__":
    main()
