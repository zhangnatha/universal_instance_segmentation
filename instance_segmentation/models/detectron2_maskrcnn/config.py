"""Detectron2 Mask R-CNN 配置解析与构建模块。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

# 使用公共包路径，使配置不依赖启动器。
from instance_segmentation.paths import ensure_pretrained, pretrained_path, require_pretrained


# 官方 Detectron2/FAIR 纯骨干网络权重检查点。这些检查点仅初始化特征提取器，
# 而特定任务的 ROI/RPN 预测头仍为用户的目标类别进行随机初始化。
OFFICIAL_PRETRAINED_WEIGHTS = {
    "r50": "r50.pkl",
    "r101": "r101.pkl",
    # Detectron2 0.6 没有 R-152 的 ImageNet 预训练权重条目。R-101 具有相同的
    # stem/stage 通道形状，因此兼容层可正常加载，R-152 额外的 res4 模块保持随机初始化。
    "r152": "r101.pkl",
    "x101": "x101.pkl",
    "x152": "x101.pkl",
}
OFFICIAL_PRETRAINED_URLS = {
    "r50": "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl",
    "r101": "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl",
    "r152": "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl",
    "x101": "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl",
    "x152": "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl",
}

TIMM_BACKBONES = {
    "res2net50": "res2net50_26w_4s",
    "regnety4gf": "regnety_040",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
    "swin_large": "swin_large_patch4_window7_224",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "convnext_large": "convnext_large",
    "convnext_xlarge": "convnext_xlarge",
    "swin_t": "swin_tiny_patch4_window7_224",
    "swin_s": "swin_small_patch4_window7_224",
    "swin_b": "swin_base_patch4_window7_224",
    "swin_l": "swin_large_patch4_window7_224",
}


def official_pretrained_weights(backbone: str) -> str:
    """仅为原生骨干网络返回官方预训练初始化权重。

    R152 刻意映射至官方 R101 权重以进行形状兼容的部分初始化；Detectron2 0.6 目录中无 R152 权重。
    绝不使用项目 output/ 检查点路径。
    """
    try:
        filename = OFFICIAL_PRETRAINED_WEIGHTS[backbone]
        return ensure_pretrained(
            pretrained_path("detectron2", filename),
            url=OFFICIAL_PRETRAINED_URLS.get(backbone),
            hint="Download the matching official Detectron2/FAIR checkpoint into the repository's pretrained/detectron2 directory.",
        )
    except KeyError as error:
        raise ValueError(
            f"Official Detectron2 pretraining is only defined for native backbones; got {backbone!r}"
        ) from error


def add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    weights_required: bool = False,
    backbone_required: bool = False,
    include_weights: bool = True,
    include_score_threshold: bool = True,
) -> None:
    """添加训练、推理、评估和导出共用的模型命令行参数。"""
    parser.add_argument("--classes", nargs="+", required=True, help="Class names in class-ID order")
    parser.add_argument(
        "--backbone",
        choices=[
            "r50", "r101", "r152", "x101", "x152",
            *TIMM_BACKBONES,
        ],
        required=backbone_required,
        default=None if backbone_required else "r101",
        help=(
            "FPN backbone; must exactly match the checkpoint"
            + ("" if backbone_required else " (default: r101)")
        ),
    )
    parser.add_argument(
        "--backbone-impl",
        choices=("timm", "native"),
        default=None,
        help=(
            "Implementation for modern backbones; native is the default for "
            "scratch/project training, while timm is selected for official "
            "pretraining (or may be specified explicitly)"
        ),
    )
    if include_weights:
        parser.add_argument("--weights", required=weights_required, help="PyTorch .pth checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--min-size", type=int, default=480, help="Resized short edge; default: 480")
    parser.add_argument("--max-size", type=int, default=640, help="Maximum resized long edge; default: 640")
    if include_score_threshold:
        parser.add_argument("--score-threshold", type=float, default=0.5, help="Model score threshold")
    parser.add_argument("--max-detections", type=int, default=100, help="Maximum detections per image")
    parser.add_argument(
        "--small-object-anchors", action=argparse.BooleanOptionalAction, default=None,
        help="Use lower-scale FPN anchors, including 16-pixel P2 coverage, while preserving large-object scales",
    )
    parser.add_argument(
        "--angle-head", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Enable or disable the angle head. When omitted for inference, "
            "the setting is inferred from the checkpoint; training defaults to off."
        ),
    )
    # 此处显式设为 None：推理时可区分缺省值与显式的 --no-angle-head 并采用检查点的 bin 数量。build_cfg 为训练提供默认值。
    parser.add_argument("--angle-bins", type=int, default=None, help="Number of angle classification bins")
    parser.add_argument("--angle-period", type=float, default=None, help="Angular range period in degrees (e.g. 180.0 or 360.0)")
    parser.add_argument("--angle-loss-weight", type=float, default=None, help="Loss weight factor for angle head")
    parser.add_argument("--angle-lr-factor", type=float, default=None, help="Learning rate multiplier for angle head parameters")
    parser.add_argument(
        "--angle-label-source", choices=("annotation", "mask"), default="annotation",
        help=(
            "Angle supervision source: annotation uses only explicit LabelMe "
            "angle fields; mask derives an unoriented PCA axis from polygons"
        ),
    )


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """添加可安全暴露在 CLI 中的 Detectron2 训练调优参数。"""
    parser.add_argument(
        "--optimizer", choices=["sgd", "adamw"], default="adamw",
        help="Optimizer (default: AdamW, matching the original Swin training recipe)",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum factor")
    parser.add_argument(
        "--nesterov", action=argparse.BooleanOptionalAction, default=False,
        help="Use Nesterov momentum for SGD",
    )
    parser.add_argument("--bias-lr-factor", type=float, default=1.0, help="Learning rate multiplier for bias parameters")
    parser.add_argument("--weight-decay-norm", type=float, default=0.0, help="Weight decay for normalization layers")
    parser.add_argument("--weight-decay-bias", type=float, help="Weight decay for bias parameters")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="Learning rate decay factor")
    parser.add_argument(
        "--lr-scheduler", choices=["WarmupMultiStepLR", "WarmupCosineLR"],
        default="WarmupMultiStepLR",
        help="Learning rate scheduler type",
    )
    parser.add_argument("--warmup-factor", type=float, default=0.001, help="Warmup starting multiplier")
    parser.add_argument("--warmup-method", choices=["constant", "linear"], default="linear", help="Warmup interpolation method")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1 parameter")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Adam beta2 parameter")
    parser.add_argument("--adam-eps", type=float, default=1e-8, help="Adam epsilon stability factor")
    parser.add_argument(
        "--clip-gradients", action=argparse.BooleanOptionalAction, default=True,
        help="Enable gradient clipping",
    )
    parser.add_argument(
        "--gradient-clip-type", choices=["norm", "value", "full_model"], default="norm",
        help="Type of gradient clipping",
    )
    parser.add_argument("--gradient-clip-value", type=float, help="Threshold value for gradient clipping")
    parser.add_argument("--gradient-clip-norm-type", type=float, default=2.0, help="p-norm type for gradient clipping")

    parser.add_argument(
        "--random-flip", choices=["none", "horizontal", "vertical", "both"],
        default="horizontal", help="Random image flip used during training",
    )
    parser.add_argument(
        "--train-size-sampling", choices=["choice", "range"], default="choice",
        help="How Detectron2 samples the training short edge",
    )
    parser.add_argument("--mask-format", choices=["polygon", "bitmask"], default="polygon", help="Internal mask representation format")
    parser.add_argument(
        "--aspect-ratio-grouping", action=argparse.BooleanOptionalAction, default=True,
        help="Group images with similar aspect ratio into the same batch",
    )
    parser.add_argument(
        "--filter-empty-annotations", action=argparse.BooleanOptionalAction, default=True,
        help="Filter out training images without instances",
    )
    parser.add_argument(
        "--sampler-train", choices=["TrainingSampler", "RepeatFactorTrainingSampler"],
        default="TrainingSampler",
        help="Training dataset sampler",
    )
    parser.add_argument("--repeat-sqrt", action=argparse.BooleanOptionalAction, default=True, help="Take square root of repeat factor in sampler")
    parser.add_argument("--repeat-threshold", type=float, default=0.0, help="Category frequency threshold for RepeatFactorTrainingSampler")

    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold for RoI heads")
    parser.add_argument("--roi-batch-size-per-image", type=int, default=512, help="RoI batch size per image")
    parser.add_argument("--roi-positive-fraction", type=float, default=0.25, help="Positive proposal fraction for RoI heads")
    parser.add_argument("--rpn-batch-size-per-image", type=int, default=256, help="RPN anchor batch size per image")
    parser.add_argument("--rpn-positive-fraction", type=float, default=0.5, help="Positive fraction of anchors for RPN")
    parser.add_argument("--rpn-nms-threshold", type=float, default=0.7, help="NMS threshold used by RPN")
    parser.add_argument("--rpn-pre-nms-topk-train", type=int, default=12000, help="RPN pre-NMS proposals to keep during training")
    parser.add_argument("--rpn-post-nms-topk-train", type=int, default=2000, help="RPN post-NMS proposals to keep during training")
    parser.add_argument("--rpn-pre-nms-topk-test", type=int, default=6000, help="RPN pre-NMS proposals to keep during inference")
    parser.add_argument("--rpn-post-nms-topk-test", type=int, default=1000, help="RPN post-NMS proposals to keep during inference")
    parser.add_argument(
        "--focus-crop-class",
        help="Class to center occasional object-aware training crops on, e.g. class_name",
    )
    parser.add_argument(
        "--focus-crop-classes", nargs="+", default=None,
        help="Classes to center occasional object-aware training crops on, e.g. class1 class2",
    )
    parser.add_argument(
        "--focus-crop-prob", type=float, default=0.0,
        help="Probability of an object-aware crop per training image; disabled by default",
    )
    parser.add_argument(
        "--focus-crop-scale", type=float, default=4.0,
        help="Approximate crop side multiplier around the selected object's box",
    )
    parser.add_argument(
        "--focus-crop-min-size", nargs=2, type=int, default=[320, 240], metavar=("WIDTH", "HEIGHT"),
        help="Minimum width/height of an object-aware crop before resizing",
    )
    parser.add_argument(
        "--augment-clahe", action=argparse.BooleanOptionalAction, default=False,
        help="Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)",
    )
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE contrast clip limit")
    parser.add_argument("--clahe-prob", type=float, default=0.3, help="Probability of applying CLAHE augmentation")
    parser.add_argument(
        "--augment-erasing", action=argparse.BooleanOptionalAction, default=False,
        help="Randomly erase small rectangular patches to simulate occlusions",
    )
    parser.add_argument("--erasing-prob", type=float, default=0.25, help="Probability of applying random erasing")
    parser.add_argument("--erasing-smin", type=float, default=0.01, help="Minimum erasing area ratio")
    parser.add_argument("--erasing-smax", type=float, default=0.04, help="Maximum erasing area ratio")
    parser.add_argument("--freeze-at", type=int, help="Backbone stages to freeze; default follows initialization mode")
    parser.add_argument("--resnet-norm", choices=["FrozenBN", "BN", "SyncBN"], help="Normalization layer type for ResNet backbone")


def parse_class_thresholds(
    items: list[str] | None,
    classes: list[str],
    *,
    argument: str = "class-thresholds",
) -> dict[str, float]:
    """解析 ``类别=阈值`` 参数，并尽早报告拼写或范围错误。"""
    thresholds: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --{argument} value: {item}; expected CLASS=VALUE")
        name, raw_value = item.split("=", 1)
        if name not in classes:
            raise SystemExit(f"--{argument} contains an unknown class: {name}")
        try:
            value = float(raw_value)
        except ValueError as error:
            raise SystemExit(f"--{argument} value for {name} is not numeric: {raw_value}") from error
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{argument} value for {name} must be between 0 and 1")
        if name in thresholds:
            raise SystemExit(f"--{argument} contains duplicate class: {name}")
        if argument == "class-iou-thresholds" and value <= 0.0:
            raise SystemExit(f"--{argument} value for {name} must be greater than 0")
        thresholds[name] = value
    return thresholds


def parse_class_limits(
    items: list[str] | None,
    classes: list[str],
    *,
    argument: str = "class-max-detections",
) -> dict[str, int]:
    """解析 ``类别=整数`` 参数，用于可选的每类实例数量上限。"""
    limits: dict[str, int] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --{argument} value: {item}; expected CLASS=INTEGER")
        name, raw_value = item.split("=", 1)
        if name not in classes:
            raise SystemExit(f"--{argument} contains an unknown class: {name}")
        try:
            value = int(raw_value)
        except ValueError as error:
            raise SystemExit(f"--{argument} value for {name} is not an integer: {raw_value}") from error
        if value <= 0:
            raise SystemExit(f"--{argument} value for {name} must be greater than zero")
        limits[name] = value
    return limits


def experiment_from_args(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    """将命令行命名空间转换成现有模型构建代码使用的字典。"""
    exp = {
        key: value for key, value in vars(args).items()
        if value is not None and key not in {"class_thresholds", "class_iou_thresholds"}
    }
    for key in ("train_dir", "val_dir", "hard_negative_dir", "weights", "output_dir"):
        value = exp.get(key)
        if value:
            # 保持面向用户的路径可移植性。Detectron2 支持相对路径，
            # 在此解析绝对路径会将宿主机路径泄漏到日志和生成的配置文件中。
            exp[key] = str(Path(value).expanduser())
    if hasattr(args, "class_thresholds"):
        exp["class_thresholds"] = parse_class_thresholds(args.class_thresholds, args.classes)
    if hasattr(args, "class_iou_thresholds"):
        exp["class_iou_thresholds"] = parse_class_thresholds(
            args.class_iou_thresholds, args.classes, argument="class-iou-thresholds"
        )
    if hasattr(args, "class_max_detections"):
        exp["class_max_detections"] = parse_class_limits(
            args.class_max_detections, args.classes
        )
    exp.update(extra)
    return exp


def build_cfg(exp: dict[str, Any], *, training: bool):
    """根据实验字典配置构建 Detectron2 CfgNode 对象。"""
    # 导入会触发 AngleROIHeads 注册。
    from instance_segmentation.models.detectron2_maskrcnn import angle_head  # noqa: F401
    from instance_segmentation.models.detectron2_maskrcnn import backbones as merged_backbones
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.MODEL.PRETRAINED_PATH = ""
    backbone = merged_backbones.canonical_backbone(exp.get("backbone", "r101"))
    # 为所有原生 ResNet/ResNeXt 变体保留统一的 Detectron2 model-zoo 入口点。
    # R152/X101/X152 在 Detectron2 0.6 中没有独立的 YAML；其架构由 RESNETS 选项确定并由相同的原生构建器构建。
    native_backbones = {
        "r50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "r101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "r152": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "x101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "x152": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    }
    model_config = native_backbones.get(
        backbone, "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    backbone_overrides = {
        "r50": {"DEPTH": 50, "NUM_GROUPS": 1, "WIDTH_PER_GROUP": 64, "STRIDE_IN_1X1": True},
        "r101": {"DEPTH": 101, "NUM_GROUPS": 1, "WIDTH_PER_GROUP": 64, "STRIDE_IN_1X1": True},
        "r152": {"DEPTH": 152, "NUM_GROUPS": 1, "WIDTH_PER_GROUP": 64, "STRIDE_IN_1X1": True},
        "x101": {"DEPTH": 101, "NUM_GROUPS": 32, "WIDTH_PER_GROUP": 8, "STRIDE_IN_1X1": False},
        "x152": {"DEPTH": 152, "NUM_GROUPS": 32, "WIDTH_PER_GROUP": 8, "STRIDE_IN_1X1": False},
    }
    if backbone in backbone_overrides:
        for key, value in backbone_overrides[backbone].items():
            setattr(cfg.MODEL.RESNETS, key, value)
        # --backbone-impl 仅用于选择现代骨干网络的实现方式。ResNet/ResNeXt 始终使用 Detectron2 原生构建器。
        backbone_impl = "native"
    else:
        # 迁移的原生构建器是现代骨干网络的标准实现。保持 timm 可作为显式兼容性选项。
        # 官方现代骨干网络权重为 timm 检查点，因此请求预训练时缺省实现解析为 timm；从零训练默认使用 native。
        backbone_impl = exp.get("backbone_impl")
        if backbone_impl is None:
            backbone_impl = "timm" if exp.get("pretrained_backbone") else "native"
        if exp.get("pretrained_backbone") and backbone_impl == "native":
            raise ValueError(
                "--pretrained-backbone for modern backbones requires "
                "--backbone-impl timm; native backbones support scratch or "
                "project-checkpoint initialization only"
            )
        if backbone_impl == "native":
            # 合并的多骨干网络实现是标准原生路径；timm 仍作为显式选项保留。
            info = merged_backbones.get_backbone_info(backbone)
            merged_backbones.register_backbones()
            cfg.MODEL.BACKBONE.NAME = info["builder"]
            cfg.MODEL.PRETRAINED = bool(exp.get("pretrained_backbone", False))
        else:
            # 延迟导入：此模块会注册 timm Detectron2 构建器，原生配置不应引入或修改 timm 注册表和配置模式。
            from instance_segmentation.models.detectron2_maskrcnn import timm_backbone  # noqa: F401
            from detectron2.config import CfgNode

            cfg.MODEL.BACKBONE.NAME = "build_timm_fpn_backbone"
            cfg.MODEL.TIMM = CfgNode()
            cfg.MODEL.TIMM.NAME = TIMM_BACKBONES[backbone]
            cfg.MODEL.TIMM.PRETRAINED = False
            cfg.MODEL.TIMM.PRETRAINED_PATH = ""
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(exp["classes"])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(exp.get("score_threshold", 0.5))
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(exp.get("nms_threshold", 0.5))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(exp.get("roi_batch_size_per_image", 512))
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = float(exp.get("roi_positive_fraction", 0.25))
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = int(exp.get("rpn_batch_size_per_image", 256))
    cfg.MODEL.RPN.POSITIVE_FRACTION = float(exp.get("rpn_positive_fraction", 0.5))
    cfg.MODEL.RPN.NMS_THRESH = float(exp.get("rpn_nms_threshold", 0.7))
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = int(exp.get("rpn_pre_nms_topk_train", 12000))
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = int(exp.get("rpn_pre_nms_topk_test", 6000))
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = int(exp.get("rpn_post_nms_topk_train", 2000))
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = int(exp.get("rpn_post_nms_topk_test", 1000))
    if exp.get("small_object_anchors", False):
        # Detectron2 0.6 要求每个 FPN 层级的 Anchor 数量相同。保留各标准尺度并在各层增加一个较小尺度；
        # 为 P2 引入 16px 覆盖，同时保留 P6 上的 512px 大目标 Anchor。
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
            [16, 32], [32, 64], [64, 128], [128, 256], [256, 512]
        ]
    cfg.TEST.DETECTIONS_PER_IMAGE = int(exp.get("max_detections", 100))
    min_size = int(exp.get("min_size", 480))
    max_size = int(exp.get("max_size", 640))
    if exp.get("multi_scale", False):
        # Detectron2 每张图像采样一个短边尺寸。保持 32 像素步长并包含两端，与 Multi-Scale 选项保持一致。
        scales = tuple(range(min_size, max_size + 1, 32))
        if not scales or scales[-1] != max_size:
            scales = scales + (max_size,)
        cfg.INPUT.MIN_SIZE_TRAIN = scales
    else:
        cfg.INPUT.MIN_SIZE_TRAIN = (min_size,)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = exp.get("train_size_sampling", "choice")
    cfg.INPUT.MIN_SIZE_TEST = int(exp.get("min_size", 480))
    cfg.INPUT.MAX_SIZE_TRAIN = int(exp.get("max_size", 640))
    cfg.INPUT.MAX_SIZE_TEST = int(exp.get("max_size", 640))
    # 原生骨干网络保持其原有的 BGR 输入约定。timm ImageNet 检查点使用 RGB，
    # 因此让 Detectron2 直接读取 RGB 并在进入 timm 适配器前应用匹配的 RGB 归一化。
    if backbone_impl == "timm":
        cfg.INPUT.FORMAT = "RGB"
        cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]
        cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]
    else:
        cfg.INPUT.FORMAT = "BGR"
        if backbone not in native_backbones:
            # 保留原生检查点归一化参数约定；原生 ResNet/ResNeXt 均值和方差来自 COCO YAML。
            cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]
            cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]
    cfg.INPUT.RANDOM_FLIP = exp.get("random_flip", "none")
    cfg.INPUT.MASK_FORMAT = exp.get("mask_format", "polygon")
    configured_weights = exp.get("weights", "")
    if exp.get("pretrained_backbone"):
        if configured_weights:
            raise ValueError(
                "--pretrained-backbone cannot be combined with a project checkpoint"
            )
        if backbone in native_backbones:
            configured_weights = official_pretrained_weights(backbone)
        else:
            if backbone_impl != "timm":
                # 作为防御性断言保留，以备未来重构实现选择逻辑时使用。
                raise ValueError(
                    "official pretraining for modern backbones requires the timm implementation"
                )
            model_name = TIMM_BACKBONES[backbone]
            try:
                import timm
                timm_cfg = timm.get_pretrained_cfg(model_name)
                timm_url = getattr(timm_cfg, "url", None)
                if timm_url is None and isinstance(timm_cfg, dict):
                    timm_url = timm_cfg.get("url")
            except Exception:
                timm_url = None
            local_path = ensure_pretrained(
                pretrained_path("timm", f"{model_name}.pth"),
                url=timm_url,
                hint="Download the matching official timm checkpoint into the repository's pretrained/timm directory.",
            )
            cfg.MODEL.TIMM.PRETRAINED = True
            cfg.MODEL.TIMM.PRETRAINED_PATH = local_path
            cfg.MODEL.PRETRAINED_PATH = local_path
    cfg.MODEL.WEIGHTS = configured_weights
    # COCO 模板默认使用 FrozenBN，它假设一定会加载已经归一化好的预训练
    # backbone。若在没有权重时从零训练，FrozenBN 实际是恒等变换，R101 的
    # 特征会在首次前向中膨胀到 1e7 量级并让 RPN 在第二步产生 Inf/NaN。
    # 真正从零训练时启用可学习 BN 且不冻结 stem/res2；保存后的 BN 统计量可
    # 由推理阶段的 FrozenBN 等价读取，因而不改变归档模型的推理框架。
    has_pretraining = bool(cfg.MODEL.WEIGHTS) or (
        hasattr(cfg.MODEL, "TIMM") and bool(cfg.MODEL.TIMM.PRETRAINED)
    )
    if training and not has_pretraining:
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
    # Detectron2 没有完整的 R152 权重：超出 R101 的模块保持随机初始化。
    # FrozenBN 对这些模块是恒等变换并可能导致其激活不稳定，因此 R152 部分初始化必须学习 BN 统计量。
    if training and backbone == "r152" and exp.get("pretrained_backbone"):
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
    if exp.get("resnet_norm"):
        cfg.MODEL.RESNETS.NORM = exp["resnet_norm"]
    if exp.get("freeze_at") is not None:
        cfg.MODEL.BACKBONE.FREEZE_AT = int(exp["freeze_at"])
    cfg.MODEL.DEVICE = (
        "cuda" if exp.get("device", "auto") == "auto" and torch.cuda.is_available()
        else "cpu" if exp.get("device", "auto") == "auto"
        else exp["device"]
    )
    cfg.OUTPUT_DIR = exp.get("output_dir", "output/training")
    cfg.SEED = int(exp.get("seed", 42))
    cfg.DATALOADER.NUM_WORKERS = int(exp.get("num_workers", 4))
    # 这些参数供 ReproTrainer 的数据加载器适配器使用。
    # 保持在 Detectron2 0.6 配置模式之外并在调用处过滤不支持的关键字参数。
    cfg.DATALOADER.PIN_MEMORY = exp.get("pin_memory")
    cfg.DATALOADER.PERSISTENT_WORKERS = bool(exp.get("persistent_workers", True))
    cfg.DATALOADER.PREFETCH_FACTOR = int(exp.get("prefetch_factor", 2))
    cfg.THROUGHPUT_TF32 = bool(exp.get("tf32", True))
    cfg.THROUGHPUT_CUDNN_BENCHMARK = bool(exp.get("cudnn_benchmark", True))
    cfg.THROUGHPUT_CHANNELS_LAST = bool(exp.get("channels_last", False))
    cfg.SOLVER.IMS_PER_BATCH = int(exp.get("batch_size", 4))
    cfg.SOLVER.BASE_LR = float(exp.get("base_lr", 1e-4))
    cfg.SOLVER.MAX_ITER = int(exp.get("max_iter", 20000))
    cfg.SOLVER.CHECKPOINT_PERIOD = int(exp.get("checkpoint_period", 1000))
    max_iter = int(exp.get("max_iter", 20000))
    configured_steps = exp.get("lr_steps")
    if configured_steps is None:
        configured_steps = [int(max_iter * 0.7), int(max_iter * 0.9)]
    cfg.SOLVER.STEPS = tuple(step for step in configured_steps if 0 < int(step) < max_iter)
    cfg.SOLVER.WARMUP_ITERS = int(exp.get("warmup_iters", 1000))
    cfg.SOLVER.WARMUP_FACTOR = float(exp.get("warmup_factor", 0.001))
    cfg.SOLVER.WARMUP_METHOD = exp.get("warmup_method", "linear")
    cfg.SOLVER.GAMMA = float(exp.get("lr_gamma", 0.1))
    cfg.SOLVER.LR_SCHEDULER_NAME = exp.get("lr_scheduler", "WarmupMultiStepLR")
    cfg.SOLVER.MOMENTUM = float(exp.get("momentum", 0.9))
    cfg.SOLVER.NESTEROV = bool(exp.get("nesterov", False))
    cfg.SOLVER.BIAS_LR_FACTOR = float(exp.get("bias_lr_factor", 1.0))
    cfg.SOLVER.WEIGHT_DECAY_NORM = float(exp.get("weight_decay_norm", 0.0))
    if exp.get("weight_decay_bias") is not None:
        cfg.SOLVER.WEIGHT_DECAY_BIAS = float(exp["weight_decay_bias"])
    cfg.SOLVER.WEIGHT_DECAY = float(exp.get("weight_decay", 0.05))
    cfg.SOLVER.AMP.ENABLED = bool(exp.get("amp", False))
    cfg.MODEL.ROI_HEADS.NAME = "AngleROIHeads" if exp.get("angle_head", False) else "StandardROIHeads"
    cfg.ANGLE_HEAD = cfg.MODEL.ROI_HEADS.NAME == "AngleROIHeads"
    cfg.ANGLE_BINS = int(exp.get("angle_bins", 72))
    cfg.ANGLE_PERIOD = float(exp.get("angle_period", 360.0))
    cfg.ANGLE_LOSS_WEIGHT = float(exp.get("angle_loss_weight", 1.0))
    cfg.ANGLE_LR_FACTOR = float(exp.get("angle_lr_factor", 0.01))
    cfg.ANGLE_LABEL_SOURCE = str(exp.get("angle_label_source", "annotation"))
    if cfg.ANGLE_LABEL_SOURCE not in {"annotation", "mask"}:
        raise ValueError("angle_label_source must be 'annotation' or 'mask'")
    if training and exp.get("angle_head") and cfg.ANGLE_LABEL_SOURCE == "mask" and cfg.ANGLE_PERIOD != 180.0:
        raise ValueError(
            "mask/PCA angle labels are unoriented and require ANGLE_PERIOD=180; "
            "use angle_label_source='annotation' for directed 360-degree labels"
        )
    angle_classes = exp.get("angle_classes")
    if isinstance(angle_classes, str):
        angle_classes = [angle_classes]
    if exp.get("angle_head"):
        angle_classes = list(exp["classes"] if angle_classes is None else angle_classes)
    else:
        angle_classes = []
    unknown_angle_classes = set(angle_classes) - set(exp["classes"])
    if unknown_angle_classes:
        raise ValueError(
            "Angle classes must be included in --classes: "
            + ", ".join(sorted(unknown_angle_classes))
        )
    cfg.ANGLE_CLASS_IDS = tuple(exp["classes"].index(name) for name in angle_classes)
    cfg.AUGMENT_ROTATION = bool(exp.get("augment_rotation", True))
    cfg.AUGMENT_ROTATION_RANGE = tuple(exp.get("rotation_range", [-5.0, 5.0]))
    cfg.AUGMENT_ROTATION_PROB = float(exp.get("rotation_prob", 0.5))
    cfg.AUGMENT_BRIGHTNESS = bool(exp.get("augment_brightness", True))
    cfg.AUGMENT_BRIGHTNESS_RANGE = tuple(exp.get("brightness_range", [0.8, 1.2]))
    cfg.AUGMENT_BRIGHTNESS_PROB = float(exp.get("brightness_prob", 0.5))
    cfg.AUGMENT_CONTRAST = bool(exp.get("augment_contrast", True))
    cfg.AUGMENT_CONTRAST_RANGE = tuple(exp.get("contrast_range", [0.7, 1.3]))
    cfg.AUGMENT_CONTRAST_PROB = float(exp.get("contrast_prob", 0.5))
    cfg.AUGMENT_TRANSLATION = bool(exp.get("augment_translation", True))
    cfg.AUGMENT_TRANSLATION_RANGE = tuple(exp.get("translation_range", [-0.2, 0.2]))
    cfg.AUGMENT_TRANSLATION_PROB = float(exp.get("translation_prob", 0.5))
    cfg.AUGMENT_LOW_RESOLUTION = bool(exp.get("augment_low_resolution", True))
    cfg.AUGMENT_LOW_RESOLUTION_SCALE = float(exp.get("low_resolution_scale", 0.6))
    cfg.AUGMENT_LOW_RESOLUTION_PROB = float(exp.get("low_resolution_prob", 0.5))
    cfg.AUGMENT_CLAHE = bool(exp.get("augment_clahe", False))
    cfg.AUGMENT_CLAHE_CLIP_LIMIT = float(exp.get("clahe_clip_limit", 2.0))
    cfg.AUGMENT_CLAHE_PROB = float(exp.get("clahe_prob", 0.3))
    cfg.AUGMENT_ERASING = bool(exp.get("augment_erasing", False))
    cfg.AUGMENT_ERASING_SMIN = float(exp.get("erasing_smin", 0.01))
    cfg.AUGMENT_ERASING_SMAX = float(exp.get("erasing_smax", 0.04))
    cfg.AUGMENT_ERASING_PROB = float(exp.get("erasing_prob", 0.25))
    copy_paste_classes = tuple(exp.get("copy_paste_classes") or exp["classes"])
    copy_paste_enabled = bool(exp.get("augment_copy_paste", False))
    unknown_copy_paste_classes = set(copy_paste_classes) - set(exp["classes"])
    if copy_paste_enabled and unknown_copy_paste_classes:
        raise ValueError(
            "Copy-Paste classes must be included in --classes: "
            + ", ".join(sorted(unknown_copy_paste_classes))
        )
    cfg.AUGMENT_COPY_PASTE = copy_paste_enabled
    cfg.AUGMENT_COPY_PASTE_PROB = float(exp.get("copy_paste_prob", 0.25))
    cfg.AUGMENT_COPY_PASTE_CLASS_IDS = tuple(
        exp["classes"].index(name) for name in copy_paste_classes
        if name in exp["classes"]
    )
    cfg.AUGMENT_COPY_PASTE_MAX_INSTANCES = int(exp.get("copy_paste_max_instances", 2))
    cfg.AUGMENT_COPY_PASTE_MAX_BBOX_IOU = float(exp.get("copy_paste_max_bbox_iou", 0.05))
    cfg.SMALL_OBJECT_ANCHORS = bool(exp.get("small_object_anchors", False))
    focus_classes = exp.get("focus_crop_classes")
    if focus_classes is None and exp.get("focus_crop_class"):
        focus_classes = [exp["focus_crop_class"]]
    if focus_classes:
        cfg.FOCUS_CROP_CLASSES = tuple(c for c in focus_classes if c in exp["classes"])
        cfg.FOCUS_CROP_CLASS_IDS = tuple(
            exp["classes"].index(c) for c in focus_classes if c in exp["classes"]
        )
    else:
        cfg.FOCUS_CROP_CLASSES = ()
        cfg.FOCUS_CROP_CLASS_IDS = ()
    cfg.FOCUS_CROP_CLASS = exp.get("focus_crop_class") or ""
    cfg.FOCUS_CROP_CLASS_ID = (
        int(exp["classes"].index(exp["focus_crop_class"]))
        if exp.get("focus_crop_class") in exp["classes"] else -1
    )
    cfg.FOCUS_CROP_PROB = float(exp.get("focus_crop_prob", 0.0))
    cfg.FOCUS_CROP_SCALE = float(exp.get("focus_crop_scale", 4.0))
    cfg.FOCUS_CROP_MIN_SIZE = tuple(int(value) for value in exp.get("focus_crop_min_size", [320, 240]))
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = bool(exp.get("clip_gradients", True))
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = exp.get("gradient_clip_type", "norm")
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = float(
        exp.get("gradient_clip_value", exp.get("gradient_clip_norm", 5.0))
    )
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = float(exp.get("gradient_clip_norm_type", 2.0))
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = bool(exp.get("aspect_ratio_grouping", True))
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = bool(exp.get("filter_empty_annotations", True))
    # 挖掘出的难负样本是有意创建的空 LabelMe 记录。在训练加载器中保留它们，
    # 其他情况保持原有默认行为。
    if training and exp.get("hard_negative_dir"):
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.SAMPLER_TRAIN = exp.get("sampler_train", "TrainingSampler")
    cfg.DATALOADER.REPEAT_SQRT = bool(exp.get("repeat_sqrt", True))
    cfg.DATALOADER.REPEAT_THRESHOLD = float(exp.get("repeat_threshold", 0.0))
    if training:
        experiment = exp.get("experiment", "dataset")
        cfg.DATASETS.TRAIN = (f"{experiment}_train",)
        cfg.DATASETS.TEST = ((f"{experiment}_val",) if exp.get("val_dir") else ())
        cfg.TEST.EVAL_PERIOD = int(exp.get("eval_period", 500))
    cfg.freeze()
    return cfg
