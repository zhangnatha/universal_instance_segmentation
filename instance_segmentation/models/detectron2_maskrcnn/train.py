#!/usr/bin/env python3
"""Detectron2 Mask R-CNN 训练脚本，支持多种骨干网络、自动切分、难负样本挖掘与早停机制。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch

from .config import (
    add_model_arguments,
    add_training_arguments,
    build_cfg,
    experiment_from_args,
)
from instance_segmentation.data.labelme import (
    ensure_disjoint_datasets,
    ensure_hard_negative_disjoint,
    missing_classes,
    register_labelme,
    split_labelme_dataset,
    validate_labelme,
    write_split_manifest,
)
from .run_logging import setup_run_logging
from .gpu_health import explain_cuda_failure, require_gpu_power_limit
from ...paths import PRETRAINED_ROOT, REPOSITORY_ROOT

CODE_ROOT = REPOSITORY_ROOT
TRAINING_OUTPUT_ROOT = CODE_ROOT / "output"
PRETRAINED_MODEL_ROOT = PRETRAINED_ROOT


def sanitize_download_proxy_environment() -> list[str]:
    """规范化下载代理环境变量以适配 httpx 与 Hugging Face。"""
    notices = []
    has_http_proxy = any(
        os.environ.get(name)
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
    )
    for name in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(name, "")
        if not value.lower().startswith("socks://"):
            continue
        if has_http_proxy:
            # httpx 在存在 HTTPS 专用代理时仍会解析 ALL_PROXY，并在请求前拒绝过时的 socks:// 协议。
            # 仅移除 ALL_PROXY 可使 HTTP(S)_PROXY 正常生效。
            os.environ.pop(name, None)
            notices.append(
                f"Ignored {name}=socks://...; using configured HTTP(S)_PROXY for downloads"
            )
        else:
            raise SystemExit(
                f"{name} uses unsupported socks:// syntax. Change it to socks5:// "
                "and install SOCKS support, or configure HTTP_PROXY/HTTPS_PROXY."
            )
    return notices


def configure_pretrained_cache() -> Path:
    """将所有支持的模型下载器缓存路径定向到本地目录。"""
    for notice in sanitize_download_proxy_environment():
        print(f"Download proxy: {notice}")
    root = PRETRAINED_MODEL_ROOT.resolve()
    directories = {
        "TORCH_HOME": root / "torch",
        "FVCORE_CACHE": root / "detectron2",
        "HF_HOME": root / "huggingface",
        "HF_HUB_CACHE": root / "huggingface" / "hub",
    }
    for variable, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        # 此处刻意不使用 setdefault：下载的骨干网络权重必须存放在当前仓库中，即使环境中配置了全局缓存。
        os.environ[variable] = str(directory)
    return root


def probability(value: str) -> float:
    """验证概率值参数（[0, 1] 区间浮点数）。"""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("probability must be between 0 and 1")
    return parsed


def split_ratio(value: str) -> float:
    """验证数据集切分比例参数（(0, 1) 区间浮点数）。"""
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("split ratio must be greater than 0 and less than 1")
    return parsed


def training_output_dir(value: str) -> str:
    """将所有训练产物限制在项目 output/ 目录内。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        result = path
    else:
        parts = path.parts
        if parts and parts[0] in {"output", "results"}:
            parts = parts[1:]
        result = Path("output").joinpath(*parts)
        resolved = (CODE_ROOT / result).resolve()
    try:
        resolved.relative_to(TRAINING_OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "training output must be inside the project output/ directory"
        ) from error
    return str(result)


def validate_class_coverage(train_stats, val_stats, classes):
    """验证训练集与验证集的类别覆盖情况。"""
    missing_train = missing_classes(train_stats, classes)
    missing_val = missing_classes(val_stats, classes) if val_stats else []
    if missing_train:
        raise SystemExit(
            "Class configuration does not match LabelMe annotations; training stopped. "
            f"Training set is missing valid instances for: {', '.join(missing_train)}. "
            "Please fix --classes or add annotations; otherwise these classes cannot be learned."
        )
    if missing_val:
        print(
            "WARNING: Validation set is missing valid instances for: "
            f"{', '.join(missing_val)}. Training continues; validation COCO AP "
            "for these classes cannot be evaluated and may show as NaN/NA in reports.",
            file=sys.stderr,
        )


def parse_args(argv=None):
    """解析训练命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN R101-FPN instance segmentation model")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir/last_checkpoint")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--weights", help="Pretrained .pth checkpoint for transfer learning")
    initialization.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help=(
            "Initialize only the backbone from official Detectron2/FAIR/timm "
            "ImageNet weights; never uses a project-trained checkpoint"
        ),
    )
    initialization.add_argument("--from-scratch", action="store_true", help="Do not load pretrained weights")
    parser.add_argument("--experiment", default="training", help="Dataset registration name")
    parser.add_argument(
        "--train-dir", required=True,
        help="LabelMe training dataset directory, or the complete source directory with --auto-split",
    )
    parser.add_argument(
        "--hard-negative-dir",
        help=(
            "Optional directory of empty LabelMe background crops generated only from "
            "--train-dir predictions; validation/test sources are rejected"
        ),
    )
    parser.add_argument(
        "--hard-negative-repeat", type=int, default=1,
        help="Repeat the mined hard-negative records in the training sampler (default: 1)",
    )
    split_mode = parser.add_mutually_exclusive_group()
    split_mode.add_argument("--val-dir", help="LabelMe validation directory; never used for updates")
    split_mode.add_argument(
        "--auto-split", action="store_true",
        help="Split --train-dir into training and validation samples without copying files",
    )
    parser.add_argument(
        "--val-ratio", type=split_ratio, default=0.2,
        help="Validation ratio used by --auto-split (default: 0.2)",
    )
    parser.add_argument(
        "--split-seed", type=int, default=None,
        help="Random seed used by --auto-split; defaults to --seed",
    )
    parser.add_argument(
        "--output-dir", type=training_output_dir, default=training_output_dir("training"),
        help="Training artifact directory under project output/; default: output/training",
    )
    parser.add_argument("--max-iter", type=int, default=20000, help="Maximum number of training iterations")
    parser.add_argument("--base-lr", type=float, default=None, help="Base learning rate; derived from batch size when omitted")
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-GPU batch size; 4 is the recommended AMP setting for a 12 GB GPU",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="CPU data-loader workers; 4 keeps CUDA fed without excessive host memory use",
    )
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=None,
        help="Pin loader host memory (default: enabled automatically for CUDA)",
    )
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True,
        help="Keep DataLoader workers alive between epochs when workers > 0",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="Batches prefetched per DataLoader worker; ignored when workers=0",
    )
    parser.add_argument(
        "--tf32", action=argparse.BooleanOptionalAction, default=True,
        help="Allow TF32 matrix/convolution kernels on supported NVIDIA GPUs",
    )
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True,
        help="Benchmark cuDNN kernels for the configured image-size workload",
    )
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=False,
        help="Reserved; Detectron2 0.6 per-image CHW input is not converted unsafely",
    )
    parser.add_argument("--checkpoint-period", type=int, default=1000, help="Save a checkpoint every N iterations")
    parser.add_argument("--eval-period", type=int, default=500, help="Run validation evaluation every N iterations")
    parser.add_argument(
        "--save-best", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Save the checkpoint with the best validation metric as model_best.pth "
            "(default: enabled; use --no-save-best to disable)"
        ),
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="Stop after this many validation evaluations without improvement; 0 disables",
    )
    parser.add_argument(
        "--early-stop-metric", default="segm/AP",
        help="Validation metric monitored by --save-best/--early-stop-patience (default: segm/AP)",
    )
    parser.add_argument("--lr-steps", nargs="*", type=int, default=None, help="Iteration milestones at which learning rate decays")
    parser.add_argument("--warmup-iters", type=int, default=200, help="Number of warmup iterations")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay factor")
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0, help="Maximum gradient norm threshold")
    parser.add_argument(
        "--multi-scale", action=argparse.BooleanOptionalAction, default=False,
        help="Randomly choose training resize short edges between min-size and max-size",
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use automatic mixed precision to improve CUDA training throughput",
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True,
        help="Show a terminal training progress bar",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--augment", action=argparse.BooleanOptionalAction, default=True,
        help="Enable or disable training augmentation",
    )
    parser.add_argument(
        "--augment-rotation", action=argparse.BooleanOptionalAction, default=False,
        help="Enable random rotation augmentation",
    )
    parser.add_argument("--rotation-range", nargs=2, type=float, default=[-5.0, 5.0], help="Random rotation angle range in degrees")
    parser.add_argument("--rotation-prob", type=probability, default=0.5, help="Probability of applying random rotation")
    parser.add_argument(
        "--augment-brightness", action=argparse.BooleanOptionalAction, default=True,
        help="Enable random brightness augmentation",
    )
    parser.add_argument("--brightness-range", nargs=2, type=float, default=[0.85, 1.15], help="Brightness factor range")
    parser.add_argument("--brightness-prob", type=probability, default=0.5, help="Probability of applying random brightness")
    parser.add_argument(
        "--augment-contrast", action=argparse.BooleanOptionalAction, default=True,
        help="Enable random contrast augmentation",
    )
    parser.add_argument("--contrast-range", nargs=2, type=float, default=[0.85, 1.15], help="Contrast factor range")
    parser.add_argument("--contrast-prob", type=probability, default=0.5, help="Probability of applying random contrast")
    parser.add_argument(
        "--augment-translation", action=argparse.BooleanOptionalAction, default=False,
        help="Enable random translation augmentation",
    )
    parser.add_argument("--translation-range", nargs=2, type=float, default=[-0.2, 0.2], help="Translation fraction range")
    parser.add_argument("--translation-prob", type=probability, default=0.5, help="Probability of applying random translation")
    parser.add_argument(
        "--augment-low-resolution", action=argparse.BooleanOptionalAction, default=False,
        help="Enable low-resolution augmentation",
    )
    parser.add_argument("--low-resolution-scale", type=float, default=0.6, help="Scale factor for low-resolution augmentation")
    parser.add_argument("--low-resolution-prob", type=probability, default=0.5, help="Probability of applying low-resolution augmentation")
    parser.add_argument(
        "--augment-copy-paste", action=argparse.BooleanOptionalAction, default=False,
        help="Paste masked instances from the registered training dataset during training",
    )
    parser.add_argument("--copy-paste-prob", type=probability, default=0.25, help="Probability of applying Copy-Paste augmentation")
    parser.add_argument(
        "--copy-paste-classes", nargs="+", default=None,
        help="Classes eligible as Copy-Paste sources (default: all configured classes)",
    )
    parser.add_argument("--copy-paste-max-instances", type=int, default=2, help="Maximum number of instances to paste per image")
    parser.add_argument(
        "--copy-paste-max-bbox-iou", type=float, default=0.05,
        help="Reject a paste when its bbox IoU with an existing instance exceeds this value",
    )
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration and dataset annotations, then exit")
    parser.add_argument(
        "--max-gpu-power-watts",
        type=float,
        help=(
            "Require an administrator-configured NVIDIA power limit at or below this "
            "value before CUDA training starts"
        ),
    )
    add_model_arguments(parser, include_weights=False)
    parser.add_argument(
        "--angle-classes", nargs="+", default=None,
        help=(
            "Classes supervised by the optional angle head; omitted means all "
            "classes when --angle-head is enabled"
        ),
    )
    add_training_arguments(parser)
    return parser.parse_args(argv)


def validate_augmentation_args(args: argparse.Namespace) -> None:
    """校验数据增强参数的有效性。"""
    for name in ("rotation_range", "brightness_range", "contrast_range", "translation_range"):
        lower, upper = getattr(args, name)
        if lower > upper:
            raise SystemExit(f"--{name.replace('_', '-')} minimum must not exceed maximum")
    if args.brightness_range[0] < 0:
        raise SystemExit("--brightness-range values must be non-negative")
    if args.contrast_range[0] < 0:
        raise SystemExit("--contrast-range values must be non-negative")
    if not -1.0 <= args.translation_range[0] <= args.translation_range[1] <= 1.0:
        raise SystemExit("--translation-range values must be between -1 and 1")
    if not 0.0 < args.low_resolution_scale <= 1.0:
        raise SystemExit("--low-resolution-scale must be greater than 0 and at most 1")


def validate_training_args(args: argparse.Namespace) -> None:
    """校验训练与模型超参数的有效性。"""
    if args.min_size <= 0 or args.max_size <= 0 or args.min_size > args.max_size:
        raise SystemExit("--min-size and --max-size must be positive, with min-size <= max-size")
    if args.max_detections <= 0:
        raise SystemExit("--max-detections must be greater than zero")
    if len(set(args.classes)) != len(args.classes):
        raise SystemExit("--classes must not contain duplicate names")
    if args.score_threshold < 0.0 or args.score_threshold > 1.0:
        raise SystemExit("--score-threshold must be between 0 and 1")
    for name in ("max_iter", "batch_size", "checkpoint_period", "eval_period"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    if args.prefetch_factor <= 0:
        raise SystemExit("--prefetch-factor must be greater than zero")
    if args.hard_negative_repeat < 1:
        raise SystemExit("--hard-negative-repeat must be at least 1")
    if args.copy_paste_max_instances < 0:
        raise SystemExit("--copy-paste-max-instances must be non-negative")
    if not 0.0 <= args.copy_paste_max_bbox_iou <= 1.0:
        raise SystemExit("--copy-paste-max-bbox-iou must be between 0 and 1")
    if not 0.0 <= args.focus_crop_prob <= 1.0:
        raise SystemExit("--focus-crop-prob must be between 0 and 1")
    if args.focus_crop_class and args.focus_crop_class not in args.classes:
        raise SystemExit("--focus-crop-class must be one of --classes")
    if args.focus_crop_scale < 1.0:
        raise SystemExit("--focus-crop-scale must be at least 1")
    if len(args.focus_crop_min_size) != 2 or any(value <= 0 for value in args.focus_crop_min_size):
        raise SystemExit("--focus-crop-min-size values must be positive")
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be non-negative")
    if args.lr_steps and any(step < 0 for step in args.lr_steps):
        raise SystemExit("--lr-steps values must be non-negative")
    # None 表示在 build_cfg 中使用基于 max-iter 推导的默认值；显式空列表表示不衰减。
    if args.lr_steps is not None and args.lr_steps != sorted(set(args.lr_steps)):
        raise SystemExit("--lr-steps values must be strictly increasing")
    if args.repeat_threshold < 0:
        raise SystemExit("--repeat-threshold must be non-negative")
    # --weight-decay 是可选的，因为 build_cfg 会在参数校验后提供默认值。
    if (args.weight_decay is not None and args.weight_decay < 0) or args.weight_decay_norm < 0:
        raise SystemExit("weight decay values must be non-negative")
    if args.weight_decay_bias is not None and args.weight_decay_bias < 0:
        raise SystemExit("--weight-decay-bias must be non-negative")
    if (args.angle_bins is not None and args.angle_bins <= 0) or (
        args.angle_period is not None and args.angle_period <= 0
    ):
        raise SystemExit("--angle-bins and --angle-period must be positive")
    if (args.angle_loss_weight is not None and args.angle_loss_weight < 0) or (
        args.angle_lr_factor is not None and args.angle_lr_factor < 0
    ):
        raise SystemExit("--angle-loss-weight and --angle-lr-factor must be non-negative")
    if args.angle_classes:
        unknown = sorted(set(args.angle_classes) - set(args.classes))
        if unknown:
            raise SystemExit(
                "--angle-classes contains unknown classes: " + ", ".join(unknown)
            )
    if args.angle_classes and args.angle_head is not True:
        raise SystemExit("--angle-classes requires --angle-head")
    effective_period = 360.0 if args.angle_period is None else float(args.angle_period)
    if args.angle_head and args.angle_label_source == "mask" and effective_period != 180.0:
        raise SystemExit(
            "--angle-label-source mask derives an unoriented PCA axis and requires "
            "--angle-period 180; use annotation for directed 360-degree labels"
        )
    for name in (
        "nms_threshold", "roi_positive_fraction", "rpn_positive_fraction",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    for name in (
        "roi_batch_size_per_image", "rpn_batch_size_per_image",
        "rpn_pre_nms_topk_train", "rpn_post_nms_topk_train",
        "rpn_pre_nms_topk_test", "rpn_post_nms_topk_test",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero")
    if args.freeze_at is not None and not 0 <= args.freeze_at <= 5:
        raise SystemExit("--freeze-at must be between 0 and 5")
    if args.optimizer == "sgd" and args.nesterov and args.momentum <= 0:
        raise SystemExit("--nesterov requires --momentum > 0")
    if args.optimizer == "adamw" and args.nesterov:
        raise SystemExit("--nesterov is only valid with --optimizer sgd")
    if args.momentum < 0 or args.lr_gamma <= 0 or args.warmup_factor <= 0:
        raise SystemExit("momentum must be non-negative; lr-gamma and warmup-factor must be positive")
    if args.bias_lr_factor < 0 or args.adam_eps <= 0:
        raise SystemExit("--bias-lr-factor must be non-negative and --adam-eps positive")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        raise SystemExit("--adam-beta1 and --adam-beta2 must be in [0, 1)")
    if args.gradient_clip_norm_type <= 0:
        raise SystemExit("--gradient-clip-norm-type must be greater than zero")
    if args.gradient_clip_value is not None and args.gradient_clip_value <= 0:
        raise SystemExit("--gradient-clip-value must be greater than zero")
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be greater than 0 and less than 1")


def main(argv=None):
    """训练主函数。"""
    args = parse_args(argv)
    # 训练输出必须保持可见。日志 tee 已将相同数据写入磁盘，静默终端只会让启动失败看起来像卡死。
    setup_run_logging("TRAIN", terminal_output=True)
    started_at = datetime.now()
    validate_augmentation_args(args)
    validate_training_args(args)
    if (args.from_scratch or args.pretrained_backbone) and args.resume:
        raise SystemExit("--from-scratch/--pretrained-backbone cannot be used with --resume")
    if args.early_stop_patience < 0:
        raise SystemExit("--early-stop-patience must be non-negative")
    if (args.save_best or args.early_stop_patience > 0) and not (args.val_dir or args.auto_split):
        raise SystemExit("--save-best and --early-stop-patience require --val-dir or --auto-split")
    exp = experiment_from_args(args)
    source_train_dir = exp["train_dir"]
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    if args.auto_split:
        train_paths, val_paths = split_labelme_dataset(
            source_train_dir,
            val_ratio=args.val_ratio,
            seed=split_seed,
            classes=exp["classes"],
        )
        exp["train_dir"] = train_paths
        exp["val_dir"] = val_paths
    exp["weights"] = "" if args.from_scratch or args.pretrained_backbone or not args.weights else exp["weights"]
    exp["pretrained_backbone"] = bool(args.pretrained_backbone)
    if args.pretrained_backbone:
        cache_root = configure_pretrained_cache()
        print(f"Pretrained model cache: {cache_root} (excluded from Git)")
    if exp.get("val_dir"):
        ensure_disjoint_datasets(
            exp["train_dir"], exp["val_dir"],
            extra_train_dirs=[exp["hard_negative_dir"]] if exp.get("hard_negative_dir") else (),
        )
    elif exp.get("hard_negative_dir"):
        # 在没有验证集目录时同样执行图像重叠检查，避免难负样本目录重复引入训练图像。
        ensure_hard_negative_disjoint(exp["train_dir"], exp["hard_negative_dir"])
    train_stats = validate_labelme(exp["train_dir"], exp["classes"])
    val_stats = (
        validate_labelme(exp["val_dir"], exp["classes"], strict_classes=False)
        if exp.get("val_dir")
        else None
    )
    print(json.dumps({"train": train_stats, "val": val_stats}, ensure_ascii=False, indent=2))
    validate_class_coverage(train_stats, val_stats, exp["classes"])
    split_manifest = None
    if args.auto_split:
        split_manifest = write_split_manifest(
            Path(exp["output_dir"]) / "dataset_split.json",
            source_train_dir,
            exp["train_dir"],
            exp["val_dir"],
            args.val_ratio,
            split_seed,
        )
        print(
            "Auto split: "
            f"source={source_train_dir}, train={train_stats['images']} images, "
            f"val={val_stats['images']} images, ratio={args.val_ratio:g}, seed={split_seed}"
        )
        print(f"Split manifest: {split_manifest}")
    print(f"Training dataset: {source_train_dir if args.auto_split else exp['train_dir']}")
    if exp.get("hard_negative_dir"):
        print(f"Hard-negative crops: {exp['hard_negative_dir']} (training-only)")
    if exp.get("val_dir"):
        print(f"Validation dataset: {exp['val_dir']} (evaluation only; no parameter updates)")
    if exp.get("pretrained_backbone"):
        print(f"Initialization: official backbone pretraining ({args.backbone}); project checkpoint forbidden")
    else:
        print("Initialization: " + ("training from scratch" if not exp.get("weights") else f"loading weights {exp['weights']}"))
    if exp.get("angle_head"):
        angle_classes = exp.get("angle_classes") or exp["classes"]
        print(
            "Angle head: enabled "
            f"(classes={','.join(angle_classes)}, source={exp.get('angle_label_source', 'annotation')}, "
            f"bins={exp.get('angle_bins', 72)}, period={exp.get('angle_period', 360.0)}). "
            "Inference must repeat --angle-head, --angle-bins and --angle-period; "
            "add --angle-classes and --angle-source prediction to output learned angles."
        )
    else:
        print(
            "Angle head: disabled (default). Inference must not add --angle-head, "
            "--angle-classes or --angle-source."
        )
    if args.validate_only:
        return

    cuda_training = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.channels_last:
        raise SystemExit(
            "--channels-last is not enabled for Detectron2 0.6: its mapper emits "
            "per-image CHW tensors, and an unsafe 3D-to-channels-last conversion "
            "would add copies or alter model semantics."
        )
    if cuda_training:
        # 训练负载在少数固定图像尺度上重复卷积。启用 benchmark 可为各尺度保留最快 kernel；
        # TF32 提升新架构 NVIDIA GPU 的吞吐量。
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        if args.tf32:
            torch.set_float32_matmul_precision("high")
        print(
            "Throughput: "
            f"tf32={bool(args.tf32)}, cudnn_benchmark={bool(args.cudnn_benchmark)}, "
            f"workers={args.num_workers}, pin_memory={args.pin_memory if args.pin_memory is not None else True}, "
            f"persistent_workers={bool(args.persistent_workers and args.num_workers > 0)}, "
            f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'disabled'}"
        )
    else:
        print(
            "Throughput: CPU mode; tf32/cudnn_benchmark disabled, "
            f"workers={args.num_workers}, persistent_workers="
            f"{bool(args.persistent_workers and args.num_workers > 0)}"
        )

    if args.device in {"auto", "cuda"} and args.max_gpu_power_watts is not None:
        require_gpu_power_limit(args.max_gpu_power_watts)
    if args.amp:
        # Detectron2 0.6 仍调用旧版 AMP 别名。PyTorch 2.5 在每个迭代都会打印警告，产生巨量日志和无用 I/O。
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.cuda\.amp\.(autocast|GradScaler).*",
            category=FutureWarning,
        )

    cfg = build_cfg(exp, training=True)
    train_sources = list(exp["train_dir"]) if args.auto_split else [exp["train_dir"]]
    if exp.get("hard_negative_dir"):
        train_sources.extend(
            [exp["hard_negative_dir"]] * int(exp.get("hard_negative_repeat", 1))
        )
    register_labelme(
        cfg.DATASETS.TRAIN[0], train_sources, exp["classes"], cfg.ANGLE_BINS, cfg.ANGLE_PERIOD
    )
    if cfg.DATASETS.TEST:
        register_labelme(
            cfg.DATASETS.TEST[0], exp["val_dir"], exp["classes"],
            cfg.ANGLE_BINS, cfg.ANGLE_PERIOD, strict_classes=False
        )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    from .trainer import ReproTrainer

    ReproTrainer.augment = bool(exp.get("augment", True))
    ReproTrainer.progress = bool(args.progress)
    ReproTrainer.optimizer_name = args.optimizer
    ReproTrainer.adam_betas = (float(args.adam_beta1), float(args.adam_beta2))
    ReproTrainer.adam_eps = float(args.adam_eps)
    ReproTrainer.save_best = bool(args.save_best)
    ReproTrainer.early_stop_patience = int(args.early_stop_patience)
    ReproTrainer.early_stop_metric = str(args.early_stop_metric)
    trainer = ReproTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    try:
        trainer.train()
    except BaseException as error:
        diagnosis = explain_cuda_failure(error, started_at)
        if diagnosis:
            print(f"FATAL GPU DIAGNOSIS: {diagnosis}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
