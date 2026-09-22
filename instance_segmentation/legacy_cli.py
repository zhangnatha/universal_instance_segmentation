"""Roboflow 深度学习流水线 - 终端 CLI 套件
包含数据集划分、预处理、数据增强、模型训练、评估及推理测试的完整端到端工具。
"""

import os
import sys
import time
import argparse
import json
import copy
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

# 确保项目根目录在 sys.path 中
BASE_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, BASE_DIR)

import numpy as np
import cv2

from .data.dataset import DatasetManager, ImageSample
from .data.preprocessing import PreprocessingPipeline
from .data.augmentation import AugmentationPipeline
from .models.catalog import get_available_models, get_model_config, MODEL_REGISTRY
from .models.trainer import Trainer
from .evaluation.evaluator import ModelEvaluator


def print_banner():
    print(r"""
================================================================================
   ____   ___  ____   ___  _____ _     _____        __   ____ _     ___ 
  |  _ \ / _ \| __ ) / _ \|  ___| |   / _ \ \      / /  / ___| |   |_ _|
  | |_) | | | |  _ \| | | | |_  | |  | | | \ \ /\ / /  | |   | |    | | 
  |  _ <| |_| | |_) | |_| |  _| | |__| |_| |\ V  V /   | |___| |___ | | 
  |_| \_\\___/|____/ \___/|_|   |_____\___/  \_/\_/     \____|_____|___|
       Roboflow Pure Terminal Deep Learning Pipeline (CLI v2.0)
================================================================================
""")


def resolve_dataset_dir(requested_path: Optional[str] = None) -> str:
    """根据用户输入或候选默认位置查找并解析数据集目录。"""
    if requested_path and requested_path.lower() not in ["auto", "none", ""]:
        candidates = [
            requested_path,
            os.path.join(BASE_DIR, requested_path),
            os.path.join(BASE_DIR, "..", requested_path)
        ]
        for p in candidates:
            if os.path.exists(p) and os.path.isdir(p):
                return os.path.normpath(p)
        return requested_path

    # 自动探测候选数据集目录
    default_candidates = [
        os.path.join(BASE_DIR, "data"),
        os.path.join(BASE_DIR, "dataset"),
        os.path.join(BASE_DIR, "datasets", "current_dataset"),
        os.path.join(BASE_DIR, "datasets", "preprocessed_dataset")
    ]
    for c in default_candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return os.path.normpath(c)

    return "data"


# ==========================================
# 1. 数据集拆分 (Dataset Splitting)
# ==========================================
def cmd_split(args):
    data_dir = resolve_dataset_dir(getattr(args, "data", None))

    print(f"\n📂 [1/5] Loading and Splitting Dataset from: {data_dir}")
    print(f"⚙️  Split Ratios: Train={args.train}%, Valid={args.valid}%, Test={args.test}% (Seed={args.seed})")

    mgr = DatasetManager()
    summary = mgr.load_from_directory(data_dir)

    split_res = mgr.split_dataset(
        train_ratio=args.train / 100.0,
        valid_ratio=args.valid / 100.0,
        test_ratio=args.test / 100.0,
        seed=args.seed
    )

    out_dir = args.output or os.path.join(BASE_DIR, "datasets", "current_dataset")
    yaml_path = mgr.export_yolo_dataset(out_dir, task="segment")

    print("\n" + "=" * 68)
    print(" 📊 DATASET SPLIT & DISTRIBUTION SUMMARY")
    print("=" * 68)
    print(f" • Total Source Images:      {summary['total_images']}")
    print(f" • Total Labeled Instances:  {summary['total_annotations']}")
    print(f" • Labeled Categories ({len(summary['classes'])}):   {', '.join(summary['classes'])}")
    print("-" * 68)
    print(f" {'Split':<12} | {'Count':<10} | {'Percentage':<12} | {'Directory'}")
    print("-" * 68)
    rel_out = os.path.relpath(out_dir, BASE_DIR) if os.path.isabs(out_dir) else out_dir
    rel_yaml = os.path.relpath(yaml_path, BASE_DIR) if os.path.isabs(yaml_path) else yaml_path
    print(f" {'Train':<12} | {split_res['train']:<10} | {args.train:>5.1f}%       | {rel_out}/train")
    print(f" {'Valid':<12} | {split_res['valid']:<10} | {args.valid:>5.1f}%       | {rel_out}/valid")
    print(f" {'Test':<12} | {split_res['test']:<10} | {args.test:>5.1f}%       | {rel_out}/test")
    print("=" * 68)

    print("\n🏷️  Per-Class Instance Distribution:")
    for cls_name, count in summary['class_counts'].items():
        print(f"   - {cls_name:<12}: {count:>4d} instances")

    print(f"\n✅ Dataset successfully split & exported to: {rel_out}")
    print(f"📄 Dataset YAML config generated: {rel_yaml}\n")
    return mgr, out_dir, yaml_path


# ==========================================
# 2. 数据预处理 (Data Preprocessing)
# ==========================================
def _parse_grid(value: Optional[str]) -> tuple:
    """将切片网格解析为 ROWSxCOLS 格式，与预处理界面保持一致。"""
    if not value:
        return 1, 1
    parts = str(value).lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise ValueError("--tile must use the ROWSxCOLS format, for example 2x3")
    rows, cols = max(1, int(parts[0])), max(1, int(parts[1]))
    return rows, cols


_RESIZE_MODES = {
    "stretch",
    "fill",
    "fit_within",
    "fit_reflect",
    "fit_black",
    "fit_white",
}


def _parse_resize_option(value) -> tuple:
    """解析合并的调整尺寸选项：MODE SIZE，或 SIZE（默认为 stretch）。"""
    if value is None:
        return "stretch", 432, 432

    tokens = [str(item).strip() for item in (value if isinstance(value, (list, tuple)) else [value]) if str(item).strip()]
    if len(tokens) == 1 and ":" in tokens[0]:
        tokens = [part.strip() for part in tokens[0].split(":", 1)]

    if len(tokens) == 1:
        mode, size = "stretch", tokens[0]
    elif len(tokens) == 2:
        if tokens[0].lower() in _RESIZE_MODES:
            mode, size = tokens[0].lower(), tokens[1]
        elif tokens[1].lower() in _RESIZE_MODES:
            size, mode = tokens[0], tokens[1].lower()
        else:
            raise ValueError("--resize must use MODE SIZE, for example: --resize stretch 512x512")
    else:
        raise ValueError("--resize accepts SIZE or MODE SIZE, for example: --resize stretch 512x512")

    size_text = str(size).lower().replace("×", "x")
    if "x" in size_text:
        width_text, height_text = size_text.split("x", 1)
    else:
        width_text = height_text = size_text
    try:
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise ValueError("Resize SIZE must be an integer or WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise ValueError("Resize WIDTH and HEIGHT must be positive")
    return mode, width, height


def _parse_rename_pairs(values: Optional[List[str]]) -> Dict[str, str]:
    """解析重复指定的 OLD=NEW 类别重命名选项。"""
    result = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--rename-class values must use OLD=NEW format")
        old, new = value.split("=", 1)
        if old.strip() and new.strip():
            result[old.strip()] = new.strip()
    return result


def _parse_tag_rules(values: Optional[List[str]]) -> List[Dict[str, str]]:
    """解析按标签过滤时重复指定的 TAG=MODE 规则。"""
    rules: List[Dict[str, str]] = []
    valid_modes = {"require", "exclude", "allow"}
    for value in values or []:
        separator = "=" if "=" in value else ":"
        if separator not in value:
            raise ValueError("--filter-tag values must use TAG=MODE")
        tag, mode = value.split(separator, 1)
        tag, mode = tag.strip(), mode.strip().lower()
        if not tag or mode not in valid_modes:
            raise ValueError("--filter-tag mode must be require, exclude, or allow")
        rules.append({"tag": tag, "mode": mode})
    return rules


def _build_preprocessing_pipeline(args) -> PreprocessingPipeline:
    pipeline = PreprocessingPipeline()

    # 自动方向校正必须在任何空间操作之前运行，以便切片/裁剪/调整大小
    # 基于物理像素方向和匹配的标注点进行。
    if getattr(args, "auto_orient", False):
        pipeline.add_step("auto_orient", {})

    tile = getattr(args, "tile", None)
    if tile:
        rows, cols = _parse_grid(tile)
        pipeline.add_step("tile", {"grid_rows": rows, "grid_cols": cols})

    if getattr(args, "isolate_objects", False):
        pipeline.add_step("isolate_objects", {})

    static_crop = getattr(args, "static_crop", None)
    if static_crop:
        h_min, h_max, v_min, v_max = static_crop
        pipeline.add_step("static_crop", {
            "h_min": h_min, "h_max": h_max, "v_min": v_min, "v_max": v_max
        })

    dynamic_class = getattr(args, "dynamic_crop_class", None)
    if isinstance(dynamic_class, (list, tuple)):
        if len(dynamic_class) > 1:
            raise ValueError("--dynamic-crop-class accepts exactly one ROI label")
        dynamic_class = dynamic_class[0] if dynamic_class else None
    if dynamic_class:
        pipeline.add_step("dynamic_crop", {"class": dynamic_class})

    resize = getattr(args, "resize", None)
    if resize:
        mode, width, height = _parse_resize_option(resize)
        pipeline.add_step("resize", {
            "mode": mode,
            "width": width,
            "height": height,
        })

    if getattr(args, "grayscale", False):
        pipeline.add_step("grayscale", {})

    if getattr(args, "contrast", False):
        pipeline.add_step("contrast", {"type": getattr(args, "contrast_type", "adaptive")})

    include_classes = getattr(args, "include_classes", None) or []
    exclude_classes = getattr(args, "exclude_classes", None) or []
    rename_map = _parse_rename_pairs(getattr(args, "rename_class", None))
    regex_pattern = getattr(args, "class_regex", None) or ""
    regex_replace = getattr(args, "class_replace", None) or ""
    if include_classes or exclude_classes or rename_map or regex_pattern:
        include_map = {label: True for label in include_classes}
        include_map.update({label: False for label in exclude_classes})
        pipeline.add_step("modify_classes", {
            "include_classes": include_map,
            "rename_map": rename_map,
            "regex_pattern": regex_pattern,
            "regex_replace": regex_replace,
        })

    if getattr(args, "filter_null", None) is not None:
        pipeline.add_step("filter_null", {"min_percent": args.filter_null})

    tag_rules = _parse_tag_rules(getattr(args, "filter_tag", None))
    if tag_rules:
        pipeline.add_step("filter_tag", {"rules": tag_rules})

    sample_config = {}
    for split_name, attr_name in (("train", "sample_train"), ("valid", "sample_valid"), ("test", "sample_test")):
        value = getattr(args, attr_name, None)
        if value is not None:
            sample_config[f"{split_name}_percent"] = value
    if sample_config:
        pipeline.add_step("random_sample", sample_config)

    return pipeline


def _apply_random_sample_by_split(splits: Dict[str, List[Any]], config: Dict[str, Any], seed: Optional[int] = None) -> None:
    """按划分独立保留精确百分比大小的随机子集。"""
    rng = random.Random(seed)
    for split_name, percent_key in (
        ("train", "train_percent"),
        ("valid", "valid_percent"),
        ("test", "test_percent"),
    ):
        samples = list(splits.get(split_name, []))
        percent = float(np.clip(config.get(percent_key, 100.0), 0, 100))
        target_count = int(round(len(samples) * percent / 100.0))
        target_count = max(0, min(len(samples), target_count))
        if target_count == len(samples):
            splits[split_name] = samples
        elif target_count == 0:
            splits[split_name] = []
        else:
            selected_indices = sorted(rng.sample(range(len(samples)), target_count))
            splits[split_name] = [samples[index] for index in selected_indices]


def _apply_filter_null_by_split(
    splits: Dict[str, List[Any]], min_percent: float, seed: Optional[int] = None
) -> None:
    """在每个数据集划分中按精确百分比过滤无标注（空）图像。

    ``min_percent`` 遵循本项目使用的 Roboflow 界面语义：
    0% 保留所有空图像，100% 移除所有空图像。有标注的图像始终予以保留。
    通过“隔离对象”（Isolate Objects）生成的分类裁剪图像在设计上不包含检测标注，
    因此其分类元数据也使其免受此过滤器的影响。
    """
    rng = random.Random(seed)
    percent = float(np.clip(min_percent, 0, 100))

    for split_name in ("train", "valid", "test"):
        samples = list(splits.get(split_name, []))
        null_indices = [
            index for index, sample in enumerate(samples)
            if sample.is_null and not hasattr(sample, "classification_label")
        ]
        remove_count = int(round(len(null_indices) * percent / 100.0))
        remove_count = max(0, min(len(null_indices), remove_count))

        if remove_count:
            remove_indices = set(rng.sample(null_indices, remove_count))
            splits[split_name] = [
                sample for index, sample in enumerate(samples)
                if index not in remove_indices
            ]
        else:
            splits[split_name] = samples


def cmd_preprocess(args, in_mgr: Optional[DatasetManager] = None):
    pipeline = _build_preprocessing_pipeline(args)
    filter_null_step = next((step for step in pipeline.steps if step["name"] == "filter_null"), None)
    random_sample_step = next((step for step in pipeline.steps if step["name"] == "random_sample"), None)
    transform_pipeline = PreprocessingPipeline(
        [step for step in pipeline.steps if step["name"] not in {"filter_null", "random_sample"}]
    )

    if in_mgr is not None:
        mgr = in_mgr
        src_label = "In-memory Split Dataset"
    else:
        # 确定源路径：用户 --data > datasets/current_dataset > 原始数据
        user_data = getattr(args, "data", None)
        if user_data and user_data not in ["auto", "none", ""]:
            data_dir = resolve_dataset_dir(user_data)
            # 若用户传入原始数据目录，检查是否存在 datasets/current_dataset 以保留划分
            if os.path.exists(os.path.join(BASE_DIR, "datasets", "current_dataset", "train")) and not os.path.exists(os.path.join(data_dir, "train")):
                print(f"💡 Notice: Using previously partitioned dataset from datasets/current_dataset to preserve split.")
                data_dir = os.path.join(BASE_DIR, "datasets", "current_dataset")
            elif os.path.exists(os.path.join(BASE_DIR, "exports", "current_dataset", "train")) and not os.path.exists(os.path.join(data_dir, "train")):
                data_dir = os.path.join(BASE_DIR, "exports", "current_dataset")
        elif os.path.exists(os.path.join(BASE_DIR, "datasets", "current_dataset", "train")):
            data_dir = os.path.join(BASE_DIR, "datasets", "current_dataset")
        elif os.path.exists(os.path.join(BASE_DIR, "exports", "current_dataset", "train")):
            data_dir = os.path.join(BASE_DIR, "exports", "current_dataset")
        else:
            data_dir = resolve_dataset_dir(None)

        src_label = os.path.relpath(data_dir, BASE_DIR) if os.path.isabs(data_dir) else data_dir
        mgr = DatasetManager()
        mgr.load_from_directory(data_dir)

    print(f"\n⚙️  [2/5] Configuring & Executing Data Preprocessing from: {src_label}...")

    # 对所有划分应用预处理转换
    for split_name in ["train", "valid", "test"]:
        orig_samples = mgr.splits.get(split_name, [])
        processed_list = []
        for s in orig_samples:
            res = transform_pipeline.process_sample(s)
            processed_list.extend(res)
        mgr.splits[split_name] = processed_list

    if filter_null_step:
        _apply_filter_null_by_split(
            mgr.splits,
            filter_null_step["config"].get("min_percent", 100.0),
            seed=getattr(args, "seed", None),
        )

    if random_sample_step:
        _apply_random_sample_by_split(
            mgr.splits,
            random_sample_step["config"],
            seed=getattr(args, "seed", None),
        )

    # 类别编辑和过滤可能会更改导出的类别集合。
    discovered = sorted({ann.label for samples in mgr.splits.values() for sample in samples for ann in sample.annotations})
    mgr._finalize_classes(discovered)

    print("\n" + "=" * 68)
    print(" 🛠️  APPLIED PREPROCESSING PIPELINE")
    print("=" * 68)
    for idx, step in enumerate(pipeline.steps, start=1):
        print(f" {idx}. {step['name'].upper():<14} | Config: {step['config']}")
    resize_step = next((step for step in pipeline.steps if step["name"] == "resize"), None)
    if resize_step:
        resize_cfg = resize_step["config"]
        print(f" • Target Resolution: {resize_cfg['width']} × {resize_cfg['height']} ({resize_cfg['mode']})")
    print(f" • Processed Samples: Train={len(mgr.splits['train'])}, Valid={len(mgr.splits['valid'])}, Test={len(mgr.splits['test'])}")
    print("=" * 68)

    out_dir = args.output or os.path.join(BASE_DIR, "datasets", "preprocessed_dataset")
    os.makedirs(out_dir, exist_ok=True)
    isolate_enabled = any(step["name"] == "isolate_objects" for step in pipeline.steps)
    if isolate_enabled:
        classification_dir = os.path.join(out_dir, "classification")
        mgr.export_classification_dataset(classification_dir)
        yaml_path = classification_dir
        rel_classification = os.path.relpath(classification_dir, BASE_DIR) if os.path.isabs(classification_dir) else classification_dir
        print("⚠️  Isolate Objects removed all bounding boxes for classification crops.")
        print(f"📁 Classification output: {rel_classification}/<split>/<class>/")
    else:
        yaml_path = mgr.export_yolo_dataset(out_dir, task="segment")
    rel_out = os.path.relpath(out_dir, BASE_DIR) if os.path.isabs(out_dir) else out_dir

    print(f"\n✅ Preprocessing completed. Preprocessed dataset ready at: {rel_out}\n")
    return mgr, out_dir, yaml_path


# ==========================================
# 3. 数据增强 (Data Augmentation)
# ==========================================
def _build_augmentation_pipeline(args) -> AugmentationPipeline:
    seed = getattr(args, "seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    aug = AugmentationPipeline(multiplier=max(1, int(getattr(args, "multiplier", 1))))

    flip_h = bool(getattr(args, "flip_h", False))
    flip_v = bool(getattr(args, "flip_v", False))
    if flip_h or flip_v:
        aug.set_config("flip", {"horizontal": flip_h, "vertical": flip_v})

    if any(getattr(args, name, False) for name in ("rotate90_cw", "rotate90_ccw", "rotate90_ud")):
        aug.set_config("rotate90", {
            "clockwise": bool(getattr(args, "rotate90_cw", False)),
            "counter_clockwise": bool(getattr(args, "rotate90_ccw", False)),
            "upside_down": bool(getattr(args, "rotate90_ud", False)),
        })

    crop = getattr(args, "crop", None)
    if crop:
        aug.set_config("crop", {"min_zoom": crop[0], "max_zoom": crop[1]})

    rotation = getattr(args, "rotation", None)
    if rotation is not None and float(rotation) > 0:
        aug.set_config("rotation", {"angle": rotation})

    shear = getattr(args, "shear", None)
    if shear:
        aug.set_config("shear", {"horizontal": shear[0], "vertical": shear[1]})

    grayscale_percent = getattr(args, "grayscale_percent", None)
    if grayscale_percent is not None:
        aug.set_config("grayscale", {"percent": grayscale_percent})

    for arg_name, config_name, key in (
        ("hue", "hue", "angle"),
        ("saturation", "saturation", "percent"),
        ("exposure", "exposure", "percent"),
        ("blur", "blur", "pixels"),
        ("noise", "noise", "percent"),
        ("camera_gain", "camera_gain", "variance"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None and float(value) > 0:
            aug.set_config(config_name, {key: value})

    brightness = getattr(args, "brightness", None)
    if brightness is not None and float(brightness) > 0:
        brighten = bool(getattr(args, "brighten", False))
        darken = bool(getattr(args, "darken", False))
        if not brighten and not darken:
            brighten = darken = True
        aug.set_config("brightness", {
            "percent": brightness, "brighten": brighten, "darken": darken
        })

    cutout = getattr(args, "cutout", None)
    if cutout:
        aug.set_config("cutout", {"percent": cutout[0], "count": cutout[1]})

    if getattr(args, "mosaic", False):
        aug.set_config("mosaic", {})

    motion_blur = getattr(args, "motion_blur", None)
    if motion_blur:
        aug.set_config("motion_blur", {
            "length": motion_blur[0], "angle": motion_blur[1], "frames": motion_blur[2]
        })

    bbox_configs = {
        "bbox_flip_h": bool(getattr(args, "bbox_flip", False) or getattr(args, "bbox_flip_h", False)),
        "bbox_flip_v": bool(getattr(args, "bbox_flip_v", False)),
        "bbox_rotate90": any(getattr(args, name, False) for name in (
            "bbox_rotate90_cw", "bbox_rotate90_ccw", "bbox_rotate90_ud"
        )),
        "bbox_crop": getattr(args, "bbox_crop", None),
        "bbox_rotation": getattr(args, "bbox_rotation", None),
        "bbox_shear": getattr(args, "bbox_shear", None),
        "bbox_brightness": getattr(args, "bbox_brightness", None),
        "bbox_brighten": bool(getattr(args, "bbox_brighten", False)),
        "bbox_darken": bool(getattr(args, "bbox_darken", False)),
        "bbox_exposure": getattr(args, "bbox_exposure", None),
        "bbox_blur": getattr(args, "bbox_blur", None),
        "bbox_noise": getattr(args, "bbox_noise", None),
        "bbox_motion_blur": getattr(args, "bbox_motion_blur", None),
        "bbox_camera_gain": getattr(args, "bbox_camera_gain", None),
    }
    if bbox_configs["bbox_flip_h"] or bbox_configs["bbox_flip_v"]:
        aug.set_config("bbox_flip", {
            "horizontal": bbox_configs["bbox_flip_h"],
            "vertical": bbox_configs["bbox_flip_v"],
        })
    if bbox_configs["bbox_rotate90"]:
        aug.set_config("bbox_rotate90", {
            "clockwise": bool(getattr(args, "bbox_rotate90_cw", False)),
            "counter_clockwise": bool(getattr(args, "bbox_rotate90_ccw", False)),
            "upside_down": bool(getattr(args, "bbox_rotate90_ud", False)),
        })
    if bbox_configs["bbox_crop"]:
        value = bbox_configs["bbox_crop"]
        aug.set_config("bbox_crop", {"min_zoom": value[0], "max_zoom": value[1]})
    if bbox_configs["bbox_rotation"] is not None and float(bbox_configs["bbox_rotation"]) > 0:
        aug.set_config("bbox_rotation", {"angle": bbox_configs["bbox_rotation"]})
    if bbox_configs["bbox_shear"]:
        value = bbox_configs["bbox_shear"]
        aug.set_config("bbox_shear", {"horizontal": value[0], "vertical": value[1]})
    for name in ("bbox_brightness", "bbox_exposure", "bbox_blur", "bbox_noise", "bbox_camera_gain"):
        value = bbox_configs[name]
        if value is not None and float(value) > 0:
            key = "pixels" if name == "bbox_blur" else ("variance" if name == "bbox_camera_gain" else "percent")
            config = {key: value}
            if name == "bbox_brightness":
                brighten = bbox_configs["bbox_brighten"]
                darken = bbox_configs["bbox_darken"]
                if not brighten and not darken:
                    brighten = darken = True
                config.update({"brighten": brighten, "darken": darken})
            aug.set_config(name, config)
    if bbox_configs["bbox_motion_blur"]:
        value = list(bbox_configs["bbox_motion_blur"])
        if len(value) not in (2, 3):
            raise ValueError("--bbox-motion-blur requires LENGTH ANGLE [FRAMES]")
        aug.set_config("bbox_motion_blur", {
            "length": value[0], "angle": value[1], "frames": value[2] if len(value) == 3 else 1,
        })

    return aug


def cmd_augment(args, in_mgr: Optional[DatasetManager] = None):
    aug = _build_augmentation_pipeline(args)

    if in_mgr is not None:
        mgr = in_mgr
        src_label = "In-memory Preprocessed Dataset"
    else:
        # 确定源路径：用户 --data > datasets/preprocessed_dataset > datasets/current_dataset > 原始数据
        user_data = getattr(args, "data", None)
        if user_data and user_data not in ["auto", "none", ""]:
            data_dir = resolve_dataset_dir(user_data)
            if os.path.exists(os.path.join(BASE_DIR, "datasets", "preprocessed_dataset", "train")) and not os.path.exists(os.path.join(data_dir, "train")):
                print(f"💡 Notice: Using preprocessed dataset from datasets/preprocessed_dataset to preserve resolution & transforms.")
                data_dir = os.path.join(BASE_DIR, "datasets", "preprocessed_dataset")
            elif os.path.exists(os.path.join(BASE_DIR, "datasets", "current_dataset", "train")) and not os.path.exists(os.path.join(data_dir, "train")):
                data_dir = os.path.join(BASE_DIR, "datasets", "current_dataset")
        elif os.path.exists(os.path.join(BASE_DIR, "datasets", "preprocessed_dataset", "train")):
            data_dir = os.path.join(BASE_DIR, "datasets", "preprocessed_dataset")
        elif os.path.exists(os.path.join(BASE_DIR, "datasets", "current_dataset", "train")):
            data_dir = os.path.join(BASE_DIR, "datasets", "current_dataset")
        elif os.path.exists(os.path.join(BASE_DIR, "exports", "preprocessed_dataset", "train")):
            data_dir = os.path.join(BASE_DIR, "exports", "preprocessed_dataset")
        elif os.path.exists(os.path.join(BASE_DIR, "exports", "current_dataset", "train")):
            data_dir = os.path.join(BASE_DIR, "exports", "current_dataset")
        else:
            data_dir = resolve_dataset_dir(None)

        src_label = os.path.relpath(data_dir, BASE_DIR) if os.path.isabs(data_dir) else data_dir
        mgr = DatasetManager()
        mgr.load_from_directory(data_dir)

    print(f"\n🎨 [3/5] Configuring & Executing Data Augmentation from: {src_label}...")

    train_count_before = len(mgr.splits.get("train", []))
    # 仅对训练集进行数据增强；验证集和测试集保持不变
    train_samples = mgr.splits.get("train", [])
    mgr.splits["train"] = aug.augment_dataset(train_samples, all_train_samples=train_samples)
    total_train_after = len(mgr.splits["train"])

    print("\n" + "=" * 68)
    print(" 🚀 DATA AUGMENTATION SPECIFICATION")
    print("=" * 68)
    print(f" • Multiplier:           {args.multiplier}x")
    print(f" • Base Training Images: {train_count_before}")
    print(f" • Augmented Train Set:  {total_train_after} training images (+{total_train_after - train_count_before} augmented)")
    print(f" • Valid / Test Sets:    {len(mgr.splits.get('valid', []))} Valid, {len(mgr.splits.get('test', []))} Test (Untouched)")
    print("-" * 68)
    print(" Active Transformations:")
    if aug.configs:
        for aug_name, config in aug.configs.items():
            print(f"   ✓ {aug_name}: {config}")
    else:
        print("   (none; original training images are retained)")
    print("=" * 68)

    out_dir = args.output or os.path.join(BASE_DIR, "datasets", "augmented_dataset")
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = mgr.export_yolo_dataset(out_dir, task="segment")
    rel_out = os.path.relpath(out_dir, BASE_DIR) if os.path.isabs(out_dir) else out_dir

    print(f"\n✅ Data Augmentation completed. Augmented dataset ready at: {rel_out}\n")
    return mgr, out_dir, yaml_path


# ==========================================
# 4. 模型训练与实时评估 (Model Training & Real-time Evaluation)
# ==========================================
def cmd_train(args):
    # 确定最相关的训练数据集 YAML 配置
    user_data = getattr(args, "data", None)
    yaml_path = None
    mgr = None

    if user_data and str(user_data).endswith((".yaml", ".yml")) and os.path.exists(user_data):
        yaml_path = user_data
    elif user_data and os.path.isdir(user_data) and os.path.exists(os.path.join(user_data, "data.yaml")):
        yaml_path = os.path.join(user_data, "data.yaml")
    elif user_data and user_data not in ["auto", "none", ""]:
        # 若用户传入原始数据目录，检查是否已生成增强或预处理数据集
        cand_yamls = [
            os.path.join(BASE_DIR, "datasets", "augmented_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "datasets", "preprocessed_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "datasets", "current_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "exports", "augmented_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "exports", "preprocessed_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "exports", "current_dataset", "data.yaml")
        ]
        for cy in cand_yamls:
            if os.path.exists(cy):
                yaml_path = cy
                print(f"💡 Notice: Found cascading pipeline dataset at {os.path.relpath(cy, BASE_DIR)}. Training with pipeline dataset.")
                break
        
        if yaml_path is None:
            data_dir = resolve_dataset_dir(user_data)
            mgr = DatasetManager()
            mgr.load_from_directory(data_dir)
            export_dir = os.path.join(BASE_DIR, "datasets", "current_dataset")
            yaml_path = mgr.export_yolo_dataset(export_dir, task="segment")
    else:
        # 级联自动探测：augmented > preprocessed > current > 原始数据
        cand_yamls = [
            os.path.join(BASE_DIR, "datasets", "augmented_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "datasets", "preprocessed_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "datasets", "current_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "exports", "augmented_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "exports", "preprocessed_dataset", "data.yaml"),
            os.path.join(BASE_DIR, "exports", "current_dataset", "data.yaml")
        ]
        for cy in cand_yamls:
            if os.path.exists(cy):
                yaml_path = cy
                break

        if yaml_path is None:
            data_dir = resolve_dataset_dir(None)
            mgr = DatasetManager()
            mgr.load_from_directory(data_dir)
            mgr.split_dataset(train_ratio=0.70, valid_ratio=0.20, test_ratio=0.10)
            export_dir = os.path.join(BASE_DIR, "datasets", "current_dataset")
            yaml_path = mgr.export_yolo_dataset(export_dir, task="segment")

    if mgr is None:
        dataset_folder = os.path.dirname(yaml_path)
        mgr = DatasetManager()
        try:
            mgr.load_from_directory(dataset_folder)
        except Exception:
            pass

    rel_yaml = os.path.relpath(yaml_path, BASE_DIR) if os.path.isabs(yaml_path) else yaml_path
    print(f"\n🚀 [4/5] Starting 100% Real Deep Learning Training with dataset: {rel_yaml}...")

    model_key = args.model
    size_id = args.size
    epochs = args.epochs
    batch_size = args.batch
    img_size = args.imgsz
    device = args.device

    print("\n" + "=" * 76)
    print(" 🛠️  TRAINING CONFIGURATION & HARDWARE SETUP")
    print("=" * 76)
    if model_key == "rf-detr":
        print(f" • Architecture:   Roboflow RF-DETR Detection Transformer ({size_id.capitalize()})")
        print(" • Backbone:       DINOv2 Visual Transformer (ViT) + Deformable Decoder")
        print(" • Framework:      Roboflow Native rfdetr (Apache-2.0)")
    else:
        print(f" • Architecture:   {model_key.upper()} Instance Segmentation ({size_id.capitalize()})")
        print(" • Framework:      Ultralytics YOLO Engine")

    print(f" • Epochs:         {epochs}")
    print(f" • Batch Size:     {batch_size}")
    print(f" • Image Size:     {img_size} × {img_size}")
    print(f" • Target Device:  {device}")
    print("=" * 76 + "\n")

    trainer = Trainer(output_root=os.path.join(BASE_DIR, "runs"))

    # 控制台实时进度回调
    print(" [Progress/Epoch] |   Elapsed  |  Remaining | Step Speed |   VRAM Used  |  mAP@50  |    F1   ")
    print("------------------------------------------------------------------------------------------")

    def terminal_callback(state):
        ep = state.get("current_epoch", 0)
        tot = state.get("total_epochs", epochs)
        cur_step = state.get("current_step", 0)
        tot_step = state.get("total_steps_all", 0)
        
        elapsed = state.get("elapsed_time", "00m 00s")
        rem = state.get("time_remaining", "Calculating...")
        speed = state.get("speed", "0.0 it/s")
        gpu_mem = state.get("gpu_memory", "N/A")
        m = state.get("metrics", {})
        
        map50_str = f"{m.get('map50', 0.0):>5.1f}%" if ep > 0 else "  --  "
        f1_str = f"{m.get('f1', 0.0):>5.1f}%" if ep > 0 else "  --  "

        step_tag = f"Epoch [{ep:2d}/{tot:2d}]"
        if cur_step > 0 and tot_step > 0:
            pct = (cur_step / tot_step) * 100.0
            step_tag = f"[{ep:2d}/{tot:2d}|{pct:4.1f}%]"

        print(f" {step_tag:<16} | {elapsed:>10} | {rem:>10} | {speed:>10} | {gpu_mem:>12} | {map50_str} | {f1_str}")

    trainer.register_callback(terminal_callback)
    
    trainer.start_training(
        yaml_path=yaml_path,
        model_key=model_key,
        size_id=size_id,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        weights=getattr(args, "weights", None),
        resume=getattr(args, "resume", None),
        lr=getattr(args, "lr", None),
        eval_interval=getattr(args, "eval_interval", 1),
        amp_dtype=getattr(args, "amp_dtype", "bf16"),
        freeze_encoder=getattr(args, "freeze_encoder", False),
        cls_loss_coef=getattr(args, "cls_loss_coef", None),
    )

    # 监控直到训练完成
    while trainer.state.is_training:
        time.sleep(0.5)

    if trainer.state.has_error:
        print(f"\n❌ Training Error encountered: {trainer.state.error_message}")
        return None

    rel_saved = os.path.relpath(trainer.state.saved_model_path, BASE_DIR) if os.path.isabs(trainer.state.saved_model_path) else trainer.state.saved_model_path
    print("\n-----------------------------------------------------------------------------------------")
    print(f"🎉 Training completed successfully! Weights saved at: {rel_saved}")

    # 运行全面的模型评估并打印格式化报告
    print("\n" + "=" * 76)
    print(" 📊 COMPREHENSIVE TEST SET & VALIDATION EVALUATION REPORT")
    print("=" * 76)

    evaluator = ModelEvaluator(saved_dir=os.path.join(BASE_DIR, "saved_models"))
    report = evaluator.evaluate_model(
        dataset_manager=mgr,
        model_path=trainer.state.saved_model_path,
        model_name=trainer.state.model_type,
        save_results=True
    )

    vm = report["validation_report"]["metrics"]
    tm = report["test_report"]["metrics"]

    print(f"\n🏆 Overall Performance:")
    print(f" • Test Set mAP@50:       {tm['map50']}%")
    print(f" • Test Set mAP@50:95:    {tm['map50_95']}%")
    print(f" • Test Set Precision:    {tm['precision']}%")
    print(f" • Test Set Recall:       {tm['recall']}%")
    print(f" • Test Set F1 Score:     {tm['f1']}%")

    print("\n📋 Per-Class Performance Breakdown (Test Set):")
    print(" " + "-" * 72)
    print(f" {'Class':<12} | {'Images':<8} | {'Instances':<10} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'mAP@50':<8}")
    print(" " + "-" * 72)
    for row in report["test_report"]["class_breakdown"]:
        print(f" {row['class']:<12} | {row['images']:<8} | {row['instances']:<10} | {row['precision']:>5.1f}% | {row['recall']:>5.1f}% | {row['f1']:>5.1f}% | {row['map50']:>5.1f}%")
    print(" " + "-" * 72)

    best_weights_path = os.path.join("saved_models", "latest", "best.pt")
    json_report_path = os.path.join("saved_models", "latest", "evaluation_report.json")
    md_report_path = os.path.join("saved_models", "latest", "evaluation_report.md")

    print("\n💾 Model Artifacts Exported:")
    print(f" • Checkpoint:  {best_weights_path}")
    print(f" • JSON Report: {json_report_path}")
    print(f" • MD Report:   {md_report_path}\n")

    return report


# ==========================================
# 5. 模型推理与可视化预测 (Model Inference & Visual Prediction)
# ==========================================
def cmd_predict(args):
    weights_path = args.weights or os.path.join(BASE_DIR, "saved_models", "latest", "best.pt")
    source = args.source
    conf_threshold = args.conf

    if not source:
        # 在候选目录中搜索可用的测试图像（优先保留原始分辨率）
        img_candidates_dirs = [
            os.path.join(BASE_DIR, "datasets", "current_dataset", "test"),
            os.path.join(BASE_DIR, "datasets", "current_dataset", "valid"),
            os.path.join(BASE_DIR, "data"),
            os.path.join(BASE_DIR, "dataset"),
            os.path.join(BASE_DIR, "datasets", "preprocessed_dataset", "test"),
            os.path.join(BASE_DIR, "datasets", "augmented_dataset", "test")
        ]
        for cdir in img_candidates_dirs:
            if os.path.exists(cdir) and os.path.isdir(cdir):
                imgs = [f for f in os.listdir(cdir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                if imgs:
                    imgs.sort()
                    source = os.path.join(cdir, imgs[0])
                    break

    if not source or not os.path.exists(source):
        print("❌ Error: Valid input image source not found. Specify --source <image_path_or_dir>.")
        return

    rel_weights = os.path.relpath(weights_path, BASE_DIR) if os.path.isabs(weights_path) else weights_path
    rel_source = os.path.relpath(source, BASE_DIR) if os.path.isabs(source) and source.startswith(BASE_DIR) else source

    evaluator = ModelEvaluator()
    model_obj = evaluator.load_model(weights_path)

    iou_threshold = getattr(args, "iou", 0.50)
    mask_alpha = getattr(args, "mask_alpha", 0.40)

    # 目录 / 批量预测模式
    if os.path.isdir(source):
        img_files = [f for f in os.listdir(source) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        img_files.sort()
        if not img_files:
            print(f"❌ No supported images found in directory: {source}")
            return

        batch_cap = getattr(args, "save_samples", 0)
        target_files = img_files[:batch_cap] if batch_cap > 0 else img_files

        print(f"\n🔍 [5/5] Running Batch Model Inference on {len(target_files)} images from: {rel_source}")
        print(f" • Model Checkpoint: {rel_weights}")
        print(f" • Confidence Limit: {conf_threshold * 100:.0f}%")
        print(f" • Mask Transparency: {mask_alpha * 100:.0f}% Alpha")
        if iou_threshold is not None and 0.0 < iou_threshold < 1.0:
            print(f" • Mask NMS Limit:   {iou_threshold * 100:.0f}% (Keep Highest Score)")
        if model_obj:
            print(f" • Framework Loaded: {model_obj['type'].upper()} ({len(model_obj['classes'])} classes: {', '.join(model_obj['classes'])})")

        out_dir = args.output or os.path.join(BASE_DIR, "output_predictions")
        os.makedirs(out_dir, exist_ok=True)
        total_dets = 0

        save_json = not getattr(args, "no_json", False)

        for idx, fname in enumerate(target_files, 1):
            fpath = os.path.join(source, fname)
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                continue
            res = evaluator.predict_image_array(
                img_bgr,
                conf_threshold=conf_threshold,
                model_obj=model_obj,
                weights_path=weights_path,
                iou_threshold=iou_threshold,
                mask_alpha=mask_alpha
            )
            preds = [p for p in res.get("predictions", []) if p.get("confidence", 0.0) >= conf_threshold]
            total_dets += len(preds)

            annotated_bgr = evaluator.draw_annotated_image(img_bgr, preds, conf_threshold=conf_threshold, mask_alpha=mask_alpha)
            out_path = os.path.join(out_dir, f"result_{Path(fname).stem}.jpg")
            cv2.imwrite(out_path, annotated_bgr)

            if save_json:
                h, w = img_bgr.shape[:2]
                detection_records = []
                shapes_records = []
                for p in preds:
                    cname = str(p.get("class", "default"))
                    cid = int(p.get("class_id", 0))
                    score = float(p.get("confidence", 0.0))
                    cx, cy, bw, bh = p["x"], p["y"], p["width"], p["height"]
                    x0 = float(cx - bw / 2.0)
                    y0 = float(cy - bh / 2.0)
                    x1 = float(cx + bw / 2.0)
                    y1 = float(cy + bh / 2.0)
                    pts_data = p.get("points", [])
                    if pts_data and len(pts_data) >= 3:
                        pts_list = [[float(pt["x"]), float(pt["y"])] for pt in pts_data]
                        contours_xy = [[[int(pt["x"]), int(pt["y"])] for pt in pts_data]]
                    else:
                        pts_list = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                        contours_xy = [[[int(x0), int(y0)], [int(x1), int(y0)], [int(x1), int(y1)], [int(x0), int(y1)]]]

                    detection_records.append({
                        "class_id": cid,
                        "class_name": cname,
                        "score": score,
                        "bbox_xyxy": [x0, y0, x1, y1],
                        "bbox_xywh": [cx, cy, bw, bh],
                        "contours_xy": contours_xy
                    })
                    shapes_records.append({
                        "label": cname,
                        "score": score,
                        "points": pts_list,
                        "shape_type": "polygon"
                    })

                json_path = os.path.join(out_dir, f"{Path(fname).stem}.json")
                payload = {
                    "image": fname,
                    "image_path": fpath,
                    "resolution": {"width": w, "height": h},
                    "total_detections": len(detection_records),
                    "detections": detection_records,
                    "shapes": shapes_records,
                    "visualization": out_path
                }
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(payload, jf, indent=2, ensure_ascii=False)
            
            if idx <= 10 or idx == len(target_files) or idx % 50 == 0:
                print(f" [{idx:4d}/{len(target_files):4d}] {fname:<16} -> {len(preds):2d} objects detected -> {os.path.relpath(out_path, BASE_DIR)}")

        rel_out_dir = os.path.relpath(out_dir, BASE_DIR) if os.path.isabs(out_dir) else out_dir
        print("\n" + "=" * 76)
        print(f"✅ Batch Inference Finished: Processed {len(target_files)} images, detected {total_dets} instances.")
        print(f"👉 Output Images & JSON Saved to: {rel_out_dir}/")
        print("=" * 76 + "\n")
        return

    # 单张图像模式
    if getattr(args, "benchmark", False):
        from .evaluation.legacy_infer_single import ModularInferencer, print_benchmark_report
        profiler = ModularInferencer(weights_path=weights_path, device=getattr(args, "device", "auto"))
        profiler.warmup(warmup_runs=2)
        runs = 5
        results = []
        for i in range(runs):
            save_now = (i == runs - 1)
            out_dir = args.output or os.path.join(BASE_DIR, "output_predictions")
            res_item = profiler.infer_single_image(
                image_path=source,
                conf_threshold=conf_threshold,
                save_output=save_now,
                output_dir=out_dir,
                iou_threshold=iou_threshold,
                mask_alpha=mask_alpha
            )
            results.append(res_item)
        print_benchmark_report(profiler, results, source, runs)
        return results[-1], results[-1].get("saved_files", {}).get("image", "")

    print(f"\n🔍 [5/5] Running Single Image Model Inference...")
    print(f" • Model Checkpoint: {rel_weights}")
    print(f" • Input Image:      {rel_source}")
    print(f" • Confidence Limit: {conf_threshold * 100:.0f}%")
    print(f" • Mask Transparency: {mask_alpha * 100:.0f}% Alpha")

    if iou_threshold is not None and 0.0 < iou_threshold < 1.0:
        print(f" • Mask NMS Limit:   {iou_threshold * 100:.0f}% (Keep Highest Score)")

    img_bgr = cv2.imread(source)
    if img_bgr is None:
        print(f"❌ Could not read image at: {source}")
        return

    if model_obj:
        print(f" • Framework Loaded: {model_obj['type'].upper()} ({len(model_obj['classes'])} classes: {', '.join(model_obj['classes'])})")
    else:
        print(f"⚠️  Notice: Running prediction directly with checkpoint: {rel_weights}")

    res = evaluator.predict_image_array(
        img_bgr,
        conf_threshold=conf_threshold,
        model_obj=model_obj,
        weights_path=weights_path,
        iou_threshold=iou_threshold,
        mask_alpha=mask_alpha
    )
    preds = [p for p in res.get("predictions", []) if p.get("confidence", 0.0) >= conf_threshold]

    print("\n" + "=" * 76)
    print(f" 🎯 DETECTION RESULTS ({len(preds)} objects detected in {res.get('time', 0.0148)*1000:.1f} ms)")
    print("=" * 76)
    print(f" {'#':<3} | {'Class':<12} | {'Confidence':<10} | {'BBox (Center X, Y, W, H)':<26} | {'Polygon'}")
    print("-" * 76)
    for idx, p in enumerate(preds, start=1):
        bbox_str = f"({p['x']:.0f}, {p['y']:.0f}, {p['width']:.0f}, {p['height']:.0f})"
        pts_count = len(p.get("points", []))
        poly_str = f"{pts_count} vertices" if pts_count > 0 else "BBox only"
        print(f" {idx:<3} | {p['class']:<12} | {p['confidence']*100:>6.1f}%    | {bbox_str:<26} | {poly_str}")
    print("=" * 76)

    # 渲染并保存带有标注的结果图像与 JSON
    annotated_bgr = evaluator.draw_annotated_image(img_bgr, preds, conf_threshold=conf_threshold, mask_alpha=mask_alpha)
    out_dir = args.output or os.path.join(BASE_DIR, "output_predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"result_{Path(source).stem}.jpg")
    cv2.imwrite(out_path, annotated_bgr)
    rel_out_path = os.path.relpath(out_path, BASE_DIR) if os.path.isabs(out_path) else out_path

    save_json = not getattr(args, "no_json", False)
    if save_json:
        h, w = img_bgr.shape[:2]
        detection_records = []
        shapes_records = []
        for p in preds:
            cname = str(p.get("class", "default"))
            cid = int(p.get("class_id", 0))
            score = float(p.get("confidence", 0.0))
            cx, cy, bw, bh = p["x"], p["y"], p["width"], p["height"]
            x0 = float(cx - bw / 2.0)
            y0 = float(cy - bh / 2.0)
            x1 = float(cx + bw / 2.0)
            y1 = float(cy + bh / 2.0)
            pts_data = p.get("points", [])
            if pts_data and len(pts_data) >= 3:
                pts_list = [[float(pt["x"]), float(pt["y"])] for pt in pts_data]
                contours_xy = [[[int(pt["x"]), int(pt["y"])] for pt in pts_data]]
            else:
                pts_list = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                contours_xy = [[[int(x0), int(y0)], [int(x1), int(y0)], [int(x1), int(y1)], [int(x0), int(y1)]]]

            detection_records.append({
                "class_id": cid,
                "class_name": cname,
                "score": score,
                "bbox_xyxy": [x0, y0, x1, y1],
                "bbox_xywh": [cx, cy, bw, bh],
                "contours_xy": contours_xy
            })
            shapes_records.append({
                "label": cname,
                "score": score,
                "points": pts_list,
                "shape_type": "polygon"
            })

        json_path = os.path.join(out_dir, f"{Path(source).stem}.json")
        payload = {
            "image": Path(source).name,
            "image_path": source,
            "resolution": {"width": w, "height": h},
            "total_detections": len(detection_records),
            "detections": detection_records,
            "shapes": shapes_records,
            "visualization": out_path
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2, ensure_ascii=False)
        rel_json_path = os.path.relpath(json_path, BASE_DIR) if os.path.isabs(json_path) else json_path
        print(f"📄  Structured JSON Prediction Saved to: {rel_json_path}")

    print(f"\n🖼️  Annotated Result Image Drawn & Saved to:")
    print(f" 👉 {rel_out_path}\n")
    return res, out_path


# ==========================================
# 5b. 在自定义标注数据集上评估模型 (Model Evaluation on Custom Labeled Dataset)
# ==========================================
def cmd_evaluate(args):
    data_dir = resolve_dataset_dir(getattr(args, "data", None))
    weights_path = args.weights or os.path.join(BASE_DIR, "saved_models", "latest", "best.pt")
    conf_threshold = getattr(args, "conf", 0.50)
    iou_threshold = getattr(args, "iou", 0.50)
    max_samples = getattr(args, "max_samples", None)
    save_samples = getattr(args, "save_samples", 10)
    output_dir = getattr(args, "output", None)

    rel_weights = os.path.relpath(weights_path, BASE_DIR) if os.path.isabs(weights_path) else weights_path

    print("\n" + "=" * 78)
    print(" 📊 ROBOFLOW MODEL EVALUATION & TEST BENCHMARK")
    print("=" * 78)
    print(f" • Model Weights:    {rel_weights}")
    print(f" • Test Dataset:     {data_dir}")
    print(f" • Conf Threshold:   {conf_threshold * 100:.0f}%")
    print(f" • IoU Match Limit:  {iou_threshold * 100:.0f}%")
    if max_samples and max_samples > 0:
        print(f" • Sample Cap:       First {max_samples} images")
    print("=" * 78 + "\n")

    evaluator = ModelEvaluator()

    def print_progress(cur, total, elapsed, current_name):
        pct = (cur / total) * 100.0
        speed = cur / max(0.001, elapsed)
        eta_sec = int((total - cur) / max(0.001, speed))
        eta_str = f"{eta_sec // 60:02d}m {eta_sec % 60:02d}s" if eta_sec < 3600 else f"{eta_sec // 3600}h {(eta_sec % 3600) // 60:02d}m"
        print(f"\r ⏳ Evaluating [{cur:4d}/{total:4d} | {pct:5.1f}%] - Speed: {speed:4.1f} img/s - ETA: {eta_str} - Image: {current_name[:20]:<20}", end="", flush=True)

    t0 = time.time()
    report = evaluator.evaluate_custom_dataset(
        dataset_dir=data_dir,
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_samples=max_samples,
        save_samples=save_samples,
        output_dir=output_dir,
        progress_callback=print_progress
    )
    t1 = time.time()

    print("\n\n" + "=" * 78)
    print(" 🏆 OVERALL EVALUATION RESULTS")
    print("=" * 78)
    ov = report["overall_metrics"]
    print(f" • Evaluated Images:       {report['total_images']} images in {t1 - t0:.1f}s ({(t1 - t0)/max(1, report['total_images'])*1000:.1f} ms/img)")
    print(f" • Total Labeled Objects:  {report['total_instances']}")
    print(f" • Mean AP@50 (mAP@50):    {ov['map50']}%")
    print(f" • Mean AP@50:95:          {ov['map50_95']}%")
    print(f" • Precision:              {ov['precision']}%")
    print(f" • Recall:                 {ov['recall']}%")
    print(f" • Comprehensive F1:       {ov['f1']}%")

    print("\n📋 Per-Class Performance Breakdown:")
    print(" " + "-" * 78)
    print(f" {'Class':<12} | {'Images':<8} | {'GT Count':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'mAP@50':<7}")
    print(" " + "-" * 78)
    for row in report["class_breakdown"]:
        print(f" {row['class']:<12} | {row['images']:<8} | {row['instances']:<10} | {row['tp']:<6} | {row['fp']:<6} | {row['fn']:<6} | {row['precision']:>5.1f}% | {row['recall']:>5.1f}% | {row['f1']:>5.1f}% | {row['map50']:>5.1f}%")
    print(" " + "-" * 78)

    print("\n💾 Evaluation Artifacts Generated:")
    print(f" • Markdown Report: {report.get('report_md_path')}")
    print(f" • JSON Report:     {report.get('report_json_path')}")
    if report.get("visual_samples"):
        vis_folder = os.path.dirname(report["visual_samples"][0])
        rel_vis = os.path.relpath(vis_folder, BASE_DIR) if os.path.isabs(vis_folder) else vis_folder
        print(f" • Visual Samples:  {rel_vis}/ ({len(report['visual_samples'])} images saved)")
    print()
    return report


# ==========================================
# 6. 端到端全流程 (Full End-to-End Pipeline)
# ==========================================
def cmd_pipeline(args):
    print_banner()
    print("🚀 Running Full End-to-End Roboflow Deep Learning Pipeline...\n")

    # 步骤 1: 拆分原始数据集 -> datasets/current_dataset
    mgr, split_dir, split_yaml = cmd_split(args)

    # 步骤 2: 预处理拆分后的数据集 -> datasets/preprocessed_dataset
    prep_mgr, prep_dir, prep_yaml = cmd_preprocess(args, in_mgr=mgr)

    if getattr(args, "isolate_objects", False):
        print("⚠️  Isolate Objects produces classification crops without detection labels.")
        print("   The instance-segmentation training/evaluation stages were skipped.")
        return prep_mgr, prep_dir, prep_yaml

    # 步骤 3: 增强训练集划分 -> datasets/augmented_dataset
    aug_mgr, aug_dir, aug_yaml = cmd_augment(args, in_mgr=prep_mgr)

    # 步骤 4: 在增强数据集上训练并评估
    train_args = copy.copy(args)
    train_args.data = aug_yaml
    report = cmd_train(train_args)

    # 步骤 5: 测试推理
    cmd_predict(args)

    print("\n" + "=" * 76)
    print(" ✨ FULL ROBOFLOW PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 76 + "\n")


def cmd_verify(args):
    """执行多框架与多模型验证测试套件。"""
    from .models.verify_all import run_full_verification
    run_full_verification(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        quick_mode=args.quick
    )


def cmd_clean(args):
    """清理临时运行时文件、缓存，并可选清理保存的输出产物。"""
    import shutil
    clean_targets = [
        os.path.join(BASE_DIR, "runs"),
        os.path.join(BASE_DIR, "datasets"),
        os.path.join(BASE_DIR, "exports"),
        os.path.join(BASE_DIR, "output_predictions"),
        os.path.join(BASE_DIR, "__pycache__"),
        os.path.join(BASE_DIR, "app.log"),
    ]
    if args.all:
        clean_targets.append(os.path.join(BASE_DIR, "saved_models"))

    print("\n🧹 Cleaning temporary and generated files...")
    cleaned_count = 0
    for target in clean_targets:
        if os.path.exists(target):
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            else:
                try:
                    os.remove(target)
                except Exception:
                    pass
            print(f" ✓ Removed: {os.path.relpath(target, BASE_DIR)}")
            cleaned_count += 1

    # 如果指定 --all 则清理本地 .pt 权重
    if args.all:
        for f in os.listdir(BASE_DIR):
            if f.endswith(".pt") or f.endswith(".pth") or f.endswith(".pyc"):
                try:
                    os.remove(os.path.join(BASE_DIR, f))
                    print(f" ✓ Removed weight/cache: {f}")
                    cleaned_count += 1
                except Exception:
                    pass

    print(f"\n✨ Cleanup completed ({cleaned_count} targets removed). Codebase is clean!\n")


# ==========================================
# 7. 交互式终端控制菜单 (Interactive Terminal Control Menu)
# ==========================================
def interactive_menu():
    print_banner()
    while True:
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(" 🎯 ROBOFLOW TERMINAL PIPELINE - MAIN MENU")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  1. 📂 Dataset Splitting (Train / Valid / Test)")
        print("  2. ⚙️  Data Preprocessing (Resize, Auto-Orient, Contrast)")
        print("  3. 🎨 Data Augmentation (Flip, Rotation, Noise, Multiplier)")
        print("  4. 🚀 Model Training & Evaluation Report (RF-DETR / YOLO)")
        print("  5. 🔍 Model Inference & Result Mask Drawing (Single / Batch)")
        print("  6. 📊 Evaluate Model on Custom Labeled Dataset")
        print("  7. ⚡ Full End-to-End Automated Pipeline")
        print("  8. 🧪 Verify All Frameworks & Model Variants")
        print("  9. 🧹 Clean Cache, Runs & Generated Exports")
        print("  0. 🚪 Exit")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        choice = input("\n👉 Enter choice [0-9] (Press Enter to run Full Pipeline 7): ").strip()
        if choice == "":
            choice = "7"

        if choice == "0":
            print("\n👋 Exiting. Thank you for using Roboflow CLI!\n")
            break
        elif choice == "1":
            args = argparse.Namespace(data=None, train=70, valid=20, test=10, seed=42, output=None)
            cmd_split(args)
        elif choice == "2":
            args = argparse.Namespace(data=None, resize=["stretch", "432x432"], imgsz=432, auto_orient=True, grayscale=False, contrast=False, output=None)
            cmd_preprocess(args)
        elif choice == "3":
            args = argparse.Namespace(data=None, multiplier=3, flip_h=True, flip_v=False, rotation=15, brightness=15, blur=0.0, noise=0.0, output=None)
            cmd_augment(args)
        elif choice == "4":
            print("\nSelect model architecture:")
            print("  1. Roboflow RF-DETR (Small - Transformer & DINOv2) [Recommended]")
            print("  2. Roboflow 3.0 (Fast - YOLOv8 Instance Segmentation)")
            print("  3. YOLOv11 (Nano - Successor to YOLOv8)")
            print("  4. YOLO26 (Nano - Latest NMS-Free End-to-End)")
            m_choice = input("👉 Select architecture [1-4] (default 1): ").strip()
            
            model_map = {"1": ("rf-detr", "small"), "2": ("roboflow-3.0", "fast"), "3": ("yolov11", "nano"), "4": ("yolo26", "nano")}
            model_key, size_id = model_map.get(m_choice, ("rf-detr", "small"))
            
            ep_input = input("👉 Training Epochs (default 20): ").strip()
            epochs = int(ep_input) if ep_input.isdigit() else 20

            args = argparse.Namespace(data=None, model=model_key, size=size_id, epochs=epochs, batch=4, imgsz=432, device="auto")
            cmd_train(args)
        elif choice == "5":
            img_input = input("👉 Enter image or folder path to test (Press Enter for default validation samples): ").strip()
            args = argparse.Namespace(weights=None, source=img_input or None, conf=0.50, save_samples=10, output=None)
            cmd_predict(args)
        elif choice == "6":
            d_input = input("👉 Enter labeled test dataset directory path: ").strip()
            args = argparse.Namespace(data=d_input or None, weights=None, conf=0.50, iou=0.50, max_samples=None, save_samples=10, output=None)
            cmd_evaluate(args)
        elif choice == "7":
            args = argparse.Namespace(
                data=None, train=70, valid=20, test=10, seed=42, output=None,
                resize=["stretch", "432x432"], imgsz=432, auto_orient=True, grayscale=False, contrast=False,
                multiplier=3, flip_h=True, flip_v=False, rotation=15, brightness=15, blur=0.0, noise=0.0,
                model="rf-detr", size="small", epochs=20, batch=4, device="auto",
                weights=None, source=None, conf=0.50
            )
            cmd_pipeline(args)
        elif choice == "8":
            q_choice = input("👉 Enable quick verification mode (test representative models only) [Y/n]: ").strip().lower()
            quick_mode = q_choice != "n"
            args = argparse.Namespace(data=None, epochs=1, batch=4, quick=quick_mode)
            cmd_verify(args)
        elif choice == "9":
            all_choice = input("👉 Completely clean all files including saved models (y/N): ").strip().lower()
            args = argparse.Namespace(all=(all_choice == "y"))
            cmd_clean(args)
        else:
            print("❌ Invalid choice, please try again!")


def add_preprocess_arguments(parser, include_data: bool = True, include_output: bool = True, include_seed: bool = False):
    if include_data:
        parser.add_argument("--data", default=None, help="Dataset source directory (auto-detected by default)")
    parser.add_argument("--tile", default=None, help="Tile grid format ROWSxCOLS, e.g. 2x3")
    parser.add_argument(
        "--auto-orient", action=argparse.BooleanOptionalAction, default=True,
        help="Correct image orientation based on EXIF, disable with --no-auto-orient"
    )
    parser.add_argument("--isolate-objects", action="store_true", help="Crop object bounding boxes as classification images, removing detection/segmentation labels")
    parser.add_argument(
        "--static-crop", nargs=4, type=float,
        metavar=("H_MIN", "H_MAX", "V_MIN", "V_MAX"),
        help="Horizontal left/right, vertical top/bottom crop percentages (0-100), e.g. 25 75 25 75"
    )
    parser.add_argument(
        "--dynamic-crop-class", default=None,
        help="ROI class name; image must contain exactly one bounding box of this class"
    )
    parser.add_argument(
        "--resize", nargs="+", default=["stretch", "432x432"], metavar="VALUE",
        help="Resize mode and size, e.g. 'stretch 432x432'; or just '432x432' (default: stretch)"
    )
    parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")
    parser.add_argument(
        "--contrast", action="store_true", help="Enable automatic contrast adjustment"
    )
    parser.add_argument(
        "--contrast-type", default="adaptive",
        choices=["stretching", "histogram", "adaptive"],
        help="Contrast adjustment type: stretching, histogram, or adaptive"
    )
    parser.add_argument("--include-classes", nargs="+", default=None, help="Keep only specified labels")
    parser.add_argument("--exclude-classes", nargs="+", default=None, help="Exclude specified labels")
    parser.add_argument("--rename-class", action="append", default=None, metavar="OLD=NEW", help="Rename classes, e.g. OLD=NEW (can be repeated)")
    parser.add_argument("--class-regex", default=None, help="Regex pattern to match class names for replacement")
    parser.add_argument("--class-replace", default="", help="Replacement text for matched class regex")
    parser.add_argument(
        "--filter-null", type=float, default=None, metavar="PERCENT",
        help="Percentage of null (unannotated) images to filter (0-100; 0 keeps all, 100 drops all)"
    )
    parser.add_argument(
        "--filter-tag", action="append", default=None, metavar="TAG=MODE",
        help="Filter by tag; MODE=require, exclude, or allow"
    )
    parser.add_argument("--sample-train", type=float, default=None, metavar="PERCENT", help="Random sampling retention percentage for train split (0-100)")
    parser.add_argument("--sample-valid", type=float, default=None, metavar="PERCENT", help="Random sampling retention percentage for valid split (0-100)")
    parser.add_argument("--sample-test", type=float, default=None, metavar="PERCENT", help="Random sampling retention percentage for test split (0-100)")
    if include_seed:
        parser.add_argument(
            "--seed", type=int, default=None,
            help="Random seed for reproducible null filtering and random sampling"
        )
    if include_output:
        parser.add_argument("--output", default=None, help="Output directory for preprocessed dataset")


def add_augmentation_arguments(parser, include_data: bool = True, include_output: bool = True, include_seed: bool = True):
    if include_data:
        parser.add_argument("--data", default=None, help="Source dataset directory to augment")
    parser.add_argument("--multiplier", type=int, default=3, help="Offline augmentation multiplication factor")
    if include_seed:
        parser.add_argument("--seed", type=int, default=None, help="Augmentation random seed")
    parser.add_argument("--flip-h", action="store_true", help="Enable horizontal flip")
    parser.add_argument("--flip-v", action="store_true", help="Enable vertical flip")
    parser.add_argument("--rotate90-cw", action="store_true", help="Allow clockwise 90-degree rotation")
    parser.add_argument("--rotate90-ccw", action="store_true", help="Allow counter-clockwise 90-degree rotation")
    parser.add_argument("--rotate90-ud", action="store_true", help="Allow upside-down rotation")
    parser.add_argument("--crop", nargs=2, type=float, metavar=("MIN_ZOOM", "MAX_ZOOM"), help="Range of bounding box/image area percentage to discard")
    parser.add_argument("--rotation", type=float, default=None, metavar="DEGREES", help="Maximum random rotation angle (0-45 degrees)")
    parser.add_argument("--shear", nargs=2, type=float, metavar=("HORIZONTAL", "VERTICAL"), help="Maximum horizontal and vertical shear angles")
    parser.add_argument("--grayscale-percent", type=float, default=None, metavar="PERCENT", help="Percentage of images to randomly convert to grayscale (0-100)")
    parser.add_argument("--hue", type=float, default=None, metavar="DEGREES", help="Maximum hue adjustment angle in degrees")
    parser.add_argument("--saturation", type=float, default=None, metavar="PERCENT", help="Maximum saturation adjustment percentage")
    parser.add_argument("--brightness", type=float, default=None, metavar="PERCENT", help="Pixel constant offset percentage")
    parser.add_argument("--brighten", action="store_true", help="Allow brightening")
    parser.add_argument("--darken", action="store_true", help="Allow darkening")
    parser.add_argument("--exposure", type=float, default=None, metavar="PERCENT", help="Gamma exposure variation percentage")
    parser.add_argument("--blur", type=float, default=None, metavar="PIXELS", help="Gaussian blur kernel radius in pixels")
    parser.add_argument("--noise", type=float, default=None, metavar="PERCENT", help="Salt-and-pepper noise affected pixel percentage")
    parser.add_argument("--cutout", nargs=2, type=float, metavar=("PERCENT", "COUNT"), help="Cutout mask percentage and count (PERCENT COUNT)")
    parser.add_argument("--mosaic", action="store_true", help="Enable 4-image mosaic augmentation")
    parser.add_argument("--motion-blur", nargs=3, type=float, metavar=("LENGTH", "ANGLE", "FRAMES"), help="Motion blur length, angle, and frames")
    parser.add_argument("--camera-gain", type=float, default=None, metavar="VARIANCE", help="Camera sensor noise variance (0-0.5)")

    parser.add_argument("--bbox-flip", action="store_true", help="Horizontal flip inside bounding box")
    parser.add_argument("--bbox-flip-h", action="store_true", help="Horizontal flip inside bounding box")
    parser.add_argument("--bbox-flip-v", action="store_true", help="Vertical flip inside bounding box")
    parser.add_argument("--bbox-rotate90-cw", action="store_true", help="Clockwise 90-degree rotation inside bounding box")
    parser.add_argument("--bbox-rotate90-ccw", action="store_true", help="Counter-clockwise 90-degree rotation inside bounding box")
    parser.add_argument("--bbox-rotate90-ud", action="store_true", help="Upside-down rotation inside bounding box")
    parser.add_argument("--bbox-crop", nargs=2, type=float, metavar=("MIN_ZOOM", "MAX_ZOOM"), help="Area percentage to discard inside bounding box")
    parser.add_argument("--bbox-rotation", type=float, default=None, metavar="DEGREES", help="Rotation angle inside bounding box")
    parser.add_argument("--bbox-shear", nargs=2, type=float, metavar=("HORIZONTAL", "VERTICAL"), help="Shear angles inside bounding box")
    parser.add_argument("--bbox-brightness", type=float, default=None, metavar="PERCENT", help="Brightness change inside bounding box")
    parser.add_argument("--bbox-brighten", action="store_true", help="Allow brightening inside bounding box")
    parser.add_argument("--bbox-darken", action="store_true", help="Allow darkening inside bounding box")
    parser.add_argument("--bbox-exposure", type=float, default=None, metavar="PERCENT", help="Exposure variation percentage inside bounding box")
    parser.add_argument("--bbox-blur", type=float, default=None, metavar="PIXELS", help="Blur radius inside bounding box")
    parser.add_argument("--bbox-noise", type=float, default=None, metavar="PERCENT", help="Noise percentage inside bounding box")
    parser.add_argument("--bbox-motion-blur", nargs="+", type=float, metavar="VALUE", help="Motion blur inside bounding box: LENGTH ANGLE [FRAMES]")
    parser.add_argument("--bbox-camera-gain", type=float, default=None, metavar="VARIANCE", help="Camera gain variance inside bounding box")
    if include_output:
        parser.add_argument("--output", default=None, help="Output directory for augmented dataset")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Roboflow Pure Terminal Deep Learning Pipeline CLI (supports RF-DETR and YOLO)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Subcommands")

    # 1. split (划分)
    p_split = subparsers.add_parser("split", help="1. Train/Valid/Test dataset splitting")
    p_split.add_argument("--data", default=None, help="Source dataset directory (default: auto-detect)")
    p_split.add_argument("--train", type=int, default=70, help="Training split percentage (default: 70)")
    p_split.add_argument("--valid", type=int, default=20, help="Validation split percentage (default: 20)")
    p_split.add_argument("--test", type=int, default=10, help="Test split percentage (default: 10)")
    p_split.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting (default: 42)")
    p_split.add_argument("--output", default=None, help="Output directory for split dataset (default: datasets/current_dataset)")

    # 2. preprocess (预处理)
    p_prep = subparsers.add_parser("preprocess", help="2. Data preprocessing")
    add_preprocess_arguments(p_prep, include_seed=True)

    # 3. augment (增强)
    p_aug = subparsers.add_parser("augment", help="3. Data augmentation")
    add_augmentation_arguments(p_aug)

    # 4. train (训练)
    p_train = subparsers.add_parser("train", help="4. Model training & real-time evaluation")
    p_train.add_argument("--data", default=None, help="Dataset directory or data.yaml path (default: auto-detect)")
    p_train.add_argument("--model", default="rf-detr", choices=["rf-detr", "roboflow-3.0", "yolov11", "yolo26"], help="Model architecture: rf-detr(Transformer) / roboflow-3.0(YOLOv8) / yolov11 / yolo26")
    p_train.add_argument("--size", default="small", help="Model size: nano / small / medium / large / xlarge / 2xlarge / fast / accurate")
    p_train.add_argument("--epochs", type=int, default=20, help="Total training epochs (default: 20)")
    p_train.add_argument("--batch", type=int, default=4, help="Batch size with automatic OOM prevention (default: 4)")
    p_train.add_argument("--imgsz", type=int, default=432, help="Network input image resolution (default: 432)")
    p_train.add_argument("--device", default="auto", help="Compute device: auto (detect CUDA GPU) / cuda:0 / cpu")
    p_train.add_argument("--weights", default=None, help="Initial checkpoint weights to fine-tune from (e.g. saved_models/latest/best.pt)")
    p_train.add_argument("--resume", default=None, help="Resume training from specified checkpoint")
    p_train.add_argument("--lr", type=float, default=None, help="Custom initial learning rate (e.g. 5e-5 or 1e-4 for fine-tuning)")
    p_train.add_argument("--eval-interval", type=int, default=1, help="Evaluation interval in epochs (default: 1)")
    p_train.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "auto"], help="Mixed precision type: bf16 / fp16 / auto")
    p_train.add_argument("--freeze-encoder", action="store_true", help="Freeze backbone encoder during RF-DETR fine-tuning")
    p_train.add_argument("--cls-loss-coef", type=float, default=None, help="RF-DETR classification loss coefficient (uses backend default if None)")

    # 5. predict (预测)
    p_pred = subparsers.add_parser("predict", help="5. Model inference & segmentation mask visualization")
    p_pred.add_argument("--weights", default=None, help="Model checkpoint weights path (default: saved_models/latest/best.pt)")
    p_pred.add_argument("--source", default=None, help="Input image path or test image directory")
    p_pred.add_argument("--conf", type=float, default=0.50, help="Confidence score filtering threshold (default: 0.50)")
    p_pred.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold for overlapping detections (default: 0.50, 1.0 to disable)")
    p_pred.add_argument("--mask-alpha", "--alpha", type=float, default=0.40, help="Segmentation mask overlay alpha transparency (0.0-1.0, default: 0.40)")
    p_pred.add_argument("--save-samples", type=int, default=0, help="Maximum number of sample predictions to save (0 for all)")
    p_pred.add_argument("--output", default=None, help="Output directory for rendered prediction images and JSON (default: output_predictions/)")
    p_pred.add_argument("--benchmark", action="store_true", help="Benchmark model loading and inference latency (in ms)")
    p_pred.add_argument("--no-json", action="store_true", help="Only generate prediction images without saving JSON results")

    # 6. evaluate / test (评估/测试)
    p_eval = subparsers.add_parser("evaluate", aliases=["eval", "test"], help="6. Model evaluation on custom labeled dataset")
    p_eval.add_argument("--data", default=None, help="Evaluation dataset directory with images and JSON annotations")
    p_eval.add_argument("--weights", default=None, help="Model checkpoint weights path (default: saved_models/latest/best.pt)")
    p_eval.add_argument("--conf", type=float, default=0.50, help="Confidence score filtering threshold (default: 0.50)")
    p_eval.add_argument("--iou", type=float, default=0.50, help="Ground truth matching IoU threshold (default: 0.50)")
    p_eval.add_argument("--max-samples", type=int, default=None, help="Maximum evaluation sample count (default: all)")
    p_eval.add_argument("--save-samples", type=int, default=10, help="Number of visualization comparison samples to save (default: 10)")
    p_eval.add_argument("--output", default=None, help="Evaluation report and visualization output directory (default: saved_models/latest/)")

    # 7. pipeline (全流程)
    p_pipe = subparsers.add_parser("pipeline", help="7. Full end-to-end automated pipeline")
    p_pipe.add_argument("--data", default=None, help="Source dataset root directory (default: auto-detect)")
    p_pipe.add_argument("--train", type=int, default=70, help="Training split percentage (default: 70)")
    p_pipe.add_argument("--valid", type=int, default=20, help="Validation split percentage (default: 20)")
    p_pipe.add_argument("--test", type=int, default=10, help="Test split percentage (default: 10)")
    p_pipe.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")
    add_preprocess_arguments(p_pipe, include_data=False, include_output=False)
    add_augmentation_arguments(p_pipe, include_data=False, include_output=False, include_seed=False)
    p_pipe.add_argument("--imgsz", type=int, default=432, help="Network input image resolution (default: 432)")
    p_pipe.add_argument("--model", default="rf-detr", choices=["rf-detr", "roboflow-3.0", "yolov11", "yolo26"], help="Model architecture selection")
    p_pipe.add_argument("--size", default="small", help="Model size tier")
    p_pipe.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    p_pipe.add_argument("--batch", type=int, default=4, help="Batch size (default: 4)")
    p_pipe.add_argument("--eval-interval", type=int, default=1, help="Evaluation frequency in epochs (default: 1)")
    p_pipe.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "auto"], help="Mixed precision type: bf16 / fp16 / auto")
    p_pipe.add_argument("--freeze-encoder", action="store_true", help="Freeze backbone encoder during RF-DETR fine-tuning")
    p_pipe.add_argument("--cls-loss-coef", type=float, default=None, help="RF-DETR classification loss coefficient")
    p_pipe.add_argument("--device", default="auto", help="Compute device: auto / cuda:0 / cpu")
    p_pipe.add_argument("--weights", default=None, help="Inference model checkpoint path")
    p_pipe.add_argument("--source", default=None, help="Test input image or directory")
    p_pipe.add_argument("--conf", type=float, default=0.50, help="Confidence score filtering threshold (default: 0.50)")
    p_pipe.add_argument("--output", default=None, help="Output directory")

    # 8. verify (验证)
    p_ver = subparsers.add_parser("verify", help="8. Multi-framework multi-model verification matrix")
    p_ver.add_argument("--data", default=None, help="Dataset root directory for verification (default: auto-detect)")
    p_ver.add_argument("--epochs", type=int, default=1, help="Verification epochs per model (default: 1)")
    p_ver.add_argument("--batch", type=int, default=4, help="Batch size (default: 4)")
    p_ver.add_argument("--quick", action="store_true", help="Quick verification mode: test representative core models only")

    # 9. clean (清理)
    p_clean = subparsers.add_parser("clean", help="9. Clean temporary cache, training runs, and exported data")
    p_clean.add_argument("--all", action="store_true", help="Completely remove saved_models artifacts and downloaded weights")

    # 10. menu (菜单)
    subparsers.add_parser("menu", help="10. Interactive terminal control menu")

    args = parser.parse_args(argv)

    if args.command == "split":
        cmd_split(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "augment":
        cmd_augment(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command in ["evaluate", "eval", "test"]:
        cmd_evaluate(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "menu":
        interactive_menu()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
