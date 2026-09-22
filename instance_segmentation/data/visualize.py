"""图像预处理与数据增强的可视化冒烟测试工具。

该工具专为单张图像设计，生成可视化测试结果文件而非直接修改数据集。
工具复用了生产环境中的预处理与数据增强实现，方便工程师直观检查训练流水线中实际执行的具体操作。

使用示例:
    python visualize_transforms.py --source image.jpg \
        --scope preprocess --method resize \
        --params '{"mode":"fit_reflect","width":640,"height":640}'

    python visualize_transforms.py --source image.jpg \
        --scope augment --all --output previews --seed 42

    python visualize_transforms.py --source image.jpg \
        --scope bbox --all --bbox 100 80 300 280 \
        --output previews --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

import sys

try:
    from .augmentation import AugmentationPipeline
    from .classes import load_class_names
    from .dataset import AnnotationItem, ImageSample
    from .preprocessing import PreprocessingPipeline
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from instance_segmentation.data.augmentation import AugmentationPipeline
    from instance_segmentation.data.classes import load_class_names
    from instance_segmentation.data.dataset import AnnotationItem, ImageSample
    from instance_segmentation.data.preprocessing import PreprocessingPipeline



PREPROCESS_METHODS = [
    "auto_orient",
    "resize",
    "grayscale",
    "contrast",
    "tile",
    "static_crop",
    "dynamic_crop",
    "isolate_objects",
    "modify_classes",
    "filter_null",
    "filter_tag",
    "random_sample",
]

IMAGE_AUGMENT_METHODS = [
    "flip",
    "rotate90",
    "crop",
    "rotation",
    "shear",
    "grayscale",
    "hue",
    "saturation",
    "brightness",
    "exposure",
    "blur",
    "noise",
    "cutout",
    "mosaic",
    "motion_blur",
    "camera_gain",
]

BBOX_AUGMENT_METHODS = [
    "flip",
    "rotate90",
    "crop",
    "rotation",
    "shear",
    "brightness",
    "exposure",
    "blur",
    "noise",
    "motion_blur",
    "camera_gain",
]


PREPROCESS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "auto_orient": {},
    "resize": {"mode": "stretch", "width": 640, "height": 640},
    "grayscale": {},
    "contrast": {"type": "adaptive"},
    "tile": {"grid_rows": 2, "grid_cols": 3},
    "static_crop": {"h_min": 25, "h_max": 75, "v_min": 25, "v_max": 75},
    "dynamic_crop": {},
    "isolate_objects": {},
    "modify_classes": {},
    "filter_null": {"min_percent": 0},
    "filter_tag": {"rules": []},
    "random_sample": {"train_percent": 100, "valid_percent": 100, "test_percent": 100},
}

IMAGE_AUGMENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "flip": {"horizontal": True, "vertical": False},
    "rotate90": {"clockwise": True, "counter_clockwise": False, "upside_down": False},
    "crop": {"min_zoom": 20, "max_zoom": 20},
    "rotation": {"angle": 15},
    "shear": {"horizontal": 10, "vertical": 10},
    "grayscale": {"percent": 100},
    "hue": {"angle": 15},
    "saturation": {"percent": 25},
    "brightness": {"percent": 15, "brighten": True, "darken": True},
    "exposure": {"percent": 15},
    "blur": {"pixels": 2.5},
    "noise": {"percent": 1},
    "cutout": {"percent": 10, "count": 3},
    "mosaic": {},
    "motion_blur": {"length": 15, "angle": 0, "frames": 2},
    "camera_gain": {"variance": 0.05},
}

BBOX_AUGMENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "flip": {"horizontal": True, "vertical": False},
    "rotate90": {"clockwise": True, "counter_clockwise": False, "upside_down": False},
    "crop": {"min_zoom": 20, "max_zoom": 20},
    "rotation": {"angle": 15},
    "shear": {"horizontal": 10, "vertical": 10},
    "brightness": {"percent": 15, "brighten": True, "darken": True},
    "exposure": {"percent": 15},
    "blur": {"pixels": 2.5},
    "noise": {"percent": 1},
    "motion_blur": {"length": 15, "angle": 0, "frames": 2},
    "camera_gain": {"variance": 0.05},
}


def _slug(value: str) -> str:
    """生成安全的文件名前缀字符串。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "result"


def _load_labelme_annotations(path: Path) -> List[AnnotationItem]:
    """从 LabelMe JSON 文件中加载多边形和矩形标注列表。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations: List[AnnotationItem] = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "rectangle" and len(points) >= 2:
            (x1, y1), (x2, y2) = points[0], points[1]
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            shape_type = "polygon"
        if len(points) >= 3:
            annotations.append(
                AnnotationItem(
                    label=str(shape.get("label", "object")),
                    points=points,
                    shape_type=shape_type,
                )
            )
    return annotations


def _load_yolo_annotations(path: Path, width: int, height: int) -> List[AnnotationItem]:
    """从 YOLO 文本标注文件中加载标注列表并反归一化为像素坐标。"""
    class_names = load_class_names()
    annotations: List[AnnotationItem] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        values = raw_line.strip().split()
        if len(values) < 5:
            continue
        try:
            class_id, cx, cy, bw, bh = map(float, values[:5])
        except ValueError:
            continue
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
        class_index = int(class_id)
        label = class_names[class_index] if 0 <= class_index < len(class_names) else f"class_{class_index}"
        annotations.append(
            AnnotationItem(label=label, points=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        )
    return annotations


def _load_annotations(path: Optional[Path], width: int, height: int) -> List[AnnotationItem]:
    """自动识别并从指定标注文件中加载标注列表。"""
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    if path.suffix.lower() == ".txt":
        return _load_yolo_annotations(path, width, height)
    return _load_labelme_annotations(path)


def _add_boxes(
    annotations: List[AnnotationItem], boxes: Optional[Sequence[Sequence[float]]], label: str
) -> None:
    """向标注列表中追加额外的边界框标注。"""
    for box in boxes or []:
        x1, y1, x2, y2 = map(float, box)
        annotations.append(
            AnnotationItem(
                label=label,
                points=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                shape_type="polygon",
            )
        )


def _image_of(sample: ImageSample) -> np.ndarray:
    """获取样本的图像副本并统一转换为 3 通道 BGR 格式。"""
    image = getattr(sample, "_cached_img", None)
    if image is None:
        image = cv2.imread(sample.image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Unable to read image: {sample.image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _draw_sample(sample: ImageSample, title: str) -> np.ndarray:
    """在样本图像上绘制标注多边形、类别名称及顶部标题栏。"""
    image = _image_of(sample)
    canvas = image.copy()
    colors = [(0, 220, 0), (0, 165, 255), (255, 80, 0), (255, 0, 180)]
    for index, ann in enumerate(sample.annotations):
        points = np.asarray(ann.points, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        if len(points) >= 2:
            color = colors[index % len(colors)]
            cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)
            x, y = points.reshape(-1, 2).min(axis=0)
            cv2.putText(
                canvas,
                str(ann.label),
                (max(0, int(x)), max(18, int(y) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    bar_height = 34
    result = np.zeros((canvas.shape[0] + bar_height, canvas.shape[1], 3), dtype=np.uint8)
    result[bar_height:] = canvas
    cv2.putText(
        result,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def _fit_height(image: np.ndarray, height: int = 480) -> np.ndarray:
    """等比例缩放图像至指定高度。"""
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _comparison(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """水平拼接处理前后的两张图像生成对比图。"""
    left = _fit_height(before)
    right = _fit_height(after)
    target_height = max(left.shape[0], right.shape[0])
    if left.shape[0] != target_height:
        left = _fit_height(left, target_height)
    if right.shape[0] != target_height:
        right = _fit_height(right, target_height)
    return cv2.hconcat([left, right])


def _read_json_object(value: Optional[str], name: str) -> Dict[str, Any]:
    """解析 JSON 字符串并确保其为字典对象。"""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _load_config_file(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """加载配置文件并解析为以方法名为键的配置字典。"""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--config must contain a JSON object keyed by method name")
    result = {}
    for method, config in data.items():
        if not isinstance(config, dict):
            raise ValueError(f"Config for {method} must be a JSON object")
        result[str(method)] = config
    return result


def _merge_config(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并默认配置与用户覆盖配置。"""
    merged = copy.deepcopy(default)
    merged.update(override)
    return merged


def _base_sample(source: Path, annotations: List[AnnotationItem]) -> ImageSample:
    """根据图像路径和标注列表构建基础 ImageSample 样本对象。"""
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {source}")
    height, width = image.shape[:2]
    sample = ImageSample(str(source), width, height, annotations, sample_id=source.stem)
    sample._cached_img = image
    sample.split = "train"
    return sample


def _run_preprocess(sample: ImageSample, method: str, config: Dict[str, Any]) -> List[ImageSample]:
    """调用预处理流水线执行单项预处理。"""
    if method == "dynamic_crop" and not config.get("class") and sample.annotations:
        config["class"] = sample.annotations[0].label
    pipeline = PreprocessingPipeline()
    pipeline.add_step(method, config)
    return pipeline.process_sample(sample)


def _run_image_augment(sample: ImageSample, method: str, config: Dict[str, Any], seed: int) -> List[ImageSample]:
    """调用增强流水线执行单项图像级增强。"""
    random.seed(seed)
    np.random.seed(seed)
    pipeline = AugmentationPipeline({method: config}, multiplier=2)
    pool = [sample, sample, sample, sample] if method == "mosaic" else [sample]
    return [pipeline._apply_all_augs(sample, version=1, pool=pool)]


def _run_bbox_augment(sample: ImageSample, method: str, config: Dict[str, Any], seed: int) -> List[ImageSample]:
    """调用增强流水线执行单项边界框级增强。"""
    random.seed(seed)
    np.random.seed(seed)
    result = AugmentationPipeline()._clone_sample(sample, version=1)
    result._cached_img = AugmentationPipeline().apply_bbox_augmentation(
        _image_of(sample), sample.annotations, f"bbox_{method}", config
    )
    result.height, result.width = result._cached_img.shape[:2]
    return [result]


def _save_result(
    output_dir: Path,
    source_sample: ImageSample,
    result_sample: ImageSample,
    method: str,
    result_index: int,
) -> Dict[str, Any]:
    """保存处理后图像及前后对比图，并返回结果记录元数据。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{result_index:02d}_{_slug(method)}"
    before = _draw_sample(source_sample, "original")
    after = _draw_sample(result_sample, f"{method} result {result_index}")
    result_path = output_dir / f"{prefix}.png"
    compare_path = output_dir / f"{prefix}_compare.png"
    cv2.imwrite(str(result_path), after)
    cv2.imwrite(str(compare_path), _comparison(before, after))
    return {
        "method": method,
        "index": result_index,
        "result": str(result_path),
        "comparison": str(compare_path),
        "width": int(result_sample.width),
        "height": int(result_sample.height),
        "annotations": len(result_sample.annotations),
    }


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Visual smoke-test tool for single-image preprocessing and augmentation, outputting original, processed, and comparison images."
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--source", help="Path to a single test image")
    source_group.add_argument(
        "--dataset-dir",
        help="Directory of labeled images; automatically loads LabelMe JSON annotations with matching stems",
    )
    parser.add_argument(
        "--sample",
        help="Specific image filename (with or without extension) in --dataset-dir to test; defaults to the first image sorted by name",
    )
    parser.add_argument("--output", default="transform_previews", help="Output directory for visual results")
    parser.add_argument(
        "--scope",
        required=False,
        choices=["preprocess", "augment", "bbox"],
        help="preprocess=image preprocessing; augment=image-level augmentation; bbox=bounding box-level augmentation",
    )
    selection = parser.add_mutually_exclusive_group(required=False)
    selection.add_argument("--method", help="Test a single method; see --help-methods for available names")
    selection.add_argument("--all", action="store_true", help="Run all methods for the chosen scope")
    parser.add_argument("--params", help="JSON string of parameters for a single method, e.g. '{\"width\":640,\"height\":640}'")
    parser.add_argument("--config", help="JSON file containing parameters for all methods, formatted as {\"method\": {...}}")
    parser.add_argument("--annotations", help="Optional LabelMe JSON or YOLO TXT annotation file")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Repeatable bounding box coordinates [x1, y1, x2, y2]; used for bbox scope when --annotations is omitted",
    )
    parser.add_argument("--bbox-label", default="object", help="Label for --bbox boxes, defaults to 'object'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--help-methods",
        action="store_true",
        help="Print supported methods for all three scopes and exit",
    )
    return parser


def _print_methods() -> None:
    """打印支持的所有方法名称。"""
    print("preprocess:", ", ".join(PREPROCESS_METHODS))
    print("augment:   ", ", ".join(IMAGE_AUGMENT_METHODS))
    print("bbox:      ", ", ".join(BBOX_AUGMENT_METHODS))


def main(argv=None) -> int:
    """命令行主执行入口。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.help_methods:
        _print_methods()
        return 0
    if not args.scope or (not args.method and not args.all):
        parser.error("Must provide --scope along with either --method or --all; see --help-methods for options")
    if not args.source and not args.dataset_dir:
        parser.error("Must provide either --source or --dataset-dir")
    if args.sample and not args.dataset_dir:
        parser.error("--sample can only be used with --dataset-dir")
    if args.params and args.all:
        parser.error("--params can only be used with --method; use --config for batch execution")

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        if not dataset_dir.is_dir():
            parser.error(f"Dataset directory does not exist: {dataset_dir}")
        image_suffixes = {".bmp", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
        image_paths = sorted(
            path for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_suffixes
        )
        if not image_paths:
            parser.error(f"No usable images found in dataset directory: {dataset_dir}")
        if args.sample:
            requested = Path(args.sample).name
            matches = [path for path in image_paths if path.name == requested or path.stem == requested]
            if not matches:
                parser.error(f"Sample not found in dataset directory: {args.sample}")
            source = matches[0]
        else:
            source = image_paths[0]
        if not args.annotations:
            paired_json = source.with_suffix(".json")
            if paired_json.exists():
                args.annotations = str(paired_json)
    else:
        source = Path(args.source).expanduser().resolve()
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        parser.error(f"Unable to read image: {source}")
    height, width = image.shape[:2]
    annotations = _load_annotations(Path(args.annotations).expanduser().resolve() if args.annotations else None, width, height)
    _add_boxes(annotations, args.bbox, args.bbox_label)
    source_sample = _base_sample(source, annotations)

    method_groups = {
        "preprocess": PREPROCESS_METHODS,
        "augment": IMAGE_AUGMENT_METHODS,
        "bbox": BBOX_AUGMENT_METHODS,
    }
    methods = method_groups[args.scope] if args.all else [args.method]
    assert all(method is not None for method in methods)
    available = set(method_groups[args.scope])
    unknown = [method for method in methods if method not in available]
    if unknown:
        parser.error(f"Scope {args.scope} does not support method(s): {', '.join(unknown)}")
    if args.scope == "bbox" and not source_sample.annotations:
        parser.error("bbox scope requires --annotations or at least one --bbox")

    config_file = _load_config_file(Path(args.config).expanduser().resolve() if args.config else None)
    single_params = _read_json_object(args.params, "--params")
    output_root = Path(args.output).expanduser().resolve() / args.scope
    output_root.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_root / "00_original.png"), _draw_sample(source_sample, "original"))

    manifest: Dict[str, Any] = {
        "source": str(source),
        "scope": args.scope,
        "seed": args.seed,
        "results": [],
        "skipped": [],
    }

    for method in methods:
        if args.scope == "preprocess":
            defaults = PREPROCESS_DEFAULTS[method]
        elif args.scope == "augment":
            defaults = IMAGE_AUGMENT_DEFAULTS[method]
        else:
            defaults = BBOX_AUGMENT_DEFAULTS[method]
        config = _merge_config(defaults, config_file.get(method, {}))
        if not args.all:
            config = _merge_config(config, single_params)

        try:
            if args.scope == "preprocess":
                results = _run_preprocess(source_sample, method, config)
            elif args.scope == "augment":
                results = _run_image_augment(source_sample, method, config, args.seed)
            else:
                results = _run_bbox_augment(source_sample, method, config, args.seed)
        except Exception as exc:  # 若单个可选方法不适用，跳过并继续保持批处理可用
            manifest["skipped"].append({"method": method, "reason": str(exc)})
            print(f"[SKIP] {method}: {exc}")
            continue

        if not results:
            reason = "This method requires annotations or matching criteria; no output generated for this image"
            manifest["skipped"].append({"method": method, "reason": reason})
            print(f"[SKIP] {method}: {reason}")
            continue

        method_dir = output_root / _slug(method)
        for result_index, result_sample in enumerate(results, start=1):
            item = _save_result(method_dir, source_sample, result_sample, method, result_index)
            item["config"] = config
            manifest["results"].append(item)
        print(f"[DONE] {method}: {len(results)} result(s) -> {method_dir}")

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nVisualizations saved to: {output_root}")
    print(f"Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
