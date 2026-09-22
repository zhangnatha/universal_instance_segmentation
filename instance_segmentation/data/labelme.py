"""LabelMe 格式标注加载、角度计算、数据集注册与样本隔离验证模块。

提供多边形主轴角度提取、按类别分层拆分、Detectron2 数据集格式转换以及数据集隔离校验等功能。
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from collections.abc import Sequence

import cv2
import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def polygon_angle(points) -> float:
    """返回多边形长轴方向，范围 [0, 180)，图像坐标系中 90° 指向下方。"""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    centered = pts - pts.mean(axis=0)
    covariance = centered.T @ centered
    axis = np.linalg.eigh(covariance)[1][:, -1]
    return float(math.degrees(math.atan2(axis[1], axis[0])) % 180.0)


def angle_to_bin(angle: float, bins: int = 72, period: float = 360.0) -> int:
    """将连续角度值映射到离散的 bin 索引。"""
    return int(round((angle % period) / period * bins)) % bins


def shape_angle(shape: dict, points) -> float:
    """优先读取 LabelMe/X-AnyLabeling 扩展角度，否则由 mask 主轴自动生成。"""
    explicit = explicit_shape_angle(shape)
    if explicit is not None:
        return explicit % 360.0
    return polygon_angle(points)


def explicit_shape_angle(shape: dict) -> float | None:
    """若 shape 中包含有效的人工指定角度，则返回该有限浮点数值。

    将此函数与 :func:`shape_angle` 分离非常关键：后者在仅有掩码训练时会主动
    回退到 PCA 主轴，而 ``annotation`` 标签源必须能够忽略缺失的角度字段。
    """
    attributes = shape.get("attributes") or {}
    for value in (
        shape.get("angle"),
        shape.get("angle_degrees"),
        attributes.get("angle"),
        attributes.get("angle_degrees"),
    ):
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def find_image(json_path: Path, image_path: str | None) -> Path:
    """根据标注文件和记录的图片文件名寻找存在的图像文件。"""
    if image_path:
        candidate = json_path.parent / Path(image_path).name
        if candidate.exists():
            return candidate
    for suffix in IMAGE_SUFFIXES:
        for variant in (suffix, suffix.upper()):
            candidate = json_path.with_suffix(variant)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No image found for annotation: {json_path}")


def _annotation_paths(directory: str | Path | Sequence[str | Path]) -> list[Path]:
    """递归或扁平遍历目录，收集所有的 LabelMe JSON 标注文件路径。"""
    directories = [directory] if isinstance(directory, (str, Path)) else list(directory)
    paths = []
    for root in directories:
        path = Path(root).expanduser()
        if path.is_file():
            if path.suffix.lower() == ".json":
                paths.append(path)
            continue
        paths.extend(path.glob("*.json"))
    return sorted(paths)


def split_labelme_dataset(
    directory: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    classes: Sequence[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    """按标注文件将一个 LabelMe 目录可复现地拆成训练集和验证集。

    返回的列表直接供 ``load_labelme``/``register_labelme`` 使用，因此不会
    复制或修改原始图片和 JSON 文件。图像与标注始终以同一个 JSON 样本为
    单位划分。提供 ``classes`` 时，使用按类别分层的拆分，尽量保证每个
    配置类别同时出现在训练集和验证集中。
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be greater than 0 and less than 1")

    annotation_paths = _annotation_paths(directory)
    if len(annotation_paths) < 2:
        raise ValueError(
            f"At least 2 LabelMe JSON files are required to split {directory}; "
            f"found {len(annotation_paths)}"
        )

    shuffled = list(annotation_paths)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, min(len(shuffled) - 1, math.ceil(len(shuffled) * val_ratio)))

    if classes:
        class_names = tuple(classes)
        class_sets = {
            path: _labelme_class_presence(path, class_names)
            for path in shuffled
        }
        class_counts = {
            name: sum(name in labels for labels in class_sets.values())
            for name in class_names
        }
        impossible = [name for name, count in class_counts.items() if count < 2]
        if impossible:
            raise ValueError(
                "Stratified split requires at least 2 valid labeled samples for "
                "each class in both subsets; insufficient classes: "
                + ", ".join(impossible)
            )

        selected = []
        selected_set = set()
        shuffle_rank = {path: index for index, path in enumerate(shuffled)}
        selected_class_counts = {name: 0 for name in class_names}
        covered = set()
        while len(selected) < val_count and covered != set(class_names):
            candidates = []
            for path in shuffled:
                if path in selected_set:
                    continue
                labels = class_sets[path]
                if any(
                    name in labels and class_counts[name] - selected_class_counts[name] <= 1
                    for name in class_names
                ):
                    continue
                new_labels = labels - covered
                if new_labels:
                    candidates.append((len(new_labels), path))
            if not candidates:
                missing = sorted(set(class_names) - covered)
                raise ValueError(
                    "Unable to create a stratified validation split containing "
                    "all classes; missing classes: " + ", ".join(missing)
                )
            _, selected_path = max(
                candidates,
                key=lambda item: (item[0], -shuffle_rank[item[1]]),
            )
            selected.append(selected_path)
            selected_set.add(selected_path)
            for name in class_sets[selected_path]:
                selected_class_counts[name] += 1
            covered.update(class_sets[selected_path])

        for path in shuffled:
            if len(selected) >= val_count:
                break
            if path in selected_set:
                continue
            labels = class_sets[path]
            if any(
                name in labels and class_counts[name] - selected_class_counts[name] <= 1
                for name in class_names
            ):
                continue
            selected.append(path)
            selected_set.add(path)
            for name in labels:
                selected_class_counts[name] += 1
        if len(selected) < val_count:
            raise ValueError(
                "Unable to create the requested stratified split while keeping "
                "every class in the training subset"
            )
        val_paths = sorted(selected)
        train_paths = sorted(path for path in shuffled if path not in selected_set)
    else:
        val_paths = sorted(shuffled[:val_count])
        train_paths = sorted(shuffled[val_count:])
    return train_paths, val_paths


def _labelme_class_presence(path: Path, classes: Sequence[str]) -> set[str]:
    """返回一个 LabelMe JSON 中有有效多边形的配置类别。"""
    with path.open("r", encoding="utf-8") as file:
        item = json.load(file)
    class_names = set(classes)
    return {
        shape.get("label")
        for shape in item.get("shapes", [])
        if shape.get("label") in class_names
        and shape.get("shape_type", "polygon") == "polygon"
        and len(shape.get("points", [])) >= 3
    }


def write_split_manifest(
    path: str | Path,
    source_directory: str | Path,
    train_paths: Sequence[str | Path],
    val_paths: Sequence[str | Path],
    val_ratio: float,
    seed: int,
) -> Path:
    """保存自动划分结果，便于复现实验和审计数据边界。"""
    manifest_path = Path(path).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_directory": str(Path(source_directory).expanduser()),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "train_annotations": [str(Path(item)) for item in train_paths],
        "val_annotations": [str(Path(item)) for item in val_paths],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def dataset_image_paths(directory: str | Path | Sequence[str | Path]) -> set[Path]:
    """返回 LabelMe 目录实际引用的图片路径，用于检查数据集隔离。"""
    paths = set()
    for json_path in _annotation_paths(directory):
        with json_path.open("r", encoding="utf-8") as file:
            item = json.load(file)
        paths.add(find_image(json_path, item.get("imagePath")))
    return paths


def ensure_disjoint_datasets(
    train_dir: str | Path,
    val_dir: str | Path,
    extra_train_dirs: Sequence[str | Path] = (),
) -> None:
    """禁止训练集和验证集引用同一张图片。"""
    train_paths = dataset_image_paths(train_dir)
    val_paths = dataset_image_paths(val_dir)
    overlap = sorted(train_paths & val_paths)
    if overlap:
        examples = ", ".join(path.name for path in overlap[:5])
        raise ValueError(
            f"train_dir and val_dir share {len(overlap)} image(s): {examples}. "
            "The validation set must be isolated from training."
        )
    for extra_dir in extra_train_dirs:
        extra_paths = dataset_image_paths(extra_dir)
        extra_overlap = sorted(extra_paths & val_paths)
        if extra_overlap:
            examples = ", ".join(path.name for path in extra_overlap[:5])
            raise ValueError(
                f"hard-negative directory and val_dir share {len(extra_overlap)} image(s): "
                f"{examples}. Hard negatives must be generated from training images only."
            )
        train_overlap = sorted(extra_paths & train_paths)
        if train_overlap:
            examples = ", ".join(path.name for path in train_overlap[:5])
            raise ValueError(
                f"hard-negative directory and train_dir share {len(train_overlap)} image(s): "
                f"{examples}. Use newly generated background crops, not source images."
            )


def ensure_hard_negative_disjoint(
    train_dir: str | Path,
    hard_negative_dir: str | Path,
) -> None:
    """检查并拒绝复用源训练图像的难负样本文件。"""
    train_paths = dataset_image_paths(train_dir)
    hard_paths = dataset_image_paths(hard_negative_dir)
    overlap = sorted(train_paths & hard_paths)
    if overlap:
        examples = ", ".join(path.name for path in overlap[:5])
        raise ValueError(
            f"hard-negative directory and train_dir share {len(overlap)} image(s): "
            f"{examples}. Use newly generated background crops, not source images."
        )


def load_labelme(
    directory: str | Path | Sequence[str | Path],
    classes: list[str],
    angle_bins: int = 72,
    angle_period: float = 360.0,
    strict_classes: bool = True,
):
    """将一个 LabelMe 目录读成 Detectron2 dataset dict 列表。"""
    from detectron2.structures import BoxMode

    class_to_id = {name: idx for idx, name in enumerate(classes)}
    records = []
    for image_id, json_path in enumerate(_annotation_paths(directory)):
        with json_path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        image_path = find_image(json_path, item.get("imagePath"))
        height, width = item.get("imageHeight"), item.get("imageWidth")
        if not height or not width:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Unable to read image: {image_path}")
            height, width = image.shape[:2]
        annotations = []
        for shape in item.get("shapes", []):
            label = shape.get("label")
            if label not in class_to_id:
                if strict_classes:
                    raise ValueError(f"{json_path.name}: class {label!r} is not configured")
                continue
            if shape.get("shape_type", "polygon") != "polygon":
                continue
            points = np.asarray(shape.get("points", []), dtype=np.float32)
            if len(points) < 3:
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            explicit_angle = explicit_shape_angle(shape)
            angle = shape_angle(shape, points)
            annotations.append({
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [points.reshape(-1).tolist()],
                "category_id": class_to_id[label],
                "iscrowd": 0,
                # 没有人工角度字段时，shape_angle 会使用多边形长轴自动生成标签。
                "angle_degrees": angle,
                "angle_bin": angle_to_bin(angle, angle_bins, angle_period),
                "angle_explicit": explicit_angle is not None,
            })
        records.append({
            "file_name": str(image_path),
            "image_id": image_id,
            "height": int(height),
            "width": int(width),
            "annotations": annotations,
        })
    if not records:
        raise ValueError(f"No LabelMe JSON files found in {directory}")
    return records


def register_labelme(
    name: str,
    directory: str | Path | Sequence[str | Path],
    classes: list[str],
    angle_bins: int = 72,
    angle_period: float = 360.0,
    strict_classes: bool = True,
):
    """向 Detectron2 数据集注册表中注册 LabelMe 数据集及元数据。"""
    from detectron2.data import DatasetCatalog, MetadataCatalog

    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    DatasetCatalog.register(
        name,
        lambda: load_labelme(
            directory, classes, angle_bins, angle_period, strict_classes
        ),
    )
    MetadataCatalog.get(name).set(thing_classes=classes, evaluator_type="coco")


def validate_labelme(
    directory: str | Path | Sequence[str | Path],
    classes: list[str],
    *,
    strict_classes: bool = True,
) -> dict:
    """校验 LabelMe 数据集并统计图像数、实例数及各类别数量。"""
    records = load_labelme(directory, classes, strict_classes=strict_classes)
    counts = {name: 0 for name in classes}
    empty = 0
    for record in records:
        if not record["annotations"]:
            empty += 1
        for ann in record["annotations"]:
            counts[classes[ann["category_id"]]] += 1
    return {"images": len(records), "instances": sum(counts.values()), "classes": counts, "empty": empty}


def missing_classes(stats: dict, classes: list[str]) -> list[str]:
    """返回没有任何有效多边形实例的配置类别。"""
    counts = stats.get("classes", {})
    return [name for name in classes if int(counts.get(name, 0)) == 0]
