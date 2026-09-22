"""基于训练集预测挖掘假阳性背景难负样本切片的工具模块。

仅利用训练集自身的假阳性预测切片生成背景样本，防止验证集/测试集数据泄漏。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import cv2

from .labelme import find_image


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    """计算两个 ``xyxy`` 轴对齐边界框之间的交并比（IoU）。"""
    ax0, ay0, ax1, ay1 = map(float, first)
    bx0, by0, bx1, by1 = map(float, second)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _intersects(first: Iterable[float], second: Iterable[float]) -> bool:
    """判断两个边界框是否存在相交区域。"""
    ax0, ay0, ax1, ay1 = map(float, first)
    bx0, by0, bx1, by1 = map(float, second)
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def candidate_is_background(candidate: Iterable[float], ground_truth: list[list[float]]) -> bool:
    """判断候选切片是否为纯背景（与所有真实标注框均无重叠）。"""
    return not any(_intersects(candidate, box) for box in ground_truth)


def _labelme_boxes(path: Path) -> tuple[Path, list[list[float]]]:
    """从 LabelMe 标注中解析所有有效多边形的外接边界框。"""
    with path.open("r", encoding="utf-8") as file:
        item = json.load(file)
    image = find_image(path, item.get("imagePath")).resolve()
    boxes = []
    for shape in item.get("shapes", []):
        points = shape.get("points") or []
        if shape.get("shape_type", "polygon") != "polygon" or len(points) < 3:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return image, boxes


def _crop_window(box, width: int, height: int, margin: float, min_size: int) -> list[int]:
    """计算围绕检测预测框的切片窗口坐标并约束在图像范围内。"""
    x0, y0, x1, y1 = map(float, box)
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    crop_width = max(float(min_size), (x1 - x0) * margin)
    crop_height = max(float(min_size), (y1 - y0) * margin)
    crop_width = min(float(width), crop_width)
    crop_height = min(float(height), crop_height)
    left = int(round(max(0.0, min(width - crop_width, center_x - crop_width / 2.0))))
    top = int(round(max(0.0, min(height - crop_height, center_y - crop_height / 2.0))))
    right = min(width, left + int(round(crop_width)))
    bottom = min(height, top + int(round(crop_height)))
    return [left, top, right, bottom]


def mine_hard_negative_crops(
    train_dir: str | Path,
    prediction_dir: str | Path,
    output_dir: str | Path,
    *,
    class_name: str = "all",
    score_threshold: float = 0.5,
    margin: float = 2.0,
    min_size: int = 64,
    max_per_image: int = 4,
    seed: int = 42,
) -> dict[str, int]:
    """仅基于 ``train_dir`` 的预测结果挖掘背景负样本切片。

    源图像校验严格：每个预测 JSON 必须指向其 LabelMe 标注在 ``train_dir`` 中的图像。
    若包含验证集/测试集的预测，将在写入切片前直接报错，避免数据泄漏。
    """
    train_root = Path(train_dir).expanduser().resolve()
    prediction_root = Path(prediction_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not train_root.is_dir():
        raise ValueError(f"Training directory does not exist: {train_root}")
    if not prediction_root.is_dir():
        raise ValueError(f"Prediction directory does not exist: {prediction_root}")
    if score_threshold < 0 or score_threshold > 1:
        raise ValueError("score_threshold must be between 0 and 1")
    if margin < 1 or min_size <= 0 or max_per_image <= 0:
        raise ValueError("margin must be >= 1, min_size/max_per_image must be positive")

    source_boxes = {}
    for annotation_path in sorted(train_root.glob("*.json")):
        image, boxes = _labelme_boxes(annotation_path)
        source_boxes[image] = boxes
    if not source_boxes:
        raise ValueError(f"No LabelMe annotations found in {train_root}")

    prediction_paths = sorted(prediction_root.rglob("*.json"))
    if not prediction_paths:
        raise ValueError(f"No prediction JSON files found in {prediction_root}")

    candidates_by_source = {}
    foreign_sources = []
    for prediction_path in prediction_paths:
        with prediction_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        source_raw = payload.get("image")
        if not source_raw:
            continue
        source = Path(source_raw).expanduser().resolve()
        if source not in source_boxes:
            foreign_sources.append(str(source))
            continue
        candidates = []
        filter_classes = (
            None
            if class_name in (None, "all", "ALL")
            else ({class_name} if isinstance(class_name, str) else set(class_name))
        )
        for detection in payload.get("detections", []):
            if filter_classes and detection.get("class_name") not in filter_classes:
                continue
            if float(detection.get("score", 0.0)) < score_threshold:
                continue
            box = detection.get("bbox_xyxy")
            if not box or len(box) != 4:
                continue
            candidates.append((float(detection.get("score", 0.0)), list(map(float, box))))
        candidates_by_source.setdefault(source, []).extend(candidates)
    if foreign_sources:
        example = ", ".join(foreign_sources[:3])
        raise ValueError(
            "Prediction directory contains images outside the training set; "
            f"refusing to mine to prevent validation/test leakage: {example}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = 0
    considered = 0
    skipped_overlap = 0
    manifest_path = output_root / "manifest.ndjson"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for source in sorted(candidates_by_source, key=str):
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            ground_truth = source_boxes[source]
            candidates = sorted(candidates_by_source[source], key=lambda item: (-item[0], item[1]))
            selected = []
            for score, box in candidates:
                considered += 1
                crop = _crop_window(box, width, height, margin, min_size)
                if crop[2] <= crop[0] or crop[3] <= crop[1]:
                    continue
                if not candidate_is_background(crop, ground_truth):
                    skipped_overlap += 1
                    continue
                if any(box_iou(crop, old_crop) > 0.5 for old_crop, _ in selected):
                    continue
                selected.append((crop, score))
            rng.shuffle(selected)
            selected = selected[:max_per_image]
            for index, (crop, score) in enumerate(selected):
                left, top, right, bottom = crop
                name = f"{source.stem}__hn_{index:02d}_{class_name}.png"
                image_path = output_root / name
                if not cv2.imwrite(str(image_path), image[top:bottom, left:right]):
                    raise OSError(f"Unable to write hard-negative crop: {image_path}")
                annotation_path = image_path.with_suffix(".json")
                annotation = {
                    "version": "5.0.0",
                    "flags": {},
                    "shapes": [],
                    "imagePath": image_path.name,
                    "imageHeight": bottom - top,
                    "imageWidth": right - left,
                }
                with annotation_path.open("w", encoding="utf-8") as file:
                    json.dump(annotation, file, ensure_ascii=False, indent=2)
                manifest.write(json.dumps({
                    "image": str(image_path),
                    "source": str(source),
                    "class_name": class_name,
                    "score": score,
                    "crop_xyxy": crop,
                }, ensure_ascii=False) + "\n")
                written += 1
    return {"predictions_considered": considered, "overlap_rejected": skipped_overlap, "crops_written": written}
