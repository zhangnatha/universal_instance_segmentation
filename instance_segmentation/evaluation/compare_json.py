#!/usr/bin/env python3
"""对比 LabelMe 真实标注 JSON 与模型推理预测 JSON 文件。"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from instance_segmentation.run_logging import setup_run_logging


DEFAULT_IGNORED_LABELS: tuple[str, ...] = ()
PREDICTION_AGGREGATE_FILES = {"predictions.json"}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# 状态颜色遵循参考分析工具定义。为每个发现的类别单独生成确定性的
# HSV 调色板，因此任意标签集均无需硬编码颜色表。
MATCHED_GT_COLOR = (0, 255, 0)       # 绿色：已匹配 GT
MISSED_GT_COLOR = (0, 0, 255)       # 红色：漏检 GT
FALSE_POSITIVE_COLOR = (255, 0, 255)  # 洋红色：误检预测
LOW_CONFIDENCE_COLOR = (0, 215, 255)  # 黄色：与漏检 GT 关联的低置信度过滤预测


def build_class_palette(classes: list[str]) -> dict[str, tuple[int, int, int]]:
    """构建各类别专属显示颜色，同时保留语义化的 GT / 错误标识颜色。"""
    palette: dict[str, tuple[int, int, int]] = {}
    count = max(1, len(classes))
    # 为类别无关语义保留的 HSV 色相范围：
    # 红色=漏检，绿色=已匹配 GT，洋红=假正例。其余色相进行确定性采样，
    # 确保任意类别集合均能获得稳定的专属颜色。
    safe_hues = [*range(21, 45), *range(76, 140)]
    for index, class_name in enumerate(classes):
        position = int(round(index * (len(safe_hues) - 1) / max(1, count - 1)))
        hue = safe_hues[position]
        hsv = np.array([[[hue, 205, 235]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette[class_name] = tuple(int(channel) for channel in bgr)
    return palette


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalized_stem(path: Path, prediction_side: bool) -> str:
    stem = path.stem
    if prediction_side and stem.endswith("_r"):
        return stem[:-2]
    return stem


IGNORED_JSON_NAMES = {
    "predictions.json",
    "_annotations.coco.json",
    "dataset_split.json",
    "coco_instances_results.json",
    "metrics.json",
}


def collect_json_files(directory: Path, prediction_side: bool) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in IGNORED_JSON_NAMES:
            continue
        if prediction_side and path.name in PREDICTION_AGGREGATE_FILES:
            continue
        key = normalized_stem(path, prediction_side)
        previous = files.get(key)
        # 当两种变体同时存在时，优先使用显式的 *_r 预测文件。
        if previous is None or (prediction_side and path.stem.endswith("_r")):
            files[key] = path
    return files


def labels_in_json(path: Path) -> set[str]:
    item = read_json(path)
    labels = set()
    if isinstance(item, list):
        for entry in item:
            name = entry.get("class_name") or entry.get("label") or entry.get("category_name")
            if name not in (None, ""):
                labels.add(str(name))
        return labels

    if "detections" in item:
        for entry in item.get("detections") or []:
            name = entry.get("class_name") or entry.get("label") or entry.get("category_name")
            if name not in (None, ""):
                labels.add(str(name))
    if "predictions" in item:
        for entry in item.get("predictions") or []:
            name = entry.get("class_name") or entry.get("label") or entry.get("category_name")
            if name not in (None, ""):
                labels.add(str(name))
    if "shapes" in item:
        for shape in item.get("shapes") or []:
            name = shape.get("label") or shape.get("class_name")
            if name not in (None, ""):
                labels.add(str(name))
    if "annotations" in item:
        for entry in item.get("annotations") or []:
            name = entry.get("class_name") or entry.get("label") or entry.get("category_name")
            if name not in (None, ""):
                labels.add(str(name))
    return labels


class ProgressReporter:
    """渲染无第三方依赖的命令行进度条，且不污染输出报告。"""

    def __init__(self, total: int, label: str) -> None:
        self.total = max(0, total)
        self.label = label
        self.started_at = time.monotonic()
        self.last_rendered = -1
        self.last_render_at = 0.0
        self.minimum_step = max(1, self.total // 100)
        self.terminal = getattr(sys.stdout, "terminal", sys.stdout)
        try:
            self.is_tty = bool(self.terminal.isatty())
        except (AttributeError, OSError):
            self.is_tty = False
        self.update(0, force=True)

    @staticmethod
    def format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def update(self, current: int, force: bool = False) -> None:
        current = min(max(0, current), self.total)
        now = time.monotonic()
        if (
            not force
            and current != self.total
            and current - self.last_rendered < self.minimum_step
            and now - self.last_render_at < 0.1
        ):
            return
        self.last_rendered = current
        self.last_render_at = now
        elapsed = now - self.started_at
        ratio = current / self.total if self.total else 1.0
        filled = int(round(28 * ratio))
        bar = "#" * filled + "-" * (28 - filled)
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - current) / rate if rate > 0 else 0.0
        message = (
            f"{self.label}: |{bar}| {current}/{self.total} "
            f"{ratio * 100:6.2f}% elapsed={self.format_duration(elapsed)} "
            f"ETA={self.format_duration(eta) if current < self.total else '00:00'}"
        )
        if self.is_tty:
            self.terminal.write("\r[COMPARE_JSON] " + message)
            if current >= self.total:
                self.terminal.write("\n")
            self.terminal.flush()
        else:
            print(message, flush=True)

    def finish(self) -> None:
        if self.last_rendered != self.total:
            self.update(self.total, force=True)


def discover_classes(
    gt_files: dict[str, Path],
    prediction_files: dict[str, Path],
    progress_callback: Callable[[int], None] | None = None,
) -> list[str]:
    labels: set[str] = set()
    paths = [*gt_files.values(), *prediction_files.values()]
    for index, path in enumerate(paths, 1):
        labels.update(labels_in_json(path))
        if progress_callback is not None:
            progress_callback(index)
    return sorted(labels)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return float(intersection / union) if union else 0.0


def polygon_mask(points: list[Any], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(array) >= 3:
        polygon = np.rint(array).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def gt_mask(shape: dict[str, Any], height: int, width: int) -> np.ndarray:
    shape_type = shape.get("shape_type", "polygon")
    points = shape.get("points") or []
    if shape_type == "rectangle" and len(points) >= 2:
        mask = np.zeros((height, width), dtype=np.uint8)
        first, second = np.rint(np.asarray(points[:2], dtype=np.float32)).astype(np.int32)
        cv2.rectangle(mask, tuple(first), tuple(second), 1, cv2.FILLED)
        return mask.astype(bool)
    return polygon_mask(points, height, width)


def decode_rle(rle: dict[str, Any] | list[Any], height: int, width: int) -> np.ndarray | None:
    # C++ 后端输出的是简单的行优先 RLE：数值交替表示背景/前景游程，
    # 且默认以背景开始。这并非 COCO 压缩 RLE，因此在尝试 pycocotools 之前先进行解码。
    if isinstance(rle, list) and all(isinstance(value, int) and not isinstance(value, bool) for value in rle):
        expected = height * width
        if height <= 0 or width <= 0:
            print(
                f"[COMPARE_JSON] Skipping invalid row-major mask_rle: invalid dimensions {height}x{width}",
                file=sys.stderr,
            )
            return None
        if any(value < 0 for value in rle):
            print(
                "[COMPARE_JSON] Skipping invalid row-major mask_rle: counts must be non-negative",
                file=sys.stderr,
            )
            return None
        total = sum(rle)
        if total != expected:
            print(
                f"[COMPARE_JSON] Skipping invalid row-major mask_rle: run total {total} != image size {expected}",
                file=sys.stderr,
            )
            return None
        flat = np.zeros(expected, dtype=bool)
        offset = 0
        foreground = False
        for count in rle:
            if foreground and count:
                flat[offset : offset + count] = True
            offset += count
            foreground = not foreground
        return flat.reshape((height, width))

    try:
        from pycocotools import mask as mask_utils
        if isinstance(rle, dict):
            rle_dict = dict(rle)
            if isinstance(rle_dict.get("counts"), str):
                rle_dict["counts"] = rle_dict["counts"].encode("utf-8")
            decoded = mask_utils.decode(rle_dict)
            if decoded is not None:
                if decoded.shape == (height, width):
                    return decoded.astype(bool)
                elif decoded.shape == (width, height):
                    return decoded.T.astype(bool)
        elif isinstance(rle, list):
            rles = mask_utils.frPyObjects(rle, height, width)
            decoded = mask_utils.decode(rles)
            if decoded is not None:
                if len(decoded.shape) == 3:
                    decoded = np.any(decoded, axis=2)
                return decoded.astype(bool)
    except Exception:
        pass
    return None


def prediction_mask(detection: dict[str, Any], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    # 1. 直接掩码数组 (Direct mask array)
    if isinstance(detection.get("mask"), np.ndarray):
        m = detection["mask"]
        if m.shape == (height, width):
            return m.astype(bool)

    # 2. RLE 掩码 (RLE mask)
    rle = detection.get("mask_rle") or (
        detection.get("segmentation")
        if isinstance(detection.get("segmentation"), dict)
        else None
    )
    if isinstance(rle, (dict, list)):
        decoded = decode_rle(rle, height, width)
        if decoded is not None and decoded.any():
            return decoded

    # 3. contours_xy 轮廓点格式
    for contour in detection.get("contours_xy") or []:
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    if mask.any():
        return mask.astype(bool)

    # 4. mask_polygons 或 COCO 多边形分割列表格式
    polygons = detection.get("mask_polygons") or (
        detection.get("segmentation")
        if isinstance(detection.get("segmentation"), list)
        else None
    )
    if polygons:
        for poly in polygons:
            points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3:
                cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        if mask.any():
            return mask.astype(bool)

    # 5. LabelMe / Solomon shapes 标注图形格式
    points = detection.get("points") or []
    shape_type = detection.get("shape_type", "polygon")
    if shape_type == "rectangle" and len(points) >= 2:
        first, second = np.rint(np.asarray(points[:2], dtype=np.float32)).astype(np.int32)
        cv2.rectangle(mask, tuple(first), tuple(second), 1, cv2.FILLED)
    elif shape_type == "polygon":
        polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def extract_score(detection: dict[str, Any]) -> float | None:
    for key in ("score", "confidence", "conf", "prob", "probability", "score_val"):
        if detection.get(key) is not None:
            try:
                return float(detection[key])
            except (TypeError, ValueError):
                pass
    if isinstance(detection.get("flags"), dict):
        for key in ("score", "confidence", "conf"):
            if detection["flags"].get(key) is not None:
                try:
                    return float(detection["flags"][key])
                except (TypeError, ValueError):
                    pass
    if isinstance(detection.get("attributes"), dict):
        for key in ("score", "confidence", "conf"):
            if detection["attributes"].get(key) is not None:
                try:
                    return float(detection["attributes"][key])
                except (TypeError, ValueError):
                    pass
    return None


def prediction_entries(item: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """归一化仓库、Solomon 以及单图像预测 JSON 记录。"""
    if isinstance(item, list):
        return item
    if "detections" in item:
        return item.get("detections") or []
    if "predictions" in item:
        return item.get("predictions") or []
    if "annotations" in item:
        return item.get("annotations") or []

    return [
        {
            "class_name": shape.get("label") or shape.get("class_name"),
            "score": extract_score(shape),
            "shape_type": shape.get("shape_type", "polygon"),
            "points": shape.get("points") or [],
        }
        for shape in item.get("shapes") or []
    ]


def find_image(json_path: Path, item: dict[str, Any]) -> Path | None:
    image_path = item.get("imagePath") or item.get("image") or item.get("file_name")
    if image_path:
        candidate = json_path.parent / Path(image_path).name
        if candidate.exists():
            return candidate
    for suffix in IMAGE_SUFFIXES:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    # Windows 导出文件通常保留大写图像后缀（例如 ``.BMP``），
    # 而 JSON 文件主名保持不变。
    try:
        for candidate in json_path.parent.iterdir():
            if (
                candidate.is_file()
                and candidate.suffix.lower() in IMAGE_SUFFIXES
                and candidate.stem.lower() == json_path.stem.lower()
            ):
                return candidate
    except OSError:
        pass
    return None


def extract_dimensions(item: dict[str, Any]) -> tuple[int, int]:
    height = int(item.get("imageHeight") or item.get("height") or 0)
    width = int(item.get("imageWidth") or item.get("width") or 0)
    if (height <= 0 or width <= 0) and isinstance(item.get("resolution"), dict):
        res = item["resolution"]
        height = int(res.get("height") or 0)
        width = int(res.get("width") or 0)
    return height, width


def load_ground_truth(path: Path, classes: list[str]) -> tuple[Path | None, list[dict[str, Any]], tuple[int, int]]:
    item = read_json(path)
    image_path = find_image(path, item)
    height, width = extract_dimensions(item)
    if image_path is not None and (height <= 0 or width <= 0):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Missing image dimensions in {path}")

    instances = []
    shapes = item.get("shapes") or item.get("annotations") or []
    for index, shape in enumerate(shapes):
        class_name = shape.get("label") or shape.get("class_name")
        if class_name not in classes:
            continue
        mask = gt_mask(shape, height, width)
        if not mask.any():
            continue
        instances.append({
            "index": index,
            "class_name": class_name,
            "mask": mask,
            "bbox": bbox_from_mask(mask),
            "area": int(mask.sum()),
            "points": shape.get("points") or [],
        })
    return image_path, instances, (height, width)


def load_prediction_metadata(path: Path) -> tuple[Path | None, tuple[int, int]]:
    item = read_json(path)
    image_path = find_image(path, item)
    height, width = extract_dimensions(item)
    if image_path is not None and (height <= 0 or width <= 0):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Missing image dimensions in {path}")
    return image_path, (height, width)


def load_predictions(
    path: Path,
    classes: list[str],
    height: int,
    width: int,
    score_threshold: float | None = 0.0,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    item = read_json(path)
    instances = []
    for index, detection in enumerate(prediction_entries(item)):
        class_name = detection.get("class_name") or detection.get("label") or detection.get("category_name")
        if class_name not in classes:
            continue
        score = extract_score(detection)
        if score_threshold is not None and score is not None and score < score_threshold:
            continue
        mask = prediction_mask(detection, height, width)
        if not mask.any():
            continue

        contours = []
        if detection.get("contours_xy"):
            contours = detection.get("contours_xy")
        elif detection.get("mask_polygons"):
            contours = detection.get("mask_polygons")
        elif detection.get("points"):
            contours = [detection.get("points")]
        elif isinstance(detection.get("segmentation"), list):
            contours = detection.get("segmentation")
        else:
            cv_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c.reshape(-1, 2).tolist() for c in cv_contours]

        instances.append({
            "index": index,
            "class_name": class_name,
            "score": score,
            "mask": mask,
            "bbox": bbox_from_mask(mask),
            "area": int(mask.sum()),
            "contours": contours,
        })
    return instances


def match_class_instances(
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    # 按预测置信度降序排序，然后将每个预测分配给同一类别下最佳且未使用的 GT。
    # 这与参考分析工具采用的一对一匹配策略完全一致。
    prediction_order = sorted(
        range(len(pred_instances)),
        key=lambda index: (
            -float(pred_instances[index]["score"])
            if pred_instances[index].get("score") is not None
            else 0.0,
            index,
        ),
    )
    used_gt, used_pred = set(), set()
    matches = []
    for pred_index in prediction_order:
        pred = pred_instances[pred_index]
        best_gt_index = None
        best_iou = 0.0
        for gt_index, gt in enumerate(gt_instances):
            if gt_index in used_gt:
                continue
            iou = mask_iou(pred["mask"], gt["mask"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is None or best_iou < iou_threshold:
            continue
        used_gt.add(best_gt_index)
        used_pred.add(pred_index)
        matches.append((best_gt_index, pred_index, float(best_iou)))

    # 在固定的 IoU 阈值下，每个未匹配的 GT 记为 FN，每个未匹配的预测记为 FP。
    # 特别地，低重叠度的同类别对贡献一个 FN 和一个 FP；
    # 这是标准的阈值化检测定义，而非第二个匹配类别。
    unmatched_gt = [
        gt_index for gt_index in range(len(gt_instances))
        if gt_index not in used_gt
    ]

    unmatched_pred = [index for index in range(len(pred_instances)) if index not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def match_low_confidence_instances(
    gt_instances: list[dict[str, Any]],
    low_confidence_predictions: list[dict[str, Any]],
    unmatched_gt: list[int],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int]]:
    """将过滤掉的低分预测与仍处于漏检状态的 GT 建立关联。

    低分预测不计入 TP/FP 统计，因为分数阈值已将其从评估预测集中滤除。
    保留它们仅用于在掩码达到 IoU 阈值时，将漏检的 GT 解释为 ``FN_LOW_SCORE``。
    匹配为一对一方式，且仅使用尚未被有效预测匹配的 GT；
    这在相邻 GT 轮廓发生轻微重叠时尤为重要。
    """
    available_gt = set(unmatched_gt)
    prediction_order = sorted(
        range(len(low_confidence_predictions)),
        key=lambda index: (
            -float(low_confidence_predictions[index]["score"])
            if low_confidence_predictions[index].get("score") is not None
            else 0.0,
            index,
        ),
    )
    matches: list[tuple[int, int, float]] = []
    for pred_index in prediction_order:
        pred = low_confidence_predictions[pred_index]
        best_gt_index = None
        best_iou = 0.0
        for gt_index in available_gt:
            iou = mask_iou(pred["mask"], gt_instances[gt_index]["mask"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is None or best_iou < iou_threshold:
            continue
        available_gt.remove(best_gt_index)
        matches.append((best_gt_index, pred_index, float(best_iou)))

    return matches, [gt_index for gt_index in unmatched_gt if gt_index in available_gt]


def best_other_class_iou(
    instance: dict[str, Any],
    candidates: list[dict[str, Any]],
    own_class: str,
) -> tuple[float, str | None, int | None]:
    best = (0.0, None, None)
    for index, candidate in enumerate(candidates):
        if candidate["class_name"] == own_class:
            continue
        score = mask_iou(instance["mask"], candidate["mask"])
        if score > best[0]:
            best = (score, candidate["class_name"], index)
    return best


def init_confusion_matrix(classes: list[str]) -> dict[str, dict[str, int]]:
    """创建带有显式背景标签的 YOLO 风格类别混淆矩阵。

    导出的 CSV 遵循行表示真实标注 ``GT``、列表示预测 ``PRED`` 的易读约定。
    Ultralytics 内部存储相同信息的转置格式（行为预测，列为真实），
    但两种约定均使用额外的背景行/列进行 FP 和 FN 统计。
    """
    labels = [*classes, "background"]
    return {
        gt_class: {pred_class: 0 for pred_class in labels}
        for gt_class in labels
    }


def build_confusion_matrix(
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    classes: list[str],
    iou_threshold: float,
) -> dict[str, dict[str, int]]:
    """构建单张图像在指定阈值下的实例分割混淆矩阵。

    匹配采用类别不可知且一对一机制，与 YOLO 评估保持一致：

    * 满足 IoU 阈值的匹配对累加至 ``GT 类别 -> 预测类别``。
      对角线为 TP，非对角线单元格为分类错误。
    * 未匹配的有效预测累加至 ``background -> 预测类别`` (FP)，包括重复预测。
    * 未匹配的 GT 累加至 ``GT 类别 -> background`` (FN)。

    现有评测策略将具有正 IoU 但未达到阈值的同类有效预测仅视为定位 FP，
    而非同时产生第二个 FN。因此此类预测会抑制对应背景列的累加，
    而已被其他 GT 消耗的预测无法抑制第二个 GT。低置信度预测在此函数被调用前
    已被调用方排除。
    """
    matrix = init_confusion_matrix(classes)
    background = "background"
    ordered_predictions = sorted(
        enumerate(pred_instances),
        key=lambda item: (
            -float(item[1]["score"])
            if item[1].get("score") is not None
            else 0.0,
            item[0],
        ),
    )

    # 优先选择全局 IoU 最大的匹配对。这与 YOLO 使用的去重策略一致，
    # 并保证一个 GT/预测最多贡献到一个矩阵单元格中。
    candidate_pairs = []
    for pred_index, prediction in ordered_predictions:
        for gt_index, ground_truth in enumerate(gt_instances):
            iou = mask_iou(prediction["mask"], ground_truth["mask"])
            if iou >= iou_threshold:
                candidate_pairs.append((float(iou), pred_index, gt_index))
    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    for iou, pred_index, gt_index in candidate_pairs:
        if pred_index in used_predictions or gt_index in used_gt:
            continue
        prediction = pred_instances[pred_index]
        ground_truth = gt_instances[gt_index]
        matrix[ground_truth["class_name"]][prediction["class_name"]] += 1
        used_predictions.add(pred_index)
        used_gt.add(gt_index)

    unmatched_predictions = [
        pred_index
        for pred_index, _ in ordered_predictions
        if pred_index not in used_predictions
    ]
    for pred_index in unmatched_predictions:
        matrix[background][pred_instances[pred_index]["class_name"]] += 1

    # 单个未匹配的同类预测最多解释一个未匹配的 GT。
    # 这可处理相邻 GT 轮廓之间的轻微重叠，而不会掩盖新增 GT 的真实漏检。
    suppressible_gt: set[int] = set()
    for pred_index in unmatched_predictions:
        prediction = pred_instances[pred_index]
        best_gt_index = None
        best_iou = 0.0
        for gt_index, ground_truth in enumerate(gt_instances):
            if gt_index in used_gt or gt_index in suppressible_gt:
                continue
            if ground_truth["class_name"] != prediction["class_name"]:
                continue
            iou = mask_iou(prediction["mask"], ground_truth["mask"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou > 0.0:
            suppressible_gt.add(best_gt_index)

    for gt_index, ground_truth in enumerate(gt_instances):
        if gt_index not in used_gt and gt_index not in suppressible_gt:
            matrix[ground_truth["class_name"]][background] += 1
    return matrix


def init_stats(classes: list[str]) -> dict[str, Any]:
    return {
        "images": 0,
        "gt_count": 0,
        "pred_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "cross_class_fp": 0,
        "cross_class_fn": 0,
        "background_fp": 0,
        "duplicate_fp": 0,
        "plain_fn": 0,
        "low_confidence_fn": 0,
        "iou_sum": 0.0,
        "iou_count": 0,
        "classes": {
            class_name: {
                "gt_count": 0,
                "pred_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "cross_class_fp": 0,
                "cross_class_fn": 0,
                "background_fp": 0,
                "duplicate_fp": 0,
                "plain_fn": 0,
                "low_confidence_fn": 0,
                "iou_sum": 0.0,
                "iou_count": 0,
            }
            for class_name in classes
        },
    }


def update_stats(stats: dict[str, Any], class_name: str, field: str, value: int | float = 1) -> None:
    stats[field] += value
    stats["classes"][class_name][field] += value


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    result["classes"] = {}
    for class_name, values in stats["classes"].items():
        item = dict(values)
        item["precision"] = item["tp"] / (item["tp"] + item["fp"]) if item["tp"] + item["fp"] else 0.0
        item["recall"] = item["tp"] / (item["tp"] + item["fn"]) if item["tp"] + item["fn"] else 0.0
        item["f1"] = (
            2 * item["precision"] * item["recall"] / (item["precision"] + item["recall"])
            if item["precision"] + item["recall"] else 0.0
        )
        item["mean_iou_tp"] = item["iou_sum"] / item["iou_count"] if item["iou_count"] else 0.0
        result["classes"][class_name] = item
    result["precision"] = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
    result["recall"] = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
    result["f1"] = (
        2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
        if result["precision"] + result["recall"] else 0.0
    )
    result["mean_iou_tp"] = stats["iou_sum"] / stats["iou_count"] if stats["iou_count"] else 0.0
    return result


def update_instance_stats(
    stats: dict[str, Any],
    gt_by_class: dict[str, list[dict[str, Any]]],
    pred_by_class: dict[str, list[dict[str, Any]]],
    matches_by_class: dict[str, list[tuple[int, int, float]]],
    unmatched_gt_by_class: dict[str, list[int]],
    unmatched_pred_by_class: dict[str, list[int]],
    iou_threshold: float,
    cross_class_threshold: float,
    *,
    low_confidence_matches_by_class: dict[str, list[tuple[int, int, float]]] | None = None,
    low_confidence_pred_by_class: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    low_confidence_matches_by_class = low_confidence_matches_by_class or {
        class_name: [] for class_name in gt_by_class
    }
    low_confidence_pred_by_class = low_confidence_pred_by_class or {
        class_name: [] for class_name in gt_by_class
    }
    rows = []
    all_gt = [instance for values in gt_by_class.values() for instance in values]
    unmatched_pred_instances = [
        pred_by_class[class_name][pred_index]
        for class_name, pred_indices in unmatched_pred_by_class.items()
        for pred_index in pred_indices
    ]
    for class_name in gt_by_class:
        update_stats(stats, class_name, "gt_count", len(gt_by_class[class_name]))
        update_stats(stats, class_name, "pred_count", len(pred_by_class[class_name]))
        for gt_index, pred_index, iou in matches_by_class[class_name]:
            update_stats(stats, class_name, "tp")
            update_stats(stats, class_name, "iou_sum", iou)
            update_stats(stats, class_name, "iou_count")
            rows.append({
                "class_name": class_name,
                "status": "TP",
                "gt_index": gt_by_class[class_name][gt_index]["index"],
                "pred_index": pred_by_class[class_name][pred_index]["index"],
                "iou": iou,
                "score": pred_by_class[class_name][pred_index].get("score"),
            })
        low_confidence_pred_indices = {
            pred["index"]
            for _, pred_index, _ in low_confidence_matches_by_class[class_name]
            for pred in [low_confidence_pred_by_class[class_name][pred_index]]
        }
        for gt_index, pred_index, iou in low_confidence_matches_by_class[class_name]:
            pred = low_confidence_pred_by_class[class_name][pred_index]
            update_stats(stats, class_name, "fn")
            update_stats(stats, class_name, "low_confidence_fn")
            rows.append({
                "class_name": class_name,
                "status": "FN_LOW_SCORE",
                "gt_index": gt_by_class[class_name][gt_index]["index"],
                "pred_index": pred["index"],
                "iou": iou,
                "score": pred.get("score"),
                "other_class": None,
            })
        for gt_index in unmatched_gt_by_class[class_name]:
            gt = gt_by_class[class_name][gt_index]
            best_iou, best_class, best_pred_index = best_other_class_iou(
                gt, unmatched_pred_instances, class_name
            )
            cross_class = best_iou >= cross_class_threshold
            available_active_same_predictions = [
                pred_by_class[class_name][pred_index]
                for pred_index in unmatched_pred_by_class[class_name]
            ]
            available_low_same_predictions = [
                pred
                for pred in low_confidence_pred_by_class[class_name]
                if pred["index"] not in low_confidence_pred_indices
            ]
            best_active_same_iou = max(
                (
                    mask_iou(pred["mask"], gt["mask"])
                    for pred in available_active_same_predictions
                ),
                default=0.0,
            )
            best_low_same_iou = 0.0
            best_low_same_prediction = None
            for prediction in available_low_same_predictions:
                low_iou = mask_iou(prediction["mask"], gt["mask"])
                if low_iou > best_low_same_iou:
                    best_low_same_iou = low_iou
                    best_low_same_prediction = prediction
            if cross_class:
                update_stats(stats, class_name, "fn")
                status = "FN_WRONG_CLASS"
                update_stats(stats, class_name, "cross_class_fn")
                row_iou = best_iou
                row_score = unmatched_pred_instances[best_pred_index].get("score")
            elif best_active_same_iou > 0.0:
                # 同类别的有效预测已在下方记录为 FP。
                # 这属于定位不准，而非对同一 GT 产生第二个 FN。
                # 此处特意排除了已被其他 GT 消耗的预测，
                # 因此轻微的 GT 轮廓重叠不会掩盖真正漏检的目标。
                continue
            elif best_low_same_iou > 0.0:
                update_stats(stats, class_name, "fn")
                status = "FN_LOW_SCORE"
                update_stats(stats, class_name, "low_confidence_fn")
                row_iou = best_low_same_iou
                row_score = best_low_same_prediction.get("score")
            else:
                update_stats(stats, class_name, "fn")
                status = "FN"
                update_stats(stats, class_name, "plain_fn")
                row_iou = 0.0
                row_score = None
            rows.append({
                "class_name": class_name,
                "status": status,
                "gt_index": gt["index"],
                "pred_index": (
                    unmatched_pred_instances[best_pred_index]["index"]
                    if cross_class
                    else best_low_same_prediction["index"]
                    if status == "FN_LOW_SCORE" and best_low_same_prediction is not None
                    else None
                ),
                "iou": row_iou,
                "score": row_score,
                "other_class": best_class if cross_class else None,
            })
        for pred_index in unmatched_pred_by_class[class_name]:
            pred = pred_by_class[class_name][pred_index]
            best_same_iou = 0.0
            best_same_gt_index = None
            for gt_index, gt in enumerate(gt_by_class[class_name]):
                same_iou = mask_iou(pred["mask"], gt["mask"])
                if same_iou > best_same_iou:
                    best_same_iou = same_iou
                    best_same_gt_index = gt_index
            best_other_iou, best_class, best_gt_index = best_other_class_iou(pred, all_gt, class_name)
            if best_same_iou >= iou_threshold:
                status = "FP_DUPLICATE"
                update_stats(stats, class_name, "duplicate_fp")
                row_gt_index = (
                    gt_by_class[class_name][best_same_gt_index]["index"]
                    if best_same_gt_index is not None else None
                )
                row_other_class = None
                best_iou = best_same_iou
            elif best_other_iou >= cross_class_threshold and best_other_iou > best_same_iou:
                status = "FP_WRONG_CLASS"
                update_stats(stats, class_name, "cross_class_fp")
                row_gt_index = all_gt[best_gt_index]["index"] if best_gt_index is not None else None
                row_other_class = best_class
                best_iou = best_other_iou
            else:
                status = "FP"
                update_stats(stats, class_name, "background_fp")
                row_gt_index = None
                row_other_class = None
                best_iou = max(best_same_iou, best_other_iou)
            update_stats(stats, class_name, "fp")
            rows.append({
                "class_name": class_name,
                "status": status,
                "gt_index": row_gt_index,
                "pred_index": pred["index"],
                "iou": best_iou,
                "score": pred.get("score"),
                "other_class": row_other_class,
            })
    return rows


def draw_contour(
    image: np.ndarray,
    points: list[Any],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if not points:
        return
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(array) < 2:
        return
    cv2.polylines(
        image,
        [np.rint(array).astype(np.int32)],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_prediction_contour(
    image: np.ndarray,
    points: list[Any],
    class_color: tuple[int, int, int],
    error_color: tuple[int, int, int] | None,
    thickness: int,
) -> None:
    """预测框采用类别颜色绘制，若存在误差则叠加大对比度的中心线。"""
    if error_color is None:
        draw_contour(image, points, class_color, thickness)
        return
    draw_contour(image, points, class_color, thickness + 2)
    draw_contour(image, points, error_color, thickness)


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    border_color: tuple[int, int, int] | None = None,
    underline: bool = False,
) -> None:
    x, y = max(0, int(origin[0])), max(15, int(origin[1]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.42, 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = min(y, image.shape[0] - 2)
    cv2.rectangle(image, (x, y - height - baseline - 3), (min(image.shape[1] - 1, x + width + 4), y + 1), color, cv2.FILLED)
    if border_color is not None:
        cv2.rectangle(
            image,
            (x, y - height - baseline - 3),
            (min(image.shape[1] - 1, x + width + 4), y + 1),
            border_color,
            1,
        )
    cv2.putText(image, text, (x + 2, y - baseline - 1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    if underline:
        cv2.line(
            image,
            (x + 2, y - baseline + 1),
            (min(image.shape[1] - 1, x + 2 + width), y - baseline + 1),
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def draw_sidebar(
    height: int,
    lines: list[
        tuple[
            str,
            tuple[int, int, int],
            tuple[int, int, int] | None,
        ]
        | tuple[
            str,
            tuple[int, int, int],
            tuple[int, int, int] | None,
            bool,
        ]
    ],
    overlap: bool,
    thresholds: dict[str, float] | None = None,
    min_width: int = 300,
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    font_thickness = 1

    max_text_w = 0
    for item in lines:
        (tw, _), _ = cv2.getTextSize(item[0], font, font_scale, font_thickness)
        if tw > max_text_w:
            max_text_w = tw

    # 阈值字符串宽度
    if thresholds:
        iou_t = thresholds.get("iou", 0.50)
        score_t = thresholds.get("score", 0.00)
        thresh_text = f"IoU: {iou_t:.2f}  Score: {score_t:.2f}"
    else:
        thresh_text = "IoU: 0.50  Score: 0.00"
    (thresh_w, _), _ = cv2.getTextSize(thresh_text, font, 0.36, 1)

    needed_width = max(max_text_w + 64, thresh_w + 24)
    sidebar_width = max(min_width, needed_width)

    sidebar = np.full((height, sidebar_width, 3), (35, 30, 26), dtype=np.uint8)

    # 标题栏区域
    cv2.putText(sidebar, "INSTANCE ANALYSIS", (12, 18), font, 0.42, (220, 235, 255), 1, cv2.LINE_AA)
    overlap_str = "YES" if overlap else "NO"
    overlap_text = f"Overlap: {overlap_str}"
    overlap_color = (100, 160, 255) if overlap else (160, 160, 160)
    (otw, _), _ = cv2.getTextSize(overlap_text, font, 0.38, 1)
    cv2.putText(sidebar, overlap_text, (max(12, sidebar_width - otw - 12), 18), font, 0.38, overlap_color, 1, cv2.LINE_AA)

    # 阈值显示行
    cv2.putText(sidebar, thresh_text, (12, 35), font, 0.36, (140, 185, 230), 1, cv2.LINE_AA)

    # 标题栏下方的水平分割线
    cv2.line(sidebar, (10, 44), (sidebar_width - 10, 44), (65, 60, 55), 1, cv2.LINE_AA)

    # 渲染检测明细详情
    margin_top = 64
    line_height = 20
    swatch = 10
    class_swatch = 6
    max_rows = max(1, (height - margin_top - 10) // line_height)
    visible = lines[:max_rows]

    for index, item in enumerate(visible):
        text = item[0]
        status_color = item[1]
        class_color = item[2]
        item_underline = item[3] if len(item) > 3 else False

        center_y = margin_top + index * line_height
        cv2.rectangle(
            sidebar,
            (12, center_y - swatch // 2),
            (12 + swatch, center_y + swatch // 2),
            status_color,
            cv2.FILLED,
        )
        text_x = 12 + swatch + 8
        if class_color is not None:
            cv2.rectangle(
                sidebar,
                (text_x, center_y - class_swatch // 2),
                (text_x + class_swatch, center_y + class_swatch // 2),
                class_color,
                cv2.FILLED,
            )
            text_x += class_swatch + 6

        cv2.putText(
            sidebar,
            text,
            (text_x, center_y + 4),
            font,
            font_scale,
            (240, 240, 240),
            font_thickness,
            cv2.LINE_AA,
        )
        if item_underline:
            (tw, _), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.line(
                sidebar,
                (text_x, center_y + 6),
                (min(sidebar_width - 6, text_x + tw), center_y + 6),
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

    # 分隔左侧边栏与主图像的垂直分割线
    cv2.line(sidebar, (sidebar_width - 1, 0), (sidebar_width - 1, height), (75, 70, 65), 2, cv2.LINE_AA)
    return sidebar


def draw_overlay(
    image: np.ndarray,
    gt_by_class: dict[str, list[dict[str, Any]]],
    pred_by_class: dict[str, list[dict[str, Any]]],
    match_rows: list[dict[str, Any]],
    overlap: bool,
    class_colors: dict[str, tuple[int, int, int]],
    thresholds: dict[str, float] | None = None,
    show_overlay_text: bool = True,
) -> np.ndarray:
    output = image.copy()

    gt_rows = {
        (row["class_name"], row["gt_index"]): row
        for row in match_rows
        if row.get("gt_index") is not None
    }
    pred_rows = {
        (row["class_name"], row["pred_index"]): row
        for row in match_rows
        if row.get("pred_index") is not None
    }
    gt_instances = sorted(
        [instance for instances in gt_by_class.values() for instance in instances],
        key=lambda instance: instance["index"],
    )
    pred_instances = sorted(
        [instance for instances in pred_by_class.values() for instance in instances],
        key=lambda instance: instance["index"],
    )
    gt_tags = {
        (instance["class_name"], instance["index"]): f"G{index + 1}"
        for index, instance in enumerate(gt_instances)
    }
    pred_tags = {
        (instance["class_name"], instance["index"]): f"P{index + 1}"
        for index, instance in enumerate(pred_instances)
    }

    legend: list[tuple[str, tuple[int, int, int], tuple[int, int, int] | None, bool]] = []
    for instance in gt_instances:
        key = (instance["class_name"], instance["index"])
        row = gt_rows.get(key, {})
        status = row.get("status", "")
        is_fn = status.startswith("FN")
        label_color = MISSED_GT_COLOR if is_fn else MATCHED_GT_COLOR
        draw_contour(
            output,
            instance.get("points") or [],
            label_color,
            3,
        )
        draw_label(output, gt_tags[key], instance["bbox"][:2], label_color, MATCHED_GT_COLOR, underline=is_fn)
        gt_desc = (
            f"{gt_tags[key]} {instance['class_name']} {status}"
            if is_fn
            else f"{gt_tags[key]} {instance['class_name']}"
        )
        legend.append((
            gt_desc,
            label_color,
            None,
            is_fn,
        ))
    for instance in pred_instances:
        key = (instance["class_name"], instance["index"])
        row = pred_rows.get(key, {})
        status = row.get("status", "")
        matched = row.get("status") == "TP"
        low_confidence = status == "FN_LOW_SCORE"
        class_color = class_colors[instance["class_name"]]
        error_color = (
            None
            if matched
            else LOW_CONFIDENCE_COLOR
            if low_confidence
            else FALSE_POSITIVE_COLOR
        )
        for contour in instance.get("contours") or []:
            draw_prediction_contour(output, contour, class_color, error_color, 2)
        x0, y0, _, _ = instance["bbox"]
        label_color = class_color if matched else LOW_CONFIDENCE_COLOR if low_confidence else FALSE_POSITIVE_COLOR
        label_border = None if matched else class_color
        draw_label(output, pred_tags[key], (x0, y0), label_color, label_border, underline=not matched)
        score_val = float(instance.get("score") if instance.get("score") is not None else 0.0)
        iou_val = float(row.get("iou", 0.0) or 0.0)
        if matched:
            status = f"TP IoU={iou_val:.3f}"
        elif row.get("status") == "FN_LOW_SCORE":
            status = f"FN_LOW_SCORE IoU={iou_val:.3f}"
        elif row.get("status") == "FP_DUPLICATE":
            status = f"FP_DUPLICATE IoU={iou_val:.3f}"
        elif row.get("status") == "FP_WRONG_CLASS":
            status = f"FP_WRONG_CLASS IoU={iou_val:.3f}"
        else:
            status = f"FP IoU={iou_val:.3f}"
        legend.append((
            f"{pred_tags[key]} {instance['class_name']} {score_val:.3f} {status}",
            label_color,
            class_color,
            not matched,
        ))

    if show_overlay_text:
        sidebar = draw_sidebar(output.shape[0], legend, overlap, thresholds=thresholds)
        return np.hstack([sidebar, output])
    return output


def image_has_cross_class_overlap(
    gt_by_class: dict[str, list[dict[str, Any]]], threshold: float,
) -> tuple[bool, int, int]:
    """检测任意两个不同发现类别之间的 GT 重叠情况。"""
    pairs = 0
    mask_pairs = 0
    class_names = list(gt_by_class)
    for left_index, left_class in enumerate(class_names):
        for right_class in class_names[left_index + 1:]:
            for left in gt_by_class[left_class]:
                for right in gt_by_class[right_class]:
                    if bbox_iou(left["bbox"], right["bbox"]) >= threshold:
                        pairs += 1
                    if np.logical_and(left["mask"], right["mask"]).any():
                        mask_pairs += 1
    return pairs > 0, pairs, mask_pairs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary(summary: dict[str, Any], subset: str) -> list[dict[str, Any]]:
    rows = []
    for class_name, values in summary["classes"].items():
        rows.append({"subset": subset, "class": class_name, **values})
    rows.append({"subset": subset, "class": "overall", **{key: value for key, value in summary.items() if key != "classes"}})
    return rows


SIMPLE_REPORT_FIELDS = [
    "Class",
    "GT",
    "PRED",
    "TP",
    "FN",
    "FP",
    "Precision",
    "Recall",
    "F1-Score",
]


def simple_report_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    def display_ratio(value: float) -> str:
        return f"{float(value):.3f}"

    rows = []
    for class_name, values in summary["classes"].items():
        rows.append({
            "Class": class_name,
            "GT": values["gt_count"],
            "PRED": values["pred_count"],
            "TP": values["tp"],
            "FN": values["fn"],
            "FP": values["fp"],
            "Precision": display_ratio(values["precision"]),
            "Recall": display_ratio(values["recall"]),
            "F1-Score": display_ratio(values["f1"]),
        })
    rows.append({
        "Class": "TOTAL",
        "GT": summary["gt_count"],
        "PRED": summary["pred_count"],
        "TP": summary["tp"],
        "FN": summary["fn"],
        "FP": summary["fp"],
        "Precision": display_ratio(summary["precision"]),
        "Recall": display_ratio(summary["recall"]),
        "F1-Score": display_ratio(summary["f1"]),
    })
    return rows


def write_html_report(
    path: Path,
    all_summary: dict[str, Any],
    overlap_summary: dict[str, Any],
    missing_prediction: int,
    missing_ground_truth: int,
    iou_threshold: float = 0.50,
    score_threshold: float | str = 0.00,
) -> None:
    score_threshold_str = (
        f"{score_threshold:.2f}"
        if isinstance(score_threshold, (int, float))
        else str(score_threshold)
    )

    def render_table(summary: dict[str, Any]) -> str:
        rows = simple_report_rows(summary)
        header = "".join(f"<th>{html.escape(field)}</th>" for field in SIMPLE_REPORT_FIELDS)
        body = []
        for row in rows:
            cells = []
            for field in SIMPLE_REPORT_FIELDS:
                value = row[field]
                if field in {"Precision", "Recall", "F1-Score", "F1"}:
                    value = f"{float(value):.3f}"
                cells.append(f"<td>{html.escape(str(value))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"

    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Instance Segmentation Analysis Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 28px 40px; color: #2d3748; background-color: #f8fafc; line-height: 1.6; }}
.container {{ max-width: 1100px; margin: 0 auto; background: #ffffff; padding: 32px 40px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }}
h2 {{ color: #1a202c; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-top: 0; }}
h3 {{ color: #2b6cb0; margin-top: 28px; margin-bottom: 6px; }}
.section-desc {{ color: #4a5568; font-size: 14px; margin-bottom: 12px; }}
table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; min-width: 820px; }}
th, td {{ border: 1px solid #cbd5e0; padding: 8px 12px; text-align: right; font-size: 14px; }}
th {{ background: #ebf8ff; color: #2c5282; text-align: center; font-weight: 600; }}
td:first-child {{ text-align: left; font-weight: 600; color: #2d3748; }}
tr:last-child {{ background: #edf2f7; font-weight: 700; color: #1a202c; }}
tr:hover {{ background-color: #f7fafc; }}
.meta {{ margin: 8px 0 16px; font-size: 14px; color: #4a5568; background: #edf2f7; padding: 8px 14px; border-radius: 4px; display: inline-block; }}
.notes-box {{ background: #f7fafc; border-left: 4px solid #3182ce; padding: 18px 22px; margin-top: 36px; border-radius: 0 6px 6px 0; }}
.notes-box h4 {{ margin-top: 0; margin-bottom: 12px; color: #2c5282; font-size: 16px; }}
.notes-box ul {{ margin: 0; padding-left: 20px; font-size: 13.5px; color: #4a5568; }}
.notes-box li {{ margin-bottom: 6px; }}
.notes-box code {{ background: #e2e8f0; padding: 2px 5px; border-radius: 3px; font-size: 12.5px; }}
</style>
</head>
<body>
<div class="container">
<h2>Instance Segmentation Analysis Report</h2>
<div class="meta"><strong>Evaluation Overview:</strong> Images: {all_summary["images"]} &nbsp;&nbsp;|&nbsp;&nbsp; Match IoU Threshold: {iou_threshold:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; Score Threshold: {score_threshold_str} &nbsp;&nbsp;|&nbsp;&nbsp; Missing Predictions: {missing_prediction} &nbsp;&nbsp;|&nbsp;&nbsp; Missing Ground Truth: {missing_ground_truth}</div>

<h3>Evaluation Results (All Images)</h3>
<div class="section-desc">Overall statistical metrics across all {all_summary["images"]} evaluation images.</div>
{render_table(all_summary)}

<div class="notes-box">
<h4>Metrics & Terminology Reference</h4>
<ul>
  <li><strong>GT (Ground Truth):</strong> Total count of annotated target instances.</li>
  <li><strong>PRED (Predictions):</strong> Total count of model predicted instances.</li>
  <li><strong>TP (True Positive):</strong> Predicted category matches GT and Mask IoU ≥ threshold (default 0.50), matched one-to-one in descending score order.</li>
  <li><strong>FN (False Negative):</strong> Ground truth targets not matched at the IoU threshold.</li>
  <li><strong>FP (False Positive):</strong> Model predictions not matched to any ground truth target at the IoU threshold.</li>
  <li><strong>Precision:</strong> Proportion of correct predictions among all predictions for this class (<code>TP / (TP + FP)</code>).</li>
  <li><strong>Recall:</strong> Proportion of true instances detected by the model (<code>TP / (TP + FN)</code>).</li>
  <li><strong>F1-Score:</strong> Harmonic mean of Precision and Recall (<code>2 * Precision * Recall / (Precision + Recall)</code>).</li>
</ul>
</div>
</div>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LabelMe GT JSON with inference prediction JSON")
    parser.add_argument("--ground-truth-dir", "--gt-dir", required=True, help="Directory containing LabelMe JSON and source images")
    parser.add_argument("--prediction-dir", "--pred-dir", required=True, help="Directory containing inference JSON files")
    parser.add_argument("--output-dir", "--output", required=True, help="Directory for reports and contour overlays")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Classes to evaluate; omitted means all labels found in GT and prediction JSON files",
    )
    parser.add_argument(
        "--ignore-labels",
        nargs="*",
        default=list(DEFAULT_IGNORED_LABELS),
        help="Optional labels to exclude; no labels are excluded by default",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="Same-class mask IoU threshold for TP (default: 0.50)")
    parser.add_argument("--score-threshold", "--conf-threshold", type=float, default=0.0, help="Minimum prediction confidence score threshold (default: 0.0)")
    parser.add_argument(
        "--class-conf",
        nargs="*",
        default=None,
        help="Per-class confidence score thresholds, e.g. class1=0.80 class2=0.50 or class1=0.80,class2=0.50",
    )
    parser.add_argument(
        "--cross-class-iou-threshold",
        type=float,
        default=0.10,
        help="IoU threshold for wrong-class diagnostics",
    )
    parser.add_argument(
        "--overlap-bbox-iou-threshold",
        type=float,
        default=0.20,
        help="GT cross-class bbox IoU threshold for hard-image subset",
    )
    parser.add_argument(
        "--visualize",
        choices=["errors", "all", "none"],
        default="all",
        help="Contour overlay output mode; all matches the reference analysis workflow",
    )
    parser.add_argument(
        "--error-dir",
        default=None,
        help="Directory to save overlay images for images with errors (missed/false positive instances); defaults to <output-dir>/errors",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N normalized filenames after sorting; 0 means all",
    )
    parser.add_argument(
        "--hide-overlay-text",
        action="store_true",
        help="Draw status-colored contours and instance tags without the legend panel",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_run_logging("COMPARE_JSON")
    args = parse_args(argv)
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise SystemExit("--iou-threshold must be between 0 and 1")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise SystemExit("--score-threshold must be between 0 and 1")
    if not 0.0 <= args.cross_class_iou_threshold <= 1.0:
        raise SystemExit("--cross-class-iou-threshold must be between 0 and 1")
    if args.limit < 0:
        raise SystemExit("--limit must be zero or a positive integer")

    thresholds_info = {
        "iou": args.iou_threshold,
        "score": args.score_threshold,
    }

    gt_dir = Path(args.ground_truth_dir).expanduser()
    pred_dir = Path(args.prediction_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    overlay_dir = output_dir / "visualizations"
    error_dir = Path(args.error_dir).expanduser() if getattr(args, "error_dir", None) else output_dir / "errors"

    gt_files = collect_json_files(gt_dir, prediction_side=False)
    pred_files = collect_json_files(pred_dir, prediction_side=True)
    all_names = sorted(set(gt_files) | set(pred_files))
    if not all_names:
        raise SystemExit("No JSON files found in the supplied directories")
    names = all_names[:args.limit] if args.limit else all_names

    ignored_labels = {label for label in args.ignore_labels if label}
    discovery_progress = ProgressReporter(
        len(gt_files) + len(pred_files), "Discovering labels"
    )
    discovered_classes = discover_classes(
        gt_files, pred_files, discovery_progress.update
    )
    discovery_progress.finish()
    if args.classes is None:
        args.classes = discovered_classes
    else:
        args.classes = list(dict.fromkeys(args.classes))
    args.classes = [class_name for class_name in args.classes if class_name not in ignored_labels]
    if not args.classes:
        raise SystemExit("No evaluation classes remain after applying --ignore-labels")

    from instance_segmentation.models.thresholds import class_thresholds
    conf_map = (
        class_thresholds(args.classes, args.class_conf, args.score_threshold, name="class-conf")
        if args.class_conf
        else {}
    )

    class_colors = build_class_palette(args.classes)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize != "none":
        overlay_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

    all_stats = init_stats(args.classes)
    overlap_stats = init_stats(args.classes)
    confusion = init_confusion_matrix(args.classes)
    image_rows, match_rows, detail_rows = [], [], []
    missing_prediction = []
    missing_ground_truth = []

    analysis_progress = ProgressReporter(len(names), "Analyzing images")
    for index, name in enumerate(names, 1):
        gt_path = gt_files.get(name)
        pred_path = pred_files.get(name)
        display_name = gt_path.name if gt_path is not None else pred_path.name
        if gt_path is not None:
            image_path, gt_instances, (height, width) = load_ground_truth(gt_path, args.classes)
        else:
            image_path, (height, width) = load_prediction_metadata(pred_path)
            gt_instances = []
            missing_ground_truth.append(display_name)
        raw_pred_instances = (
            load_predictions(pred_path, args.classes, height, width, score_threshold=None)
            if pred_path
            else []
        )
        pred_instances = [
            prediction
            for prediction in raw_pred_instances
            if prediction.get("score") is None
            or prediction["score"] >= conf_map.get(prediction.get("class_name"), args.score_threshold)
        ]
        low_confidence_pred_instances = [
            prediction
            for prediction in raw_pred_instances
            if prediction.get("score") is not None
            and prediction["score"] < conf_map.get(prediction.get("class_name"), args.score_threshold)
        ]
        if pred_path is None:
            missing_prediction.append(display_name)

        gt_by_class = {class_name: [] for class_name in args.classes}
        pred_by_class = {class_name: [] for class_name in args.classes}
        low_confidence_pred_by_class = {class_name: [] for class_name in args.classes}
        for instance in gt_instances:
            gt_by_class[instance["class_name"]].append(instance)
        for instance in pred_instances:
            pred_by_class[instance["class_name"]].append(instance)
        for instance in low_confidence_pred_instances:
            low_confidence_pred_by_class[instance["class_name"]].append(instance)

        matches_by_class = {}
        low_confidence_matches_by_class = {}
        unmatched_gt_by_class = {}
        unmatched_pred_by_class = {}
        for class_name in args.classes:
            matches, unmatched_gt, unmatched_pred = match_class_instances(
                gt_by_class[class_name], pred_by_class[class_name],
                args.iou_threshold,
            )
            low_confidence_matches, unmatched_gt = match_low_confidence_instances(
                gt_by_class[class_name],
                low_confidence_pred_by_class[class_name],
                unmatched_gt,
                args.iou_threshold,
            )
            matches_by_class[class_name] = matches
            low_confidence_matches_by_class[class_name] = low_confidence_matches
            unmatched_gt_by_class[class_name] = unmatched_gt
            unmatched_pred_by_class[class_name] = unmatched_pred

        low_confidence_match_ids = {
            prediction["index"]
            for class_name in args.classes
            for _, pred_index, _ in low_confidence_matches_by_class[class_name]
            for prediction in [low_confidence_pred_by_class[class_name][pred_index]]
        }
        display_pred_by_class = {
            class_name: [
                *pred_by_class[class_name],
                *[
                    prediction
                    for prediction in low_confidence_pred_by_class[class_name]
                    if prediction["index"] in low_confidence_match_ids
                ],
            ]
            for class_name in args.classes
        }

        overlap, overlap_pairs, mask_overlap_pairs = image_has_cross_class_overlap(
            gt_by_class, args.overlap_bbox_iou_threshold
        )
        all_stats["images"] += 1
        if overlap:
            overlap_stats["images"] += 1
        rows = update_instance_stats(
            all_stats, gt_by_class, pred_by_class, matches_by_class,
            unmatched_gt_by_class, unmatched_pred_by_class,
            args.iou_threshold, args.cross_class_iou_threshold,
            low_confidence_matches_by_class=low_confidence_matches_by_class,
            low_confidence_pred_by_class=low_confidence_pred_by_class,
        )
        if overlap:
            overlap_rows = update_instance_stats(
                overlap_stats, gt_by_class, pred_by_class, matches_by_class,
                unmatched_gt_by_class, unmatched_pred_by_class,
                args.iou_threshold, args.cross_class_iou_threshold,
                low_confidence_matches_by_class=low_confidence_matches_by_class,
                low_confidence_pred_by_class=low_confidence_pred_by_class,
            )
        else:
            overlap_rows = []
        image_confusion = build_confusion_matrix(
            gt_instances,
            pred_instances,
            args.classes,
            args.iou_threshold,
        )
        for gt_class, values in image_confusion.items():
            for pred_class, count in values.items():
                confusion[gt_class][pred_class] += count
        for row in rows:
            row["image"] = display_name
            match_rows.append(row)

        error_count = sum(row["status"] != "TP" for row in rows)
        cross_class_errors = sum(row["status"] in {"FP_WRONG_CLASS", "FN_WRONG_CLASS"} for row in rows)
        image_row = {
            "image": display_name,
            "gt_file": str(gt_path),
            "prediction_file": str(pred_path) if pred_path else "",
            "class_overlap": int(overlap),
            "class_bbox_overlap_pairs": overlap_pairs,
            "class_mask_overlap_pairs": mask_overlap_pairs,
            "error_count": error_count,
            "cross_class_error_count": cross_class_errors,
        }
        for class_name in args.classes:
            class_rows = [row for row in rows if row["class_name"] == class_name]
            class_matches = [row for row in class_rows if row["status"] == "TP"]
            image_row.update({
                f"gt_{class_name}": len(gt_by_class[class_name]),
                f"pred_{class_name}": len(pred_by_class[class_name]),
                f"tp_{class_name}": len(class_matches),
                f"fp_{class_name}": sum(row["status"].startswith("FP") for row in class_rows),
                f"fn_{class_name}": sum(row["status"].startswith("FN") for row in class_rows),
                f"mean_iou_{class_name}": float(np.mean([row["iou"] for row in class_matches])) if class_matches else 0.0,
            })
        image_rows.append(image_row)
        detail_rows.append({
            "image": display_name,
            "image_path": str(image_path) if image_path else "",
            "class_overlap": overlap,
            "class_bbox_overlap_pairs": overlap_pairs,
            "class_mask_overlap_pairs": mask_overlap_pairs,
            "matches": rows,
            "overlap_matches": overlap_rows,
        })

        is_error_image = error_count > 0
        should_visualize = args.visualize == "all" or (
            args.visualize == "errors" and (error_count > 0 or overlap)
        )
        if (should_visualize or is_error_image) and args.visualize != "none":
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path is not None else None
            if image is None:
                image = np.full((height, width, 3), 245, dtype=np.uint8)
            overlay = draw_overlay(
                image,
                gt_by_class,
                display_pred_by_class,
                rows,
                overlap,
                class_colors,
                thresholds=thresholds_info,
                show_overlay_text=not args.hide_overlay_text,
            )
            if should_visualize:
                cv2.imwrite(str(overlay_dir / f"{name}_analysis.png"), overlay)
            if is_error_image:
                cv2.imwrite(str(error_dir / f"{name}_analysis.png"), overlay)
        analysis_progress.update(index)
    analysis_progress.finish()

    summary = finalize_stats(all_stats)
    overlap_summary = finalize_stats(overlap_stats)
    write_csv(
        output_dir / "analysis_report.csv",
        simple_report_rows(summary),
        SIMPLE_REPORT_FIELDS,
    )
    write_csv(
        output_dir / "analysis_report_overlap.csv",
        simple_report_rows(overlap_summary),
        SIMPLE_REPORT_FIELDS,
    )
    score_thresh_display = (
        ", ".join(f"{k}={v:.2f}" for k, v in conf_map.items())
        if conf_map
        else args.score_threshold
    )
    write_html_report(
        output_dir / "analysis_report.html",
        summary,
        overlap_summary,
        len(missing_prediction),
        len(missing_ground_truth),
        iou_threshold=args.iou_threshold,
        score_threshold=score_thresh_display,
    )
    summary_payload = {
        "ground_truth_dir": str(gt_dir),
        "prediction_dir": str(pred_dir),
        "iou_threshold": args.iou_threshold,
        "score_threshold": conf_map if conf_map else args.score_threshold,
        "cross_class_iou_threshold": args.cross_class_iou_threshold,
        "overlap_bbox_iou_threshold": args.overlap_bbox_iou_threshold,
        "limit": args.limit,
        "available_filenames": len(all_names),
        "classes": args.classes,
        "class_colors_bgr": class_colors,
        "ignored_labels": sorted(ignored_labels),
        "summary": summary,
        "class_overlap_subset": overlap_summary,
        "confusion_matrix": confusion,
        "missing_prediction_json": missing_prediction,
        "missing_ground_truth_json": missing_ground_truth,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "details.json").write_text(
        json.dumps(detail_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary_rows = flatten_summary(summary, "all") + flatten_summary(overlap_summary, "class_overlap")
    summary_fields = [
        "subset", "class", "images", "gt_count", "pred_count", "tp", "fp", "fn",
        "precision", "recall", "f1", "mean_iou_tp", "cross_class_fp", "cross_class_fn",
        "background_fp", "duplicate_fp", "plain_fn", "low_confidence_fn",
        "iou_sum", "iou_count",
    ]
    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    image_fields = list(image_rows[0].keys()) if image_rows else ["image"]
    write_csv(output_dir / "per_image.csv", image_rows, image_fields)
    match_fields = ["image", "class_name", "status", "gt_index", "pred_index", "iou", "score", "other_class"]
    write_csv(output_dir / "matches.csv", match_rows, match_fields)
    confusion_rows = [{"gt_class": gt_class, **values} for gt_class, values in confusion.items()]
    write_csv(
        output_dir / "confusion_matrix.csv",
        confusion_rows,
        ["gt_class", *args.classes, "background"],
    )

    print(json.dumps({
        "images": summary["images"],
        "limit": args.limit,
        "class_overlap_images": overlap_summary["images"],
        "ignored_labels": sorted(ignored_labels),
        "missing_prediction_json": len(missing_prediction),
        "missing_ground_truth_json": len(missing_ground_truth),
        "classes": summary["classes"],
        "reports": {
            "analysis_report_csv": str(output_dir / "analysis_report.csv"),
            "analysis_report_overlap_csv": str(output_dir / "analysis_report_overlap.csv"),
            "analysis_report_html": str(output_dir / "analysis_report.html"),
            "summary_csv": str(output_dir / "summary.csv"),
            "per_image_csv": str(output_dir / "per_image.csv"),
            "matches_csv": str(output_dir / "matches.csv"),
            "confusion_matrix_csv": str(output_dir / "confusion_matrix.csv"),
            "visualizations": str(overlay_dir) if args.visualize != "none" else None,
            "errors": str(error_dir) if args.visualize != "none" else None,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
