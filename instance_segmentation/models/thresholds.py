"""Python 推理后端的通用阈值解析器与边界框后处理工具。"""

from __future__ import annotations

from typing import Iterable, Mapping

from ..schema import Prediction


def class_thresholds(
    classes: Iterable[str], configured: Mapping[str, float] | Iterable[str] | None,
    default: float, *, name: str,
) -> dict[str, float]:
    """解析按类别配置的阈值字典（支持映射表或 'class=val' 字符串列表）。"""
    names = [str(value) for value in classes]
    if len(set(names)) != len(names):
        raise ValueError("classes must not contain duplicate names")
    result = {value: float(default) for value in names}
    if isinstance(configured, Mapping):
        items = configured.items()
    else:
        raw_items = configured if isinstance(configured, (list, tuple, set)) else ([configured] if configured else [])
        items = (item.split("=", 1) if isinstance(item, str) and "=" in item else ("", item) for value in raw_items for item in str(value).split(","))
    seen = set()
    for class_name, raw in items:
        if class_name not in result:
            raise ValueError(f"{name} contains unknown class: {class_name}")
        if class_name in seen:
            raise ValueError(f"{name} contains duplicate class: {class_name}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} value for {class_name} is not numeric: {raw}") from error
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} value for {class_name} must be between 0 and 1")
        if name.endswith("iou") and value <= 0.0:
            raise ValueError(f"{name} value for {class_name} must be greater than 0")
        result[class_name] = value
        seen.add(class_name)
    return result


def bbox_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """计算两个边界框 (x0, y0, x1, y1) 之间的 IoU 交并比。"""
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    area = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(0.0, min(ly1, ry1) - max(ly0, ry0))
    union = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0) + max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0) - area
    return area / union if union > 0 else 0.0


def filter_predictions(
    predictions: Iterable[Prediction], classes: Iterable[str],
    confidences: Mapping[str, float], ious: Mapping[str, float],
) -> list[Prediction]:
    """按置信度过滤后执行按类别贪心边界框 NMS；IoU=1 时禁用 NMS。"""
    groups: dict[str, list[Prediction]] = {str(name): [] for name in classes}
    for prediction in predictions:
        name = prediction.class_name
        if name not in groups:
            raise ValueError(f"prediction contains unknown class: {name}")
        if prediction.score >= confidences[name]:
            groups[name].append(prediction)
    selected: list[Prediction] = []
    for name, candidates in groups.items():
        candidates.sort(key=lambda item: item.score, reverse=True)
        threshold = ious[name]
        for candidate in candidates:
            if threshold < 1.0 and any(
                kept.image_id == candidate.image_id
                and kept.class_name == name
                and bbox_iou(candidate.bbox, kept.bbox) > threshold
                for kept in selected
            ):
                continue
            selected.append(candidate)
    return sorted(selected, key=lambda item: item.score, reverse=True)
