"""通用实例分割评测指标计算模块。包含掩码与边界框 IoU 计算、实例一对一贪心匹配及精确率、召回率、F1 分数评估。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

from ..schema import MetricReport, Prediction


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个二值掩码之间的交并比 (IoU)。"""
    left = np.asarray(left, dtype=bool); right = np.asarray(right, dtype=bool)
    if left.shape != right.shape:
        raise ValueError(f"mask shapes differ: {left.shape} != {right.shape}")
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _bbox_iou(left, right) -> float:
    """计算两个边界框 (x0, y0, x1, y1) 之间的交并比 (IoU)。"""
    lx0, ly0, lx1, ly1 = left; rx0, ry0, rx1, ry1 = right
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(0.0, min(ly1, ry1) - max(ly0, ry0))
    area_left = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    area_right = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def _overlap(gt: Prediction, pred: Prediction) -> float:
    """计算真实标注与预测目标之间的重叠度（优先使用掩码 IoU，其次使用边界框 IoU）。"""
    if gt.mask is not None and pred.mask is not None:
        try:
            return mask_iou(np.asarray(gt.mask), np.asarray(pred.mask))
        except (ValueError, TypeError):
            pass
    return _bbox_iou(gt.bbox, pred.bbox)


def match_instances(ground_truth: Iterable[Prediction], predictions: Iterable[Prediction], iou_threshold: float = 0.5) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """按置信度从高到低对同类别真实标注与预测结果进行一对一贪心匹配。

    返回:
        (匹配对列表 [(gt_idx, pred_idx, iou), ...], 未匹配的 gt 索引列表, 未匹配的 pred 索引列表)
    """
    gt, pred = list(ground_truth), sorted(predictions, key=lambda item: item.score, reverse=True)
    used: set[int] = set(); matches = []
    for pred_index, candidate in enumerate(pred):
        best = max(((idx, _overlap(target, candidate)) for idx, target in enumerate(gt) if idx not in used and target.class_id == candidate.class_id), key=lambda item: item[1], default=(-1, 0.0))
        if best[0] >= 0 and best[1] >= iou_threshold:
            used.add(best[0]); matches.append((best[0], pred_index, float(best[1])))
    matched_gt = {item[0] for item in matches}; matched_pred = {item[1] for item in matches}
    return matches, [i for i in range(len(gt)) if i not in matched_gt], [i for i in range(len(pred)) if i not in matched_pred]


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """根据真正例 (TP)、假正例 (FP) 和假负例 (FN) 计算精确率、召回率和 F1 分数。"""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate_predictions(ground_truth: Iterable[Prediction], predictions: Iterable[Prediction], classes: Iterable[str], iou_threshold: float = 0.5) -> MetricReport:
    """评估预测集合并在指定 IoU 阈值下生成整体与分门别类的评测报告。"""
    gt, pred = list(ground_truth), list(predictions)
    per_class = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for class_name in classes:
        class_gt = [item for item in gt if item.class_name == class_name]
        class_pred = [item for item in pred if item.class_name == class_name]
        matches, missing, extra = match_instances(class_gt, class_pred, iou_threshold)
        values = {"tp": len(matches), "fp": len(extra), "fn": len(missing)}
        values.update(precision_recall_f1(**values))
        per_class[class_name] = values
        for key in total:
            total[key] += values[key]
    metrics = dict(total); metrics.update(precision_recall_f1(**total))
    metrics["images"] = len({item.image_id for item in gt} | {item.image_id for item in pred})
    return MetricReport(metrics, per_class, {"iou_threshold": iou_threshold})
