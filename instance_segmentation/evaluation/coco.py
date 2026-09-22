"""带有可选 pycocotools 依赖项的 COCO 指标适配器。"""

from __future__ import annotations

from typing import Any


def evaluate_coco(annotation_file: str, prediction_file: str, *, iou_types: tuple[str, ...] = ("bbox", "segm")) -> dict[str, Any]:
    """使用 COCO 标准协议评估预测结果。"""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("COCO evaluation requires the optional 'coco' extra (pycocotools)") from error
    ground_truth = COCO(annotation_file)
    predictions = ground_truth.loadRes(prediction_file)
    result = {}
    for iou_type in iou_types:
        evaluator = COCOeval(ground_truth, predictions, iou_type)
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        result[iou_type] = {"AP": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]), "AP75": float(evaluator.stats[2])}
    return result
