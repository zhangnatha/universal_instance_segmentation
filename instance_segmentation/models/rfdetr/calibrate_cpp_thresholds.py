#!/usr/bin/env python3
"""在 YOLO 验证集划分上校准 C++ 预测置信度阈值。

该工具读取已生成的 C++ JSON 预测结果档案，不运行模型。
它采用与 compare_json 相同的 np.rint / cv2.fillPoly 约定转换验证集的归一化 YOLO 多边形，
然后独立扫描每个类别的固定置信度阈值。匹配采用标准的同类别、按置信度排序、一对一掩码匹配（IoU >= 0.50）。

阈值选择策略严格限定在验证集上：
* 每个类别以全局基线阈值（默认 0.50）为起点；
* 相比基线，每个类别允许的最大新增假阳性（FP）预算为 max(1, ceil(gt_count * fp_budget_rate))；
* 在可行阈值集合中，最小化 FP + FN，随后最大化 F1、精确率、召回率，最后优先选择更高的阈值作为确定性决策；
* 输出完整指标曲线及 FP/FN 帕累托前沿以供审计。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instance_segmentation.evaluation.compare_json import (  # noqa: E402
    finalize_stats,
    init_stats,
    load_predictions,
    match_class_instances,
    update_instance_stats,
)
from instance_segmentation.data.classes import load_class_names

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
PREDICTION_AGGREGATE_FILES = {"predictions.json", "inference_manifest.json", "summary.json", "details.json", "manifest.json"}
SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 C++ 预测置信度阈值校准的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True, help="Directory containing C++ per-image JSON files")
    parser.add_argument("--valid-dir", required=True, help="YOLO validation directory with images/labels/<stem>.txt")
    parser.add_argument("--output-dir", required=True, help="Calibration artifact directory")
    parser.add_argument("--classes", nargs="+", default=None, help="Class names in YOLO id order (supports arbitrary number of classes; defaults to classes.names)")
    parser.add_argument("--baseline-threshold", type=float, default=0.50, help="Global baseline threshold (default: 0.50)")
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="Same-class mask IoU threshold (default: 0.50)")
    parser.add_argument(
        "--fp-budget-rate",
        type=float,
        default=0.01,
        help="Allowed FP increase as a fraction of class GT count (default: 0.01; minimum budget is one FP)",
    )
    parser.add_argument("--fp-budget-min", type=int, default=1, help="Minimum absolute FP budget per class (default: 1)")
    parser.add_argument("--grid-step", type=float, default=0.005, help="Fixed sweep grid step (default: 0.005)")
    return parser.parse_args(argv)


def image_files(directory: Path) -> list[Path]:
    """返回目录下所有图像文件列表。"""
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def label_to_instances(label_path: Path, height: int, width: int, classes: list[str]) -> list[dict[str, Any]]:
    """将归一化的 YOLO 多边形栅格化为 compare_json 兼容的实例掩码。"""
    instances: list[dict[str, Any]] = []
    if not label_path.exists():
        return instances
    for index, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        fields = raw.strip().split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            continue
        try:
            class_id = int(float(fields[0]))
            values = np.asarray([float(value) for value in fields[1:]], dtype=np.float32).reshape(-1, 2)
        except ValueError:
            continue
        if not 0 <= class_id < len(classes):
            continue
        values[:, 0] *= float(width)
        values[:, 1] *= float(height)
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(values) >= 3:
            cv2.fillPoly(mask, [np.rint(values).astype(np.int32)], 1)
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        instances.append(
            {
                "index": index,
                "class_name": classes[class_id],
                "mask": mask.astype(bool),
                "bbox": (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
                "area": int(mask.sum()),
            }
        )
    return instances


def load_validation(valid_dir: Path, classes: list[str]) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, tuple[int, int]], list[str]]:
    """加载验证集的全部真实标注与图像分辨率。"""
    names: list[str] = []
    gt_by_name: dict[str, list[dict[str, Any]]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    missing_labels: list[str] = []
    for image_path in image_files(valid_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        name = image_path.stem
        label_path = valid_dir / "labels" / f"{name}.txt"
        names.append(name)
        sizes[name] = (height, width)
        gt_by_name[name] = label_to_instances(label_path, height, width, classes)
        if not label_path.exists():
            missing_labels.append(name)
    return sorted(names), gt_by_name, sizes, sorted(missing_labels)


def collect_prediction_files(prediction_dir: Path) -> dict[str, Path]:
    """收集指定目录下所有单张图像的 C++ 预测 JSON 文件。"""
    files: dict[str, Path] = {}
    for path in sorted(prediction_dir.glob("*.json")):
        if path.name in PREDICTION_AGGREGATE_FILES:
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(item, dict) or "detections" not in item:
            continue
        files[path.stem] = path
    return files


def load_prediction_records(
    names: list[str],
    prediction_files: dict[str, Path],
    sizes: dict[str, tuple[int, int]],
    classes: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """加载所有图像对应的预测记录列表。"""
    pred_by_name: dict[str, list[dict[str, Any]]] = {}
    missing_predictions: list[str] = []
    for name in names:
        path = prediction_files.get(name)
        if path is None:
            pred_by_name[name] = []
            missing_predictions.append(name)
            continue
        height, width = sizes[name]
        pred_by_name[name] = load_predictions(path, classes, height, width)
    return pred_by_name, sorted(missing_predictions)


def threshold_candidates(records: list[dict[str, Any]], baseline: float, grid_step: float) -> list[float]:
    """基于基线值与网格步长生成候选阈值集合。"""
    scores = sorted({float(record.get("score", 0.0)) for record in records if 0.0 <= float(record.get("score", 0.0)) <= 1.0})
    candidates = {0.0, 1.0, float(baseline)}
    if grid_step > 0.0:
        count = int(math.floor(1.0 / grid_step + 1e-9))
        candidates.update(round(index * grid_step, 12) for index in range(count + 1))
    candidates.update(scores)
    candidates.update(float(np.nextafter(score, math.inf)) for score in scores if score < 1.0)
    return sorted(value for value in candidates if 0.0 <= value <= 1.0)


def _class_curve(
    class_name: str,
    candidates: list[float],
    names: list[str],
    gt_by_name: dict[str, list[dict[str, Any]]],
    pred_by_name: dict[str, list[dict[str, Any]]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """对每张图像执行一次贪心匹配遍历，高效计算所有阈值指标。"""
    gt_total = 0
    by_score: dict[float, list[tuple[int, float]]] = defaultdict(list)
    for name in names:
        gt_instances = [item for item in gt_by_name.get(name, []) if item["class_name"] == class_name]
        predictions = [item for item in pred_by_name.get(name, []) if item["class_name"] == class_name]
        gt_total += len(gt_instances)
        ordered = sorted(enumerate(predictions), key=lambda item: (-float(item[1].get("score", 0.0)), item[0]))
        used_gt: set[int] = set()
        for _, prediction in ordered:
            best_gt = None
            best_iou = 0.0
            for gt_index, ground_truth in enumerate(gt_instances):
                if gt_index in used_gt:
                    continue
                intersection = np.logical_and(prediction["mask"], ground_truth["mask"]).sum()
                union = np.logical_or(prediction["mask"], ground_truth["mask"]).sum()
                iou = float(intersection / union) if union else 0.0
                if iou > best_iou:
                    best_iou, best_gt = iou, gt_index
            score = float(prediction.get("score", 0.0))
            if best_gt is not None and best_iou >= iou_threshold:
                used_gt.add(best_gt)
                by_score[score].append((1, best_iou))
            else:
                by_score[score].append((0, 0.0))

    descending_scores = sorted(by_score, reverse=True)
    ascending_scores = sorted(by_score)
    cumulative: dict[float, tuple[int, int, float, int]] = {}
    tp = fp = 0
    iou_sum = 0.0
    iou_count = 0
    for score in descending_scores:
        for is_tp, iou in by_score[score]:
            if is_tp:
                tp += 1
                iou_sum += iou
                iou_count += 1
            else:
                fp += 1
        cumulative[score] = (tp, fp, iou_sum, iou_count)

    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        index = bisect.bisect_left(ascending_scores, threshold)
        if index < len(ascending_scores):
            selected_score = ascending_scores[index]
            tp, fp, iou_sum, iou_count = cumulative[selected_score]
            pred_total = tp + fp
        else:
            tp = fp = pred_total = iou_count = 0
            iou_sum = 0.0
        fn = gt_total - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class_name": class_name,
                "threshold": float(threshold),
                "gt": int(gt_total),
                "pred": int(pred_total),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "errors": int(fp + fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mean_iou_tp": float(iou_sum / iou_count) if iou_count else 0.0,
            }
        )
    return rows


def pareto_frontier(rows: list[dict[str, Any]]) -> set[float]:
    """计算指标曲线中关于 (FP, FN) 的帕累托前沿阈值集合。"""
    frontier: set[float] = set()
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other["fp"] <= candidate["fp"]
            and other["fn"] <= candidate["fn"]
            and (other["fp"] < candidate["fp"] or other["fn"] < candidate["fn"])
            for other in rows
        )
        if not dominated:
            frontier.add(float(candidate["threshold"]))
    return frontier


def select_threshold(rows: list[dict[str, Any]], baseline: float, budget: int) -> tuple[dict[str, Any], dict[str, Any], set[float]]:
    """在假阳性预算约束内执行词典序优化选择最佳阈值。"""
    baseline_row = min(rows, key=lambda row: abs(float(row["threshold"]) - baseline))
    max_fp = int(baseline_row["fp"] + budget)
    for row in rows:
        row["baseline"] = bool(abs(float(row["threshold"]) - baseline) < 1e-12)
        row["baseline_fp"] = int(baseline_row["fp"])
        row["fp_delta_vs_baseline"] = int(row["fp"] - baseline_row["fp"])
        row["fp_budget"] = int(budget)
        row["constraint_max_fp"] = max_fp
        row["constraint_satisfied"] = bool(row["fp"] <= max_fp)
    frontier = pareto_frontier(rows)
    for row in rows:
        row["pareto_frontier"] = bool(float(row["threshold"]) in frontier)
    feasible = [row for row in rows if row["constraint_satisfied"]]
    selected = min(feasible, key=lambda row: (row["errors"], -row["f1"], -row["precision"], -row["recall"], -row["threshold"]))
    decision = {
        "threshold": float(selected["threshold"]),
        "baseline_threshold": float(baseline),
        "baseline_fp": int(baseline_row["fp"]),
        "baseline_fn": int(baseline_row["fn"]),
        "baseline_tp": int(baseline_row["tp"]),
        "fp_budget": int(budget),
        "constraint_max_fp": max_fp,
        "feasible_candidate_count": len(feasible),
        "pareto_candidate_count": sum(1 for row in rows if row["pareto_frontier"]),
        "selected": {key: selected[key] for key in ("gt", "pred", "tp", "fp", "fn", "errors", "precision", "recall", "f1", "mean_iou_tp")},
        "selection_policy": "min_fp_plus_fn_then_max_f1_then_precision_then_recall_then_highest_threshold",
    }
    return selected, decision, frontier


def evaluate_thresholds(
    names: list[str],
    gt_by_name: dict[str, list[dict[str, Any]]],
    pred_by_name: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float],
    classes: list[str],
    iou_threshold: float,
) -> dict[str, Any]:
    """在给定各类别选定阈值下计算全局验证集指标报告。"""
    stats = init_stats(classes)
    for name in names:
        gt_by_class = {class_name: [item for item in gt_by_name.get(name, []) if item["class_name"] == class_name] for class_name in classes}
        pred_by_class = {
            class_name: [item for item in pred_by_name.get(name, []) if item["class_name"] == class_name and float(item.get("score", 0.0)) >= thresholds[class_name]]
            for class_name in classes
        }
        matches: dict[str, list[tuple[int, int, float]]] = {}
        unmatched_gt: dict[str, list[int]] = {}
        unmatched_pred: dict[str, list[int]] = {}
        for class_name in classes:
            matches[class_name], unmatched_gt[class_name], unmatched_pred[class_name] = match_class_instances(
                gt_by_class[class_name], pred_by_class[class_name], iou_threshold
            )
        stats["images"] += 1
        update_instance_stats(
            stats,
            gt_by_class,
            pred_by_class,
            matches,
            unmatched_gt,
            unmatched_pred,
            iou_threshold,
            0.30,
        )
    return finalize_stats(stats)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """将数据行格式化导出为 CSV 文件。"""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_selected_predictions(
    prediction_files: dict[str, Path],
    output_dir: Path,
    thresholds: dict[str, float],
) -> tuple[int, int]:
    """应用选定阈值过滤每张图像的预测并将结果输出至指定目录。"""
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    before = after = 0
    for name, source in sorted(prediction_files.items()):
        item = json.loads(source.read_text(encoding="utf-8-sig"))
        detections = item.get("detections") or []
        kept = [detection for detection in detections if float(detection.get("score", 0.0)) >= thresholds.get(str(detection.get("class_name")), 1.0)]
        before += len(detections)
        after += len(kept)
        item["detections"] = kept
        item["thresholds"] = thresholds
        item["threshold_source"] = "instance_segmentation.models.rfdetr.calibrate_cpp_thresholds"
        (output_dir / source.name).write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return before, after


def main(argv: list[str] | None = None) -> None:
    """C++ 预测阈值校准主函数。"""
    args = parse_args(argv)
    classes = list(args.classes) if args.classes else load_class_names()
    if not classes:
        raise ValueError("Classes must be provided via --classes or configured in classes.names")
    if len(set(classes)) != len(classes):
        raise SystemExit("--classes must contain unique names")
    for value, name in ((args.baseline_threshold, "baseline-threshold"), (args.iou_threshold, "iou-threshold"), (args.fp_budget_rate, "fp-budget-rate")):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name} must be between 0 and 1")
    if args.fp_budget_min < 0:
        raise SystemExit("--fp-budget-min must be non-negative")
    if args.grid_step <= 0.0 or args.grid_step > 1.0:
        raise SystemExit("--grid-step must be in (0, 1]")

    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    valid_dir = Path(args.valid_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not prediction_dir.is_dir():
        raise SystemExit(f"Prediction directory does not exist: {prediction_dir}")
    if not valid_dir.is_dir():
        raise SystemExit(f"Validation directory does not exist: {valid_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    names, gt_by_name, sizes, missing_labels = load_validation(valid_dir, classes)
    if not names:
        raise SystemExit(f"No validation images found: {valid_dir}")
    prediction_files = collect_prediction_files(prediction_dir)
    pred_by_name, missing_predictions = load_prediction_records(names, prediction_files, sizes, classes)

    curves: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for class_name in classes:
        records = [prediction for predictions in pred_by_name.values() for prediction in predictions if prediction["class_name"] == class_name]
        candidates = threshold_candidates(records, args.baseline_threshold, args.grid_step)
        rows = _class_curve(class_name, candidates, names, gt_by_name, pred_by_name, args.iou_threshold)
        gt_count = int(rows[0]["gt"]) if rows else 0
        budget = max(args.fp_budget_min, int(math.ceil(gt_count * args.fp_budget_rate)))
        selected, decision, _ = select_threshold(rows, args.baseline_threshold, budget)
        decision["candidate_count"] = len(candidates)
        decision["raw_prediction_count"] = len(records)
        decisions[class_name] = decision
        thresholds[class_name] = float(selected["threshold"])
        curves.extend(rows)
        print(
            f"[select] {class_name}: threshold={selected['threshold']:.9f} "
            f"baseline_fp={decision['baseline_fp']} budget={budget} "
            f"TP={selected['tp']} FP={selected['fp']} FN={selected['fn']} F1={selected['f1']:.6f}",
            flush=True,
        )

    selected_summary = evaluate_thresholds(names, gt_by_name, pred_by_name, thresholds, classes, args.iou_threshold)
    selected_before, selected_after = write_selected_predictions(prediction_files, output_dir / "selected_predictions", thresholds)

    curve_fields = [
        "class_name", "threshold", "gt", "pred", "tp", "fp", "fn", "errors", "precision", "recall", "f1", "mean_iou_tp",
        "baseline", "baseline_fp", "fp_delta_vs_baseline", "fp_budget", "constraint_max_fp", "constraint_satisfied", "pareto_frontier",
    ]
    write_csv(output_dir / "threshold_curves.csv", curves, curve_fields)
    pareto_rows = [row for row in curves if row.get("pareto_frontier")]
    write_csv(output_dir / "pareto_frontier.csv", pareto_rows, curve_fields)
    (output_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "prediction_dir": str(prediction_dir),
                "valid_dir": str(valid_dir),
                "classes": classes,
                "baseline_threshold": args.baseline_threshold,
                "iou_threshold": args.iou_threshold,
                "fp_budget_rate": args.fp_budget_rate,
                "fp_budget_min": args.fp_budget_min,
                "thresholds": thresholds,
                "selection": decisions,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "valid_selected_summary.json").write_text(json.dumps(selected_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": "instance_segmentation.models.rfdetr.calibrate_cpp_thresholds",
        "prediction_dir": str(prediction_dir),
        "valid_dir": str(valid_dir),
        "output_dir": str(output_dir),
        "classes": classes,
        "matching": {
            "same_class": True,
            "mask_iou_threshold": args.iou_threshold,
            "policy": "compare_json.match_class_instances",
            "prediction_order": "descending score, then archive order",
            "yolo_polygon_rasterization": "np.rint + cv2.fillPoly",
        },
        "selection_policy": {
            "baseline_threshold": args.baseline_threshold,
            "fp_budget_formula": "max(fp_budget_min, ceil(gt_count * fp_budget_rate))",
            "fp_budget_rate": args.fp_budget_rate,
            "fp_budget_min": args.fp_budget_min,
            "objective": "minimize FP+FN, then maximize F1, precision, recall, then highest threshold",
            "uses_sm_test_ground_truth": False,
        },
        "validation_images": len(names),
        "prediction_json_files": len(prediction_files),
        "missing_predictions": missing_predictions,
        "missing_labels": missing_labels,
        "raw_prediction_count": selected_before,
        "selected_prediction_count": selected_after,
        "thresholds": thresholds,
        "selection": decisions,
        "selected_validation_summary": selected_summary,
        "artifacts": {
            "thresholds": "thresholds.json",
            "curves": "threshold_curves.csv",
            "pareto_frontier": "pareto_frontier.csv",
            "selected_predictions": "selected_predictions/",
            "selected_validation_summary": "valid_selected_summary.json",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"thresholds": thresholds, "selected_validation_summary": selected_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
