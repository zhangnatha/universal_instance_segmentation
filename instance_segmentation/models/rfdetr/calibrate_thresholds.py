#!/usr/bin/env python3
"""校准 RF-DETR 检查点的固定按类别置信度阈值。

校准划分严格限制在提供的验证集上。
首先在较低的全局置信度阈值下生成原始预测，然后通过复用仓库中 compare_json 的多边形栅格化
和按置信度排序的一对一掩码匹配规则评估所有候选阈值。
可选的独立测试集验证阶段将冻结的阈值应用一次，仅用于生成报告，不影响已选定的阈值。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

# 当此文件作为脚本运行时，将仓库根目录加入导入路径
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instance_segmentation.evaluation.compare_json import (  # noqa: E402
    finalize_stats,
    init_stats,
    load_ground_truth,
    load_predictions,
    mask_iou,
    match_class_instances,
    update_instance_stats,
)
from instance_segmentation.evaluation.evaluator import ModelEvaluator  # noqa: E402
from instance_segmentation.data.classes import load_class_names

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析置信度阈值校准的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="EMA checkpoint to calibrate")
    parser.add_argument("--valid-dir", required=True, help="Flat validation dir with labels/<stem>.txt")
    parser.add_argument("--sm-dir", required=True, help="SM_Test dir containing source images and LabelMe JSON GT")
    parser.add_argument("--output-dir", required=True, help="Calibration artifact directory")
    parser.add_argument("--classes", nargs="+", default=None, help="Class names to calibrate in YOLO class id order")
    parser.add_argument("--raw-conf", type=float, default=0.05, help="Global generation confidence (default: 0.05)")
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="Same-class mask IoU TP threshold")
    parser.add_argument("--cross-class-iou-threshold", type=float, default=0.30, help="Cross-class diagnostic threshold")
    parser.add_argument("--device", default="auto", help="Kept for manifest compatibility; evaluator selects device")
    parser.add_argument("--no-sm-test", action="store_true", help="Skip the report-only SM_Test validation")
    parser.add_argument("--cpp-baseline", default=None, help="Optional path to historical C++ baseline summary JSON")
    return parser.parse_args(argv)


def ensure_empty_dir(path: Path) -> None:
    """创建空目录，清理已存在的子文件或目录。"""
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def image_files(directory: Path) -> list[Path]:
    """返回目录中所有符合支持后缀的图像文件列表。"""
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def label_to_instances(label_path: Path, height: int, width: int, classes: list[str]) -> list[dict[str, Any]]:
    """将归一化的 YOLO 多边形行转换为 compare_json 兼容的多边形实例。"""
    instances: list[dict[str, Any]] = []
    if not label_path.exists():
        return instances
    for index, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        fields = raw.strip().split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            continue
        try:
            class_id = int(float(fields[0]))
            values = np.asarray([float(x) for x in fields[1:]], dtype=np.float32).reshape(-1, 2)
        except ValueError:
            continue
        points = values.copy()
        points[:, 0] *= float(width)
        points[:, 1] *= float(height)
        # compare_json.polygon_mask: np.rint -> int32 -> cv2.fillPoly 语义
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        if not mask.any():
            continue
        class_name = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"
        ys, xs = np.nonzero(mask)
        instances.append({
            "index": index,
            "class_name": class_name,
            "mask": mask.astype(bool),
            "bbox": (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
            "area": int(mask.sum()),
            "points": points.tolist(),
        })
    return instances


def valid_ground_truth(image_path: Path, classes: list[str]) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    """读取验证集真实标注实例与图像尺寸。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read validation image: {image_path}")
    height, width = image.shape[:2]
    return label_to_instances(image_path.parent / "labels" / f"{image_path.stem}.txt", height, width, classes), (height, width)


def prediction_payload(image_path: Path, image: np.ndarray, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """将单张图像的模型预测转换为标准 JSON 载荷格式。"""
    height, width = image.shape[:2]
    records = []
    for p in predictions:
        points = p.get("points") or []
        if len(points) < 3:
            continue
        records.append({
            "class_id": int(p.get("class_id", 0)),
            "class_name": str(p.get("class", "object")),
            "score": float(p.get("confidence", 0.0)),
            "bbox_xyxy": [
                float(p.get("x", 0.0) - p.get("width", 0.0) / 2.0),
                float(p.get("y", 0.0) - p.get("height", 0.0) / 2.0),
                float(p.get("x", 0.0) + p.get("width", 0.0) / 2.0),
                float(p.get("y", 0.0) + p.get("height", 0.0) / 2.0),
            ],
            "contours_xy": [[[float(pt.get("x", 0.0)), float(pt.get("y", 0.0))] for pt in points]],
        })
    image_path = image_path.resolve()
    return {
        "file": str(image_path),
        "imagePath": image_path.name,
        "image": str(image_path),
        "width": int(width),
        "height": int(height),
        "imageWidth": int(width),
        "imageHeight": int(height),
        "detections": records,
    }


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """将字典对象以 UTF-8 编码写入格式化 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate_predictions(
    model_obj: dict[str, Any],
    source_dir: Path,
    output_dir: Path,
    raw_conf: float,
    yolo_valid: bool = False,
) -> tuple[int, int, float]:
    """生成低置信度原始预测档案。"""
    ensure_empty_dir(output_dir)
    files = image_files(source_dir)
    total_predictions = 0
    t0 = time.time()
    for index, image_path in enumerate(files, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        result = model_obj["evaluator"].predict_image_array(
            image, conf_threshold=raw_conf, model_obj=model_obj["model_obj"], return_annotated=False
        )
        predictions = result.get("predictions", [])
        payload = prediction_payload(image_path, image, predictions)
        total_predictions += len(payload["detections"])
        dump_json(output_dir / f"{image_path.stem}.json", payload)
        if index == 1 or index == len(files) or index % 100 == 0:
            elapsed = time.time() - t0
            print(f"[generate] {source_dir.name}: {index}/{len(files)} images, {total_predictions} masks, {elapsed:.1f}s", flush=True)
    return len(files), total_predictions, time.time() - t0


def threshold_candidates(records: Iterable[dict[str, Any]], raw_conf: float) -> list[float]:
    """基于网格步长与预测得分节点生成候选阈值列表。"""
    scores = sorted({float(r["score"]) for r in records if float(r["score"]) >= raw_conf})
    candidates = {float(raw_conf), 1.0}
    # 固定网格使得曲线易于跨轮次比较；精确的得分节点使选定点独立于网格分辨率
    candidates.update(round(x / 1000.0, 3) for x in range(math.ceil(raw_conf * 1000), 1001, 5))
    candidates.update(scores)
    for score in scores:
        if score < 1.0:
            candidates.add(float(np.nextafter(score, math.inf)))
    return sorted(x for x in candidates if raw_conf <= x <= 1.0)


def valid_records(raw_dir: Path, valid_dir: Path, classes: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """按文件名加载原始预测记录与真实标注掩码。"""
    raw_by_name: dict[str, dict[str, Any]] = {}
    gt_by_name: dict[str, list[dict[str, Any]]] = {}
    pred_by_name: dict[str, list[dict[str, Any]]] = {}
    for image_path in image_files(valid_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        raw_path = raw_dir / f"{image_path.stem}.json"
        item = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {"detections": []}
        raw_by_name[image_path.stem] = item
        gt_by_name[image_path.stem] = label_to_instances(valid_dir / "labels" / f"{image_path.stem}.txt", height, width, classes)
        pred_by_name[image_path.stem] = load_predictions(raw_path, classes, height, width) if raw_path.exists() else []
    return raw_by_name, gt_by_name, pred_by_name


def evaluate_threshold(
    image_names: list[str],
    gt_by_name: dict[str, list[dict[str, Any]]],
    pred_by_name: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float],
    iou_threshold: float,
    cross_class_iou_threshold: float,
    classes: list[str],
) -> dict[str, Any]:
    """在指定分类阈值配置下计算全局评估统计指标。"""
    stats = init_stats(classes)
    for name in image_names:
        gt_instances = gt_by_name.get(name, [])
        selected = {
            c: [p for p in pred_by_name.get(name, []) if p["class_name"] == c and float(p.get("score", 0.0)) >= thresholds[c]]
            for c in classes
        }
        gt_by_class = {c: [x for x in gt_instances if x["class_name"] == c] for c in classes}
        pred_by_class = selected
        matches = {}
        unmatched_gt = {}
        unmatched_pred = {}
        for c in classes:
            matches[c], unmatched_gt[c], unmatched_pred[c] = match_class_instances(gt_by_class[c], pred_by_class[c], iou_threshold)
        stats["images"] += 1
        update_instance_stats(stats, gt_by_class, pred_by_class, matches, unmatched_gt, unmatched_pred, iou_threshold, cross_class_iou_threshold)
    return finalize_stats(stats)


def choose_threshold(curve: list[dict[str, Any]], class_name: str) -> tuple[dict[str, Any], str]:
    """应用确定性词典序选择策略选出最佳阈值。"""
    exact = [r for r in curve if r["fp"] == 0 and r["fn"] == 0]
    if exact:
        return max(exact, key=lambda r: r["threshold"]), "exact_zero_fp_fn"
    # 安全约束回退：优先零误报，其次最小漏报
    zero_fp = [r for r in curve if r["fp"] == 0]
    if zero_fp and class_name == "leg":
        return min(zero_fp, key=lambda r: (r["fn"], -r["f1"], -r["precision"], -r["threshold"])), "zero_fp_fallback"
    return min(curve, key=lambda r: (r["fn"] + r["fp"], -r["f1"], -r["precision"], -r["threshold"])), "min_error_then_f1_precision_threshold"


def make_curve(
    class_name: str,
    candidates: list[float],
    image_names: list[str],
    gt_by_name: dict[str, list[dict[str, Any]]],
    pred_by_name: dict[str, list[dict[str, Any]]],
    iou_threshold: float,
    cross_class_iou_threshold: float,
) -> list[dict[str, Any]]:
    """计算单个类别在所有候选阈值下的指标曲线。"""
    # 置信度排序匹配具有前缀稳定性：降低阈值只会追加更低置信度的预测，因此较早的匹配不会改变。
    # 预先计算每个预测的 TP/FP 状态，然后聚合得分节点。
    gt_total = 0
    by_score: dict[float, list[tuple[int, float]]] = defaultdict(list)
    for name in image_names:
        gt = [x for x in gt_by_name.get(name, []) if x["class_name"] == class_name]
        pred = [x for x in pred_by_name.get(name, []) if x["class_name"] == class_name]
        gt_total += len(gt)
        ordered = sorted(enumerate(pred), key=lambda item: (-float(item[1].get("score", 0.0)), item[0]))
        used_gt: set[int] = set()
        for _, item in ordered:
            best_gt = None
            best_iou = 0.0
            for gt_index, gt_item in enumerate(gt):
                if gt_index in used_gt:
                    continue
                iou = mask_iou(item["mask"], gt_item["mask"])
                if iou > best_iou:
                    best_iou, best_gt = iou, gt_index
            score = float(item.get("score", 0.0))
            if best_gt is not None and best_iou >= iou_threshold:
                used_gt.add(best_gt)
                by_score[score].append((1, best_iou))
            else:
                by_score[score].append((0, 0.0))

    cumulative: dict[float, tuple[int, int, float, int]] = {}
    tp = fp = 0
    iou_sum = 0.0
    iou_count = 0
    pred_total = 0
    for score in sorted(by_score, reverse=True):
        for is_tp, iou in by_score[score]:
            pred_total += 1
            if is_tp:
                tp += 1
                iou_sum += iou
                iou_count += 1
            else:
                fp += 1
        cumulative[score] = (tp, fp, iou_sum, iou_count)

    rows = []
    for threshold in candidates:
        # 仅包含满足 score >= threshold 的得分组
        eligible = [score for score in cumulative if score >= threshold]
        if eligible:
            score = min(eligible)
            tp, fp, iou_sum, iou_count = cumulative[score]
            pred_total = tp + fp
        else:
            tp = fp = pred_total = iou_count = 0
            iou_sum = 0.0
        fn = gt_total - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
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
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """将数据行导出为 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def apply_thresholds(raw_dir: Path, output_dir: Path, thresholds: dict[str, float]) -> tuple[int, int]:
    """对原始 JSON 预测应用指定分类阈值并保存过滤后的结果。"""
    ensure_empty_dir(output_dir)
    total_before = total_after = 0
    for raw_path in sorted(raw_dir.glob("*.json")):
        item = json.loads(raw_path.read_text(encoding="utf-8"))
        detections = item.get("detections") or []
        total_before += len(detections)
        kept = [d for d in detections if float(d.get("score", 0.0)) >= thresholds.get(str(d.get("class_name")), 1.0)]
        total_after += len(kept)
        item["detections"] = kept
        item["thresholds"] = thresholds
        dump_json(output_dir / raw_path.name, item)
    return total_before, total_after


def run_compare(gt_dir: Path, pred_dir: Path, output_dir: Path, iou_threshold: float, cross_class_iou_threshold: float, classes: list[str]) -> dict[str, Any]:
    """调用 compare_json 评估器运行一次评估（关闭可视化）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "instance_segmentation.evaluation.compare_json",
        "--ground-truth-dir", str(gt_dir), "--prediction-dir", str(pred_dir),
        "--output-dir", str(output_dir), "--classes", *classes,
        "--iou-threshold", str(iou_threshold),
        "--cross-class-iou-threshold", str(cross_class_iou_threshold),
        "--visualize", "none",
    ]
    proc = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, check=True)
    summary_path = output_dir / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def cpp_baseline(path: Path) -> dict[str, Any] | None:
    """加载历史 C++ 基线摘要数据。"""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("summary", payload)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    """阈值校准主入口函数。"""
    args = parse_args(argv)
    for value, name in ((args.raw_conf, "raw-conf"), (args.iou_threshold, "iou-threshold"), (args.cross_class_iou_threshold, "cross-class-iou-threshold")):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name} must be between 0 and 1")
    weights = Path(args.weights).expanduser().resolve()
    valid_dir = Path(args.valid_dir).expanduser().resolve()
    sm_dir = Path(args.sm_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_raw = output_dir / "valid_raw_conf005"
    sm_raw = output_dir / "sm_test_raw_conf005"
    valid_thresholded = output_dir / "valid_selected"
    sm_thresholded = output_dir / "sm_test_selected"

    classes = list(args.classes) if args.classes else load_class_names()
    if not classes:
        raise ValueError("Classes must be provided via --classes or configured in classes.names")

    print(f"[load] checkpoint={weights}", flush=True)
    evaluator = ModelEvaluator()
    model = evaluator.load_model(str(weights))
    if model is None:
        raise SystemExit(f"Could not load checkpoint: {weights}")
    wrapped = {"evaluator": evaluator, "model_obj": model}

    valid_images, valid_raw_count, valid_elapsed = generate_predictions(model_obj=wrapped, source_dir=valid_dir, output_dir=valid_raw, raw_conf=args.raw_conf)
    raw_by_name, gt_by_name, pred_by_name = valid_records(valid_raw, valid_dir, classes)
    names = sorted(gt_by_name)
    curves: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for class_name in classes:
        records = [p for preds in pred_by_name.values() for p in preds if p["class_name"] == class_name]
        candidates = threshold_candidates(records, args.raw_conf)
        curve = make_curve(class_name, candidates, names, gt_by_name, pred_by_name, args.iou_threshold, args.cross_class_iou_threshold)
        choice, policy = choose_threshold(curve, class_name)
        for row in curve:
            row["selected"] = bool(abs(float(row["threshold"]) - float(choice["threshold"])) < 1e-12)
        curves.extend(curve)
        selected[class_name] = {
            **choice,
            "selection_policy": policy,
            "candidate_count": len(candidates),
            "constraint_satisfied": bool(choice["fp"] == 0 and choice["fn"] == 0) if (choice["fp"] == 0 and choice["fn"] == 0) else None,
        }
        print(f"[select] {class_name}: threshold={choice['threshold']:.9f} TP={choice['tp']} FP={choice['fp']} FN={choice['fn']} F1={choice['f1']:.6f} policy={policy}", flush=True)

    thresholds = {class_name: float(selected[class_name]["threshold"]) for class_name in classes}
    write_csv(output_dir / "threshold_curves.csv", curves, ["class_name", "threshold", "gt", "pred", "tp", "fp", "fn", "errors", "precision", "recall", "f1", "mean_iou_tp", "selected"])
    (output_dir / "thresholds.json").write_text(json.dumps({"checkpoint": str(weights), "valid_dir": str(valid_dir), "raw_conf": args.raw_conf, "iou_threshold": args.iou_threshold, "thresholds": thresholds, "selection": selected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    valid_before, valid_after = apply_thresholds(valid_raw, valid_thresholded, thresholds)
    valid_summary = evaluate_threshold(names, gt_by_name, pred_by_name, thresholds, args.iou_threshold, args.cross_class_iou_threshold, classes)
    (output_dir / "valid_selected_summary.json").write_text(json.dumps(valid_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result: dict[str, Any] = {
        "checkpoint": str(weights), "valid_dir": str(valid_dir), "raw_conf": args.raw_conf,
        "valid_images": valid_images, "valid_raw_predictions": valid_before, "valid_selected_predictions": valid_after,
        "valid_generation_seconds": valid_elapsed, "thresholds": thresholds,
        "valid_summary": valid_summary,
    }

    if not args.no_sm_test:
        sm_images, sm_raw_count, sm_elapsed = generate_predictions(model_obj=wrapped, source_dir=sm_dir, output_dir=sm_raw, raw_conf=args.raw_conf)
        sm_before, sm_after = apply_thresholds(sm_raw, sm_thresholded, thresholds)
        sm_eval = run_compare(sm_dir, sm_thresholded, output_dir / "sm_test_eval", args.iou_threshold, args.cross_class_iou_threshold, classes)
        cpp_path = Path(args.cpp_baseline).expanduser().resolve() if args.cpp_baseline else (ROOT / "results/cpp_evalution/summary.json")
        cpp = cpp_baseline(cpp_path)
        result["sm_test"] = {
            "source_dir": str(sm_dir), "images": sm_images, "raw_predictions": sm_before,
            "selected_predictions": sm_after, "generation_seconds": sm_elapsed, "evaluation": sm_eval,
            "historical_cpp_baseline": cpp,
        }
        (output_dir / "sm_test_summary.json").write_text(json.dumps(result["sm_test"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if cpp is not None:
            comparison_rows = []
            ours = sm_eval["summary"]
            for class_name in [*classes, "overall"]:
                ours_item = ours["classes"].get(class_name, ours)
                cpp_item = cpp["classes"].get(class_name, cpp)
                comparison_rows.append({
                    "class_name": class_name,
                    "ours_tp": ours_item["tp"], "cpp_tp": cpp_item["tp"], "delta_tp": ours_item["tp"] - cpp_item["tp"],
                    "ours_fp": ours_item["fp"], "cpp_fp": cpp_item["fp"], "delta_fp": ours_item["fp"] - cpp_item["fp"],
                    "ours_fn": ours_item["fn"], "cpp_fn": cpp_item["fn"], "delta_fn": ours_item["fn"] - cpp_item["fn"],
                    "ours_precision": ours_item["precision"], "cpp_precision": cpp_item["precision"],
                    "ours_recall": ours_item["recall"], "cpp_recall": cpp_item["recall"],
                    "ours_f1": ours_item["f1"], "cpp_f1": cpp_item["f1"],
                    "ours_mean_iou_tp": ours_item["mean_iou_tp"], "cpp_mean_iou_tp": cpp_item["mean_iou_tp"],
                })
            write_csv(output_dir / "comparison_vs_cpp.csv", comparison_rows, list(comparison_rows[0]))
            (output_dir / "comparison_vs_cpp.json").write_text(json.dumps(comparison_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (output_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
