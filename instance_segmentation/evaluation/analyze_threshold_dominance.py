#!/usr/bin/env python3
"""基于现有曲线 CSV 分析严格的每类别阈值支配关系。

这是一个专用的离线报告生成器：它仅读取
``threshold_curves.csv``，既不运行推理也不访问任何测试集标注。
对于每个类别，它报告满足 ``FP <= baseline_FP`` 且 ``FN <= baseline_FN``
（基线为阈值 0.5）的最佳阈值，以及两个方向上的严格改善子集。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RELATIONS = (
    "weak_fp_fn_baseline",
    "fp_strict_down_fn_nonincrease",
    "fn_strict_down_fp_nonincrease",
)
METRIC_FIELDS = (
    "threshold",
    "gt",
    "pred",
    "tp",
    "fp",
    "fn",
    "errors",
    "precision",
    "recall",
    "f1",
    "mean_iou_tp",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Analyze strict per-class threshold dominance from an existing curve CSV.")
    parser.add_argument("--curves", required=True, help="Existing threshold_curves.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for strict_dominance.json/csv")
    parser.add_argument("--baseline-threshold", type=float, default=0.5, help="Baseline threshold to compare against (default: 0.5)")
    return parser.parse_args(argv)


def as_row(raw: dict[str, str]) -> dict[str, Any]:
    """将原始 CSV 字典转换为具有正确类型的度量记录。"""
    row: dict[str, Any] = {"class_name": raw["class_name"]}
    for field in METRIC_FIELDS:
        row[field] = float(raw[field]) if field in {"threshold", "precision", "recall", "f1", "mean_iou_tp"} else int(raw[field])
    return row


def select(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """根据确定性优先级策略从候选记录中选择最优阈值配置。"""
    if not rows:
        return None
    # 要求的确定性策略：优先最小化 FP+FN，其次最大化 F1，最后选取最高阈值。
    # 此处不添加额外的精度决胜逻辑。
    return min(rows, key=lambda row: (row["errors"], -row["f1"], -row["threshold"]))


def public_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """提取用于公开导出的字段子集。"""
    if row is None:
        return None
    return {field: row[field] for field in ("threshold", *METRIC_FIELDS[1:])}


def main(argv: list[str] | None = None) -> None:
    """主执行逻辑：读取指标曲线并生成严格支配分析报告。"""
    args = parse_args(argv)
    curves_path = Path(args.curves).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not curves_path.is_file():
        raise SystemExit(f"Missing curve CSV: {curves_path}")
    if not 0.0 <= args.baseline_threshold <= 1.0:
        raise SystemExit("--baseline-threshold must be between 0 and 1")

    by_class: dict[str, list[dict[str, Any]]] = {}
    with curves_path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = as_row(raw)
            by_class.setdefault(row["class_name"], []).append(row)

    report_classes: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for class_name in sorted(by_class):
        rows = by_class[class_name]
        baseline = min(rows, key=lambda row: abs(row["threshold"] - args.baseline_threshold))
        weak = [row for row in rows if row["fp"] <= baseline["fp"] and row["fn"] <= baseline["fn"]]
        fp_down = [row for row in rows if row["fp"] < baseline["fp"] and row["fn"] <= baseline["fn"]]
        fn_down = [row for row in rows if row["fn"] < baseline["fn"] and row["fp"] <= baseline["fp"]]
        groups = {
            "weak_fp_fn_baseline": weak,
            "fp_strict_down_fn_nonincrease": fp_down,
            "fn_strict_down_fp_nonincrease": fn_down,
        }
        class_report: dict[str, Any] = {
            "baseline": public_row(baseline),
            "baseline_row_threshold_distance": abs(baseline["threshold"] - args.baseline_threshold),
        }
        for relation, candidates in groups.items():
            chosen = select(candidates)
            strict_dimensions: list[str] = []
            if chosen is not None:
                if chosen["fp"] < baseline["fp"]:
                    strict_dimensions.append("fp")
                if chosen["fn"] < baseline["fn"]:
                    strict_dimensions.append("fn")
            entry = {
                "candidate_count": len(candidates),
                "has_strict_improvement": bool(strict_dimensions),
                "strict_improvement_dimensions": strict_dimensions,
                "selected": public_row(chosen),
            }
            class_report[relation] = entry
            csv_rows.append(
                {
                    "class_name": class_name,
                    "relation": relation,
                    "baseline_threshold": baseline["threshold"],
                    "baseline_fp": baseline["fp"],
                    "baseline_fn": baseline["fn"],
                    "candidate_count": len(candidates),
                    "has_strict_improvement": bool(strict_dimensions),
                    "strict_improvement_dimensions": "+".join(strict_dimensions),
                    "selected_threshold": chosen["threshold"] if chosen else "",
                    "selected_tp": chosen["tp"] if chosen else "",
                    "selected_fp": chosen["fp"] if chosen else "",
                    "selected_fn": chosen["fn"] if chosen else "",
                    "selected_errors": chosen["errors"] if chosen else "",
                    "selected_precision": chosen["precision"] if chosen else "",
                    "selected_recall": chosen["recall"] if chosen else "",
                    "selected_f1": chosen["f1"] if chosen else "",
                    "delta_fp": chosen["fp"] - baseline["fp"] if chosen else "",
                    "delta_fn": chosen["fn"] - baseline["fn"] if chosen else "",
                }
            )
        report_classes[class_name] = class_report

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "source_curves": str(curves_path),
        "baseline_threshold": args.baseline_threshold,
        "selection_priority": ["minimum FP+FN", "maximum F1", "highest threshold"],
        "classes": report_classes,
        "sm_test_ground_truth_used": False,
        "inference_rerun": False,
    }
    (output_dir / "strict_dominance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "class_name", "relation", "baseline_threshold", "baseline_fp", "baseline_fn", "candidate_count",
        "has_strict_improvement", "strict_improvement_dimensions", "selected_threshold", "selected_tp",
        "selected_fp", "selected_fn", "selected_errors", "selected_precision", "selected_recall", "selected_f1",
        "delta_fp", "delta_fn",
    ]
    with (output_dir / "strict_dominance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
