#!/usr/bin/env python3
"""绘制 RF-DETR、Ultralytics 或 Detectron2 的训练指标。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

# 每条曲线保存所有可用的（周期/迭代次数，数值）数据点。
# 按行保留数据可保留稀疏验证记录和事件日志中的重复迭代次数。
SERIES = {
    "loss": [], "lr": [], "bbox_ap": [], "mask_ap": [],
    "bbox_ap50": [], "mask_ap50": [], "recall": [],
}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def append_row(data, row, x, backend):
    if x is None:
        return

    def add(series, *keys):
        for key in keys:
            value = number(row.get(key))
            if value is not None:
                data[series].append((x, value))
                return

    if backend == "rfdetr":
        add("loss", "train/loss")
        add("lr", "train/lr")
        add("bbox_ap", "val/ema_mAP_50_95", "val/mAP_50_95")
        add("mask_ap", "val/ema_segm_mAP_50_95", "val/segm_mAP_50_95")
        add("bbox_ap50", "val/ema_mAP_50", "val/mAP_50")
        add("mask_ap50", "val/ema_segm_mAP_50", "val/segm_mAP_50")
        add("recall", "val/ema_mAR", "val/mAR")
    elif backend == "ultralytics":
        # Ultralytics 提供各分项损失，而非单一总损失。
        for key, label in (("train/box_loss", "box loss"), ("train/seg_loss", "mask loss"),
                           ("train/cls_loss", "class loss"), ("train/dfl_loss", "DFL loss")):
            value = number(row.get(key))
            if value is not None:
                data["loss"].append((x, value, label))
        add("lr", "lr/pg0", "lr/pg1", "lr/pg2")
        add("bbox_ap", "metrics/mAP50-95(B)", "metrics/mAP50-95")
        add("mask_ap", "metrics/mAP50-95(M)")
        add("bbox_ap50", "metrics/mAP50(B)", "metrics/mAP50")
        add("mask_ap50", "metrics/mAP50(M)")
        add("recall", "metrics/recall(M)", "metrics/recall(B)", "metrics/recall")
    else:
        add("loss", "total_loss")
        add("lr", "lr")
        add("bbox_ap", "bbox/AP")
        add("mask_ap", "segm/AP")
        add("bbox_ap50", "bbox/AP50")
        add("mask_ap50", "segm/AP50")
        add("recall", "segm/AR", "bbox/AR")


def read_metrics(path: Path):
    """读取 RF-DETR/Ultralytics CSV 或 Detectron2 JSONL，并整理为可绘制的数据序列。"""
    if path.suffix.lower() in {".json", ".jsonl"}:
        backend, data = "detectron2", {key: [] for key in SERIES}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                append_row(data, row, number(row.get("iteration")), backend)
        return backend, data

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        backend = "ultralytics" if any(key.startswith("metrics/") for key in fields) else "rfdetr"
        data = {key: [] for key in SERIES}
        for row in reader:
            x = number(row.get("epoch"))
            append_row(data, row, x, backend)
    return backend, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="RF-DETR/Ultralytics CSV or Detectron2 metrics.json")
    parser.add_argument("--output", type=Path, required=True, help="PNG output path")
    args = parser.parse_args()

    backend, data = read_metrics(args.metrics)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x_label = "iteration" if backend == "detectron2" else "epoch"
    ap_unit = "percent (0–100)" if backend == "detectron2" else "fraction (0–1)"

    def plot_metric(ax, series, label):
        points = data[series]
        if not points:
            return
        if series == "loss" and len(points[0]) == 3:
            # 此图表面板会同时显示多个 Ultralytics 损失分项。
            by_label = {}
            for x, y, name in points:
                by_label.setdefault(name, ([], []))[0].append(x)
                by_label[name][1].append(y)
            for name, (xs, ys) in by_label.items():
                ax.plot(xs, ys, marker=".", label=name)
        else:
            x, y = zip(*points)
            ax.plot(x, y, marker=".", label=label)

    plot_metric(axes[0, 0], "loss", "train loss")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel(x_label)
    if data["loss"] and len(data["loss"][0]) == 3:
        axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    plot_metric(axes[0, 1], "bbox_ap", "bbox AP")
    plot_metric(axes[0, 1], "mask_ap", "mask AP")
    axes[0, 1].set_title("Validation AP@[.50:.95]")
    axes[0, 1].set_xlabel(x_label)
    axes[0, 1].set_ylabel(f"AP ({ap_unit})")
    if data["bbox_ap"] or data["mask_ap"]:
        axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    plot_metric(axes[1, 0], "bbox_ap50", "bbox AP50")
    plot_metric(axes[1, 0], "mask_ap50", "mask AP50")
    axes[1, 0].set_title("Validation AP50")
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_ylabel(f"AP50 ({ap_unit})")
    if data["bbox_ap50"] or data["mask_ap50"]:
        axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    plot_metric(axes[1, 1], "lr", "learning rate")
    plot_metric(axes[1, 1], "recall", "mask recall / AR")
    axes[1, 1].set_title("Learning rate / recall")
    axes[1, 1].set_xlabel(x_label)
    if data["lr"] or data["recall"]:
        axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(f"{args.metrics.name} ({backend})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
