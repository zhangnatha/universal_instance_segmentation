#!/usr/bin/env python3
"""Plot RF-DETR metrics.csv training curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fields = ["epoch", "train/loss", "train/lr", "val/ema_mAP_50_95",
              "val/ema_segm_mAP_50_95", "val/ema_mAP_50", "val/ema_segm_mAP_50",
              "val/ema_mAR"]
    values = {field: [] for field in fields}
    for row in rows:
        epoch = row.get("epoch", "")
        for field in fields:
            if field == "epoch":
                continue
            value = row.get(field, "")
            if value:
                values[field].append((float(epoch), float(value)))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="RF-DETR metrics.csv")
    parser.add_argument("--output", type=Path, required=True, help="PNG output path")
    args = parser.parse_args()

    data = read_metrics(args.metrics)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    def plot_metric(ax, field: str, label: str) -> None:
        points = data[field]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, marker=".", label=label)

    plot_metric(axes[0, 0], "train/loss", "train loss")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].grid(alpha=0.25)

    plot_metric(axes[0, 1], "val/ema_mAP_50_95", "bbox mAP50:95")
    plot_metric(axes[0, 1], "val/ema_segm_mAP_50_95", "mask mAP50:95")
    axes[0, 1].set_title("Validation AP@[.50:.95]")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    plot_metric(axes[1, 0], "val/ema_mAP_50", "bbox mAP50")
    plot_metric(axes[1, 0], "val/ema_segm_mAP_50", "mask mAP50")
    axes[1, 0].set_title("Validation AP50")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    plot_metric(axes[1, 1], "train/lr", "learning rate")
    plot_metric(axes[1, 1], "val/ema_mAR", "mask mAR")
    axes[1, 1].set_title("Learning rate / recall")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(args.metrics.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
