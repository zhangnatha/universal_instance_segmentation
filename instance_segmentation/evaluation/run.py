#!/usr/bin/env python3
"""
Roboflow 流程 - 评估运行脚本
与 instance-segmentation-repro 评估格式匹配。
对比真实标注 LabelMe JSON 与模型预测 JSON，
并输出 analysis_report.csv、summary.csv、per_image.csv、matches.csv、
confusion_matrix.csv 以及 analysis_report.html。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from .compare_json import main as compare_main


if __name__ == "__main__":
    compare_main()
