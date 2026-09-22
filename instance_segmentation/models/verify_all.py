#!/usr/bin/env python3
"""
Roboflow 多框架与多模型架构验证套件。
在所有支持的框架和变体上测试并验证完整的模型训练、检查点导出、评估与可视化推理：
  1. Roboflow RF-DETR (Nano, Small, Medium, Large, X Large, 2X Large)
  2. Roboflow 3.0 (Fast, Accurate, Medium, Large, X Large)
  3. YOLOv11 (Nano, Small, Medium, Large, X Large)
  4. YOLO26 (Nano, Small, Medium, Large, X Large)
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# 仓库根目录路径
BASE_DIR = str(Path(__file__).resolve().parents[2])
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import torch
from ..data.dataset import DatasetManager
from .trainer import Trainer
from ..evaluation.evaluator import ModelEvaluator
from .catalog import MODEL_REGISTRY, get_available_models


def verify_model(
    mgr: DatasetManager,
    yaml_path: str,
    model_key: str,
    size_id: str,
    epochs: int = 1,
    batch_size: int = 4,
    device: str = "auto"
) -> Dict[str, Any]:
    """针对单个模型变体运行端到端验证流程。"""
    model_opt = MODEL_REGISTRY.get(model_key)
    model_name = model_opt.name if model_opt else model_key
    size_cfg = next((s for s in model_opt.sizes if s["id"] == size_id), {"name": size_id, "params": "N/A"})
    
    framework_name = "Roboflow Native rfdetr" if model_key == "rf-detr" else "Ultralytics YOLO Engine"
    
    print("\n" + "=" * 78)
    print(f" 🧪 VERIFYING: {model_name} ({size_cfg['name']})")
    print("=" * 78)
    print(f" • Framework:   {framework_name}")
    print(f" • Model Key:   {model_key} | Size: {size_id}")
    print(f" • Parameters:  {size_cfg.get('params', 'N/A')}")
    print(f" • Epochs:      {epochs} (Fast Validation Mode)")
    print(f" • Batch Size:  {batch_size}")
    print(f" • Device:      {device} (GPU/CUDA: {torch.cuda.is_available()})")
    print("-" * 78)

    t0 = time.time()
    trainer = Trainer(output_root=os.path.join(BASE_DIR, "runs"))
    
    success = False
    error_msg = ""
    saved_weights = ""
    eval_metrics = {}
    preds_count = 0

    try:
        # 1. 训练模型
        trainer.start_training(
            yaml_path=yaml_path,
            model_key=model_key,
            size_id=size_id,
            epochs=epochs,
            batch_size=batch_size,
            img_size=432,
            device=device
        )

        while trainer.state.is_training:
            time.sleep(0.3)

        if trainer.state.has_error:
            raise RuntimeError(trainer.state.error_message)

        saved_weights = trainer.state.saved_model_path or ""
        if not os.path.exists(saved_weights):
            latest_best = os.path.join(BASE_DIR, "saved_models", "latest", "best.pt")
            if os.path.exists(latest_best):
                saved_weights = latest_best
            else:
                raise FileNotFoundError(f"Checkpoint was not generated at: {saved_weights}")

        # 2. 评估模型
        evaluator = ModelEvaluator(saved_dir=os.path.join(BASE_DIR, "saved_models"))
        report = evaluator.evaluate_model(
            dataset_manager=mgr,
            model_path=saved_weights,
            model_name=trainer.state.model_type,
            save_results=False
        )
        eval_metrics = report["test_report"]["metrics"]

        # 3. 测试推理
        test_dir = os.path.join(BASE_DIR, "datasets", "current_dataset", "valid")
        if not os.path.exists(test_dir):
            test_dir = os.path.join(BASE_DIR, "exports", "current_dataset", "valid")
        imgs = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))] if os.path.exists(test_dir) else []
        if imgs:
            sample_path = os.path.join(test_dir, imgs[0])
            img_bgr = cv2.imread(sample_path)
            if img_bgr is not None:
                res = evaluator.predict_image_array(img_bgr, conf_threshold=0.50, weights_path=saved_weights)
                preds = res.get("predictions", [])
                preds_count = len(preds)
                # 绘制标注图像并保存
                annotated = evaluator.draw_annotated_image(img_bgr, preds, conf_threshold=0.50)
                out_dir = os.path.join(BASE_DIR, "output_predictions", "verification")
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, f"verify_{model_key}_{size_id}.jpg"), annotated)

        success = True
    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"❌ Error during verification: {e}")

    elapsed = round(time.time() - t0, 1)
    status_str = "PASSED" if success else "FAILED"
    print(f"\nResult: [{status_str}] in {elapsed}s | Weights: {os.path.basename(saved_weights) if saved_weights else 'None'} | Test Preds: {preds_count} objects")

    return {
        "framework": framework_name,
        "model_key": model_key,
        "model_name": model_name,
        "size_id": size_id,
        "size_name": size_cfg["name"],
        "params": size_cfg.get("params", "N/A"),
        "status": status_str,
        "success": success,
        "duration_sec": elapsed,
        "saved_weights": saved_weights,
        "error_message": error_msg,
        "metrics": eval_metrics,
        "sample_preds": preds_count
    }


def resolve_dataset_dir(requested_path: Optional[str] = None) -> str:
    """根据用户输入或候选默认位置查找并解析数据集目录。"""
    if requested_path and requested_path.lower() not in ["auto", "none", ""]:
        candidates = [
            requested_path,
            os.path.join(BASE_DIR, requested_path),
            os.path.join(BASE_DIR, "..", requested_path)
        ]
        for p in candidates:
            if os.path.exists(p) and os.path.isdir(p):
                return os.path.normpath(p)
        return requested_path

    # 自动探测候选数据集目录
    default_candidates = [
        os.path.join(BASE_DIR, "data"),
        os.path.join(BASE_DIR, "dataset"),
        os.path.join(BASE_DIR, "datasets", "current_dataset")
    ]
    for c in default_candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return os.path.normpath(c)

    return "data"


def run_full_verification(
    data_dir: Optional[str] = None,
    epochs: int = 1,
    batch_size: int = 4,
    quick_mode: bool = False
):
    """验证所有支持的框架与模型架构。"""
    data_dir = resolve_dataset_dir(data_dir)

    print("\n" + "=" * 80)
    print(" 🚀 STARTING ROBOFLOW MULTI-FRAMEWORK & MULTI-MODEL VERIFICATION SUITE")
    print("=" * 80)
    print(f" • Dataset Source: {data_dir}")
    print(f" • Epochs per Run: {epochs}")
    print(f" • Quick Mode:     {quick_mode}")
    print(f" • CUDA Device:    {'Available (cuda:0)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    # 1. 准备数据集
    mgr = DatasetManager()
    mgr.load_from_directory(data_dir)
    mgr.split_dataset(train_ratio=0.70, valid_ratio=0.20, test_ratio=0.10)
    export_dir = os.path.join(BASE_DIR, "exports", "current_dataset")
    yaml_path = mgr.export_yolo_dataset(export_dir, task="segment")

    # 定义测试矩阵
    if quick_mode:
        # 快速模式：测试各框架的代表性尺寸
        test_targets = [
            ("rf-detr", "small"),
            ("roboflow-3.0", "fast"),
            ("yolov11", "nano"),
            ("yolo26", "nano")
        ]
    else:
        # 完整模式：测试所有主要变体
        test_targets = [
            # 1. Roboflow RF-DETR（Transformer 架构）
            ("rf-detr", "nano"),
            ("rf-detr", "small"),
            ("rf-detr", "medium"),
            ("rf-detr", "large"),
            ("rf-detr", "xlarge"),
            ("rf-detr", "2xlarge"),
            # 2. Roboflow 3.0 (兼容 YOLOv8)
            ("roboflow-3.0", "fast"),
            ("roboflow-3.0", "accurate"),
            ("roboflow-3.0", "medium"),
            ("roboflow-3.0", "large"),
            ("roboflow-3.0", "xlarge"),
            # 3. YOLOv11（Ultralytics 架构）
            ("yolov11", "nano"),
            ("yolov11", "small"),
            ("yolov11", "medium"),
            ("yolov11", "large"),
            ("yolov11", "xlarge"),
            # 4. YOLO26（Ultralytics 架构）
            ("yolo26", "nano"),
            ("yolo26", "small"),
            ("yolo26", "medium"),
            ("yolo26", "large"),
            ("yolo26", "xlarge"),
        ]

    results = []
    total_start = time.time()

    for idx, (m_key, s_id) in enumerate(test_targets, start=1):
        print(f"\n>>> Progress: [{idx}/{len(test_targets)}] Testing {m_key} ({s_id})...")
        res = verify_model(
            mgr=mgr,
            yaml_path=yaml_path,
            model_key=m_key,
            size_id=s_id,
            epochs=epochs,
            batch_size=batch_size,
            device="auto"
        )
        results.append(res)

    total_duration = round(time.time() - total_start, 1)
    passed_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - passed_count

    # 打印格式化验证矩阵表
    print("\n\n" + "=" * 90)
    print(" 📋 ROBOFLOW ALL FRAMEWORKS & MODELS VERIFICATION MATRIX")
    print("=" * 90)
    print(f" {'#':<3} | {'Framework':<22} | {'Model':<14} | {'Size':<10} | {'Params':<8} | {'Status':<8} | {'Time (s)':<8}")
    print("-" * 90)
    for idx, r in enumerate(results, start=1):
        status_badge = "✅ PASS" if r["success"] else "❌ FAIL"
        print(f" {idx:<3} | {r['framework']:<22} | {r['model_key']:<14} | {r['size_name']:<10} | {r['params']:<8} | {status_badge:<8} | {r['duration_sec']:>6.1f}s")
    print("=" * 90)
    print(f" 📊 Final Summary: Total Models Tested: {len(results)} | Passed: {passed_count} | Failed: {failed_count} | Total Time: {total_duration}s")
    print("=" * 90 + "\n")

    # 在导出的 JSON 中将 saved_weights 转换为相对路径
    serializable_results = []
    for r in results:
        r_copy = dict(r)
        if r_copy.get("saved_weights") and os.path.isabs(r_copy["saved_weights"]):
            r_copy["saved_weights"] = os.path.relpath(r_copy["saved_weights"], BASE_DIR)
        serializable_results.append(r_copy)

    # 保存验证报告 JSON 与 Markdown
    saved_dir = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(saved_dir, exist_ok=True)
    report_json_path = os.path.join(saved_dir, "model_verification_matrix.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tested": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "duration_sec": total_duration,
            "results": serializable_results
        }, f, indent=2, ensure_ascii=False)

    report_md_path = os.path.join(saved_dir, "model_verification_matrix.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Roboflow Multi-Framework & Model Architecture Verification Matrix\n\n")
        f.write(f"**Verification Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Total Tested:** {len(results)} | **Passed:** {passed_count} | **Failed:** {failed_count} | **Duration:** {total_duration}s\n\n")
        f.write("| # | Framework | Model Key | Size | Params | Status | Duration | Checkpoint |\n")
        f.write("| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for idx, r in enumerate(results, start=1):
            st = "✅ PASSED" if r["success"] else "❌ FAILED"
            f.write(f"| {idx} | {r['framework']} | `{r['model_key']}` | {r['size_name']} | {r['params']} | {st} | {r['duration_sec']}s | `{os.path.basename(r['saved_weights'])}` |\n")

    print(f"📄 Verification Matrix Report exported to:")
    print(f" 👉 {report_json_path}")
    print(f" 👉 {report_md_path}\n")

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Roboflow multi-framework and multi-model verification pipeline (supports RF-DETR, Roboflow 3.0, YOLOv11, YOLO26)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data", default=None, help="Source dataset directory for verification (default: auto-detect)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for fast validation training per model (default: 1)")
    parser.add_argument("--batch", type=int, default=4, help="Training batch size (default: 4)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: test only representative models from each framework")
    args = parser.parse_args(argv)

    run_full_verification(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        quick_mode=args.quick
    )


if __name__ == "__main__":
    main()
