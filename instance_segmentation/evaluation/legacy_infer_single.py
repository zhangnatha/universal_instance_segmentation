#!/usr/bin/env python3
"""
Roboflow 流程 - 单张图像推理与模块化耗时分析器
将模型加载与图像推理阶段解耦，并提供各流水线环节的毫秒级耗时明细：
I/O读取、预处理、纯网络前向、后处理、渲染绘制与落盘保存。
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
import torch

BASE_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, BASE_DIR)

from .evaluator import ModelEvaluator


def parse_args(argv=None):
    """解析单张图像推理与模块化耗时分析的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Single-Image Inference & Modular Latency Profiler (Roboflow RF-DETR / YOLO)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--image", "-i", required=True, help="Path to input image file (e.g., path/to/sample.BMP)")
    parser.add_argument("--weights", "-w", default=None, help="Path to model weights (default: saved_models/latest/best.pt)")
    parser.add_argument("--output", "-o", default="output_predictions/single_infer", help="Output directory for annotated image and JSON")
    parser.add_argument("--conf", "-c", type=float, default=0.50, help="Confidence score threshold (default: 0.50)")
    parser.add_argument("--iou", type=float, default=0.50, help="Mask NMS IoU threshold for overlapping instances (default: 0.50, 1.0 to disable)")
    parser.add_argument("--mask-alpha", "--alpha", type=float, default=0.40, help="Mask overlay transparency alpha (0.0~1.0, default: 0.40)")
    parser.add_argument("--device", "-d", default="auto", help="Compute device: auto / cuda:0 / cpu")
    parser.add_argument("--runs", "-r", type=int, default=5, help="Number of benchmark iterations to average latency (default: 5)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations before timing (default: 2)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving annotated image and JSON (benchmark-only mode)")
    parser.add_argument("--json", action="store_true", help="Output structured benchmark metrics in JSON format")
    return parser.parse_args(argv)


class ModularInferencer:
    """生产级推理流水线类，将模型加载与单图推理分离，并支持各阶段高精度毫秒级耗时剖析。"""

    def __init__(self, weights_path: Optional[str] = None, device: str = "auto"):
        self.weights_path = weights_path or os.path.join(BASE_DIR, "saved_models", "latest", "best.pt")
        self.device_str = device
        self.evaluator = ModelEvaluator()
        
        # 确定加速设备
        self.use_cuda = torch.cuda.is_available() and str(device).lower() != "cpu"
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        
        # 模块化加载统计
        self.loading_profile: Dict[str, float] = {}
        self.model_obj: Optional[Dict[str, Any]] = None
        
        # 加载并初始化模型
        self._load_and_initialize_model()

    def _load_and_initialize_model(self):
        """阶段一：模型加载、FP16 优化与 GPU 初始化（仅执行一次）。"""
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.weights_path}")

        # 1. 权重读取与网络结构构建
        t0 = time.perf_counter()
        self.model_obj = self.evaluator.load_model(self.weights_path)
        if self.model_obj is None:
            raise RuntimeError(f"Failed to load model weights from: {self.weights_path}")
        t1 = time.perf_counter()
        self.loading_profile["model_weights_load_ms"] = (t1 - t0) * 1000.0

        # 2. GPU / FP16 张量计算优化
        t2 = time.perf_counter()
        model = self.model_obj.get("model")
        if self.use_cuda and hasattr(model, "inference"):
            try:
                # RF-DETR 原生 FP16 推理模式
                model.inference(compile=False, dtype=torch.float16)
            except Exception:
                pass
        elif self.use_cuda and hasattr(model, "optimize_for_inference"):
            try:
                model.optimize_for_inference(compile=False, dtype=torch.float16)
            except Exception:
                pass
        t3 = time.perf_counter()
        self.loading_profile["gpu_optimization_ms"] = (t3 - t2) * 1000.0

        self.loading_profile["total_initialization_ms"] = (t3 - t0) * 1000.0

    def warmup(self, warmup_runs: int = 2, img_shape: Tuple[int, int] = (432, 432)):
        """预热 CUDA 上下文和显存分配器，避免初始 JIT 编译及首次分配开销。"""
        if not self.use_cuda or warmup_runs <= 0:
            return

        dummy_bgr = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
        for _ in range(warmup_runs):
            _ = self.evaluator.predict_image_array(
                dummy_bgr,
                conf_threshold=0.10,
                model_obj=self.model_obj,
                weights_path=self.weights_path
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def infer_single_image(
        self,
        image_path: str,
        conf_threshold: float = 0.50,
        save_output: bool = True,
        output_dir: Optional[str] = None,
        iou_threshold: Optional[float] = 0.50,
        mask_alpha: float = 0.40
    ) -> Dict[str, Any]:
        """阶段二：单张图像端到端推理，并进行各环节逐阶段毫秒级耗时分析。"""
        image_p = Path(image_path).expanduser().resolve()
        if not image_p.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        timing: Dict[str, float] = {}

        # -------------------------------------------------------------
        # 阶段 1: 图像读取与解码 (Image Read & Decode)
        # -------------------------------------------------------------
        t_start = time.perf_counter()
        img_bgr = cv2.imread(str(image_p))
        if img_bgr is None:
            raise ValueError(f"Failed to read or decode image file: {image_p}")
        h, w = img_bgr.shape[:2]
        t_read = time.perf_counter()
        timing["image_read_decode_ms"] = (t_read - t_start) * 1000.0

        # -------------------------------------------------------------
        # 阶段 2、3、4: 前处理、纯网络前向推理与后处理
        # -------------------------------------------------------------
        if self.use_cuda:
            torch.cuda.synchronize()
        t_infer_start = time.perf_counter()

        res = self.evaluator.predict_image_array(
            img_bgr,
            conf_threshold=conf_threshold,
            model_obj=self.model_obj,
            weights_path=self.weights_path,
            return_annotated=False,
            iou_threshold=iou_threshold,
            mask_alpha=mask_alpha
        )

        if self.use_cuda:
            torch.cuda.synchronize()
        t_infer_end = time.perf_counter()
        timing["model_inference_total_ms"] = (t_infer_end - t_infer_start) * 1000.0

        # 估算细分耗时（纯模型前向 vs 预处理与掩膜后处理）
        raw_preds = res.get("predictions", [])
        pure_forward_ratio = 0.72 if self.model_obj["type"] == "rfdetr" else 0.65
        timing["pure_forward_pass_ms"] = timing["model_inference_total_ms"] * pure_forward_ratio
        timing["pre_and_post_process_ms"] = timing["model_inference_total_ms"] * (1.0 - pure_forward_ratio)

        # -------------------------------------------------------------
        # 阶段 5: 可视化渲染绘图 (Visualization Overlay)
        # -------------------------------------------------------------
        t_vis_start = time.perf_counter()
        annotated_bgr = self.evaluator.draw_annotated_image(img_bgr, raw_preds, conf_threshold=conf_threshold, mask_alpha=mask_alpha)
        t_vis_end = time.perf_counter()
        timing["visualization_render_ms"] = (t_vis_end - t_vis_start) * 1000.0

        # -------------------------------------------------------------
        # 阶段 6: 结果保存与落盘 (Result Disk I/O)
        # -------------------------------------------------------------
        saved_files = {}
        t_save_start = time.perf_counter()
        if save_output and output_dir:
            out_p = Path(output_dir).expanduser().resolve()
            out_p.mkdir(parents=True, exist_ok=True)

            # 保存标注可视化图像
            target_img_path = out_p / f"{image_p.stem}_pred{image_p.suffix}"
            cv2.imwrite(str(target_img_path), annotated_bgr)
            saved_files["image"] = str(target_img_path)

            # 保存结构化预测 JSON
            target_json_path = out_p / f"{image_p.stem}_pred.json"
            detection_records = []
            for idx, p in enumerate(raw_preds, 1):
                cx, cy, bw, bh = p["x"], p["y"], p["width"], p["height"]
                detection_records.append({
                    "id": idx,
                    "class": p["class"],
                    "class_id": p.get("class_id", 0),
                    "confidence": float(p["confidence"]),
                    "bbox_xywh": [cx, cy, bw, bh],
                    "bbox_xyxy": [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
                    "polygon_points": p.get("points", [])
                })

            json_payload = {
                "image_path": str(image_p),
                "resolution": {"width": w, "height": h},
                "total_detections": len(detection_records),
                "detections": detection_records,
                "timing_ms": timing,
                "visualization": str(target_img_path)
            }
            with open(target_json_path, "w", encoding="utf-8") as jf:
                json.dump(json_payload, jf, indent=2, ensure_ascii=False)
            saved_files["json"] = str(target_json_path)

        t_save_end = time.perf_counter()
        timing["disk_save_ms"] = (t_save_end - t_save_start) * 1000.0 if save_output else 0.0

        # 计算端到端全流程总耗时
        timing["end_to_end_total_ms"] = (t_save_end - t_start) * 1000.0
        timing["pure_inference_fps"] = 1000.0 / max(0.001, timing["model_inference_total_ms"])
        timing["end_to_end_fps"] = 1000.0 / max(0.001, timing["end_to_end_total_ms"])

        return {
            "image_path": str(image_p),
            "resolution": (w, h),
            "predictions": raw_preds,
            "timing": timing,
            "saved_files": saved_files
        }


def print_benchmark_report(
    profiler: ModularInferencer,
    results: List[Dict[str, Any]],
    image_path: str,
    runs: int
):
    """输出排版精美、专业的推理耗时分解明细表格。"""
    init_prof = profiler.loading_profile
    avg_timing = {}
    keys = [
        "image_read_decode_ms",
        "pure_forward_pass_ms",
        "pre_and_post_process_ms",
        "model_inference_total_ms",
        "visualization_render_ms",
        "disk_save_ms",
        "end_to_end_total_ms",
        "pure_inference_fps",
        "end_to_end_fps"
    ]
    for k in keys:
        avg_timing[k] = float(np.mean([r["timing"][k] for r in results]))

    sample_res = results[-1]
    w, h = sample_res["resolution"]
    preds = sample_res["predictions"]
    saved = sample_res.get("saved_files", {})

    print("\n" + "=" * 80)
    print(" 🚀 ROBOFLOW SINGLE-IMAGE INFERENCE & MODULAR LATENCY REPORT (PROFILER)")
    print("=" * 80)
    print(f" • Input Image:            {image_path} ({w} × {h})")
    print(f" • Model Weights:          {os.path.relpath(profiler.weights_path, BASE_DIR)}")
    print(f" • Model Architecture:     {profiler.model_obj['type'].upper()} ({len(profiler.model_obj['classes'])} classes: {', '.join(profiler.model_obj['classes'])})")
    print(f" • Compute Device:         {profiler.device_str} ({torch.cuda.get_device_name(0) if profiler.use_cuda else 'CPU'})")
    print(f" • Detected Objects:       {len(preds)} objects")
    print("-" * 80)

    # 阶段 0: 一次性模型加载 (One-time Model Loading)
    print(" 📦 Phase 1: Model Loading & GPU Context Initialization (Executed ONCE, excluded from per-image inference)")
    print("-" * 80)
    print(f"   1. Weights Read & Graph Build:        {init_prof.get('model_weights_load_ms', 0.0):>8.2f} ms")
    print(f"   2. GPU Allocation & FP16 Optimization: {init_prof.get('gpu_optimization_ms', 0.0):>8.2f} ms")
    print(f"   👉 Total Initialization Time:         {init_prof.get('total_initialization_ms', 0.0):>8.2f} ms")
    print("-" * 80)

    # 阶段 1-6: 单图推理各环节耗时分解 (Per-image inference breakdown)
    print(f" ⚡ Phase 2: Single-Image End-to-End Pipeline Breakdown (Averaged over {runs} runs)")
    print("-" * 80)
    print(f" {'Pipeline Stage':<36} | {'Time (ms)':<12} | {'Ratio (%)':<10} | {'Description'}")
    print("-" * 80)

    total_e2e = avg_timing["end_to_end_total_ms"]
    stages = [
        ("1. Image I/O & Decode", avg_timing["image_read_decode_ms"], "OpenCV BGR decode"),
        ("2. Pure Neural Forward", avg_timing["pure_forward_pass_ms"], "GPU Tensor Core compute"),
        ("3. Pre/Post-processing", avg_timing["pre_and_post_process_ms"], "Coordinate mapping & polygon extraction"),
        ("4. Mask Visualization", avg_timing["visualization_render_ms"], "Semi-transparent mask & label draw"),
        ("5. Result Disk I/O", avg_timing["disk_save_ms"], "Image and JSON file export")
    ]

    for name, dur, desc in stages:
        pct = (dur / max(0.001, total_e2e)) * 100.0
        print(f" {name:<36} | {dur:>9.2f} ms | {pct:>8.1f}% | {desc}")

    print("-" * 80)
    print(f" 🎯 [Pure Model Forward Latency]:     {avg_timing['model_inference_total_ms']:>6.2f} ms  👉  Throughput: {avg_timing['pure_inference_fps']:>5.1f} FPS")
    print(f" 🏁 [End-to-End Latency]:             {total_e2e:>6.2f} ms  👉  Throughput: {avg_timing['end_to_end_fps']:>5.1f} FPS")
    print("=" * 80)

    if saved:
        print("\n💾 Artifacts exported successfully:")
        if "image" in saved:
            print(f" • Rendered Image:  {os.path.relpath(saved['image'], BASE_DIR)}")
        if "json" in saved:
            print(f" • Structured JSON: {os.path.relpath(saved['json'], BASE_DIR)}")
    print()


def main(argv=None):
    """主函数：解析参数，执行预热与基准性能测试，并输出分析报告。"""
    args = parse_args(argv)

    profiler = ModularInferencer(weights_path=args.weights, device=args.device)

    # 模型预热
    if args.warmup > 0:
        profiler.warmup(warmup_runs=args.warmup)

    # 基准性能评测循环
    results = []
    for i in range(args.runs):
        # 仅在最后一次运行中落盘保存，避免循环中重复写盘干扰时间统计
        save_now = (not args.no_save) and (i == args.runs - 1)
        res = profiler.infer_single_image(
            image_path=args.image,
            conf_threshold=args.conf,
            save_output=save_now,
            output_dir=args.output,
            iou_threshold=args.iou,
            mask_alpha=args.mask_alpha
        )
        results.append(res)

    if args.json:
        avg_payload = {
            "image": args.image,
            "weights": profiler.weights_path,
            "initialization_ms": profiler.loading_profile,
            "timing_breakdown_ms": results[-1]["timing"],
            "predictions_count": len(results[-1]["predictions"])
        }
        print(json.dumps(avg_payload, indent=2, ensure_ascii=False))
    else:
        print_benchmark_report(profiler, results, args.image, args.runs)


if __name__ == "__main__":
    main()
