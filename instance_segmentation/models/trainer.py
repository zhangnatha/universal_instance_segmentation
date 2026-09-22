"""
Roboflow 流程训练引擎模块。
提供实时模型训练、逐轮次指标跟踪与实时流式传输支持。
适配前端训练可视化规范。
"""

import os
import sys
import time
import math
import json
import random
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

import numpy as np
import torch


def format_time_duration(seconds: float) -> str:
    """将秒数格式化为人类易读的时长字符串：Xh Ym Zs。"""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes:02d}m {secs:02d}s"
    else:
        return f"{secs:02d}s"


class TrainingState:
    """封装实时训练状态、预计剩余时间（ETA）、训练速度和历史记录。"""
    def __init__(self):
        self.is_training = False
        self.is_completed = False
        self.has_error = False
        self.error_message = ""

        # 元数据与进度
        self.model_url = "roboflow-project/instance-segmentation-v1"
        self.checkpoint = "yolov8n-seg.pt"
        self.updated_on = datetime.now().strftime("%b %d, %Y, %I:%M %p")
        self.model_type = "Roboflow 3.0 Instance Segmentation (Fast)"
        self.total_epochs = 20
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps_all = 0
        self.start_time = 0.0
        self.elapsed_time_str = "00m 00s"
        self.time_remaining_str = "Calculating..."
        self.it_speed_str = "0.0 it/s"
        self.gpu_memory_str = "0.0 / 0.0 GB"

        # 当前顶层指标
        self.metrics = {
            "map50": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "map50_95": 0.0
        }

        # 绘制图表的轮次历史记录（性能及各项损失曲线）
        self.history = {
            "epochs": [],
            "map50": [],
            "map50_95": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "dice_loss": [],
            "class_loss": [],
            "box_location_loss": [],
            "box_loss": [],
            "object_loss": []
        }

        self.saved_model_path = ""
        self.run_dir = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_training": self.is_training,
            "is_completed": self.is_completed,
            "has_error": self.has_error,
            "error_message": self.error_message,
            "model_url": self.model_url,
            "checkpoint": self.checkpoint,
            "updated_on": self.updated_on,
            "model_type": self.model_type,
            "total_epochs": self.total_epochs,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "total_steps_all": self.total_steps_all,
            "elapsed_time": self.elapsed_time_str,
            "time_remaining": self.time_remaining_str,
            "speed": self.it_speed_str,
            "gpu_memory": self.gpu_memory_str,
            "metrics": self.metrics,
            "history": self.history,
            "saved_model_path": self.saved_model_path
        }


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = str(PROJECT_ROOT)


def _cuda_architecture_error(device: str) -> Optional[str]:
    """当所选 GPU 的计算能力未包含在当前安装的 PyTorch wheel 中时，返回可操作的错误提示。

    当安装的 wheel 能枚举出新 GPU 但未针对该计算能力编译 kernel 时，``torch.cuda.is_available()`` 仍可能返回 True。
    在 RF-DETR 构建 CUDA 张量前捕获此情况，避免用户遇到晦涩且迟发的 CUDA 错误。
    """
    requested = str(device or "auto").strip().lower()
    if requested == "cpu":
        return None

    if not torch.cuda.is_available():
        if requested == "auto":
            return None
        return (
            f"CUDA device '{device}' was requested, but PyTorch reports CUDA "
            "unavailable. Check the NVIDIA driver and the installed PyTorch "
            "CUDA wheel."
        )

    try:
        cuda_index = 0
        if requested.startswith("cuda:"):
            cuda_index = int(requested.split(":", 1)[1])
        capability = torch.cuda.get_device_capability(cuda_index)
        target_arch = f"sm_{capability[0]}{capability[1]}"
        compiled_arches = set(torch.cuda.get_arch_list())
        if target_arch not in compiled_arches:
            compiled = ", ".join(sorted(compiled_arches)) or "<none>"
            return (
                "The installed PyTorch CUDA wheel does not contain kernels for "
                f"the selected GPU (compute capability {capability[0]}.{capability[1]}, "
                f"{target_arch}). Detected PyTorch {torch.__version__} "
                f"(CUDA {torch.version.cuda}); compiled architectures: {compiled}. "
                "For an RTX 50/Blackwell GPU, install a PyTorch 2.7+ CUDA 12.8 "
                "wheel, for example: "
                "python -m pip install --upgrade --force-reinstall "
                "torch==2.7.1 torchvision==0.22.1 "
                "--index-url https://download.pytorch.org/whl/cu128"
            )
    except Exception as exc:
        return f"Unable to validate CUDA device compatibility before training: {exc}"

    return None


class Trainer:
    """在后台线程中管理模型训练生命周期。"""

    def __init__(self, output_root: Optional[str] = None):
        self.output_root = os.path.abspath(output_root) if output_root else os.path.join(BASE_DIR, "runs")
        os.makedirs(self.output_root, exist_ok=True)
        self.state = TrainingState()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def register_callback(self, cb: Callable[[Dict[str, Any]], None]):
        """注册状态更新回调函数。"""
        self._callbacks.append(cb)

    def _broadcast(self):
        """向所有已注册的回调广播当前训练状态。"""
        data = self.state.to_dict()
        for cb in self._callbacks:
            try:
                cb(data)
            except Exception:
                pass

    def start_training(
        self,
        yaml_path: str,
        model_key: str = "rf-detr",
        size_id: str = "small",
        epochs: int = 20,
        batch_size: int = 8,
        img_size: int = 432,
        device: str = "auto",
        weights: Optional[str] = None,
        resume: Optional[str] = None,
        lr: Optional[float] = None,
        eval_interval: int = 1,
        amp_dtype: str = "bf16",
        freeze_encoder: bool = False,
        cls_loss_coef: Optional[float] = None
    ) -> bool:
        """启动异步真实 PyTorch 模型训练。"""
        if self.state.is_training:
            return False

        self._stop_requested = False
        self.state = TrainingState()
        self.state.is_training = True
        self.state.total_epochs = epochs
        self.state.start_time = time.time()
        self.state.updated_on = datetime.now().strftime("%b %d, %Y, %I:%M %p")
        self.state.model_url = "roboflow-project/instance-segmentation-v1"

        # 模型架构映射
        if model_key == "rf-detr":
            # Roboflow RF-DETR: 实时 Transformer 架构 (DINOv2 主干 + DETR 解码器)
            self.state.model_type = f"Roboflow RF-DETR Detection Transformer ({size_id.capitalize()})"
            self.state.checkpoint = os.path.basename(weights) if weights else "Objects365 Pretrained Weights"
        elif model_key == "yolo26":
            self.state.model_type = f"YOLO26 Instance Segmentation ({size_id.capitalize()})"
            self.state.checkpoint = os.path.basename(weights) if weights else "yolo26-seg.pt"
        elif model_key == "yolov11":
            self.state.model_type = f"YOLOv11 Instance Segmentation ({size_id.capitalize()})"
            self.state.checkpoint = os.path.basename(weights) if weights else "yolo11-seg.pt"
        elif model_key == "sam3":
            self.state.model_type = f"SAM 3 Foundation Model ({size_id.capitalize()})"
            self.state.checkpoint = os.path.basename(weights) if weights else "sam3_base.pt"
        else:
            self.state.model_type = f"Roboflow 3.0 Instance Segmentation ({size_id.capitalize()})"
            self.state.checkpoint = os.path.basename(weights) if weights else "yolov8-seg.pt"

        self._thread = threading.Thread(
            target=self._run_training_worker,
            args=(yaml_path, model_key, size_id, epochs, batch_size, img_size, device, weights, resume, lr, eval_interval, amp_dtype, freeze_encoder, cls_loss_coef),
            daemon=True
        )
        self._thread.start()
        return True

    def stop_training(self):
        """请求停止训练 / 早停。"""
        self._stop_requested = True
        self.state.is_training = False
        self.state.is_completed = True
        self.state.time_remaining_str = "Early stopped by user"
        self._broadcast()

    def _run_training_worker(
        self,
        yaml_path: str,
        model_key: str,
        size_id: str,
        epochs: int,
        batch_size: int,
        img_size: int,
        device: str,
        weights: Optional[str] = None,
        resume: Optional[str] = None,
        lr: Optional[float] = None,
        eval_interval: int = 1,
        amp_dtype: str = "bf16",
        freeze_encoder: bool = False,
        cls_loss_coef: Optional[float] = None
    ):
        try:
            import shutil
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir = os.path.join(self.output_root, run_name)
            os.makedirs(run_dir, exist_ok=True)
            self.state.run_dir = run_dir

            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"Dataset export configuration not found at: {yaml_path}")

            dataset_root = os.path.dirname(yaml_path)

            # 分发至对应的模型引擎
            if model_key == "rf-detr":
                # Roboflow RF-DETR: 官方原生 rfdetr 框架 (DINOv2 + Transformer 解码器)
                self._train_with_rfdetr(dataset_root, size_id, epochs, batch_size, img_size, device, run_dir, weights, resume, lr, eval_interval, amp_dtype, freeze_encoder, cls_loss_coef)
            else:
                # 训练 YOLO 模型（Roboflow 3.0 / YOLOv8 / YOLOv11 / YOLO26）：Ultralytics YOLO 框架
                self._train_with_ultralytics(yaml_path, model_key, size_id, epochs, batch_size, img_size, device, run_dir, weights, resume, lr, eval_interval, amp_dtype)

            self.state.is_training = False
            self.state.is_completed = True
            self.state.time_remaining_str = "Training complete (Real Weights Exported)"
            self._broadcast()

        except Exception as e:
            self.state.is_training = False
            self.state.has_error = True
            self.state.error_message = str(e)
            self.state.time_remaining_str = f"Error: {str(e)}"
            self._broadcast()

    def _train_with_rfdetr(
        self,
        dataset_dir: str,
        size_id: str,
        epochs: int,
        batch_size: int,
        img_size: int,
        device: str,
        run_dir: str,
        weights: Optional[str] = None,
        resume: Optional[str] = None,
        lr: Optional[float] = None,
        eval_interval: int = 1,
        amp_dtype: str = "bf16",
        freeze_encoder: bool = False,
        cls_loss_coef: Optional[float] = None
    ):
        """使用 Roboflow 官方 rfdetr Transformer 框架（DINOv2 + PyTorch Lightning）执行真实训练。"""
        import os
        import gc
        import shutil
        # 1. 防止 PyTorch CUDA 显存碎片化并清理缓存
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        gc.collect()

        # 硬件级 PyTorch 矩阵加速（针对 Ampere / Ada / Blackwell 等架构 NVIDIA GPU）
        if torch.cuda.is_available() and str(device).lower() != "cpu":
            try:
                torch.set_float32_matmul_precision("high")
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        compatibility_error = _cuda_architecture_error(device)
        if compatibility_error:
            raise RuntimeError(compatibility_error)

        if str(device).lower() in ["cuda", "gpu", "cuda:0"] and not torch.cuda.is_available():
            raise RuntimeError(
                "❌ GPU Training Requested, but CUDA is unavailable! "
                "The NVIDIA driver is not loaded in current kernel (uname -r). "
                "Please run 'sudo dkms autoinstall && sudo modprobe nvidia' or reboot to enable GPU."
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            from rfdetr.variants import (
                RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge,
                RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge,
                RFDETRSegXLarge, RFDETRSeg2XLarge
            )
        except ImportError:
            raise ImportError(
                "Package 'rfdetr' is not installed. Please install it using: pip install rfdetr\n"
                "Or train with YOLO models: python cli.py train --model roboflow-3.0"
            )

        # 本实例分割流程默认使用官方 RF-DETR 实例分割模型
        variant_map = {
            "nano": RFDETRSegNano,
            "small": RFDETRSegSmall,
            "medium": RFDETRSegMedium,
            "large": RFDETRSegLarge,
            "xlarge": RFDETRSegXLarge,
            "2xlarge": RFDETRSeg2XLarge
        }
        if size_id not in variant_map:
            raise ValueError(
                f"Unsupported RF-DETR segmentation size: {size_id}. "
                f"Supported sizes: {', '.join(variant_map)}"
            )
        ModelCls = variant_map[size_id]
        # RF-DETR 在构造函数中接收预训练权重和冻结编码器参数
        init_weights = weights if weights and os.path.exists(weights) else None
        model_kwargs = {"freeze_encoder": freeze_encoder}
        if init_weights:
            model_kwargs["pretrain_weights"] = init_weights
        model = ModelCls(**model_kwargs)

        accel = "gpu" if torch.cuda.is_available() and str(device).lower() != "cpu" else "cpu"
        if str(device).lower() != "cpu" and accel == "cpu":
            print("\n⚠️  WARNING: Running on CPU because CUDA is not available. For GPU speedup, ensure nvidia driver is loaded.")
        devices = 1

        # 2. 基于 GPU 显存容量的自适应批次与 Worker 策略
        vram_gb = 0.0
        if torch.cuda.is_available() and accel == "gpu":
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                pass

        target_effective_batch = max(1, batch_size)
        if vram_gb >= 20.0:  # 高显存档位
            max_phys_batch = 16
            workers = 8
        elif vram_gb >= 10.0:  # 中等显存档位
            max_phys_batch = 4
            workers = 4
        else:  # 6GB ~ 8GB 显存档位
            max_phys_batch = 2
            workers = 2

        if accel == "gpu" and target_effective_batch > max_phys_batch:
            safe_micro_batch = max_phys_batch
            grad_accum = max(1, target_effective_batch // safe_micro_batch)
        else:
            safe_micro_batch = target_effective_batch
            grad_accum = 1

        print(f" • Hardware Adaptive Config: Physical Batch={safe_micro_batch}, Grad Accum={grad_accum} (Effective Batch={safe_micro_batch*grad_accum}), Workers={workers}, AMP={amp_dtype}, Eval Interval={eval_interval}, GPU VRAM={vram_gb:.1f} GB")

        # 后台监听 metrics.csv 实现指标实时流式推送
        stop_watcher = threading.Event()
        def _metrics_watcher():
            csv_path = os.path.join(run_dir, "metrics.csv")
            last_epoch = 0
            last_step = 0

            while not stop_watcher.is_set():
                if os.path.exists(csv_path):
                    try:
                        import csv
                        with open(csv_path, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                step_str = row.get("step", "")
                                if step_str:
                                    cur_step = int(step_str)
                                    if cur_step > last_step:
                                        last_step = cur_step
                                        self.state.current_step = cur_step

                                ep_str = row.get("epoch", "")
                                if ep_str:
                                    ep = int(ep_str) + 1
                                    if ep > last_epoch:
                                        last_epoch = ep
                                        self.state.current_epoch = ep

                                        m_map50 = float(row.get("val/mAP_50") or row.get("val/ema_mAP_50") or 0.0) * 100.0
                                        m_map50_95 = float(row.get("val/mAP_50_95") or row.get("val/ema_mAP_50_95") or 0.0) * 100.0
                                        m_prec = float(row.get("val/precision") or 0.0) * 100.0
                                        m_rec = float(row.get("val/recall") or 0.0) * 100.0
                                        m_f1 = float(row.get("val/F1") or 0.0) * 100.0

                                        d_loss = float(row.get("val/loss_giou") or row.get("train/loss_giou") or 0.25)
                                        c_loss = float(row.get("val/loss_ce") or row.get("train/loss_ce") or 0.50)
                                        b_loss = float(row.get("val/loss_bbox") or row.get("train/loss_bbox") or 0.20)

                                        self.state.metrics = {
                                            "map50": round(m_map50, 1),
                                            "precision": round(m_prec, 1),
                                            "recall": round(m_rec, 1),
                                            "f1": round(m_f1, 1),
                                            "map50_95": round(m_map50_95, 1)
                                        }
                                        self.state.history["epochs"].append(ep)
                                        self.state.history["map50"].append(round(m_map50, 1))
                                        self.state.history["map50_95"].append(round(m_map50_95, 1))
                                        self.state.history["precision"].append(round(m_prec, 1))
                                        self.state.history["recall"].append(round(m_rec, 1))
                                        self.state.history["f1"].append(round(m_f1, 1))
                                        self.state.history["dice_loss"].append(round(d_loss, 3))
                                        self.state.history["class_loss"].append(round(c_loss, 3))
                                        self.state.history["box_location_loss"].append(round(b_loss, 3))
                                        self.state.history["box_loss"].append(round(b_loss, 3))
                                        self.state.history["object_loss"].append(round(d_loss, 3))

                        # 实时计算耗时、GPU 显存占用与进度
                        now = time.time()
                        elapsed = max(0.1, now - self.state.start_time)
                        self.state.elapsed_time_str = format_time_duration(elapsed)

                        # GPU 显存使用量
                        if torch.cuda.is_available():
                            mem_used = torch.cuda.memory_reserved(0) / (1024 ** 3)
                            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                            self.state.gpu_memory_str = f"{mem_used:.1f}/{mem_total:.1f} GB"

                        # 迭代速度与预计剩余时间
                        if last_step > 0:
                            speed = last_step / elapsed
                            self.state.it_speed_str = f"{speed:.1f} it/s"
                            total_target_steps = epochs * steps_per_epoch
                            self.state.total_steps_all = total_target_steps
                            rem_steps = max(0, total_target_steps - last_step)
                            rem_sec = rem_steps / max(0.01, speed)
                            self.state.time_remaining_str = format_time_duration(rem_sec)
                        elif last_epoch > 0:
                            avg_per_epoch = elapsed / max(1, last_epoch)
                            rem_sec = max(0, int((epochs - last_epoch) * avg_per_epoch))
                            self.state.time_remaining_str = format_time_duration(rem_sec)

                        self._broadcast()
                    except Exception:
                        pass
                time.sleep(1.0)

        watcher_thread = threading.Thread(target=_metrics_watcher, daemon=True)
        watcher_thread.start()

        try:
            # 使用原生 Roboflow rfdetr 框架训练模型
            train_kwargs = {
                "dataset_dir": dataset_dir,
                "resolution": int(img_size),
                "epochs": epochs,
                "batch_size": safe_micro_batch,
                "grad_accum_steps": grad_accum,
                "num_workers": workers,
                "output_dir": run_dir,
                "accelerator": accel,
                "devices": devices,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 4,
                "amp_dtype": amp_dtype or "bf16",
                "fp16_eval": True,
                "eval_interval": eval_interval,
            }
            if resume and os.path.exists(resume):
                train_kwargs["resume"] = resume
            if lr is not None and lr > 0:
                train_kwargs["lr"] = lr
            if cls_loss_coef is not None and cls_loss_coef > 0:
                train_kwargs["cls_loss_coef"] = cls_loss_coef

            model.train(**train_kwargs)
        finally:
            stop_watcher.set()

        # 定位保存的最佳权重文件
        saved_path = os.path.join(run_dir, "checkpoint_best_total.pth")
        if not os.path.exists(saved_path):
            saved_path = os.path.join(run_dir, "checkpoint_best_regular.pth")
            if not os.path.exists(saved_path):
                for root, dirs, files in os.walk(run_dir):
                    for f in files:
                        if f.endswith(".pth") or f.endswith(".pt") or f.endswith(".ckpt"):
                            saved_path = os.path.join(root, f)
                            break

        if os.path.exists(saved_path):
            self.state.saved_model_path = saved_path
            latest_dir = os.path.join(BASE_DIR, "saved_models", "latest")
            os.makedirs(latest_dir, exist_ok=True)
            shutil.copy2(saved_path, os.path.join(latest_dir, "best.pt"))
        else:
            self.state.saved_model_path = os.path.join(run_dir, "best.pt")

    def _train_with_ultralytics(
        self,
        yaml_path: str,
        model_key: str,
        size_id: str,
        epochs: int,
        batch_size: int,
        img_size: int,
        device: str,
        run_dir: str,
        weights: Optional[str] = None,
        resume: Optional[str] = None,
        lr: Optional[float] = None,
        eval_interval: int = 1,
        amp_dtype: str = "bf16",
        freeze_encoder: bool = False,
        cls_loss_coef: Optional[float] = None
    ):
        """使用 Ultralytics YOLO 框架执行 YOLO 系列模型的真实训练。"""
        import shutil
        import torch
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Package 'ultralytics' is not installed. Please install it using: pip install ultralytics"
            )

        compatibility_error = _cuda_architecture_error(device)
        if compatibility_error:
            raise RuntimeError(compatibility_error)

        # PyTorch 硬件矩阵计算加速
        if torch.cuda.is_available() and str(device).lower() != "cpu":
            try:
                torch.set_float32_matmul_precision("high")
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        # 选择基础权重与架构
        if weights and os.path.exists(weights):
            weight_file = weights
            model = YOLO(weight_file)
        elif model_key == "yolov11":
            weight_map = {
                "nano": "yolo11n-seg.pt",
                "small": "yolo11s-seg.pt",
                "medium": "yolo11m-seg.pt",
                "large": "yolo11l-seg.pt",
                "xlarge": "yolo11x-seg.pt",
            }
            if size_id not in weight_map:
                raise ValueError(
                    f"Unsupported YOLOv11 size: {size_id}. "
                    f"Supported sizes: {', '.join(weight_map)}"
                )
            weight_file = weight_map[size_id]
            model = YOLO(weight_file)
        elif model_key == "yolo26":
            size_suffix = {
                "nano": "n",
                "small": "s",
                "medium": "m",
                "large": "l",
                "xlarge": "x",
            }
            if size_id not in size_suffix:
                raise ValueError(
                    f"Unsupported YOLO26 size: {size_id}. "
                    f"Supported sizes: {', '.join(size_suffix)}"
                )
            suffix = size_suffix[size_id]
            weight_candidates = [
                os.path.join(BASE_DIR, f"yolo26{suffix}-seg.pt"),
                os.path.join(BASE_DIR, "..", f"yolo26{suffix}-seg.pt"),
                f"yolo26{suffix}-seg.pt",
                os.path.join(BASE_DIR, f"yolo26{suffix}.pt"),
                os.path.join(BASE_DIR, "..", f"yolo26{suffix}.pt"),
                f"yolo26{suffix}.pt",
            ]
            weight_file = next((w for w in weight_candidates if os.path.exists(w)), f"yolo26{suffix}-seg.pt")
            model = YOLO(weight_file)
        else:
            # Roboflow 3.0: 兼容 YOLOv8 的实例分割
            weight_map = {
                "fast": "yolov8n-seg.pt",
                "accurate": "yolov8m-seg.pt",
                "medium": "yolov8m-seg.pt",
                "large": "yolov8l-seg.pt",
                "xlarge": "yolov8x-seg.pt"
            }
            if size_id not in weight_map:
                raise ValueError(
                    f"Unsupported Roboflow 3.0 size: {size_id}. "
                    f"Supported sizes: {', '.join(weight_map)}"
                )
            weight_file = weight_map[size_id]
            model = YOLO(weight_file)

        # 设备选择（若可用则使用 CUDA GPU）
        dev = "0" if torch.cuda.is_available() and device != "cpu" else "cpu"

        # 自定义训练回调，用于向看板进行实时流式传输
        def on_fit_epoch_end(trainer_obj):
            try:
                if self._stop_requested:
                    raise KeyboardInterrupt("Training stopped by user.")

                ep = trainer_obj.epoch + 1
                self.state.current_epoch = ep

                # 计算已耗时与预计剩余时间
                elapsed = time.time() - self.state.start_time
                avg_per_epoch = elapsed / max(1, ep)
                remaining_sec = max(0, int((epochs - ep) * avg_per_epoch))
                mins, secs = divmod(remaining_sec, 60)
                self.state.time_remaining_str = f"{mins} minutes remaining..." if mins > 0 else f"{secs} seconds remaining..."

                # 从训练器提取真实指标
                metrics_dict = trainer_obj.metrics if hasattr(trainer_obj, 'metrics') else {}
                m_map50 = float(metrics_dict.get('metrics/mAP50(M)', metrics_dict.get('metrics/mAP50(B)', metrics_dict.get('metrics/mAP50', 0.0))))
                m_map50_95 = float(metrics_dict.get('metrics/mAP50-95(M)', metrics_dict.get('metrics/mAP50-95(B)', metrics_dict.get('metrics/mAP50-95', 0.0))))
                m_prec = float(metrics_dict.get('metrics/precision(M)', metrics_dict.get('metrics/precision(B)', metrics_dict.get('metrics/precision', 0.0))))
                m_rec = float(metrics_dict.get('metrics/recall(M)', metrics_dict.get('metrics/recall(B)', metrics_dict.get('metrics/recall', 0.0))))
                f1 = (2 * m_prec * m_rec / (m_prec + m_rec)) if (m_prec + m_rec) > 0 else 0.0

                # 损失项（GIoU/Dice 损失、分类损失、边界框损失）
                d_loss, c_loss, b_loss = 0.25, 0.50, 0.20
                if hasattr(trainer_obj, 'loss_items') and trainer_obj.loss_items is not None:
                    try:
                        li = trainer_obj.loss_items
                        if hasattr(li, 'tolist'):
                            li = li.tolist()
                        if len(li) > 0: d_loss = float(li[0])
                        if len(li) > 1: c_loss = float(li[1])
                        if len(li) > 2: b_loss = float(li[2])
                    except Exception:
                        pass

                # 记录指标历史
                self.state.metrics = {
                    "map50": round(m_map50 * 100.0, 1),
                    "precision": round(m_prec * 100.0, 1),
                    "recall": round(m_rec * 100.0, 1),
                    "f1": round(f1 * 100.0, 1),
                    "map50_95": round(m_map50_95 * 100.0, 1)
                }

                self.state.history["epochs"].append(ep)
                self.state.history["map50"].append(round(m_map50 * 100.0, 1))
                self.state.history["map50_95"].append(round(m_map50_95 * 100.0, 1))
                self.state.history["precision"].append(round(m_prec * 100.0, 1))
                self.state.history["recall"].append(round(m_rec * 100.0, 1))
                self.state.history["f1"].append(round(f1 * 100.0, 1))
                self.state.history["dice_loss"].append(round(d_loss, 3))
                self.state.history["class_loss"].append(round(c_loss, 3))
                self.state.history["box_location_loss"].append(round(b_loss, 3))
                self.state.history["box_loss"].append(round(b_loss, 3))
                self.state.history["object_loss"].append(round(d_loss, 3))

                self._broadcast()
            except KeyboardInterrupt:
                raise
            except Exception:
                pass

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        # 执行真实 PyTorch 模型训练
        model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=dev,
            workers=8,
            half=True,
            project=self.output_root,
            name=os.path.basename(run_dir),
            exist_ok=True,
            verbose=False
        )

        saved_path = os.path.join(run_dir, "weights", "best.pt")
        if not os.path.exists(saved_path):
            for root, dirs, files in os.walk(run_dir):
                if "best.pt" in files:
                    saved_path = os.path.join(root, "best.pt")
                    break

        if os.path.exists(saved_path):
            self.state.saved_model_path = saved_path
            # 同步权重至 saved_models/latest/best.pt
            latest_dir = os.path.join(BASE_DIR, "saved_models", "latest")
            os.makedirs(latest_dir, exist_ok=True)
            shutil.copy2(saved_path, os.path.join(latest_dir, "best.pt"))
        else:
            self.state.saved_model_path = os.path.join(run_dir, "best.pt")
