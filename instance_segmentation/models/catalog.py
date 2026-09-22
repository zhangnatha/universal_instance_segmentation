"""
模型注册与架构定义模块。
支持 Roboflow RF-DETR 与 Roboflow 3.0（兼容 YOLOv8）等多种实例分割模型架构。
"""

from typing import Dict, List, Any, Optional
import os


class ModelOption:
    """表示模型架构选项。"""
    def __init__(self, key: str, name: str, description: List[str], recommended: bool, sizes: List[Dict[str, Any]]):
        self.key = key
        self.name = name
        self.description = description
        self.recommended = recommended
        self.sizes = sizes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "recommended": self.recommended,
            "sizes": self.sizes
        }


# 对应模型架构选型的模型注册表
MODEL_REGISTRY: Dict[str, ModelOption] = {
    "rf-detr": ModelOption(
        key="rf-detr",
        name="Roboflow RF-DETR (Instance Segmentation)",
        description=[
            "Official RF-DETR Segmentation Transformer (DINOv2 + Decoder + Segmentation Head)",
            "Real-time instance segmentation with pixel-accurate polygon mask prediction"
        ],
        recommended=True,
        sizes=[
            {"id": "nano", "name": "Nano", "badge": "Open Source (Apache-2.0)", "weight": "rf-detr-seg-nano.pt", "params": "3.2M", "task": "segment"},
            {"id": "small", "name": "Small", "badge": "Open Source (Apache-2.0)", "weight": "rf-detr-seg-small.pt", "params": "19.8M", "task": "segment"},
            {"id": "medium", "name": "Medium", "badge": "Open Source (Apache-2.0)", "weight": "rf-detr-seg-medium.pt", "params": "31.5M", "task": "segment"},
            {"id": "large", "name": "Large", "badge": "Open Source (Apache-2.0)", "weight": "rf-detr-seg-large.pt", "params": "32.0M", "task": "segment"},
            {"id": "xlarge", "name": "X Large", "badge": "Open Source (Apache-2.0)", "weight": "rf-detr-seg-xlarge.pt", "params": "67.0M", "task": "segment"},
            {"id": "2xlarge", "name": "2X Large", "badge": "Open Source (Apache-2.0)", "weight": "rf-detr-seg-2xlarge.pt", "params": "110.0M", "task": "segment"},
        ]
    ),
    "yolo26": ModelOption(
        key="yolo26",
        name="YOLO26",
        description=[
            "Latest from Ultralytics",
            "End-to-end NMS-free inference",
            "Faster CPU Inference"
        ],
        recommended=False,
        sizes=[
            {"id": "nano", "name": "Nano", "badge": "Open Source (AGPL-3.0)", "weight": "yolo26n-seg.pt", "params": "3.1M", "task": "segment"},
            {"id": "small", "name": "Small", "badge": "Full Access (Free Unlocked)", "weight": "yolo26s-seg.pt", "params": "11.2M", "task": "segment"},
            {"id": "medium", "name": "Medium", "badge": "Full Access (Free Unlocked)", "weight": "yolo26m-seg.pt", "params": "25.9M", "task": "segment"},
            {"id": "large", "name": "Large", "badge": "Full Access (Free Unlocked)", "weight": "yolo26l-seg.pt", "params": "43.7M", "task": "segment"},
            {"id": "xlarge", "name": "X Large", "badge": "Full Access (Free Unlocked)", "weight": "yolo26x-seg.pt", "params": "68.4M", "task": "segment"},
        ]
    ),
    "roboflow-3.0": ModelOption(
        key="roboflow-3.0",
        name="Roboflow 3.0",
        description=[
            "YOLOv8-compatible",
            "Custom performance enhancements",
            "Balance of speed & accuracy"
        ],
        recommended=False,
        sizes=[
            {"id": "fast", "name": "Fast", "badge": "Commercially Available (AGPL-3.0)", "weight": "yolov8n-seg.pt", "params": "3.4M", "task": "segment"},
            {"id": "accurate", "name": "Accurate", "badge": "Commercially Available (AGPL-3.0)", "weight": "yolov8m-seg.pt", "params": "27.3M", "task": "segment"},
            {"id": "medium", "name": "Medium", "badge": "Only available on Paid Plans (Free Unlocked)", "weight": "yolov8m-seg.pt", "params": "27.3M", "task": "segment"},
            {"id": "large", "name": "Large", "badge": "Only available on Paid Plans (Free Unlocked)", "weight": "yolov8l-seg.pt", "params": "46.0M", "task": "segment"},
            {"id": "xlarge", "name": "X Large", "badge": "Only available on Paid Plans (Free Unlocked)", "weight": "yolov8x-seg.pt", "params": "71.8M", "task": "segment"},
        ]
    ),
    "yolov11": ModelOption(
        key="yolov11",
        name="YOLOv11",
        description=[
            "Successor to YOLOv8",
            "Fast, efficient inference",
            "Optimized C3k2 & C2PSA attention"
        ],
        recommended=False,
        sizes=[
            {"id": "nano", "name": "Nano", "badge": "Commercially Available (AGPL-3.0)", "weight": "yolo11n-seg.pt", "params": "2.9M", "task": "segment"},
            {"id": "small", "name": "Small", "badge": "Full Access (Free Unlocked)", "weight": "yolo11s-seg.pt", "params": "9.4M", "task": "segment"},
            {"id": "medium", "name": "Medium", "badge": "Full Access (Free Unlocked)", "weight": "yolo11m-seg.pt", "params": "20.1M", "task": "segment"},
            {"id": "large", "name": "Large", "badge": "Full Access (Free Unlocked)", "weight": "yolo11l-seg.pt", "params": "27.6M", "task": "segment"},
            {"id": "xlarge", "name": "X Large", "badge": "Full Access (Free Unlocked)", "weight": "yolo11x-seg.pt", "params": "56.9M", "task": "segment"},
        ]
    ),
    "sam3": ModelOption(
        key="sam3",
        name="SAM 3",
        description=[
            "Segment Anything 3",
            "Experimental instance segmentation",
            "Zero-shot promptable mask generation"
        ],
        recommended=False,
        sizes=[
            {"id": "base", "name": "Base", "badge": "Open Source (Apache-2.0)", "weight": "sam3_base.pt", "params": "86M", "task": "segment"},
            {"id": "large", "name": "Large", "badge": "Open Source (Apache-2.0)", "weight": "sam3_large.pt", "params": "308M", "task": "segment"},
        ]
    )
}


def get_available_models() -> List[Dict[str, Any]]:
    """返回可用模型及其尺寸变体列表。"""
    return [m.to_dict() for m in MODEL_REGISTRY.values()]


def get_model_config(model_key: str, size_id: str) -> Dict[str, Any]:
    """获取指定模型与尺寸的具体配置。"""
    model = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["roboflow-3.0"])
    size_cfg = next((s for s in model.sizes if s["id"] == size_id), model.sizes[0])
    return {
        "model_key": model.key,
        "model_name": model.name,
        "size_id": size_cfg["id"],
        "size_name": size_cfg["name"],
        "weight": size_cfg["weight"],
        "task": size_cfg.get("task", "segment"),
        "params": size_cfg.get("params", "")
    }
