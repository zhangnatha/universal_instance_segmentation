"""模型后端注册中心，用于按名称查找与枚举可用的实例分割后端。"""

from __future__ import annotations

from typing import Type

from .base import InstanceSegmentationBackend


# 支持的后端类型映射字典
BACKEND_TYPES: dict[str, str] = {
    "detectron2": "instance_segmentation.models.detectron2_maskrcnn:Detectron2MaskRCNNBackend",
    "ultralytics": "instance_segmentation.models.ultralytics_yolo:UltralyticsBackend",
    "rfdetr": "instance_segmentation.models.rfdetr:RFDETRBackend",
}


def list_backends() -> dict[str, dict[str, object]]:
    """列出所有支持的模型后端及其能力配置。"""
    from .detectron2_maskrcnn import Detectron2MaskRCNNBackend
    from .ultralytics_yolo import UltralyticsBackend
    from .rfdetr import RFDETRBackend
    return {name: {"capabilities": backend().capabilities} for name, backend in (("detectron2", Detectron2MaskRCNNBackend), ("ultralytics", UltralyticsBackend), ("rfdetr", RFDETRBackend))}


def get_backend(name: str) -> InstanceSegmentationBackend:
    """根据名称获取对应的实例分割后端实例。"""
    normalized = name.lower().replace("-", "_")
    if normalized in {"maskrcnn", "detectron2_maskrcnn", "detectron2"}:
        from .detectron2_maskrcnn import Detectron2MaskRCNNBackend
        return Detectron2MaskRCNNBackend()
    if normalized in {"yolo", "ultralytics"}:
        from .ultralytics_yolo import UltralyticsBackend
        return UltralyticsBackend()
    if normalized in {"rf_detr", "rfdetr"}:
        from .rfdetr import RFDETRBackend
        return RFDETRBackend()
    raise KeyError(f"unknown backend {name!r}; choose detectron2, ultralytics, or rfdetr")
