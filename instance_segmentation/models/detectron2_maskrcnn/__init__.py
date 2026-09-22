"""Detectron2 Mask R-CNN 标准后端。

迁移的算法模块保留了角度预测头、小目标 Anchor 优化、聚焦裁剪（Focus Crop）、
Copy-Paste 数据增强、难负样本挖掘、自动混合精度（AMP）、早停机制、QDQ INT8 导出
以及 C++ ONNX Runtime 部署能力。重量级依赖项采用延迟加载。
"""

from __future__ import annotations

from typing import Any, Mapping

# 本仓库支持的 Detectron2 版本仍引用了已被移除的 ``np.bool`` 别名。
# 在不绑定过时 NumPy 版本的前提下保持与 NumPy 1.26+ 的兼容性。
try:
    import numpy as _np
    if "bool" not in _np.__dict__:
        _np.bool = bool  # type: ignore[attr-defined]
except Exception:
    pass

from ..base import InstanceSegmentationBackend
from ...schema import Artifact, BackendCapabilities, ModelManifest
from .backbones import canonical_backbone, get_backbone_info, list_supported_backbones


class Detectron2MaskRCNNBackend(InstanceSegmentationBackend):
    """Detectron2 Mask R-CNN 实例分割后端。"""
    name = "detectron2"
    capabilities = BackendCapabilities(train=True, predict=True, evaluate=True, export=("onnx", "onnx-int8-qdq"), preprocess=True)

    @staticmethod
    def _manifest(config: Mapping[str, Any], checkpoint: str | None = None) -> ModelManifest:
        """根据配置和权重路径生成模型清单对象。"""
        requested = str(config.get("backbone", "r101"))
        canonical = canonical_backbone(requested)
        classes = tuple(str(x) for x in config.get("classes", ()))
        metric = str(config.get("best_metric", "segm/AP"))
        return ModelManifest("detectron2", "maskrcnn", classes, canonical, tuple(config["input_size"]) if config.get("input_size") else None, checkpoint or config.get("weights"), metric, config.get("best_metric_value"), dict(config))

    @staticmethod
    def _argv(config: Mapping[str, Any]) -> list[str]:
        """将字典配置转换为命令行参数列表。"""
        args: list[str] = []
        for key, value in config.items():
            if value is None or key in {"backend", "input_size", "best_metric", "best_metric_value", "format"}:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value: args.append(flag)
            elif isinstance(value, Mapping):
                args.extend([flag, *[f"{name}={raw}" for name, raw in value.items()]])
            elif isinstance(value, (list, tuple)):
                args.extend([flag, *map(str, value)])
            else:
                args.extend([flag, str(value)])
        return args

    def train(self, config: Mapping[str, Any]):
        """执行模型训练流程。"""
        from .train import main
        args = dict(config); args.setdefault("backbone", "r101")
        main(self._argv(args))
        return self._manifest(args)

    def predict(self, config: Mapping[str, Any]):
        """执行模型推理预测。"""
        from .infer import main
        args = dict(config); args.setdefault("backbone", "r101")
        main(self._argv(args))
        return []

    def evaluate(self, config: Mapping[str, Any]):
        """执行模型评估。"""
        from .evaluate import main
        main(self._argv(config))
        return None

    def export(self, config: Mapping[str, Any]):
        """导出模型为 ONNX 或量化格式。"""
        from .export import main
        args = dict(config); args.setdefault("opset", 16)
        if args.get("format") == "onnx-int8-qdq": args["quantization"] = "int8"
        main(self._argv(args))
        output = str(args.get("output", "model.onnx"))
        return Artifact(output, str(args.get("format", "onnx")), self._manifest(args, output))


__all__ = ["Detectron2MaskRCNNBackend", "canonical_backbone", "get_backbone_info", "list_supported_backbones"]
