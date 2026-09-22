"""实例分割模型后端基础抽象基类定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..schema import Artifact, BackendCapabilities, ModelManifest, Prediction, MetricReport


class BackendCapabilityError(RuntimeError):
    """当模型后端无法执行所请求的操作时引发此异常。"""


class InstanceSegmentationBackend(ABC):
    """实例分割后端抽象基类。"""
    name: str
    capabilities: BackendCapabilities

    def capability(self, operation: str) -> bool:
        """检查后端是否支持指定的操作功能。"""
        value = getattr(self.capabilities, operation, ())
        return bool(value)

    def require(self, operation: str) -> None:
        """断言后端支持指定操作，否则抛出异常。"""
        if not self.capability(operation):
            raise BackendCapabilityError(f"backend {self.name!r} does not support {operation!r}")

    @abstractmethod
    def train(self, config: Mapping[str, Any]) -> Artifact | Any:
        """使用给定配置训练模型。"""
        ...

    @abstractmethod
    def predict(self, config: Mapping[str, Any]) -> list[Prediction] | Any:
        """使用给定配置执行推理预测。"""
        ...

    @abstractmethod
    def evaluate(self, config: Mapping[str, Any]) -> MetricReport | Any:
        """评估模型在验证/测试集上的指标。"""
        ...

    @abstractmethod
    def export(self, config: Mapping[str, Any]) -> Artifact | Any:
        """将模型导出为目标部署格式（如 ONNX、TensorRT 等）。"""
        ...
