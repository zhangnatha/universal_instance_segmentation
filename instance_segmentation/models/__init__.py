"""统一模型后端注册表，提供多种实例分割模型实现。"""

from .base import BackendCapabilityError, InstanceSegmentationBackend
from .registry import get_backend, list_backends

__all__ = [
    "BackendCapabilityError", "InstanceSegmentationBackend", "get_backend", "list_backends",
]
