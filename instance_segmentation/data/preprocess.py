"""后端无关的预处理基础模块；几何变换支持标注自适应感知。

提供轻量级尺寸调整规约 ResizeSpec 以及完整的预处理流水线调用入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .labelme_core import ImageSample


@dataclass(frozen=True)
class ResizeSpec:
    """图像与标注缩放规约，包含目标宽高及缩放填充模式。"""
    width: int
    height: int
    mode: Literal["stretch", "fit_within", "fit_black", "fit_white"] = "stretch"


def resize_sample(sample: ImageSample, spec: ResizeSpec) -> ImageSample:
    """依据缩放规约等比例或拉伸缩放样本图像及对应的所有多边形标注坐标。"""
    if sample.width <= 0 or sample.height <= 0:
        raise ValueError("sample dimensions must be known")
    sx, sy = spec.width / sample.width, spec.height / sample.height
    annotations = [type(a)(a.label, [[x * sx, y * sy] for x, y in a.points], a.shape_type, a.angle_degrees) for a in sample.annotations]
    return type(sample)(sample.image_path, sample.annotation_path, spec.width, spec.height, annotations, sample.image_id, list(sample.tags))


def run_full_preprocessing(*args, **kwargs):
    """调用完整的预处理流水线实现，而非精简版本。"""
    from .preprocessing import PreprocessingPipeline
    return PreprocessingPipeline(*args, **kwargs)
