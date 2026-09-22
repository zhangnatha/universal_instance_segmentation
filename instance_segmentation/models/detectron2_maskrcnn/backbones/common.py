"""Detectron2 骨干网络通用基类与 FPN 构建函数。"""

from __future__ import annotations

from typing import Dict, Sequence
import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import Backbone, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool


class BaseFeatureBackbone(Backbone):
    """
    将特征提取器封装为 Detectron2 主干网络，并提供四个标准特征阶段。
    """

    def __init__(
        self,
        feature_module: nn.Module,
        feature_channels: Sequence[int],
        feature_strides: Sequence[int] = (4, 8, 16, 32),
        out_features: Sequence[str] = ("res2", "res3", "res4", "res5"),
    ):
        super().__init__()
        self.feature_module = feature_module
        self._out_features = tuple(out_features)
        self._feature_channels = tuple(feature_channels)
        self._feature_strides = tuple(feature_strides)
        assert len(self._out_features) == len(self._feature_channels) == len(self._feature_strides)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.feature_module(x)
        if isinstance(outputs, (list, tuple)):
            # 底层模块返回过多阶段时，仅保留与输出特征数量相同的末尾阶段
            if len(outputs) > len(self._out_features):
                outputs = outputs[-len(self._out_features):]
            return {name: feat for name, feat in zip(self._out_features, outputs)}
        elif isinstance(outputs, dict):
            return {name: outputs[name] for name in self._out_features if name in outputs}
        raise TypeError(f"Unsupported feature module output type: {type(outputs)}")

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(channels=c, stride=s)
            for name, c, s in zip(self._out_features, self._feature_channels, self._feature_strides)
        }

    @property
    def size_divisibility(self) -> int:
        return 32


def build_fpn_from_bottom_up(
    bottom_up: Backbone,
    out_channels: int = 256,
    norm: str = "",
    top_block: nn.Module | None = None,
    fuse_type: str = "sum",
    in_features: Sequence[str] = ("res2", "res3", "res4", "res5"),
) -> FPN:
    """在四阶段自底向上主干网络之上构建 Detectron2 特征金字塔。"""
    if top_block is None:
        top_block = LastLevelMaxPool()
    return FPN(
        bottom_up=bottom_up,
        in_features=list(in_features),
        out_channels=out_channels,
        norm=norm,
        top_block=top_block,
        fuse_type=fuse_type,
    )
