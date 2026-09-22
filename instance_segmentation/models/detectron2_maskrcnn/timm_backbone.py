"""基于 timm 的 Detectron2 骨干网络与 FPN 适配模块。"""

from __future__ import annotations

import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool


class TimmFeatureBackbone(Backbone):
    """将步长为 4..32 的四个 timm 特征阶段暴露给 Detectron2。"""

    def __init__(self, model_name: str, *, pretrained: bool = False, pretrained_path: str = "") -> None:
        super().__init__()
        import timm

        model_kwargs = {"strict_img_size": False} if model_name.startswith("swin_") else {}
        probe = timm.create_model(
            model_name, features_only=True, pretrained=False, **model_kwargs
        )
        reductions = probe.feature_info.reduction()
        indices = tuple(reductions.index(stride) for stride in (4, 8, 16, 32))
        channels = [probe.feature_info.channels()[i] for i in indices]
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            dummy_out = probe(dummy)
            self._needs_permute = tuple(dummy_out[i].shape[-1] == channels[idx] for idx, i in enumerate(indices))
        del probe
        if pretrained and pretrained_path:
            model_kwargs["pretrained_cfg_overlay"] = {"file": pretrained_path}
        self.model = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=indices,
            **model_kwargs,
        )
        self._out_features = ("res2", "res3", "res4", "res5")
        self._out_feature_channels = dict(zip(self._out_features, channels))
        self._out_feature_strides = dict(zip(self._out_features, (4, 8, 16, 32)))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.model(images)
        features = [
            feature.permute(0, 3, 1, 2).contiguous() if needs_permute else feature
            for feature, needs_permute in zip(features, self._needs_permute)
        ]
        return dict(zip(self._out_features, features))

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self) -> int:
        return 32


@BACKBONE_REGISTRY.register()
def build_timm_fpn_backbone(cfg, input_shape):
    """构建基于 timm 特征提取器的 FPN 主干网络。"""
    bottom_up = TimmFeatureBackbone(
        cfg.MODEL.TIMM.NAME,
        pretrained=bool(cfg.MODEL.TIMM.PRETRAINED),
        pretrained_path=str(getattr(cfg.MODEL.TIMM, "PRETRAINED_PATH", "")),
    )
    return FPN(
        bottom_up=bottom_up,
        in_features=["res2", "res3", "res4", "res5"],
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
