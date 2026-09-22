"""RegNet 系列骨干网络构建模块。"""

from __future__ import annotations

from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .common import BaseFeatureBackbone, build_fpn_from_bottom_up


class RegNetYFeatureBackbone(BaseFeatureBackbone):
    """具有四级特征的 RegNetY 自底向上主干网络。"""

    def __init__(self, model_name: str = "regnety_040", pretrained: bool = False, pretrained_path: str = ""):
        import timm
        kwargs = {"pretrained_cfg_overlay": {"file": pretrained_path}} if pretrained and pretrained_path else {}
        model = timm.create_model(
            model_name,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            pretrained=pretrained,
            **kwargs,
        )
        channels = model.feature_info.channels()
        strides = model.feature_info.reduction()
        super().__init__(
            feature_module=model,
            feature_channels=channels,
            feature_strides=strides,
            out_features=("res2", "res3", "res4", "res5"),
        )


@BACKBONE_REGISTRY.register()
def build_regnety4gf_fpn_backbone(cfg, input_shape=None):
    """构建 RegNetY-4.0GF 特征金字塔主干网络。"""
    pretrained = getattr(cfg.MODEL, "PRETRAINED", False)
    bottom_up = RegNetYFeatureBackbone(model_name="regnety_040", pretrained=pretrained, pretrained_path=str(getattr(cfg.MODEL, "PRETRAINED_PATH", "")))
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS if cfg is not None else 256
    norm = cfg.MODEL.FPN.NORM if cfg is not None else ""
    fuse_type = cfg.MODEL.FPN.FUSE_TYPE if cfg is not None else "sum"

    return build_fpn_from_bottom_up(
        bottom_up=bottom_up,
        out_channels=out_channels,
        norm=norm,
        top_block=LastLevelMaxPool(),
        fuse_type=fuse_type,
    )
