"""ConvNeXt 系列骨干网络构建模块。"""

from __future__ import annotations

from detectron2.modeling import BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .common import BaseFeatureBackbone, build_fpn_from_bottom_up


class ConvNeXtFeatureBackbone(BaseFeatureBackbone):
    """具有四级特征的 ConvNeXt 自底向上主干网络。"""

    def __init__(self, model_name: str = "convnext_tiny", pretrained: bool = False, pretrained_path: str = ""):
        import timm
        kwargs = {"pretrained_cfg_overlay": {"file": pretrained_path}} if pretrained and pretrained_path else {}
        model = timm.create_model(
            model_name,
            features_only=True,
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


def _build_convnext_fpn_backbone(model_name: str, cfg=None, input_shape=None) -> FPN:
    """构建指定变体的 ConvNeXt 特征金字塔主干网络。"""
    pretrained = getattr(cfg.MODEL, "PRETRAINED", False) if cfg is not None else False
    bottom_up = ConvNeXtFeatureBackbone(model_name=model_name, pretrained=pretrained, pretrained_path=str(getattr(cfg.MODEL, "PRETRAINED_PATH", "")))
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


@BACKBONE_REGISTRY.register()
def build_convnext_tiny_fpn_backbone(cfg, input_shape=None):
    """构建 ConvNeXt-Tiny 特征金字塔主干网络。"""
    return _build_convnext_fpn_backbone("convnext_tiny", cfg, input_shape)


@BACKBONE_REGISTRY.register()
def build_convnext_small_fpn_backbone(cfg, input_shape=None):
    """构建 ConvNeXt-Small 特征金字塔主干网络。"""
    return _build_convnext_fpn_backbone("convnext_small", cfg, input_shape)


@BACKBONE_REGISTRY.register()
def build_convnext_base_fpn_backbone(cfg, input_shape=None):
    """构建 ConvNeXt-Base 特征金字塔主干网络。"""
    return _build_convnext_fpn_backbone("convnext_base", cfg, input_shape)


@BACKBONE_REGISTRY.register()
def build_convnext_large_fpn_backbone(cfg, input_shape=None):
    """构建 ConvNeXt-Large 特征金字塔主干网络。"""
    return _build_convnext_fpn_backbone("convnext_large", cfg, input_shape)


@BACKBONE_REGISTRY.register()
def build_convnext_xlarge_fpn_backbone(cfg, input_shape=None):
    """构建 ConvNeXt-XLarge 特征金字塔主干网络。"""
    return _build_convnext_fpn_backbone("convnext_xlarge", cfg, input_shape)
