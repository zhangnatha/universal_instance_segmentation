"""ResNet 系列骨干网络构建模块。"""

from __future__ import annotations

from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .common import build_fpn_from_bottom_up


def _build_resnet_fpn_backbone(depth: int, cfg=None, input_shape=None) -> FPN:
    """
    构建深度为 50、101 或 152 的 ResNet 特征金字塔主干网络。
    """
    if cfg is None:
        from detectron2.config import get_cfg
        from detectron2.model_zoo import get_config_file
        cfg = get_cfg()
        cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # 克隆配置并设置网络深度
    cfg = cfg.clone()
    cfg.defrost()
    cfg.MODEL.RESNETS.DEPTH = depth
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RESNETS.NUM_GROUPS = 1
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256

    in_shape = input_shape if input_shape is not None else ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    bottom_up = build_resnet_backbone(cfg, in_shape)

    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    norm = cfg.MODEL.FPN.NORM
    fuse_type = cfg.MODEL.FPN.FUSE_TYPE

    return build_fpn_from_bottom_up(
        bottom_up=bottom_up,
        out_channels=out_channels,
        norm=norm,
        top_block=LastLevelMaxPool(),
        fuse_type=fuse_type,
    )


@BACKBONE_REGISTRY.register()
def build_r50_fpn_backbone(cfg, input_shape=None):
    """构建 ResNet-50 特征金字塔主干网络。"""
    return _build_resnet_fpn_backbone(depth=50, cfg=cfg, input_shape=input_shape)


@BACKBONE_REGISTRY.register()
def build_r101_fpn_backbone(cfg, input_shape=None):
    """构建 ResNet-101 特征金字塔主干网络。"""
    return _build_resnet_fpn_backbone(depth=101, cfg=cfg, input_shape=input_shape)


@BACKBONE_REGISTRY.register()
def build_r152_fpn_backbone(cfg, input_shape=None):
    """构建 ResNet-152 特征金字塔主干网络。"""
    return _build_resnet_fpn_backbone(depth=152, cfg=cfg, input_shape=input_shape)
