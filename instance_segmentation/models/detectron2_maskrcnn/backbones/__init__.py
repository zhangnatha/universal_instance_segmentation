"""Detectron2 骨干网络注册表（支持延迟导入实现）。"""

from __future__ import annotations

from typing import Any

_ALIASES = {"swin_tiny": "swin_t", "swin_small": "swin_s", "swin_base": "swin_b", "swin_large": "swin_l"}
_SPECS: dict[str, dict[str, Any]] = {
    "r50": {"family": "resnet", "builder": "build_r50_fpn_backbone", "pixel_mean": [103.530, 116.280, 123.675], "pixel_std": [1, 1, 1]},
    "r101": {"family": "resnet", "builder": "build_r101_fpn_backbone", "pixel_mean": [103.530, 116.280, 123.675], "pixel_std": [1, 1, 1]},
    "r152": {"family": "resnet", "builder": "build_r152_fpn_backbone", "pixel_mean": [103.530, 116.280, 123.675], "pixel_std": [1, 1, 1]},
    "x101": {"family": "resnext", "builder": "build_x101_fpn_backbone", "pixel_mean": [103.530, 116.280, 123.675], "pixel_std": [1, 1, 1]},
    "x152": {"family": "resnext", "builder": "build_x152_fpn_backbone", "pixel_mean": [103.530, 116.280, 123.675], "pixel_std": [1, 1, 1]},
    "res2net50": {"family": "res2net", "builder": "build_res2net50_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "regnety4gf": {"family": "regnet", "builder": "build_regnety4gf_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "swin_t": {"family": "swin", "builder": "build_swin_t_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "swin_s": {"family": "swin", "builder": "build_swin_s_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "swin_b": {"family": "swin", "builder": "build_swin_b_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "swin_l": {"family": "swin", "builder": "build_swin_l_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "convnext_tiny": {"family": "convnext", "builder": "build_convnext_tiny_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "convnext_small": {"family": "convnext", "builder": "build_convnext_small_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "convnext_base": {"family": "convnext", "builder": "build_convnext_base_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "convnext_large": {"family": "convnext", "builder": "build_convnext_large_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
    "convnext_xlarge": {"family": "convnext", "builder": "build_convnext_xlarge_fpn_backbone", "pixel_mean": [123.675, 116.28, 103.53], "pixel_std": [58.395, 57.12, 57.375]},
}


def canonical_backbone(name: str) -> str:
    """获取标准化的骨干网络名称。"""
    value = _ALIASES.get(str(name).lower().replace("-", "_"), str(name).lower().replace("-", "_"))
    if value not in _SPECS:
        raise ValueError(f"unknown backbone {name!r}; supported: {', '.join(list_supported_backbones())}")
    return value


def list_supported_backbones(*, include_aliases: bool = True) -> list[str]:
    """列出支持的骨干网络名称列表。"""
    return sorted(list(_SPECS) + (list(_ALIASES) if include_aliases else []))


def get_backbone_info(name: str) -> dict[str, Any]:
    """获取指定骨干网络的配置信息。"""
    canonical = canonical_backbone(name)
    return {**_SPECS[canonical], "canonical_id": canonical, "requested_id": str(name)}


def register_backbones() -> None:
    """导入所有已迁移的构建器，触发 Detectron2 注册。"""
    from . import common, convnext, regnet, res2net, resnet, resnext, swin  # noqa: F401


def apply_backbone_to_cfg(cfg, name: str, pretrained: bool = False) -> dict[str, Any]:
    """将骨干网络配置应用到 Detectron2 CfgNode。"""
    info = get_backbone_info(name)
    register_backbones()
    cfg.MODEL.BACKBONE.NAME = info["builder"]
    cfg.MODEL.PRETRAINED = pretrained
    cfg.MODEL.PIXEL_MEAN = list(info["pixel_mean"])
    cfg.MODEL.PIXEL_STD = list(info["pixel_std"])
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    return info
