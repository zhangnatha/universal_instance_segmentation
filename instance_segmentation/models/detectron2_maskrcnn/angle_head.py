"""Detectron2 Mask R-CNN 旋转/方向预测头（Angle Head）及 ROIHeads 实现。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads, select_foreground_proposals


class AngleHead(nn.Module):
    """共享的 ``bins`` 类 ROI 方向分类器。

    原仓库使用 ``num_classes * 18`` 输出并选取类别切片。
    本后端保留统一的共享方向空间：方向监督已由 ``ANGLE_CLASS_IDS`` 限制，
    共享预测头保持了既有目标检查点的形状与语义。
    """

    def __init__(self, channels: int, resolution: int, bins: int):
        super().__init__()
        self.bins = bins
        self.predictor = nn.Linear(channels * resolution * resolution, bins)
        # 角度头直接接收 256x7x7 的 FPN 特征；从零训练时 0.01 会让初始
        # 72 类 logits 过宽、角度损失明显压过其余分支。较小初始化使初始
        # 交叉熵接近 log(72)，加载归档 checkpoint 时该值会被权重覆盖。
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.constant_(self.predictor.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.flatten(features, start_dim=1))


@ROI_HEADS_REGISTRY.register()
class AngleROIHeads(StandardROIHeads):
    """Mask R-CNN ROIHeads 加可配置周期的方向分类分支。"""

    @configurable
    def __init__(
        self, *, angle_bins: int, angle_period: float, angle_loss_weight: float,
        angle_class_ids=(), **kwargs
    ):
        super().__init__(**kwargs)
        resolution = self.box_pooler.output_size[0]
        channels = self.box_head.fc1.in_features // (resolution * resolution)
        self.angle_head = AngleHead(channels, resolution, angle_bins)
        self.angle_bins = angle_bins
        self.angle_period = angle_period
        self.angle_loss_weight = angle_loss_weight
        # 在检查点中持久化语义配置。仅凭权重形状无法区分
        # 36-bin/180度模型与 36-bin/360度模型，若隐式选错会导致推理异常。
        self.register_buffer(
            "angle_period_meta", torch.tensor([angle_period], dtype=torch.float32)
        )
        class_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        class_ids = tuple(int(v) for v in angle_class_ids)
        if not class_ids:
            class_ids = tuple(range(self.num_classes))
        class_mask[list(class_ids)] = True
        self.register_buffer("angle_class_mask_meta", class_mask)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update(
            angle_bins=cfg.ANGLE_BINS,
            angle_period=cfg.ANGLE_PERIOD,
            angle_loss_weight=cfg.ANGLE_LOSS_WEIGHT,
            angle_class_ids=cfg.ANGLE_CLASS_IDS,
        )
        return ret

    def _forward_angle(self, features, instances):
        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            boxes = [x.proposal_boxes for x in instances]
        else:
            boxes = [x.pred_boxes for x in instances]
        counts = [len(x) for x in instances]
        if sum(counts) == 0:
            if self.training:
                return {"loss_angle": sum(v.sum() for v in features.values()) * 0.0}
            return instances
        feature_list = [features[f] for f in self.box_in_features]
        pooled = self.box_pooler(feature_list, boxes)
        # 角度标签可能来自手工方向或 mask 主轴，噪声显著高于检测标签。
        # 角度分支只更新自身，避免其梯度在 warmup 后反向污染共享 FPN/RPN。
        if self.training:
            pooled = pooled.detach()
        logits = self.angle_head(pooled)
        if self.training:
            targets = torch.cat([x.gt_angle_bins for x in instances], dim=0)
            valid = targets >= 0
            if not valid.any():
                # 全忽略批次执行 cross_entropy 会返回 NaN。保持计算图连接的零值梯度，
                # 以便在所选类别没有手工标注角度时优化器和缩放器仍能正常工作。
                return {"loss_angle": logits.sum() * 0.0}
            return {
                "loss_angle": F.cross_entropy(logits[valid], targets[valid])
                * self.angle_loss_weight
            }
        bins = logits.argmax(dim=1)
        confidence = logits.softmax(dim=1).amax(dim=1)
        for instance, bin_values, conf_values in zip(
            instances, bins.split(counts), confidence.split(counts)
        ):
            instance.pred_angle_bins = bin_values
            instance.pred_angles = bin_values.float() * (self.angle_period / self.angle_bins)
            instance.angle_scores = conf_values
        return instances

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            assert targets is not None
            proposals = self.label_and_sample_proposals(proposals, targets)
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            losses.update(self._forward_angle(features, proposals))
            return proposals, losses
        pred_instances = self._forward_box(features, proposals)
        pred_instances = super().forward_with_given_boxes(features, pred_instances)
        pred_instances = self._forward_angle(features, pred_instances)
        return pred_instances, {}
