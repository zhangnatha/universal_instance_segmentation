"""Detectron2 Mask R-CNN 训练器扩展，支持小目标 Anchor 迁移、独立学习率、早停与验证评估。"""

from __future__ import annotations

import json
import inspect
import sys
from math import ceil
from pathlib import Path

import torch

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping

from .mapper import AngleDatasetMapper


def _angle_lr_factor(name, cfg):
    return float(cfg.ANGLE_LR_FACTOR) if name.startswith("roi_heads.angle_head.") else 1.0


def _apply_named_lr_factors(params, model, cfg):
    """为 Detectron2 0.6 应用基于参数完整名称的学习率缩放因子。

    Detectron2 0.6 没有 ``lr_factor_func`` 参数，且其 ``overrides`` 映射仅包含局部名称。
    通过按参数拆分参数组，使方向分类头的缩放因子不会泄漏至其他无关参数，同时保留权重衰减和偏置设置。
    """
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    adapted = []
    for group in params:
        base_lr = group.get("lr")
        for parameter in group["params"]:
            name = names.get(id(parameter), "")
            factor = _angle_lr_factor(name, cfg)
            new_group = {key: value for key, value in group.items() if key != "params"}
            new_group["params"] = [parameter]
            if base_lr is not None:
                new_group["lr"] = float(base_lr) * factor
            adapted.append(new_group)
    return adapted


def _build_optimizer_params(model, cfg, builder=None):
    """在不同 Detectron2 API 版本中构建具有相同语义的优化器参数组。"""
    builder = builder or get_default_optimizer_params
    kwargs = {
        "base_lr": cfg.SOLVER.BASE_LR,
        "weight_decay_norm": cfg.SOLVER.WEIGHT_DECAY_NORM,
        "bias_lr_factor": cfg.SOLVER.BIAS_LR_FACTOR,
        "weight_decay_bias": cfg.SOLVER.WEIGHT_DECAY_BIAS,
    }
    supports_lr_factor = "lr_factor_func" in inspect.signature(builder).parameters
    if supports_lr_factor:
        kwargs["lr_factor_func"] = lambda name: _angle_lr_factor(name, cfg)
    params = builder(model, **kwargs)
    return params if supports_lr_factor else _apply_named_lr_factors(params, model, cfg)


def _filter_loader_kwargs(
    builder, *, mapper, num_workers, use_cuda, pin_memory=None,
    persistent_workers=True, prefetch_factor=2,
):
    """仅在当前安装的 Detectron2 支持时传递数据加载器调优参数。"""
    values = {
        "mapper": mapper,
        "pin_memory": use_cuda if pin_memory is None else bool(pin_memory),
        "persistent_workers": bool(persistent_workers and num_workers > 0),
        # 当 num_workers == 0 时，PyTorch DataLoader 不接受 prefetch_factor 参数。
        "prefetch_factor": int(prefetch_factor),
    }
    parameters = inspect.signature(builder).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    return {
        name: value for name, value in values.items()
        if (name in parameters or accepts_kwargs)
        and not (name in {"persistent_workers", "prefetch_factor"} and num_workers == 0)
    }


def adapt_rpn_head_for_anchor_count(checkpoint, model):
    """在实验增加 Anchor 尺度时扩展基线 RPN 预测头。

    标准 R101-FPN 配置在每个位置有 3 个 Anchor，而小目标配置有 6 个 Anchor。
    直接从现有检查点加载会导致跳过目标性与回归头。复制已学得的卷积核可保留已有检测能力并为新增 Anchor 提供合理初始化。
    """
    checkpoint_model = checkpoint.get("model")
    if checkpoint_model is None:
        return []

    model_state = model.state_dict()
    # DistributedDataParallel 可能通过 module 暴露包装后的 state dict，而检查点中的键没有前缀。
    if not any(key in model_state for key in (
        "proposal_generator.rpn_head.objectness_logits.weight",
        "proposal_generator.rpn_head.anchor_deltas.weight",
    )) and hasattr(model, "module"):
        model_state = model.module.state_dict()
    adapted = []
    for key in (
        "proposal_generator.rpn_head.objectness_logits.weight",
        "proposal_generator.rpn_head.objectness_logits.bias",
        "proposal_generator.rpn_head.anchor_deltas.weight",
        "proposal_generator.rpn_head.anchor_deltas.bias",
    ):
        source = checkpoint_model.get(key)
        target = model_state.get(key)
        if source is None or target is None or tuple(source.shape) == tuple(target.shape):
            continue

        if key.endswith("objectness_logits.weight") or key.endswith("objectness_logits.bias"):
            source_count = source.shape[0]
            target_count = target.shape[0]
            if source_count <= 0 or target_count < source_count:
                continue
            repeats = ceil(target_count / source_count)
            expanded = torch.cat([source] * repeats, dim=0)[:target_count]
        else:
            source_count = source.shape[0] // 4
            target_count = target.shape[0] // 4
            if source.shape[0] != source_count * 4 or target.shape[0] != target_count * 4:
                continue
            if source_count <= 0 or target_count < source_count:
                continue
            repeats = ceil(target_count / source_count)
            source = source.reshape(source_count, 4, *source.shape[1:])
            expanded = torch.cat([source] * repeats, dim=0)[:target_count]
            expanded = expanded.reshape(target.shape)

        checkpoint_model[key] = expanded.contiguous()
        adapted.append(key)
    return adapted


class TrainingProgressHook(HookBase):
    """仅在终端显示训练进度，不把动态控制字符写入日志。"""

    def before_train(self):
        from tqdm import tqdm

        self.bar = tqdm(
            total=self.trainer.max_iter,
            initial=self.trainer.start_iter,
            desc="Training",
            unit="iter",
            dynamic_ncols=True,
            file=sys.__stderr__,
        )

    def after_step(self):
        if (self.trainer.iter + 1) % 20 == 0:
            total_loss = self.trainer.storage.history("total_loss").median(20)
            self.bar.set_postfix(total_loss=f"{total_loss:.4g}", refresh=False)
        self.bar.update(1)

    def after_train(self):
        self.bar.close()


class BestValidationHook(HookBase):
    """保存验证指标最佳模型，并可在验证指标长期不提升时早停。"""

    def __init__(self, eval_period: int, metric_name: str, patience: int, save_best: bool):
        self.eval_period = eval_period
        self.metric_name = metric_name
        self.patience = patience
        self.save_best = save_best
        self.best_value = float("-inf")
        self.bad_evaluations = 0

    def restore_from_metrics(self, output_dir):
        """从 metrics.json 恢复历史最佳指标和停滞计数，以便断点续训保留早停语义。"""
        metrics_path = Path(output_dir) / "metrics.json"
        if not metrics_path.is_file():
            return
        evaluations = []
        try:
            with metrics_path.open("r", encoding="utf-8") as file:
                for line in file:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if self.metric_name not in record or "iteration" not in record:
                        continue
                    try:
                        value = float(record[self.metric_name])
                        iteration = int(record["iteration"])
                    except (TypeError, ValueError):
                        continue
                    evaluations.append((iteration, value))
        except OSError:
            return
        if not evaluations:
            return

        best_value = float("-inf")
        bad_evaluations = 0
        for _, value in sorted(evaluations):
            if value == value and value > best_value:  # 有限数值或正无穷
                best_value = value
                bad_evaluations = 0
            else:
                bad_evaluations += 1
            self.best_value = best_value
            self.bad_evaluations = bad_evaluations

    def after_step(self):
        iteration = self.trainer.iter + 1
        is_eval_step = iteration % self.eval_period == 0 or iteration == self.trainer.max_iter
        if not is_eval_step:
            return

        latest = self.trainer.storage.latest().get(self.metric_name)
        if latest is None:
            print(
                f"WARNING: validation metric {self.metric_name!r} was not produced; "
                "best-checkpoint/early-stop logic is skipped for this evaluation.",
                file=sys.stderr,
            )
            return
        value = float(latest[0] if isinstance(latest, tuple) else latest)
        if value != value:  # 非有效数值（NaN）
            self.bad_evaluations += 1
            improved = False
        else:
            improved = value > self.best_value
            if improved:
                self.best_value = value
                self.bad_evaluations = 0
                if self.save_best:
                    self.trainer.checkpointer.save("model_best")
                    print(
                        f"Best validation {self.metric_name}={value:.6g} at iter {iteration}; "
                        "saved model_best.pth"
                    )
            else:
                self.bad_evaluations += 1

        if self.patience > 0 and not improved and self.bad_evaluations >= self.patience:
            print(
                f"Early stopping at iter {iteration}: validation {self.metric_name}="
                f"{value:.6g} did not improve for {self.bad_evaluations} evaluation(s)."
            )
            # Detectron2 0.6 在调用 TrainerBase.train() 时按值传递 max_iter，因此在此修改 max_iter 无法终止循环。
            # 将共享的迭代计数器直接跳至最终循环迭代，TrainerBase 在执行完钩子后自增一次并正常退出。
            self.trainer.iter = self.trainer.max_iter - 1


class ReproTrainer(DefaultTrainer):
    """支持增强配置、角度头以及最佳检查点保存的 Detectron2 训练器。"""
    augment = True
    progress = True
    # 验证功能对正常训练流程可用；即使用户直接调用也保留参考训练器的保存最佳模型行为。
    save_best = True
    early_stop_patience = 0
    # 跨不同 IoU 阈值选择 COCO mask AP；仅使用 AP50 可能会选出粗召回率高但定位质量差的检查点。
    early_stop_metric = "segm/AP"
    optimizer_name = "sgd"
    adam_betas = (0.9, 0.999)
    adam_eps = 1e-8

    def resume_or_load(self, resume=True):
        """加载迁移权重，并在 Anchor 发生变化时保持已学得的 RPN 权重。"""
        is_new_run = not (resume and self.checkpointer.has_checkpoint())
        weights = str(self.cfg.MODEL.WEIGHTS)
        is_local_project_checkpoint = bool(weights) and "://" not in weights
        if (
            self.cfg.SMALL_OBJECT_ANCHORS
            and is_new_run
            and is_local_project_checkpoint
        ):
            # DetectionCheckpointer._load_file 通常由其公开的 load() 方法调用。
            # 我们需要在文件加载和模型加载之间修改内存中的状态字典，因此使用其底层的 PyTorch 加载器并在此规范化单状态字典情形。
            raw_loader = getattr(self.checkpointer, "_torch_load", None)
            checkpoint = (
                raw_loader(self.cfg.MODEL.WEIGHTS)
                if raw_loader is not None
                else torch.load(self.cfg.MODEL.WEIGHTS, map_location="cpu", weights_only=False)
            )
            if "model" not in checkpoint:
                checkpoint = {"model": checkpoint}
            adapted = adapt_rpn_head_for_anchor_count(checkpoint, self.model)
            incompatible = self.checkpointer._load_model(checkpoint)
            self.checkpointer._log_incompatible_keys(incompatible)
            if adapted:
                print(
                    "Expanded baseline RPN weights for small-object anchors: "
                    + ", ".join(adapted)
                )
            return
        super().resume_or_load(resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            for hook in self._hooks:
                if isinstance(hook, BestValidationHook):
                    hook.restore_from_metrics(self.cfg.OUTPUT_DIR)
                    break

    def build_hooks(self):
        """构建训练 Hook 列表，包括早停、最佳权重保存和终端进度条。"""
        hooks = super().build_hooks()
        if self.cfg.DATASETS.TEST and (self.save_best or self.early_stop_patience > 0):
            eval_hook_index = next(
                (index for index, hook in enumerate(hooks) if hook.__class__.__name__ == "EvalHook"),
                None,
            )
            if eval_hook_index is not None:
                hooks.insert(
                    eval_hook_index + 1,
                    BestValidationHook(
                        eval_period=int(self.cfg.TEST.EVAL_PERIOD),
                        metric_name=self.early_stop_metric,
                        patience=self.early_stop_patience,
                        save_best=self.save_best,
                    ),
                )
        if self.progress:
            hooks.insert(0, TrainingProgressHook())
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        """构建训练数据加载器，应用角度标签映射与数据增强。"""
        num_workers = int(cfg.DATALOADER.NUM_WORKERS)
        use_cuda = str(cfg.MODEL.DEVICE).startswith("cuda")
        mapper = AngleDatasetMapper(cfg, is_train=True, augment=cls.augment)
        loader_kwargs = _filter_loader_kwargs(
            build_detection_train_loader,
            mapper=mapper,
            num_workers=num_workers,
            use_cuda=use_cuda,
            pin_memory=getattr(cfg.DATALOADER, "PIN_MEMORY", None),
            persistent_workers=getattr(cfg.DATALOADER, "PERSISTENT_WORKERS", True),
            prefetch_factor=getattr(cfg.DATALOADER, "PREFETCH_FACTOR", 2),
        )
        return build_detection_train_loader(cfg, **loader_kwargs)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """构建验证/测试数据加载器。"""
        data_loader = super().build_test_loader(cfg, dataset_name)
        if not cls.progress:
            return data_loader
        from tqdm import tqdm

        return tqdm(
            data_loader,
            total=len(data_loader),
            desc="Validation",
            unit="image",
            dynamic_ncols=True,
            file=sys.__stderr__,
        )

    @classmethod
    def build_optimizer(cls, cfg, model):
        """角度分类头使用独立小学习率，并为所有参数启用梯度范数裁剪。"""
        params = _build_optimizer_params(model, cfg)
        optimizer_class = torch.optim.AdamW if cls.optimizer_name == "adamw" else torch.optim.SGD
        optimizer_type = maybe_add_gradient_clipping(cfg, optimizer_class)
        if cls.optimizer_name == "adamw":
            return optimizer_type(
                params,
                lr=cfg.SOLVER.BASE_LR,
                betas=cls.adam_betas,
                eps=cls.adam_eps,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        return optimizer_type(
            params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """构建 COCO 格式评估器。"""
        return COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR)
