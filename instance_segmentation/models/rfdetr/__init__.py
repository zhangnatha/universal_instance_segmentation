"""RF-DETR 实例分割模型后端封装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..base import InstanceSegmentationBackend, BackendCapabilityError
from ...schema import Artifact, BackendCapabilities, Prediction
from ...paths import PRETRAINED_ROOT, ensure_pretrained
from ..thresholds import class_thresholds, filter_predictions


VARIANTS = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall", "medium": "RFDETRSegMedium", "large": "RFDETRSegLarge", "xlarge": "RFDETRSegXLarge", "2xlarge": "RFDETRSeg2XLarge"}
PRETRAINED_ASSETS = {
    "nano": ("rf-detr-seg-nano.pt", "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth"),
    "small": ("rf-detr-seg-small.pt", "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth"),
    "medium": ("rf-detr-seg-medium.pt", "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth"),
    "large": ("rf-detr-seg-large.pt", "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth"),
    "xlarge": ("rf-detr-seg-xlarge.pt", "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth"),
    "2xlarge": ("rf-detr-seg-xxlarge.pt", "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth"),
}


class RFDETRBackend(InstanceSegmentationBackend):
    """基于 Roboflow RF-DETR 的实例分割后端实现。"""
    name = "rfdetr"
    capabilities = BackendCapabilities(train=True, predict=True, evaluate=True, export=("onnx",), preprocess=True)

    def _model(self, config: Mapping[str, Any]):
        """加载或初始化 RF-DETR 模型实例。"""
        try:
            from rfdetr import variants
        except ImportError as error:
            raise RuntimeError("RF-DETR backend requires the optional 'rfdetr' extra") from error
        size = str(config.get("size", "small")); name = VARIANTS.get(size)
        if name is None:
            raise ValueError(f"unsupported RF-DETR segmentation size: {size}")
        kwargs = {}
        import os
        os.environ.setdefault("RF_HOME", str(PRETRAINED_ROOT / "rfdetr"))
        weights = config.get("weights") or config.get("pretrain_weights")
        if not weights and not config.get("from_scratch"):
            filename, url = PRETRAINED_ASSETS[size]
            weights = ensure_pretrained(
                PRETRAINED_ROOT / "rfdetr" / filename,
                url=url,
                hint="Place the official RF-DETR checkpoint in pretrained/rfdetr or pass --weights explicitly.",
            )
        elif weights and not config.get("from_scratch") and not Path(weights).is_file():
            target_path = Path(weights)
            matched_asset = next((v for v in PRETRAINED_ASSETS.values() if v[0] == target_path.name), None)
            if not matched_asset and size in PRETRAINED_ASSETS:
                matched_asset = PRETRAINED_ASSETS[size]
            if matched_asset:
                weights = ensure_pretrained(
                    target_path,
                    url=matched_asset[1],
                    hint=f"Place the official RF-DETR checkpoint in {target_path} or pass a valid path.",
                )
        if weights:
            kwargs["pretrain_weights"] = str(weights)
        return getattr(variants, name)(**kwargs)

    def train(self, config: Mapping[str, Any]):
        """执行 RF-DETR 实例分割训练。"""
        model = self._model(config)
        kwargs = {"dataset_dir": str(config["data"]), "resolution": int(config.get("resolution", 640)), "epochs": int(config.get("epochs", 20)), "batch_size": int(config.get("batch_size", 4)), "output_dir": str(config.get("output_dir", "runs/rfdetr")), "accelerator": config.get("accelerator", "auto"), "devices": config.get("devices", 1), "eval_interval": int(config.get("eval_interval", 1))}
        for key in ("pretrain_weights", "resume", "lr", "aug_config", "augmentation_backend", "seed", "notes"):
            if config.get(key) is not None:
                kwargs[key] = config[key]
        return model.train(**kwargs)

    def predict(self, config: Mapping[str, Any]):
        """执行 RF-DETR 实例分割预测与后处理。"""
        model = self._model(config)
        if not hasattr(model, "predict"):
            raise BackendCapabilityError("installed RF-DETR version has no predict API")
        # RF-DETR 不同版本中推理阈值的参数名称略有差异。仅传递当前安装版本支持的参数，以确保通用命令行选项在不同版本间的兼容性。
        import inspect
        signature = inspect.signature(model.predict)
        kwargs = {}
        classes = [str(value) for value in config.get("classes", [])]
        confidences = class_thresholds(classes, config.get("class_conf", config.get("class_thresholds")), float(config.get("confidence_threshold", config.get("score_threshold", 0.5))), name="class-conf") if classes else {}
        ious = class_thresholds(classes, config.get("class_iou", config.get("class_iou_thresholds")), float(config.get("iou_threshold", 1.0)), name="class-iou") if classes else {}
        confidence = min(confidences.values(), default=config.get("confidence_threshold", config.get("score_threshold")))
        # RF-DETR 各发布版本仅公开了全局查询阈值。在此放宽 NMS 并在 API 返回记录时应用所需的按类别策略。
        iou = 1.0
        if confidence is not None:
            for name in ("threshold", "confidence", "conf", "score_threshold"):
                if name in signature.parameters:
                    kwargs[name] = float(confidence)
                    break
        if iou is not None:
            for name in ("iou_threshold", "iou", "nms_threshold"):
                if name in signature.parameters:
                    kwargs[name] = float(iou)
                    break
        result = model.predict(str(config["source"]), **kwargs)
        if classes and isinstance(result, list) and all(isinstance(item, Prediction) for item in result):
            return filter_predictions(result, classes, confidences, ious)
        return result

    def evaluate(self, config: Mapping[str, Any]):
        """评估 RF-DETR 模型在指定验证集上的性能。"""
        model = self._model(config)
        if not hasattr(model, "evaluate"):
            raise BackendCapabilityError("installed RF-DETR version has no evaluate API")
        return model.evaluate(str(config["data"]))

    def export(self, config: Mapping[str, Any]):
        """将 RF-DETR 模型导出为 ONNX 格式。"""
        model = self._model(config)
        if not hasattr(model, "export"):
            raise BackendCapabilityError("installed RF-DETR version has no export API")
        output_dir = str(config.get("output_dir") or config.get("output") or "runs/rfdetr_export")
        output_name = config.get("output_name")
        if output_name:
            output_name = str(output_name)
        result = model.export(
            output_dir=output_dir,
            opset_version=int(config.get("opset_version", 17)),
            shape=(int(config.get("height", config.get("resolution", 640))), int(config.get("width", config.get("resolution", 640)))),
            batch_size=int(config.get("batch_size", 1)),
            dynamic_batch=bool(config.get("dynamic_batch", False)),
            fp16=bool(config.get("fp16", True)),
            output_name=output_name,
            verbose=bool(config.get("verbose", True)),
        )
        return Artifact(str(result), "onnx")
