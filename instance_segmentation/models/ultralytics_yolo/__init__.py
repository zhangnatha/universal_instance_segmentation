"""Ultralytics YOLO 实例分割模型后端封装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..base import InstanceSegmentationBackend
from ...schema import Artifact, BackendCapabilities, ModelManifest, Prediction
from ...paths import ensure_pretrained, pretrained_path
from ..thresholds import class_thresholds, filter_predictions


MODEL_NAMES = {
    "yolov11": {"nano": "yolo11n-seg.pt", "small": "yolo11s-seg.pt", "medium": "yolo11m-seg.pt", "large": "yolo11l-seg.pt", "xlarge": "yolo11x-seg.pt"},
    "yolo26": {"nano": "yolo26n-seg.pt", "small": "yolo26s-seg.pt", "medium": "yolo26m-seg.pt", "large": "yolo26l-seg.pt", "xlarge": "yolo26x-seg.pt"},
    "roboflow-3.0": {"fast": "yolov8n-seg.pt", "accurate": "yolov8m-seg.pt", "medium": "yolov8m-seg.pt", "large": "yolov8l-seg.pt", "xlarge": "yolov8x-seg.pt"},
}
OFFICIAL_ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"


class UltralyticsBackend(InstanceSegmentationBackend):
    """基于 Ultralytics YOLO 的实例分割模型后端实现。"""
    name = "ultralytics"
    capabilities = BackendCapabilities(train=True, predict=True, evaluate=True, export=("onnx", "engine", "torchscript"), preprocess=True)

    @staticmethod
    def _model(config: Mapping[str, Any]):
        """加载或初始化 YOLO 模型实例。"""
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Ultralytics backend requires the optional 'ultralytics' extra") from error
        path = config.get("weights") or config.get("model")
        if not path:
            model_id = str(config.get("model_id", "yolov11")); size = str(config.get("size", "nano"))
            if config.get("from_scratch"):
                yaml_prefix = {"yolov11": "yolo11", "yolo26": "yolo26", "roboflow-3.0": "yolov8"}.get(model_id, "yolo11")
                yaml_size = {"fast": "n", "accurate": "m", "nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}.get(size, "n")
                path = config.get("model_config") or f"{yaml_prefix}{yaml_size}-seg.yaml"
            else:
                filename = MODEL_NAMES.get(model_id, MODEL_NAMES["yolov11"]).get(size, "yolo11n-seg.pt")
                path = ensure_pretrained(
                    pretrained_path("ultralytics", filename),
                    url=OFFICIAL_ASSET_BASE + filename,
                    hint="Place the official Ultralytics checkpoint in pretrained/ultralytics or pass --weights explicitly.",
                )
        elif not Path(path).is_file() and not config.get("from_scratch"):
            target = Path(path)
            if target.name.endswith("-seg.pt"):
                path = ensure_pretrained(
                    target,
                    url=OFFICIAL_ASSET_BASE + target.name,
                    hint="Place the official Ultralytics checkpoint in pretrained/ultralytics or pass --weights explicitly.",
                )
        return YOLO(path)

    def train(self, config: Mapping[str, Any]):
        """执行 YOLO 实例分割训练。"""
        model = self._model(config)
        train_kwargs = {
            "data": str(config["data"]), "epochs": int(config.get("epochs", 20)),
            "imgsz": config.get("imgsz", 640), "batch": config.get("batch", 8),
            "device": config.get("device", "auto"), "project": config.get("project", "runs"),
            "name": config.get("name", "instance_segmentation"), "amp": config.get("amp", True),
            "resume": config.get("resume", False), "workers": int(config.get("workers", 0)),
        }
        for key in ("fraction", "cache", "verbose"):
            if key in config:
                train_kwargs[key] = config[key]
        result = model.train(**train_kwargs)
        return result

    def predict(self, config: Mapping[str, Any]):
        """执行 YOLO 实例分割预测与后处理。"""
        import cv2
        import numpy as np
        model = self._model(config)
        classes = [str(value) for value in config.get("classes", [])]
        confs = class_thresholds(classes, config.get("class_conf", config.get("class_thresholds")), float(config.get("conf", 0.25)), name="class-conf") if classes else {}
        ious = class_thresholds(classes, config.get("class_iou", config.get("class_iou_thresholds")), float(config.get("iou", 0.5)), name="class-iou") if classes else {}
        # 保留按类别低阈值所需的全部候选框；下方的过滤及按类别 NMS 是最终后处理。
        results = model.predict(source=str(config["source"]), conf=min(confs.values(), default=float(config.get("conf", 0.25))), iou=1.0, device=config.get("device", "auto"), save=config.get("save", False), verbose=config.get("verbose", False))
        predictions = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None); masks = getattr(result, "masks", None)
            if boxes is None:
                continue
            for index, (box, score, class_id) in enumerate(zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())):
                cid = int(class_id); name = classes[cid] if cid < len(classes) else str(names.get(cid, cid))
                mask = None
                if masks is not None and len(masks.data) > index:
                    # Ultralytics 的 masks.data 处于网络分辨率，而边界框和 Python 可视化工具使用原始图像坐标。在暴露后端通用的 Prediction 接口前缩放掩码。
                    mask = masks.data[index].cpu().numpy().astype(np.float32)
                    original_shape = getattr(result, "orig_shape", None)
                    if original_shape is not None and tuple(mask.shape[-2:]) != tuple(original_shape):
                        mask = cv2.resize(mask, (int(original_shape[1]), int(original_shape[0])), interpolation=cv2.INTER_NEAREST)
                    mask = mask >= 0.5
                predictions.append(Prediction(str(getattr(result, "path", "")), cid, name, float(score), tuple(float(v) for v in box), mask))
        if not classes:
            classes = sorted({prediction.class_name for prediction in predictions})
            confs = {name: float(config.get("conf", 0.25)) for name in classes}
        filtered = filter_predictions(predictions, classes, confs, ious)
        if config.get("output"):
            import json
            from collections import defaultdict
            from pathlib import Path
            from instance_segmentation.models.rfdetr.infer_dataset import (
                mask_to_coco_rle,
                render_visual_overlay,
            )

            out_dir = Path(config["output"]).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            by_image = defaultdict(list)
            for p in filtered:
                by_image[p.image_path].append(p)

            all_img_paths = [str(getattr(r, "path", "")) for r in results if getattr(r, "path", None)]
            for img_path_str in all_img_paths:
                if img_path_str and img_path_str not in by_image:
                    by_image[img_path_str] = []

            for img_path_str, preds in by_image.items():
                if not img_path_str:
                    continue
                img_path = Path(img_path_str)
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2] if img is not None else (0, 0)
                detections_payload = []
                render_candidates = []
                for p in preds:
                    item = {
                        "class_id": p.class_id,
                        "class_name": p.class_name,
                        "score": round(float(p.score), 4),
                        "bbox_xyxy": [round(float(v), 2) for v in p.bbox],
                    }
                    if p.mask is not None:
                        item["mask_rle"] = mask_to_coco_rle(p.mask)
                        item["mask_area"] = int(p.mask.sum())
                    detections_payload.append(item)
                    render_candidates.append({
                        "class_id": p.class_id,
                        "class_name": p.class_name,
                        "score": float(p.score),
                        "bbox_xyxy": list(p.bbox),
                        "mask_np": p.mask,
                    })

                payload = {
                    "file": str(img_path),
                    "imagePath": img_path.name,
                    "image": str(img_path),
                    "width": w,
                    "height": h,
                    "imageWidth": w,
                    "imageHeight": h,
                    "detections": detections_payload,
                }
                out_json = out_dir / f"{img_path.stem}.json"
                with open(out_json, "w", encoding="utf-8") as fp:
                    json.dump(payload, fp, indent=2)

                if config.get("draw", True) and img is not None:
                    vis_img = render_visual_overlay(img, render_candidates)
                    cv2.imwrite(str(out_dir / f"{img_path.stem}_vis.jpg"), vis_img)

        return filtered

    def evaluate(self, config: Mapping[str, Any]):
        """评估 YOLO 模型在指定验证集上的性能。"""
        return self._model(config).val(data=str(config["data"]), split=config.get("split", "val"), imgsz=config.get("imgsz", 640), device=config.get("device", "auto"))

    def export(self, config: Mapping[str, Any]):
        """将 YOLO 模型导出为指定格式（如 ONNX）。"""
        model = self._model(config)
        export_format = str(config.get("format", "onnx"))
        image_size = int(config.get("imgsz", 640))
        path = model.export(format=export_format, imgsz=image_size, dynamic=config.get("dynamic", False), int8=config.get("int8", False))
        names = config.get("classes")
        if not names:
            raw_names = getattr(model, "names", {})
            names = [raw_names[key] for key in sorted(raw_names)] if isinstance(raw_names, dict) else list(raw_names)
        names = tuple(str(name) for name in (names or ()))
        manifest = ModelManifest(
            "ultralytics", "yolo-seg", names, input_size=(image_size, image_size),
            checkpoint=str(config.get("weights") or config.get("model") or ""),
            options={
                "format": export_format,
                "preprocess": "RGB float32 [0,1], letterbox pad=114",
                "outputs": {
                    "detections": "rank-3 [1,C,A] or [1,A,C], C=4+classes+M (or +1 objectness)",
                    "prototypes": "rank-4 [1,M,Mh,Mw] float32 mask logits",
                },
                "coordinate_space": "letterboxed input pixels (cx,cy,w,h)",
                "postprocess": "class confidence, class-wise greedy NMS, prototype sigmoid and mask threshold",
            },
        )
        manifest_path = Path(str(path)).with_suffix(Path(str(path)).suffix + ".manifest.json")
        manifest.write(manifest_path)
        return Artifact(str(path), export_format, manifest)
