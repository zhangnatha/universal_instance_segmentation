"""数据集格式转换模块。

支持将样本数据结构转换为 COCO、YOLO 等标注格式，并提供流水线导出外观接口。
"""

from __future__ import annotations

from typing import Iterable

from .labelme_core import ImageSample


def export_yolo_dataset(manager, output: str, *, task: str = "segment"):
    """完整流水线导出接口的统一外观函数，保持原逻辑行为不变。"""
    return manager.export_yolo_dataset(output, task=task)


def load_pipeline_dataset(directory: str):
    """加载指定目录中的数据集管理器实例。"""
    from .dataset import DatasetManager
    return DatasetManager(directory)


def to_coco(samples: Iterable[ImageSample], classes: list[str]) -> dict:
    """将图像样本集合转换为标准 COCO 标注字典格式。"""
    class_to_id = {name: index + 1 for index, name in enumerate(classes)}
    images, annotations = [], []
    for image_id, sample in enumerate(samples, 1):
        images.append({
            "id": image_id,
            "file_name": sample.image_path.name,
            "width": sample.width,
            "height": sample.height,
            "tags": sample.tags,
        })
        for ann_id, ann in enumerate(sample.annotations, len(annotations) + 1):
            if ann.label not in class_to_id:
                raise ValueError(f"unknown class: {ann.label}")
            x0, y0, x1, y1 = ann.bbox
            segmentation = [[value for point in ann.points for value in point]]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_to_id[ann.label],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": max(0.0, (x1 - x0) * (y1 - y0)),
                "segmentation": segmentation,
                "iscrowd": 0,
            })
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(classes)],
    }


def to_yolo(sample: ImageSample, classes: list[str]) -> list[str]:
    """将单个图像样本的标注转换为 YOLO 分割多边形文本行列表。"""
    class_to_id = {name: index for index, name in enumerate(classes)}
    rows = []
    for ann in sample.annotations:
        x0, y0, x1, y1 = ann.bbox
        if ann.label not in class_to_id or sample.width <= 0 or sample.height <= 0:
            continue
        polygon = " ".join(f"{x / sample.width:.8f} {y / sample.height:.8f}" for x, y in ann.points)
        rows.append(f"{class_to_id[ann.label]} {polygon}")
    return rows
