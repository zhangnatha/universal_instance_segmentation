"""LabelMe 格式核心数据结构与解析工具。

定义单标注实例 Annotation 与单图样本 ImageSample，并提供快速解析 LabelMe JSON 和扫描样本目录的基础方法。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


@dataclass
class Annotation:
    """单个实例标注数据结构，包含类别、坐标点集、图形类型及可选方向角。"""
    label: str
    points: list[list[float]]
    shape_type: str = "polygon"
    angle_degrees: float | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """返回多边形的外接矩形框坐标 (xmin, ymin, xmax, ymax)。"""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class ImageSample:
    """单张图像样本数据结构，包含图像路径、标注路径、图像尺寸及所有实例标注。"""
    image_path: Path
    annotation_path: Path | None
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)
    image_id: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def is_null(self) -> bool:
        """若样本中没有任何实例标注则返回 True。"""
        return not self.annotations


def _image_for(annotation_path: Path, item: dict[str, Any]) -> Path:
    """根据标注文件及 JSON 内容中记录的 imagePath 寻找对应的图像文件。"""
    image_path = item.get("imagePath")
    candidates = []
    if image_path:
        candidates.extend((annotation_path.parent / str(image_path), annotation_path.parent / Path(str(image_path)).name))
    candidates.extend(annotation_path.with_suffix(ext) for ext in IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image for annotation: {annotation_path}")


def parse_labelme(path: str | Path, *, classes: Iterable[str] | None = None, strict: bool = True) -> ImageSample:
    """解析单个 LabelMe JSON 文件并返回 ImageSample 对象。"""
    annotation_path = Path(path)
    item = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    configured = set(classes or ())
    annotations: list[Annotation] = []
    for shape in item.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        points = shape.get("points") or []
        if not label or len(points) < 2:
            continue
        if configured and label not in configured:
            if strict:
                raise ValueError(f"{annotation_path.name}: class {label!r} is not configured")
            continue
        if shape.get("shape_type", "polygon") not in {"polygon", "rectangle"}:
            continue
        angle = shape.get("angle_degrees", shape.get("angle"))
        attrs = shape.get("attributes") or {}
        if angle is None:
            angle = attrs.get("angle_degrees", attrs.get("angle"))
        annotations.append(Annotation(label, [[float(x), float(y)] for x, y in points], shape.get("shape_type", "polygon"), None if angle is None else float(angle)))
    image_path = _image_for(annotation_path, item)
    width, height = int(item.get("imageWidth") or 0), int(item.get("imageHeight") or 0)
    if not width or not height:
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception:
            width, height = 0, 0
    return ImageSample(image_path, annotation_path, width, height, annotations, annotation_path.stem, list(item.get("tags") or []))


def discover_samples(directory: str | Path) -> list[ImageSample]:
    """扫描指定目录下的所有 LabelMe JSON 文件（排除 split.json）并返回解析后的样本列表。"""
    root = Path(directory)
    return [parse_labelme(path) for path in sorted(root.glob("*.json")) if path.name != "split.json"]
