from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class Prediction:
    """与后端无关的实例预测结果。

    Mask 可以是 numpy 数组、COCO RLE 字典或多边形列表。
    该模式在序列化之前刻意保持表示的无损性。
    """

    image_id: str
    class_id: int
    class_name: str
    score: float
    bbox: tuple[float, float, float, float]
    mask: Any = None
    angle_degrees: float | None = None

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["bbox"] = list(self.bbox)
        if hasattr(self.mask, "tolist"):
            item["mask"] = self.mask.tolist()
        return item

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> "Prediction":
        bbox = tuple(float(v) for v in item.get("bbox", (0, 0, 0, 0)))
        return cls(
            image_id=str(item.get("image_id", item.get("image", ""))),
            class_id=int(item.get("class_id", item.get("category_id", 0))),
            class_name=str(item.get("class_name", item.get("label", ""))),
            score=float(item.get("score", 1.0)), bbox=bbox,
            mask=item.get("mask", item.get("segmentation")),
            angle_degrees=item.get("angle_degrees"),
        )


@dataclass(frozen=True)
class ModelManifest:
    """独立于框架检查点格式的可复现模型标识。"""

    backend: str
    model_id: str
    classes: tuple[str, ...]
    canonical_backbone: str | None = None
    input_size: tuple[int, int] | None = None
    checkpoint: str | None = None
    metric_name: str = "segm/AP50"
    metric_value: float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["classes"] = list(self.classes)
        if self.input_size is not None:
            value["input_size"] = list(self.input_size)
        return value

    def write(self, path: str | Path) -> Path:
        import json
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


@dataclass(frozen=True)
class BackendCapabilities:
    """后端协议能力。"""
    train: bool = False
    predict: bool = False
    evaluate: bool = False
    export: tuple[str, ...] = ()
    preprocess: bool = True


@dataclass(frozen=True)
class MetricReport:
    """指标报告。"""
    metrics: Mapping[str, float]
    per_class: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    """模型产物。"""
    path: str
    kind: str
    manifest: ModelManifest | None = None
