"""数据集划分清单与采样划分模块。

提供确定性的数据集划分、清单持久化以及训练集/验证集拆分功能。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .labelme_core import ImageSample, discover_samples


@dataclass(frozen=True)
class SplitManifest:
    """数据集划分清单数据类，记录源目录、随机种子、划分比例及各子集样本 ID。"""
    source_directory: str
    seed: int
    validation_ratio: float
    train: tuple[str, ...]
    valid: tuple[str, ...]
    test: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """将划分清单转换为字典格式。"""
        return {
            "source_directory": self.source_directory,
            "seed": self.seed,
            "validation_ratio": self.validation_ratio,
            "train": list(self.train),
            "valid": list(self.valid),
            "test": list(self.test),
        }

    def write(self, path: str | Path) -> Path:
        """将划分清单写入指定的 JSON 文件。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


def split_samples(samples: Sequence[ImageSample], validation_ratio: float = 0.2, seed: int = 42) -> tuple[list[ImageSample], list[ImageSample]]:
    """根据验证集比例和随机种子将样本列表划分为训练集和验证集。"""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * validation_ratio)))
    return sorted(shuffled[valid_count:], key=lambda s: s.image_id), sorted(shuffled[:valid_count], key=lambda s: s.image_id)


def create_split_manifest(directory: str | Path, validation_ratio: float = 0.2, seed: int = 42, output: str | Path | None = None) -> SplitManifest:
    """扫描指定目录中的样本，执行划分并生成 SplitManifest 实例，可选写入清单文件。"""
    samples = discover_samples(directory)
    train, valid = split_samples(samples, validation_ratio, seed)
    manifest = SplitManifest(str(Path(directory)), seed, validation_ratio, tuple(s.image_id for s in train), tuple(s.image_id for s in valid))
    if output:
        manifest.write(output)
    return manifest
