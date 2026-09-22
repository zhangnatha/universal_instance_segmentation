"""类别名称管理与校验模块。

提供从配置文件读取类别名称列表、去重清洗以及与标注中发现的类别进行一致性校验的功能。
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable


def _clean_names(lines: Iterable[str]) -> list[str]:
    """清洗类别行数据，去除注释和首尾空格，并保持顺序去重。"""
    names: list[str] = []
    for raw in lines:
        name = str(raw).split("#", 1)[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def load_class_names(path: str | Path | None = None) -> list[str]:
    """从指定文件或仓库本地候选路径中加载类别名称列表。"""
    candidates = [Path(path)] if path else []
    candidates.extend([Path(__file__).resolve().parent / "classes.names", Path.cwd() / "classes.names"])
    for candidate in candidates:
        if candidate.is_file():
            names = _clean_names(candidate.read_text(encoding="utf-8-sig").splitlines())
            if names:
                return names
    if path:
        raise ValueError(f"classes file is empty: {path}")
    return []


def validate_classes(labels: Iterable[str], classes: Iterable[str]) -> None:
    """校验标注中的类别是否均已在配置的类别列表中定义。"""
    configured = set(classes)
    unknown = sorted({str(label) for label in labels} - configured)
    if unknown:
        raise ValueError("Labels missing from classes file: " + ", ".join(unknown))


def resolve_class_names(discovered: Iterable[str] | None = None, configured_path: str | Path | None = None) -> list[str]:
    """解析并确定最终生效的类别列表，优先采用配置文件，若未配置则使用数据集中发现的类别。"""
    return load_class_names(configured_path) or _clean_names(discovered or [])


def validate_discovered_labels(discovered: Iterable[str], configured: Iterable[str] | None = None) -> list[str]:
    """比对数据集中发现的类别标签与配置的类别，返回所有未在配置中定义的未知类别名称列表。"""
    configured_set = set(configured if configured is not None else load_class_names())
    if not configured_set:
        return []
    return sorted({str(label).strip() for label in discovered if str(label).strip()} - configured_set)
