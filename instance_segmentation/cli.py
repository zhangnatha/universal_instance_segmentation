from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 扁平化 cli/ 目录后，保持 ``instance_segmentation.cli.main`` 仍可导入。
sys.modules.setdefault(__name__ + ".main", sys.modules[__name__])

from .data import create_split_manifest
from .models import get_backend, list_backends
from .models.base import BackendCapabilityError
from .models.detectron2_maskrcnn import list_supported_backbones


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified in-process instance segmentation CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("capabilities", help="enumerate backend protocol capabilities")
    tool = sub.add_parser("tool", help="run a packaged data, evaluation, or RF-DETR utility")
    tool.add_argument(
        "name",
        choices=(
            "rare-resample", "real-resample", "mine-hard-negatives",
            "calibrate", "analyze-thresholds", "rfdetr-calibrate",
            "rfdetr-calibrate-cpp", "rfdetr-export", "convert-to-labelme",
            "swin-export",
        ),
    )
    tool.add_argument("args", nargs=argparse.REMAINDER)
    run = sub.add_parser("run", help="call a backend Python API directly")
    run.add_argument("backend")
    run.add_argument("operation", choices=("split", "preprocess", "augment", "train", "predict", "infer", "evaluate", "export", "pipeline", "verify", "clean", "menu"))
    run.add_argument("--config", help="JSON object containing backend options")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("args", nargs=argparse.REMAINDER)
    # 兼容性别名保留了熟悉的命令形态，同时调用相同的进程内 API（不存在子进程边界）。
    for name in ("train", "infer", "predict", "evaluate", "export", "preprocess"):
        alias = sub.add_parser(name, help=f"compatibility alias for run {name}")
        alias.add_argument("--backend", default="detectron2" if name not in {"train", "infer", "predict"} else "ultralytics")
        alias.add_argument("--config", required=True, help="JSON object containing backend options")
    return parser


def _config(parsed: argparse.Namespace) -> dict[str, Any]:
    raw = parsed.config
    if raw and raw.startswith("@"):
        try:
            raw = Path(raw[1:]).expanduser().read_text(encoding="utf-8")
        except OSError as error:
            raise SystemExit(f"cannot read config file {parsed.config!r}: {error}") from error
    value = json.loads(raw) if raw else {}
    if not isinstance(value, dict):
        raise SystemExit("--config must be a JSON object")
    return value


def _legacy_backend(name: str) -> str:
    aliases = {"pipeline": "ultralytics", "maskrcnn": "detectron2", "multi-backbone": "detectron2", "multi_backbone": "detectron2"}
    return aliases.get(name, name)


def _model_module(name: str) -> str:
    """返回用于展示和日志记录的规范浅层模型目录名称。"""
    return {
        "detectron2": "detectron2_maskrcnn",
        "maskrcnn": "detectron2_maskrcnn",
        "ultralytics": "ultralytics_yolo",
        "yolo": "ultralytics_yolo",
        "rfdetr": "rfdetr",
        "rf_detr": "rfdetr",
    }.get(name.lower().replace("-", "_"), name)


def _print_capabilities() -> None:
    for name, item in list_backends().items():
        capabilities = item["capabilities"]
        print(f"{name}: train={capabilities.train} predict={capabilities.predict} evaluate={capabilities.evaluate} export={','.join(capabilities.export) or 'none'} preprocess={capabilities.preprocess}")
    print("detectron2 backbones:", ", ".join(list_supported_backbones()))


def _preprocess(config: dict[str, Any]) -> dict[str, Any]:
    directory = config.get("data") or config.get("source")
    if not directory:
        raise SystemExit("preprocess requires data/source in --config")
    manifest = create_split_manifest(directory, float(config.get("validation_ratio", 0.2)), int(config.get("seed", 42)), config.get("manifest"))
    return manifest.to_dict()


def _apply_gpu_memory_limit(config: dict[str, Any]) -> None:
    """在模型加载前应用可选的进程局部 CUDA 分配器上限限制。"""
    value = config.pop("max_gpu_memory_fraction", None)
    if value is None:
        return
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise SystemExit("max_gpu_memory_fraction must be greater than 0 and at most 1")
    try:
        import torch
    except ImportError as error:
        raise SystemExit("max_gpu_memory_fraction requires PyTorch") from error
    if not torch.cuda.is_available():
        raise SystemExit("max_gpu_memory_fraction was requested but CUDA is unavailable")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(
        f"GPU memory safety cap: {fraction:.0%} of device 0 "
        f"({total_gib * fraction:.2f} GiB of {total_gib:.2f} GiB)"
    )


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    if parsed.command:
        from instance_segmentation.run_logging import setup_run_logging
        setup_run_logging(f"CLI_{parsed.command.upper()}")
    if parsed.command == "capabilities":
        _print_capabilities(); return 0
    if parsed.command == "tool":
        modules = {
            "rare-resample": "instance_segmentation.data.rare_resample",
            "real-resample": "instance_segmentation.data.real_resample",
            "mine-hard-negatives": "instance_segmentation.data.mine_hard_negatives",
            "calibrate": "instance_segmentation.evaluation.calibrate_thresholds",
            "analyze-thresholds": "instance_segmentation.evaluation.analyze_threshold_dominance",
            "rfdetr-calibrate": "instance_segmentation.models.rfdetr.calibrate_thresholds",
            "rfdetr-calibrate-cpp": "instance_segmentation.models.rfdetr.calibrate_cpp_thresholds",
            "rfdetr-export": "instance_segmentation.models.rfdetr.export",
            "convert-to-labelme": "instance_segmentation.data.convert_to_labelme",
            "swin-export": "instance_segmentation.models.detectron2_maskrcnn.export_swin",
        }
        from importlib import import_module
        module = import_module(modules[parsed.name])
        module.main(list(parsed.args))
        return 0
    if parsed.command in {"train", "infer", "predict", "evaluate", "export", "preprocess"}:
        backend_name, operation, config, dry_run = parsed.backend, parsed.command, _config(parsed), False
    elif parsed.command == "run":
        # ``argparse.REMAINDER`` 有意保留旧版后端标志，
        # 但它也会捕获操作后编写的统一选项（自然且记录在文档中的 CLI 顺序）。
        # 在转发给后端之前恢复这些选项。
        remainder = list(parsed.args)
        if remainder and remainder[0] == "--":
            remainder = remainder[1:]
        if parsed.config is None and "--config" in remainder:
            index = remainder.index("--config")
            if index + 1 >= len(remainder):
                raise SystemExit("--config requires a JSON object")
            parsed.config = remainder[index + 1]
            del remainder[index:index + 2]
        remainder = [arg for arg in remainder if arg != "--dry-run"]
        backend_name, operation, config = parsed.backend, parsed.operation, _config(parsed)
        dry_run = parsed.dry_run or "--dry-run" in parsed.args
        # 旧的 argparse 命令仍可在操作后传递简单标志；
        # 首选 JSON 格式，因其能精确保持列表和布尔类型。
        if remainder:
            config.setdefault("legacy_args", remainder)
    else:
        _parser().print_help(); return 0
    operation = "predict" if operation == "infer" else operation
    if backend_name == "pipeline" and operation in {"split", "preprocess", "augment", "train", "predict", "evaluate", "pipeline", "verify", "clean", "menu"}:
        if dry_run:
            print(f"[dry-run] instance_segmentation.legacy_cli.main(['{operation}', ...])")
            return 0
        _apply_gpu_memory_limit(config)
        from .legacy_cli import main as pipeline_main
        forwarded = list(config.get("legacy_args", []))
        if not forwarded and config:
            for key, value in config.items():
                if key == "legacy_args" or value is None:
                    continue
                flag = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    if value: forwarded.append(flag)
                elif isinstance(value, (list, tuple)):
                    forwarded.extend([flag, *map(str, value)])
                else:
                    forwarded.extend([flag, str(value)])
        pipeline_main([operation, *forwarded])
        return 0
    backend_name = _legacy_backend(backend_name)
    if operation == "preprocess":
        if dry_run: print("[dry-run] instance_segmentation.data.create_split_manifest"); return 0
        print(json.dumps(_preprocess(config), ensure_ascii=False, indent=2)); return 0
    if not dry_run:
        _apply_gpu_memory_limit(config)
    backend = get_backend(backend_name)
    if dry_run:
        if operation == "export" and not backend.capabilities.export:
            print(f"[dry-run] CapabilityError: backend {backend.name!r} has no export capability")
            return 2
        print(f"[dry-run] instance_segmentation.models.{_model_module(backend_name)}.{operation}(config)"); return 0
    try:
        result = getattr(backend, operation)(config)
    except BackendCapabilityError as error:
        raise SystemExit(str(error)) from error
    if result is not None:
        if hasattr(result, "to_dict"): result = result.to_dict()
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
