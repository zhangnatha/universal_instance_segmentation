"""仓库根目录脚本所使用的小型参数风格启动器。"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .cli import main as unified_main


def _value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_extra(tokens: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise SystemExit(f"unexpected positional argument: {token}")
        key = token[2:].replace("-", "_")
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            config[key] = True
            index += 1
        else:
            values = []
            index += 1
            while index < len(tokens) and not tokens[index].startswith("--"):
                values.append(_value(tokens[index]))
                index += 1
            config[key] = values[0] if len(values) == 1 else values
    return config


def run_backend_operation(operation: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Parameter-style instance segmentation {operation} launcher"
    )
    parser.add_argument("--backend", choices=("detectron2", "ultralytics", "rfdetr"), default="detectron2")
    parser.add_argument("--config", help="JSON object or @config.json; extra --parameters are also accepted")
    parser.add_argument("--dry-run", action="store_true")
    parsed, remainder = parser.parse_known_args(argv)
    config = {}
    if parsed.config:
        raw = parsed.config
        if raw.startswith("@"):
            from pathlib import Path
            raw = Path(raw[1:]).expanduser().read_text(encoding="utf-8")
        config = json.loads(raw)
        if not isinstance(config, dict):
            raise SystemExit("--config must be a JSON object")
    config.update(_parse_extra(remainder))
    if operation == "predict":
        if parsed.backend == "detectron2":
            if "class_conf" in config and "class_thresholds" not in config:
                config["class_thresholds"] = config.pop("class_conf")
            if "class_iou" in config and "class_iou_thresholds" not in config:
                config["class_iou_thresholds"] = config.pop("class_iou")
        confidence = config.pop("confidence_threshold", config.pop("score_threshold", None))
        iou = config.pop("iou_threshold", config.pop("nms_threshold", None))
        if parsed.backend == "detectron2":
            classes = [str(name) for name in config.get("classes", [])]
            if confidence is not None and "class_thresholds" not in config:
                config["class_thresholds"] = [f"{name}={float(confidence)}" for name in classes]
            if iou is not None and "class_iou_thresholds" not in config:
                config["class_iou_thresholds"] = [f"{name}={float(iou)}" for name in classes]
        elif parsed.backend == "ultralytics":
            if confidence is not None:
                config["conf"] = confidence
            if iou is not None:
                config["iou"] = iou
        else:
            if confidence is not None:
                config["confidence_threshold"] = confidence
            if iou is not None:
                config["iou_threshold"] = iou
    command = ["run", parsed.backend, operation, "--config", json.dumps(config, ensure_ascii=False)]
    if parsed.dry_run:
        command.append("--dry-run")
    return unified_main(command)


def run_preprocess(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parameter-style dataset preprocessing launcher")
    parser.add_argument("--config", help="JSON object or @config.json")
    parsed, remainder = parser.parse_known_args(argv)
    config = {}
    if parsed.config:
        raw = parsed.config
        if raw.startswith("@"):
            from pathlib import Path
            raw = Path(raw[1:]).expanduser().read_text(encoding="utf-8")
        config = json.loads(raw)
    config.update(_parse_extra(remainder))
    return unified_main(["preprocess", "--backend", "ultralytics", "--config", json.dumps(config, ensure_ascii=False)])
