#!/usr/bin/env python3
"""将 C++ Mask R-CNN 推理生成的 JSON 文件转换为 X-AnyLabeling / LabelMe JSON 格式。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def convert_detection_to_labelme(
    orig_data: dict[str, Any],
    json_path: Path,
) -> dict[str, Any]:
    """将单张图片的 C++ / Python 推理 JSON 记录转换为 LabelMe / X-AnyLabeling 数据结构规范。"""
    image_name = Path(orig_data.get("file") or orig_data.get("image") or f"{json_path.stem}.png").name
    height = orig_data.get("height")
    width = orig_data.get("width")
    if height is None or width is None:
        # 尝试从同级或关联图像文件读取真实尺寸
        img_candidates = [
            json_path.parent / image_name,
            Path(orig_data.get("image", "")) if orig_data.get("image") else None,
        ]
        found_hw = None
        for cand in img_candidates:
            if cand and cand.is_file():
                img = cv2.imread(str(cand))
                if img is not None:
                    found_hw = (img.shape[0], img.shape[1])
                    break
        if found_hw:
            height, width = found_hw
        else:
            height = height or 480
            width = width or 640
    height = int(height)
    width = int(width)

    shapes = []
    for det in orig_data.get("detections", []):
        label = det.get("class_name")
        polygons = det.get("mask_polygons") or det.get("contours_xy")
        if not polygons and "mask_rle" in det:
            rle = det["mask_rle"]
            if isinstance(rle, list) and height > 0 and width > 0 and sum(rle) == height * width:
                flat = np.zeros(height * width, dtype=np.uint8)
                offset = 0
                fg = False
                for c in rle:
                    if fg and c:
                        flat[offset : offset + c] = 1
                    offset += c
                    fg = not fg
                mask = flat.reshape((height, width))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygons = [[[float(pt[0][0]), float(pt[0][1])] for pt in cnt] for cnt in contours]

        for poly in (polygons or []):
            if len(poly) < 3:
                continue
            if len(poly) > 60:
                pts = np.asarray(poly, dtype=np.float32)
                approx = cv2.approxPolyDP(pts, 1.0, True)
                if len(approx) >= 3:
                    poly = approx.reshape(-1, 2).tolist()
            shape = {
                "label": label,
                "score": float(det["score"]) if det.get("score") is not None else None,
                "points": [[float(pt[0]), float(pt[1])] for pt in poly],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
            shapes.append(shape)

    return {
        "version": "4.0.5",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
    }


def process_file(
    json_path: Path,
    output_dir: Path | None = None,
    backup: bool = False,
) -> int:
    """转换单个 JSON 文件并写入 output_dir 或就地覆盖。

    返回创建的有效多边形标注数量。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        orig_data = json.load(f)

    converted = convert_detection_to_labelme(orig_data, json_path)
    target_path = output_dir / json_path.name if output_dir else json_path

    if backup and target_path == json_path:
        backup_path = json_path.with_suffix(".json.bak")
        shutil.copy2(json_path, backup_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    tmp_path.replace(target_path)

    return len(converted["shapes"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Convert C++ Mask R-CNN inference JSONs to X-AnyLabeling / LabelMe format.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input JSON file or directory containing JSON files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory. If omitted or same as input directory, files are overwritten in-place.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a .bak backup file before overwriting in-place.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """主函数执行转换流程。"""
    args = parse_args(argv)
    input_path = args.input.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        files = [input_path]
        out_dir = args.output_dir.resolve() if args.output_dir else None
    else:
        files = sorted(input_path.glob("*.json"))
        out_dir = args.output_dir.resolve() if args.output_dir else None

    if not files:
        logger.warning("No JSON files found in %s", input_path)
        return

    logger.info("Found %d JSON file(s) to process", len(files))
    total_shapes = 0
    try:
        from tqdm import tqdm
        file_iter = tqdm(files, desc="Converting to LabelMe", unit="file")
    except ImportError:
        file_iter = files

    for jf in file_iter:
        shapes_count = process_file(jf, output_dir=out_dir, backup=args.backup)
        total_shapes += shapes_count

    logger.info("Successfully converted %d file(s) with %d total shapes", len(files), total_shapes)


if __name__ == "__main__":
    main()
