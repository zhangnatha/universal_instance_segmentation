#!/usr/bin/env python3
"""将 Detectron2 Mask R-CNN 模型导出为 ONNX 模型，支持 FP32、FP16 和 INT8 QDQ 量化。"""

from __future__ import annotations

import argparse
import tempfile
import types
from pathlib import Path

import cv2
import onnx
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from .config import add_model_arguments, build_cfg, experiment_from_args
from .run_logging import setup_run_logging


class OnnxModel(torch.nn.Module):
    """将 Detectron2 的结构化输入输出转换为 ONNX 张量。"""

    def __init__(self, model, height: int, width: int):
        super().__init__()
        self.model = model
        self.height = height
        self.width = width

    def forward(self, image):
        inputs = [{"image": image, "height": self.height, "width": self.width}]
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, None)
        instances = self.model.roi_heads._forward_box(features, proposals)[0]
        heads = self.model.roi_heads
        if hasattr(heads, "angle_head"):
            pooled = heads.box_pooler(
                [features[name] for name in heads.box_in_features], [instances.pred_boxes]
            )
            angle_logits = heads.angle_head(pooled)
            angle_probabilities = angle_logits.softmax(dim=1)
            angle_confidence, angle_bins = angle_probabilities.max(dim=1)
            instances.pred_angles = angle_bins.to(dtype=angle_logits.dtype) * (
                heads.angle_period / heads.angle_bins
            )
            instances.angle_scores = angle_confidence
        mask_features = heads.mask_pooler(
            [features[name] for name in heads.mask_in_features],
            [instances.pred_boxes],
        )
        mask_logits = heads.mask_head.layers(mask_features)
        indices = torch.arange(mask_logits.shape[0], device=instances.pred_classes.device)
        mask_probs = mask_logits[indices, instances.pred_classes][:, None].sigmoid()
        outputs = (
            instances.pred_boxes.tensor,
            instances.scores,
            instances.pred_classes,
            mask_probs,
        )
        if hasattr(heads, "angle_head"):
            angle_values = getattr(instances, "pred_angles", instances.scores)
            angle_scores = getattr(instances, "angle_scores", instances.scores)
            outputs += (
                angle_values,
                angle_scores,
                heads.angle_class_mask_meta.to(dtype=angle_values.dtype)
                + angle_values.sum() * 0.0,
            )
        return outputs


def parse_args(argv=None):
    """解析 ONNX 导出命令行参数。"""
    parser = argparse.ArgumentParser(description="Export a fixed-resolution Detectron2 ONNX model")
    parser.add_argument("--sample", required=True, help="Sample image used for input size and tracing")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument(
        "--opset", "--opset-version", dest="opset", type=int, default=16,
        help="ONNX opset version; default: 16",
    )
    parser.add_argument(
        "--quantization", choices=["none", "int8"], default="none",
        help="Quantization mode: none for FP32 or int8 for static QDQ quantization",
    )
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "int8"], default=None,
        help="Output precision: fp32, fp16, or static-QDQ int8; overrides --quantization",
    )
    parser.add_argument(
        "--calibration-data",
        help="INT8 calibration image or directory; required for int8",
    )
    parser.add_argument(
        "--calibration-limit", type=int, default=100,
        help="Maximum number of INT8 calibration images; default: 100",
    )
    parser.add_argument(
        "--int8-per-channel", action="store_true",
        help="Enable per-channel INT8 weight quantization",
    )
    parser.add_argument(
        "--no-constant-folding", action="store_true",
        help="Disable ONNX constant folding",
    )
    add_model_arguments(parser, weights_required=True)
    parser.set_defaults(device="cpu")
    return parser.parse_args(argv)


def calibration_paths(path: str, limit: int) -> list[Path]:
    """获取用于 INT8 量化校准的图像文件路径列表。"""
    source = Path(path).expanduser()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        suffixes = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    else:
        raise SystemExit(f"Calibration data does not exist: {source}")
    if limit <= 0:
        raise SystemExit("--calibration-limit must be greater than zero")
    paths = paths[:limit]
    if not paths:
        raise SystemExit(f"No usable images found in calibration data: {source}")
    return paths


class ImageCalibrationReader:
    """ONNX Runtime 静态量化校准数据读取器。"""

    def __init__(self, paths: list[Path], height: int, width: int, image_format: str = "BGR"):
        self.paths = paths
        self.height = height
        self.width = width
        self.image_format = image_format
        self.index = 0

    def get_next(self):
        if self.index >= len(self.paths):
            return None
        path = self.paths[self.index]
        self.index += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read calibration image: {path}")
        if image.shape[:2] != (self.height, self.width):
            raise RuntimeError(
                f"Calibration image {path} is {image.shape[1]}x{image.shape[0]}; "
                f"expected {self.width}x{self.height}"
            )
        if self.image_format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = image.astype("float32").transpose(2, 0, 1)
        return {"image": tensor}

    def rewind(self):
        self.index = 0


def quantize_model(source: Path, output: Path, args, height: int, width: int, image_format: str):
    """对导出的 ONNX 模型执行静态 INT8 QDQ 量化。"""
    if args.quantization == "int8":
        if not args.calibration_data:
            raise SystemExit("--quantization int8 requires --calibration-data")
        if args.int8_per_channel and not hasattr(onnx, "mapping"):
            # ORT 1.17.3 使用旧的 onnx.mapping API；ONNX 1.19+ 已移至
            # onnx._mapping。构造只包含量化器所需字段的兼容视图。
            from onnx import _mapping
            onnx.mapping = types.SimpleNamespace(
                TENSOR_TYPE_TO_NP_TYPE={
                    key: value.np_dtype for key, value in _mapping.TENSOR_TYPE_MAP.items()
                }
            )
        try:
            from onnxruntime.quantization import (
                CalibrationMethod,
                QuantFormat,
                QuantType,
                quantize_static,
            )
        except ImportError as error:
            raise SystemExit(
                "INT8 export requires the onnxruntime Python package. Recreate the environment or run: "
                "pip install onnxruntime"
            ) from error
        paths = calibration_paths(args.calibration_data, args.calibration_limit)
        reader = ImageCalibrationReader(paths, height, width, image_format)
        quantize_static(
            str(source),
            str(output),
            reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            per_channel=args.int8_per_channel,
            calibrate_method=CalibrationMethod.MinMax,
        )
        print(f"INT8 calibration images: {len(paths)}")
        print("Warning: INT8 introduces accuracy loss; validate accuracy with representative data.")
        return

    raise RuntimeError(f"Unsupported quantization mode: {args.quantization}")


def main(argv=None):
    """ONNX 导出主函数：构建模型、执行 Trace 导出并进行精度转换或量化。"""
    setup_run_logging("EXPORT_ONNX")
    args = parse_args(argv)
    if not 11 <= args.opset <= 20:
        raise SystemExit("--opset-version must be between 11 and 20; 16 is recommended")
    precision = args.precision or ("int8" if args.quantization == "int8" else "fp32")
    if precision == "int8":
        args.quantization = "int8"
    if precision == "int8" and not args.calibration_data:
        raise SystemExit("--quantization int8 requires --calibration-data")
    if args.calibration_limit <= 0:
        raise SystemExit("--calibration-limit must be greater than zero")
    exp = experiment_from_args(args)
    cfg = build_cfg(exp, training=False)
    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    image = cv2.imread(str(Path(args.sample).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Unable to read sample image: {args.sample}")
    height, width = image.shape[:2]
    if min(height, width) != cfg.INPUT.MIN_SIZE_TEST or max(height, width) > cfg.INPUT.MAX_SIZE_TEST:
        raise SystemExit(
            f"Sample image is {width}x{height}. Export requires the model input size: "
            f"short edge {cfg.INPUT.MIN_SIZE_TEST}, long edge at most {cfg.INPUT.MAX_SIZE_TEST}."
        )
    tensor = torch.as_tensor(
        (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if cfg.INPUT.FORMAT == "RGB" else image
        ).astype("float32").transpose(2, 0, 1),
        device=cfg.MODEL.DEVICE,
    )
    wrapper = OnnxModel(model, height, width).eval()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=".onnx-export-fp32-", suffix=".onnx",
        dir=str(output.parent), delete=False,
    )
    temporary.close()
    export_path = Path(temporary.name)
    quantized_path = None
    inferred_path = export_path.with_name(export_path.stem + "-inferred.onnx")
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (tensor,),
                str(export_path),
                opset_version=args.opset,
                input_names=["image"],
                output_names=(
                    ["boxes", "scores", "classes", "mask_probs", "angles", "angle_scores", "angle_classes"]
                    if hasattr(model.roi_heads, "angle_head")
                    else ["boxes", "scores", "classes", "mask_probs"]
                ),
                dynamic_axes={
                    "boxes": {0: "detections"},
                    "scores": {0: "detections"},
                    "classes": {0: "detections"},
                    "mask_probs": {0: "detections"},
                    **({"angles": {0: "detections"}} if hasattr(model.roi_heads, "angle_head") else {}),
                    **({"angle_scores": {0: "detections"}} if hasattr(model.roi_heads, "angle_head") else {}),
                },
                do_constant_folding=not args.no_constant_folding,
            )
        onnx.checker.check_model(onnx.load(str(export_path)))
        final_path = export_path
        if precision == "fp16":
            try:
                from onnxruntime.transformers.float16 import convert_float_to_float16
            except ImportError as error:
                raise SystemExit("--precision fp16 requires onnxruntime") from error
            model_fp32 = onnx.load(str(export_path))
            model_fp16 = convert_float_to_float16(model_fp32, keep_io_types=True)
            onnx.save(model_fp16, str(inferred_path))
            onnx.checker.check_model(model_fp16)
            final_path = inferred_path
        if precision == "int8":
            quantized = tempfile.NamedTemporaryFile(
                prefix=".onnx-export-int8-", suffix=".onnx",
                dir=str(output.parent), delete=False,
            )
            quantized.close()
            quantized_path = Path(quantized.name)
            quantize_model(export_path, quantized_path, args, height, width, cfg.INPUT.FORMAT)
            onnx.checker.check_model(onnx.load(str(quantized_path)))
            final_path = quantized_path
        final_path.replace(output)
        if final_path == export_path:
            export_path = None
        else:
            quantized_path = None
    finally:
        if export_path is not None:
            export_path.unlink(missing_ok=True)
        if quantized_path is not None:
            quantized_path.unlink(missing_ok=True)
        inferred_path.unlink(missing_ok=True)
    print(f"ONNX export completed: {output}")
    print(f"Fixed input: {cfg.INPUT.FORMAT} FP32 [3, {height}, {width}]")
    print(f"Model precision: {precision.upper()}")
    print(f"Opset version: {args.opset}")
    print(f"File size: {output.stat().st_size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
