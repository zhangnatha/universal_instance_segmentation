from __future__ import annotations

from pathlib import Path

ONNX_OUTPUTS = ("boxes", "classes", "mask_probs", "scores")


def dynamic_output_axes(names: tuple[str, ...] = ONNX_OUTPUTS) -> dict[str, dict[int, str]]:
    return {name: {0: "detections"} for name in names}


def reorder_outputs(path: str | Path, output_order: tuple[str, ...] = ONNX_OUTPUTS) -> Path:
    """在不改变张量值的前提下验证并重排 ONNX 图输出。"""
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX graph operations require the optional 'onnx' extra") from error
    model_path = Path(path)
    model = onnx.load(str(model_path))
    outputs = {item.name: item for item in model.graph.output}
    if set(outputs) != set(output_order):
        raise ValueError(f"unexpected ONNX outputs: {sorted(outputs)}")
    model.graph.ClearField("output")
    model.graph.output.extend(outputs[name] for name in output_order)
    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))
    return model_path
