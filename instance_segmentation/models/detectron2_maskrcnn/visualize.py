"""实例分割预测结果的可视化与 JSON 结果保存模块。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # 调色板和几何辅助函数在最小依赖环境中仍然可用
    cv2 = None

def _hsv_to_rgb(hue: int, saturation: int = 100, value: int = 100) -> tuple[int, int, int]:
    """精确匹配 HSV 转 RGB 颜色转换逻辑。"""
    rgb_max = np.float32(value) * np.float32(2.55)
    rgb_min = rgb_max * np.float32(100 - saturation) / np.float32(100.0)
    sector = hue // 60
    difference = hue % 60
    adjustment = (rgb_max - rgb_min) * np.float32(difference) / np.float32(60.0)
    if sector == 0:
        red, green, blue = rgb_max, rgb_min + adjustment, rgb_min
    elif sector == 1:
        red, green, blue = rgb_max - adjustment, rgb_max, rgb_min
    elif sector == 2:
        red, green, blue = rgb_min, rgb_max, rgb_min + adjustment
    elif sector == 3:
        red, green, blue = rgb_min, rgb_max - adjustment, rgb_max
    elif sector == 4:
        red, green, blue = rgb_min + adjustment, rgb_min, rgb_max
    else:
        red, green, blue = rgb_max, rgb_min, rgb_max - adjustment
    return int(red), int(green), int(blue)


def class_colors(class_count: int) -> list[tuple[int, int, int]]:
    """以 OpenCV 的 BGR 通道顺序生成参考调色板。"""
    maximum = max(1, int(class_count))
    # 参考实现返回 RGB；生成的图像使用 OpenCV BGR 顺序。
    return [tuple(reversed(_hsv_to_rgb(int(360.0 / maximum * index), 100, 100)))
            for index in range(max(0, int(class_count)))]


def class_color(class_id: int, class_name: str = "", class_count: int | None = None) -> tuple[int, int, int]:
    """返回稳定的类别索引颜色；保留 ``class_name`` 参数以保持 API 兼容。"""
    del class_name
    colors = class_colors(max(int(class_id) + 1, 1) if class_count is None else class_count)
    return colors[int(class_id) % len(colors)]


def draw_label(image, text: str, origin: tuple[int, int], color: tuple[int, int, int]):
    """在图像指定位置绘制带背景框的文本标签。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, box_y = origin
    x = max(0, min(x, image.shape[1] - text_width - 4))
    box_y = max(text_height + baseline + 2, min(box_y, image.shape[0] - 1))
    top = box_y - text_height - baseline - 2
    cv2.rectangle(image, (x, top), (x + text_width + 4, box_y), color, cv2.FILLED)
    cv2.putText(image, text, (x + 2, box_y - baseline - 1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def mask_angle(mask: np.ndarray) -> float | None:
    """返回 mask 长轴在图像坐标系中的角度。

    角度限制到 [0, 180)，因此箭头会始终指向图像下方，与原工具
    对无方向 mask 长轴的显示规则一致。
    """
    yx = np.argwhere(mask)
    if len(yx) < 2:
        return None
    xy = yx[:, ::-1].astype(np.float32)
    centered = xy - xy.mean(axis=0)
    covariance = centered.T @ centered
    axis = np.linalg.eigh(covariance)[1][:, -1]
    return float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)


def mask_center(mask: np.ndarray) -> tuple[int, int] | None:
    """返回 mask 像素的质心坐标，而不是检测框中心。"""
    yx = np.argwhere(mask)
    if len(yx) == 0:
        return None
    center_xy = yx[:, ::-1].mean(axis=0)
    return tuple(np.rint(center_xy).astype(int))


def arrow_points(
    box: np.ndarray,
    angle_degrees: float,
    *,
    center: tuple[int, int] | None = None,
    zero_at_up: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """以 mask 质心为起点、检测框长边一半为长度生成方向箭头。

    模型角度和 mask 主轴都采用图像坐标系 ``atan2(y, x)`` 约定，即向右
    为 0°、向下为 90°。保留 ``zero_at_up`` 仅兼容旧调用。
    """
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float32)
    if center is None:
        center_xy = np.asarray(((x0 + x1) / 2.0, (y0 + y1) / 2.0), dtype=np.float32)
    else:
        center_xy = np.asarray(center, dtype=np.float32)
    # 使用长边的一半，使箭头大致落在实例内部，避免从质心伸出检测框。
    length = 0.5 * max(float(x1 - x0), float(y1 - y0))
    radians = np.radians(angle_degrees)
    direction = (
        np.asarray((np.sin(radians), -np.cos(radians)), dtype=np.float32)
        if zero_at_up
        else np.asarray((np.cos(radians), np.sin(radians)), dtype=np.float32)
    )
    end = center_xy + length * direction
    return tuple(np.rint(center_xy).astype(int)), tuple(np.rint(end).astype(int))


def draw_predictions(
    image,
    instances,
    classes,
    class_thresholds=None,
    alpha=0.35,
    angle_classes=None,
    angle_source="prediction",
):
    """在图像上绘制实例分割预测结果（边界框、分割掩码、类别置信度及方向箭头）。"""
    output = image.copy()
    class_thresholds = class_thresholds or {}
    if len(instances) == 0:
        return output, []
    instances = instances.to("cpu")
    records = []
    arrows = []
    angle_classes = set(angle_classes or [])
    for index in range(len(instances)):
        class_id = int(instances.pred_classes[index])
        class_name = classes[class_id]
        score = float(instances.scores[index])
        if score < float(class_thresholds.get(class_name, 0.0)):
            continue
        box = instances.pred_boxes.tensor[index].numpy()
        mask = instances.pred_masks[index].numpy().astype(bool)
        color = class_color(class_id, class_name, len(classes))

        color_layer = output.copy()
        color_layer[mask] = color
        output = cv2.addWeighted(color_layer, alpha, output, 1.0 - alpha, 0)
        x0, y0, x1, y1 = np.rint(box).astype(int)
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
        draw_label(output, f"{class_name} {score:.2f}", (x0, y0), color)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_xy = [contour.reshape(-1, 2).tolist() for contour in contours if len(contour) >= 3]
        record = {
            "class_id": class_id,
            "class_name": class_name,
            "score": score,
            "bbox_xyxy": box.tolist(),
            "contours_xy": contours_xy,
            "mask_polygons": contours_xy,
        }
        if class_name in angle_classes:
            angle = None
            angle_score = None
            if angle_source == "mask":
                angle = mask_angle(mask)
            elif hasattr(instances, "pred_angles"):
                angle = float(instances.pred_angles[index])
                if hasattr(instances, "angle_scores"):
                    angle_score = float(instances.angle_scores[index])
            if angle is not None:
                center = mask_center(mask)
                if center is not None:
                    # 训练标签和 mask_angle 均以向右为 0°、向下为 90°。
                    # 不能把模型角度再转换成“正上方为 0°”，否则箭头会旋转 90°。
                    start, end = arrow_points(box, angle, center=center)
                    arrows.append((start, end))
                    record["angle_degrees"] = angle
                    if angle_score is not None:
                        record["angle_score"] = angle_score
        records.append(record)

    # 最后绘制，避免后续实例的半透明 mask 将箭头颜色冲淡。
    for start, end in arrows:
        cv2.arrowedLine(output, start, end, (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.25)
    return output, records


def save_json(
    path: str | Path,
    source: str | Path,
    visualization: str | Path,
    detections: list[dict],
    width: int | None = None,
    height: int | None = None,
):
    """将检测与分割结果保存为标准 JSON 格式。"""
    payload = {
        "file": str(Path(source).expanduser()),
        "imagePath": Path(source).name,
        "image": str(Path(source).expanduser()),
        "detections": detections,
        "visualization": str(Path(visualization).expanduser()),
    }
    if width is not None:
        payload["width"] = int(width)
    if height is not None:
        payload["height"] = int(height)
        payload["imageHeight"] = int(height)
    if width is not None:
        payload["imageWidth"] = int(width)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
