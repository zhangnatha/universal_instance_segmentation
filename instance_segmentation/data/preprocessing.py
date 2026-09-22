"""
Roboflow 流水线图像预处理模块
实现与 Roboflow UI 规范严格对齐的全部 12 种预处理技术：
1. 自动方向校正 (Auto-Orient)
2. 尺寸调整 (Resize: 拉伸、居中裁剪填充、自适应容纳、反射边缘填充、黑边填充、白边填充)
3. 灰度化 (Grayscale)
4. 自动对比度调整 (Auto-Adjust Contrast: 对比度拉伸、直方图均衡化、自适应均衡化)
5. 图像切片分块 (Tile: 网格行数 x 列数)
6. 静态裁剪 (Static Crop: 水平与垂直范围百分比)
7. 动态裁剪 (Dynamic Crop: 基于类别的感兴趣区域 ROI 裁剪)
8. 目标抠图隔离 (Isolate Objects: 提取边界框切片作为无标注的分类图像)
9. 类别修改 (Modify Classes: 正则重命名、包含/排除及强制重命名映射)
10. 空样本过滤 (Filter Null: 最小标注比例过滤)
11. 标签过滤 (Filter by Tag: 依据图像标签进行包含、排除或放行)
12. 随机采样 (Random Sample: 训练集/验证集/测试集子集抽样)
"""

import re
import copy
import base64
import random
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps

from .dataset import ImageSample, AnnotationItem


class PreprocessingPipeline:
    """可配置的预处理流水线，依次应用于图像及其对应的多边形标注。"""

    def __init__(self, steps: Optional[List[Dict[str, Any]]] = None):
        self.steps = steps or []

    def add_step(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        self.steps.append({"name": step_name, "config": config or {}})

    def clear(self):
        self.steps = []

    def process_sample(self, sample: ImageSample) -> List[ImageSample]:
        """
        按顺序对单个样本依次执行所有配置的预处理步骤。
        返回样本列表（部分步骤如切片分块或目标抠图会生成多个样本）。
        """
        current_samples = [self._clone_sample(sample)]

        for step in self.steps:
            name = step["name"]
            cfg = step.get("config", {})
            next_samples = []

            for s in current_samples:
                res = self._apply_step(s, name, cfg)
                if isinstance(res, list):
                    next_samples.extend(res)
                elif res is not None:
                    next_samples.append(res)
            current_samples = next_samples

        return current_samples

    def _clone_sample(self, sample: ImageSample) -> ImageSample:
        new_annotations = [
            AnnotationItem(
                label=ann.label,
                points=[list(p) for p in ann.points],
                shape_type=ann.shape_type,
                score=ann.score
            )
            for ann in sample.annotations
        ]
        new_sample = ImageSample(
            image_path=sample.image_path,
            width=sample.width,
            height=sample.height,
            annotations=new_annotations,
            sample_id=sample.sample_id,
            tags=list(getattr(sample, "tags", [])),
        )
        new_sample.split = sample.split
        if hasattr(sample, "classification_label"):
            new_sample.classification_label = sample.classification_label
        # 若修改过图像，则将内存中的 numpy 图像对象进行深拷贝缓存
        if hasattr(sample, "_cached_img"):
            new_sample._cached_img = sample._cached_img.copy()
        return new_sample

    def _get_image(self, sample: ImageSample) -> np.ndarray:
        if hasattr(sample, "_cached_img") and sample._cached_img is not None:
            return sample._cached_img.copy()
        img = cv2.imread(sample.image_path)
        if img is None:
            img = np.zeros((sample.height, sample.width, 3), dtype=np.uint8)
        return img

    @staticmethod
    def _clip_polygon(points: List[List[float]], x_min: float, y_min: float,
                      x_max: float, y_max: float) -> List[List[float]]:
        """将多边形裁剪到轴对齐矩形边界内。"""
        polygon = [[float(x), float(y)] for x, y in points]
        if len(polygon) < 3:
            return []

        def clip_edge(vertices, inside, intersect):
            if not vertices:
                return []
            result = []
            previous = vertices[-1]
            previous_inside = inside(previous)
            for current in vertices:
                current_inside = inside(current)
                if current_inside:
                    if not previous_inside:
                        result.append(intersect(previous, current))
                    result.append(current)
                elif previous_inside:
                    result.append(intersect(previous, current))
                previous = current
                previous_inside = current_inside
            return result

        def vertical_intersection(a, b, x_value):
            dx = b[0] - a[0]
            t = 0.0 if abs(dx) < 1e-12 else (x_value - a[0]) / dx
            return [x_value, a[1] + t * (b[1] - a[1])]

        def horizontal_intersection(a, b, y_value):
            dy = b[1] - a[1]
            t = 0.0 if abs(dy) < 1e-12 else (y_value - a[1]) / dy
            return [a[0] + t * (b[0] - a[0]), y_value]

        polygon = clip_edge(
            polygon, lambda p: p[0] >= x_min,
            lambda a, b: vertical_intersection(a, b, x_min)
        )
        polygon = clip_edge(
            polygon, lambda p: p[0] <= x_max,
            lambda a, b: vertical_intersection(a, b, x_max)
        )
        polygon = clip_edge(
            polygon, lambda p: p[1] >= y_min,
            lambda a, b: horizontal_intersection(a, b, y_min)
        )
        polygon = clip_edge(
            polygon, lambda p: p[1] <= y_max,
            lambda a, b: horizontal_intersection(a, b, y_max)
        )
        return polygon if len(polygon) >= 3 else []

    @staticmethod
    def _valid_annotation(label: str, points: List[List[float]], shape_type: str,
                          score: Optional[float] = None) -> Optional[AnnotationItem]:
        """校验多边形顶点数量及有效面积，构建合法的标注项。"""
        if len(points) < 3:
            return None
        area = abs(cv2.contourArea(np.asarray(points, dtype=np.float32)))
        if area <= 1e-6:
            return None
        return AnnotationItem(label=label, points=points, shape_type=shape_type, score=score)

    @staticmethod
    def _exif_transform_points(annotations: List[AnnotationItem], orientation: int,
                               width: int, height: int) -> Tuple[int, int]:
        """将 EXIF 方向变换应用到所有标注坐标上。"""
        swaps_dimensions = orientation in {5, 6, 7, 8}
        for ann in annotations:
            transformed = []
            for x, y in ann.points:
                if orientation == 2:
                    point = [width - 1 - x, y]
                elif orientation == 3:
                    point = [width - 1 - x, height - 1 - y]
                elif orientation == 4:
                    point = [x, height - 1 - y]
                elif orientation == 5:
                    point = [y, x]
                elif orientation == 6:
                    point = [height - 1 - y, x]
                elif orientation == 7:
                    point = [height - 1 - y, width - 1 - x]
                elif orientation == 8:
                    point = [y, width - 1 - x]
                else:
                    point = [x, y]
                transformed.append(point)
            ann.points = transformed
        return (height, width) if swaps_dimensions else (width, height)

    def _apply_step(self, sample: ImageSample, name: str, cfg: Dict[str, Any]) -> Optional[Any]:
        if name == "auto_orient":
            return self.apply_auto_orient(sample, cfg)
        elif name == "resize":
            return self.apply_resize(sample, cfg)
        elif name == "grayscale":
            return self.apply_grayscale(sample, cfg)
        elif name == "contrast":
            return self.apply_contrast(sample, cfg)
        elif name == "tile":
            return self.apply_tile(sample, cfg)
        elif name == "static_crop":
            return self.apply_static_crop(sample, cfg)
        elif name == "dynamic_crop":
            return self.apply_dynamic_crop(sample, cfg)
        elif name == "isolate_objects":
            return self.apply_isolate_objects(sample, cfg)
        elif name == "modify_classes":
            return self.apply_modify_classes(sample, cfg)
        elif name == "filter_null":
            return self.apply_filter_null(sample, cfg)
        elif name == "filter_tag":
            return self.apply_filter_tag(sample, cfg)
        elif name == "random_sample":
            return self.apply_random_sample(sample, cfg)
        else:
            return sample

    # 1. 自动方向校正 (Auto-Orient)
    def apply_auto_orient(self, sample: ImageSample, cfg: Dict[str, Any]) -> ImageSample:
        """根据 EXIF 方向校正图像朝向，并同步变换所有标注坐标点。"""
        try:
            pil_img = Image.open(sample.image_path)
            exif = pil_img.getexif()
            orientation = int(exif.get(274, 1)) if exif else 1
            original_width, original_height = pil_img.size
            trans_img = ImageOps.exif_transpose(pil_img)
            raw = np.array(trans_img)
            if raw.ndim == 2:
                np_img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            elif raw.shape[2] == 4:
                np_img = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
            else:
                np_img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            new_width, new_height = self._exif_transform_points(
                sample.annotations, orientation, original_width, original_height
            )
            sample._cached_img = np_img
            sample.width, sample.height = new_width, new_height
            # 缓存的 OpenCV 图像写入时已去除源 EXIF 方向标签，以便下游组件直接使用标准像素。
        except Exception:
            pass
        return sample

    # 2. 尺寸调整 (Resize)
    def apply_resize(self, sample: ImageSample, cfg: Dict[str, Any]) -> ImageSample:
        mode = cfg.get("mode", "stretch")
        mode = {
            "fill": "fill_crop",
            "fit": "fit_black",
        }.get(mode, mode)
        target_w = max(1, int(cfg.get("width", 640)))
        target_h = max(1, int(cfg.get("height", 640)))

        img = self._get_image(sample)
        orig_h, orig_w = img.shape[:2]

        if orig_w <= 0 or orig_h <= 0:
            return sample

        if mode == "stretch":
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # 变换坐标点
            for ann in sample.annotations:
                points = [[p[0] * scale_x, p[1] * scale_y] for p in ann.points]
                clipped = self._clip_polygon(points, 0, 0, target_w - 1, target_h - 1)
                ann.points = clipped or points

        elif mode == "fill_crop":
            # 等比例缩放使较短边与目标尺寸对齐，然后居中裁剪多余区域
            scale = max(target_w / orig_w, target_h / orig_h)
            nw = max(1, int(round(orig_w * scale)))
            nh = max(1, int(round(orig_h * scale)))
            scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

            # 居中裁剪偏移量
            dx = (nw - target_w) // 2
            dy = (nh - target_h) // 2
            resized_img = scaled[dy:dy + target_h, dx:dx + target_w]

            # 将多边形顶点变换并裁剪至裁剪后的画布内
            new_annots = []
            for ann in sample.annotations:
                pts = [[p[0] * scale - dx, p[1] * scale - dy] for p in ann.points]
                clipped_pts = self._clip_polygon(pts, 0, 0, target_w - 1, target_h - 1)
                clipped_ann = self._valid_annotation(ann.label, clipped_pts, ann.shape_type, ann.score)
                if clipped_ann is not None:
                    new_annots.append(clipped_ann)
            sample.annotations = new_annots

        elif mode == "fit_within":
            # 保持宽高比且不进行边缘填充。生成的图像在一维上可能小于目标画布
            scale = min(target_w / orig_w, target_h / orig_h)
            nw = max(1, int(round(orig_w * scale)))
            nh = max(1, int(round(orig_h * scale)))
            resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            for ann in sample.annotations:
                ann.points = [[p[0] * scale, p[1] * scale] for p in ann.points]

        elif mode in ["fit_reflect", "fit_black", "fit_white"]:
            # 边缘填充等比缩放
            scale = min(target_w / orig_w, target_h / orig_h)
            nw = max(1, int(round(orig_w * scale)))
            nh = max(1, int(round(orig_h * scale)))
            scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

            dx = (target_w - nw) // 2
            dy = (target_h - nh) // 2

            if mode == "fit_white":
                canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
            elif mode == "fit_reflect":
                canvas = cv2.copyMakeBorder(
                    scaled, dy, target_h - nh - dy, dx, target_w - nw - dx,
                    cv2.BORDER_REFLECT_101
                )
            else:  # fit_within 或 fit_black
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            if mode != "fit_reflect":
                canvas[dy:dy + nh, dx:dx + nw] = scaled
            resized_img = canvas

            if mode == "fit_reflect":
                # BORDER_REFLECT_101 会在填充区域生成反射的目标像素。
                # Roboflow 会将标注镜像到反射区域，因此为每条边创建裁剪副本。
                new_annots = []
                x_modes = [("identity", dx, dx + nw - 1)]
                y_modes = [("identity", dy, dy + nh - 1)]
                if dx > 0:
                    x_modes.append(("left", 0, dx - 1))
                if target_w - (dx + nw) > 0:
                    x_modes.append(("right", dx + nw, target_w - 1))
                if dy > 0:
                    y_modes.append(("top", 0, dy - 1))
                if target_h - (dy + nh) > 0:
                    y_modes.append(("bottom", dy + nh, target_h - 1))

                for ann in sample.annotations:
                    source_points = [[p[0] * scale, p[1] * scale] for p in ann.points]
                    for x_mode, x_clip_min, x_clip_max in x_modes:
                        for y_mode, y_clip_min, y_clip_max in y_modes:
                            transformed = []
                            for x, y in source_points:
                                if x_mode == "identity":
                                    nx = x + dx
                                elif x_mode == "left":
                                    nx = dx - x
                                else:  # 右边界反射 (right)
                                    nx = 2.0 * (dx + nw - 1) - (x + dx)

                                if y_mode == "identity":
                                    ny = y + dy
                                elif y_mode == "top":
                                    ny = dy - y
                                else:  # 下边界反射 (bottom)
                                    ny = 2.0 * (dy + nh - 1) - (y + dy)
                                transformed.append([nx, ny])

                            clipped = self._clip_polygon(
                                transformed,
                                x_clip_min,
                                y_clip_min,
                                x_clip_max,
                                y_clip_max,
                            )
                            clipped_ann = self._valid_annotation(
                                ann.label, clipped, ann.shape_type, ann.score
                            )
                            if clipped_ann is not None:
                                new_annots.append(clipped_ann)
                sample.annotations = new_annots
            else:
                for ann in sample.annotations:
                    points = [[p[0] * scale + dx, p[1] * scale + dy] for p in ann.points]
                    ann.points = self._clip_polygon(points, 0, 0, target_w - 1, target_h - 1) or points
        else:
            resized_img = cv2.resize(img, (target_w, target_h))
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            for ann in sample.annotations:
                ann.points = [[p[0] * scale_x, p[1] * scale_y] for p in ann.points]

        sample._cached_img = resized_img
        sample.height, sample.width = resized_img.shape[:2]
        return sample

    # 3. 灰度化 (Grayscale)
    def apply_grayscale(self, sample: ImageSample, cfg: Dict[str, Any]) -> ImageSample:
        img = self._get_image(sample)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Roboflow 采用 RGB 亮度权重计算灰度。缓存的 OpenCV
            # 图像为 BGR 格式，因此显式对 R/G/B 分别应用权重。
            gray = np.clip(
                img[:, :, 2].astype(np.float32) * 0.2125
                + img[:, :, 1].astype(np.float32) * 0.7154
                + img[:, :, 0].astype(np.float32) * 0.0721,
                0,
                255,
            ).astype(np.uint8)
            sample._cached_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return sample

    # 4. 自动对比度调整 (Auto-Adjust Contrast)
    def apply_contrast(self, sample: ImageSample, cfg: Dict[str, Any]) -> ImageSample:
        ctype = cfg.get("type", "adaptive")  # 可选类型：stretching（对比度拉伸）、histogram（直方图均衡化）、adaptive（自适应均衡化）
        img = self._get_image(sample)

        def adjust_channel(channel: np.ndarray) -> np.ndarray:
            channel = np.asarray(channel, dtype=np.uint8)
            if ctype == "stretching":
                # 与 UI 定义对齐：使用第 2 和第 98 百分位数进行拉伸。
                p2, p98 = np.percentile(channel, (2, 98))
                if p98 > p2:
                    return np.clip((channel.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
                return channel.copy()
            if ctype == "histogram":
                return cv2.equalizeHist(channel)
            if ctype == "adaptive":
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(channel)
            raise ValueError(f"Unsupported contrast type: {ctype}")

        if img.ndim == 2:
            sample._cached_img = adjust_channel(img)
        elif img.ndim == 3 and img.shape[2] == 1:
            sample._cached_img = adjust_channel(img[:, :, 0])[:, :, None]
        elif img.ndim == 3:
            # 在 LAB 颜色空间中仅调整亮度通道 L，以保持色调与饱和度不变。
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = adjust_channel(l_channel)
            merged_lab = cv2.merge((l_channel, a_channel, b_channel))
            sample._cached_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        return sample

    # 5. 图像切片分块 (Tile)
    def apply_tile(self, sample: ImageSample, cfg: Dict[str, Any]) -> List[ImageSample]:
        # UI 规范中顺序为 行数 x 列数（例如 2x3 表示 2 行 3 列）。
        # 读取旧版 grid_y/grid_x 键以保持向后兼容性。
        grid_rows = max(1, int(cfg.get("grid_rows", cfg.get("grid_y", 2))))
        grid_cols = max(1, int(cfg.get("grid_cols", cfg.get("grid_x", 2))))
        if grid_rows <= 1 and grid_cols <= 1:
            return [sample]

        img = self._get_image(sample)
        h, w = img.shape[:2]
        grid_cols = min(grid_cols, max(1, w))
        grid_rows = min(grid_rows, max(1, h))
        tile_w = w // grid_cols
        tile_h = h // grid_rows

        tiles = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                x1 = col * tile_w
                y1 = row * tile_h
                x2 = w if col == grid_cols - 1 else (col + 1) * tile_w
                y2 = h if row == grid_rows - 1 else (row + 1) * tile_h

                tile_img = img[y1:y2, x1:x2].copy()
                th, tw = tile_img.shape[:2]

                tile_annots = []
                for ann in sample.annotations:
                    translated = [[p[0] - x1, p[1] - y1] for p in ann.points]
                    new_pts = self._clip_polygon(translated, 0, 0, tw - 1, th - 1)
                    clipped_ann = self._valid_annotation(ann.label, new_pts, ann.shape_type, ann.score)
                    if clipped_ann is not None:
                        tile_annots.append(clipped_ann)

                tile_sample = ImageSample(
                    image_path=sample.image_path,
                    width=tw,
                    height=th,
                    annotations=tile_annots,
                    sample_id=f"{sample.sample_id}_t_{col}_{row}",
                    tags=list(getattr(sample, "tags", [])),
                )
                tile_sample._cached_img = tile_img
                tile_sample.split = sample.split
                tiles.append(tile_sample)

        return tiles

    # 6. 静态裁剪 (Static Crop)
    def apply_static_crop(self, sample: ImageSample, cfg: Dict[str, Any]) -> ImageSample:
        h_min = float(np.clip(cfg.get("h_min", 25.0), 0, 100)) / 100.0
        h_max = float(np.clip(cfg.get("h_max", 75.0), 0, 100)) / 100.0
        v_min = float(np.clip(cfg.get("v_min", 25.0), 0, 100)) / 100.0
        v_max = float(np.clip(cfg.get("v_max", 75.0), 0, 100)) / 100.0
        if h_min >= h_max or v_min >= v_max:
            return sample

        img = self._get_image(sample)
        h, w = img.shape[:2]

        # 水平值对应左右边界，垂直值对应上下边界。UI 参数基于百分比，
        # 不附加无关的最小裁剪尺寸限制。
        x1 = int(round(w * h_min))
        x2 = int(round(w * h_max))
        y1 = int(round(h * v_min))
        y2 = int(round(h * v_max))

        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return sample

        cropped_img = img[y1:y2, x1:x2].copy()
        cw, ch = cropped_img.shape[1], cropped_img.shape[0]

        new_annots = []
        for ann in sample.annotations:
            pts = [[p[0] - x1, p[1] - y1] for p in ann.points]
            clipped = self._clip_polygon(pts, 0, 0, cw - 1, ch - 1)
            clipped_ann = self._valid_annotation(ann.label, clipped, ann.shape_type, ann.score)
            if clipped_ann is not None:
                new_annots.append(clipped_ann)

        sample._cached_img = cropped_img
        sample.width = cw
        sample.height = ch
        sample.annotations = new_annots
        return sample

    # 7. 动态裁剪 (Dynamic Crop)
    def apply_dynamic_crop(self, sample: ImageSample, cfg: Dict[str, Any]) -> Optional[ImageSample]:
        # UI 指定单个选中的 ROI 目标类别。保持对旧版 {"classes": [label]}
        # 的兼容支持，但绝不将多个不同类别合并为一个目标。
        target_class = cfg.get("class")
        if not target_class:
            configured_classes = cfg.get("classes") or []
            if isinstance(configured_classes, str):
                configured_classes = [configured_classes]
            if len(configured_classes) == 1:
                target_class = configured_classes[0]
        if not target_class:
            return None

        matching = [ann for ann in sample.annotations if ann.label == target_class]
        if len(matching) != 1:
            return None

        img = self._get_image(sample)
        h, w = img.shape[:2]

        bbox = matching[0].to_bbox()
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        padding = max(0.0, float(cfg.get("padding_percent", 0.0))) / 100.0
        pad_x = bw * padding
        pad_y = bh * padding

        x1 = max(0, int(np.floor(bbox[0] - pad_x)))
        y1 = max(0, int(np.floor(bbox[1] - pad_y)))
        x2 = min(w, int(np.ceil(bbox[2] + pad_x)) + 1)
        y2 = min(h, int(np.ceil(bbox[3] + pad_y)) + 1)

        if x2 <= x1 or y2 <= y1:
            return sample

        cropped = img[y1:y2, x1:x2].copy()
        cw, ch = cropped.shape[1], cropped.shape[0]

        new_annots = []
        for ann in sample.annotations:
            pts = [[p[0] - x1, p[1] - y1] for p in ann.points]
            clipped = self._clip_polygon(pts, 0, 0, cw - 1, ch - 1)
            clipped_ann = self._valid_annotation(ann.label, clipped, ann.shape_type, ann.score)
            if clipped_ann is not None:
                new_annots.append(clipped_ann)

        sample._cached_img = cropped
        sample.width = cw
        sample.height = ch
        sample.annotations = new_annots
        return sample

    # 8. 目标抠图隔离 (Isolate Objects)
    def apply_isolate_objects(self, sample: ImageSample, cfg: Dict[str, Any]) -> List[ImageSample]:
        """为每个目标提取一个边界框切片，并清除所有检测标注。

        Roboflow 的 Isolate Objects 功能将检测/分割数据集转换为分类任务切片。
        源类别保留在样本元数据中以便按类别目录导出，但生成的样本本身不再保留检测框或多边形。
        """
        if not sample.annotations:
            return []

        img = self._get_image(sample)
        h, w = img.shape[:2]
        isolated_samples = []

        for idx, ann in enumerate(sample.annotations):
            bbox = ann.to_bbox()
            # 像素坐标在远端角落是闭区间；采用 ceil + 1 避免紧贴边界的目标被截断 1 个像素。
            x1 = max(0, int(np.floor(bbox[0])))
            y1 = max(0, int(np.floor(bbox[1])))
            x2 = min(w, int(np.ceil(bbox[2])) + 1)
            y2 = min(h, int(np.ceil(bbox[3])) + 1)

            if x2 <= x1 or y2 <= y1:
                continue

            obj_img = img[y1:y2, x1:x2].copy()
            ow, oh = obj_img.shape[1], obj_img.shape[0]

            iso_s = ImageSample(
                image_path=sample.image_path,
                width=ow,
                height=oh,
                annotations=[],
                sample_id=f"{sample.sample_id}_obj_{idx}",
                tags=list(getattr(sample, "tags", [])),
            )
            iso_s._cached_img = obj_img
            iso_s.split = sample.split
            iso_s.classification_label = ann.label
            isolated_samples.append(iso_s)

        return isolated_samples

    # 9. 类别修改 (Modify Classes)
    def apply_modify_classes(self, sample: ImageSample, cfg: Dict[str, Any]) -> Optional[ImageSample]:
        include_classes = cfg.get("include_classes", {})  # 字典格式 {class_name: bool}
        rename_map = cfg.get("rename_map", {})  # 字典格式 {class_name: new_name}
        regex_pattern = cfg.get("regex_pattern", "")
        regex_replace = cfg.get("regex_replace", "")

        # Isolate Objects 将源类别存储为元数据，因为其输出按设计不包含标注。
        # 类别过滤与重命名同样应用于该元数据。
        if hasattr(sample, "classification_label"):
            label = sample.classification_label
            if include_classes and not include_classes.get(label, False):
                return None
            if regex_pattern:
                label = re.sub(regex_pattern, regex_replace, label)
            if label in rename_map and rename_map[label].strip():
                label = rename_map[label].strip()
            sample.classification_label = label
            return sample

        new_annots = []
        for ann in sample.annotations:
            # 检查是否包含
            if include_classes and not include_classes.get(ann.label, False):
                continue

            label = ann.label
            # 正则表达式替换
            if regex_pattern:
                label = re.sub(regex_pattern, regex_replace, label)

            # 显式重命名映射
            if label in rename_map and rename_map[label].strip():
                label = rename_map[label].strip()

            new_ann = AnnotationItem(label=label, points=ann.points, shape_type=ann.shape_type, score=ann.score)
            new_annots.append(new_ann)

        sample.annotations = new_annots
        return sample

    # 10. 空样本过滤 (Filter Null)
    def apply_filter_null(self, sample: ImageSample, cfg: Dict[str, Any]) -> Optional[ImageSample]:
        if hasattr(sample, "classification_label"):
            return sample
        min_percent = float(np.clip(cfg.get("min_percent", 100.0), 0, 100))  # 0% 保留全部，100% 过滤所有无标注空样本
        if sample.is_null:
            # 以 min_percent / 100 的概率丢弃空样本
            if random.random() < (min_percent / 100.0):
                return None
        return sample

    # 11. 标签过滤 (Filter by Tag)
    def apply_filter_tag(self, sample: ImageSample, cfg: Dict[str, Any]) -> Optional[ImageSample]:
        """应用类似 Roboflow 的 Require（必须包含）、Exclude（必须排除）和 Allow（允许包含）标签过滤规则。"""
        tags = set(getattr(sample, "tags", []))
        rules = cfg.get("rules", [])
        required = {rule["tag"] for rule in rules if rule.get("mode") == "require"}
        excluded = {rule["tag"] for rule in rules if rule.get("mode") == "exclude"}
        allowed = {rule["tag"] for rule in rules if rule.get("mode") == "allow"}

        if required and not required.issubset(tags):
            return None
        if excluded.intersection(tags):
            return None
        if allowed and not allowed.intersection(tags):
            return None
        return sample

    # 12. 随机采样 (Random Sample)
    def apply_random_sample(self, sample: ImageSample, cfg: Dict[str, Any]) -> Optional[ImageSample]:
        train_p = float(np.clip(cfg.get("train_percent", 100.0), 0, 100)) / 100.0
        valid_p = float(np.clip(cfg.get("valid_percent", 100.0), 0, 100)) / 100.0
        test_p = float(np.clip(cfg.get("test_percent", 100.0), 0, 100)) / 100.0

        p = train_p if sample.split == "train" else (valid_p if sample.split == "valid" else test_p)
        if random.random() > p:
            return None
        return sample


def preview_preprocessing(sample: ImageSample, step_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """为指定的预处理步骤生成处理前后的 base64 缩略图预览。"""
    pipeline = PreprocessingPipeline()
    pipeline.add_step(step_name, config)

    # 渲染原图
    orig_img = cv2.imread(sample.image_path)
    if orig_img is None:
        orig_img = np.zeros((sample.height, sample.width, 3), dtype=np.uint8)

    # 执行预处理
    cloned = pipeline._clone_sample(sample)
    results = pipeline.process_sample(cloned)
    processed_sample = results[0] if results else cloned
    proc_img = pipeline._get_image(processed_sample)

    def encode_b64(img_np: np.ndarray) -> str:
        # 调整预览缩略图尺寸
        th = 320
        h, w = img_np.shape[:2]
        tw = int(round(w * (th / max(1, h))))
        thumb = cv2.resize(img_np, (tw, th))
        _, buf = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buf).decode('utf-8')

    return {
        "original": f"data:image/jpeg;base64,{encode_b64(orig_img)}",
        "processed": f"data:image/jpeg;base64,{encode_b64(proc_img)}",
        "step": step_name,
        "config": config
    }
