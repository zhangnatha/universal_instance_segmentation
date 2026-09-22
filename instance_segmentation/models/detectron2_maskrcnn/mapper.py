"""Detectron2 数据集映射器（Dataset Mapper），支持数据增强、聚焦裁剪、Copy-Paste 与方向标签生成。"""

from __future__ import annotations

import copy

import numpy as np
import torch

from detectron2.data import detection_utils as utils, transforms as T
from fvcore.transforms.transform import CropTransform, NoOpTransform, Transform

from instance_segmentation.data.labelme import angle_to_bin, polygon_angle


def _bbox_iou(left, right):
    """计算两个边界框之间的交并比（IoU）。"""
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0.0, min(ly1, ry1) - max(ly0, ry0)
    )
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _annotation_mask(annotation, height, width):
    """根据标注多边形生成二值掩码数组。"""
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in annotation.get("segmentation") or []:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


class LowResolutionTransform(Transform):
    """只降低图像清晰度，不改变标注坐标。"""

    def __init__(self, scale: float):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, image, interp=None):
        import cv2

        h, w = image.shape[:2]
        small = cv2.resize(image, (max(1, round(w * self.scale)), max(1, round(h * self.scale))))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class RandomLowResolution(T.Augmentation):
    """随机低分辨率数据增强。"""
    def __init__(self, scale=0.6, probability=0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        if np.random.random() >= self.probability:
            return NoOpTransform()
        return LowResolutionTransform(float(self.scale))


class TranslationTransform(Transform):
    """平移图像和实例坐标，保持输出尺寸不变。"""

    def __init__(self, dx: float, dy: float):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, image, interp=None):
        import cv2

        h, w = image.shape[:2]
        matrix = np.asarray([[1.0, 0.0, self.dx], [0.0, 1.0, self.dy]], dtype=np.float32)
        return cv2.warpAffine(
            image, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def apply_coords(self, coords):
        return coords + np.asarray([self.dx, self.dy], dtype=np.float32)

    def apply_segmentation(self, segmentation):
        import cv2

        h, w = segmentation.shape[:2]
        matrix = np.asarray([[1.0, 0.0, self.dx], [0.0, 1.0, self.dy]], dtype=np.float32)
        return cv2.warpAffine(
            segmentation, matrix, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def inverse(self):
        return TranslationTransform(-self.dx, -self.dy)


class RandomTranslation(T.Augmentation):
    """随机平移数据增强。"""
    def __init__(self, min_fraction=-0.2, max_fraction=0.2, probability=0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        if np.random.random() >= self.probability:
            return NoOpTransform()
        h, w = image.shape[:2]
        dx = float(np.random.uniform(self.min_fraction, self.max_fraction) * w)
        dy = float(np.random.uniform(self.min_fraction, self.max_fraction) * h)
        return TranslationTransform(dx, dy)


class CLAHETransform(Transform):
    """自适应局部直方图均衡化（在 LAB 颜色空间的 L 通道处理，保留色度）。"""

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        image_format: str = "BGR",
    ):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, image, interp=None):
        import cv2

        if image.ndim != 3 or image.shape[2] != 3:
            return image
        to_lab = cv2.COLOR_RGB2LAB if self.image_format == "RGB" else cv2.COLOR_BGR2LAB
        from_lab = cv2.COLOR_LAB2RGB if self.image_format == "RGB" else cv2.COLOR_LAB2BGR
        lab = cv2.cvtColor(image, to_lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, from_lab)

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class RandomCLAHE(T.Augmentation):
    """随机 CLAHE 直方图均衡化数据增强。"""
    def __init__(
        self,
        clip_limit: float = 2.0,
        probability: float = 0.3,
        image_format: str = "BGR",
    ):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        if np.random.random() >= self.probability:
            return NoOpTransform()
        return CLAHETransform(self.clip_limit, image_format=self.image_format)


class RandomErasingTransform(Transform):
    """随机小矩形擦除（模拟栏杆、管路遮挡）。"""

    def __init__(self, s_min: float = 0.01, s_max: float = 0.04, r_min: float = 0.3, r_max: float = 3.3):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, image, interp=None):
        h, w = image.shape[:2]
        area = h * w
        target_area = float(np.random.uniform(self.s_min, self.s_max) * area)
        aspect_ratio = float(np.random.uniform(self.r_min, self.r_max))
        eh = int(round(np.sqrt(target_area * aspect_ratio)))
        ew = int(round(np.sqrt(target_area / aspect_ratio)))
        if eh < h and ew < w:
            y1 = int(np.random.randint(0, h - eh))
            x1 = int(np.random.randint(0, w - ew))
            image = np.array(image, copy=True)
            color = np.random.randint(30, 80, (3,), dtype=image.dtype)
            image[y1 : y1 + eh, x1 : x1 + ew] = color
        return image

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class RandomErasing(T.Augmentation):
    """随机矩形擦除数据增强。"""
    def __init__(self, s_min: float = 0.01, s_max: float = 0.04, probability: float = 0.25):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        if np.random.random() >= self.probability:
            return NoOpTransform()
        return RandomErasingTransform(self.s_min, self.s_max)


def build_augmentations(cfg, enabled: bool):
    """根据配置构建训练或评估期的数据增强管线。"""
    augmentations = [T.ResizeShortestEdge(
        cfg.INPUT.MIN_SIZE_TRAIN,
        cfg.INPUT.MAX_SIZE_TRAIN,
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
    )]
    if enabled:
        if cfg.INPUT.RANDOM_FLIP == "horizontal":
            augmentations.append(T.RandomFlip(horizontal=True, vertical=False))
        elif cfg.INPUT.RANDOM_FLIP == "vertical":
            augmentations.append(T.RandomFlip(horizontal=False, vertical=True))
        elif cfg.INPUT.RANDOM_FLIP == "both":
            augmentations.append(T.RandomFlip(horizontal=True, vertical=True))
        if cfg.AUGMENT_ROTATION and cfg.AUGMENT_ROTATION_PROB > 0:
            augmentations.append(T.RandomApply(
                T.RandomRotation(angle=list(cfg.AUGMENT_ROTATION_RANGE), expand=False),
                prob=cfg.AUGMENT_ROTATION_PROB,
            ))
        if cfg.AUGMENT_BRIGHTNESS and cfg.AUGMENT_BRIGHTNESS_PROB > 0:
            augmentations.append(T.RandomApply(
                T.RandomBrightness(*cfg.AUGMENT_BRIGHTNESS_RANGE),
                prob=cfg.AUGMENT_BRIGHTNESS_PROB,
            ))
        if cfg.AUGMENT_CONTRAST and cfg.AUGMENT_CONTRAST_PROB > 0:
            augmentations.append(T.RandomApply(
                T.RandomContrast(*cfg.AUGMENT_CONTRAST_RANGE),
                prob=cfg.AUGMENT_CONTRAST_PROB,
            ))
        if getattr(cfg, "AUGMENT_CLAHE", False) and getattr(cfg, "AUGMENT_CLAHE_PROB", 0.0) > 0:
            augmentations.append(RandomCLAHE(
                clip_limit=getattr(cfg, "AUGMENT_CLAHE_CLIP_LIMIT", 2.0),
                probability=getattr(cfg, "AUGMENT_CLAHE_PROB", 0.3),
                image_format=getattr(cfg.INPUT, "FORMAT", "BGR"),
            ))
        if getattr(cfg, "AUGMENT_ERASING", False) and getattr(cfg, "AUGMENT_ERASING_PROB", 0.0) > 0:
            augmentations.append(RandomErasing(
                s_min=getattr(cfg, "AUGMENT_ERASING_SMIN", 0.01),
                s_max=getattr(cfg, "AUGMENT_ERASING_SMAX", 0.04),
                probability=getattr(cfg, "AUGMENT_ERASING_PROB", 0.25),
            ))
        if cfg.AUGMENT_TRANSLATION and cfg.AUGMENT_TRANSLATION_PROB > 0:
            augmentations.append(RandomTranslation(
                *cfg.AUGMENT_TRANSLATION_RANGE,
                probability=cfg.AUGMENT_TRANSLATION_PROB,
            ))
        if cfg.AUGMENT_LOW_RESOLUTION and cfg.AUGMENT_LOW_RESOLUTION_PROB > 0:
            augmentations.append(RandomLowResolution(
                cfg.AUGMENT_LOW_RESOLUTION_SCALE,
                probability=cfg.AUGMENT_LOW_RESOLUTION_PROB,
            ))
    return augmentations


class AngleDatasetMapper:
    """Detectron2 mapper，并从增强后的多边形自动生成方向标签。"""

    def __init__(self, cfg, *, is_train: bool, augment: bool = False):
        self.is_train = is_train
        self.angle_bins = int(cfg.ANGLE_BINS)
        self.angle_period = float(cfg.ANGLE_PERIOD)
        self.angle_label_source = str(getattr(cfg, "ANGLE_LABEL_SOURCE", "annotation"))
        self.angle_class_ids = frozenset(int(v) for v in getattr(cfg, "ANGLE_CLASS_IDS", ()))
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.image_format = cfg.INPUT.FORMAT
        self.augmentations = T.AugmentationList(
            build_augmentations(cfg, augment) if is_train else []
        )
        self.focus_crop_class_id = int(getattr(cfg, "FOCUS_CROP_CLASS_ID", -1))
        self.focus_crop_class_ids = tuple(getattr(cfg, "FOCUS_CROP_CLASS_IDS", ()))
        if not self.focus_crop_class_ids and self.focus_crop_class_id >= 0:
            self.focus_crop_class_ids = (self.focus_crop_class_id,)
        self.focus_crop_prob = float(getattr(cfg, "FOCUS_CROP_PROB", 0.0))
        self.focus_crop_scale = float(getattr(cfg, "FOCUS_CROP_SCALE", 4.0))
        self.focus_crop_min_size = tuple(getattr(cfg, "FOCUS_CROP_MIN_SIZE", (320, 240)))
        self.copy_paste_enabled = bool(getattr(cfg, "AUGMENT_COPY_PASTE", False)) and is_train
        self.copy_paste_probability = float(getattr(cfg, "AUGMENT_COPY_PASTE_PROB", 0.25))
        self.copy_paste_class_ids = tuple(getattr(cfg, "AUGMENT_COPY_PASTE_CLASS_IDS", ()))
        self.copy_paste_max_instances = int(getattr(cfg, "AUGMENT_COPY_PASTE_MAX_INSTANCES", 2))
        self.copy_paste_max_bbox_iou = float(getattr(cfg, "AUGMENT_COPY_PASTE_MAX_BBOX_IOU", 0.05))
        self.copy_paste_max_attempts = max(4, self.copy_paste_max_instances * 8)
        self.copy_paste_records = []
        self.copy_paste_candidates = []
        if self.copy_paste_enabled and self.copy_paste_class_ids:
            from detectron2.data import DatasetCatalog

            train_names = tuple(getattr(cfg.DATASETS, "TRAIN", ()))
            if train_names:
                # 仅从 DATASETS.TRAIN 中解析样本；验证集/测试集目录绝不会被此增强方式使用。
                self.copy_paste_records = DatasetCatalog.get(train_names[0])
                self.copy_paste_candidates = [
                    (record, annotation)
                    for record in self.copy_paste_records
                    for annotation in record.get("annotations") or []
                    if int(annotation.get("category_id", -1)) in self.copy_paste_class_ids
                ]

    def _copy_paste(self, dataset_dict, image):
        """仅从训练集目录中复制带边界框和多边形掩码的实例并粘贴到当前图像。"""
        if (
            not self.copy_paste_enabled
            or not self.copy_paste_candidates
            or self.copy_paste_probability <= 0
            or np.random.random() >= self.copy_paste_probability
            or self.copy_paste_max_instances <= 0
        ):
            return image
        height, width = image.shape[:2]
        existing = list(dataset_dict.get("annotations") or [])
        cur_file = dataset_dict.get("file_name")
        candidates = [
            cand for cand in self.copy_paste_candidates
            if cand[0].get("file_name") != cur_file
        ] or self.copy_paste_candidates
        if not candidates:
            return image
        # Detectron2 的序列化数据集 worker 可能会返回由只读缓冲区支持的 OpenCV 数组。
        # 仅在数据增强路径上进行数据拷贝。
        image = np.array(image, copy=True)
        sample_count = min(len(candidates), self.copy_paste_max_attempts)
        sample_indices = np.random.choice(len(candidates), size=sample_count, replace=False)
        pasted = 0
        for idx in sample_indices:
            if pasted >= self.copy_paste_max_instances:
                break
            source_record, source_annotation = candidates[idx]
            source_image = utils.read_image(source_record["file_name"], format=self.image_format)
            source_height, source_width = source_image.shape[:2]
            mask = _annotation_mask(source_annotation, source_height, source_width)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            sx0, sx1 = int(xs.min()), int(xs.max()) + 1
            sy0, sy1 = int(ys.min()), int(ys.max()) + 1
            crop_width, crop_height = sx1 - sx0, sy1 - sy0
            if crop_width > width or crop_height > height:
                continue
            patch = source_image[sy0:sy1, sx0:sx1]
            patch_mask = mask[sy0:sy1, sx0:sx1]
            for _ in range(self.copy_paste_max_attempts):
                x0 = int(np.random.randint(0, width - crop_width + 1))
                y0 = int(np.random.randint(0, height - crop_height + 1))
                new_bbox = (x0, y0, x0 + crop_width, y0 + crop_height)
                if all(
                    _bbox_iou(new_bbox, annotation.get("bbox", (0, 0, 0, 0)))
                    <= self.copy_paste_max_bbox_iou
                    for annotation in existing
                ):
                    break
            else:
                continue
            image[y0:y0 + crop_height, x0:x0 + crop_width][patch_mask] = patch[patch_mask]
            transformed = copy.deepcopy(source_annotation)
            transformed["bbox"] = [
                float(x0 + float(source_annotation["bbox"][0]) - sx0),
                float(y0 + float(source_annotation["bbox"][1]) - sy0),
                float(x0 + float(source_annotation["bbox"][2]) - sx0),
                float(y0 + float(source_annotation["bbox"][3]) - sy0),
            ]
            transformed["segmentation"] = [
                [
                    float(value + (x0 - sx0 if index % 2 == 0 else y0 - sy0))
                    for index, value in enumerate(polygon)
                ]
                for polygon in transformed.get("segmentation") or []
            ]
            existing.append(transformed)
            pasted += 1
        dataset_dict["annotations"] = existing
        return image

    def _focus_crop(self, dataset_dict, image):
        """生成保留全图训练的低比例特定目标局部聚焦放大样本。

        只保留裁剪中基本完整的实例，避免把被截断的大目标作为错误监督；
        目标类别和概率均为显式配置，默认完全关闭。
        """
        if not self.focus_crop_class_ids or self.focus_crop_prob <= 0:
            return image
        if np.random.random() >= self.focus_crop_prob:
            return image
        annotations = dataset_dict.get("annotations") or []
        candidates = [
            annotation for annotation in annotations
            if int(annotation.get("category_id", -1)) in self.focus_crop_class_ids
        ]
        if not candidates:
            return image

        target = candidates[np.random.randint(len(candidates))]
        x0, y0, x1, y1 = map(float, target["bbox"])
        height, width = image.shape[:2]
        min_width, min_height = self.focus_crop_min_size
        crop_height = max(float(min_height), (y1 - y0) * self.focus_crop_scale)
        crop_width = max(float(min_width), (x1 - x0) * self.focus_crop_scale)
        # 使裁剪区域接近原始宽高比，避免缩放引入额外的几何畸变。
        crop_width = max(crop_width, crop_height * width / height)
        crop_height = max(crop_height, crop_width * height / width)
        crop_width = min(width, int(round(crop_width)))
        crop_height = min(height, int(round(crop_height)))
        center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        crop_x0 = int(round(np.clip(center_x - crop_width / 2.0, 0, width - crop_width)))
        crop_y0 = int(round(np.clip(center_y - crop_height / 2.0, 0, height - crop_height)))
        crop_transform = CropTransform(crop_x0, crop_y0, crop_width, crop_height)

        kept = []
        target_kept = False
        for annotation in annotations:
            ax0, ay0, ax1, ay1 = map(float, annotation["bbox"])
            box_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
            ix0, iy0 = max(ax0, crop_x0), max(ay0, crop_y0)
            ix1, iy1 = min(ax1, crop_x0 + crop_width), min(ay1, crop_y0 + crop_height)
            coverage = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) / box_area
            if coverage < 0.8:
                continue
            transformed = copy.deepcopy(annotation)
            # 在部分 Detectron2/fvcore 版本中 CropTransform.apply_polygons 依赖 shapely。
            # 此处裁剪仅为平移操作，因此直接变换 LabelMe 多边形坐标以避免额外依赖。
            transformed["bbox"] = [
                float(np.clip(ax0 - crop_x0, 0, crop_width - 1)),
                float(np.clip(ay0 - crop_y0, 0, crop_height - 1)),
                float(np.clip(ax1 - crop_x0, 0, crop_width - 1)),
                float(np.clip(ay1 - crop_y0, 0, crop_height - 1)),
            ]
            segmentation = []
            for polygon in transformed.get("segmentation", []):
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                points = crop_transform.apply_coords(points)
                segmentation.append(points.reshape(-1).tolist())
            transformed["segmentation"] = segmentation
            kept.append(transformed)
            if transformed is not None and annotation is target:
                target_kept = True
        if not target_kept:
            return image
        dataset_dict["annotations"] = kept
        dataset_dict["height"] = crop_height
        dataset_dict["width"] = crop_width
        return crop_transform.apply_image(image)

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        transformed, angles = [], []
        for annotation in dataset_dict.pop("annotations"):
            if annotation.get("iscrowd", 0):
                continue
            category_id = int(annotation.get("category_id", -1))
            angle_is_valid = category_id in self.angle_class_ids
            angle_is_explicit = bool(annotation.get("angle_explicit", False))
            if angle_is_valid and self.angle_label_source == "annotation" and angle_is_explicit:
                # 将方向向量一起做几何变换，旋转/翻转增强后角度仍保持正确。
                angle = float(annotation["angle_degrees"])
                radians = np.radians(angle)
                bbox = np.asarray(annotation["bbox"], dtype=np.float32).reshape(2, 2)
                center = bbox.mean(axis=0)
                direction = center + np.asarray([np.cos(radians), np.sin(radians)]) * 20.0
                mapped = transforms.apply_coords(np.stack([center, direction]))
                vector = mapped[1] - mapped[0]
                angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % self.angle_period)
            annotation = utils.transform_instance_annotations(annotation, transforms, image_shape)
            polygons = annotation.get("segmentation") or []
            if angle_is_valid and self.angle_label_source == "mask":
                angle = (
                    polygon_angle(np.asarray(max(polygons, key=len)).reshape(-1, 2))
                    if polygons else 0.0
                )
            elif not (angle_is_valid and self.angle_label_source == "annotation" and angle_is_explicit):
                # -1 为 AngleROIHeads 使用的忽略占位值。
                # 避免未标注或未选中的类别向方向分支贡献虚假监督信号。
                angle = None
            transformed.append(annotation)
            angles.append(
                angle_to_bin(angle, self.angle_bins, self.angle_period)
                if angle is not None else -1
            )
        instances = utils.annotations_to_instances(
            transformed, image_shape, mask_format=self.mask_format
        )
        instances.gt_angle_bins = torch.as_tensor(angles, dtype=torch.int64)
        instances, _ = utils.filter_empty_instances(instances, return_mask=True)
        dataset_dict["instances"] = instances

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        if self.is_train:
            image = self._copy_paste(dataset_dict, image)
            image = self._focus_crop(dataset_dict, image)
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict
