"""
Roboflow 流水线数据增强模块
实现与 Roboflow UI 规范严格对齐的全部图像级 (Image-Level) 与边界框级 (Bounding Box-Level) 数据增强技术：

图像级增强：
1. 翻转 (Flip: 水平翻转、垂直翻转)
2. 90° 旋转 (90° Rotate: 顺时针、逆时针、上下颠倒)
3. 裁剪缩放 (Crop: 最小缩放比例 %、最大缩放比例 %)
4. 任意角度旋转 (Rotation: ±角度 °)
5. 错切变换 (Shear: 水平 ±°、垂直 ±°)
6. 灰度化 (Grayscale)
7. 色调调整 (Hue: ±角度 °)
8. 饱和度调整 (Saturation: ±%)
9. 亮度调整 (Brightness: ±%、提亮/变暗)
10. 曝光度调整 (Exposure: ±%)
11. 高斯模糊 (Blur: 像素核半径 px)
12. 椒盐噪声 (Noise: 像素百分比 %)
13. 区域遮挡 (Cutout: 遮挡面积占比 %、遮挡块数量)
14. 马赛克增强 (Mosaic: 4 图拼接)
15. 运动模糊 (Motion Blur: 模糊长度 px、运动角度 °、帧数)
16. 相机增益 (Camera Gain: 方差)

边界框级增强：
1. 边界框：翻转 (Bounding Box: Flip)
2. 边界框：90° 旋转 (Bounding Box: 90° Rotate)
3. 边界框：裁剪缩放 (Bounding Box: Crop)
4. 边界框：任意角度旋转 (Bounding Box: Rotation)
5. 边界框：错切变换 (Bounding Box: Shear)
6. 边界框：亮度调整 (Bounding Box: Brightness)
7. 边界框：曝光度调整 (Bounding Box: Exposure)
8. 边界框：高斯模糊 (Bounding Box: Blur)
9. 边界框：椒盐噪声 (Bounding Box: Noise)
10. 边界框：运动模糊 (Bounding Box: Motion Blur)
11. 边界框：相机增益 (Bounding Box: Camera Gain)

边界框级算子在固定的裁剪 ROI 局部区域上复用对应的图像级像素与几何算子。
ROI 外部的像素及标注坐标保持不变。
"""

import math
import copy
import base64
import hashlib
import random
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageEnhance

from .dataset import ImageSample, AnnotationItem


class AugmentationPipeline:
    """可配置的数据增强流水线。"""

    def __init__(self, aug_configs: Optional[Dict[str, Any]] = None, multiplier: int = 3):
        self.configs = aug_configs or {}
        self.multiplier = multiplier

    def set_config(self, aug_name: str, config: Dict[str, Any]):
        self.configs[aug_name] = config

    def remove_config(self, aug_name: str):
        if aug_name in self.configs:
            del self.configs[aug_name]

    def clear(self):
        self.configs = {}

    def augment_dataset(self, samples: List[ImageSample], all_train_samples: Optional[List[ImageSample]] = None) -> List[ImageSample]:
        """
        根据倍率 multiplier 生成训练样本的增强版本。
        自动剔除重复的图像/标注结果，与 Roboflow 数据集版本生成行为保持一致。
        """
        if not self.configs or self.multiplier <= 1:
            return samples

        augmented_samples: List[ImageSample] = []
        pool = all_train_samples if all_train_samples else samples
        signatures = set()

        for s in samples:
            original_signature = self._sample_signature(s)
            if original_signature not in signatures:
                # 保留原样本
                augmented_samples.append(s)
                signatures.add(original_signature)

            # 生成 (multiplier - 1) 个增强副本
            for i in range(1, self.multiplier):
                aug_s = self._apply_all_augs(s, version=i, pool=pool)
                signature = self._sample_signature(aug_s)
                if signature in signatures:
                    continue
                augmented_samples.append(aug_s)
                signatures.add(signature)

        return augmented_samples

    def _sample_signature(self, sample: ImageSample) -> str:
        """对图像像素与标注标签进行哈希计算（排除样本 ID），用于数据去重。"""
        image = np.ascontiguousarray(self._get_image(sample))
        digest = hashlib.sha256()
        digest.update(str(image.shape).encode("utf-8"))
        digest.update(str(image.dtype).encode("utf-8"))
        digest.update(image.tobytes())
        for ann in sample.annotations:
            digest.update(str(ann.label).encode("utf-8"))
            digest.update(str(ann.shape_type).encode("utf-8"))
            digest.update(np.asarray(ann.points, dtype=np.float64).tobytes())
        digest.update(str(getattr(sample, "classification_label", "")).encode("utf-8"))
        return digest.hexdigest()

    def _clone_sample(self, sample: ImageSample, version: int = 1) -> ImageSample:
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
            sample_id=f"{sample.sample_id}_aug{version}",
            tags=list(getattr(sample, "tags", [])),
        )
        new_sample.split = sample.split
        if hasattr(sample, "_cached_img") and sample._cached_img is not None:
            new_sample._cached_img = sample._cached_img.copy()
        return new_sample

    def _get_image(self, sample: ImageSample) -> np.ndarray:
        if hasattr(sample, "_cached_img") and sample._cached_img is not None:
            return sample._cached_img.copy()
        img = cv2.imread(sample.image_path)
        if img is None:
            img = np.zeros((sample.height, sample.width, 3), dtype=np.uint8)
        return img

    def _apply_all_augs(self, sample: ImageSample, version: int, pool: List[ImageSample]) -> ImageSample:
        aug_sample = self._clone_sample(sample, version)
        img = self._get_image(aug_sample)

        # 图像级数据增强
        if "flip" in self.configs:
            img, aug_sample.annotations = self.apply_flip(img, aug_sample.annotations, self.configs["flip"])

        if "rotate90" in self.configs:
            img, aug_sample.annotations = self.apply_rotate90(img, aug_sample.annotations, self.configs["rotate90"])

        if "rotation" in self.configs:
            img, aug_sample.annotations = self.apply_rotation(img, aug_sample.annotations, self.configs["rotation"])

        if "shear" in self.configs:
            img, aug_sample.annotations = self.apply_shear(img, aug_sample.annotations, self.configs["shear"])

        if "crop" in self.configs:
            img, aug_sample.annotations = self.apply_crop(img, aug_sample.annotations, self.configs["crop"])

        if "grayscale" in self.configs:
            img = self.apply_grayscale(img, self.configs["grayscale"])

        if "hue" in self.configs:
            img = self.apply_hue(img, self.configs["hue"])

        if "saturation" in self.configs:
            img = self.apply_saturation(img, self.configs["saturation"])

        if "brightness" in self.configs:
            img = self.apply_brightness(img, self.configs["brightness"])

        if "exposure" in self.configs:
            img = self.apply_exposure(img, self.configs["exposure"])

        if "blur" in self.configs:
            img = self.apply_blur(img, self.configs["blur"])

        if "noise" in self.configs:
            img = self.apply_noise(img, self.configs["noise"])

        if "motion_blur" in self.configs:
            img = self.apply_motion_blur(img, self.configs["motion_blur"])

        if "camera_gain" in self.configs:
            img = self.apply_camera_gain(img, self.configs["camera_gain"])

        if "cutout" in self.configs:
            img = self.apply_cutout(img, self.configs["cutout"])

        if "mosaic" in self.configs:
            img, aug_sample.annotations = self.apply_mosaic(img, aug_sample.annotations, pool)

        # 边界框级数据增强
        bbox_augs = [k for k in self.configs if k.startswith("bbox_")]
        for bkey in bbox_augs:
            img = self.apply_bbox_augmentation(img, aug_sample.annotations, bkey, self.configs[bkey])

        aug_sample._cached_img = img
        aug_sample.height, aug_sample.width = img.shape[:2]
        return aug_sample

    # 1. 翻转 (Flip)
    def apply_flip(self, img: np.ndarray, annots: List[AnnotationItem], cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[AnnotationItem]]:
        h, w = img.shape[:2]
        do_h = cfg.get("horizontal", True) and random.random() > 0.5
        do_v = cfg.get("vertical", False) and random.random() > 0.5

        if not do_h and not do_v:
            return img, annots

        if do_h and do_v:
            img = cv2.flip(img, -1)
            for ann in annots:
                ann.points = [[w - 1 - p[0], h - 1 - p[1]] for p in ann.points]
        elif do_h:
            img = cv2.flip(img, 1)
            for ann in annots:
                ann.points = [[w - 1 - p[0], p[1]] for p in ann.points]
        elif do_v:
            img = cv2.flip(img, 0)
            for ann in annots:
                ann.points = [[p[0], h - 1 - p[1]] for p in ann.points]

        return img, annots

    # 2. 90° 旋转 (90° Rotate)
    def apply_rotate90(self, img: np.ndarray, annots: List[AnnotationItem], cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[AnnotationItem]]:
        choices = []
        if cfg.get("clockwise", True):
            choices.append(cv2.ROTATE_90_CLOCKWISE)
        if cfg.get("counter_clockwise", True):
            choices.append(cv2.ROTATE_90_COUNTERCLOCKWISE)
        if cfg.get("upside_down", False):
            choices.append(cv2.ROTATE_180)

        if not choices:
            return img, annots

        rot = random.choice(choices)
        h, w = img.shape[:2]

        if rot == cv2.ROTATE_90_CLOCKWISE:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            for ann in annots:
                ann.points = [[h - 1 - p[1], p[0]] for p in ann.points]
        elif rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            for ann in annots:
                ann.points = [[p[1], w - 1 - p[0]] for p in ann.points]
        elif rot == cv2.ROTATE_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
            for ann in annots:
                ann.points = [[w - 1 - p[0], h - 1 - p[1]] for p in ann.points]

        return img, annots

    # 3. 任意角度旋转 (Rotation)
    def apply_rotation(self, img: np.ndarray, annots: List[AnnotationItem], cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[AnnotationItem]]:
        max_angle = float(cfg.get("angle", 15.0))
        angle = random.uniform(-max_angle, max_angle)
        if abs(angle) < 0.1:
            return img, annots

        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        for ann in annots:
            new_pts = []
            for p in ann.points:
                nx = M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2]
                ny = M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]
                new_pts.append([float(np.clip(nx, 0, w - 1)), float(np.clip(ny, 0, h - 1))])
            ann.points = new_pts

        return img, annots

    # 4. 错切变换 (Shear)
    def apply_shear(self, img: np.ndarray, annots: List[AnnotationItem], cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[AnnotationItem]]:
        max_h = float(cfg.get("horizontal", 10.0))
        max_v = float(cfg.get("vertical", 10.0))
        sh_x = math.tan(math.radians(random.uniform(-max_h, max_h)))
        sh_y = math.tan(math.radians(random.uniform(-max_v, max_v)))

        h, w = img.shape[:2]
        M = np.array([[1.0, sh_x, 0.0], [sh_y, 1.0, 0.0]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        for ann in annots:
            new_pts = []
            for p in ann.points:
                nx = M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2]
                ny = M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]
                new_pts.append([float(np.clip(nx, 0, w - 1)), float(np.clip(ny, 0, h - 1))])
            ann.points = new_pts

        return img, annots

    # 5. 裁剪缩放 (Crop)
    def apply_crop(self, img: np.ndarray, annots: List[AnnotationItem], cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[AnnotationItem]]:
        min_zoom = float(np.clip(cfg.get("min_zoom", 0.0), 0.0, 99.0)) / 100.0
        max_zoom = float(np.clip(cfg.get("max_zoom", 20.0), 0.0, 99.0)) / 100.0
        if min_zoom > max_zoom:
            min_zoom, max_zoom = max_zoom, min_zoom
        zoom = random.uniform(min_zoom, max_zoom)

        if zoom <= 0:
            return img, annots

        h, w = img.shape[:2]
        # Roboflow 基于移除图像面积的比例定义裁剪。各维度使用平方根比例，
        # 使保留区域具有请求的面积分数，而不是两次乘百分比。
        retained_fraction = max(1e-6, 1.0 - zoom)
        side_fraction = math.sqrt(retained_fraction)
        crop_w = max(1, min(w, int(round(w * side_fraction))))
        crop_h = max(1, min(h, int(round(h * side_fraction))))

        dx = random.randint(0, max(0, w - crop_w))
        dy = random.randint(0, max(0, h - crop_h))

        cropped = img[dy:dy + crop_h, dx:dx + crop_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        scale_x = w / max(1, crop_w)
        scale_y = h / max(1, crop_h)

        new_annots = []
        for ann in annots:
            pts = [[(p[0] - dx) * scale_x, (p[1] - dy) * scale_y] for p in ann.points]
            clipped_pts = [[np.clip(p[0], 0, w - 1), np.clip(p[1], 0, h - 1)] for p in pts]
            poly_np = np.array(clipped_pts, dtype=np.int32)
            if cv2.contourArea(poly_np) > 10:
                ann.points = clipped_pts
                new_annots.append(ann)

        return resized, new_annots

    # 6. 灰度化 (Grayscale)
    def apply_grayscale(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        probability = float(cfg.get("percent", cfg.get("probability", 50.0))) / 100.0
        probability = float(np.clip(probability, 0.0, 1.0))
        if random.random() < probability:
            if img.ndim == 2:
                gray = img.astype(np.uint8, copy=True)
            else:
                # 使用 Roboflow 文档记录的 RGB 亮度权重；OpenCV 数组存储格式为 BGR。
                gray = np.clip(
                    img[:, :, 2].astype(np.float32) * 0.2125
                    + img[:, :, 1].astype(np.float32) * 0.7154
                    + img[:, :, 0].astype(np.float32) * 0.0721,
                    0,
                    255,
                ).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

    # 7. 色调调整 (Hue)
    def apply_hue(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        max_deg = float(cfg.get("angle", 15.0))
        shift = random.uniform(-max_deg, max_deg)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift / 2.0) % 180.0
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 8. 饱和度调整 (Saturation)
    def apply_saturation(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        pct = float(cfg.get("percent", 25.0)) / 100.0
        factor = random.uniform(1.0 - pct, 1.0 + pct)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 9. 亮度调整 (Brightness)
    def apply_brightness(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        pct = float(np.clip(cfg.get("percent", 15.0), 0.0, 99.0)) / 100.0
        brighten = cfg.get("brighten", True)
        darken = cfg.get("darken", True)

        offsets = []
        if brighten:
            offsets.append((0.0, pct))
        if darken:
            offsets.append((-pct, 0.0))
        if not offsets:
            return img

        low = min(item[0] for item in offsets)
        high = max(item[1] for item in offsets)
        offset = random.uniform(low, high) * 255.0
        return np.clip(img.astype(np.float32) + offset, 0, 255).astype(np.uint8)

    # 10. 曝光度调整 (Exposure)
    def apply_exposure(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        pct = float(np.clip(cfg.get("percent", 16.0), 0.0, 100.0)) / 100.0
        amount = random.uniform(0.0, pct)
        brighten = random.random() >= 0.5
        if brighten:
            gamma = max(1e-6, 1.0 - amount)
        else:
            gamma = 1.0 / max(1e-6, 1.0 - amount)

        table = np.arange(256, dtype=np.float32) / 255.0
        table = np.clip(np.power(table, gamma) * 255.0, 0, 255).astype(np.uint8)
        return cv2.LUT(img, table)

    # 11. 高斯模糊 (Blur)
    def apply_blur(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        px = float(cfg.get("pixels", 2.5))
        ksize = int(px * 2) + 1
        ksize = max(1, ksize if ksize % 2 == 1 else ksize + 1)
        return cv2.GaussianBlur(img, (ksize, ksize), px)

    # 12. 椒盐噪声 (Noise)
    def apply_noise(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        # Roboflow 将噪声定义为椒盐噪声：百分比为受影响的图像像素比例，
        # 而非施加到每个通道/像素的高斯噪声幅度。
        pct = float(np.clip(cfg.get("percent", 0.1), 0.0, 25.0)) / 100.0
        height, width = img.shape[:2]
        count = min(height * width, int(round(height * width * pct)))
        if count <= 0:
            return img.copy()

        flat_indices = np.random.choice(height * width, count, replace=False)
        salt = np.random.random(count) >= 0.5
        noisy = img.copy()
        flat = noisy.reshape(height * width, *noisy.shape[2:]) if noisy.ndim > 2 else noisy.reshape(height * width)
        if noisy.ndim > 2:
            flat[flat_indices[salt]] = 255
            flat[flat_indices[~salt]] = 0
        else:
            flat[flat_indices[salt]] = 255
            flat[flat_indices[~salt]] = 0
        return noisy

    # 13. 区域遮挡 (Cutout)
    def apply_cutout(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        pct = float(cfg.get("percent", 10.0)) / 100.0
        count = int(cfg.get("count", 3))
        h, w = img.shape[:2]
        res = img.copy()

        box_area = h * w * pct
        box_side = int(math.sqrt(box_area))

        for _ in range(count):
            bh = random.randint(max(5, box_side // 2), max(10, box_side * 2))
            bw = random.randint(max(5, box_side // 2), max(10, box_side * 2))
            x1 = random.randint(0, max(0, w - bw))
            y1 = random.randint(0, max(0, h - bh))
            res[y1:y1 + bh, x1:x1 + bw] = (0, 0, 0)

        return res

    # 14. 运动模糊 (Motion Blur)
    def apply_motion_blur(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        length = int(np.clip(round(float(cfg.get("length", 15))), 1, 1000))
        angle = float(np.clip(cfg.get("angle", 0.0), 0.0, 90.0))
        frames = int(np.clip(round(float(cfg.get("frames", 1))), 1, 5))
        if length <= 1:
            return img

        # 构建运动模糊卷积核
        kernel = np.zeros((length, length), dtype=np.float32)
        center = length // 2
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        for i in range(length):
            offset = i - center
            x = int(round(center + offset * cos_a))
            y = int(round(center + offset * sin_a))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1.0

        ksum = np.sum(kernel)
        if ksum > 0:
            kernel /= ksum
        else:
            kernel[center, center] = 1.0

        # 沿配置的运动方向对不同的曝光相位取平均。
        # 这也使得垂直（90 度）多帧模糊与水平模糊表现不同。
        height, width = img.shape[:2]
        result = np.zeros_like(img, dtype=np.float32)
        for frame_idx in range(frames):
            phase_offset = frame_idx - (frames - 1) / 2.0
            shift_x = phase_offset * cos_a
            shift_y = phase_offset * sin_a
            matrix = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
            phase = cv2.warpAffine(
                img, matrix, (width, height),
                borderMode=cv2.BORDER_REFLECT_101,
            )
            result += cv2.filter2D(phase, -1, kernel).astype(np.float32)
        return np.clip(result / frames, 0, 255).astype(np.uint8)

    # 15. 相机增益 (Camera Gain)
    def apply_camera_gain(self, img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        variance = float(cfg.get("variance", 0.05))
        std = math.sqrt(variance) * 255.0
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        gain_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return gain_img

    # 16. 马赛克增强 (Mosaic)
    def apply_mosaic(self, img: np.ndarray, annots: List[AnnotationItem], pool: List[ImageSample]) -> Tuple[np.ndarray, List[AnnotationItem]]:
        if len(pool) < 4:
            return img, annots

        h, w = img.shape[:2]
        xc = int(random.uniform(w * 0.4, w * 0.6))
        yc = int(random.uniform(h * 0.4, h * 0.6))

        selected = [random.choice(pool) for _ in range(3)]
        all_imgs = [img] + [self._get_image(s) for s in selected]
        all_annots = [annots] + [
            [
                AnnotationItem(
                    label=ann.label,
                    points=[list(p) for p in ann.points],
                    shape_type=ann.shape_type,
                    score=ann.score,
                )
                for ann in s.annotations
            ]
            for s in selected
        ]

        mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
        mosaic_annots = []

        # 4 个象限：左上、右上、左下、右下
        quads = [
            (0, 0, xc, yc),       # 左上 (TL)
            (xc, 0, w, yc),       # 右上 (TR)
            (0, yc, xc, h),       # 左下 (BL)
            (xc, yc, w, h)        # 右下 (BR)
        ]

        for i, (qx1, qy1, qx2, qy2) in enumerate(quads):
            sub_img = all_imgs[i]
            if sub_img is None:
                continue
            sh, sw = sub_img.shape[:2]
            qw = qx2 - qx1
            qh = qy2 - qy1

            scaled_sub = cv2.resize(sub_img, (qw, qh))
            mosaic_img[qy1:qy2, qx1:qx2] = scaled_sub

            sx = qw / max(1, sw)
            sy = qh / max(1, sh)

            for ann in all_annots[i]:
                new_pts = [[p[0] * sx + qx1, p[1] * sy + qy1] for p in ann.points]
                poly_np = np.array(new_pts, dtype=np.int32)
                if cv2.contourArea(poly_np) > 10:
                    mosaic_annots.append(AnnotationItem(
                        label=ann.label,
                        points=new_pts,
                        shape_type=ann.shape_type,
                        score=ann.score
                    ))

        return mosaic_img, mosaic_annots

    # 边界框级数据增强 (Bounding Box-Level Augmentations)
    @staticmethod
    def _bbox_bounds(ann: AnnotationItem, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        """返回标注边界框裁剪后的半开区间像素 ROI (x1, y1, x2, y2)。"""
        bbox = ann.to_bbox()
        x1 = max(0, min(width - 1, int(np.floor(bbox[0]))))
        y1 = max(0, min(height - 1, int(np.floor(bbox[1]))))
        x2 = min(width, max(x1 + 1, int(np.ceil(bbox[2])) + 1))
        y2 = min(height, max(y1 + 1, int(np.ceil(bbox[3])) + 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _apply_bbox_roi_augmentation(self, roi: np.ndarray, aug_key: str, cfg: Dict[str, Any]) -> np.ndarray:
        """对单个 ROI 区域应用对应的图像级增强算子。

        几何算子传入空标注列表，因为检测标注仍处于原始图像坐标系中；
        仅改变固定边界框内部的像素内容。
        """
        if aug_key == "bbox_flip":
            return self.apply_flip(roi, [], cfg)[0]
        if aug_key == "bbox_rotate90":
            transformed, _ = self.apply_rotate90(roi, [], cfg)
            if transformed.shape[:2] != roi.shape[:2]:
                transformed = cv2.resize(
                    transformed, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            return transformed
        if aug_key == "bbox_crop":
            return self.apply_crop(roi, [], cfg)[0]
        if aug_key == "bbox_rotation":
            return self.apply_rotation(roi, [], cfg)[0]
        if aug_key == "bbox_shear":
            return self.apply_shear(roi, [], cfg)[0]
        if aug_key == "bbox_brightness":
            return self.apply_brightness(roi, cfg)
        if aug_key == "bbox_exposure":
            return self.apply_exposure(roi, cfg)
        if aug_key == "bbox_blur":
            return self.apply_blur(roi, cfg)
        if aug_key == "bbox_noise":
            return self.apply_noise(roi, cfg)
        if aug_key == "bbox_motion_blur":
            return self.apply_motion_blur(roi, cfg)
        if aug_key == "bbox_camera_gain":
            return self.apply_camera_gain(roi, cfg)
        return roi

    def apply_bbox_augmentation(self, img: np.ndarray, annots: List[AnnotationItem], aug_key: str, cfg: Dict[str, Any]) -> np.ndarray:
        """在保持图像几何形状不变的前提下增强各边界框内部的像素。

        原始图像、所有 ROI 外部的像素以及所有标注坐标均保持不变。
        算子与参数语义与其对应的图像级算子共享。
        """
        if not annots:
            return img

        res = img.copy()
        h, w = img.shape[:2]

        for ann in annots:
            bounds = self._bbox_bounds(ann, w, h)
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds

            roi = res[y1:y2, x1:x2].copy()
            transformed = self._apply_bbox_roi_augmentation(roi, aug_key, cfg)
            if transformed.shape[:2] != roi.shape[:2]:
                transformed = cv2.resize(
                    transformed, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            res[y1:y2, x1:x2] = transformed

        return res


def preview_augmentation(sample: ImageSample, aug_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """为指定的数据增强步骤生成增强前后的 base64 缩略图预览。"""
    pipeline = AugmentationPipeline()
    pipeline.set_config(aug_name, config)

    orig_img = cv2.imread(sample.image_path)
    if orig_img is None:
        orig_img = np.zeros((sample.height, sample.width, 3), dtype=np.uint8)

    cloned = pipeline._clone_sample(sample)
    aug_res = pipeline._apply_all_augs(cloned, version=1, pool=[sample])
    proc_img = aug_res._cached_img if hasattr(aug_res, "_cached_img") else orig_img

    def encode_b64(img_np: np.ndarray) -> str:
        th = 320
        h, w = img_np.shape[:2]
        tw = int(round(w * (th / max(1, h))))
        thumb = cv2.resize(img_np, (tw, th))
        _, buf = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buf).decode('utf-8')

    return {
        "original": f"data:image/jpeg;base64,{encode_b64(orig_img)}",
        "augmented": f"data:image/jpeg;base64,{encode_b64(proc_img)}",
        "aug_name": aug_name,
        "config": config
    }
