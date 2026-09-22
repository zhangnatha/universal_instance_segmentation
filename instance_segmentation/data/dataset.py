"""
Roboflow 流水线数据集管理模块
负责已标注数据集的加载、解析、可视化渲染、子集划分与多格式导出。
支持 LabelMe JSON、YOLO TXT 以及 COCO 格式。
"""

import os
import json
import random
import shutil
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
from PIL import Image
from .classes import load_class_names, validate_discovered_labels


class AnnotationItem:
    """单个实例标注数据结构（多边形或边界框）。"""
    def __init__(self, label: str, points: List[List[float]], shape_type: str = "polygon", score: Optional[float] = None):
        self.label = label
        self.points = [list(map(float, p)) for p in points]
        self.shape_type = shape_type
        self.score = score

    def to_bbox(self) -> List[float]:
        """返回 [xmin, ymin, xmax, ymax] 边界框坐标。"""
        pts = np.array(self.points)
        xmin = float(np.min(pts[:, 0]))
        ymin = float(np.min(pts[:, 1]))
        xmax = float(np.max(pts[:, 0]))
        ymax = float(np.max(pts[:, 1]))
        return [xmin, ymin, xmax, ymax]

    def to_dict(self) -> Dict[str, Any]:
        """将标注转换为字典格式。"""
        return {
            "label": self.label,
            "points": self.points,
            "shape_type": self.shape_type,
            "score": self.score
        }


class ImageSample:
    """单张图像样本数据结构，包含图像元信息及其所有实例标注。"""
    def __init__(
        self,
        image_path: str,
        width: int,
        height: int,
        annotations: List[AnnotationItem],
        sample_id: str = "",
        tags: Optional[List[str]] = None,
    ):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.annotations = annotations
        self.sample_id = sample_id or Path(image_path).stem
        if isinstance(tags, str):
            tags = [tags]
        self.tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        self.split = "train"  # 划分类型：训练集、验证集或测试集 (train, valid, test)

    @property
    def is_null(self) -> bool:
        """若样本没有任何有效实例标注则返回 True。"""
        return len(self.annotations) == 0

    def get_classes(self) -> List[str]:
        """获取当前样本中包含的所有类别名称列表。"""
        return [ann.label for ann in self.annotations]


class DatasetManager:
    """管理数据集的完整生命周期：加载、检查、划分与导出。"""

    IMAGE_EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.webp', '.tiff', '.BMP', '.PNG', '.JPG', '.JPEG'}

    def __init__(self, source_dir: Optional[str] = None):
        self.source_dir = source_dir
        self.samples: List[ImageSample] = []
        self.classes: List[str] = []
        self.split_ratios = {"train": 0.70, "valid": 0.20, "test": 0.10}
        self.splits: Dict[str, List[ImageSample]] = {"train": [], "valid": [], "test": []}

        if source_dir:
            self.load_from_directory(source_dir)

    def _finalize_classes(self, discovered: List[str]) -> None:
        """将可编辑的 classes.names 配置规则应用到数据集中发现的类别标签。"""
        configured = load_class_names()
        unknown = validate_discovered_labels(discovered, configured)
        if unknown:
            raise ValueError(
                "Dataset labels are missing from classes.names: "
                + ", ".join(unknown)
                + ". Add them to classes.names before continuing."
            )
        self.classes = configured or sorted(set(discovered)) or ["class_0"]

    def load_from_directory(self, directory: str) -> Dict[str, Any]:
        """从用户提供的数据集目录中加载图像和标注文件。"""
        candidate_paths = [
            directory,
            os.path.join(os.getcwd(), directory),
            os.path.join(os.path.dirname(__file__), directory),
            os.path.join(os.path.dirname(__file__), "..", directory),
            os.path.join(os.path.dirname(__file__), "data"),
            os.path.join(os.path.dirname(__file__), "dataset"),
            os.path.join(os.path.dirname(__file__), "datasets", "current_dataset")
        ]
        chosen_dir = None
        for p in candidate_paths:
            if os.path.exists(p) and os.path.isdir(p):
                chosen_dir = os.path.normpath(p)
                break

        if chosen_dir is None:
            raise FileNotFoundError(f"Dataset directory '{directory}' not found in candidate locations: {candidate_paths}")

        self.source_dir = chosen_dir
        self.samples = []
        self.splits = {"train": [], "valid": [], "test": []}
        class_set = set()

        # 检查 chosen_dir 是否为已划分的数据集目录（例如 exports/current_dataset）
        if os.path.exists(os.path.join(chosen_dir, "train")) and os.path.exists(os.path.join(chosen_dir, "valid")):
            for split_name in ["train", "valid", "test"]:
                split_dir = os.path.join(chosen_dir, split_name)
                if not os.path.exists(split_dir):
                    continue

                coco_json = os.path.join(split_dir, "_annotations.coco.json")
                if os.path.exists(coco_json):
                    try:
                        with open(coco_json, 'r', encoding='utf-8') as cf:
                            cdata = json.load(cf)
                        cat_id_to_name = {c["id"]: c["name"] for c in cdata.get("categories", [])}
                        img_id_to_anns = {}
                        for ann in cdata.get("annotations", []):
                            img_id = ann["image_id"]
                            cname = cat_id_to_name.get(ann.get("category_id"))
                            if not cname:
                                continue
                            class_set.add(cname)
                            seg = ann.get("segmentation", [])
                            pts = []
                            if seg and isinstance(seg[0], list):
                                flat_pts = seg[0]
                                pts = [[flat_pts[i], flat_pts[i+1]] for i in range(0, len(flat_pts)-1, 2)]
                            elif "bbox" in ann:
                                bx, by, bw, bh = ann["bbox"]
                                pts = [[bx, by], [bx+bw, by], [bx+bw, by+bh], [bx, by+bh]]
                            if pts:
                                img_id_to_anns.setdefault(img_id, []).append(
                                    AnnotationItem(label=cname, points=pts, shape_type="polygon")
                                )

                        for im_info in cdata.get("images", []):
                            fname = im_info["file_name"]
                            im_path = os.path.join(split_dir, "images", fname)
                            if not os.path.exists(im_path):
                                im_path = os.path.join(split_dir, fname)

                            s = ImageSample(
                                image_path=im_path,
                                width=im_info.get("width", 640),
                                height=im_info.get("height", 640),
                                annotations=img_id_to_anns.get(im_info["id"], []),
                                sample_id=Path(fname).stem,
                                tags=im_info.get("tags", []),
                            )
                            s.split = split_name
                            self.samples.append(s)
                            self.splits[split_name].append(s)
                    except Exception as e:
                        print(f"Warning: error loading COCO json in {split_name}: {e}")

            self._finalize_classes(list(class_set))
            return self.get_summary()

        # 查找扁平目录中的所有图像文件
        all_files = os.listdir(self.source_dir)
        image_files = [f for f in all_files if Path(f).suffix in self.IMAGE_EXTENSIONS]
        image_files.sort()

        for img_name in image_files:
            img_path = os.path.join(self.source_dir, img_name)
            stem = Path(img_name).stem
            json_name = f"{stem}.json"
            json_path = os.path.join(self.source_dir, json_name)

            annotations: List[AnnotationItem] = []
            width, height = 0, 0
            tags: List[str] = []

            # 尝试读取同名 JSON 标注文件
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                    width = data.get("imageWidth", 0)
                    height = data.get("imageHeight", 0)
                    tags = data.get("tags", [])
                    for shape in data.get("shapes", []):
                        lbl = str(shape.get("label", "")).strip()
                        if not lbl:
                            continue
                        pts = shape.get("points", [])
                        stype = shape.get("shape_type", "polygon")
                        score = shape.get("score", None)
                        if pts:
                            annotations.append(AnnotationItem(label=lbl, points=pts, shape_type=stype, score=score))
                            class_set.add(lbl)
                except Exception as e:
                    print(f"Warning: Failed to parse {json_path}: {e}")

            # 若 JSON 中缺少 width/height，则尝试读取图像获取尺寸
            if width == 0 or height == 0:
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception:
                    width, height = 640, 480

            sample = ImageSample(
                image_path=img_path,
                width=width,
                height=height,
                annotations=annotations,
                sample_id=stem,
                tags=tags,
            )
            self.samples.append(sample)

        self._finalize_classes(list(class_set))

        # 默认比例划分
        self.split_dataset(train_ratio=0.70, valid_ratio=0.20, test_ratio=0.10)
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """获取数据集的统计摘要信息。"""
        total_images = len(self.samples)
        class_counts: Dict[str, int] = {c: 0 for c in self.classes}
        null_images = 0
        total_annotations = 0

        for sample in self.samples:
            if sample.is_null:
                null_images += 1
            for ann in sample.annotations:
                class_counts[ann.label] = class_counts.get(ann.label, 0) + 1
                total_annotations += 1

        split_counts = {
            "train": len(self.splits.get("train", [])),
            "valid": len(self.splits.get("valid", [])),
            "test": len(self.splits.get("test", []))
        }

        return {
            "source_dir": self.source_dir,
            "total_images": total_images,
            "total_annotations": total_annotations,
            "null_images": null_images,
            "classes": self.classes,
            "class_counts": class_counts,
            "split_ratios": self.split_ratios,
            "split_counts": split_counts
        }

    def split_dataset(self, train_ratio: float = 0.70, valid_ratio: float = 0.20, test_ratio: float = 0.10, seed: int = 42) -> Dict[str, int]:
        """将数据集按指定比例划分为训练集 (train)、验证集 (valid) 与测试集 (test)。"""
        # 归一化比例使总和为 1.0
        total_ratio = train_ratio + valid_ratio + test_ratio
        if total_ratio <= 0:
            train_ratio, valid_ratio, test_ratio = 0.70, 0.20, 0.10
        else:
            train_ratio /= total_ratio
            valid_ratio /= total_ratio
            test_ratio /= total_ratio

        self.split_ratios = {"train": train_ratio, "valid": valid_ratio, "test": test_ratio}

        rng = random.Random(seed)
        shuffled = list(self.samples)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_valid = int(round(n * valid_ratio))
        n_test = n - n_train - n_valid
        if n_test < 0:
            n_test = 0
            n_valid = n - n_train

        train_samples = shuffled[:n_train]
        valid_samples = shuffled[n_train:n_train + n_valid]
        test_samples = shuffled[n_train + n_valid:]

        for s in train_samples:
            s.split = "train"
        for s in valid_samples:
            s.split = "valid"
        for s in test_samples:
            s.split = "test"

        self.splits = {
            "train": train_samples,
            "valid": valid_samples,
            "test": test_samples
        }

        return {
            "train": len(train_samples),
            "valid": len(valid_samples),
            "test": len(test_samples)
        }

    def render_sample(self, sample: ImageSample, show_boxes: bool = True, show_polygons: bool = True, show_labels: bool = True) -> np.ndarray:
        """在图像上渲染多边形、边界框及类别标签。"""
        img = cv2.imread(sample.image_path)
        if img is None:
            # 创建空白占位图像
            img = np.zeros((sample.height, sample.width, 3), dtype=np.uint8)

        # 类别颜色调色板
        colors = [
            (255, 56, 56), (255, 157, 15), (255, 112, 31), (255, 178, 29),
            (52, 209, 183), (11, 201, 226), (0, 159, 255), (124, 77, 255)
        ]

        class_to_color = {}
        for idx, cls_name in enumerate(self.classes):
            class_to_color[cls_name] = colors[idx % len(colors)]

        overlay = img.copy()
        for ann in sample.annotations:
            color = class_to_color.get(ann.label, (0, 255, 0))
            pts = np.array(ann.points, dtype=np.int32)

            if show_polygons and len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

            if show_boxes:
                bbox = ann.to_bbox()
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                bbox = ann.to_bbox()
                x1, y1 = int(bbox[0]), int(bbox[1])
                label_text = f"{ann.label}"
                if ann.score is not None:
                    label_text += f" {ann.score:.2f}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), color, -1)
                cv2.putText(img, label_text, (x1 + 3, max(th, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 多边形半透明混合
        if show_polygons:
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        return img

    def export_yolo_dataset(self, output_dir: str, task: str = "segment") -> str:
        """
        将数据集导出为通用结构，同时支持：
        1. Roboflow RF-DETR (train/_annotations.coco.json, valid/_annotations.coco.json, train/images)
        2. Ultralytics YOLO (images/train, labels/train, data.yaml)
        """
        import json
        yaml_path = os.path.join(output_dir, "data.yaml")
        os.makedirs(output_dir, exist_ok=True)
        class_to_id = {c: i for i, c in enumerate(self.classes)}
        categories_coco = [{"id": i, "name": c, "supercategory": "none"} for i, c in enumerate(self.classes)]

        ann_global_id = 1

        for split_name in ["train", "valid", "test"]:
            # YOLO 格式目录
            yolo_split = "val" if split_name == "valid" else split_name
            img_dir_yolo = os.path.join(output_dir, "images", yolo_split)
            lbl_dir_yolo = os.path.join(output_dir, "labels", yolo_split)
            os.makedirs(img_dir_yolo, exist_ok=True)
            os.makedirs(lbl_dir_yolo, exist_ok=True)

            # Roboflow RF-DETR 目录
            rf_dir = os.path.join(output_dir, split_name)
            rf_img_dir = os.path.join(rf_dir, "images")
            rf_lbl_dir = os.path.join(rf_dir, "labels")
            os.makedirs(rf_img_dir, exist_ok=True)
            os.makedirs(rf_lbl_dir, exist_ok=True)

            coco_images = []
            coco_annotations = []

            for img_idx, sample in enumerate(self.splits.get(split_name, []), start=1):
                ext = Path(sample.image_path).suffix
                file_name = f"{sample.sample_id}{ext}"
                dest_img_path = os.path.join(img_dir_yolo, file_name)
                dest_rf_root_img = os.path.join(rf_dir, file_name)
                dest_rf_img_path = os.path.join(rf_img_dir, file_name)

                # 保存或拷贝图像到目标路径
                if hasattr(sample, "_cached_img") and sample._cached_img is not None:
                    cv2.imwrite(dest_img_path, sample._cached_img)
                    cv2.imwrite(dest_rf_root_img, sample._cached_img)
                    cv2.imwrite(dest_rf_img_path, sample._cached_img)
                elif os.path.exists(sample.image_path):
                    shutil.copy2(sample.image_path, dest_img_path)
                    shutil.copy2(sample.image_path, dest_rf_root_img)
                    shutil.copy2(sample.image_path, dest_rf_img_path)

                # COCO 图像条目
                coco_images.append({
                    "id": img_idx,
                    "file_name": file_name,
                    "width": sample.width,
                    "height": sample.height,
                    "tags": list(getattr(sample, "tags", [])),
                })

                # 写入 YOLO 标签文本文件
                dest_lbl_path = os.path.join(lbl_dir_yolo, f"{sample.sample_id}.txt")
                dest_rf_lbl_path = os.path.join(rf_lbl_dir, f"{sample.sample_id}.txt")

                with open(dest_lbl_path, "w", encoding="utf-8") as lf, open(dest_rf_lbl_path, "w", encoding="utf-8") as rflf:
                    for ann in sample.annotations:
                        cls_id = class_to_id.get(ann.label, 0)
                        bbox = ann.to_bbox()
                        bw = (bbox[2] - bbox[0]) / max(1, sample.width)
                        bh = (bbox[3] - bbox[1]) / max(1, sample.height)
                        bx = (bbox[0] + bbox[2]) / 2.0 / max(1, sample.width)
                        by = (bbox[1] + bbox[3]) / 2.0 / max(1, sample.height)

                        if task == "segment" and len(ann.points) >= 3:
                            coords = []
                            poly_flat = []
                            for p in ann.points:
                                nx = np.clip(p[0] / max(1, sample.width), 0.0, 1.0)
                                ny = np.clip(p[1] / max(1, sample.height), 0.0, 1.0)
                                coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
                                poly_flat.extend([float(p[0]), float(p[1])])
                            
                            lbl_line = f"{cls_id} " + " ".join(coords) + "\n"
                            lf.write(lbl_line)
                            rflf.write(lbl_line)

                            # COCO 多边形标注
                            coco_annotations.append({
                                "id": ann_global_id,
                                "image_id": img_idx,
                                "category_id": cls_id,
                                "segmentation": [poly_flat],
                                "area": float(max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))),
                                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                                "iscrowd": 0
                            })
                            ann_global_id += 1
                        else:
                            lbl_line = f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n"
                            lf.write(lbl_line)
                            rflf.write(lbl_line)

                            coco_annotations.append({
                                "id": ann_global_id,
                                "image_id": img_idx,
                                "category_id": cls_id,
                                "segmentation": [],
                                "area": float(max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))),
                                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                                "iscrowd": 0
                            })
                            ann_global_id += 1

            # 为 Roboflow RF-DETR 生成 COCO JSON 文件
            coco_json_path = os.path.join(rf_dir, "_annotations.coco.json")
            with open(coco_json_path, "w", encoding="utf-8") as cjf:
                json.dump({
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "categories": categories_coco
                }, cjf)

        # 为 YOLO 与 RF-DETR 生成 data.yaml（采用相对路径以提高跨机器移植性）
        try:
            rel_output_path = os.path.relpath(output_dir, os.getcwd())
        except Exception:
            rel_output_path = output_dir

        yaml_content = f"""path: {rel_output_path}
train: images/train
val: images/val
test: images/test

names:
"""
        for i, c in enumerate(self.classes):
            yaml_content += f"  {i}: {c}\n"

        with open(yaml_path, "w", encoding="utf-8") as yf:
            yf.write(yaml_content)

        # 同时将 data.yaml 拷贝至根目录及输出目录
        with open(os.path.join(output_dir, "data.yml"), "w", encoding="utf-8") as yf2:
            yf2.write(yaml_content)

        return yaml_path

    @staticmethod
    def _classification_folder_name(label: str) -> str:
        """将类别名称转换为安全且跨平台的目录名称。"""
        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(label).strip())
        return safe.strip("._") or "class"

    def export_classification_dataset(self, output_dir: str) -> str:
        """将 Isolate Objects 生成的切片样本按 split/class 目录结构导出为图像分类数据集。

        隔离抠图后的样本携带 ``classification_label`` 元数据且无检测标注。
        此类输出按设计与 YOLO/RF-DETR 分割导出相互独立，因为分割模型需要边界框或多边形标注。
        """
        os.makedirs(output_dir, exist_ok=True)
        manifest = {"classes": [], "label_folders": {}, "splits": {}}
        labels_seen = set()

        for split_name in ["train", "valid", "test"]:
            split_records = []
            for sample in self.splits.get(split_name, []):
                label = str(getattr(sample, "classification_label", "")).strip()
                if not label:
                    continue

                folder = self._classification_folder_name(label)
                labels_seen.add(label)
                manifest["label_folders"][label] = folder
                class_dir = os.path.join(output_dir, split_name, folder)
                os.makedirs(class_dir, exist_ok=True)

                if hasattr(sample, "_cached_img") and sample._cached_img is not None:
                    filename = f"{sample.sample_id}.jpg"
                    destination = os.path.join(class_dir, filename)
                    if not cv2.imwrite(destination, sample._cached_img):
                        continue
                elif os.path.exists(sample.image_path):
                    suffix = Path(sample.image_path).suffix or ".jpg"
                    filename = f"{sample.sample_id}{suffix}"
                    destination = os.path.join(class_dir, filename)
                    shutil.copy2(sample.image_path, destination)
                else:
                    continue

                split_records.append({
                    "file": os.path.relpath(destination, output_dir),
                    "label": label,
                    "source": sample.image_path,
                })

            manifest["splits"][split_name] = split_records

        manifest["classes"] = sorted(labels_seen)
        with open(os.path.join(output_dir, "classes.names"), "w", encoding="utf-8") as cf:
            cf.write("\n".join(manifest["classes"]) + ("\n" if manifest["classes"] else ""))
        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        return output_dir
