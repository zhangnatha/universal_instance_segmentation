# Ultralytics YOLO segmentation C++ 推理

该目标面向 Ultralytics YOLOv8/YOLO11/YOLO26 `*-seg` 的 ONNX 导出。输入是固定 `image[1,3,H,W]`，RGB、`float32`、`[0,1]`，使用保持纵横比的灰色 `114` letterbox。标准导出产生两个输出：

| 输出 | 形状 | 含义 |
|---|---|---|
| detections | `[1,C,A]` 或 `[1,A,C]` | 每个候选的 `cx,cy,w,h`（letterbox 坐标）、类别分数和 mask coefficients |
| prototypes | `[1,M,Mh,Mw]` | 原型 mask logits |

其中 `C=4+num_classes+M`（YOLOv8/11/26，无 objectness）或 `5+num_classes+M`（旧 YOLOv5-seg，有 objectness）；`A` 是候选数量，`M` 从 prototype 自动校验。坐标通常是输入像素坐标；如果导出图输出归一化坐标（所有坐标绝对值不超过 2），程序按输入宽高还原。类别顺序必须与 `--classes` 和导出模型一致。

程序自动识别两个 rank/layout，若输出多于一个 rank-3 或 rank-4 张量、类别数/原型通道不匹配、输入空间尺寸是动态维度，会 fail-fast。后处理是每类置信度筛选、按类别 greedy NMS、系数线性组合原型、sigmoid、阈值化、letterbox 逆映射和统一 JSON RLE；不会把 YOLO 输出当作 Detectron2 的 `mask_probs`。

```bash
cmake --build cpp/build --target ultralytics_yolo_infer -j2
cpp/build/ultralytics_yolo_infer --model yolo11n-seg.onnx --input image.jpg \
  --output results/cpp_yolo --classes class1,class2,class3,class4 \
  --device cpu --score-threshold 0.25 --iou-threshold 0.45
```

`--class-conf class1=0.45,class2=0.35` 和 `--class-iou class1=0.50,class2=1.0` 可逐类覆盖全局值；未配置类别回退到全局值，IoU=1 禁用同类 bbox greedy NMS。

CUDA 目标为 `ultralytics_yolo_infer_cuda`，须用 `-DBUILD_GPU=ON -DONNXRUNTIME_GPU_ROOT=...` 配置。模型无须在仓库内提交。
