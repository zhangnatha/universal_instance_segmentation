# Detectron2 Mask R-CNN C++ 推理

该程序只消费仓库 Detectron2 导出器生成的固定分辨率 ONNX：输入节点为 `image[3,H,W]`（CHW、BGR/RGB 取决于导出配置，默认 BGR），输出节点按名称为 `boxes[N,4]`（xyxy，原图坐标）、`scores[N]`、`classes[N]`（int64）和 `mask_probs[N,1,Mh,Mw]`。导出脚本会将检测数设为动态维度，但空间尺寸固定为样本图像尺寸。C++ 端会严格校验节点和形状，避免误把 YOLO 或 RF-DETR 模型送入本目标。

输入支持单张图像或目录批处理，目录内按文件名排序并忽略 LabelMe JSON 等非图像文件；支持 BMP、PNG、JPEG、TIFF 和 WebP。输入尺寸必须与 ONNX 导出样本的固定 H×W 一致，程序不会静默 resize（尺寸不符会明确报错，避免框和 mask 错位）。单图模式沿用 `--output` 指定的图像路径；目录模式将每张图输出为 `<stem>.png` 和同名 JSON。

从 `cpp/` 目录统一构建：

```bash
cmake -S cpp -B cpp/build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake --build cpp/build --target detectron2_maskrcnn_infer -j2
cpp/build/detectron2_maskrcnn_infer --model model.onnx --input images/ \
  --output results/detectron2 --device cpu \
  --classes class1,class2,class3,class4 --score-threshold 0.5 \
  --class-conf class1=0.45,class2=0.35 --class-iou class1=0.50,class2=1.0
```

GPU 版本使用 `-DBUILD_GPU=ON -DONNXRUNTIME_GPU_ROOT=...`，目标名为 `detectron2_maskrcnn_infer_cuda`。CUDA 目标要求 GPU 版 ONNX Runtime；CPU 目标不加载 CUDA provider。

`--classes` 支持逗号分隔的类别列表，或直接传入 `classes.names` 文本文件路径（例如 `--classes /path/to/classes.names`）。
输出 JSON 除包含 `mask_rle` 游程编码外，还包含图像 `width`、`height` 以及多边形坐标 `mask_polygons`，方便直接下游解析。

`--class-conf` 执行逐类 score 过滤，`--class-iou` 对 ONNX 返回的候选执行逐类 bbox greedy NMS；该 Detectron2 ONNX 导出协议已在图内完成候选/NMS，C++ 无法恢复已删除候选，因此逐类 NMS只能作用于导出图保留下来的候选。需要完整候选集时，应导出关闭内部 NMS的图。

## 实用工具

1. **转换为 LabelMe / X-AnyLabeling 格式**：
```bash
python -m instance_segmentation.cli tool convert-to-labelme -i results/detectron2 -o results/labelme
```

2. **便捷导出 Swin-S Mask R-CNN ONNX**：
```bash
python -m instance_segmentation.cli tool swin-export \
  --weights output/model_best.pth \
  --output output/onnx/model.onnx \
  --classes class1 class2 class3 class4
```

3. **对齐真值评估**：
```bash
python scripts/compare_json.py \
  --ground-truth-dir /path/to/LabelMe_GT \
  --prediction-dir results/detectron2 \
  --output-dir results/evaluation
```
