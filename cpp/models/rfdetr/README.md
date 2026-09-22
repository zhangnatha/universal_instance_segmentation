# RF-DETR C++ 推理

该程序消费 RF-DETR 1.9.3 官方实例分割导出模型，协议与另外两个后端独立且不兼容：输入为固定 `input[1,3,H,W]`，RGB、ImageNet mean/std；输出为 `dets[1,Q,4]`（归一化 cx/cy/w/h）、`labels[1,Q,C+1]`（C 类 logits + no-object）和可选 `masks[1,Q,Mh,Mw]` mask logits。Q、Mh、Mw 从图形运行时形状读取，类别数由 `--classes` 校验。H/W 必须可被 4 整除；查询选择、置信度筛选、框解码、mask resize/裁剪和统一 JSON RLE 均在 C++ 中完成。

从仓库根目录统一构建：

```bash
cmake -S cpp -B cpp/build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-1.20.x \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake --build cpp/build --target rfdetr_infer -j2
cpp/build/rfdetr_infer --model runs/onnx_export/rfdetr-seg-small.onnx \
  --input /path/to/image_or_dir --output results/cpp_rfdetr \
  --classes class1,class2,class3,class4 --score-threshold 0.50
```

`--class-conf`/`--class-iou` 使用逗号分隔的 `CLASS=VALUE`；未配置类别沿用全局值，IoU=1 禁用同类 bbox greedy NMS，不跨类抑制。

CUDA 目标为 `rfdetr_infer_cuda`，须用 `-DBUILD_GPU=ON -DONNXRUNTIME_GPU_ROOT=...` 配置；`--device cuda` 在 CPU 目标上会明确失败。`--no-draw` 禁止写可视化图像。
