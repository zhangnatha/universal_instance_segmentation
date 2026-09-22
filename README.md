# universal_instance_segmentation
通用工业级实例分割框架，**深度支持 CNN（卷积神经网络：Mask R-CNN, ConvNeXt, YOLO）与 Transformer（自注意力机制：Swin Transformer, RF-DETR）两大主流深度学习架构**。提供 **Detectron2 Mask R-CNN**、**Ultralytics YOLO-seg** 和 **RF-DETR** 三大高性能后端支持。项目打通了**自建数据集划分与预处理、长尾均衡与物理遮挡仿真、在线数据增强、模型微调训练、ONNX 跨平台导出、C++/Python 双端推理**以及 **LabelMe 自动化对比评测**的全流程闭环。代码架构纯净通用，用户可直接在自己的自定义工业或自动化视觉数据集上开箱即用。
