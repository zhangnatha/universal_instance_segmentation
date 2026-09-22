# C++ 推理引擎与跨平台部署

本项目提供统一的 C++ 推理实现，三个模型协议分别置于 `models/` 目录下，由 `CMakeLists.txt` 统一发现 ONNX Runtime 与 OpenCV 依赖并组织构建。各模型拥有独立的输入前处理与后处理逻辑，避免协议串用。

```text
cpp/
├── CMakeLists.txt                 # 统一 CMake 入口；跨平台依赖发现、编译配置与一键打包
├── common/
│   ├── glibc_compat.cpp           # Linux glibc 2.27 (Ubuntu 18.04+) 兼容层符号重定向
│   ├── run_logger.hpp             # 跨平台双向终端/日志分流记录器（支持 Windows 与 Linux）
│   └── colors.hpp                 # 动态类别调色板与 HSV 转换
├── models/
│   ├── detectron2_maskrcnn/       # Detectron2：boxes/scores/classes/mask_probs
│   │   ├── src/main.cpp
│   │   └── README.md
│   ├── ultralytics_yolo/          # Ultralytics YOLO-seg：detections/prototypes
│   │   ├── src/main.cpp
│   │   └── README.md
│   └── rfdetr/                    # RF-DETR 1.9.3：dets/labels/masks
│       ├── src/main.cpp
│       └── README.md
├── scripts/                       # 依赖拉取、一键打包与系统优化脚本
│   ├── fetch_onnxruntime.sh / .ps1
│   ├── fetch_opencv.sh / .ps1
│   ├── package_linux.sh           # Linux 一键打包部署脚本
│   └── package_windows.ps1 / .bat # Windows 一键打包部署脚本
└── 3rdparty/                      # 外部依赖解压/安装目录（不纳入 Git）
```

---

## 1. 跨平台兼容性保障

### 1.1 Ubuntu 18.04+ (glibc 2.27+) 支持
- **glibc 符号重定向兼容层**：内置 `common/glibc_compat.cpp`，通过链接器 `--wrap=__libc_start_main` 将主入口重定向绑定至 `GLIBC_2.2.5` 符号，彻底解决在 Ubuntu 20.04/22.04/24.04 编译后的二进制移植到 Ubuntu 18.04 运行时报错 `GLIBC_2.34 not found` 的问题。
- **旧版 GCC 支持**：针对 Ubuntu 18.04 默认的 GCC 7.x/8.x 自动链接 `stdc++fs`，保障 C++17 `std::filesystem` 顺畅编译。
- **自定位 RPATH**：二进制文件自动注入 `$ORIGIN/lib` 与 `$ORIGIN/../lib`，解压后可直接定位同级动态库，无需全局污染宿主机的 `LD_LIBRARY_PATH`。

### 1.2 Windows 10 与 Windows 11 支持
- **原生 MSVC / Visual Studio 支持**：自动定义 `NOMINMAX`、`_CRT_SECURE_NO_WARNINGS` 及 `_USE_MATH_DEFINES`。
- **跨平台时间与目录操作**：日志系统自适应 Windows `localtime_s` 与 Linux `localtime_r`；文件目录操作采用标准 `std::filesystem`，天然兼容 `\` 与 `/` 路径分隔符。
- **DLL 依赖自包含**：打包时自动提取所有运行时依赖 DLL（ONNX Runtime / OpenCV / CUDA）置于可执行文件同级目录，解压即用。

---

## 2. 编译构建

### 2.1 准备第三方依赖
可使用 `cpp/scripts/` 目录下的自动化拉取脚本（亦可自行安装并指定路径）：

**Linux (Ubuntu 18.04+)：**
```bash
# 拉取并安装 ONNX Runtime CPU 版
bash cpp/scripts/fetch_onnxruntime.sh

# 编译或安装轻量化 OpenCV 4.5.5
bash cpp/scripts/fetch_opencv.sh
```

**Windows 10 / 11：**
```powershell
# PowerShell 环境下拉取 Windows 版依赖
.\cpp\scripts\fetch_onnxruntime.ps1
.\cpp\scripts\fetch_opencv.ps1
```

### 2.2 Linux 编译

```bash
# 1. 快速构建 CPU 版本（CMake 会自动检测 3rdparty 目录下的依赖）
cmake -S cpp -B cpp/build
cmake --build cpp/build -j4

# 2. 构建 GPU (CUDA) 版本
cmake -S cpp -B cpp/build -DBUILD_GPU=ON
cmake --build cpp/build -j4
```

### 2.3 Windows 编译

```cmd
:: 使用 Visual Studio Developer Command Prompt 或普通 CMD (已安装 CMake & MSVC)
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build cpp/build --config Release --parallel 4
```

构建生成的可执行文件：

| CMake Target | 对应协议 | 入口源码 |
|---|---|---|
| `detectron2_maskrcnn_infer` | Detectron2 Mask R-CNN ONNX | `models/detectron2_maskrcnn/src/main.cpp` |
| `ultralytics_yolo_infer` | Ultralytics YOLO-seg ONNX | `models/ultralytics_yolo/src/main.cpp` |
| `rfdetr_infer` | RF-DETR 1.9.3 官方导出 ONNX | `models/rfdetr/src/main.cpp` |
| `*_cuda` | 上述对应目标的 CUDA GPU 加速版本 | （启用 `BUILD_GPU=ON` 时生成） |

---

## 3. C++ 可执行程序一键打包（在新机器上独立运行）

为了在没有开发环境与编译工具链的纯净新机器（Ubuntu 18+ 或 Windows 10/11）上运行 C++ 推理程序，本项目提供了一键打包机制，自动提取可执行程序、ONNX Runtime 运行时库、OpenCV 核心动态库、CUDA 运行时库、模型权重、测试工具与启动脚本，输出即开即用的独立免安装发布包。

### 3.1 一键打包命令

**方式一：通过 CMake 一键打包（推荐，跨平台通用）**
```bash
# 编译完成后直接执行打包目标
cmake --build cpp/build --target package_infer
```

**方式二：通过独立脚本一键打包**

* **Linux 环境（在项目根目录执行）：**
  ```bash
  bash scripts/package_cpp.sh
  # 或：bash cpp/scripts/package_linux.sh --build-dir cpp/build --force
  ```

* **Windows 环境（CMD 或 PowerShell）：**
  ```cmd
  scripts\package_cpp.bat
  :: 或：powershell -ExecutionPolicy Bypass -File cpp\scripts\package_windows.ps1 -BuildDir cpp\build -Force
  ```

打包产物将输出至：
- Linux：`dist/universal_instance_segmentation_linux-x86_64/`
- Windows：`dist\universal_instance_segmentation_windows-x64\`

---

## 4. 在纯净新机器上部署与运行

将打包生成的 `dist` 目录压缩（如 tar.gz 或 zip）拷贝至目标新机，解压即可运行：

### 4.1 新 Linux 机器运行示例 (Ubuntu 18.04+)
```bash
cd universal_instance_segmentation_linux-x86_64

# 方法 A：直接执行二进制（已内嵌 $ORIGIN RPATH，自动加载 lib/ 目录动态库）
./detectron2_maskrcnn_infer \
  --model models/model.onnx \
  --input /path/to/test_images \
  --output results/pred_d2 \
  --classes class1,class2,class3 \
  --class-conf class1=0.60,class2=0.50 \
  --class-iou class1=0.50,class2=0.50 \
  --device cpu

# 方法 B：通过统一启动包装脚本执行
./run.sh ultralytics_yolo_infer \
  --model models/yolo.onnx \
  --input /path/to/test_images \
  --output results/pred_yolo \
  --classes class1,class2 \
  --device cpu

./run.sh rfdetr_infer \
  --model models/rfdetr.onnx \
  --input /path/to/test_images \
  --output results/pred_rfdetr \
  --classes class1,class2 \
  --device cpu
```

### 4.2 新 Windows 机器运行示例 (Windows 10 / 11)
```cmd
cd universal_instance_segmentation_windows-x64

:: 通过 run.bat 启动器执行（自动配置 DLL 搜索路径）
run.bat detectron2_maskrcnn_infer ^
  --model models\model.onnx ^
  --input C:\data\images ^
  --output results\pred_d2 ^
  --classes class1,class2 ^
  --device cpu

run.bat ultralytics_yolo_infer ^
  --model models\yolo.onnx ^
  --input C:\data\images ^
  --output results\pred_yolo ^
  --classes class1,class2 ^
  --device cpu

run.bat rfdetr_infer ^
  --model models\rfdetr.onnx ^
  --input C:\data\images ^
  --output results\pred_rfdetr ^
  --classes class1,class2 ^
  --device cpu
```
