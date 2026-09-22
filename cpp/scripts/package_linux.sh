#!/usr/bin/env bash
# 一键打包已编译的 Linux C++ 推理可执行程序及其依赖库。
# 生成的部署包为自包含免安装包，支持直接复制到 Ubuntu 18.04+ 及其他主流 Linux x86_64 新机器上运行。
# 宿主机只需具备基础 glibc (>=2.27) 与 NVIDIA 内核驱动（GPU 模式）。
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Package built Linux C++ inference executables and runtime libraries.
The output package is self-contained and ready to run on target machines (Ubuntu 18.04+).

Usage: package_linux.sh [options]
  --build-dir DIR       CMake build directory (default: cpp/build, build-cuda, build)
  --output DIR          Target package directory (default: dist/universal_instance_segmentation_linux-x86_64)
  --target NAME         Specific target to package (default: all available executables)
  --mode cpu|cuda       Runtime mode (default: auto-detected from build)
  --ort-root DIR        ONNX Runtime root directory (default: read from cache or 3rdparty)
  --opencv-root DIR     OpenCV root directory (default: read from cache or 3rdparty)
  --cuda-root DIR       CUDA toolkit / runtime root directory (for GPU packages)
  --runtime-dir DIR     Additional runtime library directory (may be repeated)
  --model PATH          Model file to bundle into the package
  --no-models           Do not bundle any model file
  --force               Overwrite existing output directory
  -h, --help            Show this help message
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CPP_DIR}/.." && pwd)"

BUILD_DIR=""
OUTPUT_DIR="${PROJECT_ROOT}/dist/universal_instance_segmentation_linux-x86_64"
TARGET_NAME=""
MODE=""
ORT_ROOT=""
OPENCV_ROOT=""
CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-}}"
MODEL_PATH=""
NO_MODELS=0
FORCE=0
EXTRA_RUNTIME_DIRS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir|--output|--target|--mode|--ort-root|--opencv-root|--cuda-root|--runtime-dir|--model)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --build-dir) BUILD_DIR="$2" ;;
                --output) OUTPUT_DIR="$2" ;;
                --target) TARGET_NAME="$2" ;;
                --mode) MODE="$2" ;;
                --ort-root) ORT_ROOT="$2" ;;
                --opencv-root) OPENCV_ROOT="$2" ;;
                --cuda-root) CUDA_ROOT="$2" ;;
                --runtime-dir) EXTRA_RUNTIME_DIRS+=("$2") ;;
                --model) MODEL_PATH="$2" ;;
            esac
            shift 2 ;;
        --no-models) NO_MODELS=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

# 自动发现 CMake 构建目录
if [[ -z "$BUILD_DIR" ]]; then
    for candidate in "${CPP_DIR}/build" "${PROJECT_ROOT}/build" "${PROJECT_ROOT}/build-cuda" "${CPP_DIR}/build-cuda"; do
        if [[ -f "$candidate/CMakeCache.txt" ]]; then
            BUILD_DIR="$candidate"
            break
        fi
    done
fi
[[ -n "$BUILD_DIR" && -f "$BUILD_DIR/CMakeCache.txt" ]] || {
    echo "CMakeCache.txt not found. Please build the project first or specify --build-dir DIR" >&2
    exit 1
}
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

# 读取 CMakeCache.txt 中的配置变量
cache_value() {
    local key="$1" value
    value="$(sed -n -E "s#^${key}:[^=]*=(.*)#\1#p" "$BUILD_DIR/CMakeCache.txt" | tail -n 1)"
    printf '%s' "$value"
}

# 发现待打包的目标可执行文件列表
KNOWN_TARGETS=(
    "detectron2_maskrcnn_infer"
    "detectron2_maskrcnn_infer_cuda"
    "ultralytics_yolo_infer"
    "ultralytics_yolo_infer_cuda"
    "rfdetr_infer"
    "rfdetr_infer_cuda"
)

FOUND_BINS=()
if [[ -n "$TARGET_NAME" && "$TARGET_NAME" != "all" ]]; then
    if [[ -x "${BUILD_DIR}/${TARGET_NAME}" ]]; then
        FOUND_BINS+=("${BUILD_DIR}/${TARGET_NAME}")
    else
        found="$(find "$BUILD_DIR" -maxdepth 3 -type f -name "$TARGET_NAME" -perm -u+x -print -quit 2>/dev/null || true)"
        if [[ -n "$found" && -x "$found" ]]; then
            FOUND_BINS+=("$found")
        fi
    fi
else
    for tgt in "${KNOWN_TARGETS[@]}"; do
        if [[ -x "${BUILD_DIR}/${tgt}" ]]; then
            FOUND_BINS+=("${BUILD_DIR}/${tgt}")
        else
            found="$(find "$BUILD_DIR" -maxdepth 3 -type f -name "$tgt" -perm -u+x -print -quit 2>/dev/null || true)"
            if [[ -n "$found" && -x "$found" ]]; then
                FOUND_BINS+=("$found")
            fi
        fi
    done
fi

if [[ ${#FOUND_BINS[@]} -eq 0 ]]; then
    echo "No inference executables found in $BUILD_DIR." >&2
    echo "Please build at least one target before packaging." >&2
    exit 1
fi

echo "Found executables to package: ${#FOUND_BINS[@]}"
for b in "${FOUND_BINS[@]}"; do
    echo "  - $(basename "$b")"
done

# 解析运行模式（CPU 或 CUDA）
if [[ -z "$MODE" ]]; then
    cached_gpu="$(cache_value BUILD_GPU)"
    case "${cached_gpu,,}" in
        on|true|1) MODE="cuda" ;;
        *)
            # 检查待打包二进制是否包含 _cuda
            has_cuda=0
            for b in "${FOUND_BINS[@]}"; do
                if [[ "$(basename "$b")" == *"_cuda"* ]]; then
                    has_cuda=1
                    break
                fi
            done
            if [[ $has_cuda -eq 1 ]]; then MODE="cuda"; else MODE="cpu"; fi
            ;;
    esac
fi

# 自动推导或回退 ONNX Runtime 根目录
ORT_ROOT="${ORT_ROOT:-$(cache_value ONNXRUNTIME_ROOT)}"
if [[ "$MODE" == "cuda" && -z "$ORT_ROOT" ]]; then
    ORT_ROOT="$(cache_value ONNXRUNTIME_GPU_ROOT)"
fi
if [[ -z "$ORT_ROOT" || ! -d "$ORT_ROOT" ]]; then
    for candidate in "${CPP_DIR}/3rdparty/onnxruntime-linux-x64-1.20.0" \
                     "${CPP_DIR}/3rdparty/onnxruntime-cuda11-1.20.0" \
                     "${CPP_DIR}/3rdparty/onnxruntime"; do
        if [[ -d "$candidate/lib" ]]; then
            ORT_ROOT="$candidate"
            break
        fi
    done
fi

# 自动推导或回退 OpenCV 根目录
OPENCV_DIR_VAL="$(cache_value OpenCV_DIR)"
if [[ -n "$OPENCV_DIR_VAL" && -z "$OPENCV_ROOT" ]]; then
    # OpenCV_DIR 通常指向 <root>/lib/cmake/opencv4
    cand_root="$(cd "$OPENCV_DIR_VAL/../../.." 2>/dev/null && pwd || true)"
    if [[ -d "$cand_root/lib" ]]; then
        OPENCV_ROOT="$cand_root"
    fi
fi
if [[ -z "$OPENCV_ROOT" || ! -d "$OPENCV_ROOT" ]]; then
    for candidate in "${CPP_DIR}/3rdparty/opencv-install-4.5.5" \
                     "${CPP_DIR}/3rdparty/opencv"; do
        if [[ -d "$candidate/lib" ]]; then
            OPENCV_ROOT="$candidate"
            break
        fi
    done
fi

# 自动推导 CUDA 11 运行时库目录
CUDA_RUNTIME_DIR="$(cache_value ONNXRUNTIME_GPU_RUNTIME_ROOT)"
if [[ -n "$CUDA_RUNTIME_DIR" && -d "$CUDA_RUNTIME_DIR/lib" ]]; then
    EXTRA_RUNTIME_DIRS+=("$CUDA_RUNTIME_DIR/lib")
elif [[ -d "${CPP_DIR}/3rdparty/cuda11-runtime/lib" ]]; then
    EXTRA_RUNTIME_DIRS+=("${CPP_DIR}/3rdparty/cuda11-runtime/lib")
fi

echo "Packaging configuration:"
echo "  Mode:        $MODE"
echo "  ORT Root:    ${ORT_ROOT:-<none>}"
echo "  OpenCV Root: ${OPENCV_ROOT:-<none>}"
echo "  Output:      $OUTPUT_DIR"

# 准备输出目录结构
if [[ -e "$OUTPUT_DIR" ]]; then
    if [[ "$FORCE" -eq 1 ]]; then
        rm -rf -- "$OUTPUT_DIR"
    else
        echo "Output directory exists: $OUTPUT_DIR (use --force to overwrite)" >&2
        exit 1
    fi
fi
mkdir -p "$OUTPUT_DIR/bin" "$OUTPUT_DIR/lib" "$OUTPUT_DIR/models"

# 拷贝可执行文件
for bin_path in "${FOUND_BINS[@]}"; do
    bin_name="$(basename "$bin_path")"
    cp -a "$bin_path" "$OUTPUT_DIR/bin/"
    cp -a "$bin_path" "$OUTPUT_DIR/"
done

# 动态库拷贝与软链接解析辅助函数
copy_file() {
    local source="$1" destination="$OUTPUT_DIR/lib/$(basename "$1")" real
    [[ -e "$source" || -L "$source" ]] || return 0
    [[ -e "$destination" || -L "$destination" ]] || cp -a -- "$source" "$destination"
    if [[ -L "$source" ]]; then
        real="$(readlink -f "$source" 2>/dev/null || true)"
        if [[ -n "$real" && -f "$real" ]]; then
            destination="$OUTPUT_DIR/lib/$(basename "$real")"
            [[ -e "$destination" || -L "$destination" ]] || cp -a -- "$real" "$destination"
        fi
    fi
}

copy_glob() {
    local pattern="$1" directory glob item
    directory="${pattern%/*}"
    glob="${pattern##*/}"
    [[ -d "$directory" ]] || return 0
    while IFS= read -r -d '' item; do copy_file "$item"; done < <(
        find "$directory" -maxdepth 1 \( -type f -o -type l \) -name "$glob" -print0 2>/dev/null || true
    )
}

# 判定是否属于宿主机操作系统基础库（glibc/libstdc++/NVIDIA 内核驱动驱动）
is_host_library() {
    local base
    base="$(basename "$1")"
    case "$base" in
        libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|ld-linux*|libgcc_s.so*) return 0 ;;
        libcuda.so*|libnvidia-*|linux-vdso.so*) return 0 ;;
    esac
    case "$1" in
        /lib/*|/lib64/*) return 0 ;;
        /usr/lib/*|/usr/lib64/*)
            if [[ "$base" == libopencv_* || "$base" == libonnxruntime* ]]; then
                return 1
            fi
            return 0 ;;
        *) return 1 ;;
    esac
}

declare -A VISITED=()
copy_dependency_tree() {
    local source="$1" real dep
    [[ -e "$source" || -L "$source" ]] || return 0
    real="$(readlink -f "$source" 2>/dev/null || true)"
    [[ -n "$real" && -f "$real" ]] || return 0
    is_host_library "$real" && return 0
    [[ -n "${VISITED[$real]:-}" ]] && return 0
    VISITED["$real"]=1
    local destination="$OUTPUT_DIR/lib/$(basename "$source")"
    [[ -e "$destination" || -L "$destination" ]] || cp -a -- "$source" "$destination"
    if [[ -L "$source" ]]; then
        destination="$OUTPUT_DIR/lib/$(basename "$real")"
        [[ -e "$destination" || -L "$destination" ]] || cp -a -- "$real" "$destination"
    fi
    while IFS= read -r dep; do
        [[ -n "$dep" ]] || continue
        copy_dependency_tree "$dep"
    done < <(LD_LIBRARY_PATH="$LDD_LIBRARY_PATH" ldd "$source" 2>/dev/null |
        awk '$2 == "=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}')
}

# 构建 ldd 解析搜索路径
LDD_LIBRARY_PATH=""
if [[ -n "$ORT_ROOT" && -d "$ORT_ROOT/lib" ]]; then
    LDD_LIBRARY_PATH="${LDD_LIBRARY_PATH:+${LDD_LIBRARY_PATH}:}${ORT_ROOT}/lib"
fi
if [[ -n "$OPENCV_ROOT" && -d "$OPENCV_ROOT/lib" ]]; then
    LDD_LIBRARY_PATH="${LDD_LIBRARY_PATH:+${LDD_LIBRARY_PATH}:}${OPENCV_ROOT}/lib"
fi
for d in "${EXTRA_RUNTIME_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        LDD_LIBRARY_PATH="${LDD_LIBRARY_PATH:+${LDD_LIBRARY_PATH}:}${d}"
    fi
done
if [[ -n "$CUDA_ROOT" ]]; then
    for d in "$CUDA_ROOT/lib64" "$CUDA_ROOT/targets/x86_64-linux/lib"; do
        if [[ -d "$d" ]]; then
            LDD_LIBRARY_PATH="${LDD_LIBRARY_PATH:+${LDD_LIBRARY_PATH}:}${d}"
        fi
    done
fi
LDD_LIBRARY_PATH="${LDD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 复制 ONNX Runtime 动态库
if [[ -n "$ORT_ROOT" && -d "$ORT_ROOT/lib" ]]; then
    copy_glob "$ORT_ROOT/lib/libonnxruntime.so*"
    copy_glob "$ORT_ROOT/lib/libonnxruntime_providers_*.so*"
fi

# 复制 OpenCV 动态库
if [[ -n "$OPENCV_ROOT" && -d "$OPENCV_ROOT/lib" ]]; then
    copy_glob "$OPENCV_ROOT/lib/libopencv_*.so*"
fi

# 复制额外运行时（如 CUDA 11 独立运行时库）
for d in "${EXTRA_RUNTIME_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        copy_glob "$d/*.so*"
    fi
done

# 通过 ldd 递归提取所有二进制及 provider 的动态链接依赖
for bin_path in "${FOUND_BINS[@]}"; do
    while IFS= read -r dependency; do
        [[ -n "$dependency" ]] || continue
        copy_dependency_tree "$dependency"
    done < <(LD_LIBRARY_PATH="$LDD_LIBRARY_PATH" ldd "$bin_path" 2>/dev/null |
        awk '$2 == "=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}')
done

for provider in "$OUTPUT_DIR"/lib/libonnxruntime_providers_*.so*; do
    if [[ -f "$provider" ]]; then
        copy_dependency_tree "$provider"
    fi
done

# 使用 patchelf 修正 RPATH，确保可执行文件在新机上无论从根目录还是 bin/ 调用均能自定位 lib/
if command -v patchelf >/dev/null 2>&1; then
    force_flag=""
    if patchelf --help 2>&1 | grep -q -- '--force-rpath'; then
        force_flag="--force-rpath"
    fi
    for bin in "$OUTPUT_DIR"/* "$OUTPUT_DIR"/bin/*; do
        if [[ -f "$bin" && -x "$bin" && "$(basename "$bin")" != *.sh ]]; then
            patchelf $force_flag --set-rpath '$ORIGIN/lib:$ORIGIN/../lib' "$bin" 2>/dev/null || true
        fi
    done
    for so in "$OUTPUT_DIR"/lib/*.so*; do
        if [[ -f "$so" && ! -L "$so" ]]; then
            patchelf $force_flag --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
        fi
    done
    echo "Configured RPATH using patchelf."
else
    echo "Warning: patchelf not found; run.sh launcher will configure LD_LIBRARY_PATH." >&2
fi

# 拷贝可选的模型文件
if [[ "$NO_MODELS" -eq 0 ]]; then
    if [[ -n "$MODEL_PATH" ]]; then
        if [[ -f "$MODEL_PATH" ]]; then
            cp -a "$MODEL_PATH" "$OUTPUT_DIR/models/"
            echo "Bundled model: $MODEL_PATH"
        else
            echo "Warning: Specified model not found: $MODEL_PATH" >&2
        fi
    elif [[ -f "${PROJECT_ROOT}/benchmark_model/model.onnx" ]]; then
        cp -a "${PROJECT_ROOT}/benchmark_model/model.onnx" "$OUTPUT_DIR/models/"
        echo "Bundled default benchmark model."
    fi
fi

# 拷贝类别定义文件（若存在）
if [[ -f "${PROJECT_ROOT}/classes.names" ]]; then
    cp -a "${PROJECT_ROOT}/classes.names" "$OUTPUT_DIR/"
fi

# 拷贝配套后处理与评估工具脚本
for script in compare_json.py convert_to_labelme.py; do
    if [[ -f "${PROJECT_ROOT}/scripts/$script" ]]; then
        cp -a "${PROJECT_ROOT}/scripts/$script" "$OUTPUT_DIR/"
        chmod +x "$OUTPUT_DIR/$script"
    fi
done

# 生成通用启动脚本 run.sh
cat > "$OUTPUT_DIR/run.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

usage() {
    cat << 'HELP_EOF'
Universal Instance Segmentation - Linux C++ Runtime Launcher
Usage:
  ./run.sh <target_executable> [options...]
  ./<target_executable> [options...]

Available executables in this package:
HELP_EOF
    for exe in "$ROOT"/bin/* "$ROOT"/*; do
        if [[ -f "$exe" && -x "$exe" && "$(basename "$exe")" != *.sh && "$(basename "$exe")" != *.py ]]; then
            echo "  - $(basename "$exe")"
        fi
    done | sort -u
    cat << 'HELP_EOF'

Examples:
  # Detectron2 Mask R-CNN:
  ./run.sh detectron2_maskrcnn_infer --model models/model.onnx --input images/ --output results/d2 --classes class1,class2

  # Ultralytics YOLO-seg:
  ./run.sh ultralytics_yolo_infer --model models/yolo.onnx --input images/ --output results/yolo --classes class1,class2

  # RF-DETR:
  ./run.sh rfdetr_infer --model models/rfdetr.onnx --input images/ --output results/rfdetr --classes class1,class2
HELP_EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

TARGET="$1"
shift

BIN_EXE=""
if [[ -x "$ROOT/$TARGET" ]]; then
    BIN_EXE="$ROOT/$TARGET"
elif [[ -x "$ROOT/bin/$TARGET" ]]; then
    BIN_EXE="$ROOT/bin/$TARGET"
fi

if [[ -z "$BIN_EXE" ]]; then
    echo "Error: Executable '$TARGET' not found in package." >&2
    usage >&2
    exit 1
fi

exec "$BIN_EXE" "$@"
EOF
chmod +x "$OUTPUT_DIR/run.sh"

# 生成部署包详细说明文档
cat > "$OUTPUT_DIR/README.txt" << 'EOF'
================================================================================
通用实例分割 (Universal Instance Segmentation) C++ Linux 独立部署包
================================================================================

本部署包包含编译完成的 C++ 高性能推理可执行程序、ONNX Runtime 运行时、OpenCV 核心动态库、
CUDA/cuDNN 运行时库（若包含 GPU 目标）以及一键运行包装脚本与后处理工具。
目标机器无需配置复杂开发环境或拉取源码，解压即可直接运行。

宿主机基础环境要求：
  - Linux x86_64（兼容 Ubuntu 18.04 及以上版本、CentOS 7+、Debian 10+）
  - glibc >= 2.27
  - GPU 模式要求宿主机已安装 NVIDIA 官方显卡驱动

--------------------------------------------------------------------------------
一、快速运行指南
--------------------------------------------------------------------------------

部署包内所有二进制可执行程序均已注入 $ORIGIN 运行时动态链接库路径（RPATH）。
您可以直接执行目标程序，也可以通过 run.sh 包装脚本调用：

1. Detectron2 Mask R-CNN 推理：
   ./detectron2_maskrcnn_infer \
     --model models/model.onnx \
     --input /path/to/test_images \
     --output results/pred_d2 \
     --classes class1,class2,class3 \
     --score-threshold 0.50 \
     --class-conf class1=0.60,class2=0.50 \
     --class-iou class1=0.50,class2=0.50 \
     --device cpu

2. Ultralytics YOLO-seg 推理：
   ./ultralytics_yolo_infer \
     --model models/yolo.onnx \
     --input /path/to/test_images \
     --output results/pred_yolo \
     --classes class1,class2,class3 \
     --score-threshold 0.25 \
     --iou-threshold 0.45 \
     --device cpu

3. RF-DETR 推理：
   ./rfdetr_infer \
     --model models/rfdetr.onnx \
     --input /path/to/test_images \
     --output results/pred_rfdetr \
     --classes class1,class2,class3 \
     --score-threshold 0.40 \
     --iou-threshold 0.50 \
     --device cpu

4. GPU 目标加速（如包含 *_cuda 可执行程序）：
   将上述目标名称替换为 *_infer_cuda，并将 --device 参数设置为 cuda 即可。

5. 统一入口脚本调用：
   ./run.sh detectron2_maskrcnn_infer --model models/model.onnx ...

--------------------------------------------------------------------------------
二、评估与格式转换脚本
--------------------------------------------------------------------------------
- compare_json.py：将推理输出的 JSON 预测与 LabelMe 真值标注对标，计算 Precision/Recall/F1 并生成报告。
- convert_to_labelme.py：将 C++ 推理 JSON 一键转换为标准 LabelMe 多边形格式，便于人工复核修正。
EOF

# 生成打包清单文件
cat > "$OUTPUT_DIR/manifest.txt" << EOF
packaged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mode=$MODE
build_dir=$BUILD_DIR
ort_root=${ORT_ROOT:-none}
opencv_root=${OPENCV_ROOT:-none}
executables=$(printf '%s ' "${FOUND_BINS[@]}")
EOF

echo "================================================================================"
echo "Linux package created successfully: $OUTPUT_DIR"
echo "Package size:"
du -sh "$OUTPUT_DIR"
echo "================================================================================"
