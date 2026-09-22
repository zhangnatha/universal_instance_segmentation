#!/usr/bin/env bash
# 一键编译并安装适用于本项目的轻量化 OpenCV 动态库。
# 默认安装至 3rdparty/opencv 目录，供 CMakeLists.txt 通过 -DOpenCV_DIR 发现。
# 编译时自动启用内置图像编解码器（zlib/png/jpeg/tiff），无需宿主机多媒体库，
# 确保在 Ubuntu 18.04、20.04、22.04 等不同 Linux 环境下具备最大兼容性。
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: install_opencv.sh [options]

Compile and install a lightweight OpenCV runtime for C++ inference.

Options:
  -v, --version VERSION      OpenCV version (default: 4.5.5)
  -p, --prefix, --install-dir DIR
                             Target installation directory (default: <project_root>/3rdparty/opencv)
  -b, --build-dir DIR        CMake build directory (default: <project_root>/build-opencv)
  -j, --jobs N               Number of parallel compilation jobs (default: nproc)
  --keep-temp                Keep temporary build files and downloaded archive after installation
  -h, --help                 Show this help message and exit

Environment variables:
  OPENCV_VERSION             OpenCV version (default: 4.5.5)
  INSTALL_DIR                Target installation directory
  BUILD_DIR                  CMake build directory
  BUILD_JOBS                 Number of parallel compilation jobs
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENCV_VERSION="${OPENCV_VERSION:-4.5.5}"
INSTALL_DIR="${INSTALL_DIR:-${PROJECT_ROOT}/3rdparty/opencv}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build-opencv}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 2)}"
KEEP_TEMP=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--version)
            [[ $# -ge 2 ]] || { echo "Error: missing value for $1" >&2; exit 2; }
            OPENCV_VERSION="$2"
            shift 2
            ;;
        -p|--prefix|--install-dir)
            [[ $# -ge 2 ]] || { echo "Error: missing value for $1" >&2; exit 2; }
            INSTALL_DIR="$2"
            shift 2
            ;;
        -b|--build-dir)
            [[ $# -ge 2 ]] || { echo "Error: missing value for $1" >&2; exit 2; }
            BUILD_DIR="$2"
            shift 2
            ;;
        -j|--jobs)
            [[ $# -ge 2 ]] || { echo "Error: missing value for $1" >&2; exit 2; }
            BUILD_JOBS="$2"
            shift 2
            ;;
        --keep-temp|--no-clean)
            KEEP_TEMP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# 将路径转换为绝对路径
mkdir -p "${INSTALL_DIR}" "${BUILD_DIR}" "${PROJECT_ROOT}/3rdparty"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"

# 检查系统架构
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "Warning: Current architecture is $(uname -m). Recommended target is x86_64." >&2
fi

# 检查基础编译工具链依赖
check_command() {
    local cmd="$1"
    local package_hint="$2"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Error: Required command '${cmd}' was not found." >&2
        echo "Please install prerequisites: sudo apt-get update && sudo apt-get install -y ${package_hint}" >&2
        exit 1
    fi
}

check_command "cmake" "cmake"
check_command "make" "build-essential"
check_command "gcc" "build-essential"
check_command "g++" "build-essential"
check_command "tar" "tar"

# 检查下载工具（优先 wget，次选 curl）
DOWNLOADER=""
if command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
else
    echo "Error: Neither 'wget' nor 'curl' was found. Please install wget or curl." >&2
    exit 1
fi

ARCHIVE_NAME="opencv-${OPENCV_VERSION}.tar.gz"
ARCHIVE_PATH="${PROJECT_ROOT}/3rdparty/${ARCHIVE_NAME}"
RELEASE_URL="https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz"
SOURCE_DIR="${BUILD_DIR}/opencv-${OPENCV_VERSION}"
CMAKE_BUILD_DIR="${BUILD_DIR}/build"

echo "=========================================="
echo "OpenCV Source Build & Installation"
echo "  Version:     ${OPENCV_VERSION}"
echo "  Source:      ${SOURCE_DIR}"
echo "  Build Dir:   ${CMAKE_BUILD_DIR}"
echo "  Install Dir: ${INSTALL_DIR}"
echo "  Jobs:        ${BUILD_JOBS}"
echo "=========================================="

# 下载 OpenCV 源码压缩包（若本地不存在则下载）
if [[ ! -s "${ARCHIVE_PATH}" ]]; then
    echo "Downloading OpenCV ${OPENCV_VERSION} source archive..."
    if [[ "${DOWNLOADER}" == "wget" ]]; then
        wget -c "${RELEASE_URL}" -O "${ARCHIVE_PATH}"
    else
        curl -fSL "${RELEASE_URL}" -o "${ARCHIVE_PATH}"
    fi
else
    echo "Found existing source archive, skipping download: ${ARCHIVE_PATH}"
fi

# 解压源码包
if [[ ! -f "${SOURCE_DIR}/CMakeLists.txt" ]]; then
    echo "Extracting source archive into ${BUILD_DIR}..."
    tar -xzf "${ARCHIVE_PATH}" -C "${BUILD_DIR}"
else
    echo "Found extracted source tree, skipping extraction: ${SOURCE_DIR}"
fi

mkdir -p "${CMAKE_BUILD_DIR}"

# 执行 CMake 配置：仅编译核心推理所需模块，禁用无关测试、工具与外置多媒体依赖
echo "Configuring OpenCV with CMake..."
cmake -S "${SOURCE_DIR}" -B "${CMAKE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_LIST=core,imgproc,imgcodecs \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_PACKAGE=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_JAVA=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_1394=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_GTK=OFF \
    -DWITH_OPENEXR=OFF \
    -DWITH_TBB=OFF \
    -DWITH_IPP=OFF \
    -DWITH_ITT=OFF \
    -DWITH_WEBP=OFF \
    -DBUILD_ZLIB=ON \
    -DBUILD_PNG=ON \
    -DBUILD_JPEG=ON \
    -DBUILD_TIFF=ON \
    -DOPENCV_GENERATE_PKGCONFIG=ON

# 编译 OpenCV
echo "Building OpenCV with ${BUILD_JOBS} parallel job(s)..."
cmake --build "${CMAKE_BUILD_DIR}" -j"${BUILD_JOBS}"

# 安装至目标目录
echo "Installing OpenCV into ${INSTALL_DIR}..."
cmake --build "${CMAKE_BUILD_DIR}" --target install

# 校验核心动态库与 CMake 配置文件完整性
echo "Verifying installation integrity..."
MISSING=0
for lib in "libopencv_core.so" "libopencv_imgproc.so" "libopencv_imgcodecs.so"; do
    if ! find "${INSTALL_DIR}/lib" -maxdepth 2 -name "${lib}*" | grep -q .; then
        echo "Error: Required library '${lib}' not found under ${INSTALL_DIR}/lib" >&2
        MISSING=1
    fi
done

if [[ ! -f "${INSTALL_DIR}/lib/cmake/opencv4/OpenCVConfig.cmake" && \
      ! -f "${INSTALL_DIR}/lib64/cmake/opencv4/OpenCVConfig.cmake" && \
      ! -f "${INSTALL_DIR}/share/opencv4/OpenCVConfig.cmake" ]]; then
    echo "Error: OpenCVConfig.cmake not found in install tree: ${INSTALL_DIR}" >&2
    MISSING=1
fi

if [[ "${MISSING}" -ne 0 ]]; then
    echo "Error: OpenCV installation verification failed." >&2
    exit 1
fi

# 安装校验成功后，自动清除构建临时目录与下载的源码压缩包
if [[ "${KEEP_TEMP}" -eq 0 ]]; then
    echo "Cleaning up temporary build directory and downloaded archive..."
    rm -rf "${BUILD_DIR}"
    rm -f "${ARCHIVE_PATH}"
fi

echo "=========================================="
echo "OpenCV ${OPENCV_VERSION} build and installation completed successfully."
echo "Install path: ${INSTALL_DIR}"
echo ""
echo "To build the project with this OpenCV:"
echo "  cmake -S cpp -B cpp/build \\"
echo "    -DONNXRUNTIME_ROOT=cpp/3rdparty/onnxruntime \\"
echo "    -DOpenCV_DIR=${INSTALL_DIR}/lib/cmake/opencv4"
echo "=========================================="
