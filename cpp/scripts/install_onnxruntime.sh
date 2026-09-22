#!/bin/bash
# 下载并安装 ONNX Runtime 的 CPU 与 CUDA GPU 软件包到 3rdparty。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
ORT_VERSION="${ORT_VERSION:-1.28.0}"
CUDA_MAJOR_MINOR="${CUDA_MAJOR_MINOR:-13}"
INSTALL_GPU_DIR="${INSTALL_GPU_DIR:-${PROJECT_ROOT}/3rdparty/onnxruntime-cuda${CUDA_MAJOR_MINOR}-${ORT_VERSION}}"
INSTALL_CPU_DIR="${INSTALL_CPU_DIR:-${PROJECT_ROOT}/3rdparty/onnxruntime}"
CPU_TGZ="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
GPU_TGZ="onnxruntime-linux-x64-gpu_cuda${CUDA_MAJOR_MINOR}-${ORT_VERSION}.tgz"
RELEASE_URL="${RELEASE_URL:-https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}}"
CPU_URL="${CPU_URL:-${RELEASE_URL}/${CPU_TGZ}}"
GPU_URL="${GPU_URL:-${RELEASE_URL}/${GPU_TGZ}}"

cd "${PROJECT_ROOT}"
mkdir -p 3rdparty

download_if_needed() {
    local url="$1"
    local file="$2"
    if [[ ! -s "${file}" ]]; then
        echo "Downloading: ${file}"
        wget -c "${url}" -O "${file}"
    else
        echo "Found local file, skipping download: ${file}"
    fi
}

extract_into() {
    local archive="$1"
    local dest="$2"
    local tmp
    tmp="$(mktemp -d "${PROJECT_ROOT}/.ort-extract.XXXXXX")"
    tar -xzf "${archive}" -C "${tmp}"
    # 官方压缩包包含一个顶层目录，自动识别目录名称。
    local top
    top="$(find "${tmp}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [[ -z "${top}" ]]; then
        echo "Could not find ONNX Runtime directory in archive: ${archive}" >&2
        rm -rf "${tmp}"
        exit 1
    fi
    rm -rf "${dest}"
    mkdir -p "${dest}"
    # 仅移动目录内容，使 CMake 使用的目标路径保持稳定。
    mv "${top}"/* "${dest}/"
    rm -rf "${tmp}"
}

echo "=========================================="
echo "Installing ONNX Runtime ${ORT_VERSION}"
echo "  CPU directory: ${INSTALL_CPU_DIR}"
echo "  GPU directory: ${INSTALL_GPU_DIR}"
echo "=========================================="

download_if_needed "${CPU_URL}" "${CPU_TGZ}"
download_if_needed "${GPU_URL}" "${GPU_TGZ}"

echo "Extracting CPU package..."
extract_into "${CPU_TGZ}" "${INSTALL_CPU_DIR}"
echo "Extracting CUDA GPU package..."
extract_into "${GPU_TGZ}" "${INSTALL_GPU_DIR}"

if [[ ! -f "${INSTALL_CPU_DIR}/include/onnxruntime_cxx_api.h" ]] || \
   [[ ! -f "${INSTALL_CPU_DIR}/lib/libonnxruntime.so" ]]; then
    echo "CPU ONNX Runtime installation is incomplete." >&2
    exit 1
fi
if [[ ! -f "${INSTALL_GPU_DIR}/include/onnxruntime_cxx_api.h" ]] || \
   [[ ! -f "${INSTALL_GPU_DIR}/lib/libonnxruntime.so" ]]; then
    echo "GPU ONNX Runtime installation is incomplete." >&2
    exit 1
fi

# 安装校验成功后，自动清除下载的临时压缩包
echo "Cleaning up temporary download archives..."
rm -f "${CPU_TGZ}" "${GPU_TGZ}"

echo "ONNX Runtime ${ORT_VERSION} installation completed."
echo "  CPU: ${INSTALL_CPU_DIR}"
echo "  CUDA: ${INSTALL_GPU_DIR}"
echo "Pass the following to CMake: -DONNXRUNTIME_ROOT=${INSTALL_CPU_DIR} -DONNXRUNTIME_GPU_ROOT=${INSTALL_GPU_DIR} -DBUILD_GPU=ON"
