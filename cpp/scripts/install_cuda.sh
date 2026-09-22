#!/usr/bin/env bash
# 安装 CUDA 工具包；版本、系统版本和安装目录均可通过环境变量覆盖。
#
# 本脚本只安装工具包，不安装或替换 NVIDIA 内核驱动，也不修改系统 CUDA 软链接。
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-13.0}"
CUDA_PACKAGE_VERSION="${CUDA_PACKAGE_VERSION:-13-0}"
CUDA_TOOLKIT_PACKAGE="${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit-${CUDA_PACKAGE_VERSION}}"
PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${CUDA_KEYRING_DEB}"
CUDA_INSTALL_PATH="/usr/local/cuda-${CUDA_VERSION}"
CUDA_KEYRING_LOCAL_PATH="${PROJECT_ROOT}/${CUDA_KEYRING_DEB}"

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This installer only supports Linux x86_64." >&2
    exit 1
fi

if [[ ! -r /etc/os-release ]]; then
    echo "Unable to identify operating system." >&2
    exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "Current OS is not the targeted Ubuntu release: ${PRETTY_NAME:-unknown}." >&2
    exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "Installing the system toolkit requires sudo." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "Installing CUDA ${CUDA_VERSION} Toolkit"
echo "Install directory: ${CUDA_INSTALL_PATH}"
echo "Driver: Unchanged"
echo "=========================================="

if [[ -x "${CUDA_INSTALL_PATH}/bin/nvcc" ]] &&
   "${CUDA_INSTALL_PATH}/bin/nvcc" --version 2>/dev/null | grep -q "release ${CUDA_VERSION}"; then
    echo "CUDA ${CUDA_VERSION} toolkit already exists, skipping package installation."
else
    if [[ ! -s "${CUDA_KEYRING_LOCAL_PATH}" ]]; then
        echo "Downloading CUDA repository keyring..."
        wget -c "${CUDA_KEYRING_URL}" -O "${CUDA_KEYRING_LOCAL_PATH}"
    else
        echo "Found local repository keyring, skipping download."
    fi

    echo "Registering NVIDIA CUDA repository..."
    sudo dpkg -i "${CUDA_KEYRING_LOCAL_PATH}"

    echo "Updating APT repository metadata..."
    sudo apt-get update

    echo "Installing ${CUDA_TOOLKIT_PACKAGE}..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${CUDA_TOOLKIT_PACKAGE}"
fi

if [[ ! -x "${CUDA_INSTALL_PATH}/bin/nvcc" ]]; then
    echo "CUDA installation completed, but ${CUDA_INSTALL_PATH}/bin/nvcc was not found." >&2
    exit 1
fi

"${CUDA_INSTALL_PATH}/bin/nvcc" --version

# 安装校验成功后，自动清除下载的临时安装包
echo "Cleaning up temporary download files..."
rm -f "${CUDA_KEYRING_LOCAL_PATH}"

echo "=========================================="
echo "CUDA ${CUDA_VERSION} toolkit installation completed."
echo "Existing CUDA system symlinks were not modified."
echo "Run the following in your current shell:"
echo "  export CUDA_HOME=${CUDA_INSTALL_PATH}"
echo "  export PATH=\${CUDA_HOME}/bin:\${PATH}"
echo "  export LD_LIBRARY_PATH=\${CUDA_HOME}/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}"
echo "=========================================="
