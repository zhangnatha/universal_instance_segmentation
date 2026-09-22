#!/usr/bin/env bash
# 一键打包 C++ 推理可执行程序部署包（Linux x86_64）
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Starting C++ inference executable packaging..."
bash "${ROOT_DIR}/cpp/scripts/package_linux.sh" "$@"
