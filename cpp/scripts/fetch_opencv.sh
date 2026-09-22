#!/usr/bin/env bash
set -euo pipefail
# OpenCV 4.5.5 是兼容 Ubuntu 18.04 标准环境的稳健基准版本。
ver="${1:-4.5.5}"; [[ "$ver" == "4.5.5" ]] || { echo "This script pins OpenCV 4.5.5" >&2; exit 2; }
root="$(cd "$(dirname "$0")/.." && pwd)"; third="$root/3rdparty"; src="$third/opencv-$ver"; prefix="$third/opencv-install-$ver"
mkdir -p "$third"
if [[ ! -d "$src" ]]; then
  archive="$third/opencv-$ver.tar.gz"; url="https://github.com/opencv/opencv/archive/refs/tags/$ver.tar.gz"
  curl --fail --location --retry 3 --proto '=https' --tlsv1.2 "$url" -o "$archive"
  tar -xzf "$archive" -C "$third"; rm -f "$archive"
fi
if [[ ! -f "$prefix/lib/cmake/opencv4/OpenCVConfig.cmake" ]]; then
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" -DBUILD_LIST=core,imgproc,imgcodecs -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_OPENGL=OFF -DWITH_FFMPEG=OFF -DWITH_V4L=OFF -DWITH_CUDA=OFF -DWITH_IPP=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DBUILD_SHARED_LIBS=ON
  cmake --build "$src/build" --parallel "${JOBS:-2}"; cmake --install "$src/build"
fi
echo "OpenCV_DIR=$prefix/lib/cmake/opencv4"
