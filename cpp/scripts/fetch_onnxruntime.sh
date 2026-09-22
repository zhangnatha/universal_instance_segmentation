#!/usr/bin/env bash
set -euo pipefail
ver="${1:-1.20.0}"; sha256="${ORT_SHA256:-}"
[[ "$ver" == "1.20.0" ]] || { echo "This script pins ORT 1.20.0; got $ver" >&2; exit 2; }
root="$(cd "$(dirname "$0")/.." && pwd)"; third="$root/3rdparty"; out="$third/onnxruntime-linux-x64-$ver"; mkdir -p "$third"
if [[ ! -f "$out/include/onnxruntime_cxx_api.h" ]]; then
  archive="$third/ort-$ver.tgz"; url="https://github.com/microsoft/onnxruntime/releases/download/v${ver}/onnxruntime-linux-x64-${ver}.tgz"
  curl --fail --location --retry 3 --proto '=https' --tlsv1.2 "$url" -o "$archive"
  if [[ -n "$sha256" ]] && ! echo "$sha256  $archive" | sha256sum -c -; then echo "ORT archive SHA256 mismatch" >&2; exit 3; fi
  tar -xzf "$archive" -C "$third"; rm -f "$archive"
fi
ln -sfn "onnxruntime-linux-x64-$ver" "$third/onnxruntime"; echo "ORT_ROOT=$third/onnxruntime"
