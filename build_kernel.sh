#!/usr/bin/env bash
# Build the minimal FalcoCV native kernel (_falco_kernel) on Linux/macOS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build/kernel"
BIN_DIR="$ROOT/cpp/bin"
PYTHON="${PYTHON:-python3}"

cmake -S "$ROOT/cpp/kernel" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$PYTHON"
cmake --build "$BUILD_DIR" -j "${JOBS:-$(nproc 2>/dev/null || echo 4)}"

mkdir -p "$BIN_DIR"
shopt -s nullglob
for f in "$BIN_DIR"/_falco_kernel*.so "$BIN_DIR"/_falco_kernel*.dylib; do
    ext="${f##*.}"
    cp -f "$f" "$BIN_DIR/_falco_kernel.$ext"
    echo "[OK] $BIN_DIR/_falco_kernel.$ext"
    exit 0
done
echo "[ERROR] _falco_kernel shared library not found" >&2
exit 1
