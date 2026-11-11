#!/usr/bin/env bash
# Build a FalcoCV AppImage from the PyInstaller onedir bundle.
#
#   packaging/linux/build_appimage.sh [bundle-dir]
#
# Requires appimagetool on PATH (or APPIMAGETOOL=/path/to/appimagetool).
# The PyInstaller bundle must already exist (scripts/build_standalone.py).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUNDLE=${1:-"$ROOT/dist/standalone/FalcoCV"}
VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/src/falcocv/version.py")
APPDIR="$ROOT/build/appimage/FalcoCV.AppDir"
ICON="$ROOT/src/falcocv/ui/assets/favicon.png"
OUT="$ROOT/dist/standalone/FalcoCV-$VERSION-linux-x86_64.AppImage"
TOOL=${APPIMAGETOOL:-appimagetool}

if [ ! -d "$BUNDLE" ]; then
    echo "bundle not found: $BUNDLE" >&2
    exit 1
fi
if ! command -v "$TOOL" >/dev/null 2>&1; then
    echo "appimagetool not found; set APPIMAGETOOL or install it" >&2
    exit 1
fi

rm -rf "$APPDIR"
mkdir -p \
    "$APPDIR/usr/bin" \
    "$APPDIR/usr/share/applications" \
    "$APPDIR/usr/share/icons/hicolor/256x256/apps" \
    "$APPDIR/usr/share/mime/packages"

cp -a "$BUNDLE/." "$APPDIR/usr/bin/"
cp "$ROOT/packaging/linux/AppRun" "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun"
cp "$ROOT/packaging/linux/falcocv.desktop" "$APPDIR/falcocv.desktop"
cp "$ROOT/packaging/linux/falcocv.desktop" \
    "$APPDIR/usr/share/applications/falcocv.desktop"
cp "$ICON" "$APPDIR/falcocv.png"
cp "$ICON" "$APPDIR/usr/share/icons/hicolor/256x256/apps/falcocv.png"
cp "$ROOT/packaging/linux/falcocv-mime.xml" \
    "$APPDIR/usr/share/mime/packages/falcocv-mime.xml"

mkdir -p "$(dirname "$OUT")"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" "$APPDIR" "$OUT"
echo "$OUT"
