#!/usr/bin/env bash
# mojo/build.sh — compile all Mojo modules into shared libraries
#
# Usage: ./build.sh [module...]
#   ./build.sh              # build all modules
#   ./build.sh vector_utils # build only vector_utils
#   ./build.sh sprt bm25    # build specific modules
#
# With Mojo 1.0.0b2+, you don't NEED to build .so files at all:
# the mojo.importer Python hook auto-compiles .mojo files on import.
# This build script is for the full Mojo SDK which supports
# SIMD intrinsics and UnsafePointer origin inference.
#
# Output: each module compiles to mojo/<module>.so, discoverable
#         by the Python bridge layers via sys.path manipulation.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------
if ! command -v mojo &>/dev/null; then
	echo "❌ 'mojo' not found on PATH."
	echo ""
	echo "   Install Mojo 1.0.0b2+ via:"
	echo ""
	echo "       uv pip install mojo --prerelease allow"
	echo ""
	echo "   This makes 'mojo' available in your venv at:"
	echo "       .venv/bin/mojo"
	echo ""
	echo "   The pip distribution supports Python interop but lacks the"
	echo "   SIMD stdlib for full acceleration. dspytools falls back to"
	echo "   pure Python automatically when .so files are absent."
	echo ""
	exit 1
fi

MOJO_BIN="$(which mojo)"

# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------
if [ $# -eq 0 ]; then
	MODULES=("vector_utils" "sprt" "bm25")
else
	MODULES=("$@")
fi

echo "🔧 Mojo build — target directory: $DIR"
echo "  Compiler: $MOJO_BIN"
echo ""

for MODULE in "${MODULES[@]}"; do
	SRC="${MODULE}.mojo"
	OUT="${MODULE}.so"

	if [ ! -f "$SRC" ]; then
		echo "  ⚠  Source not found: $SRC — skipping"
		continue
	fi

	echo "  → Compiling $SRC -> $OUT ..."
	mojo build "$SRC" --emit shared-lib -o "$OUT"
	echo "    ✓ $OUT"
done

echo ""
echo "✅ Build complete. Shared libraries in: $DIR"
ls -lh "$DIR"/*.so 2>/dev/null || echo "   (no .so files produced)"
echo ""
echo "💡 Tip: The mojo.importer hook auto-compiles without build.sh."
echo "   Just run: python3 -c \"import mojo.importer; import vector_utils\""
