#!/usr/bin/env bash
# Pinned Linux container build of the OpenWordle-v1 SCOWL lists.
#
# Phase A (network on): pull the pinned debian base image.
# Phase B (network off): extract the hash-verified SCOWL tarball, build the
# wordlists with the upstream generator, and copy final/english-words.* out to
# data/sources/openwordle-v1/scowl-v1-rel-2020.12.07/ beneath the checkout.
#
# Inputs and outputs must both live beneath the repository root. Native macOS
# builds are not a sanctioned path: BSD grep silently corrupts SCOWL's variant
# tables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DOCKER="${DOCKER:-docker}"
if ! command -v "$DOCKER" >/dev/null 2>&1; then
    for candidate in /Applications/Docker.app/Contents/Resources/bin/docker ~/.docker/bin/docker; do
        if [ -x "$candidate" ]; then DOCKER="$candidate"; break; fi
    done
fi

IMAGE="openwordle-scowl-build:pin1"
BASE_IMAGE="debian:bookworm-slim"
BASE_DIGEST="sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241"
SOURCES="$REPO_ROOT/data/sources/openwordle-v1"
TARBALL="$SOURCES/scowl-v1-rel-2020.12.07-5ef55f9c.tar.gz"
EXPECTED_SHA="933cd8edab9297fc8cb6358d7cb3d05a50371107d1431af61de09eaa875043ac"
EXTRACT_DIR="$SOURCES/scowl-v1-rel-2020.12.07"
OUTPUT_DIR_NAME="${1:-}"

usage() {
    echo "usage: tools/corpus/build-linux.sh <output-dir-under-checkout>" >&2
    exit 2
}

[ -n "$OUTPUT_DIR_NAME" ] || usage
case "$OUTPUT_DIR_NAME" in
    /*|..*) echo "refusing output outside the checkout: $OUTPUT_DIR_NAME" >&2; exit 2 ;;
esac
OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR_NAME"
mkdir -p "$OUTPUT_DIR"

for path in "$TARBALL" "$SOURCES/WordNet-3.0.tar.gz"; do
    [ -f "$path" ] || { echo "missing source archive: $path (run tools/corpus/fetch.py)" >&2; exit 1; }
done

ACTUAL_SHA="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
[ "$ACTUAL_SHA" = "$EXPECTED_SHA" ] || {
    echo "SCOWL tarball digest mismatch: $ACTUAL_SHA" >&2
    exit 1
}

echo "== phase A: image pin check (network only if missing) =="
BASE_OK="$("$DOCKER" image inspect "$BASE_IMAGE@$BASE_DIGEST" --format ok 2>/dev/null || true)"
BUILDER_OK="$("$DOCKER" image inspect "$IMAGE" --format ok 2>/dev/null || true)"
if [ "$BASE_OK" = "ok" ] && [ "$BUILDER_OK" = "ok" ]; then
    echo "pinned builder image already present: $IMAGE"
elif [ "$BASE_OK" = "ok" ]; then
    echo "base present; rebuilding builder image $IMAGE"
    "$DOCKER" build -f "$SCRIPT_DIR/Containerfile" -t "$IMAGE" "$SCRIPT_DIR"
else
    echo "pulling base $BASE_IMAGE@$BASE_DIGEST and building $IMAGE"
    "$DOCKER" pull "$BASE_IMAGE@$BASE_DIGEST"
    "$DOCKER" build -f "$SCRIPT_DIR/Containerfile" -t "$IMAGE" "$SCRIPT_DIR"
fi

echo "== verify the builder sits on the pinned base digest =="
"$DOCKER" image inspect "$IMAGE" --format 'builder layers: {{len .RootFS.Layers}}'

echo "== phase B: build (network off) =="
BUILD_LOG="$OUTPUT_DIR/build.log"
{
    "$DOCKER" run \
        --rm \
        --network none \
        --volume "$REPO_ROOT:/probe:rw" \
        --workdir /probe \
        "$IMAGE" \
        "/probe/tools/corpus/container-build-inner.sh"
} 2>&1 | tee "$BUILD_LOG"

echo "container build complete; outputs under $OUTPUT_DIR"
