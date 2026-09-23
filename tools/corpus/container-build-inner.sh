#!/usr/bin/env bash
# In-container half of the pinned SCOWL build. Runs with --network none.
# Extracts the verified tarball, builds final/english-words.* exactly as the
# upstream generator does, and copies them beneath /probe (the checkout).

set -euo pipefail

PROBE="/probe"
SOURCES="$PROBE/data/sources/openwordle-v1"
TARBALL="$SOURCES/scowl-v1-rel-2020.12.07-5ef55f9c.tar.gz"
EXTRACT_DIR="$SOURCES/scowl-v1-rel-2020.12.07"
OUT="$SOURCES/scowl-v1-rel-2020.12.07-build"

rm -rf "$OUT"
mkdir -p "$OUT" "$EXTRACT_DIR"

echo "== extracting SCOWL v1 =="
STAGE="/tmp/scowl-extract"
rm -rf "$STAGE"
mkdir -p "$STAGE"
tar -xzf "$TARBALL" -C "$STAGE"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
cp -a "$STAGE"/wordlist-*/. "$EXTRACT_DIR/"
rm -rf "$STAGE"
cd "$EXTRACT_DIR/scowl"

if [ ! -d l ] || [ ! -f l/levels ]; then
    echo "SCOWL tree incomplete after extraction" >&2
    exit 1
fi

echo "== make prep + all (GNU toolchain, no network) =="
cd "$EXTRACT_DIR"
make all
cd scowl

echo "== copying final lists =="
for size in 10 20 35 40 50 55 60; do
    cp "final/english-words.$size" "$OUT/"
done

ls -la "$OUT"
