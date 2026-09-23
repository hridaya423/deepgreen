#!/usr/bin/env bash
# Continue the @5022 pipeline once the normal certificate lands.
# Usage: nohup tools/continue-5022.sh > /tmp/certlogs/continue-5022.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

CERT_DIR=artifacts/oracle/openwordle-v1/openwordle-v1.uniform.normal.full-compiled@5022
CERT="$CERT_DIR/certificate.json"
UNIVERSE=benchmarks/openwordle-v1/frozen-universe
CONFIG=configs/train/adaptive-5022.toml
LOG=/tmp/certlogs/continue-5022.log

echo "[wait] $CERT"
while [ ! -s "$CERT" ]; do sleep 60; done
echo "[landed] $(date)"

uv run --no-sync wordle oracle verify --certificate "$CERT" --frozen-root "$UNIVERSE"

SHA=$(shasum -a 256 "$CERT" | awk '{print $1}')
echo "[cert-sha] $SHA"
python3 - "$CONFIG" "$SHA" <<'PY'
import pathlib, sys
path, sha = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
assert "NORMAL_CERT_SHA256_PLACEHOLDER" in text
path.write_text(text.replace("NORMAL_CERT_SHA256_PLACEHOLDER", sha))
print(f"[config] bound normal sha in {path}")
PY

echo "[train] launching adaptive campaign"
uv run --no-sync wordle train adaptive --config "$CONFIG" \
  --runs-root runs --output-release artifacts/releases/champion-5022-certified
echo "[done] $(date)"
