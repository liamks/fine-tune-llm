#!/usr/bin/env bash
# reset_unsloth_env.sh — clean Unsloth/TRL state and align deps
set -euo pipefail

# --- Config (override via env vars when calling the script) ---
PROJECT_DIR="${PROJECT_DIR:-$PWD}"                 # path to your repo

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"       # your venv path
PYTHON="${PYTHON:-$VENV_PATH/bin/python}"          # python in that venv
PIP="${PIP:-$VENV_PATH/bin/pip}"                   # pip in that venv
# Pin a conservative stack? set PIN_TRL=1 when calling the script.
PIN_TRL="${PIN_TRL:-0}"

# These reduce flakiness between runs
export UNSLOTH_COMPILE_DISABLE="${UNSLOTH_COMPILE_DISABLE:-1}"
export UNSLOTH_DISABLE_AUTO_UPDATES="${UNSLOTH_DISABLE_AUTO_UPDATES:-1}"

echo "==> Project: $PROJECT_DIR"
echo "==> Venv:    $VENV_PATH"
echo

# 0) Sanity checks
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "ERROR: venv not found at $VENV_PATH. Set VENV_PATH or create it." >&2
  exit 1
fi

# 1) Kill stragglers from THIS project only (safer than global pkill)
echo "==> Killing leftover python/accelerate processes for this project (if any)…"
PIDS=$(pgrep -af "(python|accelerate)" | grep -F "$PROJECT_DIR" | awk '{print $1}' || true)
if [[ -n "${PIDS:-}" ]]; then
  echo "$PIDS" | xargs -r kill -TERM || true
  sleep 1
  # hard kill if they’re stubborn
  PIDS=$(pgrep -af "(python|accelerate)" | grep -F "$PROJECT_DIR" | awk '{print $1}' || true)
  [[ -n "${PIDS:-}" ]] && echo "$PIDS" | xargs -r kill -KILL || true
else
  echo "(none)"
fi

# 2) Nuke stale caches that trigger Unsloth RL patcher weirdness
echo "==> Clearing Unsloth/torch caches…"
rm -rf \
  "$PROJECT_DIR/unsloth_compiled_cache" \
  /tmp/unsloth_compiled_cache \
  "$HOME/.cache/unsloth_zoo" \
  "$HOME/.cache/torch_extensions" \
  "$HOME/.cache/torch/inductor"

# 3) Activate venv
echo "==> Activating venv…"
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

# 4) Hard reinstall Unsloth + Zoo (their recommendation)
echo "==> Reinstalling unsloth + unsloth_zoo (force, no-deps, no-cache)…"
uv pip install --upgrade --force-reinstall --no-deps --no-cache-dir unsloth unsloth_zoo

# 5) Align TRL/Transformers/Accelerate/PEFT
if [[ "$PIN_TRL" == "1" ]]; then
  echo "==> Installing pinned compatibility set (safer if you saw RL patch errors)…"
  uv pip install -U \
    "trl==0.14.0" \
    "transformers>=4.49,<4.56" \
    "accelerate>=0.27,<1.0" \
    "peft<0.13"
else
  echo "==> Installing latest TRL/Transformers/Accelerate/PEFT…"
  uv pip install -U trl transformers accelerate peft
fi

# 6) Quick preflight to ensure imports are clean
echo "==> Preflight import check…"
$PYTHON - <<'PY'
import os
import unsloth, trl, transformers
print("unsloth:", getattr(unsloth, "__version__", "unknown"))
print("trl:", trl.__version__, "transformers:", transformers.__version__)
from unsloth import FastLanguageModel
print("Unsloth import OK")
PY

cat <<'MSG'

✅ Reset complete.

Recommended:
- Keep these in your shell profile (~/.bashrc or ~/.zshrc):
    export UNSLOTH_COMPILE_DISABLE=1
    export UNSLOTH_DISABLE_AUTO_UPDATES=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

- Launch training as a module (keeps package context stable):
    accelerate launch -m scripts.train -- --config configs/train.default.yaml

If you still see RL patcher AttributeErrors, rerun this script with PIN_TRL=1.
MSG
