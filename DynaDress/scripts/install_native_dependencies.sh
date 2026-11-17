#!/usr/bin/env bash
# Reproduce the pinned project dependencies without modifying Isaac's Python installation.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ISAAC_NATIVE_HOME=${ISAAC_NATIVE_HOME:-/root/isaacsim}
[[ -x "$ISAAC_NATIVE_HOME/python.sh" ]]
grep -q '^6\.0\.1' "$ISAAC_NATIVE_HOME/VERSION" || { echo 'Expected preinstalled Isaac Sim 6.0.1'; exit 1; }
mkdir -p "$ROOT/native/python"
export LD_LIBRARY_PATH="$ISAAC_NATIVE_HOME/kit/python/lib:$ISAAC_NATIVE_HOME/kit:${LD_LIBRARY_PATH:-}"
# Isaac supplies NumPy, Torch, CUDA libraries and common transitive packages.
# Do not let pip upgrade those bundled components.
exec "$ISAAC_NATIVE_HOME/python.sh" -m pip install --target "$ROOT/native/python" --upgrade --no-deps -r "$ROOT/upstream/requirements-runtime.txt"
