#!/usr/bin/env bash
# Offline data/model tools using the same bundled Python/Torch as native Isaac.
# Does not touch a USD stage, prepare assets or take the simulation runtime lock.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ISAAC_NATIVE_HOME=${ISAAC_NATIVE_HOME:-/root/isaacsim}
TORCH_NATIVE_BUNDLE="$ISAAC_NATIVE_HOME/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"
M2_COMPUTE_LIBS="$TORCH_NATIVE_BUNDLE/torch/lib"
for lib in "$TORCH_NATIVE_BUNDLE"/nvidia/*/lib; do
    [[ ! -d "$lib" ]] || M2_COMPUTE_LIBS+=":$lib"
done
export LD_LIBRARY_PATH="$ISAAC_NATIVE_HOME/kit/python/lib:$ISAAC_NATIVE_HOME/kit:$ISAAC_NATIVE_HOME/kit/lib:$M2_COMPUTE_LIBS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$TORCH_NATIVE_BUNDLE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$ISAAC_NATIVE_HOME/python.sh" "$@"
