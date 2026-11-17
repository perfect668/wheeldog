#!/usr/bin/env bash
# Native adapter for the unchanged upstream PhyRC sources and pinned assets.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM="$ROOT/upstream"
ISAAC_NATIVE_HOME=${ISAAC_NATIVE_HOME:-/root/isaacsim}
MODE=${1:-help}
[[ $# == 0 ]] || shift
# Prefer the matching bundled Python and Torch/CUDA libraries over system copies.
TORCH_NATIVE_BUNDLE="$ISAAC_NATIVE_HOME/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"
NATIVE_COMPUTE_LIBS="$TORCH_NATIVE_BUNDLE/torch/lib"
for lib in "$TORCH_NATIVE_BUNDLE"/nvidia/*/lib; do
    [[ ! -d "$lib" ]] || NATIVE_COMPUTE_LIBS+=":$lib"
done
export LD_LIBRARY_PATH="$ISAAC_NATIVE_HOME/kit/python/lib:$ISAAC_NATIVE_HOME/kit:$ISAAC_NATIVE_HOME/kit/lib:$NATIVE_COMPUTE_LIBS:${LD_LIBRARY_PATH:-}"
export ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
export OMNICLIENT_HUB_MODE=${OMNICLIENT_HUB_MODE:-disabled}
export PYTHONPATH="$UPSTREAM/sitecustomize:$ROOT/native/python:$UPSTREAM/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PHYRC_SOURCE_COMMIT=$(git -C "$UPSTREAM" rev-parse HEAD)
export PHYRC_SOURCE_DIRTY=0
if [[ -n "$(git -C "$UPSTREAM" status --porcelain --untracked-files=no)" ]]; then export PHYRC_SOURCE_DIRTY=1; fi
export MPLCONFIGDIR="$UPSTREAM/cache/mpl"
mkdir -p "$UPSTREAM/cache/mpl" "$UPSTREAM/output" "$UPSTREAM/.runtime/PhyRC_Sim" "$UPSTREAM/.runtime/RCareWorld-2.0"
link_path() {
    local target=$1 path=$2
    if [[ -e "$path" || -L "$path" ]]; then
        [[ "$(readlink -f "$path")" == "$(readlink -f "$target")" ]] || { echo "Refusing to replace existing path: $path" >&2; exit 1; }
    else
        ln -s "$target" "$path"
    fi
}
case "$MODE" in
    help) echo 'Usage: scripts/run_native.sh {prepare|smoke|gui|python SCRIPT [ARGS...]|probe}'; exit 0;;
    prepare|smoke|gui|python|probe) ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2;;
esac
# Upstream scripts have container-absolute paths. Symlinks preserve them without edits.
link_path "$ISAAC_NATIVE_HOME" /isaac-sim
link_path "$UPSTREAM/.runtime" /workspace
link_path "$UPSTREAM" /project
link_path "$UPSTREAM/scripts" /scripts
link_path "$UPSTREAM/output" /output
exec 9>"$UPSTREAM/cache/runtime.lock"
# M2 workers only read the prepared runtime; prepare and all legacy runs stay exclusive.
if [[ "${DYNADRESS_READ_ONLY_RUNTIME:-0}" == 1 ]]; then
    [[ "$MODE" == python ]] || { echo 'Shared runtime lock is only allowed for prepared Python runs.' >&2; exit 2; }
    flock -s -n 9 || { echo 'Exclusive runtime operation is active.' >&2; exit 2; }
else
flock -n 9 || { echo 'Another prepare/run holds the upstream runtime lock.' >&2; exit 2; }
fi
if [[ "$MODE" == prepare ]]; then
    /usr/bin/python3 "$UPSTREAM/scripts/fetch_assets.py" --check
    cp -a "$UPSTREAM/src/PhyRC_Sim/." "$UPSTREAM/.runtime/PhyRC_Sim/"
    mkdir -p "$UPSTREAM/.runtime/RCareWorld-2.0/assets/robots" "$UPSTREAM/.runtime/PhyRC_Sim/Assets/Human/Mesh/manikin_exports" "$UPSTREAM/.runtime/PhyRC_Sim/Assets/Material"
    cp -a "$UPSTREAM/assets/custom/robots/." "$UPSTREAM/.runtime/RCareWorld-2.0/assets/robots/"
    cp -a "$UPSTREAM/assets/custom/human/." "$UPSTREAM/.runtime/PhyRC_Sim/Assets/Human/Mesh/manikin_exports/"
    cp -a "$UPSTREAM/assets/downloads/Material/." "$UPSTREAM/.runtime/PhyRC_Sim/Assets/Material/"
    install -D -m 664 "$UPSTREAM/assets/downloads/2K-tiling_30_basecolor.jpg" "$UPSTREAM/.runtime/PhyRC_Sim/Assets/Scene/kitchen/kitchen_7/textures/2K-tiling_30_basecolor.jpg"
    usd=("$ISAAC_NATIVE_HOME"/extscache/omni.usd.libs-*/)
    export PYTHONPATH="${usd[0]}:$PYTHONPATH"
    export LD_LIBRARY_PATH="${usd[0]}/bin:$ISAAC_NATIVE_HOME/kit/lib:${LD_LIBRARY_PATH:-}"
    cd "$UPSTREAM/.runtime/PhyRC_Sim"
    exec "$ISAAC_NATIVE_HOME/python.sh" /scripts/prepare_assets.py
fi
cd "$UPSTREAM/.runtime/PhyRC_Sim"
case "$MODE" in
    probe) exec "$ISAAC_NATIVE_HOME/python.sh" "$ROOT/scripts/native_probe.py" --allow-root;;
    smoke) export STRETCH4_HEADLESS=1; exec "$ISAAC_NATIVE_HOME/python.sh" /scripts/smoke.py --allow-root;;
    gui) export STRETCH4_HEADLESS=0; export DISPLAY=${DISPLAY:-:20}; exec "$ISAAC_NATIVE_HOME/python.sh" Env_StandAlone/Teleop_TShirt_Stretch4_Env.py --allow-root;;
    python) export STRETCH4_HEADLESS=1; exec "$ISAAC_NATIVE_HOME/python.sh" "$@" --allow-root;;
esac
