#!/usr/bin/env bash
# Small, deterministic diagnostic batch. These are NOT sleeve-success episodes.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/setup_logs/m1"
for stiffness in 4 12; do
    for condition in pulse free contact collision-disabled; do
        extra=()
        case "$condition" in
            free) extra=(--no-pulse);;
            contact) extra=(--contact);;
            collision-disabled) extra=(--contact --disable-human-collision);;
        esac
        "$ROOT/scripts/run_native.sh" python "$ROOT/scripts/m1_human_test.py" --stiffness "$stiffness" "${extra[@]}" > "$ROOT/setup_logs/m1/batch_k${stiffness}_${condition}.log" 2>&1
    done
done
