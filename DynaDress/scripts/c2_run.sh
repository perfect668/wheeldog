#!/usr/bin/env bash
# Run from the native server project. One Kit process only; failures are retained.
set -euo pipefail
C2_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$C2_ROOT"
mkdir -p setup_logs/c2
./scripts/m2_model_python.sh scripts/c2_replay.py --calibrate-only --calibrate > setup_logs/c2/calibration.log 2>&1
./scripts/m2_model_python.sh scripts/c2_replay.py --calibrate-only --calibration-steps 20 --calibrate > setup_logs/c2/probe_calibration.log 2>&1
./scripts/m2_model_python.sh scripts/c2_replay.py --fixed-mu .3 --split train --limit 2 --output artifacts/c2/replay_fixed_mu > setup_logs/c2/replay_fixed_mu.log 2>&1
./scripts/m2_model_python.sh scripts/c2_replay.py > setup_logs/c2/replay.log 2>&1
./scripts/m2_model_python.sh -m unittest discover -s tests -v > setup_logs/c2/tests.log 2>&1
./scripts/run_native.sh python "$C2_ROOT/scripts/m2_collect_episode.py" --batch-plan "$C2_ROOT/dynadress/configs/c2_smoke_final/plan.json" > setup_logs/c2/smoke_final.log 2>&1
./scripts/m2_model_python.sh -c 'import json,pathlib; p=pathlib.Path("artifacts/c2/smoke_final"); x=[json.loads(f.read_text()) for f in p.glob("*/probe_selection.json")]; assert len(x)==3 and {v["mode"] for v in x}=={"passive","fixed","active"}; assert all(v["stage"] in ("insert","advance") for v in x); print("C2_THREE_MODE_SMOKE_PASS")' > setup_logs/c2/smoke_receipt.log 2>&1
./scripts/run_native.sh python "$C2_ROOT/scripts/m2_collect_episode.py" --batch-plan "$C2_ROOT/dynadress/configs/c2_paired/plan.json" > setup_logs/c2/paired.log 2>&1
./scripts/m2_model_python.sh scripts/c2_evaluate.py > setup_logs/c2/evaluation.log 2>&1
./scripts/m2_model_python.sh scripts/c2_diagnose_mu.py > setup_logs/c2/mu_diagnosis.log 2>&1
./scripts/m2_model_python.sh scripts/c2_plots.py > setup_logs/c2/plots.log 2>&1
./scripts/m2_model_python.sh scripts/c2_video.py --episode artifacts/c2/paired/c2_paired_p00_active --output artifacts/c2/video > setup_logs/c2/video.log 2>&1
./scripts/m2_model_python.sh scripts/c2_export_review.py > setup_logs/c2/export.log 2>&1
