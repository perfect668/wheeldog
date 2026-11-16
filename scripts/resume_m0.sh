#!/usr/bin/env bash
# Continue on an adequately provisioned, Docker-capable GPU instance.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LOG="$ROOT/setup_logs/resume_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOG"
exec > >(tee "$LOG/resume.log") 2>&1
printf 'M0 continuation %s\n' "$(date -Is)"
command -v docker
command -v nvidia-smi
gpu=$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader)
[[ -n "$gpu" ]] || { echo 'GPU query returned no data; refusing to report an empty nvidia-smi as success.'; exit 1; }
printf '%s\n' "$gpu"
docker info > "$LOG/docker_info.log"
docker run --rm --gpus all ubuntu:24.04 nvidia-smi 2>&1 | tee "$LOG/gpu_container.log"
/usr/bin/python3 - "$ROOT" <<'PY'
import pathlib, shutil, sys
m = int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:'))) * 1024
limit = pathlib.Path('/sys/fs/cgroup/memory.max')
if limit.exists() and limit.read_text().strip() != 'max':
    m = min(m, int(limit.read_text()))
free = shutil.disk_usage(sys.argv[1]).free
print(f'Effective memory: {m/2**30:.2f} GiB; workspace free: {free/2**30:.2f} GiB')
if m < 32*2**30 or free < 100*2**30:
    raise SystemExit('Provision at least 32 GiB RAM and 100 GiB free SSD space before building.')
PY
cd "$ROOT/upstream"
printf 'Upstream commit: '; git rev-parse HEAD
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Upstream tracked source is modified; inspect before continuing.'; exit 1; }
for stage in doctor build prepare smoke; do
    printf 'Running %s\n' "$stage"
    set +e
    ./run.sh "$stage" 2>&1 | tee "$LOG/$stage.log"
    rc=${PIPESTATUS[0]}
    set -e
    printf 'EXIT_CODE=%s\n' "$rc" | tee -a "$LOG/$stage.log"
    [[ "$rc" == 0 ]] || exit "$rc"
done
grep -q '^PHYRC-SMOKE-PASS ' "$LOG/smoke.log"
docker image inspect phyrc-2027:isaac-6.0.1 > "$LOG/image_inspect.json"
printf 'Upstream smoke passed; logs: %s\n' "$LOG"
printf 'Images/video and simulation resource measurements still need a separate real capture run.\n'
