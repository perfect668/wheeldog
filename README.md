# Recoverability-Aware Whole-Body Assistance

当前交付为 **M1-C**：H1 URDF 内的名义施力预算与有限时间协同扶手路径分析。
结果、假设与证据边界见 [M1-C 报告](docs/M1_CAPABILITY_RESULTS.md) 和
[数学模型](docs/M1_CAPABILITY_MODEL.md)，原始数据位于 `results/m1c/run/`。
M0 与 M1-A/B 原始结果保持冻结。M1-C 完成后等待审阅，未进入 M1-D。

## M0 / EXP0

CPU 上的可复现模型实验：固定 CoP 的二维 LIPM + 有界辅助合力，比较相同 150 N 静态能力下的响应时间常数、力变化率和命令延迟。M0 所有参数为人为 toy model 假设，不使用 H1 模型；没有真实人体实验或新控制算法。

研究约定见 [AGENTS.md](AGENTS.md)、[RESEARCH_BRIEF](docs/RESEARCH_BRIEF.md)；实际方程、单位、采样时序、oracle/因果反馈边界见 [EXP0_MODEL](docs/EXP0_MODEL.md)。正式结果见 [EXP0_RESULTS](docs/EXP0_RESULTS.md)。原仓库交接文件原文保留，整理到了根目录，原交接 README 位于 `docs/ORIGINAL_HANDOFF.md`。

## 安装和运行

Python >=3.10；numpy/scipy/matplotlib/PyYAML，测试用 pytest，不使用 GPU。通用安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH  # 避免本机 ROS 路径注入其他版本的 pytest 插件
python -m pip install -e '.[test]'
python -m pytest
python -m recoverability --config configs/exp0.yaml --output results/reproduction --seed 0
```

优先复用已有科学计算环境，安装本项目时可用 `python -m pip install --no-deps --no-build-isolation -e .`。本机实际使用已有 Python 与本地缓存包进行离线安装，精确版本见 `requirements-exp0.txt` 和结果 metadata；不需要重新下载机器人或仿真资产。已存在的 `.venv` 可直接 `source .venv/bin/activate`。

一条实验命令运行参数扫描并生成 PNG/PDF 图。拒绝覆盖非空目录，复现请使用新 output 路径。CLI 的 `--seed` 被保存；当前实验是确定性的，不用随机噪声。

仅运行 YAML 中的额外单个案例集合（未来速度脉冲和短时外力脉冲）：

```bash
python -m recoverability --config configs/exp0.yaml --output results/single --seed 0 --mode single
```

仅从已有原始数据重画，不重新求解：

```bash
python -m recoverability --output results/exp0 --mode plots
```

步长收敛、二分边界与全扫描审计：

```bash
python scripts/validate_exp0.py --config configs/exp0.yaml --input results/exp0 --output results/validation_reproduction --seed 0
python scripts/validate_exp0.py --output results/validation --plots-only
```

## 输出与解释

- `metrics.csv`：每个扰动/执行器/时域/方法的状态、参数、计时和误差。
- `trajectories.npz`：`cXXXXX_state/force/command/jumps_m_s/disturbance_N`；`rows_json` 与 CSV 同行；`metadata_json` 可独立解释 NPZ。不使用 pickle。
- `metadata.json / config.yaml / summary.json`：配置、seed、git hash、源文件哈希、软件版本、统计。
- `boundaries.csv / figures/`：扫描网格边界、热图、典型轨迹、求解诊断和反馈结果。
- `results/validation/convergence.csv`：dt=0.05/0.02/0.01/0.005 s 的边界二分，区间宽<=0.0005 m/s；`audit.json` 检查对称性与本扫描中的能力/时域一致性。

离线 oracle 的 `INFEASIBLE` 和 `SOLVER_ERROR` 分开；反馈 `NOT_RECOVERED` 不是不可行证明。对于不可行 oracle，不生成虚构状态轨迹。终端箱并非已验证不变集，结果仅说明有限时域的模型内返回。

`src/recoverability/` 包含独立的人体、执行器、扰动、oracle、反馈、指标、实验和绘图模块。`tests/` 用解析式、独立 ODE、镜像、约束篡改和故障注入验证；`configs/` 为全部实验假设。审阅后停在 M0，不自动进入 M1。

## M1-A / M1-B: supplied asset and static kinematics

Work is on `m1/asset-kinematics`, based on the reviewed M0 branch. Scope:
URDF/manual audit, FK/Jacobian finite differences, bounded dual IK, and redundant
static configurations at identical world flange targets. No dynamics/control.
See `docs/M1_ASSET_AUDIT.md`, `docs/M1_KINEMATICS_RESULTS.md`, and
`docs/M0_REVIEW_ADDENDUM.md`. Vendor archive and manual remain local; source
hashes and mesh mapping are saved. Place the supplied archive at
`assets/h1/source.zip` (see `assets/h1/README.md`).

A clean Python 3.11 environment can install the tested dependencies:

```bash
python3.11 -m venv .venv-m1
env -u PYTHONPATH .venv-m1/bin/python -m pip install -r requirements-m1.txt
env -u PYTHONPATH .venv-m1/bin/python -m pip install -e . --no-deps
# Choose a new empty output directory; runs never overwrite saved experiments.
env -u PYTHONPATH .venv-m1/bin/python -m pytest -q
env -u PYTHONPATH .venv-m1/bin/python -m h1_kinematics --config configs/m1_kinematics.yaml --output results/m1_reproduction --seed 20261111
env -u PYTHONPATH .venv-m1/bin/python -m h1_kinematics.plots results/m1_reproduction
.venv/bin/python scripts/verify_frozen_exp0.py
```

On the development laptop `.venv-m1` reuses the installed `rl` environment's
MuJoCo/numerical packages and the installed pure Python trimesh package. Only
missing python-fcl was installed. No GPU or MuJoCo dynamic stepping is used.
`ACCEPTED_CONDITIONAL` means URDF limits, target tolerances and supplied mesh
surface clearance pass under the recorded exclusions; it is not physical robot
feasibility, solid collision certification, force capability, or stability.

## M1-C: nominal force and cooperative motion capability

Branch `m1/capability-reachability` extends the validated M1-B geometry with
arm-budget force LPs, cooperative instantaneous velocity LPs, offline finite-time
path continuation and all-pair analysis. Shared torso loads are summed and
recorded; their unknown limits, gravity and base stability are not modeled.
Motion outputs are sampled, verified local path lower bounds, not global
reachable sets, hardware capability or causal human recovery validation.

Reuse `.venv-m1`, `requirements-m1.txt` and local assets. Run from the repository
root; choose a new output directory for reproduction:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
env -u PYTHONPATH .venv-m1/bin/python -m pytest -q
env -u PYTHONPATH .venv-m1/bin/python -m h1_capability --output results/m1c_reproduction --stage force
env -u PYTHONPATH .venv-m1/bin/python -m h1_capability --output results/m1c_reproduction --stage reach --workers 4
env -u PYTHONPATH .venv-m1/bin/python -m h1_capability.analysis results/m1c_reproduction --no-plots
env -u PYTHONPATH .venv-m1/bin/python -m h1_capability --output results/m1c_reproduction --stage sensitivity --workers 4
env -u PYTHONPATH .venv-m1/bin/python -m h1_capability.analysis results/m1c_reproduction
env -u PYTHONPATH .venv-m1/bin/python -m h1_capability.plots results/m1c_reproduction --postures
env -u PYTHONPATH .venv-m1/bin/python -m h1_capability.verification results/m1c_reproduction --workers 4
env -u PYTHONPATH .venv-m1/bin/python scripts/validate_m1c_numerics.py results/m1c_reproduction
# Verify delivered artifacts, without changing them:
env -u PYTHONPATH .venv-m1/bin/python scripts/freeze_m1c.py results/m1c --verify
```

Main and sensitivity runs resume by case ID. Reusing a directory with a different
configuration raises an error; changed algorithm semantics require a new output.
All 48 initial candidates and 1128 pair filters are retained. The three examples
are selected by aggregate six-direction motion differences after observing data;
all 185 force-matched pairs remain available for review.

## M1-D: coupled lateral human–H1 recovery

Branch `m1/coupled-recovery` reuses M0's LIPM and M1-C's geometry/constraints.
A bilateral spring–damper contact produces actual modeled forces from handle
position/velocity relative to the human. All contact/dynamic parameters remain
explicit assumptions. See `docs/M1_COUPLED_RECOVERY_MODEL.md`,
`docs/M1_COUPLED_RECOVERY_RESULTS.md`, and `docs/M1_COUPLED_TEST_REPORT.md`.
Offline known-disturbance LPs and causal feedback are reported separately.
Restricted path failure is not global unrecoverability.

Reuse the local audited archive, `.venv-m1` and `requirements-m1.txt`; no asset
or GPU simulator download is required. Run from the repository root:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
env -u PYTHONPATH .venv-m1/bin/python -m pytest -q
# Choose a new directory; prepare refuses an existing saved config.
env -u PYTHONPATH .venv-m1/bin/python -m h1_coupled --output results/m1d_reproduction --stage prepare
env -u PYTHONPATH .venv-m1/bin/python -m h1_coupled --output results/m1d_reproduction --stage main --workers 4
env -u PYTHONPATH .venv-m1/bin/python -m h1_coupled --output results/m1d_reproduction --stage sensitivity --workers 4
env -u PYTHONPATH .venv-m1/bin/python -m h1_coupled --output results/m1d_reproduction --stage future --workers 4
env -u PYTHONPATH .venv-m1/bin/python -m h1_coupled --output results/m1d_reproduction --stage boundaries
# Original knot-only optimization and its midpoint geometry correction:
env -u PYTHONPATH .venv-m1/bin/python -m h1_coupled.direct results/m1d_reproduction --workers 2
env -u PYTHONPATH .venv-m1/bin/python -m h1_coupled.direct results/m1d_reproduction --override configs/m1_direct_refined.yaml --workers 2
env -u PYTHONPATH .venv-m1/bin/python -m h1_coupled.passive results/m1d_reproduction
env -u PYTHONPATH .venv-m1/bin/python scripts/probe_m1d_direct.py results/m1d_reproduction
env -u PYTHONPATH .venv-m1/bin/python scripts/validate_m1d.py results/m1d_reproduction
env -u PYTHONPATH .venv-m1/bin/python -m h1_coupled.analysis results/m1d_reproduction
# Verify the delivered data without rerunning experiments:
env -u PYTHONPATH .venv-m1/bin/python scripts/freeze_m1d.py --verify
```

`results/m1d/run/cases/` retains solver/feedback statuses and failure reasons;
`paths/` contains coupled states, actual contact forces, true FK handle traces,
joints/base, constraint usage and independent human ODE trajectories. Direct
optimizer iterations (including rejected ones) are in `direct_paths/`.
`boundaries.csv` contains local success/failure brackets, not global proofs;
`figures/` contains data-generated PNG/PDF plots. M0/M1/M1-C stay frozen.
This milestone stops for review; no subsequent method or hardware phase starts.
