# Actual run commands and source commits

CPU only; reused local `.venv-m1` and audited assets. From repository root,
all Python commands used `env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-m1/bin/python`.

- `-m h1_selection.experiment --stage prepare --output results/m2a` (a828c5a).
- `-m h1_selection.experiment --stage score --output results/m2a --workers 4` (a828c5a).
- `-m h1_selection.experiment --stage select --output results/m2a` (5716852).
- `-m h1_selection.experiment --stage test --output results/m2a --workers 4` (5716852).
- `-m h1_selection.experiment --stage analysis --output results/m2a` (final analysis: 962f296; post-hoc diagnostic additions, no policy/model retuning).
- `-m h1_selection.audit --output results/m2a` (final audit: c367f28; adds successful full observation windows).
- `-m pytest -q` (9e882e6; 81 passed in 25.40 s; later changes affect plots/documentation/audit selection only).

Each calculation stage has its source SHA and start UTC in the corresponding
provenance JSON. The holdout selections and all simulation/controller parameters
were fixed before the first test result. Main data are not rewritten by analysis.

- `-m h1_selection.repeatability --output results/m2a --workers 4` (f361810; 36 unchanged-parameter diagnostic reruns, no primary data replacement).
