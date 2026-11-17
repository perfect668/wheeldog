# M1.1 lightweight review evidence

The current controller produced five successful sleeve episodes and one genuine
initial-alignment failure. See `docs/M11_REPORT.md` and `analysis/analysis.json`.

The Git review package contains code, reports, actual rendered images, response
plots, compact numeric curves, result/config/validation files for all six runs,
and full synchronized JSONL plus final FEM nodes for the nominal and failure
example. `scripts/m11_validate.py` can audit these two complete examples offline.
Batch raw JSONL and video remain on the simulation server; their SHA256 hashes are
in `docs/M11_SOURCE_MANIFEST.json`.

No robot meshes, Isaac binaries, downloaded PhyRC assets, full RGB frame sequences
or MP4 files are added to Git. All six complete raw records and videos are kept at:

`/root/gpufree-data/dynadress/artifacts/m11/<episode-id>/`

Local copies of the six continuous MP4 files and synchronized records are under:

`/home/user/桌面/DynaDress/artifacts/m11/<episode-id>/`

The archived failed trial `20261117T144124_trajectory01` used the earlier alignment
controller. It stays on the server, with its failed video, and is excluded from
the six current-controller examples.
