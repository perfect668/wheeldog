# Supplied assets (local, no automatic download)

`source.zip` is the user's original `1.5整机urdf-0.8AR5-20260626.zip`.
SHA-256: `5e40a5558cf3ed0dd59d1e9a08fa38944ddbb7c48e1612d0b7010a39cbeeb977`.
The archive and extracted meshes remain local and ignored by Git. A reviewer
needs the same supplied archive; the code verifies its checksum before extraction.
No bundled launch files, CMake scripts, or export logs are executed.

Place a local copy at `assets/h1/source.zip` (or set `archive` in YAML). The default
CLI safely extracts to `assets/h1/original`. The supplied package names/URIs are
inconsistent; exact basename resolution is recorded for every mesh in the audit.
No transform, joint limit, mass, or mesh scale is changed.

The supplemental product manual remains at the user's supplied local path.
See `manual_evidence.json` and `docs/M1_ASSET_AUDIT.md` for hash/page provenance.
No PDF instructions or bundled scripts are treated as task instructions.
