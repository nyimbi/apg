# DVRL Works Plan

## Current State

The parent DVRL package has the executable lifecycle packet, contract, service,
view models, app entrypoint, semantic metadata, manifest, release report, and
focused tests. This folder contains supporting work reports and one local
`cap_spec.md`, which made automated scans treat it as an undocumented package.

## Plan

1. Add local documentation that identifies this folder as a working-artifact
   packet.
2. Replace the local `cap_spec.md` with a source-of-truth pointer to the parent
   DVRL capability.
3. Preserve existing reports as evidence without promoting them to runtime
   capability status.
4. Verify that package-gap scans no longer report this folder as missing
   README, specification, or plan docs.
5. Commit and push the documentation cleanup as its own coherent slice.

## Verification

```bash
python - <<'PY'
from pathlib import Path
missing = []
for d in sorted(Path("capabilities").rglob("*")):
    if not d.is_dir():
        continue
    if any(part.startswith(".") or part in {"__pycache__", "tests", "docs"} for part in d.parts):
        continue
    files = {p.name for p in d.iterdir() if p.is_file()}
    has_cap = any(name in files for name in {"capability_contract.py", "cap_spec.md", "package_manifest.json", "semantic_model.json"})
    if not has_cap:
        continue
    needed = [name for name in ["README.md", "SPECIFICATION.md", "PLAN.md"] if name not in files]
    if needed:
        missing.append((str(d), ",".join(needed)))
print("count", len(missing))
PY

git diff --check -- capabilities/common/dvrl/works docs/progress_log.md
```
