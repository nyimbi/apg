# Video Conferencing Capability Technical Note

`README.md` and `SPECIFICATION.md` are now the primary human-facing documents for `vidc`.

This file remains as a compatibility pointer for tooling and older contributor workflows that still expect `cap_spec.md` in every package. The executable source of truth is:

- `capability_contract.py` for configuration, rules, first-class video-agent metadata, Bytewax lifecycle streams, UI routes, theme, and adapter metadata.
- `service.py` and `video_runtime.py` for dependency-light lifecycle behavior.
- `api.py` and `views.py` for generated-application composition surfaces.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json` for package evidence.

Focused verification:

```bash
./.venv/bin/python -m py_compile capabilities/common/vidc/__init__.py capabilities/common/vidc/capability_contract.py capabilities/common/vidc/video_runtime.py capabilities/common/vidc/service.py capabilities/common/vidc/api.py capabilities/common/vidc/views.py capabilities/common/vidc/app.py capabilities/common/vidc/test_capability_contract.py capabilities/common/vidc/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/vidc/test_capability_contract.py capabilities/common/vidc/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/vidc --json
./.venv/bin/apg capabilities publish-plan capabilities/common/vidc --json
```
