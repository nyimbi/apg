# Help and Knowledge Base Capability Technical Note

`README.md` and `SPECIFICATION.md` are now the primary human-facing documents for `help`.

This file remains as a compatibility pointer for tooling and older contributor workflows that still expect `cap_spec.md` in every package. The executable source of truth is:

- `capability_contract.py` for configuration, rules, first-class help-agent metadata, Bytewax lifecycle streams, UI routes, theme, and adapter metadata.
- `service.py`, `models.py`, and `help_runtime.py` for dependency-light lifecycle behavior.
- `api.py` and `views.py` for generated-application composition surfaces.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json` for package evidence.

Focused verification:

```bash
./.venv/bin/python -m py_compile capabilities/common/help/__init__.py capabilities/common/help/capability_contract.py capabilities/common/help/models.py capabilities/common/help/help_runtime.py capabilities/common/help/service.py capabilities/common/help/api.py capabilities/common/help/views.py capabilities/common/help/app.py capabilities/common/help/test_capability_contract.py capabilities/common/help/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/help/test_capability_contract.py capabilities/common/help/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/help --json
./.venv/bin/apg capabilities publish-plan capabilities/common/help --json
```
