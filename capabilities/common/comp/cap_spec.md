# Compliance Management Capability Spec Pointer

The active specification is `SPECIFICATION.md`.

Use this file as the stable compatibility pointer for older capability tooling
that still expects `cap_spec.md`.

## Current Packet

`comp` now owns a coherent compliance lifecycle packet:

- tenant-scoped frameworks, obligations, and policy versions;
- tenant-scoped controls with DLP linkage and testing cadence;
- encrypted immutable evidence records;
- control assessments and evidence freshness checks;
- findings, escalation, remediation, and resolution evidence;
- report preparation, independent approval, attestation, publication, and
  critical-finding blocking;
- first-class provider-neutral compliance agents for `codex`, `claude_code`,
  `opencode`, and `pi`;
- Bytewax lifecycle-batch validation for compliance mutations;
- hashed audit-event metadata;
- UI route, view-model, theme, and adapter metadata;
- Bytewax as the required event-stream adapter for batch compliance mutation.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/comp/__init__.py capabilities/common/comp/capability_contract.py capabilities/common/comp/compliance_engine.py capabilities/common/comp/models.py capabilities/common/comp/service.py capabilities/common/comp/api.py capabilities/common/comp/views.py capabilities/common/comp/app.py capabilities/common/comp/test_capability_contract.py capabilities/common/comp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/comp/test_capability_contract.py capabilities/common/comp/tests/test_package_contract.py
./.venv/bin/python capabilities/common/comp/app.py
./.venv/bin/apg capabilities inspect comp --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/comp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/comp --json
```
