# Zero Trust Network Access Capability Spec Pointer

The active specification is `SPECIFICATION.md`.

Use this file as the stable compatibility pointer for older capability tooling
that still expects `cap_spec.md`.

## Current Packet

`ztna` now owns a coherent zero-trust lifecycle packet:

- tenant-scoped identities;
- verified, privileged, suspended, and MFA-complete identity state;
- device posture, trust, compliance, management, attestation, and quarantine
  state;
- protected resources with policies, access levels, sensitivity, and network
  segments;
- access requests with deterministic allow, deny, and review decisions;
- independent access review;
- governed session start, reevaluation, revocation, and closure;
- first-class provider-neutral zero-trust agents;
- Bytewax lifecycle-batch validation for zero-trust mutations;
- append-only audit events;
- UI route, view-model, theme, and adapter metadata;
- Bytewax as the required event-stream adapter for batch zero-trust mutation
  and lifecycle-batch validation.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/ztna/__init__.py capabilities/common/ztna/capability_contract.py capabilities/common/ztna/zero_trust_runtime.py capabilities/common/ztna/models.py capabilities/common/ztna/service.py capabilities/common/ztna/api.py capabilities/common/ztna/views.py capabilities/common/ztna/app.py capabilities/common/ztna/test_capability_contract.py capabilities/common/ztna/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/ztna/test_capability_contract.py capabilities/common/ztna/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.ztna import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/ztna --json
./.venv/bin/apg capabilities publish-plan capabilities/common/ztna --json
```
