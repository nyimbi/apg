# ZTNA Capability Plan

## Goal

Deliver one coherent lifecycle and guardrail packet for `ztna`: specification,
contract, runtime, API helpers, view models, tests, generated evidence, review,
and progress log entry.

## Implementation Packets

### 1. Contract

- Expand configuration sections for identity, device, resource, access,
  session, segmentation, review, security, governance, observability, adapters,
  UI, and theme.
- Keep deterministic rules terse but explicit.
- Add routes for identity console, review queue, and audit timeline.
- Declare Bytewax as the batch/event stream adapter.
- Keep adapters pluggable and side-effect-free.

### 2. Runtime

- Enforce tenant context at registration and access time.
- Key identity, device, resource, request, session, and audit records by tenant.
- Evaluate rules through `evaluate_capability_rules()`.
- Deny hard failures and route review-required states without silently
  approving.
- Track audit events for state changes.
- Preserve compatibility helpers by mapping generic records to protected
  resources.

### 3. API And Views

- Pass through request controls for MFA, review evidence, JIT approval,
  least-privilege scope, explicit decision evidence, and risk score.
- Add data-only view models for identity console, review queue, and audit.
- Keep UI models independent of web frameworks and databases.

### 4. Documentation

- Add `README.md` for practical use.
- Add `SPECIFICATION.md` for capability definition and acceptance criteria.
- Add `PLAN.md` for implementation and review sequencing.
- Replace `cap_spec.md` with a pointer to the active specification and runtime
  proof commands.

### 5. Tests

- Cover contract shape, route count, rule count, theme, adapters, and
  registration metadata.
- Cover standard access lifecycle.
- Cover privileged MFA and independent review.
- Cover high-risk review before session start.
- Cover request-specific guardrails and duplicate pending review blocking.
- Cover tenant-local IDs and cross-tenant denial.
- Cover API helpers and view models.

### 6. Evidence

- Regenerate `semantic_model.json` from `app.semantic_model()`.
- Regenerate `release_report.json` from `app.self_test()` and contract counts.
- Regenerate `package_manifest.json` from package files and runtime metadata.
- Run focused package proof commands.

## Review Checklist

- Contract and runtime rules agree.
- No stale embedded rule/route evidence remains.
- Privileged access cannot bypass review without JIT/review evidence.
- Cross-tenant records cannot be mixed.
- Request-specific rules execute because runtime sets `operation`.
- API helpers expose all service controls needed by generated apps.
- View models cover every declared route family.
- Docs describe current behavior, not planned provider integrations.
- Stale-marker scan has no primary-slice hits.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/ztna/__init__.py capabilities/common/ztna/capability_contract.py capabilities/common/ztna/zero_trust_runtime.py capabilities/common/ztna/models.py capabilities/common/ztna/service.py capabilities/common/ztna/api.py capabilities/common/ztna/views.py capabilities/common/ztna/app.py capabilities/common/ztna/test_capability_contract.py capabilities/common/ztna/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/ztna/test_capability_contract.py capabilities/common/ztna/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.ztna import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/ztna --json
./.venv/bin/apg capabilities publish-plan capabilities/common/ztna --json
git diff --check -- capabilities/common/ztna docs/progress_log.md
```

## Not In This Packet

- Persistent database adapters.
- Live identity provider integrations.
- Live endpoint posture providers.
- Live gateway, service mesh, or packet filtering enforcement.
- Browser-rendered UI validation.
- Bytewax runtime deployment.
- Performance, scale, interoperability, or security certification.
