# COMP Capability Plan

## Goal

Deliver one coherent lifecycle and guardrail packet for `comp`: specification,
contract, runtime, API helpers, view models, tests, generated evidence, review,
and progress log entry. This packet extends the existing compliance lifecycle
with first-class provider-neutral AI agents and Bytewax lifecycle validation.

## Implementation Packets

### 1. Contract

- Expand configuration sections for frameworks, controls, evidence,
  assessments, findings, reporting, exceptions, security, governance,
  observability, adapters, UI, and theme.
- Add deterministic rules for lifecycle and guardrail coverage.
- Declare Bytewax as the batch/event stream adapter.
- Add first-class agent metadata for `codex`, `claude_code`, `opencode`, and
  `pi`.
- Add Bytewax lifecycle stream metadata for bulk compliance mutations.
- Add UI routes for assessments, exceptions, exports, audit, agents, and
  lifecycle batches.

### 2. Runtime

- Key all package records by tenant-qualified internal keys while preserving
  public business IDs.
- Route framework, control, evidence, assessment, finding, report, attestation,
  publish, and resolution operations through `evaluate_capability_rules()`.
- Record hashed audit events for state changes.
- Register scoped compliance agents with owner, purpose, role, runtime,
  contribution disclosure, and privileged-review status.
- Validate Bytewax lifecycle batches and record accepted/denied evidence.
- Preserve deterministic dependency-light behavior.

### 3. API And Views

- Expose finding resolution through `api.py`.
- Expose compliance-agent registration and lifecycle-batch validation through
  `api.py`.
- Add view models for frameworks, controls, evidence, assessments, findings,
  reports, attestations, agents, lifecycle batches, audit, and settings.
- Keep view models data-only and framework-neutral.

### 4. Documentation

- Add `README.md` for practical use.
- Add `SPECIFICATION.md` for capability definition and acceptance criteria.
- Add `PLAN.md` for implementation and review sequencing.
- Replace `cap_spec.md` with a pointer to the active specification and proof
  commands.

### 5. Tests

- Cover contract shape, route count, rule count, theme, adapters, and
  registration metadata.
- Cover full positive compliance lifecycle.
- Cover evidence encryption, immutable reference, stale evidence, DLP linkage,
  report approval, attestation, independent approval, critical finding blocking,
  finding resolution, and tenant-local IDs.
- Rename stale package-test terminology.
- Cover supported/unsupported compliance-agent runtimes and roles.
- Cover privileged agent review state.
- Cover Bytewax-only lifecycle batches, empty batches, and unsupported
  operations.

### 6. Evidence

- Regenerate `semantic_model.json` from `app.semantic_model()`.
- Regenerate `release_report.json` from `app.self_test()` and contract counts.
- Regenerate `package_manifest.json` from package files and runtime metadata.
- Run focused package proof commands.

## Review Checklist

- Contract and runtime rules agree.
- Agent and streaming manifests are exposed by contract, semantic model, and
  registration metadata.
- Tenant-local storage cannot collide on repeated IDs.
- Report publication cannot bypass approval, attestation, or critical-finding
  blocks.
- Evidence guardrails are enforced before assessment.
- Regulated controls cannot bypass DLP linkage.
- API helpers cover service operations.
- View models cover declared route families.
- Docs describe current behavior, provider-neutral agent integration points,
  and Bytewax-only lifecycle handling.
- Stale-marker scan has no primary-slice hits.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/comp/__init__.py capabilities/common/comp/capability_contract.py capabilities/common/comp/compliance_engine.py capabilities/common/comp/models.py capabilities/common/comp/service.py capabilities/common/comp/api.py capabilities/common/comp/views.py capabilities/common/comp/app.py capabilities/common/comp/test_capability_contract.py capabilities/common/comp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/comp/test_capability_contract.py capabilities/common/comp/tests/test_package_contract.py
./.venv/bin/python capabilities/common/comp/app.py
./.venv/bin/apg capabilities inspect comp --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/comp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/comp --json
git diff --check -- capabilities/common/comp docs/progress_log.md
```

## Not In This Packet

- Persistent database adapters.
- Live regulator submissions.
- Live document repository storage.
- Live DLP, audit, workflow, or identity provider integrations.
- Browser-rendered UI validation.
- Bytewax runtime deployment.
- External `codex`, `claude_code`, `opencode`, or `pi` invocation.
- Performance, scale, interoperability, or certification work.
