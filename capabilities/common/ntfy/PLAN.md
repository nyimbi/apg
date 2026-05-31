# NTFY Capability Plan

## Goal

Deliver one coherent lifecycle and guardrail packet for `ntfy`: specification,
contract, runtime, API helpers, view models, tests, generated evidence, review,
and progress log entry. This packet extends the existing notification lifecycle
with first-class provider-neutral AI agents and Bytewax lifecycle validation.

## Implementation Packets

### 1. Contract

- Expand configuration sections for channels, delivery, preferences, templates,
  campaigns, security, governance, observability, adapters, UI, and theme.
- Add deterministic rules for lifecycle and guardrail coverage.
- Declare Bytewax as the batch/event stream adapter.
- Add first-class agent metadata for `codex`, `claude_code`, `opencode`, and
  `pi`.
- Add Bytewax lifecycle stream metadata for bulk notification mutations.
- Add UI routes for suppression lists, agents, lifecycle batches, and audit.

### 2. Runtime

- Add `notification_runtime.py` with tenant-scoped records and deterministic
  lifecycle operations.
- Keep live delivery providers behind adapters.
- Route channel, preference, template, message, campaign, and audit decisions
  through `evaluate_capability_rules()`.
- Register scoped notification agents with owner, purpose, role, runtime,
  contribution disclosure, and privileged-review status.
- Validate Bytewax lifecycle batches and record accepted/denied evidence.

### 3. API And Views

- Add `package_api.py` for dependency-light generated-app calls.
- Add `view_models.py` for data-only generated UI payloads.
- Expose notification-agent registration and lifecycle-batch validation
  through `package_api.py`.
- Preserve existing Flask/API and FAB view modules as production integration
  surfaces.

### 4. Documentation

- Add `README.md` for practical use.
- Add `SPECIFICATION.md` for capability definition and acceptance criteria.
- Add `PLAN.md` for implementation and review sequencing.
- Replace `cap_spec.md` with a pointer to the active specification and proof
  commands.

### 5. Tests

- Cover contract shape, route count, rule count, theme, adapters, and
  registration metadata.
- Cover channel, preference, template, message, campaign, and audit lifecycle.
- Cover consent, encryption, provider health, idempotency, webhook signature,
  campaign approval, and large-batch review guardrails.
- Cover supported/unsupported notification-agent runtimes and roles.
- Cover privileged agent review state.
- Cover Bytewax-only lifecycle batches, empty batches, and unsupported
  operations.
- Rename stale package-test terminology.

### 6. Evidence

- Regenerate `semantic_model.json` from `app.semantic_model()`.
- Regenerate `release_report.json` from `app.self_test()` and contract counts.
- Regenerate `package_manifest.json` from package files and runtime metadata.
- Run focused package proof commands.

## Review Checklist

- Contract and runtime rules agree.
- Agent and streaming manifests are exposed by contract, semantic model, and
  registration metadata.
- Generated apps use dependency-light runtime/API/view-model files.
- Live providers remain adapter boundaries.
- Marketing sends cannot bypass opt-in or unsubscribe.
- Sensitive payloads require encryption.
- Campaigns cannot bypass approval.
- Large batches route to review.
- Docs describe current behavior, provider-neutral agent integration points,
  and Bytewax-only lifecycle handling.
- Stale-marker scan has no primary-slice hits.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/ntfy/__init__.py capabilities/common/ntfy/capability_contract.py capabilities/common/ntfy/notification_runtime.py capabilities/common/ntfy/package_api.py capabilities/common/ntfy/view_models.py capabilities/common/ntfy/app.py capabilities/common/ntfy/test_capability_contract.py capabilities/common/ntfy/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/ntfy/test_capability_contract.py capabilities/common/ntfy/tests/test_package_contract.py
./.venv/bin/python capabilities/common/ntfy/app.py
./.venv/bin/apg capabilities inspect ntfy --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/ntfy --json
./.venv/bin/apg capabilities publish-plan capabilities/common/ntfy --json
git diff --check -- capabilities/common/ntfy docs/progress_log.md
```

## Not In This Packet

- Live provider delivery.
- Live personalization model execution.
- Live WebSocket server validation.
- Persistent database adapters.
- Browser-rendered UI validation.
- Bytewax runtime deployment.
- External `codex`, `claude_code`, `opencode`, or `pi` invocation.
- Performance, scale, deliverability, or provider certification work.
