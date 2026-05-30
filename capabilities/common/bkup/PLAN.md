# BKUP Capability Development Plan

## Current State

BKUP has a dependency-light backup and restore runtime with tenant-qualified
plans, snapshots, restores, restore approvals, retention dispositions,
continuity reports, audit evidence, API helpers, view models, package
evidence, and focused tests.

The current packet closes the remaining composition gap for AI-assisted backup
governance: backup agents must be first-class, governed participants; batch
backup mutation must declare the Bytewax lifecycle stream; generated evidence
must expose practical provides/requires metadata; and the package needs a root
README that explains how builders compose BKUP into executable applications.

## Packet: Backup-Agent And Bytewax Governance

Deliver a focused lifecycle and guardrail packet:

- add a practical root `README.md`;
- keep `SPECIFICATION.md`, `PLAN.md`, and `cap_spec.md` aligned with the
  executable package surface;
- add first-class AI backup-agent configuration for Codex, Claude Code,
  OpenCode, and Pi style runtimes;
- require backup-agent registration, supported runtime, supported role,
  explicit scope, and contribution disclosure;
- add `BackupAgent` model state;
- add service, API-helper, and view-model methods for backup-agent
  registration and listing;
- add Bytewax stream metadata to the contract and generated semantic model;
- require Bytewax for batch backup mutation intent;
- expose backup-agent, audit, analytics, settings, and stream UI state;
- add provides/requires metadata for APG composition;
- refresh generated package evidence from the live contract;
- extend focused tests for the new rules and view/API surfaces;
- record progress evidence and commit the verified slice.

## Implementation Steps

1. Extend `capability_contract.py` with backup-agent, governance,
   observability, adapter, UI, theme, provides/requires, and Bytewax stream
   metadata.
2. Extend the rule engine with backup-agent registration/runtime/role/scope/
   disclosure guardrails and Bytewax batch mutation enforcement.
3. Add `BackupAgent` to `models.py`.
4. Extend `BkupService` with tenant-qualified backup-agent state,
   registration, listing, dashboard counts, token normalization, and batch
   mutation validation.
5. Extend `api.py` with backup-agent and batch mutation helpers.
6. Extend `views.py` with backup-agent, analytics, audit, settings, and stream
   surfaces.
7. Update focused tests to cover contract shape, rule evaluation, service,
   API helpers, view models, generated package evidence, and Bytewax metadata.
8. Replace stale `cap_spec.md` content with a pointer to the active spec.
9. Add `README.md` and update `SPECIFICATION.md`.
10. Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and
    `release_report.json` from the contract.
11. Run focused compile, package tests, self-test, implementation audit,
    publish-plan, stale-marker scan for touched files, and diff checks.

## Review Checklist

- Backup-agent runtime and role values normalize predictable CLI/provider
  names.
- Unsupported agent runtimes fail closed.
- Unsupported agent roles fail closed.
- Missing backup-agent scope fails closed.
- Undisclosed backup-agent contribution fails closed.
- Batch backup mutation fails unless `event_stream` is `bytewax`.
- Tenant-qualified state remains intact for plans, snapshots, restores,
  approvals, dispositions, reports, backup agents, and audit records.
- Production restore still requires approved matching restore approval
  evidence.
- Retention disposition still respects legal hold and independent review.
- Stale restore-test review still fails closed into review-required state.
- API helpers expose the same behavior as service methods.
- View models expose dashboard, plans, snapshots, restores, approvals,
  retention, agents, analytics, settings, stream, theme, and audit state.
- Generated semantic model exposes current route names, provides/requires
  metadata, backup-agent configuration, and Bytewax stream metadata.
- Storage providers, schedulers, orchestration engines, databases, and web
  servers remain adapter boundaries.

## Verification Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/bkup/__init__.py capabilities/common/bkup/capability_contract.py capabilities/common/bkup/models.py capabilities/common/bkup/backup_engine.py capabilities/common/bkup/service.py capabilities/common/bkup/api.py capabilities/common/bkup/views.py capabilities/common/bkup/app.py capabilities/common/bkup/test_capability_contract.py capabilities/common/bkup/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/bkup/test_capability_contract.py capabilities/common/bkup/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.bkup import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bkup --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bkup --json
git diff --check -- capabilities/common/bkup docs/progress_log.md
```
