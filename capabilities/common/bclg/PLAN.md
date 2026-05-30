# BCLG Capability Development Plan

## Current State

BCLG has a dependency-light ledger runtime with ledger registration,
key-custody binding, deterministic transaction/block hashes, high-value
transaction review state, smart contract deployment approval, contract/rule/
theme metadata, API helpers, view models, package evidence, and focused tests.

The current packet closes the remaining composition gap for AI-assisted ledger
governance: ledger agents must be first-class, governed participants; batch
ledger mutation must declare the Bytewax lifecycle stream; generated evidence
must expose practical provides/requires metadata; and the package needs a root
README that explains how builders compose BCLG into executable applications.

## Packet: Ledger-Agent And Bytewax Governance

Deliver a focused lifecycle and guardrail packet:

- add a practical root `README.md`;
- keep `SPECIFICATION.md`, `PLAN.md`, and `cap_spec.md` aligned with the
  executable package surface;
- add first-class AI ledger-agent configuration for Codex, Claude Code,
  OpenCode, and Pi style runtimes;
- require ledger-agent registration, supported runtime, supported role,
  explicit scope, and contribution disclosure;
- add `LedgerAgent` model state;
- add service, API-helper, and view-model methods for ledger-agent
  registration and listing;
- add Bytewax stream metadata to the contract and generated semantic model;
- require Bytewax for batch ledger mutation intent;
- expose ledger-agent, audit, analytics, settings, and stream UI state;
- add provides/requires metadata for APG composition;
- refresh generated package evidence from the live contract;
- extend focused tests for the new rules and view/API surfaces;
- record progress evidence and commit the verified slice.

## Implementation Steps

1. Extend `capability_contract.py` with ledger-agent, governance,
   observability, adapter, UI, theme, provides/requires, and Bytewax stream
   metadata.
2. Extend the rule engine with ledger-agent registration/runtime/role/scope/
   disclosure guardrails and Bytewax batch mutation enforcement.
3. Add `LedgerAgent` to `models.py`.
4. Extend `BclgService` with tenant-qualified ledger-agent state,
   registration, listing, dashboard counts, token normalization, and batch
   mutation validation.
5. Extend `api.py` with ledger-agent and batch mutation helpers.
6. Extend `views.py` with ledger-agent, analytics, audit, settings, and stream
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

- Ledger-agent runtime and role values normalize predictable CLI/provider
  names.
- Unsupported agent runtimes fail closed.
- Unsupported agent roles fail closed.
- Missing ledger-agent scope fails closed.
- Undisclosed ledger-agent contribution fails closed.
- Batch ledger mutation fails unless `event_stream` is `bytewax`.
- Tenant-qualified state remains intact for ledger, custody, transaction,
  approval, contract, ledger-agent, head, and audit records.
- High-value transaction commit still requires approved matching review
  evidence.
- Contract deployment still requires approved matching deployment evidence.
- Reviewers still cannot approve their own transaction or deployment requests.
- API helpers expose the same behavior as service methods.
- View models expose dashboard, ledger, custody, transaction, approval,
  contract, agent, analytics, settings, stream, theme, and audit state.
- Generated semantic model exposes current route names, provides/requires
  metadata, ledger-agent configuration, and Bytewax stream metadata.
- Live chain nodes, HSMs, wallets, compliance engines, and web servers remain
  adapter boundaries.

## Verification Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/bclg/__init__.py capabilities/common/bclg/capability_contract.py capabilities/common/bclg/models.py capabilities/common/bclg/ledger_engine.py capabilities/common/bclg/service.py capabilities/common/bclg/api.py capabilities/common/bclg/views.py capabilities/common/bclg/app.py capabilities/common/bclg/test_capability_contract.py capabilities/common/bclg/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/bclg/test_capability_contract.py capabilities/common/bclg/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.bclg import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bclg --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bclg --json
git diff --check -- capabilities/common/bclg docs/progress_log.md
```
