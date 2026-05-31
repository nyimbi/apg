# Capability Development Plan

This plan operationalizes the capability specification. It is the shared
sequence for developing all APG capabilities methodically and in parallel.

## Current Baseline

The capability registry currently exposes 109 contracts. The global target is
for every contract to remain valid, package-complete, domain-specific,
publish-plan-ready, documented, and reviewable.

Baseline commands:

```bash
./.venv/bin/apg capabilities list --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities lifecycle-audit --json
./.venv/bin/apg tooling audit --json
```

## Per-Capability Build Cycle

Apply this cycle to one package at a time:

1. **Specification**
   - Add or update `SPECIFICATION.md`.
   - Define users, outcomes, service boundary, data ownership, rules, UI,
     theme, adapters, test gates, and non-goals.

2. **Plan**
   - Add or update `PLAN.md`.
   - Break work into small packets: models, service, API, views, tests, docs,
     review fixes, and publish evidence.

3. **Implementation**
   - Keep changes inside the package root unless a shared contract requires a
     coordinated change.
   - Implement domain runtime behavior, not generic generated record storage.
   - Keep provider integrations behind adapters.

4. **Focused verification**
   - Run package tests.
   - Run root implementation audit.
   - Run publish-plan.
   - Run diff whitespace checks.

5. **Code review**
   - Review changed files for domain correctness, tenant boundaries, rule
     enforcement, audit events, API/view consistency, stale docs, and missing
     negative tests.
   - Fix all concrete review findings before moving on.

6. **Global evidence**
   - Run global capability audits when implementation-depth counts or package
     completeness changed.
   - Run tooling audit when shared CLI, compiler, docs, or generated evidence
     changed.

7. **Commit**
   - Stage exact files.
   - Commit with Lore trailers.
   - Push completed, verified slices regularly.

## Parallel Execution Model

Use multiple agents when work can be isolated by package root.

Safe parallel packets:

- `capabilities/common/accs/` specification and review;
- `capabilities/common/agnt/` service/test improvements;
- `capabilities/fin/glr/general_ledger/` lifecycle deepening;
- `capabilities/composition/events/` Bytewax event semantics;
- documentation-only capability index updates.

Unsafe parallel packets without coordination:

- changing `capabilities/capability_contract_registry.py`;
- changing CLI command names or JSON formats;
- renaming capability IDs, route names, rule IDs, or theme names;
- refreshing generated evidence for the same package from two agents;
- changing shared docs that define acceptance gates.

## First Pass Queue

Use the registry order unless an audit reports a failure:

1. `accs` - Accessibility Services
2. `agnt` - AI Agent Composition
3. `aicr` - AI Core Framework
4. `anom` - Anomaly Detection
5. `apig` - APG Intelligent Gateway
6. Continue through `./.venv/bin/apg capabilities list --json`.

For each package, create or verify these local artifacts before implementation:

- `SPECIFICATION.md`;
- `PLAN.md`;
- `cap_spec.md`;
- positive lifecycle tests;
- negative guardrail tests;
- publish-plan evidence.

## Review Checklist

Before committing a capability package:

- Contract shape validates.
- Service methods enforce contract rules.
- Tenant context is required on tenant-sensitive operations.
- High-risk or destructive operations have review or denial paths.
- Audit events or evidence records are produced for important decisions.
- API helpers expose the same behavior as service methods.
- View models expose routes, records, actions, rule state, and theme metadata.
- Tests cover at least one successful lifecycle and key guardrail failures.
- `cap_spec.md`, `SPECIFICATION.md`, and `PLAN.md` reflect current behavior.
- Focused commands passed and were inspected.

## Completion Criteria

The full capability-development goal is complete only when every capability has
been reviewed through this cycle and the repository still passes:

```bash
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities lifecycle-audit --json
./.venv/bin/apg tooling audit --json
./.venv/bin/apg docs audit --json
```
