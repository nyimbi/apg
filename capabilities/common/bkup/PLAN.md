# BKUP Implementation Plan

## Scope

Implement one coherent backup and restore governance packet:

1. Tenant-qualified state for plans, snapshots, restores, continuity reports, approvals, dispositions, and audit events.
2. Explicit restore approval lifecycle for production and high-risk recovery.
3. Independent review guardrails for restore review and retention disposition.
4. Legal-hold-aware snapshot disposition.
5. API/view/contract/theme/semantic evidence updates.
6. Focused package tests and proof.

## Non-Goals

- Do not integrate cloud storage providers.
- Do not build real filesystem, VM, database, or Kubernetes restore adapters.
- Do not add dependencies.
- Do not run full repository suites while battery is constrained.

## Implementation Steps

### 1. Extend Models

Add lifecycle records for:

- restore approval;
- retention disposition.

Add approval/reviewer evidence to restore records and disposition status to snapshots.

### 2. Harden Service State

Convert in-memory stores to tenant-qualified keys and add helper methods for:

- duplicate detection;
- tenant-specific lookup;
- sorted listing;
- rule reason extraction.

### 3. Implement Restore Approval

Add request/decision methods. Production restore execution must require a matching approved approval record. Caller approval booleans must not bypass this gate.

### 4. Implement Retention Disposition

Add request/decision methods. Legal hold must block disposition. Approved deletion should transition the snapshot out of `available` state. Rejected dispositions must not mutate the snapshot.

### 5. Update Composition Surfaces

Update:

- `api.py`;
- `views.py`;
- `capability_contract.py`;
- `__init__.py`;
- `app.py`;
- `package_manifest.json`;
- `semantic_model.json`;
- `release_report.json`;
- `cap_spec.md`;
- package contract tests.

### 6. Review and Proof

Run only focused checks:

- `py_compile` on changed BKUP package files;
- focused BKUP pytest files;
- `apg capabilities implementation-audit --root capabilities/common/bkup --json`;
- `apg capabilities publish-plan capabilities/common/bkup --json`;
- stale baseline-marker search;
- `git diff --check`.

Review the diff manually and fix blocking guardrail gaps before commit.
