# BKUP Capability Specification

## Purpose

BKUP provides governed backup, restore, retention, and continuity operations for generated APG applications. It must let applications define backup plans, create encrypted snapshots, prove restore readiness, request production restores, review risky recovery operations, and dispose of snapshots under retention and legal-hold controls.

The package runtime is dependency-light and executable without storage providers, schedulers, cloud APIs, orchestration engines, or production databases. Production deployments bind those systems through adapters that honor the same contract.

## Composable Capabilities

BKUP exposes these application components:

- Backup plan manager for tenant-scoped schedules, source inventories, owners, RPO targets, retention policy, and legal-hold state.
- Snapshot vault for encrypted snapshots, integrity evidence, lineage, source mapping, region placement, and availability state.
- Restore console for point-in-time restore requests, production approval gates, stale restore-test review, RTO evidence, and completed/rejected recovery state.
- Restore approval queue for independent approval of production or high-risk restore operations.
- Retention disposition queue for legal-hold-aware snapshot deletion or archival approvals.
- Continuity reporting for restore tests, RPO/RTO findings, stale test detection, and audit evidence.
- Backup-agent governance for Codex, Claude Code, OpenCode, and Pi style runtimes with explicit role, scope, disclosure, and audit evidence.
- Bytewax lifecycle stream metadata for batch backup mutation and generated application composition.
- Dashboard and view models that generated APG applications can render directly.

## Lifecycle

### 1. Backup Plan Creation

A valid plan requires:

- tenant context;
- accountable owner;
- schedule;
- non-empty source inventory;
- positive retention period;
- RPO target.

Plan IDs are tenant-local. Two tenants may use the same plan ID without collision.

### 2. Snapshot Creation

A valid snapshot requires:

- tenant context;
- existing plan and source in the plan source inventory;
- encryption;
- integrity check evidence;
- size evidence;
- lineage metadata;
- region metadata.

Unencrypted or failed-integrity snapshots are rejected and never enter the available vault.

### 3. Restore Request

Restore requests require:

- tenant context;
- existing available snapshot;
- integrity check;
- requester;
- target environment;
- optional point-in-time value.

Production restores require an approved matching restore approval record. Caller-supplied booleans such as `approval_recorded=True` are not trusted as governance evidence.

Stale restore-test context creates a pending review state. Review decisions must be independent and include notes.

### 4. Restore Approval

Restore approval records are explicit tenant-scoped governance objects. A valid restore approval:

- matches tenant, snapshot, target environment, and point-in-time value;
- is requested by the restore operator;
- is decided by an independent reviewer;
- includes reviewer notes;
- is approved before production restore execution.

Rejected restore approvals cannot be used for restore execution.

### 5. Continuity Reporting

Restore tests record RPO/RTO findings and stale-test status. Reports are tenant-qualified and auditable. Stale restore tests must not be treated as current evidence without review.

### 6. Retention Disposition

Snapshot disposition requires explicit approval. The runtime supports:

- retention deletion;
- archival handoff;
- legal-hold blocking;
- independent reviewer decision;
- reviewer notes;
- snapshot status transition after approval.

Snapshots under legal hold cannot be deleted or archived through disposition approval.

### 7. Audit Evidence

Every plan, snapshot, restore, approval, disposition, and continuity transition emits tenant-scoped audit evidence with:

- event ID;
- subject ID;
- event type;
- actor;
- decision;
- reasons;
- metadata.

Production deployments may forward these events to AUDL.

### 8. Backup-Agent Registration

AI backup agents are governed lifecycle records, not hidden automation. A valid
agent registration requires:

- tenant context;
- supported runtime: `codex`, `claude_code`, `opencode`, or `pi`;
- supported role: `plan_reviewer`, `snapshot_reviewer`, `restore_reviewer`,
  `retention_reviewer`, or `continuity_reviewer`;
- explicit operating scope;
- contribution disclosure;
- optional policy reference.

Generated applications must display agent scope and disclosure on review
surfaces so human approvers can distinguish agent-assisted summaries from
direct reviewer decisions.

### 9. Streaming

BKUP publishes Bytewax lifecycle stream metadata through the capability
contract and generated semantic model:

- processor: `bytewax`;
- topic: `apg.bkup.lifecycle`;
- state collections: plans, snapshots, restores, restore approvals, retention
  dispositions, continuity reports, backup agents, and audit events;
- events: plan creation, snapshot creation, restore approval decisions,
  restore requests, restore review approvals, restore-test records, retention
  disposition decisions, and backup-agent registration;
- batch mutation guardrail: `batch_backup_mutation_requires_bytewax`.

## Rule Engine

The capability contract must expose deterministic rules for:

- tenant context;
- accountable plan owner;
- snapshot encryption;
- restore integrity checks;
- production restore approval;
- stale restore test review;
- independent restore reviewer;
- legal-hold retention block;
- independent retention reviewer.
- backup-agent registration, supported runtime, supported role, scope, and
  disclosure;
- lifecycle state-change audit evidence;
- Bytewax batch backup mutation.

The runtime must enforce equivalent behavior and fail closed.

## UI and Theming

BKUP must expose UI routes and theme components for:

- dashboard;
- plan manager;
- snapshot vault;
- backup runs;
- restore console;
- restore approval queue;
- retention policy;
- retention disposition queue;
- backup-agent panel;
- continuity reports;
- audit;
- analytics;
- settings.

Theme components must support compact operational surfaces for RPO/RTO state, encryption state, lineage, restore approvals, legal hold, retention disposition, and continuity findings.

## Adapter Boundaries

The package runtime must not implement or require:

- cloud storage provider APIs;
- filesystem snapshots;
- Kubernetes/VM/database restore orchestration;
- schedulers;
- production persistence;
- legal/regulatory advice;
- live disaster-recovery execution.
- live Bytewax topology execution.

Those concerns belong behind adapters.

## Acceptance Criteria

- The package has `SPECIFICATION.md` and `PLAN.md`.
- The package has a practical root `README.md`.
- The service stores records by tenant-qualified keys.
- Restore approval and retention disposition are explicit lifecycle records, not raw booleans.
- Backup agents can be registered with supported runtime, supported role,
  scope, disclosure, and policy evidence.
- Unsupported backup-agent runtime, missing scope, or undisclosed contribution
  fails closed.
- Batch backup mutation validation accepts Bytewax and denies other stream
  providers.
- API helpers and view models expose approval/disposition queues.
- Generated semantic model exposes backup-agent route, provides/requires
  metadata, and Bytewax stream metadata.
- Contract, semantic model, release report, and manifest include the new surfaces.
- Focused tests prove positive and negative governance paths.
- Implementation audit and publish-plan pass for `capabilities/common/bkup`.

## Focused Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/bkup/__init__.py capabilities/common/bkup/capability_contract.py capabilities/common/bkup/models.py capabilities/common/bkup/backup_engine.py capabilities/common/bkup/service.py capabilities/common/bkup/api.py capabilities/common/bkup/views.py capabilities/common/bkup/app.py capabilities/common/bkup/test_capability_contract.py capabilities/common/bkup/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/bkup/test_capability_contract.py capabilities/common/bkup/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.bkup import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bkup --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bkup --json
```
