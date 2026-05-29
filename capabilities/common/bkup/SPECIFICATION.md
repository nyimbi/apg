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
- continuity reports;
- audit;
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

Those concerns belong behind adapters.

## Acceptance Criteria

- The package has `SPECIFICATION.md` and `PLAN.md`.
- The service stores records by tenant-qualified keys.
- Restore approval and retention disposition are explicit lifecycle records, not raw booleans.
- API helpers and view models expose approval/disposition queues.
- Contract, semantic model, release report, and manifest include the new surfaces.
- Focused tests prove positive and negative governance paths.
- Implementation audit and publish-plan pass for `capabilities/common/bkup`.
