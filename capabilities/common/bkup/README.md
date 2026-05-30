# BKUP - Backup And Restore

BKUP provides governed backup, restore, retention, and continuity operations
for APG applications. It covers tenant backup plans, encrypted snapshots,
restore approval, stale restore-test review, retention disposition, legal-hold
controls, continuity reporting, audit events, Bytewax lifecycle stream
metadata, AI backup-agent registration, UI route metadata, and visual theming.

The package is dependency-light. It does not require cloud storage APIs,
filesystem snapshot drivers, schedulers, Kubernetes/database restore
orchestration, durable databases, web servers, or live Bytewax workers.
Production deployments connect those systems through adapters after BKUP has
validated tenant context, plan ownership, source inventory, encryption,
integrity, approval, retention, stream, and audit guardrails.

## What BKUP Provides

- Tenant-qualified backup plan registry with owner, schedule, source
  inventory, retention, RPO, status, and legal-hold state.
- Snapshot vault records with encryption, integrity, lineage, region, size, and
  deterministic hash evidence.
- Production restore approval workflow with requester, reviewer, decision,
  notes, target environment, and point-in-time matching.
- Restore execution with integrity checks, stale restore-test review, RTO
  evidence, and completed or pending-review state.
- Retention disposition workflow for archive/delete decisions with legal-hold
  blocking and independent reviewer evidence.
- Continuity reports for RPO/RTO findings and restore-test freshness.
- AI backup-agent registration for `codex`, `claude_code`, `opencode`, and
  `pi` runtimes with explicit role, scope, disclosure, and policy evidence.
- Bytewax lifecycle stream metadata for batch backup mutation and generated
  application composition.
- API helpers and route-ready view models for generated APG Python
  applications.
- UI route metadata for dashboard, plans, snapshots, backup runs, restore,
  approvals, retention, agents, reports, audit, analytics, and settings.

## Package Structure

- `SPECIFICATION.md` defines functional scope, lifecycle rules, adapter
  boundaries, and acceptance criteria.
- `PLAN.md` records the implementation and review sequence for this packet.
- `cap_spec.md` points older tooling to the active specification.
- `capability_contract.py` declares configuration, guardrails, UI routes,
  theme, provides/requires metadata, and Bytewax stream metadata.
- `models.py` defines tenant-scoped backup records.
- `backup_engine.py` provides deterministic snapshot hashes and continuity
  findings.
- `service.py` implements the executable lifecycle.
- `api.py` exposes generated-application helper calls.
- `views.py` exposes route-ready UI state.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` provide package publication evidence.
- `test_capability_contract.py` and `tests/test_package_contract.py` provide
  focused verification.

## Basic Usage

```python
from capabilities.common.bkup.service import BkupService

service = BkupService()
tenant_id = "tenant-backup"

plan = service.create_backup_plan(
    plan_id="core-db",
    tenant_id=tenant_id,
    name="Core Database",
    owner="platform-owner",
    schedule="0 * * * *",
    sources=["core-primary", "core-replica"],
    retention_days=35,
    rpo_minutes=30,
)

snapshot = service.create_snapshot(
    snapshot_id="snap-core",
    tenant_id=tenant_id,
    plan_id=plan["id"],
    source_id="core-primary",
    size_bytes=2048,
    encrypted=True,
    integrity_check_passed=True,
    data_fingerprint="core-v1",
)

assert snapshot["status"] == "available"
assert len(snapshot["snapshot_hash"]) == 64
```

## Production Restore Approval

Production restores require an explicit matching restore approval record.
Caller-supplied approval booleans do not bypass the lifecycle.

```python
approval_request = service.request_restore_approval(
    approval_id="restore-approval-1",
    tenant_id=tenant_id,
    snapshot_id=snapshot["id"],
    target_environment="production",
    requested_by="recovery-owner",
    justification="Approved recovery window.",
    point_in_time="2026-05-30T00:00:00Z",
)

approval = service.decide_restore_approval(
    approval_id=approval_request["id"],
    tenant_id=tenant_id,
    reviewer="continuity-reviewer",
    decision="approved",
    notes="Integrity and rollback checks passed.",
)

restore = service.restore_snapshot(
    restore_id="restore-1",
    tenant_id=tenant_id,
    snapshot_id=snapshot["id"],
    target_environment="production",
    requested_by="recovery-owner",
    point_in_time="2026-05-30T00:00:00Z",
    integrity_check_passed=True,
    approval_id=approval["id"],
)

assert restore["status"] == "completed"
```

## Backup-Agent Governance

BKUP treats AI backup agents as governed participants in continuity workflows.
An agent must declare a supported runtime, supported role, explicit scope, and
contribution disclosure before it can be shown in generated application
surfaces.

```python
agent = service.register_backup_agent(
    agent_id="restore-review-agent",
    tenant_id=tenant_id,
    name="Restore Review Agent",
    runtime="claude-code",
    role="restore-reviewer",
    scope="Summarize production restore approval evidence.",
    contribution_disclosed=True,
    policy_ref="bkup-agent-policy",
)

assert agent["runtime"] == "claude_code"
assert agent["role"] == "restore_reviewer"
```

## Bytewax Guardrail

Batch backup mutation must declare the Bytewax lifecycle stream.

```python
service.validate_batch_backup_mutation(
    tenant_id=tenant_id,
    event_stream="bytewax",
    mutation_count=2,
)
```

## Composition Contract

`get_capability_contract()` returns the executable APG contract:

- `provides`: backup plan governance, snapshot vault, restore governance,
  retention governance, continuity reporting, and backup agents.
- `requires`: ENCR, CONF, and AUDL.
- `configuration`: plan, snapshot, restore, backup-agent, governance,
  observability, adapter, UI, and theme settings.
- `rule_engine`: deterministic guardrails for tenant context, plan owner,
  snapshot encryption/integrity, restore approval, stale restore tests,
  retention/legal hold, backup agents, audit, and Bytewax batch mutation.
- `ui`: route metadata for generated APG Python applications.
- `theme`: compact continuity operations tokens and component metadata.
- `streaming`: Bytewax processor, topic, state collections, lifecycle events,
  and batch mutation guardrail.

## Verification

Focused checks for this package:

```bash
./.venv/bin/python -m py_compile capabilities/common/bkup/__init__.py capabilities/common/bkup/capability_contract.py capabilities/common/bkup/models.py capabilities/common/bkup/backup_engine.py capabilities/common/bkup/service.py capabilities/common/bkup/api.py capabilities/common/bkup/views.py capabilities/common/bkup/app.py capabilities/common/bkup/test_capability_contract.py capabilities/common/bkup/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/bkup/test_capability_contract.py capabilities/common/bkup/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.bkup import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bkup --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bkup --json
```

Full repository suites, live storage providers, schedulers, orchestration
engines, rendered browser UI, live Bytewax workers, and load tests are separate
integration concerns.
