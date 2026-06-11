# BKUP User Guide

**Capability**: Backup & Recovery (`bkup`) | **Domain**: `common` | **Version**: 1.1.0
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft

---

## Overview

BKUP provides governed backup, restore, retention, and continuity operations for APG
applications. It is dependency-light by design: the service layer stores all state in
memory and exposes a clean async API. Production deployments attach real storage, cloud,
and scheduler adapters without changing the service contract.

**Core capabilities**

- Tenant-scoped backup plan registry with owner, schedule, RPO, sources, retention, and
  legal-hold state.
- Snapshot vault: full, incremental, and differential backup types with lineage tracking.
- Production restore approval workflow: requester, reviewer, decision, point-in-time.
- GFS (Grandfather-Father-Son) retention policies with automatic expiry enforcement.
- Encryption-at-rest and offsite sync records.
- RPO/RTO measurement, compliance reports (SOC2, ISO 27001), and DR runbooks.
- Audit trail for every mutation.

**Advanced capabilities (v1.1)**

- Immutable Merkle-chain snapshot ledger with tamper evidence.
- Decimal-precise FinOps cost estimation and per-plan cost breakdown.
- SLA breach alerting with configurable warning/critical thresholds.
- Statistical anomaly detection (z-score) per backup type.
- WORM compliance locking with automatic expiry.
- Asyncio-bounded parallel backup execution.
- Continuous Data Protection (CDP) journal and sub-second point-in-time restore.
- Multi-region replication with quorum tracking.
- Backup Policy as Code (BPaC): JSON bundle export/import for GitOps.

---

## Installation

```bash
pip install apg-common-bkup
```

---

## Quick Start

```python
import asyncio
from capabilities.common.bkup.service import BkupService

async def main():
    svc = BkupService(actor_id="platform-admin", tenant_id="acme")

    # 1. Create a plan
    plan = await svc.create_backup_plan(
        name="Core Database",
        sources=["db-primary", "db-replica"],
        retention_days=30,
        rpo_minutes=60,
        owner="dba-team",
    )

    # 2. Schedule it
    await svc.backup_schedule(plan["plan_id"], "0 * * * *", backup_type="full")

    # 3. Run a backup
    snap = await svc.backup_run(plan["plan_id"])
    assert snap["status"] == "available"

    # 4. Check RPO compliance
    rpo = await svc.rpo_check(plan["plan_id"])
    print(f"RPO met: {rpo['rpo_met']}, gap: {rpo['gap_minutes']} min")

    # 5. Dashboard
    dash = await svc.dashboard()
    print(f"Total snapshots: {dash['total_snapshots']}")

asyncio.run(main())
```

---

## Backup Plan Lifecycle

### Creating a plan

```python
plan = await svc.create_backup_plan(
    name="Finance DB",
    sources=["finance-primary"],
    retention_days=90,
    rpo_minutes=30,
    owner="finance-ops",
)
```

### Scheduling

```python
schedule = await svc.backup_schedule(
    plan["plan_id"],
    cron_expression="0 2 * * *",  # 02:00 daily
    backup_type="full",
)
```

### Running backups

```python
# Full backup
full = await svc.backup_run(plan_id, backup_type="full")

# Incremental from a parent snapshot
incr = await svc.incremental_backup(plan_id, parent_snapshot_id=full["snapshot_id"])

# Differential from a base full snapshot
diff = await svc.differential_backup(plan_id, base_snapshot_id=full["snapshot_id"])
```

### Retention policy (GFS)

```python
await svc.retention_policy(
    plan_id,
    daily_copies=7,
    weekly_copies=4,
    monthly_copies=12,
    yearly_copies=3,
)
# Enforce TTL expiry on demand
result = await svc.enforce_expiry(plan_id)
print(f"Expired {result['expired_count']} snapshots")
```

---

## Restore Operations

### Standard restore

```python
restore = await svc.restore_from(
    snapshot_id=snap["snapshot_id"],
    target_environment="staging",
    requested_by="ops-team",
)
```

### Point-in-time restore

```python
restore = await svc.point_in_time_restore(
    plan_id=plan_id,
    target_datetime="2026-06-10T14:30:00",
    target_environment="staging",
    requested_by="ops-team",
)
```

### Approval workflow

```python
# Request and approve
await svc.approve_restore(restore["restore_id"], reviewer="security-team", notes="Approved")
```

### Sandbox test restore

```python
result = await svc.test_restore(snap["snapshot_id"], sandbox_environment="sandbox")
assert result["passed"] is True
```

---

## Compliance & Governance

### Encryption at rest

```python
await svc.encryption_at_rest(snap["snapshot_id"], key_ref="kms://prod-key")
```

### Legal hold

```python
await svc.legal_hold(plan_id, hold=True, reason="Litigation hold – case 2026-XYZ")
# Lift hold
await svc.legal_hold(plan_id, hold=False)
```

### WORM locking

Prevents deletion or expiry of a snapshot before a fixed timestamp — required by SEC
17a-4, FINRA, and similar regulations.

```python
await svc.worm_lock(
    snap["snapshot_id"],
    lock_until="2027-01-01T00:00:00",
    reason="SEC-17a-4 compliance",
)
# List all active locks
locked = await svc.list_worm_locked_snapshots()
```

Attempting `bulk_delete_snapshots` or `enforce_expiry` on a WORM-locked snapshot raises
`ValueError` with the lock expiry timestamp.

### Compliance report

```python
report = await svc.compliance_report(framework="SOC2")
print(f"Encryption rate: {report['encryption_rate']}")
print(f"Offsite rate: {report['offsite_rate']}")
print(f"DR tests executed: {report['dr_tests_executed']}")
```

---

## Disaster Recovery

### DR test

```python
dr = await svc.disaster_recovery_test(plan_id, scenario="full_site_failure")
assert dr["passed"] is True
```

### Full DR runbook

```python
runbook = await svc.dr_runbook_execute(plan_id, scenario="region_failure")
print(f"Overall pass: {runbook['overall_pass']}")
print(f"RTO: {runbook['rto_estimate']['estimated_rto_minutes']} min")
```

---

## Immutable Merkle Ledger

Append each snapshot to the chain immediately after creation. The ledger is per-tenant
and append-only. `verify_ledger()` recomputes every SHA-256 link and returns the index
of the first broken entry.

```python
await svc.ledger_append(snap["snapshot_id"])
result = await svc.verify_ledger()
assert result["valid"] is True
assert result["entry_count"] >= 1
```

If a snapshot record is modified retroactively, `verify_ledger()` returns:

```python
{"valid": False, "tampered_at_index": 3, "reason": "hash_mismatch", "verified_at": "..."}
```

---

## FinOps: Cost Estimation

All monetary values use `decimal.Decimal` with `ROUND_HALF_UP` — no float rounding
accumulation across large snapshot estates.

```python
from decimal import Decimal

# Per-plan cost
cost = await svc.estimate_backup_cost(
    plan_id,
    storage_cost_per_gb=Decimal("0.023"),
    egress_cost_per_gb=Decimal("0.009"),
)
print(f"Monthly cost: ${cost['total_monthly_cost_usd']}")

# Tenant-wide breakdown
report = await svc.cost_breakdown_report(
    storage_cost_per_gb=Decimal("0.023"),
    egress_cost_per_gb=Decimal("0.009"),
)
print(f"Grand total: ${report['grand_total_monthly_cost_usd']}")
```

---

## SLA Breach Alerting

```python
# warn_pct=0.8 means warn when 80% of RPO budget is consumed
result = await svc.sla_breach_check(plan_id, warn_pct=0.8)
# result["severity"]: "ok" | "warning" | "critical" | "unknown"

# Query breach history
critical_events = await svc.list_sla_events(severity="critical")
```

---

## Anomaly Detection

Detects snapshots whose `size_bytes` deviates more than `z_threshold` standard
deviations from the per-backup-type rolling mean. Requires ≥ 3 data points per type.
Typical signal: ransomware encrypting data (massive size spike) or misconfigured source.

```python
anomalies = await svc.detect_anomalies(plan_id, z_threshold=3.0)
for a in anomalies:
    print(f"[{a['severity']}] snapshot {a['snapshot_id']} z={a['z_score']}")

# Query anomaly log
all_critical = await svc.list_anomalies(severity="critical")
```

---

## Parallel Backup Execution

Run all sources in a plan concurrently, bounded by `max_concurrency` to prevent
resource exhaustion. Failed sources are recorded without aborting the run.

```python
result = await svc.parallel_backup_run(
    plan_id,
    backup_type="full",
    max_concurrency=4,
)
print(f"{result['succeeded']}/{result['total_sources']} sources succeeded")
for outcome in result["outcomes"]:
    print(f"  {outcome['source']}: {outcome['status']} in {outcome['duration_ms']}ms")
```

---

## Continuous Data Protection (CDP)

CDP journals every write event, enabling recovery to any second within the retention
window. Enable CDP on the plan before recording events.

```python
# Enable CDP
await svc.update_plan(plan_id, cdp_enabled=True)

# Stream change events (call from your adapter/WAL reader)
await svc.journal_write_event(plan_id, "db-primary", "row-update: orders.id=42", bytes_changed=1024)
await svc.journal_write_event(plan_id, "db-primary", "row-insert: orders.id=43", bytes_changed=512)

# Stats
stats = await svc.cdp_journal_stats(plan_id)
print(f"Journal events: {stats['event_count']}, total bytes: {stats['total_bytes']}")

# Sub-second point-in-time restore
restore = await svc.cdp_restore_to_second(
    plan_id,
    target_datetime="2026-06-11T14:30:45",
    target_environment="staging",
    requested_by="ops",
)
print(f"Replay {restore['replay_event_count']} CDP events to reach target second")
```

---

## Multi-Region Replication

```python
rep = await svc.replicate_to_regions(
    snap["snapshot_id"],
    regions=["us-east-1", "eu-west-1", "ap-southeast-1"],
    quorum=2,
)
assert rep["quorum_met"] is True
# Later, verify quorum
assert await svc.quorum_met(snap["snapshot_id"]) is True
```

---

## Backup Policy as Code (BPaC)

Export plan definitions as a versioned JSON bundle for GitOps / version control:

```python
bundle_json = await svc.export_policy_bundle([plan_id])

# On another environment or tenant:
result = await svc.import_policy_bundle(bundle_json, conflict_mode="skip")
print(f"Created: {result['created']}, skipped: {result['skipped']}")

# Overwrite existing plans:
result = await svc.import_policy_bundle(bundle_json, conflict_mode="overwrite")
```

The bundle schema (`policy_version: "1.0"`) includes plan, schedule, and retention
policy records. Idempotent re-import with the same `plan_id` + `conflict_mode="skip"` is
a no-op — safe for CI pipelines.

---

## Audit Trail

Every method emits an audit event. Query with optional type filter:

```python
events = await svc.audit_trail()
dr_events = await svc.audit_trail(event_type="dr_runbook_executed")
```

---

## Reporting & Export

```python
# Snapshot catalogue (filtered by plan)
catalogue = await svc.backup_catalogue(plan_id=plan_id)

# Summary report
report = await svc.backup_report(plan_id=plan_id)

# Storage utilisation per plan
storage = await svc.storage_utilisation()

# Export snapshots to CSV
csv_data = await svc.export_snapshots_csv()

# Export plan report to JSON
json_data = await svc.export_plan_report_json()
```

---

## Configuration Reference

All configuration is tenant-scoped. Use the `conf` capability or environment variables
prefixed with `BKUP_`.

| Key | Default | Description |
|-----|---------|-------------|
| `BKUP_DEFAULT_RETENTION_DAYS` | `30` | Retention if not specified on plan |
| `BKUP_DEFAULT_RPO_MINUTES` | `60` | RPO target if not specified on plan |
| `BKUP_ENCRYPTION_KEY_REF` | `kms://default` | Default KMS key reference |
| `BKUP_MAX_PARALLEL_CONCURRENCY` | `4` | Default `max_concurrency` for parallel runs |

---

## Composition

```apg
use bkup;
```

`bkup` requires: `encr`, `conf`, `audl`.

`bkup` provides: `backup_plan_governance`, `snapshot_vault`, `restore_governance`,
`retention_governance`, `continuity_reporting`.

---

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/bkup/service.py
./.venv/bin/pytest -q capabilities/common/bkup/test_capability_contract.py capabilities/common/bkup/tests/test_package_contract.py
```
