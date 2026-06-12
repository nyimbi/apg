# APG Real-Time Monitoring

`intel_monitoring` is an executable APG capability for lawful, defensive
real-time monitoring workflows. Compose into generated APG applications that
need security monitoring, fraud detection, public-safety monitoring, compliance
monitoring, availability monitoring, or operational incident triage.

## What It Provides

- Authority, monitoring policy, source, watch, event, signal, incident,
  referral, dissemination, review, and AI-agent workflows.
- Deterministic rules enforcing tenant context, lawful authority, source access
  review, evidence, approvals, Bytewax lifecycle routing, and AI-agent guardrails.
- Adaptive baseline engine: per-watch rolling confidence statistics
  (mean, stddev, p95, p99) with auto-computed detection thresholds.
- Entity watchlist: first-class monitored entities (IP, domain, person, org)
  with hit-count tracking and last-seen timestamps.
- Watch expression versioning: full history with rollback and analyst attribution.
- Structured alert suppression with reinstatement.
- Tamper-evident audit ledger via SHA-256 hash chaining.
- Retention policy enforcement: ephemeral (7d), standard (90d), long-term (365d).
- Composite health score (0–100) synthesising stale watches, SLA breaches, and FP rates.
- Severity heatmap: time-bucketed cross-tabulation for dashboard rendering.
- API helpers and view models callable without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/monitoring/app.py
./.venv/bin/pytest -q capabilities/intel/monitoring/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_monitoring --json
```

### Basic integration

```python
from capabilities.intel.monitoring import RealTimeMonitoringService

service = RealTimeMonitoringService()
authority = service.record_authority(
    "auth-1", "tenant-a",
    "security_monitoring_authority", "monitoring-scope",
    "confidential", "approver-1", "2026-12-31", "evidence-auth",
)
policy = service.record_policy(
    "pol-1", "tenant-a", "detection", "Keyword Watch Policy",
    "medium", "auth-1", "evidence-pol",
)
source = service.register_source(
    "src-1", "tenant-a", "open_source", "https://feeds.example.com",
    "owner-1", "auth-1", "access-review-1", "evidence-src",
)
```

### Async operational methods

```python
import asyncio

async def run():
    svc = RealTimeMonitoringService(tenant_id="tenant-a")
    # ... register authority/policy/source first ...

    # Start entity watchlist monitoring
    entity = await svc.add_to_watchlist("domain", "evil.example.com", ["ransomware", "c2"], "high")

    # Adaptive baseline
    baseline = await svc.update_watch_baseline(entity["watch_id"])
    print(baseline["adaptive_threshold"])

    # Severity heatmap for dashboard
    heatmap = await svc.severity_heatmap(granularity="1h", periods=24)

    # Composite health score
    health = await svc.composite_health_score()
    print(health["health_status"])   # "healthy" | "degraded" | "critical"

    # Seal the audit ledger for compliance
    seal = await svc.seal_audit_ledger("2026-12-31T23:59:59+00:00")
    verify = await svc.verify_audit_ledger(seal["ledger_root"])
    assert verify["valid"]

asyncio.run(run())
```

## Async Method Reference

| Method | Purpose |
|--------|---------|
| `start_monitor` | Create keyword watch for a target |
| `stop_monitor` | Deactivate watch |
| `monitor_alert` | Ingest inbound alert payload |
| `alert_triage` | Assign analyst to alert |
| `escalate_alert` | Escalate alert; promote to incident if signal exists |
| `escalation_auto` | Auto-escalate by severity floor |
| `bulk_monitor` | Start monitors for multiple targets |
| `monitor_health_check` | Counts, coverage, signal rates |
| `false_positive_rate` | FP rate for a monitor over period |
| `flag_false_positive` | Register event fingerprint as FP |
| `monitor_analytics` | Aggregated stats over period |
| `batch_alert_processing` | Enrich batch of alert IDs |
| `watch_coverage_report` | Per-watch event and signal counts |
| `source_health_summary` | Event volume per registered source |
| `alert_suppress` | Suppress alerts for duration |
| `alert_correlate` | Correlate alerts for common root cause |
| `threshold_adapt` | Update detection threshold |
| `shift_pattern_detect` | Temporal attack window analysis |
| `monitoring_schedule` | Schedule periodic checks |
| `anomaly_root_cause` | Root-cause analysis for incident |
| `sla_breach_alert` | Identify unresolved incidents past SLA |
| `monitor_export` | Export monitoring state |
| `capacity_forecast` | Forecast utilisation |
| `incident_timeline` | Build incident → signal → event → watch chain |
| **`update_watch_baseline`** | Adaptive confidence baseline |
| **`update_watch_expression`** | Version-controlled keyword update |
| **`unsuppress_monitor`** | Early suppression reinstatement |
| **`add_to_watchlist`** | Add entity to entity watchlist |
| **`remove_from_watchlist`** | Remove entity from watchlist |
| **`watchlist_report`** | Hit counts + last-seen per entity |
| **`severity_heatmap`** | Time-bucketed severity matrix |
| **`seal_audit_ledger`** | SHA-256 hash-chain audit seal |
| **`verify_audit_ledger`** | Verify sealed ledger integrity |
| **`enforce_retention`** | Purge expired records by TTL class |
| **`composite_health_score`** | 0–100 synthesised health score |

Bold = new in v1.2.0.

## Guardrails

Defensive and compliance-first. Does not implement destructive response,
autonomous enforcement, privacy bypass, data exfiltration, unauthorized
monitoring expansion, account actions, or takedowns. AI-agent actions
requesting those scopes are denied by the rule engine.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/intel/monitoring/*.py capabilities/intel/monitoring/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/monitoring/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/monitoring --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/monitoring --json
```

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements: Intel Monitoring Capability
- **I2.** Adaptive Baseline Engine
- **I3.** Multi-Tenant Watch Namespace Isolation
- **I4.** Streaming Event Ingestion Pipeline
- **I5.** Keyword Watch Versioning
- **I6.** Alert Deduplication Registry
- **I7.** Structured Suppression with Reinstatement
- **I8.** Signal Enrichment Pipeline
- **I9.** Cross-Tenant Federated Watch Sharing
- **I10.** Watchlist-Driven Entity Monitoring
- **I11.** Incident Playbook Integration
- **I12.** Confidence Score Calibration
- **I13.** Retention Policy Enforcement
- **I14.** Real-Time Severity Heatmap
- **I15.** Composite Health Score

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
