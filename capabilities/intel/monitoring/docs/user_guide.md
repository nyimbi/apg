# Real-Time Monitoring — User Guide

**Capability ID**: `intel_monitoring` | **Domain**: `intel` | **Version**: `1.2.0`
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft

---

## Description

`intel_monitoring` is an executable APG capability for lawful, defensive
real-time monitoring workflows. It is composable into any APG-generated
application that needs security monitoring, fraud detection, public-safety
monitoring, compliance monitoring, availability monitoring, or operational
incident triage.

All operations are tenant-scoped, evidence-linked, and policy-gated. The
capability cannot perform destructive response, autonomous enforcement,
privacy bypass, data exfiltration, or takedowns — the rule engine denies
such requests at the boundary.

---

## Installation

```bash
pip install apg-intel-monitoring
```

---

## Architecture

```
Authority → Policy → Source → Watch → Event → Signal → Incident → Referral
                                                                  ↓
                                                            Dissemination
```

Every object in the chain requires an evidence reference and is audit-logged
to an append-only ledger that can be sealed and verified via SHA-256
hash-chaining.

---

## Quick Start

```python
from capabilities.intel.monitoring import RealTimeMonitoringService
import asyncio

svc = RealTimeMonitoringService(tenant_id="acme", actor_id="analyst-1")

# 1. Establish lawful authority
svc.record_authority(
    "auth-1", "acme",
    "security_monitoring_authority", "network-perimeter",
    "confidential", "ciso-1", "2027-12-31", "evidence-ciso-auth",
)

# 2. Define monitoring policy
svc.record_policy(
    "pol-1", "acme", "detection", "Threat Intel Watch Policy",
    "medium", "auth-1", "evidence-pol-1",
)

# 3. Register data source
svc.register_source(
    "src-1", "acme", "open_source", "https://threatfeeds.example.com",
    "soc-team", "auth-1", "access-review-2026", "evidence-src-1",
)

# 4. Start keyword-based monitoring
async def setup():
    monitor = await svc.start_monitor(
        target_type="domain",
        target_id="malicious.example.com",
        keywords=["exfil", "c2", "ransomware"],
        channels=["soc-slack", "pagerduty"],
    )
    print(monitor["expression"])  # "exfil" OR "c2" OR "ransomware"

asyncio.run(setup())
```

---

## Core Workflow

### 1. Authority and Policy Setup

Before any monitoring can start, a lawful authority must be recorded and a
policy must reference it. Both require evidence references.

```python
svc.record_authority(authority_id, tenant_id, authority_type, scope_reference,
                     classification, approver_id, expires_at, evidence_reference)
svc.record_policy(policy_id, tenant_id, policy_type, name, severity_floor,
                  authority_id, evidence_reference)
```

Supported `authority_type` values: see `SUPPORTED_AUTHORITY_TYPES` in
`capability_contract.py`.

### 2. Source Registration

Every watch must reference a registered source. Sources require an access
review reference — this is the mandatory compliance gate.

```python
svc.register_source(source_id, tenant_id, source_type, source_reference,
                    owner_id, authority_id, access_review_reference, evidence_reference)
```

### 3. Watch Creation

Watches connect a policy, a source, and a detection expression.

```python
svc.record_watch(watch_id, tenant_id, policy_id, source_id, watch_type,
                 watch_expression, retention_class, evidence_reference)
```

`retention_class` controls TTL for retention enforcement:
- `ephemeral` — 7 days
- `standard` — 90 days  
- `long_term` — 365 days
- `permanent` — never purged

### 4. Event and Signal Ingestion

Events are raw observations from a watch. Signals are analyst assessments
applied to events with a severity rating.

```python
svc.record_event(event_id, tenant_id, watch_id, event_type, event_reference,
                 event_fingerprint, observed_at, confidence_score, evidence_reference)
svc.record_signal(signal_id, tenant_id, event_id, signal_type, severity,
                  confidence_score, analyst_id, evidence_reference)
```

### 5. Incident Management

Incidents are elevated from signals. From there they flow to referrals and
disseminations.

```python
svc.record_incident(incident_id, tenant_id, signal_id, incident_type,
                    severity, confidence_score, analyst_id, evidence_reference)
svc.record_referral(referral_id, tenant_id, incident_id, referral_type,
                    recipient, approval_reference, evidence_reference)
svc.record_dissemination(dissemination_id, tenant_id, incident_id, audience,
                         release_marking, approval_reference, evidence_reference)
```

---

## Async Operational Methods

### Alert Lifecycle

```python
# Ingest from external feed
record = await svc.monitor_alert({
    "event_id": "ev-001", "watch_id": "watch-1",
    "event_type": "keyword_match", "event_reference": "https://feed/item/1",
    "fingerprint": "sha256:abc123", "observed_at": "2026-06-01T12:00:00Z",
    "confidence": 0.82,
})

# Assign triage
triage = await svc.alert_triage("ev-001", "analyst-jane")

# Escalate to SOC team
escalation = await svc.escalate_alert("ev-001", "soc-tier-2")

# Auto-escalate all high/critical signals
result = await svc.escalation_auto(["sig-1", "sig-2", "sig-3"], severity_floor="high")
```

### Entity Watchlists

The watchlist provides entity-centric monitoring on top of the expression-based
watch system.

```python
# Add domain to watchlist
entity = await svc.add_to_watchlist(
    entity_type="domain",
    entity_id="suspicious.tld",
    keywords=["phishing", "credential-harvest"],
    risk_tier="high",
)

# Report hits across all watchlist entities
report = await svc.watchlist_report()
# [{entity_id, hit_count, last_seen, risk_tier, watch_id, ...}, ...]

# Remove when no longer needed
await svc.remove_from_watchlist("suspicious.tld")
```

### Adaptive Baselines

Rather than static thresholds, build per-watch baselines from historical data:

```python
baseline = await svc.update_watch_baseline("watch-domain-1", window="7d")
# {mean: 0.72, stddev: 0.08, p95: 0.91, p99: 0.97, adaptive_threshold: 0.84}

# Incoming event with confidence < adaptive_threshold → low-confidence alert
```

### Watch Expression Versioning

```python
updated = await svc.update_watch_expression(
    watch_id="watch-1",
    new_expression='"c2" OR "beacon" OR "exfil" OR "lateral"',
    change_reason="Added lateral movement keywords per threat hunt TH-2026-042",
    analyst_id="analyst-jane",
)
# History available at svc._watch_history["watch-1"]
```

### Alert Suppression

```python
# Suppress during maintenance window
suppression = await svc.alert_suppress("watch-1", duration_minutes=120, reason="Planned maintenance")

# Reinstate early
await svc.unsuppress_monitor("watch-1")
```

### Batch Processing and Correlation

```python
# Process a batch of incoming alert IDs
batch = await svc.batch_alert_processing(["ev-001", "ev-002", "ev-003"])

# Find common root cause across related alerts
correlation = await svc.alert_correlate(["ev-001", "ev-002", "ev-003"])
print(correlation["dominant_watch"], correlation["dominant_severity"])
```

### Health and Analytics

```python
# Composite 0-100 health score
health = await svc.composite_health_score()
# health_status: "healthy" | "degraded" | "critical"

# Severity heatmap for last 24 hours (hourly buckets)
heatmap = await svc.severity_heatmap(granularity="1h", periods=24)
# matrix: [{bucket: "2026-06-01T12:00Z", severities: {high: 3, medium: 8}}, ...]

# SLA breach detection
breaches = await svc.sla_breach_alert(sla_hours=4.0)

# Capacity forecast
forecast = await svc.capacity_forecast(period="30d")
```

### Temporal Analysis

```python
# Detect unusual temporal patterns in alert volume
pattern = await svc.shift_pattern_detect(period="7d")
# unusual_night_activity: True if 00:00-06:00 UTC volume > 09:00-18:00 UTC volume
```

---

## Compliance and Retention

### Retention Enforcement

```python
# Dry-run first to see what would be purged
report = await svc.enforce_retention(dry_run=True)
print(report["eligible_event_count"])

# Apply purge
await svc.enforce_retention(dry_run=False)
```

### Tamper-Evident Audit Ledger

For regulatory admissibility, seal the audit log periodically:

```python
# Seal at end of period
seal = await svc.seal_audit_ledger("2026-06-30T23:59:59+00:00")
# {ledger_root: "a3f8...", entry_count: 1247, sealed_at: ...}

# Verify at any time
check = await svc.verify_audit_ledger(seal["ledger_root"])
assert check["valid"]  # False if any entry was modified
```

---

## Dashboard Integration

### Dashboard Summary (synchronous)

```python
summary = svc.dashboard_summary("acme")
# {watch_count, event_count, signal_count, incident_count, ...}
```

### Timeline Reconstruction

```python
timeline = await svc.incident_timeline("inc-001")
# {incident: {...}, signal: {...}, event: {...}, watch: {...}}
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-monitoring/dashboard` | `intel_monitoring:view` | Overview |
| `/intel-monitoring/authorities` | `intel_monitoring:authorities` | Governance |
| `/intel-monitoring/policies` | `intel_monitoring:policies` | Planning |
| `/intel-monitoring/sources` | `intel_monitoring:sources` | Sources |
| `/intel-monitoring/watches` | `intel_monitoring:watches` | Detection |
| `/intel-monitoring/events` | `intel_monitoring:events` | Detection |
| `/intel-monitoring/signals` | `intel_monitoring:signals` | Analysis |
| `/intel-monitoring/incidents` | `intel_monitoring:incidents` | Response |
| `/intel-monitoring/watchlist` | `intel_monitoring:watchlist` | Watchlist |

---

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or environment
variables prefixed with `INTEL_MONITORING_`.

Optional integrations:
- `OLLAMA_BASE_URL` — enables `ml_alert_triage` for AI-powered FP reduction.
- `db_url` — PostgreSQL DSN for persistent storage via the `store` adapter.

---

## Composability

Reference in `.apg` source files:

```apg
use intel_monitoring;
```

Composes with: `intel_alerts`, `intel_correlation`, `intel_prediction`,
`intel_dashboard`, `intel_reporting`, `intel_threats`.

---

## Further Reading

- `service.py` — Complete business logic (sync CRUD + async operational methods)
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Rule engine and supported type registries
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 documented improvement vectors
- `README.md` — Quick reference
