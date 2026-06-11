# APG Audit Log (audl) — User Guide

© 2025 Datacraft  www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

---

## 1. Introduction

`audl` is the foundational audit capability for the APG platform. It provides:

- Append-only, tamper-evident event storage with SHA-256 checksum + chain hash
- Multi-tenant isolation enforced at every read/write path
- GDPR Art. 15/17/20/25 compliance primitives
- Decimal-precise financial event logging
- Real-time SIEM stream and alert routing
- SOX, GDPR, HIPAA, PCI-DSS, and SOC-2 compliance reporting

All public methods on `AuditLoggingService` are `async`. The service requires no external process to run standalone — PostgreSQL and NATS are optional enhancements.

---

## 2. Instantiation

```python
from capabilities.common.audl.service import AuditLoggingService

# db_session: SQLAlchemy async session, or None for in-memory-only mode
svc = AuditLoggingService(
	db_session=None,          # or: async_session_factory()
	tenant_id="acme-corp",    # must be non-empty
	actor_id="system-init",   # must be non-empty
)
```

`tenant_id` and `actor_id` are fixed for the lifetime of the service instance. To act as a different actor within the same tenant, construct a new instance sharing the same `db_session`.

---

## 3. Logging Events

### 3.1 Single Event

```python
from capabilities.common.audl.models import AuditEventCreate, AuditLevel, AuditEventType, EventSource

ev = await svc.log_event(
	who     = "alice@acme.com",
	what    = "update_profile",
	on_what = "user-profile-99",
	how     = AuditEventType.DATA_UPDATE,
	where   = "197.248.100.5",
	when    = None,    # defaults to now(UTC)
	result  = True,
	payload = AuditEventCreate(
		tenant_id    = "acme-corp",
		level        = AuditLevel.INFO,
		event_type   = AuditEventType.DATA_UPDATE,
		source       = EventSource.APG_CORE,
		category     = "user_management",
		action       = "update_profile",
		compliance_tags = ["gdpr"],
		contains_pii    = True,
	),
)
print(ev.id, ev.checksum)
```

### 3.2 Batch Ingestion

```python
events = [AuditEventCreate(...), AuditEventCreate(...)]
results = await svc.immutable_log_write(events)
# Max 10 000 per batch; raises ValueError if exceeded
```

### 3.3 Financial Events (Decimal precision)

```python
from decimal import Decimal

ev = await svc.log_financial_event(
	who      = "payment-svc",
	what     = "charge",
	on_what  = "invoice-001",
	how      = AuditEventType.DATA_UPDATE,
	where    = None,
	when     = None,
	result   = True,
	amount   = Decimal("4999.99"),
	currency = "KES",
)
# ev.details["monetary_amount"] is a Decimal-string quantized to 8 dp
# ev.details["monetary_currency"] == "KES"
```

Amount is included in the event's checksum pre-image — any post-write mutation is detectable.

---

## 4. Integrity Verification

### 4.1 Full Chain Check

```python
report = await svc.log_integrity_check()
# {integrity: "intact" | "broken", chain_breaks: [...], events_checked: N}
```

### 4.2 Single Event Verify

```python
result = await svc.tamper_proof_verify(event_id="01JXF...")
# {status: "clean" | "suspect", checksum_ok: bool, stored: ..., expected: ...}
```

### 4.3 Batch Checksum Verify

```python
verdict = await svc.verify_event_checksum_batch(event_ids=["id1", "id2"])
# {clean: N, suspect: N, not_found: N, verdicts: {"id1": "clean", ...}}
```

### 4.4 Chain Anchoring

```python
anchor = await svc.chain_anchoring_record()
# Submit anchor["anchor_hash"] to an external notary / blockchain.
# anchor["payload_bytes"] is the hex-encoded pre-image for later verification.
```

---

## 5. Querying

### 5.1 Structured Search

```python
from capabilities.common.audl.models import AuditQueryCreate

result = await svc.audit_trail_search(AuditQueryCreate(
	tenant_id   = "acme-corp",
	event_types = [AuditEventType.USER_FAILED_LOGIN],
	success     = False,
	date_start  = datetime.now(timezone.utc) - timedelta(hours=24),
	date_end    = datetime.now(timezone.utc),
	limit       = 50,
	requested_by = "security-analyst",
))
print(result.total_count, result.has_more)
```

### 5.2 Advanced Search (risk band + anomaly)

```python
result = await svc.audit_query_advanced(
	filters          = query_create,
	risk_band        = "high",       # low | medium | high | critical
	anomaly_threshold = 0.6,
)
```

### 5.3 Actor / Resource Shortcuts

```python
events = await svc.search_by_actor("alice@acme.com", limit=100)
events = await svc.search_by_resource("invoice-001")
```

### 5.4 Session Timeline

```python
timeline = await svc.actor_session_timeline(
	actor_id   = "alice@acme.com",
	session_id = "sess-abc123",
)
# [{event_id, timestamp, event_type, action, delta_ms, risk_score}, ...]
```

---

## 6. Compliance Reporting

### 6.1 Generate a Report

```python
from capabilities.common.audl.models import ComplianceFramework

report = await svc.audit_report_generate(
	period_start = datetime(2025, 1, 1, tzinfo=timezone.utc),
	period_end   = datetime(2025, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
	framework    = ComplianceFramework.SOC_2,
	requested_by = "compliance-officer",
	include_recommendations = True,
)
print(report.violation_count, report.summary)
```

### 6.2 Gap Analysis

```python
gap = await svc.compliance_gap_analysis(
	framework    = ComplianceFramework.PCI_DSS,
	period_start = ...,
	period_end   = ...,
)
# {coverage_pct: 85.7, status: "non_compliant", gaps: [{event_type, count, met}, ...]}
```

### 6.3 Risk Summary

```python
summary = await svc.risk_summary(period_start, period_end)
print(summary.high_risk_count, summary.compliance_violations)
```

---

## 7. GDPR & Privacy Operations

### 7.1 Subject Access Request (Art. 15)

```python
from capabilities.common.audl.models import DataSubjectRequestCreate, DSRType

dsr = await svc.gdpr_data_subject_access(
	DataSubjectRequestCreate(
		tenant_id     = "acme-corp",
		dsr_type      = DSRType.ACCESS,
		subject_id    = "user-99",
		requested_by  = "user-99",
		justification = "Right of access under GDPR Art. 15",
	),
	is_admin = False,
)
print(dsr.response_data["event_count"])
```

### 7.2 Erasure Impact (Art. 17)

```python
impact = await svc.right_to_erasure_audit_impact("user-99")
# pii_pseudonymisable: [list of event IDs that can have PII scrubbed]
# erasure_blocked: count — these cannot be deleted (Art. 17(3)(b) exemption)
```

### 7.3 GDPR Log Erasure

```python
result = await svc.gdpr_log_erasure(
	subject_id    = "user-99",
	justification = "Erasure request verified, Art. 17 approved",
	dry_run       = True,    # inspect impact first
)
```

### 7.4 Actor Pseudonymisation

```python
result = await svc.pseudonymise_actor(
	real_actor_id = "alice@acme.com",
	pseudonym     = "anon-7f3a",
	justification = "Analytics export — identity not required",
	dry_run       = False,
)
```

### 7.5 PII Field Masking

```python
result = await svc.pii_mask_in_logs(
	subject_id     = "user-99",
	fields_to_mask = ["email", "phone", "ip_address"],
)
```

---

## 8. Security Operations

### 8.1 Tamper Detection Scan

```python
from capabilities.common.audl.models import TamperDetectionCreate

scan = await svc.tamper_detection(TamperDetectionCreate(
	tenant_id  = "acme-corp",
	scan_type  = "on-demand",
	scanned_by = "security-bot",
))
print(scan.status, scan.events_suspect)
```

### 8.2 Velocity Check

```python
status = await svc.velocity_check(
	actor_id       = "api-key-XYZ",
	window_minutes = 5,
	threshold      = 200,
)
if status["breached"]:
	# Revoke key, alert SOC
```

### 8.3 High Risk Events

```python
events = await svc.high_risk_events(threshold=0.7, limit=50)
```

### 8.4 Anomaly Detection

```python
anomalous = await svc.anomaly_in_audit(threshold=0.7, period_start=..., period_end=...)
```

---

## 9. Legal Hold Management

### 9.1 Apply Hold

```python
ev = await svc.set_legal_hold(event_id, hold=True, reason="Litigation hold — case #2025-001")
```

### 9.2 Bulk Hold

```python
result = await svc.bulk_set_legal_hold(
	event_ids = ["id1", "id2", "id3"],
	hold      = True,
	reason    = "Regulatory investigation",
)
```

### 9.3 Rotate Custodian

```python
result = await svc.rotate_legal_hold_ownership(
	event_id    = "id1",
	new_steward = "bob@acme.com",
	reason      = "Alice resigned; Bob is new custodian",
)
# ev.details["legal_hold_custody_log"] updated with transfer record
```

---

## 10. Evidence Packages

```python
from capabilities.common.audl.models import EvidencePackageCreate

pkg = await svc.evidence_package_export(EvidencePackageCreate(
	tenant_id    = "acme-corp",
	name         = "Case-2025-001 Evidence",
	event_ids    = [ev.id for ev in events],
	requested_by = "legal@acme.com",
	reason       = "e-Discovery production",
	legal_matter = "Case-2025-001",
	include_chain = True,
))
print(pkg.file_checksum, pkg.sealed_at)
```

---

## 11. Retention Policies

```python
from capabilities.common.audl.models import RetentionPolicyCreate, RetentionAction, AuditEventType

policy = await svc.create_retention_policy(RetentionPolicyCreate(
	tenant_id          = "acme-corp",
	name               = "7-Year Financial Retention",
	event_types        = [AuditEventType.DATA_UPDATE],
	data_classifications = ["financial"],
	retain_days        = 2555,        # 7 years
	archive_after_days = 365,
	action_on_expiry   = RetentionAction.ARCHIVE,
))

# Run enforcement
summary = await svc.retention_enforcement()
print(summary["archived"], summary["skipped_legal_hold"])
```

---

## 12. SIEM Integration

```python
# Subscribe to real-time stream
async def consume_siem():
	async for event in svc.real_time_siem_stream(risk_threshold=0.5):
		# Forward to Splunk / Elastic / Panther
		print(event.event_id, event.risk_score)

# Or push a stored event manually
result = await svc.realtime_siem_push(event_id)
```

---

## 13. Event Delta (drift detection)

```python
# Capture snapshot at T0
snapshot = (await svc.get_event(event_id)).model_dump(mode="json")

# ... time passes ...

# Check for drift
delta = await svc.event_delta(event_id, reference=snapshot)
if delta["has_drift"]:
	print("DRIFT DETECTED:", delta["changed"])
```

---

## 14. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | unset | NATS JetStream URL; enables event bus publishing |
| `APG_SIGNING_KEY_PATH` | unset | PEM private key for evidence package signatures |
| `APG_AUDIT_LOG_LEVEL` | `INFO` | Python log level for `[audl]` messages |
| `DATABASE_URL` | unset | SQLAlchemy async URL; enables durable PostgreSQL storage |

---

## 15. Running Tests

```bash
# Fast unit tests (no DB required)
uv run pytest -vxs capabilities/common/audl/tests/ci/

# Full test suite
uv run pytest -vxs capabilities/common/audl/tests/

# Type checks
uv run pyright capabilities/common/audl/service.py capabilities/common/audl/models.py
```

---

## 16. Key Design Decisions

- **No mutation of core event fields.** `actor_id`, `resource_id`, `event_type`, `created_at`, `success` are immutable once written. Only `details` and `tags` blobs can be pseudonymised/masked.
- **Decimal for money.** `log_financial_event` quantizes to 8 dp using `ROUND_HALF_UP`. The stringified value is stored in `details["monetary_amount"]` and included in the checksum.
- **guard_tenant_id on every cross-boundary call.** The `guard_tenant_id` helper from `capabilities.common.reliability` raises `PermissionError` (not `AssertionError`) so callers can catch it explicitly.
- **Tabs, not spaces.** All Python source in this capability uses tabs per project standard.
- **Legal hold is append-only.** There is no programmatic path to hard-delete an event under legal hold. `rotate_legal_hold_ownership` records custodian handoffs without lifting the hold.
