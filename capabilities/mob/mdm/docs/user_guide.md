# Mobile Device Management — User Guide

**Capability ID**: `mob_mdm` | **Domain**: `mob` | **Version**: `2.0.0`
**© 2025 Datacraft** | www.datacraft.co.ke

---

## Description

The Mobile Device Management (MDM) capability provides an enterprise-grade device lifecycle runtime. It covers device enrolment across multiple platforms and methods; deterministic policy creation, activation, and assignment; continuous compliance evaluation with automatic alert generation; silent app distribution; remote wipe with mandatory dual approval; MDM configuration profile deployment; device inventory registry; device groups for bulk policy targeting; certificate lifecycle tracking; and a structured audit log — all tenant-scoped.

---

## Installation

```bash
pip install apg-mob-mdm
```

---

## Quick Start

```python
import asyncio
from apg_mob_mdm import MobileDeviceManagementService
from apg_mob_mdm.models import DeviceEnrolmentCreate

svc = MobileDeviceManagementService()

async def main():
    device = await svc.enrol_device(DeviceEnrolmentCreate(
        tenant_id="acme",
        serial_number="SN-001",
        device_type="laptop",
        os_platform="macos",
        os_version="14.5",
        ownership_type="corporate",
        enrolment_method="dep",
        approval_reference="TICKET-001",
        created_by="admin",
    ))
    print(device.id, device.enrolment_state)

asyncio.run(main())
```

---

## Service Methods Reference

### Contract Helpers

| Method | Signature | Description |
|--------|-----------|-------------|
| `describe` | `(tenant_id) -> dict` | Return full capability contract |
| `evaluate` | `(context) -> dict` | Evaluate business rules against context dict |

---

### Device Enrolment

#### `enrol_device`
```python
device = await svc.enrol_device(DeviceEnrolmentCreate(
    tenant_id="acme",
    serial_number="SN-001",
    device_type="laptop",        # laptop | smartphone | tablet | desktop | kiosk | iot
    os_platform="macos",         # macos | ios | android | windows | linux | chromeos
    os_version="14.5",
    ownership_type="corporate",  # corporate | byod | copo | loaner
    enrolment_method="dep",      # dep | zero_touch | qr | nfc | manual | bulk_csv
    approval_reference="TICKET-001",
    created_by="admin",
))
```

#### `bulk_enrol_devices`
Enrol many devices in one call. Duplicate serials are skipped rather than errored.
```python
result = await svc.bulk_enrol_devices([
    DeviceEnrolmentCreate(tenant_id="acme", serial_number="SN-A", ...),
    DeviceEnrolmentCreate(tenant_id="acme", serial_number="SN-B", ...),
])
# result: {total, succeeded, failed, duplicates, results: [{serial_number, status, device_id?, error?}]}
```

#### `get_device`
```python
device = await svc.get_device(tenant_id="acme", device_id="<uuid>")
```

#### `list_devices`
```python
devices = await svc.list_devices(
    tenant_id="acme",
    os_platform="ios",           # optional
    enrolment_state="enrolled",  # optional
    ownership_type="byod",       # optional
)
```

#### `update_device`
```python
device = await svc.update_device("acme", device_id, DeviceUpdate(
    os_version="17.0",
    location="Nairobi HQ",
    updated_by="admin",
))
```

#### `unenrol_device` / `suspend_device`
```python
await svc.unenrol_device("acme", device_id, unenrolled_by="admin")
await svc.suspend_device("acme", device_id, suspended_by="security")
```

#### `lock_device` / `unlock_device`
```python
await svc.lock_device("acme", device_id, reason="reported_stolen", locked_by="soc")
await svc.unlock_device("acme", device_id, pin="123456", unlocked_by="admin")
```

---

### Device Health Score

Returns a weighted 0–100 score across compliance, enrolment state, last-seen recency, and open alerts.

```python
score = await svc.get_device_health_score("acme", device_id)
# {health_score: 80, grade: "A", components: {compliance, enrolment, last_seen_recency, open_alerts}}
```

Grade thresholds: A ≥ 80 | B ≥ 60 | C ≥ 40 | F < 40.

---

### Device Groups

Groups allow bulk policy targeting without per-device assignments.

```python
# Create group
group = await svc.create_device_group(
    tenant_id="acme", name="EMEA Laptops",
    description="All EMEA corporate laptops", created_by="admin",
)

# Add devices
await svc.add_device_to_group("acme", group["id"], device_id, added_by="admin")

# Assign policy to all group members
result = await svc.assign_policy_to_group(
    tenant_id="acme",
    group_id=group["id"],
    policy_id=policy.id,
    assigned_by="admin",
    created_by="admin",
)
# result: {device_count, results: [{device_id, assignment_id?, status, error?}]}
```

---

### Policies

#### Create and activate
```python
policy = await svc.create_policy(PolicyCreate(
    tenant_id="acme",
    name="Corporate Security Baseline",
    policy_type="security",      # security | network | app | kiosk | update | custom
    configuration={"min_pin_length": 6, "screen_lock_timeout": 300},
    platform_targets=["ios", "android"],
    created_by="admin",
))

policy = await svc.activate_policy(
    "acme", policy.id,
    approval_reference="CAB-2025-001",
    activated_by="ciso",
)
```

#### Assign to device
```python
assignment = await svc.assign_policy(PolicyAssignmentCreate(
    tenant_id="acme",
    policy_id=policy.id,
    device_id=device.id,
    assigned_by="admin",
    created_by="admin",
))
```

---

### Compliance Evaluation

```python
record = await svc.evaluate_compliance(ComplianceEvaluationCreate(
    tenant_id="acme",
    device_id=device.id,
    evaluator_id="compliance-engine",
    findings=[
        {"rule": "screen_lock", "severity": "high", "detail": "screen lock disabled"},
    ],
    created_by="system",
))
# Non-compliant findings auto-raise a high-severity MDM alert
```

---

### App Distribution

```python
dist = await svc.distribute_app(AppDistributionCreate(
    tenant_id="acme",
    app_bundle_id="com.corp.vpn",
    app_name="Corporate VPN",
    app_version="3.1.0",
    device_id=device.id,
    distribution_type="required",  # required | available | blocked
    silent_install=True,
    created_by="admin",
))
```

---

### Remote Wipe

Wipe requires dual approval. Cancel before execution if issued in error.

```python
# Request
wipe = await svc.request_wipe(WipeRequestCreate(
    tenant_id="acme",
    device_id=device.id,
    wipe_type="full_wipe",          # full_wipe | selective_wipe | corporate_wipe | factory_reset
    approval_reference="MGMT-001",
    second_approval_reference="SEC-007",
    justification="device_reported_stolen",
    requested_by="admin",
    created_by="admin",
))

# Cancel (while pending)
wipe = await svc.cancel_wipe(
    "acme", wipe.id,
    cancelled_by="admin",
    reason="device_recovered",
)

# Execute (after dual approval confirmed)
wipe = await svc.execute_wipe("acme", wipe.id, executed_by="soc")
```

---

### MDM Profiles

```python
profile = await svc.create_profile(MdmProfileCreate(
    tenant_id="acme",
    name="Corp WiFi",
    profile_type="wifi",      # config | certificate | vpn | wifi | email | custom
    platform="ios",
    payload={"ssid": "CorpNet", "security": "WPA2-Enterprise"},
    created_by="admin",
))

profile = await svc.deploy_profile("acme", profile.id, device.id, deployed_by="admin")
```

---

### Certificate Lifecycle

Track certificate expiry and receive automatic alerts at 30 days and 0 days.

```python
from datetime import datetime, timedelta

cert = await svc.track_certificate(
    tenant_id="acme",
    device_id=device.id,
    cert_serial="ABC123",
    issuer="Corp CA",
    not_before=datetime.utcnow(),
    not_after=datetime.utcnow() + timedelta(days=25),   # triggers high alert
    tracked_by="pki-bot",
)

certs = await svc.list_certificates("acme", device_id=device.id, status="valid")
```

Alert rules:
- `days_remaining <= 7` → severity `high`
- `7 < days_remaining <= 30` → severity `medium`
- `days_remaining <= 0` → severity `critical`

---

### Alerts

```python
alerts = await svc.list_alerts("acme", resolved=False)
alert = await svc.resolve_alert("acme", alert_id, resolved_by="soc")
```

---

### Audit Log

```python
entries = await svc.get_audit_log(
    tenant_id="acme",
    entity_id=device.id,    # optional — filter by entity
    event_type="wipe_completed",  # optional
    limit=50,               # 1–1000, newest-first
)
```

---

### Analytics and Reporting

```python
# Dashboard KPI card
kpi = await svc.mdm_kpi_summary("acme")

# Full dashboard
dash = await svc.dashboard_summary("acme")

# Analytics by period
analytics = await svc.mdm_analytics("acme", period="2025-Q2")

# Compliance report
report = await svc.mdm_compliance_report("acme", period="2025-06")
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mob-mdm/dashboard` | `mob_mdm:view` | Overview |
| `/mob-mdm/devices` | `mob_mdm:devices:list` | Devices |
| `/mob-mdm/devices/<device_id>` | `mob_mdm:devices:view` | Devices |
| `/mob-mdm/devices/<device_id>/health` | `mob_mdm:devices:view` | Devices |
| `/mob-mdm/enrolment` | `mob_mdm:enrolment:manage` | Devices |
| `/mob-mdm/groups` | `mob_mdm:groups:manage` | Devices |
| `/mob-mdm/policies` | `mob_mdm:policies:list` | Policies |
| `/mob-mdm/policies/<policy_id>` | `mob_mdm:policies:view` | Policies |
| `/mob-mdm/compliance` | `mob_mdm:compliance:view` | Compliance |
| `/mob-mdm/compliance/<device_id>` | `mob_mdm:compliance:view` | Compliance |
| `/mob-mdm/certificates` | `mob_mdm:certs:view` | Security |
| `/mob-mdm/audit-log` | `mob_mdm:audit:view` | Security |

---

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed `MOB_MDM_`.

| Key | Default | Description |
|-----|---------|-------------|
| `devices.approval_required_for_enrolment` | `true` | Require approval before device enrolment |
| `policies.approval_required` | `true` | Require approval before policy activation |
| `remote_actions.wipe_requires_dual_approval` | `true` | Two approvers for any wipe |
| `compliance.evaluation_interval_minutes` | `60` | How often compliance is re-evaluated |
| `compliance.grace_period_hours` | `24` | Grace before non-compliant blocking |
| `certificates.expiry_warning_days` | `30` | Days before expiry to raise alert |
| `governance.cross_tenant_access_denied` | `true` | Prevent cross-tenant data access |

---

## Interoperability

```apg
use mob_mdm;
```

`mob_mdm` integrates with:
- `auth` — token validation and caller identity
- `audl` — receives every state-changing event
- `ntfy` — alert dispatch (push / email / sms)
- `comp` — regulatory compliance framework mapping
- `wflo` — multi-stage approval workflows
- `mqeb` — event streaming to fleet health dashboards
- `moni` — operational monitoring
- `mob_map` — biometric enrolment gating via `device_enrolled` status

---

## Further Reading

- `service.py` — Business logic (all async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `capability_contract.py` — Business rules and supported enumerations
- `README.md` — Quick reference and composability notes
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 architectural improvements
