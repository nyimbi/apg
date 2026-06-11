# Remote Workforce

**Capability ID**: `mob_rwf` | **Domain**: `mob` | **Version**: `1.0.0`

## Description

The Remote Workforce (RWF) capability provides a complete remote and hybrid work governance runtime. It manages remote work policy authoring, activation, and employee acknowledgment; VPN access provisioning with MFA enforcement and split-tunneling prevention; consent-based productivity tracking; equipment requisition with per-employee limits; digital onboarding orchestration with step tracking; remote compliance checks; and remote incident management — all governed by tenant-scoped deterministic rules with full audit trails.

## Installation

```bash
pip install apg-mob-rwf
```

## Provides

- `remote_work_policy_management`
- `vpn_access_governance`
- `productivity_tracking_workflow`
- `equipment_requisition_workflow`
- `digital_onboarding_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mob-rwf/dashboard` | `mob_rwf:view` | Overview |
| `/mob-rwf/policies` | `mob_rwf:policies:list` | Policies |
| `/mob-rwf/policies/<policy_id>` | `mob_rwf:policies:view` | Policies |
| `/mob-rwf/policies/<policy_id>/acknowledge` | `mob_rwf:policies:acknowledge` | Policies |
| `/mob-rwf/vpn` | `mob_rwf:vpn:list` | VPN |
| `/mob-rwf/vpn/provision` | `mob_rwf:vpn:provision` | VPN |
| `/mob-rwf/productivity` | `mob_rwf:productivity:view` | Productivity |
| `/mob-rwf/productivity/<employee_id>` | `mob_rwf:productivity:view` | Productivity |

## Key Service Methods

### Contract & Rules
- `describe()` — return the full capability contract for a tenant
- `evaluate(context)` — evaluate domain rules against an arbitrary operation context

### Work Policies
- `create_work_policy(payload)` — author a new remote work policy in `draft` state
- `activate_work_policy(tenant_id, policy_id, approval_reference, activated_by)` — publish after approval
- `update_work_policy(tenant_id, policy_id, payload)` — edit; version counter auto-increments
- `list_work_policies(tenant_id, policy_type?, state?)` — filtered listing
- `get_work_policy(tenant_id, policy_id)` — retrieve single policy
- `acknowledge_policy(payload)` — record employee acknowledgment with IP/device trail
- `list_acknowledgments(tenant_id, policy_id?, employee_id?)` — audit acknowledgments

### VPN Access
- `provision_vpn(payload)` — provision VPN; requires MFA, rejects split-tunneling
- `revoke_vpn(tenant_id, access_id, reason, revoked_by)` — immediate revocation
- `start_vpn_session(tenant_id, access_id, client_ip?)` — open session
- `end_vpn_session(tenant_id, session_id, bytes_in, bytes_out)` — close and record transfer stats
- `list_vpn_access(tenant_id, employee_id?, state?)` — filter VPN records

### Productivity
- `record_productivity_metric(payload)` — consent-gated metric recording
- `get_productivity_summary(tenant_id, employee_id)` — per-metric averages
- `list_productivity_metrics(tenant_id, employee_id?, metric_type?)` — raw listing

### Equipment Requisition
- `request_equipment(payload)` — submit; enforces per-employee item limit (default 5)
- `approve_equipment(tenant_id, req_id, approval_reference, approved_by)` — approval increments count
- `ship_equipment(tenant_id, req_id, asset_tag)` — mark dispatched
- `deliver_equipment(tenant_id, req_id)` — confirm receipt
- `return_equipment(tenant_id, req_id, returned_by)` — decrements employee count
- `list_equipment(tenant_id, employee_id?, state?)` — full lifecycle view

### Digital Onboarding
- `start_onboarding(payload)` — manager-approval-gated; creates all standard pending steps
- `complete_onboarding_step(payload)` — mark step done; auto-transitions to `completed` when all clear
- `list_onboarding_records(tenant_id, state?, employee_id?)` — progress overview
- `get_onboarding_record(tenant_id, record_id)` — detail view
- `bulk_start_onboarding(tenant_id, payloads)` — onboard many employees; returns partial-failure report

### Compliance & Incidents
- `record_compliance_check(payload)` — log result; auto-schedules next due date (30 days)
- `list_compliance_checks(tenant_id, employee_id?, check_type?, result?)` — compliance audit
- `raise_incident(payload)` — open incident with severity classification
- `resolve_incident(tenant_id, incident_id, resolution_notes, resolved_by)` — close with notes
- `list_incidents(tenant_id, employee_id?, incident_type?, state?)` — incident tracker

### Field Shifts
- `check_in_shift(tenant_id, employee_id, lat?, lon?, site_id?)` — GPS-verified shift start
- `check_out_shift(tenant_id, shift_id, lat?, lon?)` — auto-computes duration in minutes
- `get_active_shifts(tenant_id, site_id?)` — live site headcount; filter by work site

### Field Tasks
- `assign_field_task(tenant_id, employee_id, task_title, task_type, due_at?, geo_region?, priority?, dependencies?)` — create geo-scoped task with dependency list
- `complete_field_task(tenant_id, task_id, outcome_notes?, completed_by?)` — close task
- `list_field_tasks(tenant_id, employee_id?, state?, geo_region?)` — multi-filter listing

### Field Certifications
- `record_certification(tenant_id, employee_id, cert_type, issuer, issued_at, expiry_date?)` — log licence/cert; expired certs auto-raise a compliance incident
- `list_certifications(tenant_id, employee_id?, cert_type?, state?)` — validity-aware listing

### Route Optimisation
- `optimize_field_route(tenant_id, waypoints, start_point?)` — pure-Python nearest-neighbour + 2-opt TSP; returns ordered route with total Euclidean distance and algorithm label

### Offline Sync
- `enqueue_offline_operation(tenant_id, employee_id, operation, payload, logical_clock?)` — buffer operations captured without connectivity, tagged with a logical clock for causal ordering
- `sync_offline_queue(tenant_id, employee_id)` — replay buffered ops in causal order; returns `{succeeded, failed, errors}` per operation

### Audit & Observability
- `export_audit_log(tenant_id, event_type?, since?, until?, format?)` — filtered export with per-batch SHA-256 checksum for tamper detection
- `health_check(tenant_id?)` — structured `{status, checks, audit_event_count}` suitable for Kubernetes probes

### Analytics & Dashboard
- `dashboard_summary(tenant_id)` — top-level KPI counts
- `rwf_analytics(tenant_id, period)` — period-scoped aggregate metrics
- `productivity_report(tenant_id, employee_id, period)` — per-employee productivity narrative
- `security_compliance_remote(tenant_id, employee_id)` — security posture snapshot with issue list

## Field Worker Workflow (end-to-end example)

```python
from capabilities.mob.rwf.service import RemoteWorkforceService

svc = RemoteWorkforceService()

# 1. Worker checks in at a site
shift = await svc.check_in_shift("acme", "emp-42", lat=-1.286389, lon=36.817223, site_id="nbi-hq")

# 2. Dispatcher assigns an inspection task
task = await svc.assign_field_task(
    "acme", "emp-42", "Inspect meter bank 7", "inspection",
    geo_region="nairobi-central", priority="high"
)

# 3. Worker completes the task
await svc.complete_field_task("acme", task["task_id"], outcome_notes="All meters functional")

# 4. Optimise remaining waypoints before driving to next site
route = await svc.optimize_field_route("acme", [
    {"lat": -1.29, "lon": 36.82, "label": "Site A"},
    {"lat": -1.31, "lon": 36.85, "label": "Site B"},
    {"lat": -1.27, "lon": 36.79, "label": "Site C"},
])

# 5. Check out
await svc.check_out_shift("acme", shift["shift_id"])

# 6. Export audit trail at end of day
log = await svc.export_audit_log("acme", since=datetime(2026, 6, 11))
```

## Offline Workflow

When a field worker loses connectivity, operations are buffered locally:

```python
await svc.enqueue_offline_operation(
    "acme", "emp-42",
    operation="complete_field_task",
    payload={"tenant_id": "acme", "task_id": "...", "outcome_notes": "done offline"},
    logical_clock=5,
)
# ... regain connectivity ...
result = await svc.sync_offline_queue("acme", "emp-42")
# result = {"succeeded": 1, "failed": 0, "errors": []}
```

## Interoperability

`mob_rwf` integrates with other APG capabilities through the composition engine:

```apg
use mob_rwf;
```

Compose with:
- `mob_mdm` — device enrolment gates VPN provisioning
- `mob_map` — biometric enrollment as an onboarding step
- `fint` — equipment asset depreciation and write-downs
- `wflo` — multi-stage approval for policies and equipment
- `schd` — compliance check cadence scheduling
- `ntfy` — incident and SLA-breach push alerts
- `nlpc` — semantic search over work policy content
- `mqeb` — event streaming to `moni` dashboards

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or `MOB_RWF_*` env vars.

| Key | Default | Description |
|-----|---------|-------------|
| `vpn.mfa_required` | `true` | MFA enforced before VPN provisioning |
| `vpn.split_tunneling_allowed` | `false` | Split-tunneling always denied |
| `vpn.max_session_hours` | `12` | Max VPN session length |
| `productivity.tracking_consent_required` | `true` | Consent gate for metric recording |
| `equipment.max_items_per_employee` | `5` | Per-worker equipment ceiling |
| `compliance.check_interval_days` | `30` | Auto-scheduled next-due interval |
| `governance.onboarding_requires_manager_approval` | `true` | Manager gates onboarding start |

## Further Reading

- `service.py` — Business logic implementation (all async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference and API surface
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvement proposals with rationale
