# Personnel & HR Management — User Guide

**Capability ID**: `government_per` | **Domain**: `government` | **Version**: `1.1.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero | **Contact**: nyimbi@gmail.com

---

## Description

`government_per` is the APG capability for civil service human resource management, payroll integration, performance management, and disciplinary proceedings. It also provides the original Permits Management workflows (building, environmental, occupation permits) that share this namespace.

The capability enforces statutory compliance with:
- **Kenya Employment Act 2007** (disciplinary due process, leave entitlements)
- **Public Service Commission Regulations** (appointment types, probation, confirmation)
- **Pensions Act Cap 189** and **NPF/LGPS schemes** (terminal benefits)
- **Kenya Revenue Authority PAYE** graduated bands (2024 Finance Act)
- **NHIF, NSSF, and Housing Levy** statutory deductions

---

## Installation

```bash
pip install apg-government-per
```

Or within the APG monorepo:

```bash
cd capabilities/government/per
uv sync
```

---

## Quick Start

```python
from apg_government_per.service import GovernmentPerService
import asyncio

svc = GovernmentPerService(tenant_id="nairobi_county", actor_id="hr_officer_001")

async def demo():
    # Record a new permanent appointment
    appointment = await svc.record_employee_appointment(
        employee_id="EMP-00142",
        post_id="POST-D3-005",
        appointment_type="permanent",
        department_id="DEPT-FINANCE",
        grade="D3",
        effective_date="2026-01-01",
        salary_step=1,
        probation_months=6,
    )
    print(appointment)

asyncio.run(demo())
```

---

## Core HR Workflows

### 1. Employee Appointment

Create a civil service appointment. Supported types: `permanent`, `contract`, `secondment`, `acting`, `internship`.

```python
appointment = await svc.record_employee_appointment(
    employee_id="EMP-00142",
    post_id="POST-D3-005",
    appointment_type="permanent",
    department_id="DEPT-FINANCE",
    grade="D3",
    effective_date="2026-01-01",
)
# Returns appointment dict with probation_end computed automatically.
# NATS event: apg.government.per.appointment.created
```

**Establishment control**: if approved headcount ceiling is exceeded, the appointment is blocked and an `EstablishmentBreachAttempted` event is emitted.

---

### 2. Payroll Processing

Initiate a payroll run for one or more departments. Bytewax processes the run on NATS:

```python
run = await svc.process_payroll_run(
    pay_period="2026-05",
    department_ids=["DEPT-FINANCE", "DEPT-WORKS"],
    run_type="regular",   # regular | supplementary | off_cycle | final_settlement
)
# NATS event: apg.government.per.payroll.run
# Bytewax dataflow emits PayslipGenerated per employee
```

#### Statutory Deductions

Compute PAYE, NHIF, NSSF, and Housing Levy for a given gross pay:

```python
deductions = await svc.compute_statutory_deductions(
    employee_id="EMP-00142",
    gross_pay=75_000.0,
    jurisdiction="kenya_central",
)
# Returns: paye, nhif, nssf, housing_levy_employee, net_pay
# NATS event: apg.government.per.payroll.netpay
```

Supported jurisdictions: `kenya_central` (more via `conf` capability).

---

### 3. Grade Increment Processing

Increment an employee's salary step based on annual appraisal:

```python
increment = await svc.process_grade_increment(
    employee_id="EMP-00142",
    current_grade="D3",
    current_step=4,
    appraisal_score=3.8,    # out of 5.0; threshold is 3.0
    appraisal_period="2025-2026",
)
# approved=True → new_step=5
# approved=False → new_step unchanged, withhold_reason set
# requires_treasury_concurrence=True if new_gross > KES 50,000
# NATS event: apg.government.per.payroll.increment
```

---

### 4. Disciplinary Case Management

Open a disciplinary case. The FSM enforces due-process timelines:

```python
case = await svc.open_disciplinary_case(
    employee_id="EMP-00142",
    allegation="Insubordination and failure to follow lawful instructions",
    complainant_id="SUPERVISOR-007",
    evidence_refs=["EVD-001", "EVD-002"],
    severity="major",   # minor | major | gross_misconduct
)
# Returns state="allegation_raised", investigation_deadline, appeal_window_days
# NATS event: apg.government.per.disciplinary.opened
```

**FSM States**: `allegation_raised → investigation → hearing_scheduled → hearing_held → outcome_issued → appeal_window → closed`

**Due-process timelines by severity**:

| Severity | Investigation (days) | Hearing Notice (days) | Appeal Window (days) |
|---|---|---|---|
| `minor` | 14 | 7 | 14 |
| `major` | 30 | 14 | 21 |
| `gross_misconduct` | 30 | 14 | 21 |

---

### 5. Leave Balance Computation

```python
balance = await svc.compute_leave_balance(
    employee_id="EMP-00142",
    leave_type="annual",   # annual | sick | maternity | paternity | compassionate | study | unpaid
    as_at_date="2026-06-01",
)
# Annual leave accrues at 1.75 days/month (21 days/year), carry-over cap 10 days.
# NATS event processed by Bytewax accrual dataflow.
```

---

### 6. Terminal Benefits Calculation

```python
benefits = await svc.calculate_terminal_benefits(
    employee_id="EMP-00142",
    years_of_service=30.5,
    final_basic_salary=85_000.0,
    scheme="cap189",        # cap189 | npf | lgps
    age_at_retirement=60,
)
# Returns: annual_pension, monthly_pension, gratuity, commutation_lump_sum,
#          reduced_monthly_pension_if_commuted
```

**Scheme comparison**:

| Scheme | Coverage | Gratuity Factor |
|---|---|---|
| `cap189` | Kenya central government | 31/480 |
| `npf` | National Pension Fund members | 25/480 |
| `lgps` | Local government officers | 3/80 |

---

### 7. Cross-Agency Secondment

```python
secondment = await svc.record_secondment(
    employee_id="EMP-00142",
    origin_agency_id="AGENCY-NLC",
    destination_agency_id="AGENCY-KNBS",
    effective_date="2026-07-01",
    reversion_date="2027-06-30",
    payroll_responsibility="destination",  # origin | destination | split
)
# Bytewax schedules automatic reversion on reversion_date.
# NATS event: apg.government.per.secondment.created
```

---

### 8. Workforce Headcount Report

```python
report = await svc.workforce_headcount_report(
    department_id="DEPT-FINANCE",   # optional
    grade="D3",                      # optional
)
# Returns: total_active_appointments, establishment_ceiling, over_establishment flag
```

---

### 9. Performance Appraisal Summary

```python
summary = await svc.performance_appraisal_summary(
    department_id="DEPT-FINANCE",
    appraisal_period="2025-2026",
)
# Returns rating band distribution: outstanding / exceeds / meets / below / unsatisfactory
# NATS event: apg.government.per.performance.summary (feeds intel attrition ML)
```

---

## Permit Management Workflows

The original permits management workflows remain fully supported.

### Submit a Permit Application

```python
app = svc.apply_permit(
    applicant_id="CONTRACTOR-001",
    permit_type="building",
    property_details={"address": "Plot 123, Westlands, Nairobi"},
    documents=["structural_drawings.pdf", "site_plan.pdf"],
)
```

### Issue a Permit

```python
permit = svc.issue_permit(
    permit_id="PER-001",
    tenant_id="nairobi_county",
    application_id=app["id"],
    permit_type="building",
    permit_number="BLD-2026-00123",
    holder_id="CONTRACTOR-001",
    site_reference="Plot 123",
    issued_date="2026-06-01",
    expiry_date="2027-06-01",
    evidence_reference="approved_drawings_v2.pdf",
)
```

### Schedule & Record an Inspection

```python
insp = await svc.inspection_scheduling(
    permit_id=permit["permit_id"],
    inspection_type="structural",
    scheduled_date="2026-08-15",
    inspector_id="INSPECTOR-007",
)

# After inspection:
outcome = svc.record_inspection_outcome(
    inspection_id=insp["inspection_id"],
    tenant_id="nairobi_county",
    outcome="pass",
    findings="All structural elements conform to approved drawings.",
)
```

---

## Streaming Architecture

All HR and permit events are published to **NATS JetStream** and processed by **Bytewax** dataflows. This provides:
- Sub-second payroll GL posting (vs. 3–5 day batch cycles)
- Real-time leave balance accrual
- Date-scheduled secondment reversion (no manual intervention)
- Event-sourced audit trail with full replay capability

To observe the event stream locally:

```bash
# Subscribe to all HR events
nats sub "apg.government.per.>"

# Payroll events only
nats sub "apg.government.per.payroll.*"
```

---

## Configuration Reference

Set via `conf` capability or `GOVERNMENT_PER_*` environment variables:

| Key | Default | Description |
|---|---|---|
| `GOVERNMENT_PER_ESTABLISHMENT_CEILING` | `500` | Max appointments before over-establishment |
| `GOVERNMENT_PER_INCREMENT_THRESHOLD` | `3.0` | Minimum appraisal score for increment |
| `GOVERNMENT_PER_TREASURY_THRESHOLD` | `50000` | Gross salary flagging Treasury concurrence |
| `GOVERNMENT_PER_DISCIPLINARY_APPEAL_DAYS` | `14` | Default disciplinary appeal window |
| `GOVERNMENT_PER_PAYROLL_JURISDICTION` | `kenya_central` | Statutory deduction ruleset |
| `GOVERNMENT_PER_DEFAULT_PENSION_SCHEME` | `cap189` | Default pension scheme |
| `OLLAMA_BASE_URL` | — | Enables ML-powered approval prediction |

---

## Permissions Reference

| Permission | Grants |
|---|---|
| `government_per:appoint` | Create and manage appointments |
| `government_per:payroll` | Initiate payroll runs, compute deductions |
| `government_per:increment` | Process salary increments |
| `government_per:disciplinary` | Open and manage disciplinary cases |
| `government_per:leave` | Compute leave balances |
| `government_per:benefits` | Calculate terminal benefits |
| `government_per:secondment` | Create and manage secondments |
| `government_per:headcount` | View headcount reports |
| `government_per:appraise` | Generate appraisal summaries |
| `government_per:apply` | Submit permit applications |
| `government_per:permits` | View permit register |
| `government_per:inspect` | Schedule and record inspections |
| `government_per:compliance` | Record compliance assessments |
| `government_per:enforce` | Initiate enforcement actions |
| `government_per:view` | Read-only access to all resources |

---

## Testing

```bash
# Run CI test suite
uv run pytest -vxs tests/ci

# Type checking
uv run pyright
```

All tests in `tests/ci/` use real objects and pytest fixtures — no mocks except LLM calls.

---

## Further Reading

- `service.py` — Business logic (HR + permits)
- `models.py` — In-memory data models
- `capability_contract.py` — Policy rules and supported enumerations
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised enhancement proposals
- `README.md` — Quick reference
