# Employee Data Management Capability

`chr_employee_data_management` is the APG capability packet for governed employee
profiles, organization structure, personal information, emergency contacts,
employment history, skills, certifications, data-quality issues, and employee
data agents. It keeps the package boundary dependency-light so generated APG
applications can compose it immediately while production deployments attach
durable HRIS, identity, payroll, benefits, workflow, audit, notification, and
Bytewax topology through adapters.

## What It Provides

- Tenant-scoped departments with ownership and cost-center controls.
- Position lifecycle with department linkage, job level, headcount, and
  compensation-band review.
- Employee profile lifecycle with employee number, legal name, work email,
  department, position, manager, employment type, work mode, and hire date.
- Status transitions for leave, suspension, termination, alumni, and other
  supported workforce states.
- Personal-information records with country, effective date, and privacy basis.
- Emergency contact records.
- Employment history events with reason and approval guardrails for sensitive
  events.
- Skill and certification records with evidence and expiry guardrails.
- Data-quality issue workflow by domain, severity, owner, and employee.
- First-class employee data agents for Codex, Claude Code, OpenCode, and Pi
  review teams.
- APG UI route metadata, framework-neutral screen models, compact theme tokens,
  semantic metadata, package manifest, and release evidence.

## Package Layout

- `SPECIFICATION.md` defines records, workflows, rules, UI, events, adapter
  boundaries, and acceptance criteria.
- `PLAN.md` records the implementation and review plan for this lifecycle
  packet.
- `cap_spec.md` summarizes the current executable runtime contract.
- `capability_contract.py` exposes the executable APG contract and deterministic
  rule engine.
- `service.py` implements the dependency-light lifecycle service.
- `api.py` exposes composition helpers and legacy endpoint shims.
- `views.py` exposes framework-neutral screen models and legacy view shims.
- `app.py` exposes semantic model, component manifest, and self-test.
- `tests/test_package_contract.py` verifies the package contract, lifecycle,
  guardrails, API, views, and app surface.

## Runtime Lifecycle

1. Create departments with owner and cost center.
2. Create positions under same-tenant departments.
3. Create employee profiles under same-tenant departments and positions.
4. Record personal information, emergency contacts, and employment history.
5. Assign skills and certifications.
6. Track data-quality issues and route high-severity issues to owners.
7. Register employee data agents that inspect, prepare, and recommend within
   explicit human-approval boundaries.

## Usage

```python
from capabilities.hcm.chr.employee_data_management import EmployeeDataManagementService

service = EmployeeDataManagementService()

department = service.create_department(
	"department-hr",
	"tenant-a",
	"HR",
	"Human Resources",
	"hr-owner",
	"HR-000",
)
position = service.create_position(
	"position-hrbp",
	"tenant-a",
	"HRBP",
	"HR Business Partner",
	department["id"],
	"professional",
)
employee = service.create_employee(
	"employee-1",
	"tenant-a",
	"EMP-0001",
	"Amina",
	"Otieno",
	"amina.otieno@example.com",
	department["id"],
	position["id"],
	"2026-01-01",
	"manager-1",
)
service.record_emergency_contact(
	"contact-1",
	"tenant-a",
	employee["id"],
	"Sam Otieno",
	"Spouse",
	"+254700000000",
)
print(service.dashboard_summary("tenant-a"))
```

Generated APG applications can use `api.py`:

```python
from capabilities.hcm.chr.employee_data_management import api

status = api.capability_status("tenant-a")
records = api.list_records("employees", "tenant-a")
```

## Guardrails

- Tenant context is required.
- Write operations require policy context.
- Departments require code, name, owner, and cost center.
- Positions require code, title, same-tenant department, job level, and
  nonnegative authorized headcount.
- Compensation-band positions require review evidence.
- Employees require employee number, first name, last name, valid work email,
  same-tenant department, same-tenant position, manager for non-executives, hire
  date, supported employment type, and supported work mode.
- Sensitive status changes require review.
- Personal information requires employee, country, effective date, and privacy
  basis.
- Emergency contacts require employee, name, relationship, and phone.
- Sensitive employment-history events require reason; termination requires
  approval.
- Expert and master skills require evidence.
- Expiring certifications require expiry date and supported status.
- Data-quality issues require supported domain, severity, and owner for high
  severity.
- Employee batches and events require Bytewax metadata.
- Employee agents must use supported runtimes and roles.
- Privileged employee-agent actions require recorded human approval.

## Integration Boundary

This package does not start a live HRIS or identity workflow by default.
Production deployments should bind these concerns through adapters:

- identity, authorization, and tenant policy;
- audit vault and event replication;
- payroll, benefits, workforce planning, and onboarding systems;
- document stores for contracts and evidence;
- privacy policy and data retention engines;
- notification and workflow routing;
- durable Bytewax topology and event sinks;
- AI-agent runtime orchestration.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/hcm/chr/employee_data_management/__init__.py capabilities/hcm/chr/employee_data_management/capability_contract.py capabilities/hcm/chr/employee_data_management/service.py capabilities/hcm/chr/employee_data_management/api.py capabilities/hcm/chr/employee_data_management/views.py capabilities/hcm/chr/employee_data_management/app.py capabilities/hcm/chr/employee_data_management/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/hcm/chr/employee_data_management/tests/test_package_contract.py
./.venv/bin/python capabilities/hcm/chr/employee_data_management/app.py
./.venv/bin/apg capabilities inspect chr_employee_data_management --json
./.venv/bin/apg capabilities publish-plan capabilities/hcm/chr/employee_data_management --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/hcm/chr/employee_data_management --json
```

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements — chr_employee_data_management
- **I2.** Async-First Service Layer
- **I3.** Pydantic v2 Input/Output Models
- **I4.** Pluggable Persistence Adapter
- **I5.** Event-Driven Audit Bus
- **I6.** Structured Observability (OpenTelemetry)
- **I7.** Row-Level Multi-Tenancy Enforcement
- **I8.** GDPR / Data-Residency Controls
- **I9.** Position Vacancy Tracking
- **I10.** Leave Balance Engine
- **I11.** Onboarding Workflow Orchestration
- **I12.** Payroll Run Aggregation
- **I13.** Org Chart Flattened Search
- **I14.** Headcount Budget vs. Actual
- **I15.** Contract Expiry Alerting

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
