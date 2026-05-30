# Employee Data Management Runtime Summary

`chr_employee_data_management` is a package-backed APG capability for employee
master data, organization structure, sensitive profile data, talent profile
records, data-quality governance, and employee-data agent review.

## Runtime Contract

- Capability id: `chr_employee_data_management`
- Display name: `Employee Data Management`
- Version: `2.1.0`
- Target: `python`
- Entrypoint: `app.py`
- Service: `service.py`
- API helpers: `api.py`
- View models: `views.py`
- Stream processor: `bytewax`
- Event stream: `apg.hcm.chr.employee.lifecycle`
- Theme: `employee_data_control`

## Provides

- `employee_profile_lifecycle`
- `employee_identity_registry`
- `department_lifecycle`
- `position_lifecycle`
- `employment_history_lifecycle`
- `employee_skill_lifecycle`
- `employee_certification_lifecycle`
- `employee_contact_lifecycle`
- `employee_data_quality_workflow`
- `employee_dashboard_service`
- `employee_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `workflow`
- `document_management`
- `identity_access`
- `privacy_policy`

## Screens

The package exposes dashboard, employees, departments, positions, personal
information, contacts, history, skills, certifications, data quality, agents,
and settings routes under `/hcm/employees`.

## Guardrail Coverage

The deterministic rule engine covers tenant context, write policy attachment,
department completeness, position completeness, compensation-band review,
employee identity, email format, organization membership, manager requirement,
supported employment type and work mode, sensitive status changes, personal
privacy basis, emergency contacts, employment history, skills, certifications,
data-quality ownership, Bytewax routing, employee-agent runtime and role, and
privileged employee-agent approval.

## Adapter Boundary

The package intentionally avoids live HRIS, payroll, benefits, identity,
workflow, audit, notification, data vault, and AI-runtime imports at the top
level. Production applications attach those systems through APG adapters while
using this packet for the executable lifecycle contract and composition surface.
