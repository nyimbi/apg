# Employee Data Management Specification

## Purpose

`chr_employee_data_management` gives APG applications a governed employee master
data capability for profiles, departments, positions, personal information,
emergency contacts, employment history, skills, certifications, data-quality
workflows, and employee-data agent review. The package must run without external
services and make every production integration boundary explicit.

## Capability Identity

- Capability id: `chr_employee_data_management`
- Display name: `Employee Data Management`
- Version: `2.1.0`
- Target: `python`
- Profile: `capability`
- Event stream: `apg.hcm.chr.employee.lifecycle`
- Stream processor: `bytewax`
- Theme: `employee_data_control`

## Domain Records

### Department

Fields: `id`, `tenant_id`, `code`, `name`, `owner_id`, `cost_center`,
`parent_department_id`, `status`, and `created_at`.

### Position

Fields: `id`, `tenant_id`, `code`, `title`, `department_id`, `job_level`,
`authorized_headcount`, `compensation_band`, `reviewed_by`, `status`, and
`created_at`.

### Employee

Fields: `id`, `tenant_id`, `employee_number`, `first_name`, `last_name`,
`full_name`, `work_email`, `department_id`, `position_id`, `manager_id`,
`hire_date`, `employment_type`, `work_mode`, `executive`, `metadata`, `status`,
`created_at`, and `updated_at`.

Supported employment types are full time, part time, contractor, intern, and
temporary. Supported work modes are onsite, hybrid, remote, and field.

### Personal Information

Fields: `id`, `tenant_id`, `employee_id`, `country`, `effective_date`,
`privacy_basis`, `fields`, `status`, and `created_at`.

### Emergency Contact

Fields: `id`, `tenant_id`, `employee_id`, `name`, `relationship`, `phone`,
`status`, and `created_at`.

### Employment History

Fields: `id`, `tenant_id`, `employee_id`, `event_type`, `effective_date`,
`reason`, `approved_by`, `status`, and `created_at`.

### Skill

Fields: `id`, `tenant_id`, `employee_id`, `skill_name`, `level`, `evidence`,
`status`, and `created_at`. Supported levels are awareness, working,
practitioner, expert, and master.

### Certification

Fields: `id`, `tenant_id`, `employee_id`, `name`, `issuer`, `issued_on`,
`expires_on`, `status`, and `created_at`.

### Data-Quality Issue

Fields: `id`, `tenant_id`, `domain`, `severity`, `description`, `owner_id`,
`employee_id`, `status`, and `created_at`. Supported domains are identity,
employment, organization, skills, certifications, contacts, and privacy.

### Employee Agent

Fields: `id`, `tenant_id`, `name`, `runtime`, `role`, `scope`, `status`, and
`created_at`. Supported runtimes are Codex, Claude Code, OpenCode, and Pi.
Supported roles are profile steward, data-quality reviewer, organization design
reviewer, skills reviewer, compliance reviewer, and onboarding reviewer.

## Lifecycle Workflows

### Organization Structure

1. Create department with code, name, owner, and cost center.
2. Create positions under same-tenant departments.
3. Require review for positions carrying compensation-band metadata.
4. Keep organization records available for payroll, benefits, workflow, and
   identity adapters.

### Employee Profile

1. Create employee with identity, organization, manager, hire-date, employment
   type, and work-mode data.
2. Require a manager for non-executives.
3. Enforce supported employment type and work mode.
4. Change employee status with review for sensitive states.

### Sensitive Profile Data

1. Record personal information with country, effective date, and privacy basis.
2. Record emergency contacts with name, relationship, and phone.
3. Record employment history with event type and effective date.
4. Require reasons for sensitive history events and approval for termination.

### Talent Profile

1. Assign skills with supported proficiency levels.
2. Require evidence for expert and master skills.
3. Assign certifications with issuer, issue date, expiry rules, and supported
   status.

### Data Quality And Agents

1. Record data-quality issues by supported domain and severity.
2. Require owner for high and critical severity.
3. Register employee data agents with supported runtime and role.
4. Limit agent scope to inspection, preparation, validation, and recommendation.
5. Require human approval for privileged agent actions.

## Rule Engine

The deterministic rule engine returns:

- `decision`: allow, deny, or require_review;
- `matched_rules`: ordered matching rule names;
- `effects`: rule effects with reason and required action.

Rules cover tenant context, write policy attachment, department completeness,
position completeness, compensation-band review, employee identity, email
format, organization membership, manager requirement, status changes, personal
privacy basis, emergency contacts, employment history, skills, certifications,
data-quality ownership, Bytewax routing, agent runtime and role support, and
privileged-agent approval.

## UI Contract

The capability exposes APG screen metadata for dashboard, employees,
departments, positions, personal info, contacts, history, skills,
certifications, data quality, agents, and settings. `views.py` returns
framework-neutral screen models so generated Python applications can render the
capability without importing Flask-AppBuilder.

## Event Contract

Lifecycle events use stream `apg.hcm.chr.employee.lifecycle`, key `tenant_id`,
and processor `bytewax`. Events cover department creation, position creation,
employee creation, status changes, sensitive profile data, employment history,
skills, certifications, data-quality issues, and employee-agent registration.

## Acceptance Criteria

- Top-level imports must not require Flask, SQLAlchemy, databases, Redis, HRIS
  systems, or AI runtimes.
- `get_capability_contract()` must expose configuration, rules, UI, theme,
  streaming, provides, and requires.
- The service must enforce guardrails before state changes.
- API helpers, view models, semantic model, manifest, and self-test must be
  executable.
- Bytewax must be the only lifecycle stream processor named by the contract.
- Tests must cover contract shape, guardrails, lifecycle execution, API helpers,
  view models, app self-test, Bytewax metadata, and employee-agent metadata.
