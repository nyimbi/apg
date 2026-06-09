# School Management

**Capability ID**: `education_sch_mgmt` | **Domain**: `education` | **Version**: `1.0.0`

## Description

The School Management capability provides end-to-end administration for educational institutions: student records and lifecycle management, structured admissions workflows with capacity control, fee generation and payment tracking, staff administration, academic calendar management, document vault with consent-gated sharing, multi-channel communications, and reporting. All operations are tenant-scoped with strict governance on sensitive actions (expulsion, fee waivers, student data exports).

## Installation

```bash
pip install apg-education-sch_mgmt
```

## Provides

- `student_records_workflow`
- `admissions_workflow`
- `fee_management_workflow`
- `parent_portal_workflow`
- `staff_administration_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/sch-mgmt/dashboard` | `education_sch_mgmt:view` | Overview |
| `/sch-mgmt/students` | `education_sch_mgmt:view_students` | Students |
| `/sch-mgmt/students/<student_id>` | `education_sch_mgmt:view_students` | Students |
| `/sch-mgmt/admissions` | `education_sch_mgmt:manage_admissions` | Admissions |
| `/sch-mgmt/fees` | `education_sch_mgmt:manage_fees` | Finance |
| `/sch-mgmt/fees/invoices` | `education_sch_mgmt:manage_fees` | Finance |
| `/sch-mgmt/parent-portal` | `education_sch_mgmt:parent_access` | Community |
| `/sch-mgmt/staff` | `education_sch_mgmt:manage_staff` | Human Resources |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_student()`
- `get_student()`
- `list_students()`
- `update_student_status()`
- `submit_application()`
- `update_admission_status()`
- `list_admissions()`
- `generate_fee_invoice()`

_(See `service.py` for complete API.)_

## Interoperability

`education_sch_mgmt` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use education_sch_mgmt;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `EDUCATION_SCH_MGMT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
