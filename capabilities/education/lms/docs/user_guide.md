# Learning Management System

**Capability ID**: `education_lms` | **Domain**: `education` | **Version**: `1.0.0`

## Description

The LMS capability provides full lifecycle management for online and blended learning: course authoring, content delivery (including SCORM 1.2/2004 and xAPI), learner enrolment, assessment creation and grading, certificate issuance, learning path orchestration, and per-learner analytics. It enforces governance rules around grade overrides, certificate eligibility, analytics consent, and cross-tenant isolation.

## Installation

```bash
pip install apg-education-lms
```

## Provides

- `course_lifecycle_workflow`
- `content_delivery_workflow`
- `enrolment_workflow`
- `assessment_workflow`
- `grading_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/lms/dashboard` | `education_lms:view` | Overview |
| `/lms/courses` | `education_lms:view` | Learning |
| `/lms/courses/create` | `education_lms:manage_courses` | Learning |
| `/lms/courses/<course_id>` | `education_lms:view` | Learning |
| `/lms/courses/<course_id>/content` | `education_lms:manage_content` | Learning |
| `/lms/enrolments` | `education_lms:manage_enrolments` | Administration |
| `/lms/assessments` | `education_lms:manage_assessments` | Assessment |
| `/lms/submissions` | `education_lms:grade` | Assessment |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_course()`
- `get_course()`
- `list_courses()`
- `update_course()`
- `publish_course()`
- `archive_course()`
- `add_content_item()`
- `list_course_content()`

_(See `service.py` for complete API.)_

## Interoperability

`education_lms` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use education_lms;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `EDUCATION_LMS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
