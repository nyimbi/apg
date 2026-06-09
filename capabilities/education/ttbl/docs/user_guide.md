# Timetabling & Scheduling

**Capability ID**: `education_ttbl` | **Domain**: `education` | **Version**: `1.0.0`

## Description

The Timetabling capability provides constraint-based timetable generation and management for educational institutions. It supports master, class, teacher, room, and exam timetables; hard and soft constraint modelling; automated conflict detection (teacher double-booking, room double-booking, student group overlaps); conflict resolution workflows; room inventory management; teacher-consent-gated substitution management; multi-format export (iCal, CSV, PDF, JSON, HTML, Excel); and approval-gated publication. Publication is hard-blocked when any unresolved conflict remains.

## Installation

```bash
pip install apg-education-ttbl
```

## Provides

- `timetable_generation_workflow`
- `constraint_management_workflow`
- `room_allocation_workflow`
- `teacher_assignment_workflow`
- `conflict_detection_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ttbl/dashboard` | `education_ttbl:view` | Overview |
| `/ttbl/timetables` | `education_ttbl:view` | Timetables |
| `/ttbl/timetables/<timetable_id>/build` | `education_ttbl:manage_timetables` | Timetables |
| `/ttbl/timetables/<timetable_id>/view` | `education_ttbl:view` | Timetables |
| `/ttbl/constraints` | `education_ttbl:manage_constraints` | Configuration |
| `/ttbl/rooms` | `education_ttbl:manage_rooms` | Resources |
| `/ttbl/conflicts` | `education_ttbl:resolve_conflicts` | Operations |
| `/ttbl/substitutions` | `education_ttbl:manage_substitutions` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_timetable()`
- `get_timetable()`
- `list_timetables()`
- `publish_timetable()`
- `add_constraint()`
- `remove_constraint()`
- `list_constraints()`
- `create_room()`

_(See `service.py` for complete API.)_

## Interoperability

`education_ttbl` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use education_ttbl;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `EDUCATION_TTBL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
