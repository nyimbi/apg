# Timetabling & Scheduling

## Overview

The Timetabling capability provides constraint-based timetable generation and management for educational institutions. It supports master, class, teacher, room, and exam timetables; hard and soft constraint modelling; automated conflict detection (teacher double-booking, room double-booking, student group overlaps); conflict resolution workflows; room inventory management; teacher-consent-gated substitution management; multi-format export (iCal, CSV, PDF, JSON, HTML, Excel); and approval-gated publication. Publication is hard-blocked when any unresolved conflict remains.

## Capability ID

`education_ttbl`

## Provides

| Service | Description |
|---|---|
| `timetable_generation_workflow` | Constraint-driven timetable creation via configurable algorithms |
| `constraint_management_workflow` | Hard/soft constraint CRUD with approval-gated removal |
| `room_allocation_workflow` | Room inventory and capacity-checked allocation |
| `teacher_assignment_workflow` | Teacher-period-subject assignment with qualification checks |
| `conflict_detection_workflow` | Automatic detection of double-bookings and group overlaps |
| `conflict_resolution_workflow` | Structured resolution strategies with audit trail |
| `substitution_management_workflow` | Absence-driven substitution with teacher consent enforcement |
| `timetable_publication_workflow` | Approval-gated publication with zero-conflict precondition |
| `exam_scheduling_workflow` | Exam period scheduling with room confirmation requirement |

## Requires

| Capability | Reason |
|---|---|
| `auth` | Access control for timetable editors and viewers |
| `audl` | Audit trail for all schedule mutations |
| `mten` | Tenant isolation |
| `conf` | Configuration management |
| `ntfy` | Substitution notifications to teachers |
| `wflo` | Approval workflows for publication and constraint removal |
| `moni` | Generation job monitoring |
| `mqeb` | Event stream for conflict and publication events |
| `schd` | Integration with academic calendar for term boundaries |
| `comp` | Regulatory compliance for exam scheduling |

## Configuration

| Key | Default | Description |
|---|---|---|
| `timetables.publish_requires_zero_conflicts` | `true` | Block publication if any hard conflict unresolved |
| `timetables.publish_requires_approval` | `true` | Approval reference required to publish |
| `constraints.removal_requires_approval` | `true` | Constraint removal needs approval |
| `rooms.capacity_check_required` | `true` | Capacity verified before room allocation |
| `rooms.cross_tenant_booking_denied` | `true` | Cannot book rooms from another tenant |
| `substitutions.consent_required` | `true` | Teacher must consent before substitution assigned |
| `time_slots.supported_durations_minutes` | `[30..120]` | Only supported period lengths accepted |

## API Routes

| Path | Method | Description | Permission |
|---|---|---|---|
| `/api/ttbl/dashboard` | GET | Dashboard summary | `education_ttbl:view` |
| `/api/ttbl/timetables` | GET | List timetables | `education_ttbl:view` |
| `/api/ttbl/timetables` | POST | Create timetable | `education_ttbl:manage_timetables` |
| `/api/ttbl/timetables/<id>` | GET | Get timetable | `education_ttbl:view` |
| `/api/ttbl/timetables/<id>/publish` | POST | Publish timetable | `education_ttbl:manage_timetables` |
| `/api/ttbl/timetables/<id>/constraints` | GET/POST | List/add constraints | `education_ttbl:manage_constraints` |
| `/api/ttbl/constraints/<id>` | DELETE | Remove constraint | `education_ttbl:manage_constraints` |
| `/api/ttbl/rooms` | GET/POST | List/create rooms | `education_ttbl:manage_rooms` |
| `/api/ttbl/timetables/<id>/slots` | GET/POST | List/create time slots | `education_ttbl:manage_timetables` |
| `/api/ttbl/timetables/<id>/entries` | GET/POST | List/assign entries | `education_ttbl:manage_timetables` |
| `/api/ttbl/timetables/<id>/conflicts` | GET | List conflicts | `education_ttbl:resolve_conflicts` |
| `/api/ttbl/conflicts/<id>/resolve` | PUT | Resolve conflict | `education_ttbl:resolve_conflicts` |
| `/api/ttbl/substitutions` | GET/POST | List/request substitutions | `education_ttbl:manage_substitutions` |
| `/api/ttbl/substitutions/<id>/assign` | PUT | Assign substitute | `education_ttbl:manage_substitutions` |
| `/api/ttbl/timetables/<id>/export` | GET | Export timetable | `education_ttbl:export` |
| `/api/ttbl/agents` | POST | Register agent | `education_ttbl:admin` |

## Business Rules

| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | `tenant_context_present=false` | deny |
| `timetable_publish_requires_zero_conflicts` | publish, unresolved conflicts exist | deny — resolve all conflicts |
| `timetable_publish_requires_approval` | publish, no approval reference | deny — obtain approval |
| `constraint_removal_requires_approval` | remove constraint, no approval | deny — obtain approval |
| `room_capacity_check_required` | allocate room, check not performed | deny — perform capacity check |
| `cross_tenant_room_booking_denied` | room tenant differs from requestor | deny |
| `substitution_requires_teacher_consent` | assign substitution, no consent | deny — record consent |
| `exam_schedule_requires_room_confirmation` | publish exam schedule, unconfirmed rooms | deny — confirm rooms |
| `generation_algorithm_not_supported` | generate, unsupported algorithm | deny — select supported algorithm |
| `privileged_agent_action_requires_human_approval` | agent, privileged, unapproved | deny |

## Data Models

| Model | Key Fields |
|---|---|
| `TimetableCreate` | id, timetable_type, status, generation_algorithm, approval_reference, published_at |
| `ConstraintCreate` | id, timetable_id, constraint_type, entity_id, entity_type, is_hard, weight |
| `RoomCreate` | id, code, room_type, capacity, amenities, is_available |
| `TimeSlotCreate` | id, timetable_id, day_of_week, start_time, end_time, duration_minutes, period_number |
| `ScheduleEntryCreate` | id, time_slot_id, room_id, teacher_id, subject_id, student_group_id |
| `ConflictCreate` | id, conflict_type, entry_ids, severity, resolution_type, resolved_at |
| `SubstitutionRequestCreate` | id, original_entry_id, absent_teacher_id, substitute_teacher_id, teacher_consent_recorded |
| `TtblAgent` | id, runtime, role, scope |

## Streaming Events

| Event | Trigger |
|---|---|
| `timetable_created` | Timetable created in draft |
| `timetable_generation_started` | Algorithm run initiated |
| `timetable_generation_completed` | Algorithm run complete |
| `conflict_detected` | Double-booking or overlap found during entry assignment |
| `conflict_resolved` | Conflict marked resolved with strategy |
| `constraint_added` | Constraint registered |
| `room_allocated` | Schedule entry assigned to room |
| `teacher_assigned` | Teacher assigned to schedule entry |
| `substitution_requested` | Absent teacher substitution requested |
| `substitution_assigned` | Substitute teacher assigned with consent |
| `timetable_published` | Timetable status set to published |

## Edge Cases Handled

- Publication is hard-blocked when any hard conflict remains unresolved — no bypass via approval alone.
- Conflict detection runs automatically on every `assign_entry` call, checking teacher, room, and student group overlaps against all existing same-slot entries.
- Constraint removal (not just add) requires an explicit approval reference to prevent accidental soft deletion.
- Room capacity check flag must be explicitly set `true` by the caller — the service does not auto-check against student group size (external HR/SIS data required).
- Substitution assignment requires `teacher_consent_recorded=True`; the service does not infer consent from any prior state.
- Cross-tenant room booking is hard-denied; shared facility scenarios require a dedicated shared-tenant setup.
- Slot duration must be in the supported list (multiples of 5 from 30 to 120); non-standard durations rejected at rule evaluation time.
- Export payload includes full slot map and entries for offline rendering — format conversion to iCal/PDF/CSV delegated to downstream consumers.

## Composability Notes

- **education_sch_mgmt**: Staff records supply teacher IDs for schedule entries; grade levels drive class/group definitions.
- **education_lms**: Live session content items in LMS can reference timetable entry IDs for blended learning scheduling.
- **schd**: Academic calendar term boundaries from `sch_mgmt` calendar events feed into `schd` to constrain generation windows.
- **ntfy**: Substitution assignments and timetable publications broadcast via `ntfy` to affected teachers and students.
- **wflo**: Publication approval and constraint removal approval workflows execute in `wflo`.
- **mqeb** / bytewax: All conflict and publication events published to `apg.education.ttbl.lifecycle` for dashboards and downstream analytics.
