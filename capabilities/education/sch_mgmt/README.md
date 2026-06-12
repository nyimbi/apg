# School Management

## Overview

The School Management capability provides end-to-end administration for educational institutions: student records and lifecycle management, structured admissions workflows with capacity control, fee generation and payment tracking, staff administration, academic calendar management, document vault with consent-gated sharing, multi-channel communications, and reporting. All operations are tenant-scoped with strict governance on sensitive actions (expulsion, fee waivers, student data exports).

## Capability ID

`education_sch_mgmt`

## Provides

| Service | Description |
|---|---|
| `student_records_workflow` | Full student lifecycle from enrolment to alumni/graduation |
| `admissions_workflow` | Application intake, review, shortlisting, offering, and acceptance |
| `fee_management_workflow` | Invoice generation, payment recording, waiver and refund with approvals |
| `parent_portal_workflow` | Read-only guardian access to student profile, fees, and calendar |
| `staff_administration_workflow` | Staff record creation, role management, and status tracking |
| `academic_calendar_workflow` | Term, holiday, and event management by academic year |
| `document_management_workflow` | Secure document upload with confidentiality and consent controls |
| `communications_workflow` | Multi-channel bulk and targeted communications |
| `reporting_workflow` | Attendance, fee, transcript, and roster reports |

## Requires

| Capability | Reason |
|---|---|
| `auth` | Authentication and access control |
| `audl` | Audit trail for all student and financial records |
| `mten` | Tenant isolation for school data |
| `conf` | Configuration management |
| `ntfy` | Fee reminders and admission notifications |
| `wflo` | Approval workflows for expulsion and fee waivers |
| `comp` | Data protection compliance for student records |
| `mqeb` | Event stream for fee and admission lifecycle events |
| `schd` | Academic calendar scheduling |

## Configuration

| Key | Default | Description |
|---|---|---|
| `students.expulsion_requires_approval` | `true` | Expulsion must have approval reference |
| `admissions.capacity_check_required` | `true` | Class capacity verified before offering admission |
| `fees.waiver_requires_approval` | `true` | Fee waiver must have approval reference |
| `fees.refund_requires_approval` | `true` | Fee refund must have approval reference |
| `documents.sharing_requires_consent` | `true` | Consent required to share any document |
| `governance.student_data_export_requires_consent` | `true` | Consent required before bulk data export |
| `governance.cross_tenant_record_access_denied` | `true` | Hard block on cross-tenant record access |

## API Routes

| Path | Method | Description | Permission |
|---|---|---|---|
| `/api/sch-mgmt/dashboard` | GET | Dashboard summary | `education_sch_mgmt:view` |
| `/api/sch-mgmt/students` | GET | List students | `education_sch_mgmt:view_students` |
| `/api/sch-mgmt/students` | POST | Register student | `education_sch_mgmt:manage_students` |
| `/api/sch-mgmt/students/<id>` | GET | Student profile | `education_sch_mgmt:view_students` |
| `/api/sch-mgmt/students/<id>/status` | PUT | Update student status | `education_sch_mgmt:manage_students` |
| `/api/sch-mgmt/admissions` | GET/POST | List/submit applications | `education_sch_mgmt:manage_admissions` |
| `/api/sch-mgmt/admissions/<id>/status` | PUT | Update admission status | `education_sch_mgmt:manage_admissions` |
| `/api/sch-mgmt/fees` | GET/POST | List/generate invoices | `education_sch_mgmt:manage_fees` |
| `/api/sch-mgmt/fees/<id>/pay` | PUT | Record payment | `education_sch_mgmt:manage_fees` |
| `/api/sch-mgmt/fees/<id>/waive` | PUT | Waive fee | `education_sch_mgmt:manage_fees` |
| `/api/sch-mgmt/staff` | GET/POST | List/create staff records | `education_sch_mgmt:manage_staff` |
| `/api/sch-mgmt/calendar` | GET/POST | List/create calendar events | `education_sch_mgmt:manage_calendar` |
| `/api/sch-mgmt/documents` | POST | Upload document | `education_sch_mgmt:manage_documents` |
| `/api/sch-mgmt/communications` | POST | Dispatch communication | `education_sch_mgmt:send_communications` |
| `/api/sch-mgmt/agents` | POST | Register agent | `education_sch_mgmt:admin` |

## Business Rules

| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | `tenant_context_present=false` | deny |
| `expulsion_requires_approval` | `new_status=expelled`, no approval | deny — obtain approval |
| `admission_offer_requires_capacity_check` | offer, `capacity_available=false` | deny — verify capacity |
| `fee_waiver_requires_approval` | waive fee, no approval | deny — obtain approval |
| `fee_refund_requires_approval` | refund fee, no approval | deny — obtain approval |
| `document_sharing_requires_consent` | share doc, `consent_recorded=false` | deny — record consent |
| `cross_tenant_record_access_denied` | tenant mismatch on record access | deny |
| `student_data_export_requires_consent` | export data, `consent_recorded=false` | deny — record consent |
| `unsupported_staff_role` | role not in supported list | deny — select supported role |
| `privileged_agent_action_requires_human_approval` | agent, privileged, unapproved | deny |

## Data Models

| Model | Key Fields |
|---|---|
| `StudentCreate` | id, student_number, grade_level, status, guardian_ids, special_needs |
| `AdmissionApplicationCreate` | id, grade_level_applying, status, reviewer_id, offer_reference |
| `FeeInvoiceCreate` | id, student_id, fee_type, amount, currency, status, academic_year, term |
| `StaffRecordCreate` | id, staff_number, role, status, subjects, qualifications |
| `CalendarEventCreate` | id, event_type, start_date, end_date, academic_year, term, is_public |
| `DocumentCreate` | id, owner_id, owner_type, document_type, is_confidential, consent_recorded |
| `CommunicationCreate` | id, channel, recipient_ids, recipient_groups, sent_at, is_draft |
| `SchMgmtAgent` | id, runtime, role, scope |

## Streaming Events

| Event | Trigger |
|---|---|
| `student_enrolled` | New student record created |
| `student_status_changed` | Student status updated |
| `admission_submitted` | Application submitted |
| `admission_decision_recorded` | Status moved to offered/accepted/rejected |
| `fee_invoice_generated` | Fee invoice created |
| `fee_payment_recorded` | Payment reference attached |
| `staff_record_created` | Staff member registered |
| `calendar_event_published` | Calendar event created |
| `document_uploaded` | Document added to vault |
| `communication_dispatched` | Message sent or scheduled |

## Edge Cases Handled

- Expulsion status transition blocked without explicit approval reference.
- Capacity check enforced before any admission offer is issued; no bypass path.
- Fee waivers and refunds both require distinct approval references (not reusable).
- Document sharing consent is tracked at the document level, not globally.
- Cross-tenant student record access is hard-denied regardless of auth role.
- Student data export (bulk) requires per-export consent recording.
- Unsupported grade levels rejected at model validation time via `AfterValidator`.
- Communications to `carrier_pigeon` or any unsupported channel fail fast at rule evaluation.

## Composability Notes

- **education_lms**: `student_id` from this capability maps to `learner_id` in LMS enrolments.
- **education_ttbl**: Staff records feed teacher IDs into timetable assignments.
- **ntfy**: Fee overdue notices, admission decisions, and event announcements routed through `ntfy`.
- **wflo**: Expulsion and fee-waiver approval workflows execute in `wflo`.
- **comp**: GDPR/data protection compliance checks for student data handled by `comp`.
- **mqeb** / bytewax: Admission and fee lifecycle events published to `apg.education.sch_mgmt.lifecycle`.

---

## World-Class Enhancements (v2.0)

- **I1.** School Management — World-Class Improvements
- **I2.** Predictive Attendance Risk Scoring
- **I3.** Bulk Grade Import and Grade-Book Management
- **I4.** Parent/Guardian Self-Service Portal API
- **I5.** Automated Fee Reminder Workflow
- **I6.** Timetable / Class-Schedule Management
- **I7.** Student Learning Outcomes / Competency Tracking
- **I8.** Digital Consent Management
- **I9.** Health and Medical Records Management
- **I10.** Automated Report Card Generation with PDF Output
- **I11.** Multi-Campus / Branch School Support
- **I12.** Real-Time Notification Dispatch with Delivery Tracking
- **I13.** Student Cohort and Progression Tracking
- **I14.** Integration Hooks / Webhook Outbox
- **I15.** AI-Powered Admission Scoring and Shortlisting

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
