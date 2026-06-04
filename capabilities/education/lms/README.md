# Learning Management System

## Overview

The LMS capability provides full lifecycle management for online and blended learning: course authoring, content delivery (including SCORM 1.2/2004 and xAPI), learner enrolment, assessment creation and grading, certificate issuance, learning path orchestration, and per-learner analytics. It enforces governance rules around grade overrides, certificate eligibility, analytics consent, and cross-tenant isolation.

## Capability ID

`education_lms`

## Provides

| Service | Description |
|---|---|
| `course_lifecycle_workflow` | Create, review, publish, and archive courses |
| `content_delivery_workflow` | Add and serve SCORM, xAPI, video, document, quiz content |
| `enrolment_workflow` | Open, paid, voucher, and invitation enrolments with capacity enforcement |
| `assessment_workflow` | Formative and summative assessments with configurable attempts |
| `grading_workflow` | Score recording, feedback, and approval-gated overrides |
| `certificate_issuance_workflow` | Completion-verified certificate generation |
| `learner_analytics_workflow` | Consent-gated per-learner progress and completion analytics |
| `scorm_xapi_compliance_workflow` | SCORM 1.2/2004 and xAPI 1.0.3 compliance checks |
| `learning_path_workflow` | Ordered multi-course learning paths |
| `cohort_management_workflow` | Cohort-based course grouping and scheduling |

## Requires

| Capability | Reason |
|---|---|
| `auth` | User authentication and permission checks |
| `audl` | Immutable audit trail for all state changes |
| `mten` | Multi-tenant configuration isolation |
| `conf` | Runtime configuration management |
| `ntfy` | Enrolment and grade notifications |
| `wflo` | Course review/publish approval workflows |
| `nlpc` | Search and NLP-assisted content tagging |
| `moni` | LMS operational monitoring and alerting |
| `comp` | SCORM/xAPI regulatory compliance checks |
| `mqeb` | Bytewax event stream for progress tracking |
| `schd` | Scheduled cohort sessions and deadlines |

## Configuration

| Key | Default | Description |
|---|---|---|
| `courses.review_before_publish` | `true` | Require review approval before publishing |
| `enrolments.cross_tenant_denied` | `true` | Block enrolments across tenant boundaries |
| `assessments.grading_schemes` | all | Supported grading schemes |
| `certificates.completion_required` | `true` | Enforce completion criteria before issuing |
| `governance.grade_override_requires_approval` | `true` | All grade overrides need approval reference |
| `governance.analytics_export_requires_consent` | `true` | Learner consent required for exports |

## API Routes

| Path | Method | Description | Permission |
|---|---|---|---|
| `/api/lms/contract` | GET | Capability contract | `education_lms:view` |
| `/api/lms/dashboard` | GET | Dashboard summary | `education_lms:view` |
| `/api/lms/courses` | GET | List courses | `education_lms:view` |
| `/api/lms/courses` | POST | Create course | `education_lms:manage_courses` |
| `/api/lms/courses/<id>` | GET | Course detail | `education_lms:view` |
| `/api/lms/courses/<id>` | PUT | Update course | `education_lms:manage_courses` |
| `/api/lms/courses/<id>/publish` | POST | Publish course | `education_lms:manage_courses` |
| `/api/lms/courses/<id>/content` | GET/POST | List/add content | `education_lms:manage_content` |
| `/api/lms/enrolments` | GET/POST | List/create enrolments | `education_lms:manage_enrolments` |
| `/api/lms/enrolments/<id>` | DELETE | Withdraw enrolment | `education_lms:manage_enrolments` |
| `/api/lms/assessments` | POST | Create assessment | `education_lms:manage_assessments` |
| `/api/lms/submissions` | GET/POST | List/submit assessments | `education_lms:grade` |
| `/api/lms/submissions/<id>/grade` | PUT | Grade submission | `education_lms:grade` |
| `/api/lms/certificates` | GET/POST | List/issue certificates | `education_lms:manage_certificates` |
| `/api/lms/progress` | POST | Record xAPI/SCORM progress | `education_lms:submit` |
| `/api/lms/analytics/learner/<id>` | GET | Learner analytics | `education_lms:analytics` |
| `/api/lms/paths` | GET/POST | Learning paths | `education_lms:manage_paths` |
| `/api/lms/agents` | POST | Register agent | `education_lms:admin` |

## Business Rules

| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | `tenant_context_present=false` | deny — attach tenant context |
| `lms_write_requires_policy` | write op, no policy | deny — attach policy |
| `course_publish_requires_review` | publish, `review_approved=false` | deny — submit for review |
| `paid_enrolment_requires_payment_reference` | paid type, no payment_reference | deny — provide payment reference |
| `cross_tenant_enrolment_denied` | learner and course tenants differ | deny — enrol within tenant |
| `grade_override_requires_approval` | second grade, no approval_reference | deny — obtain approval |
| `certificate_requires_completion` | issue cert, `completion_criteria_met=false` | deny — meet criteria |
| `analytics_export_requires_consent` | export analytics, `consent_recorded=false` | deny — record consent |
| `scorm_compliance_check_required` | publish SCORM, unchecked | deny — run compliance check |
| `privileged_agent_action_requires_human_approval` | agent action, privileged, unapproved | deny — record approval |

## Data Models

| Model | Key Fields |
|---|---|
| `CourseCreate` | id, tenant_id, title, code, course_type, status, grading_scheme, passing_score |
| `ContentItemCreate` | id, course_id, content_type, scorm_version, order_index, compliance_checked |
| `EnrolmentCreate` | id, course_id, learner_id, enrolment_type, payment_reference, completion_percentage |
| `AssessmentCreate` | id, course_id, assessment_type, max_score, weight_percent, attempts_allowed |
| `SubmissionCreate` | id, assessment_id, learner_id, score, graded_by, override_approval |
| `CertificateCreate` | id, enrolment_id, certificate_type, verification_code, issuer_id |
| `LearnerProgressCreate` | id, content_item_id, completion_percentage, time_spent_minutes, xapi_statement |
| `LearningPathCreate` | id, course_ids, required_course_ids, is_published |
| `LmsAgent` | id, runtime, role, scope |

## Streaming Events

| Event | Trigger |
|---|---|
| `course_created` | Course created in draft |
| `course_published` | Course status set to published |
| `content_item_added` | Content item added to course |
| `enrolment_recorded` | Learner enrolled in course |
| `enrolment_withdrawn` | Enrolment withdrawn |
| `assessment_submitted` | Learner submits assessment |
| `grade_recorded` | Submission graded |
| `certificate_issued` | Certificate issued |
| `learner_progress_updated` | xAPI/SCORM progress recorded |

## Edge Cases Handled

- Capacity enforcement: enrolment blocked when `max_enrolments` reached.
- SCORM version validation: only supported versions (1.2, 2004 3rd/4th) accepted.
- Grade override on an already-graded submission requires explicit `override_approval` reference.
- Paid enrolments validated for presence of `payment_reference` before activation.
- Cross-tenant enrolment is hard-denied regardless of policy_attached state.
- Learner analytics export gated on per-learner consent record.
- SCORM content publish blocked until compliance check flag is set.
- Tenant isolation: all store lookups use `(tenant_id, entity_id)` composite keys.

## Composability Notes

- **education_sch_mgmt**: Learner records from `sch_mgmt` feed into LMS enrolments via `learner_id`.
- **education_ttbl**: Live session content items can reference scheduled slots from `ttbl`.
- **auth**: All permission checks delegate to `auth` service; LMS only enforces capability-level rules.
- **ntfy**: Grade notifications and certificate issuance events routed through `ntfy`.
- **wflo**: Course review/publish approval workflows execute via `wflo`.
- **mqeb** / bytewax: All lifecycle events published to `apg.education.lms.lifecycle` stream for downstream analytics consumers.
