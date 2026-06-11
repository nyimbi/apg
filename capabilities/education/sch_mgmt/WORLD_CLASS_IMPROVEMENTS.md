# School Management — World-Class Improvements

15 high-impact improvements grounded in observable gaps vs. leading EdTech platforms (PowerSchool, Infinite Campus, Alma, Blackbaud, SchoolMint, Brightwheel, Edsby, Gradelink).

---

## 1. Predictive Attendance Risk Scoring

**Category**: Analytics / ML

**Justification**: Chronic absenteeism (>10% of days missed) is the strongest early predictor of dropout. Schools need automatic flagging before the pattern is irreversible. Infinite Campus surfaces an "At-Risk" band on the student dashboard; APG currently has no forward-looking signal.

**Implementation**: Aggregate rolling attendance records per student. Compute a risk score (0–100) using a configurable threshold model: `days_absent / enrolled_days`. Publish a `student_attendance_risk_flagged` event when score crosses 80. Expose `async def attendance_risk_score(tenant_id, student_id, academic_year)`.

**Competitor reference**: Infinite Campus "Early Warning System" attendance sub-score.

---

## 2. Bulk Grade Import and Grade-Book Management

**Category**: Academic Records

**Justification**: Teachers spend 15–30 % of prep time on manual score entry. PowerSchool and Alma both support CSV/JSON bulk grade import with column mapping and validation. APG has single-record exam result recording only — no grade-book aggregate, no subject weighting, no running GPA.

**Implementation**: `async def bulk_import_grades(tenant_id, class_id, term, rows: list[dict])` — validates student IDs, subject codes, and score ranges; persists to a `GradeEntry` model; recomputes term GPA per student. Add `async def get_student_gpa(tenant_id, student_id, academic_year)` returning weighted and unweighted GPA.

**Competitor reference**: PowerSchool "Grade Book" bulk upload; Alma gradebook import.

---

## 3. Parent/Guardian Self-Service Portal API

**Category**: Parent Engagement

**Justification**: Brightwheel and Edsby show that 70 %+ of parent-school communication now flows through mobile portals. APG's `parent_communication` method is teacher→parent only; there is no inbound parent request, no read-receipt tracking, and no parent-visible fee or attendance view.

**Implementation**: `async def parent_portal_summary(tenant_id, guardian_id)` — aggregates linked students' attendance rates, outstanding fees, upcoming calendar events, and unread communications. Add `async def record_parent_read_receipt(tenant_id, communication_id, guardian_id)`. Gate all portal reads with `record_tenant_matches_requestor_tenant` rule.

**Competitor reference**: Brightwheel parent app; Edsby family portal.

---

## 4. Automated Fee Reminder Workflow

**Category**: Fee Management / Workflow

**Justification**: Manual follow-up on overdue invoices wastes bursar time and creates inconsistent enforcement. Blackbaud Tuition Management and FACTS automate tiered reminder sequences (T+7, T+14, T+30 days). APG marks invoices `overdue` with no downstream escalation.

**Implementation**: `async def trigger_fee_reminders(tenant_id, term, dry_run=False)` — queries all `overdue` invoices, groups by student, computes days overdue, dispatches graded communications (SMS at T+7, email+SMS at T+14, formal letter at T+30) via `dispatch_communication`, and records reminder events in the audit log.

**Competitor reference**: FACTS Management automated billing; Blackbaud Smart Tuition.

---

## 5. Timetable / Class-Schedule Management

**Category**: Academic Administration

**Justification**: Timetable management is universally cited as the most time-intensive admin task in K-12. PowerSchool Scheduling, Edval, and Infinite Campus automate period-slot assignments, teacher conflicts, and room utilisation. APG has no timetable object.

**Implementation**: `async def create_timetable_slot(tenant_id, academic_year, term, day_of_week, period, class_id, subject, teacher_id, room, created_by)` — enforces no double-booking via in-memory conflict check. `async def get_class_schedule(tenant_id, class_id, term)` returns ordered slot list. Publish `timetable_slot_created` event.

**Competitor reference**: PowerSchool Scheduling Engine; Edval AI scheduler.

---

## 6. Student Learning Outcomes / Competency Tracking

**Category**: Academic Records / Standards Alignment

**Justification**: Competency-based grading is now mandated in several African national curricula (Kenya CBC, Rwanda CBE). Alma and Brightwheel track per-standard mastery levels. APG scores exams with a blunt A–E grade; there is no subject-standard linkage.

**Implementation**: `async def record_competency_assessment(tenant_id, student_id, standard_code, mastery_level: int, evidence_ref, assessed_by)` — validates `mastery_level` in [1, 4], stores `CompetencyRecord`. `async def get_competency_profile(tenant_id, student_id, term)` returns a dict of `{standard_code: mastery_level}`.

**Competitor reference**: Alma competency-based grading; Infinite Campus standards-based report cards.

---

## 7. Digital Consent Management

**Category**: Compliance / Data Protection

**Justification**: Kenya's Data Protection Act 2019 and GDPR both require explicit, recorded consent for processing minors' data. APG's `share_document` checks `consent_recorded` as a boolean flag but does not capture *what* was consented to, *by whom*, or *when*. Regulators require a full consent audit trail.

**Implementation**: `async def record_consent(tenant_id, guardian_id, student_id, consent_type, scope, consented_at, ip_address)` — stores a `ConsentRecord` with SHA-256 hash of the consent payload for tamper evidence. `async def check_consent(tenant_id, student_id, consent_type)` returns the active consent record or `None`. All data export and sharing methods call `check_consent` before proceeding.

**Competitor reference**: SchoolMint FERPA/COPPA consent flows; Blackbaud GDPR consent manager.

---

## 8. Health and Medical Records Management

**Category**: Student Welfare

**Justification**: Schools in most jurisdictions are legally required to maintain health records including vaccination status, allergies, and medication administration logs. Infinite Campus Health, Skyward Health, and SchoolDude all provide this module. APG has a freetext `medical_notes` field — no structure, no vaccination tracking, no medication log.

**Implementation**: `async def record_medical_event(tenant_id, student_id, event_type, description, administered_by, date)` — validates `event_type` in `{vaccination, allergy_update, medication_administered, injury, sick_visit}`. `async def get_health_summary(tenant_id, student_id)` returns structured health profile with vaccination list, allergy flags, and recent events.

**Competitor reference**: Infinite Campus Health module; Skyward Student Health.

---

## 9. Automated Report Card Generation with PDF Output

**Category**: Reporting

**Justification**: `generate_report_card` returns a dict with no grades because it has no grade-book integration and no output artefact. Gradelink, PowerSchool, and Alma generate formatted PDF report cards in one click with subject scores, teacher comments, attendance summary, and signatures.

**Implementation**: Extend `generate_report_card` to join grade entries (from improvement #2), attendance records, and teacher comments. Add `async def render_report_card_pdf(tenant_id, student_id, term)` — assembles a `ReportCardDocument` payload and calls a configured PDF renderer service (e.g. WeasyPrint or a LaTeX template). Return a `file_reference` suitable for upload via `upload_document`.

**Competitor reference**: PowerSchool Report Cards; Gradelink PDF transcripts.

---

## 10. Multi-Campus / Branch School Support

**Category**: Multi-Tenancy / Architecture

**Justification**: Large school groups (e.g. Nova Pioneer, Bridge International Academies) operate dozens of campuses under one administrative entity. PowerSchool SIS supports campus-level isolation within a district. APG's `tenant_id` conflates district and campus — there is no sub-tenant scoping.

**Implementation**: Add `campus_id: str | None` to all major models. `async def create_campus(tenant_id, campus_id, name, address, principal_id, created_by)`. Filter list methods by `campus_id` when provided. Enforce that a student's `campus_id` matches the requester's campus scope via a `campus_context_matches` rule.

**Competitor reference**: PowerSchool Multi-District; Infinite Campus Campus Configuration.

---

## 11. Real-Time Notification Dispatch with Delivery Tracking

**Category**: Communications

**Justification**: APG dispatches communications to an in-memory store with no actual delivery. Edsby and Brightwheel show delivery receipts (sent, delivered, read) per channel. Without delivery tracking, schools cannot prove a parent received a safeguarding notice — a legal liability.

**Implementation**: `async def dispatch_communication` extended: after persisting, call an injected `NotificationGateway` with `await gateway.send(channel, recipients, subject, body)` returning a list of `DeliveryReceipt` dicts. Persist receipts. `async def get_communication_delivery_status(tenant_id, communication_id)` returns per-recipient delivery state.

**Competitor reference**: Edsby delivery receipts; Remind (now ParentSquare) read tracking.

---

## 12. Student Cohort and Progression Tracking

**Category**: Academic Records / Longitudinal Analytics

**Justification**: Schools need to track a student's journey across years — grade promotions, repetitions, and lateral transfers — to produce longitudinal reports for regulatory bodies (NEMIS in Kenya requires cohort data). APG has no progression history; `update_student_status` overwrites without preserving prior state.

**Implementation**: `async def promote_student(tenant_id, student_id, from_grade, to_grade, academic_year, created_by)` — appends a `ProgressionEvent` to an immutable history list on the student record. `async def get_progression_history(tenant_id, student_id)` returns the full chain. Repeating a grade creates a `RepetitionEvent` with a mandatory `reason` field.

**Competitor reference**: Infinite Campus Grade History; NEMIS cohort tracking in Kenyan schools.

---

## 13. Integration Hooks / Webhook Outbox

**Category**: Interoperability / Event-Driven Architecture

**Justification**: Third-party systems (LMS, payment gateways, government EMIS portals) need reliable event delivery. APG emits audit events to an in-memory list that is lost on restart. The outbox pattern (used by Stripe, Shopify, and modern EdTech platforms) guarantees at-least-once delivery.

**Implementation**: Replace `self.audit_events: list` with an `OutboxEntry` model list. `async def flush_outbox(tenant_id, consumer_id)` marks entries as delivered and returns them. `async def register_webhook(tenant_id, url, event_types, secret)` stores a `WebhookSubscription`. On each `_audit()` call, fan out to matching subscriptions via `await webhook_client.post(url, payload, signature)`.

**Competitor reference**: Stripe webhook outbox; Schoology LTI/webhook integrations.

---

## 14. AI-Powered Admission Scoring and Shortlisting

**Category**: Admissions / AI

**Justification**: SchoolMint and Ravenna automate application scoring and shortlisting using configurable criteria weighting (academic history 40 %, interview 30 %, proximity 20 %, sibling preference 10 %). APG's admission workflow has no scoring — reviewers must read every application manually.

**Implementation**: `async def score_admission_application(tenant_id, application_id, criteria: dict[str, float])` — computes a weighted composite score from application fields, stores `AdmissionScore` on the application. `async def shortlist_admissions(tenant_id, academic_year, grade_level, top_n, criteria)` — ranks all `submitted` applications by score, transitions top N to `shortlisted`, remainder to `waitlisted`, dispatches notification per applicant.

**Competitor reference**: SchoolMint Enroll scoring engine; Ravenna automated shortlisting.

---

## 15. Financial Analytics and Budget Variance Reporting

**Category**: Financial Management

**Justification**: School bursars need actual vs. expected fee collection ratios, debtor aging reports, and term-over-term trend analysis. Blackbaud Financial Edge and SchoolAdmin provide these out of the box. APG's `generate_school_report` returns raw totals with no aging, no variance, and no trend line.

**Implementation**: `async def fee_collection_analytics(tenant_id, academic_year)` — computes per-term collection rates, identifies top 10 outstanding debtors, groups overdue invoices into aging buckets (0–30, 31–60, 61–90, 90+ days), and calculates term-over-term collection rate delta. `async def debtor_aging_report(tenant_id, term, as_of_date)` returns a structured aging ledger suitable for export.

**Competitor reference**: Blackbaud Financial Edge; SchoolAdmin bursary module.
