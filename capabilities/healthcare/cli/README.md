# Clinical Management

## Overview
Clinical workflow orchestration capability providing care plan management, clinical protocol activation, workflow task tracking, clinical decision support (CDS) alerts, structured handoff management, and care team coordination. Enforces structured SBAR handoff format and requires team assignment before care plan activation.

## Capability ID
`healthcare_cli`

## Provides
- care_plan_management: Multidisciplinary care plan lifecycle (draft → active → completed/revoked) with intervention tracking
- clinical_workflow_orchestration: Task-level workflow tracking with overdue detection and state transitions
- protocol_adherence_tracking: Evidence-based protocol activation (sepsis bundle, stroke, MI, etc.) with completion tracking
- clinical_decision_support: Real-time CDS alerts (sepsis screening, deterioration, drug dosing, guideline reminders) with acknowledgement
- care_team_management: Assign and track multidisciplinary team members per patient care plan
- clinical_handoff_management: Structured SBAR handoffs with acknowledgement tracking across shift changes, transfers, and discharges
- intervention_tracking: Add and track clinical interventions (medication, procedure, education, therapy, etc.) within care plans
- deterioration_alerting: Early warning CDS alerts for patient deterioration with escalation support

## Requires
- auth: Clinical role-based access control
- audl: Audit trail for all care plan and protocol modifications
- mten: Multi-tenant isolation
- conf: Tenant-specific protocol library configuration
- ntfy: Overdue workflow and critical CDS alert notifications
- wflo: Care plan approval and protocol activation workflows
- nlpc: NLP-assisted documentation and protocol search
- moni: Operational monitoring for workflow SLA tracking
- mqeb: Event emission for downstream analytics and EMR

## Configuration

| Key | Description |
|-----|-------------|
| care_plans.multidisciplinary_team_required | Require at least one team member before activation |
| protocols.evidence_required | Require evidence reference for all protocols |
| workflows.overdue_alert_enabled | Alert when workflow tasks pass due_at |
| handoffs.structured_format_required | Enforce structured_format_used=True for all handoffs |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/cli/care-plans | List care plans | healthcare_cli:care_plans |
| POST | /api/healthcare/cli/care-plans | Create care plan | healthcare_cli:care_plans_write |
| GET | /api/healthcare/cli/care-plans/<id> | Care plan detail | healthcare_cli:care_plans |
| POST | /api/healthcare/cli/care-plans/<id>/activate | Activate | healthcare_cli:care_plans_write |
| POST | /api/healthcare/cli/care-plans/<id>/complete | Complete | healthcare_cli:care_plans_write |
| POST | /api/healthcare/cli/care-plans/<id>/interventions | Add intervention | healthcare_cli:care_plans_write |
| GET | /api/healthcare/cli/protocols | List protocols | healthcare_cli:protocols |
| POST | /api/healthcare/cli/protocols | Create/activate protocol | healthcare_cli:protocols |
| GET | /api/healthcare/cli/workflows | List workflows | healthcare_cli:workflows |
| POST | /api/healthcare/cli/workflows | Create workflow task | healthcare_cli:workflows |
| POST | /api/healthcare/cli/workflows/<id>/transition | Transition state | healthcare_cli:workflows |
| GET | /api/healthcare/cli/cds-alerts | List CDS alerts | healthcare_cli:cds |
| POST | /api/healthcare/cli/cds-alerts | Create CDS alert | healthcare_cli:cds |
| POST | /api/healthcare/cli/cds-alerts/<id>/acknowledge | Acknowledge | healthcare_cli:cds |
| GET | /api/healthcare/cli/handoffs | List handoffs | healthcare_cli:handoffs |
| POST | /api/healthcare/cli/handoffs | Record handoff | healthcare_cli:handoffs |
| POST | /api/healthcare/cli/handoffs/<id>/acknowledge | Acknowledge handoff | healthcare_cli:handoffs |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| care_plan_requires_team_member | operation=activate_care_plan, team_member_assigned=False | deny |
| protocol_activation_requires_criteria | operation=activate_protocol, activation_criteria_met=False | deny |
| handoff_requires_structured_format | operation=record_handoff, structured_format_used=False | deny |
| revoked_care_plan_not_editable | operation=update_care_plan, care_plan_status=revoked | deny |
| completed_protocol_not_re_activatable | operation=activate_protocol, protocol_status=completed | deny |
| overdue_workflow_alert | operation=check_workflow, workflow_state=overdue | warn |
| decision_support_evidence_required | operation=create_cds_alert, evidence_reference_present=False | deny |

## Data Models
- CarePlanCreate/Response: patient_id, goals, care_team_ids, interventions list, adherence_status, status
- ProtocolCreate/Response: protocol_type, activation_criteria, steps, evidence_reference, status, activated_at
- ClinicalWorkflowCreate/Response: patient_id, care_plan_id, assigned_to, due_at, state, completed_at
- CDSAlertCreate/Response: cds_type, priority, message, evidence_reference, suggested_action, acknowledged_by
- HandoffCreate/Response: handoff_type, SBAR fields, structured_format_used, acknowledged_by

## Streaming Events
- care_plan_created, care_plan_activated, care_plan_completed
- protocol_activated, workflow_state_changed, intervention_completed
- handoff_recorded, cds_alert_triggered, deterioration_alert_fired

## Edge Cases Handled
- Care plan activation requires at least one care_team_id; empty list is hard denied
- Handoff structured_format_used=False is denied at rule layer regardless of content quality
- Revoked care plans cannot be edited; a new care plan must be created
- CDS alerts require evidence_reference to prevent unsupported recommendations
- Workflow overdue detection compares due_at to UTC now at dashboard summary time

## Composability Notes
Care plans reference ICD-10 codes from `healthcare_emr` problem lists. CDS alerts consume lab results from `healthcare_lab` (critical values) and vital signs. Protocol activation events trigger workflow tasks that are tracked alongside `healthcare_pmt` ADT events.

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements — Clinical Management (healthcare_cli)
- **I2.** Early Warning Score (EWS) Engine
- **I3.** FHIR R4 Resource Serialisation
- **I4.** Admission-to-Discharge Acuity Timeline
- **I5.** Constraint-Based Bed Management Integration
- **I6.** Clinical Documentation Quality Scorer
- **I7.** Protocol Deviation Detection and Auto-Alert
- **I8.** Structured Consent Management
- **I9.** Multi-Factor Risk Stratification
- **I10.** Outcome Tracking and Readmission Prediction
- **I11.** Clinical Task Escalation Engine
- **I12.** Antimicrobial Stewardship Tracker
- **I13.** Structured Adverse Event Reporting
- **I14.** Pre-Operative Checklist Automation
- **I15.** Care Bundle Compliance Dashboard

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
