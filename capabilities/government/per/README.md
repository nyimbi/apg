# Personnel & HR Management (government_per)

## Overview

Civil service HR, payroll integration, performance management, and disciplinary case management for government agencies. Covers the full employee lifecycle — from appointment through increment processing, secondment, terminal benefits calculation, and disciplinary proceedings — while enforcing statutory compliance (Employment Act, Public Service Commission Regulations, Pensions Act Cap 189).

Also provides the original **Permits Management** workflows (building, environmental, occupation, demolition permits) that share this capability namespace.

## Capability ID
`government_per`

## Version
`1.1.0`

## Provides

### HR & Personnel
- `employee_appointment_workflow`: Civil service appointment creation with FSM lifecycle
- `payroll_run_workflow`: Payroll processing with Bytewax/NATS streaming
- `grade_increment_workflow`: Annual increment processing with appraisal gating
- `disciplinary_case_workflow`: Due-process disciplinary FSM with appeal windows
- `leave_accrual_workflow`: Leave balance computation with carry-over rules
- `terminal_benefits_workflow`: Pension and gratuity calculation (Cap 189 / NPF / LGPS)
- `secondment_transfer_workflow`: Cross-agency secondment with automatic payroll handoff
- `headcount_establishment_workflow`: Workforce headcount vs. establishment ceiling enforcement
- `performance_appraisal_workflow`: KPI-based appraisal with rating band distribution
- `statutory_deductions_workflow`: PAYE / NHIF / NSSF / Housing Levy computation

### Permits Management (original)
- `permit_application_workflow`: Permit application intake and assessment
- `permit_issuance_workflow`: Issue permit with conditions attached
- `conditional_approval_workflow`: Track and enforce permit conditions
- `inspection_scheduling_workflow`: Construction-phase inspection management
- `permit_compliance_monitoring_workflow`: Ongoing compliance against conditions
- `permit_revocation_workflow`: Revoke permits for serious breaches
- `permits_review_workflow`: Governance review of permit decisions
- `enforcement_action_workflow`: Stop-work orders and enforcement notices

## Requires

| Capability | Reason |
|---|---|
| `auth` | Employee and officer RBAC |
| `audl` | HR decision and payroll audit trail |
| `mten` | Tenant-scoped employee registry |
| `conf` | Grade/step salary matrix, pension scheme parameters, deduction rate tables |
| `ntfy` | Payslip distribution, increment notifications, disciplinary hearing notices |
| `wflo` | Appointment and approval workflow orchestration |
| `schd` | Inspection and hearing scheduling |
| `government_bud` | Establishment ceiling and personnel emoluments vote check |
| `intel` | Workforce cost analytics and attrition modelling |
| `mqeb` | Event streaming via NATS + Bytewax |

## Streaming Architecture

All significant HR events are published to **NATS JetStream** subjects and processed by **Bytewax** dataflows:

| NATS Subject | Event | Consumer |
|---|---|---|
| `apg.government.per.appointment.created` | `AppointmentCreated` | `government_bud` (emoluments debit) |
| `apg.government.per.payroll.run` | `PayrollRunInitiated` | Bytewax payroll dataflow |
| `apg.government.per.payroll.netpay` | `NetPayComputed` | `government_bud`, `audl` |
| `apg.government.per.payroll.increment` | `SalaryIncrementApproved/Withheld` | `conf` (grade matrix update) |
| `apg.government.per.disciplinary.*` | `CaseOpened/Closed/Appealed` | `government_cas`, `audl` |
| `apg.government.per.leave.*` | `LeaveAccrued/Taken/Balanced` | Bytewax accrual dataflow |
| `apg.government.per.secondment.*` | `SecondmentActivated/Reverted` | Both agency payroll processors |
| `apg.government.per.performance.summary` | `AppraisalSummarised` | `intel` (attrition ML) |

## API Routes

| Path | Method | Description | Permission |
|---|---|---|---|
| `/government-per/appointments` | `GET/POST` | Employee appointments | `government_per:appoint` |
| `/government-per/payroll/runs` | `GET/POST` | Payroll run management | `government_per:payroll` |
| `/government-per/payroll/deductions` | `POST` | Statutory deduction computation | `government_per:payroll` |
| `/government-per/increments` | `POST` | Grade increment processing | `government_per:increment` |
| `/government-per/disciplinary` | `GET/POST` | Disciplinary case management | `government_per:disciplinary` |
| `/government-per/leave/balances` | `GET/POST` | Leave balance computation | `government_per:leave` |
| `/government-per/benefits/terminal` | `POST` | Terminal benefits calculation | `government_per:benefits` |
| `/government-per/secondments` | `GET/POST` | Secondment management | `government_per:secondment` |
| `/government-per/headcount` | `GET` | Workforce headcount report | `government_per:headcount` |
| `/government-per/appraisals` | `GET/POST` | Performance appraisal summary | `government_per:appraise` |
| `/government-per/applications` | `GET/POST` | Permit applications | `government_per:apply` |
| `/government-per/permits` | `GET` | Permit register | `government_per:permits` |
| `/government-per/inspections` | `GET/POST` | Inspection schedule | `government_per:inspect` |
| `/government-per/compliance` | `GET/POST` | Compliance monitoring | `government_per:compliance` |
| `/government-per/enforcement` | `GET/POST` | Enforcement actions | `government_per:enforce` |

## Business Rules

### HR Rules
| Rule | Condition | Effect |
|---|---|---|
| `appraisal_score_required_for_increment` | `appraisal_score < 3.0` | Increment withheld |
| `disciplinary_hold_blocks_increment` | Active disciplinary case | Increment suspended |
| `establishment_ceiling_enforced` | `headcount >= ceiling` | Appointment blocked |
| `payroll_requires_active_appointment` | No active appointment | Payroll excluded |
| `secondment_reversion_scheduled` | `reversion_date` set | Auto-reverts via Bytewax |
| `treasury_concurrence_required` | `new_gross > KES 50,000` | Flag for Treasury approval |

### Permit Rules (original)
| Rule | Condition | Effect |
|---|---|---|
| `application_fee_required` | `fee_paid=False` | deny |
| `construction_before_permit_denied` | `permit_active=False` | deny |
| `occupation_final_inspection_required` | `final_inspection_passed=False` | deny |
| `duplicate_permit_denied` | `duplicate_detected=True` | deny |

## New Service Methods (v1.1.0)

```python
# HR & Personnel
await svc.record_employee_appointment(employee_id, post_id, "permanent", dept_id, "D3", "2026-01-01")
await svc.process_payroll_run("2026-05", ["dept-001", "dept-002"])
await svc.open_disciplinary_case(employee_id, "Insubordination", complainant_id, ["evd-1"])
await svc.compute_leave_balance(employee_id, "annual", "2026-06-01")
await svc.calculate_terminal_benefits(employee_id, 30.5, 85_000.0, "cap189", 60)
await svc.process_grade_increment(employee_id, "D3", 4, 3.8, "2025-2026")
await svc.record_secondment(employee_id, "agency-A", "agency-B", "2026-07-01", "2027-06-30")
await svc.workforce_headcount_report(department_id="dept-001")
await svc.performance_appraisal_summary("dept-001", "2025-2026")
await svc.compute_statutory_deductions(employee_id, 75_000.0, "kenya_central")
```

## Data Models

### HR Models
- `Appointment`: `appointment_id, employee_id, post_id, appointment_type, department_id, grade, salary_step, effective_date, probation_end, status`
- `DisciplinaryCase`: `case_id, employee_id, allegation, severity, state, investigation_deadline, appeal_window_days`
- `LeaveBalance`: `employee_id, leave_type, available_balance, accrual_rate, as_at_date`
- `PayrollRun`: `run_id, pay_period, department_ids, run_type, status`
- `TerminalBenefits`: `employee_id, scheme, annual_pension, gratuity, commutation_lump_sum`
- `SecondmentRecord`: `secondment_id, employee_id, origin_agency_id, destination_agency_id, effective_date, reversion_date`

### Permit Models (original)
- `PermitApplication`, `Permit`, `PermitCondition`, `PermitInspection`, `ComplianceRecord`, `EnforcementAction`, `PermitReview`, `PermitsAgent`

## Composability Notes

Composes with:
- `government_bud` — Personnel emoluments vote check; payroll GL posting
- `government_lic` — Contractor licence verification before permit issuance
- `government_con` — Construction contracts reference building permits
- `government_cas` — Disciplinary cases linked to case management
- `government_trn` — Training needs fed from skills gap analysis
- `intel` — Workforce attrition ML models; permit pattern analysis

## Configuration

All configuration is tenant-scoped via the `conf` capability or `GOVERNMENT_PER_*` environment variables.

| Key | Description | Default |
|---|---|---|
| `governance.establishment_ceiling_enforced` | Block over-establishment appointments | `true` |
| `governance.treasury_concurrence_threshold` | Gross salary above which Treasury flag is raised | `50000` |
| `governance.increment_pass_threshold` | Minimum appraisal score for increment | `3.0` |
| `governance.disciplinary_appeal_window_days` | Default appeal window for disciplinary outcomes | `14` |
| `payroll.jurisdiction` | Statutory deduction jurisdiction | `kenya_central` |
| `pension.default_scheme` | Default pension scheme | `cap189` |
| `governance.construction_before_permit_denied` | Commencement without permit always blocked | `true` |
| `governance.occupation_before_final_inspection_denied` | Occupation requires final inspection pass | `true` |

## World-Class Improvements

See `WORLD_CLASS_IMPROVEMENTS.md` for 15 prioritised enhancements including:
- AI-powered performance prediction & succession planning (Ollama/Mistral)
- Real-time payroll streaming (NATS + Bytewax)
- Due-process disciplinary FSM
- Multi-jurisdiction statutory deductions engine
- Predictive attrition modelling (HR analytics dashboard)
