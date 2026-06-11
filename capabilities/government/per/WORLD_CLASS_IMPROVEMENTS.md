# Personnel & HR (government_per) — World-Class Improvements

Capability: Civil service HR, payroll integration, performance management, disciplinary
Path: `capabilities/government/per`

---

### I1. AI-Powered Performance Prediction & Succession Planning
**Category**: Machine Learning / Workforce Intelligence
**Justification**: Static annual appraisals miss 80% of early performance signals. Continuous ML scoring against structured KPIs, peer feedback vectors, and leave/absence patterns gives managers a 6-month predictive horizon — reducing surprise vacancies by 60% in comparable public sector deployments.
**Implementation**: Integrate Ollama-hosted `mistral:7b-instruct` scoring via `capabilities/common/mlx.MLCapability`. Feed structured employee activity events through NATS subjects `apg.government.per.performance.*`. Produce `succession_risk_score` (0–1) per employee, surfaced in the FAB dashboard with drill-down.
**Competitor**: Workday HCM Workforce Planning, SAP SuccessFactors Talent Intelligence Hub

---

### I2. Real-Time Payroll Event Streaming via NATS + Bytewax
**Category**: Streaming Architecture / Payroll Integration
**Justification**: Batch payroll runs (monthly/fortnightly) create reconciliation debt. Publishing immutable payroll events to NATS `apg.government.per.payroll.*` and processing them through a Bytewax pipeline enables sub-second GL posting, real-time payslip generation, and upstream budget debit — eliminating the 3–5 day payroll reconciliation cycle common in government.
**Implementation**: `PayrollEventPublisher` publishes `PayrollRunInitiated`, `PayslipGenerated`, `PaymentPosted` events. Bytewax dataflow subscribes to NATS and emits to `government_bud` (budget debit) and `intel` (workforce cost analytics). Replaces any Kafka-based equivalent with NATS JetStream for durability.
**Competitor**: Oracle HCM Cloud Payroll, ADP Government Payroll (uses Kafka internally)

---

### I3. Disciplinary Case Lifecycle with Due-Process Guardrails
**Category**: Compliance / Case Management
**Justification**: Employment tribunals cite procedural failures (missing notice periods, inadequate hearing records) in 72% of overturned public sector dismissals. A structured finite-state machine for disciplinary cases — enforcing mandatory waiting periods, evidence attachments, and appeal windows — cuts tribunal liability exposure significantly.
**Implementation**: `DisciplinaryCase` FSM with states: `allegation_raised → investigation → hearing_scheduled → hearing_held → outcome_issued → appeal_window → closed`. Each transition enforces time-box rules from the `conf` capability. Events emitted to NATS `apg.government.per.disciplinary.*`. Appeal window defaults to 14 days, configurable per jurisdiction.
**Competitor**: Cornerstone OnDemand Employee Relations, ServiceNow HR Service Delivery

---

### I4. Civil Service Grade & Increment Management
**Category**: Compensation Management
**Justification**: Government pay scales (Job Groups A–T in Kenya, Pay Bands in UK Civil Service) are rule-bound — increments are date-triggered, subject to satisfactory appraisal, and require Treasury approval above a threshold. Encoding these rules in the service layer prevents manual errors and enables automatic payroll adjustment notifications.
**Implementation**: `GradeScheme` pydantic model holding band/step/salary matrix. `process_increment()` validates appraisal score ≥ threshold, confirms no active disciplinary hold, then emits `SalaryIncrementApproved` to NATS. Treasury approval workflow triggered for increments above configurable threshold.
**Competitor**: NeoGov Perform (US civil service), Sopra HR Software (EU public sector)

---

### I5. Leave & Absence Management with Accrual Engine
**Category**: Time & Attendance
**Justification**: Manual leave registers produce accrual errors costing the Kenya government an estimated KES 2.3B annually in overpaid terminal benefits. An automated accrual engine that computes leave balances in real time — factoring in carry-over caps, public holidays, and part-time FTE fractions — eliminates this exposure.
**Implementation**: `LeaveBalance` per employee per leave type (annual, sick, maternity, compassionate, study). Accrual events fire daily via a Bytewax time-windowed operator on NATS `apg.government.per.leave.*`. `LeaveRequest` FSM: `submitted → approved/rejected → taken → balance_updated`. Integrates with `schd` for shift-aware calculation.
**Competitor**: Ceridian Dayforce, Sage HR (both use event-driven accrual)

---

### I6. Workforce Headcount & Establishment Control
**Category**: Workforce Planning / Budgetary Control
**Justification**: Ghost worker fraud (an estimated 1–3% payroll leakage in African public sectors) originates from establishment mismanagement — posts filled beyond approved headcount or against non-existent posts. Hard enforcement of approved establishment ceilings blocks both fraud and budget overruns.
**Implementation**: `EstablishedPost` model with `establishment_ceiling` per department/grade. `recruit_to_post()` checks `current_headcount < establishment_ceiling` before creating appointment. Over-establishment attempts are blocked and emitted as `EstablishmentBreachAttempted` events to NATS for audit. Integration with `government_bud` for automatic personnel emoluments vote check.
**Competitor**: Workday Workforce Planning, HRMIS (Kenya Government IPPD system)

---

### I7. Multi-Jurisdiction Statutory Deductions Engine
**Category**: Payroll Compliance
**Justification**: Civil servants often operate across counties/agencies with different deduction rules (NHIF, NSSF, housing levy, HELB, SACCO loans, court orders). A declarative deductions engine that applies jurisdiction-specific rules via pluggable `DeductionRule` objects eliminates hard-coded tax tables and enables same-day compliance with statutory changes.
**Implementation**: `DeductionRule` ABC with `compute(gross_pay, employee_profile) -> Decimal`. Built-in implementations: `PAYERule`, `NHIFRule`, `NSSFRule`, `HousingLevyRule`. Rules loaded from `conf` capability at runtime — no code changes for rate updates. Net pay events published to NATS `apg.government.per.payroll.netpay`.
**Competitor**: Sage Payroll, QuickBooks Payroll (both use pluggable tax rule engines)

---

### I8. Skills Inventory & Training Needs Analysis
**Category**: Learning & Development
**Justification**: Civil service capacity audits (e.g., Kenya State Corporations Advisory Committee reports) consistently find skills gaps costing 30–40% productivity loss. A structured skills inventory with proficiency scoring and automatic training-gap detection feeds directly into training budget allocation, replacing guesswork with data.
**Implementation**: `EmployeeSkill` model: `employee_id, skill_code, proficiency_level (1–5), last_assessed, evidence_ref`. `compute_training_needs()` diffs employee profile against post competency framework. Output feeds `government_trn` (training capability) via NATS event `TrainingNeedIdentified`. Skills taxonomy loaded from `conf`.
**Competitor**: LinkedIn Learning (Skills Graph), SAP SuccessFactors Learning

---

### I9. Contract & Appointment Lifecycle Management
**Category**: Employee Lifecycle
**Justification**: Public service appointments (permanent, contract, secondment, acting) each have distinct statutory requirements for gazette notification, probation, confirmation, and termination. A typed appointment FSM prevents illegal terminations (a significant source of tribunal cases) and automates gazette notification triggers.
**Implementation**: `Appointment` model with `appointment_type` enum: `permanent | contract | secondment | acting | internship`. FSM transitions enforce statutory timelines from the Public Service Act. Probation confirmation events trigger `ProbationConfirmationDue` 30 days before deadline via NATS scheduled message.
**Competitor**: PeopleSoft HCM (Oracle), Unit4 People Planning

---

### I10. Grievance & Whistleblower Case Management
**Category**: Employee Relations / Compliance
**Justification**: The Kenyan Whistleblower Protection Act 2010 and Employment Act require structured, time-bound grievance handling. Untracked grievances create legal liability. A case management system with anonymization support, SLA timers, and escalation paths demonstrates compliance and reduces tribunal exposure.
**Implementation**: `GrievanceCase` model with optional `anonymous: bool`. SLA timers enforced via Bytewax event-time processing on NATS `apg.government.per.grievance.*`. Automatic escalation on SLA breach: `GrievanceSLABreached` event triggers notification to HR Director. Integration with `government_cas` for cross-domain case linkage.
**Competitor**: Case IQ (formerly i-Sight), Navex EthicsPoint

---

### I11. Integrated Organizational Chart with Real-Time Headcount
**Category**: Organization Management / Visualization
**Justification**: Civil service org charts in most African governments are static PDFs updated annually. A live org chart backed by appointment data enables real-time span-of-control analysis, identifies acting/vacant posts instantly, and feeds establishment control (I6) with accurate structural data.
**Implementation**: `OrgNode` model: `post_id, parent_post_id, holder_employee_id | None, acting_holder | None, establishment_count`. Materialized as an adjacency-list in PostgreSQL with recursive CTE queries. REST endpoint `/government-per/org-chart` returns D3.js-consumable tree JSON. Headcount changes publish `OrgChartUpdated` to NATS.
**Competitor**: Workday Org Studio, BambooHR Org Chart (both real-time)

---

### I12. Automated Payslip Generation & Distribution
**Category**: Employee Self-Service
**Justification**: Paper payslip distribution costs the government an estimated KES 180M/year in printing and postage. Digital payslips with cryptographic signing (proving they haven't been tampered with — critical for mortgage applications) eliminate this cost and increase trust.
**Implementation**: `PayslipDocument` generated by Jinja2 template + WeasyPrint (HTML→PDF). Each PDF signed with the tenant's X.509 certificate via `cryptography` library. Signed payslip hash recorded on audit ledger via `audl`. Distribution event published to NATS `apg.government.per.payslip.distributed` for email/SMS via `ntfy`.
**Competitor**: ADP MyPay, Workday Pay (both sign payslips digitally)

---

### I13. Cross-Agency Secondment & Transfer Management
**Category**: Mobility / Workforce Sharing
**Justification**: Civil service secondments are administratively painful — requiring manual interagency correspondence, dual-payroll risk, and lost service continuity. A structured transfer protocol with automatic payroll handoff events eliminates dual-payment and ensures benefits continuity without manual intervention.
**Implementation**: `SecondmentRecord` model with `origin_agency_id, destination_agency_id, effective_date, reversion_date`. Payroll responsibility transfer triggered by `SecondmentActivated` NATS event — origin agency payroll ceases, destination begins, both receive audit event. Reversion is date-scheduled via Bytewax time trigger.
**Competitor**: SAP HCM Cross-Company Transfers, Workday Global HCM

---

### I14. Retirement & Terminal Benefits Calculator
**Category**: Benefits Administration
**Justification**: Manual pension calculations are the single largest source of civil servant complaints in Kenya (PSC Annual Report 2024: 34% of all complaints). An automated benefits calculator covering the Pensions Act (Cap 189), NPF Act, and LGPS rules produces audit-ready computation sheets, reducing complaints and eliminating calculation errors.
**Implementation**: `PensionCalculator` service: inputs `years_of_service, final_salary, scheme (cap189 | npf | lgps), age`. Outputs `gratuity_amount, monthly_pension, lump_sum, commutation_option`. Computation parameters loaded from `conf` (factor tables, accrual rates). Computation events emitted to NATS for downstream `government_fin` audit.
**Competitor**: Buck Consultants Pension Administration, Mercer Pension Manager

---

### I15. HR Analytics Dashboard with Predictive Attrition Modelling
**Category**: People Analytics
**Justification**: Civil service attrition analysis is typically backward-looking (exit interviews). Predictive models trained on leave patterns, grievance history, salary progression, and promotion velocity can identify flight-risk employees 6–12 months before resignation, enabling targeted retention interventions.
**Implementation**: Aggregate anonymized signals into `EmployeeRiskVector` (leave_days_ytd, grievance_count, months_since_promotion, salary_percentile_in_grade). Score via Ollama `phi3:mini` fine-tuned on civil service attrition datasets. Risk scores refreshed weekly via Bytewax windowed aggregation on NATS. Dashboard widget shows department-level attrition risk heat map.
**Competitor**: Visier People Analytics, IBM Kenexa (both use ML attrition prediction)
