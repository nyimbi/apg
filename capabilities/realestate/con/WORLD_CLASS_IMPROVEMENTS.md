# Construction Management (realestate_con) — World-Class Improvements

### I1. Real-Time Defect Snagging with Photo AI Triage
**Category**: AI/Quality Assurance
**Justification**: Current defect tracking is text-only. Photo evidence attached to snag items should auto-classify severity (critical/major/minor), tag the trade responsible (electrical/plumbing/finishes), and generate a resolution SLA — cutting QS inspection time by 80%.
**Implementation**: On snag item creation, forward attached image refs to a locally-hosted vision model (via Ollama `llava` or `moondream`). Model returns severity, trade category, and estimated rectification effort. Results stored on `SnagItem` and surface in the snagging dashboard without round-tripping to a cloud API.
**Competitor**: Procore Inspections, Snagr, PlanGrid Punch List

---

### I2. Critical-Path Schedule Engine with Float Monitoring
**Category**: Project Management
**Justification**: Milestones today are independent records with no dependency graph. A CPM engine that tracks total float, free float, and critical-path tasks lets PMs spot schedule slippage weeks before it hits the programme, reducing delay claims by 40-60%.
**Implementation**: Add `predecessor_ids`, `successor_ids`, and `lag_days` to `MilestoneCreate`. A dedicated `CriticalPathService` uses a topological sort + forward/backward pass (ES/EF/LS/LF) on the milestone DAG. Float values are recomputed async on every milestone status change and published as NATS events on subject `con.schedule.float_alert`.
**Competitor**: Primavera P6, MS Project, Procore Schedule

---

### I3. Earned Value Management (EVM) Dashboard
**Category**: Cost Control / Analytics
**Justification**: Budget vs. actual spend is not currently tracked as a time-series. EVM metrics (PV, EV, AC, CPI, SPI) give site managers a single number to judge project health and feed investor reporting — the absence of this is why most construction projects finish 20%+ over budget.
**Implementation**: Introduce `ProgressSnapshot` model (period, planned_value, earned_value, actual_cost, budget_at_completion). `calculate_evm` async method computes CPI/SPI/EAC/VAC from stored snapshots. Integration with `realestate_acc` pulls certified payment postings as AC. NATS event `con.evm.variance_alert` fires when CPI < 0.85.
**Competitor**: Oracle Primavera Unifier, e-Builder, Kahua

---

### I4. Automated NEC/JBCC/FIDIC Contract Clause Compliance Checker
**Category**: Legal / Compliance
**Justification**: Construction contracts in Kenya (JBCC, NEC3/4, FIDIC Red/Yellow) have mandatory notice periods, compensation event windows, and programme submission deadlines. Missing them is the leading cause of contractor claims. Auto-detecting approaching deadlines from clause metadata eliminates manual calendar tracking.
**Implementation**: Clause library gains `contract_standard` (NEC4/JBCC/FIDIC), `notice_period_days`, and `trigger_condition` fields. A scheduled async job (`check_clause_deadlines`) scans active contracts nightly, computes trigger dates from clause metadata, and publishes NATS events `con.clause.deadline_approaching` with T-14, T-7, T-1 horizon alerts.
**Competitor**: Exari, Icertis, Kira Systems

---

### I5. Subcontractor Work Package & Back-to-Back Contract Linking
**Category**: Supply Chain / Subcontracting
**Justification**: Main contracts and subcontracts are currently unlinked. Back-to-back linking lets the system enforce that subcontract payment terms mirror main contract terms, propagate variation orders downstream automatically, and flag when subcontractor scope creep creates main-contract exposure.
**Implementation**: Add `parent_contract_id` and `back_to_back` flag to `ContractCreate`. New `propagate_variation_downstream` async method fans out approved VOs from main contract to all linked subcontracts, applying a configurable markup factor. NATS events chain: `con.variation.approved` → `con.subcontract.vo_propagated`.
**Competitor**: Procore Commitments, Sage 300 Construction, Viewpoint

---

### I6. Contractor Performance Scorecard with Weighted KPIs
**Category**: Contractor Management
**Justification**: `ContractorResponse.performance_score` is a single decimal with no audit trail. A multi-axis scorecard (quality 30%, programme 25%, safety 20%, commercial 15%, sustainability 10%) with per-contract history enables data-driven grading and shortlisting, reducing contractor-related defects by 35%.
**Implementation**: `ContractorScorecard` model stores per-contract scores by axis. `compute_contractor_scorecard` async method aggregates across completed contracts with configurable weights. Rolling 24-month window with exponential time-decay weighting. Grade changes triggered automatically when rolling score crosses grade thresholds. Scores published via NATS `con.contractor.score_updated`.
**Competitor**: Achilles, Builders Profile, Jaggaer Supplier Performance

---

### I7. Payment Certificate Workflow with Cashflow Forecasting
**Category**: Financial Management
**Justification**: Interim payment certificates (IPCs) drive construction cashflow but are currently implicit in milestones. Formalising the IPC workflow (application → certification → payment → final account) with linked cashflow S-curves gives developers, lenders, and PMs a live view of committed spend vs. planned disbursement.
**Implementation**: `PaymentCertificate` model (application_ref, period_end, gross_value, retention_deduction, variations_included, net_certified, status). `issue_payment_certificate` and `certify_payment_application` async methods. `generate_cashflow_forecast` async method produces monthly PV/EV/AC series up to contract completion. Integration with `realestate_acc` posts certified amounts as AP invoices.
**Competitor**: Procore Finance, Aconex Payment, Causeway Tradex

---

### I8. Defect Liability Period (DLP) Tracker with Automated Closure
**Category**: Post-Completion / Quality
**Justification**: DLP management is manual today. Missed DLP expiry dates mean retention moneys sit unreleased costing contractors working capital, and unresolved defects at expiry become the developer's cost. Automated DLP tracking closes the gap between practical completion and final account.
**Implementation**: `DefectLiabilityRecord` model per contract with `dlp_start`, `dlp_end`, linked `snag_items`, and `outstanding_count`. `check_dlp_expiry` nightly job publishes NATS events at T-30/T-14/T-1. `close_dlp` async method validates zero outstanding snags before clearing defect liability (prerequisite to `release_retention`).
**Competitor**: PlanGrid, SnagR, Procore Closeout

---

### I9. NATS-Based Real-Time Event Stream for Construction Events
**Category**: Integration / Streaming
**Justification**: All construction events (VO approval, milestone completion, default notice, dispute) should be available to downstream systems (accounting, reporting, mobile site apps) in sub-second latency. NATS JetStream provides durable, at-least-once delivery at far lower operational overhead than self-hosted alternatives.
**Implementation**: `ConEventPublisher` wraps `nats.py` client. On every mutating service method, publish a typed CloudEvent to subjects `con.contract.*`, `con.milestone.*`, `con.variation.*`, `con.dispute.*`, `con.snag.*`. Schema registry in `domain/events.py`. Consumer groups for `acc` (payment posting), `ntfy` (notifications), and `audit` (immutable log).
**Competitor**: Kafka + Procore Webhooks, Aconex Notifications, Oracle Event Hub

---

### I10. Risk Register with Monte Carlo Schedule/Cost Simulation
**Category**: Risk Management
**Justification**: Construction risks (ground conditions, supply chain, weather, regulatory) are unmodelled. A risk register with likelihood/impact scoring, linked to schedule and cost items, combined with Monte Carlo simulation, produces P50/P80/P90 cost and completion date forecasts demanded by lenders and insurers.
**Implementation**: `RiskItem` model (risk_id, category, probability, impact_cost, impact_days, mitigation_action, owner). `run_monte_carlo_simulation` async method performs N=10,000 trials (configurable), sampling triangular distributions for each risk item. Returns P50/P80/P90 completion date and cost distributions. Computationally intensive work offloaded to async executor (`asyncio.run_in_executor`).
**Competitor**: Safran Risk, ARM (Active Risk Manager), Oracle Risk Management

---

### I11. BIM/IFC Document Integration with Drawing Register
**Category**: Document Management
**Justification**: Construction projects generate thousands of drawings, specs, and RFIs. A drawing register that tracks revisions, superseded drawings, and IFC model linkages prevents construction from proceeding on out-of-date information — the root cause of 30% of rework.
**Implementation**: `Drawing` model (drawing_number, revision, title, discipline, document_id, superseded_by, ifc_element_ids). `register_drawing` and `supersede_drawing` async methods. `get_current_drawing_set` returns only the latest revision per drawing number. NATS event `con.drawing.superseded` alerts field teams. IFC element IDs enable future BIM viewer integration.
**Competitor**: Aconex, Autodesk Construction Cloud, Procore Documents

---

### I12. Delay Analysis Engine (As-Planned vs. As-Built)
**Category**: Claims Management
**Justification**: Extension of time (EOT) claims require contemporaneous delay analysis. An automated impacted as-planned analysis compares the baseline programme to actual milestone completion dates, identifies delay events, and attributes responsibility — reducing claim preparation from weeks to hours.
**Implementation**: `DelayEvent` model (event_type: employer_risk/neutral_risk/contractor_risk, impact_days, linked_milestone_ids, causation_narrative). `analyse_delays` async method performs impacted-as-planned analysis: reconstructs the as-built programme from milestone completion dates, computes delay float per critical path activity, and categorises by risk ownership. Output feeds `extension_of_time_claim` generation.
**Competitor**: Oracle Primavera Claim Digger, Deltek Acumen Fuse, 4c Claims

---

### I13. Automated Quantity Surveying (QS) Cost Benchmarking
**Category**: Cost Management / AI
**Justification**: Budget estimates and BoQ items lack market benchmarks. Comparing approved contract rates to regional benchmarks (from a locally stored schedule of rates) flags over/under-pricing at tender stage and provides audit evidence for procurement compliance.
**Implementation**: `BoQItem` model (item_code, description, unit, qty, rate, amount, benchmark_rate, variance_pct). `benchmark_boq_rates` async method loads the current schedule of rates from `conf` capability, computes variance per item, and flags items outside ±15% tolerance. Ollama `mistral` model used to generate narrative commentary on significant variances. Results stored as `CostBenchmarkReport`.
**Competitor**: BCIS (RICS), Uniformat, Autodesk Insight for Cost

---

### I14. Multi-Party Notice Management with Tracked Delivery
**Category**: Legal / Notices
**Justification**: Construction contracts require formal notices (compensation events, early warnings, default notices) to be served within strict contractual windows. Manual email tracking is inadmissible as evidence. A notice registry with delivery confirmation and contractual clock management provides a complete audit trail.
**Implementation**: `FormalNotice` model (notice_type, contractual_basis, served_by, served_on, delivery_method: registered_post/email/hand, acknowledgement_ref, contractual_deadline, response_deadline). `serve_notice` async method records notice, starts response clock. `check_notice_responses` nightly job fires NATS event `con.notice.response_overdue` when response deadline passed without acknowledgement.
**Competitor**: Procore Correspondence, Aconex Workflow, Exigent Contract Intelligence

---

### I15. Snagging-to-Handover Digital Certificate Workflow
**Category**: Handover / Commissioning
**Justification**: Practical completion (PC) certificates are the financial trigger for 50%+ of retention release and final payments. The current flow has no formal checklist-to-certificate pipeline. A structured workflow from outstanding snag count zero → commissioning sign-off → PC certificate issuance → DLP start eliminates disputes over completion date.
**Implementation**: `HandoverChecklist` model (sections: snags/commissioning/documentation/O&M-manuals, section_status, sign_off_by). `generate_pc_certificate` async method validates all checklist sections complete and snag count is zero before issuing `PracticalCompletionCertificate` (certificate_ref, issued_date, dlp_start_date, retention_release_trigger). Published as NATS event `con.handover.pc_certificate_issued`, triggering `realestate_acc` retention release workflow.
**Competitor**: Procore Closeout, PlanGrid, Fieldwire Handover
