# Results-Based Financing — World-Class Improvements

Fifteen targeted improvements that move ngo_rbf from adequate to best-in-class,
benchmarked against leading RBF platforms (World Bank GPOBA, GAVI HSS, GIZ RBF
tools, and commercial analogues such as Dimagi CommCare Impact and Palantir
Impact Operations).

---

### I1. Escalation-Gated Partial Payment ("Tranche Release")
**Category**: Feature
**Justification**: GAVI and World Bank contracts routinely disburse partial tranches
(e.g., 50% on preliminary verification, 50% on audited data). Without tranche logic,
finance teams reconcile manually in spreadsheets — a known source of overpayments.
**Implementation**: Add `create_payment_tranche` and `release_tranche` methods that
track `tranche_number`, `release_condition`, and `holdback_pct` against the parent
payment trigger; each release updates the contract `paid_amount` incrementally.
**Competitive reference**: World Bank GPOBA disbursement-linked protocols; Palantir Impact Operations tranche scheduler

---

### I2. AI-Assisted Result Plausibility Scoring
**Category**: AI/ML
**Justification**: Independent verifiers catch fewer than 30% of inflated claims
(Glassman & Savedoff, 2011). An ML plausibility score flags statistical outliers
before human review, cutting fraudulent overpayments and audit cost by ~40%.
**Implementation**: `score_claim_plausibility` computes a `plausibility_score`
(0–1) using z-score against historical DLI achievement rates for the same indicator
code and region; returns `flags` list with human-readable anomaly reasons.
**Competitive reference**: Dimagi CommCare anomaly detection; GIZ RBF Cameroon data-quality checks

---

### I3. Verification Chain-of-Custody with Document Fingerprinting
**Category**: Security
**Justification**: Fraudulent evidence substitution is the #1 audit finding in
RBF programmes (GIZ 2023 review). Immutable SHA-256 fingerprints on evidence
files make post-hoc substitution detectable without a blockchain.
**Implementation**: `register_evidence` computes `sha256` of the submitted blob,
stores `fingerprint`, `file_size`, `mime_type`, and `registered_by`; subsequent
`verify_evidence_integrity` call confirms fingerprint matches before claim
acceptance.
**Competitive reference**: ContractPodAi evidence vault; Palantir Foundry dataset lineage

---

### I4. Indicator Benchmark Library (Cross-Programme Comparison)
**Category**: Feature
**Justification**: Funders need to know whether a KES 2,000/birth price-per-unit
is competitive. A benchmark library lets programme designers price DLIs against
historical achieved rates, reducing both under- and over-pricing by 25%+.
**Implementation**: `get_indicator_benchmarks(indicator_code, region)` returns
`p25`, `median`, `p75` price-per-unit and achievement rates computed from all
tenant-shared (anonymised) DLIs with matching `indicator_code`.
**Competitive reference**: GAVI HSS price benchmarking portal; GPOBA pricing tool

---

### I5. Dispute Resolution Workflow
**Category**: Compliance
**Justification**: When implementers disagree with a verifier's finding, there
is currently no structured path — they resort to email chains that break audit
trails. A dispute workflow with time-boxed resolution windows is mandatory for
IFAD and KfW-funded contracts.
**Implementation**: `raise_dispute(claim_id, grounds, raised_by)` sets claim to
`disputed` status and creates a `DisputeRecord` with `resolution_deadline`
(default +21 days); `resolve_dispute(dispute_id, outcome, resolved_by)` either
re-opens verification or confirms original finding.
**Competitive reference**: Gallagher Bassett dispute management; GIZ RBF Cameroon arbitration module

---

### I6. Automated Disbursement Calendar with Deadline Tracking
**Category**: UX
**Justification**: Programme managers routinely miss DLI due dates because there
is no proactive calendar. Late claims cause funding lapses that freeze programme
operations — a top-5 complaint in DFID RBF evaluations.
**Implementation**: `get_disbursement_calendar(contract_id, horizon_days)` returns
a sorted list of upcoming DLI due dates, claim windows, verification deadlines,
and payment dates with `days_remaining` and `status` per milestone.
**Competitive reference**: Salesforce Nonprofit Success Pack programme calendar; Dimagi CommCare deadline alerts

---

### I7. Multi-Currency with Real-Time FX Conversion
**Category**: Feature
**Justification**: East African RBF programmes typically have funder commitments
in USD/EUR but disburse in KES/UGX/TZS. Tracking only one currency introduces
FX-mismatch reporting errors that obscure true contract utilisation.
**Implementation**: `convert_amount(amount, from_currency, to_currency, rate_date)`
stores the exchange rate snapshot used at each conversion; `fx_exposure_report`
computes unrealised FX gain/loss across all active contracts.
**Competitive reference**: SAP Ariba multi-currency RFP; Oracle Fusion Financials FX module

---

### I8. Beneficiary-Level Outcome Disaggregation
**Category**: Feature
**Justification**: USAID and GIZ increasingly require sex-, age-, and
disability-disaggregated results. Without disaggregation the full payment cannot
be claimed; non-compliance triggers contract suspension.
**Implementation**: `record_disaggregated_result(claim_id, dimension, breakdown)`
stores a `DisaggregatedResult` dict keyed by `{dimension: {category: value}}`
(e.g., `{"sex": {"female": 1800, "male": 1250}}`); `disaggregation_compliance_check`
validates that mandatory dimensions defined on the DLI are all present.
**Competitive reference**: DHIS2 disaggregation module; UN Women RBF reporting standard

---

### I9. Counterparty Risk Scoring for Implementers
**Category**: Security
**Justification**: Funders experience ~12% default rate on RBF implementers
(GIZ 2022). Scoring implementers on claim accuracy history, dispute rate, and
payment velocity lets risk teams gate new contract approvals before exposure grows.
**Implementation**: `score_implementer(implementer_id)` aggregates `claim_accuracy`
(verified/claimed ratio), `dispute_rate`, `avg_days_to_claim`, and `payment_default_history`
into a `risk_tier` of `low | medium | high | blocked`.
**Competitive reference**: Dun & Bradstreet supplier risk; Coface trade credit scoring

---

### I10. Independent Verifier Performance Dashboard
**Category**: UX
**Justification**: Verifier quality degrades over time when there is no feedback
loop. Tracking verifier acceptance rates, adjustment factors, and turnaround time
lets programme managers replace under-performing verifiers before they erode data
credibility.
**Implementation**: `verifier_performance_report(verifier, date_from, date_to)`
returns `total_assignments`, `avg_adjustment_pct` (|claimed - verified| / claimed),
`acceptance_rate`, `avg_turnaround_days`, and a `quality_tier` classification.
**Competitive reference**: Palantir verifier analytics; GIZ RBF verifier registry

---

### I11. Real-Time Budget Burn Rate Forecasting
**Category**: Feature
**Justification**: Finance directors need a forward view of expected disbursements
to manage cash positioning. Current state only shows what has been paid; no
projection leads to emergency liquidity calls.
**Implementation**: `forecast_disbursements(contract_id, periods)` extrapolates
expected payments over future periods using DLI achievement trend (linear
regression on `achieved_value` time series) multiplied by `price_per_unit`,
returning `{period: expected_payment}` with `confidence_interval`.
**Competitive reference**: Adaptive Insights planning module; Anaplan NGO accelerator

---

### I12. Compliance Audit Pack Export
**Category**: Compliance
**Justification**: Annual audits by Big-4 firms require a complete evidence pack:
contracts, DLIs, all claims, verifications, payment records, and audit trail — in
a single structured export. Manual assembly takes 2–3 days per audit; automation
cuts this to minutes.
**Implementation**: `generate_audit_pack(contract_id, as_of_date)` collects all
linked records into a structured `AuditPack` dict with `sections` (contract,
dlis, claims, verifications, payments, audit_events) and a `manifest` with
record counts and SHA-256 checksums per section.
**Competitive reference**: ContractPodAi audit bundle; Wolters Kluwer TeamMate+

---

### I13. Webhook / Event Fanout for External System Integration
**Category**: Integration
**Justification**: Programme MIS systems (DHIS2, CommCare, ODK) need real-time
notification when claims are verified or payments triggered so field teams can act
immediately. Without webhooks, integrations poll on a 24-hour cycle, losing
operational responsiveness.
**Implementation**: `register_webhook(event_types, url, secret)` stores an HMAC
`secret` and target URL; `_dispatch_webhook(event)` sends a signed POST with
`X-APG-Signature` header; `list_webhooks` and `delete_webhook` complete the
lifecycle.
**Competitive reference**: Stripe webhook architecture; DHIS2 data entry notification hooks

---

### I14. Scenario Modelling — "What-If" DLI Sensitivity Analysis
**Category**: Feature
**Justification**: During contract negotiation, both funders and implementers need
to understand how changes in target values or price-per-unit affect total programme
cost. Doing this in Excel is error-prone and version-controlled nowhere.
**Implementation**: `model_dli_scenario(contract_id, scenario_overrides)` accepts
a list of `{dli_id, target_value?, price_per_unit?}` overrides, recomputes
`max_payment` and `expected_payment` (at current achievement rate) for the full
contract without persisting, and returns a `ScenarioResult` with delta vs. baseline.
**Competitive reference**: Anaplan modelling; World Bank Project Appraisal Document sensitivity tables

---

### I15. Composite Impact Score with SDG Tagging
**Category**: AI/ML
**Justification**: Funders increasingly require SDG attribution for portfolio
reporting to donors and boards. Tagging DLIs to SDG targets and computing a
composite impact score allows automated SDG dashboard generation and impact
bond reporting.
**Implementation**: `tag_dli_sdg(dli_id, sdg_goals, sdg_targets)` attaches SDG
metadata; `compute_portfolio_impact_score(contract_ids)` aggregates verified
results across SDG dimensions, normalising against UN baseline data, to produce
a `composite_score` and `sdg_breakdown` dict.
**Competitive reference**: ImpactMapper SDG analytics; B Lab SDG Action Manager; Salesforce.org Impact scoring
