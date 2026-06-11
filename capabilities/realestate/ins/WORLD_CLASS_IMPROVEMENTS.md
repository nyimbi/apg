# Property Insurance (realestate_ins) — World Class Improvements

**Capability**: `realestate_ins` | **Version**: 1.0.0 → 2.0.0

---

## 1. Parametric Insurance Triggers

Replace manual claim lodgement for catastrophe perils (flood, earthquake, wind) with oracle-driven parametric triggers. When a sensor reading or third-party data feed (e.g., rainfall index, seismic magnitude) exceeds a pre-agreed threshold, claims are auto-lodged and pre-approved without loss adjuster involvement. Eliminates 4–6 week settlement lag for nat-cat events. Implement as `parametric_trigger_evaluate()` consuming an event payload from `mqeb`.

## 2. AI-Powered Under-Insurance Detection

Replace the static 90%-of-rebuild-cost heuristic with a valuation model that ingests construction cost indices (BCIS, Rawlinson's), inflation rates, and comparable property data from `realestate_val`. Run quarterly. Flag properties where declared sum insured has drifted more than 15% below current reinstatement estimate and auto-generate an endorsement request. Reduces average portfolio under-insurance from 35% to under 5%.

## 3. Real-Time Insurer Solvency Monitoring

Currently `InsurerGrade` is set manually. Integrate IRA (Insurance Regulatory Authority) and AM Best API feeds to automatically update insurer grade when solvency margin drops below threshold, triggers insurer suspension and initiates policy rebroking workflow. Prevents concentration risk from writing premiums to an about-to-fail insurer. Implement as `sync_insurer_solvency_ratings()` on a daily cron.

## 4. Claims Fraud Detection Engine

Apply ML anomaly detection on claim patterns: duplicate incident dates across tenants, claimed asset not on schedule at incident date, estimated loss > insured value, loss adjuster appointed by claimant. Score each claim 0–100 on submission; route high-score claims to senior adjuster automatically. Expected fraud savings: 8–12% of settled claims. New method: `score_claim_fraud_risk()`.

## 5. Structured Subrogation Workflow

Once a claim is settled, if the loss was caused by a third party (contractor negligence, tenant damage), automatically open a subrogation recovery file, assign to legal team, track correspondence and recovery amounts. Reduce net claim cost by typical 15–20% subrogation recovery rate. Implement `initiate_subrogation()` and `record_subrogation_recovery()`.

## 6. Multi-Layer Reinsurance Treaty Modelling

Model quota-share and excess-of-loss reinsurance treaties at portfolio level. For each claim, calculate gross and net (post-reinsurance) exposure automatically, apply reinstatement premiums, track reinstatement provisions exhaustion. Essential for self-insured retention (SIR) and captive management. New models: `ReinsuranceTreaty`, `ReinsuranceLayer`. Method: `apply_reinsurance_recoveries()`.

## 7. Dynamic Premium Rating Engine

Replace flat `annual_premium` field with an actuarial rating workflow: base rate tables by peril × construction type × location, apply experience rating adjustment (claims history multiplier), apply portfolio discount for multi-property. Produces a structured premium schedule rather than a single figure. Auditable, defensible, and eliminates broker over-discount. Method: `compute_actuarial_premium()`.

## 8. Compliance Certificate Automation

Auto-generate IRA-compliant policy certificates, mortgage lender endorsement certificates, and loss payee clauses on demand in PDF/HTML format. Pull data from `PolicyResponse` + `InsurerResponse`, apply tenant branding, watermark draft vs. issued. Eliminates 2-day turnaround for certificate requests. Method: `issue_certificate()` returning a structured document payload for `docx`/`pdf` skill rendering.

## 9. Portfolio Stress Testing

Run scenario analysis: "what if a 1-in-100-year flood hits all properties in Westlands?" Apply PML (Probable Maximum Loss) factors by peril × location, sum exposures, compare to reinsurance recoveries. Output maximum net retained loss per scenario. Answers the CFO question before the event. Method: `run_portfolio_stress_test()`.

## 10. Renewal Automation Workflow

Extend `get_renewal_pipeline()` from read-only to action-triggering: at 90 days send broker RFQ, at 60 days collate quotes into `market_options`, at 30 days trigger approval workflow, at 7 days bind or escalate. Each stage transition is an event on `mqeb`. Replaces a spreadsheet-and-email renewal process with a fully audited automated pipeline. Method: `advance_renewal_stage()`.

## 11. Loss Run Generation

Produce structured loss run reports (5-year claims history by policy/property) in the format required by underwriters during renewal negotiation. Include frequency, severity, cause of loss breakdown, trend line. Currently no structured loss run exists — underwriters derive it manually. Method: `generate_loss_run()`.

## 12. Tenant-Level Insurance Liability Apportionment

For multi-tenanted properties, calculate each tenant's proportional insurance liability based on their lease area, link to `realestate_lea` lease records, and post charges to `realestate_acc` via service call. Currently premium allocation stops at unit level; this closes the last-mile to billing. Method: `apportion_insurance_to_tenants()`.

## 13. Digital Evidence Vault for Claims

Store claim evidence (photos, videos, contractor reports, police abstracts) as structured `ClaimEvidence` records with SHA-256 hash, upload timestamp, and chain-of-custody log. Prevent evidence tampering. Currently `evidence_ids` is a bare list of strings with no validation. New model: `ClaimEvidence`. Methods: `attach_claim_evidence()`, `verify_evidence_integrity()`.

## 14. Broker Performance Scorecard

Track broker KPIs: renewal retention rate, quote turnaround days, claims support rating, commission vs. market. Aggregate across all policies placed via each `broker_id`. Surface in dashboard. Enables data-driven broker panel management. Method: `get_broker_scorecard()`.

## 15. Event-Sourced Audit Log

Replace mutable dict updates in the store with an append-only event log: every state transition emits a typed `InsEvent` (policy_created, claim_status_changed, etc.) with actor, timestamp, and before/after diff. Current store allows silent dict mutation with no before-state. Replay events to reconstruct any historical state. Integrate with `audl` capability via `mqeb`. Foundational for regulatory audit and dispute resolution.
