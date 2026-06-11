# Distribution & Agency Management — World-Class Improvements

Fifteen targeted improvements to make ins_dst a category leader in African and emerging-market insurer distribution platforms.

---

### I1. Multi-Tier Hierarchy with Override Chain
**Category**: Feature
**Justification**: Real agency networks have 3–5 levels (National Manager → Regional → Branch → Agent → Sub-Agent). Flat supervisor_id is insufficient; insurers waste hours reconciling overrides and escalations manually.
**Implementation**: Add `hierarchy_path` (materialised path string, e.g. `"/nat1/reg3/br7/agt-001"`) to each agent record; implement `get_hierarchy_subtree` that resolves all descendants in one pass, enabling downstream commission splits and reporting roll-ups.
**Competitive reference**: Swiss Re iptiQ distribution API, Majesco Distribution Management

---

### I2. Tiered Commission Schedule Engine
**Category**: Feature
**Justification**: Fixed-rate commissions miss volume-incentive structures that drive 30–40% more premium from top agents; competitors offer slab-rate and retrospective bonus schemes.
**Implementation**: Introduce `commission_schedules` with `slab_tiers` (list of `{up_to, rate}`) and an `apply_commission_schedule` method that iterates slabs, calculates blended rate, and records which tier fired.
**Competitive reference**: Majesco Distribution Management, FINEOS Distribution

---

### I3. Automated Compliance Expiry Alerts
**Category**: Compliance
**Justification**: IRA Kenya mandates licence renewal within 90 days; missed renewals carry KSh 500,000 fines. Manual tracking is error-prone; automated alerts cut exposure.
**Implementation**: Add `scan_compliance_expiry_alerts` that returns structured alert objects (`{agent_id, compliance_type, expiry_date, days_remaining, severity}`) bucketed by severity (critical ≤7d, warning ≤30d, notice ≤90d).
**Competitive reference**: Gallagher Bassett Compliance Hub, Applied Epic

---

### I4. Agent Scorecard & Ranking
**Category**: AI/ML
**Justification**: Performance percentile ranking lets branch managers focus coaching on the bottom quartile and retain the top decile before competitors poach them.
**Implementation**: Add `rank_agents_by_performance` that computes a composite score (premium attainment × 0.5 + persistency_rate × 0.3 + compliance_score × 0.2), sorts agents within a peer group, and returns percentile ranks.
**Competitive reference**: Majesco Sales Performance Management, SalesForce Financial Services Cloud

---

### I5. Clawback / Commission Reversal with Lapse Tracking
**Category**: Compliance
**Justification**: When a policy lapses within 90 days, IRA regulations require commission clawback; failure to track exposes insurers to regulatory sanctions and inflated commission payouts.
**Implementation**: Add `initiate_clawback` that links a reversed commission to a `lapse_event`, creates a negative `dst_commission` record, adjusts agent lifetime totals, and emits a `commission_clawback_initiated` audit event.
**Competitive reference**: Majesco Distribution Management, Oracle Insurance Policy Administration

---

### I6. Real-Time Production Dashboard Metrics
**Category**: Performance
**Justification**: Insurers that can see live premium velocity by agent/branch react to shortfalls 10× faster than those running monthly reports, directly improving loss ratios.
**Implementation**: Add `production_dashboard` method returning rolling 30/90/365-day premium aggregates, commission liability (pending + approved), top-10 agents by premium, and product mix breakdown — all computed in a single pass over in-memory data.
**Competitive reference**: Majesco Distribution Management, Sapiens DistributionHub

---

### I7. E&O / Professional Indemnity Register
**Category**: Compliance
**Justification**: Errors & Omissions claims are rising; tracking PI coverage per agent with automatic suspension when coverage lapses protects the insurer from vicarious liability.
**Implementation**: Add `record_pi_coverage` and `check_pi_coverage_status` methods; auto-set agent status to `pi_lapsed` when PI expiry passes, blocking new commission computation.
**Competitive reference**: Applied Epic, Vertafore AMS360

---

### I8. Geospatial Territory Assignment
**Category**: Feature
**Justification**: Territory conflicts between agents cost 8–12% of premium in dispute resolution; explicit county/ward assignments enforce non-overlapping territories and power map visualisations.
**Implementation**: Add `territory` field (`{country, region, county, ward_codes: list[str]}`) to agent records; add `assign_territory` and `check_territory_conflict` methods that detect ward overlap across active agents.
**Competitive reference**: Salesforce Maps for Insurance, OneShield Distribution

---

### I9. Bulk Commission Import & Reconciliation
**Category**: Feature
**Justification**: Large brokers submit monthly commission statements as CSV/Excel; manual entry causes 3–5% error rate and 2-week settlement delays.
**Implementation**: Add `bulk_import_commissions` accepting a list of dicts, validating each row against agent registry and product catalogue, returning `{imported, failed, warnings}` summary with per-row error detail.
**Competitive reference**: Duck Creek Distribution, Sapiens DistributionHub

---

### I10. Agent Wallet & Settlement Ledger
**Category**: Feature
**Justification**: Mobile-first agents in emerging markets expect real-time balance visibility and M-Pesa settlement; a wallet ledger eliminates reconciliation disputes and enables instant pay-outs.
**Implementation**: Add `credit_agent_wallet` / `debit_agent_wallet` / `get_wallet_balance` methods maintaining a double-entry `dst_ledger_entry` list per agent with running balance, enabling M-Pesa or EFT settlement integrations.
**Competitive reference**: bolttech Distribution Platform, Jumo InsurTech

---

### I11. Incentive Campaign Management
**Category**: Feature
**Justification**: Time-bound bonus campaigns (e.g., "sell 10 motor policies in June, earn 5% bonus") are the primary lever for seasonal premium spikes; without automation the insurer overpays or miscalculates.
**Implementation**: Add `create_incentive_campaign` (with `start_date`, `end_date`, `qualifying_products`, `threshold_count`, `bonus_rate`) and `evaluate_agent_campaign_eligibility` that snapshots period commissions and computes campaign bonuses.
**Competitive reference**: Majesco Sales Performance Management, Vilocify

---

### I12. Persistency Rate Tracking
**Category**: AI/ML
**Justification**: Persistency (renewal rate) is the single highest-signal predictor of agent quality; IRA Kenya uses it for licence grading; insurers with >80% persistency have 20–25% lower loss ratios.
**Implementation**: Add `record_policy_renewal` / `record_policy_lapse` and `compute_persistency_rate` methods that track a rolling 12-month renewal cohort per agent, returning `{agent_id, policies_due, renewed, lapsed, persistency_pct}`.
**Competitive reference**: Majesco Distribution Management, Guidewire InsuranceSuite

---

### I13. Digital Agent Onboarding Checklist
**Category**: UX
**Justification**: Incomplete onboarding (missing KYC docs, unsigned agreements) causes compliance failures; a structured checklist with per-step status cuts average activation time from 5 days to same-day.
**Implementation**: Add `create_onboarding_checklist` generating a `dst_onboarding` record with standard steps (kyc_docs, ira_licence_upload, agreement_signed, bank_details, training_completed); add `update_onboarding_step` and `get_onboarding_status` computing overall percentage complete.
**Competitive reference**: Socotra Platform, bolttech Onboarding

---

### I14. Commission Statement PDF Generation Metadata
**Category**: Feature
**Justification**: Agents demand monthly commission statements for tax and audit purposes; automated statement metadata (statement period, line items, totals, tax deductions) eliminates manual preparation costing 20+ man-hours/month.
**Implementation**: Add `generate_commission_statement` that aggregates all paid commissions for an agent in a period, computes withholding tax (WHT) at 5% per KRA rules, and returns a structured statement dict ready for PDF rendering by the document capability.
**Competitive reference**: Applied Epic, Vertafore AMS360

---

### I15. Cross-Capability Integration Hooks
**Category**: Integration
**Justification**: Distribution is the upstream source for policy, claims, and finance data; well-defined integration hooks eliminate brittle point-to-point wiring and let new capabilities compose without modifying service.py.
**Implementation**: Add `get_agent_for_policy_integration` (returns minimal agent context for ins_pol), `get_commission_for_finance_integration` (returns payable amounts for ins_fin), and an `integration_manifest` method listing available hooks and their schemas.
**Competitive reference**: Majesco Enterprise Platform, Sapiens IDIT
