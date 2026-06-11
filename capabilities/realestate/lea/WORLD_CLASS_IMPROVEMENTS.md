# Lease Management — World-Class Improvement Opportunities

15 high-impact improvements to elevate `realestate_lea` from production-grade to world-class.

---

## 1. Persistent Database Adapter (SQLAlchemy Async)

**Current**: In-memory dict store only. Production use requires injecting a real DB session.
**Improvement**: Implement a `PostgresLeaseStore` adapter backed by SQLAlchemy async ORM. Use the existing `db_session` parameter to switch transparently. Store adapters implement a `LeaseStoreProtocol` interface so the service never couples to persistence.
**Impact**: Production readiness, horizontal scaling, ACID transactions, audit trail permanence.

---

## 2. Event Sourcing for Lease Lifecycle Transitions

**Current**: Lease state mutations overwrite dict fields in place with no event log.
**Improvement**: Emit domain events (`LeaseCreated`, `LeaseExecuted`, `LeaseAmended`, `LeaseTerminated`, etc.) to a `lease_events` store collection and optionally to an MQ bus. Reconstruct lease state from the event log for full audit replay.
**Impact**: Immutable audit trail, temporal queries ("what did the lease look like on 2024-03-01?"), compliance with IFRS 16 disclosure history requirements.

---

## 3. Structured Rent Escalation Projections

**Current**: `apply_rent_escalation` applies one escalation at a time with no look-ahead schedule.
**Improvement**: Add `project_rent_escalation_schedule(lease_id, years)` that produces a full multi-year rent projection table accounting for compounded fixed %, CPI-linked, and stepped escalations with configurable CPI forecasts.
**Impact**: Enables landlord/investor cash-flow modelling and covenant testing without leaving the capability.

---

## 4. Dilapidation Provision Calculation

**Current**: No dilapidation handling despite it being listed in README provides.
**Improvement**: Implement `calculate_dilapidation_provision(lease_id, schedule_type)` covering pre-lease, interim, and terminal schedules. Integrate with IFRS 37 provisions: recognise liability at commencement when there is a present obligation (restoration clause in lease).
**Impact**: Closes a gap between README and implementation; required for full IFRS compliance and property management workflows.

---

## 5. Lease Comparison / Benchmarking Against ERV

**Current**: `lease_cost_analysis` uses estimated occupancy cost components.
**Improvement**: Add `benchmark_against_erv(lease_id, erv_per_sqm, market_data)` that computes passing rent vs. Estimated Rental Value, over/under-rented status, reversion potential, and time-to-reversion. Feed from `realestate_val` or manual input.
**Impact**: Core REIT and institutional landlord metric. Directly informs rent review strategy and portfolio valuation.

---

## 6. Full Amortisation Schedule Generation (All Periods)

**Current**: `calculate_lease_liability` only returns the first 12 periods of the schedule.
**Improvement**: Add `full_amortisation_schedule(lease_id)` returning every period from commencement to expiry with opening balance, interest, principal, payment, and closing balance. Support quarterly/annual summarisation. Include CSV/JSON export.
**Impact**: Required for period-end accounting, auditor deliverables, and management accounts.

---

## 7. Lease Covenant Monitoring

**Current**: No financial covenant tracking on leases.
**Improvement**: Add `record_covenant(lease_id, covenant_type, threshold, test_date)` and `test_covenant_compliance(lease_id, as_of_date)` covering rent cover ratio, DSCR, net worth minimums, and occupancy thresholds. Alert when covenant headroom falls below configured buffer.
**Impact**: Critical for commercial lending against leased assets and landlord protection in high-value leases.

---

## 8. Automated Lease Abstract Extraction via LLM

**Current**: `create_abstraction` stores manually provided fields; AI integration is a stub.
**Improvement**: Add `extract_lease_abstract_llm(document_text, model)` that calls a locally hosted Ollama model (Llama 3/Mistral) with a structured extraction prompt and maps the JSON response to `LeaseAbstractionCreate`. Human verification step remains mandatory.
**Impact**: Reduces abstraction time from hours to minutes. Directly uses the Ollama-first strategy from project guidelines.

---

## 9. Multi-Currency Lease Portfolio with FX Translation

**Current**: Currency stored as a label; no FX translation or multi-currency aggregation.
**Improvement**: Add `translate_portfolio_to_reporting_currency(target_currency, fx_rates)` that converts all lease obligations, ROU assets, and liabilities to a single reporting currency using provided FX rates. Expose both functional and presentation currency columns.
**Impact**: Essential for multinational portfolio holders. Required for consolidated IFRS 16 disclosures.

---

## 10. Lease Abstraction Quality Scoring

**Current**: Abstraction is binary: pending/verified. No quality signal.
**Improvement**: Add `score_abstraction_quality(abstraction_id)` that checks completeness of key fields (rent, dates, break/renewal options, rent review dates, permitted use, alienation rights) and returns a score 0–100 with per-field gap analysis.
**Impact**: Drives targeted re-abstraction, reduces errors reaching the rent review and IFRS 16 calculation pipelines.

---

## 11. Break Option Financial Penalty Modelling

**Current**: Break penalty is a single flat value stored in `options`.
**Improvement**: Add `model_break_option_cost(lease_id, break_date)` returning total cost of exercising break: penalty payments, unamortised incentives to be repaid, dilapidations estimate, fit-out write-off, and relocation costs. Compare against cost of staying to produce a break/stay NPV matrix.
**Impact**: Gives occupiers and advisors a data-driven decision framework. Differentiates the product from basic lease admin tools.

---

## 12. Lease Performance KPI Dashboard

**Current**: `lease_portfolio_summary` provides counts and totals but no KPIs.
**Improvement**: Add `lease_kpi_dashboard(tenant_id, period)` computing: vacancy rate, occupancy cost ratio (OCR), rent collection efficiency, lease incentive payback period, WAULT, reversion yield, and void liability. Return time-series for trend analysis.
**Impact**: One-stop executive dashboard. Replaces manual spreadsheet reporting used by most property managers.

---

## 13. Holding-Over Detection and Notification

**Current**: No mechanism to detect or handle leases that pass their expiry date without renewal or termination.
**Improvement**: Add `detect_holding_over(as_of_date)` that identifies active leases past expiry and transitions them to `holding_over` status. Record holding-over rent (typically higher, often last passing rent + uplift). Trigger notification events.
**Impact**: Prevents silent revenue leakage and ensures accurate IFRS 16 reassessment of the lease term in holding-over situations.

---

## 14. Lease Data Validation and Integrity Checks

**Current**: Assertions at method entry but no cross-record consistency checks.
**Improvement**: Add `validate_lease_data_integrity(lease_id)` that verifies: rent review dates fall within lease term, option exercise windows don't extend beyond expiry, IFRS 16 liability + accumulated principal reconciles to zero at expiry, escalation records are non-overlapping, sublease term ≤ head lease term.
**Impact**: Catches data corruption and manual entry errors before they flow into financial statements. Supports internal audit and external audit readiness.

---

## 15. Lease Obligation Sensitivity Analysis

**Current**: IFRS 16 calculations use a single discount rate with no sensitivity.
**Improvement**: Add `discount_rate_sensitivity(lease_id, rate_range)` computing lease liability and ROU asset at each rate in the range (e.g. 3%–9% in 0.5% steps). Return a matrix showing liability delta per 100bps. Expose a breakeven rate at which operating vs. finance classification flips.
**Impact**: Directly supports auditor queries, treasury stress-testing, and the IFRS 16 sensitivity disclosure (IFRS 7 analogy). Differentiates the service from simple amortisation table generators.

---

*Prepared 2026-06-11. Priority: items 1, 2, 6, 8, 13 for Q3; remaining for Q4.*
