# Agricultural Credit Scoring — World-Class Improvements

15 improvements that make agr_crd 10× better than any competing agri-fintech platform.

---

### I1. Satellite-Derived Yield Verification
**Category**: AI/ML
**Justification**: Eliminates self-reported yield fraud — the single largest source of agricultural credit loss in emerging markets. Lenders who verify yields via remote sensing cut default rates by 30-45% (IFC, 2023).
**Implementation**: Integrate NDVI time-series from Sentinel-2 (via ESA STAC API) keyed on farm parcel centroid; compute a `ndvi_consistency_score` that re-weights the `yield_level` factor in `score_farmer`, applying a fraud-flag multiplier (0.5×) when declared yield diverges >40% from NDVI-implied yield.
**Competitive reference**: Hello Tractor / Apollo Agriculture (Kenya) — both now gate disbursement on satellite-verified field health.

---

### I2. Repayment Schedule Generator with Amortisation
**Category**: Feature
**Justification**: Seasonal loans have irregular cash-flow windows (post-harvest). A static equal-instalment schedule causes unnecessary defaults; bullet + grace-period schedules aligned to harvest cycles reduce default by 22% (FAO, 2022).
**Implementation**: `generate_repayment_schedule(loan_id, schedule_type)` produces an amortisation table (Decimal arithmetic) supporting `equal`, `bullet`, `graduated` and `harvest_aligned` types; persisted as `_schedules` and used by `record_repayment` to compute days-past-due accurately.
**Competitive reference**: MoKash (Uganda) / Equity Bank Kenya — harvest-aligned bullet products with grace periods.

---

### I3. Early-Warning Default Prediction
**Category**: AI/ML
**Justification**: Reactive collections are 3× more expensive than pre-emptive intervention. Rule-based triggers (missed instalment + weather shock + commodity price drop) identify at-risk loans 45 days before default, allowing loan restructuring.
**Implementation**: `predict_default_risk(loan_id)` scores a loan on: days-past-due, commodity price index deviation, recent NDVI anomaly, group peer-default rate, and borrower's rolling repayment volatility; returns `risk_level` (LOW/MEDIUM/HIGH/CRITICAL) plus recommended intervention.
**Competitive reference**: Branch International / Tala — ML-based early-warning embedded in loan lifecycle.

---

### I4. Commodity Price Index Integration
**Category**: Integration
**Justification**: Loan sizing and interest-rate setting without commodity prices is guesswork. Maize at KES 28/kg vs KES 42/kg changes repayment capacity by 50%; dynamic pricing reduces adverse selection.
**Implementation**: `update_commodity_prices(crop_type, price_kes_per_kg, source)` maintains an in-memory price ledger; `score_farmer` optionally re-weights `revenue` factor using `(current_price / baseline_price)` multiplier, and `apply_for_loan` stamps the prevailing price at origination for audit traceability.
**Competitive reference**: Acre Africa / OKO Insurance — commodity price feed gates indemnity and loan triggers.

---

### I5. Weather/Index Insurance Linkage
**Category**: Integration
**Justification**: Parametric insurance bundled with credit reduces lender risk capital requirement by 40%, enabling lower rates. Borrowers with insurance coverage should receive preferential scoring.
**Implementation**: `link_insurance_policy(farmer_id, policy_id, provider, coverage_type, sum_insured_kes)` persists an `_insurance_policies` registry; `score_farmer` adds up to 8 bonus points for active parametric coverage; `predict_default_risk` reduces risk score for insured borrowers; `trigger_insurance_claim(policy_id, event_type, loss_pct)` emits a claim event.
**Competitive reference**: Pula (pan-Africa) — insurance-credit bundle with automatic satellite-triggered payouts.

---

### I6. Group Solidarity Scoring with Peer-Pressure Index
**Category**: Feature
**Justification**: Grameen-model groups perform 18% better when scored on group-level cohesion (meeting attendance, cross-guarantee history) rather than just individual scores. Most platforms ignore intra-group dynamics.
**Implementation**: `score_group(group_id)` computes: average member credit score, score dispersion (penalise high variance), historical group-level repayment rate, and a `peer_pressure_index` (fraction of members with prior settled loans); produces a group rating and maximum group loan quantum.
**Competitive reference**: Musoni (Kenya) — group credit limit set by cohesion metrics, not just aggregate score.

---

### I7. Loan Portfolio Risk Concentration Alerts
**Category**: Compliance
**Justification**: Regulators (CBK, RBA Kenya) require concentration limits: max 25% of portfolio to single crop type, max 15% to single geography. Automated alerts prevent regulatory breach before it occurs.
**Implementation**: `check_concentration_limits(limits_config)` inspects active loans by `crop_types` and `farm_parcel_ids` prefix, computes concentration ratios as Decimal percentages, and returns a list of `ConcentrationBreach` objects (entity, actual_pct, limit_pct, severity) suitable for dashboard surfacing and audit.
**Competitive reference**: KWFT (Kenya Women Finance Trust) — regulatory concentration reporting mandated by CBK Prudential Guidelines.

---

### I8. Multi-Currency Loan Support with FX Hedging Flags
**Category**: Feature
**Justification**: Agribusiness value chains routinely involve USD-denominated input purchases (fertiliser) while revenue is KES. Mismatched currencies are a hidden default driver; flagging FX exposure at origination is a baseline risk management requirement.
**Implementation**: `apply_for_loan` accepts `disbursement_currency` and `repayment_currency` independently; when they differ, a `fx_risk_flag=True` is set and `recommended_rate_pct` is inflated by 1.5pp as FX risk premium; `get_portfolio_summary` reports `fx_exposed_balance_kes` using a `_fx_rates` table updated via `update_fx_rate(from_ccy, to_ccy, rate)`.
**Competitive reference**: Citi AgriFinance / IFC SME Banking — dual-currency agri-loan products with embedded FX risk buffers.

---

### I9. Collateral Value Haircut and Pledging Enforcement
**Category**: Compliance
**Justification**: Accepting face-value collateral estimates inflates coverage ratios. Regulated lenders apply asset-class haircuts (land 30%, livestock 50%, equipment 40%) enforced at loan approval, not just recorded for optics.
**Implementation**: `apply_haircut(col_id, asset_type)` applies a configurable haircut schedule (Decimal multipliers) to compute `haircut_value_kes`; `approve_loan(loan_id, approved_amount)` validates that total pledged collateral haircut value ≥ approved_amount × `min_coverage_ratio` (default 1.2), raising `CollateralCoverageError` if insufficient.
**Competitive reference**: Land Bank of South Africa — mandatory haircut table enforced by credit policy engine.

---

### I10. Bulk Score Refresh with Staleness Enforcement
**Category**: Performance
**Justification**: Credit scores older than 90 days are regulatory dead weight (CBK Prudential Guideline on Consumer Credit). Batch refresh across all profiles catches seasonal income changes and prevents stale scores from driving bad approvals.
**Implementation**: `bulk_rescore(max_staleness_days, batch_size)` iterates profiles with `last_scored_at` older than threshold (or None), calls `score_farmer` in async batches of `batch_size`, and returns a `BulkScoreResult` with counts of refreshed, skipped, and failed profiles along with processing duration_ms.
**Competitive reference**: Equity Bank Kenya — nightly credit score refresh pipeline for HELB and agri-lending portfolios.

---

### I11. Farmer Financial Statement Ingestion (Mobile Money Parsing)
**Category**: AI/ML
**Justification**: M-Pesa / Airtel Money statement history is the de-facto financial statement for smallholders. Parsing transaction velocity, seasonal income spikes, and outflow patterns adds 15-20 points of scoring signal beyond declared revenue.
**Implementation**: `ingest_mobile_money_statement(farmer_id, transactions)` accepts a list of `{amount, direction, timestamp, counterparty_type}` dicts; computes `avg_monthly_inflow`, `income_seasonality_index` (coefficient of variation of monthly inflows), `savings_rate_pct`, and `irregular_outflow_flag`; persists as `_mobile_statements[farmer_id]` and feeds new factors into `score_farmer`.
**Competitive reference**: Tala / Branch International — mobile money transaction scoring with 100+ derived features.

---

### I12. Amortisation-Aware Loan Restructuring
**Category**: Feature
**Justification**: Forced defaults from inability to restructure cost lenders 3-7× more than renegotiated terms. A formal restructuring workflow with audit trail satisfies both regulatory requirements and investor covenants.
**Implementation**: `restructure_loan(loan_id, new_duration_months, grace_period_days, reason)` validates loan status is `repaying` or `disbursed`; computes new schedule via `generate_repayment_schedule`; records original terms in `restructure_history` list on the loan; emits `loan.restructured` event; updates `status` to `restructured`; re-triggers `predict_default_risk` post-restructure.
**Competitive reference**: Musoni / MFI standard — loan restructuring module with full audit trail required by AMFI Kenya Code of Practice.

---

### I13. KYC / AML Watchlist Screening
**Category**: Security
**Justification**: Agricultural lending programmes under CBK supervision require CDD (Customer Due Diligence) at onboarding. An unscreened borrower database exposes the platform to regulatory sanctions and reputational damage.
**Implementation**: `screen_farmer_kyc(farmer_id, national_id, phone)` checks farmer identifiers against an internal `_watchlist` (extensible to external INTERPOL/OFAC feeds); returns `{status: CLEAR|FLAGGED|REVIEW_REQUIRED, hits: [...]}` and blocks `apply_for_loan` for FLAGGED status; all screening events are appended to `_audit` with `event_type="kyc.screen"`.
**Competitive reference**: NCBA Bank Kenya / Standard Chartered — mandatory KYC screening at agricultural loan origination.

---

### I14. Seasonal Loan Window Enforcement
**Category**: Compliance
**Justification**: Disbursing a planting-season loan in October (post-harvest) is operationally nonsensical and a common fraud vector. Enforcing crop-calendar-aware disbursement windows eliminates this category of fraud entirely.
**Implementation**: `validate_disbursement_window(loan_id, crop_type, region)` looks up a `_crop_calendars` table (planting/harvest month ranges per crop/region); returns `{valid: bool, reason: str, suggested_disbursement_date: str}`; `approve_loan` calls this validation and surfaces a `DISBURSEMENT_WINDOW_MISMATCH` warning in the response if the loan application date is outside the optimal window.
**Competitive reference**: One Acre Fund (East Africa) — strict disbursement calendar enforcement keyed to regional planting windows.

---

### I15. Regulatory Report Generation (CBK Quarterly Return)
**Category**: Compliance
**Justification**: CBK supervised institutions must file quarterly agricultural credit returns (Form CBK-MFI-12). Manual generation takes 3-5 days per quarter. Automated generation from live data eliminates reporting lag and transcription errors.
**Implementation**: `generate_regulatory_report(report_type, period_start, period_end)` produces a structured dict matching CBK Form MFI-12 schema: portfolio composition by crop, gender, region; NPL ratio; concentration ratios; group vs individual split; average loan size; disbursement counts — all computed from `_loans`, `_groups`, `_profiles` in O(n) passes, returned as JSON and optionally serialised to CSV.
**Competitive reference**: KWFT / Faulu Kenya — automated CBK quarterly return generation mandated under Microfinance Act 2006.
