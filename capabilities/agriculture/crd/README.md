# Agricultural Credit Scoring (agr_crd)

Yield-based credit scoring, seasonal loan products, group lending, collateral registry.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/crd/health | Health check |
| GET | /api/agriculture/crd/profiles | List profiles |
| POST | /api/agriculture/crd/profiles | Create profile |
| GET | /api/agriculture/crd/profiles/{id} | Get profile |
| PUT | /api/agriculture/crd/profiles/{id} | Update profile |
| DELETE | /api/agriculture/crd/profiles/{id} | Delete profile |
| POST | /api/agriculture/crd/score/{farmer_id} | Score farmer |
| GET | /api/agriculture/crd/loans | List loans |
| POST | /api/agriculture/crd/loans | Apply for loan |
| GET | /api/agriculture/crd/loans/{id} | Get loan |
| PUT | /api/agriculture/crd/loans/{id} | Update loan |
| POST | /api/agriculture/crd/loans/{id}/repayment | Record repayment |
| GET | /api/agriculture/crd/collateral | List collateral |
| POST | /api/agriculture/crd/collateral | Register collateral |
| GET | /api/agriculture/crd/groups | Group loans |
| POST | /api/agriculture/crd/groups | Create group loan |
| GET | /api/agriculture/crd/portfolio | Portfolio summary |
| GET | /api/agriculture/crd/audit | Audit log |

## World-Class Enhancements (v2.0)

15 improvements that elevate agr_crd above competing agri-fintech platforms.

**I1. Satellite-Derived Yield Verification** — NDVI time-series from Sentinel-2 re-weights `yield_level` and applies 0.5× fraud-flag multiplier when declared yield diverges >40% from remote-sensing estimate [AI/ML]

**I2. Repayment Schedule Generator with Amortisation** — `generate_repayment_schedule()` produces amortisation tables for `equal`, `bullet`, `graduated`, and `harvest_aligned` schedule types using Decimal arithmetic [Feature]

**I3. Early-Warning Default Prediction** — `predict_default_risk()` scores loans on DPD, commodity deviation, NDVI anomaly, peer default rate, and repayment volatility; returns LOW/MEDIUM/HIGH/CRITICAL + intervention recommendation [AI/ML]

**I4. Commodity Price Index Integration** — `update_commodity_prices()` maintains a live price ledger; `score_farmer` re-weights revenue factor by `(current_price / baseline_price)` and stamps prevailing price at loan origination [Integration]

**I5. Weather/Index Insurance Linkage** — `link_insurance_policy()` / `trigger_insurance_claim()` add up to 8 bonus scoring points for active parametric coverage and reduce `predict_default_risk` output for insured borrowers [Integration]

**I6. Group Solidarity Scoring with Peer-Pressure Index** — `score_group()` computes score dispersion, historical group repayment rate, and `peer_pressure_index` to set group loan quanta beyond simple score aggregation [Feature]

**I7. Loan Portfolio Risk Concentration Alerts** — `check_concentration_limits()` enforces CBK max 25% single-crop / 15% single-geography rules, returning typed `ConcentrationBreach` objects before a regulatory breach occurs [Compliance]

**I8. Multi-Currency Loan Support with FX Hedging Flags** — `apply_for_loan` accepts independent `disbursement_currency` / `repayment_currency`; mismatches set `fx_risk_flag=True` and add 1.5 pp FX premium to recommended rate [Feature]

**I9. Collateral Value Haircut and Pledging Enforcement** — `apply_haircut()` applies asset-class haircut schedules (land 30%, livestock 50%, equipment 40%); `approve_loan` blocks if haircut coverage < 1.2× approved amount [Compliance]

**I10. Bulk Score Refresh with Staleness Enforcement** — `bulk_rescore()` async-batches profiles older than a configurable threshold and returns `BulkScoreResult` with refreshed/skipped/failed counts and duration_ms [Performance]

**I11. Farmer Financial Statement Ingestion (Mobile Money Parsing)** — `ingest_mobile_money_statement()` derives `avg_monthly_inflow`, `income_seasonality_index`, `savings_rate_pct`, and `irregular_outflow_flag` from M-Pesa/Airtel transaction history [AI/ML]

**I12. Amortisation-Aware Loan Restructuring** — `restructure_loan()` recomputes schedule, records original terms in `restructure_history`, emits `loan.restructured` event, and re-triggers `predict_default_risk` post-restructure [Feature]

**I13. KYC / AML Watchlist Screening** — `screen_farmer_kyc()` checks identifiers against an internal watchlist (extensible to INTERPOL/OFAC), blocks `apply_for_loan` for FLAGGED status, and appends all events to audit log [Security]

**I14. Seasonal Loan Window Enforcement** — `validate_disbursement_window()` looks up crop-calendar tables per crop/region; `approve_loan` surfaces `DISBURSEMENT_WINDOW_MISMATCH` warnings for out-of-window applications [Compliance]

**I15. Regulatory Report Generation (CBK Quarterly Return)** — `generate_regulatory_report()` produces a CBK Form MFI-12-conformant JSON/CSV: portfolio by crop/gender/region, NPL ratio, concentration ratios, and disbursement counts in O(n) passes [Compliance]

## New Methods

Three high-impact v2.0 methods available on `AgriCreditService`.

### `predict_default_risk`

Scores an active loan across five risk dimensions and recommends pre-emptive intervention 45+ days before default.

```python
svc = AgriCreditService()

result = await svc.predict_default_risk("loan_abc123")
# {
#   "loan_id": "loan_abc123",
#   "risk_level": "HIGH",           # LOW | MEDIUM | HIGH | CRITICAL
#   "composite_score": 72.4,
#   "factors": {
#     "days_past_due": 0,
#     "commodity_price_deviation": -18.5,
#     "ndvi_anomaly": True,
#     "group_peer_default_rate": 0.12,
#     "repayment_volatility": 0.41
#   },
#   "recommended_intervention": "Schedule restructuring call; flag for loan officer review"
# }
```

### `bulk_rescore`

Async-batch refreshes all farmer credit profiles whose scores are stale beyond a configurable threshold — mandatory under CBK 90-day staleness rule.

```python
svc = AgriCreditService()

result = await svc.bulk_rescore(max_staleness_days=90, batch_size=50)
# {
#   "refreshed": 412,
#   "skipped": 38,      # scored within threshold
#   "failed": 3,        # missing profile data
#   "duration_ms": 1847
# }
```

### `generate_regulatory_report`

Produces a CBK Form MFI-12-conformant report for a given quarter. Eliminates 3-5 days of manual quarterly preparation.

```python
svc = AgriCreditService()

report = await svc.generate_regulatory_report(
    report_type="CBK_MFI_12",
    period_start="2025-01-01",
    period_end="2025-03-31",
)
# {
#   "period": "Q1 2025",
#   "portfolio_by_crop": {"maize": 142, "tea": 87, ...},
#   "portfolio_by_gender": {"F": 168, "M": 61},
#   "npl_ratio": 0.043,
#   "concentration_ratios": {"single_crop_max": 0.21, "single_region_max": 0.13},
#   "group_vs_individual_split": {"group": 0.58, "individual": 0.42},
#   "avg_loan_size_kes": 42500,
#   "disbursement_count": 229
# }
```
