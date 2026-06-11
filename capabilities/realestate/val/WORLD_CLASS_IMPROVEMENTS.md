# Property Valuation — World-Class Improvement Plan

**Capability**: `realestate_val` | **Date**: 2026-06-11 | **Author**: Nyimbi Odero

---

## 1. Hedonic Regression Pricing Model

**Gap**: `mass_appraisal_run` assigns `estimated_value: 0.0` — a placeholder with no actual model.

**Fix**: Implement a full ordinary-least-squares hedonic regression using property attributes (floor area, age, location, bedrooms, condition score) fitted against comparable transaction data stored in `_store["comparables"]`. Expose `fit_hedonic_model()` and `predict_hedonic_value()` methods. Return R², coefficient table, and prediction interval alongside the point estimate. This is the global standard for mass valuation (IAAO Std 6).

---

## 2. Automated Valuation Model (AVM) with Confidence Bands

**Gap**: No true AVM; comparable analysis returns a raw average.

**Fix**: Add `run_avm()` that uses spatial distance weighting (IDW), time-adjustment factors (monthly CPI or transaction date decay), and property attribute adjustment grids to produce a confidence-graded AVM estimate. Return `value_low`, `value_central`, `value_high`, and a `confidence` tier (`low / medium / high / very_high`) so consumers can gate downstream decisions (e.g. auto-approve mortgage only when `confidence == very_high`).

---

## 3. IRR Computation via Newton-Raphson Bisection

**Gap**: `_compute_dcf` always returns `irr=None`.

**Fix**: Add `_compute_irr(cash_flows: list[Decimal], purchase_price: Decimal) -> Decimal | None` using a bisection solver between 0% and 100% with 1 bp convergence. Populate the `irr` field on `DcfModelResponse`. IRR is a first-class metric for institutional real estate investment decisions (IPF, RICS VPS 4).

---

## 4. Sensitivity Analysis Grid for DCF

**Gap**: Single-point DCF output with no stress-testing capability.

**Fix**: Add `dcf_sensitivity_analysis()` that sweeps `discount_rate` and `exit_yield` across ±150 bps in 50 bp steps, returning a 2D capital-value grid. Include a `recommended_range_low` / `recommended_range_high` at ±1 standard deviation. Institutional investors require this before investment committee sign-off.

---

## 5. Residual Land Value Method

**Gap**: `ValuationMethod.residual_method` exists as an enum but has no implementation.

**Fix**: Add `residual_land_valuation()`: GDV − build costs − finance costs − developer profit − transaction costs = residual land value. Accept GDV as either a direct input or computed from a DCF on the completed development. Essential for development sites, planning gain analysis, and affordable housing viability assessments.

---

## 6. Reinstatement Cost Assessment (RCA)

**Gap**: `ReportType.reinstatement_cost_assessment` is defined but `service.py` has no RCA logic.

**Fix**: Add `calculate_reinstatement_cost()` using BCIS elemental cost rates per property type and region. Apply gross external area, age obsolescence factor, professional fees uplift (12–15%), and debris removal. Output per-element cost breakdown. This feeds directly into `realestate_ins` insurance sum insured.

---

## 7. Comparable Adjustment Matrix Engine

**Gap**: `ComparableCreate.adjustments` is a freeform dict; no standardised adjustment methodology.

**Fix**: Add `apply_comparable_adjustments()` implementing a paired sales adjustment grid: time adjustment (date-to-date capital growth), size adjustment (per-sqm bracket), condition adjustment (RICS 1–3 scale), and location adjustment (postal sector index). Return adjusted price, adjustment narrative, and a `reliability_score` (0–100) based on how many adjustments were applied.

---

## 8. Portfolio-Level Valuation Variance Reporting

**Gap**: `valuation_analytics()` returns aggregate counts but no movement analysis.

**Fix**: Add `portfolio_variance_report()` comparing current roll values against prior-period values to compute like-for-like capital growth, acquisition/disposal effects, and revaluation surplus/deficit. Format output per IFRS 13 and IAS 40 disclosure requirements. Essential for REIT and fund reporting.

---

## 9. Valuation Uncertainty / HABU Analysis

**Gap**: No highest-and-best-use (HABU) or valuation uncertainty quantification.

**Fix**: Add `habu_analysis()` that evaluates 3–5 alternative uses (current use, planning permitted use, achievable alternate use) and ranks by net present value. Attach a `valuation_uncertainty` flag (`low / medium / high`) per RICS VPS 3 when multiple uses are plausible within ±20% of each other.

---

## 10. Market Rent Review Modelling

**Gap**: `ValuationPurpose.rental_review` exists but there is no rent review engine.

**Fix**: Add `model_rent_review()` that takes current passing rent, open market rent from comparable evidence, rent review clause (upwards-only, upwards/downwards, CPI-linked, fixed step), and calculates revised rent, uplift percentage, and next review date. Supports lease liability remeasurement triggers under IFRS 16.

---

## 11. Structured Valuation Report Generation (PDF/JSON)

**Gap**: No output beyond raw dict; no audit-ready report artefact.

**Fix**: Add `generate_valuation_report()` that assembles a structured JSON report (serialisable to PDF via WeasyPrint/ReportLab) containing: instruction details, methodology, comparable evidence table, sensitivity analysis, RICS Red Book disclaimer, valuer signature block, and digital timestamp. Store the report hash in `_store` for immutability verification.

---

## 12. Bulk Comparable Import with Deduplication

**Gap**: `add_comparable()` handles one record at a time; no deduplication logic.

**Fix**: Add `bulk_import_comparables()` accepting a list of `ComparableCreate` objects. Before insertion, run a fuzzy duplicate check on (address, transaction_date, price) within ±2% price tolerance and ±30 days. Return `inserted`, `skipped_duplicate`, and `validation_errors` counts. Critical when ingesting CoreLogic/Land Registry feeds.

---

## 13. Yield Curve & Equivalent Yield Calculator

**Gap**: `calculate_yield()` computes only net initial yield; equivalent yield requires term/reversion DCF.

**Fix**: Add `calculate_equivalent_yield()` that models term income (at passing rent) and reversion (at market rent) using a dual-rate annuity structure, then solves for the internal rate of return across the full income profile. Return NIY, equivalent yield, reversionary yield, and running yield in a single call — the standard RICS yield matrix.

---

## 14. Revaluation Trigger Detector

**Gap**: `revaluation_cycle()` must be manually called; no automated trigger detection.

**Fix**: Add `detect_revaluation_triggers()` that scans the valuation roll for: (a) entries older than `max_age_months` config, (b) properties with active planning permissions since last valuation, (c) leases with >10% rent change since last valuation, (d) IFRS reporting date proximity within 30 days. Returns a prioritised list of properties requiring revaluation with trigger reason and urgency score.

---

## 15. Real-Time Market Data Integration Adapter

**Gap**: All market evidence is manually added via `add_comparable()`; no live data feeds.

**Fix**: Add `sync_market_data()` with a pluggable adapter interface (`MarketDataAdapter` abstract base class) supporting: (a) Land Registry HMLR bulk download, (b) MSCI/IPD index integration, (c) CoStar API feed. Each adapter normalises to `ComparableCreate` schema and calls `bulk_import_comparables()`. Enables automated evidence refresh without manual data entry — a critical gap between AVM-grade and bespoke-valuation-grade systems.

---

## Priority Matrix

| # | Improvement | Effort | Business Impact | Standard |
|---|-------------|--------|-----------------|----------|
| 3 | IRR Computation | Low | High | RICS VPS 4 |
| 13 | Equivalent Yield | Low | High | RICS |
| 4 | DCF Sensitivity Grid | Medium | High | Investment Cmte |
| 7 | Comparable Adjustments | Medium | High | RICS GN 2 |
| 2 | AVM Confidence Bands | Medium | High | Basel III |
| 1 | Hedonic Regression | High | Very High | IAAO Std 6 |
| 5 | Residual Land Value | Medium | High | RICS VPS 12 |
| 6 | Reinstatement Cost | Medium | High | BCIS |
| 10 | Rent Review Modelling | Medium | High | IFRS 16 |
| 8 | Portfolio Variance | Medium | High | IAS 40 |
| 9 | HABU Analysis | High | Medium | RICS VPS 3 |
| 11 | Report Generation | High | High | RICS Red Book |
| 12 | Bulk Import | Low | Medium | Ops |
| 14 | Trigger Detector | Medium | Medium | IFRS 13 |
| 15 | Market Data Adapter | High | Very High | Strategic |
