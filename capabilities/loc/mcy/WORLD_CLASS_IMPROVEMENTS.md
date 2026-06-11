# World-Class Improvements: loc_mcy (Multi-Currency Management)

**Capability**: Multi-Currency Management (`loc_mcy`)
**Author**: Nyimbi Odero — Datacraft © 2025

---

## 1. Real-Time Rate Feed Ingestion Pipeline

**Problem**: Exchange rates are only recorded manually. No live feed integration.

**Solution**: Async `ingest_rate_feed()` method that polls configurable rate sources (CBK, ECB, Bloomberg, XE) on a schedule, deduplicates by currency-pair/date, and emits `exchange_rate_ingested` events. Use `httpx.AsyncClient` with retry+backoff. Store source metadata (provider, fetch_timestamp, latency_ms) per rate row.

**Impact**: Eliminates manual rate entry lag; enables intraday rate refresh for high-volume FX operations.

---

## 2. Volatility-Adjusted FX Risk Scoring

**Problem**: `fx_risk_report()` uses static thresholds (1M/100K) without market-context sensitivity.

**Solution**: Compute rolling 30-day rate volatility (std-dev of daily close rates) per currency pair and weight exposure by volatility coefficient. Expose `volatility_score` and `risk_adjusted_exposure` per account. Flag currencies whose 7-day annualised vol > 15% as "elevated".

**Impact**: Treasury teams get risk signals grounded in market reality, not arbitrary balance thresholds.

---

## 3. Cross-Rate Triangulation Engine

**Problem**: `convert_amount()` only supports direct and simple inverse lookups. Multi-hop conversions (KES → EUR via USD) silently fail.

**Solution**: Implement Dijkstra's shortest-path over the rate graph where edge weights are bid-ask spreads. `convert_amount_via_path()` returns the full conversion chain, cumulative spread cost, and path confidence score. Cache the graph with a TTL equal to the shortest rate expiry in the graph.

**Impact**: Eliminates "no rate found" failures for exotic pairs; surfaces the true cost of multi-leg conversions.

---

## 4. Bulk Rate Upload with Idempotency Keys

**Problem**: Recording rates one-at-a-time is O(n) round-trips. No deduplication on retry.

**Solution**: `bulk_record_exchange_rates()` accepts a list of `ExchangeRateCreate` payloads plus an `upload_batch_id`. Each rate is keyed on `(tenant_id, from_currency, to_currency, effective_date, rate_type)`. Duplicate submissions within a batch return the existing record without error. Return a `BulkUploadResult` with counts: created, skipped_duplicate, rejected.

**Impact**: Rate feed imports that contain 10k+ daily rates become reliable and re-entrant.

---

## 5. IFRS 21 / ASC 830 Compliance Assertions

**Problem**: Translation and revaluation methods are accepted strings without standards-body alignment enforcement.

**Solution**: Add a `compliance_check()` method that, given entity financial statements and the applied method, verifies: (a) closing rate used for balance sheet items, (b) average rate for P&L, (c) translation reserve correctly routed to OCI. Returns a structured `ComplianceReport` with pass/fail per assertion and cite references.

**Impact**: Audit readiness is built into the capability, not bolted on post-factum by the finance team.

---

## 6. Stale Rate Detection and Alerting

**Problem**: The service silently uses expired rates when `expiry_date` is past. No warning surfaced.

**Solution**: `detect_stale_rates()` method returns all rates where `expiry_date < today` or rates with no `expiry_date` older than a configurable staleness window (default: 3 business days). Integrates with `ntfy` capability to push alerts. Also surfaces as a dashboard badge on `dashboard_summary()`.

**Impact**: Prevents financial misstatements caused by month-old spot rates applied to current-period transactions.

---

## 7. Hedging Instrument Registry

**Problem**: `hedge_effectiveness_monitor()` references `self._hedges` which does not exist in the service — it is a placeholder. No hedge lifecycle management exists.

**Solution**: Implement full hedge registry: `register_hedge()`, `list_hedges()`, `close_hedge()`, `mark_hedge_expired()`. Model: `HedgeInstrumentCreate` / `HedgeInstrumentResponse` with fields `instrument_type` (forward, option, swap), `notional_amount`, `strike_rate`, `maturity_date`, `counterparty_id`, `hedge_designation` (cash_flow, fair_value, net_investment).

**Impact**: Closes the gap between treasury operations and accounting — hedge effectiveness calculation becomes computable, not mocked.

---

## 8. Period-Close Checklist Automation

**Problem**: Period-end FX close requires coordinating rate upload, revaluation, translation, and reporting in sequence. Currently manual.

**Solution**: `run_period_close()` orchestrates the workflow: (1) assert all active rates are non-stale, (2) trigger revaluation for all entities, (3) trigger translation for consolidating entities, (4) generate FX gain/loss report, (5) return a `PeriodCloseResult` with status per step and any blocking exceptions. Each step is a named checkpoint in the audit log.

**Impact**: Reduces period-close from a 3-day manual process to a supervised automated pipeline.

---

## 9. Decimal-Precise Arithmetic Throughout

**Problem**: `service.py` and `models.py` use `float` for all monetary amounts. IEEE 754 rounding errors accumulate across revaluation calculations.

**Solution**: Replace `float` with `Decimal` for all monetary fields (`rate`, `converted_amount`, `fx_gain_amount`, etc.). Apply `ROUND_HALF_EVEN` (banker's rounding) at the point of persistence, not calculation. Add a `DecimalAmount` annotated type with range and precision constraints.

**Impact**: Eliminates ±0.0001 discrepancies that trigger reconciliation failures in downstream GL systems.

---

## 10. Multi-Entity Consolidation Roll-Up

**Problem**: `currency_exposure_summary()` is scoped to a single entity. No roll-up across legal entities.

**Solution**: `consolidated_exposure_summary()` accepts a list of `entity_ids`, translates each entity's exposures to the consolidation currency using the applicable rate, eliminates intercompany balances via `intercompany_pairs` input, and returns a `ConsolidatedExposureReport`. Supports partial consolidation (select entities only).

**Impact**: Group treasury can see net FX exposure across subsidiaries without running manual Excel consolidations.

---

## 11. Rate History Versioning and Audit Trail

**Problem**: `update_currency()` and rate records overwrite data. There is no history of what the rate was before update.

**Solution**: Implement append-only rate versioning: each update creates a new rate record with `superseded_by` pointing to the new ID. `get_rate_history()` returns the full chain for a currency pair. Deleted/superseded rates are flagged `is_active=False` but never removed. The audit stream includes `previous_rate` and `new_rate` in event payload.

**Impact**: Auditors and regulators can reconstruct exactly which rate was in effect for any transaction on any day.

---

## 12. Tenant-Scoped Rate Policy Engine

**Problem**: Rate validation rules are hardcoded in `_enforce()`. No per-tenant customisation of thresholds (e.g., max allowed daily rate move percentage).

**Solution**: `RatePolicy` model with fields: `max_daily_move_pct` (default 5%), `require_dual_approval_above` (amount threshold for manual rates), `auto_expire_days` (default 30). `apply_rate_policy()` validates a candidate rate against the tenant's active policy before persisting. Policies are versioned and auditable.

**Impact**: Large multinationals with different subsidiary risk tolerances can enforce different controls within the same deployment.

---

## 13. Currency Pair Correlation Matrix

**Problem**: No analytical tools for understanding relationships between currency pairs.

**Solution**: `compute_correlation_matrix()` accepts a list of currency pairs and a lookback window (e.g., 90 days). Returns a `CorrelationMatrix` pydantic model with pair-wise Pearson correlation coefficients computed from stored historical rates. Flags highly correlated pairs (>0.85) as potential double-hedging risks.

**Impact**: Portfolio-level FX risk management: treasury avoids hedging correlated pairs independently when a single macro hedge suffices.

---

## 14. Streaming Event Enrichment

**Problem**: `McyAuditEvent` emits minimal data — only `event_type`, `reference_id`, `actor_id`. Downstream consumers must re-query to reconstruct context.

**Solution**: Enrich `_emit()` to accept an optional `payload: dict[str, Any]` and populate it with before/after snapshots for state-change events (rate_recorded, revaluation_posted, etc.). Add `correlation_id` (traceable across a multi-step workflow like period-close) and `causation_id` (the event that triggered this one) to every event. Use `msgspec` for zero-copy serialisation before publishing to the bytewax stream.

**Impact**: Downstream analytics, alerting, and reconciliation systems get self-contained events — no N+1 query patterns against the MCY service.

---

## 15. Multi-Currency Balance Sheet Projection

**Problem**: No forward-looking analytics. The service only reports historical FX impact.

**Solution**: `project_balance_sheet_fx_impact()` accepts a set of open FX balances, a projection horizon (e.g., 90 days), and either (a) forward rates from hedge instruments or (b) a simple random-walk Monte Carlo scenario set. Returns expected P&L impact distribution (P10/P50/P90) with scenario breakdowns by currency. Powered by async numpy operations via `anyio.to_thread.run_sync`.

**Impact**: CFOs can quantify forward FX exposure under different rate scenarios before committing to hedging strategy — transforms MCY from a record-keeping tool to a decision-support system.
