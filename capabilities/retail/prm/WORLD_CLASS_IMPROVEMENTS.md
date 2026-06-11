# Promotions Management — World-Class Improvements

**Capability**: `retail_prm` | **Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. Real-Time Dynamic Pricing Engine

**Current gap**: Pricing rules are static priority-ordered adjustments with no runtime price elasticity awareness.

**Improvement**: Implement a demand-sensing dynamic pricing engine that observes real-time sell-through rates, time-to-expiry (for perishables/seasonal), and competitor price feeds to automatically propose and apply price adjustments within pre-approved corridors. The engine should expose `compute_dynamic_price(sku, channel, context)` and honour the existing `margin_floor_pct` as a hard floor. Integrate with the `moni` capability for sell-through velocity signals.

---

## 2. Multi-Buy / Bundle Offer Types

**Current gap**: Promotion types are flat — there is no native "buy N get M" or "bundle of SKUs at bundle price" logic.

**Improvement**: Add `BundleOffer` and `MultiBuyOffer` models with quantity threshold evaluation at basket apply time. `apply_promotion` should accept a `line_items: list[LineItem]` parameter and return per-line discount breakdowns. Bundle pricing requires SKU co-occurrence rules and must respect `excluded_skus` from the parent promotion.

---

## 3. Promotion Canary / A-B Testing Framework

**Current gap**: There is no controlled experiment infrastructure — all customers in an eligible segment receive the same treatment.

**Improvement**: Add `PrmExperiment` model with `control_pct` and `treatment_pct` fields. `check_promotion_eligibility` should perform deterministic bucket assignment (hash of `customer_id + experiment_id mod 100`) and return `variant: "control" | "treatment"`. Effectiveness records should carry `experiment_id` and `variant` for downstream causal inference.

---

## 4. Promotion Fatigue Detection

**Current gap**: The system has no awareness of over-promotioning individual customers, which degrades perceived value and margin.

**Improvement**: Track `customer_promotion_exposure` as a rolling 30-day count of promotions applied per customer. Expose `get_customer_promotion_fatigue(customer_id)` returning fatigue score (0-100), and add a `max_customer_exposures_per_month` gate in `check_promotion_eligibility`. Integrate with `retail_loy` loyalty tier data to apply tighter fatigue limits for high-value segments.

---

## 5. Cascade Markdown Automation

**Current gap**: `cascade_enabled` exists on the model but no automated cascade progression logic is implemented — it must be triggered manually.

**Improvement**: Implement `advance_markdown_cascade(markdown_id)` that reads `cascade_interval_days` and `cascade_increment_pct`, checks the current date against the last cascade step, and applies the next markdown tier up to a configurable `cascade_floor_margin_pct`. Schedule via the `schd` capability with daily invocation.

---

## 6. Promotion ROI Attribution Pipeline

**Current gap**: `PrmEffectivenessRecord` stores ROI as a manually supplied scalar; there is no automated attribution pipeline.

**Improvement**: Implement `compute_promotion_roi(promotion_id, period_start, period_end)` that aggregates `total_discount_issued`, `incremental_revenue` from transaction records, and estimates cannibilisation using a counterfactual baseline (matching control group from A/B experiments or synthetic control via pre-period average). Write the result as a `PrmEffectivenessResponse` record automatically.

---

## 7. Budget Burn Rate Alerting

**Current gap**: Budget state is tracked but there is no proactive alerting when a promotion is burning through its budget faster than expected.

**Improvement**: Implement `check_budget_burn_rate(promotion_id)` that computes current daily burn vs. expected daily burn (budget_cap / promotion_days). Return a `burn_rate_health` enum: `on_track | accelerating | exhaustion_imminent`. Publish an event to the `ntfy` capability when `exhaustion_imminent` is detected, and gate new applications when < 5% of budget remains.

---

## 8. Competitor Price Parity Monitoring

**Current gap**: `_competitor_prices` is initialised as an empty list but is never populated or evaluated.

**Improvement**: Add `ingest_competitor_price(sku, competitor, price, source_url, captured_at)` to record external price observations. Implement `compute_price_gap_analysis(sku)` that returns the tenant's current price vs. competitor median/min/max, and `suggest_competitive_markdown(sku)` that proposes a markdown depth to achieve price parity within the `floor_margin_pct` constraint.

---

## 9. Approval Workflow SLA Tracking

**Current gap**: Promotions can sit in `pending_review` indefinitely with no SLA enforcement.

**Improvement**: Record `submitted_at` timestamp on state transition to `pending_review`. Implement `list_overdue_approvals(tenant_id, sla_hours)` that returns promotions whose `submitted_at` is older than `sla_hours`. Integrate with `wflo` and `ntfy` capabilities to escalate overdue approvals to a configurable approver escalation chain.

---

## 10. Personalised Coupon Generation at Scale

**Current gap**: `coupon_issue` generates one coupon per call; bulk personalised campaigns require N sequential calls.

**Improvement**: Add `bulk_issue_coupons(segment_id, discount_pct, expiry, max_count)` that generates up to `max_count` unique coupon codes in a single async call, verifying uniqueness in a batch-safe way. Support templated code prefixes (e.g. `VIP-` for loyalty segment) and return a summary with `issued_count`, `duplicate_skipped`, and a signed S3/GCS URL to a CSV export of codes.

---

## 11. Promotion Conflict Pre-flight Validator

**Current gap**: Stacking conflicts are only discovered at cart-apply time, causing a degraded checkout experience.

**Improvement**: Add `preflight_promotion_plan(promotion_ids)` that simulates the full stacking evaluation for a proposed set of active promotions before any are activated, and returns a conflict matrix with resolution suggestions. This prevents merchants from accidentally creating an impossible promotion combination in the catalogue.

---

## 12. Tenant Analytics Benchmarking

**Current gap**: `promotion_analytics` returns absolute metrics with no benchmarking context.

**Improvement**: Implement `get_benchmark_analytics(tenant_id, period)` that computes percentile rank for key KPIs (redemption rate, coupon redemption rate, avg discount depth, ROI) against anonymised cross-tenant median and top-quartile values. Returns `{metric, tenant_value, peer_median, peer_p75, percentile_rank}` per KPI to help merchandisers contextualise performance.

---

## 13. Promotion Lifecycle Event Bus

**Current gap**: `_log_promotion_event` emits only to Python logging; no downstream consumers can react to promotion state changes.

**Improvement**: Replace the internal log calls with structured CloudEvents published via the `mqeb` capability. Event schema should follow `retail_prm.promotion.{activated|expired|budget_exhausted|margin_breach}` topic naming. This enables `retail_pos`, `retail_omc`, and `retail_loy` to subscribe and react in real time (e.g. strip promotion from active carts when it expires mid-day).

---

## 14. Margin-Aware Promotion Simulation

**Current gap**: Merchants cannot preview the margin impact of a new promotion before activating it.

**Improvement**: Implement `simulate_promotion_impact(promotion_id, expected_redemptions, avg_basket_value)` that projects total discount outlay, effective margin, budget utilisation at exhaustion, and break-even redemption count. Returns a `PrmSimulationResult` with `projected_roi`, `margin_floor_headroom`, and `confidence_interval` (narrow if historical redemption data is available, wide otherwise).

---

## 15. Immutable Audit Trail via Append-Only Ledger

**Current gap**: Promotion records are mutated in-place in `_promotions` dict with only an `updated_at` timestamp; there is no change history.

**Improvement**: Introduce an `_audit_ledger: list[dict]` append-only log that records every field change as `{entity_type, entity_id, field, old_value, new_value, changed_by, changed_at}`. Expose `get_promotion_audit_trail(promotion_id)` and route all writes through a `_record_change()` helper. In production, persist ledger entries to a PostgreSQL `prm_audit_log` table via the `audl` capability for compliance and dispute resolution.
