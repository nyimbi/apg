# Procurement Management (scm_prc)

Full source-to-pay cycle: RFQ issuance and competitive scoring, purchase order lifecycle, goods receipt, three-way invoice matching, vendor evaluation, contract compliance with spend-down tracking, multi-currency normalisation, SLA monitoring, tamper-evident audit chain, delivery performance, and process cycle time analytics.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/prc/health | Health check |
| GET | /api/scm/prc/describe | Capability contract |
| GET | /api/scm/prc/rfqs | List RFQs |
| POST | /api/scm/prc/rfqs | Create RFQ |
| GET | /api/scm/prc/rfqs/{id} | Get RFQ |
| POST | /api/scm/prc/rfqs/{id}/issue | Issue RFQ to vendors |
| POST | /api/scm/prc/rfqs/{id}/responses | Record vendor response |
| POST | /api/scm/prc/rfqs/{id}/score | Weighted competitive scorecard |
| POST | /api/scm/prc/rfqs/{id}/award | Award RFQ |
| GET | /api/scm/prc/purchase-orders | List purchase orders |
| POST | /api/scm/prc/purchase-orders | Create PO |
| POST | /api/scm/prc/purchase-orders/bulk | Bulk create POs |
| GET | /api/scm/prc/purchase-orders/{id} | Get PO |
| PUT | /api/scm/prc/purchase-orders/{id} | Update PO |
| DELETE | /api/scm/prc/purchase-orders/{id} | Cancel PO |
| POST | /api/scm/prc/purchase-orders/{id}/send | Send PO to vendor |
| POST | /api/scm/prc/purchase-orders/{id}/acknowledge | Record vendor acknowledgement |
| POST | /api/scm/prc/purchase-orders/{id}/receive | Record goods receipt |
| PUT | /api/scm/prc/purchase-orders/{id}/delivery-schedule | Set delivery schedule |
| GET | /api/scm/prc/three-way-matches | List 3-way matches |
| POST | /api/scm/prc/three-way-matches | Create 3-way match |
| POST | /api/scm/prc/three-way-matches/{id}/resolve | Resolve disputed match |
| GET | /api/scm/prc/vendor-evaluations | List evaluations |
| POST | /api/scm/prc/vendor-evaluations | Create evaluation |
| GET | /api/scm/prc/contracts | List contracts |
| POST | /api/scm/prc/contracts | Create contract |
| GET | /api/scm/prc/contracts/{id}/spend-status | Contract spend-down status |
| POST | /api/scm/prc/exchange-rates | Set exchange rate |
| GET | /api/scm/prc/analytics/spend | Spend analytics (transaction currency) |
| GET | /api/scm/prc/analytics/spend/normalised | Multi-currency normalised spend |
| GET | /api/scm/prc/analytics/dashboard | Procurement KPI dashboard |
| GET | /api/scm/prc/analytics/delivery-performance | On-time delivery rates by vendor |
| GET | /api/scm/prc/analytics/cycle-times | Process cycle time analytics |
| POST | /api/scm/prc/sla/configure | Configure SLA thresholds |
| GET | /api/scm/prc/sla/breaches | SLA breach and warning scan |
| GET | /api/scm/prc/audit-events | Audit events |
| GET | /api/scm/prc/audit-events/verify-chain | Verify tamper-evident audit chain |

## Status Flows

```
RFQ:  draft → issued → responses_received → awarded | cancelled

PO:   draft → sent → acknowledged → partially_received
                                  → received → invoiced → closed | cancelled

3WM:  pending → approved | rejected
```

## Key Design Decisions

- **Audit chain**: every event carries `prev_hash` and `hash` (SHA-256). `verify_audit_chain` recomputes the full chain to detect tampering.
- **Tolerances**: three-way match uses 1% (auto-approve) / 5% (partial) / >5% (disputed) bands.
- **Contract alerts**: `get_contract_spend_status` emits `contract_nearing_limit_80pct` / `_95pct` events automatically.
- **Multi-currency**: `set_exchange_rate` stores forward and reverse rates; `normalised_spend_analytics` converts all PO values to a single reporting currency.
- **SLA**: configurable per document type via `configure_sla`; `check_sla_breaches` returns `warning` (>75% elapsed) and `breached` (>100% elapsed) items.

## World-Class Enhancements (v2.0)

1. **Dynamic Tolerance Tiers** — configurable per-tenant match bands by commodity class, vendor tier, or PO value; replaces hardcoded 1%/5% thresholds.
2. **Supplier Risk Scoring Engine** — composite risk score from late delivery rate, dispute frequency, financial health, and geopolitical exposure.
3. **Automated PO Approval Workflows** — configurable single/two-tier approval chains triggered by PO value, category, or vendor risk; full audit of delegation and escalation.
4. **Contract Spend-Down Tracking** — real-time consumed-vs-ceiling with `contract_nearing_limit` events at 80%/95% utilisation.
5. **RFQ Comparative Scoring (Weighted Criteria)** — configurable weighted scorecard across price, lead time, quality, and sustainability; produces auditable award recommendation.
6. **Catalog Integration & Punchout Support** — line items linked to approved-supplier catalog; off-catalog items flagged for additional approval.
7. **Delivery Schedule & Milestone Tracking** — ordered receipt schedules per PO line; on-time delivery rate computed per vendor with automatic overdue escalation.
8. **Invoice Discounting & Early-Payment Programs** — models 2/10 NET30 terms, surfaces discount windows in AP worklist, records captured yield.
9. **Spend Forecasting with Seasonality** — forward spend forecast by vendor/category using historical PO patterns, open commitments, and contract run-rates.
10. **Commodity Price Index Benchmarking** — compares quoted unit prices against external indexes; flags quotes above a configurable percentage deviation.
11. **ESG / Sustainability Supplier Scorecard** — extends vendor evaluation with carbon, labour standards, and diversity dimensions for CSRD/scope-3 reporting.
12. **Exception-Based Alerts & SLA Engine** — document-type SLAs (e.g. PO acknowledgement 48 h); imminent-breach warnings and escalation event emission.
13. **Audit Trail Immutability & Tamper Evidence** — append-only log with SHA-256 chaining; cryptographically tamper-evident without a full blockchain.
14. **Multi-Currency Normalisation** — transaction and reporting currency stored per PO with dated exchange rates; aggregated analytics free of FX distortion.
15. **Procurement Process Mining & Bottleneck Detection** — actual process graph reconstructed from audit events; cycle times compared against targets to surface bottlenecks.

## New Methods

### `score_rfq_responses` — weighted competitive award scorecard

Ranks all vendor responses to an RFQ by a configurable weighted scorecard. Default weights: price 50%, lead time 25%, quality 15%, sustainability 10%. Weights must sum to 1.0.

```python
svc = ProcurementService(tenant_id="acme")

# record two responses before scoring
await svc.record_rfq_response(rfq_id, vendor_id="v-001",
    quoted_lines=[{"item_id": "SKU-A", "unit_price": 10.0, "quantity": 100}],
    currency="USD", lead_time_days=7, quality_score=8.5, sustainability_score=7.0)

result = await svc.score_rfq_responses(
    rfq_id,
    weights={"price": 0.60, "lead_time": 0.20, "quality": 0.10, "sustainability": 0.10},
)
# result["ranked"][0] is the recommended vendor with composite_score and rank
winner = result["recommended_vendor"]
```

### `get_contract_spend_status` — real-time spend-down with auto-alerts

Aggregates all non-cancelled POs against a contract vendor within the contract date window, returns consumed/remaining/utilisation, and automatically emits `contract_nearing_limit_80pct` or `contract_nearing_limit_95pct` audit events when thresholds are crossed.

```python
status = await svc.get_contract_spend_status(contract_id="ctr-abc123")
# {
#   "ceiling": 500000, "consumed": 412000, "remaining": 88000,
#   "utilisation_pct": 82.4, "alert_level": "80pct", ...
# }
if status["alert_level"]:
    print(f"Contract at {status['utilisation_pct']}% — consider renewal negotiation")
```

### `check_sla_breaches` — SLA breach and warning scan

Scans all open documents against configured SLA windows. Returns `warning` items (>75% of SLA elapsed) and `breached` items (>100% elapsed) with elapsed hours and the document reference for triage.

```python
# tighten acknowledgement SLA to 24 h for this tenant
await svc.configure_sla({"po_acknowledgement": 24, "rfq_response": 120})

report = await svc.check_sla_breaches()
for item in report["breached"]:
    print(f"SLA BREACH: {item['record_type']} {item['record_id']} "
          f"— {item['elapsed_hours']:.1f}h / {item['sla_hours']}h limit")
for item in report["warnings"]:
    print(f"SLA WARNING: {item['record_id']} at {item['pct_elapsed']:.0f}% of window")
```
