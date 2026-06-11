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
