# Returns & Reverse Logistics (scm_rrl)

RMA processing, refurbishment workflow, disposal management, credit notes, reverse shipment tracking.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/rrl/health | Health check |
| GET | /api/scm/rrl/describe | Capability contract |
| GET | /api/scm/rrl/rmas | List RMAs |
| POST | /api/scm/rrl/rmas | Create RMA |
| GET | /api/scm/rrl/rmas/{id} | Get RMA |
| PUT | /api/scm/rrl/rmas/{id} | Update RMA |
| DELETE | /api/scm/rrl/rmas/{id} | Close RMA |
| POST | /api/scm/rrl/rmas/{id}/approve | Approve RMA |
| POST | /api/scm/rrl/rmas/{id}/reject | Reject RMA |
| POST | /api/scm/rrl/rmas/{id}/receive | Receive returned goods |
| POST | /api/scm/rrl/rmas/{id}/resolve | Resolve RMA |
| GET | /api/scm/rrl/refurbishments | List refurbishments |
| POST | /api/scm/rrl/refurbishments | Create refurbishment |
| POST | /api/scm/rrl/refurbishments/{id}/complete | Complete refurbishment |
| GET | /api/scm/rrl/disposals | List disposals |
| POST | /api/scm/rrl/disposals | Create disposal |
| GET | /api/scm/rrl/credit-notes | List credit notes |
| POST | /api/scm/rrl/credit-notes | Issue credit note |
| GET | /api/scm/rrl/analytics | Returns analytics |
| GET | /api/scm/rrl/audit-events | Audit events |

## World-Class Enhancements (v2.0)

1. **Predictive Return Rate Forecasting** — ML-based 7/14/30-day return volume forecasts per SKU, region, and reason code.
2. **Intelligent Disposition Engine (IDE)** — Rules-driven scoring of residual value, refurb ROI, WEEE compliance, and contractual obligations to rank disposition options with audit-trail rationale.
3. **Customer Self-Service Portal API** — `POST /rmas/self-service` with JWT entitlement validation, policy enforcement, auto-generated return labels, and status webhooks.
4. **SLA Tracking & Breach Alerting** — Per-RMA `sla_response_due`/`sla_receipt_due`/`sla_resolution_due` with background breach alerting and `sla_compliance_rate` metric.
5. **Reverse Logistics Cost Optimisation** — `cost_optimise_shipment()` clusters RMAs by geography and recommends carrier consolidation with projected savings via `scm_frm` rate cards.
6. **Serialised Item & Batch Traceability** — `serial_number`/`batch_id` on RMA items; `trace_item()` returns full lifecycle from sale to disposal; cross-checks warranty and recall flags against `scm_inv`.
7. **Green Returns Score & ESG Reporting** — `green_score` (CO₂e saved, landfill diverted) per record; tenant-level `esg_report()` endpoint for monthly sustainability KPIs.
8. **Multi-Currency Credit Note Management** — FX rate capture at issuance, `void_credit_note()`, `partially_apply_credit_note()`, and `credit_note_reconciliation()` for AR integration.
9. **Photo & Document Evidence Capture** — `attach_evidence()` stores images/documents per RMA; enables pluggable vision-model condition grading (Ollama LLaVA locally).
10. **Carrier Integration Adapters** — `CarrierAdapter` interface for DHL/FedEx/Aramex/mock; label generation, tracking webhook ingestion, and normalised shipment state machine.
11. **Configurable Return Policy Engine** — `ReturnPolicy` model with SKU glob patterns, return windows, auto-approve thresholds; `create_rma()` raises `PolicyViolationError` on violation.
12. **Bulk Operations & Async Job Queue** — `bulk_approve_rmas()`, `bulk_resolve_rmas()`, `bulk_complete_refurbishments()` via `asyncio.Queue` with retries, backoff, and job progress endpoint.
13. **Returns Fraud Detection** — `fraud_score()` returns 0–100 risk score from signals: return frequency, missing evidence, repeat SKU, out-of-window, high-value; emits `fraud_risk_detected` audit events.
14. **Refurbishment Cost & Profitability Analytics** — `refurbishment_profitability()` breaks down avg refurb cost, resale uplift, cycle time, and net margin per SKU with configurable time granularity.
15. **Event-Driven Integration Bus** — Pluggable `EventBus` (in-process / Redis Streams / Kafka / NATS) publishing CloudEvents 1.0 for downstream `scm_inv`, `fin_ar`, `crm_cst` consumers.

## New Methods

### `bulk_create_rmas` — high-throughput RMA ingestion

Processes a batch concurrently via `asyncio.gather`; returns partial results on failure.

```python
svc = ReturnsService(tenant_id="acme")

result = await svc.bulk_create_rmas(
    rmas_data=[
        {
            "order_id": "ORD-001",
            "customer_id": "CUST-42",
            "items": [{"sku": "LAPTOP-X1", "qty": 1}],
            "reason_code": "defective",
            "requested_resolution": "replacement",
        },
        {
            "order_id": "ORD-002",
            "customer_id": "CUST-77",
            "items": [{"sku": "MOUSE-M5", "qty": 2}],
            "reason_code": "wrong_item",
            "requested_resolution": "refund",
        },
    ]
)
# result = {"created": [...], "errors": [...], "total": 2, "failed": 0}
```

### `returns_analytics` — real-time returns KPIs

Aggregates RMA volume by reason, status, and resolution; includes credit issued and pending workloads.

```python
stats = await svc.returns_analytics()
# {
#   "total_rmas": 142,
#   "by_reason": {"defective": 80, "wrong_item": 35, ...},
#   "by_status": {"pending": 12, "resolved": 110, ...},
#   "total_credit_issued": 18450.00,
#   "pending_refurbishments": 7,
#   "pending_disposals": 3,
# }
```

### `create_reverse_shipment` — book a return carrier shipment

Links a carrier booking to an approved RMA and seeds the internal shipment state machine.

```python
shipment = await svc.create_reverse_shipment(
    rma_id="rma-a3f9c12d8b4e",
    carrier="dhl",
    tracking_number="1Z999AA10123456784",
    estimated_delivery="2026-06-19",
)
# Emits "reverse_shipment_created" audit event.
# shipment["status"] starts as "booked"; progresses via carrier webhooks.
```
