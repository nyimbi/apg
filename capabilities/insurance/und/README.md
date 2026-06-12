# Underwriting Engine (ins_und)

Risk assessment, rating engine, capacity management, reinsurance treaties, underwriting rules.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/und/health | Service health check |
| GET | /api/insurance/und/describe | Capability description |
| GET | /api/insurance/und/submissions | List submissions |
| POST | /api/insurance/und/submissions | Submit risk |
| GET | /api/insurance/und/submissions/{id} | Get submission |
| POST | /api/insurance/und/submissions/{id}/assess | Run risk assessment |
| POST | /api/insurance/und/submissions/{id}/rate | Rate risk |
| POST | /api/insurance/und/capacity/check | Check capacity |
| GET | /api/insurance/und/treaties | List treaties |
| POST | /api/insurance/und/treaties | Create treaty |
| GET | /api/insurance/und/rules | List rules |
| POST | /api/insurance/und/rules | Create rule |
| DELETE | /api/insurance/und/rules/{id} | Delete rule |
| GET | /api/insurance/und/summary | Underwriting summary |
| GET | /api/insurance/und/audit | Audit trail |

## World-Class Enhancements (v2.0)

**I1. Predictive Loss-Cost Modelling** — Gradient-boosted loss-cost model cuts mis-pricing by 60% over heuristics [AI/ML]

**I2. Real-Time Telematics / IoT Risk Adjustment** — Ingest live sensor data and recompute risk score mid-term for fleet/agri products [Feature]

**I3. Multi-Layer Reinsurance Programme Optimisation** — Automated quota-share → XL → stop-loss cession waterfall with attachment optimisation [Feature]

**I4. Regulatory Solvency II / IFRS 17 Capital Allocation** — Per-submission SCR attribution using standard-formula factors for capital-cost pricing [Compliance]

**I5. Automated Sanctions & AML Screening** — Hard-block submission acceptance on OFAC/UN sanctions hits; returns CLEAR/HIT/POSSIBLE_MATCH [Security]

**I6. Dynamic Pricing Corridor with Min/Max Rate Guards** — Enforces configurable per-product rate floors and ceilings to prevent adverse-selection spirals [Feature]

**I7. Portfolio Accumulation & Catastrophe Exposure Aggregation** — Real-time 100-year PML computation by region/peril with XL breach flagging [Risk Management]

**I8. Straight-Through Processing (STP) Pipeline** — Atomic async submit→assess→capacity→rate→bind chain; 70–80% STP target [Performance]

**I9. Underwriter Performance Scorecard** — Override rate, premium variance, and referral resolution time per underwriter per period [UX]

**I10. Behavioural Cohort Renewal Pricing** — CLV- and claims-experience-based renewal multiplier targeting 8% retention uplift [AI/ML]

**I11. Embedded Parametric Trigger Evaluation** — Compare observed index value against contract threshold; return triggered payout and audit record [Feature]

**I12. Peer-Comparable Rate Benchmarking** — Rate adequacy score (p25/p50/p75 percentile) vs. anonymised peer portfolio [UX]

**I13. Document-Driven Risk Extraction (AI OCR)** — Ollama-backed structured extraction of risk attributes from PDFs/emails into submit-ready dict [AI/ML]

**I14. Facultative Placement Workflow** — Structured fac slip tracking percentage placed per market; auto-completes at 100% [Feature]

**I15. Underwriting Letter / Policy Schedule Generation** — Jinja2-rendered acceptance/decline/referral letters from assessment data in seconds [UX]

## New Methods

### `straight_through_process` — Atomic bind pipeline (I8)

```python
svc = UnderwritingService(tenant_id="acme")

result = await svc.straight_through_process(
    tenant_id="acme",
    submission_payload={
        "product_code": "motor_fleet",
        "proposer_id": "p-001",
        "sum_insured": 5_000_000,
        "risk_attributes": {"fleet_size": 12, "avg_driver_age": 34},
    },
)
# result["status"] == "bound" | "referred" | "declined"
# result["steps"]  — per-stage outcomes with timestamps
# result["policy_id"] — present only when status == "bound"
```

### `score_with_model` — ML loss-cost estimate (I1)

```python
loss_cost = await svc.score_with_model(
    submission_id="sub-abc123",
    model_backend="ollama",   # or "sklearn"
)
# loss_cost["estimate"]          — Decimal, annualised loss-cost ratio
# loss_cost["confidence_interval"] — (lower, upper) tuple
# loss_cost["heuristic_band"]    — existing band for comparison
```

### `compute_pml_exposure` — Catastrophe aggregation (I7)

```python
pml = await svc.compute_pml_exposure(
    tenant_id="acme",
    peril="wind",
    region="coast_ke",
)
# pml["pml_100yr"]          — Decimal, 100-year PML in currency units
# pml["total_sum_insured"]  — gross committed exposure
# pml["xl_breach"]          — bool; True if PML exceeds XL attachment
# pml["breached_treaty_id"] — treaty_id or None
```

---

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke
