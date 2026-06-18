# Supplier Relationship Management (scm_srm)

Supplier scorecard, risk assessment, collaboration portal, performance reviews, preferred supplier status, ESG scoring, contract lifecycle, development plans, segmentation, and benchmarking.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/srm/health | Health check |
| GET | /api/scm/srm/describe | Capability contract |
| GET | /api/scm/srm/suppliers | List suppliers |
| POST | /api/scm/srm/suppliers | Create supplier |
| GET | /api/scm/srm/suppliers/{id} | Get supplier |
| PUT | /api/scm/srm/suppliers/{id} | Update supplier |
| DELETE | /api/scm/srm/suppliers/{id} | Deactivate supplier |
| POST | /api/scm/srm/suppliers/{id}/approve | Approve supplier |
| POST | /api/scm/srm/suppliers/{id}/suspend | Suspend supplier |
| POST | /api/scm/srm/suppliers/{id}/preferred | Set preferred status |
| POST | /api/scm/srm/suppliers/bulk | Bulk-create suppliers |
| GET | /api/scm/srm/suppliers/segment | Segment portfolio |
| GET | /api/scm/srm/suppliers/concentration-risk | Concentration risk report |
| GET | /api/scm/srm/suppliers/{id}/benchmark | Benchmark vs peers |
| GET | /api/scm/srm/scorecards | List scorecards |
| POST | /api/scm/srm/scorecards | Create scorecard |
| GET | /api/scm/srm/scorecards/{supplier_id}/trend | Score trend series |
| GET | /api/scm/srm/risk-assessments | List risks |
| POST | /api/scm/srm/risk-assessments | Create risk assessment |
| POST | /api/scm/srm/risk-assessments/{id}/review | Review risk |
| GET | /api/scm/srm/risk-heatmap | Portfolio risk heatmap |
| GET | /api/scm/srm/messages | List collaboration messages |
| POST | /api/scm/srm/messages | Send message |
| GET | /api/scm/srm/performance-reviews | List reviews |
| POST | /api/scm/srm/performance-reviews | Create review |
| GET | /api/scm/srm/certifications | List certifications |
| POST | /api/scm/srm/certifications | Add certification |
| GET | /api/scm/srm/esg-scores | List ESG scores |
| POST | /api/scm/srm/esg-scores | Record ESG score |
| GET | /api/scm/srm/contracts | List contracts |
| POST | /api/scm/srm/contracts | Register contract |
| GET | /api/scm/srm/development-plans | List development plans |
| POST | /api/scm/srm/development-plans | Create development plan |
| PUT | /api/scm/srm/development-plans/{id}/progress | Update plan progress |
| GET | /api/scm/srm/escalations | List escalations |
| POST | /api/scm/srm/escalations | Raise escalation |
| POST | /api/scm/srm/escalations/{id}/resolve | Resolve escalation |
| POST | /api/scm/srm/onboarding | Start onboarding workflow |
| PUT | /api/scm/srm/onboarding/{id}/items/{idx} | Complete onboarding item |
| GET | /api/scm/srm/analytics | Supplier analytics |
| GET | /api/scm/srm/audit-events | Audit events |

## Key Constants

| Constant | Values |
|----------|--------|
| SUPPLIER_CATEGORIES | raw_material, packaging, services, technology, logistics, equipment, consumables |
| RISK_LEVELS | low, medium, high, critical |
| RISK_CATEGORIES | financial, geopolitical, operational, compliance, esg, concentration |
| SUPPLIER_STATUSES | active, pending_approval, probation, suspended, blacklisted, inactive |
| MESSAGE_TYPES | general, forecast_share, po_update, complaint, escalation, nda, performance_review |
| SEGMENT_STRATEGIES | risk_score, spend_category, geography |

## World-Class Enhancements (v2.0)

1. **Supplier Segmentation (Kraljic Matrix)** — `segment_suppliers()` maps risk vs. score into strategic/leverage/bottleneck/non-critical quadrants to drive differentiated engagement.
2. **Scorecard Trending & Regression Detection** — `scorecard_trend()` classifies trajectory as improving/declining/stable and triggers auto-probation recommendations on 2-period decline.
3. **Concentration Risk Detection** — `concentration_risk_report()` flags single-source categories and countries exceeding configurable share thresholds.
4. **Supplier Development Plans** — `create_development_plan()` / `update_development_plan_progress()` provide milestone-tracked remediation with budget, owner, and target score.
5. **Contract Lifecycle Management** — `register_contract()` / `list_contracts(expiring_within_days=90)` track values, auto-renew flags, and notice periods; expiring contracts surface in `health_check()`.
6. **Full ESG Scoring** — `record_esg_score()` captures E/S/G sub-scores with a weighted composite (40/30/30) and evidence URLs for CSRD/SFDR reporting.
7. **Formal Escalation Management** — `raise_escalation()` / `resolve_escalation()` provide full lifecycle with severity, due dates, and resolution notes; open escalations surface in `health_check()`.
8. **Supplier Benchmarking** — `benchmark_supplier()` computes per-dimension delta between a supplier and named category peers, contextualising absolute scores.
9. **Structured Onboarding Workflow** — `start_onboarding()` / `complete_onboarding_item()` enforce a configurable checklist (NDA, certs, bank details, site audit) before approval is permitted.
10. **Portfolio Risk Heatmap** — `risk_heatmap()` returns a category × severity matrix consumable by any BI tool in a single API call.
11. **Webhook / Event Bus Integration** — async `_emit()` publishes CloudEvents-formatted payloads to a configurable webhook URL or broker (Bytewax, NATS, Redis Streams).
12. **Composite Supplier Health Score** — `H = 0.4×scorecard + 0.2×risk_penalty_inverse + 0.2×esg + 0.1×no_open_escalations + 0.1×cert_current` surfaced on every supplier record.
13. **Expiring Certification Alerts** — `get_expiring_certifications(within_days=60)` returns certs with `days_remaining` for 90/60/30-day notification dispatch.
14. **Multi-Currency Contract Value Normalisation** — FX-rate integration normalises all contract values to a base currency for portfolio spend analytics.
15. **Audit Trail Immutability & Export** — append-only PostgreSQL audit store with SHA-256 hash chaining; `export_audit_events(format="jsonl"|"csv")` satisfies ISO 27001 / SOX requirements.

## New Methods

### `segment_suppliers` — Kraljic portfolio segmentation

```python
svc = SRMService(tenant_id="acme")

result = await svc.segment_suppliers(
    strategy="risk_score",   # or "spend_category" / "geography"
    tenant_id="acme",
)
# result["segments"] -> {"strategic": [...], "leverage": [...],
#                         "bottleneck": [...], "non_critical": [...]}
# result["summary"]  -> per-segment counts and recommended actions
```

### `scorecard_trend` — performance trajectory with early-warning

```python
trend = await svc.scorecard_trend(
    supplier_id="sup_01j...",
    periods=6,           # last N scorecards
    tenant_id="acme",
)
# trend["trajectory"]          -> "declining" | "improving" | "stable"
# trend["dimension_series"]    -> {dimension: [score, ...], ...}
# trend["recommendation"]      -> "place_on_probation" | None
```

### `risk_heatmap` — CPO-level portfolio exposure in one call

```python
heatmap = await svc.risk_heatmap(tenant_id="acme")
# heatmap["matrix"]   -> {category: {low: N, medium: N, high: N, critical: N}}
# heatmap["hotspots"] -> [{"category": "logistics", "severity": "critical", "count": 3}]
# heatmap["total_assessed"] -> int
```
