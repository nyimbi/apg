# leg_ctr — Contract Lifecycle Management

Drafting, review, redlining, approval workflow, e-signature, renewal alerts, obligations tracking.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/ctr/health | Health check |
| GET | /api/legal/ctr/contracts | List contracts |
| GET | /api/legal/ctr/contracts/{id} | Get contract |
| POST | /api/legal/ctr/contracts | Create contract |
| PUT | /api/legal/ctr/contracts/{id} | Update contract |
| DELETE | /api/legal/ctr/contracts/{id} | Archive contract |
| POST | /api/legal/ctr/contracts/{id}/submit | Submit for review |
| POST | /api/legal/ctr/contracts/{id}/execute | Execute contract |
| POST | /api/legal/ctr/contracts/{id}/terminate | Terminate contract |
| GET | /api/legal/ctr/contracts/{id}/redlines | List redlines |
| POST | /api/legal/ctr/redlines | Create redline |
| POST | /api/legal/ctr/redlines/{id}/resolve | Resolve redline |
| POST | /api/legal/ctr/obligations | Create obligation |
| POST | /api/legal/ctr/approvals | Request approval |
| POST | /api/legal/ctr/approvals/{id}/decide | Decide approval |
| GET | /api/legal/ctr/expiring | Expiring contracts |
| GET | /api/legal/ctr/dashboard | Contract dashboard |
| GET | /api/legal/ctr/audit | Audit events |

## Service Class

`ContractLifecycleService` — full contract lifecycle from draft to execution, with version history, redlining, multi-level approval, e-signature, renewals, and obligation tracking.

---

## World-Class Enhancements (v2.0)

**I1. AI-Powered Clause Risk Scoring** — Score contracts 0–100 across six risk dimensions using local Ollama LLM; composite `risk_score` on every record. [AI/ML]

**I2. Playbook-Driven Deviation Detection** — Compare redline text against tenant playbook tiers at `create_redline` time; emit `playbook_deviation_flag` with severity. [Compliance]

**I3. Obligation Calendar with iCalendar Feed** — RFC 5545 VCALENDAR feed per tenant with VALARM triggers at configurable lead times; `/calendar.ics` endpoint for calendar app subscription. [Feature]

**I4. Counterparty Risk Intelligence Integration** — Accept external health signals (credit, sanctions, litigation); surface `counterparty_risk_flag` and block execution on `sanctions_hit=True`. [Integration]

**I5. Templated Contract Generation from Structured Inputs** — Jinja2 templates per `(tenant, contract_type)` with typed variable slots; `generate_from_template` produces playbook-compliant versioned drafts. [Feature]

**I6. Decimal-Accurate Financial Milestone Tracking** — Replace `float` monetary fields with `Decimal`; `ContractMilestone` records with `record_milestone_payment` and running-balance summary. [Compliance]

**I7. E-Signature Audit Trail with SHA-256 Hash Chain** — Rolling `SHA-256(prev_hash + signatory + timestamp + content_hash)` chain on every signature; `verify_signature_chain` for tamper detection. [Security]

**I8. LLM Metadata Extraction from Counterparty Paper** — `extract_contract_metadata` sends raw text to local Ollama, parses structured JSON into contract fields, returns a populated draft. [AI/ML]

**I9. Approval Delegation and Escalation Engine** — `delegate_approval` with audit trail; `escalate_approval` auto-promotes on SLA breach per `(contract_type, value_band)` DoA matrix. [Compliance]

**I10. Semantic Full-Text Contract Search** — Per-tenant inverted index over title, description, tags, and redline text; `search_contracts` returns ranked results with matched field highlights. [Performance]

**I11. Contract Performance Scorecard** — `record_performance_event` (SLA breach, late payment, obligation miss, dispute) aggregates into a weighted 0–100 scorecard with trend direction. [Feature]

**I12. Jurisdiction-Aware Compliance Checklist** — `JURISDICTION_RULES` registry keyed by `(jurisdiction, contract_type)`; `run_compliance_scan` returns structured passed/warnings/failures report. [Compliance]

**I13. Automated Redline Conflict Detection** — On `create_redline`, detect open redlines on the same `(contract_id, section_ref)`; set `conflict_flag=True` and populate `conflicting_redline_ids`. [AI/ML]

**I14. Webhook Notification Bus for Contract Lifecycle Events** — HMAC-SHA256 signed fan-out to registered endpoints with exponential-backoff retry; `register_webhook` / `list_webhooks` management. [Integration]

**I15. Renewal Forecast and Portfolio Value-at-Risk Dashboard** — `renewal_forecast` aggregates expiry by quarter, sums `Decimal` values, and computes `value_at_risk` for unscheduled renewals. [Feature]

---

## New Methods

### `score_clause_risks` — AI Risk Scoring

```python
svc = ContractLifecycleService(tenant_id="acme")

# Score a contract across six risk dimensions via local Ollama
result = await svc.score_clause_risks(
    tenant_id="acme",
    contract_id="ctr_01j...",
    model="mistral",          # any locally-hosted Ollama model
)
# Returns:
# {
#   "contract_id": "ctr_01j...",
#   "risk_score": 72,          # composite 0–100
#   "dimensions": {
#     "liability": 85, "ip": 60, "termination": 70,
#     "payment": 55, "data_protection": 80, "jurisdiction": 65
#   },
#   "snapshot_version": 3
# }
```

### `extract_contract_metadata` — LLM Metadata Extraction

```python
raw_text = open("counterparty_draft.txt").read()

draft = await svc.extract_contract_metadata(
    tenant_id="acme",
    raw_text=raw_text,
    model="mistral",
)
# Returns a populated draft record with auto-extracted fields:
# effective_date, expiry_date, value, governing_law,
# payment_terms, termination_notice_days — ready for review/edit.
print(draft["governing_law"])   # e.g. "Laws of Kenya"
print(draft["termination_notice_days"])  # e.g. 30
```

### `renewal_forecast` — Portfolio Value-at-Risk

```python
forecast = await svc.renewal_forecast(tenant_id="acme")
# Returns per-quarter breakdown:
# {
#   "quarters": [
#     {
#       "quarter": "2026-Q3",
#       "expiring_count": 12,
#       "auto_renewing": 7,
#       "decision_required": 5,
#       "total_value": Decimal("4500000.00"),
#       "value_at_risk": Decimal("1800000.00")
#     },
#     ...
#   ],
#   "portfolio_value_at_risk": Decimal("3200000.00")
# }
for q in forecast["quarters"]:
    print(q["quarter"], q["value_at_risk"])
```
