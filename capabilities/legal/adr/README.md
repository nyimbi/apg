# leg_adr — ADR / Dispute Resolution

Arbitration case management, mediation workflows, settlement tracking, award enforcement.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/adr/health | Health check |
| GET | /api/legal/adr/cases | List cases |
| GET | /api/legal/adr/cases/{id} | Get case |
| POST | /api/legal/adr/cases | File case |
| PUT | /api/legal/adr/cases/{id} | Update case |
| DELETE | /api/legal/adr/cases/{id} | Close case |
| POST | /api/legal/adr/cases/{id}/advance | Advance status |
| GET | /api/legal/adr/cases/{id}/neutrals | List neutrals |
| POST | /api/legal/adr/neutrals | Appoint neutral |
| POST | /api/legal/adr/neutrals/{id}/challenge | Challenge neutral |
| DELETE | /api/legal/adr/neutrals/{id} | Remove neutral |
| GET | /api/legal/adr/cases/{id}/proceedings | List proceedings |
| POST | /api/legal/adr/proceedings | Schedule proceeding |
| POST | /api/legal/adr/proceedings/{id}/conclude | Conclude proceeding |
| DELETE | /api/legal/adr/proceedings/{id} | Cancel proceeding |
| GET | /api/legal/adr/awards | List awards |
| GET | /api/legal/adr/awards/{id} | Get award |
| POST | /api/legal/adr/awards | Render award |
| PUT | /api/legal/adr/awards/{id} | Update award |
| DELETE | /api/legal/adr/awards/{id} | Set aside award |
| POST | /api/legal/adr/awards/{id}/challenge | Challenge award |
| POST | /api/legal/adr/awards/{id}/enforce | File enforcement |
| GET | /api/legal/adr/settlements | List settlements |
| POST | /api/legal/adr/settlements | Record settlement |
| PUT | /api/legal/adr/settlements/{id} | Update settlement |
| DELETE | /api/legal/adr/settlements/{id} | Void settlement |
| GET | /api/legal/adr/dashboard | ADR dashboard |
| GET | /api/legal/adr/audit | Audit events |

## Service Class

`ADRDisputeResolutionService` — supports arbitration, mediation, conciliation, expert determination. Auto-generates case numbers (ARB-YYYY-NNNN), tracks panel constitution, proceedings, awards with set-aside/enforcement lifecycle, and negotiated settlements.

## World-Class Enhancements (v2.0)

**I1. Decimal-precision monetary fields** — Replace all `float` monetary fields with `Decimal` to eliminate rounding errors on cross-border enforcement filings (IFRS 9 / IAS 37). [Compliance]

**I2. Statute-of-limitations guardian** — `check_limitation_period()` computes days remaining under a configurable `(governing_law, cause_of_action)` rule table and returns `{days_remaining, deadline, status: ok|warning|expired}`. [Compliance]

**I3. Procedural deadline calendar** — `create_deadline` / `list_overdue_deadlines` replace spreadsheet-based deadline tracking with automated overdue surfacing and webhook dispatch. [Feature]

**I4. Multi-party / multi-claimant support** — `add_party()` with `party_role ∈ {claimant, respondent, third_party, intervener}` enables ICC/SIAC multi-party arbitrations; replaces single claimant/respondent schema. [Feature]

**I5. Cost ledger with running exposure** — `record_cost_entry()` / `get_case_costs()` track neutral fees, institution charges, and disbursements in real time with `Decimal` aggregation by `payable_by`. [Feature/Performance]

**I6. Conflict-of-interest disclosure registry** — `record_coi_disclosure()` / `list_coi_disclosures()` satisfy IBA Guidelines and ISO-accredited institution requirements for electronic disclosure tracking. [Security/Compliance]

**I7. Settlement installment plan tracking** — `add_settlement_installment()` / `mark_installment_paid()` / `list_overdue_installments()` model structured payment schedules beyond the single `settlement_amount` field. [Feature/Compliance]

**I8. New York Convention enforcement eligibility** — `check_enforcement_eligibility()` cross-references a bundled `NYC_CONTRACTING_STATES` set at enforcement-filing time; no external API required. [Compliance/Feature]

**I9. Document bundle with hash integrity** — `add_document()` stores SHA-256 hash at submission time; `list_documents()` returns chain-of-custody metadata for award enforceability and set-aside defence. [Security]

**I10. Case risk score (0–100)** — `compute_case_risk_score()` returns `{score, risk_band, factors}` from weighted signals: overdue deadlines, challenged neutrals, panel delay, award overdue. [AI/ML/UX]

**I11. Procedural timetable (Gantt-ready)** — `create_timetable_item()` / `get_timetable()` provide a structured milestone schedule with `days_remaining`, replacing Word-document timetables. [Feature/UX]

**I12. Seat-specific default rules engine** — `SEAT_RULES` dict keyed by institution code (UNCITRAL, ICC, LCIA, SIAC, Nairobi Centre) auto-populates preliminary deadlines on `create_case`. [Feature/Compliance]

**I13. Multi-currency award conversion with FX audit trail** — `record_currency_conversion()` / `get_award_in_currency()` store the rate and date to prevent post-award enforcement disputes. [Feature/Performance]

**I14. Panel registry with specialisation filtering** — `register_panel_member()` / `search_panel_registry()` expose SIAC/LCIA-style neutral search by specialisation, language, and seat accreditation. [Integration/Feature]

**I15. Case-level SLA monitoring** — `compute_case_sla_status()` measures elapsed time at each status transition against configurable institution SLAs, returning `{sla_status: on_track|at_risk|breached, days_elapsed, days_budget, transitions}`. [Performance]

## New Methods

Three high-impact v2 methods and their intended usage:

### `check_limitation_period` — Time-bar compliance gate

```python
svc = ADRDisputeResolutionService(tenant_id="acme")

result = await svc.check_limitation_period(
    tenant_id="acme",
    case_id="case-001",
    governing_law="KE",
    cause_of_action="breach_of_contract",
    incident_date="2023-04-15",
)
# {"days_remaining": 12, "deadline": "2026-04-15", "status": "warning"}

if result["status"] == "expired":
    raise ValueError("Filing is time-barred — cannot proceed")
```

### `compute_case_risk_score` — Case health for general counsel dashboards

```python
risk = await svc.compute_case_risk_score(
    tenant_id="acme",
    case_id="case-001",
)
# {"score": 65, "risk_band": "high", "factors": [
#     {"factor": "overdue_deadlines", "count": 2, "points": 50},
#     {"factor": "no_panel_after_30_days", "points": 15},
# ]}

if risk["score"] >= 60:
    await notify_gcounsel(case_id="case-001", risk=risk)
```

### `check_enforcement_eligibility` — NYC Convention guard at enforcement filing

```python
eligibility = await svc.check_enforcement_eligibility(
    tenant_id="acme",
    award_id="award-007",
    jurisdiction="DE",  # Germany
)
# {"eligible": True, "basis": "New York Convention 1958", "caveats": []}

if not eligibility["eligible"]:
    raise ValueError(f"Award not enforceable in {jurisdiction}: {eligibility['caveats']}")

await svc.file_enforcement_action(
    tenant_id="acme",
    award_id="award-007",
    jurisdiction="DE",
    filed_by="counsel-002",
)
```
