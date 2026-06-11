# SACCO Guarantor Management (`gua`)

**Capability ID:** `fintech_sacco_gua`  
**Domain:** fintech / sacco  
**URL prefix:** `/api/fintech/sacco/gua`

Manages the full lifecycle of guarantor obligations in SACCO lending: consent request, acceptance with savings freeze, exposure tracking, substitution, GL posting on default call, and automatic release on loan closure.

## Core flows

```
Loan officer → request_guarantee()  → guarantor receives consent request
Guarantor    → accept_guarantee()   → savings frozen, ActiveGuarantee created
                                    or
             → decline_guarantee()  → no savings touched, request closed

Loan repaid  → process_automatic_releases() → savings unfrozen, notices sent

Borrower defaults → call_guarantee() → DR Guarantor Savings / CR Loan Recovery

Guarantor leaving → substitute_guarantor() → old released, new request created
```

## Eligibility rules (defaults, tunable)

| Rule | Default |
|------|---------|
| Free savings must cover | 100 % of guarantee amount |
| Total exposure ceiling | 3 × share capital |
| Absolute default ceiling | KES 500,000 |
| Defaulter bar | Any loan in arrears blocks eligibility |
| At-risk DPD threshold | > 30 days |

Override per member: `POST /exposure-limit`.

## Key endpoints

| Method | Path | Action |
|--------|------|--------|
| POST | `/requests` | Request consent from guarantor |
| POST | `/requests/{id}/accept` | Accept with PIN verification |
| POST | `/requests/{id}/decline` | Decline request |
| POST | `/requests/{id}/cancel` | Cancel pending request |
| POST | `/guarantees/{id}/release` | Release savings |
| POST | `/guarantees/{id}/substitute` | Replace guarantor |
| POST | `/guarantees/{id}/call` | Default recovery + GL post |
| POST | `/guarantees/{id}/notice` | Send warning/call/release notice |
| GET  | `/exposure/{member_id}` | Current exposure snapshot |
| POST | `/eligibility` | Eligibility check |
| GET  | `/at-risk` | Guarantees on loans with DPD > 30 |
| GET  | `/metrics` | Portfolio statistics |
| POST | `/process-releases` | Nightly auto-release run |

## APG integration

- **`mem`** capability — member active status, share capital  
- **`dep`** capability — savings balance for cover calculation  
- **`lnd`** capability — loan status/DPD for at-risk and auto-release  
- **`ntfy`** capability — SMS/push notices via `domain/adapters.py`  
- **`common/reliability`** — `guard_tenant_id`, circuit breaker  
- **`common/nats`** — audit event streaming  

## Running tests

```bash
uv run pytest -vxs capabilities/fintech/sacco/gua/tests/
```

© 2025 Datacraft — Nyimbi Odero
