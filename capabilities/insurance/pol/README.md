# Policy Administration (ins_pol)

Policy lifecycle management: issuance, endorsements, renewals, cancellations, reinstatements, and document generation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/pol/health | Service health check |
| GET | /api/insurance/pol/describe | Capability description |
| GET | /api/insurance/pol/policies | List policies |
| POST | /api/insurance/pol/policies | Issue new policy |
| GET | /api/insurance/pol/policies/{id} | Get policy detail |
| PUT | /api/insurance/pol/policies/{id} | Update policy |
| DELETE | /api/insurance/pol/policies/{id} | Void draft policy |
| POST | /api/insurance/pol/policies/{id}/endorse | Create endorsement |
| POST | /api/insurance/pol/policies/{id}/renew | Initiate renewal |
| POST | /api/insurance/pol/policies/{id}/cancel | Cancel policy |
| POST | /api/insurance/pol/policies/{id}/reinstate | Reinstate policy |
| POST | /api/insurance/pol/policies/{id}/documents | Generate document |
| GET | /api/insurance/pol/portfolio/summary | Portfolio metrics |
| GET | /api/insurance/pol/audit | Audit trail |

## Core Service Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_policy` | `(tenant_id, policy_data) -> dict` | Issue a new policy |
| `get_policy` | `(tenant_id, policy_id) -> dict` | Retrieve policy by ID |
| `get_policy_by_number` | `(tenant_id, policy_number) -> dict` | Lookup by policy number |
| `update_policy` | `(tenant_id, policy_id, updates) -> dict` | Update policy fields |
| `delete_policy` | `(tenant_id, policy_id, reason) -> dict` | Void a draft policy |
| `list_policies` | `(tenant_id, status?, product_code?) -> list` | Filtered policy listing |
| `list_policies_by_insured` | `(tenant_id, insured_id) -> list` | Policies for an insured |
| `create_endorsement` | `(tenant_id, policy_id, ...) -> dict` | Mid-term endorsement |
| `initiate_renewal` | `(tenant_id, policy_id, ...) -> dict` | Start renewal process |
| `cancel_policy` | `(tenant_id, policy_id, ...) -> dict` | Cancel with audit trail |
| `reinstate_policy` | `(tenant_id, policy_id, ...) -> dict` | Reinstate lapsed policy |
| `lapse_policy` | `(tenant_id, policy_id, reason?) -> dict` | Mark policy as lapsed |
| `expire_policies` | `(tenant_id) -> list` | Batch-expire past-end-date policies |
| `generate_document` | `(tenant_id, policy_id, ...) -> dict` | Generate policy document |
| `search_policies` | `(tenant_id, query) -> list` | Full-text policy search |
| `bulk_issue_policies` | `(tenant_id, policies) -> dict` | Batch issuance |
| `portfolio_summary` | `(tenant_id) -> dict` | Aggregate portfolio metrics |

---

## World-Class Enhancements (v2.0)

**I1. AI-Driven Premium Re-Rating on Endorsement** — Real-time re-pricing via a pluggable `RatingEngine.rate()` coroutine on every mid-term change; stores old and AI-derived premium delta. [AI/ML]

**I2. Proactive Renewal Scoring & Churn Prediction** — `score_renewal_risk()` returns a `[0,1]` churn probability with `low/medium/high` band, computed 90 days before expiry from lapse history and premium delta. [AI/ML]

**I3. Regulatory Compliance Guardrails (IRA Kenya / IAIS)** — `validate_regulatory_compliance()` enforces product-specific sum-insured floors, premium caps, and mandatory clauses via a configurable `ComplianceRuleset` before any record persists. [Compliance]

**I4. Instalment Schedule & Premium Financing Tracking** — `create_instalment_schedule()` / `record_instalment_payment()` manage dated instalment records, recompute outstanding balances, and auto-unlapse on arrears clearance. [Feature]

**I5. Co-Insurance & Treaty Reinsurance Split Tracking** — `set_coinsurance_structure()` / `get_coinsurance_structure()` attach validated lead/follower share records (must sum to 100%) for IRA statutory returns. [Feature]

**I6. Digital Policy Wallet (QR / PDF / Passkit)** — `generate_digital_certificate()` produces an HMAC-signed certificate with `download_url` and `verify_url`; signature verification requires no DB round-trip. [UX]

**I7. No-Claim Bonus (NCB) Ledger** — `update_ncb()` increments or resets claim-free years and recomputes the discount pct via a configurable step table; value flows automatically into `initiate_renewal()`. [Feature]

**I8. Lien & Financier Notification Workflow** — `register_lien()` stores financier details; `_notify_lienholders()` fires on every cancellation and lapse, emitting a `lienholder_notified` audit event. [Integration]

**I9. Policy Comparison & Version Diffing** — `snapshot_policy()` persists immutable version records; `diff_policy_versions()` returns a structured `{field: {from, to}}` diff between any two snapshots. [Feature]

**I10. Automated Expiry & Grace-Period Engine** — `get_effective_status()` computes live status (active → grace → lapsed) without DB mutation; `run_expiry_engine()` batch-transitions only fully-elapsed policies. [Automation]

**I11. Multi-Currency & FX Rate Support** — `convert_policy_currency()` recomputes premium, sum insured, and instalment balances at a supplied Decimal FX rate, recording the applied rate in the audit trail. [Feature]

**I12. STP Eligibility Scoring** — `stp_score()` gates auto-bind eligibility by sum-insured threshold, product type, credit tier, and prior claims; `create_policy(auto_bind=True)` checks it before issuance. [AI/ML]

**I13. Claims-Loss Ratio Linkage** — `attach_claim_summary()` stores incurred/paid aggregates from `ins_clm`; `compute_loss_ratio()` returns `total_incurred / premium` as Decimal, flagging above-threshold ratios. [Integration]

**I14. Beneficiary & Succession Management** — `set_beneficiaries()` replaces the beneficiary register (validates 100% pct sum), records an endorsement; `get_beneficiaries()` returns the ordered list with effective date. [Feature]

**I15. Embedded Telematics / IoT Data Hooks** — `ingest_telematics_snapshot()` accepts a `{period, odometer_km, risk_events, score}` payload, recomputes the period premium via a `TelematicsBand` table, and auto-creates an adjustment endorsement when change exceeds materiality threshold. [Integration]

---

## New Methods

Three high-impact async methods from v2.0 illustrating the service extension pattern.

### `score_renewal_risk` — Churn Prediction

```python
svc = PolicyAdministrationService(tenant_id="acme")

result = await svc.score_renewal_risk(
    tenant_id="acme",
    policy_id="pol_01J2X...",
)
# {
#   "score": 0.73,
#   "risk_band": "high",
#   "factors": {
#     "days_to_expiry": 22,
#     "lapse_count": 2,
#     "premium_delta_pct": 18.4,
#     "endorsement_frequency": 3
#   }
# }
```

Scores range `[0, 1]`; `risk_band` is `low / medium / high`. Feed results into a retention campaign queue when `risk_band == "high"`.

---

### `get_effective_status` — Real-Time Grace-Period Engine

```python
status = await svc.get_effective_status(
    tenant_id="acme",
    policy_id="pol_01J2X...",
    grace_period_days=30,   # overrides product default
)
# {
#   "stored_status": "active",
#   "effective_status": "grace",
#   "end_date": "2026-05-10",
#   "grace_expires": "2026-06-09",
#   "days_remaining": 3
# }
```

Read-only — does not mutate the record. Call `run_expiry_engine(tenant_id)` in a scheduled task to batch-persist terminal transitions.

---

### `ingest_telematics_snapshot` — Usage-Based Premium Adjustment

```python
adjustment = await svc.ingest_telematics_snapshot(
    tenant_id="acme",
    policy_id="pol_01J2X...",
    snapshot={
        "period": "2026-05",
        "odometer_km": 1240,
        "risk_events": 2,       # hard-braking / speeding events
        "score": 0.61,          # normalised telematics score
    },
)
# {
#   "adjustment_endorsement_id": "end_01J3Y...",
#   "period_premium_before": "4200.00",
#   "period_premium_after": "3528.00",
#   "delta_pct": -16.0,
#   "band_applied": "moderate_risk"
# }
```

An endorsement record is created only when `abs(delta_pct) >= materiality_threshold` (default 5%). Below threshold, the snapshot is logged but no premium record is written.
