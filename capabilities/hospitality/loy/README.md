# Guest Loyalty Programme (hos_loy)

Points accrual, tier management, redemption, partner rewards, and recognition preferences.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/loy/health | Health check |
| GET | /api/hospitality/loy/members | List members |
| POST | /api/hospitality/loy/members/enroll | Enroll member |
| GET | /api/hospitality/loy/members/{id} | Get member |
| PUT | /api/hospitality/loy/members/{id} | Update member |
| DELETE | /api/hospitality/loy/members/{id} | Deactivate member |
| GET | /api/hospitality/loy/members/{id}/transactions | List transactions |
| POST | /api/hospitality/loy/members/{id}/earn | Earn points |
| POST | /api/hospitality/loy/members/{id}/redeem | Redeem points |
| POST | /api/hospitality/loy/members/{id}/adjust | Adjust points |
| POST | /api/hospitality/loy/members/{id}/tier-upgrade | Force tier upgrade |
| GET | /api/hospitality/loy/members/{id}/preferences | Get preferences |
| PUT | /api/hospitality/loy/members/{id}/preferences | Set preferences |
| GET | /api/hospitality/loy/partners | List partners |
| POST | /api/hospitality/loy/partners | Create partner |
| POST | /api/hospitality/loy/members/{id}/partner-earn | Partner points |
| POST | /api/hospitality/loy/bonus-campaigns | Create campaign |
| GET | /api/hospitality/loy/bonus-campaigns | List campaigns |
| GET | /api/hospitality/loy/tier-distribution | Tier report |
| GET | /api/hospitality/loy/dashboard | Dashboard |

## World-Class Enhancements (v2.0)

**I1. Decimal-Safe Monetary Arithmetic** — Replace float with `Decimal` for all monetary fields; `ROUND_HALF_UP` quantisation to 2dp on all point-to-cash conversions. [Compliance]

**I2. FIFO Points Expiry Ledger with Advance Notice** — Track point blocks with `expires_at`; `sweep_expired_points` emits 60/30-day `points_expiry_notice` events before FIFO expiry. [Compliance]

**I3. Tier Downgrade Protection with Grace Period** — `evaluate_tier_retention` sets `tier_grace_until` instead of immediate downgrade; emits `tier_downgrade_warning` for CRM retention actions. [Feature]

**I4. Real-Time Spend Projection to Next Tier** — `tier_progress_for_member` returns `{current_tier, next_tier, points_needed, nights_needed, spend_needed_kes}` computed live using Decimal arithmetic. [UX]

**I5. Points Transfer and Account Pooling** — `transfer_points(from, to, points, fee_pct)` atomic double-ledger debit/credit; `create_pool` and `add_member_to_pool` for corporate/family pooling. [Feature]

**I6. Digital Wallet Token (QR-Scannable Loyalty Card)** — `generate_wallet_token` issues a signed HMAC-SHA256 JWT (24h TTL) with tier and balance; `verify_wallet_token` for front-desk validation. [UX]

**I7. Stay-Based Milestone Awards** — `check_and_award_milestones` evaluates a configurable milestone table post-earn; idempotent bonus grants with `milestone_achieved` events. [Feature]

**I8. Velocity-Based Fraud Detection** — `assess_transaction_risk` checks same-day earn frequency, duplicate `reference_id`, and velocity vs. baseline; returns `RiskVerdict`, emits `fraud_flag` on HIGH. [Security]

**I9. AI-Driven Churn Risk Score** — `compute_churn_risk` scores on RFM dimensions; returns `{risk_score, risk_tier: low|medium|high, recommended_action}` with no external ML dependency. [AI/ML]

**I10. Configurable Challenge Engine** — `create_challenge`, `opt_in_challenge`, `evaluate_challenge_progress`, `complete_challenge`; completion triggers automatic bonus award and `challenge_completed` event. [Feature]

**I11. Partner Points Conversion (Airline Mile Transfer)** — `convert_to_partner_currency(member_id, partner_id, points, target_account_ref)` deducts at configured ratio and returns partner API-formatted payload. [Integration]

**I12. Birthday and Anniversary Lifecycle Rewards** — `check_lifecycle_rewards` matches `date_of_birth` and `enrollment_anniversary`; issues configurable bonus points with one-per-year idempotency guard. [UX]

**I13. GDPR/PDPA Consent Capture and PII Erasure** — `record_consent` persists immutable consent trail; `request_data_erasure` SHA-256 anonymises PII while retaining anonymised financial ledger. [Compliance / Security]

**I14. Multi-Currency Earn Normalisation** — `earn_points_multi_currency(member_id, amount, currency_code, ...)` normalises to KES via per-tenant `fx_rates`; `set_fx_rate` for operator updates. [Feature / Compliance]

**I15. Configurable Redemption Blackout Rules** — `create_blackout_rule(date_from, date_to, affected_offer_types, property_ids)` enforced on every `redeem_points` / `redeem_offer` call; returns next eligible date on violation. [Feature / Compliance]

## New Methods

Three high-impact async methods from v2.0 worth integrating first:

### `earn_points` — Tier-Aware Accrual with Campaign Multipliers

```python
svc = LoyaltyService(tenant_id="nairobi_grand")

# Earn points for a KES 12,500 stay (3 nights); active campaigns apply automatically
txn = await svc.earn_points(
    member_id="mbr_01j...",
    spend_amount=12500.0,
    description="Room 204 – 3-night stay",
    reference_id="RES-20260601-0042",
    nights=3,
)
# returns: {"id": ..., "transaction_type": "earn", "points": 1250, "running_balance": 4800, ...}
```

### `redeem_points` — Balance-Guarded Redemption

```python
# Redeem 500 points against F&B charge (1 pt = 0.05 KES → KES 25 discount)
redemption = await svc.redeem_points(
    member_id="mbr_01j...",
    points=500,
    description="F&B credit – Table 7",
    reference_id="POS-20260601-0119",
)
# raises ValueError("insufficient_points:...") if balance too low
# returns: {"cash_value": 25.0, "running_balance": 4300, ...}
```

### `adjust_points` — Auditable Manual Correction

```python
# Goodwill credit after a service failure; negative delta for corrections
adj = await svc.adjust_points(
    member_id="mbr_01j...",
    points_delta=200,
    reason="Service recovery – delayed check-in 2026-06-01",
    adjusted_by="mgr_frontdesk_01",
)
# records transaction_type="adjust"; negative adjustments floor at 0 (no negative balance)
```
