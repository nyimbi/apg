# Loyalty & Rewards — User Guide

**Capability ID**: `retail_loy` | **Domain**: `retail` | **Version**: `1.1.0`
**© 2025 Datacraft** | nyimbi@gmail.com | www.datacraft.co.ke

---

## Description

End-to-end loyalty programme management for retail tenants. Covers member lifecycle (enrolment through merge and GDPR deletion), points earn/redeem across all channels including batch and referral, tier management with upgrade/downgrade grace, campaign authoring and ROI measurement, a tiered reward catalogue, coalition partner exchange, CLV segmentation, points liability accounting, and duplicate detection.

---

## Installation

```bash
pip install apg-retail-loy
```

---

## Quick Start

```python
import asyncio
from apg_retail_loy.service import LoyService
from apg_retail_loy.models import (
    LoyProgrammeCreate, LoyMemberCreate, LoyTierCreate,
)

async def main():
    svc = LoyService(tenant_id="acme", actor_id="admin")

    # 1. Create programme
    prog = await svc.create_programme(LoyProgrammeCreate(
        tenant_id="acme",
        name="Acme Rewards",
        programme_type="points",
        created_by="admin",
    ))

    # 2. Create tiers
    bronze = await svc.create_tier(LoyTierCreate(
        tenant_id="acme", programme_id=prog.id,
        tier_name="bronze", earn_multiplier=1.0, qualification_points=0,
        created_by="admin",
    ))
    silver = await svc.create_tier(LoyTierCreate(
        tenant_id="acme", programme_id=prog.id,
        tier_name="silver", earn_multiplier=1.5, qualification_points=5000,
        created_by="admin",
    ))

    # 3. Enrol member
    member = await svc.enrol_member(LoyMemberCreate(
        tenant_id="acme", programme_id=prog.id,
        external_customer_id="cust_001",
        first_name="Jane", last_name="Doe",
        email="jane@example.com", mobile="+254700000001",
        consent_recorded=True, identity_verified=True,
        created_by="pos_terminal_1",
    ))

    # 4. Earn points at POS
    txn = await svc.earn_points(member.id, "pos_txn_001", spend_amount=150.0)
    print(txn)  # {"points": 150, "balance_after": 150, ...}

asyncio.run(main())
```

---

## Provides

- `loyalty_member_enrolment`
- `loyalty_points_earn`
- `loyalty_points_earn_batch`
- `loyalty_points_redeem`
- `loyalty_tier_management`
- `loyalty_campaign_management`
- `loyalty_referral_earn`
- `loyalty_partner_coalition`
- `loyalty_clv_analytics`
- `loyalty_expiry_management`
- `loyalty_reward_catalogue`
- `loyalty_transaction_ledger`
- `loyalty_liability_report`
- `loyalty_member_merge`
- `loyalty_privacy`

---

## Requires

- `auth` — member authentication and operator permissions
- `audl` — immutable audit trail for all point mutations
- `mten` — tenant context isolation
- `conf` — programme and tier configuration
- `ntfy` — tier-upgrade, expiry, and churn notifications

---

## Core Concepts

### Points Economy

| Constant | Default | Description |
|---|---|---|
| `POINTS_CASH_RATE` | 0.01 | 1 point = $0.01 (configurable per programme) |
| `COALITION_TRANSFER_FEE_PCT` | 0.05 | 5% fee on outbound coalition transfers |
| `max_earn_per_transaction` | 100,000 | Hard cap per earn (programme config) |
| `max_redeem_per_transaction` | 50,000 | Hard cap per redeem (programme config) |

### Tier Lifecycle

```
Enrol (bronze)
  → earn_points() [rolling window earn accumulates]
    → tier_upgrade_check()  [upgrade when threshold met]
  → earn stops / slows
    → tier_downgrade_check() [grace period starts]
      → grace_period_active (downgrade_grace_days)
        → tier_downgraded (if no recovery)
```

### Coalition Transfer

Outbound transfer deducts `points` from member balance, applies `COALITION_TRANSFER_FEE_PCT`, and records a `coalition_transfer` transaction. Settlement is batched per `settlement_frequency_days` on `LoyPartnerResponse`.

---

## Service API Reference

### Programme

```python
await svc.create_programme(data: LoyProgrammeCreate) -> LoyProgrammeResponse
await svc.get_programme(tenant_id, programme_id) -> LoyProgrammeResponse | None
await svc.list_programmes(tenant_id) -> list[LoyProgrammeResponse]
```

### Member Lifecycle

```python
await svc.enrol_member(data: LoyMemberCreate) -> LoyMemberResponse
# consent_recorded=True, identity_verified=True required

await svc.get_member(tenant_id, member_id) -> LoyMemberResponse | None
await svc.get_member_by_number(tenant_id, member_number) -> LoyMemberResponse | None
await svc.update_member(tenant_id, member_id, data: LoyMemberUpdate) -> LoyMemberResponse | None
await svc.list_members(tenant_id, programme_id=None) -> list[LoyMemberResponse]
await svc.freeze_member(tenant_id, member_id, reason, by) -> LoyMemberResponse | None
await svc.reactivate_member(tenant_id, member_id, by) -> LoyMemberResponse | None
await svc.get_member_summary(tenant_id, member_id) -> dict  # profile + balance + CLV + recent txns
```

### Points Transactions

```python
# Single earn
await svc.earn_points(customer_id, transaction_id, spend_amount, bonus_multiplier=1.0) -> dict

# Batch earn — partial-failure model: errors collected, successes committed individually
await svc.batch_earn_points(
    earn_records=[
        {"customer_id": "...", "transaction_id": "...", "spend_amount": 200.0, "bonus_multiplier": 1.5},
        ...
    ],
    programme_id="prog_001",
    idempotency_key="batch_2026-06-01",
) -> dict  # {succeeded, failed, total_points_issued, results, errors}

# Redeem
await svc.redeem_points(customer_id, points_to_redeem, redemption_type) -> dict

# Balance
await svc.points_balance(customer_id) -> dict  # {points_balance, cash_equivalent, current_tier}

# Admin adjustment (prevents negative balance)
await svc.adjust_points(data: LoyTransactionCreate) -> LoyTransactionResponse

# History
await svc.get_transaction_history(tenant_id, member_id, limit=50) -> list[LoyTransactionResponse]
```

### Tiers

```python
await svc.create_tier(data: LoyTierCreate) -> LoyTierResponse
await svc.list_tiers(tenant_id, programme_id) -> list[LoyTierResponse]
await svc.assign_member_tier(tenant_id, member_id, tier_id, by) -> LoyMemberResponse | None
await svc.tier_progress(customer_id) -> dict  # {current_tier, next_tier, points_to_next, progress_pct}
await svc.tier_upgrade_check(customer_id) -> dict  # evaluates and applies upgrade if qualified
await svc.tier_downgrade_check(customer_id) -> dict
# Respects downgrade_grace_days from LoyTierCreate.
# Returns action: "no_action" | "grace_period_started" | "grace_period_active" | "downgraded" | "grace_period_cleared"
```

### Campaigns

```python
await svc.create_campaign(data: LoyCampaignCreate) -> LoyCampaignResponse
await svc.approve_campaign(tenant_id, campaign_id, by) -> LoyCampaignResponse | None
await svc.activate_campaign(tenant_id, campaign_id) -> LoyCampaignResponse | None
# Must be approved before activation — assertion failure otherwise.
await svc.list_campaigns(tenant_id, programme_id=None) -> list[LoyCampaignResponse]

# ROI measurement
await svc.record_campaign_attribution(tenant_id, campaign_id, member_id, transaction_id, incremental_revenue) -> dict
await svc.get_campaign_roi(tenant_id, campaign_id) -> dict
# Returns: {total_incremental_revenue, points_cost_currency, gross_roi}
# gross_roi = (incremental_revenue - points_cost) / points_cost
```

### Rewards

```python
await svc.create_reward(data: LoyRewardCreate) -> LoyRewardResponse
await svc.list_rewards(tenant_id, programme_id) -> list[LoyRewardResponse]  # status="available" only

# Tier-gated catalogue — respects min_tier_name and allowed_segments on each reward
await svc.list_rewards_for_member(tenant_id, programme_id, member_id) -> list[LoyRewardResponse]
# Tier rank: bronze(0) < silver(1) < gold(2) < platinum(3)
```

### Partners & Coalition

```python
await svc.register_partner(data: LoyPartnerCreate) -> LoyPartnerResponse
await svc.list_partners(tenant_id, programme_id) -> list[LoyPartnerResponse]
await svc.coalition_transfer(customer_id, points, partner_programme) -> dict
# Deducts points + 5% fee; logs coalition_transfer record.
```

### Referral Earn

```python
# Generate code (idempotent)
await svc.generate_referral_code(tenant_id, member_id) -> dict
# {"referral_code": "REF-ABCD1234-XY12", "already_existed": False}

# Award bonuses after referee's qualifying spend
await svc.process_referral_earn(
    tenant_id,
    referral_code="REF-ABCD1234-XY12",
    referee_member_id="mem_002",
    qualifying_spend=100.0,
    referrer_bonus=500,   # configurable
    referee_bonus=200,    # configurable
) -> dict
# Referral depth capped at 1 (no pyramid exploitation)
```

### CLV

```python
await svc.record_clv_segment(data: LoyClvSegmentRecord) -> LoyClvSegmentResponse
# clv_segment: "high_value" | "medium_value" | "standard" | "at_risk" | "lapsed"

await svc.get_clv_segment(tenant_id, member_id) -> LoyClvSegmentResponse | None
```

### Points Expiry

```python
await svc.point_expiry_management(customer_id, expiry_date) -> dict
# Expires earn transactions older than expiry_date; posts expiry transaction.

await svc.expire_points(tenant_id, programme_id, dry_run=False) -> dict
# Bulk expiry for inactive members (last transaction > 365 days).
# dry_run=True returns affected list without mutating balances.
```

### Points Liability Report

```python
await svc.points_liability_report(programme_id) -> dict
# Returns:
# {
#   total_outstanding_points, gross_liability_currency,
#   breakage_rate,  # derived from historical expiry runs (default 5%)
#   expected_breakage_currency, net_liability_currency,
#   tier_breakdown: {tier_name: {member_count, points, liability}},
#   scenario_analysis: {optimistic_80pct, base_60pct, conservative_40pct},
# }
```

### Analytics

```python
await svc.loyalty_analytics(programme_id, period) -> dict
# period: "2026-06" (YYYY-MM)
# Returns: active/frozen/inactive counts, earn/redeem volumes, tier distribution,
#          CLV distribution, churn_risk_count, coalition_transfers.

await svc.personalised_offer(customer_id, offer_type) -> dict
# offer_type: "bonus_points" | "discount_voucher" | "free_product" | "tier_accelerator"
# Offer parameters scale with clv_segment and tier.

await svc.analytics_summary(tenant_id, period="monthly") -> dict
```

### Member Merge & Deduplication

```python
# Find candidates
await svc.find_duplicate_candidates(tenant_id, programme_id) -> list[dict]
# Returns groups: [{match_reason: "email_match"|"mobile_match"|"name_match", member_ids: [...]}]

# Merge (irreversible — secondary becomes status="merged")
await svc.merge_members(
    tenant_id,
    primary_member_id="mem_001",
    secondary_member_id="mem_002",
    merged_by="admin_user",
) -> dict
# {points_transferred, lifetime_points_merged, transactions_retargeted}
```

### Privacy & GDPR / DPA Compliance

```python
# Consent withdrawal — freezes member, schedules data deletion
await svc.withdraw_consent(tenant_id, member_id, withdrawn_by) -> dict

# DSAR — full data export (profile, transactions, CLV, offers, coalition transfers)
await svc.export_member_data(tenant_id, member_id) -> dict
```

---

## Business Rules

| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | No tenant context | deny |
| `enrolment_requires_consent` | `consent_recorded=False` | AssertionError |
| `enrolment_requires_identity` | `identity_verified=False` | AssertionError |
| `earn_requires_active_member` | `status != "active"` | AssertionError |
| `earn_requires_receipt` | `receipt_reference` missing (domain txn) | AssertionError |
| `redeem_requires_sufficient_balance` | `balance < points_to_redeem` | AssertionError |
| `redeem_frozen_member_denied` | `status = "frozen"` | AssertionError |
| `negative_balance_denied` | adjustment yields negative | AssertionError |
| `tier_skip_denied` | tier jump without approval | not implemented by default — override `assign_member_tier` |
| `campaign_requires_approval` | activate without `approval_status="approved"` | AssertionError |
| `cross_tenant_access_denied` | entity `tenant_id` != request `tenant_id` | returns `None` |
| `coalition_transfer_requires_balance` | `balance < points` | AssertionError |
| `referral_no_self_refer` | referrer == referee | AssertionError |
| `referral_already_referred` | `referred_by` already set | AssertionError |
| `merge_same_programme_only` | different `programme_id` | AssertionError |
| `merge_secondary_not_already_merged` | `status = "merged"` | AssertionError |
| `downgrade_grace_enforced` | `downgrade_grace_days` not elapsed | action = "grace_period_active" |

---

## Configuration Keys

All keys are tenant-scoped. Set via `conf` capability or `RETAIL_LOY_*` environment variables.

| Key | Default | Description |
|---|---|---|
| `points_currency` | PTS | Display symbol |
| `points_to_currency_rate` | 0.01 | Redemption rate (1 PTS = $0.01) |
| `max_earn_per_transaction` | 100,000 | Hard earn cap |
| `max_redeem_per_transaction` | 50,000 | Hard redeem cap |
| `coalition_transfer_fee_pct` | 0.05 | Outbound coalition fee |
| `expiry.default_policy` | rolling_activity | Expiry policy |
| `expiry.default_rolling_days` | 365 | Inactivity window |
| `tiers.qualification_window_days` | 365 | Rolling earn window |
| `tiers.downgrade_grace_days` | 90 | Grace before downgrade |
| `referral.referrer_bonus` | 500 | Default referrer bonus points |
| `referral.referee_bonus` | 200 | Default referee bonus points |

---

## UI Routes

| Path | Permission | Nav Group |
|---|---|---|
| `/retail-loy/dashboard` | `retail_loy:view` | Overview |
| `/retail-loy/members` | `retail_loy:view` | Members |
| `/retail-loy/members/<id>` | `retail_loy:view` | Members |
| `/retail-loy/members/enrol` | `retail_loy:write` | Members |
| `/retail-loy/members/<id>/export` | `retail_loy:admin` | Members |
| `/retail-loy/transactions` | `retail_loy:view` | Transactions |
| `/retail-loy/earn` | `retail_loy:write` | Transactions |
| `/retail-loy/earn/batch` | `retail_loy:write` | Transactions |
| `/retail-loy/redeem` | `retail_loy:write` | Transactions |
| `/retail-loy/tiers` | `retail_loy:admin` | Programme |
| `/retail-loy/campaigns` | `retail_loy:view` | Campaigns |
| `/retail-loy/campaigns/<id>/roi` | `retail_loy:admin` | Campaigns |
| `/retail-loy/rewards` | `retail_loy:view` | Rewards |
| `/retail-loy/reports/liability` | `retail_loy:admin` | Reports |

---

## Streaming Events

| Event | Trigger |
|---|---|
| `member_enrolled` | New member onboarded |
| `points_earned` | Earn transaction posted |
| `points_earned_batch` | Batch earn run completed |
| `points_redeemed` | Redeem transaction posted |
| `points_expired` | Expiry run completed |
| `points_adjusted` | Admin adjustment applied |
| `tier_upgraded` | Tier upgrade applied |
| `tier_downgraded` | Tier downgrade applied post-grace |
| `downgrade_scheduled` | Grace period started |
| `campaign_triggered` | Campaign applied to transaction |
| `clv_segment_changed` | Member moved CLV segment |
| `referral_completed` | Referral bonuses awarded |
| `member_merged` | Duplicate merge completed |
| `consent_withdrawn` | Member consent withdrawn |

---

## Composability

```apg
use retail_loy;
```

| Capability | Integration Point |
|---|---|
| **retail_pos** | POS triggers `earn_points` / `redeem_points` at checkout; EOD batch via `batch_earn_points` |
| **retail_omc** | Online orders trigger earn; referral codes tracked per channel |
| **retail_prm** | Issues `loyalty_multiplier` campaigns that interact with earn rates |
| **retail_sin** | CLV segments used as audience targeting and reward gating |
| **schd** | Scheduled expiry runs, CLV recalculation, downgrade checks |
| **ntfy** | Tier-upgrade, expiry warning, churn win-back notifications |
| **audl** | All point mutations emitted as immutable audit events |

---

## Further Reading

- `service.py` — Business logic (all async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 roadmap improvements with industry benchmarks
- `SPECIFICATION.md` — Full capability specification
- `cap_spec.md` — Compact capability spec
