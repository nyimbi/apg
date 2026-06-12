# Loyalty & Rewards

## Overview
Provides end-to-end loyalty programme management for retail tenants: member enrolment with consent and identity verification, points earn/redeem/adjust/batch transactions, tier qualification and downgrade management, coalition partner integration, referral earn, targeted campaign authoring with approval workflows and ROI measurement, a tiered reward catalogue, customer lifetime value (CLV) segmentation, points liability reporting, member merge/deduplication, and configurable points-expiry policies. All operations are tenant-isolated, streamed to Bytewax, and governed by 28+ deterministic rules with full GDPR/DPA consent lifecycle support.

## Capability ID
`retail_loy`

## Provides
| Service | Description |
|---|---|
| loyalty_member_enrolment | Consent-gated member onboarding with identity verification |
| loyalty_points_earn | POS and partner earn transactions with receipt validation |
| loyalty_points_earn_batch | High-volume batch earn with partial-failure model |
| loyalty_points_redeem | Balance-validated redemption across mechanisms |
| loyalty_tier_management | Tier upgrade, downgrade with grace period, and skip protection |
| loyalty_campaign_management | Campaign authoring, approval, activation, and ROI measurement |
| loyalty_partner_coalition | Multi-partner earn/redeem with SLA and settlement |
| loyalty_referral_earn | Referral code generation and referee/referrer bonus processing |
| loyalty_clv_analytics | RFM-based CLV scoring and segment assignment |
| loyalty_expiry_management | Rolling-activity and calendar-year expiry with dry-run |
| loyalty_reward_catalogue | Tier-gated and segment-gated reward stock and validity management |
| loyalty_transaction_ledger | Immutable earn/redeem/adjust/coalition/referral audit ledger |
| loyalty_liability_report | Points float, breakage estimate, and net liability in currency |
| loyalty_member_merge | Duplicate detection and member merge with transaction retargeting |
| loyalty_privacy | GDPR/DPA consent withdrawal, data export, and deletion scheduling |

## Requires
| Capability | Reason |
|---|---|
| auth | Member authentication and operator permissions |
| audl | Immutable audit trail for all point mutations |
| mten | Tenant context isolation |
| conf | Programme and tier configuration |
| ntfy | Tier-upgrade and expiry notifications |
| wflo | Campaign approval workflow |
| mqeb | Bytewax event stream for batch earn |
| moni | Points balance monitoring and anomaly alerts |
| schd | Scheduled expiry runs and CLV recalculation |

## Configuration
| Key | Default | Description |
|---|---|---|
| points_currency | PTS | Display symbol for points |
| max_earn_per_transaction | 100,000 | Hard cap per earn transaction |
| max_redeem_per_transaction | 50,000 | Hard cap per redeem transaction |
| expiry.default_policy | rolling_activity | Expiry policy applied to new programmes |
| expiry.default_rolling_days | 365 | Inactivity window before expiry |
| tiers.qualification_window_days | 365 | Window for tier qualification points |
| tiers.downgrade_grace_days | 90 | Grace period before tier downgrade |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /retail-loy/api/v1/contract | GET | Capability contract | public |
| /retail-loy/api/v1/programmes | GET/POST | List/create programmes | retail_loy:view/write |
| /retail-loy/api/v1/members | GET/POST | List members / enrol | retail_loy:view/write |
| /retail-loy/api/v1/members/<id> | GET/PUT/DELETE | Member detail/update/deactivate | retail_loy:view/write |
| /retail-loy/api/v1/members/<id>/transactions | GET | Transaction ledger | retail_loy:view |
| /retail-loy/api/v1/members/<id>/summary | GET | Full member summary | retail_loy:view |
| /retail-loy/api/v1/members/<id>/export | GET | GDPR data export | retail_loy:admin |
| /retail-loy/api/v1/members/<id>/consent/withdraw | POST | Withdraw consent | retail_loy:admin |
| /retail-loy/api/v1/members/<id>/referral-code | GET/POST | Get/generate referral code | retail_loy:write |
| /retail-loy/api/v1/members/merge | POST | Merge duplicate members | retail_loy:admin |
| /retail-loy/api/v1/members/duplicates | GET | List duplicate candidates | retail_loy:admin |
| /retail-loy/api/v1/transactions/earn | POST | Post earn transaction | retail_loy:write |
| /retail-loy/api/v1/transactions/earn/batch | POST | Batch earn (high volume) | retail_loy:write |
| /retail-loy/api/v1/transactions/redeem | POST | Post redeem transaction | retail_loy:write |
| /retail-loy/api/v1/transactions/adjust | POST | Administrative adjustment | retail_loy:write |
| /retail-loy/api/v1/transactions/referral | POST | Process referral earn | retail_loy:write |
| /retail-loy/api/v1/tiers | GET/POST | List/create tiers | retail_loy:view/admin |
| /retail-loy/api/v1/tiers/<id>/downgrade-check | POST | Run downgrade check | retail_loy:admin |
| /retail-loy/api/v1/campaigns | GET/POST | List/create campaigns | retail_loy:view/write |
| /retail-loy/api/v1/campaigns/<id>/approve | POST | Approve campaign | retail_loy:admin |
| /retail-loy/api/v1/campaigns/<id>/activate | POST | Activate campaign | retail_loy:admin |
| /retail-loy/api/v1/campaigns/<id>/roi | GET | Campaign ROI report | retail_loy:admin |
| /retail-loy/api/v1/campaigns/<id>/attribution | POST | Record campaign attribution | retail_loy:write |
| /retail-loy/api/v1/partners | GET/POST | List/register partners | retail_loy:admin |
| /retail-loy/api/v1/rewards | GET/POST | List/create rewards | retail_loy:view/write |
| /retail-loy/api/v1/rewards/for-member/<id> | GET | Tier-gated rewards for member | retail_loy:view |
| /retail-loy/api/v1/clv/<member_id> | GET | Get CLV segment | retail_loy:view |
| /retail-loy/api/v1/expiry/run | POST | Run expiry (dry_run default) | retail_loy:admin |
| /retail-loy/api/v1/reports/liability | GET | Points liability report | retail_loy:admin |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| tenant_context_required | tenant_context_present=False | deny |
| write_requires_policy | write op, no policy | deny |
| enrolment_requires_consent | enrol without consent | deny |
| earn_requires_receipt | earn without receipt_reference | deny |
| redeem_requires_sufficient_balance | balance < points | deny |
| redeem_frozen_member_denied | member status=frozen | deny |
| negative_balance_denied | adjustment yields negative | deny |
| tier_skip_denied | tier jump without approval | deny |
| campaign_requires_approval | activate without approved status | deny |
| cross_tenant_access_denied | cross-tenant operation | deny |

## Data Models
| Model | Key Fields |
|---|---|
| LoyProgrammeResponse | id, tenant_id, name, programme_type, points_currency |
| LoyMemberResponse | id, member_number, points_balance, current_tier_name, status, clv_segment |
| LoyTierResponse | id, tier_name, earn_multiplier, qualification_points |
| LoyTransactionResponse | id, transaction_type, points, balance_after, tier_at_time |
| LoyCampaignResponse | id, campaign_type, approval_status, points_issued_to_date |
| LoyPartnerResponse | id, partner_role, earn_rate, sla_reference |
| LoyRewardResponse | id, points_cost, redeem_mechanism, status |
| LoyClvSegmentResponse | id, clv_score, clv_segment, predicted_12m_revenue |

## Streaming Events
- `member_enrolled` — new member onboarded
- `points_earned` — earn transaction posted
- `points_redeemed` — redeem transaction posted
- `points_expired` — expiry run completed
- `points_adjusted` — admin adjustment applied
- `tier_upgraded` / `tier_downgraded` — tier change recorded
- `campaign_triggered` — campaign applied to transaction
- `clv_segment_changed` — member moved CLV segment

## Edge Cases Handled
- Earn on inactive member: denied at service layer assertion
- Redeem on frozen member: denied with explicit status check
- Adjustment yielding negative balance: denied
- Tier skip (e.g. bronze → gold): requires override approval flag
- Tier downgrade during grace period: blocked until grace expires
- Campaign activation without prior approval: assertion failure
- Campaign budget already consumed: activation blocked
- Batch earn without Bytewax stream: guardrail enforced

## Composability Notes
- **retail_pos** triggers earn/redeem via the transaction ledger at checkout; batch earn at EOD
- **retail_omc** triggers earn on online orders via the same earn API; referral codes tracked per channel
- **retail_prm** can issue loyalty_multiplier campaigns that interact with earn rates
- **retail_sin** CLV segments can be used as audience targeting in campaigns and tiered reward gating
- CLV recalculation is scheduled via **schd** and can trigger tier reassignments and churn interventions
- **retail_loy** publishes `member_merged`, `consent_withdrawn`, `referral_completed`, and `downgrade_scheduled` events to the Bytewax stream for downstream consumers

## New Service Methods (v1.1)
| Method | Description |
|---|---|
| `batch_earn_points()` | Process list of earn records with partial-failure model |
| `tier_downgrade_check()` | Evaluate and execute tier downgrade with grace period |
| `points_liability_report()` | Finance-grade outstanding liability with breakage estimate |
| `merge_members()` | Merge duplicate member accounts, retarget transactions |
| `find_duplicate_candidates()` | Fuzzy-match duplicate detection by email/mobile/name |
| `generate_referral_code()` | Idempotent referral code generation per member |
| `process_referral_earn()` | Award referrer and referee bonuses on qualifying spend |
| `record_campaign_attribution()` | Link transactions to campaigns for ROI tracking |
| `get_campaign_roi()` | Compute incremental revenue ROI vs. points cost |
| `list_rewards_for_member()` | Tier-gated and segment-gated reward catalogue |
| `withdraw_consent()` | GDPR consent withdrawal with freeze and deletion scheduling |
| `export_member_data()` | GDPR DSAR — full member data export |

---

## World-Class Enhancements (v2.0)

1. **Gamified Streak & Challenge Engine** — time-boxed micro-challenges with streak targets, bonus multipliers, and badge unlocks to drive purchase frequency
2. **Dynamic Points Pricing (Yield Management)** — `PointsPricingEngine` adjusts redemption rates in real-time based on liability float, demand pressure, and inventory levels
3. **Fraud Detection — Velocity & Graph Anomaly Rules** — inline velocity gates on earn/redeem plus async graph scan for synthetic member rings
4. **Referral & Social Earn** — unique referral codes with two-level tree, configurable bonuses, and leaderboard surface
5. **Omnichannel Earn Deduplication** — receipt-hash idempotency index prevents double-crediting across POS, online, mobile, and e-receipt channels
6. **Coalition Real-Time Point Conversion API** — `CoalitionExchangeEngine` with bidirectional transfer, per-partner rates, settlement batching, and reconciliation ledger
7. **Predictive Churn Intervention** — scheduler-driven win-back offers triggered by RFM recency threshold with acceptance-rate tracking
8. **Tier Downgrade Grace Period Enforcement** — `tier_downgrade_check` with rolling-window qualification, `downgrade_scheduled_at` timestamp, and notification on execution
9. **Points Float & Liability Reporting** — actuarial breakage model with Monte Carlo scenario, segmented by tier and CLV, exportable for finance systems
10. **Batch Earn via Event Stream (Bytewax / Kafka)** — async generator batch processing with partial-failure model and `batch_earn_completed` stream event
11. **Member Merge & Duplicate Detection** — fuzzy-match duplicate candidates, balance/transaction retargeting, immutable audit on merge
12. **Tiered Reward Gating** — `min_tier_name` and `allowed_segments` on rewards; `list_rewards_for_member` enforces eligibility at catalogue and redemption
13. **Campaign ROI Measurement** — `record_campaign_attribution` links transactions to campaigns; `get_campaign_roi` computes incremental revenue vs. points cost
14. **Privacy & Consent Lifecycle Management** — `ConsentRecord` model with version/timestamp/IP, `withdraw_consent` freeze + deletion scheduling, GDPR DSAR export
15. **Real-Time Personalisation via Contextual Bandits** — epsilon-greedy bandit over `[tier, clv_segment, recency_days, balance_ratio]` replaces static offer rules; falls back when `min_trials` not met

---

## New Methods

### `batch_earn_points` — High-volume POS reconciliation

```python
result = await svc.batch_earn_points(
    earn_records=[
        {"customer_id": "mbr_001", "transaction_id": "pos_txn_9001", "spend_amount": 250.0},
        {"customer_id": "mbr_002", "transaction_id": "pos_txn_9002", "spend_amount": 85.50, "bonus_multiplier": 2.0},
    ],
    programme_id="prog_abc",
    idempotency_key="eod-batch-2026-06-12",
)
# result["succeeded"], result["failed"], result["total_points_issued"], result["errors"]
```

Partial failures are collected in `errors`; successes are committed individually — no large rollbacks on a single bad record.

### `merge_members` — Deduplicate member accounts

```python
result = await svc.merge_members(
    tenant_id="tenant_x",
    primary_member_id="mbr_001",
    secondary_member_id="mbr_007",   # duplicate enrolled via app
    merged_by="ops@retailer.com",
)
# secondary status → "merged"; primary balance += transferred_points; all transactions retargeted
```

Use `find_duplicate_candidates(tenant_id, programme_id)` first to surface fuzzy-matched pairs before merging.

### `get_campaign_roi` — Measure incremental revenue vs. points cost

```python
roi = await svc.get_campaign_roi(tenant_id="tenant_x", campaign_id="camp_spring25")
# roi["gross_roi"]  → 1.42  (142% return)
# roi["incremental_revenue"], roi["points_cost_currency"], roi["attributed_transactions"]
```

Call `record_campaign_attribution(tenant_id, campaign_id, member_id, transaction_id, incremental_revenue)` on each qualifying earn to feed this report.
