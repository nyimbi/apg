# Loyalty & Rewards

## Overview
Provides end-to-end loyalty programme management for retail tenants: member enrolment with consent and identity verification, points earn/redeem/adjust transactions, tier qualification and downgrade management, coalition partner integration, targeted campaign authoring with approval workflows, a reward catalogue, customer lifetime value (CLV) segmentation, and configurable points-expiry policies. All operations are tenant-isolated, streamed to Bytewax, and governed by 28 deterministic rules.

## Capability ID
`retail_loy`

## Provides
| Service | Description |
|---|---|
| loyalty_member_enrolment | Consent-gated member onboarding with identity verification |
| loyalty_points_earn | POS and partner earn transactions with receipt validation |
| loyalty_points_redeem | Balance-validated redemption across mechanisms |
| loyalty_tier_management | Tier assignment with skip protection and downgrade grace |
| loyalty_campaign_management | Campaign authoring, approval, and activation lifecycle |
| loyalty_partner_coalition | Multi-partner earn/redeem with SLA and settlement |
| loyalty_clv_analytics | RFM-based CLV scoring and segment assignment |
| loyalty_expiry_management | Rolling-activity and calendar-year expiry with dry-run |
| loyalty_reward_catalogue | Reward stock and validity management |
| loyalty_transaction_ledger | Immutable earn/redeem/adjust audit ledger |

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
| /retail-loy/api/v1/transactions/earn | POST | Post earn transaction | retail_loy:write |
| /retail-loy/api/v1/transactions/redeem | POST | Post redeem transaction | retail_loy:write |
| /retail-loy/api/v1/transactions/adjust | POST | Administrative adjustment | retail_loy:write |
| /retail-loy/api/v1/tiers | GET/POST | List/create tiers | retail_loy:view/admin |
| /retail-loy/api/v1/campaigns | GET/POST | List/create campaigns | retail_loy:view/write |
| /retail-loy/api/v1/campaigns/<id>/approve | POST | Approve campaign | retail_loy:admin |
| /retail-loy/api/v1/campaigns/<id>/activate | POST | Activate campaign | retail_loy:admin |
| /retail-loy/api/v1/partners | GET/POST | List/register partners | retail_loy:admin |
| /retail-loy/api/v1/rewards | GET/POST | List/create rewards | retail_loy:view/write |
| /retail-loy/api/v1/clv/<member_id> | GET | Get CLV segment | retail_loy:view |
| /retail-loy/api/v1/expiry/run | POST | Run expiry (dry_run default) | retail_loy:admin |

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
- **retail_pos** triggers earn/redeem via the transaction ledger at checkout
- **retail_omc** triggers earn on online orders via the same earn API
- **retail_prm** can issue loyalty_multiplier campaigns that interact with earn rates
- **retail_sin** CLV segments can be used as audience targeting in campaigns
- CLV recalculation is scheduled via **schd** and can trigger tier reassignments
