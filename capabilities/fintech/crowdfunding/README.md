# Crowdfunding Platform

## Overview
Crowdfunding Platform manages the full alternative-finance campaign lifecycle: issuer due diligence, campaign publishing across equity, debt, reward, donation, and revenue-share structures, investor disclosure management, commitment recording, escrow funding, milestone-gated payouts, equity allocation, investor reporting, regulatory limits, and campaign moderation.

Investor commitments require KYC and explicit risk acknowledgement. Payouts are gated behind milestone evidence and human approval. All platform lifecycle events stream to `apg.fintech.crowdfunding.lifecycle` via Bytewax.

**Capability ID**: `fintech_crowdfunding`  
**Version**: 2.0.0

## Features
- Multi-type campaigns: equity, debt, reward, donation, revenue-share
- CMA Kenya Crowdfunding Regulations 2022 compliance (KES 500K per-campaign limit, KES 3M annual platform limit)
- KYC-gated issuer onboarding with beneficial owner evidence
- Milestone-gated escrow disbursement with human approval
- Pro-rata equity allocation and investor returns reporting
- Campaign analytics: commitment size distribution, payout history, compliance alerts
- Secondary market listing, investment certificates, bulk campaign approval
- Bytewax streaming event bus for all lifecycle events
- AI agent registration for disclosure review and escrow release automation

## Provides
| Service | Description |
|---------|-------------|
| crowdfunding_issuer_workflow | Onboard issuers with KYC, beneficial owner, and risk rating evidence |
| crowdfunding_campaign_workflow | Publish campaigns with type, currency, target, and disclosure requirements |
| crowdfunding_disclosure_workflow | Record and review offering memoranda, risk factors, financials, and use-of-funds documents |
| crowdfunding_commitment_workflow | Record investor commitments with KYC, risk acknowledgement, and positive amount controls |
| crowdfunding_escrow_workflow | Record escrow funding linked to funded commitments and wallet references |
| crowdfunding_milestone_workflow | Track campaign milestones with evidence and review requirements |
| crowdfunding_payout_workflow | Authorize payouts against milestones with positive amount and approval controls |
| crowdfunding_investor_update_workflow | Publish investor updates linked to campaign disclosures |
| crowdfunding_compliance_workflow | Record and review compliance alerts with severity controls |
| crowdfunding_review_workflow | Governance reviews for disclosures, milestones, and compliance alerts |
| crowdfunding_agent_workflow | Register AI agents for issuer due diligence, disclosure review, and escrow release |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Investor and issuer notifications |
| nlpc | NLP for disclosure and narrative analysis |
| keym | Key management |
| fintech_payments | Payment execution for commitments and payouts |
| fintech_wallets | Escrow wallet management |
| fintech_kyc | Issuer and investor identity verification |
| fintech_aml | AML screening for issuers and investors |
| fintech_fraud | Fraud screening |
| fintech_portfolio | Investor portfolio integration |
| fintech_wealth | Wealth management for accredited investors |
| bia | Business intelligence and analytics |
| fin_rpt | Financial reporting |

## Quick Start

```python
from capabilities.fintech.crowdfunding.service import CrowdfundingService

svc = CrowdfundingService(tenant_id="acme", actor_id="admin")

# Onboard an issuer
issuer = await svc.onboard_issuer(
    issuer_id="iss_001",
    name="Acme Solar Ltd",
    kyc_reference="kyc_ref_001",
    beneficial_owner_reference="ubo_ref_001",
    risk_rating_reference="risk_ref_001",
)

# Launch a campaign
campaign = await svc.launch_campaign(
    creator_id="iss_001",
    title="Acme Solar Series A",
    goal_amount=5_000_000.0,
    currency="KES",
    deadline="2026-12-31T23:59:00+03:00",
    campaign_type="equity",
)

# Approve it
await svc.campaign_moderation(campaign["id"], "approve", "disclosure review passed")

# Record an investor contribution
result = await svc.contribute(
    contributor_id="inv_001",
    campaign_id=campaign["id"],
    amount=50_000.0,
    payment_method="mpesa",
)
```

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-crowdfunding/dashboard | GET | fintech_crowdfunding:view | Overview |
| issuers | /fintech-crowdfunding/issuers | GET/POST | fintech_crowdfunding:issuers | Issuers |
| campaigns | /fintech-crowdfunding/campaigns | GET/POST | fintech_crowdfunding:campaigns | Campaigns |
| disclosures | /fintech-crowdfunding/disclosures | GET/POST | fintech_crowdfunding:disclosures | Campaigns |
| commitments | /fintech-crowdfunding/commitments | GET/POST | fintech_crowdfunding:commitments | Investors |
| escrow | /fintech-crowdfunding/escrow | GET/POST | fintech_crowdfunding:escrow | Funds |
| milestones | /fintech-crowdfunding/milestones | GET/POST | fintech_crowdfunding:milestones | Funds |
| payouts | /fintech-crowdfunding/payouts | GET/POST | fintech_crowdfunding:payouts | Funds |
| updates | /fintech-crowdfunding/updates | GET/POST | fintech_crowdfunding:updates | Investors |
| compliance | /fintech-crowdfunding/compliance | GET/POST | fintech_crowdfunding:compliance | Governance |
| reviews | /fintech-crowdfunding/reviews | GET/POST | fintech_crowdfunding:reviews | Governance |
| agents | /fintech-crowdfunding/agents | GET/POST | fintech_crowdfunding:admin | Automation |
| settings | /fintech-crowdfunding/settings | GET/POST | fintech_crowdfunding:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| issuer_kyc_required | Issuer without KYC | deny |
| issuer_owner_required | Issuer without beneficial owner evidence | deny |
| issuer_risk_rating_required | Issuer without risk rating | deny |
| campaign_disclosure_required | Campaign without disclosure | deny |
| campaign_positive_target | Campaign with zero or negative target | deny |
| commitment_investor_kyc_required | Commitment without investor KYC | deny |
| commitment_risk_ack_required | Commitment without risk acknowledgement | deny |
| escrow_commitment_required | Escrow without funded commitment | deny |
| payout_approval_required | Payout without human approval | deny |
| update_disclosure_required | Investor update without disclosure reference | deny |
| crowdfunding_batch_requires_bytewax | Batch without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| Issuer | id, name, kyc_reference, beneficial_owner_reference, risk_rating, status |
| Campaign | id, issuer_id, name, campaign_type, target_amount, currency, disclosure_reference, status |
| Disclosure | id, campaign_id, disclosure_type, evidence_reference, status |
| InvestorCommitment | id, campaign_id, investor_reference, amount, currency, investor_kyc_reference, risk_ack_reference, status |
| EscrowFunding | id, commitment_id, wallet_reference, amount |
| Milestone | id, campaign_id, title, evidence_reference, status |
| Payout | id, campaign_id, milestone_id, amount, approval_reference, status |
| InvestorUpdate | id, campaign_id, disclosure_reference, recipient_scope |
| ComplianceAlert | id, severity, evidence_reference, status |

## Streaming Events
Events emitted to `apg.fintech.crowdfunding.lifecycle` via Bytewax.
| Event | Trigger |
|-------|---------|
| issuer_onboarded | Issuer passes due diligence |
| campaign_published | Campaign opened to investors |
| disclosure_recorded | Disclosure document attached |
| investor_commitment_recorded | Investor commitment recorded |
| escrow_funding_recorded | Escrow funded from commitment |
| milestone_recorded | Campaign milestone evidenced |
| payout_authorized | Payout authorized against milestone |
| investor_update_published | Update sent to investors |
| crowdfunding_compliance_alert_recorded | Compliance alert raised |
| crowdfunding_review_recorded | Governance review completed |
| crowdfunding_agent_registered | AI agent registered |

## New Methods

### `campaign_analytics(campaign_id)`
Full analytics for a campaign: commitment size distribution, payout history, compliance alert count.
```python
analytics = await svc.campaign_analytics("camp_001")
# {total_raised_minor, avg_commitment_minor, commitment_size_distribution, compliance_alert_count, ...}
```

### `equity_share_allocation(campaign_id)`
Pro-rata ownership table for equity/revenue-share campaigns based on funded commitments.
```python
alloc = await svc.equity_share_allocation("camp_001")
# {"allocations": [{"investor_id": "inv_001", "ownership_pct": 12.5, "share_units": 1250.0}, ...]}
```

### `regulatory_limits_check(contributor_id, campaign_id, amount)`
Validate a proposed contribution against CMA Kenya 2022 limits before accepting funds.
```python
check = await svc.regulatory_limits_check("inv_001", "camp_001", 400_000.0)
# {"within_limits": True, "violations": [], "warnings": [...]}
```

### `cma_crowdfunding_return(period)`
Generate a CMA Kenya periodic regulatory return covering all closed campaigns.
```python
report = await svc.cma_crowdfunding_return("2026-Q2")
# {"report_type": "CMA_CROWDFUNDING_RETURN", "total_campaigns": 14, "total_raised_minor": ..., "status": "draft"}
```

### `investor_accreditation_check(investor_id, net_worth, annual_income)`
Classify an investor as accredited/retail under CMA thresholds (KES 5M net worth or KES 1M income).
```python
acc = await svc.investor_accreditation_check("inv_001", net_worth=6_000_000.0, annual_income=800_000.0)
# {"accredited": True, ...}
```

## World-Class Enhancements (v2.0)

1. **Dynamic Tiered Fee Engine** — Configurable fee schedules by campaign type, volume tier, and issuer history; replaces the flat 3% hard-code.
2. **Real-Time Funding Velocity & Momentum Scoring** — Sliding-window velocity (1h/6h/24h/7d) and a 0–100 momentum score; declining momentum triggers issuer alerts.
3. **Investor Sentiment & Engagement Analytics** — Engagement signals (update reads, disclosure downloads, Q&A) aggregated into a per-campaign sentiment index correlated with conversion rates.
4. **Automated Beneficial Ownership Graph** — Graph-based UBO resolution for multi-level corporate chains; detects circular ownership, undisclosed related parties, and PEPs at campaign launch.
5. **Milestone Evidence Verification with AI-Assisted Review** — Structured evidence pipeline with completeness scoring and an AI reviewer agent that produces pass/fail recommendations with citations.
6. **Cross-Campaign Investor Risk Aggregation** — Real-time portfolio-level exposure tracking; enforces per-campaign CMA limit and sector concentration limits (max 40% in one industry).
7. **Smart Contract-Ready Escrow Integration** — Escrow adapter supporting both wallet-reference and EVM smart contract models; milestone completions trigger on-chain release events.
8. **Sophisticated Investor Fast-Track Workflow** — Dedicated onboarding for accredited investors with higher limits (up to KES 50M) and pre-public "professional tranche" access.
9. **Campaign A/B Testing Framework** — Statistically controlled variant testing across reward structures, equity percentages, and pitch narratives; auto-migrates investor pool to winning variant.
10. **Regulatory Filing Automation Pipeline** — Fully automated CMA Kenya return: queries closed campaigns, computes mandatory tables, signs, and submits to the CMA API; human review only for exceptions.
11. **Investor Communication Drip Workflow** — Milestone-triggered templated drip: pre-launch teasers, 25%/50%/75%/100% progress updates, completion and payout notifications; all delivery receipts logged.
12. **Revenue-Share Distribution Engine** — Periodic pro-rata distribution engine for revenue-share campaigns; handles minimum thresholds, withheld tax, cumulative return caps, and escrow reconciliation.
13. **Campaign Clone & Template Library** — Clone a past campaign or bootstrap from a sector-specific best-practice template; financial targets and dates reset, everything else carries forward.
14. **Dispute Resolution & Investor Protection Workflow** — Structured investor dispute process (misrepresentation, milestone non-delivery, payout delay) with case IDs, independent reviewer routing, and remediation options including partial refunds and campaign suspension.
15. **Predictive Campaign Failure Early Warning System** — ML-informed early warning at 7/14/30 days post-launch using sector, velocity, disclosure completeness, and KYC tier features; high-risk campaigns escalated to a campaign support team.

## Edge Cases Handled
- Campaign disclosure required at publication — campaigns cannot open to investors without a linked disclosure
- Beneficial owner evidence is separate from issuer KYC — entity KYC alone does not satisfy the UBO requirement
- Investor updates must reference a disclosure document — freeform updates not anchored to a filed disclosure are rejected
- Escrow funding requires a commitment in `funded` status, not just `pledged`
- Payout approval required regardless of amount — no low-value exemption

## Composability
- **Upstream**: `fintech_kyc` and `fintech_aml` provide identity evidence; `fintech_wallets` manages escrow accounts; `fintech_payments` executes payouts
- **Downstream**: `fintech_portfolio` receives investment records; `fintech_wealth` supports accredited investor workflows
- **Peer**: Commonly deployed alongside `fintech_portfolio` (investor holdings) and `fintech_compliance` (securities disclosure obligations)

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| campaigns.supported_types | list | equity, debt, reward, donation, revenue_share | Campaign structures |
| campaigns.supported_currencies | list | USD, KES, EUR, GBP, NGN, GHS, ZAR | Supported currencies |
| commitments.supported_statuses | list | pledged, funded, cancelled, refunded | Commitment states |
| disclosures.supported_types | list | offering_memo, risk_factors, financials, use_of_funds, issuer_update | Disclosure categories |
| compliance.supported_severities | list | low, medium, high, critical | Alert severity levels |

## Development Notes
- Campaign types `equity` and `debt` imply securities regulation requirements; the capability enforces disclosure and KYC but does not perform securities registration — that remains an adapter boundary
- `recipient_scope` on investor updates: `all`, `committed`, or `funded`; service layer enforces filtering
- Commitment statuses (`pledged`, `funded`, `cancelled`, `refunded`) are service-layer transitions; the rule engine validates that escrow funding links to a `funded` commitment
- `CrowdfundingPlatformService` is an alias for `CrowdfundingService` for backward compatibility

---
© 2025 Datacraft | www.datacraft.co.ke
