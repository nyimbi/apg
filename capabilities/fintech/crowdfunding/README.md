# Crowdfunding Platform

## Overview
Crowdfunding Platform manages the lifecycle of alternative finance campaigns: issuer due diligence, campaign publishing across equity, debt, reward, donation, and revenue-share structures, investor disclosure management, commitment recording, escrow funding, milestone tracking, payout authorization, investor updates, compliance alerts, and review workflows. It is designed for regulated crowdfunding operations where every campaign requires disclosure review before investors can commit.

Investor commitments require KYC and explicit risk acknowledgement. Payouts are gated behind milestone evidence and human approval. All platform lifecycle events stream to `apg.fintech.crowdfunding.lifecycle` via Bytewax.

## Capability ID
`fintech_crowdfunding`  Version: 1.1.0

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

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| campaigns.supported_types | list | equity, debt, reward, donation, revenue_share | Campaign structures |
| campaigns.supported_currencies | list | USD, KES, EUR, GBP, NGN, GHS, ZAR | Supported currencies |
| commitments.supported_statuses | list | pledged, funded, cancelled, refunded | Commitment states |
| disclosures.supported_types | list | offering_memo, risk_factors, financials, use_of_funds, issuer_update | Disclosure categories |
| compliance.supported_severities | list | low, medium, high, critical | Alert severity levels |

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
Events emitted to the fintech event stream via Bytewax.
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

## Edge Cases Handled
- Campaign disclosure is required at campaign publication — a campaign cannot be opened to investors without a linked disclosure, preventing uninformed commitments
- Beneficial owner evidence is separate from issuer KYC — entity KYC alone does not satisfy the UBO requirement
- Investor updates must reference a disclosure document — freeform updates that are not anchored to a filed disclosure are rejected
- Escrow funding requires a commitment in `funded` status, not just `pledged` — the funding record is rejected if the commitment has not been confirmed as funded
- Payout approval is required regardless of payout amount — there is no low-value payout exemption; every payout must carry an approval reference

## Composability
- **Upstream**: `fintech_kyc` and `fintech_aml` provide issuer and investor identity evidence; `fintech_wallets` manages escrow accounts; `fintech_payments` executes payouts
- **Downstream**: `fintech_portfolio` receives investment records for investor portfolio tracking; `fintech_wealth` supports accredited investor workflows
- **Peer**: Commonly deployed alongside `fintech_portfolio` (investor holdings) and `fintech_compliance` (securities disclosure obligations)

## Development Notes
- Campaign types `equity` and `debt` imply securities regulation requirements; the capability enforces disclosure and KYC but does not perform securities registration — that remains an adapter boundary
- Disclosure review (`review_required: True` in configuration) means a disclosure must have a recorded review before the campaign can be published
- `recipient_scope` on investor updates can be `all`, `committed`, or `funded`; the service layer enforces filtering — the rule engine only checks the field is present
- Commitment statuses (`pledged`, `funded`, `cancelled`, `refunded`) are transitions managed by the service; the rule engine validates that escrow funding links to a `funded` commitment
