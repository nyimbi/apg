# Crowdfunding Platform

Crowdfunding Platform is an executable APG capability for issuer onboarding,
campaign publishing, disclosures, investor commitments, escrow funding,
milestones, payout authorization, investor updates, compliance alerts, review
workflows, and AI-assisted platform governance.

The package is dependency-light and can run inside generated Python
applications. Production deployments bind the adapter keys in the capability
contract to APG identity, audit, notifications, language/NLP, key management,
payments, wallets, KYC, AML, fraud, portfolio, wealth, analytics, reporting,
and Bytewax services.

## Use

```python
from capabilities.fintech.crowdfunding import CrowdfundingPlatformService

service = CrowdfundingPlatformService()
issuer = service.onboard_issuer(
    "issuer-1", "tenant-1", "Solar Cooperative", "kyc-1", "ubo-1", "risk-1"
)
campaign = service.publish_campaign(
    "campaign-1", "tenant-1", issuer["id"], "Solar Mini Grid",
    "revenue_share", 50000000, "USD", "offering-memo-1"
)
commitment = service.record_commitment(
    "commitment-1", "tenant-1", campaign["id"], "investor-1",
    250000, "USD", "investor-kyc-1", "risk-ack-1"
)
service.record_escrow_funding(
    "funding-1", "tenant-1", commitment["id"], "wallet-1", 250000
)
```

## Capability Surfaces

- Issuer onboarding with KYC, beneficial-owner, and risk-rating evidence.
- Campaign publishing for equity, debt, reward, donation, and revenue-share
  offers.
- Disclosure recording for offering memoranda, risk factors, financials, use
  of funds, and issuer updates.
- Investor commitments with KYC, risk acknowledgement, positive amount, and
  currency controls.
- Escrow funding, milestone evidence, payout authorization, and investor-update
  workflows.
- Compliance alerts and review workflows for governance escalation.
- Provider-neutral AI agent registration across Codex, Claude Code, OpenCode,
  and Pi runtimes.
- Dashboard, issuer, campaign, disclosure, commitment, escrow, milestone,
  payout, update, compliance, review, settings, and agent view models.
- Deterministic rule engine and Bytewax lifecycle stream metadata.

## Integration Boundaries

Live payment capture, wallet settlement, securities registration, investor
accreditation checks, document signing, regulator filing, tax reporting,
secondary trading, and durable Bytewax workers stay behind adapter boundaries.
