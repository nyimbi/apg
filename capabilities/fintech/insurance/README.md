# InsurTech

InsurTech is an executable APG capability for policyholder onboarding, product
publishing, quote generation, policy binding, premium recording, claim intake,
document evidence, risk assessment, reinsurance attachment, compliance alerts,
reviews, and AI-assisted insurance operations.

The package is dependency-light and can run inside generated Python
applications. Production deployments bind the adapter keys in the capability
contract to APG identity, audit, notifications, language/NLP, key management,
payments, wallets, KYC, AML, fraud, analytics, reporting, and Bytewax services.

## Use

```python
from capabilities.fintech.insurance import InsurTechService

service = InsurTechService()
holder = service.onboard_policyholder(
    "holder-1", "tenant-1", "Amina Holder", "kyc-1", "contact-1", "risk-1"
)
product = service.publish_product(
    "product-1", "tenant-1", "Motor Protect", "motor", "coverage-1", "pricing-1"
)
quote = service.generate_quote(
    "quote-1", "tenant-1", holder["id"], product["id"], 120000, "USD",
    "underwriting-1"
)
service.bind_policy("policy-1", "tenant-1", quote["id"], "2026-06-01", "payment-1")
```

## Capability Surfaces

- Policyholder onboarding with KYC, contact, and risk-profile evidence.
- Insurance product publishing for life, health, property, motor, travel, crop,
  and microinsurance products.
- Quote generation with policyholder, product, premium, currency, and
  underwriting evidence.
- Policy binding, premium recording, and payment-reference controls.
- Claim intake with policy, claim type, amount, loss date, and evidence.
- Document, risk-assessment, reinsurance, compliance-alert, and review
  workflows.
- Provider-neutral AI agent registration across Codex, Claude Code, OpenCode,
  and Pi runtimes.
- Dashboard, policyholder, product, quote, policy, premium, claim, document,
  risk, reinsurance, compliance, review, settings, and agent view models.
- Deterministic rule engine and Bytewax lifecycle stream metadata.

## Integration Boundaries

Live payment capture, external underwriting engines, repair networks, medical
networks, reinsurance bordereaux, regulator filing, actuarial reserving,
document signing, and durable Bytewax workers stay behind adapter boundaries.
