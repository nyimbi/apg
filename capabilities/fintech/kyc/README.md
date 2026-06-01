# Know Your Customer

Know Your Customer provides APG-generated applications with tenant-scoped
customer identity profiles, consent-backed onboarding, document verification,
sanctions/PEP/adverse-media screening, KYC risk scoring, verification
decisions, enhanced due diligence, and governed AI KYC agents.

The package is dependency-light at the generated-application boundary. It can
run without live document vendors, biometric providers, sanctions feeds,
databases, web frameworks, or queue brokers. Live integrations remain adapter
work behind `auth`, `audl`, `cons`, `ntfy`, `biop`, `cvsn`, `nlpc`, `keym`,
`fintech_payments`, and `fintech_wallets`.

## Runtime Files

- `capability_contract.py`: configuration, deterministic rules, routes, theme,
  dependencies, and Bytewax streaming metadata.
- `models.py`: profile, document, screening, decision, and evidence dataclasses.
- `kyc_runtime.py`: code, country, confidence, and risk helper functions.
- `service.py`: executable KYC lifecycle and guardrail enforcement.
- `api.py`: process-local helper functions for generated applications.
- `views.py`: framework-neutral view models.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused package tests.

## Public Lifecycle

1. `open_profile`
2. `register_document`
3. `record_screening`
4. `score_risk`
5. `record_decision`
6. `register_kyc_agent`
7. `validate_batch`
8. `dashboard_summary`

## Guardrails

The rule engine denies or requires review for missing tenant context, missing
write policy, missing subject/legal-name/country/consent evidence, unsupported
customer or document types, missing tokenized document references, missing
document subject extraction, low document confidence, missing profiles,
screening hits without review, invalid risk scores, high-risk profiles without
enhanced due diligence review, verification decisions without required identity,
address, screening, risk, and consent evidence, unresolved review flags,
non-Bytewax lifecycle batches, unsupported KYC-agent runtimes or roles, and
privileged agent actions without human approval.

## AI KYC Agents

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles include KYC operations, document review, sanctions review, risk review,
and onboarding review. Agents can prepare and recommend actions, but privileged
actions require human approval evidence.

## Bytewax Streaming

All batch and lifecycle metadata uses:

- processor: `bytewax`
- stream: `apg.fintech.kyc.lifecycle`
- key: `tenant_id`

Kafka is intentionally not part of this package boundary.

## Example

```python
from capabilities.fintech.kyc import KnowYourCustomerService

svc = KnowYourCustomerService()
profile = svc.open_profile("kyc-a", "tenant-a", "customer-a", "Amina Njeri", "individual", "KE", "consent-a")
svc.register_document("doc-id", "tenant-a", profile["id"], "national_id", "vault://doc/id", "Amina Njeri", 0.93)
svc.register_document("doc-address", "tenant-a", profile["id"], "utility_bill", "vault://doc/address", "Amina Njeri", 0.91)
svc.record_screening("screen-a", "tenant-a", profile["id"])
svc.score_risk("risk-a", "tenant-a", profile["id"], 22)
svc.record_decision("decision-a", "tenant-a", profile["id"], "approve", 22)
```
