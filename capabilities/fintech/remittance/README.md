# APG Cross-Border Remittance

`fintech_remittance` is the APG capability for executable cross-border money
movement workflows. It packages corridor governance, FX quotes, transfer
creation, AML/Fraud review, payout release, refund handling, UI metadata,
visual theming, Bytewax lifecycle metadata, and provider-neutral AI-agent
composition into one composable application component.

Kafka is intentionally not part of this package. Remittance lifecycle events
are modeled as Bytewax streams so generated APG applications can use the same
event-processing strategy as the surrounding fintech capabilities.

## What It Provides

- Corridor and currency eligibility checks.
- FX quote creation with amount, rate, fee, expiry, and quote-lock evidence.
- Transfer creation with sender, beneficiary, KYC, AML, fraud, funding,
  payout-method, purpose, and source-of-funds evidence.
- Payout release with provider receipt and settlement reference.
- Refund/return filing with reason and reviewer evidence.
- Deterministic rule evaluation for every high-impact workflow.
- UI/view metadata for dashboard, corridors, quotes, transfers, payouts,
  refunds, agents, and settings.
- Visual theme metadata for remittance operations consoles.
- Provider-neutral AI-agent registration for Codex, Claude Code, OpenCode, and
  Pi.

## Package Files

- `capability_contract.py` defines dependencies, configuration, rules, UI,
  theme, and Bytewax metadata.
- `models.py` defines quotes, transfers, refunds, and evidence records.
- `remittance_runtime.py` normalizes countries, currencies, amounts, corridors,
  payout methods, risk bands, and lifecycle recommendations.
- `service.py` enforces rules during local lifecycle operations.
- `api.py` exposes process-local helpers for generated apps.
- `views.py` exposes framework-neutral view models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and
  `self_test()`.
- `tests/test_package_contract.py` verifies the executable contract and
  runtime.

## Usage

```python
from capabilities.fintech.remittance import RemittanceService

service = RemittanceService()
quote = service.create_quote(
	"quote-1", "tenant-a", "KE", "UG", "KES", "UGX", 1000, 28.5, 20, "2026-06-02T00:00:00Z"
)
transfer = service.create_transfer(
	"transfer-1", "tenant-a", quote["id"], "sender-1", "beneficiary-1",
	"sender-kyc", "beneficiary-kyc", "wallet-hold-1", "mobile_money",
	"family_support", "salary", "aml-clear", "clear"
)
payout = service.release_payout("transfer-1", "tenant-a", "provider-receipt-1", "settlement-1")
```

## Rules

The package exposes deterministic rules for tenant context, write policy,
supported corridors/currencies, quote amount/rate/fee/expiry, transfer quote
lock, sender and beneficiary evidence, KYC, funding, payout method, purpose,
source of funds, AML evidence, sanctions blocking, fraud decisions, high-value
approval, payout settlement, refund evidence, Bytewax routing, supported
agent runtimes/roles, and privileged-agent approval.

## Composition

Remittance composes with:

- `fintech_payments` for funding and provider authorization.
- `fintech_wallets` for stored-value funding and payout rails.
- `fintech_kyc` for sender and beneficiary identity evidence.
- `fintech_aml` for sanctions, PEP, adverse-media, and typology review.
- `fintech_fraud` for fraud-risk and account-takeover decisions.
- `auth`, `audl`, `ntfy`, `nlpc`, and `keym` for platform governance.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/fintech/remittance/__init__.py capabilities/fintech/remittance/capability_contract.py capabilities/fintech/remittance/models.py capabilities/fintech/remittance/remittance_runtime.py capabilities/fintech/remittance/service.py capabilities/fintech/remittance/api.py capabilities/fintech/remittance/views.py capabilities/fintech/remittance/app.py capabilities/fintech/remittance/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/remittance/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/remittance/app.py
./.venv/bin/apg capabilities inspect fintech_remittance --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/remittance --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/remittance --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/remittance --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

## Deferred Adapter Work

- Live FX quote providers and treasury liquidity.
- Live bank, card, wallet, and mobile-money payout providers.
- Live sanctions/PEP/adverse-media providers.
- Regulator filing and travel-rule adapters.
- Durable Bytewax deployment and treasury reconciliation.
