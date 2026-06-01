# APG Digital Cards

`fintech_cards` is the APG capability for executable card issuing and card
operations workflows. It packages card program governance, cardholder onboarding,
virtual/physical card issuance, token lifecycle, authorization controls,
dispute intake, UI metadata, visual theming, Bytewax lifecycle metadata, and
provider-neutral AI-agent composition into one composable application component.

Kafka is intentionally not part of this package. Card lifecycle and
authorization events are modeled as Bytewax streams so generated APG
applications can use the same event-processing strategy as the surrounding
fintech capabilities.

## What It Provides

- Card program registration with owner, BIN range, supported currency, and
  settlement account evidence.
- Cardholder onboarding linked to KYC and supported issuing countries.
- Virtual and physical card issuance linked to wallet and funding account
  references.
- Token provisioning for wallet, device, merchant, and network-token use cases.
- Authorization decisions with amount, currency, merchant category, fraud, AML,
  and limit override guardrails.
- Dispute filing with transaction, reason, evidence, and reviewer assignment.
- Deterministic rule evaluation for high-impact card workflows.
- UI/view metadata for dashboard, programs, cardholders, cards, tokens,
  authorizations, disputes, agents, and settings.
- Provider-neutral card-agent registration for Codex, Claude Code, OpenCode,
  and Pi.

## Package Files

- `capability_contract.py` defines dependencies, configuration, rules, UI,
  theme, and Bytewax metadata.
- `models.py` defines programs, cardholders, cards, tokens, authorizations,
  disputes, and evidence.
- `cards_runtime.py` normalizes card-domain values and decisions.
- `service.py` enforces rules during local lifecycle operations.
- `api.py` exposes process-local helpers for generated apps.
- `views.py` exposes framework-neutral view models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and
  `self_test()`.
- `tests/test_package_contract.py` verifies the executable contract and
  runtime.

## Usage

```python
from capabilities.fintech.cards import CardService

service = CardService()
program = service.register_program(
	"program-1", "tenant-a", "Everyday Debit", "issuer-ops", "411111", "KES", "settlement-1"
)
holder = service.onboard_cardholder("holder-1", "tenant-a", "customer-1", "kyc-1", "KE")
card = service.issue_card(
	"card-1", "tenant-a", program["id"], holder["id"], "virtual", "debit",
	"wallet-1", "funding-1", consent_reference="consent-1"
)
auth = service.authorize_transaction(
	"auth-1", "tenant-a", card["id"], 500, "KES", "grocery",
	"fraud-clear", "aml-clear", fraud_decision="clear", aml_result="clear"
)
```

## Composition

Digital Cards composes with:

- `fintech_payments` for authorization, capture, settlement, and dispute
  handoff.
- `fintech_wallets` for wallet funding, holds, and tokenized wallet cards.
- `fintech_kyc` for cardholder identity evidence.
- `fintech_aml` for restricted-party screening.
- `fintech_fraud` for authorization-risk and account-takeover decisions.
- `auth`, `audl`, `ntfy`, `nlpc`, `keym`, and `encr` for platform governance.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/fintech/cards/__init__.py capabilities/fintech/cards/capability_contract.py capabilities/fintech/cards/models.py capabilities/fintech/cards/cards_runtime.py capabilities/fintech/cards/service.py capabilities/fintech/cards/api.py capabilities/fintech/cards/views.py capabilities/fintech/cards/app.py capabilities/fintech/cards/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/cards/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/cards/app.py
./.venv/bin/apg capabilities inspect fintech_cards --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/cards --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/cards --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/cards --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

## Deferred Adapter Work

- Live card issuer processor and network certification.
- Token-service-provider, 3DS, embossing, and card-personalization adapters.
- PCI DSS production zone implementation.
- Network chargeback submission and clearing-file reconciliation.
- Durable Bytewax deployment.
