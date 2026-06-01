# Digital Cards Executable Capability

## Runtime Contract

- Capability ID: `fintech_cards`
- Display name: `Digital Cards`
- Version: `1.1.0`
- Target: `python`
- Event stream: `apg.fintech.cards.lifecycle`
- Stream processor: `bytewax`

## Provides

- `card_program_governance`
- `cardholder_card_lifecycle`
- `tokenized_card_credentialing`
- `card_authorization_control`
- `card_dispute_workflow`
- `card_agent_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `encr`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`

## Executable Surface

- `capability_contract.py` exposes configuration, dependencies, rules, UI,
  theme, and Bytewax metadata.
- `models.py` defines card programs, cardholders, cards, tokens,
  authorizations, disputes, and evidence records.
- `cards_runtime.py` normalizes card-domain values and decisions.
- `service.py` enforces deterministic card rules during local lifecycle
  methods.
- `api.py` exposes process-local helper functions for generated apps.
- `views.py` exposes framework-neutral dashboard, card, and rule view models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and `self_test()`.

## Rule Coverage

The package exposes 39 deterministic rules covering tenant context, write
policy, program owner/BIN/currency/settlement evidence, cardholder customer/KYC
and country support, card program/holder/card type/product/wallet/funding/
consent/shipping evidence, token card/type/reference/key/device evidence,
authorization card/amount/currency/merchant/fraud/AML/limit/high-risk review,
dispute transaction/reason/evidence/reviewer evidence, Bytewax routing,
supported agent runtime/role, and privileged-agent approval.
