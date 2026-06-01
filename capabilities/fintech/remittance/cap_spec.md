# Cross-Border Remittance Executable Capability

## Runtime Contract

- Capability ID: `fintech_remittance`
- Display name: `Cross-Border Remittance`
- Version: `1.1.0`
- Target: `python`
- Event stream: `apg.fintech.remittance.lifecycle`
- Stream processor: `bytewax`

## Provides

- `remittance_corridor_governance`
- `remittance_quote_lifecycle`
- `cross_border_transfer_workflow`
- `remittance_payout_workflow`
- `remittance_refund_workflow`
- `remittance_agent_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`

## Executable Surface

- `capability_contract.py` exposes configuration, dependencies, rules, UI,
  theme, and Bytewax metadata.
- `models.py` defines quotes, transfers, refunds, and evidence records.
- `remittance_runtime.py` normalizes remittance domain values and decisions.
- `service.py` enforces deterministic remittance rules during local lifecycle
  methods.
- `api.py` exposes process-local helper functions for generated apps.
- `views.py` exposes framework-neutral dashboard, transfer, and rule view
  models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and `self_test()`.

## Rule Coverage

The package exposes 35 deterministic rules covering tenant context, write
policy, supported corridors/currencies, quote amount/rate/fee/expiry, transfer
quote lock, sender and beneficiary references, sender and beneficiary KYC,
funding, payout method, purpose, source-of-funds evidence, AML screen,
sanctions blocking, fraud decision support, blocked fraud denial, AML/fraud
review approvals, high-value approval, payout receipt/settlement, refund
reason/reviewer, Bytewax routing, supported agent runtime/role, and privileged
agent approval.
