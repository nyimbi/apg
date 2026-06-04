# Fraud Detection Capability Specification

## Purpose

`fintech_fraud` is the APG capability for transaction fraud prevention, account
takeover detection, device and behavioral risk, decisioning, fraud case
investigation, chargeback evidence, and AI-assisted fraud operations. It makes
fraud controls first-class and composable for generated fintech applications
without binding the platform to one card processor, wallet provider, ML vendor,
message broker, or AI-agent runtime.

The package must be executable locally. A generated APG application can inspect
the contract, run the service, evaluate deterministic rules, mount UI/view
metadata, and publish release evidence without live payment rails, device
fingerprinting providers, card networks, or model services.

## Users

- Fraud operations analysts reviewing risky transactions and account events.
- Risk managers configuring thresholds, interventions, and escalation paths.
- Chargeback and disputes teams collecting evidence.
- Product engineers composing payment, wallet, KYC, AML, and fraud capabilities.
- AI agents assisting with signal summarization, evidence collation, and case
  prioritization under deterministic policy guardrails.

## Functional Scope

The first executable packet covers:

- Tenant-scoped fraud signal scoring for payment, wallet, account, refund,
  chargeback, and device events.
- Deterministic intervention decisions: approve, step-up, hold, block, and
  review.
- Fraud case creation, investigation state, resolution, and evidence references.
- Guardrails for account takeover, synthetic identity, mule-account, chargeback
  abuse, device anomaly, velocity, geography, high-risk KYC, and AML linkage.
- Provider-neutral fraud agent registration for Codex, Claude Code, OpenCode,
  and Pi runtimes.
- Framework-neutral UI route metadata, view models, theme tokens, and route
  permissions.
- Bytewax stream metadata for fraud lifecycle/event processing.

## Out Of Scope For This Packet

- Live model inference and training.
- Live device-fingerprinting or behavioral biometric providers.
- Live card-network chargeback submission.
- Durable Bytewax topology deployment.
- Graph entity resolution across institutions.

Those concerns remain explicit adapter boundaries.

## Dependencies

Required APG capabilities:

- `auth` for identity, challenge, and authorization.
- `audl` for durable audit trails.
- `ntfy` for alerts and review notifications.
- `nlpc` for narrative and evidence assistance.
- `keym` for protected references.
- `fintech_payments` for payment events.
- `fintech_wallets` for wallet transfers and balances.
- `fintech_kyc` for identity risk and customer profile evidence.
- `fintech_aml` for AML alert/case linkage.

The lifecycle processor is Bytewax. Bytewax is intentionally not part of this
capability contract.

## Configuration

The contract exposes tenant-safe defaults for:

- Scoring thresholds for review, step-up, hold, and block actions.
- Supported signal types, channels, and event sources.
- Velocity, device, geography, chargeback, and account-takeover indicators.
- Intervention policy and supported decisions.
- Case policy and supported case types.
- AI-agent runtime and role support.
- Governance requirements for tenant context, policy evidence, KYC link, audit,
  and human approval for high-impact interventions.
- UI route and theme metadata.
- Adapter boundaries for APG dependencies and Bytewax.

## Deterministic Rule Engine

The rule engine must deny or require review for:

- Missing tenant context.
- Writes without fraud policy evidence.
- Signals without subject, source reference, KYC profile, currency, or positive
  amount where money is present.
- Unsupported signal type or channel.
- Risk scores outside 0-100.
- High-risk scores without review.
- Velocity, device, geography, AML, chargeback, and account-takeover indicators
  without review evidence.
- Unsupported fraud decisions.
- Step-up decisions without challenge references.
- Hold/block decisions without reason and human approval.
- Case creation without signal, supported case type, investigator, and evidence.
- Case resolution without disposition and reviewer.
- Fraud batches or events not routed through Bytewax.
- Unsupported fraud agent runtime or role.
- Privileged fraud-agent actions without human approval.

Service methods must enforce the same rule names exposed by the contract.

## UI Requirements

The capability publishes route metadata for:

- Dashboard
- Signals
- Decisions
- Cases
- Chargebacks
- Devices
- AI agents
- Settings

View models are plain dictionaries so APG can mount them into generated Python
applications or future UI shells without a framework dependency.

## Runtime Evidence

The package must provide:

- `app.py` self-test.
- `semantic_model.json`.
- `package_manifest.json`.
- `release_report.json`.
- Focused package tests.
- `cap_spec.md` with proof commands and known gaps.

## Review Criteria

The capability is serviceable when:

- `get_capability_contract()` validates in the APG registry.
- The service can score fraud signals, make intervention decisions, open and
  resolve cases, register agents, validate Bytewax batches, and produce
  dashboard summaries.
- Tests prove guardrails for tenant context, policy, KYC linkage, Bytewax,
  high-risk decisions, intervention approval, and agent runtimes.
- No placeholder marker files or baseline-only implementation remains.
