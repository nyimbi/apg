# Anti Money Laundering Capability Specification

## Purpose

`fintech_aml` is the APG capability for transaction monitoring, AML alert
triage, sanctions escalation, case investigation, suspicious activity report
workflows, typology rules, and AI-assisted compliance operations. It turns
payments, wallet activity, and KYC profiles into reviewable AML evidence without
binding generated applications to one vendor, country, model provider, or queue.

The capability is intentionally executable. A generated APG application must be
able to import the package, inspect its contract, execute its local runtime,
mount its UI route metadata, evaluate deterministic rules, and produce release
evidence without live screening, banking, or regulator credentials.

## Users

- AML operations analysts triaging alerts and monitoring customers.
- Financial-crime investigators managing cases and SAR/STR evidence.
- Compliance officers approving escalations and regulatory filings.
- Product engineers composing fintech applications from APG capabilities.
- AI agents assisting with typology analysis, narrative drafting, evidence
  collation, and queue prioritization under human approval.

## Functional Scope

The first executable packet covers:

- Tenant-scoped transaction monitoring.
- Alert creation from transactions, KYC risk, sanctions hits, velocity, and
  structuring indicators.
- Alert triage into close, review, escalate, or case actions.
- Case creation and investigation state.
- SAR/STR draft lifecycle with mandatory human approval.
- Typology rules for high-value activity, velocity, structuring, high-risk KYC,
  sanctions exposure, mule-account behavior, and agent-assisted review.
- Provider-neutral AML agent registration for Codex, Claude Code, OpenCode, and
  Pi runtimes.
- Framework-neutral UI route metadata, view models, theme tokens, and route
  permissions.
- Bytewax stream metadata for lifecycle/event processing.

## Out Of Scope For This Packet

- Live regulator submission.
- Live sanctions/PEP/adverse-media provider integrations.
- Durable Bytewax topology deployment.
- Graph-database entity resolution.
- ML model training or real-time feature store execution.

Those surfaces are represented by adapter boundaries and deterministic local
runtime behavior so they can be implemented later without changing the APG
contract.

## Dependencies

Required APG capabilities:

- `auth` for identity and authorization.
- `audl` for durable audit trails.
- `ntfy` for alert and review notifications.
- `nlpc` for narrative/entity assistance.
- `keym` for protected references and evidence tokens.
- `fintech_payments` for payment events.
- `fintech_wallets` for wallet activity and balances.
- `fintech_kyc` for customer profiles, screening, and risk scores.

The event processor is Bytewax. Bytewax is intentionally not part of this
capability contract.

## Configuration

The contract exposes tenant-safe defaults for:

- Monitoring thresholds: high-value amount, velocity window, velocity count,
  velocity amount, structuring amount, structuring count, and high-risk score.
- Alert policy: supported alert types, severity set, evidence requirements, and
  auto-close restrictions.
- Case policy: case types, mandatory investigator assignment, review evidence,
  and SAR eligibility.
- SAR policy: human approval, narrative, subject, evidence, and jurisdiction
  requirements.
- Agent policy: supported runtimes, supported roles, human approval for
  privileged actions, and agent event emission.
- Governance: tenant context, policy attachment, KYC linkage, audit, and
  approval requirements.
- UI and theme metadata.
- Adapter boundaries for APG dependencies and Bytewax.

## Deterministic Rule Engine

The rule engine must deny or require review for:

- Missing tenant context.
- Writes without AML policy evidence.
- Transactions without subject, amount, currency, or source references.
- Transaction monitoring without linked KYC evidence.
- High-value transactions above configured thresholds.
- Velocity and structuring indicators.
- Sanctions or high-risk KYC activity without review.
- Unsupported alert type or severity.
- Alerts closed without a disposition reason.
- Case creation without alert evidence.
- Case escalation without investigator assignment.
- SAR drafts without case, subject, jurisdiction, evidence, narrative, or human
  approval.
- AML batches or events not routed through Bytewax.
- Unsupported AML agent runtime or role.
- Privileged AML agent actions without human approval.

Service methods must enforce the same rule names exposed by the contract.

## UI Requirements

The capability publishes route metadata for:

- Dashboard
- Alerts
- Monitoring
- Cases
- SAR/STR workflow
- Typologies
- AI agents
- Settings

View models must be framework-neutral dictionaries so the compiler can mount
them into generated applications without depending on Django, Flask, React, or a
specific UI stack.

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
- The local service can monitor transactions, create alerts, triage alerts,
  open cases, draft SARs, register agents, validate Bytewax batches, and produce
  dashboard summaries.
- Tests prove guardrails for tenant context, policy, KYC linkage, Bytewax,
  high-risk activity, SAR approval, and agent runtimes.
- The package has no placeholder marker files or baseline-only implementation.
