# Dark Web Monitoring Capability Specification

## Purpose

Dark Web Monitoring (`intel_darkweb`) enables APG applications to compose
lawful, defensive monitoring workflows for exposure, brand abuse, fraud
markets, threat actor chatter, data leaks, and infrastructure indicators. It
records authority, monitoring programs, monitored sources, passive
observations, exposure indicators, marketplace and threat actor assessments,
referrals, dissemination, reviews, lifecycle events, UI metadata, theming, and
provider-neutral AI-agent participation.

The capability is executable without live network credentials. Generated
applications can use the local runtime for tests and workflows, then provide
adapters for approved ingestion, evidence storage, enrichment, search, and
dissemination.

## Users

- Security analysts monitoring credential exposure, data leaks, fraud, and
  threat actor claims.
- Incident response teams triaging breach or brand-abuse indicators.
- Compliance reviewers validating authority, access review, evidence, and
  release approvals.
- Application builders composing APG security, fraud, legal, or public-safety
  applications.
- AI-agent operators who need provider-neutral automation with deterministic
  guardrails.

## Functional Scope

`intel_darkweb` provides:

- Authority records with classification, approver, expiry, and evidence.
- Monitoring programs for brand protection, credential exposure, data leaks,
  fraud markets, watchlists, threat actor tracking, vulnerability chatter, and
  executive protection.
- Hidden-service source registration with network/source type, custodian,
  lawful authority, access review, and evidence.
- Passive observation records with references, content fingerprints,
  observation time, confidence, and evidence.
- Exposure indicators with type, risk, analyst, confidence, and evidence.
- Marketplace risk and threat actor assessments.
- Referral, dissemination, and review workflows with approval evidence.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Deterministic rule evaluation for all write-path guardrails.
- Bytewax lifecycle metadata for composable event processing.
- UI route metadata, compact view models, and theme tokens.

## Out Of Scope

The capability does not implement live dark-web crawling, marketplace
participation, credential use, exploit procurement, contraband transactions,
private-data collection, identity resolution, anti-bot evasion, account
automation, or doxxing. These are denied by rule where appropriate or left
behind explicit adapter contracts that require separate review.

## Lifecycle

1. Record lawful authority.
2. Record monitoring program under that authority.
3. Register passive source with access review.
4. Record observation evidence linked to program and source.
5. Record exposure indicator from observation evidence.
6. Record marketplace risk and/or threat actor assessment.
7. Record referral or dissemination with approval.
8. Record review outcome.
9. Emit Bytewax lifecycle metadata for every accepted mutation.
10. Allow AI agents only inside configured roles and approved scopes.

## Rules

All service methods evaluate rules before mutating state. Guardrails require
tenant context, write policy, lawful authority, supported types, access review,
program/source authority alignment, observation fingerprint/evidence,
confidence scores between 0 and 1, analyst ownership, referral/dissemination
approvals, review evidence, Bytewax batch routing, supported AI-agent runtimes
and roles, human approval for privileged agent actions, and denial of
credential use, exploit procurement, contraband transactions, evasion, and
doxxing scopes.

## UI And Theme

The capability exposes generated-screen metadata for dashboards, authorities,
programs, sources, observations, indicators, marketplace risk, threat actors,
referrals, dissemination, reviews, agents, and settings. Theme tokens are
compact, operational, and suitable for dense security workflows.

## Adapter Boundaries

Adapters own network access, source ingestion, evidence stores, storage, search,
translation, NLP enrichment, GraphRAG projections, notifications,
dissemination delivery, legal hold, case-management writes, and durable Bytewax
worker topology.
