# Social Media Intelligence Capability Specification

## Purpose

Social Media Intelligence (`intel_socint`) enables APG applications to compose
lawful public or otherwise authorized social-source intelligence workflows. It
records the authority for collection, the topics being monitored, public or
authorized social sources, post-level evidence, social signals, influence and
network assessments, referrals, dissemination, reviews, lifecycle events, UI
metadata, theming, and provider-neutral AI-agent participation.

The capability is intentionally executable without live platform credentials.
Generated applications can use the in-memory runtime for local orchestration and
tests, then replace external integrations through adapters.

## Users

- Intelligence analysts who need traceable topic, source, post, signal, and
  assessment workflows.
- Compliance reviewers who need authority, terms-review, evidence, approval,
  and dissemination gates.
- Product builders composing ERP, public-safety, fraud, crisis, brand-risk, or
  policy-monitoring applications.
- AI-agent operators who need provider-neutral agent roles with deterministic
  guardrails.

## Functional Scope

`intel_socint` provides:

- Lawful authority records with classification, approver, expiry, and evidence.
- Topic planning for brands, events, threats, public safety, disinformation,
  fraud, crises, and policy monitoring.
- Source registration for accounts, pages, groups, hashtags, keywords, public
  channels, and public sites across supported platform types.
- Post evidence ledgers with post reference, content fingerprint, observation
  time, confidence score, and evidence reference.
- Signal recording for trends, sentiment shifts, coordination,
  misinformation, threats, fraud, crises, and bot-like activity.
- Influence and network assessments with analyst, confidence, risk, and
  evidence fields.
- Referral, dissemination, and review workflows with approval and release
  evidence.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Deterministic rule evaluation for all write-path guardrails.
- Bytewax lifecycle metadata for composable event processing.
- UI route metadata, compact view models, and theme tokens for generated apps.

## Out Of Scope

The capability does not perform live scraping, platform login, cookie
collection, anti-bot evasion, account automation, direct messaging, takedown
actions, private-data collection, identity resolution, harassment, doxxing, or
platform abuse. Those are either denied by rule or left behind explicit adapter
contracts that must be reviewed before implementation.

## Lifecycle

1. Record authority.
2. Record monitoring topic under that authority.
3. Register public or authorized source with terms review.
4. Record post evidence linked to topic and source.
5. Record signal from post evidence.
6. Record influence and/or network assessment.
7. Record referral or dissemination with approval.
8. Record review outcome.
9. Emit Bytewax lifecycle metadata for every accepted mutation.
10. Allow AI agents only inside configured roles and approved scopes.

## Rules

All service methods evaluate rules before mutating state. Guardrails require
tenant context, write policy, lawful authority, supported types, source terms
review, topic/source authority alignment, post fingerprint/evidence, confidence
scores between 0 and 1, analyst ownership, referral/dissemination approvals,
review evidence, Bytewax batch routing, supported AI-agent runtimes and roles,
human approval for privileged agent actions, and denial of harassment,
doxxing, platform abuse, and evasion scopes.

## UI And Theme

The capability exposes generated-screen metadata for dashboards, authorities,
topics, sources, posts, signals, influence, networks, referrals,
dissemination, reviews, agents, and settings. Theme tokens are compact,
operational, and suitable for dense intelligence workflows.

## Adapter Boundaries

Adapters own live platform APIs, source ingestion, storage, search,
translation, NLP enrichment, GraphRAG projections, evidence stores, notification
delivery, dissemination delivery, identity-resolution integrations, and durable
Bytewax worker topology.
