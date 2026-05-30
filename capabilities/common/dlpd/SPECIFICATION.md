# DLPD Capability Specification

## Purpose

DLPD provides first-class data loss prevention composition for APG
applications. It lets generated apps classify sensitive content, enforce
tenant-scoped egress policies, quarantine risky transfers, open incidents, and
audit decisions without binding the generated app to one vendor DLP product.

## Scope

DLPD owns:

- DLP policy registration and channel/classifier binding.
- Built-in and custom classifier governance.
- Deterministic generated-app classification helpers.
- Egress inspection and decisioning.
- Quarantine metadata, legal hold, incident response, and review routing.
- Digest-backed audit events.
- Contract-derived UI routes, view payloads, theme tokens, and package evidence.

DLPD does not own:

- Live network interception or mail gateway enforcement.
- Long-term quarantine storage or raw-content retention.
- Production NLP/model providers.
- Persistent database migrations.
- Compliance case-management workflows outside the capability boundary.

## Configuration

The contract must expose tenant-scoped configuration for data patterns,
policies, channels, classification, response, quarantine, incidents, reviews,
security, governance, observability, adapters, UI, and theme.

Required adapter evidence:

- `service.DlpdService` for generated runtime execution.
- `dlp_engine.py` for dependency-light classifier helpers.
- `api.py` and `views.py` for generated API/view payloads.
- `bytewax` for event-stream composition.
- `secu`, `encr`, `nlpc`, `anom`, `audl`, `mqeb`, `srch`, `comp`, `moni`,
  and `cach` as integration adapter points.

## Runtime Lifecycle

Policy lifecycle:

1. Register a tenant-local policy.
2. Require owner and channels.
3. Route missing classifiers for review.
4. Audit policy state changes.

Classifier lifecycle:

1. Register built-in or custom classifiers.
2. Require sensitivity labels and pattern keys.
3. Require review for custom classifiers.
4. Use deterministic local patterns in the generated runtime.

Inspection lifecycle:

1. Inspect only tenant-local active policies.
2. Ensure the channel is covered by the policy.
3. Classify content into hits, severity, and labels.
4. Enforce classification, high-severity, secret, destination, and large-export
   guardrails.
5. Quarantine or open incidents when required.

Quarantine and incident lifecycle:

1. Store content hash and metadata only.
2. Require encrypted quarantine.
3. Apply legal hold by default.
4. Resolve incidents only with resolution notes.
5. Record digest-backed audit events.

## Rules

The rule engine is deterministic and returns `allow`, `require_review`, or
`deny`. Rules must cover tenant context, policies, classifiers, inspections,
classification, channels, destinations, quarantine, incidents, reviews,
Bytewax batch mutation, cross-tenant isolation, raw-content retention, and
audit requirements.

## UI

The route manifest must include dashboard, policies, classifiers, channels,
inspections, incidents, quarantine, reviews, legal hold, analytics, audit, and
settings. View helpers must return dependency-light dictionaries suitable for
generated Python apps.

## Theming

The default theme is `dlpd_data_protection_ops`. It defines compact density,
8px radius, protection/status color tokens, and named theme components for
classifier grids, policy matrices, channel flows, inspection tables, incident
queues, quarantine vaults, review queues, legal holds, and audit timelines.

## Verification Requirements

The focused packet is serviceable when:

- The contract shape validates.
- The package self-test passes.
- The rule count is at least 30.
- The route count is at least 12.
- Bytewax adapter evidence is present.
- The runtime can register classifiers, register policies, inspect egress,
  quarantine content, open/resolve incidents, audit actions, and isolate
  tenants.
- Focused DLPD tests pass without requiring full repository execution.
