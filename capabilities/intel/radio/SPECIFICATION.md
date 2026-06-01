# Radio Intelligence Listener Capability Specification

## Purpose

Radio Intelligence Listener (`intel_radio`) enables APG applications to compose
lawful, passive radio-monitoring workflows for public-safety, spectrum
management, interference review, emergency monitoring, asset-signal tracking,
and partner-feed analysis. It records authority, band plans, receivers,
collection sessions, signal observations, classifications, event assessments,
referrals, dissemination, reviews, lifecycle events, UI metadata, theming, and
provider-neutral AI-agent participation.

The capability is executable without live radio hardware. Generated
applications can use the local runtime for tests and workflows, then provide
adapters for approved receivers, SDR drivers, evidence storage, enrichment,
geospatial analysis, search, and dissemination.

## Users

- Public-safety and spectrum analysts monitoring authorized frequencies.
- Incident response teams triaging interference, distress, spoofing suspicion,
  or emergency signal events.
- Compliance reviewers validating authority, calibration, evidence, and release
  approvals.
- Application builders composing APG security, public-safety, logistics, or
  operational-monitoring products.
- AI-agent operators who need provider-neutral automation with deterministic
  guardrails.

## Functional Scope

`intel_radio` provides:

- Authority records with classification, approver, expiry, and evidence.
- Band plans with supported band types and frequency ranges.
- Receiver registration with site, custodian, calibration, authority, and
  evidence.
- Collection sessions linked to authorized bands and receivers.
- Signal observations with frequency, signal type, fingerprint, observation
  time, confidence, and evidence.
- Transmission classification and event assessment workflows.
- Referral, dissemination, and review workflows with approval evidence.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Deterministic rule evaluation for all write-path guardrails.
- Bytewax lifecycle metadata for composable event processing.
- UI route metadata, compact view models, and theme tokens.

## Out Of Scope

The capability does not implement live receiver control, SDR drivers,
transmission, jamming, spoofing, interference, decryption, protected
communication interception, or unauthorized collection. These are denied by
rule where appropriate or left behind explicit adapter contracts that require
separate review.

## Lifecycle

1. Record lawful authority.
2. Record authorized band plan and frequency range.
3. Register receiver under the same authority.
4. Record collection session for the authorized band and receiver.
5. Record signal observation within the band range.
6. Record transmission classification.
7. Record radio event assessment.
8. Record referral or dissemination with approval.
9. Record review outcome.
10. Emit Bytewax lifecycle metadata for every accepted mutation.
11. Allow AI agents only inside configured roles and approved scopes.

## Rules

All service methods evaluate rules before mutating state. Guardrails require
tenant context, write policy, lawful authority, supported types, valid frequency
ranges, calibrated receivers, band/receiver authority alignment,
session/observation evidence, observation frequencies inside the band plan,
confidence scores between 0 and 1, analyst ownership, referral/dissemination
approvals, review evidence, Bytewax batch routing, supported AI-agent runtimes
and roles, human approval for privileged agent actions, and denial of
transmission, unauthorized interception, decryption, jamming, spoofing, and
interference scopes.

## UI And Theme

The capability exposes generated-screen metadata for dashboards, authorities,
band plans, receivers, sessions, observations, classifications, events,
referrals, dissemination, reviews, agents, and settings. Theme tokens are
compact, operational, and suitable for dense radio-monitoring workflows.

## Adapter Boundaries

Adapters own receiver and SDR integration, signal capture, demodulation,
recording storage, geolocation, enrichment, NLP/translation, GraphRAG
projections, notifications, dissemination delivery, regulatory reporting,
maintenance tickets, and durable Bytewax worker topology.
