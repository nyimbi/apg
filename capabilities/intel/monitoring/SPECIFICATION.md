# Real-Time Monitoring Capability Specification

## Purpose

Real-Time Monitoring (`intel_monitoring`) enables APG applications to compose
lawful, defensive monitoring workflows across authorized event, log, metric,
telemetry, partner, case, sensor, and API streams. It records authority,
policies, sources, watches, events, signals, incidents, referrals,
dissemination, reviews, lifecycle events, UI metadata, theming, and
provider-neutral AI-agent participation.

The capability is executable without live stream connectors. Generated
applications can use the local runtime for tests and workflows, then provide
adapters for approved event ingestion, Bytewax topologies, evidence storage,
notifications, case-management writes, search, and dissemination.

## Users

- Security, fraud, public-safety, compliance, and operations analysts.
- Incident response teams triaging correlated events and incidents.
- Compliance reviewers validating authority, source access, evidence, and
  release approvals.
- Application builders composing APG operational, security, fraud, and
  public-safety monitoring products.
- AI-agent operators who need provider-neutral automation with deterministic
  guardrails.

## Functional Scope

`intel_monitoring` provides:

- Authority records with classification, approver, expiry, and evidence.
- Monitoring policies with policy type, severity floor, authority, and
  evidence.
- Source registration with owner, lawful authority, access review, and
  evidence.
- Watch definitions with policy/source alignment, expression, retention class,
  and evidence.
- Event records with references, fingerprints, observation time, confidence,
  and evidence.
- Signal and incident workflows with analyst, severity, confidence, and
  evidence fields.
- Referral, dissemination, and review workflows with approval evidence.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Deterministic rule evaluation for all write-path guardrails.
- Bytewax lifecycle metadata for composable event processing.
- UI route metadata, compact view models, and theme tokens.

## Out Of Scope

The capability does not implement live collectors, destructive response,
autonomous enforcement, data exfiltration, privacy bypass, account actions,
takedowns, or unauthorized monitoring expansion. These are denied by rule where
appropriate or left behind explicit adapter contracts that require separate
review.

## Lifecycle

1. Record lawful authority.
2. Record monitoring policy under that authority.
3. Register authorized monitoring source with access review.
4. Record watch aligned to policy and source.
5. Record event evidence.
6. Record signal from event evidence.
7. Record incident.
8. Record referral or dissemination with approval.
9. Record review outcome.
10. Emit Bytewax lifecycle metadata for every accepted mutation.
11. Allow AI agents only inside configured roles and approved scopes.

## Rules

All service methods evaluate rules before mutating state. Guardrails require
tenant context, write policy, lawful authority, supported types, source access
review, policy/source authority alignment, watch expression, event
fingerprint/evidence, confidence scores between 0 and 1, analyst ownership,
referral/dissemination approvals, review evidence, Bytewax batch routing,
supported AI-agent runtimes and roles, human approval for privileged agent
actions, and denial of destructive actions, autonomous enforcement, privacy
bypass, data exfiltration, unauthorized expansion, account actions, and
takedowns.

## UI And Theme

The capability exposes generated-screen metadata for dashboards, authorities,
policies, sources, watches, events, signals, incidents, referrals,
dissemination, reviews, agents, and settings. Theme tokens are compact,
operational, and suitable for dense monitoring workflows.

## Adapter Boundaries

Adapters own event collectors, log collectors, metric collectors, telemetry
connectors, partner feeds, stream persistence, Bytewax worker topology,
notification delivery, case-management writes, GraphRAG projection,
dashboards, search, and dissemination delivery.
