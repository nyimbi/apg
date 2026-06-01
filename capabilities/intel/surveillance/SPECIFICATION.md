# Digital Surveillance Capability Specification

## Purpose

Digital Surveillance (`intel_surveillance`) enables APG applications to compose
lawful, defensive monitoring workflows for authorized assets, facilities,
endpoints, accounts, devices, public areas, network segments, and cloud
resources. It records authority, programs, monitored assets, sensors,
observations, alerts, risk assessments, referrals, dissemination, reviews,
lifecycle events, UI metadata, theming, and provider-neutral AI-agent
participation.

The capability is executable without live sensors or surveillance systems.
Generated applications can use the local runtime for tests and workflows, then
provide adapters for approved cameras, endpoint telemetry, network sensors,
access-control feeds, evidence storage, enrichment, search, and dissemination.

## Users

- Security analysts monitoring authorized assets and facilities.
- Incident response teams triaging alerts from approved sensors.
- Compliance reviewers validating authority, privacy review, evidence, and
  release approvals.
- Application builders composing APG security, public-safety, facility,
  endpoint, fraud, or operational-monitoring products.
- AI-agent operators who need provider-neutral automation with deterministic
  guardrails.

## Functional Scope

`intel_surveillance` provides:

- Authority records with classification, approver, expiry, and evidence.
- Surveillance programs for asset protection, facility monitoring, endpoint
  monitoring, fraud monitoring, public safety, incident watch, compliance
  monitoring, and executive protection.
- Monitored assets with owner, lawful authority, privacy review, and evidence.
- Sensor registration with asset, custodian, calibration, and evidence.
- Observation records with references, content fingerprints, observation time,
  confidence, and evidence.
- Alert and risk assessment workflows with analyst, confidence, risk, and
  evidence fields.
- Referral, dissemination, and review workflows with approval evidence.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Deterministic rule evaluation for all write-path guardrails.
- Bytewax lifecycle metadata for composable event processing.
- UI route metadata, compact view models, and theme tokens.

## Out Of Scope

The capability does not implement covert tracking, stalking, spyware,
credential capture, bypass, biometric identification, exfiltration, live sensor
control, or unauthorized monitoring. These are denied by rule where
appropriate or left behind explicit adapter contracts that require separate
review.

## Lifecycle

1. Record lawful authority.
2. Record surveillance program under that authority.
3. Record monitored asset with privacy review.
4. Register approved sensor for the asset.
5. Record observation linked to program and sensor.
6. Record alert from observation evidence.
7. Record risk assessment.
8. Record referral or dissemination with approval.
9. Record review outcome.
10. Emit Bytewax lifecycle metadata for every accepted mutation.
11. Allow AI agents only inside configured roles and approved scopes.

## Rules

All service methods evaluate rules before mutating state. Guardrails require
tenant context, write policy, lawful authority, supported types, privacy
review, calibrated sensors, program/sensor authority alignment, observation
fingerprint/evidence, confidence scores between 0 and 1, analyst ownership,
referral/dissemination approvals, review evidence, Bytewax batch routing,
supported AI-agent runtimes and roles, human approval for privileged agent
actions, and denial of covert tracking, stalking, spyware, credential capture,
bypass, biometric identification, and exfiltration scopes.

## UI And Theme

The capability exposes generated-screen metadata for dashboards, authorities,
programs, assets, sensors, observations, alerts, risk, referrals,
dissemination, reviews, agents, and settings. Theme tokens are compact,
operational, and suitable for dense monitoring workflows.

## Adapter Boundaries

Adapters own camera, endpoint, network, access-control, location, telemetry,
partner-feed, evidence-store, Computer Vision, NLP/translation, GraphRAG,
geospatial, notification, case-management, dissemination, and durable Bytewax
worker integrations.
