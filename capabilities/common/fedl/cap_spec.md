# Federated Learning Capability Specification

- **Capability Name**: Federated Learning
- **Capability ID**: `fedl`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`fedl` provides executable privacy-preserving federated learning orchestration
for APG applications. It manages tenant-scoped federations, participant
attestation, data-residency boundaries, training-round approval, differential
privacy budgets, participant model updates, poisoning-signal quarantine, secure
aggregation, model registration, and auditable governance evidence.

The package is dependency-light by design. It proves the FEDL domain model,
rule enforcement, API helpers, UI view models, deterministic aggregation
evidence, semantic-model publication, and publish-plan readiness while keeping
production ML platforms, key management, monitoring, audit persistence, and
distributed training runtimes behind APG integration boundaries.

## Provided Services

- `federation_coordination`: create tenant-aware federated learning groups with
  coordinator, model family, objective metric, privacy limit, and residency
  policy.
- `participant_attestation`: register participant nodes only after attestation,
  contract reference, and data-residency checks.
- `training_round_monitoring`: start approved training rounds once minimum
  participant requirements and privacy budget checks pass.
- `secure_aggregation`: aggregate accepted participant updates only when secure
  aggregation is enabled and no poisoning signal is present.
- `privacy_budgeting`: track privacy epsilon allocated to rounds and spent by
  completed aggregations.
- `poisoning_defense`: quarantine suspicious updates and block aggregation when
  poisoning evidence exists.
- `federated_model_governance`: register deterministic model versions produced
  by secure aggregation.

## Required Services

- `tenant_context`: all executable operations require a tenant identifier.
- `aicr`: downstream AI core model governance and inference handoff.
- `mlcm`: model lifecycle registration and promotion boundary.
- `encr`: secure channel and aggregation key boundary.
- `mten`: tenant isolation and residency governance.

## Optional Services

- `moni`: runtime monitoring and participant health telemetry.
- `audl`: append-only audit trail persistence.
- `keym`: production key management and secure aggregation material.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. The contract includes:

- `federation`: coordinator enablement, participant attestation, and minimum
  participant requirements.
- `privacy`: secure aggregation, differential privacy, and maximum privacy
  epsilon requirements.
- `training`: training-round approval, update validation, and poisoning defense
  controls.
- `governance`: tenant context, data residency, audit, and participant contract
  requirements.
- `ui`: federation console, round monitor, privacy budget, and participant map
  toggles.
- `theme`: tenant-overridable privacy-mesh visual theme tokens.

## Rules

- `tenant_context_required`: deny operations without tenant context.
- `participant_requires_attestation`: deny joining a federation without
  participant attestation.
- `round_requires_minimum_participants`: deny starting a round with fewer than
  three participants.
- `secure_aggregation_required`: deny aggregation without secure aggregation.
- `privacy_budget_requires_review`: require review when privacy epsilon exceeds
  the configured threshold and no review is recorded.
- `poisoning_signal_blocks_round`: deny aggregation when poisoning signals are
  detected.

## UI

The package exposes APG Python view models for:

- Dashboard: route, rule, theme, summary, and operational state.
- Federation console: federations, participants, residency state, and lifecycle
  status.
- Participant map: attested participants, compute profile, region, and contract
  references.
- Training round monitor: rounds, updates, secure aggregation, and aggregation
  status.
- Privacy budget console: spent epsilon, active round epsilon, and review
  controls.
- Security console: accepted updates, quarantined updates, and poisoning
  evidence.
- Federated model registry: registered model versions and aggregation evidence.

## Theme

The package uses the `fedl_privacy_mesh` APG theme contract. It defines compact
operational UI tokens and component-level visual contracts for participant
cards, round timelines, privacy budget meters, and federation topology views.

## External Runtime Boundary

The repository runtime uses deterministic hashing for update digests,
aggregation evidence, and model version identifiers. Production deployments
should wire APG adapters for distributed training runtimes, secure aggregation
protocols, encryption/key management, participant telemetry, audit vaults, and
model lifecycle promotion without changing the FEDL capability contract.
