# FEDL Capability Specification

## Purpose

Federated Learning provides the APG privacy-preserving AI training substrate.
It lets applications learn from distributed participant data without central
data movement, while enforcing attestation, residency, privacy-budget, secure
aggregation, poisoning-defense, MLCM release, audit, UI, and theming contracts.
FEDL also treats AI agents as first-class federation governance actors so
Codex, Claude Code, OpenCode, Pi, and later provider runtimes can participate
through an accountable, tenant-scoped, policy-checked composition surface.

## Lifecycle

The coherent lifecycle packet is:

1. Create a tenant-scoped federation with coordinator, objective, model family,
   privacy budget, and data-residency regions.
2. Register attested participants with contracts, allowed regions, and compute
   profile metadata.
3. Start approved training rounds only when the minimum participant count is
   met and privacy controls are within policy.
4. Accept participant updates for active rounds and compute deterministic
   update digests.
5. Quarantine poisoned or low-quality updates.
6. Securely aggregate accepted updates when every round participant has
   submitted valid evidence.
7. Register the resulting federated model version.
8. Release federated models through MLCM linkage with approval and privacy
   review evidence.
9. Retire federations after impact review.
10. Register federation agents with runtime, role, scope, owner, purpose,
    contribution disclosure, and privileged-role review status.
11. Validate FEDL lifecycle mutation batches through Bytewax-backed stream
    contracts before accepting generated-app or agent-authored changes.
12. Emit audit evidence for state changes and policy decisions.

## Functional Requirements

- The capability must expose an executable contract through
  `get_capability_contract()`.
- Configuration must cover federation, participants, privacy, training,
  aggregation, model release, agents, streaming, governance, observability,
  adapters, UI, and theme.
- The deterministic rule engine must remain dependency-light.
- Generated applications must be able to call `FedlService` without external
  infrastructure.
- UI routes must cover dashboard, federations, participants, attestation,
  rounds, updates, aggregation, privacy, security, models, release, agents,
  lifecycle, audit, and settings.
- The theme must provide privacy-mesh console tokens and component hints.
- Package evidence must be generated from the current contract.
- Bytewax must be the event-stream adapter; broker-specific queue is intentionally not used.
- Agent composition must remain provider-neutral through the AICR adapter
  contract, with current runtime codes for `codex`, `claude_code`,
  `opencode`, and `pi`.
- Privileged federation agent roles must be allowed only with explicit human
  approval evidence or held in pending review.

## Runtime Requirements

- `FedlService` owns in-process records for federations, participants, rounds,
  model updates, aggregations, federated models, releases, federation agents,
  lifecycle-batch evidence, and audit events.
- Service methods must raise `PermissionError` for policy denials and `KeyError`
  for missing tenant-scoped records.
- Summary and list calls must remain tenant-scoped.
- Compatibility calls `create_record()` and `list_records()` must continue to
  map to federation behavior for older package callers.
- `register_federation_agent()` must normalize runtime and role tokens, retain
  contribution disclosure, mark privileged agents without approval as
  `pending_review`, and audit the registration.
- `validate_fedl_lifecycle_batch()` must reject non-Bytewax streams, reject
  empty batches, restrict operations to the declared streaming manifest, and
  retain denied-batch evidence for governance dashboards.

## Guardrails

The contract must include guardrails for tenant context, coordinator evidence,
model-family metadata, objective metric, data residency, privacy budget,
participant attestation, participant contract, participant region, compute
profile, minimum participants, round approval, secure aggregation, privacy
review, federation budget limits, running rounds, round participants, sample
counts, quality scores, quality review, complete update sets, poisoning
signals, aggregate digests, MLCM release linkage, release approval, release
privacy review, retirement impact review, cross-tenant participation, and
Bytewax event streaming. The agent guardrail packet must additionally cover
unsupported agent runtimes, unsupported agent roles, missing agent scope,
missing owner, missing purpose, undisclosed machine contributions,
privileged-role human approval, and non-Bytewax FEDL lifecycle mutation
batches.

## Composition Interfaces

- AICR: AI runtime and model-family integration.
- MLCM: release target for federated model versions.
- ENCR: encryption and secure aggregation support.
- MTEN: tenant boundary and participant tenancy.
- AUTH: permission enforcement.
- AUDL: audit sink for federated training evidence.
- MONI: metrics, health, and privacy budget observability.
- Bytewax: event stream for federated round and participant events.
- Provider-neutral AI agents: Codex, Claude Code, OpenCode, Pi, and future
  runtimes connect through AICR-owned adapters rather than hard-coded SDKs.

## Non-Goals For This Packet

- Live distributed training.
- Live Bytewax stream execution.
- Live external agent runtime execution.
- Persistent database schema migration.
- Browser-rendered UI implementation.
- Real secure multiparty computation or homomorphic encryption.
- Load, latency, convergence, poisoning, privacy, and throughput benchmarking.
