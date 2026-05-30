# EDGE Edge Computing Packet Plan

## Scope

Build `edge` as a coherent lifecycle and guardrail packet for APG applications
that need edge nodes, fleets, signed workloads, deployment placement, offline
sync, resource pressure, AI-agent review, UI metadata, theme metadata, Bytewax
stream governance, and publishable package evidence.

## Implementation Packets

1. Specification and contract
   - Replace stale narrative in `cap_spec.md` with a pointer to the active
     specification.
   - Add `SPECIFICATION.md` for the normative behavior.
   - Expand `capability_contract.py` with configuration, rules, UI routes,
     theme metadata, provides/requires, agent metadata, and Bytewax streaming.

2. Dependency-light service
   - Preserve node, fleet, workload, deployment, sync, audit, and view-model
     behavior already present in `EdgeService`.
   - Add edge-agent data contracts and service methods.
   - Add batch mutation validation tied to the Bytewax stream guardrail.
   - Keep physical devices, runtimes, telemetry stores, distribution, and
     stream workers behind adapters.

3. Package entrypoint and helper surfaces
   - Make `__init__.py` export the expanded contract, service, agent model, and
     stream metadata.
   - Extend API helpers and view models with edge-agent and batch mutation
     surfaces.

4. Documentation and generated evidence
   - Add root package `README.md` with practical usage and composition notes.
   - Refresh semantic model, package manifest, and release evidence from the
     live contract.
   - Update the progress log with proof commands and review notes.

5. Focused proof and review
   - Extend focused contract/service tests without invoking physical device or
     remote runtime fixtures.
   - Run compile checks, focused tests, semantic probes, implementation audit,
     publish plan, stale-marker scan, and diff checks.
   - Review tenant isolation, attestation, secure transport, signed artifact,
     quota, offline sync review, AI-agent boundaries, Bytewax guardrails, import
     behavior, and generated evidence consistency.

## Out Of Scope

- Physical device enrollment and live attestation providers.
- Container, process, or model runtime execution.
- Durable telemetry stores and remote update systems.
- Live Bytewax topology deployment.
- Browser-rendered UI.
- Full repository test suite.

## Review Checklist

- Contract is registry-valid and APG Python route metadata uses practical
  targets.
- Dependency-light package import does not start device, runtime, telemetry, or
  stream services.
- Nodes require owner, attestation, location policy, and secure transport.
- Fleets require owner and policy version.
- Workloads require owner, signed artifacts, deployment policy, and quota.
- Deployments enforce health and capacity.
- Sync requires conflict and cache policy.
- Long offline windows require review.
- AI-agent guardrails include runtime, role, scope, registration, and
  contribution disclosure.
- Batch mutation is rejected unless the event stream is Bytewax.
- Generated semantic evidence matches the executable contract.
