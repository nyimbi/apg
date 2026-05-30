# LOGT Capability Packet Plan

## Scope

Build one coherent LOGT lifecycle and guardrail packet that turns the existing
logging and tracing runtime into a documented, composable APG capability with
explicit Bytewax stream governance, AI observability agents, tenant-local
state, UI view contracts, and focused proof.

## Implementation Steps

1. Specification and documentation
   - Add `README.md` with usage, composition, agent, and verification guidance.
   - Add `SPECIFICATION.md` with normative domain model, rules, UI, theme,
     stream, and acceptance criteria.
   - Add `PLAN.md` for this packet.
   - Convert `cap_spec.md` into a compatibility pointer.

2. Executable contract
   - Expand configuration with LOGT agents, observability, adapters, and
     Bytewax streaming.
   - Add provides/requires to the contract.
   - Add deterministic guardrails for pipelines, logs, traces, spans, queries,
     exports, AI agents, audit, and batch mutation.
   - Add route and theme metadata for agents and audit.

3. Runtime
   - Add `LogtAgent` model.
   - Store runtime state under tenant-local keys.
   - Enforce pipeline owner, schema, Bytewax stream, sampling policy, trace
     context, span duration, log service, query actor, export approval,
     AI-agent, and batch mutation policies.
   - Preserve audit evidence for lifecycle changes.

4. API and UI
   - Add API helpers for agent registration, audit listing, and Bytewax batch
     validation.
   - Add view models for AI agents, audit trail, and diagnostic policy.
   - Include streaming metadata in dashboard view models.

5. Generated evidence
   - Regenerate `app.py`, `semantic_model.json`, `release_report.json`, and
     `package_manifest.json` from the executable contract.
   - Ensure package manifest includes local docs and focused tests.

6. Verification and review
   - Compile only LOGT package files.
   - Run focused LOGT pytest.
   - Probe service runtime, semantic model, implementation audit, publish plan,
     stale markers, and whitespace checks.
   - Review the diff for tenant isolation, contract/runtime drift, stale
     language, and unverified claims.

## Out Of Scope

- Live log collectors or trace collector connections.
- Persistent search-index writes.
- Durable audit-store writes.
- Rendered browser UI.
- Live Bytewax worker execution.
- Full repository test suite while running on battery.
