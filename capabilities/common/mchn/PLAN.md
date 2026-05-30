# MCHN Capability Packet Plan

## Scope

Build one coherent MCHN lifecycle and guardrail packet that turns the existing
multi-channel output runtime into a documented, composable APG capability with
explicit Bytewax stream governance, AI output agents, tenant-local state, UI
view contracts, and focused proof.

## Implementation Steps

1. Specification and documentation
   - Add `README.md` with usage, composition, agent, and verification guidance.
   - Add `SPECIFICATION.md` with normative domain model, rules, UI, theme,
     stream, and acceptance criteria.
   - Add `PLAN.md` for this packet.
   - Convert `cap_spec.md` into a compatibility pointer.

2. Executable contract
   - Expand configuration with MCHN agents, observability, adapters, and
     Bytewax streaming.
   - Add provides/requires to the contract.
   - Add deterministic guardrails for channels, templates, policies, rendering,
     delivery, receipts, AI agents, audit, and batch mutation.
   - Add route and theme metadata for agents and audit.

3. Runtime
   - Add `MchnAgent` model.
   - Store runtime state under tenant-local keys.
   - Enforce channel owner, provider reference, template approval, template
     content, policy limits, compliance reference, recipient identity,
     encryption, delivery actor, rendered output presence, Bytewax stream,
     receipt reference, AI-agent, and batch mutation policies.
   - Preserve audit evidence for lifecycle changes.

4. API and UI
   - Add API helpers for agent registration, audit listing, and Bytewax batch
     validation.
   - Add view models for AI agents, audit trail, and delivery governance.
   - Include streaming metadata in dashboard view models.

5. Generated evidence
   - Regenerate `app.py`, `semantic_model.json`, `release_report.json`, and
     `package_manifest.json` from the executable contract.
   - Ensure package manifest includes local docs and focused tests.

6. Verification and review
   - Compile only MCHN package files.
   - Run focused MCHN pytest.
   - Probe service runtime, semantic model, implementation audit, publish plan,
     stale markers, and whitespace checks.
   - Review the diff for tenant isolation, contract/runtime drift, stale
     language, and unverified claims.

## Out Of Scope

- Live notification provider calls.
- Live document rendering or print systems.
- Durable audit-store writes.
- Rendered browser UI.
- Live Bytewax worker execution.
- Full repository test suite while running on battery.
