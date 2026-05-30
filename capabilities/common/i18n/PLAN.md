# I18N Capability Packet Plan

## Scope

Build one coherent I18N lifecycle and guardrail packet that turns the existing
localization runtime into a documented, composable APG capability with explicit
language policy, AI localization agents, Bytewax stream metadata, tenant-local
state, UI/view contracts, and focused proof.

## Implementation Steps

1. Specification and documentation
   - Add `README.md` with usage, composition, agent, and verification guidance.
   - Add `SPECIFICATION.md` with normative domain model, rules, UI, theme,
     stream, and acceptance criteria.
   - Add `PLAN.md` for this packet.
   - Convert `cap_spec.md` into a compatibility pointer.

2. Executable contract
   - Expand configuration with language policy, African language codes, I18N
     agents, observability, adapters, and Bytewax streaming.
   - Add provides/requires to the contract.
   - Add deterministic guardrails for locale, translation, glossary,
     publication, coverage, AI agents, audit, and batch mutation.
   - Add route and theme metadata for agents and audit.

3. Runtime
   - Add `I18nAgent` and `I18nAuditEvent` models.
   - Store runtime state under tenant-local keys.
   - Enforce supported language codes, glossary ownership, translation key and
     text presence, publication approver, AI-agent registration, and Bytewax
     batch mutation policy.
   - Record audit events for lifecycle changes.

4. API and UI
   - Add API helpers for agent registration, audit listing, and Bytewax batch
     validation.
   - Add view models for AI agents, audit trail, and language policy.
   - Include streaming metadata in dashboard view models.

5. Generated evidence
   - Regenerate `app.py`, `semantic_model.json`, `release_report.json`, and
     `package_manifest.json` from the executable contract.
   - Ensure package manifest includes local docs and focused tests.

6. Verification and review
   - Compile only I18N package files.
   - Run focused I18N pytest.
   - Probe service runtime, semantic model, implementation audit, publish plan,
     stale markers, and whitespace checks.
   - Review the diff for tenant isolation, contract/runtime drift, stale
     language, and unverified claims.

## Out Of Scope

- Live machine-translation provider calls.
- Live NLP provider calls.
- Durable audit-store writes.
- Rendered browser UI.
- Live Bytewax worker execution.
- Full repository test suite while running on battery.
