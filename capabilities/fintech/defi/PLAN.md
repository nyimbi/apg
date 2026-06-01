# Decentralized Finance Build Plan

## Packet 1: Contract And Rule Surface

- Define `fintech_defi` metadata, dependencies, configuration, UI routes,
  theme, Bytewax streaming metadata, and deterministic rules.
- Keep AI-agent runtimes provider-neutral: `codex`, `claude_code`, `opencode`,
  and `pi`.
- Require tenant context, policy attachment, evidence, and human approval for
  privileged agent actions.

## Packet 2: Executable Runtime

- Add dependency-light dataclasses for protocols, positions, actions, yield
  strategies, rewards, governance votes, risk assessments, reviews, and agents.
- Implement service methods that evaluate rules before mutating in-memory state.
- Add API helper functions that generated Python applications can call without
  importing a web framework.

## Packet 3: Composition And UI

- Publish view models for dashboard, DeFi console, and agent workbench.
- Publish `app.py` with self-test, component manifest, and semantic model.
- Generate `semantic_model.json` from the package entrypoint.

## Packet 4: Documentation And Evidence

- Write README, specification, capability spec, package manifest, and release
  evidence.
- Update the FinTech capability registry and root capabilities README counts.
- Record implementation and review notes in `docs/progress_log.md`.

## Packet 5: Review And Verification

- Run `py_compile` for package files.
- Run the focused DeFi package tests.
- Run app self-test, inspect, publish-plan, package implementation audit, and
  lifecycle audit.
- Run focused global implementation and strict package audits.
- Scan for stale placeholders and disallowed messaging terminology.
- Commit and push only the verified DeFi slice.

## Review Checklist

- Rule evaluation happens before each state mutation.
- Tenant scoping prevents cross-tenant object reuse.
- Live chain/provider behavior is adapter-only.
- UI routes and theme metadata are complete enough for composition.
- The capability is discoverable by APG CLI inspection and strict package
  audits.
