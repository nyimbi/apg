# GRPH Capability Development Plan

## Objectives

Build GRPH into a coherent APG capability packet with documentation, executable
contract, runtime lifecycle, guardrails, generated UI models, package evidence,
focused tests, review, and progress-log evidence.

## Work Items

1. Document the capability.
   - Add `README.md`.
   - Add `SPECIFICATION.md`.
   - Add this implementation plan.
   - Replace `cap_spec.md` with a short pointer to the packet docs and contract.

2. Expand the executable contract.
	- Add complete configuration sections.
	- Expand deterministic rules to cover lifecycle and guardrails.
	- Add first-class graph-agent composition metadata for Codex, Claude Code,
	  opencode, Pi, and future provider-neutral runtimes.
	- Add Bytewax lifecycle stream metadata and processor guardrails.
	- Expand UI routes to 14 generated-app screens.
	- Add adapter evidence, including Bytewax event streaming.
	- Add graph-specific theme components.

3. Harden runtime behavior.
   - Enforce required schema, node, edge, traversal, lineage, and audit inputs.
	- Add executable review-evidence paths.
	- Add provider-neutral graph-agent registration state.
	- Add Bytewax lifecycle batch validation state.
	- Record audit events for mutations and review-relevant actions.
	- Keep the runtime dependency-light and deterministic.

4. Expand UI and API surfaces.
	- Add route-aligned view models.
	- Expose API helpers for schema, node, edge, traversal, lineage, quality,
	  graph agents, lifecycle batches, audit, and summary surfaces.

5. Refresh package evidence.
   - Replace static app evidence with contract-derived semantic models.
   - Refresh `semantic_model.json`, `release_report.json`, and
     `package_manifest.json`.

6. Verify and review.
   - Run focused compile and pytest checks for GRPH only.
   - Run package audit and publish-plan commands.
   - Run stale-marker and whitespace checks.
   - Use a review agent for the final packet review.
   - Fix review findings before committing.

## Battery-Conscious Verification

Run focused checks only:

- `py_compile` for GRPH modules and tests.
- GRPH unit/package tests.
- `app.self_test()`.
- `apg capabilities implementation-audit --root capabilities/common/grph --json`.
- `apg capabilities publish-plan capabilities/common/grph --json`.
- service smoke for schema -> node -> edge -> traversal -> graph agent ->
  Bytewax lifecycle batch.
- stale-marker scan over GRPH files.
- `git diff --check`.

Full repository tests, live graph database checks, rendered browser UI, live
Bytewax streams, external Codex/Claude Code/OpenCode/Pi clients, external
adapter calls, migrations, and performance benchmarks are deferred.
