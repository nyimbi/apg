# Intelligence Reporting Build Plan

## Packet 1: Contract And Specification

- Define `intel_reporting` as a Python executable capability package.
- Publish a deterministic contract with configuration, taxonomy, UI routes,
  theme tokens, Bytewax lifecycle metadata, dependencies, provided workflows,
  and AI-agent runtime support.
- Document purpose, scope, lifecycle, guardrails, composition boundaries, and
  known adapter exclusions.

## Packet 2: Runtime And Domain Model

- Add tenant-keyed dataclasses for authorities, workspaces, templates,
  products, sections, citations, approvals, distributions, publications,
  reviews, and agents.
- Add a service that enforces rules before mutation and emits Bytewax
  audit-event metadata for lifecycle records.
- Keep implementation deterministic, inspectable, and adapter-friendly.

## Packet 3: Composition Surface

- Add process-local API helpers for generated APG applications.
- Add dashboard, reporting console, and AI-agent workbench view models.
- Add `app.py` with semantic-model generation, component manifest, and self-test.
- Generate `semantic_model.json`, `package_manifest.json`, and
  `release_report.json`.

## Packet 4: Verification And Review

- Add focused tests for contract shape, rule denial paths, full lifecycle
  execution, tenant isolation, guardrail rejection, API/view execution, and
  publishable app entrypoint.
- Run py_compile, app self-test, JSON validation, focused pytest, APG inspect,
  APG publish-plan, implementation audit, lifecycle audit, strict package
  audit, stale-marker scan, disallowed messaging scan, and `git diff --check`.

## Packet 5: Catalog And Progress

- Add `reporting` to implemented Intel subcapabilities.
- Update the capabilities catalog counts and Intel description.
- Record verification evidence and review notes in `docs/progress_log.md`.
- Commit and push the verified slice using the Lore commit protocol.

