# SCRP Development Plan

## Objective

Turn SCRP into a coherent lifecycle and guardrail packet for governed data
harvesting in executable APG applications.

## Build Steps

1. Specify the packet.
   - Define source, extractor, job, run, result, handoff, harvest-agent, audit,
     UI, theme, rule, and Bytewax stream requirements.
   - Keep live scraping, vault, DLP, scheduler, and ETL providers behind
     adapters.

2. Align the capability contract.
   - Add harvest-agent, governance, observability, adapter, UI, theme, and
     Bytewax stream configuration.
   - Add deterministic rules for the lifecycle and guardrails.
   - Ensure rule matching supports numeric and inequality suffixes.

3. Complete the executable runtime.
   - Add the `HarvestAgent` model.
   - Extend `ScrpService` with tenant-safe agent registration and guarded job
     state changes.
   - Preserve source, extractor, job, run, result, handoff, and audit flows.
   - Keep state in-memory and dependency-light for package tooling.

4. Complete composition surfaces.
   - Extend API helpers for agents, job state changes, listings, and audit.
   - Extend view models for agents, audit, analytics, settings, and Bytewax
     stream metadata.
   - Update capability registration metadata, permissions, endpoints, optional
     dependencies, and capabilities.

5. Refresh package evidence.
   - Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the contract.
   - Confirm generated evidence includes Bytewax, harvest agents, routes, and
     expanded rules.

6. Review and verify.
   - Run focused compile checks and SCRP package tests.
   - Run package self-test, implementation audit, publish-plan, stale-marker
     search, and `git diff --check`.
   - Fix emergent issues before committing.

## Risks And Controls

- Live scraping can produce side effects. Keep the package metadata-only and
  require future adapters for network access.
- AI-agent automation can obscure accountability. Require registration, scope,
  supported runtime/role, policy reference, contribution disclosure, and audit.
- Cross-tenant state can leak if IDs are reused. Key tenant-local stores by
  stable tenant-qualified identifiers.
- Batch processing can drift away from the platform stream standard. Require
  Bytewax in the stream contract and rule engine.
- Battery constraints limit verification scope. Run focused SCRP checks now and
  leave broader repository and live-adapter checks documented as not run.

## Completion Evidence

- Focused compile and pytest checks pass.
- Package self-test passes.
- Generated semantic model confirms:
  - `streaming.processor == "bytewax"`
  - supported harvest-agent runtimes include Codex, Claude Code, OpenCode, Pi
  - `/scrp/agents` is exposed
- Implementation audit reports no SCRP errors or warnings.
- Publish-plan reports SCRP is side-effect free.
- Stale-marker search returns no matches.
- Progress log records the packet and known verification gaps.

