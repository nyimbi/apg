# MDM Follow-Up Work

This file tracks follow-up work after the executable MDM capability packet.
The source of truth for current behavior is `SPECIFICATION.md`,
`capability_contract.py`, `service.py`, `view_models.py`, `app.py`, and the
focused package tests.

## Next Runtime Work

- Connect `MdmService` guardrail decisions to the durable async `MDMService`
  persistence path.
- Add production Bytewax flow definitions for entity, quality, duplicate,
  golden-record, publish, and data-agent lifecycle streams.
- Add adapter shims for Codex, Claude Code, opencode, Pi, and later
  APG-compatible AI-agent runtimes.
- Add durable data-agent registration storage with tenant isolation and audit
  replay.
- Add adapter-backed quality scoring and duplicate matching while preserving
  contract-level deny decisions.
- Add metadata catalog and lineage graph adapters.
- Add cache invalidation and lookup acceleration adapters.
- Add security, authorization, and audit-service integration once AUTH, SECU,
  AUDL, and related capabilities are available in generated apps.

## UI Work

- Render the generated data-agent roster in the APG application shell.
- Render the Bytewax lifecycle batch monitor in the APG application shell.
- Add side-by-side steward review panels for human and machine contributions.
- Add explainable rule traces for denied publish, merge, retirement, and agent
  registration actions.

## Verification Work

- Add live persistence tests when running on AC power.
- Add Bytewax topology tests with real stream inputs and replay fixtures.
- Add rendered UI checks after the generated application shell is stable.
- Add performance and concurrency tests for large tenant datasets.
- Run the broader repository suite after the capability sequence reaches a
  stable milestone.

## Open Design Decisions

- Define the APG-wide contract for external AI-agent runtime adapters.
- Decide how data-agent credentials and delegated permissions should be
  provisioned across AUTH, SECU, AUDL, KEYM, and MDM.
- Decide how Bytewax state snapshots should be retained and restored for
  mastered data replay.
- Decide whether golden-record survivorship policies should become reusable
  policy objects shared with other data-governance capabilities.
