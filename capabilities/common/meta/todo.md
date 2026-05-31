# META Follow-Up Work

This file tracks follow-up work after the executable META capability packet.
The source of truth for current behavior is `SPECIFICATION.md`,
`capability_contract.py`, `service.py`, `view_models.py`, `app.py`, and the
focused package tests.

## Next Runtime Work

- Connect `MetaService` guardrail decisions to durable persistence in
  `APGMetadataService`.
- Add production Bytewax flow definitions for asset, discovery,
  classification, lineage, quality, certification, glossary, and catalog-agent
  lifecycle streams.
- Add adapter shims for Codex, Claude Code, opencode, Pi, and later
  APG-compatible AI-agent runtimes.
- Add durable catalog-agent registration storage with tenant isolation and
  audit replay.
- Add adapter-backed discovery, classification, lineage, quality, and search
  engines while preserving contract-level deny decisions.
- Add metadata-store, graph-store, and search-index adapters.
- Add APG MDM, ETLP, CONN, AUDL, AUTH, MONI, and notification integration once
  those generated-app contracts stabilize.

## UI Work

- Render the generated catalog-agent roster in the APG application shell.
- Render the Bytewax lifecycle batch monitor in the APG application shell.
- Add side-by-side steward review panels for human and machine contributions.
- Add explainable rule traces for denied classification, certification,
  publish, retirement, and agent-registration actions.

## Verification Work

- Add live persistence tests when running on AC power.
- Add Bytewax topology tests with real stream inputs and replay fixtures.
- Add rendered UI checks after the generated application shell is stable.
- Add connector, search-index, graph-store, and classification-engine
  integration tests.
- Add performance and concurrency tests for large tenant catalogs.
- Run the broader repository suite after the capability sequence reaches a
  stable milestone.

## Open Design Decisions

- Define the APG-wide contract for external AI-agent runtime adapters.
- Decide how catalog-agent credentials and delegated permissions should be
  provisioned across AUTH, SECU, AUDL, KEYM, MDM, and META.
- Decide how Bytewax state snapshots should be retained and restored for
  metadata catalog replay.
- Decide whether glossary and classification policies should become reusable
  governance objects shared with MDM and ETLP.
