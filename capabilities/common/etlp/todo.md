# ETLP Follow-Up Work

This file tracks follow-up work after the executable ETLP capability packet.
The source of truth for current behavior is `SPECIFICATION.md`,
`capability_contract.py`, `service.py`, `view_models.py`, `app.py`, and the
focused package tests.

## Next Runtime Work

- Connect `ETLPLifecycleService` guardrail decisions to durable persistence in
  `ETLPService`.
- Add production Bytewax flow definitions for pipeline, datasource, mapping,
  execution, quality, publish, replay, and pipeline-agent lifecycle streams.
- Add adapter shims for Codex, Claude Code, opencode, Pi, and later
  APG-compatible AI-agent runtimes.
- Add durable pipeline-agent registration storage with tenant isolation and
  audit replay.
- Add adapter-backed connector registry, execution engine, field mapper,
  quality profiler, lineage emitter, and secret-store integrations.
- Add APG MDM, META, MQEB, MONI, AUTH, AUDL, CONF, CACH, and notification
  integration once those generated-app contracts stabilize.

## UI Work

- Render the generated pipeline-agent roster in the APG application shell.
- Render the Bytewax lifecycle batch monitor in the APG application shell.
- Add side-by-side human and machine review panels for datasource, mapping,
  execution, quality, publish, and replay decisions.
- Add explainable rule traces for denied execution, publish, replay, retire,
  datasource, and agent-registration actions.

## Verification Work

- Add live persistence tests when running on AC power.
- Add Bytewax topology tests with real stream inputs and replay fixtures.
- Add rendered UI checks after the generated application shell is stable.
- Add connector, execution-engine, quality-engine, lineage, and secret-store
  integration tests.
- Add performance and concurrency tests for large tenant pipelines.
- Run the broader repository suite after the capability sequence reaches a
  stable milestone.

## Open Design Decisions

- Define the APG-wide contract for external AI-agent runtime adapters.
- Decide how pipeline-agent credentials and delegated permissions should be
  provisioned across AUTH, SECU, AUDL, KEYM, META, and ETLP.
- Decide how Bytewax state snapshots should be retained and restored for
  pipeline replay and backfill.
- Decide whether pipeline quality and mapping policies should become reusable
  governance objects shared with META and MDM.
