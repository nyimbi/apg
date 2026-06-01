# AUDL Capability Package Spec

The authoritative package specification is maintained in `SPECIFICATION.md`.

This file is kept as a compatibility pointer for APG package tooling that still
expects `cap_spec.md` to exist beside generated capability artifacts.

Current executable slice:

- tenant-scoped audit events with checksum enforcement;
- legal hold, regulated export, purge, and investigation governance;
- first-class audit agent registration for `codex`, `claude_code`,
  `opencode`, and `pi` runtimes;
- Bytewax lifecycle-stream metadata and guardrails for audit batches;
- durable review evidence for regulated exports, purge requests, privileged
  audit agents, denied batches, lifecycle events, and governance events;
- dependency-light API helpers and view models for generated APG apps.
