# MTEN Capability Package Spec

The authoritative package specification is maintained in `SPECIFICATION.md`.

This file is kept as a compatibility pointer for APG package tooling that still
expects `cap_spec.md` beside generated capability artifacts.

Current executable slice:

- tenant-qualified tenant registration, activation, suspension, reactivation,
  capacity review, isolation incident, and live migration governance;
- first-class tenant agent registration for `codex`, `claude_code`,
  `opencode`, and `pi` runtimes;
- privileged tenant-agent approval guardrails;
- Bytewax lifecycle-stream metadata and batch-routing guardrails;
- dependency-light API helpers and view models for generated APG applications.
