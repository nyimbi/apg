# APG SECU Capability Plan

## Slice

Implement durable SECU review evidence on top of first-class security-agent
composition and Bytewax lifecycle guardrails as the next coherent foundation
packet.

SECU already has deterministic security policy, device posture, threat,
assessment, compliance, policy exception, incident response, quarantine, and
audit lifecycle behavior. This slice makes reviewable SECU decisions durable:
AI security agents without required human approval become pending-review
records, compliance gaps and policy exceptions carry policy evidence, and
denied non-Bytewax lifecycle batches are preserved before raising.

## Implementation Steps

1. Extend `capability_contract.py`:
   - supported agent runtimes: `codex`, `claude_code`, `opencode`, `pi`;
   - supported roles for risk review, threat triage, incident response,
     compliance review, and exception review;
   - privileged agent roles and human-approval guardrail;
   - `/secu/agents` route and theme metadata;
   - Bytewax lifecycle stream metadata and batch guardrail.

2. Extend runtime state:
   - add `SecurityAgentRecord`;
   - add durable policy/review evidence to policy exceptions, compliance
     controls, security agents, security lifecycle batches, and audit events;
   - add `SecurityLifecycleBatchRecord`;
   - add tenant-qualified security-agent store;
   - add registration, listing, pending-review, audit event, and dashboard
     summary behavior;
   - add durable Bytewax lifecycle batch validation.

3. Extend generated-application surfaces:
   - add API helpers for agent registration, batch validation, pending reviews,
     and batch evidence;
   - add security-agent roster and pending-review view models;
   - expose agents, streaming, and review evidence metadata through settings
     and dashboard views;
   - include agents, streaming, and review evidence in the package semantic
     model.

4. Extend verification:
   - positive agent registration and Bytewax validation;
   - negative unsupported agent guardrails;
   - privileged agent pending-review evidence;
   - denied lifecycle batch persistence;
   - contract, app, API, view, semantic model, and package evidence checks.

5. Refresh package evidence:
   - `semantic_model.json`;
   - `release_report.json`;
   - `cap_spec.md`;
   - `README.md` and `SPECIFICATION.md`.

6. Review and focused proof:
   - py_compile focused package files;
   - focused pytest package suite;
   - app self-test;
   - inspect, implementation-audit, publish-plan;
   - full SECU Python compile;
   - service smoke;
   - diff whitespace check.

## Non-Goals For This Slice

- Live SIEM, EDR, MDM, SOAR, DLP, GRC, IAM, or AI-provider integrations.
- Production persistence.
- Live Bytewax topology execution.
- Full repository test suite.

Those remain adapter and integration concerns after the executable lifecycle
packet is stable.

## Review Risks

- Agent role/runtime normalization must be predictable for CLI-style names.
- Privileged security-agent roles must become pending review without human
  approval evidence.
- Batch lifecycle routing must reject non-Bytewax streams and preserve denial
  evidence.
- Semantic model and release evidence must stay contract-derived.
