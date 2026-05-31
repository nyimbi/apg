# APG SECU Capability Plan

## Slice

Implement first-class SECU security-agent composition and Bytewax lifecycle
guardrails as the next coherent foundation packet.

SECU already has deterministic security policy, device posture, threat,
assessment, compliance, policy exception, incident response, quarantine, and
audit lifecycle behavior. This slice makes AI security agents governed
participants in those workflows and makes batch security lifecycle intent
explicitly Bytewax-routed.

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
   - add tenant-qualified security-agent store;
   - add registration, listing, audit event, and dashboard summary behavior;
   - add Bytewax lifecycle batch validation.

3. Extend generated-application surfaces:
   - add API helpers for agent registration and batch validation;
   - add security-agent roster view model;
   - expose agents and streaming metadata through settings and dashboard views;
   - include agents and streaming in the package semantic model.

4. Extend verification:
   - positive agent registration and Bytewax validation;
   - negative unsupported/privileged agent guardrails;
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
- Privileged security-agent roles must fail closed without human approval.
- Batch lifecycle routing must reject non-Bytewax streams.
- Semantic model and release evidence must stay contract-derived.
