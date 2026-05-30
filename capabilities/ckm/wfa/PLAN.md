# CKM Workflow Automation Packet Plan

## Scope

Build `ckm_wfa` as a coherent lifecycle and guardrail packet for APG
applications that need workflow definitions, active process instances, task
orchestration, approvals, exception handling, AI-agent review, UI metadata,
theme metadata, Bytewax stream governance, and publishable package evidence.

## Implementation Packets

1. Specification and contract
   - Replace stale narrative in `cap_spec.md` with a pointer to the active
     specification.
   - Add `SPECIFICATION.md` for the normative behavior.
   - Expand `capability_contract.py` with configuration, rules, UI routes,
     theme metadata, provides/requires, and Bytewax streaming.

2. Dependency-light lifecycle
   - Add process, instance, task, approval, and WFA-agent data contracts.
   - Implement `WfaLifecycleService` for definition creation, activation,
     instance start, task creation, task completion, approval records,
     exception records, agent registration, batch mutation validation, audit
     events, and dashboard summary.
   - Keep live designers, databases, connectors, schedulers, and stream workers
     behind adapters.

3. Package entrypoint
   - Make `__init__.py` dependency-light and export the contract plus lifecycle
     surfaces.
   - Keep legacy workflow engines importable from their existing files without
     starting them during package import.

4. Documentation and generated evidence
   - Add root package `README.md` with practical usage and composition notes.
   - Refresh semantic model, package manifest, and release evidence from the
     live contract.
   - Update the progress log with proof commands and review notes.

5. Focused proof and review
   - Add a root focused contract/lifecycle test that avoids legacy designer,
     database, connector, and scheduler fixtures.
   - Run compile checks, focused tests, semantic probes, implementation audit,
     publish plan, stale-marker scan, and diff checks.
   - Review tenant isolation, process activation, task assignment, approval
     independence, exception ownership, AI-agent boundaries, Bytewax guardrails,
     import behavior, and generated evidence consistency.

## Out Of Scope

- Browser-rendered process designer.
- Durable database migrations and legacy integration test suite.
- Live connector execution.
- Production scheduler deployment.
- Live Bytewax topology deployment.
- Full repository test suite.

## Review Checklist

- Contract is registry-valid and APG Python route metadata uses practical
  targets.
- Dependency-light package import does not start database, scheduler,
  connector, designer, or stream services.
- Definitions require owner, version, and activation approval.
- Instances require active definitions and initiators.
- Human tasks require assignee or queue ownership.
- SLA-tracked tasks require due-time evidence.
- Task completion requires evidence.
- Approval records enforce reviewer independence and decision reasons.
- Exceptions require ownership.
- AI-agent guardrails include runtime, role, scope, registration, and
  contribution disclosure.
- Batch mutation is rejected unless the event stream is Bytewax.
- Generated semantic evidence matches the executable contract.
