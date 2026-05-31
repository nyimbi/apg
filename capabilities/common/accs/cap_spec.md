# ACCS Capability Packet

`accs` is the APG Accessibility Services capability. It turns accessibility
governance into executable package behavior that generated applications can
compose with their screens, rules, themes, agents, and release gates.

## Purpose

ACCS provides tenant-scoped accessibility standards, targets, deterministic
audits, findings, remediation tasks, formal reviews, governed temporary
exceptions, AI accessibility-agent registration, Bytewax lifecycle metadata,
view models, and theme metadata.

The package is intentionally dependency-light. Browser scanners,
assistive-technology previews, captioning services, external AI CLIs, ticketing
systems, durable stores, and live Bytewax workers attach through replaceable
adapters after the local package contract is proven.

## Executable Surfaces

- `capability_contract.py` publishes configuration, deterministic rules, UI
  routes, theme tokens, Bytewax stream metadata, and composition surfaces.
- `models.py` defines the local accessibility records.
- `service.py` implements the accessibility lifecycle and enforces contract
  guardrails.
- `api.py` exposes dependency-light helper functions for generated apps.
- `views.py` exposes dashboard, audit, finding, remediation, exception,
  assistive, compliance, agent, audit-trail, analytics, and settings models.
- `app.py`, `semantic_model.json`, and `release_report.json` are generated
  package evidence derived from the executable contract.

## Core Lifecycle

1. Register an accessibility standard.
2. Register tenant-owned routes, screens, content, or media targets.
3. Run deterministic audits against a selected standard.
4. Record findings with evidence, severity, owner, and remediation tasks.
5. Require formal review before closing critical findings.
6. Close findings only with resolution evidence.
7. Record approved, expiring accessibility exceptions with compensating
   controls when unresolved findings are deliberately accepted for a release.
8. Validate publication readiness and distinguish clean readiness from
   `publishable_with_exception` release governance.
9. Register scoped AI accessibility agents using supported runtimes and roles.
10. Compose the ACCS screens, rules, theme, state, and Bytewax lifecycle
    metadata into larger generated APG applications.

## Guardrails

ACCS enforces these rule families:

- tenant context and tenant isolation;
- selected accessibility standard before audit execution;
- remediation ownership for detected violations;
- contrast and media-caption publication rules;
- formal review and resolution evidence before closure;
- exception expiry and compensating-control requirements;
- accessibility-agent registration, supported runtime, supported role, scope,
  and contribution disclosure;
- audit evidence for lifecycle state changes;
- Bytewax stream metadata for batch accessibility mutations.

## AI Agent Composition

AI agents are first-class but provider-neutral. The local contract supports
`codex`, `claude_code`, `opencode`, and `pi` as runtimes and records every
agent with tenant, role, scope, disclosure, and policy metadata. ACCS does not
shell out to those providers directly; adapters own provider invocation.

## Bytewax Composition

The streaming contract declares `bytewax` as the lifecycle processor and
`apg.accs.lifecycle` as the topic. Batch accessibility mutations must carry
Bytewax metadata and fail when a broker-specific or missing stream is used.

## Focused Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/accs/__init__.py capabilities/common/accs/models.py capabilities/common/accs/accessibility_engine.py capabilities/common/accs/service.py capabilities/common/accs/api.py capabilities/common/accs/views.py capabilities/common/accs/capability_contract.py capabilities/common/accs/app.py capabilities/common/accs/test_capability_contract.py capabilities/common/accs/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/accs/test_capability_contract.py capabilities/common/accs/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.accs import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/accs --json
./.venv/bin/apg capabilities publish-plan capabilities/common/accs --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/common/accs --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

## Known Adapter Gaps

This packet does not prove live scanners, live assistive-technology previews,
external AI CLIs, durable persistence, workflow-system integration, rendered UI,
or live Bytewax execution. Those remain adapter and integration layers above
the dependency-light capability package.
