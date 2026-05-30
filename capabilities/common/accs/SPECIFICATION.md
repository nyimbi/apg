# ACCS Capability Specification

## Identity

- Capability ID: `accs`
- Display name: Accessibility Services
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `accs_accessibility_ops`

## Purpose

ACCS makes accessibility a first-class, executable APG platform concern. It
lets generated applications register accessibility standards, register UI,
content, and media targets, run deterministic audits, record findings, assign
remediation work, require formal review for critical issues, validate
publication readiness, register scoped AI accessibility agents, emit Bytewax
lifecycle metadata, and compose accessibility screens into larger applications.

The capability is intentionally provider-light. It must run locally without
external scanners, AI providers, browser automation, identity providers,
workflow engines, or stream processors. Live scanners, assistive-technology
integrations, external AI CLIs, and Bytewax workers belong behind adapters
added after the deterministic package lifecycle is proven.

## Users And Outcomes

- Product teams can audit application surfaces before release.
- Accessibility leads can triage findings by severity, owner, and status.
- Compliance teams can prove WCAG-oriented governance decisions.
- Generated APG applications can block publication of inaccessible UI.
- AI agents can participate in accessibility work only through explicit
  runtime, role, scope, disclosure, and audit registration.
- The theming system can render accessibility score, finding, remediation, and
  review surfaces consistently.

## Domain Model

The package owns these records:

- `AccessibilityStandard`: tenant-scoped WCAG, EN 301 549, or local policy
  profile.
- `AccessibilityTarget`: UI, route, media, or content surface that can be
  audited.
- `AccessibilityAudit`: deterministic audit run over one or more targets.
- `AccessibilityFinding`: audit or manual finding with severity, rule,
  evidence, owner, status, and review state.
- `RemediationTask`: tracked work item for resolving a finding.
- `AccessibilityReview`: formal decision record for critical findings.
- `AccessibilityAuditEvent`: tenant-scoped event emitted for important state
  changes.
- `AccessibilityAgent`: tenant-scoped AI-agent registration for audit review,
  remediation planning, caption review, standards advice, or release review.

## Lifecycle

The primary lifecycle is:

1. Register an accessibility standard.
2. Register one or more tenant-owned targets.
3. Run an audit against a selected standard.
4. Record findings with deterministic evidence and remediation owners.
5. Create remediation tasks automatically.
6. Require an approved formal review for critical findings before closure.
7. Close findings only with resolution evidence.
8. Validate publication readiness with the same rule engine exposed by the
   public contract.
9. Register scoped AI accessibility agents when agent assistance contributes to
   audit, remediation, content, standards, or release-review work.
10. Validate Bytewax-backed batch accessibility mutation metadata.
11. Expose dashboards, audit consoles, finding boards, remediation queues,
   assistive previews, agent panels, audit trails, analytics, settings, and
   compliance evidence.

## Rules And Guardrails

The contract rules are executable guardrails:

- `tenant_context_required`: write operations require tenant context.
- `audit_requires_standard`: audits require a known standard.
- `violation_requires_remediation_owner`: findings require remediation owners.
- `published_ui_requires_contrast`: published UI must pass contrast checks.
- `media_requires_captions`: media requires captions or transcripts.
- `critical_issue_requires_review`: critical findings require an approved
  formal review before closure.
- `finding_closure_requires_resolution`: finding closure requires resolution
  evidence.
- `accessibility_agent_requires_registration`: AI accessibility agents must be
  registered before contributing.
- `accessibility_agent_runtime_supported`: AI accessibility agents must use a
  supported runtime.
- `accessibility_agent_role_supported`: AI accessibility agents must use a
  supported role.
- `accessibility_agent_requires_scope`: AI accessibility agents require
  explicit scope.
- `accessibility_agent_requires_disclosure`: AI accessibility-agent
  contributions require disclosure.
- `accs_state_change_requires_audit`: lifecycle state changes require audit
  evidence.
- `cross_tenant_accessibility_access_denied`: tenant records cannot cross
  boundaries.
- `batch_accessibility_mutation_requires_bytewax`: batch accessibility
  mutations must use Bytewax streams.

Service methods must enforce these rules, not only publish them as metadata.

## UI And Theme

ACCS exposes route and view-model surfaces for:

- dashboard summary;
- audit console;
- findings board;
- remediation queue;
- assistive preview;
- media accessibility;
- compliance evidence;
- accessibility agent panel;
- audit trail;
- analytics;
- settings.

The `accs_accessibility_ops` theme must expose semantic tokens and component
metadata for score pills, severity bands, finding boards, compliance evidence,
semantic-tree previews, agent panels, and audit timelines.

## Adapter Boundaries

These integrations are out of process and must remain replaceable:

- browser or DOM scanners;
- assistive-technology preview providers;
- document/media captioning engines;
- AI remediation recommenders;
- AI coding/agent CLIs such as Codex, Claude Code, OpenCode, and Pi;
- Bytewax workers;
- compliance export destinations;
- enterprise workflow or ticketing systems.

Local package tests must not require those systems.

## Acceptance Gates

Focused ACCS proof:

```bash
./.venv/bin/python -m py_compile capabilities/common/accs/__init__.py capabilities/common/accs/models.py capabilities/common/accs/accessibility_engine.py capabilities/common/accs/service.py capabilities/common/accs/api.py capabilities/common/accs/views.py capabilities/common/accs/capability_contract.py capabilities/common/accs/app.py capabilities/common/accs/test_capability_contract.py capabilities/common/accs/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/accs/test_capability_contract.py capabilities/common/accs/tests
./.venv/bin/python -c "from capabilities.common.accs import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/accs --json
./.venv/bin/apg capabilities publish-plan capabilities/common/accs --json
git diff --check -- capabilities/common/accs
```

The package is ready for the next capability when the lifecycle, review
guardrails, API helpers, view models, `cap_spec.md`, and progress log all match
the executable behavior.
