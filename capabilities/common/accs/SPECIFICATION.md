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
publication readiness, and compose accessibility screens into larger
applications.

The capability is intentionally provider-light. It must run locally without
external scanners, AI providers, browser automation, identity providers, or
workflow engines. Live scanners and assistive-technology integrations belong
behind adapters added after the deterministic package lifecycle is proven.

## Users And Outcomes

- Product teams can audit application surfaces before release.
- Accessibility leads can triage findings by severity, owner, and status.
- Compliance teams can prove WCAG-oriented governance decisions.
- Generated APG applications can block publication of inaccessible UI.
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
9. Expose dashboards, audit consoles, finding boards, remediation queues,
   assistive previews, and compliance evidence.

## Rules And Guardrails

The contract rules are executable guardrails:

- `tenant_context_required`: write operations require tenant context.
- `audit_requires_standard`: audits require a known standard.
- `violation_requires_remediation_owner`: findings require remediation owners.
- `published_ui_requires_contrast`: published UI must pass contrast checks.
- `media_requires_captions`: media requires captions or transcripts.
- `critical_issue_requires_review`: critical findings require an approved
  formal review before closure.

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
- settings.

The `accs_accessibility_ops` theme must expose semantic tokens and component
metadata for score pills, severity bands, finding boards, compliance evidence,
and semantic-tree previews.

## Adapter Boundaries

These integrations are out of process and must remain replaceable:

- browser or DOM scanners;
- assistive-technology preview providers;
- document/media captioning engines;
- AI remediation recommenders;
- compliance export destinations;
- enterprise workflow or ticketing systems.

Local package tests must not require those systems.

## Acceptance Gates

Focused ACCS proof:

```bash
./.venv/bin/pytest -q capabilities/common/accs/test_capability_contract.py capabilities/common/accs/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/accs --json
./.venv/bin/apg capabilities publish-plan capabilities/common/accs --json
git diff --check -- capabilities/common/accs
```

The package is ready for the next capability when the lifecycle, review
guardrails, API helpers, view models, `cap_spec.md`, and progress log all match
the executable behavior.
