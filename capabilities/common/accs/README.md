# ACCS - Accessibility Services

ACCS makes accessibility governance an executable APG capability. It gives
generated applications a tenant-scoped way to register accessibility standards,
register UI/content/media targets, run deterministic audits, record findings,
assign remediation work, require formal review for critical findings, register
AI accessibility agents, validate publication readiness, and expose operational
views for accessibility teams.

The package is dependency-light by design. It does not call browser scanners,
assistive-technology providers, captioning engines, external AI tools, or
workflow systems directly. Those systems attach through adapters after the
local lifecycle, rules, and generated application contract are proven.

## What ACCS Provides

- Accessibility standard registry for WCAG-oriented and tenant-defined
  profiles.
- Accessibility target registry for routes, screens, content surfaces, and
  media assets.
- Deterministic audit runner for contrast, semantic labels, keyboard
  navigation, and media caption/transcript guardrails.
- Finding and remediation lifecycle with owner assignment, due dates, review
  state, closure evidence, and audit events.
- Critical-finding review workflow that blocks closure until an approved
  formal review exists.
- Publication validation that applies the same deterministic rules used by
  generated packages.
- First-class AI accessibility-agent registration for Codex, Claude Code,
  OpenCode, Pi, and future runtimes.
- Bytewax lifecycle stream metadata for batch accessibility mutation and
  package composition.
- UI/view-model surfaces for dashboards, audit console, findings board,
  remediation queue, review queue, agent panel, audit trail, analytics,
  assistive preview, compliance evidence, and settings.
- Visual theme tokens for compact accessibility operations screens.

## Core Lifecycle

1. Register or reuse an accessibility standard.
2. Register tenant-owned targets for routes, media, or content.
3. Run an audit with a selected standard and remediation owner.
4. Record findings and remediation tasks with deterministic evidence.
5. Route critical findings to formal review.
6. Close findings only when approved review and resolution evidence are
   present.
7. Validate publication readiness before release.
8. Register scoped accessibility agents when AI assistance contributes to
   audit, remediation, standards, caption, or release-review work.
9. Compose ACCS screens, theme, rules, and Bytewax stream metadata into the
   generated application.

## Runtime Example

```python
from capabilities.common.accs.service import AccsService

service = AccsService()
service.register_target(
    target_id="checkout",
    tenant_id="tenant-a",
    surface="Checkout",
    route="/checkout",
    owner="product-owner",
    published_ui=True,
    contrast_ratio=3.2,
    media_content_present=True,
    captions_available=False,
)

audit = service.run_audit(
    audit_id="checkout-audit",
    tenant_id="tenant-a",
    standard_id="wcag_2_2_aa",
    target_ids=["checkout"],
    remediation_owner="accessibility-lead",
)

finding_id = audit["finding_ids"][0]
service.record_review(
    finding_id=finding_id,
    tenant_id="tenant-a",
    reviewer="accessibility-reviewer",
    decision="approved",
    notes="Remediation evidence accepted.",
)
service.close_finding(
    finding_id=finding_id,
    tenant_id="tenant-a",
    resolution="Contrast and keyboard evidence verified.",
)
```

## Agent Example

```python
service.register_accessibility_agent(
    agent_id="release-review-agent",
    tenant_id="tenant-a",
    name="Release Accessibility Reviewer",
    runtime="codex",
    role="release_reviewer",
    scope="release accessibility gates and critical finding evidence",
    contribution_disclosed=True,
    policy_ref="accs-agent-policy",
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles are `audit_reviewer`, `remediation_planner`, `caption_reviewer`,
`standards_advisor`, and `release_reviewer`.

## Composition Contract

`get_capability_contract()` returns the executable APG contract:

- `configuration`: tenant, standards, audit, assistive, agent, governance,
  observability, adapter, UI, and theme settings.
- `rule_engine`: deterministic guardrails for tenant context, audit standards,
  remediation ownership, contrast, captions, critical review, finding closure,
  agent registration, tenant isolation, audit evidence, and Bytewax batch
  mutation.
- `ui`: APG Python route metadata and view-model module.
- `theme`: accessibility operations tokens and component metadata.
- `streaming`: Bytewax processor, topic, state collections, lifecycle events,
  and batch mutation guardrail.

## Verification

Focused checks for this package:

```bash
./.venv/bin/python -m py_compile capabilities/common/accs/__init__.py capabilities/common/accs/models.py capabilities/common/accs/accessibility_engine.py capabilities/common/accs/service.py capabilities/common/accs/api.py capabilities/common/accs/views.py capabilities/common/accs/capability_contract.py capabilities/common/accs/app.py capabilities/common/accs/test_capability_contract.py capabilities/common/accs/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/accs/test_capability_contract.py capabilities/common/accs/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.accs import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/accs --json
./.venv/bin/apg capabilities publish-plan capabilities/common/accs --json
```

Full platform suites, live browser scanning, assistive-technology providers,
captioning services, workflow systems, external AI CLIs, durable databases, and
live Bytewax execution are intentionally separate integration checks.
