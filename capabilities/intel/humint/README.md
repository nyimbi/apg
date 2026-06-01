# APG Human Intelligence

`intel_humint` is the APG package-backed capability for governed
human-intelligence applications. It composes authorities, human sources,
contact plans, contact reports, debriefings, reliability assessments, leads,
dissemination, reviews, Bytewax lifecycle metadata, UI/view models, visual
theming, and provider-neutral AI-agent automation.

## What It Provides

- Lawful authority workflow with scope, approver, expiry, classification, and
  evidence.
- Human source registry with handling status, risk level, owner, protection
  reference, authority, and evidence.
- Contact planning with objective, safety plan, approval, source-authority
  matching, and evidence.
- Contact reports, debriefings, reliability assessments, leads, dissemination,
  and review workflows.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.humint.lifecycle`.

## Use The Service

```python
from capabilities.intel.humint import HumanIntelligenceService

service = HumanIntelligenceService()
authority = service.record_authority(
	"auth-1",
	"tenant-a",
	"mission_order",
	"scope://mission",
	"secret",
	"approver-1",
	"2026-12-31",
	"evidence://authority",
)
source = service.register_source(
	"source-1",
	"tenant-a",
	"voluntary_source",
	"active",
	"medium",
	"owner-1",
	authority["id"],
	"protection://source-1",
	"evidence://source",
)
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `lawful_authority_required`,
`source_authority_mismatch`, `safety_plan_required`,
`coercive_humint_action_denied`, or `bytewax_event_stream_required`.

## Compose In Generated Apps

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Verify Locally

```bash
./.venv/bin/pytest -q capabilities/intel/humint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/humint/app.py
./.venv/bin/apg capabilities inspect intel_humint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/humint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/humint --json
```

## Production Boundaries

Field operations, source recruitment, coercive operations, covert
communications, payment handling, physical security, identity protection
infrastructure, partner case systems, storage backends, GraphRAG projections,
dissemination delivery, and durable Bytewax topology execution stay behind
adapters.
