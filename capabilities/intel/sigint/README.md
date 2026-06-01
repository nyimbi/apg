# APG Signals Intelligence

`intel_sigint` is the APG package-backed capability for governed
signals-intelligence applications. It composes authorities, sources, collection
tasks, observations, processing batches, patterns, assessments, reviews, Bytewax
lifecycle metadata, UI/view models, visual theming, and provider-neutral
AI-agent automation.

## What It Provides

- Lawful authority workflow with scope, approver, expiry, classification, and
  evidence.
- Signal source registry linked to authority.
- Collection task workflow with retention, minimization, approval, and evidence.
- Observation, processing, pattern, assessment, and review workflows.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.sigint.lifecycle`.

## Use The Service

```python
from capabilities.intel.sigint import SignalsIntelligenceService

service = SignalsIntelligenceService()
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
	"radio",
	"vhf",
	"sensor://vhf-1",
	"owner-1",
	authority["id"],
	"evidence://source",
)
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `lawful_authority_required`,
`source_authority_mismatch`, `minimization_reference_required`, or
`bytewax_event_stream_required`.

## Compose In Generated Apps

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Verify Locally

```bash
./.venv/bin/pytest -q capabilities/intel/sigint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/sigint/app.py
./.venv/bin/apg capabilities inspect intel_sigint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/sigint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/sigint --json
```

## Production Boundaries

Live receivers, lawful-intercept gateways, telecom systems, satellite feeds,
decryptors, speech processing, direction finding, storage backends, search
indexes, GraphRAG projections, dissemination delivery, and durable Bytewax
topology execution stay behind adapters.
