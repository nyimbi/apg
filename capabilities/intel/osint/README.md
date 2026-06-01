# APG Open Source Intelligence

`intel_osint` is the APG package-backed capability for governed open-source
intelligence applications. It composes requirements, sources, collection plans,
evidence, triage, assessments, dissemination, reviews, Bytewax lifecycle
metadata, UI/view models, visual theming, and provider-neutral AI-agent
automation.

## What It Provides

- Collection requirement workflow.
- Source registry with source-term review and risk tier controls.
- Collection plans linked to requirements and sources.
- Evidence ledger with fingerprints and confidence scores.
- Triage, assessment, dissemination, and review workflows.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.osint.lifecycle`.

## Use The Service

```python
from capabilities.intel.osint import OpenSourceIntelligenceService

service = OpenSourceIntelligenceService()
requirement = service.register_requirement(
	"req-1",
	"tenant-a",
	"Critical infrastructure monitoring",
	"high",
	"requester-1",
	"confidential",
	"evidence://requirement",
)
source = service.register_source(
	"source-1",
	"tenant-a",
	"news",
	"https://example.com/feed",
	"owner-1",
	"terms-review",
	"medium",
	"evidence://source",
)
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `priority_not_supported`, `terms_review_required`,
`collection_approval_required`, or `bytewax_event_stream_required`.

## Compose In Generated Apps

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Verify Locally

```bash
./.venv/bin/pytest -q capabilities/intel/osint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/osint/app.py
./.venv/bin/apg capabilities inspect intel_osint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/osint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/osint --json
```

## Production Boundaries

Live crawler execution, paid source APIs, social-platform access, search-index
queries, GraphRAG projections, storage, source-term verification, release
distribution, and durable Bytewax topology execution stay behind adapters.
