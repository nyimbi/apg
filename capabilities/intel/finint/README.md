# APG Financial Intelligence

`intel_finint` is the APG package-backed capability for governed
financial-intelligence applications. It composes authorities, financial sources,
subjects, transactions, patterns, risk assessments, referrals, dissemination,
reviews, Bytewax lifecycle metadata, UI/view models, visual theming, and
provider-neutral AI-agent automation.

## What It Provides

- Lawful authority workflow with scope, approver, expiry, classification, and
  evidence.
- Financial source registry with source type, jurisdiction, owner, authority,
  and evidence.
- Subject, transaction, pattern, risk, referral, dissemination, and review
  workflows.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.finint.lifecycle`.

## Use The Service

```python
from capabilities.intel.finint import FinancialIntelligenceService

service = FinancialIntelligenceService()
authority = service.record_authority(
	"auth-1",
	"tenant-a",
	"regulatory_authority",
	"scope://aml-review",
	"confidential",
	"approver-1",
	"2026-12-31",
	"evidence://authority",
)
source = service.register_source(
	"source-1",
	"tenant-a",
	"bank_feed",
	"KE",
	"owner-1",
	authority["id"],
	"evidence://source",
)
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `lawful_authority_required`, `authority_mismatch`,
`amount_invalid`, `funds_movement_scope_denied`, or
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
./.venv/bin/pytest -q capabilities/intel/finint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/finint/app.py
./.venv/bin/apg capabilities inspect intel_finint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/finint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/finint --json
```

## Production Boundaries

Funds movement, payment execution, account freezing, trade placement, crypto
exchange execution, sanctions-screening engines, regulatory report submission,
live bank feeds, case-management writes, storage backends, GraphRAG projections,
dissemination delivery, and durable Bytewax topology execution stay behind
adapters.
