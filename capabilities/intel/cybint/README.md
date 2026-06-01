# APG Cyber Intelligence

`intel_cybint` is the APG package-backed capability for governed defensive
cyber-intelligence applications. It composes authorities, indicators, sightings,
enrichment, threat profiles, risk assessments, incident links, dissemination,
reviews, Bytewax lifecycle metadata, UI/view models, visual theming, and
provider-neutral AI-agent automation.

## What It Provides

- Lawful authority workflow with scope, approver, expiry, classification, and
  evidence.
- Indicator registry with type, value, TLP, confidence, authority, and evidence.
- Sighting, enrichment, profile, risk, incident-link, dissemination, and review
  workflows.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.cybint.lifecycle`.

## Use The Service

```python
from capabilities.intel.cybint import CyberIntelligenceService

service = CyberIntelligenceService()
authority = service.record_authority(
	"auth-1",
	"tenant-a",
	"defensive_operations_authority",
	"scope://defensive-threat-intel",
	"confidential",
	"approver-1",
	"2026-12-31",
	"evidence://authority",
)
indicator = service.record_indicator(
	"ioc-1",
	"tenant-a",
	"domain",
	"example.invalid",
	"amber",
	0.82,
	authority["id"],
	"evidence://ioc",
)
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `lawful_authority_required`,
`confidence_score_invalid`, `offensive_or_exploit_scope_denied`, or
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
./.venv/bin/pytest -q capabilities/intel/cybint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/cybint/app.py
./.venv/bin/apg capabilities inspect intel_cybint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/cybint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/cybint --json
```

## Production Boundaries

Exploit development, payload generation, intrusion tooling, vulnerability
exploitation, credential collection, command-and-control, live SIEM/EDR/SOAR
integrations, malware sandboxes, vulnerability scanners, ticketing systems,
asset inventories, blocklist deployment, containment execution, storage
backends, GraphRAG projections, dissemination delivery, and durable Bytewax
topology execution stay behind adapters.
