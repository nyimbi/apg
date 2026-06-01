# Regulatory Technology

`fintech_regtech` is the APG package-backed Regulatory Technology capability.
It provides executable regulatory source, change, obligation mapping, policy
mapping, impact assessment, filing, submission, inquiry, response, review, and AI-agent workflows for
generated APG fintech applications.

The package is dependency-light and provider-neutral. It exposes a Python
contract, deterministic rules, runtime service methods, process-local API
helpers, UI view models, theme metadata, Bytewax lifecycle metadata, tests, and
release evidence without requiring live regulator portals, paid regulatory
feeds, document signing, external GRC suites, or AI vendors.

## What It Provides

- Regulatory source registration by regulator, jurisdiction, owner, and evidence.
- Regulatory change intake for new rules, rule updates, guidance, enforcement
  actions, consultations, and deadline changes.
- Obligation mapping with policy references, owners, and due dates.
- Impact assessment across affected APG capabilities.
- Filing preparation and submission evidence.
- Regulatory inquiry intake and approved response recording.
- Provider-neutral regulatory agents for Codex, Claude Code, OpenCode, and Pi.
- UI route metadata and theme tokens for generated RegTech consoles.

## Local Usage

Inspect the APG contract:

```bash
./.venv/bin/apg capabilities inspect fintech_regtech --json
```

Run the local self-test:

```bash
./.venv/bin/python capabilities/fintech/regtech/app.py
```

Run focused tests:

```bash
./.venv/bin/pytest -q capabilities/fintech/regtech/tests/test_package_contract.py
```

Use the service directly:

```python
from capabilities.fintech.regtech import RegulatoryTechnologyService

service = RegulatoryTechnologyService()
source = service.register_source("source-1", "tenant-a", "central_bank", "KE", "gazette-1", "owner-a", "source-evidence")
change = service.record_change("change-1", "tenant-a", source["id"], "psd2", "new_rule", "Digital credit rules", "2026-06-01", "high", "evidence-1")
service.map_obligation("mapping-1", "tenant-a", change["id"], "obligation-1", "policy-1", "owner-a", "2026-07-01")
service.assess_impact("impact-1", "tenant-a", change["id"], "fintech_lending", "high", "impact-evidence", "reviewer-a")
```

## Rule Engine

The deterministic rule engine is defined in `capability_contract.py` and
enforced by `service.py`. Rules cover tenant context, write-policy evidence,
source regulator/jurisdiction/reference/owner/evidence, change source/framework/
type/effective date/severity/evidence, obligation and policy mappings, impact
review evidence, filing owner/evidence, submission channel/acknowledgment,
inquiry evidence, approved responses, review evidence, Bytewax batch routing,
supported AI-agent runtimes and roles, and human approval for privileged agent
actions.

## Composition

The capability depends on APG auth, audit, notifications, NLP, keys, compliance,
risk, AML, KYC, and financial reporting contracts. Live regulator portals,
external feed subscriptions, signed documents, and durable Bytewax workers
remain adapter responsibilities.
