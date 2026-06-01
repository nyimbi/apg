# Anti Money Laundering

`fintech_aml` is the APG package-backed Anti Money Laundering capability. It
provides local, executable financial-crime controls that generated APG
applications can compose with payments, wallets, and KYC.

The capability is deliberately dependency-light. It exposes a stable Python
contract, deterministic rules, runtime service methods, process-local API
helpers, UI view models, theme metadata, Bytewax lifecycle metadata, and release
evidence without requiring live sanctions feeds, banking providers, regulators,
or AI vendors.

Kafka is intentionally not part of this package. AML lifecycle events are
declared for Bytewax through `apg.fintech.aml.lifecycle`.

## What It Provides

- Transaction monitoring for payment and wallet activity.
- AML alert creation and triage.
- Sanctions, high-risk KYC, velocity, structuring, large-transaction, and
  mule-account typology surfaces.
- AML case investigation workflows.
- SAR/STR draft workflows with mandatory human approval.
- Provider-neutral AML agents for Codex, Claude Code, OpenCode, and Pi.
- UI route metadata for dashboards, alert queues, monitoring consoles, cases,
  SAR workflow, typology rules, agents, and settings.
- Theme tokens and component metadata for generated application shells.
- Local tests and release evidence for compiler/package integration.

## Package Shape

```text
capabilities/fintech/aml/
  SPECIFICATION.md
  PLAN.md
  README.md
  cap_spec.md
  capability_contract.py
  models.py
  aml_runtime.py
  service.py
  api.py
  views.py
  app.py
  semantic_model.json
  package_manifest.json
  release_report.json
  tests/test_package_contract.py
```

## Local Usage

Inspect the APG contract:

```bash
./.venv/bin/apg capabilities inspect fintech_aml --json
```

Run the local self-test:

```bash
./.venv/bin/python capabilities/fintech/aml/app.py
```

Run the focused tests:

```bash
./.venv/bin/pytest -q capabilities/fintech/aml/tests/test_package_contract.py
```

Use the service directly:

```python
from capabilities.fintech.aml import AntiMoneyLaunderingService

service = AntiMoneyLaunderingService()
transaction = service.monitor_transaction(
    "txn-1",
    "tenant-a",
    "customer-a",
    "kyc-profile-a",
    120.00,
    "KES",
    "fintech_payments",
    "payment-1",
    22,
)
alert = service.create_alert_from_transaction("alert-1", "tenant-a", transaction["id"])
service.triage_alert(alert["id"], "tenant-a", "escalate", reviewer_id="analyst-a")
case = service.open_case("case-1", "tenant-a", alert["id"], "transaction_monitoring", "investigator-a")
service.draft_sar(
    "sar-1",
    "tenant-a",
    case["id"],
    "customer-a",
    "KE",
    "Suspicious activity narrative with linked evidence.",
    ["txn-1", "alert-1", "case-1"],
    "compliance-manager",
)
```

## Rule Engine

The deterministic rule engine is defined in `capability_contract.py` and is
enforced by `service.py`.

Rules cover:

- tenant context;
- write-policy evidence;
- subject, KYC, amount, currency, and source references;
- high-value, velocity, structuring, sanctions, and high-risk KYC review;
- alert type, severity, evidence, close disposition, and reviewer assignment;
- case alert, case type, and investigator requirements;
- SAR case, subject, jurisdiction, narrative, evidence, and human approval;
- Bytewax routing for AML batches and events;
- supported AI-agent runtimes and roles;
- human approval for privileged AML-agent actions.

Evaluate a rule context:

```bash
./.venv/bin/apg capabilities evaluate-rules fintech_aml \
  --context-json '{"tenant_context_present": false}' \
  --json
```

## UI Composition

The package publishes framework-neutral route and view-model metadata:

- `/fintech-aml/dashboard`
- `/fintech-aml/alerts`
- `/fintech-aml/monitoring`
- `/fintech-aml/cases`
- `/fintech-aml/sar`
- `/fintech-aml/typologies`
- `/fintech-aml/agents`
- `/fintech-aml/settings`

Generated applications can mount these screens in any shell that understands APG
route descriptors and theme tokens.

## AI Agent Composition

AML agents are first-class configuration entries, not hard-coded provider calls.
Supported runtimes are:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles are:

- `aml_ops_reviewer`
- `transaction_monitoring_analyst`
- `sanctions_reviewer`
- `case_investigator`
- `sar_reviewer`

Privileged agent actions require human approval evidence.

## Verification

Focused verification for this package:

```bash
./.venv/bin/python -m py_compile capabilities/fintech/aml/__init__.py capabilities/fintech/aml/capability_contract.py capabilities/fintech/aml/models.py capabilities/fintech/aml/aml_runtime.py capabilities/fintech/aml/service.py capabilities/fintech/aml/api.py capabilities/fintech/aml/views.py capabilities/fintech/aml/app.py capabilities/fintech/aml/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/aml/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/aml/app.py
./.venv/bin/apg capabilities inspect fintech_aml --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/aml --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/aml --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/aml --json
```

## Known Gaps

- Live sanctions/PEP/adverse-media integrations are adapter boundaries.
- Durable Bytewax topology deployment is not part of the local package.
- Regulator filing submission is represented by approved SAR draft evidence.
- Full graph/network analytics and ML model training are future packets.
