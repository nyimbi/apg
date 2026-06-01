# Fraud Detection

`fintech_fraud` is the APG package-backed Fraud Detection capability. It
provides executable fraud prevention, transaction risk decisioning, account
takeover detection, device risk, case investigation, and chargeback evidence
workflows for generated APG fintech applications.

The capability is dependency-light and provider-neutral. It exposes a Python
contract, deterministic rules, runtime service methods, process-local API
helpers, UI view models, theme metadata, Bytewax lifecycle metadata, and release
evidence without requiring live payment rails, device providers, card networks,
model vendors, or AI vendors.

Kafka is intentionally not part of this package. Fraud lifecycle events are
declared for Bytewax through `apg.fintech.fraud.lifecycle`.

## What It Provides

- Fraud signal scoring for payment, wallet, login, device, refund, and
  chargeback events.
- Transaction risk decisioning: approve, review, step-up, hold, and block.
- Account takeover, device anomaly, geography anomaly, velocity, chargeback, and
  AML-link review surfaces.
- Fraud case creation and resolution.
- Chargeback evidence workflow metadata.
- Provider-neutral fraud agents for Codex, Claude Code, OpenCode, and Pi.
- UI route metadata for dashboards, signal queues, decision consoles, cases,
  chargebacks, devices, agents, and settings.
- Theme tokens and component metadata for generated application shells.
- Local tests and release evidence for compiler/package integration.

## Package Shape

```text
capabilities/fintech/fraud/
  SPECIFICATION.md
  PLAN.md
  README.md
  cap_spec.md
  capability_contract.py
  models.py
  fraud_runtime.py
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
./.venv/bin/apg capabilities inspect fintech_fraud --json
```

Run the local self-test:

```bash
./.venv/bin/python capabilities/fintech/fraud/app.py
```

Run the focused tests:

```bash
./.venv/bin/pytest -q capabilities/fintech/fraud/tests/test_package_contract.py
```

Use the service directly:

```python
from capabilities.fintech.fraud import FraudDetectionService

service = FraudDetectionService()
signal = service.score_signal(
    "signal-1",
    "tenant-a",
    "customer-a",
    "kyc-profile-a",
    "payment",
    "mobile",
    "payment-1",
    120.00,
    "KES",
    52,
    review_id="review-1",
)
service.record_decision(
    "decision-1",
    "tenant-a",
    signal["id"],
    "step_up",
    reviewer_id="analyst-a",
    challenge_reference="challenge-1",
)
case = service.open_case(
    "case-1",
    "tenant-a",
    signal["id"],
    "transaction_fraud",
    "investigator-a",
    [signal["id"]],
)
service.resolve_case(case["id"], "tenant-a", "customer_verified", "fraud-manager")
```

## Rule Engine

The deterministic rule engine is defined in `capability_contract.py` and is
enforced by `service.py`.

Rules cover:

- tenant context;
- write-policy evidence;
- signal subject, type, channel, source, and KYC linkage;
- money-bearing event amount and currency;
- risk score range and high-risk review;
- velocity, device, geography, AML, and chargeback indicators;
- supported decisions;
- challenge evidence for step-up;
- reason and human approval for hold/block;
- case signal, type, investigator, evidence, disposition, and reviewer;
- Bytewax routing for fraud batches and events;
- supported AI-agent runtimes and roles;
- human approval for privileged fraud-agent actions.

Evaluate a rule context:

```bash
./.venv/bin/apg capabilities evaluate-rules fintech_fraud \
  --context-json '{"tenant_context_present": false}' \
  --json
```

## UI Composition

The package publishes framework-neutral route and view-model metadata:

- `/fintech-fraud/dashboard`
- `/fintech-fraud/signals`
- `/fintech-fraud/decisions`
- `/fintech-fraud/cases`
- `/fintech-fraud/chargebacks`
- `/fintech-fraud/devices`
- `/fintech-fraud/agents`
- `/fintech-fraud/settings`

Generated applications can mount these screens in any shell that understands APG
route descriptors and theme tokens.

## AI Agent Composition

Fraud agents are first-class configuration entries, not hard-coded provider
calls. Supported runtimes are:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles are:

- `fraud_ops_reviewer`
- `transaction_risk_analyst`
- `chargeback_reviewer`
- `device_risk_reviewer`
- `case_investigator`

Privileged agent actions require human approval evidence.

## Verification

Focused verification for this package:

```bash
./.venv/bin/python -m py_compile capabilities/fintech/fraud/__init__.py capabilities/fintech/fraud/capability_contract.py capabilities/fintech/fraud/models.py capabilities/fintech/fraud/fraud_runtime.py capabilities/fintech/fraud/service.py capabilities/fintech/fraud/api.py capabilities/fintech/fraud/views.py capabilities/fintech/fraud/app.py capabilities/fintech/fraud/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/fraud/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/fraud/app.py
./.venv/bin/apg capabilities inspect fintech_fraud --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/fraud --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/fraud --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/fraud --json
```

## Known Gaps

- Live model inference and training are adapter boundaries.
- Live device fingerprinting and behavioral biometrics are adapter boundaries.
- Live card-network chargeback submission is out of scope for the local package.
- Durable Bytewax topology deployment is not part of this package.
