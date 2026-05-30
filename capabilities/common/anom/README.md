# APG ANOM - Anomaly Detection

ANOM is the APG capability for governed anomaly detection across monitored
metrics, events, traces, forecast residuals, and security signals. It lets
generated applications register monitoring sources, build statistical
baselines, score observations, route severe anomalies into investigations,
record feedback, tune false positives, expose UI view models, and publish audit
evidence through deterministic guardrails.

## What It Provides

- Monitoring source registration with tenant, name, kind, owner, labels, and
  audit evidence.
- Baseline creation with source linkage, metric, sensitivity, minimum history
  checks, and reset approval governance.
- Observation scoring through deterministic anomaly thresholds and root-cause
  hints.
- Signal records with severity, status, source, baseline, observation, score,
  and tenant isolation.
- Investigation workflows with owner assignment and closure evidence.
- Feedback capture with reviewer, label, false-positive-rate checks, and tuning
  review enforcement.
- UI view models for dashboard, sources, baselines, detector, signals,
  investigations, alerts, rules, feedback, quality, audit, and settings.
- Adapter configuration for PRED, AICR, MONI, WFLO, NTFY, HLTH, CONF, AUTH,
  AUDL, CACH, and Bytewax event streaming.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this packet.
- `PLAN.md` - implementation and review plan.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and
  theme contract.
- `service.py` - `AnomService`, the dependency-light generated-app runtime.
- `anomaly_engine.py` - deterministic baseline, score, summary, and feedback
  helpers.
- `views.py` - semantic UI view models for generated applications.
- `app.py` - dynamic package evidence and self-test.
- `test_capability_contract.py` - focused executable contract coverage.
- `tests/test_package_contract.py` - package evidence and compatibility tests.

## Generated-App Usage

```python
from capabilities.common.anom.service import AnomService

service = AnomService()
source = service.register_source(
	"api_latency",
	"tenant-a",
	"API Latency",
	kind="metric",
	owner="platform",
)
baseline = service.create_baseline(
	"api_latency_baseline",
	"tenant-a",
	source["id"],
	"p95_latency_ms",
	[100.0 + (index % 5) for index in range(60)],
)
signal = service.detect(
	"signal-1",
	"tenant-a",
	source["id"],
	baseline["id"],
	"p95_latency_ms",
	180.0,
	context={"deployment": "checkout-v2", "region": "ke"},
	owner="sre-lead",
)
closed = service.close_investigation(
	f"investigate:{signal['id']}",
	"rollback checkout-v2",
	"tenant-a",
	closed_by="sre-lead",
	resolution_evidence=["incident:123", "deployment rollback completed"],
)
```

## Guardrails

ANOM blocks missing tenant context, sources without name/owner/kind, baselines
without source/metric/history/sensitivity, detections without source, baseline,
metric, or value, critical anomalies without an owner, cross-tenant detection,
investigations without signal or owner, closures without resolution, closer, or
evidence, feedback without signal/reviewer/label, baseline reset without
approval, non-Bytewax batch detection streams, alert dispatch without
notification adapter, and state changes without audit evidence. ANOM requires
review for unknown source kinds, unknown sensitivity values, high-severity
triage, unknown feedback labels, and high false-positive rates.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/anom/__init__.py capabilities/common/anom/capability_contract.py capabilities/common/anom/models.py capabilities/common/anom/anomaly_engine.py capabilities/common/anom/service.py capabilities/common/anom/api.py capabilities/common/anom/views.py capabilities/common/anom/app.py capabilities/common/anom/test_capability_contract.py capabilities/common/anom/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/anom/test_capability_contract.py capabilities/common/anom/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/anom --json
./.venv/bin/apg capabilities publish-plan capabilities/common/anom --json
```
