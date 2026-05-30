# APG PRED - Predictive Analytics

PRED is the APG capability for governed forecasting, scoring, scenario
simulation, and predictive model operations. It lets generated applications
register predictive models and feature sets, create forecasts, score entities,
compare what-if scenarios, monitor drift, expose UI view models, and publish
audit evidence through deterministic guardrails.

## What It Provides

- Predictive model registration with owner, algorithm, target, environment,
  training history, features, approval state, explainability state, and audit
  evidence.
- Feature-set registration with owner, feature names, ETLP lineage references,
  and source-system metadata.
- Forecast runs with history-size checks, positive horizon checks, long-horizon
  review, confidence-interval metadata, deterministic forecast values, and
  audit events.
- Entity scoring with approved-model checks, feature-lineage checks,
  high-impact explainability checks, deterministic scores, and audit evidence.
- Scenario simulation with baseline, adjustments, assumptions, projected score,
  and delta output.
- Drift reports with metric, threshold, score, review evidence, status, and
  audit events.
- UI view models for dashboard, forecasts, scores, features, scenarios, models,
  drift, batch scoring, explainability, governance, and audit.
- Adapter configuration for AICR, MLCM, ETLP, CONF, AUTH, AUDL, MONI, CACH, and
  Bytewax event streaming.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this packet.
- `PLAN.md` - implementation and review plan.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and
  theme contract.
- `service.py` - `PredService`, the dependency-light generated-app runtime.
- `predictive_runtime.py` - deterministic forecast, score, scenario, and drift
  helpers.
- `views.py` - semantic UI view models for generated applications.
- `app.py` - dynamic package evidence and self-test.
- `test_capability_contract.py` - focused executable contract coverage.
- `tests/test_package_contract.py` - package evidence and compatibility tests.

## Generated-App Usage

```python
from capabilities.common.pred.service import PredService

service = PredService()
model = service.register_model(
	"model-demand",
	"tenant-a",
	"Demand Forecast",
	"analytics",
	"gradient_boosted_tree",
	"daily_demand",
	environment="production",
	approved=True,
	explainability_attached=True,
	training_history_points=48,
	feature_names=["demand", "season", "promotion"],
)
features = service.register_feature_set(
	"features-demand",
	"tenant-a",
	"Demand Features",
	"analytics",
	["demand", "season", "promotion"],
	["etlp://pipelines/demand/features"],
	"etlp",
)
forecast = service.create_forecast(
	"forecast-week",
	"tenant-a",
	model["id"],
	"daily demand",
	[100 + index for index in range(24)],
	7,
)
score = service.score_entity(
	"score-order-1",
	"tenant-a",
	model["id"],
	features["id"],
	"order-1",
	{"demand": 43, "season": 12, "promotion": True},
	environment="production",
	impact="high",
	explanation_ref="explain://score-order-1",
)
```

## Guardrails

PRED blocks missing tenant context, models without owner/algorithm/target,
feature sets without owner/features/source system, forecasts without a model or
series, forecasts with insufficient history or invalid horizon, production
scoring without approved models, scoring without feature lineage, high-impact
scoring without explainability, scoring without entity or feature values,
scenarios without model/assumptions/adjustments/baseline, drift reports without
metric or threshold, non-Bytewax batch scoring streams, cross-tenant scoring,
and prediction state changes without audit evidence. PRED requires review for
short model training history, missing model feature metadata, model approval
without explainability, long forecast horizons, missing feature lineage during
feature registration, and above-threshold drift without review.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/pred/__init__.py capabilities/common/pred/capability_contract.py capabilities/common/pred/models.py capabilities/common/pred/predictive_runtime.py capabilities/common/pred/service.py capabilities/common/pred/views.py capabilities/common/pred/app.py capabilities/common/pred/test_capability_contract.py capabilities/common/pred/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/pred/test_capability_contract.py capabilities/common/pred/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/pred --json
./.venv/bin/apg capabilities publish-plan capabilities/common/pred --json
```
