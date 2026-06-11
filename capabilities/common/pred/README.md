# APG PRED - Predictive Analytics

PRED is the APG capability for governed forecasting, scoring, scenario
simulation, and predictive model operations. It lets generated applications
register predictive models and feature sets, create forecasts, score entities,
compare what-if scenarios, monitor drift, expose UI view models, and publish
audit evidence through deterministic guardrails.

## What It Provides

- Predictive model registration with owner, algorithm, target, environment,
  training history, features, approval state, explainability state, and audit
  evidence, including pending-review state for incomplete governance evidence.
- Feature-set registration with owner, feature names, ETLP lineage references,
  source-system metadata, and pending-review evidence when lineage is missing.
- Forecast runs with history-size checks, positive horizon checks, long-horizon
  review, confidence-interval metadata, deterministic forecast values, and
  audit events.
- Entity scoring with approved-model checks, feature-lineage checks,
  high-impact explainability checks, deterministic scores, and audit evidence.
- Scenario simulation with baseline, adjustments, assumptions, projected score,
  and delta output.
- Drift reports with metric, threshold, score, review evidence, status, and
  audit events, including pending-review state for unreviewed above-threshold
  drift.
- First-class AI prediction-agent composition for `codex`, `claude_code`,
  `opencode`, and `pi`, with role, scope, owner, purpose, contribution
  disclosure, and privileged-role review guardrails.
- Bytewax lifecycle batch validation for model, feature-set, forecast, score,
  scenario, drift, explainability, and prediction-agent mutations.
- UI view models for dashboard, forecasts, scores, features, scenarios, models,
  drift, batch scoring, explainability, agents, lifecycle batches, governance,
  and audit.
- Adapter configuration for AICR, MLCM, ETLP, CONF, AUTH, AUDL, MONI, CACH, and
  Bytewax event streaming.

## New in This Release

### Platt Score Calibration
`calibrate_scores()` fits a logistic sigmoid to raw scores using gradient
descent, converting hash-based deterministic scores to calibrated probabilities.
Parameters are stored per `(tenant_id, model_id)` and used in downstream
explanation and financial impact methods.

### Decimal-Precision Monetary Outcomes
`attach_monetary_outcome()` and `aggregate_monetary_impact()` track financial
consequences of scoring decisions using Python's `decimal.Decimal` with
`ROUND_HALF_EVEN` (banker's rounding). Floats are explicitly excluded from the
accumulation path — required for IFRS 13 compliance.

### Champion-Challenger A/B Routing
`register_champion_challenger()` stores a routing policy with configurable
traffic split (1–49%). `route_score_request()` uses deterministic SHA-256
entity hashing so the same entity always hits the same model arm — preventing
decision inconsistency during rollout.

### Temporal Confidence Decay
`compute_confidence_decay()` applies exponential decay (`exp(-lambda * age)`)
with a configurable half-life (default 90 days) to surface models that need
retraining before measured drift occurs. Integrated into `dashboard_summary`
recommendations.

### PSI + KL-Divergence Drift
`stream_drift_window()` computes Population Stability Index and KL-divergence
between reference and current score distributions using equal-width binning.
PSI bands: < 0.1 stable, 0.1–0.2 warning, > 0.2 critical (Basel III standard).

### Explanation Attestation Registry
`register_explanation_attestation()` creates a non-repudiable SHA-256 hash
over `(score_id, model_version_id, method, attested_by)`. High-impact scores
require ≥ 80% feature coverage. `verify_explanation_attestation()` recomputes
and compares hashes to detect tampering.

### Prediction Latency SLA Monitoring
`record_prediction_latency()` stores per-score latency with breach flags.
`compute_sla_report()` derives P50/P95/P99 percentiles and breach rate using
nearest-rank interpolation over the full latency distribution.

### Governance Lineage Graph
`build_lineage_graph()` constructs a traversable adjacency DAG linking scores
→ models → feature sets → ETL lineage refs. `trace_decision_lineage()` runs
BFS from any `score_id` back to root nodes for regulatory audit exhibits.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this packet.
- `PLAN.md` - implementation and review plan.
- `WORLD_CLASS_IMPROVEMENTS.md` - 15 world-class improvement proposals.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and
  theme contract.
- `service.py` - `PredService`, the dependency-light generated-app runtime.
- `predictive_runtime.py` - deterministic forecast, score, scenario, and drift
  helpers.
- `views.py` - semantic UI view models for generated applications.
- `app.py` - dynamic package evidence and self-test.
- `test_capability_contract.py` - focused executable contract coverage.
- `tests/test_package_contract.py` - package evidence and compatibility tests.
- `docs/user_guide.md` - comprehensive operator and developer guide.

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

# Calibrate scores to true probabilities
import asyncio
calib = asyncio.run(service.calibrate_scores(
	"tenant-a", model["id"],
	[{"predicted": score["score"], "actual": 1.0}],
))

# Attach Decimal monetary outcome
import asyncio
outcome = asyncio.run(service.attach_monetary_outcome(
	"tenant-a", score["id"], "125000.00", currency="KES",
))

# Check model confidence decay
decay = asyncio.run(service.compute_confidence_decay("tenant-a", model["id"]))

# Champion-challenger routing
asyncio.run(service.register_champion_challenger(
	"tenant-a", "policy-001", model["id"], model["id"], traffic_split_pct=10,
))
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
feature registration, and above-threshold drift without review; those outcomes
are persisted as `pending_review` records with matched rule and review-reason
evidence for generated model, forecast, drift, and governance screens. AI
prediction-agent guardrails also block unsupported runtimes, unsupported roles,
missing scope, missing owner, missing purpose, missing machine-contribution
disclosure, and route privileged roles through pending human review when
approval evidence is absent. Lifecycle mutation batches are accepted only
through the declared Bytewax processor contract.

Monetary outcome methods reject float arguments — amounts must be passed as
strings. Champion-challenger routing rejects traffic splits outside 1–49%.
Explanation attestations on high-impact scores require ≥ 80% feature coverage.

## AI Agent Composition

PRED treats predictive AI agents as first-class APG citizens. Generated
applications can compose agents from multiple rapidly changing tool runtimes
without binding forecasting, scoring, or governance logic to a single provider.
The current executable contract supports `codex`, `claude_code`, `opencode`,
and `pi`; roles include forecast review, score review, feature-lineage review,
scenario review, drift review, model-release review, explainability review,
batch-scoring review, and prediction stewardship.

The runtime stores provider-neutral agent metadata only. Live CLI/API
invocation, credential management, and remote agent orchestration belong behind
the AICR adapter boundary.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/pred/__init__.py capabilities/common/pred/capability_contract.py capabilities/common/pred/models.py capabilities/common/pred/predictive_runtime.py capabilities/common/pred/service.py capabilities/common/pred/views.py capabilities/common/pred/app.py capabilities/common/pred/test_capability_contract.py capabilities/common/pred/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/pred/test_capability_contract.py capabilities/common/pred/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/pred --json
./.venv/bin/apg capabilities publish-plan capabilities/common/pred --json
```
