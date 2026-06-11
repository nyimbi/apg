# Predictive Intelligence

`intel_prediction` is an executable APG capability package for building governed
predictive-intelligence applications. It gives generated APG apps a concrete runtime for
lawful authority, analytical workspaces, scenarios, signals, validated models, forecasts,
projections, early warnings, recommendations, reviews, Bytewax lifecycle checks, UI models,
and provider-neutral AI-agent support.

## What It Provides

**Governance workflows**

- `prediction_authority_workflow`
- `prediction_workspace_workflow`
- `prediction_review_workflow`

**Analytical workflows**

- `prediction_scenario_workflow`
- `prediction_indicator_workflow`
- `prediction_model_workflow`
- `prediction_forecast_workflow`
- `prediction_projection_workflow`
- `prediction_warning_workflow`
- `prediction_recommendation_workflow`
- `prediction_agent_workflow`

## Using The Service

```python
from capabilities.intel.prediction import PredictiveIntelligenceService

service = PredictiveIntelligenceService(tenant_id="tenant-a", actor_id="analyst-1")

# Governance chain
authority = service.record_authority(
    "auth-1", "tenant-a", "mission_order",
    "scope-ref", "confidential",
    "approver-1", "2026-12-31", "authority-evidence",
)
workspace = service.record_workspace(
    "workspace-1", "tenant-a", "threat_prediction",
    "Threat Forecasts", "confidential",
    authority["id"], "workspace-evidence",
)
scenario = service.record_scenario(
    "scen-1", "tenant-a", workspace["id"],
    "geopolitical", "APT-42 expansion ref",
    "short_term", "analyst-1", "evidence-ref",
)

# Model lifecycle
import asyncio

async def main():
    model = await service.create_prediction_model(
        model_type="gradient_boost",
        training_data={"features": ["ttps", "ioc_count"], "sample_count": 500},
        target_variable="attack_probability",
    )
    await service.train_model(model["id"], features=["ttps", "ioc_count"])
    await service.model_deployment(model["id"])

    # Single-model inference
    result = await service.prediction_run(
        model["id"], input_data={"ttps": 0.7, "ioc_count": 42}
    )
    print(result["output_probability"])

    # Ensemble across multiple models
    ensemble = await service.ensemble_predict(
        model_ids=[model["id"]],
        input_data={"ttps": 0.7, "ioc_count": 42},
    )

    # Multi-horizon consensus
    mhf = await service.multi_horizon_forecast(model["id"], {"ttps": 0.7, "ioc_count": 42})
    print(mhf["consensus_probability"])

asyncio.run(main())
```

All write operations evaluate deterministic rules before mutation. Invalid authority, missing
evidence, missing validation, missing approval, unsupported taxonomies, non-Bytewax lifecycle
routing, and unsafe AI-agent scopes raise `PermissionError`.

## Async Method Reference

| Method | Description |
|---|---|
| `create_prediction_model` | Bootstrap a model under the first available scenario |
| `train_model` | Simulate a training run (log-saturation accuracy curve) |
| `prediction_run` | Execute inference; Ollama-backed when `OLLAMA_BASE_URL` is set |
| `model_deployment` | Promote a trained model to production |
| `model_retirement` | Retire a model with audit trail |
| `model_update` | Incremental online learning step |
| `ensemble_predict` | Weighted soft-voting across multiple models with Brier-score decomposition |
| `counterfactual_analysis` | Feature-flip analysis identifying decision-boundary drivers |
| `adversarial_stress_test` | Monte Carlo perturbation robustness scoring |
| `detect_temporal_anomaly` | CUSUM + EWMA streaming anomaly detection on indicator series |
| `multi_horizon_forecast` | Consensus probability across all supported time horizons |
| `check_concept_drift` | PSI-based concept-drift detection; marks models STALE automatically |
| `scenario_analysis` | Batch-run a model over multiple scenario inputs |
| `forecast_event_probability` | Estimate event probability from indicators with horizon decay |
| `threat_trajectory` | Project threat-actor escalation trend from forecast history |
| `threat_actor_profiling` | Structured threat actor profile with risk band and trajectory |
| `early_warning_indicators` | Top indicators for a threat domain sorted by confidence |
| `indicator_correlation_matrix` | Pairwise Pearson correlations across scenario indicators |
| `prediction_accuracy_report` | Accuracy trend report for a specific model |
| `prediction_dashboard` | Consolidated per-model dashboard view |
| `prediction_analytics` | Tenant-level aggregate analytics |
| `projection_risk_matrix` | Projections grouped by risk level and probability band |
| `regulatory_compliance_scorecard` | EU AI Act / NIST AI RMF / ISO 42001 compliance grading |
| `compliance_validation` | Governance documentation completeness check |
| `analytical_assessment` | Coverage assessment over a configurable time window |
| `warning_escalation` | Escalate a warning to TACTICAL / OPERATIONAL / STRATEGIC / NATIONAL |
| `osint_collection_trigger` | Trigger OSINT collection with coverage metadata |
| `intelligence_sharing` | Share forecasts with partner organisations under classification |
| `bulk_scenario_creation` | Bulk-create up to 100 scenarios atomically |
| `horizon_extend` | Extend prediction horizon and recompute decay-adjusted probability |
| `export_forecasts` | Export forecast records to JSON or CSV |
| `health_check` | Service health and operational metrics |

## Generated Application Surfaces

- `app.semantic_model()` returns an APG semantic model for compiler output.
- `app.component_manifest()` returns a publishable component manifest.
- `app.self_test()` verifies the package entrypoint and key invariants.
- `api.py` exposes process-local helpers for generated applications.
- `views.py` exposes dashboard, console, and agent-workbench view models.

## Guardrails

The capability denies unsupported automated decisions, hallucinated forecasts, privacy
bypasses, unapproved model deployment, autonomous warnings, autonomous recommendations, and
privileged agent actions without human approval. AI agents are first-class but bounded:
supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.

## Composability

```apg
use intel_prediction;
use intel_correlation;   // knowledge graph linkage
use intel_alerts;        // warning fan-out
```

Compose with `intel_correlation` to link `PredictionForecast` nodes into the correlation
graph. Compose with `intel_alerts` to route `PredictionWarning` records through the
alerting pipeline.

## Verification

Focused verification covers Python compilation, app self-test, manifest JSON validation,
package tests, APG inspect, APG publish-plan, package implementation audit, lifecycle audit,
global implementation audit, strict package-artifact audit, stale-marker scan, disallowed
messaging scan, and `git diff --check`.
