"""UI metadata and view models for Predictive Analytics."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import PredService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"prediction_agents": service.list_prediction_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def forecast_console_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"forecasts": service.list_forecasts(tenant_id),
		"pending_review": [item for item in service.list_forecasts(tenant_id) if item["status"] == "pending_review"],
		"models": service.list_models(tenant_id),
		"route": "/pred/forecasts",
	}


def score_monitor_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"scores": service.list_scores(tenant_id),
		"feature_sets": service.list_feature_sets(tenant_id),
		"drift_reports": service.list_drift_reports(tenant_id),
		"route": "/pred/scores",
	}


def feature_registry_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"feature_sets": service.list_feature_sets(tenant_id),
		"pending_review": [item for item in service.list_feature_sets(tenant_id) if item["status"] == "pending_review"],
		"lineage_required": service.describe(tenant_id)["configuration"]["feature_sets"]["lineage_required"],
		"route": "/pred/features",
	}


def scenario_lab_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"scenarios": service.list_scenarios(tenant_id),
		"models": service.list_models(tenant_id),
		"route": "/pred/scenarios",
	}


def model_board_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"models": service.list_models(tenant_id),
		"pending_review": [item for item in service.list_models(tenant_id) if item["status"] == "pending_review"],
		"feature_sets": service.list_feature_sets(tenant_id),
		"route": "/pred/models",
	}


def drift_monitor_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"drift_reports": service.list_drift_reports(tenant_id),
		"pending_review": [item for item in service.list_drift_reports(tenant_id) if item["status"] == "pending_review"],
		"threshold_required": service.describe(tenant_id)["configuration"]["drift"]["threshold_required"],
		"route": "/pred/drift",
	}


def batch_scoring_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	contract = service.describe(tenant_id)
	event_stream = contract["configuration"]["adapters"]["event_stream"]
	return {
		"tenant_id": tenant_id,
		"event_stream": event_stream,
		"streaming": contract["streaming"],
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"scores": service.list_scores(tenant_id),
		"route": "/pred/batch",
	}


def explainability_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"models": [model for model in service.list_models(tenant_id) if model["explainability_attached"]],
		"high_impact_scores": [score for score in service.list_scores(tenant_id) if score["impact"] == "high"],
		"route": "/pred/explainability",
	}


def governance_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"drift_reports": service.list_drift_reports(tenant_id),
		"pending_reviews": {
			"models": [item for item in service.list_models(tenant_id) if item["status"] == "pending_review"],
			"feature_sets": [item for item in service.list_feature_sets(tenant_id) if item["status"] == "pending_review"],
			"forecasts": [item for item in service.list_forecasts(tenant_id) if item["status"] == "pending_review"],
			"drift_reports": [item for item in service.list_drift_reports(tenant_id) if item["status"] == "pending_review"],
			"agents": [item for item in service.list_prediction_agents(tenant_id) if item["status"] == "pending_review"],
		},
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"prediction_agents": service.list_prediction_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"route": "/pred/governance",
	}


def audit_timeline_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/pred/audit",
	}


def prediction_agent_roster_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	contract = service.describe(tenant_id)
	agents = service.list_prediction_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/pred/agents",
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
	}


def lifecycle_batch_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/pred/lifecycle",
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
	}
