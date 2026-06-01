"""UI metadata helpers for the AI Model Lifecycle Management capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import MlcmService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"models": service.list_models(tenant_id),
		"deployments": service.list_deployments(tenant_id),
		"drift_signals": service.list_drift_signals(tenant_id),
		"model_lifecycle_agents": service.list_model_lifecycle_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def registry_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	return {
		"tenant_id": tenant_id,
		"models": service.list_models(tenant_id),
		"versions": service.list_versions(tenant_id),
		"route": "/mlcm/models",
	}


def model_card_library_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	return {
		"tenant_id": tenant_id,
		"model_cards": [
			{
				"version_id": version["id"],
				"model_id": version["model_id"],
				"stage": version["stage"],
				"complete": bool(version["model_card"]),
				"model_card": version["model_card"],
			}
			for version in service.list_versions(tenant_id)
		],
		"route": "/mlcm/model-cards",
	}


def version_manager_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	return {
		"tenant_id": tenant_id,
		"versions": service.list_versions(tenant_id),
		"evaluations": service.list_evaluations(tenant_id),
		"promotions": service.list_promotion_requests(tenant_id),
		"pending_review": [
			version
			for version in service.list_versions(tenant_id)
			if version["status"] == "pending_review"
		],
		"pending_reviews": [
			version
			for version in service.list_pending_reviews(tenant_id)
			if version.get("version")
		],
		"route": "/mlcm/versions",
	}


def evaluation_console_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	return {
		"tenant_id": tenant_id,
		"minimum_eval_score": service.minimum_eval_score,
		"evaluations": service.list_evaluations(tenant_id),
		"pending_review": [
			evaluation
			for evaluation in service.list_evaluations(tenant_id)
			if evaluation["status"] == "pending_review"
		],
		"pending_reviews": [
			evaluation
			for evaluation in service.list_pending_reviews(tenant_id)
			if "score" in evaluation
		],
		"versions": service.list_versions(tenant_id),
		"route": "/mlcm/evaluation",
	}


def baseline_evidence_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	return {
		"tenant_id": tenant_id,
		"baselines": [
			{
				"version_id": version["id"],
				"baseline_ref": version["baseline_ref"],
				"training_data_ref": version["training_data_ref"],
				"evaluation_id": version["evaluation_id"],
			}
			for version in service.list_versions(tenant_id)
		],
		"evaluations": service.list_evaluations(tenant_id),
		"route": "/mlcm/baselines",
	}


def deployment_board_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	return {
		"tenant_id": tenant_id,
		"targets": service.list_targets(tenant_id),
		"deployments": service.list_deployments(tenant_id),
		"rollbacks": service.list_rollbacks(tenant_id),
		"route": "/mlcm/deployments",
	}


def promotion_board_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	return {
		"tenant_id": tenant_id,
		"promotions": service.list_promotion_requests(tenant_id),
		"versions": service.list_versions(tenant_id),
		"route": "/mlcm/promotion",
	}


def rollback_console_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	return {
		"tenant_id": tenant_id,
		"deployments": service.list_deployments(tenant_id),
		"rollbacks": service.list_rollbacks(tenant_id),
		"retirements": service.list_retirements(tenant_id),
		"route": "/mlcm/rollback",
	}


def drift_monitor_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	signals = service.list_drift_signals(tenant_id)
	return {
		"tenant_id": tenant_id,
		"signals": signals,
		"unresolved": [
			signal
			for signal in signals
			if signal["drift_detected"] and not signal["review_recorded"]
		],
		"route": "/mlcm/drift",
	}


def audit_timeline_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/mlcm/audit",
	}


def model_lifecycle_agent_roster_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_model_lifecycle_agents(tenant_id),
		"pending_reviews": [
			agent
			for agent in service.list_model_lifecycle_agents(tenant_id)
			if agent["status"] == "pending_review"
		],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"required_fields": contract["agents"]["required_fields"],
		"route": "/mlcm/agents",
	}


def lifecycle_batch_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"batches": service.list_lifecycle_batches(tenant_id),
		"denied": [
			batch
			for batch in service.list_lifecycle_batches(tenant_id)
			if batch["status"] == "denied"
		],
		"streaming": contract["streaming"],
		"required_operations": contract["streaming"]["required_operations"],
		"route": "/mlcm/lifecycle",
	}


def governance_model(
	service: MlcmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MlcmService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"versions": service.list_versions(tenant_id),
		"evaluations": service.list_evaluations(tenant_id),
		"promotions": service.list_promotion_requests(tenant_id),
		"retirements": service.list_retirements(tenant_id),
		"model_lifecycle_agents": service.list_model_lifecycle_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/mlcm/governance",
	}
