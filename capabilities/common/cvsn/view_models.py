"""Generated-app view models for APG Computer Vision."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .cvsn_runtime import CvsnService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	"""Return route metadata for generated CVSN applications."""
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"recent_assets": service.list_assets(tenant_id)[-10:],
		"recent_jobs": service.list_jobs(tenant_id)[-10:],
		"vision_agents": service.list_vision_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def asset_workbench_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	config = service.describe(tenant_id)["configuration"]["processing"]
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/assets",
		"assets": service.list_assets(tenant_id),
		"allowed_image_types": config["allowed_image_types"],
		"allowed_document_types": config["allowed_document_types"],
		"allowed_video_types": config["allowed_video_types"],
	}


def document_processing_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/documents",
		"document_assets": [asset for asset in service.list_assets(tenant_id) if asset["asset_kind"] == "document"],
		"ocr_jobs": [job for job in service.list_jobs(tenant_id) if job["processing_type"] == "ocr"],
		"ocr_config": service.describe(tenant_id)["configuration"]["ocr"],
	}


def image_analysis_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/images",
		"image_assets": [asset for asset in service.list_assets(tenant_id) if asset["asset_kind"] == "image"],
		"image_jobs": [
			job for job in service.list_jobs(tenant_id)
			if job["processing_type"] in {"object_detection", "image_classification", "barcode_qr"}
		],
		"detection_config": service.describe(tenant_id)["configuration"]["detection"],
	}


def video_analytics_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/video",
		"video_assets": [asset for asset in service.list_assets(tenant_id) if asset["asset_kind"] == "video"],
		"video_jobs": [job for job in service.list_jobs(tenant_id) if job["processing_type"] == "video_analytics"],
		"video_config": service.describe(tenant_id)["configuration"]["video"],
	}


def quality_inspection_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/quality",
		"quality_jobs": [job for job in service.list_jobs(tenant_id) if job["processing_type"] == "quality_inspection"],
		"quality_config": service.describe(tenant_id)["configuration"]["quality"],
	}


def safety_console_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/safety",
		"safety_jobs": [job for job in service.list_jobs(tenant_id) if job["processing_type"] == "factory_safety"],
		"safety_config": service.describe(tenant_id)["configuration"]["safety"],
	}


def similarity_search_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/similarity",
		"search_index_adapter": service.describe(tenant_id)["configuration"]["adapters"]["search_index"],
		"similarity_jobs": [job for job in service.list_jobs(tenant_id) if job["processing_type"] == "visual_similarity"],
	}


def review_console_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	threshold = service.describe(tenant_id)["configuration"]["vision_tasks"]["minimum_confidence_score"]
	jobs = service.list_jobs(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/review",
		"low_confidence_jobs": [job for job in jobs if job["confidence_score"] < threshold],
		"pending_review_jobs": [job for job in jobs if job["status"] == "pending_review"],
	}


def model_registry_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/models",
		"models": service.list_models(tenant_id),
		"pipelines": service.list_pipelines(tenant_id),
		"mlcm_adapter": service.describe(tenant_id)["configuration"]["adapters"]["model_lifecycle"],
	}


def governance_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/governance",
		"configuration": contract["configuration"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"vision_agents": service.list_vision_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"audit_events": service.list_audit_events(tenant_id),
	}


def audit_timeline_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/audit",
		"audit_events": service.list_audit_events(tenant_id),
	}


def vision_agent_roster_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	contract = service.describe(tenant_id)
	agents = service.list_vision_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/agents",
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
	}


def lifecycle_batch_model(service: CvsnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CvsnService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/cvsn/lifecycle",
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
	}
