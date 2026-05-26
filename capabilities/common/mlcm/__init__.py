"""APG AI Model Lifecycle Management (MLCM) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)

__version__ = "1.0.0"
__capability_id__ = "mlcm"
__capability_name__ = "AI Model Lifecycle Management"
__apg_dependencies__ = ["aicr", "moni", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "mlcm",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware AI model registry, evaluation, promotion, deployment, and drift governance",
	"category": "ai_infrastructure",
	"subcategory": "model_lifecycle",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"model_registry",
		"model_versioning",
		"model_evaluation",
		"promotion_gates",
		"deployment_board",
		"drift_monitoring",
		"model_governance"
	],
	"composition_patterns": [
		"governed_model_release",
		"continuous_evaluation",
		"drift_response",
		"rollback_orchestration"
	],
	"apis": {
		"rest": "/mlcm/api/v1",
		"websocket": "/mlcm/ws/v1"
	},
	"ui_routes": {
		"main": "/mlcm",
		"models": "/mlcm/models",
		"evaluation": "/mlcm/evaluation",
		"deployments": "/mlcm/deployments",
		"drift": "/mlcm/drift"
	},
	"permissions": [
		"mlcm:view",
		"mlcm:view_models",
		"mlcm:manage_models",
		"mlcm:evaluate",
		"mlcm:deploy",
		"mlcm:view_drift",
		"mlcm:govern",
		"mlcm:admin"
	]
}


def register_capability() -> dict[str, Any]:
	"""Register MLCM with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "mlcm",
		"aliases": ["model_lifecycle", "model_ops", "mlops"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["cach", "conf", "auth"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"model_registry": "Register tenant-scoped AI models, versions, owners, and lifecycle state",
			"promotion_gates": "Promote models through governed dev, staging, and production stages",
			"model_evaluation": "Attach evaluation baselines, scores, and release evidence",
			"drift_monitoring": "Surface drift state for review, rollback, and retraining workflows",
			"capability_rules": "Evaluate deterministic model lifecycle governance rules",
			"visual_theming": "Apply model-ops console theme tokens and components"
		},
		"endpoints": {
			"models": "/mlcm/api/v1/models",
			"versions": "/mlcm/api/v1/versions",
			"evaluations": "/mlcm/api/v1/evaluations",
			"deployments": "/mlcm/api/v1/deployments",
			"drift": "/mlcm/api/v1/drift"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get MLCM capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"capability_metadata",
	"register_capability",
	"get_capability_info",
	"get_capability_contract",
	"evaluate_capability_rules",
	"__version__",
	"__capability_id__",
	"__capability_name__",
	"__apg_dependencies__"
]
