"""APG Federated Learning (FEDL) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)

__version__ = "1.0.0"
__capability_id__ = "fedl"
__capability_name__ = "Federated Learning"
__apg_dependencies__ = ["aicr", "mlcm", "encr", "mten"]

capability_metadata: dict[str, Any] = {
	"name": "fedl",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Privacy-preserving federated learning orchestration with tenant-aware participants, secure aggregation, and model governance",
	"category": "ai_infrastructure",
	"subcategory": "federated_learning",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"federation_coordination",
		"participant_attestation",
		"secure_aggregation",
		"privacy_budgeting",
		"training_round_monitoring",
		"poisoning_defense",
		"federated_model_governance"
	],
	"composition_patterns": [
		"privacy_preserving_training",
		"cross_tenant_consent",
		"secure_model_aggregation",
		"federated_release_to_mlcm"
	],
	"apis": {
		"rest": "/fedl/api/v1",
		"websocket": "/fedl/ws/v1"
	},
	"ui_routes": {
		"main": "/fedl",
		"federations": "/fedl/federations",
		"participants": "/fedl/participants",
		"rounds": "/fedl/rounds",
		"privacy": "/fedl/privacy"
	},
	"permissions": [
		"fedl:view",
		"fedl:manage_federations",
		"fedl:view_participants",
		"fedl:run_rounds",
		"fedl:manage_privacy",
		"fedl:manage_security",
		"fedl:view_models",
		"fedl:admin"
	]
}


def register_capability() -> dict[str, Any]:
	"""Register FEDL with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "fedl",
		"aliases": ["federated_learning", "privacy_preserving_ai", "federated_training"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["moni", "audl", "keym"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"federation_coordination": "Create and run tenant-aware federated learning groups",
			"participant_attestation": "Validate participant identity, contracts, and runtime posture",
			"secure_aggregation": "Require privacy-preserving aggregation for model updates",
			"privacy_budgeting": "Track and govern differential privacy budget thresholds",
			"poisoning_defense": "Block aggregation when poisoning signals are detected",
			"capability_rules": "Evaluate deterministic federated learning governance rules",
			"visual_theming": "Apply privacy-mesh theme tokens and components"
		},
		"endpoints": {
			"federations": "/fedl/api/v1/federations",
			"participants": "/fedl/api/v1/participants",
			"rounds": "/fedl/api/v1/rounds",
			"aggregation": "/fedl/api/v1/aggregation",
			"privacy": "/fedl/api/v1/privacy"
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
	"""Get FEDL capability information for composition and marketplace discovery."""
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
