"""APG Predictive Analytics (PRED) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)

__version__ = "1.0.0"
__capability_id__ = "pred"
__capability_name__ = "Predictive Analytics"
__apg_dependencies__ = ["aicr", "mlcm", "etlp"]

capability_metadata: dict[str, Any] = {
	"name": "pred",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware forecasting, scoring, simulation, and predictive model operations",
	"category": "ai_intelligence",
	"subcategory": "predictive_analytics",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"forecasting",
		"predictive_scoring",
		"scenario_simulation",
		"feature_lineage",
		"prediction_monitoring",
		"model_explainability"
	],
	"composition_patterns": [
		"forecast_pipeline",
		"batch_scoring",
		"real_time_prediction",
		"what_if_simulation"
	],
	"permissions": [
		"pred:view",
		"pred:score",
		"pred:forecast",
		"pred:simulate",
		"pred:manage_models",
		"pred:govern",
		"pred:admin"
	]
}


def register_capability() -> dict[str, Any]:
	"""Register PRED with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "pred",
		"aliases": ["predictive_analytics", "forecasting", "predictive_scoring"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["moni", "audl", "cach", "nlpc"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"forecasting": "Run governed time-series and demand forecasts",
			"predictive_scoring": "Score tenant-scoped entities through approved models",
			"scenario_simulation": "Compare what-if scenarios with audit evidence",
			"feature_lineage": "Track prediction features back to ETLP and metadata sources",
			"capability_rules": "Evaluate deterministic predictive-analytics governance rules",
			"visual_theming": "Apply forecast-console theme tokens and components"
		},
		"endpoints": {
			"forecasts": "/pred/api/v1/forecasts",
			"scores": "/pred/api/v1/scores",
			"models": "/pred/api/v1/models",
			"scenarios": "/pred/api/v1/scenarios",
			"monitoring": "/pred/api/v1/monitoring"
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
	"""Get PRED capability information for composition and marketplace discovery."""
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
