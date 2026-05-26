"""APG Anomaly Detection (ANOM) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)

__version__ = "1.0.0"
__capability_id__ = "anom"
__capability_name__ = "Anomaly Detection"
__apg_dependencies__ = ["pred", "aicr", "moni"]

capability_metadata: dict[str, Any] = {
	"name": "anom",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware anomaly detection, alert scoring, drift signals, and investigation workflows",
	"category": "ai_intelligence",
	"subcategory": "anomaly_detection",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"metric_anomaly_detection",
		"event_anomaly_detection",
		"behavioral_baselines",
		"alert_scoring",
		"root_cause_hints",
		"investigation_workflows"
	],
	"composition_patterns": [
		"monitoring_anomaly_signal",
		"predictive_thresholding",
		"incident_enrichment",
		"feedback_training_loop"
	],
	"permissions": [
		"anom:view",
		"anom:detect",
		"anom:investigate",
		"anom:tune",
		"anom:manage_rules",
		"anom:admin"
	]
}


def register_capability() -> dict[str, Any]:
	"""Register ANOM with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "anom",
		"aliases": ["anomaly_detection", "outlier_detection", "behavioral_anomaly"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "mqeb", "hlth", "secu"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"metric_anomaly_detection": "Detect anomalies across monitored metrics and forecasts",
			"event_anomaly_detection": "Score unusual events and behavioral sequences",
			"baseline_management": "Maintain tenant-scoped statistical and model baselines",
			"investigation_workflows": "Route high-severity anomalies into governed investigations",
			"capability_rules": "Evaluate deterministic anomaly-governance rules",
			"visual_theming": "Apply anomaly-console theme tokens and components"
		},
		"endpoints": {
			"signals": "/anom/api/v1/signals",
			"baselines": "/anom/api/v1/baselines",
			"detections": "/anom/api/v1/detections",
			"investigations": "/anom/api/v1/investigations",
			"feedback": "/anom/api/v1/feedback"
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
	"""Get ANOM capability information for composition and marketplace discovery."""
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
