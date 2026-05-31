"""APG Anomaly Detection (ANOM) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from .service import AnomService

__version__ = "1.0.0"
__capability_id__ = "anom"
__capability_name__ = "Anomaly Detection"
__apg_dependencies__ = ["pred", "aicr", "moni", "conf"]

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
		"investigation_workflows",
		"feedback_tuning",
		"alert_dispatch",
		"anomaly_agent_composition",
		"lifecycle_batch_governance",
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
		"anom:audit",
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
		"optional_dependencies": ["auth", "audl", "wflo", "ntfy", "hlth", "secu", "cach"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"adapters": contract["configuration"]["adapters"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"capabilities": {
			"metric_anomaly_detection": "Detect anomalies across monitored metrics and forecasts",
			"event_anomaly_detection": "Score unusual events and behavioral sequences",
			"source_registry": "Register tenant-scoped metric, event, trace, forecast, and security sources",
			"baseline_management": "Maintain tenant-scoped statistical and model baselines",
			"investigation_workflows": "Route high-severity anomalies into governed investigations",
			"investigation_closure_governance": "Require tenant-safe closure evidence for anomaly investigations",
			"feedback_tuning": "Track false positives and require tuning review when quality degrades",
			"alert_dispatch": "Expose notification-adapter metadata for severe anomaly dispatch",
			"anomaly_agent_composition": "Register provider-neutral AI anomaly agents with runtime, role, scope, owner, purpose, disclosure, and human-review guardrails",
			"lifecycle_batch_governance": "Validate ANOM lifecycle mutations through Bytewax-only batch contracts",
			"capability_rules": "Evaluate deterministic anomaly-governance rules",
			"visual_theming": "Apply anomaly-console theme tokens and components"
		},
		"endpoints": {
			"sources": "/anom/api/v1/sources",
			"signals": "/anom/api/v1/signals",
			"baselines": "/anom/api/v1/baselines",
			"detections": "/anom/api/v1/detections",
			"investigations": "/anom/api/v1/investigations",
			"alerts": "/anom/api/v1/alerts",
			"feedback": "/anom/api/v1/feedback",
			"quality": "/anom/api/v1/quality",
			"agents": "/anom/api/v1/agents",
			"lifecycle": "/anom/api/v1/lifecycle",
			"audit_events": "/anom/api/v1/audit-events"
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
	"AnomService",
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
