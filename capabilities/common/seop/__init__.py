"""APG Security Operations capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_SEOP_AGENT_ROLES,
	SUPPORTED_SEOP_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .models import SeopAgentRecord
from .service import SeopService

__version__ = "1.0.0"
__capability_id__ = "seop"
__capability_name__ = "Security Operations"
__apg_dependencies__ = ["secu", "anom", "moni", "logt", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "seop",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant security operations, detection pipelines, incident response, threat triage, and governed remediation",
	"category": "security",
	"subcategory": "security_operations",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["detection_pipeline", "incident_response", "threat_triage", "response_playbooks", "security_posture", "seop_agents"],
	"permissions": ["seop:view", "seop:triage", "seop:respond", "seop:manage_playbooks", "seop:admin"],
}


def register_capability() -> dict[str, Any]:
	"""Register SEOP with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "seop",
		"aliases": ["secops", "security-operations", "incident-response"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["ztna", "dlpd", "comp"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"detection_pipeline": "Correlate alerts, anomalies, telemetry, and security events into detections",
			"incident_response": "Manage incident ownership, severity, response state, evidence, and closure",
			"threat_triage": "Prioritize alerts with anomaly context, confidence, and business impact",
			"response_playbooks": "Execute approved containment, isolation, and remediation playbooks",
			"seop_agents": "Compose governed AI agents into detection review, incident command, response review, playbook authoring, posture review, and compliance review lanes",
			"capability_rules": "Evaluate deterministic security-operations rules",
			"visual_theming": "Apply security operations theme tokens and components"
		},
		"endpoints": {"detections": "/seop/api/v1/detections", "incidents": "/seop/api/v1/incidents", "playbooks": "/seop/api/v1/playbooks", "responses": "/seop/api/v1/responses", "posture": "/seop/api/v1/posture", "agents": "/seop/api/v1/agents", "audit": "/seop/api/v1/audit"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": streaming_manifest(),
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get SEOP capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"SUPPORTED_SEOP_AGENT_ROLES",
	"SUPPORTED_SEOP_AGENT_RUNTIMES",
	"SeopAgentRecord",
	"SeopService",
	"capability_metadata",
	"register_capability",
	"get_capability_info",
	"get_capability_contract",
	"evaluate_capability_rules",
	"streaming_manifest",
	"__version__",
	"__capability_id__",
	"__capability_name__",
	"__apg_dependencies__",
]
