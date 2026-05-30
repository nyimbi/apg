"""APG Logging and Tracing (LOGT) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_LOGT_AGENT_ROLES,
	SUPPORTED_LOGT_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .models import LogtAgent
from .service import LogtService

__version__ = "1.0.0"
__capability_id__ = "logt"
__capability_name__ = "Logging and Tracing"
__apg_dependencies__ = ["moni", "conf", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "logt",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware structured logs, traces, spans, correlation, retention, redaction, and diagnostic search",
	"category": "infrastructure_operations",
	"subcategory": "logging_tracing",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["structured_logging", "distributed_tracing", "trace_correlation", "log_search", "diagnostic_retention", "diagnostic_exports", "logt_agents"],
	"permissions": ["logt:view", "logt:query", "logt:manage_pipelines", "logt:manage_retention", "logt:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register LOGT with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "logt",
		"aliases": ["logging", "tracing", "observability_traces"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "srch", "anom", "comp"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"structured_logging": "Collect tenant-scoped structured logs with redaction and retention controls",
			"distributed_tracing": "Capture traces, spans, service dependencies, and latency diagnostics",
			"trace_correlation": "Correlate logs, metrics, events, workflows, and requests",
			"log_search": "Search and filter diagnostics with RBAC and privacy controls",
			"diagnostic_exports": "Create approved diagnostic bundles for incident, audit, and compliance workflows",
			"logt_agents": "Register scoped AI observability agents for pipeline, log, trace, incident, privacy, and retention work",
			"capability_rules": "Evaluate deterministic logging and tracing rules",
			"event_streaming": "Emit diagnostic lifecycle events through Bytewax",
			"visual_theming": "Apply observability theme tokens and components"
		},
		"endpoints": {"logs": "/logt/api/v1/logs", "traces": "/logt/api/v1/traces", "spans": "/logt/api/v1/spans", "pipelines": "/logt/api/v1/pipelines", "retention": "/logt/api/v1/retention", "agents": "/logt/api/v1/agents", "audit": "/logt/api/v1/audit"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get LOGT capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"LogtAgent",
	"LogtService",
	"SUPPORTED_LOGT_AGENT_ROLES",
	"SUPPORTED_LOGT_AGENT_RUNTIMES",
	"capability_metadata",
	"evaluate_capability_rules",
	"get_capability_contract",
	"get_capability_info",
	"register_capability",
	"streaming_manifest",
	"__apg_dependencies__",
	"__capability_id__",
	"__capability_name__",
	"__version__",
]
