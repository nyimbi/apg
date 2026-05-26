"""APG Logging and Tracing (LOGT) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "logt"
__capability_name__ = "Logging and Tracing"
__apg_dependencies__ = ["moni", "mqeb", "conf"]

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
	"provides": ["structured_logging", "distributed_tracing", "trace_correlation", "log_search", "diagnostic_retention"],
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
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"structured_logging": "Collect tenant-scoped structured logs with redaction and retention controls",
			"distributed_tracing": "Capture traces, spans, service dependencies, and latency diagnostics",
			"trace_correlation": "Correlate logs, metrics, events, workflows, and requests",
			"log_search": "Search and filter diagnostics with RBAC and privacy controls",
			"capability_rules": "Evaluate deterministic logging and tracing rules",
			"visual_theming": "Apply observability theme tokens and components"
		},
		"endpoints": {"logs": "/logt/api/v1/logs", "traces": "/logt/api/v1/traces", "spans": "/logt/api/v1/spans", "pipelines": "/logt/api/v1/pipelines", "retention": "/logt/api/v1/retention"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get LOGT capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
