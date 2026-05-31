"""APG API Gateway and Management capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract


CAPABILITY_METADATA: dict[str, Any] = {
	"name": "apig",
	"display_name": "API Gateway & Management",
	"version": "1.0.0",
	"category": "infrastructure",
	"subcategory": "api_gateway",
	"description": "Governed API gateway control plane for APG route, traffic, security, edge, deployment, and audit workflows.",
	"author": "APG Platform Team",
	"created_at": datetime.now(timezone.utc),
	"updated_at": datetime.now(timezone.utc),
	"dependencies": [
		{"capability": "auth_rbac", "version": ">=1.0.0", "required": True, "purpose": "Authentication, authorization, and RBAC"},
		{"capability": "moni", "version": ">=1.0.0", "required": True, "purpose": "Metrics, traces, logs, and health"},
		{"capability": "mqeb", "version": ">=1.0.0", "required": True, "purpose": "Gateway event publication"},
		{"capability": "conf", "version": ">=1.0.0", "required": True, "purpose": "Configuration and service discovery"},
		{"capability": "audl", "version": ">=1.0.0", "required": True, "purpose": "Audit trails"},
		{"capability": "keym", "version": ">=1.0.0", "required": True, "purpose": "Secrets, API keys, certificates, and signing material"},
		{"capability": "cach", "version": ">=1.0.0", "required": False, "purpose": "Gateway cache policy adapters"},
		{"capability": "aicr", "version": ">=1.0.0", "required": False, "purpose": "Optional AI route optimization"},
	],
	"provides": [
		{"service": "api_gateway", "interface": "http", "description": "Governed HTTP/HTTPS route publication"},
		{"service": "traffic_management", "interface": "control_plane", "description": "Quota, canary, and traffic policy lifecycle"},
		{"service": "edge_filter_governance", "interface": "wasm", "description": "Signed edge filter governance"},
		{"service": "gateway_audit", "interface": "events", "description": "Gateway lifecycle decision evidence"},
	],
}


async def health_check() -> dict[str, Any]:
	"""Return dependency-light package health for composition."""
	contract = get_capability_contract()
	return {
		"status": "healthy",
		"timestamp": datetime.now(timezone.utc),
		"version": CAPABILITY_METADATA["version"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
	}


async def get_capability_info() -> dict[str, Any]:
	"""Get APIG capability metadata for APG composition."""
	info = CAPABILITY_METADATA.copy()
	info["contract"] = get_capability_contract()
	return info


def register_capability() -> dict[str, Any]:
	"""Register APIG with the APG composition engine."""
	contract = get_capability_contract()
	required = contract["requires"]
	optional = [
		dep["capability"] for dep in CAPABILITY_METADATA["dependencies"]
		if not dep.get("required") and dep["capability"] not in required
	]
	return {
		"name": "apig",
		"aliases": ["api_gateway", "gateway", "traffic_management"],
		"display_name": CAPABILITY_METADATA["display_name"],
		"description": CAPABILITY_METADATA["description"],
		"version": CAPABILITY_METADATA["version"],
		"dependencies": required,
		"optional_dependencies": optional,
		"provides": contract["provides"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"capabilities": {
			"upstream_lifecycle": "Register and govern upstream services",
			"consumer_lifecycle": "Register API consumers and credential controls",
			"route_lifecycle": "Request, review, activate, and retire gateway routes",
			"traffic_management": "Apply quota, rate-limit, canary, and rollback guardrails",
			"security_gateway": "Enforce auth, threat, mTLS, and signed filter controls",
			"deployment_gates": "Evaluate gateway deployment readiness and approvals",
			"gateway_agent_composition": "Register governed AI and automation agents as first-class APIG participants",
			"bytewax_lifecycle_batches": "Validate Bytewax-routed lifecycle batches before generated applications mutate APIG state",
			"capability_rules": "Evaluate deterministic gateway governance rules",
			"visual_theming": "Apply gateway-console theme tokens and components",
		},
		"endpoints": {
			"upstreams": "/apig/api/v1/upstreams",
			"consumers": "/apig/api/v1/consumers",
			"routes": "/apig/api/v1/routes",
			"traffic": "/apig/api/v1/traffic",
			"security": "/apig/api/v1/security",
			"edge": "/apig/api/v1/edge",
			"quota_reviews": "/apig/api/v1/quota-reviews",
			"canary": "/apig/api/v1/canary",
			"deployments": "/apig/api/v1/deployments",
			"analytics": "/apig/api/v1/analytics",
			"agents": "/apig/api/v1/agents",
			"lifecycle": "/apig/api/v1/lifecycle",
			"audit_events": "/apig/api/v1/audit-events",
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": [
			"apig:view",
			"apig:manage_routes",
			"apig:manage_traffic",
			"apig:manage_security",
			"apig:manage_edge",
			"apig:view_metrics",
			"apig:admin",
		],
	}


__version__ = "1.0.0"
__author__ = "APG Platform Team"
__license__ = "MIT"


__all__ = [
	"CAPABILITY_METADATA",
	"health_check",
	"get_capability_info",
	"register_capability",
	"get_capability_contract",
	"evaluate_capability_rules",
	"__version__",
	"__author__",
]
