"""APG Quantum Computing capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_QUAN_AGENT_ROLES,
	SUPPORTED_QUAN_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .models import QuanAgent
from .service import QuanService

__version__ = "1.0.0"
__capability_id__ = "quan"
__capability_name__ = "Quantum Computing"
__apg_dependencies__ = ["aicr", "encr", "keym", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "quan",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant quantum backends, circuit experiments, job governance, cryptographic transition planning, and result management",
	"category": "advanced_infrastructure",
	"subcategory": "quantum",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["quantum_backend_registry", "circuit_management", "quantum_job_orchestration", "result_analysis", "post_quantum_governance", "quan_agents"],
	"permissions": ["quan:view", "quan:experiment", "quan:run_jobs", "quan:manage_backends", "quan:admin"],
}


def register_capability() -> dict[str, Any]:
	"""Register QUAN with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "quan",
		"aliases": ["quantum", "quantum-computing", "quantum-lab"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["mlcm", "pred", "comp", "logt"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"quantum_backend_registry": "Register simulators, cloud quantum providers, quotas, and access policy",
			"circuit_management": "Manage circuit definitions, owners, versions, and experiment metadata",
			"quantum_job_orchestration": "Submit, monitor, retry, and audit quantum jobs",
			"result_analysis": "Capture measurements, confidence, and AI-assisted experiment analysis",
			"post_quantum_governance": "Govern cryptographic transition experiments and post-quantum review evidence",
			"quan_agents": "Register scoped AI quantum agents for backend, circuit, job, result, cost, and post-quantum review",
			"capability_rules": "Evaluate deterministic quantum-governance rules",
			"event_streaming": "Emit quantum lifecycle events through Bytewax",
			"visual_theming": "Apply quantum lab theme tokens and components",
		},
		"endpoints": {
			"backends": "/quan/api/v1/backends",
			"circuits": "/quan/api/v1/circuits",
			"jobs": "/quan/api/v1/jobs",
			"experiments": "/quan/api/v1/experiments",
			"results": "/quan/api/v1/results",
			"agents": "/quan/api/v1/agents",
			"audit": "/quan/api/v1/audit",
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"],
	}


def get_capability_info() -> dict[str, Any]:
	"""Get QUAN capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"QuanAgent",
	"QuanService",
	"SUPPORTED_QUAN_AGENT_ROLES",
	"SUPPORTED_QUAN_AGENT_RUNTIMES",
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
