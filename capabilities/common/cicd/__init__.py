"""APG Continuous Integration/Delivery (CICD) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import CicdService

__version__ = "1.0.0"
__capability_id__ = "cicd"
__capability_name__ = "Continuous Integration and Delivery"
__apg_dependencies__ = ["depl", "envm", "logt"]

capability_metadata: dict[str, Any] = {
	"name": "cicd",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware pipelines, builds, quality gates, artifacts, promotions, scoped AI delivery agents, release automation, and delivery governance",
	"category": "infrastructure_operations",
	"subcategory": "continuous_delivery",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["pipeline_management", "build_orchestration", "quality_gates", "artifact_promotion", "release_automation", "delivery_agents", "delivery_audit"],
	"permissions": ["cicd:view", "cicd:manage_pipelines", "cicd:run_builds", "cicd:promote", "cicd:audit", "cicd:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register CICD with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "cicd",
		"aliases": ["ci_cd", "continuous_delivery", "pipeline_automation"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["scpt", "ntfy", "comp", "edge", "bytewax", "audl", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"pipeline_management": "Define tenant-scoped build, test, package, scan, and delivery pipelines",
			"build_orchestration": "Run builds with workers, caches, secrets, logs, and trace evidence",
			"quality_gates": "Enforce tests, scans, approvals, and artifact policies before promotion",
			"release_automation": "Promote artifacts through environments and deployment capabilities",
			"delivery_agents": "Register scoped AI delivery agents across Codex, Claude Code, OpenCode, Pi, and compatible runtimes",
			"delivery_audit": "Record pipeline, build, artifact, gate, promotion, agent, and state-change evidence",
			"bytewax_streaming": "Declare Bytewax lifecycle streams for CI/CD batch and state events",
			"capability_rules": "Evaluate deterministic CI/CD governance rules",
			"visual_theming": "Apply pipeline-automation theme tokens and components"
		},
		"endpoints": {"pipelines": "/cicd/api/v1/pipelines", "builds": "/cicd/api/v1/builds", "artifacts": "/cicd/api/v1/artifacts", "gates": "/cicd/api/v1/gates", "promotions": "/cicd/api/v1/promotions", "agents": "/cicd/api/v1/agents", "audit": "/cicd/api/v1/audit"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get CICD capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["CicdService", "capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
