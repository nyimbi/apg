"""APG AI Agent Composition capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import AgntService

__version__ = "1.0.0"
__capability_id__ = "agnt"
__capability_name__ = "AI Agent Composition"
__apg_dependencies__ = ["aicr", "auth", "wflo"]

capability_metadata: dict[str, Any] = {
	"name": "agnt",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "First-class AI agent declarations, provider-neutral runtimes, team handoffs, memory policy, and governed execution",
	"category": "ai",
	"subcategory": "agent_composition",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["agent_registry", "runtime_registry", "team_composition", "handoff_graphs", "execution_plans", "runtime_approval_governance"],
	"permissions": ["agnt:view", "agnt:compose", "agnt:run", "agnt:manage_runtimes", "agnt:audit", "agnt:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register AGNT with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "agnt",
		"aliases": ["agents", "ai-agents", "agent-composition"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["nlpc", "ragn", "grag", "mchn", "logt", "secu", "bytewax", "audl", "sbox"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"agent_registry": "Register first-class AI agents with model, runtime, tools, memory, and IO contracts",
			"runtime_adapters": "Select fast-changing Codex, Claude Code, OpenCode, Pi, local, or custom agent backends by configuration",
			"runtime_approval_governance": "Request and approve external agent runtimes before use",
			"team_composition": "Compose agents into swarms, teams, and handoff graphs",
			"handoff_graphs": "Validate agent-to-agent flow edges before execution",
			"execution_plans": "Build deterministic execution plans with runtime assignments, handoffs, cost limits, and approval evidence",
			"capability_rules": "Evaluate deterministic agent-composition governance rules",
			"visual_theming": "Apply AI-agent operations theme tokens and components"
		},
		"endpoints": {"agents": "/agnt/api/v1/agents", "teams": "/agnt/api/v1/teams", "runtimes": "/agnt/api/v1/runtimes", "runtime_approvals": "/agnt/api/v1/runtime-approvals", "executions": "/agnt/api/v1/executions", "memory": "/agnt/api/v1/memory", "audit": "/agnt/api/v1/audit"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get AGNT capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["AgntService", "capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
