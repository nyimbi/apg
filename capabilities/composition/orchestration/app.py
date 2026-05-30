"""Publishable APG capability package entrypoint for Workflow Orchestration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

try:
	from .capability_contract import get_capability_contract
except ImportError:
	_contract_path = Path(__file__).resolve().parent / "capability_contract.py"
	_spec = importlib.util.spec_from_file_location("composition_orchestration_capability_contract", _contract_path)
	if _spec is None or _spec.loader is None:
		raise
	_module = importlib.util.module_from_spec(_spec)
	sys.modules[_spec.name] = _module
	_spec.loader.exec_module(_module)
	get_capability_contract = _module.get_capability_contract


def semantic_model() -> dict[str, Any]:
	"""Return the package semantic model."""
	contract = get_capability_contract()
	capability_id = contract["capability"]
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"app": {
			"name": capability_id,
			"description": "Workflow orchestration package-backed APG capability",
			"version": "2.1.0",
			"entity_count": 7,
		},
		"capabilities": {
			capability_id: {
				"name": contract["display_name"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
				"rule_engine": contract["rule_engine"],
				"rules": contract["rule_engine"]["rules"],
				"ui": contract["ui"],
				"screens": {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"]} for route in contract["ui"]["routes"]},
				"theme": contract["theme"],
				"streaming": contract["streaming"],
				"runtime": {"entrypoint": "app.py", "service": "service.py", "api": "api.py", "views": "views.py"},
			}
		},
		"composition": {
			"capability_dependencies": {capability_id: contract["requires"]},
			"agent_teams": {
				"workflow_orchestration": {
					"supported_runtimes": contract["configuration"]["automation_agents"]["supported_runtimes"],
					"supported_roles": contract["configuration"]["automation_agents"]["supported_roles"],
				}
			},
		},
		"contracts": {
			capability_id: {
				"id": capability_id,
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
			}
		},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"packages": {capability_id: {"entrypoint": "app.py", "profile": "capability"}},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"agents": {},
		"flows": {},
		"graphs": {"capability": {"kind": "capability", "nodes": 1, "edges": len(contract["requires"])}},
		"diagnostics": [],
	}


def component_manifest() -> dict[str, Any]:
	"""Return the APG component manifest for this capability package."""
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "composition_orchestration",
		"display_name": "Workflow Orchestration",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["composition_orchestration"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("composition_orchestration", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if not capability:
		errors.append("capability missing from semantic model")
	if capability.get("streaming", {}).get("processor") != "bytewax":
		errors.append("streaming processor mismatch")
	if "workflow_agents" not in capability.get("provides", []):
		errors.append("agent capability missing")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "composition_orchestration",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
