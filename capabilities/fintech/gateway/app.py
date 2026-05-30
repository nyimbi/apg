"""Publishable APG capability entrypoint for Fintech Gateway."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent


try:
	from .capability_contract import get_capability_contract
except ImportError:  # pragma: no cover - direct script execution
	spec = importlib.util.spec_from_file_location("gateway_capability_contract", PACKAGE_DIR / "capability_contract.py")
	assert spec is not None and spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	get_capability_contract = module.get_capability_contract


def semantic_model() -> dict[str, Any]:
	"""Return the package semantic model."""
	contract = get_capability_contract()
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"app": {
			"name": "fintech_gateway",
			"version": "2.1.0",
			"description": "Fintech Gateway package-backed APG capability",
			"entity_count": 12,
		},
		"capabilities": {
			"fintech_gateway": {
				"name": contract["name"],
				"version": contract["version"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
				"rules": contract["rule_engine"]["rules"],
				"rule_engine": contract["rule_engine"],
				"ui": contract["ui"],
				"screens": {
					route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"]}
					for route in contract["ui"]["routes"]
				},
				"theme": contract["theme"],
				"streaming": contract["streaming"],
				"runtime": {"entrypoint": "app.py", "service": "service.py", "api": "api.py", "views": "views.py"},
			}
		},
		"contracts": {
			"fintech_gateway": {
				"id": "fintech_gateway",
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
			}
		},
		"composition": {
			"capability_dependencies": {"fintech_gateway": contract["requires"]},
			"agent_teams": {
				"gateway_operations_review": {
					"runtimes": contract["configuration"]["gateway_agents"]["supported_runtimes"],
					"roles": contract["configuration"]["gateway_agents"]["supported_roles"],
				}
			},
			"applications": {},
		},
		"packages": {"fintech_gateway": {"entrypoint": "app.py", "profile": "capability"}},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {"capability": {"kind": "capability", "nodes": 1, "edges": len(contract["requires"])}},
		"diagnostics": [],
	}


def component_manifest() -> dict[str, Any]:
	"""Return the APG component manifest for this capability package."""
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "fintech_gateway",
		"display_name": "Fintech Gateway",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["fintech_gateway"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	capability = model.get("capabilities", {}).get("fintech_gateway", {})
	errors: list[str] = []
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if not capability:
		errors.append("capability missing from semantic model")
	if capability.get("streaming", {}).get("processor") != "bytewax":
		errors.append("streaming processor must be bytewax")
	if "gateway_agents" not in capability.get("provides", []):
		errors.append("gateway_agents provide missing")
	if "agents" not in capability.get("screens", {}):
		errors.append("agent workbench screen missing")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "fintech_gateway",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
