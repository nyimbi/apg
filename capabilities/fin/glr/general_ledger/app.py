"""Publishable APG capability entrypoint for Financial Management General Ledger."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent


try:
	from .capability_contract import get_capability_contract
except ImportError:  # pragma: no cover - direct script execution
	spec = importlib.util.spec_from_file_location("glr_capability_contract", PACKAGE_DIR / "capability_contract.py")
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
			"name": "glr_general_ledger",
			"version": "2.1.0",
			"description": "Financial Management General Ledger package-backed APG capability",
			"entity_count": 9,
		},
		"capabilities": {
			"glr_general_ledger": {
				"name": contract["name"],
				"version": contract["version"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
				"rules": contract["rule_engine"]["rules"],
				"rule_engine": contract["rule_engine"],
				"ui": contract["ui"],
				"screens": {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"]} for route in contract["ui"]["routes"]},
				"theme": contract["theme"],
				"streaming": contract["streaming"],
				"runtime": {"entrypoint": "app.py", "service": "service.py", "api": "api.py", "views": "views.py"},
			}
		},
		"contracts": {
			"glr_general_ledger": {
				"id": "glr_general_ledger",
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
			}
		},
		"composition": {
			"capability_dependencies": {"glr_general_ledger": contract["requires"]},
			"agent_teams": {
				"glr_close_review": {
					"runtimes": contract["configuration"]["glr_agents"]["supported_runtimes"],
					"roles": contract["configuration"]["glr_agents"]["supported_roles"],
				}
			},
			"applications": {},
		},
		"packages": {"glr_general_ledger": {"entrypoint": "app.py", "profile": "capability"}},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {"capability": {"kind": "capability", "nodes": 1, "edges": len(contract["requires"])}},
		"diagnostics": [],
	}


def component_manifest() -> dict[str, Any]:
	"""Return the APG component manifest for this capability package."""
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "glr_general_ledger",
		"display_name": "Financial Management General Ledger",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["glr_general_ledger"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	capability = model.get("capabilities", {}).get("glr_general_ledger", {})
	errors: list[str] = []
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if not capability:
		errors.append("capability missing from semantic model")
	if capability.get("streaming", {}).get("processor") != "bytewax":
		errors.append("streaming processor must be bytewax")
	if "glr_agents" not in capability.get("provides", []):
		errors.append("glr_agents provide missing")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "glr_general_ledger",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
