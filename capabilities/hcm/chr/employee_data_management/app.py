"""Publishable APG capability entrypoint for HCM Employee Data Management."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent


try:
	from .capability_contract import get_capability_contract
except ImportError:  # pragma: no cover - direct script execution
	spec = importlib.util.spec_from_file_location("employee_data_capability_contract", PACKAGE_DIR / "capability_contract.py")
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
			"name": "chr_employee_data_management",
			"version": "2.1.0",
			"description": "Employee Data Management package-backed APG capability",
			"entity_count": 10,
		},
		"capabilities": {
			"chr_employee_data_management": {
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
			"chr_employee_data_management": {
				"id": "chr_employee_data_management",
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
			}
		},
		"composition": {
			"capability_dependencies": {"chr_employee_data_management": contract["requires"]},
			"agent_teams": {
				"employee_data_review": {
					"runtimes": contract["configuration"]["employee_agents"]["supported_runtimes"],
					"roles": contract["configuration"]["employee_agents"]["supported_roles"],
				}
			},
			"applications": {},
		},
		"packages": {"chr_employee_data_management": {"entrypoint": "app.py", "profile": "capability"}},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {"capability": {"kind": "capability", "nodes": 1, "edges": len(contract["requires"])}},
		"diagnostics": [],
	}


def component_manifest() -> dict[str, Any]:
	"""Return the APG component manifest for this capability package."""
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "chr_employee_data_management",
		"display_name": "Employee Data Management",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["chr_employee_data_management"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	capability = model.get("capabilities", {}).get("chr_employee_data_management", {})
	errors: list[str] = []
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if not capability:
		errors.append("capability missing from semantic model")
	if capability.get("streaming", {}).get("processor") != "bytewax":
		errors.append("streaming processor must be bytewax")
	if "employee_agents" not in capability.get("provides", []):
		errors.append("employee_agents provide missing")
	if "agents" not in capability.get("screens", {}):
		errors.append("agent workbench screen missing")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "chr_employee_data_management",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
