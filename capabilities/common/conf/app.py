"""Publishable APG capability package entrypoint for Configuration Management."""

from __future__ import annotations

import json
from typing import Any

try:
	from .capability_contract import get_capability_contract
except ImportError:  # pragma: no cover - standalone package loading path
	import importlib.util
	import sys
	from pathlib import Path

	_CONTRACT_PATH = Path(__file__).with_name("capability_contract.py")
	_SPEC = importlib.util.spec_from_file_location("conf_capability_contract", _CONTRACT_PATH)
	assert _SPEC is not None
	assert _SPEC.loader is not None
	_MODULE = importlib.util.module_from_spec(_SPEC)
	sys.modules[_SPEC.name] = _MODULE
	_SPEC.loader.exec_module(_MODULE)
	get_capability_contract = _MODULE.get_capability_contract


def semantic_model() -> dict[str, Any]:
	"""Return the package semantic model from the current capability contract."""
	contract = get_capability_contract("default")
	routes = {
		route["name"]: {
			"route": route["path"],
			"component": route["component"],
			"permission": route["permission"],
		}
		for route in contract["ui"]["routes"]
	}
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"app": {
			"name": "conf",
			"version": "1.0.0",
			"description": "Configuration Management package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {
			"conf": {
				"profile": "capability",
				"entrypoint": "app.py",
			}
		},
		"capabilities": {
			"conf": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": ["conf_operations", "conf_agents"],
				"requires": [],
				"erp_modules": ["common"],
				"rule_engine": contract["rule_engine"],
				"rules": contract["rule_engine"]["rules"],
				"ui": contract["ui"],
				"screens": routes,
				"theme": contract["theme"],
				"runtime": {
					"api": "api.py",
					"entrypoint": "app.py",
					"service": "service.py",
					"views": "views.py",
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"configuration_change": "ConfigurationChange",
					"drift_remediation": "DriftRemediation",
				},
				"i18n": {},
				"master_data": {},
				"streaming": contract["streaming"],
			}
		},
		"contracts": {
			"conf": {
				"id": "conf",
				"configuration": contract["configuration"],
				"provides": ["conf_operations", "conf_agents"],
				"requires": [],
			}
		},
		"rules": {
			rule["name"]: rule
			for rule in contract["rule_engine"]["rules"]
		},
		"composition": {
			"capability_dependencies": {"conf": []},
			"applications": {},
			"agent_teams": {
				"conf_review": {
					"runtimes": contract["configuration"]["conf_agents"]["supported_runtimes"],
					"roles": contract["configuration"]["conf_agents"]["supported_roles"],
				}
			},
		},
		"deployment": {
			"source": "capability_contract.py",
			"target": "python",
		},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": 0},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"source_files": ["capability_contract.py"],
		"symbols": {
			"capability.conf": {
				"id": "capability.conf",
				"kind": "capability",
				"name": contract["display_name"],
				"file": "capability_contract.py",
				"range": {
					"start": {"line": 0, "character": 0},
					"end": {"line": 0, "character": 1},
				},
				"references": [],
			}
		},
		"agents": {"conf_review": contract["configuration"]["conf_agents"]},
		"flows": {},
		"llms": {},
		"operations": {},
		"roles": {},
		"security": {},
		"tables": {},
		"views": {},
		"diagnostics": [],
	}


def component_manifest() -> dict[str, Any]:
	"""Return the APG component manifest for this capability package."""
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "conf",
		"display_name": "Configuration Management",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["conf"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	routes = model.get("capabilities", {}).get("conf", {}).get("ui", {}).get("routes", [])
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "conf" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 12:
		errors.append("CONF semantic model route manifest is stale")
	if model.get("capabilities", {}).get("conf", {}).get("streaming", {}).get("processor") != "bytewax":
		errors.append("CONF streaming metadata must use Bytewax")
	if "conf_agents" not in model.get("capabilities", {}).get("conf", {}).get("provides", []):
		errors.append("CONF agent provide is missing")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "conf",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
