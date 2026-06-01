"""Publishable APG capability package entrypoint for Encryption Services."""

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
	_SPEC = importlib.util.spec_from_file_location("encr_capability_contract", _CONTRACT_PATH)
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
			"name": "encr",
			"version": "1.0.0",
			"description": "Encryption Services package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {
			"encr": {
				"profile": "capability",
				"entrypoint": "app.py",
			}
		},
		"capabilities": {
			"encr": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"erp_modules": ["common"],
				"rule_engine": contract["rule_engine"],
				"rules": contract["rule_engine"]["rules"],
				"ui": contract["ui"],
				"screens": routes,
				"theme": contract["theme"],
				"agents": contract["agents"],
				"review_evidence": contract["review_evidence"],
				"runtime": {
					"api": "api.py",
					"entrypoint": "app.py",
					"service": "service.py",
					"views": "views.py",
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"crypto_exception": "CryptoExceptionReviewRecord",
					"key_rotation": "KeyRotationRecord",
					"crypto_agent": "CryptoAgentRecord",
				},
				"i18n": {},
				"master_data": {},
				"streaming": contract["streaming"],
			}
		},
		"contracts": {
			"encr": {
				"id": "encr",
				"configuration": contract["configuration"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"agents": contract["agents"],
				"streaming": contract["streaming"],
				"review_evidence": contract["review_evidence"],
			}
		},
		"rules": {
			rule["name"]: rule
			for rule in contract["rule_engine"]["rules"]
		},
		"composition": {
			"capability_dependencies": {"encr": contract["requires"]},
			"applications": {},
			"agent_teams": {
				"encr_crypto_governance": {
					"capability": "encr",
					"roles": contract["agents"]["supported_roles"],
					"runtimes": contract["agents"]["supported_runtimes"],
					"guardrails": contract["agents"]["guardrails"],
				}
			},
		},
		"deployment": {
			"source": "capability_contract.py",
			"target": "python",
		},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": len(contract["requires"])},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"source_files": ["capability_contract.py"],
		"symbols": {
			"capability.encr": {
				"id": "capability.encr",
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
		"agents": {"encr_agent_contract": contract["agents"]},
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
		"name": "encr",
		"display_name": "Encryption Services",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["encr"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	routes = model.get("capabilities", {}).get("encr", {}).get("ui", {}).get("routes", [])
	approvals = model.get("capabilities", {}).get("encr", {}).get("approvals", {})
	streaming = model.get("capabilities", {}).get("encr", {}).get("streaming", {})
	agents = model.get("capabilities", {}).get("encr", {}).get("agents", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "encr" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 11:
		errors.append("ENCR semantic model route manifest is stale")
	if "crypto_exception" not in approvals or "key_rotation" not in approvals or "crypto_agent" not in approvals:
		errors.append("ENCR semantic model approval manifest is stale")
	if streaming.get("engine") != "bytewax":
		errors.append("ENCR semantic model streaming manifest is stale")
	if not agents.get("first_class"):
		errors.append("ENCR semantic model agent manifest is stale")
	if "review_evidence" not in model.get("capabilities", {}).get("encr", {}).get("provides", []):
		errors.append("ENCR review evidence provide is missing")
	if "review_evidence" not in model.get("capabilities", {}).get("encr", {}):
		errors.append("ENCR durable review evidence is missing")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "encr",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
