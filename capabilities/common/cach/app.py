"""Publishable APG capability package entrypoint for Cache Management."""

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
	_SPEC = importlib.util.spec_from_file_location("cach_capability_contract", _CONTRACT_PATH)
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
			"name": "cach",
			"version": "1.0.0",
			"description": "Cache Management package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {
			"cach": {
				"profile": "capability",
				"entrypoint": "app.py",
			}
		},
		"capabilities": {
			"cach": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"erp_modules": ["common"],
				"rule_engine": contract["rule_engine"],
				"rules": contract["rule_engine"]["rules"],
				"ui": contract["ui"],
				"screens": routes,
				"agents": {
					"cach_agent_contract": contract["agents"],
				},
				"streaming": contract["streaming"],
				"theme": contract["theme"],
				"runtime": {
					"api": "api.py",
					"entrypoint": "app.py",
					"service": "service.py",
					"views": "view_models.py",
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"warming": "CacheWarmingPlanRecord",
					"eviction": "CacheEvictionReviewRecord",
					"cache_agent": "CacheAgentRecord",
				},
				"cache_lifecycle": {
					"namespace": "CacheNamespaceRecord",
					"entry": "CacheEntryRecord",
					"lifecycle_batch": "CacheLifecycleBatchRecord",
					"audit": "CacheAuditEventRecord",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
			}
		},
		"contracts": {
			"cach": {
				"id": "cach",
				"configuration": contract["configuration"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"agents": contract["agents"],
				"streaming": contract["streaming"],
			}
		},
		"rules": {
			rule["name"]: rule
			for rule in contract["rule_engine"]["rules"]
		},
		"composition": {
			"capability_dependencies": {"cach": contract["requires"]},
			"applications": {},
			"agent_teams": {
				"cach_cache_governance": {
					"roles": contract["agents"]["supported_roles"],
					"runtimes": contract["agents"]["supported_runtimes"],
					"requires_human_approval_for": contract["agents"]["privileged_roles"],
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
			"capability.cach": {
				"id": "capability.cach",
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
		"agents": {
			"cach_agent_contract": contract["agents"],
		},
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
		"name": "cach",
		"display_name": "Cache Management",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["cach"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("cach", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	agents = capability.get("agents", {}).get("cach_agent_contract", {})
	streaming = capability.get("streaming", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "cach" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 14:
		errors.append("CACH semantic model route manifest is stale")
	if len(rules) < 22:
		errors.append("CACH semantic model rule manifest is stale")
	if "memory" not in adapters.get("supported_backends", []):
		errors.append("CACH adapter manifest must include memory backend")
	if agents.get("first_class") is not True:
		errors.append("CACH cache agents must remain first-class")
	if streaming.get("engine") != "bytewax":
		errors.append("CACH streaming manifest must remain Bytewax-first")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "cach",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
