"""Publishable APG capability package entrypoint for Graph-based RAG."""

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
	_SPEC = importlib.util.spec_from_file_location("grag_capability_contract", _CONTRACT_PATH)
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
			"name": "grag",
			"version": "1.0.0",
			"description": "Graph-based RAG package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {"grag": {"profile": "capability", "entrypoint": "app.py"}},
		"capabilities": {
			"grag": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": ["grag_operations"],
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
					"service": contract["configuration"]["adapters"]["generated_app_runtime"],
					"production_service": contract["configuration"]["adapters"]["production_runtime"],
					"helper_runtime": contract["configuration"]["adapters"]["helper_runtime"],
					"views": contract["ui"]["view_module"],
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"graph_source_retirement": "GragRecord",
					"large_result_window": "GragRecord",
					"low_retrieval_confidence": "GragRecord",
					"deep_reasoning_path": "GragRecord",
					"low_answer_confidence": "GragRecord",
					"answer_publication": "GragRecord",
				},
				"graph_rag_lifecycle": {
					"graph_source": "GragRecord",
					"vector_source": "GragRecord",
					"hybrid_query": "GragRecord",
					"reasoning_path": "GragRecord",
					"answer": "GragRecord",
					"curation": "GragRecord",
					"publication": "GragRecord",
					"audit": "GragRecord",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
				"streaming": {"engine": contract["configuration"]["adapters"]["event_stream"]},
			}
		},
		"contracts": {
			"grag": {
				"id": "grag",
				"configuration": contract["configuration"],
				"provides": ["grag_operations"],
				"requires": [],
			}
		},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"composition": {"capability_dependencies": {"grag": []}, "applications": {}, "agent_teams": {}},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": 0},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"source_files": ["capability_contract.py", "grag_runtime.py", "api.py", "views.py"],
		"symbols": {
			"capability.grag": {
				"id": "capability.grag",
				"kind": "capability",
				"name": contract["display_name"],
				"file": "capability_contract.py",
				"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
				"references": [],
			}
		},
		"agents": {},
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
		"name": "grag",
		"display_name": "Graph-based RAG",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["grag"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("grag", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "grag" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 12:
		errors.append("GRAG semantic model route manifest is stale")
	if len(rules) < 30:
		errors.append("GRAG semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("GRAG adapter manifest must use Bytewax for event streaming")
	if capability.get("runtime", {}).get("service") != "grag_runtime.GragService":
		errors.append("GRAG generated-app runtime is missing")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "grag",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
