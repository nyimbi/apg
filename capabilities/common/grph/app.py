"""Publishable APG capability package entrypoint for Graph Data Management."""

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
	_SPEC = importlib.util.spec_from_file_location("grph_capability_contract", _CONTRACT_PATH)
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
			"name": "grph",
			"version": "1.0.0",
			"description": "Graph Data Management package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {"grph": {"profile": "capability", "entrypoint": "app.py"}},
		"capabilities": {
			"grph": {
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
				"runtime": {
					"api": "api.py",
					"entrypoint": "app.py",
					"service": contract["configuration"]["adapters"]["generated_app_runtime"],
					"helper_runtime": contract["configuration"]["adapters"]["helper_runtime"],
					"views": contract["ui"]["view_module"],
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"restricted_relationship": "GraphEdge",
					"deep_traversal": "GraphTraversalResult",
					"quality_report": "GraphQualityReport",
					"schema_retire": "GraphSchema",
					"graph_agent": "GraphAgentRecord",
				},
				"review_queues": {
					"schemas": "GraphSchema.status == pending_review",
					"nodes": "GraphNode.status == pending_review",
					"edges": "GraphEdge.status == pending_review",
					"traversals": "GraphTraversalResult.status == pending_review",
					"quality_reports": "GraphQualityReport.status == pending_review",
					"graph_agents": "GraphAgentRecord.status == pending_review",
				},
				"graph_lifecycle": {
					"schema": "GraphSchema",
					"node": "GraphNode",
					"edge": "GraphEdge",
					"traversal": "GraphTraversalResult",
					"quality": "GraphQualityReport",
					"graph_agent": "GraphAgentRecord",
					"lifecycle_batch": "GrphLifecycleBatchRecord",
					"audit": "GraphAuditEventRecord",
				},
				"adapters": contract["configuration"]["adapters"],
				"agents": contract["agents"],
				"i18n": {},
				"master_data": {},
				"streaming": contract["streaming"],
			}
		},
		"contracts": {
			"grph": {
				"id": "grph",
				"configuration": contract["configuration"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"agents": contract["agents"],
				"streaming": contract["streaming"],
			}
		},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"composition": {
			"capability_dependencies": {"grph": contract["requires"]},
			"applications": {},
			"agent_teams": {
				"grph_relationship_governance": {
					"roles": contract["agents"]["supported_roles"],
					"runtimes": contract["agents"]["supported_runtimes"],
					"stream": contract["streaming"]["lifecycle_stream"],
				}
			},
		},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": 0},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"source_files": ["capability_contract.py", "models.py", "service.py", "graph_runtime.py", "views.py"],
		"symbols": {
			"capability.grph": {
				"id": "capability.grph",
				"kind": "capability",
				"name": contract["display_name"],
				"file": "capability_contract.py",
				"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
				"references": [],
			}
		},
		"agents": {"grph_relationship_governance": contract["agents"]},
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
		"name": "grph",
		"display_name": "Graph Data Management",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["grph"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("grph", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	agents = capability.get("agents", {})
	streaming = capability.get("streaming", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "grph" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 14:
		errors.append("GRPH semantic model route manifest is stale")
	if len(rules) < 43:
		errors.append("GRPH semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("GRPH adapter manifest must use Bytewax for event streaming")
	if agents.get("first_class") is not True:
		errors.append("GRPH agents must be first-class semantic citizens")
	if streaming.get("required_processor") != "bytewax":
		errors.append("GRPH lifecycle stream must require Bytewax")
	if capability.get("runtime", {}).get("service") != "service.GrphService":
		errors.append("GRPH generated-app runtime is missing")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "grph",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
