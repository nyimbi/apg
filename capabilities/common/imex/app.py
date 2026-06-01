"""Publishable APG capability package entrypoint for Import/Export."""

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
	_SPEC = importlib.util.spec_from_file_location("imex_capability_contract", _CONTRACT_PATH)
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
	provides = list(contract["provides"])
	requires = list(contract["requires"])
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"app": {
			"name": "imex",
			"version": "1.0.0",
			"description": "Import/Export package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {"imex": {"profile": "capability", "entrypoint": "app.py"}},
		"capabilities": {
			"imex": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": provides,
				"requires": requires,
				"erp_modules": ["common"],
				"rule_engine": contract["rule_engine"],
				"rules": contract["rule_engine"]["rules"],
				"agents": contract["agents"],
				"ui": contract["ui"],
				"screens": routes,
				"theme": contract["theme"],
				"review_evidence": contract["review_evidence"],
				"runtime": {
					"api": "api.py",
					"entrypoint": "app.py",
					"service": "imex_runtime.py",
					"views": "view_models.py",
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"destination": "ReviewRecord",
					"quality": "ReviewRecord",
					"capacity": "ReviewRecord",
					"purge": "ReviewRecord",
					"owner_transfer": "ReviewRecord",
					"transfer_agent_privileged_role": "TransferAgentRecord",
				},
				"transfer_lifecycle": {
					"endpoint": "TransferEndpoint",
					"mapping": "MappingProfile",
					"job": "TransferJob",
					"run": "TransferRun",
					"artifact": "TransferArtifact",
					"review": "ReviewRecord",
					"transfer_agent": "TransferAgentRecord",
					"lifecycle_batch": "TransferLifecycleBatchRecord",
					"audit": "TransferAuditEvent",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
				"streaming": {**contract["streaming"]},
			}
		},
		"contracts": {
			"imex": {
				"id": "imex",
				"configuration": contract["configuration"],
				"provides": provides,
				"requires": requires,
				"agents": contract["agents"],
				"streaming": contract["streaming"],
				"review_evidence": contract["review_evidence"],
			}
		},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"composition": {"capability_dependencies": {"imex": requires}, "applications": {}, "agent_teams": {"transfer_governance": {"supported_runtimes": contract["agents"]["supported_runtimes"], "roles": contract["agents"]["supported_roles"], "approval": contract["agents"]["approval"]}}},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": 0},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"source_files": ["capability_contract.py"],
		"symbols": {
			"capability.imex": {
				"id": "capability.imex",
				"kind": "capability",
				"name": contract["display_name"],
				"file": "capability_contract.py",
				"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
				"references": [],
			}
		},
		"agents": {"imex": contract["agents"]},
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
		"name": "imex",
		"display_name": "Import/Export",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["imex"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("imex", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	review_evidence = capability.get("review_evidence", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "imex" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 14:
		errors.append("IMEX semantic model route manifest is stale")
	if len(rules) < 38:
		errors.append("IMEX semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("IMEX adapter manifest must use Bytewax for event streaming")
	if capability.get("runtime", {}).get("service") != "imex_runtime.py":
		errors.append("IMEX generated-app runtime is missing")
	if capability.get("agents", {}).get("first_class") is not True:
		errors.append("IMEX semantic model must expose first-class transfer agents")
	if capability.get("streaming", {}).get("required_processor") != "bytewax":
		errors.append("IMEX lifecycle batches must require Bytewax")
	if "transfer_agents" not in review_evidence.get("pending_queues", []):
		errors.append("IMEX semantic model must expose transfer-agent pending review evidence")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "imex",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
