"""Publishable APG capability package entrypoint for Connection Management."""

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
	_SPEC = importlib.util.spec_from_file_location("conn_capability_contract", _CONTRACT_PATH)
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
			"name": "conn",
			"version": "1.0.0",
			"description": "Connection Management package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {
			"conn": {
				"profile": "capability",
				"entrypoint": "app.py",
			}
		},
		"capabilities": {
			"conn": {
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
				"runtime": {
					"api": "api.py",
					"entrypoint": "app.py",
					"service": "conn_runtime.py",
					"views": "view_models.py",
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"marketplace_review": "ReviewRecord",
					"activation_review": "ReviewRecord",
					"schema_review": "ReviewRecord",
					"owner_transfer": "ReviewRecord",
					"connection_retirement": "ConnectionRecord",
					"connector_agent_privileged_role": "ConnectorAgentRecord",
				},
				"connector_lifecycle": {
					"connector": "ConnectorRecord",
					"connection": "ConnectionRecord",
					"flow": "FlowRecord",
					"sync_run": "SyncRunRecord",
					"schedule": "ScheduleRecord",
					"review": "ReviewRecord",
					"connector_agent": "ConnectorAgentRecord",
					"lifecycle_batch": "ConnectorLifecycleBatchRecord",
					"audit": "ConnectorAuditEvent",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
				"streaming": {
					**contract["streaming"],
				},
			}
		},
		"contracts": {
			"conn": {
				"id": "conn",
				"configuration": contract["configuration"],
				"provides": provides,
				"requires": requires,
				"agents": contract["agents"],
				"streaming": contract["streaming"],
			}
		},
		"rules": {
			rule["name"]: rule
			for rule in contract["rule_engine"]["rules"]
		},
		"composition": {
			"capability_dependencies": {"conn": requires},
			"applications": {},
			"agent_teams": {
				"connector_governance": {
					"supported_runtimes": contract["agents"]["supported_runtimes"],
					"roles": contract["agents"]["supported_roles"],
					"approval": contract["agents"]["approval"],
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
			"capability.conn": {
				"id": "capability.conn",
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
			"conn": contract["agents"],
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
		"name": "conn",
		"display_name": "Connection Management",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["conn"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("conn", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "conn" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 14:
		errors.append("CONN semantic model route manifest is stale")
	if len(rules) < 39:
		errors.append("CONN semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("CONN adapter manifest must use Bytewax for event streaming")
	if capability.get("runtime", {}).get("service") != "conn_runtime.py":
		errors.append("CONN generated-app runtime is missing")
	if capability.get("agents", {}).get("first_class") is not True:
		errors.append("CONN semantic model must expose first-class connector agents")
	if capability.get("streaming", {}).get("required_processor") != "bytewax":
		errors.append("CONN lifecycle batches must require Bytewax")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "conn",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
