"""Publishable APG capability package entrypoint for Health Checks and Diagnostics."""

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
	_SPEC = importlib.util.spec_from_file_location("hlth_capability_contract", _CONTRACT_PATH)
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
			"name": "hlth",
			"version": "1.0.0",
			"description": "Health Checks and Diagnostics package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {
			"hlth": {
				"profile": "capability",
				"entrypoint": "app.py",
			}
		},
		"capabilities": {
			"hlth": {
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
					"health_agent_contract": contract["agents"],
				},
				"streaming": contract["streaming"],
				"review_evidence": contract["review_evidence"],
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
					"remediation": "HlthRemediationRequestRecord",
					"deployment_gate": "HlthDeploymentGateRecord",
					"health_agent": "HlthAgentRecord",
				},
				"health_lifecycle": {
					"component": "HlthComponentRecord",
					"check": "HlthCheckRecord",
					"baseline": "HlthBaselineRecord",
					"prediction": "HlthPredictionRecord",
					"alert": "HlthAlertRecord",
					"incident": "HlthIncidentRecord",
					"lifecycle_batch": "HlthLifecycleBatchRecord",
					"audit": "HlthAuditEventRecord",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
			}
		},
		"contracts": {
			"hlth": {
				"id": "hlth",
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
			"capability_dependencies": {"hlth": contract["requires"]},
			"applications": {},
			"agent_teams": {
				"hlth_reliability_governance": {
					"roles": contract["agents"]["supported_roles"],
					"runtimes": contract["agents"]["supported_runtimes"],
					"stream": contract["streaming"]["lifecycle_stream"],
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
			"capability.hlth": {
				"id": "capability.hlth",
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
			"health_agents": contract["agents"],
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
		"name": "hlth",
		"display_name": "Health Checks and Diagnostics",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["hlth"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("hlth", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	agents = capability.get("agents", {}).get("health_agent_contract", {})
	streaming = capability.get("streaming", {})
	review_evidence = capability.get("review_evidence", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "hlth" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 15:
		errors.append("HLTH semantic model route manifest is stale")
	if len(rules) < 26:
		errors.append("HLTH semantic model rule manifest is stale")
	if "moni" not in adapters.get("supported_probe_sources", []):
		errors.append("HLTH adapter manifest must include MONI probe source boundary")
	if "codex" not in agents.get("supported_runtimes", []):
		errors.append("HLTH agent manifest must include Codex runtime")
	if streaming.get("required_processor") != "bytewax":
		errors.append("HLTH streaming manifest must remain Bytewax-first")
	if "health_agents" not in review_evidence.get("pending_queues", []):
		errors.append("HLTH review evidence must expose health-agent pending queue")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "hlth",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
