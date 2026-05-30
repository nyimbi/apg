"""Publishable APG capability package entrypoint for Master Data Management."""

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
	_SPEC = importlib.util.spec_from_file_location("mdm_capability_contract", _CONTRACT_PATH)
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
			"name": "mdm",
			"version": "1.0.0",
			"description": "Master Data Management package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {
			"mdm": {
				"profile": "capability",
				"entrypoint": "app.py",
			}
		},
		"capabilities": {
			"mdm": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": ["mdm_operations"],
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
					"views": "view_models.py",
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"duplicate_review": "MdmDuplicateCandidateRecord",
					"golden_record_merge": "MdmMergeRequestRecord",
					"publish_readiness": "MdmPublishRecord",
				},
				"mdm_lifecycle": {
					"entity": "MdmEntityRecord",
					"quality": "MdmQualityRecord",
					"duplicate_candidate": "MdmDuplicateCandidateRecord",
					"golden_record": "MdmGoldenRecord",
					"merge_request": "MdmMergeRequestRecord",
					"cross_reference": "MdmCrossReferenceRecord",
					"publish": "MdmPublishRecord",
					"audit": "MdmAuditEventRecord",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
				"streaming": {
					"engine": contract["configuration"]["integration"]["event_stream_adapter"],
				},
			}
		},
		"contracts": {
			"mdm": {
				"id": "mdm",
				"configuration": contract["configuration"],
				"provides": ["mdm_operations"],
				"requires": [],
			}
		},
		"rules": {
			rule["name"]: rule
			for rule in contract["rule_engine"]["rules"]
		},
		"composition": {
			"capability_dependencies": {"mdm": []},
			"applications": {},
			"agent_teams": {},
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
			"capability.mdm": {
				"id": "capability.mdm",
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
		"name": "mdm",
		"display_name": "Master Data Management",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["mdm"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("mdm", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "mdm" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 13:
		errors.append("MDM semantic model route manifest is stale")
	if len(rules) < 15:
		errors.append("MDM semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("MDM adapter manifest must use Bytewax for event streaming")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "mdm",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
