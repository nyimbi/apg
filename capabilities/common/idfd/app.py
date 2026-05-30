"""Publishable APG capability package entrypoint for Identity Federation."""

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
	_SPEC = importlib.util.spec_from_file_location("idfd_capability_contract", _CONTRACT_PATH)
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
			"name": "idfd",
			"version": "1.0.0",
			"description": "Identity Federation package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {"idfd": {"profile": "capability", "entrypoint": "app.py"}},
		"capabilities": {
			"idfd": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": ["idfd_operations"],
				"requires": ["auth", "mfau", "encr"],
				"erp_modules": ["common"],
				"rule_engine": contract["rule_engine"],
				"rules": contract["rule_engine"]["rules"],
				"ui": contract["ui"],
				"screens": routes,
				"theme": contract["theme"],
				"runtime": {
					"entrypoint": "app.py",
					"service": contract["configuration"]["adapters"]["generated_app_runtime"],
					"helper_runtime": contract["configuration"]["adapters"]["helper_runtime"],
					"api_helpers": contract["configuration"]["adapters"]["api_helpers"],
					"views": contract["configuration"]["adapters"]["view_models"],
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"stale_metadata": "FederationReview",
					"sensitive_claim_mapping": "FederationReview",
					"high_risk_session": "FederationReview",
					"certificate_rotation": "FederationReview",
				},
				"federation_lifecycle": {
					"provider": "FederationProvider",
					"protocol": "ProviderProtocol",
					"claim_mapping": "ClaimMapping",
					"session": "FederatedSession",
					"certificate": "CertificateRecord",
					"health": "FederationHealthReport",
					"audit": "FederationAuditEvent",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
				"streaming": {"engine": contract["configuration"]["adapters"]["event_stream"]},
			}
		},
		"contracts": {
			"idfd": {
				"id": "idfd",
				"configuration": contract["configuration"],
				"provides": ["idfd_operations"],
				"requires": ["auth", "mfau", "encr"],
			}
		},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"composition": {"capability_dependencies": {"idfd": ["auth", "mfau", "encr"]}, "applications": {}, "agent_teams": {}},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": 3},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"source_files": ["capability_contract.py", "models.py", "federation_runtime.py", "service.py", "api.py", "views.py", "app.py"],
		"symbols": {
			"capability.idfd": {
				"id": "capability.idfd",
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
		"name": "idfd",
		"display_name": "Identity Federation",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["idfd"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("idfd", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "idfd" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 10:
		errors.append("IDFD semantic model route manifest is stale")
	if len(rules) < 30:
		errors.append("IDFD semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("IDFD adapter manifest must use Bytewax for event streaming")
	if capability.get("runtime", {}).get("service") != "service.IdfdService":
		errors.append("IDFD generated-app runtime is missing")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "idfd",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
