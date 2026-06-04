"""Publishable APG capability package entrypoint for API/Service Registry."""

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
	_SPEC = importlib.util.spec_from_file_location("regy_capability_contract", _CONTRACT_PATH)
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
			"name": "regy",
			"version": "1.0.0",
			"description": "API/Service Registry package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {
			"regy": {
				"profile": "capability",
				"entrypoint": "app.py",
			}
		},
		"capabilities": {
			"regy": {
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
					"service": "registry_runtime.py",
					"views": "view_models.py",
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"production_registration": "RegistryReviewRecord",
					"compatibility_review": "RegistryReviewRecord",
					"discovery_limit_review": "RegistryReviewRecord",
					"service_retirement": "RegistryServiceRecord",
					"registry_agent_privileged_role": "RegistryAgentRecord",
				},
				"registry_lifecycle": {
					"service": "RegistryServiceRecord",
					"instance": "RegistryInstanceRecord",
					"version": "RegistryVersionRecord",
					"gateway_publication": "RegistryGatewayPublication",
					"review": "RegistryReviewRecord",
					"registry_agent": "RegistryAgentRecord",
					"lifecycle_batch": "RegistryLifecycleBatchRecord",
					"audit": "RegistryAuditEvent",
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
			"regy": {
				"id": "regy",
				"configuration": contract["configuration"],
				"provides": provides,
				"requires": requires,
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
			"capability_dependencies": {"regy": requires},
			"applications": {},
			"agent_teams": {
				"registry_governance": {
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
			"capability.regy": {
				"id": "capability.regy",
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
			"regy": contract["agents"],
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
		"name": "regy",
		"display_name": "API/Service Registry",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["regy"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("regy", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	review_evidence = capability.get("review_evidence", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "regy" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 14:
		errors.append("REGY semantic model route manifest is stale")
	if len(rules) < 33:
		errors.append("REGY semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("REGY adapter manifest must use Bytewax for event streaming")
	if capability.get("runtime", {}).get("service") != "registry_runtime.py":
		errors.append("REGY generated-app runtime is missing")
	if capability.get("agents", {}).get("first_class") is not True:
		errors.append("REGY semantic model must expose first-class registry agents")
	if capability.get("streaming", {}).get("required_processor") != "bytewax":
		errors.append("REGY lifecycle batches must require Bytewax")
	if "registry_agents" not in review_evidence.get("pending_queues", []):
		errors.append("REGY semantic model must expose registry-agent pending review evidence")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "regy",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))


# ──────────────────────────────────────────────────────────────────────────────
# Standalone HTTP server — added by APG packaging pipeline
# Run: python -m <module_name>  OR  <package-name> --port 8080
# ──────────────────────────────────────────────────────────────────────────────
import argparse as _argparse

try:
	from flask import Flask as _Flask, jsonify as _jsonify, request as _request

	def create_app(config: dict | None = None):
		"""Create the standalone Flask application for this capability."""
		app = _Flask(__name__)
		if config:
			app.config.update(config)

		try:
			from .api import blueprint as _api_bp
			app.register_blueprint(_api_bp, url_prefix="/api/v1")
		except (ImportError, AttributeError):
			pass

		try:
			from .views import blueprint as _views_bp
			app.register_blueprint(_views_bp)
		except (ImportError, AttributeError):
			pass

		@app.get("/health")
		def _health():
			return _jsonify({"status": "ok", "capability": get_capability_contract().get("capability"), "version": get_capability_contract().get("version")})

		@app.get("/contract")
		def _contract():
			return _jsonify(get_capability_contract())

		@app.post("/evaluate")
		def _evaluate():
			from .capability_contract import evaluate_capability_rules
			ctx = _request.get_json(force=True, silent=True) or {}
			return _jsonify(evaluate_capability_rules(ctx))

		@app.get("/semantic-model.json")
		def _semantic_model():
			return _jsonify(semantic_model())

		return app

except ImportError:
	# Flask not installed — create_app is unavailable in this environment
	def create_app(config=None):
		raise ImportError("flask is required for standalone HTTP mode: pip install flask")


def main(argv=None):
	parser = _argparse.ArgumentParser(description=f"APG capability server")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8080)
	parser.add_argument("--debug", action="store_true")
	args = parser.parse_args(argv)
	app = create_app()
	app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
	main()
