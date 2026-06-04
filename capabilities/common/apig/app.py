"""Publishable APG capability package entrypoint for API Gateway & Management."""

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
	_SPEC = importlib.util.spec_from_file_location("apig_capability_contract", _CONTRACT_PATH)
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
			"name": "apig",
			"version": "1.0.0",
			"description": "API Gateway & Management package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {
			"apig": {
				"profile": "capability",
				"entrypoint": "app.py",
			}
		},
		"capabilities": {
			"apig": {
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
				"streaming": contract["streaming"],
				"review_evidence": contract["review_evidence"],
				"runtime": {
					"api": "api.py",
					"entrypoint": "app.py",
					"service": "gateway_runtime.py",
					"views": "view_models.py",
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"quota_review": "GatewayQuotaReview",
					"canary_review": "GatewayTrafficShiftRecord",
					"deployment_approval": "GatewayDeploymentRecord",
					"policy_review": "GatewayPolicyRecord",
					"route_retirement": "GatewayRouteRecord",
					"gateway_agent": "GatewayAgentRecord",
					"lifecycle_batch": "GatewayLifecycleBatchRecord",
				},
				"gateway_lifecycle": {
					"upstream": "GatewayUpstreamRecord",
					"consumer": "GatewayConsumerRecord",
					"route": "GatewayRouteRecord",
					"quota_review": "GatewayQuotaReview",
					"policy": "GatewayPolicyRecord",
					"traffic_shift": "GatewayTrafficShiftRecord",
					"deployment": "GatewayDeploymentRecord",
					"gateway_agent": "GatewayAgentRecord",
					"lifecycle_batch": "GatewayLifecycleBatchRecord",
					"audit": "GatewayAuditEvent",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
			}
		},
		"contracts": {
			"apig": {
				"id": "apig",
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
			"capability_dependencies": {"apig": contract["requires"]},
			"applications": {},
			"agent_teams": {
				"apig_gateway_agents": {
					"supported_runtimes": contract["agents"]["supported_runtimes"],
					"supported_roles": contract["agents"]["supported_roles"],
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
			"capability.apig": {
				"id": "capability.apig",
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
			"apig_gateway_agents": contract["agents"],
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
		"name": "apig",
		"display_name": "API Gateway & Management",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["apig"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("apig", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	review_evidence = capability.get("review_evidence", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "apig" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 15:
		errors.append("APIG semantic model route manifest is stale")
	if len(rules) < 33:
		errors.append("APIG semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("APIG adapter manifest must use Bytewax for event streaming")
	if not capability.get("agents", {}).get("first_class"):
		errors.append("APIG semantic model must expose first-class gateway-agent composition")
	if capability.get("streaming", {}).get("required_processor") != "bytewax":
		errors.append("APIG semantic model must require Bytewax lifecycle processing")
	if "gateway_agents" not in review_evidence.get("pending_queues", []):
		errors.append("APIG semantic model must expose gateway-agent pending review evidence")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "apig",
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
