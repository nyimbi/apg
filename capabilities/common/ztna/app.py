"""Publishable APG capability package entrypoint for Zero Trust Network Access."""

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
	_SPEC = importlib.util.spec_from_file_location("ztna_capability_contract", _CONTRACT_PATH)
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
			"name": "ztna",
			"version": "1.0.0",
			"description": "Zero Trust Network Access package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {"ztna": {"profile": "capability", "entrypoint": "app.py"}},
		"capabilities": {
			"ztna": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": ["ztna_operations", "zero_trust_access", "zero_trust_agent_composition"],
				"requires": ["auth", "secu", "mfau", "moni", "audl", "idfd", "anom", "mqeb", "cach"],
				"erp_modules": ["common"],
				"rule_engine": contract["rule_engine"],
				"rules": contract["rule_engine"]["rules"],
				"agents": contract["agents"],
				"streaming": contract["streaming"],
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
					"high_risk_access": "AccessReview",
					"privileged_access": "AccessReview",
					"unmanaged_privileged_device": "AccessReview",
					"session_reauth": "AccessReview",
				},
				"zero_trust_lifecycle": {
					"identity": "ZeroTrustIdentityRecord",
					"device": "ZeroTrustDeviceRecord",
					"resource": "ZeroTrustResourceRecord",
					"access_request": "ZeroTrustAccessRequestRecord",
					"session": "ZeroTrustSessionRecord",
					"audit": "ZeroTrustAuditEventRecord",
					"zero_trust_agent": "ZeroTrustAgentRecord",
					"lifecycle_batch": "ZtnaLifecycleBatchRecord",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
			}
		},
		"contracts": {
			"ztna": {
				"id": "ztna",
				"configuration": contract["configuration"],
				"provides": ["ztna_operations", "zero_trust_access", "zero_trust_agent_composition"],
				"requires": ["auth", "secu", "mfau", "moni", "audl", "idfd", "anom", "mqeb", "cach"],
				"agents": contract["agents"],
				"streaming": contract["streaming"],
			}
		},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"composition": {
			"capability_dependencies": {"ztna": ["auth", "secu", "mfau", "moni", "audl", "idfd", "anom", "mqeb", "cach"]},
			"applications": {},
			"agent_teams": {"zero_trust_governance": contract["agents"]},
		},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": 9},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"source_files": ["capability_contract.py", "models.py", "zero_trust_runtime.py", "service.py", "api.py", "views.py", "app.py"],
		"symbols": {
			"capability.ztna": {
				"id": "capability.ztna",
				"kind": "capability",
				"name": contract["display_name"],
				"file": "capability_contract.py",
				"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
				"references": [],
			}
		},
		"agents": {"zero_trust_governance": contract["agents"]},
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
		"name": "ztna",
		"display_name": "Zero Trust Network Access",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["ztna"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("ztna", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "ztna" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 13:
		errors.append("ZTNA semantic model route manifest is stale")
	if len(rules) < 42:
		errors.append("ZTNA semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("ZTNA adapter manifest must use Bytewax for event streaming")
	if not capability.get("agents", {}).get("first_class"):
		errors.append("ZTNA agent manifest must make agents first-class")
	if capability.get("streaming", {}).get("required_processor") != "bytewax":
		errors.append("ZTNA lifecycle stream must require Bytewax")
	if capability.get("runtime", {}).get("service") != "service.ZtnaService":
		errors.append("ZTNA generated-app runtime is missing")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "ztna",
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
