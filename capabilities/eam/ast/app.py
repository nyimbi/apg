"""Publishable APG capability package entrypoint for Enterprise Asset Management."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

try:
	from .capability_contract import get_capability_contract
except ImportError:
	_contract_path = Path(__file__).resolve().parent / "capability_contract.py"
	_spec = importlib.util.spec_from_file_location("eam_ast_capability_contract", _contract_path)
	if _spec is None or _spec.loader is None:
		raise
	_module = importlib.util.module_from_spec(_spec)
	sys.modules[_spec.name] = _module
	_spec.loader.exec_module(_module)
	get_capability_contract = _module.get_capability_contract


def semantic_model() -> dict[str, Any]:
	contract = get_capability_contract()
	capability_id = contract["capability"]
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"app": {"name": capability_id, "description": "Enterprise asset management package-backed APG capability", "version": "2.1.0", "entity_count": 8},
		"capabilities": {
			capability_id: {
				"name": contract["display_name"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
				"rule_engine": contract["rule_engine"],
				"rules": contract["rule_engine"]["rules"],
				"ui": contract["ui"],
				"screens": {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"]} for route in contract["ui"]["routes"]},
				"theme": contract["theme"],
				"streaming": contract["streaming"],
				"runtime": {"entrypoint": "app.py", "service": "service.py", "api": "api.py", "views": "views.py"},
			}
		},
		"composition": {
			"capability_dependencies": {capability_id: contract["requires"]},
			"agent_teams": {
				"enterprise_asset_management": {
					"supported_runtimes": contract["configuration"]["eam_agents"]["supported_runtimes"],
					"supported_roles": contract["configuration"]["eam_agents"]["supported_roles"],
				}
			},
		},
		"contracts": {capability_id: {"id": capability_id, "provides": contract["provides"], "requires": contract["requires"], "configuration": contract["configuration"]}},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"packages": {capability_id: {"entrypoint": "app.py", "profile": "capability"}},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"agents": {},
		"flows": {},
		"graphs": {"capability": {"kind": "capability", "nodes": 1, "edges": len(contract["requires"])}},
		"diagnostics": [],
	}


def component_manifest() -> dict[str, Any]:
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "eam_ast",
		"display_name": "Enterprise Asset Management",
		"target": "python",
		"interfaces": {"health": "/health", "self_test": "/self-test", "semantic_model": "/semantic-model.json"},
		"capabilities": ["eam_ast"],
	}


def self_test() -> dict[str, Any]:
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("eam_ast", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if not capability:
		errors.append("capability missing from semantic model")
	if capability.get("streaming", {}).get("processor") != "bytewax":
		errors.append("streaming processor mismatch")
	if "eam_agents" not in capability.get("provides", []):
		errors.append("agent capability missing")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {"passed": not errors, "status": "ok" if not errors else "failed", "errors": errors, "routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"], "capability": "eam_ast"}


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
