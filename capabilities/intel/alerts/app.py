"""Standalone APG Alert Management server."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent

try:
	from .capability_contract import (
		CAPABILITY_ID,
		CAPABILITY_NAME,
		CAPABILITY_VERSION,
		get_capability_contract,
		evaluate_capability_rules,
	)
except ImportError:  # pragma: no cover - direct script execution
	spec = importlib.util.spec_from_file_location("alerts_capability_contract", PACKAGE_DIR / "capability_contract.py")
	assert spec is not None and spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	CAPABILITY_ID = module.CAPABILITY_ID
	CAPABILITY_NAME = module.CAPABILITY_NAME
	CAPABILITY_VERSION = module.CAPABILITY_VERSION
	get_capability_contract = module.get_capability_contract
	evaluate_capability_rules = module.evaluate_capability_rules


def semantic_model() -> dict[str, Any]:
	contract = get_capability_contract()
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"app": {"name": "intel_alerts", "version": CAPABILITY_VERSION, "description": "Alert Management package-backed APG capability", "entity_count": 11},
		"capabilities": {
			"intel_alerts": {
				"name": contract["name"],
				"version": contract["version"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
				"rules": contract["rule_engine"]["rules"],
				"rule_engine": contract["rule_engine"],
				"ui": contract["ui"],
				"screens": {
					route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"]}
					for route in contract["ui"]["routes"]
				},
				"theme": contract["theme"],
				"streaming": contract["streaming"],
				"runtime": {"entrypoint": "app.py", "service": "service.py", "api": "api.py", "views": "views.py"},
			}
		},
		"contracts": {
			"intel_alerts": {
				"id": "intel_alerts",
				"provides": contract["provides"],
				"requires": contract["requires"],
				"configuration": contract["configuration"],
			}
		},
		"composition": {
			"capability_dependencies": {"intel_alerts": contract["requires"]},
			"agent_teams": {
				"alert_operations": {
					"runtimes": contract["configuration"]["agents"]["supported_runtimes"],
					"roles": contract["configuration"]["agents"]["supported_roles"],
				}
			},
			"applications": {},
		},
		"packages": {"intel_alerts": {"entrypoint": "app.py", "profile": "capability"}},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {"capability": {"kind": "capability", "nodes": 1, "edges": len(contract["requires"])}},
		"diagnostics": [],
	}


def component_manifest() -> dict[str, Any]:
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "intel_alerts",
		"display_name": "Alert Management",
		"target": "python",
		"interfaces": {"health": "/health", "self_test": "/self-test", "semantic_model": "/semantic-model.json"},
		"capabilities": ["intel_alerts"],
	}


def self_test() -> dict[str, Any]:
	model = semantic_model()
	manifest = component_manifest()
	capability = model.get("capabilities", {}).get("intel_alerts", {})
	errors: list[str] = []
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if not capability:
		errors.append("capability missing from semantic model")
	if capability.get("streaming", {}).get("processor") != "bytewax":
		errors.append("streaming processor must be bytewax")
	if "alert_agent_workflow" not in capability.get("provides", []):
		errors.append("alert_agent_workflow provide missing")
	if "agents" not in capability.get("screens", {}):
		errors.append("agent workbench screen missing")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": CAPABILITY_ID,
	}


# ──────────────────────────────────────────────────────────────────────────────
# Standalone HTTP server
# Run: python -m <module_name>  OR  <package-name> --port 8080
# ──────────────────────────────────────────────────────────────────────────────

try:
	from flask import Flask, jsonify, request

	def create_app(config: dict | None = None) -> Flask:
		"""Create the standalone Flask application for this capability."""
		app = Flask(__name__)
		if config:
			app.config.update(config)

		# Wire adapters — null fallbacks used when platform capabilities not installed
		from .domain.adapters import get_auth_adapter, get_audit_adapter, get_notify_adapter, get_workflow_adapter
		from .database.store import get_store

		db_url = (config or {}).get("DB_URL") or os.environ.get("APG_DATABASE_URL")
		store  = get_store(db_url)

		try:
			from .service import AlertManagementService
			svc = AlertManagementService()
			app.config["SERVICE"] = svc
		except Exception:
			pass

		try:
			from .api import blueprint as api_bp
			app.register_blueprint(api_bp, url_prefix="/api/v1")
		except (ImportError, AttributeError):
			pass

		try:
			from .views import blueprint as views_bp
			app.register_blueprint(views_bp)
		except (ImportError, AttributeError):
			pass

		@app.get("/health")
		def health():
			return jsonify({"status": "ok", "capability": CAPABILITY_ID, "version": CAPABILITY_VERSION, "standalone": True})

		@app.get("/contract")
		def contract():
			return jsonify(get_capability_contract())

		@app.post("/evaluate")
		def evaluate():
			ctx = request.get_json(force=True, silent=True) or {}
			return jsonify(evaluate_capability_rules(ctx))

		@app.get("/semantic-model.json")
		def semantic_model_route():
			return jsonify(semantic_model())

		@app.get("/openapi.json")
		def openapi_spec():
			return jsonify({
				"openapi": "3.1.0",
				"info": {"title": CAPABILITY_NAME, "version": CAPABILITY_VERSION},
				"paths": {
					"/health":   {"get":  {"summary": "Liveness probe",       "responses": {"200": {"description": "ok"}}}},
					"/contract": {"get":  {"summary": "Capability contract",   "responses": {"200": {"description": "contract"}}}},
					"/evaluate": {"post": {"summary": "Rule evaluation",       "responses": {"200": {"description": "result"}}}},
					"/api/v1":   {"get":  {"summary": "API root",              "responses": {"200": {"description": "ok"}}}},
				},
			})

		return app

except ImportError:
	def create_app(config=None):  # type: ignore[misc]
		raise ImportError("flask is required for standalone HTTP mode: pip install flask")


def main(argv=None):
	parser = argparse.ArgumentParser(description=f"APG {CAPABILITY_NAME} standalone server")
	parser.add_argument("--host",   default="127.0.0.1")
	parser.add_argument("--port",   type=int, default=8080)
	parser.add_argument("--debug",  action="store_true")
	parser.add_argument("--db-url", default=None, help="PostgreSQL URL (optional; default: in-memory)")
	parser.add_argument("--tenant", default="default", help="Default tenant ID")
	args = parser.parse_args(argv)

	app = create_app({"DB_URL": args.db_url, "DEFAULT_TENANT": args.tenant})
	print(f"APG {CAPABILITY_NAME} v{CAPABILITY_VERSION}")
	print(f"  Standalone mode: {'PostgreSQL' if args.db_url else 'InMemory'}")
	print(f"  Listening: http://{args.host}:{args.port}")
	print(f"  Contract:  http://{args.host}:{args.port}/contract")
	app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
	main()
