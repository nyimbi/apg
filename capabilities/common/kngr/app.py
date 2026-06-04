"""Publishable APG capability package entrypoint for Knowledge Graph."""

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
	_SPEC = importlib.util.spec_from_file_location("kngr_capability_contract", _CONTRACT_PATH)
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
			"name": "kngr",
			"version": "1.0.0",
			"description": "Knowledge Graph package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {"kngr": {"profile": "capability", "entrypoint": "app.py"}},
		"capabilities": {
			"kngr": {
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
				"runtime": {
					"api": "api.py",
					"entrypoint": "app.py",
					"service": contract["configuration"]["adapters"]["generated_app_runtime"],
					"helper_runtime": contract["configuration"]["adapters"]["helper_runtime"],
					"views": contract["ui"]["view_module"],
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"low_confidence_source": "KnowledgeSource",
					"low_confidence_entity": "KnowledgeEntity",
					"low_confidence_relationship": "KnowledgeRelationship",
					"low_confidence_enrichment": "SemanticEnrichment",
					"deep_reasoning": "ReasoningPath",
					"publication": "GraphPublication",
					"knowledge_agent": "KnowledgeAgentRecord",
				},
				"review_queues": {
					"sources": "KnowledgeSource.status == pending_review",
					"entities": "KnowledgeEntity.status == pending_review",
					"relationships": "KnowledgeRelationship.status == pending_review",
					"enrichments": "SemanticEnrichment.status == pending_review",
					"reasoning_paths": "ReasoningPath.status == pending_review",
					"knowledge_agents": "KnowledgeAgentRecord.status == pending_review",
				},
				"knowledge_lifecycle": {
					"source": "KnowledgeSource",
					"entity": "KnowledgeEntity",
					"relationship": "KnowledgeRelationship",
					"enrichment": "SemanticEnrichment",
					"reasoning": "ReasoningPath",
					"curation": "CurationRecord",
					"publication": "GraphPublication",
					"knowledge_agent": "KnowledgeAgentRecord",
					"lifecycle_batch": "KngrLifecycleBatchRecord",
					"audit": "KngrAuditEvent",
				},
				"adapters": contract["configuration"]["adapters"],
				"agents": contract["agents"],
				"i18n": {},
				"master_data": {},
				"streaming": contract["streaming"],
			}
		},
		"contracts": {
			"kngr": {
				"id": "kngr",
				"configuration": contract["configuration"],
				"provides": contract["provides"],
				"requires": contract["requires"],
				"agents": contract["agents"],
				"streaming": contract["streaming"],
			}
		},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"composition": {
			"capability_dependencies": {"kngr": contract["requires"]},
			"applications": {},
			"agent_teams": {
				"kngr_semantic_governance": {
					"roles": contract["agents"]["supported_roles"],
					"runtimes": contract["agents"]["supported_runtimes"],
					"stream": contract["streaming"]["lifecycle_stream"],
				}
			},
		},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": 0},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"source_files": ["capability_contract.py", "models.py", "service.py", "knowledge_runtime.py", "views.py"],
		"symbols": {
			"capability.kngr": {
				"id": "capability.kngr",
				"kind": "capability",
				"name": contract["display_name"],
				"file": "capability_contract.py",
				"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
				"references": [],
			}
		},
		"agents": {"kngr_semantic_governance": contract["agents"]},
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
		"name": "kngr",
		"display_name": "Knowledge Graph",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["kngr"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("kngr", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	agents = capability.get("agents", {})
	streaming = capability.get("streaming", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "kngr" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 14:
		errors.append("KNGR semantic model route manifest is stale")
	if len(rules) < 45:
		errors.append("KNGR semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("KNGR adapter manifest must use Bytewax for event streaming")
	if agents.get("first_class") is not True:
		errors.append("KNGR agents must be first-class semantic citizens")
	if streaming.get("required_processor") != "bytewax":
		errors.append("KNGR lifecycle stream must require Bytewax")
	if capability.get("runtime", {}).get("service") != "service.KngrService":
		errors.append("KNGR generated-app runtime is missing")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "kngr",
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
