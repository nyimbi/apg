"""Publishable APG capability package entrypoint for Help and Knowledge Base."""

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
	_SPEC = importlib.util.spec_from_file_location("help_capability_contract", _CONTRACT_PATH)
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
	dependencies = ["ragn", "srch", "nlpc", "auth", "audl", "aicr"]
	provides = ["help_operations", "support_knowledge", "help_agent_composition"]
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"app": {"name": "help", "version": "1.0.0", "description": "Help and Knowledge Base package-backed APG capability", "entity_count": 0},
		"packages": {"help": {"profile": "capability", "entrypoint": "app.py"}},
		"capabilities": {
			"help": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": provides,
				"requires": dependencies,
				"erp_modules": ["common"],
				"rule_engine": contract["rule_engine"],
				"rules": contract["rule_engine"]["rules"],
				"ui": contract["ui"],
				"screens": routes,
				"theme": contract["theme"],
				"agents": contract["agents"],
				"streaming": contract["streaming"],
				"runtime": {
					"entrypoint": "app.py",
					"service": contract["configuration"]["adapters"]["generated_app_runtime"],
					"helpers": contract["configuration"]["adapters"]["runtime_helpers"],
					"api": contract["configuration"]["adapters"]["api_helpers"],
					"views": contract["configuration"]["adapters"]["view_models"],
				},
				"help_lifecycle": {
					"source": "HelpSource",
					"article": "HelpArticle",
					"answer": "HelpAnswer",
					"feedback": "HelpFeedback",
					"localization": "HelpLocalization",
					"curation": "HelpCurationItem",
					"help_agent": "HelpAgentRecord",
					"lifecycle_batch": "HelpLifecycleBatchRecord",
					"audit": "HelpAuditEvent",
				},
				"adapters": contract["configuration"]["adapters"],
				"business_rules": [],
				"components": {},
				"approvals": {
					"source_approval": "HelpSourceApproval",
					"publication_approval": "KnowledgePublicationApproval",
					"answer_confidence_review": "AnswerConfidenceReview",
					"privileged_help_agent_review": "HelpAgentHumanApprovalReview",
				},
				"i18n": {},
				"master_data": {},
			}
		},
		"contracts": {
			"help": {
				"id": "help",
				"configuration": contract["configuration"],
				"provides": provides,
				"requires": dependencies,
				"agents": contract["agents"],
				"streaming": contract["streaming"],
			}
		},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"composition": {
			"capability_dependencies": {"help": dependencies},
			"applications": {},
			"agent_teams": {
				"support_knowledge_governance": {
					"runtimes": contract["agents"]["supported_runtimes"],
					"roles": contract["agents"]["supported_roles"],
					"adapter_contract": contract["agents"]["adapter_contract"],
				}
			},
		},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {"capability": {"kind": "capability", "nodes": 1, "edges": len(dependencies)}, "package": {"kind": "package", "nodes": 2, "edges": 1}},
		"source_files": ["capability_contract.py", "models.py", "help_runtime.py", "service.py", "api.py", "views.py", "app.py"],
		"symbols": {
			"capability.help": {
				"id": "capability.help",
				"kind": "capability",
				"name": contract["display_name"],
				"file": "capability_contract.py",
				"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
				"references": [],
			}
		},
		"agents": {"help": contract["agents"]},
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
		"name": "help",
		"display_name": "Help and Knowledge Base",
		"target": "python",
		"interfaces": {"health": "/health", "self_test": "/self-test", "semantic_model": "/semantic-model.json"},
		"capabilities": ["help"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("help", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "help" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 13:
		errors.append("HELP semantic model route manifest is stale")
	if len(rules) < 35:
		errors.append("HELP semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("HELP adapter manifest must use Bytewax for event streaming")
	if not capability.get("agents", {}).get("first_class"):
		errors.append("HELP semantic model must expose first-class agents")
	if capability.get("streaming", {}).get("required_processor") != "bytewax":
		errors.append("HELP lifecycle streaming must require Bytewax")
	if capability.get("runtime", {}).get("service") != "service.HelpService":
		errors.append("HELP generated-app runtime is missing")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "help",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
