"""Publishable APG capability package entrypoint for Notifications and Alerts."""

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
	_SPEC = importlib.util.spec_from_file_location("ntfy_capability_contract", _CONTRACT_PATH)
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
			"name": "ntfy",
			"version": "1.0.0",
			"description": "Notifications and Alerts package-backed APG capability",
			"entity_count": 0,
		},
		"packages": {"ntfy": {"profile": "capability", "entrypoint": "app.py"}},
		"capabilities": {
			"ntfy": {
				"name": contract["display_name"],
				"configuration": contract["configuration"],
				"provides": ["ntfy_operations", "notification_management", "notification_agent_composition"],
				"requires": ["mqeb", "auth", "mten", "audl", "aicr", "secu", "cach"],
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
					"helper_runtime": contract["configuration"]["adapters"]["helper_runtime"],
					"api_helpers": contract["configuration"]["adapters"]["api_helpers"],
					"views": contract["configuration"]["adapters"]["view_models"],
				},
				"business_rules": [],
				"components": {},
				"approvals": {
					"template_approval": "TemplateApproval",
					"campaign_approval": "CampaignApproval",
					"large_batch_review": "BatchReview",
					"quiet_hours_override": "DeliveryReview",
				},
				"notification_lifecycle": {
					"recipient_preference": "RecipientPreferenceRecord",
					"channel_provider": "ChannelProviderRecord",
					"template": "NotificationTemplateRecord",
					"delivery": "DeliveryRecord",
					"campaign": "CampaignRecord",
					"audit": "NotificationAuditEventRecord",
					"notification_agent": "NotificationAgentRecord",
					"lifecycle_batch": "NtfyLifecycleBatchRecord",
				},
				"adapters": contract["configuration"]["adapters"],
				"i18n": {},
				"master_data": {},
			}
		},
		"contracts": {
			"ntfy": {
				"id": "ntfy",
				"configuration": contract["configuration"],
				"provides": ["ntfy_operations", "notification_management", "notification_agent_composition"],
				"requires": ["mqeb", "auth", "mten", "audl", "aicr", "secu", "cach"],
			}
		},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"composition": {
			"capability_dependencies": {"ntfy": ["mqeb", "auth", "mten", "audl", "aicr", "secu", "cach"]},
			"applications": {},
			"agent_teams": {
				"notification_delivery_team": {
					"runtimes": contract["agents"]["supported_runtimes"],
					"roles": contract["agents"]["supported_roles"],
					"stream": contract["streaming"]["lifecycle_stream"],
				}
			},
		},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": 7},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"source_files": ["capability_contract.py", "notification_runtime.py", "package_api.py", "view_models.py", "app.py"],
		"symbols": {
			"capability.ntfy": {
				"id": "capability.ntfy",
				"kind": "capability",
				"name": contract["display_name"],
				"file": "capability_contract.py",
				"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
				"references": [],
			}
		},
		"agents": {
			"ntfy": contract["agents"],
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
		"name": "ntfy",
		"display_name": "Notifications and Alerts",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["ntfy"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	capability = model.get("capabilities", {}).get("ntfy", {})
	routes = capability.get("ui", {}).get("routes", [])
	rules = capability.get("rule_engine", {}).get("rules", [])
	adapters = capability.get("adapters", {})
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "ntfy" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	if len(routes) < 12:
		errors.append("NTFY semantic model route manifest is stale")
	if len(rules) < 40:
		errors.append("NTFY semantic model rule manifest is stale")
	if adapters.get("event_stream") != "bytewax":
		errors.append("NTFY adapter manifest must use Bytewax for event streaming")
	if not capability.get("agents", {}).get("first_class"):
		errors.append("NTFY semantic model must expose first-class agents")
	if capability.get("streaming", {}).get("required_processor") != "bytewax":
		errors.append("NTFY lifecycle streaming must require Bytewax")
	if capability.get("runtime", {}).get("service") != "notification_runtime.NotificationRuntime":
		errors.append("NTFY generated-app runtime is missing")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "ntfy",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
