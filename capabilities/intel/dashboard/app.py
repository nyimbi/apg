"""Publishable APG capability entrypoint for Intelligence Dashboard."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent

try:
	from .capability_contract import get_capability_contract
except ImportError:  # pragma: no cover
	spec = importlib.util.spec_from_file_location("dashboard_capability_contract", PACKAGE_DIR / "capability_contract.py")
	assert spec is not None and spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	get_capability_contract = module.get_capability_contract


def semantic_model() -> dict[str, Any]:
	contract = get_capability_contract()
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"app": {"name": "intel_dashboard", "version": contract["version"], "description": "Intelligence Dashboard package-backed APG capability", "entity_count": 11},
		"capabilities": {"intel_dashboard": {"name": contract["name"], "version": contract["version"], "provides": contract["provides"], "requires": contract["requires"], "configuration": contract["configuration"], "rules": contract["rule_engine"]["rules"], "rule_engine": contract["rule_engine"], "ui": contract["ui"], "screens": {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"]} for route in contract["ui"]["routes"]}, "theme": contract["theme"], "streaming": contract["streaming"], "runtime": {"entrypoint": "app.py", "service": "service.py", "api": "api.py", "views": "views.py"}}},
		"contracts": {"intel_dashboard": {"id": "intel_dashboard", "provides": contract["provides"], "requires": contract["requires"], "configuration": contract["configuration"]}},
		"composition": {"capability_dependencies": {"intel_dashboard": contract["requires"]}, "agent_teams": {"dashboard_operations": {"runtimes": contract["configuration"]["agents"]["supported_runtimes"], "roles": contract["configuration"]["agents"]["supported_roles"]}}, "applications": {}},
		"packages": {"intel_dashboard": {"entrypoint": "app.py", "profile": "capability"}},
		"deployment": {"source": "capability_contract.py", "target": "python"},
		"graphs": {"capability": {"kind": "capability", "nodes": 1, "edges": len(contract["requires"])}},
		"diagnostics": [],
	}


def component_manifest() -> dict[str, Any]:
	return {"format": "apg.component-manifest.v1", "kind": "apg.generated_application", "name": "intel_dashboard", "display_name": "Intelligence Dashboard", "target": "python", "interfaces": {"health": "/health", "self_test": "/self-test", "semantic_model": "/semantic-model.json"}, "capabilities": ["intel_dashboard"]}


def self_test() -> dict[str, Any]:
	model = semantic_model()
	manifest = component_manifest()
	capability = model.get("capabilities", {}).get("intel_dashboard", {})
	errors: list[str] = []
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if not capability:
		errors.append("capability missing from semantic model")
	if capability.get("streaming", {}).get("processor") != "bytewax":
		errors.append("streaming processor must be bytewax")
	if "dashboard_agent_workflow" not in capability.get("provides", []):
		errors.append("dashboard_agent_workflow provide missing")
	if "agents" not in capability.get("screens", {}):
		errors.append("agent workbench screen missing")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {"passed": not errors, "status": "ok" if not errors else "failed", "errors": errors, "routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"], "capability": "intel_dashboard"}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))

