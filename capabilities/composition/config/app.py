"""Publishable APG capability package entrypoint for Central Configuration Management."""

from __future__ import annotations

import json
from typing import Any


SEMANTIC_MODEL: dict[str, Any] = json.loads(r"""{"agents": {}, "app": {"description": "Central Configuration Management package-backed APG capability", "entity_count": 0, "name": "composition_config", "version": "1.0.0"}, "capabilities": {"composition_config": {"approvals": {}, "business_rules": [], "components": {}, "configuration": {"capability": {"category": "Composition", "enabled": true, "id": "composition_config", "name": "Central Configuration Management", "spec_path": "/Users/nyimbiodero/src/pjs/apg/capabilities/composition/config/cap_spec.md", "version": "1.0.0"}, "execution": {"async_supported": true, "audit_operations": true, "policy_enforced": true, "require_tenant_context": true}, "tenant_id": "default", "theme": {"allow_tenant_overrides": true, "default_theme": "composition_config_operations"}, "ui": {"enable_dashboard": true, "enable_operations": true, "enable_rules": true, "enable_settings": true}}, "erp_modules": ["composition"], "i18n": {}, "master_data": {}, "name": "Central Configuration Management", "provides": ["composition_config_operations"], "requires": [], "rule_engine": {"rules": [{"condition": {"tenant_context_present": false}, "description": "Central Configuration Management operations require tenant context.", "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}, "name": "tenant_context_required"}, {"condition": {"operation_type": "write", "policy_attached": false}, "description": "Central Configuration Management write operations require policy enforcement.", "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}, "name": "operation_policy_required"}, {"condition": {"review_recorded": false, "risk_level": "high"}, "description": "High-risk Central Configuration Management operations require review.", "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}, "name": "high_risk_requires_review"}], "type": "deterministic"}, "rules": [{"condition": {"tenant_context_present": false}, "description": "Central Configuration Management operations require tenant context.", "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}, "name": "tenant_context_required"}, {"condition": {"operation_type": "write", "policy_attached": false}, "description": "Central Configuration Management write operations require policy enforcement.", "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}, "name": "operation_policy_required"}, {"condition": {"review_recorded": false, "risk_level": "high"}, "description": "High-risk Central Configuration Management operations require review.", "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}, "name": "high_risk_requires_review"}], "runtime": {"api": "api.py", "entrypoint": "app.py", "service": "service.py", "views": "views.py"}, "screens": {"dashboard": {"component": "CapabilityDashboard", "permission": "composition_config:view", "route": "/composition-config/dashboard"}, "operations": {"component": "CapabilityOperations", "permission": "composition_config:operate", "route": "/composition-config/operations"}, "rules": {"component": "CapabilityRules", "permission": "composition_config:govern", "route": "/composition-config/rules"}, "settings": {"component": "CapabilitySettings", "permission": "composition_config:admin", "route": "/composition-config/settings"}}, "streaming": {}, "theme": {"components": {"dashboard": {"icon": "layout-dashboard", "risk_style": "policy-band", "status_indicator": "health-pill"}, "operations": {"status_style": "sla-chip", "visual": "work-queue"}, "rules": {"status_style": "decision-chip", "visual": "rule-list"}, "settings": {"density": "compact", "visual": "settings-panel"}}, "name": "composition_config_operations", "tokens": {"border.radius": "8px", "color.accent": "#C44536", "color.danger": "#C53030", "color.primary": "#28536B", "color.success": "#2F855A", "color.warning": "#B7791F", "density": "compact", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D"}}, "ui": {"api_prefix": "/composition-config/api/v1", "requires_theme": true, "routes": [{"component": "CapabilityDashboard", "name": "dashboard", "nav_group": "Overview", "path": "/composition-config/dashboard", "permission": "composition_config:view"}, {"component": "CapabilityOperations", "name": "operations", "nav_group": "Operations", "path": "/composition-config/operations", "permission": "composition_config:operate"}, {"component": "CapabilityRules", "name": "rules", "nav_group": "Governance", "path": "/composition-config/rules", "permission": "composition_config:govern"}, {"component": "CapabilitySettings", "name": "settings", "nav_group": "Administration", "path": "/composition-config/settings", "permission": "composition_config:admin"}], "shell": "apg_python", "template_roots": ["templates/", "static/"], "view_module": "views.py"}}}, "composition": {"agent_teams": {}, "applications": {}, "capability_dependencies": {"composition_config": []}}, "contracts": {"composition_config": {"configuration": {"capability": {"category": "Composition", "enabled": true, "id": "composition_config", "name": "Central Configuration Management", "spec_path": "/Users/nyimbiodero/src/pjs/apg/capabilities/composition/config/cap_spec.md", "version": "1.0.0"}, "execution": {"async_supported": true, "audit_operations": true, "policy_enforced": true, "require_tenant_context": true}, "tenant_id": "default", "theme": {"allow_tenant_overrides": true, "default_theme": "composition_config_operations"}, "ui": {"enable_dashboard": true, "enable_operations": true, "enable_rules": true, "enable_settings": true}}, "id": "composition_config", "provides": ["composition_config_operations"], "requires": []}}, "deployment": {"source": "capability_contract.py", "target": "python"}, "diagnostics": [], "flows": {}, "format": "apg.semantic-model.v1", "graphs": {"capability": {"edges": 0, "kind": "capability", "nodes": 1}, "package": {"edges": 1, "kind": "package", "nodes": 2}}, "llms": {}, "ok": true, "operations": {}, "packages": {"composition_config": {"entrypoint": "app.py", "profile": "capability"}}, "roles": {}, "rules": {"high_risk_requires_review": {"condition": {"review_recorded": false, "risk_level": "high"}, "description": "High-risk Central Configuration Management operations require review.", "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}, "name": "high_risk_requires_review"}, "operation_policy_required": {"condition": {"operation_type": "write", "policy_attached": false}, "description": "Central Configuration Management write operations require policy enforcement.", "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}, "name": "operation_policy_required"}, "tenant_context_required": {"condition": {"tenant_context_present": false}, "description": "Central Configuration Management operations require tenant context.", "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}, "name": "tenant_context_required"}}, "security": {}, "source_files": ["capability_contract.py"], "symbols": {"capability.composition_config": {"file": "capability_contract.py", "id": "capability.composition_config", "kind": "capability", "name": "Central Configuration Management", "range": {"end": {"character": 1, "line": 0}, "start": {"character": 0, "line": 0}}, "references": []}}, "tables": {}, "views": {}}""")


def semantic_model() -> dict[str, Any]:
	"""Return the package semantic model."""
	return json.loads(json.dumps(SEMANTIC_MODEL, sort_keys=True))


def component_manifest() -> dict[str, Any]:
	"""Return the APG component manifest for this capability package."""
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "composition_config",
		"display_name": "Central Configuration Management",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["composition_config"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "composition_config" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "composition_config",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
