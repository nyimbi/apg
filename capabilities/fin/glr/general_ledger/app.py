"""Publishable APG capability package entrypoint for Financial Management General Ledger."""

from __future__ import annotations

import json
from typing import Any


SEMANTIC_MODEL: dict[str, Any] = json.loads(r"""{"agents": {}, "app": {"description": "Financial Management General Ledger package-backed APG capability", "entity_count": 0, "name": "glr_general_ledger", "version": "1.0.0"}, "capabilities": {"glr_general_ledger": {"approvals": {}, "business_rules": [], "components": {}, "configuration": {"capability": {"category": "Glr", "enabled": true, "id": "glr_general_ledger", "name": "Financial Management General Ledger", "spec_path": "/Users/nyimbiodero/src/pjs/apg/capabilities/fin/glr/general_ledger/cap_spec.md", "version": "1.0.0"}, "execution": {"async_supported": true, "audit_operations": true, "policy_enforced": true, "require_tenant_context": true}, "tenant_id": "default", "theme": {"allow_tenant_overrides": true, "default_theme": "glr_general_ledger_operations"}, "ui": {"enable_dashboard": true, "enable_operations": true, "enable_rules": true, "enable_settings": true}}, "erp_modules": ["fin"], "i18n": {}, "master_data": {}, "name": "Financial Management General Ledger", "provides": ["glr_general_ledger_operations"], "requires": [], "rule_engine": {"rules": [{"condition": {"tenant_context_present": false}, "description": "Financial Management General Ledger operations require tenant context.", "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}, "name": "tenant_context_required"}, {"condition": {"operation_type": "write", "policy_attached": false}, "description": "Financial Management General Ledger write operations require policy enforcement.", "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}, "name": "operation_policy_required"}, {"condition": {"review_recorded": false, "risk_level": "high"}, "description": "High-risk Financial Management General Ledger operations require review.", "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}, "name": "high_risk_requires_review"}], "type": "deterministic"}, "rules": [{"condition": {"tenant_context_present": false}, "description": "Financial Management General Ledger operations require tenant context.", "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}, "name": "tenant_context_required"}, {"condition": {"operation_type": "write", "policy_attached": false}, "description": "Financial Management General Ledger write operations require policy enforcement.", "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}, "name": "operation_policy_required"}, {"condition": {"review_recorded": false, "risk_level": "high"}, "description": "High-risk Financial Management General Ledger operations require review.", "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}, "name": "high_risk_requires_review"}], "runtime": {"api": "api.py", "entrypoint": "app.py", "service": "service.py", "views": "views.py"}, "screens": {"dashboard": {"component": "CapabilityDashboard", "permission": "glr_general_ledger:view", "route": "/glr-general-ledger/dashboard"}, "operations": {"component": "CapabilityOperations", "permission": "glr_general_ledger:operate", "route": "/glr-general-ledger/operations"}, "rules": {"component": "CapabilityRules", "permission": "glr_general_ledger:govern", "route": "/glr-general-ledger/rules"}, "settings": {"component": "CapabilitySettings", "permission": "glr_general_ledger:admin", "route": "/glr-general-ledger/settings"}}, "streaming": {}, "theme": {"components": {"dashboard": {"icon": "layout-dashboard", "risk_style": "policy-band", "status_indicator": "health-pill"}, "operations": {"status_style": "sla-chip", "visual": "work-queue"}, "rules": {"status_style": "decision-chip", "visual": "rule-list"}, "settings": {"density": "compact", "visual": "settings-panel"}}, "name": "glr_general_ledger_operations", "tokens": {"border.radius": "8px", "color.accent": "#C44536", "color.danger": "#C53030", "color.primary": "#28536B", "color.success": "#2F855A", "color.warning": "#B7791F", "density": "compact", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D"}}, "ui": {"api_prefix": "/glr-general-ledger/api/v1", "requires_theme": true, "routes": [{"component": "CapabilityDashboard", "name": "dashboard", "nav_group": "Overview", "path": "/glr-general-ledger/dashboard", "permission": "glr_general_ledger:view"}, {"component": "CapabilityOperations", "name": "operations", "nav_group": "Operations", "path": "/glr-general-ledger/operations", "permission": "glr_general_ledger:operate"}, {"component": "CapabilityRules", "name": "rules", "nav_group": "Governance", "path": "/glr-general-ledger/rules", "permission": "glr_general_ledger:govern"}, {"component": "CapabilitySettings", "name": "settings", "nav_group": "Administration", "path": "/glr-general-ledger/settings", "permission": "glr_general_ledger:admin"}], "shell": "apg_python", "template_roots": ["templates/", "static/"], "view_module": "views.py"}}}, "composition": {"agent_teams": {}, "applications": {}, "capability_dependencies": {"glr_general_ledger": []}}, "contracts": {"glr_general_ledger": {"configuration": {"capability": {"category": "Glr", "enabled": true, "id": "glr_general_ledger", "name": "Financial Management General Ledger", "spec_path": "/Users/nyimbiodero/src/pjs/apg/capabilities/fin/glr/general_ledger/cap_spec.md", "version": "1.0.0"}, "execution": {"async_supported": true, "audit_operations": true, "policy_enforced": true, "require_tenant_context": true}, "tenant_id": "default", "theme": {"allow_tenant_overrides": true, "default_theme": "glr_general_ledger_operations"}, "ui": {"enable_dashboard": true, "enable_operations": true, "enable_rules": true, "enable_settings": true}}, "id": "glr_general_ledger", "provides": ["glr_general_ledger_operations"], "requires": []}}, "deployment": {"source": "capability_contract.py", "target": "python"}, "diagnostics": [], "flows": {}, "format": "apg.semantic-model.v1", "graphs": {"capability": {"edges": 0, "kind": "capability", "nodes": 1}, "package": {"edges": 1, "kind": "package", "nodes": 2}}, "llms": {}, "ok": true, "operations": {}, "packages": {"glr_general_ledger": {"entrypoint": "app.py", "profile": "capability"}}, "roles": {}, "rules": {"high_risk_requires_review": {"condition": {"review_recorded": false, "risk_level": "high"}, "description": "High-risk Financial Management General Ledger operations require review.", "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}, "name": "high_risk_requires_review"}, "operation_policy_required": {"condition": {"operation_type": "write", "policy_attached": false}, "description": "Financial Management General Ledger write operations require policy enforcement.", "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}, "name": "operation_policy_required"}, "tenant_context_required": {"condition": {"tenant_context_present": false}, "description": "Financial Management General Ledger operations require tenant context.", "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}, "name": "tenant_context_required"}}, "security": {}, "source_files": ["capability_contract.py"], "symbols": {"capability.glr_general_ledger": {"file": "capability_contract.py", "id": "capability.glr_general_ledger", "kind": "capability", "name": "Financial Management General Ledger", "range": {"end": {"character": 1, "line": 0}, "start": {"character": 0, "line": 0}}, "references": []}}, "tables": {}, "views": {}}""")


def semantic_model() -> dict[str, Any]:
	"""Return the package semantic model."""
	return json.loads(json.dumps(SEMANTIC_MODEL, sort_keys=True))


def component_manifest() -> dict[str, Any]:
	"""Return the APG component manifest for this capability package."""
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "glr_general_ledger",
		"display_name": "Financial Management General Ledger",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["glr_general_ledger"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "glr_general_ledger" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "glr_general_ledger",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
