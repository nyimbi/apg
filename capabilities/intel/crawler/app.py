"""Publishable APG capability package entrypoint for Crawler Capability - Revolutionary Enterprise Web Intelligence."""

from __future__ import annotations

import json
from typing import Any


SEMANTIC_MODEL: dict[str, Any] = json.loads(r"""{"agents": {}, "app": {"description": "Crawler Capability - Revolutionary Enterprise Web Intelligence package-backed APG capability", "entity_count": 0, "name": "intel_crawler", "version": "2.0.0"}, "capabilities": {"intel_crawler": {"approvals": {}, "business_rules": [], "components": {}, "configuration": {"capability": {"category": "common", "enabled": true, "id": "intel_crawler", "name": "Crawler Capability - Revolutionary Enterprise Web Intelligence", "spec_path": "/Users/nyimbiodero/src/pjs/apg/capabilities/intel/crawler/cap_spec.md", "version": "2.0.0"}, "execution": {"async_supported": true, "audit_operations": true, "policy_enforced": true, "require_tenant_context": true}, "tenant_id": "default", "theme": {"allow_tenant_overrides": true, "default_theme": "intel_crawler_operations"}, "ui": {"enable_dashboard": true, "enable_operations": true, "enable_rules": true, "enable_settings": true}}, "erp_modules": ["intel"], "i18n": {}, "master_data": {}, "name": "Crawler Capability - Revolutionary Enterprise Web Intelligence", "provides": ["intel_crawler_operations"], "requires": [], "rule_engine": {"rules": [{"condition": {"tenant_context_present": false}, "description": "Crawler Capability - Revolutionary Enterprise Web Intelligence operations require tenant context.", "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}, "name": "tenant_context_required"}, {"condition": {"operation_type": "write", "policy_attached": false}, "description": "Crawler Capability - Revolutionary Enterprise Web Intelligence write operations require policy enforcement.", "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}, "name": "operation_policy_required"}, {"condition": {"review_recorded": false, "risk_level": "high"}, "description": "High-risk Crawler Capability - Revolutionary Enterprise Web Intelligence operations require review.", "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}, "name": "high_risk_requires_review"}], "type": "deterministic"}, "rules": [{"condition": {"tenant_context_present": false}, "description": "Crawler Capability - Revolutionary Enterprise Web Intelligence operations require tenant context.", "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}, "name": "tenant_context_required"}, {"condition": {"operation_type": "write", "policy_attached": false}, "description": "Crawler Capability - Revolutionary Enterprise Web Intelligence write operations require policy enforcement.", "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}, "name": "operation_policy_required"}, {"condition": {"review_recorded": false, "risk_level": "high"}, "description": "High-risk Crawler Capability - Revolutionary Enterprise Web Intelligence operations require review.", "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}, "name": "high_risk_requires_review"}], "runtime": {"api": "api.py", "entrypoint": "app.py", "service": "service.py", "views": "views.py"}, "screens": {"dashboard": {"component": "CapabilityDashboard", "permission": "intel_crawler:view", "route": "/intel-crawler/dashboard"}, "operations": {"component": "CapabilityOperations", "permission": "intel_crawler:operate", "route": "/intel-crawler/operations"}, "rules": {"component": "CapabilityRules", "permission": "intel_crawler:govern", "route": "/intel-crawler/rules"}, "settings": {"component": "CapabilitySettings", "permission": "intel_crawler:admin", "route": "/intel-crawler/settings"}}, "streaming": {}, "theme": {"components": {"dashboard": {"icon": "layout-dashboard", "risk_style": "policy-band", "status_indicator": "health-pill"}, "operations": {"status_style": "sla-chip", "visual": "work-queue"}, "rules": {"status_style": "decision-chip", "visual": "rule-list"}, "settings": {"density": "compact", "visual": "settings-panel"}}, "name": "intel_crawler_operations", "tokens": {"border.radius": "8px", "color.accent": "#C44536", "color.danger": "#C53030", "color.primary": "#28536B", "color.success": "#2F855A", "color.warning": "#B7791F", "density": "compact", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D"}}, "ui": {"api_prefix": "/intel-crawler/api/v1", "requires_theme": true, "routes": [{"component": "CapabilityDashboard", "name": "dashboard", "nav_group": "Overview", "path": "/intel-crawler/dashboard", "permission": "intel_crawler:view"}, {"component": "CapabilityOperations", "name": "operations", "nav_group": "Operations", "path": "/intel-crawler/operations", "permission": "intel_crawler:operate"}, {"component": "CapabilityRules", "name": "rules", "nav_group": "Governance", "path": "/intel-crawler/rules", "permission": "intel_crawler:govern"}, {"component": "CapabilitySettings", "name": "settings", "nav_group": "Administration", "path": "/intel-crawler/settings", "permission": "intel_crawler:admin"}], "shell": "apg_python", "template_roots": ["templates/", "static/"], "view_module": "views.py"}}}, "composition": {"agent_teams": {}, "applications": {}, "capability_dependencies": {"intel_crawler": []}}, "contracts": {"intel_crawler": {"configuration": {"capability": {"category": "common", "enabled": true, "id": "intel_crawler", "name": "Crawler Capability - Revolutionary Enterprise Web Intelligence", "spec_path": "/Users/nyimbiodero/src/pjs/apg/capabilities/intel/crawler/cap_spec.md", "version": "2.0.0"}, "execution": {"async_supported": true, "audit_operations": true, "policy_enforced": true, "require_tenant_context": true}, "tenant_id": "default", "theme": {"allow_tenant_overrides": true, "default_theme": "intel_crawler_operations"}, "ui": {"enable_dashboard": true, "enable_operations": true, "enable_rules": true, "enable_settings": true}}, "id": "intel_crawler", "provides": ["intel_crawler_operations"], "requires": []}}, "deployment": {"source": "capability_contract.py", "target": "python"}, "diagnostics": [], "flows": {}, "format": "apg.semantic-model.v1", "graphs": {"capability": {"edges": 0, "kind": "capability", "nodes": 1}, "package": {"edges": 1, "kind": "package", "nodes": 2}}, "llms": {}, "ok": true, "operations": {}, "packages": {"intel_crawler": {"entrypoint": "app.py", "profile": "capability"}}, "roles": {}, "rules": {"high_risk_requires_review": {"condition": {"review_recorded": false, "risk_level": "high"}, "description": "High-risk Crawler Capability - Revolutionary Enterprise Web Intelligence operations require review.", "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}, "name": "high_risk_requires_review"}, "operation_policy_required": {"condition": {"operation_type": "write", "policy_attached": false}, "description": "Crawler Capability - Revolutionary Enterprise Web Intelligence write operations require policy enforcement.", "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}, "name": "operation_policy_required"}, "tenant_context_required": {"condition": {"tenant_context_present": false}, "description": "Crawler Capability - Revolutionary Enterprise Web Intelligence operations require tenant context.", "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}, "name": "tenant_context_required"}}, "security": {}, "source_files": ["capability_contract.py"], "symbols": {"capability.intel_crawler": {"file": "capability_contract.py", "id": "capability.intel_crawler", "kind": "capability", "name": "Crawler Capability - Revolutionary Enterprise Web Intelligence", "range": {"end": {"character": 1, "line": 0}, "start": {"character": 0, "line": 0}}, "references": []}}, "tables": {}, "views": {}}""")


def semantic_model() -> dict[str, Any]:
	"""Return the package semantic model."""
	return json.loads(json.dumps(SEMANTIC_MODEL, sort_keys=True))


def component_manifest() -> dict[str, Any]:
	"""Return the APG component manifest for this capability package."""
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "intel_crawler",
		"display_name": "Crawler Capability - Revolutionary Enterprise Web Intelligence",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["intel_crawler"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "intel_crawler" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "intel_crawler",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
