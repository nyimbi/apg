"""Publishable APG capability package entrypoint for Connection Management."""

from __future__ import annotations

import json
from typing import Any


SEMANTIC_MODEL: dict[str, Any] = json.loads(r"""{"agents": {}, "app": {"description": "Connection Management package-backed APG capability", "entity_count": 0, "name": "conn", "version": "1.0.0"}, "capabilities": {"conn": {"approvals": {}, "business_rules": [], "components": {}, "configuration": {"ai": {"enabled": true, "model": "qwen3:1.7b", "schema_mapping_confidence_threshold": 0.75}, "security": {"audit_enabled": true, "encrypt_credentials": true, "require_connection_test_before_activation": true}, "singer": {"default_batch_size": 1000, "health_check_interval_seconds": 60, "max_batch_size": 100000, "sync_mode": "incremental"}, "tenant_id": "default", "theme": {"allow_tenant_overrides": true, "default_theme": "conn_enterprise"}, "ui": {"enable_data_quality_view": true, "enable_lineage_view": true, "enable_marketplace": true, "enable_visual_designer": true}}, "erp_modules": ["common"], "i18n": {}, "master_data": {}, "name": "Connection Management", "provides": ["conn_operations"], "requires": [], "rule_engine": {"rules": [{"condition": {"last_test_passed": false, "requested_status": "active"}, "description": "Connections must pass a test before activation.", "effect": {"decision": "deny", "reason": "connection_test_required", "required_action": "run_connection_test"}, "name": "require_connection_test_before_activation"}, {"condition": {"contains_credentials": true, "credentials_encrypted": false}, "description": "Credential-bearing connectors require encrypted storage.", "effect": {"decision": "deny", "reason": "credentials_must_be_encrypted", "required_action": "enable_encryption"}, "name": "encrypt_credentials"}, {"condition": {"batch_size_gt": 10000, "monitoring_enabled": false}, "description": "Large synchronization batches require monitoring.", "effect": {"decision": "deny", "reason": "large_batch_requires_monitoring", "required_action": "enable_monitoring"}, "name": "large_batch_requires_monitoring"}], "type": "deterministic"}, "rules": [{"condition": {"last_test_passed": false, "requested_status": "active"}, "description": "Connections must pass a test before activation.", "effect": {"decision": "deny", "reason": "connection_test_required", "required_action": "run_connection_test"}, "name": "require_connection_test_before_activation"}, {"condition": {"contains_credentials": true, "credentials_encrypted": false}, "description": "Credential-bearing connectors require encrypted storage.", "effect": {"decision": "deny", "reason": "credentials_must_be_encrypted", "required_action": "enable_encryption"}, "name": "encrypt_credentials"}, {"condition": {"batch_size_gt": 10000, "monitoring_enabled": false}, "description": "Large synchronization batches require monitoring.", "effect": {"decision": "deny", "reason": "large_batch_requires_monitoring", "required_action": "enable_monitoring"}, "name": "large_batch_requires_monitoring"}], "runtime": {"api": "api.py", "entrypoint": "app.py", "service": "service.py", "views": "views.py"}, "screens": {"dashboard": {"component": "ConnectionDashboard", "permission": "conn:view", "route": "/conn/dashboard"}, "data_quality": {"component": "DataQualityWorkbench", "permission": "conn:view", "route": "/conn/data-quality"}, "designer": {"component": "VisualFlowDesigner", "permission": "conn:create", "route": "/conn/designer"}, "lineage": {"component": "DataLineageView", "permission": "conn:view", "route": "/conn/lineage"}, "marketplace": {"component": "ConnectorMarketplace", "permission": "conn:view", "route": "/conn/marketplace"}, "rules": {"component": "CapabilityRuleWorkbench", "permission": "conn:admin", "route": "/conn/rules"}, "settings": {"component": "CapabilitySettings", "permission": "conn:admin", "route": "/conn/settings"}}, "streaming": {}, "theme": {"components": {"connection_node": {"icon": "plug", "shape": "rounded-rectangle", "status_indicator": "left-border"}, "data_flow_edge": {"animated_when_active": "true", "line_style": "solid"}, "rule_badge": {"icon": "shield-check", "variant": "subtle"}}, "name": "conn_enterprise", "tokens": {"border.radius": "8px", "color.accent": "#8B5E34", "color.danger": "#B42318", "color.primary": "#176B87", "color.success": "#2E7D32", "color.warning": "#B26A00", "density": "compact", "surface.canvas": "#F7F9FB", "surface.panel": "#FFFFFF", "text.primary": "#1F2933", "text.secondary": "#52616B"}}, "ui": {"frontend_bundle": "frontend/src/App.tsx", "requires_theme": true, "routes": [{"component": "ConnectionDashboard", "name": "dashboard", "nav_group": "Operations", "path": "/conn/dashboard", "permission": "conn:view"}, {"component": "VisualFlowDesigner", "name": "designer", "nav_group": "Build", "path": "/conn/designer", "permission": "conn:create"}, {"component": "ConnectorMarketplace", "name": "marketplace", "nav_group": "Extend", "path": "/conn/marketplace", "permission": "conn:view"}, {"component": "DataLineageView", "name": "lineage", "nav_group": "Governance", "path": "/conn/lineage", "permission": "conn:view"}, {"component": "DataQualityWorkbench", "name": "data_quality", "nav_group": "Governance", "path": "/conn/data-quality", "permission": "conn:view"}, {"component": "CapabilityRuleWorkbench", "name": "rules", "nav_group": "Governance", "path": "/conn/rules", "permission": "conn:admin"}, {"component": "CapabilitySettings", "name": "settings", "nav_group": "Administration", "path": "/conn/settings", "permission": "conn:admin"}], "shell": "apg_python", "template_roots": ["templates/", "frontend/"]}}}, "composition": {"agent_teams": {}, "applications": {}, "capability_dependencies": {"conn": []}}, "contracts": {"conn": {"configuration": {"ai": {"enabled": true, "model": "qwen3:1.7b", "schema_mapping_confidence_threshold": 0.75}, "security": {"audit_enabled": true, "encrypt_credentials": true, "require_connection_test_before_activation": true}, "singer": {"default_batch_size": 1000, "health_check_interval_seconds": 60, "max_batch_size": 100000, "sync_mode": "incremental"}, "tenant_id": "default", "theme": {"allow_tenant_overrides": true, "default_theme": "conn_enterprise"}, "ui": {"enable_data_quality_view": true, "enable_lineage_view": true, "enable_marketplace": true, "enable_visual_designer": true}}, "id": "conn", "provides": ["conn_operations"], "requires": []}}, "deployment": {"source": "capability_contract.py", "target": "python"}, "diagnostics": [], "flows": {}, "format": "apg.semantic-model.v1", "graphs": {"capability": {"edges": 0, "kind": "capability", "nodes": 1}, "package": {"edges": 1, "kind": "package", "nodes": 2}}, "llms": {}, "ok": true, "operations": {}, "packages": {"conn": {"entrypoint": "app.py", "profile": "capability"}}, "roles": {}, "rules": {"encrypt_credentials": {"condition": {"contains_credentials": true, "credentials_encrypted": false}, "description": "Credential-bearing connectors require encrypted storage.", "effect": {"decision": "deny", "reason": "credentials_must_be_encrypted", "required_action": "enable_encryption"}, "name": "encrypt_credentials"}, "large_batch_requires_monitoring": {"condition": {"batch_size_gt": 10000, "monitoring_enabled": false}, "description": "Large synchronization batches require monitoring.", "effect": {"decision": "deny", "reason": "large_batch_requires_monitoring", "required_action": "enable_monitoring"}, "name": "large_batch_requires_monitoring"}, "require_connection_test_before_activation": {"condition": {"last_test_passed": false, "requested_status": "active"}, "description": "Connections must pass a test before activation.", "effect": {"decision": "deny", "reason": "connection_test_required", "required_action": "run_connection_test"}, "name": "require_connection_test_before_activation"}}, "security": {}, "source_files": ["capability_contract.py"], "symbols": {"capability.conn": {"file": "capability_contract.py", "id": "capability.conn", "kind": "capability", "name": "Connection Management", "range": {"end": {"character": 1, "line": 0}, "start": {"character": 0, "line": 0}}, "references": []}}, "tables": {}, "views": {}}""")


def semantic_model() -> dict[str, Any]:
	"""Return the package semantic model."""
	return json.loads(json.dumps(SEMANTIC_MODEL, sort_keys=True))


def component_manifest() -> dict[str, Any]:
	"""Return the APG component manifest for this capability package."""
	return {
		"format": "apg.component-manifest.v1",
		"kind": "apg.generated_application",
		"name": "conn",
		"display_name": "Connection Management",
		"target": "python",
		"interfaces": {
			"health": "/health",
			"self_test": "/self-test",
			"semantic_model": "/semantic-model.json",
		},
		"capabilities": ["conn"],
	}


def self_test() -> dict[str, Any]:
	"""Run a dependency-light package self-test."""
	model = semantic_model()
	manifest = component_manifest()
	errors: list[str] = []
	if model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic model format mismatch")
	if "conn" not in model.get("capabilities", {}):
		errors.append("capability missing from semantic model")
	if manifest.get("interfaces", {}).get("semantic_model") != "/semantic-model.json":
		errors.append("component manifest semantic model interface mismatch")
	return {
		"passed": not errors,
		"status": "ok" if not errors else "failed",
		"errors": errors,
		"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
		"capability": "conn",
	}


if __name__ == "__main__":
	print(json.dumps(self_test(), indent=2, sort_keys=True))
