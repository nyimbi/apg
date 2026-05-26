"""Shared capability-contract factory for spec-backed APG capabilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_THEME_TOKENS: dict[str, str] = {
	"color.primary": "#28536B",
	"color.accent": "#C44536",
	"color.success": "#2F855A",
	"color.warning": "#B7791F",
	"color.danger": "#C53030",
	"surface.canvas": "#F7F8FA",
	"surface.panel": "#FFFFFF",
	"text.primary": "#172033",
	"text.secondary": "#52606D",
	"border.radius": "8px",
	"density": "compact",
}


def build_spec_capability_contract(
	capability_path: Path,
	tenant_id: str = "default",
	overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Build the executable contract shape for a capability with a cap_spec."""
	metadata = _read_capability_metadata(capability_path)
	capability_id = metadata["capability_id"]
	route_prefix = f"/{capability_id.replace('_', '-')}"
	configuration = {
		"tenant_id": tenant_id,
		"capability": {
			"id": capability_id,
			"name": metadata["display_name"],
			"category": metadata["category"],
			"version": metadata["version"],
			"spec_path": str(capability_path / "cap_spec.md"),
			"enabled": True,
		},
		"execution": {
			"require_tenant_context": True,
			"audit_operations": True,
			"policy_enforced": True,
			"async_supported": True,
		},
		"ui": {
			"enable_dashboard": True,
			"enable_operations": True,
			"enable_rules": True,
			"enable_settings": True,
		},
		"theme": {
			"default_theme": f"{capability_id}_operations",
			"allow_tenant_overrides": True,
		},
	}
	if overrides:
		_deep_merge(configuration, overrides)
	return {
		"capability": capability_id,
		"display_name": metadata["display_name"],
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"rule_engine": {
			"type": "deterministic",
			"rules": _default_rules(metadata["display_name"]),
		},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": f"{route_prefix}/api/v1",
			"routes": [
				{"name": "dashboard", "path": f"{route_prefix}/dashboard", "component": "CapabilityDashboard", "permission": f"{capability_id}:view", "nav_group": "Overview"},
				{"name": "operations", "path": f"{route_prefix}/operations", "component": "CapabilityOperations", "permission": f"{capability_id}:operate", "nav_group": "Operations"},
				{"name": "rules", "path": f"{route_prefix}/rules", "component": "CapabilityRules", "permission": f"{capability_id}:govern", "nav_group": "Governance"},
				{"name": "settings", "path": f"{route_prefix}/settings", "component": "CapabilitySettings", "permission": f"{capability_id}:admin", "nav_group": "Administration"},
			],
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": {
			"name": f"{capability_id}_operations",
			"tokens": deepcopy(DEFAULT_THEME_TOKENS),
			"components": {
				"dashboard": {"icon": "layout-dashboard", "status_indicator": "health-pill", "risk_style": "policy-band"},
				"operations": {"visual": "work-queue", "status_style": "sla-chip"},
				"rules": {"visual": "rule-list", "status_style": "decision-chip"},
				"settings": {"visual": "settings-panel", "density": "compact"},
			},
		},
	}


def evaluate_contract_rules(rules: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate the deterministic contract-rule shape used by generated contracts."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in rules:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _read_capability_metadata(capability_path: Path) -> dict[str, str]:
	spec_path = capability_path / "cap_spec.md"
	text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
	title = _first_markdown_title(text) or capability_path.name.replace("_", " ").replace("-", " ").title()
	return {
		"capability_id": "_".join(capability_path.parts[-2:]).lower().replace("-", "_"),
		"display_name": _metadata_value(text, "Capability Name") or _clean_title(title),
		"category": _metadata_value(text, "Category") or capability_path.parts[-2].replace("_", " ").title(),
		"version": _metadata_value(text, "Version") or "1.0.0",
	}


def _first_markdown_title(text: str) -> str | None:
	for line in text.splitlines():
		if line.startswith("# "):
			return line[2:].strip()
	return None


def _metadata_value(text: str, key: str) -> str | None:
	prefixes = (f"- **{key}**:", f"**{key}:**", f"**{key}**:")
	for line in text.splitlines():
		stripped = line.strip()
		for prefix in prefixes:
			if stripped.startswith(prefix):
				return stripped[len(prefix):].strip().strip("`")
	return None


def _clean_title(title: str) -> str:
	title = title.removeprefix("APG ").strip()
	return title.replace(" - Capability Specification", "").replace(" Capability Specification", "").strip()


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": ["tenant_id", "capability", "execution", "ui", "theme"],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"capability": {"type": "object"},
			"execution": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"},
		},
	}


def _default_rules(display_name: str) -> list[dict[str, Any]]:
	return [
		{"name": "tenant_context_required", "description": f"{display_name} operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
		{"name": "operation_policy_required", "description": f"{display_name} write operations require policy enforcement.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
		{"name": "high_risk_requires_review", "description": f"High-risk {display_name} operations require review.", "condition": {"risk_level": "high", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"}},
	]


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
