"""Deterministic runtime helpers for no-code app composition."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


APP_STATUSES = ("draft", "validated", "published", "retired")
PAGE_LAYOUTS = ("responsive_grid", "form", "dashboard", "wizard", "detail")
COMPONENT_TYPES = ("text", "input", "select", "table", "chart", "button", "form", "metric", "workflow_action")
SOURCE_TYPES = ("entity", "query", "api", "event", "file")


def stable_id(prefix: str, *parts: object) -> str:
	"""Create short deterministic identifiers for repeatable APG tests and demos."""
	seed = "|".join(str(part) for part in parts if part is not None)
	digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def normalize_app_status(status: str | None) -> str:
	value = (status or "draft").strip().lower()
	if value not in APP_STATUSES:
		raise ValueError(f"unsupported_app_status:{value}")
	return value


def normalize_layout(layout: str | None) -> str:
	value = (layout or "responsive_grid").strip().lower()
	if value not in PAGE_LAYOUTS:
		raise ValueError(f"unsupported_page_layout:{value}")
	return value


def normalize_component_type(component_type: str | None) -> str:
	value = (component_type or "text").strip().lower()
	if value not in COMPONENT_TYPES:
		raise ValueError(f"unsupported_component_type:{value}")
	return value


def normalize_source_type(source_type: str | None) -> str:
	value = (source_type or "entity").strip().lower()
	if value not in SOURCE_TYPES:
		raise ValueError(f"unsupported_source_type:{value}")
	return value


def normalize_route(route: str) -> str:
	value = route.strip()
	if not value:
		raise ValueError("route_required")
	return value if value.startswith("/") else f"/{value}"


def bump_patch_version(version: str) -> str:
	parts = version.split(".")
	if len(parts) != 3 or not all(part.isdigit() for part in parts):
		return "0.1.0"
	major, minor, patch = (int(part) for part in parts)
	return f"{major}.{minor}.{patch + 1}"


def component_accessible(component_type: str, label: str, props: dict[str, Any]) -> bool:
	if component_type in {"input", "select", "button", "chart", "table", "workflow_action"}:
		return bool(label.strip() or str(props.get("aria_label", "")).strip())
	return True


def binding_schema_valid(schema: dict[str, Any]) -> bool:
	fields = schema.get("fields")
	return isinstance(fields, list) and all(isinstance(field, str) and field for field in fields)


def validation_checks(
	app: dict[str, Any],
	pages: list[dict[str, Any]],
	components: list[dict[str, Any]],
	bindings: list[dict[str, Any]],
	scripts: list[dict[str, Any]],
	connectors: list[dict[str, Any]],
) -> tuple[dict[str, bool], list[str]]:
	checks = {
		"has_owner": bool(app.get("owner")),
		"has_page": bool(pages),
		"has_component": bool(components),
		"theme_selected": bool(app.get("theme")),
		"accessibility_checked": bool(app.get("accessibility_checked")),
		"rbac_policy_present": bool(app.get("rbac_policy_ref")),
		"data_residency_policy_present": bool(app.get("data_residency_policy_ref")),
		"data_bindings_valid": all(binding.get("validated") for binding in bindings),
		"scripts_policy_attached": all(bool(script.get("policy_ref")) for script in scripts),
		"connectors_policy_attached": all(bool(connector.get("policy_ref")) for connector in connectors),
	}
	issues = [name for name, passed in checks.items() if not passed]
	return checks, issues


def publish_status(target_environment: str, validation_passed: bool) -> str:
	if not validation_passed:
		return "blocked"
	return "production" if target_environment == "production" else "published"
