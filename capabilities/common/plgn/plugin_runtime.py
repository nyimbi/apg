"""Deterministic runtime helpers for plugin governance."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


RELEASE_CHANNELS = ("stable", "beta", "dev", "private")
INSTALL_POLICIES = ("tenant_allowed", "admin_only", "blocked")
SENSITIVE_SCOPES = ("secrets", "payments", "identity", "filesystem:write", "network:external")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part) for part in parts if part is not None)
	digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def normalize_channel(channel: str | None) -> str:
	value = (channel or "stable").strip().lower()
	if value not in RELEASE_CHANNELS:
		raise ValueError(f"unsupported_release_channel:{value}")
	return value


def normalize_install_policy(policy: str | None) -> str:
	value = (policy or "tenant_allowed").strip().lower()
	if value not in INSTALL_POLICIES:
		raise ValueError(f"unsupported_install_policy:{value}")
	return value


def normalize_scopes(scopes: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
	return tuple(sorted({str(scope).strip().lower() for scope in scopes or () if str(scope).strip()}))


def sensitive_scopes(scopes: tuple[str, ...]) -> tuple[str, ...]:
	return tuple(scope for scope in scopes if scope in SENSITIVE_SCOPES)


def manifest_ready(plugin: dict[str, Any]) -> tuple[bool, list[str]]:
	issues: list[str] = []
	if not plugin["owner"]:
		issues.append("plugin_owner_required")
	if not plugin["manifest_schema_valid"]:
		issues.append("manifest_schema_required")
	if not plugin["signature_verified"]:
		issues.append("plugin_signature_required")
	if not plugin["dependency_validation_passed"]:
		issues.append("dependency_validation_required")
	if not plugin["supply_chain_scan_passed"]:
		issues.append("supply_chain_scan_required")
	if plugin["external_plugin"] and not plugin["external_review_recorded"]:
		issues.append("external_plugin_review_required")
	return not issues, issues


def release_readiness(
	plugin: dict[str, Any],
	permission_review_recorded: bool,
	sandbox_policy_attached: bool,
	listing_ready: bool,
) -> tuple[str, list[str]]:
	_ready, issues = manifest_ready(plugin)
	if plugin["permissions"] and not permission_review_recorded:
		issues.append("permission_review_required")
	if not sandbox_policy_attached:
		issues.append("plugin_sandbox_required")
	if not listing_ready:
		issues.append("marketplace_listing_required")
	return ("ready" if not issues else "blocked"), issues
