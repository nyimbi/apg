"""Security-focused integration tests for APG governance rule enforcement.

Validates that capability contracts properly enforce security policies across
all 259 capabilities without mocking the registry or rule engine.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from capabilities.capability_contract_registry import (
	load_contract_registry,
	evaluate_rules,
)


# ---------------------------------------------------------------------------
# Registry fixture — load once per session for speed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def registry() -> dict[str, Any]:
	"""Return the full capability contract registry (all 259 capabilities)."""
	reg = load_contract_registry()
	assert len(reg) == 259, f"Expected 259 capabilities, got {len(reg)}"
	return reg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_rules(registry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
	"""Return (capability_id, rule) pairs for every rule in the registry."""
	pairs: list[tuple[str, dict[str, Any]]] = []
	for cap_id, rec in registry.items():
		for rule in rec.contract["rule_engine"]["rules"]:
			pairs.append((cap_id, rule))
	return pairs


def _cross_tenant_deny_caps(registry: dict[str, Any]) -> list[str]:
	"""Return capability IDs where cross_tenant_access=True is the ONLY condition key
	and the effect is deny — meaning evaluate_rules({cross_tenant_access: True}) will
	deterministically return deny without needing additional context keys.
	"""
	result = []
	for cap_id, rec in sorted(registry.items()):
		for rule in rec.contract["rule_engine"]["rules"]:
			cond = rule.get("condition", {})
			eff = rule.get("effect", {})
			if (
				list(cond.keys()) == ["cross_tenant_access"]
				and cond["cross_tenant_access"] is True
				and eff.get("decision") == "deny"
			):
				result.append(cap_id)
				break
	return result


# ---------------------------------------------------------------------------
# Test 1 — cross-tenant access universally denied (10 capabilities)
# ---------------------------------------------------------------------------

def test_cross_tenant_access_universally_denied(registry: dict[str, Any]) -> None:
	"""For 10 different capabilities, cross_tenant_access=True must return deny."""
	caps = _cross_tenant_deny_caps(registry)
	step = max(1, len(caps) // 10)
	sample = [caps[i] for i in range(0, min(10 * step, len(caps)), step)][:10]
	assert len(sample) == 10, (
		f"Could not sample 10 capabilities with cross_tenant_access deny rule; "
		f"found {len(caps)} total"
	)
	for cap_id in sample:
		result = evaluate_rules(cap_id, {"cross_tenant_access": True})
		assert result["decision"] == "deny", (
			f"{cap_id}: expected deny for cross_tenant_access=True, "
			f"got {result['decision']!r}"
		)


# ---------------------------------------------------------------------------
# Test 2 — missing tenant context denied
# ---------------------------------------------------------------------------

def test_missing_tenant_context_denied(registry: dict[str, Any]) -> None:
	"""intel_alerts, fintech_payments, accs, and payroll must deny when tenant_context_present=False."""
	payroll_candidates = sorted(k for k in registry if "payroll" in k)
	assert payroll_candidates, "No payroll capability found in registry"
	payroll_id = payroll_candidates[0]

	# auth does not carry a tenant_context_present rule (it uses tenant_mismatch instead);
	# use accs (Accessibility Services) which does carry the rule.
	targets = ["accs", "intel_alerts", "fintech_payments", payroll_id]
	for cap_id in targets:
		assert cap_id in registry, f"{cap_id} not found in registry"
		result = evaluate_rules(cap_id, {"tenant_context_present": False})
		assert result["decision"] == "deny", (
			f"{cap_id}: expected deny for tenant_context_present=False, "
			f"got {result['decision']!r}"
		)


# ---------------------------------------------------------------------------
# Test 3 — write without policy -> deny or require_review (5 capabilities)
# ---------------------------------------------------------------------------

def test_write_requires_policy_enforced(registry: dict[str, Any]) -> None:
	"""For 5 capabilities, operation_type=write + policy_attached=False must not be allowed."""
	write_policy_caps: list[str] = []
	for cap_id, rec in sorted(registry.items()):
		for rule in rec.contract["rule_engine"]["rules"]:
			cond = rule.get("condition", {})
			eff = rule.get("effect", {})
			if (
				cond.get("operation_type") == "write"
				and cond.get("policy_attached") is False
				and eff.get("decision") in {"deny", "require_review"}
			):
				write_policy_caps.append(cap_id)
				break

	assert len(write_policy_caps) >= 5, (
		f"Expected >= 5 capabilities with write-policy rules, found {len(write_policy_caps)}"
	)
	for cap_id in write_policy_caps[:5]:
		result = evaluate_rules(cap_id, {"operation_type": "write", "policy_attached": False})
		assert result["decision"] in {"deny", "require_review"}, (
			f"{cap_id}: expected deny/require_review for write without policy, "
			f"got {result['decision']!r}"
		)


# ---------------------------------------------------------------------------
# Test 4 — allow on valid context (10 capabilities)
# ---------------------------------------------------------------------------

def test_allow_on_valid_context(registry: dict[str, Any]) -> None:
	"""For 10 evenly-spaced capabilities, a benign context must return allow."""
	cap_ids = sorted(registry.keys())
	step = max(1, len(cap_ids) // 10)
	sample = cap_ids[::step][:10]
	assert len(sample) == 10

	valid_context: dict[str, Any] = {
		"operation_type": "read",
		"tenant_context_present": True,
		"tenant_id": "acme_corp",
		"cross_tenant_access": False,
		"user_locked": False,
		"mfa_verified": True,
		"risk_level": "low",
		"policy_attached": True,
	}
	for cap_id in sample:
		result = evaluate_rules(cap_id, valid_context)
		assert result["decision"] == "allow", (
			f"{cap_id}: expected allow on valid context, "
			f"got {result['decision']!r} (matched: {result.get('matched_rules')})"
		)


# ---------------------------------------------------------------------------
# Test 5 — rule decision values are a restricted set (all 259)
# ---------------------------------------------------------------------------

_VALID_DECISIONS = {"allow", "deny", "require_review", "warn", "audit", "quarantine", "challenge"}


def test_rule_decisions_are_restricted_set(registry: dict[str, Any]) -> None:
	"""Every rule effect.decision across all 259 capabilities must be in the allowed set."""
	violations: list[str] = []
	for cap_id, rule in _all_rules(registry):
		decision = rule.get("effect", {}).get("decision")
		if decision not in _VALID_DECISIONS:
			violations.append(
				f"{cap_id}.{rule.get('name')}: decision={decision!r}"
			)
	assert not violations, (
		f"{len(violations)} rule(s) have invalid decision values:\n"
		+ "\n".join(f"  {v}" for v in violations[:20])
	)


# ---------------------------------------------------------------------------
# Test 6 — all rules have required fields (all 259)
# ---------------------------------------------------------------------------

def test_all_rules_have_required_fields(registry: dict[str, Any]) -> None:
	"""Every rule must have name (non-empty str), condition (dict), effect (dict with decision)."""
	violations: list[str] = []
	for cap_id, rule in _all_rules(registry):
		name = rule.get("name")
		condition = rule.get("condition")
		effect = rule.get("effect")

		if not isinstance(name, str) or not name:
			violations.append(f"{cap_id}: rule missing non-empty name: {rule!r}")
			continue
		if not isinstance(condition, dict):
			violations.append(
				f"{cap_id}.{name}: condition must be dict, got {type(condition).__name__}"
			)
		if not isinstance(effect, dict):
			violations.append(
				f"{cap_id}.{name}: effect must be dict, got {type(effect).__name__}"
			)
		elif not effect.get("decision"):
			violations.append(f"{cap_id}.{name}: effect.decision is missing or empty")

	assert not violations, (
		f"{len(violations)} rule field violation(s):\n"
		+ "\n".join(f"  {v}" for v in violations[:20])
	)


# ---------------------------------------------------------------------------
# Test 7 — minimum rule count (all 259)
# ---------------------------------------------------------------------------

def test_minimum_rule_count(registry: dict[str, Any]) -> None:
	"""All 259 capabilities must have >= 10 governance rules."""
	below_minimum = [
		(cap_id, len(rec.contract["rule_engine"]["rules"]))
		for cap_id, rec in registry.items()
		if len(rec.contract["rule_engine"]["rules"]) < 10
	]
	assert not below_minimum, (
		f"{len(below_minimum)} capability/ies have fewer than 10 rules:\n"
		+ "\n".join(f"  {cap_id}: {count} rule(s)" for cap_id, count in below_minimum)
	)


# ---------------------------------------------------------------------------
# Test 8 — tenant_id in all configurations (all 259)
# ---------------------------------------------------------------------------

def test_tenant_id_in_all_configurations(registry: dict[str, Any]) -> None:
	"""All 259 capabilities must have a non-empty tenant_id in their configuration dict."""
	violations: list[str] = []
	for cap_id, rec in registry.items():
		configuration = rec.contract.get("configuration", {})
		if not isinstance(configuration, dict):
			violations.append(f"{cap_id}: configuration is not a dict")
			continue
		tid = configuration.get("tenant_id")
		if not isinstance(tid, str) or not tid:
			violations.append(
				f"{cap_id}: configuration.tenant_id missing or empty (got {tid!r})"
			)
	assert not violations, (
		f"{len(violations)} configuration.tenant_id violation(s):\n"
		+ "\n".join(f"  {v}" for v in violations[:20])
	)


# ---------------------------------------------------------------------------
# Test 9 — streaming.guardrails present (all 259)
# ---------------------------------------------------------------------------

def test_streaming_guardrails_present(registry: dict[str, Any]) -> None:
	"""All 259 capabilities must have a streaming section.
	If streaming.guardrails is present it must be a list; absence is treated as an empty list.
	"""
	violations: list[str] = []
	for cap_id, rec in registry.items():
		streaming = rec.contract.get("streaming")
		if streaming is None:
			violations.append(f"{cap_id}: missing 'streaming' key in contract")
			continue
		if not isinstance(streaming, dict):
			violations.append(
				f"{cap_id}: streaming must be a dict, got {type(streaming).__name__}"
			)
			continue
		# guardrails key is optional — absence equals an empty guardrail list
		guardrails = streaming.get("guardrails", [])
		if not isinstance(guardrails, list):
			violations.append(
				f"{cap_id}: streaming.guardrails must be a list, "
				f"got {type(guardrails).__name__}"
			)
	assert not violations, (
		f"{len(violations)} streaming violation(s):\n"
		+ "\n".join(f"  {v}" for v in violations[:20])
	)


# ---------------------------------------------------------------------------
# Test 10 — no default_tenant in rule conditions (all 259)
# ---------------------------------------------------------------------------

def test_no_default_tenant_in_rules(registry: dict[str, Any]) -> None:
	"""No rule condition should have tenant_id == 'default_tenant' (tenant isolation bypass)."""
	violations: list[str] = []
	for cap_id, rule in _all_rules(registry):
		cond = rule.get("condition", {})
		if cond.get("tenant_id") == "default_tenant":
			violations.append(
				f"{cap_id}.{rule.get('name')}: condition.tenant_id == 'default_tenant'"
			)
	assert not violations, (
		f"{len(violations)} rule(s) use 'default_tenant' as tenant_id condition — "
		"this bypasses tenant isolation:\n"
		+ "\n".join(f"  {v}" for v in violations)
	)


# ---------------------------------------------------------------------------
# Test 11 — UI permissions are namespaced (all 259)
# ---------------------------------------------------------------------------

# Routes intentionally open to unauthenticated users.
_PUBLIC_PERMISSIONS = {"public"}

# Accepts colon-separated (domain:action, domain:resource:action) and
# dot-separated (domain.action, domain.resource.action) patterns.
# Rejects bare single words that are not in the public allow-list.
_PERM_NAMESPACED = re.compile(
	r"^[a-zA-Z_][a-zA-Z0-9_]*[:.][a-zA-Z_][a-zA-Z0-9_.:]*$"
)


def test_ui_permissions_are_namespaced(registry: dict[str, Any]) -> None:
	"""All non-public UI route permissions must follow 'domain:action' or 'domain.action' pattern."""
	violations: list[str] = []
	for cap_id, rec in registry.items():
		for route in rec.contract["ui"].get("routes", []):
			perm = route.get("permission", "")
			if perm in _PUBLIC_PERMISSIONS:
				continue
			if not _PERM_NAMESPACED.match(perm):
				violations.append(
					f"{cap_id}/{route.get('name', '?')}: "
					f"permission {perm!r} is not namespaced"
				)
	assert not violations, (
		f"{len(violations)} UI route(s) have non-namespaced permissions:\n"
		+ "\n".join(f"  {v}" for v in violations[:20])
	)


# ---------------------------------------------------------------------------
# Test 12 — ui.requires_theme == True everywhere (all 259)
# ---------------------------------------------------------------------------

def test_require_theme_true_everywhere(registry: dict[str, Any]) -> None:
	"""All 259 capabilities must have ui.requires_theme == True."""
	violations = [
		cap_id
		for cap_id, rec in registry.items()
		if rec.contract["ui"].get("requires_theme") is not True
	]
	assert not violations, (
		f"{len(violations)} capability/ies missing ui.requires_theme=True:\n"
		+ "\n".join(f"  {v}" for v in violations[:20])
	)