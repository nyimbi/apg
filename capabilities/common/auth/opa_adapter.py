"""OPA (Open Policy Agent) adapter for APG authorization decisions.

Replaces the in-memory RBAC engine in evaluate_capability_rules() with
policy-as-code evaluation via the OPA REST API.

The output shape is identical to the existing engine:
  {"decision": "allow"|"deny"|"require_review",
   "matched_rules": [...], "actions": [...]}

Activated when OPA_URL env var is set. The capability_contract.py's
evaluate_capability_rules() function routes to this adapter when available.

OPA policy location: policies/apg/authz.rego (mounted at /policies in docker)

Usage::

    from capabilities.common.auth.opa_adapter import evaluate_with_opa
    result = await evaluate_with_opa(context)
    # result == {"decision": "allow", "matched_rules": [...], "actions": [...]}
"""
from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger(__name__)

# OPA data path — matches the Rego package apg.authz
_OPA_POLICY_PATH = "apg/authz"


async def evaluate_with_opa(context: dict[str, Any]) -> dict[str, Any] | None:
	"""Evaluate an authorization context against OPA.

	Returns the decision dict if OPA is configured and reachable,
	returns None to signal the caller should use the built-in engine.

	Args:
		context: The same context dict passed to evaluate_capability_rules().
		         Expected keys: user, action, resource, resource_type, tenant_id, rules, ...

	Returns:
		{"decision": str, "matched_rules": list, "actions": list} or None
	"""
	opa_url = os.environ.get("OPA_URL")
	if not opa_url:
		return None

	try:
		import httpx

		url = f"{opa_url.rstrip('/')}/v1/data/{_OPA_POLICY_PATH}"
		async with httpx.AsyncClient(timeout=2.0) as client:
			response = await client.post(url, json={"input": context})
			response.raise_for_status()

		result = response.json().get("result", {})

		decision = result.get("decision", "deny")
		if not isinstance(decision, str):
			decision = "allow" if result.get("allow") else "deny"

		return {
			"decision": decision,
			"matched_rules": result.get("matched_rules", []),
			"actions": result.get("actions", []),
		}

	except Exception as exc:
		# OPA timeout or unavailable — fail open with a warning and deny.
		# Failing closed (deny all) is the safe default for auth systems.
		_log.warning(
			"OPA evaluation failed (%s) — falling back to built-in engine", exc
		)
		return None


def build_opa_context(
	user: dict[str, Any],
	action: str,
	resource: str,
	resource_type: str = "",
	tenant_id: str = "",
	capability_id: str = "",
	extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Build a well-typed OPA input context from caller parameters.

	Standardizes the shape so all capabilities produce consistent OPA inputs.
	"""
	return {
		"user": {
			"id": user.get("user_id") or user.get("id", ""),
			"tenant_id": user.get("tenant_id", tenant_id),
			"roles": user.get("roles", []),
		},
		"action": action,
		"resource": resource,
		"resource_type": resource_type,
		"capability_id": capability_id,
		"context": {
			"tenant_id": tenant_id,
			**(extra or {}),
		},
	}
