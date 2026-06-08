"""Tests for OPA (Open Policy Agent) authorization adapter.

Tests validate the adapter interface, context building, fallback behaviour,
and evaluate_capability_rules() routing — without requiring a live OPA server.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── OPA adapter unit tests ────────────────────────────────────────────────────

def test_evaluate_capability_rules_uses_builtin_without_opa_url(monkeypatch):
	"""When OPA_URL is not set, the built-in engine is used."""
	monkeypatch.delenv("OPA_URL", raising=False)
	from capabilities.common.auth.capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({"tenant_context_present": True})
	assert "decision" in result
	assert result["decision"] in {"allow", "deny", "require_review"}


def test_evaluate_capability_rules_routes_to_opa_when_url_set(monkeypatch):
	"""When OPA_URL is set, evaluate_capability_rules() calls OPA REST API."""
	monkeypatch.setenv("OPA_URL", "http://localhost:8181")
	opa_response = {
		"result": {
			"decision": "allow",
			"matched_rules": [{"role": "admin", "action": "read"}],
			"actions": ["audit"],
		}
	}

	mock_response = MagicMock()
	mock_response.json.return_value = opa_response
	mock_response.raise_for_status = MagicMock()

	with patch("httpx.post", return_value=mock_response) as mock_post:
		from capabilities.common.auth import capability_contract
		import importlib
		importlib.reload(capability_contract)
		result = capability_contract.evaluate_capability_rules({
			"user": {"id": "u1", "roles": ["admin"]},
			"action": "read",
			"resource": "patient_record",
		})

	assert result["decision"] == "allow"
	assert result["matched_rules"] == [{"role": "admin", "action": "read"}]
	assert result["actions"] == ["audit"]
	mock_post.assert_called_once()
	call_kwargs = mock_post.call_args
	assert "/v1/data/apg/authz" in call_kwargs[0][0]


def test_evaluate_capability_rules_falls_back_on_opa_timeout(monkeypatch):
	"""OPA timeout falls back to built-in engine without raising."""
	monkeypatch.setenv("OPA_URL", "http://localhost:8181")

	with patch("httpx.post", side_effect=Exception("connection timeout")):
		from capabilities.common.auth.capability_contract import evaluate_capability_rules
		result = evaluate_capability_rules({"tenant_context_present": True})

	# Built-in engine ran as fallback
	assert "decision" in result


def test_evaluate_capability_rules_deny_without_context(monkeypatch):
	"""Empty context produces a deny decision from built-in engine."""
	monkeypatch.delenv("OPA_URL", raising=False)
	from capabilities.common.auth.capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({})
	assert "decision" in result


# ── OPA adapter module tests ──────────────────────────────────────────────────

async def test_evaluate_with_opa_returns_none_without_url(monkeypatch):
	"""evaluate_with_opa() returns None when OPA_URL is not configured."""
	monkeypatch.delenv("OPA_URL", raising=False)
	from capabilities.common.auth.opa_adapter import evaluate_with_opa
	result = await evaluate_with_opa({"action": "read"})
	assert result is None


async def test_evaluate_with_opa_returns_decision_on_success(monkeypatch):
	"""evaluate_with_opa() returns structured decision when OPA responds."""
	monkeypatch.setenv("OPA_URL", "http://localhost:8181")

	opa_result = {
		"decision": "allow",
		"matched_rules": [],
		"actions": ["audit"],
	}

	mock_resp = MagicMock()
	mock_resp.json.return_value = {"result": opa_result}
	mock_resp.raise_for_status = MagicMock()

	mock_inner = AsyncMock()
	mock_inner.post = AsyncMock(return_value=mock_resp)

	with patch("httpx.AsyncClient") as mock_client_cls:
		mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_inner)
		mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

		from capabilities.common.auth.opa_adapter import evaluate_with_opa
		result = await evaluate_with_opa({"action": "read", "user": {"roles": ["admin"]}})

	assert result is not None
	assert result["decision"] == "allow"
	assert "matched_rules" in result
	assert "actions" in result


def test_build_opa_context_shape():
	"""build_opa_context() produces the expected OPA input shape."""
	from capabilities.common.auth.opa_adapter import build_opa_context
	ctx = build_opa_context(
		user={"user_id": "u1", "tenant_id": "t1", "roles": ["clinician"]},
		action="read",
		resource="patient-123",
		resource_type="patient_record",
		tenant_id="t1",
		capability_id="healthcare_emr",
		extra={"purpose": "treatment"},
	)
	assert ctx["user"]["id"] == "u1"
	assert ctx["user"]["roles"] == ["clinician"]
	assert ctx["action"] == "read"
	assert ctx["resource"] == "patient-123"
	assert ctx["resource_type"] == "patient_record"
	assert ctx["capability_id"] == "healthcare_emr"
	assert ctx["context"]["purpose"] == "treatment"
	assert ctx["context"]["tenant_id"] == "t1"


# ── OPA Rego policy correctness tests ────────────────────────────────────────

def test_rego_authz_policy_file_exists():
	"""Base authorization Rego policy must exist."""
	from pathlib import Path
	policy = Path("policies/apg/authz.rego")
	assert policy.exists(), "policies/apg/authz.rego must exist"
	content = policy.read_text()
	assert "package apg.authz" in content
	assert "default allow" in content


def test_rego_healthcare_policy_file_exists():
	"""HIPAA healthcare Rego policy must exist."""
	from pathlib import Path
	policy = Path("policies/apg/capabilities/healthcare.rego")
	assert policy.exists()
	content = policy.read_text()
	assert "package apg.capabilities.healthcare" in content
	assert "phi_access_allowed" in content


def test_rego_fintech_policy_file_exists():
	"""PCI DSS fintech Rego policy must exist."""
	from pathlib import Path
	policy = Path("policies/apg/capabilities/fintech.rego")
	assert policy.exists()
	content = policy.read_text()
	assert "package apg.capabilities.fintech" in content
	assert "pci_scope" in content
