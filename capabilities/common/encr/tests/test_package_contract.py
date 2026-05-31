"""ENCR package contract and dependency-light runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.encr import api, views
from capabilities.common.encr.service import EncrService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_is_valid():
	module = _load_module("package_contract_encr", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "encr"
	assert len(contract["ui"]["routes"]) >= 12
	assert len(contract["rule_engine"]["rules"]) >= 14
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["engine"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_encr", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text())
	committed_report = json.loads((PACKAGE_DIR / "release_report.json").read_text())
	capability = model["capabilities"]["encr"]

	assert self_test["passed"] is True
	assert committed_model == model
	assert committed_report["ok"] is True
	assert committed_report["evidence"]["contracts"]["capability_contract"]["rule_count"] >= 14
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "encr" in model["capabilities"]
	assert len(capability["ui"]["routes"]) >= 12
	assert capability["approvals"]["crypto_exception"] == "CryptoExceptionReviewRecord"
	assert capability["approvals"]["key_rotation"] == "KeyRotationRecord"
	assert capability["approvals"]["crypto_agent"] == "CryptoAgentRecord"
	assert capability["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert capability["streaming"]["engine"] == "bytewax"


def test_encryption_lifecycle_records_domains_operations_reviews_rotations_and_audit():
	service = EncrService()

	domain = service.register_key_domain(
		tenant_id="tenant-a",
		domain_id="finance-pii",
		name="Finance PII",
		owner="security-admin",
		algorithm="CRYSTALS-Kyber-768",
		data_classification="restricted",
		entropy_quality=0.99,
	)
	allowed = service.evaluate_crypto_operation(
		tenant_id="tenant-a",
		operation_id="encrypt-invoice",
		operation_type="encrypt",
		key_domain_id=domain["id"],
		data_classification="restricted",
	)
	legacy = service.evaluate_crypto_operation(
		tenant_id="tenant-a",
		operation_id="legacy-partner",
		operation_type="encrypt",
		key_domain_id=domain["id"],
		algorithm="RSA-2048",
		algorithm_family="legacy",
		data_classification="internal",
	)
	review = service.request_crypto_exception(
		tenant_id="tenant-a",
		review_id="legacy-partner-review",
		operation_id=legacy["id"],
		requested_by="integration-owner",
		reason="Partner migration window.",
	)
	approved = service.decide_crypto_exception(
		tenant_id="tenant-a",
		review_id=review["id"],
		reviewer="crypto-reviewer",
		decision="approved",
		notes="Approved for 30-day migration window.",
	)
	rotation = service.schedule_key_rotation(
		tenant_id="tenant-a",
		rotation_id="finance-pii-rotation",
		key_domain_id=domain["id"],
		requested_by="soc-analyst",
		reason="Active threat signal.",
	)
	completed = service.complete_key_rotation(
		tenant_id="tenant-a",
		rotation_id=rotation["id"],
		actor="key-admin",
		evidence="audit://encr/finance-pii/rotation",
	)
	agent = service.register_crypto_agent(
		tenant_id="tenant-a",
		agent_id="rotation-agent",
		name="Rotation Reviewer",
		runtime="opencode",
		role="threat-rotation-reviewer",
		scope="restricted key-domain rotation review",
		owner="secops",
		purpose="review compromise-triggered crypto rotation evidence",
		human_approval_required=True,
	)
	batch = service.validate_crypto_lifecycle_batch(
		tenant_id="tenant-a",
		event_stream="ByteWax",
		mutation_count=3,
	)
	after_rotation = service.evaluate_crypto_operation(
		tenant_id="tenant-a",
		operation_id="post-rotation",
		operation_type="encrypt",
		key_domain_id=domain["id"],
		data_classification="restricted",
		active_threat_signal=True,
	)
	summary = service.dashboard_summary("tenant-a")

	assert domain["algorithm_quantum_safe"] is True
	assert allowed["status"] == "allowed"
	assert legacy["status"] == "review_required"
	assert approved["status"] == "approved"
	assert completed["status"] == "completed"
	assert agent["runtime"] == "opencode"
	assert agent["role"] == "threat_rotation_reviewer"
	assert batch["event_stream"] == "bytewax"
	assert batch["accepted"] is True
	assert after_rotation["status"] == "allowed"
	assert summary["key_domain_count"] == 1
	assert summary["operation_count"] == 3
	assert summary["crypto_agent_count"] == 1
	assert summary["pending_exception_count"] == 0
	assert summary["scheduled_rotation_count"] == 0
	assert {event["event_type"] for event in service.list_audit_events("tenant-a")} >= {
		"key_domain_registered",
		"crypto_operation_allowed",
		"crypto_operation_review_required",
		"crypto_exception_requested",
		"crypto_exception_decided",
		"key_rotation_scheduled",
		"key_rotation_completed",
		"crypto_agent_registered",
	}


def test_guardrails_fail_closed_for_crypto_operations_reviews_and_rotations():
	service = EncrService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_key_domain("", "domain", "Domain", "owner")
	with pytest.raises(ValueError, match="key_domain_owner_required"):
		service.register_key_domain("tenant-a", "domain", "Domain", "")
	with pytest.raises(PermissionError, match="quantum_safe_algorithm_required"):
		service.register_key_domain("tenant-a", "restricted", "Restricted", "owner", "AES-256-GCM", "restricted")
	with pytest.raises(ValueError, match="entropy_quality_out_of_range"):
		service.register_key_domain("tenant-a", "bad-entropy", "Bad", "owner", entropy_quality=1.5)

	domain = service.register_key_domain("tenant-a", "internal", "Internal", "owner")
	with pytest.raises(KeyError, match="key_domain_not_found"):
		service.evaluate_crypto_operation("tenant-a", "missing-domain", "encrypt", "missing")

	restricted_denied = service.evaluate_crypto_operation(
		"tenant-a",
		"restricted-denied",
		"encrypt",
		domain["id"],
		data_classification="restricted",
	)
	plaintext_denied = service.evaluate_crypto_operation(
		"tenant-a",
		"plaintext-export",
		"export",
		domain["id"],
		plaintext_export_requested=True,
	)
	low_entropy_denied = service.evaluate_crypto_operation(
		"tenant-a",
		"low-entropy",
		"generate_key",
		domain["id"],
		entropy_quality=0.5,
	)
	threat_denied = service.evaluate_crypto_operation(
		"tenant-a",
		"active-threat",
		"encrypt",
		domain["id"],
		active_threat_signal=True,
	)
	threat_bypass_denied = service.evaluate_crypto_operation(
		"tenant-a",
		"active-threat-bypass",
		"encrypt",
		domain["id"],
		active_threat_signal=True,
	)
	with pytest.raises(ValueError, match="crypto_exception_not_required"):
		service.request_crypto_exception("tenant-a", "not-needed", plaintext_denied["id"], "owner", "Denied operation cannot be reviewed.")

	legacy = service.evaluate_crypto_operation(
		"tenant-a",
		"legacy-review",
		"encrypt",
		domain["id"],
		algorithm="RSA-2048",
		algorithm_family="legacy",
	)
	legacy_family_bypass = service.evaluate_crypto_operation(
		"tenant-a",
		"legacy-family-bypass",
		"encrypt",
		domain["id"],
		algorithm="RSA-2048",
		algorithm_family="modern",
	)
	review = service.request_crypto_exception("tenant-a", "legacy-review", legacy["id"], "requester", "Legacy partner.")
	with pytest.raises(PermissionError, match="independent_crypto_reviewer_required"):
		service.decide_crypto_exception("tenant-a", review["id"], "requester", "approved", "Self review.")
	with pytest.raises(ValueError, match="crypto_exception_notes_required"):
		service.decide_crypto_exception("tenant-a", review["id"], "reviewer", "approved", "")
	with pytest.raises(ValueError, match="key_rotation_reason_required"):
		service.schedule_key_rotation("tenant-a", "rotation", domain["id"], "requester", "")

	rotation = service.schedule_key_rotation("tenant-a", "rotation", domain["id"], "requester", "Threat signal.")
	with pytest.raises(PermissionError, match="key_rotation_evidence_required"):
		service.complete_key_rotation("tenant-a", rotation["id"], "key-admin", "")
	completed = service.complete_key_rotation("tenant-a", rotation["id"], "key-admin", "audit://rotation")
	with pytest.raises(ValueError, match="key_rotation_already_completed"):
		service.complete_key_rotation("tenant-a", completed["id"], "key-admin", "audit://rotation-again")
	with pytest.raises(PermissionError, match="crypto_agent_privileged_role_requires_human_approval"):
		service.register_crypto_agent(
			"tenant-a",
			"agent-denied",
			"Agent Denied",
			"codex",
			"exception_reviewer",
			"legacy exception review",
			"secops",
			"review crypto exceptions",
		)
	with pytest.raises(PermissionError, match="bytewax_crypto_stream_required"):
		service.validate_crypto_lifecycle_batch("tenant-a", "kafka", 1)
	with pytest.raises(ValueError, match="crypto_lifecycle_batch_empty"):
		service.validate_crypto_lifecycle_batch("tenant-a", "bytewax", 0)

	assert restricted_denied["status"] == "denied"
	assert plaintext_denied["status"] == "denied"
	assert low_entropy_denied["status"] == "denied"
	assert threat_denied["status"] == "denied"
	assert threat_bypass_denied["status"] == "denied"
	assert legacy_family_bypass["status"] == "review_required"


def test_api_and_view_models_expose_crypto_posture_surfaces():
	local_api_service = EncrService()
	api.SERVICE = local_api_service

	with pytest.raises(PermissionError, match="tenant_context_required"):
		api.register_key_domain({
			"id": "missing-tenant",
			"name": "Missing tenant",
			"owner": "secops",
		})
	domain = api.register_key_domain({
		"tenant_id": "tenant-b",
		"id": "documents",
		"name": "Documents",
		"owner": "secops",
		"algorithm": "CRYSTALS-Kyber-768",
		"data_classification": "restricted",
	})
	operation = api.evaluate_crypto_operation({
		"tenant_id": "tenant-b",
		"id": "legacy-doc",
		"operation_type": "encrypt",
		"key_domain_id": domain["id"],
		"algorithm": "RSA-2048",
		"algorithm_family": "legacy",
		"data_classification": "internal",
	})
	review = api.request_crypto_exception({
		"tenant_id": "tenant-b",
		"id": "legacy-doc-review",
		"operation_id": operation["id"],
		"requested_by": "owner",
		"reason": "Partner migration.",
	})
	api.decide_crypto_exception({
		"tenant_id": "tenant-b",
		"id": review["id"],
		"reviewer": "crypto-reviewer",
		"decision": "approved",
		"notes": "Approved temporarily.",
	})
	api_threat_bypass = api.evaluate_crypto_operation({
		"tenant_id": "tenant-b",
		"id": "threat-bypass",
		"operation_type": "encrypt",
		"key_domain_id": domain["id"],
		"active_threat_signal": True,
		"key_rotation_completed": True,
	})
	rotation = api.schedule_key_rotation({
		"tenant_id": "tenant-b",
		"id": "documents-rotation",
		"key_domain_id": domain["id"],
		"requested_by": "soc",
		"reason": "Threat signal.",
	})
	api.complete_key_rotation({
		"tenant_id": "tenant-b",
		"id": rotation["id"],
		"actor": "key-admin",
		"evidence": "audit://rotation/documents",
	})
	agent = api.register_crypto_agent({
		"tenant_id": "tenant-b",
		"id": "policy-agent",
		"name": "Policy Agent",
		"runtime": "claude-code",
		"role": "crypto-policy-reviewer",
		"scope": "crypto policy change review",
		"owner": "secops",
		"purpose": "review generated crypto policy changes",
		"contribution_disclosed": True,
	})
	batch = api.validate_crypto_lifecycle_batch({
		"tenant_id": "tenant-b",
		"event_stream": "bytewax",
		"mutation_count": 2,
	})
	with pytest.raises(PermissionError, match="tenant_context_required"):
		api.evaluate_crypto_operation({
			"id": "missing-tenant-operation",
			"operation_type": "encrypt",
			"key_domain_id": domain["id"],
			"active_threat_signal": True,
			"key_rotation_completed": True,
		})

	status = api.capability_status("tenant-b")
	posture = api.list_crypto_posture("tenant-b")
	dashboard = views.dashboard_model(tenant_id="tenant-b")
	operations = views.operations_console_model(tenant_id="tenant-b")
	keys = views.key_domain_model(tenant_id="tenant-b")
	policies = views.policy_designer_model("tenant-b")
	entropy = views.entropy_console_model(tenant_id="tenant-b")
	exceptions = views.exception_queue_model(tenant_id="tenant-b")
	rotations = views.rotation_console_model(tenant_id="tenant-b")
	homomorphic = views.homomorphic_workspace_model("tenant-b")
	analytics = views.analytics_model(tenant_id="tenant-b")
	audit = views.audit_timeline_model(tenant_id="tenant-b")
	agents = views.crypto_agents_model(tenant_id="tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["key_domain_count"] == 1
	assert status["operation_count"] == 2
	assert status["crypto_agent_count"] == 1
	assert posture["summary"]["scheduled_rotation_count"] == 0
	assert posture["crypto_agents"][0]["id"] == agent["id"]
	assert batch["required_processor"] == "bytewax"
	assert dashboard["summary"]["operation_count"] == 2
	assert dashboard["streaming"]["engine"] == "bytewax"
	assert operations["review_required"] == []
	assert keys["classifications"][-1] == "critical"
	assert policies["decision_order"] == ["deny", "require_review", "allow"]
	assert entropy["minimum_entropy_quality"] == 0.95
	assert exceptions["exception_reviews"][0]["status"] == "approved"
	assert rotations["rotations"][0]["status"] == "completed"
	assert "aggregate" in homomorphic["supported_operations"]
	assert analytics["summary"]["key_domain_count"] == 1
	assert audit["events"]
	assert agents["crypto_agents"][0]["runtime"] == "claude_code"
	assert "exception_reviewer" in agents["privileged_roles"]
	assert settings["agents"]["first_class"] is True
	assert settings["streaming"]["lifecycle_stream"] == "encr.lifecycle"
	assert settings["theme"]["name"] == "encr_quantum_guard"
	assert api_threat_bypass["status"] == "denied"
