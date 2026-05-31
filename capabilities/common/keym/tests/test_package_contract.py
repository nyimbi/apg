"""KEYM package contract and dependency-light runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.keym import api, view_models
from capabilities.common.keym.service import KeymService


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
	module = _load_module("package_contract_keym", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "keym"
	assert len(contract["ui"]["routes"]) >= 12
	assert len(contract["rule_engine"]["rules"]) >= 14
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["engine"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_keym", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text())
	committed_report = json.loads((PACKAGE_DIR / "release_report.json").read_text())
	capability = model["capabilities"]["keym"]

	assert self_test["passed"] is True
	assert committed_model == model
	assert committed_report["ok"] is True
	assert committed_report["evidence"]["contracts"]["capability_contract"]["rule_count"] >= 14
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "keym" in model["capabilities"]
	assert len(capability["ui"]["routes"]) >= 12
	assert capability["approvals"]["export"] == "ExportApprovalRecord"
	assert capability["approvals"]["rotation_exception"] == "RotationExceptionRecord"
	assert capability["approvals"]["key_agent"] == "KeymAgentRecord"
	assert capability["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert capability["streaming"]["engine"] == "bytewax"


def test_key_lifecycle_records_export_rotation_compromise_and_audit_state():
	service = KeymService()

	key = service.create_managed_key(
		tenant_id="tenant-a",
		key_id="finance-root",
		name="Finance Root",
		owner="security-admin",
		algorithm="AES-256",
		key_class="root",
		policy_ref="policy://finance-root",
		hsm_attested=True,
	)
	use_allowed = service.evaluate_key_operation("tenant-a", "use-root", key["id"], "use_key")
	export_denied = service.evaluate_key_operation("tenant-a", "export-before-approval", key["id"], "export_key")
	approval = service.request_export_approval(
		tenant_id="tenant-a",
		approval_id="export-1",
		key_id=key["id"],
		requested_by="integration-owner",
		reason="Partner wrapped migration.",
	)
	approved = service.decide_export_approval(
		tenant_id="tenant-a",
		approval_id=approval["id"],
		reviewer="key-custodian",
		decision="approved",
		notes="Approved wrapped export only.",
	)
	export_allowed = service.evaluate_key_operation("tenant-a", "export-after-approval", key["id"], "export_key")
	old_key = service.create_managed_key(
		tenant_id="tenant-a",
		key_id="old-data",
		name="Old Data",
		owner="key-admin",
		policy_ref="policy://old-data",
		rotation_age_days=120,
	)
	overdue = service.evaluate_key_operation("tenant-a", "use-old", old_key["id"], "use_key")
	exception = service.request_rotation_exception(
		tenant_id="tenant-a",
		exception_id="old-data-exception",
		key_id=old_key["id"],
		requested_by="app-owner",
		reason="Migration blackout.",
	)
	exception_approved = service.decide_rotation_exception(
		tenant_id="tenant-a",
		exception_id=exception["id"],
		reviewer="key-custodian",
		decision="approved",
		notes="Approved until migration window closes.",
	)
	overdue_allowed = service.evaluate_key_operation("tenant-a", "use-old-after-exception", old_key["id"], "use_key")
	rotation = service.schedule_rotation("tenant-a", "old-data-rotation", old_key["id"], "soc", "Age threshold.")
	completed = service.complete_rotation("tenant-a", rotation["id"], "key-admin", "audit://keym/old-data/rotation")
	compromised = service.mark_key_compromised("tenant-a", key["id"], "soc", "audit://keym/finance-root/compromise")
	compromised_denied = service.evaluate_key_operation("tenant-a", "use-compromised", key["id"], "use_key")
	agent = service.register_key_agent(
		tenant_id="tenant-a",
		agent_id="compromise-agent",
		name="Compromise Reviewer",
		runtime="opencode",
		role="compromise-responder",
		scope="compromised key response review",
		owner="secops",
		purpose="review key compromise evidence and rotation readiness",
		human_approval_required=True,
	)
	batch = service.validate_key_lifecycle_batch("tenant-a", "ByteWax", 4)
	summary = service.dashboard_summary("tenant-a")

	assert use_allowed["status"] == "allowed"
	assert export_denied["status"] == "denied"
	assert approved["status"] == "approved"
	assert export_allowed["status"] == "allowed"
	assert overdue["status"] == "review_required"
	assert exception_approved["status"] == "approved"
	assert overdue_allowed["status"] == "allowed"
	assert completed["status"] == "completed"
	assert compromised["status"] == "compromised"
	assert compromised_denied["status"] == "denied"
	assert agent["runtime"] == "opencode"
	assert agent["role"] == "compromise_responder"
	assert batch["event_stream"] == "bytewax"
	assert batch["accepted"] is True
	assert summary["key_count"] == 2
	assert summary["key_agent_count"] == 1
	assert summary["compromised_key_count"] == 1
	assert {event["event_type"] for event in service.list_audit_events("tenant-a")} >= {
		"managed_key_created",
		"export_approval_requested",
		"export_approval_decided",
		"rotation_exception_requested",
		"rotation_exception_decided",
		"key_rotation_scheduled",
		"key_rotation_completed",
		"managed_key_compromised",
		"key_agent_registered",
	}


def test_keym_guardrails_fail_closed():
	service = KeymService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_managed_key("", "key", "Key", "owner", policy_ref="policy://key")
	with pytest.raises(PermissionError, match="key_policy_required"):
		service.create_managed_key("tenant-a", "missing-policy", "Missing Policy", "owner")
	with pytest.raises(PermissionError, match="hsm_attestation_required"):
		service.create_managed_key("tenant-a", "root", "Root", "owner", key_class="root", policy_ref="policy://root")
	with pytest.raises(ValueError, match="unsupported_key_class"):
		service.create_managed_key("tenant-a", "bad", "Bad", "owner", key_class="unknown", policy_ref="policy://bad")

	key = service.create_managed_key("tenant-a", "data", "Data", "owner", policy_ref="policy://data", rotation_age_days=120)
	export_denied = service.evaluate_key_operation("tenant-a", "export", key["id"], "export_key")
	overdue = service.evaluate_key_operation("tenant-a", "overdue", key["id"], "use_key")
	with pytest.raises(ValueError, match="export_approval_reason_required"):
		service.request_export_approval("tenant-a", "export-approval", key["id"], "owner", "")
	approval = service.request_export_approval("tenant-a", "export-approval", key["id"], "owner", "Wrapped export.")
	with pytest.raises(PermissionError, match="independent_export_reviewer_required"):
		service.decide_export_approval("tenant-a", approval["id"], " OWNER ", "approved", "Self review.")
	with pytest.raises(ValueError, match="review_notes_required"):
		service.decide_export_approval("tenant-a", approval["id"], "reviewer", "approved", "")

	with pytest.raises(ValueError, match="rotation_exception_reason_required"):
		service.request_rotation_exception("tenant-a", "rotation-exception", key["id"], "owner", "")
	exception = service.request_rotation_exception("tenant-a", "rotation-exception", key["id"], "owner", "Blackout.")
	with pytest.raises(PermissionError, match="independent_rotation_exception_reviewer_required"):
		service.decide_rotation_exception("tenant-a", exception["id"], " OWNER ", "approved", "Self review.")
	with pytest.raises(ValueError, match="review_notes_required"):
		service.decide_rotation_exception("tenant-a", exception["id"], "reviewer", "approved", "")

	with pytest.raises(ValueError, match="key_rotation_reason_required"):
		service.schedule_rotation("tenant-a", "rotation", key["id"], "owner", "")
	rotation = service.schedule_rotation("tenant-a", "rotation", key["id"], "owner", "Aged key.")
	with pytest.raises(PermissionError, match="key_rotation_evidence_required"):
		service.complete_rotation("tenant-a", rotation["id"], "key-admin", "")
	completed = service.complete_rotation("tenant-a", rotation["id"], "key-admin", "audit://rotation")
	with pytest.raises(ValueError, match="key_rotation_already_completed"):
		service.complete_rotation("tenant-a", completed["id"], "key-admin", "audit://rotation-again")

	with pytest.raises(ValueError, match="compromise_evidence_required"):
		service.mark_key_compromised("tenant-a", key["id"], "soc", "")
	compromised = service.mark_key_compromised("tenant-a", key["id"], "soc", "audit://compromise")
	compromised_denied = service.evaluate_key_operation("tenant-a", "compromised-use", compromised["id"], "use_key")
	disabled = service.create_record("disabled", "tenant-a", {"owner": "secops"}, status="disabled")
	destroyed = service.create_record("destroyed", "tenant-a", {"owner": "secops"}, status="destroyed")
	disabled_denied = service.evaluate_key_operation("tenant-a", "disabled-use", disabled["id"], "use_key")
	destroyed_denied = service.evaluate_key_operation("tenant-a", "destroyed-use", destroyed["id"], "use_key")
	with pytest.raises(PermissionError, match="key_destroyed"):
		service.schedule_rotation("tenant-a", "destroyed-rotation", destroyed["id"], "secops", "Replacement required.")
	with pytest.raises(PermissionError, match="key_agent_privileged_role_requires_human_approval"):
		service.register_key_agent(
			"tenant-a",
			"agent-denied",
			"Agent Denied",
			"codex",
			"export_reviewer",
			"wrapped export review",
			"secops",
			"review key export approvals",
		)
	with pytest.raises(PermissionError, match="bytewax_key_stream_required"):
		service.validate_key_lifecycle_batch("tenant-a", "legacy_queue", 1)
	with pytest.raises(ValueError, match="key_lifecycle_batch_empty"):
		service.validate_key_lifecycle_batch("tenant-a", "bytewax", 0)

	assert export_denied["status"] == "denied"
	assert overdue["status"] == "review_required"
	assert compromised_denied["status"] == "denied"
	assert disabled_denied["status"] == "denied"
	assert disabled_denied["required_actions"] == ["reactivate_or_rotate_key"]
	assert destroyed_denied["status"] == "denied"
	assert destroyed_denied["required_actions"] == ["provision_replacement_key"]


def test_api_and_view_models_expose_key_posture_surfaces():
	local_service = KeymService()
	api.SERVICE = local_service

	with pytest.raises(PermissionError, match="tenant_context_required"):
		api.create_managed_key({"id": "missing-tenant", "name": "Missing", "owner": "secops", "policy_ref": "policy://missing"})
	key = api.create_managed_key({
		"tenant_id": "tenant-b",
		"id": "documents",
		"name": "Documents",
		"owner": "secops",
		"policy_ref": "policy://documents",
	})
	approval = api.request_export_approval({
		"tenant_id": "tenant-b",
		"id": "documents-export",
		"key_id": key["id"],
		"requested_by": "owner",
		"reason": "Wrapped partner export.",
	})
	api.decide_export_approval({
		"tenant_id": "tenant-b",
		"id": approval["id"],
		"reviewer": "custodian",
		"decision": "approved",
		"notes": "Approved wrapped export.",
	})
	rotation = api.schedule_rotation({
		"tenant_id": "tenant-b",
		"id": "documents-rotation",
		"key_id": key["id"],
		"requested_by": "soc",
		"reason": "Policy interval.",
	})
	api.complete_rotation({
		"tenant_id": "tenant-b",
		"id": rotation["id"],
		"actor": "key-admin",
		"evidence": "audit://rotation/documents",
	})
	api.mark_key_compromised({
		"tenant_id": "tenant-b",
		"key_id": key["id"],
		"actor": "soc",
		"evidence": "audit://compromise/documents",
	})
	api.evaluate_key_operation({
		"tenant_id": "tenant-b",
		"id": "use-documents",
		"key_id": key["id"],
		"operation": "use_key",
	})
	agent = api.register_key_agent({
		"tenant_id": "tenant-b",
		"id": "policy-agent",
		"name": "Policy Agent",
		"runtime": "claude-code",
		"role": "key-policy-reviewer",
		"scope": "key policy change review",
		"owner": "secops",
		"purpose": "review generated key policy changes",
		"contribution_disclosed": True,
	})
	batch = api.validate_key_lifecycle_batch({
		"tenant_id": "tenant-b",
		"event_stream": "bytewax",
		"mutation_count": 2,
	})

	status = api.capability_status("tenant-b")
	posture = api.list_key_posture("tenant-b")
	dashboard = view_models.dashboard_model(tenant_id="tenant-b")
	inventory = view_models.inventory_model(tenant_id="tenant-b")
	lifecycle = view_models.lifecycle_model(tenant_id="tenant-b")
	export_approvals = view_models.export_approval_queue_model(tenant_id="tenant-b")
	rotation_exceptions = view_models.rotation_exception_queue_model(tenant_id="tenant-b")
	hsm = view_models.hsm_console_model(tenant_id="tenant-b")
	compromise = view_models.compromise_console_model(tenant_id="tenant-b")
	audit = view_models.audit_timeline_model(tenant_id="tenant-b")
	analytics = view_models.analytics_model(tenant_id="tenant-b")
	agents = view_models.key_agents_model(tenant_id="tenant-b")
	settings = view_models.settings_model("tenant-b")

	assert status["key_count"] == 1
	assert status["key_agent_count"] == 1
	assert status["compromised_key_count"] == 1
	assert posture["summary"]["operation_count"] == 1
	assert posture["key_agents"][0]["id"] == agent["id"]
	assert batch["required_processor"] == "bytewax"
	assert dashboard["summary"]["key_count"] == 1
	assert dashboard["streaming"]["engine"] == "bytewax"
	assert inventory["key_classes"][-1] == "wrapping"
	assert lifecycle["rotations"][0]["status"] == "completed"
	assert export_approvals["export_approvals"][0]["status"] == "approved"
	assert rotation_exceptions["pending"] == []
	assert hsm["attestation_required"] is True
	assert compromise["compromised_keys"][0]["status"] == "compromised"
	assert audit["events"]
	assert analytics["summary"]["compromised_key_count"] == 1
	assert agents["key_agents"][0]["runtime"] == "claude_code"
	assert "export_reviewer" in agents["privileged_roles"]
	assert settings["agents"]["first_class"] is True
	assert settings["streaming"]["lifecycle_stream"] == "keym.lifecycle"
	assert settings["theme"]["name"] == "keym_vault_console"
