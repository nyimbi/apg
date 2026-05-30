"""Executable GRC document capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_grc_doc", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "grc_doc"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "doc_agents" in contract["provides"]
	assert "/grc-doc/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_restricted_gap():
	module = _load_module("rules_grc_doc", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "doc_batch",
		"event_stream": "queue",
	})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "create_document",
		"restricted_classification": True,
		"review_recorded": False,
	})["matched_rules"] == ["restricted_document_requires_review"]


def test_service_executes_document_lifecycle():
	service_module = _load_module("service_grc_doc", PACKAGE_DIR / "service.py")
	service = service_module.GrcDocService()

	template = service.register_template("template-1", "tenant-test", "Policy Template", "Policy body", "owner-1")
	document = service.create_document("doc-1", "tenant-test", "Access Policy", "owner-2", "Access policy", "policy", "confidential", template["id"])
	revision = service.create_revision("revision-1", "tenant-test", document["id"], "editor-1", "Updated policy", "Clarified access review")
	approved = service.approve_document(document["id"], "tenant-test", "approver-1", "Ready for publication")
	published = service.publish_document(document["id"], "tenant-test", "publisher-1")
	retention = service.assign_retention_policy("retention-1", "tenant-test", document["id"], 2555)
	grant = service.grant_access("grant-1", "tenant-test", document["id"], "group-risk", "view")
	job = service.register_processing_job("job-1", "tenant-test", document["id"], "classification")
	completed_job = service.complete_processing_job(job["id"], "tenant-test", {"classification": "confidential"})
	agent = service.register_doc_agent("tenant-test", "Document Review Agent", "codex", "document_reviewer", "review policies")

	summary = service.dashboard_summary("tenant-test")
	assert template["classification"] == "internal"
	assert revision["version"] == 2
	assert approved["approved_by"] == "approver-1"
	assert published["status"] == "published"
	assert retention["retention_days"] == 2555
	assert grant["permission"] == "view"
	assert completed_job["status"] == "completed"
	assert agent["runtime"] == "codex"
	assert summary["published_count"] == 1
	assert summary["audit_event_count"] == 10
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_grc_doc", PACKAGE_DIR / "service.py")
	service = service_module.GrcDocService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_document("doc", "", "Doc", "owner", "content")
	with pytest.raises(PermissionError, match="document_type_not_supported"):
		service.create_document("doc", "tenant-test", "Doc", "owner", "content", "unsupported")
	with pytest.raises(PermissionError, match="restricted_document_review_required"):
		service.create_document("doc", "tenant-test", "Doc", "owner", "content", "policy", "restricted")
	with pytest.raises(PermissionError, match="template_body_required"):
		service.register_template("template", "tenant-test", "Template", "", "owner")

	document = service.create_document("doc", "tenant-test", "Doc", "owner", "content", "policy", "restricted", reviewed_by="reviewer")
	with pytest.raises(PermissionError, match="segregation_of_duties_required"):
		service.approve_document(document["id"], "tenant-test", "owner", "approve")
	with pytest.raises(PermissionError, match="document_approval_required"):
		service.publish_document(document["id"], "tenant-test", "publisher")
	with pytest.raises(PermissionError, match="retention_too_short"):
		service.assign_retention_policy("retention", "tenant-test", document["id"], 30)
	with pytest.raises(PermissionError, match="restricted_access_expiry_required"):
		service.grant_access("grant", "tenant-test", document["id"], "user", "view")
	with pytest.raises(PermissionError, match="bytewax_processor_required"):
		service.register_processing_job("job", "tenant-test", document["id"], "classification", "queue")


def test_agents_batch_api_views_and_app_are_executable():
	api = _load_module("api_grc_doc", PACKAGE_DIR / "api.py")
	views = _load_module("views_grc_doc", PACKAGE_DIR / "views.py")
	app = _load_module("app_grc_doc", PACKAGE_DIR / "app.py")

	document = api.create_record({"tenant_id": "tenant-api", "id": "api-doc"})
	agent = api.register_doc_agent({
		"tenant_id": "tenant-api",
		"name": "Retention Agent",
		"runtime": "claude_code",
		"role": "retention_reviewer",
	})
	batch = api.service().validate_batch("tenant-api", 2)
	model = views.document_repository_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert document["id"] == "api-doc"
	assert agent["role"] == "retention_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["title"] == "API Document"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["grc_doc"]["screens"]["agents"]["route"] == "/grc-doc/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_grc_doc", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["grc_doc"]["streaming"]["processor"] == "bytewax"
