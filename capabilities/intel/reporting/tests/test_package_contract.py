"""Executable Intelligence Reporting capability package tests."""

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
	module = _load_module("contract_intel_reporting", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_reporting"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "reporting_agent_workflow" in contract["provides"]
	assert "/intel-reporting/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_reporting", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "reporting_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "reporting_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "reporting_agent_action", "uncited_claim_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "reporting_agent_action", "classification_downgrade_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "reporting_agent_action", "source_fabrication_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "reporting_agent_action", "privacy_bypass_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "reporting_agent_action", "autonomous_publication_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "reporting_agent_action", "unapproved_distribution_scope": True})["decision"] == "deny"


def test_service_executes_reporting_lifecycle():
	service_module = _load_module("service_intel_reporting", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceReportingService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	workspace = service.record_workspace("workspace-1", "tenant-test", "threat_reporting", "Threat Reporting", "confidential", authority["id"], "workspace-evidence")
	template = service.record_template("template-1", "tenant-test", workspace["id"], "advisory", "template-ref", "confidential", "template-evidence")
	product = service.record_product("product-1", "tenant-test", template["id"], "threat_advisory", "Threat Advisory", "author-1", "confidential", "product-evidence")
	section = service.record_section("section-1", "tenant-test", product["id"], "assessment", "section-ref", 0.82, "section-evidence")
	citation = service.record_citation("citation-1", "tenant-test", section["id"], "source_extract", "source-ref", "citation-evidence")
	approval = service.record_approval("approval-1", "tenant-test", product["id"], "classification", "approver-1", "approved", "approval-evidence")
	distribution = service.record_distribution("distribution-1", "tenant-test", product["id"], "internal", "recipient-ref", approval["id"], "distribution-evidence")
	publication = service.record_publication("publication-1", "tenant-test", distribution["id"], "portal", "publication-ref", approval["id"], "publication-evidence")
	review = service.record_review("review-1", "tenant-test", publication["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_reporting_agent("agent-1", "tenant-test", "Reporting Agent", "codex", "draft_writer", "draft support")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert workspace["workspace_type"] == "threat_reporting"
	assert template["template_type"] == "advisory"
	assert product["product_type"] == "threat_advisory"
	assert section["section_type"] == "assessment"
	assert citation["citation_type"] == "source_extract"
	assert approval["status"] == "approved"
	assert distribution["distribution_type"] == "internal"
	assert publication["publication_type"] == "portal"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_reporting", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceReportingService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_workspace("shared-workspace", "tenant-a", "threat_reporting", "Workspace A", "confidential", tenant_a["id"], "evidence-a")
	service.record_workspace("shared-workspace", "tenant-b", "partner_reporting", "Workspace B", "unclassified", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["workspace_count"] == 1
	assert dashboard_b["workspace_count"] == 1
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-a").name == "Workspace A"
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-b").name == "Workspace B"


def test_service_guardrails_reject_invalid_reporting_actions():
	service_module = _load_module("guardrail_service_intel_reporting", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceReportingService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_workspace("workspace", "tenant-test", "threat_reporting", "workspace", "confidential", "missing-auth", "evidence")
	workspace = service.record_workspace("workspace-ok", "tenant-test", "threat_reporting", "workspace", "confidential", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="template_type_not_supported"):
		service.record_template("template", "tenant-test", workspace["id"], "unknown", "template", "confidential", "evidence")
	template = service.record_template("template-ok", "tenant-test", workspace["id"], "advisory", "template", "confidential", "evidence")
	with pytest.raises(PermissionError, match="product_author_required"):
		service.record_product("product", "tenant-test", template["id"], "threat_advisory", "title", "", "confidential", "evidence")
	product = service.record_product("product-ok", "tenant-test", template["id"], "threat_advisory", "title", "author", "confidential", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_section("section", "tenant-test", product["id"], "assessment", "section", 1.8, "evidence")
	section = service.record_section("section-ok", "tenant-test", product["id"], "assessment", "section", 0.8, "evidence")
	with pytest.raises(PermissionError, match="citation_source_required"):
		service.record_citation("citation", "tenant-test", section["id"], "source_extract", "", "evidence")
	with pytest.raises(PermissionError, match="approval_approver_required"):
		service.record_approval("approval", "tenant-test", product["id"], "classification", "", "approved", "evidence")
	approval = service.record_approval("approval-ok", "tenant-test", product["id"], "classification", "approver", "approved", "evidence")
	with pytest.raises(PermissionError, match="distribution_approval_required"):
		service.record_distribution("distribution", "tenant-test", product["id"], "internal", "recipient", "", "evidence")
	distribution = service.record_distribution("distribution-ok", "tenant-test", product["id"], "internal", "recipient", approval["id"], "evidence")
	with pytest.raises(PermissionError, match="publication_approval_required"):
		service.record_publication("publication", "tenant-test", distribution["id"], "portal", "publication", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", product["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="reporting_agent_runtime_not_supported"):
		service.register_reporting_agent("agent", "tenant-test", "Bad Agent", "unsupported", "draft_writer", "scope")
	with pytest.raises(PermissionError, match="reporting_agent_scope_required"):
		service.register_reporting_agent("agent", "tenant-test", "Reporting Agent", "codex", "draft_writer", "")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="uncited_claim_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, uncited_claim_scope=True)
	with pytest.raises(PermissionError, match="classification_downgrade_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, classification_downgrade_scope=True)
	with pytest.raises(PermissionError, match="source_fabrication_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, source_fabrication_scope=True)
	with pytest.raises(PermissionError, match="privacy_bypass_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, privacy_bypass_scope=True)
	with pytest.raises(PermissionError, match="autonomous_publication_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, autonomous_publication_scope=True)
	with pytest.raises(PermissionError, match="unapproved_distribution_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unapproved_distribution_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_reporting", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_reporting", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_reporting", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	workspace = api.record_workspace({"tenant_id": "tenant-api", "workspace_id": "api-workspace", "workspace_type": "partner_reporting", "name": "Partner Reporting", "classification": "unclassified", "authority_id": authority["id"], "evidence_reference": "evidence"})
	template = api.record_template({"tenant_id": "tenant-api", "template_id": "api-template", "workspace_id": workspace["id"], "template_type": "brief", "template_reference": "template-ref", "classification": "unclassified", "evidence_reference": "evidence"})
	product = api.record_product({"tenant_id": "tenant-api", "product_id": "api-product", "template_id": template["id"], "product_type": "intelligence_brief", "title": "Brief", "author_id": "author", "classification": "unclassified", "evidence_reference": "evidence"})
	api.record_section({"tenant_id": "tenant-api", "section_id": "api-section", "product_id": product["id"], "section_type": "summary", "section_reference": "section-ref", "confidence_score": 0.72, "evidence_reference": "evidence"})
	agent = api.register_reporting_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Reporting Agent", "runtime": "claude_code", "role": "draft_writer"})
	batch = api.validate_batch({"tenant_id": "tenant-api", "item_count": 2})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.reporting_console_model(api.service(), "tenant-api")
	workbench = views.agent_workbench_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "draft_writer"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["products"][0]["id"] == product["id"]
	assert workbench["agents"][0]["id"] == agent["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_reporting"]["screens"]["agents"]["route"] == "/intel-reporting/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_reporting", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_reporting"]["streaming"]["processor"] == "bytewax"

