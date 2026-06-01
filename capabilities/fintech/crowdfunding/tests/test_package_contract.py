"""Executable Crowdfunding Platform capability package tests."""

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
	module = _load_module("contract_fintech_crowdfunding", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_crowdfunding"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "crowdfunding_commitment_workflow" in contract["provides"]
	assert "/fintech-crowdfunding/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_crowdfunding", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "crowdfunding_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "crowdfunding_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_crowdfunding_lifecycle():
	service_module = _load_module("service_fintech_crowdfunding", PACKAGE_DIR / "service.py")
	service = service_module.CrowdfundingPlatformService()

	issuer = service.onboard_issuer("issuer-1", "tenant-test", "Solar Cooperative", "kyc-1", "ubo-1", "risk-1")
	campaign = service.publish_campaign("campaign-1", "tenant-test", issuer["id"], "Solar Mini Grid", "revenue_share", 50000000, "usd", "memo-1")
	disclosure = service.record_disclosure("disclosure-1", "tenant-test", campaign["id"], "offering_memo", "evidence-1")
	commitment = service.record_commitment("commitment-1", "tenant-test", campaign["id"], "investor-1", 250000, "usd", "investor-kyc-1", "risk-ack-1")
	escrow = service.record_escrow_funding("funding-1", "tenant-test", commitment["id"], "wallet-1", 250000)
	milestone = service.record_milestone("milestone-1", "tenant-test", campaign["id"], "Permits received", "milestone-evidence-1")
	payout = service.authorize_payout("payout-1", "tenant-test", campaign["id"], milestone["id"], 100000, "approval-1")
	update = service.publish_investor_update("update-1", "tenant-test", campaign["id"], disclosure["id"], "all_investors")
	alert = service.record_compliance_alert("alert-1", "tenant-test", campaign["id"], "medium", "alert-evidence-1")
	review = service.record_review("review-1", "tenant-test", alert["id"], "reviewer-1", "approved", "review-evidence-1")
	agent = service.register_crowdfunding_agent("agent-1", "tenant-test", "Crowdfunding Agent", "codex", "crowdfunding_compliance_reviewer", "review campaigns")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert campaign["campaign_type"] == "revenue_share"
	assert campaign["currency"] == "USD"
	assert disclosure["disclosure_type"] == "offering_memo"
	assert service.commitments[commitment["id"]].status == "funded"
	assert escrow["amount_minor"] == 250000
	assert payout["status"] == "authorized"
	assert update["recipient_scope"] == "all_investors"
	assert review["status"] == "approved"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["campaign_count"] == 1
	assert summary["audit_event_count"] == 11


def test_service_guardrails_reject_invalid_crowdfunding_actions():
	service_module = _load_module("guardrail_service_fintech_crowdfunding", PACKAGE_DIR / "service.py")
	service = service_module.CrowdfundingPlatformService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.onboard_issuer("issuer", "", "Issuer", "kyc", "ubo", "risk")
	with pytest.raises(PermissionError, match="issuer_kyc_required"):
		service.onboard_issuer("issuer", "tenant-test", "Issuer", "", "ubo", "risk")
	issuer = service.onboard_issuer("issuer-ok", "tenant-test", "Issuer", "kyc", "ubo", "risk")
	with pytest.raises(PermissionError, match="campaign_type_not_supported"):
		service.publish_campaign("campaign", "tenant-test", issuer["id"], "Campaign", "unsupported", 100, "USD", "memo")
	with pytest.raises(PermissionError, match="positive_campaign_target_required"):
		service.publish_campaign("campaign", "tenant-test", issuer["id"], "Campaign", "equity", 0, "USD", "memo")
	campaign = service.publish_campaign("campaign-ok", "tenant-test", issuer["id"], "Campaign", "equity", 100, "USD", "memo")
	with pytest.raises(PermissionError, match="disclosure_type_not_supported"):
		service.record_disclosure("disclosure", "tenant-test", campaign["id"], "unsupported", "evidence")
	with pytest.raises(PermissionError, match="investor_kyc_required"):
		service.record_commitment("commitment", "tenant-test", campaign["id"], "investor", 100, "USD", "", "ack")
	commitment = service.record_commitment("commitment-ok", "tenant-test", campaign["id"], "investor", 100, "USD", "kyc", "ack")
	with pytest.raises(PermissionError, match="escrow_wallet_reference_required"):
		service.record_escrow_funding("funding", "tenant-test", commitment["id"], "", 100)
	with pytest.raises(PermissionError, match="milestone_evidence_required"):
		service.record_milestone("milestone", "tenant-test", campaign["id"], "Milestone", "")
	milestone = service.record_milestone("milestone-ok", "tenant-test", campaign["id"], "Milestone", "evidence")
	with pytest.raises(PermissionError, match="payout_approval_required"):
		service.authorize_payout("payout", "tenant-test", campaign["id"], milestone["id"], 100, "")
	with pytest.raises(PermissionError, match="update_disclosure_reference_required"):
		service.publish_investor_update("update", "tenant-test", campaign["id"], "", "all")
	with pytest.raises(PermissionError, match="compliance_severity_not_supported"):
		service.record_compliance_alert("alert", "tenant-test", campaign["id"], "unknown", "evidence")
	with pytest.raises(PermissionError, match="review_status_not_supported"):
		service.record_review("review", "tenant-test", campaign["id"], "reviewer", "maybe", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="crowdfunding_agent_runtime_not_supported"):
		service.register_crowdfunding_agent("agent", "tenant-test", "Bad Agent", "unsupported", "crowdfunding_compliance_reviewer", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_crowdfunding", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_crowdfunding", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_crowdfunding", PACKAGE_DIR / "app.py")

	issuer = api.onboard_issuer({"tenant_id": "tenant-api", "issuer_id": "api-issuer", "name": "Issuer", "kyc_reference": "kyc", "beneficial_owner_reference": "ubo", "risk_rating_reference": "risk"})
	campaign = api.publish_campaign({"tenant_id": "tenant-api", "campaign_id": "api-campaign", "issuer_id": issuer["id"], "name": "Campaign", "campaign_type": "equity", "target_amount_minor": 1000, "currency": "USD", "disclosure_reference": "memo"})
	api.record_disclosure({"tenant_id": "tenant-api", "disclosure_id": "api-disclosure", "campaign_id": campaign["id"], "disclosure_type": "offering_memo", "evidence_reference": "evidence"})
	api.record_commitment({"tenant_id": "tenant-api", "commitment_id": "api-commitment", "campaign_id": campaign["id"], "investor_id": "investor", "amount_minor": 100, "currency": "USD", "investor_kyc_reference": "kyc", "risk_ack_reference": "ack"})
	agent = api.register_crowdfunding_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Agent", "runtime": "claude_code", "role": "crowdfunding_compliance_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.crowdfunding_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "crowdfunding_compliance_reviewer"
	assert dashboard["summary"]["campaign_count"] == 1
	assert console["commitments"][0]["id"] == "api-commitment"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_crowdfunding"]["screens"]["agents"]["route"] == "/fintech-crowdfunding/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_crowdfunding", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_crowdfunding"]["streaming"]["processor"] == "bytewax"
