"""Executable InsurTech capability package tests."""

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
	module = _load_module("contract_fintech_insurance", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_insurance"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "insurance_claim_workflow" in contract["provides"]
	assert "/fintech-insurance/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_insurance", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "insurance_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "insurance_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_insurance_lifecycle():
	service_module = _load_module("service_fintech_insurance", PACKAGE_DIR / "service.py")
	service = service_module.InsurTechService()

	holder = service.onboard_policyholder("holder-1", "tenant-test", "Amina Holder", "kyc-1", "contact-1", "risk-profile-1")
	product = service.publish_product("product-1", "tenant-test", "Motor Protect", "motor", "coverage-1", "pricing-1")
	quote = service.generate_quote("quote-1", "tenant-test", holder["id"], product["id"], 120000, "usd", "underwriting-1")
	policy = service.bind_policy("policy-1", "tenant-test", quote["id"], "2026-06-01", "payment-1")
	premium = service.record_premium("premium-1", "tenant-test", policy["id"], 120000, "usd", "payment-1")
	claim = service.open_claim("claim-1", "tenant-test", policy["id"], "accident", 50000, "2026-06-10", "claim-evidence-1")
	document = service.record_document("document-1", "tenant-test", claim["id"], "proof_of_loss", "document-evidence-1")
	risk = service.record_risk_assessment("risk-1", "tenant-test", holder["id"], 72, "risk-source-1")
	reinsurance = service.record_reinsurance_attachment("reinsurance-1", "tenant-test", policy["id"], "treaty-1", 25)
	alert = service.record_compliance_alert("alert-1", "tenant-test", policy["id"], "medium", "alert-evidence-1")
	review = service.record_review("review-1", "tenant-test", alert["id"], "reviewer-1", "approved", "review-evidence-1")
	agent = service.register_insurance_agent("agent-1", "tenant-test", "Insurance Agent", "codex", "insurance_compliance_reviewer", "review policies")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert product["product_line"] == "motor"
	assert quote["currency"] == "USD"
	assert premium["amount_minor"] == 120000
	assert claim["claim_type"] == "accident"
	assert document["document_type"] == "proof_of_loss"
	assert risk["score"] == 72
	assert reinsurance["share_percent"] == 25
	assert review["status"] == "approved"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["policy_count"] == 1
	assert summary["audit_event_count"] == 12


def test_service_guardrails_reject_invalid_insurance_actions():
	service_module = _load_module("guardrail_service_fintech_insurance", PACKAGE_DIR / "service.py")
	service = service_module.InsurTechService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.onboard_policyholder("holder", "", "Holder", "kyc", "contact", "risk")
	with pytest.raises(PermissionError, match="policyholder_kyc_required"):
		service.onboard_policyholder("holder", "tenant-test", "Holder", "", "contact", "risk")
	holder = service.onboard_policyholder("holder-ok", "tenant-test", "Holder", "kyc", "contact", "risk")
	with pytest.raises(PermissionError, match="product_line_not_supported"):
		service.publish_product("product", "tenant-test", "Product", "unsupported", "coverage", "pricing")
	product = service.publish_product("product-ok", "tenant-test", "Product", "motor", "coverage", "pricing")
	with pytest.raises(PermissionError, match="positive_quote_premium_required"):
		service.generate_quote("quote", "tenant-test", holder["id"], product["id"], 0, "USD", "underwriting")
	with pytest.raises(PermissionError, match="quote_underwriting_reference_required"):
		service.generate_quote("quote-missing-underwriting", "tenant-test", holder["id"], product["id"], 100, "USD", "")
	quote = service.generate_quote("quote-ok", "tenant-test", holder["id"], product["id"], 100, "USD", "underwriting")
	with pytest.raises(PermissionError, match="policy_payment_reference_required"):
		service.bind_policy("policy", "tenant-test", quote["id"], "2026-06-01", "")
	policy = service.bind_policy("policy-ok", "tenant-test", quote["id"], "2026-06-01", "payment")
	with pytest.raises(PermissionError, match="positive_premium_amount_required"):
		service.record_premium("premium", "tenant-test", policy["id"], 0, "USD", "payment")
	with pytest.raises(PermissionError, match="premium_currency_not_supported"):
		service.record_premium("premium-currency", "tenant-test", policy["id"], 100, "XYZ", "payment")
	with pytest.raises(PermissionError, match="premium_payment_reference_required"):
		service.record_premium("premium-payment", "tenant-test", policy["id"], 100, "USD", "")
	with pytest.raises(PermissionError, match="claim_type_not_supported"):
		service.open_claim("claim", "tenant-test", policy["id"], "unsupported", 100, "2026-06-01", "evidence")
	with pytest.raises(PermissionError, match="document_type_not_supported"):
		service.record_document("document", "tenant-test", policy["id"], "unsupported", "evidence")
	with pytest.raises(PermissionError, match="risk_score_required"):
		service.record_risk_assessment("risk", "tenant-test", holder["id"], 101, "source")
	with pytest.raises(PermissionError, match="risk_source_required"):
		service.record_risk_assessment("risk-source", "tenant-test", holder["id"], 70, "")
	with pytest.raises(PermissionError, match="reinsurance_treaty_required"):
		service.record_reinsurance_attachment("reinsurance", "tenant-test", policy["id"], "", 10)
	with pytest.raises(PermissionError, match="positive_reinsurance_share_required"):
		service.record_reinsurance_attachment("reinsurance-share", "tenant-test", policy["id"], "treaty", 0)
	with pytest.raises(PermissionError, match="compliance_severity_not_supported"):
		service.record_compliance_alert("alert", "tenant-test", policy["id"], "unknown", "evidence")
	with pytest.raises(PermissionError, match="review_status_not_supported"):
		service.record_review("review", "tenant-test", policy["id"], "reviewer", "maybe", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="insurance_agent_runtime_not_supported"):
		service.register_insurance_agent("agent", "tenant-test", "Bad Agent", "unsupported", "insurance_compliance_reviewer", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_insurance", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_insurance", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_insurance", PACKAGE_DIR / "app.py")

	holder = api.onboard_policyholder({"tenant_id": "tenant-api", "policyholder_id": "api-holder", "name": "Holder", "kyc_reference": "kyc", "contact_reference": "contact", "risk_profile_reference": "risk"})
	product = api.publish_product({"tenant_id": "tenant-api", "product_id": "api-product", "name": "Product", "product_line": "motor", "coverage_terms_reference": "coverage", "pricing_reference": "pricing"})
	quote = api.generate_quote({"tenant_id": "tenant-api", "quote_id": "api-quote", "policyholder_id": holder["id"], "product_id": product["id"], "premium_minor": 100, "currency": "USD", "underwriting_reference": "underwriting"})
	policy = api.bind_policy({"tenant_id": "tenant-api", "policy_id": "api-policy", "quote_id": quote["id"], "effective_date": "2026-06-01", "payment_reference": "payment"})
	api.open_claim({"tenant_id": "tenant-api", "claim_id": "api-claim", "policy_id": policy["id"], "claim_type": "accident", "amount_minor": 50, "loss_date": "2026-06-10", "evidence_reference": "evidence"})
	agent = api.register_insurance_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Agent", "runtime": "claude_code", "role": "insurance_compliance_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.insurance_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "insurance_compliance_reviewer"
	assert dashboard["summary"]["policy_count"] == 1
	assert console["claims"][0]["id"] == "api-claim"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_insurance"]["screens"]["agents"]["route"] == "/fintech-insurance/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_insurance", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_insurance"]["streaming"]["processor"] == "bytewax"
