"""Regression coverage for the SCRP executable capability contract."""

from capabilities.common.scrp import register_capability
from capabilities.common.scrp.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-scrp", {"extraction": {"result_retention_days": 14}})

	assert contract["capability"] == "scrp"
	assert contract["configuration"]["tenant_id"] == "tenant-scrp"
	assert contract["configuration"]["extraction"]["result_retention_days"] == 14
	assert contract["configuration_schema"]["required"] == ["tenant_id", "sources", "extraction", "compliance", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "sources", "jobs", "extractors", "pipelines", "compliance", "results", "settings"}
	assert contract["theme"]["name"] == "scrp_harvest_ops"


def test_rule_engine_enforces_scrp_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_source", "source_owner_assigned": False, "terms_evidence_present": False, "pii_expected": True, "pii_policy_attached": False, "sensitive_source": True, "source_review_recorded": False})
	job_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "run_harvest", "schedule_policy_attached": False, "terms_evidence_present": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "source_requires_owner", "source_terms_required", "pii_requires_handling_policy", "sensitive_source_requires_review"}
	assert job_result["matched_rules"] == ["harvest_requires_schedule_policy"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "scrp"
	assert "etlp" in registration["dependencies"]
	assert registration["ui_components"]["extractors"] == "/scrp/extractors"
	assert "scrp:run_jobs" in registration["permissions"]
