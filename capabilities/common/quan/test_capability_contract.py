"""Regression coverage for the QUAN executable capability contract."""

from capabilities.common.quan import register_capability
from capabilities.common.quan.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-quan", {"jobs": {"shot_limit": 20000}})

	assert contract["capability"] == "quan"
	assert contract["configuration"]["tenant_id"] == "tenant-quan"
	assert contract["configuration"]["jobs"]["shot_limit"] == 20000
	assert contract["configuration_schema"]["required"] == ["tenant_id", "backends", "circuits", "jobs", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "backends", "circuits", "jobs", "experiments", "results", "governance", "settings"}
	assert contract["theme"]["name"] == "quan_quantum_lab"


def test_rule_engine_enforces_quan_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_circuit", "circuit_owner_assigned": False, "sensitive_input_present": True, "encryption_applied": False, "shot_count": 20000, "job_review_recorded": False})
	backend_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_backend", "backend_approved": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "circuit_requires_owner", "sensitive_input_requires_encryption", "large_job_requires_review"}
	assert backend_result["matched_rules"] == ["backend_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "quan"
	assert "aicr" in registration["dependencies"]
	assert registration["ui_components"]["jobs"] == "/quan/jobs"
	assert "quan:run_jobs" in registration["permissions"]
