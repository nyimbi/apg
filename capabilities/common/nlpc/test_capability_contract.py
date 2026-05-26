"""Regression coverage for the NLPC executable capability contract."""

from capabilities.common.nlpc import register_capability
from capabilities.common.nlpc.capability_contract import (
	SUPPORTED_LANGUAGES,
	evaluate_capability_rules,
	get_capability_contract
)


AFRICAN_LANGUAGE_CODES = {
	"af", "aa", "ak", "am", "bm", "ee", "ff", "ha", "ig", "kr",
	"ki", "rw", "rn", "kg", "ln", "lg", "mg", "ny", "om", "sg",
	"sn", "so", "st", "sw", "ss", "ti", "ts", "tn", "tw", "ve",
	"wo", "xh", "yo", "zu", "kab", "kam", "luo", "mas", "mer",
	"mos", "nus", "suk", "tzm", "tig", "umb"
}


def test_contract_exposes_configuration_rules_ui_theme_and_languages():
	contract = get_capability_contract("tenant-text", {"processing": {"max_document_chars": 5000}})

	assert contract["capability"] == "nlpc"
	assert contract["configuration"]["tenant_id"] == "tenant-text"
	assert contract["configuration"]["processing"]["max_document_chars"] == 5000
	assert AFRICAN_LANGUAGE_CODES <= set(SUPPORTED_LANGUAGES)
	assert AFRICAN_LANGUAGE_CODES <= set(contract["configuration"]["processing"]["supported_languages"])
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"processing",
		"tasks",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"process",
		"documents",
		"annotations",
		"models",
		"languages",
		"governance",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/nlpc/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "language_coverage_map" in contract["theme"]["components"]


def test_rule_engine_enforces_nlp_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "process_document",
		"language_known": False,
		"task": "text_generation",
		"safety_policy_attached": False,
		"confidence_score": 0.25,
		"human_review_recorded": False,
		"document_count": 100,
		"async_queue_enabled": False
	})
	pii_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"task": "pii_detection",
		"redaction_policy_attached": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"language_required_or_detected",
		"generation_requires_safety_policy",
		"low_confidence_requires_review",
		"large_batch_requires_async_queue"
	}
	assert pii_result["decision"] == "deny"
	assert pii_result["matched_rules"] == ["pii_requires_redaction_policy"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "nlpc"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "nlpc_text_intelligence"
	assert registration["ui_components"]["languages"] == "/nlpc/languages"
	assert "aicr" in registration["dependencies"]
	assert "nlpc:process" in registration["permissions"]
