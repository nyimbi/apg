"""Regression coverage for the HELP executable capability contract."""

from capabilities.common.help import register_capability
from capabilities.common.help.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-help", {"content": {"freshness_review_days": 45}})

	assert contract["capability"] == "help"
	assert contract["configuration"]["tenant_id"] == "tenant-help"
	assert contract["configuration"]["content"]["freshness_review_days"] == 45
	assert contract["configuration_schema"]["required"] == ["tenant_id", "content", "answers", "search", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "home", "articles", "editor", "answers", "curation", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/help/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "answer_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_help_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_article",
		"article_owner_assigned": False,
		"publication_approved": False,
		"citations_present": False,
		"restricted_content_present": True,
		"rbac_filter_applied": False,
		"article_age_days": 120,
		"freshness_review_recorded": False
	})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_article", "publication_approved": False})
	answer_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "generate_answer", "citations_present": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "article_requires_owner", "restricted_content_requires_filtering", "stale_article_requires_review"}
	assert publish_result["matched_rules"] == ["publication_requires_approval"]
	assert answer_result["matched_rules"] == ["answer_requires_citations"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "help"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "help_support_knowledge"
	assert registration["ui_components"]["answers"] == "/help/answers"
	assert "ragn" in registration["dependencies"]
	assert "help:ask" in registration["permissions"]
