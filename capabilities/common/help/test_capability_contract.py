"""Regression coverage for the HELP executable capability contract."""

import pytest

from capabilities.common.help import register_capability
from capabilities.common.help.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.help.service import HelpService
from capabilities.common.help.views import (
	answer_console_model,
	article_editor_model,
	curation_queue_model,
	dashboard_model,
	help_center_model,
	support_analytics_model,
)


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


def test_help_lifecycle_is_executable():
	service = HelpService()

	article = service.create_article(
		article_id="article-password",
		tenant_id="tenant-help",
		title="Reset a password",
		body="Users reset passwords from account settings after confirming their email.",
		owner_id="owner-support",
		topics=["password", "account"],
		visibility="internal",
		source_ids=["runbook-1"],
	)
	published = service.publish_article(
		article_id=article["id"],
		tenant_id="tenant-help",
		approver_id="publisher-1",
		publication_approved=True,
	)
	search_hits = service.search_articles("tenant-help", "reset password")
	answer = service.generate_answer("answer-1", "tenant-help", "How do I reset my password?")
	feedback = service.record_feedback(
		feedback_id="feedback-1",
		tenant_id="tenant-help",
		user_id="user-1",
		rating=2,
		comment="Needs a screenshot",
		article_id=article["id"],
	)
	summary = service.dashboard_summary("tenant-help")

	assert article["status"] == "draft"
	assert published["status"] == "published"
	assert search_hits[0]["article"]["id"] == "article-password"
	assert answer["blocked"] is False
	assert answer["citations"][0]["article_id"] == "article-password"
	assert feedback["requires_review"] is True
	assert summary == {
		"tenant_id": "tenant-help",
		"article_count": 1,
		"published_article_count": 1,
		"answer_count": 1,
		"blocked_answer_count": 0,
		"feedback_count": 1,
		"open_curation_count": 1,
	}
	assert dashboard_model(service, "tenant-help")["summary"] == summary
	assert help_center_model(service, "tenant-help")["articles"][0]["id"] == "article-password"
	assert article_editor_model(service, "tenant-help")["visibility_options"] == ["public", "internal", "restricted"]
	assert answer_console_model(service, "tenant-help")["requires_citations"] is True
	assert curation_queue_model(service, "tenant-help")["curation_items"]
	assert support_analytics_model(service, "tenant-help")["summary"] == summary


def test_help_service_enforces_policy_guardrails():
	service = HelpService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_article("", "", "Title", "Body", "owner")
	with pytest.raises(PermissionError, match="article_owner_required"):
		service.create_article("article-no-owner", "tenant-help", "Title", "Body", "")

	service.create_article(
		"article-restricted",
		"tenant-help",
		"Restricted payroll help",
		"Payroll support guidance is restricted to authorized users.",
		"owner-support",
		topics=["payroll"],
		visibility="restricted",
	)

	with pytest.raises(PermissionError, match="publication_approval_required"):
		service.publish_article("article-restricted", "tenant-help", "publisher-1", False)
	with pytest.raises(PermissionError, match="rbac_filter_required"):
		service.publish_article(
			"article-restricted",
			"tenant-help",
			"publisher-1",
			True,
			rbac_filter_applied=False,
		)
	with pytest.raises(PermissionError, match="freshness_review_required"):
		service.publish_article(
			"article-restricted",
			"tenant-help",
			"publisher-1",
			True,
			article_age_days=120,
			freshness_review_recorded=False,
		)

	service.publish_article("article-restricted", "tenant-help", "publisher-1", True)
	with pytest.raises(PermissionError, match="rbac_filter_required"):
		service.search_articles(
			tenant_id="tenant-help",
			query="payroll",
			include_restricted=True,
			rbac_filter_applied=False,
		)
	with pytest.raises(PermissionError, match="citations_required"):
		service.generate_answer("answer-empty", "tenant-help", "unknown topic")
	with pytest.raises(ValueError, match="rating_out_of_range"):
		service.record_feedback("feedback-bad", "tenant-help", "user-1", 6, article_id="article-restricted")
	with pytest.raises(PermissionError, match="article_missing"):
		service.record_feedback("feedback-cross", "other-tenant", "user-1", 3, article_id="article-restricted")
