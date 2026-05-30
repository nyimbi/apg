"""Regression coverage for the HELP executable capability contract."""

import pytest

from capabilities.common.help import register_capability
from capabilities.common.help.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.help.service import HelpService
from capabilities.common.help.views import (
	answer_console_model,
	article_editor_model,
	audit_model,
	curation_queue_model,
	dashboard_model,
	help_center_model,
	localization_workbench_model,
	settings_model,
	source_registry_model,
	support_analytics_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-help", {"content": {"freshness_review_days": 45}})

	assert contract["capability"] == "help"
	assert contract["configuration"]["tenant_id"] == "tenant-help"
	assert contract["configuration"]["content"]["freshness_review_days"] == 45
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "content", "sources", "answers", "search", "feedback", "localization", "governance", "observability", "adapters", "ui", "theme"}
	assert contract["configuration"]["governance"]["batch_event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["retrieval_augmented_generation"] == "ragn"
	assert len(contract["rule_engine"]["rules"]) >= 29
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "home", "articles", "editor", "sources", "answers", "localization", "curation", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/help/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "answer_panel" in contract["theme"]["components"]
	assert "source_registry" in contract["theme"]["components"]


def test_rule_engine_enforces_help_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "batch_help_mutation",
		"source_owner_assigned": False,
		"source_uri_present": False,
		"source_approval_required": True,
		"source_approved": False,
		"article_owner_assigned": False,
		"article_title_present": False,
		"article_body_present": False,
		"publication_approved": False,
		"query_present": False,
		"citations_present": False,
		"answer_confidence": 0.2,
		"unsafe_answer_detected": True,
		"restricted_content_present": True,
		"rbac_filter_applied": False,
		"query_logging_enabled": False,
		"article_age_days": 120,
		"freshness_review_recorded": False,
		"feedback_user_present": False,
		"feedback_rating": 0,
		"feedback_review_opened": False,
		"locale_supported": False,
		"translator_assigned": False,
		"fallback_locale_configured": False,
		"reviewer_present": False,
		"curation_evidence_present": False,
		"audit_event_recorded": False,
		"cross_tenant_access": True,
		"state_change_requested": True,
		"event_stream": "legacy_bus",
	})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_article", "publication_approved": False})
	answer_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "generate_answer", "citations_present": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"source_requires_approval",
		"restricted_content_requires_filtering",
		"stale_article_requires_review",
		"cross_tenant_help_access_denied",
		"help_state_change_requires_audit",
		"batch_help_mutation_requires_bytewax",
	}
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
	assert registration["ui_components"]["sources"] == "/help/sources"
	assert "ragn" in registration["dependencies"]
	assert "help:ask" in registration["permissions"]


def test_help_lifecycle_is_executable():
	service = HelpService()

	source = service.register_source(
		source_id="source-runbook-1",
		tenant_id="tenant-help",
		title="Account runbook",
		uri="kb://runbooks/account",
		owner_id="owner-support",
	)
	approved_source = service.approve_source(source["id"], "tenant-help", "publisher-1")
	article = service.create_article(
		article_id="article-password",
		tenant_id="tenant-help",
		title="Reset a password",
		body="Users reset passwords from account settings after confirming their email.",
		owner_id="owner-support",
		topics=["password", "account"],
		visibility="internal",
		source_ids=[approved_source["id"]],
	)
	published = service.publish_article(
		article_id=article["id"],
		tenant_id="tenant-help",
		approver_id="publisher-1",
		publication_approved=True,
	)
	search_hits = service.search_articles("tenant-help", "reset password")
	answer = service.generate_answer("answer-1", "tenant-help", "How do I reset my password?")
	localization = service.localize_article(
		"loc-1",
		"tenant-help",
		article["id"],
		"fr",
		"Reinitialiser un mot de passe",
		"Les utilisateurs reinitialisent les mots de passe depuis les parametres du compte.",
		"translator-1",
	)
	feedback = service.record_feedback(
		feedback_id="feedback-1",
		tenant_id="tenant-help",
		user_id="user-1",
		rating=2,
		comment="Needs a screenshot",
		article_id=article["id"],
	)
	curation = service.list_curation_items("tenant-help")[0]
	closed_curation = service.close_curation_item(curation["id"], "tenant-help", "reviewer-1", ["feedback reviewed"])
	summary = service.dashboard_summary("tenant-help")

	assert source["approved"] is False
	assert approved_source["approved"] is True
	assert article["status"] == "draft"
	assert published["status"] == "published"
	assert search_hits[0]["article"]["id"] == "article-password"
	assert answer["blocked"] is False
	assert answer["citations"][0]["article_id"] == "article-password"
	assert localization["locale"] == "fr"
	assert feedback["requires_review"] is True
	assert closed_curation["status"] == "closed"
	assert summary == {
		"tenant_id": "tenant-help",
		"source_count": 1,
		"approved_source_count": 1,
		"article_count": 1,
		"published_article_count": 1,
		"answer_count": 1,
		"blocked_answer_count": 0,
		"feedback_count": 1,
		"localization_count": 1,
		"open_curation_count": 0,
		"audit_event_count": 8,
	}
	assert dashboard_model(service, "tenant-help")["summary"] == summary
	assert help_center_model(service, "tenant-help")["articles"][0]["id"] == "article-password"
	assert source_registry_model(service, "tenant-help")["sources"][0]["approved"] is True
	assert article_editor_model(service, "tenant-help")["visibility_options"] == ["public", "internal", "restricted"]
	assert answer_console_model(service, "tenant-help")["requires_citations"] is True
	assert localization_workbench_model(service, "tenant-help")["localizations"][0]["locale"] == "fr"
	assert curation_queue_model(service, "tenant-help")["curation_items"]
	assert audit_model(service, "tenant-help")["event_stream"] == "bytewax"
	assert support_analytics_model(service, "tenant-help")["summary"] == summary
	assert settings_model("tenant-help")["configuration"]["tenant_id"] == "tenant-help"


def test_help_service_enforces_policy_guardrails():
	service = HelpService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_article("", "", "Title", "Body", "owner")
	with pytest.raises(PermissionError, match="article_owner_required"):
		service.create_article("article-no-owner", "tenant-help", "Title", "Body", "")
	with pytest.raises(PermissionError, match="source_uri_required"):
		service.register_source("source-no-uri", "tenant-help", "Source", "", "owner")

	source = service.register_source("source-restricted", "tenant-help", "Payroll source", "kb://payroll", "owner-support", visibility="restricted")
	with pytest.raises(PermissionError, match="source_approval_required"):
		service.create_article(
			"article-unapproved-source",
			"tenant-help",
			"Unapproved source article",
			"Content from unapproved source.",
			"owner-support",
			source_ids=[source["id"]],
		)
	service.approve_source(source["id"], "tenant-help", "publisher-1")

	service.create_article(
		"article-restricted",
		"tenant-help",
		"Restricted payroll help",
		"Payroll support guidance is restricted to authorized users.",
		"owner-support",
		topics=["payroll"],
		visibility="restricted",
		source_ids=[source["id"]],
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
	with pytest.raises(PermissionError, match="rating_out_of_range"):
		service.record_feedback("feedback-bad", "tenant-help", "user-1", 6, article_id="article-restricted")
	with pytest.raises(PermissionError, match="article_missing"):
		service.record_feedback("feedback-cross", "other-tenant", "user-1", 3, article_id="article-restricted")
	with pytest.raises(PermissionError, match="unsupported_locale"):
		service.localize_article("loc-bad", "tenant-help", "article-restricted", "zz", "Title", "Body", "translator-1")
	with pytest.raises(PermissionError, match="curation_reviewer_required"):
		curation = service.freshness_queue("tenant-help")[0]
		service.close_curation_item(curation["id"], "tenant-help", "", ["evidence"])

	rule_result = service.evaluate({
		"tenant_context_present": True,
		"operation": "batch_help_mutation",
		"event_stream": "legacy_bus",
	})
	assert rule_result["decision"] == "deny"
	assert rule_result["matched_rules"] == ["batch_help_mutation_requires_bytewax"]
