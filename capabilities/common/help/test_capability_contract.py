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
	help_agent_roster_model,
	help_center_model,
	lifecycle_batch_model,
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
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "content", "sources", "answers", "search", "feedback", "localization", "governance", "observability", "agents", "streaming", "adapters", "ui", "theme"}
	assert contract["configuration"]["governance"]["batch_event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["retrieval_augmented_generation"] == "ragn"
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "knowledge_steward" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["streaming"]["lifecycle_stream"] == "help.lifecycle"
	assert len(contract["rule_engine"]["rules"]) >= 41
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "home", "articles", "editor", "sources", "answers", "localization", "curation", "agents", "lifecycle", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/help/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "answer_panel" in contract["theme"]["components"]
	assert "source_registry" in contract["theme"]["components"]
	assert "help_agent_roster" in contract["theme"]["components"]
	assert "bytewax_lifecycle_panel" in contract["theme"]["components"]


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

	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_help_agent",
		"agent_id_present": False,
		"agent_name_present": False,
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"help_agent_runtime_supported",
		"help_agent_role_supported",
		"help_agent_requires_id",
		"help_agent_requires_name",
		"help_agent_requires_scope",
		"help_agent_requires_owner",
		"help_agent_requires_purpose",
		"help_agent_requires_contribution_disclosure",
		"help_agent_privileged_role_requires_human_approval",
	}

	batch_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_help_lifecycle_batch",
		"event_stream": "legacy_bus",
		"mutation_count": 0,
		"lifecycle_operation_supported": False,
	})
	assert batch_result["decision"] == "deny"
	assert set(batch_result["matched_rules"]) == {
		"help_lifecycle_batch_requires_mutations",
		"help_lifecycle_operation_supported",
		"bytewax_help_stream_required",
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "help"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "help_support_knowledge"
	assert registration["ui_components"]["answers"] == "/help/answers"
	assert registration["ui_components"]["sources"] == "/help/sources"
	assert registration["ui_components"]["agents"] == "/help/agents"
	assert registration["ui_components"]["lifecycle"] == "/help/lifecycle"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
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
	agent = service.register_help_agent(
		tenant_id="tenant-help",
		agent_id="help-agent-1",
		name="Knowledge Steward",
		runtime="codex",
		role="knowledge_steward",
		scope=article["id"],
		owner="publisher-1",
		purpose="Govern article, answer, curation, and safety lifecycle evidence.",
		human_approval_required=True,
	)
	batch = service.validate_help_lifecycle_batch("tenant-help", "bytewax", 3, "help_agent_batch")
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
	assert agent["status"] == "active"
	assert batch["status"] == "accepted"
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
		"help_agent_count": 1,
		"pending_help_agent_review_count": 0,
		"lifecycle_batch_count": 1,
		"denied_lifecycle_batch_count": 0,
		"audit_event_count": 10,
	}
	assert dashboard_model(service, "tenant-help")["summary"] == summary
	assert help_center_model(service, "tenant-help")["articles"][0]["id"] == "article-password"
	assert source_registry_model(service, "tenant-help")["sources"][0]["approved"] is True
	assert article_editor_model(service, "tenant-help")["visibility_options"] == ["public", "internal", "restricted"]
	assert answer_console_model(service, "tenant-help")["requires_citations"] is True
	assert localization_workbench_model(service, "tenant-help")["localizations"][0]["locale"] == "fr"
	assert curation_queue_model(service, "tenant-help")["curation_items"]
	assert help_agent_roster_model(service, "tenant-help")["active"][0]["name"] == "Knowledge Steward"
	assert lifecycle_batch_model(service, "tenant-help")["accepted"][0]["operation"] == "help_agent_batch"
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
	with pytest.raises(PermissionError, match="unsupported_help_agent_runtime"):
		service.register_help_agent("tenant-help", "bad-agent", "Bad Agent", "unknown", "answer_reviewer", "article-restricted", "owner", "Review answers.")
	with pytest.raises(PermissionError, match="help_agent_id_required"):
		service.register_help_agent("tenant-help", "", "Missing ID", "codex", "answer_reviewer", "article-restricted", "owner", "Review answers.")
	with pytest.raises(PermissionError, match="help_agent_name_required"):
		service.register_help_agent("tenant-help", "missing-name-agent", "", "codex", "answer_reviewer", "article-restricted", "owner", "Review answers.")
	with pytest.raises(PermissionError, match="help_agent_contribution_disclosure_required"):
		service.register_help_agent("tenant-help", "undisclosed-agent", "Undisclosed Agent", "codex", "answer_reviewer", "article-restricted", "owner", "Review answers.", contribution_disclosed=False)
	pending = service.register_help_agent("tenant-help", "pending-agent", "Pending Agent", "codex", "knowledge_steward", "article-restricted", "owner", "Govern privileged help evidence.")
	assert pending["status"] == "pending_review"
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_help_lifecycle_batch("tenant-help", "legacy_bus", 2, "help_agent_batch")
	with pytest.raises(PermissionError, match="help_lifecycle_batch_empty"):
		service.validate_help_lifecycle_batch("tenant-help", "bytewax", 0, "help_agent_batch")
	with pytest.raises(PermissionError, match="unsupported_help_lifecycle_operation"):
		service.validate_help_lifecycle_batch("tenant-help", "bytewax", 1, "kafka_replay")

	rule_result = service.evaluate({
		"tenant_context_present": True,
		"operation": "batch_help_mutation",
		"event_stream": "legacy_bus",
	})
	assert rule_result["decision"] == "deny"
	assert rule_result["matched_rules"] == ["batch_help_mutation_requires_bytewax"]
