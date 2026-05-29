"""Regression coverage for the I18N executable capability contract."""

import pytest

from capabilities.common.i18n import register_capability
from capabilities.common.i18n.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.i18n.service import I18nService
from capabilities.common.i18n.views import (
	coverage_dashboard_model,
	dashboard_model,
	glossary_manager_model,
	locale_console_model,
	publish_queue_model,
	translation_workbench_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-i18n", {"translations": {"minimum_coverage_percent": 90}})

	assert contract["capability"] == "i18n"
	assert contract["configuration"]["tenant_id"] == "tenant-i18n"
	assert contract["configuration"]["translations"]["minimum_coverage_percent"] == 90
	assert contract["configuration_schema"]["required"] == ["tenant_id", "locales", "translations", "publishing", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "locales", "translations", "glossaries", "coverage", "publishing", "policies", "settings"}
	assert contract["theme"]["name"] == "i18n_localization_workbench"


def test_rule_engine_enforces_i18n_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_locale", "locale_owner_assigned": False, "machine_translation_used": True, "translation_review_recorded": False, "restricted_content_present": True, "rbac_filter_applied": False, "coverage_percent": 70, "coverage_review_recorded": False})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_translations", "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "locale_requires_owner", "machine_translation_requires_review", "restricted_content_requires_filtering", "low_coverage_requires_review"}
	assert publish_result["matched_rules"] == ["publish_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "i18n"
	assert "nlpc" in registration["dependencies"]
	assert registration["ui_components"]["translations"] == "/i18n/translations"
	assert "i18n:translate" in registration["permissions"]


def test_i18n_lifecycle_is_executable():
	service = I18nService()

	default_locale = service.create_locale(
		locale_id="locale-en",
		tenant_id="tenant-i18n",
		locale_code="en-US",
		display_name="English",
		owner_id="owner-locale",
		fallback_locale="en-US",
	)
	target_locale = service.create_locale(
		locale_id="locale-sw",
		tenant_id="tenant-i18n",
		locale_code="sw-KE",
		display_name="Swahili Kenya",
		owner_id="owner-locale",
		fallback_locale="en-US",
	)
	glossary = service.add_glossary_term(
		term_id="term-ledger",
		tenant_id="tenant-i18n",
		source_term="ledger",
		localized_terms={"sw-KE": "leja"},
		owner_id="owner-language",
	)
	source_translation = service.upsert_translation(
		translation_id="tr-en-welcome",
		tenant_id="tenant-i18n",
		key="app.welcome",
		locale_code="en-US",
		source_text="Welcome",
		translated_text="Welcome",
		reviewer_id="reviewer-1",
	)
	target_translation = service.upsert_translation(
		translation_id="tr-sw-welcome",
		tenant_id="tenant-i18n",
		key="app.welcome",
		locale_code="sw-KE",
		source_text="Welcome",
		translated_text="Karibu",
		machine_translation_used=True,
		translation_review_recorded=True,
		reviewer_id="reviewer-1",
	)
	reused = service.reuse_translation_memory(
		translation_id="tr-sw-home",
		tenant_id="tenant-i18n",
		key="nav.home",
		locale_code="sw-KE",
		source_text="Welcome",
		reviewer_id="reviewer-2",
	)
	published = service.publish_translations(
		batch_id="pub-sw-1",
		tenant_id="tenant-i18n",
		locale_code="sw-KE",
		translation_ids=["tr-sw-welcome", "tr-sw-home"],
		approver_id="publisher-1",
		approval_recorded=True,
	)
	report = service.coverage_report(
		report_id="coverage-sw",
		tenant_id="tenant-i18n",
		locale_code="sw-KE",
		required_keys=["app.welcome", "nav.home", "nav.settings"],
		coverage_review_recorded=False,
	)
	resolved = service.resolve_text("tenant-i18n", "app.welcome", "sw-KE")
	summary = service.dashboard_summary("tenant-i18n")

	assert default_locale["locale_code"] == "en-US"
	assert target_locale["fallback_locale"] == "en-US"
	assert glossary["localized_terms"]["sw-KE"] == "leja"
	assert source_translation["status"] == "reviewed"
	assert target_translation["source"] == "machine"
	assert reused["source"] == "memory"
	assert published["translation_ids"] == ["tr-sw-welcome", "tr-sw-home"]
	assert report["coverage_percent"] == 66.67
	assert report["requires_review"] is True
	assert resolved["text"] == "Karibu"
	assert resolved["fallback_chain"] == ["sw-KE", "en-US"]
	assert summary == {
		"tenant_id": "tenant-i18n",
		"locale_count": 2,
		"glossary_term_count": 1,
		"translation_count": 3,
		"published_translation_count": 2,
		"coverage_report_count": 1,
		"coverage_review_count": 1,
		"publish_batch_count": 1,
	}
	assert dashboard_model(service, "tenant-i18n")["summary"] == summary
	assert locale_console_model(service, "tenant-i18n")["locales"][0]["kind"] == "locale"
	assert translation_workbench_model(service, "tenant-i18n")["requires_machine_translation_review"] is True
	assert glossary_manager_model(service, "tenant-i18n")["glossary_terms"][0]["source_term"] == "ledger"
	assert coverage_dashboard_model(service, "tenant-i18n")["coverage_reports"][0]["requires_review"] is True
	assert publish_queue_model(service, "tenant-i18n")["approval_required"] is True


def test_i18n_service_enforces_policy_guardrails():
	service = I18nService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_locale("locale-empty", "", "fr-FR", "French", "owner")
	with pytest.raises(PermissionError, match="locale_owner_required"):
		service.create_locale("locale-no-owner", "tenant-i18n", "fr-FR", "French", "")

	service.create_locale("locale-fr", "tenant-i18n", "fr-FR", "French", "owner")

	with pytest.raises(PermissionError, match="translation_review_required"):
		service.upsert_translation(
			"tr-machine-unreviewed",
			"tenant-i18n",
			"hello",
			"fr-FR",
			"Hello",
			"Bonjour",
			machine_translation_used=True,
			translation_review_recorded=False,
		)
	with pytest.raises(PermissionError, match="rbac_filter_required"):
		service.upsert_translation(
			"tr-restricted",
			"tenant-i18n",
			"payroll.title",
			"fr-FR",
			"Payroll",
			"Paie",
			restricted=True,
			rbac_filter_applied=False,
		)

	draft = service.upsert_translation(
		"tr-draft",
		"tenant-i18n",
		"draft",
		"fr-FR",
		"Draft",
		"Brouillon",
		translation_review_recorded=False,
	)
	assert draft["status"] == "draft"
	with pytest.raises(PermissionError, match="publication_approval_required"):
		service.publish_translations("pub-denied", "tenant-i18n", "fr-FR", ["tr-draft"], "publisher", False)
	with pytest.raises(PermissionError, match="translation_not_reviewed"):
		service.publish_translations("pub-draft", "tenant-i18n", "fr-FR", ["tr-draft"], "publisher", True)
	with pytest.raises(PermissionError, match="locale_missing"):
		service.upsert_translation("tr-missing-locale", "tenant-i18n", "x", "de-DE", "X", "X")
	with pytest.raises(PermissionError, match="translation_memory_miss"):
		service.reuse_translation_memory("tr-memory-miss", "tenant-i18n", "x", "fr-FR", "Missing", "reviewer")
	with pytest.raises(PermissionError, match="translation_missing"):
		service.resolve_text("tenant-i18n", "missing.key", "fr-FR")
