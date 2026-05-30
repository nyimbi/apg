"""Regression coverage for the I18N executable capability contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.common.i18n import register_capability
from capabilities.common.i18n.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.i18n.service import I18nService
from capabilities.common.i18n.views import (
	audit_trail_model,
	coverage_dashboard_model,
	dashboard_model,
	glossary_manager_model,
	i18n_agent_model,
	language_policy_model,
	locale_console_model,
	publish_queue_model,
	translation_workbench_model,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-i18n", {"translations": {"minimum_coverage_percent": 90}})

	assert contract["capability"] == "i18n"
	assert contract["configuration"]["tenant_id"] == "tenant-i18n"
	assert contract["configuration"]["translations"]["minimum_coverage_percent"] == 90
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"locales",
		"translations",
		"publishing",
		"i18n_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert len(contract["configuration"]["locales"]["african_language_codes"]) >= 40
	assert {"sw", "am", "ha", "ig", "yo", "zu", "rw", "so"} <= set(contract["configuration"]["locales"]["african_language_codes"])
	assert contract["configuration"]["i18n_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["provides"] == [
		"locale_management",
		"translation_memory",
		"content_localization",
		"language_fallbacks",
		"regional_formatting",
		"language_policy",
		"i18n_agents",
	]
	assert contract["requires"] == ["conf", "nlpc", "auth", "audl"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "locales", "translations", "glossaries", "coverage", "publishing", "agents", "audit", "policies", "settings"}
	assert contract["theme"]["name"] == "i18n_localization_workbench"


def test_rule_engine_enforces_i18n_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_locale", "locale_owner_assigned": False, "language_code_supported": False, "fallback_locale_present": False, "regional_format_present": False, "machine_translation_used": True, "translation_review_recorded": False, "restricted_content_present": True, "rbac_filter_applied": False, "coverage_percent": 70, "coverage_review_recorded": False})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_translations", "approval_recorded": False})
	agent_result = evaluate_capability_rules({"i18n_agent_present": True, "agent_runtime_supported": False})
	batch_result = evaluate_capability_rules({"requested_operation": "batch_i18n_mutation", "event_stream": "other-stream"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "locale_requires_owner", "locale_language_supported", "locale_requires_fallback", "locale_requires_regional_format", "machine_translation_requires_review", "restricted_content_requires_filtering", "low_coverage_requires_review"}
	assert publish_result["matched_rules"] == ["publish_requires_approval"]
	assert agent_result["decision"] == "deny"
	assert agent_result["matched_rules"] == ["i18n_agent_runtime_supported"]
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["batch_i18n_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "i18n"
	assert "nlpc" in registration["dependencies"]
	assert registration["ui_components"]["translations"] == "/i18n/translations"
	assert registration["ui_components"]["agents"] == "/i18n/agents"
	assert registration["streaming"]["processor"] == "bytewax"
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
	agent = service.register_i18n_agent(
		tenant_id="tenant-i18n",
		name="Swahili reviewer",
		runtime="codex",
		role="translation_reviewer",
		scope="review Swahili UI translations",
	)
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
	assert agent["runtime"] == "codex"
	assert agent["role"] == "translation_reviewer"
	assert summary == {
		"tenant_id": "tenant-i18n",
		"locale_count": 2,
		"glossary_term_count": 1,
		"translation_count": 3,
		"published_translation_count": 2,
		"coverage_report_count": 1,
		"coverage_review_count": 1,
		"publish_batch_count": 1,
		"i18n_agent_count": 1,
		"audit_event_count": 9,
		"streaming": service.describe("tenant-i18n")["streaming"],
	}
	assert service.validate_batch_i18n_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_i18n_mutation("other-stream")["decision"] == "deny"
	assert dashboard_model(service, "tenant-i18n")["summary"] == summary
	assert locale_console_model(service, "tenant-i18n")["locales"][0]["kind"] == "locale"
	assert translation_workbench_model(service, "tenant-i18n")["requires_machine_translation_review"] is True
	assert glossary_manager_model(service, "tenant-i18n")["glossary_terms"][0]["source_term"] == "ledger"
	assert coverage_dashboard_model(service, "tenant-i18n")["coverage_reports"][0]["requires_review"] is True
	assert publish_queue_model(service, "tenant-i18n")["approval_required"] is True
	assert i18n_agent_model(service, "tenant-i18n")["i18n_agents"][0]["role"] == "translation_reviewer"
	assert audit_trail_model(service, "tenant-i18n")["audit_events"][0]["kind"] == "audit_event"
	assert "sw" in language_policy_model("tenant-i18n")["african_language_codes"]


def test_i18n_service_enforces_policy_guardrails():
	service = I18nService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_locale("locale-empty", "", "fr-FR", "French", "owner")
	with pytest.raises(PermissionError, match="locale_owner_required"):
		service.create_locale("locale-no-owner", "tenant-i18n", "fr-FR", "French", "")
	with pytest.raises(PermissionError, match="language_code_not_supported"):
		service.create_locale("locale-unsupported", "tenant-i18n", "zz-ZZ", "Unsupported", "owner")

	service.create_locale("locale-fr", "tenant-i18n", "fr-FR", "French", "owner")
	with pytest.raises(PermissionError, match="glossary_owner_required"):
		service.add_glossary_term("term-no-owner", "tenant-i18n", "ledger", {"fr-FR": "registre"})

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
	with pytest.raises(PermissionError, match="translated_text_required"):
		service.upsert_translation(
			"tr-empty",
			"tenant-i18n",
			"empty",
			"fr-FR",
			"Empty",
			"",
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
	with pytest.raises(PermissionError, match="i18n_agent_runtime_not_supported"):
		service.register_i18n_agent("tenant-i18n", "Unsupported", "unsupported", "translation_reviewer", "review")


def test_lifecycle_ids_are_tenant_scoped():
	service = I18nService()

	for tenant_id, display_name, translated_text in (
		("tenant-a", "Swahili A", "Karibu A"),
		("tenant-b", "Swahili B", "Karibu B"),
	):
		service.create_locale("shared-locale", tenant_id, "sw-KE", display_name, "owner")
		service.upsert_translation("shared-translation", tenant_id, "welcome", "sw-KE", "Welcome", translated_text, reviewer_id="reviewer")
		service.register_i18n_agent(tenant_id, "Reviewer", "codex", "translation_reviewer", "review tenant translations", agent_id="shared-agent")

	assert service.list_locales("tenant-a")[0]["display_name"] == "Swahili A"
	assert service.list_locales("tenant-b")[0]["display_name"] == "Swahili B"
	assert service.list_translations("tenant-a")[0]["translated_text"] == "Karibu A"
	assert service.list_translations("tenant-b")[0]["translated_text"] == "Karibu B"
	assert service.list_i18n_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_i18n_agents("tenant-b")[0]["id"] == "shared-agent"


def test_generated_evidence_and_docs_are_current():
	app = _load_module("i18n_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["i18n"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["i18n"]["screens"]["agents"]["route"] == "/i18n/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
