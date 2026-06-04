"""Service-level tests for APG Multi-Language & Localisation."""

from __future__ import annotations

import asyncio
import sys
import os

_CAP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CAP_DIR)

from service import MultiLanguageLocalisationService
from models import (
	FormattingRuleCreate,
	LocaleConfigCreate,
	LocaleConfigUpdate,
	MlgAgentCreate,
	TerminologyCreate,
	TranslationCreate,
)


def _run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def _svc():
	return MultiLanguageLocalisationService()


TENANT = "test_tenant"


def _en_ke_payload(**kw):
	return LocaleConfigCreate(
		tenant_id=TENANT,
		locale_code="en_KE",
		language="en",
		script="latin",
		text_direction="ltr",
		date_format="DD/MM/YYYY",
		number_format="1,234.56",
		is_default=kw.get("is_default", False),
	)


def _ar_sa_payload():
	return LocaleConfigCreate(
		tenant_id=TENANT,
		locale_code="ar_SA",
		language="ar",
		script="arabic",
		text_direction="rtl",
		date_format="DD/MM/YYYY",
		number_format="1,234.56",
		is_rtl=True,
	)


def test_configure_locale():
	svc = _svc()
	locale = _run(svc.configure_locale(_en_ke_payload()))
	assert locale.locale_code == "en_KE"
	assert locale.language == "en"
	assert locale.is_active is True


def test_configure_rtl_locale():
	svc = _svc()
	locale = _run(svc.configure_locale(_ar_sa_payload()))
	assert locale.is_rtl is True
	assert locale.text_direction == "rtl"


def test_rtl_language_requires_rtl_direction():
	try:
		LocaleConfigCreate(
			tenant_id=TENANT,
			locale_code="ar_SA",
			language="ar",
			script="arabic",
			text_direction="ltr",  # wrong
			date_format="DD/MM/YYYY",
			number_format="1,234.56",
			is_rtl=False,
		)
		assert False, "expected AssertionError — ar requires rtl direction"
	except (AssertionError, ValueError):
		pass


def test_list_locales_empty():
	svc = _svc()
	assert _run(svc.list_locales(TENANT)) == []


def test_list_locales_by_language():
	svc = _svc()
	_run(svc.configure_locale(_en_ke_payload()))
	_run(svc.configure_locale(_ar_sa_payload()))
	en_locales = _run(svc.list_locales(TENANT, language="en"))
	assert len(en_locales) == 1


def test_list_rtl_locales():
	svc = _svc()
	_run(svc.configure_locale(_en_ke_payload()))
	_run(svc.configure_locale(_ar_sa_payload()))
	rtl = _run(svc.list_locales(TENANT, is_rtl=True))
	assert len(rtl) == 1
	assert rtl[0].language == "ar"


def test_only_one_default_locale():
	svc = _svc()
	l1 = _run(svc.configure_locale(_en_ke_payload(is_default=True)))
	# Configure a second default — first should become non-default
	l2 = _run(svc.configure_locale(LocaleConfigCreate(
		tenant_id=TENANT,
		locale_code="en_US",
		language="en",
		script="latin",
		text_direction="ltr",
		date_format="MM/DD/YYYY",
		number_format="1,234.56",
		is_default=True,
	)))
	refreshed_l1 = _run(svc.get_locale(TENANT, l1.id))
	assert refreshed_l1.is_default is False
	assert l2.is_default is True


def test_create_translation():
	svc = _svc()
	tr = _run(svc.create_translation(TranslationCreate(
		tenant_id=TENANT,
		translation_key="button.save",
		source_language="en",
		target_language="sw",
		content_type="ui_string",
		source_text="Save",
		translated_text="Hifadhi",
		translator_id="translator_001",
	)))
	assert tr.translation_key == "button.save"
	assert tr.status == "draft"


def test_self_translation_rejected():
	try:
		TranslationCreate(
			tenant_id=TENANT,
			translation_key="test.key",
			source_language="en",
			target_language="en",
			content_type="ui_string",
			source_text="Hello",
			translated_text="Hello",
			translator_id="t1",
		)
		assert False, "expected AssertionError"
	except (AssertionError, ValueError):
		pass


def test_translation_workflow():
	svc = _svc()
	tr = _run(svc.create_translation(TranslationCreate(
		tenant_id=TENANT,
		translation_key="nav.home",
		source_language="en",
		target_language="fr",
		content_type="ui_string",
		source_text="Home",
		translated_text="Accueil",
		translator_id="translator_001",
	)))
	# Submit
	submitted = _run(svc.submit_translation_for_review(TENANT, tr.id))
	assert submitted.status == "pending_review"
	# Approve by different reviewer
	approved = _run(svc.approve_translation(TENANT, tr.id, reviewer_id="reviewer_002"))
	assert approved.status == "approved"
	# Publish
	published = _run(svc.publish_translation(TENANT, tr.id))
	assert published.status == "published"


def test_self_review_denied():
	svc = _svc()
	tr = _run(svc.create_translation(TranslationCreate(
		tenant_id=TENANT,
		translation_key="test.key",
		source_language="en",
		target_language="de",
		content_type="ui_string",
		source_text="Test",
		translated_text="Test DE",
		translator_id="translator_001",
	)))
	_run(svc.submit_translation_for_review(TENANT, tr.id))
	try:
		_run(svc.approve_translation(TENANT, tr.id, reviewer_id="translator_001"))
		assert False, "expected PermissionError — self-review"
	except PermissionError:
		pass


def test_publish_requires_approved_status():
	svc = _svc()
	tr = _run(svc.create_translation(TranslationCreate(
		tenant_id=TENANT,
		translation_key="test.k2",
		source_language="en",
		target_language="sw",
		content_type="ui_string",
		source_text="Cancel",
		translated_text="Ghairi",
		translator_id="t1",
	)))
	try:
		_run(svc.publish_translation(TENANT, tr.id))
		assert False, "expected PermissionError — not approved"
	except PermissionError:
		pass


def test_lookup_translation():
	svc = _svc()
	tr = _run(svc.create_translation(TranslationCreate(
		tenant_id=TENANT,
		translation_key="footer.copyright",
		source_language="en",
		target_language="sw",
		content_type="ui_string",
		source_text="Copyright 2026",
		translated_text="Hakimiliki 2026",
		translator_id="t1",
	)))
	_run(svc.submit_translation_for_review(TENANT, tr.id))
	_run(svc.approve_translation(TENANT, tr.id, reviewer_id="r1"))
	_run(svc.publish_translation(TENANT, tr.id))
	found = _run(svc.lookup_translation(TENANT, "footer.copyright", "sw"))
	assert found is not None
	assert found.translated_text == "Hakimiliki 2026"


def test_configure_formatting():
	svc = _svc()
	locale = _run(svc.configure_locale(_en_ke_payload()))
	rule = _run(svc.configure_formatting(FormattingRuleCreate(
		tenant_id=TENANT,
		locale_id=locale.id,
		date_format="DD/MM/YYYY",
		number_format="1,234.56",
		thousand_separator=",",
		decimal_separator=".",
		time_format_24h=True,
		first_day_of_week=1,
	)))
	assert rule.date_format == "DD/MM/YYYY"
	assert rule.time_format_24h is True


def test_add_and_search_terminology():
	svc = _svc()
	_run(svc.add_terminology(TerminologyCreate(
		tenant_id=TENANT,
		term="Invoice",
		language="en",
		definition="A commercial document issued by a seller to a buyer",
		domain="finance",
		preferred_translation="Ankara",
	)))
	results = _run(svc.search_terminology(TENANT, "inv"))
	assert len(results) == 1
	assert results[0].term == "Invoice"


def test_register_agent():
	svc = _svc()
	agent = _run(svc.register_agent(MlgAgentCreate(
		tenant_id=TENANT,
		name="TranslationBot",
		runtime="claude_code",
		role="translation_assistant",
		scope="UI string translation",
	)))
	assert agent.name == "TranslationBot"


def test_dashboard_summary():
	svc = _svc()
	_run(svc.configure_locale(_en_ke_payload()))
	summary = _run(svc.dashboard_summary(TENANT))
	assert summary["locale_count"] == 1
	assert summary["tenant_id"] == TENANT


def test_cross_tenant_isolation():
	svc = _svc()
	_run(svc.configure_locale(_en_ke_payload()))
	other = _run(svc.list_locales("other_tenant"))
	assert other == []


def test_audit_events_recorded():
	svc = _svc()
	_run(svc.configure_locale(_en_ke_payload()))
	events = _run(svc.list_audit_events(TENANT))
	assert len(events) >= 1
