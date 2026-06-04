"""Service-level tests for APG Multi-Country Operations."""

from __future__ import annotations

import asyncio
import sys
import os
from datetime import date

_CAP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CAP_DIR)

from service import MultiCountryOperationsService
from models import (
	CountryCreate,
	EntityCreate,
	ComplianceMappingCreate,
	IntercompanyTransactionCreate,
	StatutoryReportCreate,
	McoAgentCreate,
	CountryUpdate,
	EntityUpdate,
	ComplianceMappingUpdate,
)


def _run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def _svc():
	return MultiCountryOperationsService()


TENANT = "test_tenant"


def _country_payload(**kwargs):
	return CountryCreate(
		tenant_id=TENANT,
		name=kwargs.get("name", "Kenya"),
		jurisdiction=kwargs.get("jurisdiction", "ke"),
		functional_currency=kwargs.get("functional_currency", "KES"),
		regulatory_framework=kwargs.get("regulatory_framework", "ifrs"),
	)


def _entity_payload(country_id: str, **kwargs):
	return EntityCreate(
		tenant_id=TENANT,
		name=kwargs.get("name", "Acme Kenya Ltd"),
		entity_type=kwargs.get("entity_type", "subsidiary"),
		country_id=country_id,
		registration_number=kwargs.get("registration_number", "CPR/2020/123456"),
		functional_currency=kwargs.get("functional_currency", "KES"),
	)


def test_register_country():
	svc = _svc()
	country = _run(svc.register_country(_country_payload()))
	assert country.jurisdiction == "ke"
	assert country.functional_currency == "KES"
	assert country.tenant_id == TENANT
	assert country.status == "active"


def test_register_country_uppercase_currency_normalised():
	svc = _svc()
	payload = _country_payload(functional_currency="kes")
	country = _run(svc.register_country(payload))
	assert country.functional_currency == "KES"


def test_list_countries_empty():
	svc = _svc()
	countries = _run(svc.list_countries(TENANT))
	assert countries == []


def test_list_countries_after_register():
	svc = _svc()
	_run(svc.register_country(_country_payload()))
	countries = _run(svc.list_countries(TENANT))
	assert len(countries) == 1


def test_list_countries_status_filter():
	svc = _svc()
	c = _run(svc.register_country(_country_payload()))
	_run(svc.update_country(TENANT, c.id, CountryUpdate(status="inactive")))
	active = _run(svc.list_countries(TENANT, status="active"))
	inactive = _run(svc.list_countries(TENANT, status="inactive"))
	assert len(active) == 0
	assert len(inactive) == 1


def test_get_country_not_found():
	svc = _svc()
	try:
		_run(svc.get_country(TENANT, "nonexistent"))
		assert False, "expected KeyError"
	except KeyError:
		pass


def test_register_entity_requires_existing_country():
	svc = _svc()
	try:
		_run(svc.register_entity(_entity_payload("no_such_country")))
		assert False, "expected PermissionError"
	except PermissionError:
		pass


def test_register_entity_success():
	svc = _svc()
	country = _run(svc.register_country(_country_payload()))
	entity = _run(svc.register_entity(_entity_payload(country.id)))
	assert entity.entity_type == "subsidiary"
	assert entity.country_id == country.id
	assert entity.is_active is True


def test_list_entities_by_country():
	svc = _svc()
	c_ke = _run(svc.register_country(_country_payload(name="Kenya", jurisdiction="ke")))
	c_ug = _run(svc.register_country(_country_payload(name="Uganda", jurisdiction="ug", functional_currency="UGX")))
	_run(svc.register_entity(_entity_payload(c_ke.id, name="Acme Kenya")))
	_run(svc.register_entity(_entity_payload(c_ug.id, name="Acme Uganda", functional_currency="UGX")))
	ke_entities = _run(svc.list_entities(TENANT, country_id=c_ke.id))
	assert len(ke_entities) == 1
	assert ke_entities[0].name == "Acme Kenya"


def test_record_compliance_mapping():
	svc = _svc()
	country = _run(svc.register_country(_country_payload()))
	entity = _run(svc.register_entity(_entity_payload(country.id)))
	mapping = _run(svc.record_compliance_mapping(ComplianceMappingCreate(
		tenant_id=TENANT,
		entity_id=entity.id,
		domain="tax",
		framework="ifrs",
		owner_id="cfo_001",
		next_review_date=date(2026, 12, 31),
		evidence_reference="tax_cert_2025",
	)))
	assert mapping.domain == "tax"
	assert mapping.status == "under_review"


def test_compliance_domain_invalid():
	from pydantic import ValidationError
	try:
		ComplianceMappingCreate(
			tenant_id=TENANT,
			entity_id="e1",
			domain="invalid_domain",
			framework="ifrs",
			owner_id="o1",
			next_review_date=date(2026, 1, 1),
			evidence_reference="ref",
		)
		assert False, "expected ValidationError"
	except (AssertionError, ValidationError):
		pass


def test_create_intercompany_transaction():
	svc = _svc()
	c_ke = _run(svc.register_country(_country_payload()))
	c_ug = _run(svc.register_country(_country_payload(name="Uganda", jurisdiction="ug", functional_currency="UGX")))
	orig = _run(svc.register_entity(_entity_payload(c_ke.id, name="Parent")))
	counter = _run(svc.register_entity(_entity_payload(c_ug.id, name="Sub", functional_currency="UGX")))
	txn = _run(svc.create_intercompany_transaction(IntercompanyTransactionCreate(
		tenant_id=TENANT,
		transaction_type="management_fee",
		originator_entity_id=orig.id,
		counterparty_entity_id=counter.id,
		amount=50000.0,
		currency="USD",
		transaction_date=date(2026, 3, 31),
		transfer_pricing_method="cost_plus",
		description="Annual management fee recharge",
	)))
	assert txn.status == "draft"
	assert txn.amount == 50000.0


def test_approve_intercompany_requires_pending_status():
	svc = _svc()
	c = _run(svc.register_country(_country_payload()))
	orig = _run(svc.register_entity(_entity_payload(c.id, name="A")))
	counter = _run(svc.register_entity(_entity_payload(c.id, name="B")))
	txn = _run(svc.create_intercompany_transaction(IntercompanyTransactionCreate(
		tenant_id=TENANT,
		transaction_type="loan",
		originator_entity_id=orig.id,
		counterparty_entity_id=counter.id,
		amount=1000.0,
		currency="KES",
		transaction_date=date(2026, 1, 1),
		transfer_pricing_method="comparable_uncontrolled_price",
		description="Intercompany loan",
	)))
	try:
		_run(svc.approve_intercompany_transaction(TENANT, txn.id, "approver_1", "REF_001"))
		assert False, "expected AssertionError — status must be pending_approval"
	except AssertionError:
		pass


def test_create_statutory_report():
	svc = _svc()
	country = _run(svc.register_country(_country_payload()))
	entity = _run(svc.register_entity(_entity_payload(country.id)))
	report = _run(svc.create_statutory_report(StatutoryReportCreate(
		tenant_id=TENANT,
		entity_id=entity.id,
		report_type="annual_return",
		period_start=date(2025, 1, 1),
		period_end=date(2025, 12, 31),
		due_date=date(2026, 3, 31),
		filer_id="sec_officer_001",
	)))
	assert report.status == "draft"
	assert report.report_type == "annual_return"


def test_file_statutory_report():
	svc = _svc()
	country = _run(svc.register_country(_country_payload()))
	entity = _run(svc.register_entity(_entity_payload(country.id)))
	report = _run(svc.create_statutory_report(StatutoryReportCreate(
		tenant_id=TENANT,
		entity_id=entity.id,
		report_type="tax_return",
		period_start=date(2025, 1, 1),
		period_end=date(2025, 12, 31),
		due_date=date(2026, 4, 30),
		filer_id="tax_officer_001",
	)))
	filed = _run(svc.file_statutory_report(TENANT, report.id, "tax_officer_001", date(2026, 4, 15)))
	assert filed.status == "filed"
	assert filed.filed_date == date(2026, 4, 15)


def test_register_agent():
	svc = _svc()
	agent = _run(svc.register_agent(McoAgentCreate(
		tenant_id=TENANT,
		name="ComplianceBot",
		runtime="claude_code",
		role="compliance_monitor",
		scope="statutory filing oversight",
	)))
	assert agent.name == "ComplianceBot"
	assert agent.runtime == "claude_code"


def test_dashboard_summary():
	svc = _svc()
	country = _run(svc.register_country(_country_payload()))
	_run(svc.register_entity(_entity_payload(country.id)))
	summary = _run(svc.dashboard_summary(TENANT))
	assert summary["country_count"] == 1
	assert summary["entity_count"] == 1
	assert summary["tenant_id"] == TENANT


def test_audit_events_recorded():
	svc = _svc()
	_run(svc.register_country(_country_payload()))
	events = _run(svc.list_audit_events(TENANT))
	assert len(events) >= 1
	assert events[0]["event_type"] == "country_registered"


def test_cross_tenant_isolation():
	svc = _svc()
	_run(svc.register_country(_country_payload()))
	other_countries = _run(svc.list_countries("other_tenant"))
	assert other_countries == []


def test_validate_agent_action_privileged_no_approval():
	svc = _svc()
	try:
		_run(svc.validate_agent_action(TENANT, privileged_scope=True, human_approval_recorded=False))
		assert False, "expected PermissionError"
	except PermissionError:
		pass


def test_validate_agent_action_approved():
	svc = _svc()
	result = _run(svc.validate_agent_action(TENANT, privileged_scope=True, human_approval_recorded=True))
	assert result["accepted"] is True
