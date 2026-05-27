from __future__ import annotations

import asyncio
import importlib
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from capabilities.crm.adv import api as crm_api
from capabilities.crm.adv.database import DatabaseManager
from capabilities.crm.adv.models import (
	ActivityType,
	AccountType,
	CRMAccount,
	CRMActivity,
	CRMLead,
	CRMOpportunity,
	LeadSource,
	LeadStatus,
	OpportunityStage,
)
from capabilities.crm.adv.service import CRMService


def test_crm_package_imports_reserved_sales_forecasting_module():
	import capabilities.crm as crm

	assert "for" in crm.__all__


def test_crm_adv_records_work_without_postgres_pool():
	async def exercise():
		db = DatabaseManager()

		lead = await db.create_lead(
			CRMLead(
				tenant_id="tenant-a",
				created_by="seller-1",
				first_name="Ada",
				last_name="Sales",
				email="ada@example.com",
				lead_source=LeadSource.WEBSITE,
			)
		)
		qualified = await db.update_lead(
			lead.id,
			{"lead_status": LeadStatus.QUALIFIED, "updated_by": "seller-2", "version": lead.version + 1},
			"tenant-a",
		)

		assert qualified.lead_status == LeadStatus.QUALIFIED
		assert qualified.version == 2
		assert await db.get_lead(lead.id, "tenant-b") is None

		account = await db.create_account(
			CRMAccount(
				tenant_id="tenant-a",
				created_by="seller-1",
				account_name="Acme Ltd",
				account_owner_id="seller-1",
			)
		)
		assert (await db.get_account(account.id, "tenant-a")).account_name == "Acme Ltd"

		opportunity = await db.create_opportunity(
			CRMOpportunity(
				tenant_id="tenant-a",
				created_by="seller-1",
				opportunity_name="Core ERP rollout",
				amount=Decimal("25000.00"),
				probability=40,
				close_date=date.today(),
				account_id=account.id,
				owner_id="seller-1",
			)
		)
		assert opportunity.expected_revenue == Decimal("10000.000")

		closed = await db.update_opportunity(
			opportunity.id,
			{
				"stage": OpportunityStage.CLOSED_WON,
				"is_closed": True,
				"is_won": True,
				"updated_by": "seller-2",
				"version": opportunity.version + 1,
			},
			"tenant-a",
		)
		assert closed.stage == OpportunityStage.CLOSED_WON
		assert closed.is_won is True

		activity = await db.create_activity(
			CRMActivity(
				tenant_id="tenant-a",
				created_by="seller-1",
				subject="Close plan call",
				activity_type=ActivityType.CALL,
				start_datetime=datetime.utcnow(),
				related_to_type="opportunity",
				related_to_id=opportunity.id,
				assigned_to_id="seller-1",
			)
		)
		assert activity.related_to_id == opportunity.id

	asyncio.run(exercise())


def test_crm_service_imports_with_optional_integrations_absent():
	async def exercise():
		service = CRMService()

		assert type(service.email_integration_manager).__name__ == "EmailIntegrationManager"
		assert type(service.realtime_sync).__name__ == "RealTimeSyncEngine"

		lead = await service.create_lead(
			{
				"first_name": "Maya",
				"last_name": "Buyer",
				"email": "maya@example.com",
				"lead_source": LeadSource.REFERRAL,
			},
			"tenant-service",
			"seller-1",
		)
		updated = await service.update_lead(
			lead.id,
			{"lead_status": LeadStatus.QUALIFIED},
			"tenant-service",
			"seller-2",
		)

		assert updated.lead_status == LeadStatus.QUALIFIED
		assert updated.version == 2

	asyncio.run(exercise())


def test_crm_api_lists_core_records_without_placeholders():
	async def exercise():
		service = CRMService()

		account = await service.create_account(
			{
				"account_name": "Acme Manufacturing",
				"account_type": AccountType.CUSTOMER,
				"industry": "industrial",
				"account_owner_id": "seller-1",
			},
			"tenant-api",
			"seller-1",
		)
		await service.create_account(
			{
				"account_name": "Other Tenant Account",
				"account_type": AccountType.CUSTOMER,
				"account_owner_id": "seller-2",
			},
			"tenant-other",
			"seller-2",
		)
		lead = await service.create_lead(
			{
				"first_name": "Priya",
				"last_name": "Buyer",
				"company": "Acme Manufacturing",
				"email": "priya@example.com",
				"lead_source": LeadSource.WEBSITE,
				"owner_id": "seller-1",
			},
			"tenant-api",
			"seller-1",
		)
		opportunity = await service.create_opportunity(
			{
				"opportunity_name": "Acme ERP Expansion",
				"amount": Decimal("75000.00"),
				"probability": 60,
				"close_date": date.today(),
				"account_id": account.id,
				"owner_id": "seller-1",
			},
			"tenant-api",
			"seller-1",
		)
		activity = await service.db_manager.create_activity(
			CRMActivity(
				tenant_id="tenant-api",
				created_by="seller-1",
				subject="Acme expansion call",
				activity_type=ActivityType.CALL,
				start_datetime=datetime.utcnow(),
				related_to_type="opportunity",
				related_to_id=opportunity.id,
				assigned_to_id="seller-1",
			)
		)

		accounts = await crm_api.get_accounts(
			search_term="acme",
			account_type=AccountType.CUSTOMER,
			owner_id=None,
			page=1,
			page_size=10,
			service=service,
			tenant_id="tenant-api",
		)
		leads = await crm_api.get_leads(
			search_term="priya",
			lead_source=None,
			lead_status=LeadStatus.NEW,
			owner_id="seller-1",
			page=1,
			page_size=10,
			service=service,
			tenant_id="tenant-api",
		)
		opportunities = await crm_api.get_opportunities(
			search_term="expansion",
			stage=OpportunityStage.PROSPECTING,
			account_id=account.id,
			owner_id=None,
			page=1,
			page_size=10,
			service=service,
			tenant_id="tenant-api",
		)
		activities = await crm_api.get_activities(
			search_term="call",
			activity_type=ActivityType.CALL,
			related_to_type="opportunity",
			related_to_id=opportunity.id,
			assigned_to_id="seller-1",
			page=1,
			page_size=10,
			service=service,
			tenant_id="tenant-api",
		)
		health = await crm_api.health_check(service=service)
		dashboard = await crm_api.get_dashboard(
			service=service,
			tenant_id="tenant-api",
			user_id="seller-1",
		)
		pipeline = await crm_api.get_pipeline_analytics(
			service=service,
			tenant_id="tenant-api",
			user_id="seller-1",
		)
		clock_in = await crm_api.clock_in(
			clock_in_data=crm_api.ClockInRequest(
				location={"latitude": -1.286389, "longitude": 36.817223},
				device_info={"device": "test-terminal"},
				notes="Start field sales route",
			),
			service=service,
			tenant_id="tenant-api",
			user_id="seller-1",
		)
		metrics = await crm_api.get_metrics(service=service, tenant_id="tenant-api")
		default_config = await crm_api.get_configuration(
			service=service,
			tenant_id="tenant-api",
		)
		updated_config = await crm_api.update_configuration(
			update_data={
				"default_lead_score_threshold": 82.5,
				"max_records_per_page": 250,
				"ai_recommendations_enabled": False,
			},
			service=service,
			tenant_id="tenant-api",
			user_id="admin-1",
		)
		other_tenant_config = await crm_api.get_configuration(
			service=service,
			tenant_id="tenant-other",
		)

		assert [item["id"] for item in accounts.data["items"]] == [account.id]
		assert accounts.data["total_count"] == 1
		assert accounts.data["items"][0]["account_name"] == "Acme Manufacturing"
		assert [item["id"] for item in leads.data["items"]] == [lead.id]
		assert leads.data["items"][0]["lead_status"] == LeadStatus.NEW
		assert [item["id"] for item in opportunities.data["items"]] == [opportunity.id]
		assert opportunities.data["items"][0]["expected_revenue"] == "45000.000"
		assert [item["id"] for item in activities.data["items"]] == [activity.id]
		assert health.uptime_seconds >= 0
		assert dashboard.data["record_counts"]["opportunities"] == 1
		assert dashboard.data["pipeline_value"] == "75000.00"
		assert dashboard.data["weighted_pipeline_value"] == "45000.000"
		assert dashboard.data["opportunity_stage_breakdown"] == {"prospecting": 1}
		assert pipeline.data["opportunity_count"] == 1
		assert pipeline.data["stage_breakdown"] == {"prospecting": 1}
		assert pipeline.data["weighted_pipeline_value"] == "45000.000"
		assert clock_in.data["status"] == "clocked_in"
		assert clock_in.data["tenant_id"] == "tenant-api"
		assert clock_in.data["user_id"] == "seller-1"
		assert metrics.data["record_counts"] == {
			"contacts": 0,
			"accounts": 1,
			"leads": 1,
			"opportunities": 1,
			"activities": 1,
			"time_entries": 1,
		}
		assert default_config.data["default_lead_score_threshold"] == 70.0
		assert updated_config.data["default_lead_score_threshold"] == 82.5
		assert updated_config.data["max_records_per_page"] == 250
		assert updated_config.data["ai_recommendations_enabled"] is False
		assert other_tenant_config.data["default_lead_score_threshold"] == 70.0

	asyncio.run(exercise())


def test_optional_crm_integration_modules_import_standalone():
	modules = [
		path.stem
		for path in sorted(Path("capabilities/crm/adv").glob("*.py"))
		if not path.name.startswith("__")
	]

	for module_name in modules:
		module = importlib.import_module(f"capabilities.crm.adv.{module_name}")
		assert module is not None
