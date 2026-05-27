from __future__ import annotations

import asyncio
from datetime import date, datetime
from decimal import Decimal

from capabilities.crm.adv.database import DatabaseManager
from capabilities.crm.adv.models import (
	ActivityType,
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
