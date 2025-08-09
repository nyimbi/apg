"""
APG Customer Relationship Management - Integration Tests

Comprehensive integration tests for CRM system components including
end-to-end workflows, APG integrations, and system-level functionality.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid_extensions import uuid7str

from ..capability import CustomerRelationshipManagementCapability
from ..service import CRMService
from ..database import DatabaseManager
from ..models import CRMContact, CRMLead, CRMOpportunity, LeadStatus, OpportunityStage, ContactType
from . import TEST_TENANT_ID, TEST_USER_ID


@pytest.mark.integration
class TestCapabilityIntegration:
	"""Test complete capability integration"""
	
	@pytest.mark.asyncio
	async def test_capability_initialization(self, test_database):
		"""Test complete capability initialization"""
		# Mock APG dependencies
		mock_gateway = AsyncMock()
		mock_service_discovery = AsyncMock()
		mock_event_bus = AsyncMock()
		mock_config_manager = AsyncMock()
		
		with patch('..capability.APGGateway', return_value=mock_gateway), \
			 patch('..capability.ServiceDiscovery', return_value=mock_service_discovery), \
			 patch('..capability.EventBus', return_value=mock_event_bus), \
			 patch('..capability.ConfigurationManager', return_value=mock_config_manager):
			
			capability = CustomerRelationshipManagementCapability()
			
			# Mock configuration
			mock_config_manager.get_config.return_value = {
				"database": test_database,
				"ai_insights": {"enabled": True},
				"analytics": {"enabled": True}
			}
			
			await capability.initialize()
			
			assert capability.is_healthy()
			
			# Verify all components are initialized
			health = await capability.health_check()
			assert health["status"] == "healthy"
			assert health["components"]["database"]["status"] == "healthy"
			assert health["components"]["service"]["status"] == "healthy"
			
			await capability.shutdown()
	
	@pytest.mark.asyncio
	async def test_capability_service_integration(self, test_database):
		"""Test capability service integration"""
		# Create individual components
		database_manager = DatabaseManager(test_database)
		await database_manager.initialize()
		
		mock_event_bus = AsyncMock()
		mock_ai_insights = AsyncMock()
		mock_analytics = AsyncMock()
		
		service = CRMService(
			database_manager=database_manager,
			event_bus=mock_event_bus,
			ai_insights=mock_ai_insights,
			analytics=mock_analytics
		)
		
		await service.initialize()
		
		# Test service integration
		health = await service.health_check()
		assert health["status"] == "healthy"
		
		await service.shutdown()
		await database_manager.shutdown()


@pytest.mark.integration
class TestEndToEndWorkflows:
	"""Test complete end-to-end business workflows"""
	
	@pytest.mark.asyncio
	async def test_complete_sales_cycle(self, crm_service, test_user_context, clean_database):
		"""Test complete sales cycle from lead to closed opportunity"""
		# 1. Create a lead
		lead_data = CRMLead(
			id=uuid7str(),
			tenant_id=TEST_TENANT_ID,
			first_name="John",
			last_name="Prospect",
			email="john.prospect@example.com",
			company="Prospect Corp",
			budget=Decimal('100000.00'),
			timeline="Q2 2025",
			lead_status=LeadStatus.NEW,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		created_lead = await crm_service.create_lead(lead_data, test_user_context)
		assert created_lead is not None
		assert created_lead.lead_status == LeadStatus.NEW
		
		# 2. Qualify the lead
		qualified_lead = await crm_service.qualify_lead(created_lead.id, test_user_context)
		assert qualified_lead.lead_status == LeadStatus.QUALIFIED
		
		# 3. Convert lead to contact and opportunity
		conversion_data = {
			"create_contact": True,
			"create_account": True,
			"create_opportunity": True,
			"opportunity_name": f"Opportunity from {qualified_lead.company}",
			"opportunity_amount": float(qualified_lead.budget)
		}
		
		conversion_result = await crm_service.convert_lead(
			qualified_lead.id, 
			conversion_data, 
			test_user_context
		)
		
		assert conversion_result is not None
		assert "contact_id" in conversion_result
		assert "opportunity_id" in conversion_result
		
		# 4. Verify contact was created
		contact = await crm_service.get_contact(conversion_result["contact_id"], test_user_context)
		assert contact is not None
		assert contact.first_name == qualified_lead.first_name
		assert contact.email == qualified_lead.email
		assert contact.contact_type == ContactType.LEAD  # Converted from lead
		
		# 5. Verify opportunity was created
		opportunity = await crm_service.get_opportunity(
			conversion_result["opportunity_id"], 
			test_user_context
		)
		assert opportunity is not None
		assert float(opportunity.amount) == 100000.00
		assert opportunity.stage == OpportunityStage.PROSPECTING
		
		# 6. Progress opportunity through sales stages
		stages = [
			OpportunityStage.QUALIFICATION,
			OpportunityStage.NEEDS_ANALYSIS,
			OpportunityStage.VALUE_PROPOSITION,
			OpportunityStage.PROPOSAL,
			OpportunityStage.NEGOTIATION
		]
		
		current_opportunity = opportunity
		for stage in stages:
			current_opportunity = await crm_service.update_opportunity_stage(
				current_opportunity.id,
				stage,
				test_user_context
			)
			assert current_opportunity.stage == stage
		
		# 7. Close opportunity as won
		final_opportunity = await crm_service.close_opportunity(
			current_opportunity.id,
			True,  # Won
			test_user_context
		)
		
		assert final_opportunity.is_closed is True
		assert final_opportunity.stage == OpportunityStage.CLOSED_WON
		assert final_opportunity.closed_at is not None
		
		# 8. Verify lead is marked as converted
		final_lead = await crm_service.get_lead(created_lead.id, test_user_context)
		assert final_lead.converted_at is not None
		assert final_lead.converted_contact_id == conversion_result["contact_id"]
		assert final_lead.converted_opportunity_id == conversion_result["opportunity_id"]
	
	@pytest.mark.asyncio
	async def test_marketing_campaign_workflow(self, crm_service, test_user_context, clean_database):
		"""Test complete marketing campaign workflow"""
		from ..models import CRMCampaign, CampaignType, CampaignStatus
		
		# 1. Create campaign
		campaign_data = CRMCampaign(
			id=uuid7str(),
			tenant_id=TEST_TENANT_ID,
			campaign_name="Q2 Email Campaign",
			campaign_type=CampaignType.EMAIL,
			status=CampaignStatus.DRAFT,
			description="Quarterly email marketing campaign",
			start_date=date.today(),
			end_date=date.today() + timedelta(days=30),
			budget=Decimal('10000.00'),
			expected_response_rate=Decimal('5.0'),
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		created_campaign = await crm_service.create_campaign(campaign_data, test_user_context)
		assert created_campaign is not None
		assert created_campaign.status == CampaignStatus.DRAFT
		
		# 2. Create target contacts
		contacts = []
		for i in range(5):
			contact_data = CRMContact(
				id=uuid7str(),
				tenant_id=TEST_TENANT_ID,
				first_name=f"Contact{i:02d}",
				last_name="Prospect",
				email=f"contact{i:02d}@example.com",
				company=f"Company{i:02d}",
				contact_type=ContactType.PROSPECT,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			
			contact = await crm_service.create_contact(contact_data, test_user_context)
			contacts.append(contact)
		
		# 3. Add contacts to campaign
		for contact in contacts:
			await crm_service.add_contact_to_campaign(
				created_campaign.id,
				contact.id,
				test_user_context
			)
		
		# 4. Activate campaign
		activated_campaign = await crm_service.update_campaign_status(
			created_campaign.id,
			CampaignStatus.ACTIVE,
			test_user_context
		)
		assert activated_campaign.status == CampaignStatus.ACTIVE
		
		# 5. Process campaign responses (simulate some leads)
		generated_leads = []
		for i in range(2):  # 2 out of 5 respond
			lead_data = CRMLead(
				id=uuid7str(),
				tenant_id=TEST_TENANT_ID,
				first_name=f"Lead{i:02d}",
				last_name="Generated",
				email=f"lead{i:02d}@example.com",
				company=f"LeadCompany{i:02d}",
				lead_source="email_campaign",
				campaign_id=created_campaign.id,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			
			lead = await crm_service.create_lead(lead_data, test_user_context)
			generated_leads.append(lead)
		
		# 6. Complete campaign
		completed_campaign = await crm_service.update_campaign_status(
			created_campaign.id,
			CampaignStatus.COMPLETED,
			test_user_context
		)
		assert completed_campaign.status == CampaignStatus.COMPLETED
		
		# 7. Verify campaign results
		campaign_results = await crm_service.get_campaign_results(
			created_campaign.id,
			test_user_context
		)
		assert campaign_results["leads_generated"] == 2
		assert campaign_results["response_rate"] > 0
	
	@pytest.mark.asyncio
	async def test_customer_service_workflow(self, crm_service, test_user_context, clean_database):
		"""Test customer service workflow with activities"""
		from ..models import CRMActivity, ActivityType, ActivityStatus
		
		# 1. Create customer contact
		customer_data = CRMContact(
			id=uuid7str(),
			tenant_id=TEST_TENANT_ID,
			first_name="Jane",
			last_name="Customer",
			email="jane.customer@example.com",
			company="Customer Corp",
			contact_type=ContactType.CUSTOMER,
			customer_health_score=Decimal('75.0'),
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		customer = await crm_service.create_contact(customer_data, test_user_context)
		assert customer is not None
		
		# 2. Create support activities
		activities = []
		activity_types = [ActivityType.CALL, ActivityType.EMAIL, ActivityType.MEETING]
		
		for i, activity_type in enumerate(activity_types):
			activity_data = CRMActivity(
				id=uuid7str(),
				tenant_id=TEST_TENANT_ID,
				activity_type=activity_type,
				subject=f"Customer support {activity_type.value} #{i+1}",
				description=f"Support activity for customer issue #{i+1}",
				status=ActivityStatus.PLANNED,
				contact_id=customer.id,
				due_date=datetime.utcnow() + timedelta(days=i+1),
				duration_minutes=30,
				owner_id=TEST_USER_ID,
				assigned_to_id=TEST_USER_ID,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			
			activity = await crm_service.create_activity(activity_data, test_user_context)
			activities.append(activity)
		
		# 3. Complete activities
		for activity in activities:
			completed_activity = await crm_service.complete_activity(
				activity.id,
				f"Successfully resolved {activity.subject}",
				test_user_context
			)
			assert completed_activity.status.value == "completed"
			assert completed_activity.completed_at is not None
		
		# 4. Update customer health score based on interactions
		updated_customer = await crm_service.update_contact(
			customer.id,
			{"customer_health_score": 90.0},
			test_user_context
		)
		assert updated_customer.customer_health_score == 90.0
		
		# 5. Verify activity history
		contact_activities = await crm_service.get_contact_activities(
			customer.id,
			test_user_context
		)
		assert len(contact_activities["items"]) == 3
		assert all(activity.status.value == "completed" for activity in contact_activities["items"])


@pytest.mark.integration
class TestAPGIntegrations:
	"""Test APG ecosystem integrations"""
	
	@pytest.mark.asyncio
	async def test_event_bus_integration(self, crm_service, test_user_context, clean_database):
		"""Test event publishing and subscription"""
		# Create contact should publish events
		contact_data = CRMContact(
			id=uuid7str(),
			tenant_id=TEST_TENANT_ID,
			first_name="Event",
			last_name="Test",
			email="event.test@example.com",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		created_contact = await crm_service.create_contact(contact_data, test_user_context)
		
		# Verify event was published
		crm_service.event_bus.publish.assert_called()
		
		# Check event content
		published_events = crm_service.event_bus.publish.call_args_list
		contact_created_event = None
		
		for call in published_events:
			event = call[0][0]
			if hasattr(event, 'event_type') and event.event_type == "crm.contact.created":
				contact_created_event = event
				break
		
		assert contact_created_event is not None
		assert contact_created_event.data["contact_id"] == created_contact.id
		assert contact_created_event.data["first_name"] == "Event"
	
	@pytest.mark.asyncio
	async def test_ai_insights_integration(self, crm_service, test_user_context, clean_database):
		"""Test AI insights integration"""
		# Mock AI insights
		mock_insights = {
			"engagement_score": 85.0,
			"conversion_probability": 0.75,
			"recommended_actions": [
				"Schedule follow-up call",
				"Send product demo",
				"Provide case studies"
			],
			"communication_preferences": {
				"preferred_channel": "email",
				"best_contact_time": "morning"
			}
		}
		crm_service.ai_insights.generate_contact_insights.return_value = mock_insights
		
		# Create contact
		contact_data = CRMContact(
			id=uuid7str(),
			tenant_id=TEST_TENANT_ID,
			first_name="AI",
			last_name="Insights",
			email="ai.insights@example.com",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		created_contact = await crm_service.create_contact(contact_data, test_user_context)
		
		# Get AI insights
		insights = await crm_service.get_contact_insights(created_contact.id, test_user_context)
		
		assert insights is not None
		assert insights["engagement_score"] == 85.0
		assert insights["conversion_probability"] == 0.75
		assert len(insights["recommended_actions"]) == 3
		
		# Verify AI service was called
		crm_service.ai_insights.generate_contact_insights.assert_called()
	
	@pytest.mark.asyncio
	async def test_analytics_integration(self, crm_service, test_user_context, clean_database):
		"""Test analytics tracking integration"""
		# Create and close opportunity to trigger analytics
		opportunity_data = CRMOpportunity(
			id=uuid7str(),
			tenant_id=TEST_TENANT_ID,
			opportunity_name="Analytics Test Opportunity",
			amount=Decimal('50000.00'),
			probability=Decimal('60.0'),
			close_date=date.today() + timedelta(days=30),
			stage=OpportunityStage.PROPOSAL,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		created_opportunity = await crm_service.create_opportunity(opportunity_data, test_user_context)
		
		# Close opportunity
		closed_opportunity = await crm_service.close_opportunity(
			created_opportunity.id,
			True,  # Won
			test_user_context
		)
		
		# Verify analytics were tracked
		crm_service.analytics.track_event.assert_called()
		
		# Check tracked events
		tracked_calls = crm_service.analytics.track_event.call_args_list
		
		# Should have tracked opportunity creation and closure
		event_types = [call[0][0] for call in tracked_calls]
		assert "opportunity_created" in event_types
		assert "opportunity_won" in event_types


@pytest.mark.integration
class TestDataConsistency:
	"""Test data consistency across operations"""
	
	@pytest.mark.asyncio
	async def test_lead_conversion_consistency(self, crm_service, test_user_context, clean_database):
		"""Test data consistency during lead conversion"""
		# Create lead
		lead_data = CRMLead(
			id=uuid7str(),
			tenant_id=TEST_TENANT_ID,
			first_name="Consistency",
			last_name="Test",
			email="consistency.test@example.com",
			company="Test Corp",
			budget=Decimal('75000.00'),
			timeline="Q3 2025",
			lead_status=LeadStatus.QUALIFIED,
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		created_lead = await crm_service.create_lead(lead_data, test_user_context)
		
		# Convert lead
		conversion_data = {
			"create_contact": True,
			"create_account": True,
			"create_opportunity": True,
			"opportunity_name": f"Opportunity from {created_lead.company}",
			"opportunity_amount": float(created_lead.budget)
		}
		
		conversion_result = await crm_service.convert_lead(
			created_lead.id,
			conversion_data,
			test_user_context
		)
		
		# Verify all entities are consistent
		converted_lead = await crm_service.get_lead(created_lead.id, test_user_context)
		contact = await crm_service.get_contact(conversion_result["contact_id"], test_user_context)
		opportunity = await crm_service.get_opportunity(conversion_result["opportunity_id"], test_user_context)
		
		# Check lead references
		assert converted_lead.converted_contact_id == contact.id
		assert converted_lead.converted_opportunity_id == opportunity.id
		assert converted_lead.converted_at is not None
		
		# Check data consistency
		assert contact.first_name == converted_lead.first_name
		assert contact.last_name == converted_lead.last_name
		assert contact.email == converted_lead.email
		assert contact.company == converted_lead.company
		
		assert float(opportunity.amount) == float(converted_lead.budget)
		assert opportunity.opportunity_name == f"Opportunity from {converted_lead.company}"
	
	@pytest.mark.asyncio
	async def test_cascade_delete_consistency(self, crm_service, test_user_context, clean_database):
		"""Test cascade delete consistency"""
		from ..models import CRMActivity, ActivityType, ActivityStatus
		
		# Create contact
		contact_data = CRMContact(
			id=uuid7str(),
			tenant_id=TEST_TENANT_ID,
			first_name="Delete",
			last_name="Test",
			email="delete.test@example.com",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		contact = await crm_service.create_contact(contact_data, test_user_context)
		
		# Create activities for contact
		activities = []
		for i in range(3):
			activity_data = CRMActivity(
				id=uuid7str(),
				tenant_id=TEST_TENANT_ID,
				activity_type=ActivityType.CALL,
				subject=f"Test Activity {i+1}",
				contact_id=contact.id,
				status=ActivityStatus.PLANNED,
				owner_id=TEST_USER_ID,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			
			activity = await crm_service.create_activity(activity_data, test_user_context)
			activities.append(activity)
		
		# Delete contact should cascade delete activities
		delete_result = await crm_service.delete_contact(contact.id, test_user_context)
		assert delete_result is True
		
		# Verify contact is deleted
		deleted_contact = await crm_service.get_contact(contact.id, test_user_context)
		assert deleted_contact is None
		
		# Verify activities are cascade deleted
		for activity in activities:
			deleted_activity = await crm_service.get_activity(activity.id, test_user_context)
			assert deleted_activity is None


@pytest.mark.integration
class TestConcurrencyHandling:
	"""Test concurrent operations handling"""
	
	@pytest.mark.asyncio
	async def test_concurrent_contact_creation(self, crm_service, test_user_context, clean_database):
		"""Test concurrent contact creation"""
		async def create_contact(index):
			contact_data = CRMContact(
				id=uuid7str(),
				tenant_id=TEST_TENANT_ID,
				first_name=f"Concurrent{index:02d}",
				last_name="Test",
				email=f"concurrent{index:02d}@example.com",
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			return await crm_service.create_contact(contact_data, test_user_context)
		
		# Create 10 contacts concurrently
		tasks = [create_contact(i) for i in range(10)]
		results = await asyncio.gather(*tasks)
		
		# All should succeed
		assert all(result is not None for result in results)
		assert len(set(result.id for result in results)) == 10  # All unique IDs
	
	@pytest.mark.asyncio
	async def test_concurrent_lead_conversion(self, crm_service, test_user_context, clean_database):
		"""Test concurrent lead conversions"""
		# Create multiple leads
		leads = []
		for i in range(5):
			lead_data = CRMLead(
				id=uuid7str(),
				tenant_id=TEST_TENANT_ID,
				first_name=f"ConLead{i:02d}",
				last_name="Test",
				email=f"conlead{i:02d}@example.com",
				company=f"Company{i:02d}",
				budget=Decimal('50000.00'),
				lead_status=LeadStatus.QUALIFIED,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			
			lead = await crm_service.create_lead(lead_data, test_user_context)
			leads.append(lead)
		
		# Convert all leads concurrently
		async def convert_lead(lead):
			conversion_data = {
				"create_contact": True,
				"create_account": True,
				"create_opportunity": True,
				"opportunity_name": f"Opportunity from {lead.company}",
				"opportunity_amount": float(lead.budget)
			}
			return await crm_service.convert_lead(lead.id, conversion_data, test_user_context)
		
		tasks = [convert_lead(lead) for lead in leads]
		results = await asyncio.gather(*tasks)
		
		# All conversions should succeed
		assert all(result is not None for result in results)
		assert all("contact_id" in result for result in results)
		assert all("opportunity_id" in result for result in results)
		
		# All created entities should be unique
		contact_ids = [result["contact_id"] for result in results]
		opportunity_ids = [result["opportunity_id"] for result in results]
		
		assert len(set(contact_ids)) == 5
		assert len(set(opportunity_ids)) == 5


@pytest.mark.performance
class TestIntegrationPerformance:
	"""Test integration performance characteristics"""
	
	@pytest.mark.asyncio
	async def test_bulk_workflow_performance(self, crm_service, test_user_context, clean_database, performance_timer):
		"""Test performance of bulk workflow operations"""
		performance_timer.start()
		
		# Create 50 leads and convert them
		leads = []
		for i in range(50):
			lead_data = CRMLead(
				id=uuid7str(),
				tenant_id=TEST_TENANT_ID,
				first_name=f"BulkLead{i:03d}",
				last_name="Performance",
				email=f"bulk{i:03d}@example.com",
				company=f"BulkCompany{i:03d}",
				budget=Decimal('25000.00'),
				lead_status=LeadStatus.QUALIFIED,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			
			lead = await crm_service.create_lead(lead_data, test_user_context)
			leads.append(lead)
		
		# Convert 25 of them
		conversion_tasks = []
		for lead in leads[:25]:
			conversion_data = {
				"create_contact": True,
				"create_opportunity": True,
				"opportunity_name": f"Bulk Opportunity from {lead.company}",
				"opportunity_amount": float(lead.budget)
			}
			task = crm_service.convert_lead(lead.id, conversion_data, test_user_context)
			conversion_tasks.append(task)
		
		conversion_results = await asyncio.gather(*conversion_tasks)
		
		performance_timer.stop()
		
		# Verify results
		assert len(conversion_results) == 25
		assert all(result is not None for result in conversion_results)
		
		# Should complete within reasonable time (adjust as needed)
		assert performance_timer.elapsed < 30.0  # 30 seconds for 50 leads + 25 conversions
	
	@pytest.mark.asyncio
	async def test_complex_query_performance(self, crm_service, test_user_context, clean_database, performance_timer):
		"""Test performance of complex queries"""
		# Create diverse dataset
		for i in range(100):
			# Create contact
			contact_data = CRMContact(
				id=uuid7str(),
				tenant_id=TEST_TENANT_ID,
				first_name=f"Contact{i:03d}",
				last_name="Performance",
				email=f"perf{i:03d}@example.com",
				company=f"Company{i%10}",  # 10 different companies
				contact_type=ContactType.PROSPECT if i % 2 == 0 else ContactType.CUSTOMER,
				lead_score=Decimal(str(i % 100)),
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			await crm_service.create_contact(contact_data, test_user_context)
			
			# Create opportunity for some contacts
			if i % 3 == 0:
				opportunity_data = CRMOpportunity(
					id=uuid7str(),
					tenant_id=TEST_TENANT_ID,
					opportunity_name=f"Performance Opportunity {i:03d}",
					amount=Decimal(str((i % 10 + 1) * 10000)),
					probability=Decimal(str(i % 100)),
					close_date=date.today() + timedelta(days=i % 365),
					stage=list(OpportunityStage)[i % len(list(OpportunityStage))],
					created_by=TEST_USER_ID,
					updated_by=TEST_USER_ID
				)
				await crm_service.create_opportunity(opportunity_data, test_user_context)
		
		# Test complex queries
		performance_timer.start()
		
		# Multiple concurrent complex queries
		tasks = [
			crm_service.search_contacts("Company", test_user_context),
			crm_service.list_contacts(test_user_context, filters={"contact_type": ContactType.PROSPECT}),
			crm_service.get_pipeline_analytics(test_user_context),
			crm_service.list_opportunities(test_user_context, filters={"stage": OpportunityStage.PROPOSAL})
		]
		
		results = await asyncio.gather(*tasks)
		
		performance_timer.stop()
		
		# Verify results
		assert all(result is not None for result in results)
		
		# Should complete quickly
		assert performance_timer.elapsed < 5.0  # 5 seconds for complex queries