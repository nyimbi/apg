"""
APG Customer Relationship Management - Service Tests

Comprehensive unit tests for CRM service layer including business logic,
workflows, integrations, and error handling.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
import asyncio
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid_extensions import uuid7str

from ..service import CRMService
from ..models import CRMContact, CRMAccount, CRMLead, CRMOpportunity, ContactType, LeadStatus, OpportunityStage
from . import TEST_TENANT_ID, TEST_USER_ID


@pytest.mark.unit
class TestCRMService:
	"""Test CRMService functionality"""
	
	@pytest.mark.asyncio
	async def test_service_initialization(self, database_manager):
		"""Test CRM service initialization"""
		mock_event_bus = AsyncMock()
		mock_ai_insights = AsyncMock()
		mock_analytics = AsyncMock()
		
		service = CRMService(
			database_manager=database_manager,
			event_bus=mock_event_bus,
			ai_insights=mock_ai_insights,
			analytics=mock_analytics
		)
		
		assert not service._initialized
		
		await service.initialize()
		assert service._initialized
		
		await service.shutdown()
		assert not service._initialized
	
	@pytest.mark.asyncio
	async def test_health_check(self, crm_service):
		"""Test service health check"""
		health = await crm_service.health_check()
		
		assert health["status"] == "healthy"
		assert health["initialized"] is True
		assert "database_status" in health
		assert "timestamp" in health


@pytest.mark.unit
class TestContactManagement:
	"""Test contact management operations"""
	
	@pytest.mark.asyncio
	async def test_create_contact(self, crm_service, contact_factory, test_user_context, clean_database):
		"""Test creating a contact"""
		contact_data = contact_factory()
		
		result = await crm_service.create_contact(contact_data, test_user_context)
		
		assert result is not None
		assert result.id == contact_data.id
		assert result.first_name == contact_data.first_name
		assert result.email == contact_data.email
		assert result.created_by == test_user_context.user_id
		
		# Verify AI insights were called
		crm_service.ai_insights.generate_contact_insights.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_contact(self, crm_service, contact_factory, test_user_context, clean_database):
		"""Test retrieving a contact"""
		contact = contact_factory()
		created = await crm_service.create_contact(contact, test_user_context)
		
		retrieved = await crm_service.get_contact(created.id, test_user_context)
		
		assert retrieved is not None
		assert retrieved.id == created.id
		assert retrieved.first_name == created.first_name
	
	@pytest.mark.asyncio
	async def test_update_contact(self, crm_service, contact_factory, test_user_context, clean_database):
		"""Test updating a contact"""
		contact = contact_factory()
		created = await crm_service.create_contact(contact, test_user_context)
		
		# Update contact data
		update_data = {
			"first_name": "UpdatedName",
			"email": "updated@example.com",
			"lead_score": 90.0
		}
		
		updated = await crm_service.update_contact(created.id, update_data, test_user_context)
		
		assert updated is not None
		assert updated.first_name == "UpdatedName"
		assert updated.email == "updated@example.com"
		assert updated.lead_score == 90.0
		assert updated.updated_by == test_user_context.user_id
		assert updated.version == created.version + 1
	
	@pytest.mark.asyncio
	async def test_delete_contact(self, crm_service, contact_factory, test_user_context, clean_database):
		"""Test deleting a contact"""
		contact = contact_factory()
		created = await crm_service.create_contact(contact, test_user_context)
		
		result = await crm_service.delete_contact(created.id, test_user_context)
		assert result is True
		
		# Verify contact is deleted
		deleted = await crm_service.get_contact(created.id, test_user_context)
		assert deleted is None
	
	@pytest.mark.asyncio
	async def test_list_contacts(self, crm_service, contact_factory, test_user_context, clean_database):
		"""Test listing contacts"""
		# Create multiple contacts
		contacts = []
		for i in range(5):
			contact = contact_factory(email=f"contact{i}@example.com")
			created = await crm_service.create_contact(contact, test_user_context)
			contacts.append(created)
		
		# List all contacts
		result = await crm_service.list_contacts(test_user_context)
		
		assert result["total"] == 5
		assert len(result["items"]) == 5
		
		# Test pagination
		page1 = await crm_service.list_contacts(test_user_context, skip=0, limit=3)
		assert len(page1["items"]) == 3
		
		page2 = await crm_service.list_contacts(test_user_context, skip=3, limit=3)
		assert len(page2["items"]) == 2
	
	@pytest.mark.asyncio
	async def test_search_contacts(self, crm_service, contact_factory, test_user_context, clean_database):
		"""Test contact search"""
		# Create contacts with searchable data
		contacts = [
			contact_factory(first_name="John", last_name="Doe", company="TechCorp"),
			contact_factory(first_name="Jane", last_name="Smith", company="DataCorp"),
			contact_factory(first_name="Bob", last_name="Johnson", company="TechStart")
		]
		
		for contact in contacts:
			await crm_service.create_contact(contact, test_user_context)
		
		# Search by name
		john_results = await crm_service.search_contacts("John", test_user_context)
		assert len(john_results["items"]) == 1
		assert john_results["items"][0].first_name == "John"
		
		# Search by company
		tech_results = await crm_service.search_contacts("Tech", test_user_context)
		assert len(tech_results["items"]) == 2
	
	@pytest.mark.asyncio
	async def test_contact_ai_scoring(self, crm_service, contact_factory, test_user_context, clean_database):
		"""Test AI-powered contact scoring"""
		contact = contact_factory()
		created = await crm_service.create_contact(contact, test_user_context)
		
		# Mock AI insights
		crm_service.ai_insights.calculate_lead_score.return_value = 85.0
		
		score = await crm_service.calculate_contact_score(created.id, test_user_context)
		
		assert score == 85.0
		crm_service.ai_insights.calculate_lead_score.assert_called()


@pytest.mark.unit
class TestLeadManagement:
	"""Test lead management operations"""
	
	@pytest.mark.asyncio
	async def test_create_lead(self, crm_service, lead_factory, test_user_context, clean_database):
		"""Test creating a lead"""
		lead_data = lead_factory()
		
		result = await crm_service.create_lead(lead_data, test_user_context)
		
		assert result is not None
		assert result.id == lead_data.id
		assert result.first_name == lead_data.first_name
		assert result.lead_status == LeadStatus.NEW
		assert result.created_by == test_user_context.user_id
	
	@pytest.mark.asyncio
	async def test_qualify_lead(self, crm_service, lead_factory, test_user_context, clean_database):
		"""Test lead qualification"""
		lead = lead_factory(lead_status=LeadStatus.NEW)
		created = await crm_service.create_lead(lead, test_user_context)
		
		qualified = await crm_service.qualify_lead(created.id, test_user_context)
		
		assert qualified is not None
		assert qualified.lead_status == LeadStatus.QUALIFIED
		assert qualified.updated_by == test_user_context.user_id
	
	@pytest.mark.asyncio
	async def test_convert_lead(self, crm_service, lead_factory, test_user_context, clean_database):
		"""Test lead conversion to contact and opportunity"""
		lead = lead_factory(
			lead_status=LeadStatus.QUALIFIED,
			budget=50000.00,
			timeline="Q2 2025"
		)
		created = await crm_service.create_lead(lead, test_user_context)
		
		conversion_data = {
			"create_contact": True,
			"create_account": True,
			"create_opportunity": True,
			"opportunity_name": "Converted Opportunity",
			"opportunity_amount": 75000.00
		}
		
		result = await crm_service.convert_lead(created.id, conversion_data, test_user_context)
		
		assert result is not None
		assert result["lead_id"] == created.id
		assert "contact_id" in result
		assert "account_id" in result
		assert "opportunity_id" in result
		
		# Verify lead is marked as converted
		converted_lead = await crm_service.get_lead(created.id, test_user_context)
		assert converted_lead.converted_at is not None
		assert converted_lead.converted_contact_id == result["contact_id"]
	
	@pytest.mark.asyncio
	async def test_bulk_lead_operations(self, crm_service, lead_factory, test_user_context, clean_database):
		"""Test bulk lead operations"""
		# Create multiple leads
		leads = []
		for i in range(10):
			lead = lead_factory(email=f"lead{i}@example.com")
			created = await crm_service.create_lead(lead, test_user_context)
			leads.append(created.id)
		
		# Bulk update lead status
		update_data = {"lead_status": LeadStatus.CONTACTED}
		result = await crm_service.bulk_update_leads(leads[:5], update_data, test_user_context)
		
		assert result["updated_count"] == 5
		assert result["failed_count"] == 0
		
		# Verify updates
		for lead_id in leads[:5]:
			updated_lead = await crm_service.get_lead(lead_id, test_user_context)
			assert updated_lead.lead_status == LeadStatus.CONTACTED


@pytest.mark.unit
class TestOpportunityManagement:
	"""Test opportunity management operations"""
	
	@pytest.mark.asyncio
	async def test_create_opportunity(self, crm_service, opportunity_factory, test_user_context, clean_database):
		"""Test creating an opportunity"""
		opportunity_data = opportunity_factory()
		
		result = await crm_service.create_opportunity(opportunity_data, test_user_context)
		
		assert result is not None
		assert result.id == opportunity_data.id
		assert result.opportunity_name == opportunity_data.opportunity_name
		assert result.amount == opportunity_data.amount
		assert result.created_by == test_user_context.user_id
	
	@pytest.mark.asyncio
	async def test_update_opportunity_stage(self, crm_service, opportunity_factory, test_user_context, clean_database):
		"""Test updating opportunity stage"""
		opportunity = opportunity_factory(stage=OpportunityStage.QUALIFICATION)
		created = await crm_service.create_opportunity(opportunity, test_user_context)
		
		updated = await crm_service.update_opportunity_stage(
			created.id, 
			OpportunityStage.PROPOSAL, 
			test_user_context
		)
		
		assert updated is not None
		assert updated.stage == OpportunityStage.PROPOSAL
		assert updated.updated_by == test_user_context.user_id
	
	@pytest.mark.asyncio
	async def test_close_opportunity(self, crm_service, opportunity_factory, test_user_context, clean_database):
		"""Test closing an opportunity"""
		opportunity = opportunity_factory(stage=OpportunityStage.NEGOTIATION)
		created = await crm_service.create_opportunity(opportunity, test_user_context)
		
		# Close as won
		closed = await crm_service.close_opportunity(created.id, True, test_user_context)
		
		assert closed is not None
		assert closed.is_closed is True
		assert closed.closed_at is not None
		assert closed.stage == OpportunityStage.CLOSED_WON
	
	@pytest.mark.asyncio
	async def test_opportunity_ai_predictions(self, crm_service, opportunity_factory, test_user_context, clean_database):
		"""Test AI-powered opportunity predictions"""
		opportunity = opportunity_factory()
		created = await crm_service.create_opportunity(opportunity, test_user_context)
		
		# Mock AI predictions
		crm_service.ai_insights.calculate_win_probability.return_value = 0.75
		
		prediction = await crm_service.predict_opportunity_outcome(created.id, test_user_context)
		
		assert prediction is not None
		assert prediction["win_probability"] == 0.75
		crm_service.ai_insights.calculate_win_probability.assert_called()
	
	@pytest.mark.asyncio
	async def test_sales_pipeline_analytics(self, crm_service, opportunity_factory, test_user_context, clean_database):
		"""Test sales pipeline analytics"""
		# Create opportunities in different stages
		stages_amounts = [
			(OpportunityStage.QUALIFICATION, 25000),
			(OpportunityStage.PROPOSAL, 50000),
			(OpportunityStage.NEGOTIATION, 75000),
			(OpportunityStage.CLOSED_WON, 100000)
		]
		
		for stage, amount in stages_amounts:
			opportunity = opportunity_factory(stage=stage, amount=amount)
			await crm_service.create_opportunity(opportunity, test_user_context)
		
		analytics = await crm_service.get_pipeline_analytics(test_user_context)
		
		assert analytics is not None
		assert analytics["total_opportunities"] == 4
		assert analytics["total_value"] == 250000
		assert len(analytics["stage_breakdown"]) == 4


@pytest.mark.unit
class TestActivityManagement:
	"""Test activity management operations"""
	
	@pytest.mark.asyncio
	async def test_create_activity(self, crm_service, activity_factory, test_user_context, clean_database):
		"""Test creating an activity"""
		activity_data = activity_factory()
		
		result = await crm_service.create_activity(activity_data, test_user_context)
		
		assert result is not None
		assert result.id == activity_data.id
		assert result.subject == activity_data.subject
		assert result.created_by == test_user_context.user_id
	
	@pytest.mark.asyncio
	async def test_complete_activity(self, crm_service, activity_factory, test_user_context, clean_database):
		"""Test completing an activity"""
		activity = activity_factory()
		created = await crm_service.create_activity(activity, test_user_context)
		
		completed = await crm_service.complete_activity(
			created.id, 
			"Activity completed successfully",
			test_user_context
		)
		
		assert completed is not None
		assert completed.status.value == "completed"
		assert completed.completed_at is not None
		assert completed.outcome_summary == "Activity completed successfully"
	
	@pytest.mark.asyncio
	async def test_get_overdue_activities(self, crm_service, activity_factory, test_user_context, clean_database):
		"""Test getting overdue activities"""
		from datetime import timedelta
		
		# Create overdue activity
		overdue_activity = activity_factory(
			due_date=datetime.utcnow() - timedelta(days=1),
			subject="Overdue Activity"
		)
		await crm_service.create_activity(overdue_activity, test_user_context)
		
		# Create future activity
		future_activity = activity_factory(
			due_date=datetime.utcnow() + timedelta(days=1),
			subject="Future Activity"
		)
		await crm_service.create_activity(future_activity, test_user_context)
		
		overdue_list = await crm_service.get_overdue_activities(test_user_context)
		
		assert len(overdue_list["items"]) == 1
		assert overdue_list["items"][0].subject == "Overdue Activity"


@pytest.mark.unit
class TestBusinessWorkflows:
	"""Test business workflow operations"""
	
	@pytest.mark.asyncio
	async def test_lead_to_opportunity_workflow(self, crm_service, lead_factory, test_user_context, clean_database):
		"""Test complete lead to opportunity workflow"""
		# Create lead
		lead = lead_factory(
			lead_status=LeadStatus.NEW,
			budget=100000.00,
			timeline="Q2 2025"
		)
		created_lead = await crm_service.create_lead(lead, test_user_context)
		
		# Qualify lead
		qualified_lead = await crm_service.qualify_lead(created_lead.id, test_user_context)
		assert qualified_lead.lead_status == LeadStatus.QUALIFIED
		
		# Convert lead
		conversion_data = {
			"create_contact": True,
			"create_account": True,
			"create_opportunity": True,
			"opportunity_name": f"Opportunity from {qualified_lead.company}",
			"opportunity_amount": qualified_lead.budget
		}
		
		conversion_result = await crm_service.convert_lead(
			qualified_lead.id, 
			conversion_data, 
			test_user_context
		)
		
		assert conversion_result is not None
		assert "contact_id" in conversion_result
		assert "opportunity_id" in conversion_result
		
		# Verify opportunity was created with correct data
		opportunity = await crm_service.get_opportunity(
			conversion_result["opportunity_id"], 
			test_user_context
		)
		assert opportunity is not None
		assert float(opportunity.amount) == 100000.00
	
	@pytest.mark.asyncio
	async def test_opportunity_lifecycle(self, crm_service, opportunity_factory, test_user_context, clean_database):
		"""Test complete opportunity lifecycle"""
		# Create opportunity
		opportunity = opportunity_factory(
			stage=OpportunityStage.PROSPECTING,
			amount=75000.00,
			probability=10.0
		)
		created = await crm_service.create_opportunity(opportunity, test_user_context)
		
		# Progress through stages
		stages = [
			OpportunityStage.QUALIFICATION,
			OpportunityStage.NEEDS_ANALYSIS,
			OpportunityStage.PROPOSAL,
			OpportunityStage.NEGOTIATION
		]
		
		current_opportunity = created
		for stage in stages:
			current_opportunity = await crm_service.update_opportunity_stage(
				current_opportunity.id, 
				stage, 
				test_user_context
			)
			assert current_opportunity.stage == stage
		
		# Close as won
		final_opportunity = await crm_service.close_opportunity(
			current_opportunity.id, 
			True, 
			test_user_context
		)
		
		assert final_opportunity.is_closed is True
		assert final_opportunity.stage == OpportunityStage.CLOSED_WON


@pytest.mark.unit
class TestIntegrations:
	"""Test external integrations"""
	
	@pytest.mark.asyncio
	async def test_ai_insights_integration(self, crm_service, contact_factory, test_user_context, clean_database):
		"""Test AI insights integration"""
		contact = contact_factory()
		created = await crm_service.create_contact(contact, test_user_context)
		
		# Mock AI insights response
		mock_insights = {
			"engagement_score": 85.0,
			"conversion_probability": 0.75,
			"recommended_actions": ["Schedule follow-up call", "Send case studies"]
		}
		crm_service.ai_insights.generate_contact_insights.return_value = mock_insights
		
		insights = await crm_service.get_contact_insights(created.id, test_user_context)
		
		assert insights is not None
		assert insights["engagement_score"] == 85.0
		assert len(insights["recommended_actions"]) == 2
	
	@pytest.mark.asyncio
	async def test_event_publishing(self, crm_service, contact_factory, test_user_context, clean_database):
		"""Test event publishing integration"""
		contact = contact_factory()
		
		# Create contact should publish event
		created = await crm_service.create_contact(contact, test_user_context)
		
		# Verify event was published
		crm_service.event_bus.publish.assert_called()
		
		# Check event content
		published_event = crm_service.event_bus.publish.call_args[0][0]
		assert published_event.event_type == "crm.contact.created"
		assert published_event.data["contact_id"] == created.id
	
	@pytest.mark.asyncio
	async def test_analytics_tracking(self, crm_service, opportunity_factory, test_user_context, clean_database):
		"""Test analytics tracking integration"""
		opportunity = opportunity_factory()
		created = await crm_service.create_opportunity(opportunity, test_user_context)
		
		# Close opportunity should track analytics
		await crm_service.close_opportunity(created.id, True, test_user_context)
		
		# Verify analytics were tracked
		crm_service.analytics.track_event.assert_called()


@pytest.mark.unit
class TestErrorHandling:
	"""Test error handling scenarios"""
	
	@pytest.mark.asyncio
	async def test_not_found_errors(self, crm_service, test_user_context):
		"""Test handling of not found errors"""
		non_existent_id = uuid7str()
		
		# Should return None for non-existent entities
		contact = await crm_service.get_contact(non_existent_id, test_user_context)
		assert contact is None
		
		lead = await crm_service.get_lead(non_existent_id, test_user_context)
		assert lead is None
		
		opportunity = await crm_service.get_opportunity(non_existent_id, test_user_context)
		assert opportunity is None
	
	@pytest.mark.asyncio
	async def test_validation_errors(self, crm_service, test_user_context):
		"""Test handling of validation errors"""
		# Invalid contact data
		invalid_contact = CRMContact(
			tenant_id=TEST_TENANT_ID,
			first_name="",  # Empty name should fail
			last_name="Test",
			email="invalid-email",  # Invalid email
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		result = await crm_service.create_contact(invalid_contact, test_user_context)
		assert result is None  # Should handle validation gracefully
	
	@pytest.mark.asyncio
	async def test_permission_errors(self, crm_service, contact_factory):
		"""Test handling of permission errors"""
		# Create user context without permissions
		from ..auth_integration import CRMUserContext, CRMRole
		
		limited_user = CRMUserContext(
			user_id="limited_user",
			username="limiteduser",
			email="limited@example.com",
			tenant_id=TEST_TENANT_ID,
			roles=[CRMRole.CRM_READONLY],
			permissions=set(),  # No permissions
			territories=[],
			is_active=True
		)
		
		contact = contact_factory()
		
		# Should fail due to lack of permissions
		with pytest.raises(Exception):  # Should raise permission error
			await crm_service.create_contact(contact, limited_user)
	
	@pytest.mark.asyncio
	async def test_database_error_handling(self, crm_service, contact_factory, test_user_context):
		"""Test handling of database errors"""
		# Mock database error
		crm_service.database_manager.create_contact = AsyncMock(side_effect=Exception("Database error"))
		
		contact = contact_factory()
		result = await crm_service.create_contact(contact, test_user_context)
		
		# Should handle database errors gracefully
		assert result is None


@pytest.mark.performance
class TestServicePerformance:
	"""Test service performance characteristics"""
	
	@pytest.mark.asyncio
	async def test_bulk_operations_performance(self, crm_service, contact_factory, test_user_context, clean_database, performance_timer):
		"""Test bulk operations performance"""
		# Create 50 contacts
		contacts = [contact_factory(email=f"bulk{i:03d}@example.com") for i in range(50)]
		
		performance_timer.start()
		
		# Bulk create
		tasks = [crm_service.create_contact(contact, test_user_context) for contact in contacts]
		results = await asyncio.gather(*tasks)
		
		performance_timer.stop()
		
		# Verify all succeeded
		assert all(result is not None for result in results)
		
		# Should complete within reasonable time
		assert performance_timer.elapsed < 10.0  # 10 seconds for 50 operations
	
	@pytest.mark.asyncio
	async def test_search_performance(self, crm_service, contact_factory, test_user_context, clean_database, performance_timer):
		"""Test search performance with large dataset"""
		# Create 200 contacts
		contacts = []
		for i in range(200):
			contact = contact_factory(
				first_name=f"Contact{i:03d}",
				email=f"search{i:03d}@example.com",
				company=f"Company{i%20}"
			)
			contacts.append(contact)
		
		# Bulk create
		tasks = [crm_service.create_contact(contact, test_user_context) for contact in contacts]
		await asyncio.gather(*tasks)
		
		# Test search performance
		performance_timer.start()
		
		results = await crm_service.search_contacts("Contact001", test_user_context)
		
		performance_timer.stop()
		
		# Should find the contact quickly
		assert len(results["items"]) == 1
		assert results["items"][0].first_name == "Contact001"
		
		# Search should be fast
		assert performance_timer.elapsed < 1.0  # Under 1 second