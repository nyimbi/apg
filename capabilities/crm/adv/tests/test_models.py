"""
APG Customer Relationship Management - Model Tests

Comprehensive unit tests for CRM data models including validation,
serialization, business rules, and data integrity constraints.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
from datetime import datetime, date
from decimal import Decimal
from pydantic import ValidationError
from uuid_extensions import uuid7str

from ..models import (
	CRMContact, CRMAccount, CRMLead, CRMOpportunity, CRMActivity, CRMCampaign,
	ContactType, LeadSource, LeadStatus, OpportunityStage, ActivityType, ActivityStatus,
	CampaignType, CampaignStatus, Address, PhoneNumber
)
from . import TEST_TENANT_ID, TEST_USER_ID


@pytest.mark.unit
class TestAddress:
	"""Test Address model"""
	
	def test_address_creation(self):
		"""Test creating a valid address"""
		address = Address(
			street="123 Main Street",
			city="Test City",
			state="CA",
			postal_code="90210",
			country="United States"
		)
		
		assert address.street == "123 Main Street"
		assert address.city == "Test City"
		assert address.state == "CA"
		assert address.postal_code == "90210"
		assert address.country == "United States"
	
	def test_address_validation(self):
		"""Test address validation"""
		# Test empty street
		with pytest.raises(ValidationError):
			Address(street="", city="Test City", state="CA", postal_code="90210", country="US")
		
		# Test empty city
		with pytest.raises(ValidationError):
			Address(street="123 Main St", city="", state="CA", postal_code="90210", country="US")
	
	def test_address_serialization(self):
		"""Test address serialization"""
		address = Address(
			street="123 Main Street",
			city="Test City", 
			state="CA",
			postal_code="90210",
			country="United States"
		)
		
		data = address.model_dump()
		assert data["street"] == "123 Main Street"
		assert data["city"] == "Test City"
		
		# Test deserialization
		new_address = Address(**data)
		assert new_address == address


@pytest.mark.unit
class TestPhoneNumber:
	"""Test PhoneNumber model"""
	
	def test_phone_creation(self):
		"""Test creating a valid phone number"""
		phone = PhoneNumber(
			number="+1-555-0123",
			type="mobile",
			is_primary=True
		)
		
		assert phone.number == "+1-555-0123"
		assert phone.type == "mobile"
		assert phone.is_primary is True
	
	def test_phone_validation(self):
		"""Test phone number validation"""
		# Test empty number
		with pytest.raises(ValidationError):
			PhoneNumber(number="", type="mobile")
		
		# Test invalid type
		with pytest.raises(ValidationError):
			PhoneNumber(number="+1-555-0123", type="invalid_type")
	
	def test_phone_defaults(self):
		"""Test phone number defaults"""
		phone = PhoneNumber(number="+1-555-0123")
		assert phone.type == "mobile"
		assert phone.is_primary is False


@pytest.mark.unit
class TestCRMContact:
	"""Test CRMContact model"""
	
	def test_contact_creation(self, contact_factory):
		"""Test creating a valid contact"""
		contact = contact_factory()
		
		assert contact.id is not None
		assert contact.tenant_id == TEST_TENANT_ID
		assert contact.first_name == "John"
		assert contact.last_name == "Doe"
		assert contact.email == "john.doe@example.com"
		assert contact.contact_type == ContactType.PROSPECT
		assert contact.version == 1
	
	def test_contact_validation(self):
		"""Test contact validation"""
		# Test missing required fields
		with pytest.raises(ValidationError):
			CRMContact(tenant_id=TEST_TENANT_ID)
		
		# Test invalid email
		with pytest.raises(ValidationError):
			CRMContact(
				tenant_id=TEST_TENANT_ID,
				first_name="John",
				last_name="Doe",
				email="invalid-email",
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
		
		# Test invalid lead score
		with pytest.raises(ValidationError):
			CRMContact(
				tenant_id=TEST_TENANT_ID,
				first_name="John",
				last_name="Doe",
				lead_score=150.0,  # Should be <= 100
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
	
	def test_contact_defaults(self):
		"""Test contact default values"""
		contact = CRMContact(
			tenant_id=TEST_TENANT_ID,
			first_name="John",
			last_name="Doe",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		assert contact.contact_type == ContactType.PROSPECT
		assert contact.version == 1
		assert contact.tags == []
		assert contact.custom_fields == {}
	
	def test_contact_full_name(self, contact_factory):
		"""Test contact full name property"""
		contact = contact_factory(first_name="John", last_name="Doe")
		assert contact.full_name == "John Doe"
	
	def test_contact_serialization(self, contact_factory):
		"""Test contact serialization"""
		contact = contact_factory()
		data = contact.model_dump()
		
		assert data["id"] == contact.id
		assert data["first_name"] == contact.first_name
		assert data["contact_type"] == contact.contact_type.value
		
		# Test deserialization
		new_contact = CRMContact(**data)
		assert new_contact.id == contact.id
		assert new_contact.first_name == contact.first_name


@pytest.mark.unit
class TestCRMAccount:
	"""Test CRMAccount model"""
	
	def test_account_creation(self, account_factory):
		"""Test creating a valid account"""
		account = account_factory()
		
		assert account.id is not None
		assert account.tenant_id == TEST_TENANT_ID
		assert account.account_name == "Test Account Corp"
		assert account.industry == "Technology"
		assert account.annual_revenue == Decimal('1000000.00')
		assert account.employee_count == 50
	
	def test_account_validation(self):
		"""Test account validation"""
		# Test missing required fields
		with pytest.raises(ValidationError):
			CRMAccount(tenant_id=TEST_TENANT_ID)
		
		# Test negative revenue
		with pytest.raises(ValidationError):
			CRMAccount(
				tenant_id=TEST_TENANT_ID,
				account_name="Test Account",
				annual_revenue=-1000.00,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
		
		# Test negative employee count
		with pytest.raises(ValidationError):
			CRMAccount(
				tenant_id=TEST_TENANT_ID,
				account_name="Test Account",
				employee_count=-10,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
	
	def test_account_defaults(self):
		"""Test account default values"""
		account = CRMAccount(
			tenant_id=TEST_TENANT_ID,
			account_name="Test Account",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		assert account.tags == []
		assert account.custom_fields == {}
		assert account.version == 1


@pytest.mark.unit  
class TestCRMLead:
	"""Test CRMLead model"""
	
	def test_lead_creation(self, lead_factory):
		"""Test creating a valid lead"""
		lead = lead_factory()
		
		assert lead.id is not None
		assert lead.tenant_id == TEST_TENANT_ID
		assert lead.first_name == "Jane"
		assert lead.last_name == "Smith"
		assert lead.lead_status == LeadStatus.NEW
		assert lead.lead_source == LeadSource.REFERRAL
		assert lead.budget == Decimal('50000.00')
	
	def test_lead_validation(self):
		"""Test lead validation"""
		# Test invalid email
		with pytest.raises(ValidationError):
			CRMLead(
				tenant_id=TEST_TENANT_ID,
				first_name="Jane",
				last_name="Smith",
				email="invalid-email",
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
		
		# Test negative budget
		with pytest.raises(ValidationError):
			CRMLead(
				tenant_id=TEST_TENANT_ID,
				first_name="Jane",
				last_name="Smith",
				budget=-1000.00,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
	
	def test_lead_defaults(self):
		"""Test lead default values"""
		lead = CRMLead(
			tenant_id=TEST_TENANT_ID,
			first_name="Jane",
			last_name="Smith",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		assert lead.lead_status == LeadStatus.NEW
		assert lead.tags == []
		assert lead.custom_fields == {}
		assert lead.converted_at is None
	
	def test_lead_conversion_tracking(self, lead_factory):
		"""Test lead conversion tracking"""
		lead = lead_factory()
		
		# Initially not converted
		assert not lead.is_converted
		
		# Mark as converted
		lead.converted_contact_id = uuid7str()
		lead.converted_at = datetime.utcnow()
		
		assert lead.is_converted


@pytest.mark.unit
class TestCRMOpportunity:
	"""Test CRMOpportunity model"""
	
	def test_opportunity_creation(self, opportunity_factory):
		"""Test creating a valid opportunity"""
		opportunity = opportunity_factory()
		
		assert opportunity.id is not None
		assert opportunity.tenant_id == TEST_TENANT_ID
		assert opportunity.opportunity_name == "Test Opportunity"
		assert opportunity.stage == OpportunityStage.QUALIFICATION
		assert opportunity.amount == Decimal('75000.00')
		assert opportunity.probability == Decimal('40.0')
	
	def test_opportunity_validation(self):
		"""Test opportunity validation"""
		# Test missing required fields
		with pytest.raises(ValidationError):
			CRMOpportunity(tenant_id=TEST_TENANT_ID)
		
		# Test negative amount
		with pytest.raises(ValidationError):
			CRMOpportunity(
				tenant_id=TEST_TENANT_ID,
				opportunity_name="Test Opp",
				amount=-1000.00,
				close_date=date.today(),
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
		
		# Test invalid probability
		with pytest.raises(ValidationError):
			CRMOpportunity(
				tenant_id=TEST_TENANT_ID,
				opportunity_name="Test Opp",
				probability=150.0,  # Should be <= 100
				close_date=date.today(),
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
	
	def test_opportunity_defaults(self):
		"""Test opportunity default values"""
		opportunity = CRMOpportunity(
			tenant_id=TEST_TENANT_ID,
			opportunity_name="Test Opportunity",
			close_date=date.today(),
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		assert opportunity.stage == OpportunityStage.PROSPECTING
		assert opportunity.amount == Decimal('0')
		assert opportunity.probability == Decimal('0')
		assert opportunity.is_closed is False
	
	def test_opportunity_expected_revenue(self, opportunity_factory):
		"""Test opportunity expected revenue calculation"""
		opportunity = opportunity_factory(
			amount=100000.00,
			probability=60.0
		)
		
		# Expected revenue should be calculated
		assert opportunity.expected_revenue == Decimal('60000.00')


@pytest.mark.unit
class TestCRMActivity:
	"""Test CRMActivity model"""
	
	def test_activity_creation(self, activity_factory):
		"""Test creating a valid activity"""
		activity = activity_factory()
		
		assert activity.id is not None
		assert activity.tenant_id == TEST_TENANT_ID
		assert activity.activity_type == ActivityType.CALL
		assert activity.subject == "Test Call"
		assert activity.status == ActivityStatus.PLANNED
		assert activity.duration_minutes == 30
	
	def test_activity_validation(self):
		"""Test activity validation"""
		# Test missing required fields
		with pytest.raises(ValidationError):
			CRMActivity(tenant_id=TEST_TENANT_ID)
		
		# Test invalid duration
		with pytest.raises(ValidationError):
			CRMActivity(
				tenant_id=TEST_TENANT_ID,
				activity_type=ActivityType.CALL,
				subject="Test Call",
				duration_minutes=-30,  # Should be positive
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
	
	def test_activity_defaults(self):
		"""Test activity default values"""
		activity = CRMActivity(
			tenant_id=TEST_TENANT_ID,
			activity_type=ActivityType.TASK,
			subject="Test Task",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		assert activity.status == ActivityStatus.PLANNED
		assert activity.priority == "medium"
		assert activity.tags == []
		assert activity.custom_fields == {}
	
	def test_activity_overdue_property(self, activity_factory):
		"""Test activity overdue property"""
		from datetime import timedelta
		
		# Future activity - not overdue
		future_activity = activity_factory(
			due_date=datetime.utcnow() + timedelta(days=1)
		)
		assert not future_activity.is_overdue
		
		# Past activity - overdue
		past_activity = activity_factory(
			due_date=datetime.utcnow() - timedelta(days=1),
			status=ActivityStatus.PLANNED
		)
		assert past_activity.is_overdue
		
		# Completed activity - not overdue
		completed_activity = activity_factory(
			due_date=datetime.utcnow() - timedelta(days=1),
			status=ActivityStatus.COMPLETED
		)
		assert not completed_activity.is_overdue


@pytest.mark.unit
class TestCRMCampaign:
	"""Test CRMCampaign model"""
	
	def test_campaign_creation(self, campaign_factory):
		"""Test creating a valid campaign"""
		campaign = campaign_factory()
		
		assert campaign.id is not None
		assert campaign.tenant_id == TEST_TENANT_ID
		assert campaign.campaign_name == "Test Campaign"
		assert campaign.campaign_type == CampaignType.EMAIL
		assert campaign.status == CampaignStatus.DRAFT
		assert campaign.budget == Decimal('10000.00')
	
	def test_campaign_validation(self):
		"""Test campaign validation"""
		# Test missing required fields
		with pytest.raises(ValidationError):
			CRMCampaign(tenant_id=TEST_TENANT_ID)
		
		# Test negative budget
		with pytest.raises(ValidationError):
			CRMCampaign(
				tenant_id=TEST_TENANT_ID,
				campaign_name="Test Campaign",
				budget=-1000.00,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
		
		# Test invalid date range
		with pytest.raises(ValidationError):
			CRMCampaign(
				tenant_id=TEST_TENANT_ID,
				campaign_name="Test Campaign",
				start_date=date(2025, 6, 1),
				end_date=date(2025, 5, 1),  # End before start
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
	
	def test_campaign_defaults(self):
		"""Test campaign default values"""
		campaign = CRMCampaign(
			tenant_id=TEST_TENANT_ID,
			campaign_name="Test Campaign",
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		assert campaign.campaign_type == CampaignType.EMAIL
		assert campaign.status == CampaignStatus.DRAFT
		assert campaign.tags == []
		assert campaign.custom_fields == {}
	
	def test_campaign_duration(self, campaign_factory):
		"""Test campaign duration calculation"""
		campaign = campaign_factory(
			start_date=date(2025, 1, 1),
			end_date=date(2025, 1, 31)
		)
		
		assert campaign.duration_days == 30


@pytest.mark.unit
class TestModelRelationships:
	"""Test model relationships and references"""
	
	def test_opportunity_account_reference(self, opportunity_factory, account_factory):
		"""Test opportunity-account relationship"""
		account = account_factory()
		opportunity = opportunity_factory(account_id=account.id)
		
		assert opportunity.account_id == account.id
	
	def test_activity_relationships(self, activity_factory, contact_factory, opportunity_factory):
		"""Test activity relationships to other entities"""
		contact = contact_factory()
		opportunity = opportunity_factory()
		
		activity = activity_factory(
			contact_id=contact.id,
			opportunity_id=opportunity.id
		)
		
		assert activity.contact_id == contact.id
		assert activity.opportunity_id == opportunity.id
	
	def test_lead_conversion_references(self, lead_factory, contact_factory, opportunity_factory):
		"""Test lead conversion references"""
		contact = contact_factory()
		opportunity = opportunity_factory()
		
		lead = lead_factory(
			converted_contact_id=contact.id,
			converted_opportunity_id=opportunity.id,
			converted_at=datetime.utcnow()
		)
		
		assert lead.converted_contact_id == contact.id
		assert lead.converted_opportunity_id == opportunity.id
		assert lead.is_converted


@pytest.mark.unit
class TestModelEdgeCases:
	"""Test model edge cases and boundary conditions"""
	
	def test_maximum_field_lengths(self):
		"""Test maximum field length constraints"""
		# Test very long names
		long_name = "A" * 200
		
		with pytest.raises(ValidationError):
			CRMContact(
				tenant_id=TEST_TENANT_ID,
				first_name=long_name,  # Should exceed max length
				last_name="Doe",
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
	
	def test_decimal_precision(self):
		"""Test decimal field precision"""
		# Test very precise amounts
		opportunity = CRMOpportunity(
			tenant_id=TEST_TENANT_ID,
			opportunity_name="Test",
			amount=99999999999999.99,  # Maximum precision
			close_date=date.today(),
			created_by=TEST_USER_ID,
			updated_by=TEST_USER_ID
		)
		
		assert opportunity.amount == Decimal('99999999999999.99')
	
	def test_email_edge_cases(self):
		"""Test email validation edge cases"""
		# Valid emails
		valid_emails = [
			"user@domain.com",
			"user.name@domain.co.uk", 
			"user+tag@domain.org",
			"123@domain.com"
		]
		
		for email in valid_emails:
			contact = CRMContact(
				tenant_id=TEST_TENANT_ID,
				first_name="Test",
				last_name="User",
				email=email,
				created_by=TEST_USER_ID,
				updated_by=TEST_USER_ID
			)
			assert contact.email == email
		
		# Invalid emails
		invalid_emails = [
			"invalid-email",
			"@domain.com",
			"user@",
			"user space@domain.com"
		]
		
		for email in invalid_emails:
			with pytest.raises(ValidationError):
				CRMContact(
					tenant_id=TEST_TENANT_ID,
					first_name="Test",
					last_name="User",
					email=email,
					created_by=TEST_USER_ID,
					updated_by=TEST_USER_ID
				)