"""
APG Customer Relationship Management - API Tests

Comprehensive unit tests for CRM REST API endpoints including authentication,
authorization, request/response handling, and error cases.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
import json
from datetime import datetime, date

from ..api import app, get_current_user, get_crm_service
from ..models import CRMContact, ContactType
from . import TEST_TENANT_ID, TEST_USER_ID


@pytest.fixture
def test_client():
	"""Create test client for FastAPI app"""
	return TestClient(app)


@pytest.fixture
async def async_client():
	"""Create async test client"""
	async with AsyncClient(app=app, base_url="http://test") as client:
		yield client


@pytest.fixture
def mock_crm_service():
	"""Mock CRM service for API testing"""
	service = AsyncMock()
	
	# Mock common operations
	service.create_contact = AsyncMock()
	service.get_contact = AsyncMock()
	service.update_contact = AsyncMock()
	service.delete_contact = AsyncMock()
	service.list_contacts = AsyncMock()
	service.search_contacts = AsyncMock()
	
	service.create_lead = AsyncMock()
	service.get_lead = AsyncMock()
	service.convert_lead = AsyncMock()
	
	service.create_opportunity = AsyncMock()
	service.get_opportunity = AsyncMock()
	service.close_opportunity = AsyncMock()
	
	service.health_check = AsyncMock(return_value={"status": "healthy"})
	
	return service


@pytest.fixture
def mock_auth_user(test_user_context):
	"""Mock authenticated user"""
	return test_user_context


@pytest.fixture
def override_dependencies(mock_crm_service, mock_auth_user):
	"""Override FastAPI dependencies for testing"""
	app.dependency_overrides[get_crm_service] = lambda: mock_crm_service
	app.dependency_overrides[get_current_user] = lambda: mock_auth_user
	
	yield
	
	# Clean up overrides
	app.dependency_overrides.clear()


@pytest.mark.unit
class TestHealthEndpoint:
	"""Test health check endpoint"""
	
	def test_health_check(self, test_client, override_dependencies, mock_crm_service):
		"""Test health check endpoint"""
		mock_crm_service.health_check.return_value = {
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat()
		}
		
		response = test_client.get("/health")
		
		assert response.status_code == 200
		data = response.json()
		assert data["status"] == "healthy"
		assert "timestamp" in data
	
	def test_health_check_unhealthy(self, test_client, override_dependencies, mock_crm_service):
		"""Test health check when service is unhealthy"""
		mock_crm_service.health_check.return_value = {
			"status": "unhealthy",
			"error": "Database connection failed"
		}
		
		response = test_client.get("/health")
		
		assert response.status_code == 503
		data = response.json()
		assert data["status"] == "unhealthy"


@pytest.mark.unit
class TestContactEndpoints:
	"""Test contact management endpoints"""
	
	def test_create_contact(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test creating a contact via API"""
		contact = contact_factory()
		mock_crm_service.create_contact.return_value = contact
		
		contact_data = {
			"first_name": contact.first_name,
			"last_name": contact.last_name,
			"email": contact.email,
			"phone": contact.phone,
			"company": contact.company,
			"job_title": contact.job_title,
			"contact_type": contact.contact_type.value
		}
		
		response = test_client.post("/api/v1/contacts", json=contact_data)
		
		assert response.status_code == 201
		data = response.json()
		assert data["id"] == contact.id
		assert data["first_name"] == contact.first_name
		assert data["email"] == contact.email
		
		mock_crm_service.create_contact.assert_called_once()
	
	def test_get_contact(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test retrieving a contact via API"""
		contact = contact_factory()
		mock_crm_service.get_contact.return_value = contact
		
		response = test_client.get(f"/api/v1/contacts/{contact.id}")
		
		assert response.status_code == 200
		data = response.json()
		assert data["id"] == contact.id
		assert data["first_name"] == contact.first_name
		
		mock_crm_service.get_contact.assert_called_once_with(contact.id, mock_auth_user)
	
	def test_get_contact_not_found(self, test_client, override_dependencies, mock_crm_service):
		"""Test retrieving non-existent contact"""
		mock_crm_service.get_contact.return_value = None
		
		response = test_client.get("/api/v1/contacts/non-existent-id")
		
		assert response.status_code == 404
		data = response.json()
		assert data["detail"] == "Contact not found"
	
	def test_update_contact(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test updating a contact via API"""
		contact = contact_factory()
		updated_contact = contact.model_copy()
		updated_contact.first_name = "UpdatedName"
		updated_contact.version = 2
		
		mock_crm_service.update_contact.return_value = updated_contact
		
		update_data = {
			"first_name": "UpdatedName",
			"lead_score": 85.0
		}
		
		response = test_client.put(f"/api/v1/contacts/{contact.id}", json=update_data)
		
		assert response.status_code == 200
		data = response.json()
		assert data["first_name"] == "UpdatedName"
		assert data["version"] == 2
		
		mock_crm_service.update_contact.assert_called_once()
	
	def test_delete_contact(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test deleting a contact via API"""
		contact = contact_factory()
		mock_crm_service.delete_contact.return_value = True
		
		response = test_client.delete(f"/api/v1/contacts/{contact.id}")
		
		assert response.status_code == 204
		
		mock_crm_service.delete_contact.assert_called_once_with(contact.id, mock_auth_user)
	
	def test_list_contacts(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test listing contacts via API"""
		contacts = [contact_factory() for _ in range(3)]
		mock_crm_service.list_contacts.return_value = {
			"items": contacts,
			"total": 3,
			"skip": 0,
			"limit": 10
		}
		
		response = test_client.get("/api/v1/contacts")
		
		assert response.status_code == 200
		data = response.json()
		assert len(data["items"]) == 3
		assert data["total"] == 3
		
		mock_crm_service.list_contacts.assert_called_once()
	
	def test_list_contacts_with_pagination(self, test_client, override_dependencies, mock_crm_service):
		"""Test contact listing with pagination parameters"""
		mock_crm_service.list_contacts.return_value = {
			"items": [],
			"total": 50,
			"skip": 20,
			"limit": 10
		}
		
		response = test_client.get("/api/v1/contacts?skip=20&limit=10")
		
		assert response.status_code == 200
		data = response.json()
		assert data["skip"] == 20
		assert data["limit"] == 10
		
		mock_crm_service.list_contacts.assert_called_once_with(
			mock_auth_user, skip=20, limit=10, filters=None
		)
	
	def test_search_contacts(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test contact search via API"""
		matching_contact = contact_factory(first_name="John")
		mock_crm_service.search_contacts.return_value = {
			"items": [matching_contact],
			"total": 1
		}
		
		response = test_client.get("/api/v1/contacts/search?q=John")
		
		assert response.status_code == 200
		data = response.json()
		assert len(data["items"]) == 1
		assert data["items"][0]["first_name"] == "John"
		
		mock_crm_service.search_contacts.assert_called_once_with("John", mock_auth_user)


@pytest.mark.unit
class TestLeadEndpoints:
	"""Test lead management endpoints"""
	
	def test_create_lead(self, test_client, override_dependencies, mock_crm_service, lead_factory):
		"""Test creating a lead via API"""
		lead = lead_factory()
		mock_crm_service.create_lead.return_value = lead
		
		lead_data = {
			"first_name": lead.first_name,
			"last_name": lead.last_name,
			"email": lead.email,
			"company": lead.company,
			"budget": float(lead.budget),
			"timeline": lead.timeline
		}
		
		response = test_client.post("/api/v1/leads", json=lead_data)
		
		assert response.status_code == 201
		data = response.json()
		assert data["first_name"] == lead.first_name
		assert data["company"] == lead.company
	
	def test_convert_lead(self, test_client, override_dependencies, mock_crm_service, lead_factory):
		"""Test lead conversion via API"""
		lead = lead_factory()
		conversion_result = {
			"lead_id": lead.id,
			"contact_id": "new_contact_id",
			"account_id": "new_account_id",
			"opportunity_id": "new_opportunity_id"
		}
		mock_crm_service.convert_lead.return_value = conversion_result
		
		conversion_data = {
			"create_contact": True,
			"create_account": True,
			"create_opportunity": True,
			"opportunity_name": "Converted Opportunity",
			"opportunity_amount": 50000.0
		}
		
		response = test_client.post(f"/api/v1/leads/{lead.id}/convert", json=conversion_data)
		
		assert response.status_code == 200
		data = response.json()
		assert data["lead_id"] == lead.id
		assert "contact_id" in data
		assert "opportunity_id" in data


@pytest.mark.unit
class TestOpportunityEndpoints:
	"""Test opportunity management endpoints"""
	
	def test_create_opportunity(self, test_client, override_dependencies, mock_crm_service, opportunity_factory):
		"""Test creating an opportunity via API"""
		opportunity = opportunity_factory()
		mock_crm_service.create_opportunity.return_value = opportunity
		
		opportunity_data = {
			"opportunity_name": opportunity.opportunity_name,
			"amount": float(opportunity.amount),
			"probability": float(opportunity.probability),
			"close_date": opportunity.close_date.isoformat(),
			"description": opportunity.description
		}
		
		response = test_client.post("/api/v1/opportunities", json=opportunity_data)
		
		assert response.status_code == 201
		data = response.json()
		assert data["opportunity_name"] == opportunity.opportunity_name
		assert data["amount"] == float(opportunity.amount)
	
	def test_close_opportunity(self, test_client, override_dependencies, mock_crm_service, opportunity_factory):
		"""Test closing an opportunity via API"""
		opportunity = opportunity_factory()
		closed_opportunity = opportunity.model_copy()
		closed_opportunity.is_closed = True
		closed_opportunity.closed_at = datetime.utcnow()
		
		mock_crm_service.close_opportunity.return_value = closed_opportunity
		
		close_data = {
			"won": True,
			"notes": "Successfully closed the deal"
		}
		
		response = test_client.post(f"/api/v1/opportunities/{opportunity.id}/close", json=close_data)
		
		assert response.status_code == 200
		data = response.json()
		assert data["is_closed"] is True
		assert data["closed_at"] is not None


@pytest.mark.unit
class TestValidation:
	"""Test API request validation"""
	
	def test_invalid_contact_data(self, test_client, override_dependencies, mock_crm_service):
		"""Test validation of invalid contact data"""
		invalid_data = {
			"first_name": "",  # Empty name
			"email": "invalid-email",  # Invalid email format
			"lead_score": 150.0  # Score too high
		}
		
		response = test_client.post("/api/v1/contacts", json=invalid_data)
		
		assert response.status_code == 422
		data = response.json()
		assert "detail" in data
	
	def test_missing_required_fields(self, test_client, override_dependencies, mock_crm_service):
		"""Test validation of missing required fields"""
		incomplete_data = {
			"email": "test@example.com"
			# Missing first_name and last_name
		}
		
		response = test_client.post("/api/v1/contacts", json=incomplete_data)
		
		assert response.status_code == 422
		data = response.json()
		assert "detail" in data
	
	def test_invalid_pagination_parameters(self, test_client, override_dependencies, mock_crm_service):
		"""Test validation of pagination parameters"""
		# Negative skip value
		response = test_client.get("/api/v1/contacts?skip=-1")
		assert response.status_code == 422
		
		# Invalid limit value
		response = test_client.get("/api/v1/contacts?limit=0")
		assert response.status_code == 422


@pytest.mark.unit
class TestAuthentication:
	"""Test API authentication and authorization"""
	
	def test_unauthorized_access(self, test_client):
		"""Test access without authentication"""
		# Remove auth override to test real auth
		response = test_client.get("/api/v1/contacts")
		
		# Should require authentication
		assert response.status_code == 401
	
	@patch('..api.get_current_user')
	def test_forbidden_access(self, mock_get_user, test_client, mock_crm_service):
		"""Test access with insufficient permissions"""
		from ..auth_integration import CRMUserContext, CRMRole
		
		# User without contact read permission
		limited_user = CRMUserContext(
			user_id="limited_user",
			username="limited",
			email="limited@example.com",
			tenant_id=TEST_TENANT_ID,
			roles=[CRMRole.CRM_READONLY],
			permissions=set(),  # No permissions
			territories=[],
			is_active=True
		)
		
		mock_get_user.return_value = limited_user
		app.dependency_overrides[get_crm_service] = lambda: mock_crm_service
		
		response = test_client.get("/api/v1/contacts")
		
		# Should be forbidden due to lack of permissions
		assert response.status_code == 403
		app.dependency_overrides.clear()


@pytest.mark.unit
class TestErrorHandling:
	"""Test API error handling"""
	
	def test_internal_server_error(self, test_client, override_dependencies, mock_crm_service):
		"""Test handling of internal server errors"""
		mock_crm_service.list_contacts.side_effect = Exception("Database connection failed")
		
		response = test_client.get("/api/v1/contacts")
		
		assert response.status_code == 500
		data = response.json()
		assert "detail" in data
	
	def test_service_unavailable(self, test_client, override_dependencies, mock_crm_service):
		"""Test handling when service is unavailable"""
		mock_crm_service.create_contact.side_effect = Exception("Service temporarily unavailable")
		
		contact_data = {
			"first_name": "Test",
			"last_name": "User",
			"email": "test@example.com"
		}
		
		response = test_client.post("/api/v1/contacts", json=contact_data)
		
		assert response.status_code == 500
	
	def test_malformed_json(self, test_client, override_dependencies):
		"""Test handling of malformed JSON"""
		response = test_client.post(
			"/api/v1/contacts",
			data="invalid json",
			headers={"Content-Type": "application/json"}
		)
		
		assert response.status_code == 422


@pytest.mark.unit
class TestResponseFormat:
	"""Test API response formatting"""
	
	def test_contact_response_format(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test contact response includes all required fields"""
		contact = contact_factory()
		mock_crm_service.get_contact.return_value = contact
		
		response = test_client.get(f"/api/v1/contacts/{contact.id}")
		
		assert response.status_code == 200
		data = response.json()
		
		# Check required fields are present
		required_fields = [
			"id", "tenant_id", "first_name", "last_name", "email",
			"contact_type", "created_at", "updated_at", "version"
		]
		
		for field in required_fields:
			assert field in data
		
		# Check field types
		assert isinstance(data["id"], str)
		assert isinstance(data["version"], int)
		assert isinstance(data["lead_score"], (float, type(None)))
	
	def test_list_response_format(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test list response format"""
		contacts = [contact_factory() for _ in range(3)]
		mock_crm_service.list_contacts.return_value = {
			"items": contacts,
			"total": 3,
			"skip": 0,
			"limit": 10
		}
		
		response = test_client.get("/api/v1/contacts")
		
		assert response.status_code == 200
		data = response.json()
		
		# Check pagination structure
		assert "items" in data
		assert "total" in data
		assert "skip" in data
		assert "limit" in data
		
		assert isinstance(data["items"], list)
		assert isinstance(data["total"], int)
		assert len(data["items"]) == 3


@pytest.mark.integration
class TestAPIIntegration:
	"""Integration tests for API endpoints"""
	
	@pytest.mark.asyncio
	async def test_async_endpoints(self, async_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test async API endpoints"""
		contact = contact_factory()
		mock_crm_service.create_contact.return_value = contact
		
		contact_data = {
			"first_name": contact.first_name,
			"last_name": contact.last_name,
			"email": contact.email
		}
		
		response = await async_client.post("/api/v1/contacts", json=contact_data)
		
		assert response.status_code == 201
		data = response.json()
		assert data["first_name"] == contact.first_name
	
	def test_crud_workflow(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test complete CRUD workflow"""
		contact = contact_factory()
		
		# Create
		mock_crm_service.create_contact.return_value = contact
		create_data = {
			"first_name": contact.first_name,
			"last_name": contact.last_name,
			"email": contact.email
		}
		
		create_response = test_client.post("/api/v1/contacts", json=create_data)
		assert create_response.status_code == 201
		created_contact = create_response.json()
		contact_id = created_contact["id"]
		
		# Read
		mock_crm_service.get_contact.return_value = contact
		read_response = test_client.get(f"/api/v1/contacts/{contact_id}")
		assert read_response.status_code == 200
		
		# Update
		updated_contact = contact.model_copy()
		updated_contact.first_name = "Updated"
		updated_contact.version = 2
		mock_crm_service.update_contact.return_value = updated_contact
		
		update_data = {"first_name": "Updated"}
		update_response = test_client.put(f"/api/v1/contacts/{contact_id}", json=update_data)
		assert update_response.status_code == 200
		assert update_response.json()["first_name"] == "Updated"
		
		# Delete
		mock_crm_service.delete_contact.return_value = True
		delete_response = test_client.delete(f"/api/v1/contacts/{contact_id}")
		assert delete_response.status_code == 204


@pytest.mark.performance
class TestAPIPerformance:
	"""Test API performance characteristics"""
	
	def test_response_time(self, test_client, override_dependencies, mock_crm_service, contact_factory, performance_timer):
		"""Test API response time"""
		contacts = [contact_factory() for _ in range(100)]
		mock_crm_service.list_contacts.return_value = {
			"items": contacts,
			"total": 100,
			"skip": 0,
			"limit": 100
		}
		
		performance_timer.start()
		response = test_client.get("/api/v1/contacts?limit=100")
		performance_timer.stop()
		
		assert response.status_code == 200
		assert performance_timer.elapsed < 1.0  # Should respond within 1 second
	
	def test_concurrent_requests(self, test_client, override_dependencies, mock_crm_service, contact_factory):
		"""Test handling of concurrent requests"""
		contact = contact_factory()
		mock_crm_service.get_contact.return_value = contact
		
		# Simulate concurrent requests
		import threading
		results = []
		
		def make_request():
			response = test_client.get(f"/api/v1/contacts/{contact.id}")
			results.append(response.status_code)
		
		threads = [threading.Thread(target=make_request) for _ in range(10)]
		
		for thread in threads:
			thread.start()
		
		for thread in threads:
			thread.join()
		
		# All requests should succeed
		assert all(status == 200 for status in results)
		assert len(results) == 10