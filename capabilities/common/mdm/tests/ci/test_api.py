#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - API Testing
Unit tests for FastAPI endpoints and GraphQL operations

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid_extensions import uuid7str

from fastapi.testclient import TestClient
import httpx

from ...api import create_mdm_app
from ...service import MDMService
from ...models import EntityType, EntityStatus


class TestMDMAPIEndpoints:
	"""Test FastAPI REST endpoints"""
	
	@pytest.fixture
	def mock_mdm_service(self):
		"""Create mock MDM service for API testing"""
		service = MagicMock(spec=MDMService)
		service.is_initialized = True
		
		# Mock service methods
		service.health_check = AsyncMock(return_value={
			"status": "healthy",
			"version": "1.0.0",
			"services": {"entity_service": "healthy"},
			"database": "connected",
			"uptime_seconds": 3600
		})
		
		service.entity_service = MagicMock()
		service.quality_service = MagicMock()
		service.matching_service = MagicMock()
		service.audit_service = MagicMock()
		
		return service
	
	@pytest.fixture
	def test_app(self, mock_mdm_service):
		"""Create test FastAPI application"""
		app = create_mdm_app(mock_mdm_service, config={
			"enable_auth": False,  # Disable auth for testing
			"enable_rate_limiting": False,
			"cors_origins": ["*"]
		})
		return app
	
	@pytest.fixture
	def test_client(self, test_app):
		"""Create test client"""
		return TestClient(test_app)
	
	def test_health_check_endpoint(self, test_client, mock_mdm_service):
		"""Test health check endpoint"""
		response = test_client.get("/health")
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["status"] == "healthy"
		assert data["version"] == "1.0.0"
		assert "services" in data
		assert "database" in data
		assert "uptime_seconds" in data
		
		mock_mdm_service.health_check.assert_called_once()
	
	def test_create_entity_endpoint(self, test_client, mock_mdm_service):
		"""Test entity creation endpoint"""
		entity_id = uuid7str()
		
		# Mock successful entity creation
		mock_mdm_service.entity_service.create_entity = AsyncMock(return_value={
			"status": "success",
			"entity_id": entity_id,
			"created_at": datetime.utcnow().isoformat(),
			"message": "Entity created successfully"
		})
		
		entity_data = {
			"entity_type": "person",
			"entity_name": "John Doe",
			"business_key": "PERSON-001",
			"source_system": "api_test",
			"status": "active",
			"attributes": {
				"first_name": "John",
				"last_name": "Doe",
				"email": "john.doe@example.com"
			},
			"tags": ["test", "api"],
			"data_classification": "internal"
		}
		
		response = test_client.post(
			"/api/v1/entities",
			json=entity_data,
			headers={"X-Tenant-ID": "test-tenant", "X-User-ID": "test-user"}
		)
		
		assert response.status_code == 201
		data = response.json()
		
		assert data["success"] is True
		assert data["data"]["entity_id"] == entity_id
		assert "created_at" in data["data"]
		
		# Verify service was called
		mock_mdm_service.entity_service.create_entity.assert_called_once()
	
	def test_create_entity_validation_error(self, test_client):
		"""Test entity creation with validation errors"""
		invalid_entity_data = {
			"entity_type": "invalid_type",
			"entity_name": "",  # Empty name should fail validation
			"business_key": "TEST-001"
			# Missing required fields
		}
		
		response = test_client.post(
			"/api/v1/entities",
			json=invalid_entity_data,
			headers={"X-Tenant-ID": "test-tenant", "X-User-ID": "test-user"}
		)
		
		assert response.status_code == 422  # Validation error
		data = response.json()
		
		assert "detail" in data  # FastAPI validation error format
	
	def test_get_entity_endpoint(self, test_client, mock_mdm_service):
		"""Test entity retrieval endpoint"""
		entity_id = uuid7str()
		tenant_id = "test-tenant"
		
		entity_data = {
			"entity_id": entity_id,
			"tenant_id": tenant_id,
			"entity_type": "person",
			"entity_name": "John Doe",
			"business_key": "PERSON-001",
			"source_system": "api_test",
			"status": "active",
			"attributes": {"first_name": "John", "last_name": "Doe"},
			"created_at": datetime.utcnow().isoformat(),
			"updated_at": datetime.utcnow().isoformat()
		}
		
		# Mock successful entity retrieval
		mock_mdm_service.entity_service.get_entity = AsyncMock(return_value={
			"status": "success",
			"entity": entity_data
		})
		
		response = test_client.get(
			f"/api/v1/entities/{entity_id}",
			headers={"X-Tenant-ID": tenant_id}
		)
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["success"] is True
		assert data["data"]["entity_id"] == entity_id
		assert data["data"]["entity_name"] == "John Doe"
		
		# Verify service was called with correct parameters
		mock_mdm_service.entity_service.get_entity.assert_called_once_with(
			entity_id, tenant_id, include_versions=False, include_quality=False, include_cross_refs=False
		)
	
	def test_get_entity_with_includes(self, test_client, mock_mdm_service):
		"""Test entity retrieval with optional includes"""
		entity_id = uuid7str()
		tenant_id = "test-tenant"
		
		mock_mdm_service.entity_service.get_entity = AsyncMock(return_value={
			"status": "success",
			"entity": {
				"entity_id": entity_id,
				"entity_name": "Test Entity",
				"versions": [{"version_id": uuid7str(), "version_number": 1}],
				"quality_assessment": {"overall_score": 85.0},
				"cross_references": []
			}
		})
		
		response = test_client.get(
			f"/api/v1/entities/{entity_id}",
			params={"include_versions": True, "include_quality": True, "include_cross_refs": True},
			headers={"X-Tenant-ID": tenant_id}
		)
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["success"] is True
		assert "versions" in data["data"]
		assert "quality_assessment" in data["data"]
		assert "cross_references" in data["data"]
		
		# Verify includes were passed to service
		mock_mdm_service.entity_service.get_entity.assert_called_once_with(
			entity_id, tenant_id, include_versions=True, include_quality=True, include_cross_refs=True
		)
	
	def test_get_entity_not_found(self, test_client, mock_mdm_service):
		"""Test entity retrieval for non-existent entity"""
		entity_id = uuid7str()
		tenant_id = "test-tenant"
		
		# Mock entity not found
		mock_mdm_service.entity_service.get_entity = AsyncMock(return_value={
			"status": "error",
			"error_code": "ENTITY_NOT_FOUND",
			"message": f"Entity {entity_id} not found"
		})
		
		response = test_client.get(
			f"/api/v1/entities/{entity_id}",
			headers={"X-Tenant-ID": tenant_id}
		)
		
		assert response.status_code == 404
		data = response.json()
		
		assert data["success"] is False
		assert "not found" in data["message"].lower()
	
	def test_update_entity_endpoint(self, test_client, mock_mdm_service):
		"""Test entity update endpoint"""
		entity_id = uuid7str()
		tenant_id = "test-tenant"
		user_id = "test-user"
		
		# Mock successful update
		mock_mdm_service.entity_service.update_entity = AsyncMock(return_value={
			"status": "success",
			"entity_id": entity_id,
			"updated_at": datetime.utcnow().isoformat(),
			"message": "Entity updated successfully"
		})
		
		update_data = {
			"entity_name": "Updated Name",
			"entity_description": "Updated description",
			"attributes": {
				"first_name": "John",
				"last_name": "Updated",
				"email": "john.updated@example.com"
			},
			"tags": ["updated", "test"]
		}
		
		response = test_client.put(
			f"/api/v1/entities/{entity_id}",
			json=update_data,
			headers={"X-Tenant-ID": tenant_id, "X-User-ID": user_id}
		)
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["success"] is True
		assert data["data"]["entity_id"] == entity_id
		assert "updated_at" in data["data"]
		
		# Verify service was called
		mock_mdm_service.entity_service.update_entity.assert_called_once()
	
	def test_delete_entity_endpoint(self, test_client, mock_mdm_service):
		"""Test entity deletion endpoint"""
		entity_id = uuid7str()
		tenant_id = "test-tenant"
		user_id = "test-user"
		
		# Mock successful deletion
		mock_mdm_service.entity_service.delete_entity = AsyncMock(return_value={
			"status": "success",
			"entity_id": entity_id,
			"deleted_at": datetime.utcnow().isoformat(),
			"message": "Entity deleted successfully"
		})
		
		response = test_client.delete(
			f"/api/v1/entities/{entity_id}",
			headers={"X-Tenant-ID": tenant_id, "X-User-ID": user_id}
		)
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["success"] is True
		assert data["data"]["entity_id"] == entity_id
		assert "deleted_at" in data["data"]
		
		# Verify service was called
		mock_mdm_service.entity_service.delete_entity.assert_called_once()
	
	def test_search_entities_endpoint(self, test_client, mock_mdm_service):
		"""Test entity search endpoint"""
		tenant_id = "test-tenant"
		
		search_results = {
			"status": "success",
			"entities": [
				{
					"entity_id": uuid7str(),
					"entity_name": "Entity 1",
					"entity_type": "person",
					"business_key": "PERSON-001"
				},
				{
					"entity_id": uuid7str(),
					"entity_name": "Entity 2",
					"entity_type": "person",
					"business_key": "PERSON-002"
				}
			],
			"pagination": {
				"total_count": 2,
				"offset": 0,
				"limit": 10,
				"has_next": False,
				"has_previous": False
			}
		}
		
		# Mock search results
		mock_mdm_service.entity_service.search_entities = AsyncMock(return_value=search_results)
		
		response = test_client.get(
			"/api/v1/entities/search",
			params={
				"entity_type": "person",
				"entity_name": "Entity",
				"limit": 10,
				"offset": 0
			},
			headers={"X-Tenant-ID": tenant_id}
		)
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["success"] is True
		assert len(data["data"]["entities"]) == 2
		assert data["data"]["pagination"]["total_count"] == 2
		
		# Verify search criteria were passed correctly
		mock_mdm_service.entity_service.search_entities.assert_called_once()
		search_criteria = mock_mdm_service.entity_service.search_entities.call_args[0][1]
		assert search_criteria["entity_type"] == "person"
		assert search_criteria["entity_name"] == "Entity"
		assert search_criteria["limit"] == 10
		assert search_criteria["offset"] == 0
	
	def test_quality_assessment_endpoint(self, test_client, mock_mdm_service):
		"""Test quality assessment endpoint"""
		entity_id = uuid7str()
		tenant_id = "test-tenant"
		
		quality_result = {
			"status": "success",
			"assessment_id": uuid7str(),
			"entity_id": entity_id,
			"overall_score": 85.0,
			"quality_status": "good",
			"completeness_score": 90.0,
			"accuracy_score": 80.0,
			"consistency_score": 85.0,
			"validity_score": 85.0,
			"uniqueness_score": 95.0,
			"timeliness_score": 75.0,
			"quality_issues": [],
			"recommendations": ["Update contact information"],
			"assessment_timestamp": datetime.utcnow().isoformat()
		}
		
		# Mock quality assessment
		mock_mdm_service.quality_service.assess_quality = AsyncMock(return_value=quality_result)
		
		response = test_client.post(
			f"/api/v1/entities/{entity_id}/quality/assess",
			headers={"X-Tenant-ID": tenant_id}
		)
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["success"] is True
		assert data["data"]["overall_score"] == 85.0
		assert data["data"]["quality_status"] == "good"
		assert len(data["data"]["recommendations"]) == 1
		
		# Verify service was called
		mock_mdm_service.quality_service.assess_quality.assert_called_once()
	
	def test_duplicate_detection_endpoint(self, test_client, mock_mdm_service):
		"""Test duplicate detection endpoint"""
		entity_id = uuid7str()
		tenant_id = "test-tenant"
		
		duplicate_result = {
			"status": "success",
			"detection_id": uuid7str(),
			"entity_id": entity_id,
			"total_candidates": 2,
			"high_confidence_matches": 1,
			"medium_confidence_matches": 1,
			"low_confidence_matches": 0,
			"match_candidates": [
				{
					"candidate_id": uuid7str(),
					"candidate_name": "Similar Entity",
					"match_score": 92.5,
					"confidence": "high",
					"recommended_action": "merge"
				}
			],
			"detection_timestamp": datetime.utcnow().isoformat()
		}
		
		# Mock duplicate detection
		mock_mdm_service.matching_service.find_duplicates = AsyncMock(return_value=duplicate_result)
		
		response = test_client.post(
			f"/api/v1/entities/{entity_id}/duplicates/detect",
			headers={"X-Tenant-ID": tenant_id}
		)
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["success"] is True
		assert data["data"]["total_candidates"] == 2
		assert data["data"]["high_confidence_matches"] == 1
		assert len(data["data"]["match_candidates"]) == 1
		
		# Verify service was called
		mock_mdm_service.matching_service.find_duplicates.assert_called_once()
	
	def test_batch_operations_endpoint(self, test_client, mock_mdm_service):
		"""Test batch entity operations endpoint"""
		tenant_id = "test-tenant"
		user_id = "test-user"
		
		batch_result = {
			"status": "success",
			"batch_id": uuid7str(),
			"total_processed": 3,
			"successful": 3,
			"failed": 0,
			"entity_ids": [uuid7str(), uuid7str(), uuid7str()],
			"processing_duration_ms": 250.0
		}
		
		# Mock batch creation
		mock_mdm_service.entity_service.batch_create_entities = AsyncMock(return_value=batch_result)
		
		batch_data = {
			"entities": [
				{
					"entity_type": "person",
					"entity_name": "Person 1",
					"business_key": "BATCH-001",
					"source_system": "batch_test",
					"attributes": {"index": 1}
				},
				{
					"entity_type": "person",
					"entity_name": "Person 2",
					"business_key": "BATCH-002",
					"source_system": "batch_test",
					"attributes": {"index": 2}
				},
				{
					"entity_type": "person",
					"entity_name": "Person 3",
					"business_key": "BATCH-003",
					"source_system": "batch_test",
					"attributes": {"index": 3}
				}
			]
		}
		
		response = test_client.post(
			"/api/v1/entities/batch",
			json=batch_data,
			headers={"X-Tenant-ID": tenant_id, "X-User-ID": user_id}
		)
		
		assert response.status_code == 201
		data = response.json()
		
		assert data["success"] is True
		assert data["data"]["total_processed"] == 3
		assert data["data"]["successful"] == 3
		assert data["data"]["failed"] == 0
		assert len(data["data"]["entity_ids"]) == 3
		
		# Verify service was called
		mock_mdm_service.entity_service.batch_create_entities.assert_called_once()


class TestMDMAPIAuthentication:
	"""Test API authentication and authorization"""
	
	@pytest.fixture
	def auth_enabled_app(self, mock_mdm_service):
		"""Create test app with authentication enabled"""
		app = create_mdm_app(mock_mdm_service, config={
			"enable_auth": True,
			"jwt_secret": "test-secret-key",
			"jwt_algorithm": "HS256"
		})
		return app
	
	@pytest.fixture
	def auth_client(self, auth_enabled_app):
		"""Create test client with auth enabled"""
		return TestClient(auth_enabled_app)
	
	def test_missing_tenant_header(self, test_client):
		"""Test API request without tenant header"""
		response = test_client.get("/api/v1/entities/search")
		
		assert response.status_code == 400
		data = response.json()
		assert "tenant" in data["detail"].lower()
	
	def test_missing_user_header_for_write_operations(self, test_client):
		"""Test write operation without user header"""
		entity_data = {
			"entity_type": "person",
			"entity_name": "Test Entity",
			"business_key": "TEST-001",
			"source_system": "test"
		}
		
		response = test_client.post(
			"/api/v1/entities",
			json=entity_data,
			headers={"X-Tenant-ID": "test-tenant"}
			# Missing X-User-ID header
		)
		
		assert response.status_code == 400
		data = response.json()
		assert "user" in data["detail"].lower()


class TestMDMAPIRateLimiting:
	"""Test API rate limiting"""
	
	@pytest.fixture
	def rate_limited_app(self, mock_mdm_service):
		"""Create test app with rate limiting enabled"""
		app = create_mdm_app(mock_mdm_service, config={
			"enable_auth": False,
			"enable_rate_limiting": True,
			"rate_limit_requests": 5,
			"rate_limit_window": 60  # 5 requests per minute
		})
		return app
	
	@pytest.fixture
	def rate_limited_client(self, rate_limited_app):
		"""Create test client with rate limiting"""
		return TestClient(rate_limited_app)
	
	def test_rate_limit_enforcement(self, rate_limited_client):
		"""Test rate limiting enforcement"""
		headers = {"X-Tenant-ID": "test-tenant"}
		
		# Make requests up to the limit
		for i in range(5):
			response = rate_limited_client.get("/health", headers=headers)
			assert response.status_code == 200
		
		# Next request should be rate limited
		response = rate_limited_client.get("/health", headers=headers)
		assert response.status_code == 429  # Too Many Requests
		
		data = response.json()
		assert "rate limit" in data["detail"].lower()


class TestMDMAPIGraphQL:
	"""Test GraphQL endpoint"""
	
	def test_graphql_entity_query(self, test_client, mock_mdm_service):
		"""Test GraphQL entity query"""
		entity_id = uuid7str()
		tenant_id = "test-tenant"
		
		# Mock entity retrieval
		mock_mdm_service.entity_service.get_entity = AsyncMock(return_value={
			"status": "success",
			"entity": {
				"entity_id": entity_id,
				"entity_name": "GraphQL Test Entity",
				"entity_type": "person",
				"business_key": "GQL-001",
				"attributes": {"test": "value"}
			}
		})
		
		graphql_query = {
			"query": f"""
			query GetEntity {{
				entity(entityId: "{entity_id}") {{
					entityId
					entityName
					entityType
					businessKey
					attributes
				}}
			}}
			"""
		}
		
		response = test_client.post(
			"/graphql",
			json=graphql_query,
			headers={"X-Tenant-ID": tenant_id}
		)
		
		assert response.status_code == 200
		data = response.json()
		
		assert "data" in data
		assert "entity" in data["data"]
		assert data["data"]["entity"]["entityId"] == entity_id
		assert data["data"]["entity"]["entityName"] == "GraphQL Test Entity"
	
	def test_graphql_entities_search_query(self, test_client, mock_mdm_service):
		"""Test GraphQL entities search query"""
		tenant_id = "test-tenant"
		
		# Mock search results
		mock_mdm_service.entity_service.search_entities = AsyncMock(return_value={
			"status": "success",
			"entities": [
				{
					"entity_id": uuid7str(),
					"entity_name": "Entity 1",
					"entity_type": "person",
					"business_key": "PERSON-001"
				},
				{
					"entity_id": uuid7str(),
					"entity_name": "Entity 2",
					"entity_type": "person",
					"business_key": "PERSON-002"
				}
			],
			"pagination": {
				"total_count": 2,
				"offset": 0,
				"limit": 10,
				"has_next": False,
				"has_previous": False
			}
		})
		
		graphql_query = {
			"query": """
			query SearchEntities($criteria: EntitySearchInput!) {
				searchEntities(criteria: $criteria) {
					entities {
						entityId
						entityName
						entityType
						businessKey
					}
					pagination {
						totalCount
						hasNext
						hasPrevious
					}
				}
			}
			""",
			"variables": {
				"criteria": {
					"entityType": "person",
					"limit": 10,
					"offset": 0
				}
			}
		}
		
		response = test_client.post(
			"/graphql",
			json=graphql_query,
			headers={"X-Tenant-ID": tenant_id}
		)
		
		assert response.status_code == 200
		data = response.json()
		
		assert "data" in data
		assert "searchEntities" in data["data"]
		assert len(data["data"]["searchEntities"]["entities"]) == 2
		assert data["data"]["searchEntities"]["pagination"]["totalCount"] == 2
	
	def test_graphql_error_handling(self, test_client):
		"""Test GraphQL error handling"""
		invalid_query = {
			"query": """
			query InvalidQuery {
				nonExistentField {
					someField
				}
			}
			"""
		}
		
		response = test_client.post(
			"/graphql",
			json=invalid_query,
			headers={"X-Tenant-ID": "test-tenant"}
		)
		
		assert response.status_code == 400
		data = response.json()
		
		assert "errors" in data
		assert len(data["errors"]) > 0


class TestMDMAPIMetrics:
	"""Test API metrics and monitoring"""
	
	def test_metrics_endpoint(self, test_client):
		"""Test Prometheus metrics endpoint"""
		response = test_client.get("/metrics")
		
		assert response.status_code == 200
		metrics_text = response.text
		
		# Check for expected metrics
		assert "mdm_http_requests_total" in metrics_text
		assert "mdm_http_request_duration_seconds" in metrics_text
		assert "mdm_entity_operations_total" in metrics_text
		assert "mdm_quality_assessments_total" in metrics_text
		assert "mdm_duplicate_detections_total" in metrics_text
	
	def test_request_metrics_collection(self, test_client, mock_mdm_service):
		"""Test that metrics are collected for API requests"""
		# Make several API calls
		headers = {"X-Tenant-ID": "test-tenant"}
		
		# Mock health check
		mock_mdm_service.health_check = AsyncMock(return_value={
			"status": "healthy", "version": "1.0.0"
		})
		
		for _ in range(3):
			response = test_client.get("/health", headers=headers)
			assert response.status_code == 200
		
		# Check metrics
		response = test_client.get("/metrics")
		metrics_text = response.text
		
		# Should have recorded the health check requests
		assert "mdm_http_requests_total" in metrics_text
		assert 'method="GET"' in metrics_text
		assert 'endpoint="/health"' in metrics_text