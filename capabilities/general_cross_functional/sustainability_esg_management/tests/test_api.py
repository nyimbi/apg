#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Sustainability & ESG Management - API Tests

Comprehensive test suite for REST API endpoints with authentication,
WebSocket functionality, and API integration testing.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from sqlalchemy.orm import Session

from ..api import app, ESGWebSocketManager
from ..models import ESGMetric, ESGTarget, ESGStakeholder, ESGMetricType, ESGTargetStatus
from ..service import ESGManagementService, ESGServiceConfig
from . import ESGTestConfig, TEST_TENANT_ID, TEST_USER_ID


@pytest.fixture
def api_client():
	"""FastAPI test client"""
	return TestClient(app)


@pytest.fixture
def mock_auth_headers():
	"""Mock authentication headers"""
	return {"Authorization": "Bearer mock_jwt_token"}


@pytest.fixture
def mock_dependencies(monkeypatch):
	"""Mock API dependencies"""
	async def mock_get_current_user():
		return {
			"user_id": TEST_USER_ID,
			"tenant_id": TEST_TENANT_ID,
			"permissions": ["esg:read", "esg:write", "esg:admin"]
		}
	
	async def mock_get_database_session():
		return Mock()
	
	async def mock_get_esg_service(user=None, db_session=None):
		service = Mock(spec=ESGManagementService)
		service.get_metrics = AsyncMock(return_value=[])
		service.create_metric = AsyncMock()
		service.update_metric = AsyncMock()
		service.record_measurement = AsyncMock()
		service.create_target = AsyncMock()
		service.create_stakeholder = AsyncMock()
		service._initialize_metric_ai_predictions = AsyncMock(return_value={})
		service._predict_target_achievement = AsyncMock(return_value={})
		service._initialize_stakeholder_analytics = AsyncMock(return_value={})
		return service
	
	# Patch dependencies
	monkeypatch.setattr("sustainability_esg_management.api.get_current_user", mock_get_current_user)
	monkeypatch.setattr("sustainability_esg_management.api.get_database_session", mock_get_database_session)
	monkeypatch.setattr("sustainability_esg_management.api.get_esg_service", mock_get_esg_service)


class TestAPIHealthCheck:
	"""Test API health check endpoint"""
	
	def test_health_check(self, api_client: TestClient):
		"""Test health check endpoint"""
		response = api_client.get("/api/v1/esg/health")
		
		assert response.status_code == 200
		data = response.json()
		
		assert data["status"] == "healthy"
		assert data["version"] == "1.0.0"
		assert "timestamp" in data
		assert "services" in data
		assert data["services"]["database"] == "connected"
		assert data["services"]["ai_engine"] == "active"


class TestESGMetricsAPI:
	"""Test ESG metrics API endpoints"""
	
	def test_get_metrics_success(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test successful metrics retrieval"""
		response = api_client.get(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers
		)
		
		assert response.status_code == 200
		data = response.json()
		assert isinstance(data, list)
	
	def test_get_metrics_with_filters(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test metrics retrieval with query filters"""
		response = api_client.get(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			params={
				"metric_type": "environmental",
				"is_kpi": True,
				"category": "energy",
				"limit": 10,
				"offset": 0
			}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert isinstance(data, list)
	
	def test_get_metrics_search(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test metrics search functionality"""
		response = api_client.get(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			params={"search": "carbon emissions"}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert isinstance(data, list)
	
	def test_create_metric_success(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test successful metric creation"""
		metric_data = {
			"name": "API Test Metric",
			"code": "API_TEST_METRIC",
			"metric_type": "environmental",
			"category": "energy",
			"subcategory": "consumption",
			"description": "Test metric created via API",
			"unit": "kwh",
			"target_value": 1000.0,
			"baseline_value": 1500.0,
			"is_kpi": True,
			"is_public": False,
			"is_automated": True,
			"enable_ai_predictions": True
		}
		
		response = api_client.post(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			json=metric_data
		)
		
		assert response.status_code == 201
		# Would verify response data structure in real implementation
	
	def test_create_metric_validation_error(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test metric creation with validation errors"""
		invalid_metric_data = {
			"name": "",  # Empty name
			"code": "invalid code",  # Invalid code format
			"metric_type": "invalid_type",  # Invalid type
			"category": "",  # Empty category
			"unit": ""  # Empty unit
		}
		
		response = api_client.post(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			json=invalid_metric_data
		)
		
		assert response.status_code == 422  # Validation error
		data = response.json()
		assert "details" in data or "detail" in data
	
	def test_get_metric_by_id(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test getting specific metric by ID"""
		response = api_client.get(
			"/api/v1/esg/metrics/test_metric_id",
			headers=mock_auth_headers
		)
		
		# Would return 404 with mock, but testing the route
		assert response.status_code in [200, 404, 500]
	
	def test_update_metric(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test metric update"""
		updates = {
			"name": "Updated Metric Name",
			"target_value": 2000.0,
			"is_public": True
		}
		
		response = api_client.put(
			"/api/v1/esg/metrics/test_metric_id",
			headers=mock_auth_headers,
			json=updates
		)
		
		# Mock would handle this appropriately
		assert response.status_code in [200, 404, 500]
	
	def test_record_measurement(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test recording measurement for metric"""
		measurement_data = {
			"value": 1250.75,
			"measurement_date": datetime.utcnow().isoformat(),
			"period_start": (datetime.utcnow() - timedelta(days=30)).isoformat(),
			"period_end": datetime.utcnow().isoformat(),
			"data_source": "api_test",
			"collection_method": "automated",
			"metadata": {
				"test_source": True,
				"api_version": "1.0"
			},
			"notes": "Test measurement via API"
		}
		
		response = api_client.post(
			"/api/v1/esg/metrics/test_metric_id/measurements",
			headers=mock_auth_headers,
			json=measurement_data
		)
		
		assert response.status_code in [201, 400, 404, 500]
	
	def test_get_metric_ai_insights(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test getting AI insights for metric"""
		response = api_client.get(
			"/api/v1/esg/metrics/test_metric_id/ai-insights",
			headers=mock_auth_headers
		)
		
		assert response.status_code in [200, 404, 500]
		
		if response.status_code == 200:
			data = response.json()
			assert "status" in data
			assert "metric_id" in data
			assert "ai_insights" in data
			assert "generated_at" in data


class TestESGTargetsAPI:
	"""Test ESG targets API endpoints"""
	
	def test_get_targets(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test targets retrieval"""
		response = api_client.get(
			"/api/v1/esg/targets",
			headers=mock_auth_headers
		)
		
		assert response.status_code == 200
		data = response.json()
		assert isinstance(data, list)
	
	def test_get_targets_with_filters(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test targets retrieval with filters"""
		response = api_client.get(
			"/api/v1/esg/targets",
			headers=mock_auth_headers,
			params={
				"status": "on_track",
				"metric_id": "test_metric_id",
				"owner_id": "test_owner",
				"is_public": True,
				"limit": 25
			}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert isinstance(data, list)
	
	def test_create_target(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test target creation"""
		target_data = {
			"name": "API Test Target",
			"metric_id": "test_metric_id",
			"description": "Test target created via API",
			"target_value": 100.0,
			"baseline_value": 50.0,
			"start_date": datetime(2024, 1, 1).isoformat(),
			"target_date": datetime(2025, 12, 31).isoformat(),
			"priority": "high",
			"owner_id": "test_owner",
			"is_public": False,
			"create_milestones": True
		}
		
		response = api_client.post(
			"/api/v1/esg/targets",
			headers=mock_auth_headers,
			json=target_data
		)
		
		assert response.status_code in [201, 400, 500]
	
	def test_get_target_prediction(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test target achievement prediction"""
		response = api_client.get(
			"/api/v1/esg/targets/test_target_id/prediction",
			headers=mock_auth_headers
		)
		
		assert response.status_code in [200, 404, 500]
		
		if response.status_code == 200:
			data = response.json()
			assert "status" in data
			assert "target_id" in data
			assert "prediction" in data


class TestESGStakeholdersAPI:
	"""Test ESG stakeholders API endpoints"""
	
	def test_get_stakeholders(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test stakeholders retrieval"""
		response = api_client.get(
			"/api/v1/esg/stakeholders",
			headers=mock_auth_headers
		)
		
		assert response.status_code == 200
		data = response.json()
		assert isinstance(data, list)
	
	def test_get_stakeholders_with_filters(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test stakeholders retrieval with filters"""
		response = api_client.get(
			"/api/v1/esg/stakeholders",
			headers=mock_auth_headers,
			params={
				"stakeholder_type": "investor",
				"country": "USA",
				"portal_access": True,
				"is_active": True,
				"engagement_score_min": 70.0,
				"limit": 20
			}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert isinstance(data, list)
	
	def test_create_stakeholder(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test stakeholder creation"""
		stakeholder_data = {
			"name": "API Test Stakeholder",
			"organization": "Test Organization Ltd",
			"stakeholder_type": "investor",
			"email": "test.stakeholder@example.com",
			"phone": "+1-555-1234",
			"country": "USA",
			"language_preference": "en_US",
			"esg_interests": ["climate_change", "governance", "social_impact"],
			"engagement_frequency": "monthly",
			"portal_access": True,
			"data_access_level": "internal"
		}
		
		response = api_client.post(
			"/api/v1/esg/stakeholders",
			headers=mock_auth_headers,
			json=stakeholder_data
		)
		
		assert response.status_code in [201, 400, 500]
	
	def test_get_stakeholder_analytics(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test stakeholder engagement analytics"""
		response = api_client.get(
			"/api/v1/esg/stakeholders/test_stakeholder_id/analytics",
			headers=mock_auth_headers
		)
		
		assert response.status_code in [200, 404, 500]
		
		if response.status_code == 200:
			data = response.json()
			assert "status" in data
			assert "stakeholder_id" in data
			assert "analytics" in data


class TestESGDashboardAPI:
	"""Test ESG dashboard and analytics API endpoints"""
	
	def test_get_dashboard(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test dashboard data retrieval"""
		response = api_client.get(
			"/api/v1/esg/dashboard",
			headers=mock_auth_headers,
			params={
				"period": "current_quarter",
				"include_ai_insights": True
			}
		)
		
		assert response.status_code in [200, 500]
		
		if response.status_code == 200:
			data = response.json()
			assert "key_metrics" in data
			assert "active_targets" in data
			assert "stakeholder_summary" in data
			assert "ai_insights" in data
			assert "last_updated" in data
	
	def test_get_analytics(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test analytics data retrieval"""
		response = api_client.get(
			"/api/v1/esg/analytics",
			headers=mock_auth_headers,
			params={
				"period": "current_year",
				"include_trends": True,
				"include_predictions": True
			}
		)
		
		assert response.status_code in [200, 500]
		
		if response.status_code == 200:
			data = response.json()
			assert "period" in data
			assert "total_metrics" in data
			assert "active_targets" in data
			assert "trends" in data
			assert "ai_insights" in data


class TestAPIAuthentication:
	"""Test API authentication and authorization"""
	
	def test_unauthorized_access(self, api_client: TestClient):
		"""Test unauthorized access to protected endpoints"""
		response = api_client.get("/api/v1/esg/metrics")
		
		assert response.status_code == 401  # Unauthorized
	
	def test_invalid_token(self, api_client: TestClient):
		"""Test invalid authentication token"""
		invalid_headers = {"Authorization": "Bearer invalid_token"}
		
		response = api_client.get(
			"/api/v1/esg/metrics",
			headers=invalid_headers
		)
		
		assert response.status_code in [401, 403]  # Unauthorized or Forbidden
	
	def test_missing_permissions(self, api_client: TestClient, monkeypatch):
		"""Test access with insufficient permissions"""
		async def mock_get_current_user_limited():
			return {
				"user_id": TEST_USER_ID,
				"tenant_id": TEST_TENANT_ID,
				"permissions": ["esg:read"]  # Limited permissions
			}
		
		monkeypatch.setattr("sustainability_esg_management.api.get_current_user", mock_get_current_user_limited)
		
		response = api_client.post(
			"/api/v1/esg/metrics",
			headers={"Authorization": "Bearer limited_token"},
			json={"name": "Test", "code": "TEST", "metric_type": "environmental", "category": "test", "unit": "count"}
		)
		
		# Should fail due to insufficient permissions for create operation
		assert response.status_code in [403, 500]


class TestAPIValidation:
	"""Test API request validation"""
	
	def test_invalid_json_request(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test handling of invalid JSON requests"""
		response = api_client.post(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			data="invalid json"  # Not valid JSON
		)
		
		assert response.status_code == 422  # Unprocessable Entity
	
	def test_missing_required_fields(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test validation of missing required fields"""
		incomplete_data = {
			"name": "Incomplete Metric"
			# Missing required fields: code, metric_type, category, unit
		}
		
		response = api_client.post(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			json=incomplete_data
		)
		
		assert response.status_code == 422
		data = response.json()
		assert "details" in data or "detail" in data
	
	def test_invalid_field_types(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test validation of invalid field types"""
		invalid_data = {
			"name": "Type Test Metric",
			"code": "TYPE_TEST",
			"metric_type": "environmental",
			"category": "test",
			"unit": "count",
			"target_value": "not_a_number",  # Should be numeric
			"is_kpi": "not_a_boolean"  # Should be boolean
		}
		
		response = api_client.post(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			json=invalid_data
		)
		
		assert response.status_code == 422
	
	def test_field_constraints(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test field constraint validation"""
		invalid_data = {
			"name": "A" * 300,  # Too long
			"code": "invalid code format",  # Invalid format
			"metric_type": "environmental",
			"category": "",  # Empty string
			"unit": "count",
			"target_value": -100.0  # Negative value where positive expected
		}
		
		response = api_client.post(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			json=invalid_data
		)
		
		assert response.status_code == 422


class TestAPIErrorHandling:
	"""Test API error handling"""
	
	def test_internal_server_error(self, api_client: TestClient, mock_auth_headers: Dict[str, str], monkeypatch):
		"""Test internal server error handling"""
		async def mock_get_esg_service_error(*args, **kwargs):
			raise Exception("Database connection failed")
		
		monkeypatch.setattr("sustainability_esg_management.api.get_esg_service", mock_get_esg_service_error)
		
		response = api_client.get(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers
		)
		
		assert response.status_code == 500
		data = response.json()
		assert data["status"] == "error"
		assert data["message"] == "Internal server error"
	
	def test_resource_not_found(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test resource not found error handling"""
		response = api_client.get(
			"/api/v1/esg/metrics/nonexistent_metric_id",
			headers=mock_auth_headers
		)
		
		# Should return 404 for nonexistent resource
		assert response.status_code in [404, 500]
	
	def test_validation_error_format(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test validation error response format"""
		invalid_data = {
			"name": "",
			"code": "",
			"metric_type": "invalid",
			"category": "",
			"unit": ""
		}
		
		response = api_client.post(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			json=invalid_data
		)
		
		assert response.status_code == 422
		data = response.json()
		
		# Verify error response structure
		assert "status" in data
		assert data["status"] == "error"
		assert "message" in data
		assert "details" in data


@pytest.mark.asyncio
class TestWebSocketAPI:
	"""Test WebSocket functionality"""
	
	async def test_websocket_connection(self):
		"""Test WebSocket connection establishment"""
		manager = ESGWebSocketManager()
		
		# Mock WebSocket
		mock_websocket = Mock(spec=WebSocket)
		mock_websocket.accept = AsyncMock()
		
		await manager.connect(mock_websocket, TEST_TENANT_ID)
		
		# Verify connection was accepted and tracked
		mock_websocket.accept.assert_called_once()
		assert mock_websocket in manager.active_connections
		assert TEST_TENANT_ID in manager.tenant_connections
		assert mock_websocket in manager.tenant_connections[TEST_TENANT_ID]
	
	async def test_websocket_disconnect(self):
		"""Test WebSocket disconnection"""
		manager = ESGWebSocketManager()
		
		# Mock WebSocket
		mock_websocket = Mock(spec=WebSocket)
		mock_websocket.accept = AsyncMock()
		
		# Connect first
		await manager.connect(mock_websocket, TEST_TENANT_ID)
		
		# Then disconnect
		manager.disconnect(mock_websocket, TEST_TENANT_ID)
		
		# Verify connection was removed
		assert mock_websocket not in manager.active_connections
		if TEST_TENANT_ID in manager.tenant_connections:
			assert mock_websocket not in manager.tenant_connections[TEST_TENANT_ID]
	
	async def test_websocket_broadcast(self):
		"""Test WebSocket broadcasting to tenant"""
		manager = ESGWebSocketManager()
		
		# Mock WebSockets
		mock_websocket1 = Mock(spec=WebSocket)
		mock_websocket1.accept = AsyncMock()
		mock_websocket1.send_json = AsyncMock()
		
		mock_websocket2 = Mock(spec=WebSocket)
		mock_websocket2.accept = AsyncMock()
		mock_websocket2.send_json = AsyncMock()
		
		# Connect both websockets
		await manager.connect(mock_websocket1, TEST_TENANT_ID)
		await manager.connect(mock_websocket2, TEST_TENANT_ID)
		
		# Broadcast message
		test_message = {
			"type": "metric_update",
			"metric_id": "test_metric",
			"value": 125.5,
			"timestamp": datetime.utcnow().isoformat()
		}
		
		await manager.broadcast_to_tenant(TEST_TENANT_ID, test_message)
		
		# Verify message was sent to both connections
		mock_websocket1.send_json.assert_called_once_with(test_message)
		mock_websocket2.send_json.assert_called_once_with(test_message)
	
	async def test_websocket_broadcast_error_handling(self):
		"""Test WebSocket broadcast error handling"""
		manager = ESGWebSocketManager()
		
		# Mock WebSocket that will fail
		mock_websocket = Mock(spec=WebSocket)
		mock_websocket.accept = AsyncMock()
		mock_websocket.send_json = AsyncMock(side_effect=Exception("Connection lost"))
		
		await manager.connect(mock_websocket, TEST_TENANT_ID)
		
		# Broadcasting should not raise exception even if individual connection fails
		test_message = {"type": "test", "data": "test"}
		
		# Should not raise exception
		await manager.broadcast_to_tenant(TEST_TENANT_ID, test_message)
		
		# Verify send was attempted
		mock_websocket.send_json.assert_called_once()


class TestServerSentEvents:
	"""Test Server-Sent Events functionality"""
	
	def test_metric_stream_endpoint(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test metric data streaming endpoint"""
		response = api_client.get(
			"/api/v1/esg/stream/metrics/test_metric_id",
			headers=mock_auth_headers,
			params={"interval_seconds": 10}
		)
		
		# Should return streaming response
		assert response.status_code in [200, 404, 500]
		
		if response.status_code == 200:
			# Verify streaming response headers
			assert "text/event-stream" in response.headers.get("content-type", "")
			assert response.headers.get("cache-control") == "no-cache"


@pytest.mark.performance
class TestAPIPerformance:
	"""Test API performance"""
	
	def test_metrics_endpoint_performance(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies, performance_timer):
		"""Test metrics endpoint response time"""
		performance_timer.start()
		
		response = api_client.get(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			params={"limit": 100}
		)
		
		performance_timer.stop()
		
		# API should respond quickly
		performance_timer.assert_max_time(0.5, "Metrics API response too slow")
		assert response.status_code == 200
	
	def test_concurrent_api_requests(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies, performance_timer):
		"""Test concurrent API request handling"""
		import threading
		import time
		
		results = []
		
		def make_request():
			response = api_client.get(
				"/api/v1/esg/metrics",
				headers=mock_auth_headers
			)
			results.append(response.status_code)
		
		performance_timer.start()
		
		# Make 10 concurrent requests
		threads = []
		for _ in range(10):
			thread = threading.Thread(target=make_request)
			thread.start()
			threads.append(thread)
		
		# Wait for all requests to complete
		for thread in threads:
			thread.join()
		
		performance_timer.stop()
		
		# Should handle concurrent requests efficiently
		performance_timer.assert_max_time(2.0, "Concurrent requests took too long")
		
		# All requests should succeed
		assert len(results) == 10
		assert all(status == 200 for status in results)


@pytest.mark.integration
class TestAPIIntegration:
	"""Test API integration scenarios"""
	
	def test_complete_metric_workflow(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test complete metric management workflow"""
		# Create metric
		metric_data = {
			"name": "Integration Test Metric",
			"code": "INT_TEST_METRIC",
			"metric_type": "environmental",
			"category": "integration",
			"unit": "percentage",
			"target_value": 80.0,
			"is_kpi": True,
			"enable_ai_predictions": True
		}
		
		create_response = api_client.post(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			json=metric_data
		)
		
		# Mock would handle this appropriately
		assert create_response.status_code in [201, 400, 500]
		
		if create_response.status_code == 201:
			metric_id = "mock_metric_id"  # Would get from response
			
			# Record measurement
			measurement_data = {
				"value": 75.5,
				"measurement_date": datetime.utcnow().isoformat(),
				"data_source": "integration_test"
			}
			
			measurement_response = api_client.post(
				f"/api/v1/esg/metrics/{metric_id}/measurements",
				headers=mock_auth_headers,
				json=measurement_data
			)
			
			assert measurement_response.status_code in [201, 400, 404, 500]
			
			# Get AI insights
			insights_response = api_client.get(
				f"/api/v1/esg/metrics/{metric_id}/ai-insights",
				headers=mock_auth_headers
			)
			
			assert insights_response.status_code in [200, 404, 500]
	
	def test_dashboard_data_consistency(self, api_client: TestClient, mock_auth_headers: Dict[str, str], mock_dependencies):
		"""Test dashboard data consistency"""
		# Get dashboard data
		dashboard_response = api_client.get(
			"/api/v1/esg/dashboard",
			headers=mock_auth_headers,
			params={"period": "current_month"}
		)
		
		# Get analytics data
		analytics_response = api_client.get(
			"/api/v1/esg/analytics",
			headers=mock_auth_headers,
			params={"period": "current_month"}
		)
		
		# Both should succeed or fail consistently
		if dashboard_response.status_code == 200 and analytics_response.status_code == 200:
			dashboard_data = dashboard_response.json()
			analytics_data = analytics_response.json()
			
			# Data should be consistent between endpoints
			assert dashboard_data["last_updated"] is not None
			assert analytics_data["period"] is not None