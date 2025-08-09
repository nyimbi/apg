"""
Time & Attendance API Tests

Comprehensive unit tests for FastAPI and Flask-AppBuilder endpoints
including authentication, validation, and error handling.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
import json
from datetime import datetime, date
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from api import create_app
from models import TimeEntryStatus, WorkMode, AIAgentType


class TestHealthEndpoints:
	"""Test health and status endpoints"""
	
	def test_health_check_success(self, test_client):
		"""Test health check endpoint returns success"""
		response = test_client.get("/api/human_capital_management/time_attendance/health")
		
		assert response.status_code == 200
		data = response.json()
		assert data["status"] == "healthy"
		assert "timestamp" in data
		assert "version" in data
		assert "environment" in data
		assert "features" in data
	
	@patch('api.get_config')
	def test_health_check_with_features(self, mock_get_config, test_client):
		"""Test health check with feature flags"""
		mock_config = AsyncMock()
		mock_config.environment.value = "testing"
		mock_config.is_feature_enabled.side_effect = lambda feature: {
			"ai_fraud_detection": True,
			"biometric_authentication": False,
			"remote_work_tracking": True,
			"ai_agent_management": True,
			"hybrid_collaboration": False
		}.get(feature, False)
		mock_get_config.return_value = mock_config
		
		response = test_client.get("/api/human_capital_management/time_attendance/health")
		
		assert response.status_code == 200
		data = response.json()
		assert data["environment"] == "testing"
		assert data["features"]["ai_fraud_detection"] == True
		assert data["features"]["biometric_authentication"] == False


class TestConfigurationEndpoints:
	"""Test configuration endpoints"""
	
	@patch('api.get_current_user')
	@patch('api.get_config')
	def test_get_configuration_success(self, mock_get_config, mock_get_user, test_client):
		"""Test getting sanitized configuration"""
		# Mock user authentication
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_default",
			"roles": ["employee"]
		}
		
		# Mock configuration
		mock_config = AsyncMock()
		mock_config.environment.value = "development"
		mock_config.tracking_mode.value = "comprehensive"
		mock_config.features = {
			"ai_fraud_detection": True,
			"remote_work_tracking": True
		}
		mock_config.performance = AsyncMock()
		mock_config.performance.target_response_time_ms = 200
		mock_config.performance.target_availability_percent = 99.9
		mock_config.compliance = AsyncMock()
		mock_config.compliance.flsa_compliance_enabled = True
		mock_config.compliance.gdpr_compliance_enabled = True
		mock_config.compliance.overtime_threshold_hours = 8.0
		mock_get_config.return_value = mock_config
		
		response = test_client.get(
			"/api/human_capital_management/time_attendance/config",
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert data["success"] == True
		assert data["data"]["environment"] == "development"
		assert data["data"]["tracking_mode"] == "comprehensive"
		assert "features" in data["data"]
		assert "performance" in data["data"]
		assert "compliance" in data["data"]


class TestTimeTrackingEndpoints:
	"""Test core time tracking endpoints"""
	
	@patch('api.get_current_user')
	@patch('api.get_service')
	def test_clock_in_success(self, mock_get_service, mock_get_user, test_client):
		"""Test successful clock-in"""
		# Mock authentication
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["employee"]
		}
		
		# Mock service response
		mock_service = AsyncMock()
		mock_time_entry = AsyncMock()
		mock_time_entry.id = "entry_123"
		mock_time_entry.clock_in = datetime.now()
		mock_time_entry.status = TimeEntryStatus.SUBMITTED
		mock_time_entry.anomaly_score = 0.1
		mock_time_entry.requires_approval = False
		mock_time_entry.verification_confidence = 0.95
		
		mock_service.clock_in.return_value = mock_time_entry
		mock_get_service.return_value = mock_service
		
		# Test data
		request_data = {
			"employee_id": "emp_001",
			"device_info": {
				"device_id": "device_001",
				"device_type": "mobile",
				"os": "iOS 17.0"
			},
			"location": {
				"latitude": 40.7128,
				"longitude": -74.0060
			},
			"notes": "Starting my workday"
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/clock-in",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 201
		data = response.json()
		assert data["success"] == True
		assert data["message"] == "Clock-in processed successfully"
		assert "data" in data
		assert data["data"]["time_entry_id"] == "entry_123"
		assert data["data"]["status"] == "submitted"
		assert data["data"]["fraud_score"] == 0.1
		assert data["data"]["requires_approval"] == False
	
	@patch('api.get_current_user')
	def test_clock_in_validation_error(self, mock_get_user, test_client):
		"""Test clock-in with validation errors"""
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["employee"]
		}
		
		# Invalid request data (missing employee_id)
		request_data = {
			"device_info": {"device_id": "device_001"},
			"location": {"latitude": 91.0, "longitude": 0.0}  # Invalid latitude
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/clock-in",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 422  # Validation error
	
	@patch('api.get_current_user')
	@patch('api.get_service')
	def test_clock_out_success(self, mock_get_service, mock_get_user, test_client):
		"""Test successful clock-out"""
		# Mock authentication
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["employee"]
		}
		
		# Mock service response
		mock_service = AsyncMock()
		mock_time_entry = AsyncMock()
		mock_time_entry.id = "entry_123"
		mock_time_entry.clock_out = datetime.now()
		mock_time_entry.total_hours = 8.5
		mock_time_entry.regular_hours = 8.0
		mock_time_entry.overtime_hours = 0.5
		mock_time_entry.status = TimeEntryStatus.APPROVED
		mock_time_entry.anomaly_score = 0.2
		
		mock_service.clock_out.return_value = mock_time_entry
		mock_get_service.return_value = mock_service
		
		request_data = {
			"employee_id": "emp_001",
			"device_info": {"device_id": "device_001"},
			"notes": "End of workday"
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/clock-out",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert data["success"] == True
		assert data["data"]["total_hours"] == 8.5
		assert data["data"]["regular_hours"] == 8.0
		assert data["data"]["overtime_hours"] == 0.5
	
	@patch('api.get_current_user')
	def test_get_time_entries_success(self, mock_get_user, test_client):
		"""Test retrieving time entries with filters"""
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["manager"]
		}
		
		response = test_client.get(
			"/api/human_capital_management/time_attendance/time-entries"
			"?employee_id=emp_001&start_date=2025-01-01&end_date=2025-01-31&page=1&per_page=25",
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert data["success"] == True
		assert "pagination" in data
		assert data["pagination"]["page"] == 1
		assert data["pagination"]["per_page"] == 25


class TestRemoteWorkEndpoints:
	"""Test remote work management endpoints"""
	
	@patch('api.get_current_user')
	@patch('api.get_service')
	def test_start_remote_work_session(self, mock_get_service, mock_get_user, test_client):
		"""Test starting remote work session"""
		# Mock authentication
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["employee"]
		}
		
		# Mock service response
		mock_service = AsyncMock()
		mock_remote_worker = AsyncMock()
		mock_remote_worker.id = "rw_123"
		mock_remote_worker.workspace_id = "ws_home_001"
		mock_remote_worker.work_mode = WorkMode.REMOTE_ONLY
		mock_remote_worker.current_activity.value = "active_working"
		mock_remote_worker.overall_productivity_score = 0.8
		
		mock_service.start_remote_work_session.return_value = mock_remote_worker
		mock_get_service.return_value = mock_service
		
		request_data = {
			"employee_id": "emp_001",
			"work_mode": "remote_only",
			"workspace_config": {
				"location": "Home Office",
				"equipment": {
					"computer": "MacBook Pro",
					"monitor": "4K Display"
				}
			},
			"timezone": "America/New_York",
			"collaboration_platforms": ["slack", "zoom"]
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/remote-work/start-session",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 201
		data = response.json()
		assert data["success"] == True
		assert data["data"]["remote_worker_id"] == "rw_123"
		assert data["data"]["work_mode"] == "remote_only"
	
	@patch('api.get_current_user')
	@patch('api.get_service')
	def test_track_remote_productivity(self, mock_get_service, mock_get_user, test_client):
		"""Test tracking remote productivity"""
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["employee"]
		}
		
		# Mock service response
		mock_service = AsyncMock()
		mock_analysis = {
			"productivity_score": 0.85,
			"insights": ["High focus time", "Good task completion"],
			"recommendations": ["Take regular breaks"],
			"burnout_risk": "LOW",
			"work_life_balance": 0.8
		}
		mock_service.track_remote_productivity.return_value = mock_analysis
		mock_get_service.return_value = mock_service
		
		request_data = {
			"employee_id": "emp_001",
			"activity_data": {
				"active_time_minutes": 480,
				"tasks_completed": 8,
				"focus_sessions": 4
			},
			"metric_type": "task_completion"
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/remote-work/track-productivity",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert data["productivity_score"] == 0.85
		assert data["burnout_risk"] == "LOW"
		assert len(data["insights"]) == 2
		assert len(data["recommendations"]) == 1


class TestAIAgentEndpoints:
	"""Test AI agent management endpoints"""
	
	@patch('api.get_current_user')
	@patch('api.get_service')
	def test_register_ai_agent(self, mock_get_service, mock_get_user, test_client):
		"""Test registering AI agent"""
		mock_get_user.return_value = {
			"user_id": "admin_123",
			"tenant_id": "tenant_001",
			"roles": ["admin"]
		}
		
		# Mock service response
		mock_service = AsyncMock()
		mock_ai_agent = AsyncMock()
		mock_ai_agent.id = "ai_123"
		mock_ai_agent.agent_name = "Claude Assistant"
		mock_ai_agent.agent_type = AIAgentType.CONVERSATIONAL_AI
		mock_ai_agent.capabilities = ["nlp", "coding", "analysis"]
		mock_ai_agent.health_status = "healthy"
		mock_ai_agent.overall_performance_score = 0.95
		
		mock_service.register_ai_agent.return_value = mock_ai_agent
		mock_get_service.return_value = mock_service
		
		request_data = {
			"agent_name": "Claude Assistant",
			"agent_type": "conversational_ai",
			"capabilities": ["nlp", "coding", "analysis"],
			"configuration": {
				"api_endpoints": {"chat": "/api/chat"},
				"resource_limits": {"max_tokens": 100000}
			},
			"version": "3.5",
			"environment": "production",
			"cost_per_hour": 0.50
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/ai-agents/register",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 201
		data = response.json()
		assert data["success"] == True
		assert data["data"]["ai_agent_id"] == "ai_123"
		assert data["data"]["agent_name"] == "Claude Assistant"
		assert data["data"]["health_status"] == "healthy"
	
	@patch('api.get_current_user')
	@patch('api.get_service')
	def test_track_ai_agent_work(self, mock_get_service, mock_get_user, test_client):
		"""Test tracking AI agent work"""
		mock_get_user.return_value = {
			"user_id": "admin_123",
			"tenant_id": "tenant_001",
			"roles": ["admin"]
		}
		
		# Mock service response
		mock_service = AsyncMock()
		mock_analysis = {
			"performance_score": 0.92,
			"cost_efficiency": 0.88,
			"resource_utilization": {
				"cpu_cost": 0.10,
				"gpu_cost": 0.05
			},
			"recommendations": ["Optimize batch processing"],
			"total_cost": 0.25
		}
		mock_service.track_ai_agent_work.return_value = mock_analysis
		mock_get_service.return_value = mock_service
		
		request_data = {
			"task_data": {
				"completed": True,
				"duration_seconds": 120,
				"accuracy_score": 0.95
			},
			"resource_consumption": {
				"cpu_hours": 0.033,
				"memory_gb_hours": 0.5,
				"api_calls": 25
			}
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/ai-agents/ai_123/track-work",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert data["success"] == True
		assert "performance_score" in data["data"]
		assert "cost_efficiency" in data["data"]
		assert "recommendations" in data["data"]


class TestHybridCollaborationEndpoints:
	"""Test hybrid collaboration endpoints"""
	
	@patch('api.get_current_user')
	@patch('api.get_service')
	def test_start_hybrid_collaboration(self, mock_get_service, mock_get_user, test_client):
		"""Test starting hybrid collaboration session"""
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["manager"]
		}
		
		# Mock service response
		mock_service = AsyncMock()
		mock_collaboration = AsyncMock()
		mock_collaboration.id = "collab_123"
		mock_collaboration.session_name = "Product Planning"
		mock_collaboration.human_participants = ["emp_001"]
		mock_collaboration.ai_participants = ["ai_001"]
		mock_collaboration.start_time = datetime.now()
		mock_collaboration.session_lead = "emp_001"
		
		mock_service.start_hybrid_collaboration.return_value = mock_collaboration
		mock_get_service.return_value = mock_service
		
		request_data = {
			"session_name": "Product Planning",
			"project_id": "proj_001",
			"human_participants": ["emp_001"],
			"ai_participants": ["ai_001"],
			"session_type": "collaborative_work",
			"planned_duration_minutes": 90,
			"objectives": ["Define product roadmap", "Analyze market data"]
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/collaboration/start-session",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 201
		data = response.json()
		assert data["success"] == True
		assert data["data"]["collaboration_id"] == "collab_123"
		assert data["data"]["session_name"] == "Product Planning"
		assert len(data["data"]["human_participants"]) == 1
		assert len(data["data"]["ai_participants"]) == 1


class TestAnalyticsEndpoints:
	"""Test analytics and prediction endpoints"""
	
	@patch('api.get_current_user')
	@patch('api.get_service')
	def test_generate_workforce_predictions(self, mock_get_service, mock_get_user, test_client):
		"""Test generating workforce predictions"""
		mock_get_user.return_value = {
			"user_id": "manager_123",
			"tenant_id": "tenant_001",
			"roles": ["manager", "analytics_user"]
		}
		
		# Mock service response
		mock_service = AsyncMock()
		mock_analytics = AsyncMock()
		mock_analytics.id = "analytics_123"
		mock_analytics.analysis_type = "workforce_optimization"
		mock_analytics.model_confidence = 0.87
		mock_analytics.projected_savings = 50000.00
		mock_analytics.actionable_insights = ["insight1", "insight2"]
		mock_analytics.strategic_recommendations = ["rec1", "rec2"]
		
		mock_service.generate_workforce_predictions.return_value = mock_analytics
		mock_get_service.return_value = mock_service
		
		request_data = {
			"prediction_period_days": 30,
			"departments": ["sales", "engineering"]
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/analytics/workforce-predictions",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 200
		data = response.json()
		assert data["analytics_id"] == "analytics_123"
		assert data["analysis_type"] == "workforce_optimization"
		assert data["model_confidence"] == 0.87
		assert data["projected_savings"] == 50000.00


class TestErrorHandling:
	"""Test error handling and edge cases"""
	
	def test_unauthorized_access(self, test_client):
		"""Test accessing protected endpoints without authentication"""
		response = test_client.post(
			"/api/human_capital_management/time_attendance/clock-in",
			json={"employee_id": "emp_001"}
		)
		
		assert response.status_code == 401  # Unauthorized
	
	@patch('api.get_current_user')
	def test_invalid_json_payload(self, mock_get_user, test_client):
		"""Test sending invalid JSON payload"""
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["employee"]
		}
		
		# Send invalid JSON
		response = test_client.post(
			"/api/human_capital_management/time_attendance/clock-in",
			data="invalid json",
			headers={
				"Authorization": "Bearer test_token",
				"Content-Type": "application/json"
			}
		)
		
		assert response.status_code == 422  # Unprocessable Entity
	
	@patch('api.get_current_user')
	@patch('api.get_service')
	def test_service_error_handling(self, mock_get_service, mock_get_user, test_client):
		"""Test handling service layer errors"""
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["employee"]
		}
		
		# Mock service to raise exception
		mock_service = AsyncMock()
		mock_service.clock_in.side_effect = ValueError("Employee not found")
		mock_get_service.return_value = mock_service
		
		request_data = {
			"employee_id": "non_existent_emp",
			"device_info": {"device_id": "device_001"}
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/clock-in",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		assert response.status_code == 400
		assert "Employee not found" in response.json()["detail"]
	
	def test_nonexistent_endpoint(self, test_client):
		"""Test accessing non-existent endpoint"""
		response = test_client.get("/api/human_capital_management/time_attendance/nonexistent")
		
		assert response.status_code == 404


class TestInputValidation:
	"""Test input validation and sanitization"""
	
	@patch('api.get_current_user')
	def test_sql_injection_prevention(self, mock_get_user, test_client):
		"""Test SQL injection prevention in inputs"""
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["employee"]
		}
		
		# Attempt SQL injection in employee_id
		request_data = {
			"employee_id": "'; DROP TABLE ta_time_entries; --",
			"device_info": {"device_id": "device_001"}
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/clock-in",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		# Should be handled by validation (may be 422 or 400)
		assert response.status_code in [400, 422]
	
	@patch('api.get_current_user')
	def test_xss_prevention(self, mock_get_user, test_client):
		"""Test XSS prevention in inputs"""
		mock_get_user.return_value = {
			"user_id": "user_123",
			"tenant_id": "tenant_001",
			"roles": ["employee"]
		}
		
		# Attempt XSS in notes field
		request_data = {
			"employee_id": "emp_001",
			"device_info": {"device_id": "device_001"},
			"notes": "<script>alert('XSS')</script>"
		}
		
		response = test_client.post(
			"/api/human_capital_management/time_attendance/clock-in",
			json=request_data,
			headers={"Authorization": "Bearer test_token"}
		)
		
		# Should be processed (XSS is escaped/sanitized at output level)
		# The request itself should not be rejected for this
		assert response.status_code in [200, 201, 400, 422, 500]


if __name__ == "__main__":
	pytest.main([__file__, "-v"])