"""
Time & Attendance Service Tests

Comprehensive unit tests for the Time & Attendance service layer
including business logic, AI features, and integration points.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from service import TimeAttendanceService
from models import (
	TAEmployee, TATimeEntry, TARemoteWorker, TAAIAgent, TAHybridCollaboration,
	TimeEntryStatus, TimeEntryType, WorkMode, AIAgentType, ProductivityMetric,
	RemoteWorkStatus
)
from config import TimeAttendanceConfig


class TestTimeAttendanceService:
	"""Test TimeAttendanceService class"""
	
	@pytest.mark.asyncio
	async def test_service_initialization(self, test_config):
		"""Test service initialization"""
		service = TimeAttendanceService(test_config)
		
		assert service.config == test_config
		assert service.logger is not None
		assert service._fraud_detector is None  # Not initialized yet
		assert service._predictor is None
		assert service._optimizer is None
	
	@pytest.mark.asyncio
	async def test_clock_in_success(self, service, employee_data, tenant_data):
		"""Test successful clock-in process"""
		# Mock dependencies
		mock_employee = TAEmployee(**employee_data, tenant_id=tenant_data["id"], created_by="admin")
		service._get_employee_profile = AsyncMock(return_value=mock_employee)
		service._validate_clock_in_rules = AsyncMock()
		service._process_biometric_authentication = AsyncMock(return_value={"confidence": 0.95})
		service._analyze_fraud_indicators = AsyncMock(return_value={"indicators": [], "anomaly_score": 0.1})
		service._validate_time_entry = AsyncMock(return_value={"valid": True})
		service._requires_approval = AsyncMock(return_value=False)
		service._save_time_entry = AsyncMock(side_effect=lambda x: x)
		service._send_clock_in_notification = AsyncMock()
		
		# Test clock-in
		result = await service.clock_in(
			employee_id=employee_data["id"],
			tenant_id=tenant_data["id"],
			device_info={"device_id": "test-device"},
			location={"latitude": 40.7128, "longitude": -74.0060},
			created_by=employee_data["id"]
		)
		
		# Assertions
		assert result.employee_id == employee_data["id"]
		assert result.tenant_id == tenant_data["id"]
		assert result.status == TimeEntryStatus.SUBMITTED
		assert result.entry_type == TimeEntryType.REGULAR
		assert result.clock_in is not None
		
		# Verify mocks were called
		service._get_employee_profile.assert_called_once()
		service._validate_clock_in_rules.assert_called_once()
		service._save_time_entry.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_clock_in_employee_not_found(self, service, tenant_data):
		"""Test clock-in with non-existent employee"""
		service._get_employee_profile = AsyncMock(return_value=None)
		
		with pytest.raises(ValueError, match="Employee .* not found"):
			await service.clock_in(
				employee_id="non-existent-employee",
				tenant_id=tenant_data["id"],
				device_info={"device_id": "test-device"},
				created_by="admin"
			)
	
	@pytest.mark.asyncio
	async def test_clock_in_with_fraud_detection(self, service, employee_data, tenant_data):
		"""Test clock-in with fraud detection triggered"""
		# Mock high fraud score
		mock_employee = TAEmployee(**employee_data, tenant_id=tenant_data["id"], created_by="admin")
		service._get_employee_profile = AsyncMock(return_value=mock_employee)
		service._validate_clock_in_rules = AsyncMock()
		service._analyze_fraud_indicators = AsyncMock(return_value={
			"indicators": [{"type": "location_anomaly", "severity": "HIGH"}],
			"anomaly_score": 0.8
		})
		service._validate_time_entry = AsyncMock(return_value={"valid": True})
		service._requires_approval = AsyncMock(return_value=True)
		service._save_time_entry = AsyncMock(side_effect=lambda x: x)
		service._trigger_approval_workflow = AsyncMock()
		
		result = await service.clock_in(
			employee_id=employee_data["id"],
			tenant_id=tenant_data["id"],
			device_info={"device_id": "test-device"},
			created_by=employee_data["id"]
		)
		
		# Should be in draft status due to high fraud score
		assert result.status == TimeEntryStatus.DRAFT
		assert result.anomaly_score == 0.8
		assert result.requires_approval == True
		service._trigger_approval_workflow.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_clock_out_success(self, service, employee_data, tenant_data):
		"""Test successful clock-out process"""
		# Create mock active time entry
		mock_time_entry = TATimeEntry(
			tenant_id=tenant_data["id"],
			employee_id=employee_data["id"],
			entry_date=date.today(),
			clock_in=datetime.now().replace(hour=9, minute=0),
			status=TimeEntryStatus.SUBMITTED,
			created_by=employee_data["id"]
		)
		
		# Mock dependencies
		service._get_active_time_entry = AsyncMock(return_value=mock_time_entry)
		service._calculate_work_hours = AsyncMock()
		service._apply_compliance_rules = AsyncMock()
		service._analyze_fraud_indicators = AsyncMock(return_value={"indicators": [], "anomaly_score": 0.2})
		service._save_time_entry = AsyncMock(side_effect=lambda x: x)
		service._send_clock_out_notification = AsyncMock()
		service._sync_with_payroll = AsyncMock()
		
		result = await service.clock_out(
			employee_id=employee_data["id"],
			tenant_id=tenant_data["id"],
			device_info={"device_id": "test-device"},
			created_by=employee_data["id"]
		)
		
		# Assertions
		assert result.clock_out is not None
		assert result.status == TimeEntryStatus.APPROVED  # Low fraud score
		
		# Verify mocks were called
		service._get_active_time_entry.assert_called_once()
		service._calculate_work_hours.assert_called_once()
		service._sync_with_payroll.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_clock_out_no_active_entry(self, service, employee_data, tenant_data):
		"""Test clock-out with no active time entry"""
		service._get_active_time_entry = AsyncMock(return_value=None)
		
		with pytest.raises(ValueError, match="No active time entry found"):
			await service.clock_out(
				employee_id=employee_data["id"],
				tenant_id=tenant_data["id"],
				device_info={"device_id": "test-device"},
				created_by=employee_data["id"]
			)


class TestRemoteWorkerManagement:
	"""Test remote worker management functionality"""
	
	@pytest.mark.asyncio
	async def test_start_remote_work_session(self, service, employee_data, tenant_data):
		"""Test starting remote work session"""
		# Mock dependencies
		mock_employee = TAEmployee(**employee_data, tenant_id=tenant_data["id"], created_by="admin")
		service._get_employee_profile = AsyncMock(return_value=mock_employee)
		service._setup_workspace_monitoring = AsyncMock()
		service._initialize_productivity_tracking = AsyncMock()
		service._setup_collaboration_tracking = AsyncMock()
		service._start_environmental_monitoring = AsyncMock()
		service._save_remote_worker = AsyncMock(side_effect=lambda x: x)
		service._send_remote_work_setup_notification = AsyncMock()
		
		workspace_config = {
			"location": "Home Office",
			"equipment": {"computer": "MacBook Pro", "monitor": "4K Display"},
			"timezone": "America/New_York",
			"collaboration_platforms": ["slack", "zoom"]
		}
		
		result = await service.start_remote_work_session(
			employee_id=employee_data["id"],
			tenant_id=tenant_data["id"],
			workspace_config=workspace_config,
			work_mode=WorkMode.REMOTE_ONLY,
			created_by=employee_data["id"]
		)
		
		# Assertions
		assert result.employee_id == employee_data["id"]
		assert result.tenant_id == tenant_data["id"]
		assert result.work_mode == WorkMode.REMOTE_ONLY
		assert result.current_activity == RemoteWorkStatus.ACTIVE_WORKING
		assert result.timezone == "America/New_York"
		
		# Verify setup methods were called
		service._setup_workspace_monitoring.assert_called_once()
		service._initialize_productivity_tracking.assert_called_once()
		service._setup_collaboration_tracking.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_track_remote_productivity(self, service, employee_data, tenant_data):
		"""Test tracking remote worker productivity"""
		# Create mock remote worker
		mock_remote_worker = TARemoteWorker(
			tenant_id=tenant_data["id"],
			employee_id=employee_data["id"],
			workspace_id="ws-001",
			work_mode=WorkMode.REMOTE_ONLY,
			created_by=employee_data["id"]
		)
		
		# Mock dependencies
		service._get_active_remote_worker = AsyncMock(return_value=mock_remote_worker)
		service._analyze_remote_productivity = AsyncMock(return_value={
			"score": 0.85,
			"insights": ["High focus time", "Good task completion rate"]
		})
		service._assess_burnout_risk = AsyncMock(return_value={"risk_level": "LOW"})
		service._calculate_work_life_balance = AsyncMock(return_value=0.8)
		service._generate_productivity_recommendations = AsyncMock(return_value=[
			"Take regular breaks",
			"Schedule focused work blocks"
		])
		service._save_remote_worker = AsyncMock()
		
		activity_data = {
			"active_time_minutes": 480,
			"tasks_completed": 8,
			"focus_sessions": 4,
			"break_duration_minutes": 60
		}
		
		result = await service.track_remote_productivity(
			employee_id=employee_data["id"],
			tenant_id=tenant_data["id"],
			activity_data=activity_data,
			metric_type=ProductivityMetric.TASK_COMPLETION
		)
		
		# Assertions
		assert result["productivity_score"] == 0.85
		assert result["burnout_risk"] == "LOW"
		assert result["work_life_balance"] == 0.8
		assert len(result["insights"]) == 2
		assert len(result["recommendations"]) == 2
	
	@pytest.mark.asyncio
	async def test_track_productivity_no_active_session(self, service, employee_data, tenant_data):
		"""Test tracking productivity with no active remote work session"""
		service._get_active_remote_worker = AsyncMock(return_value=None)
		
		with pytest.raises(ValueError, match="No active remote work session"):
			await service.track_remote_productivity(
				employee_id=employee_data["id"],
				tenant_id=tenant_data["id"],
				activity_data={"tasks_completed": 5}
			)


class TestAIAgentManagement:
	"""Test AI agent management functionality"""
	
	@pytest.mark.asyncio
	async def test_register_ai_agent(self, service, tenant_data):
		"""Test registering AI agent"""
		# Mock dependencies
		service._setup_ai_agent_monitoring = AsyncMock()
		service._configure_ai_agent_integrations = AsyncMock()
		service._initialize_resource_tracking = AsyncMock()
		service._save_ai_agent = AsyncMock(side_effect=lambda x: x)
		service._send_ai_agent_registration_notification = AsyncMock()
		
		configuration = {
			"api_endpoints": {"chat": "/api/chat"},
			"resource_limits": {"max_tokens": 100000},
			"version": "3.5",
			"environment": "production",
			"cost_per_hour": 0.50
		}
		
		result = await service.register_ai_agent(
			agent_name="Claude Assistant",
			agent_type=AIAgentType.CONVERSATIONAL_AI,
			capabilities=["nlp", "coding", "analysis"],
			tenant_id=tenant_data["id"],
			configuration=configuration,
			created_by="admin"
		)
		
		# Assertions
		assert result.agent_name == "Claude Assistant"
		assert result.agent_type == AIAgentType.CONVERSATIONAL_AI
		assert result.tenant_id == tenant_data["id"]
		assert result.health_status == "healthy"
		assert result.operational_cost_per_hour == Decimal("0.50")
		
		# Verify setup methods were called
		service._setup_ai_agent_monitoring.assert_called_once()
		service._configure_ai_agent_integrations.assert_called_once()
		service._initialize_resource_tracking.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_track_ai_agent_work(self, service, tenant_data):
		"""Test tracking AI agent work"""
		# Create mock AI agent
		mock_ai_agent = TAAIAgent(
			tenant_id=tenant_data["id"],
			agent_name="Test Agent",
			agent_type=AIAgentType.AUTOMATION_BOT,
			capabilities=["automation"],
			configuration={
				"api_endpoints": {"execute": "/api/execute"},
				"resource_limits": {"max_operations": 1000}
			},
			tasks_completed=10,
			total_operational_cost=Decimal("5.00"),
			created_by="admin"
		)
		
		# Mock dependencies
		service._get_ai_agent = AsyncMock(return_value=mock_ai_agent)
		service._calculate_ai_agent_costs = AsyncMock(return_value={
			"total_cost": 0.25,
			"resource_breakdown": {
				"cpu_cost": 0.10,
				"gpu_cost": 0.05,
				"api_cost": 0.10
			}
		})
		service._update_ai_agent_health = AsyncMock()
		service._analyze_ai_agent_performance = AsyncMock(return_value={
			"recommendations": ["Optimize batch processing", "Reduce API calls"]
		})
		service._save_ai_agent = AsyncMock()
		
		task_data = {
			"completed": True,
			"duration_seconds": 120,
			"accuracy_score": 0.95
		}
		
		resource_consumption = {
			"cpu_hours": 0.033,
			"gpu_hours": 0.0,
			"memory_gb_hours": 0.5,
			"api_calls": 25
		}
		
		result = await service.track_ai_agent_work(
			agent_id=mock_ai_agent.id,
			tenant_id=tenant_data["id"],
			task_data=task_data,
			resource_consumption=resource_consumption
		)
		
		# Assertions
		assert "performance_score" in result
		assert "cost_efficiency" in result
		assert "resource_utilization" in result
		assert "recommendations" in result
		assert "total_cost" in result
		
		# Verify agent was updated
		service._save_ai_agent.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_track_work_agent_not_found(self, service, tenant_data):
		"""Test tracking work for non-existent AI agent"""
		service._get_ai_agent = AsyncMock(return_value=None)
		
		with pytest.raises(ValueError, match="AI agent .* not found"):
			await service.track_ai_agent_work(
				agent_id="non-existent-agent",
				tenant_id=tenant_data["id"],
				task_data={"completed": True},
				resource_consumption={"cpu_hours": 0.1}
			)


class TestHybridCollaboration:
	"""Test hybrid collaboration functionality"""
	
	@pytest.mark.asyncio
	async def test_start_hybrid_collaboration(self, service, employee_data, ai_agent_data, tenant_data):
		"""Test starting hybrid collaboration session"""
		# Mock dependencies
		mock_employee = TAEmployee(**employee_data, tenant_id=tenant_data["id"], created_by="admin")
		mock_ai_agent = TAAIAgent(**ai_agent_data)
		
		service._get_employee_profile = AsyncMock(return_value=mock_employee)
		service._get_ai_agent = AsyncMock(return_value=mock_ai_agent)
		service._initialize_collaboration_work_allocation = AsyncMock()
		service._setup_collaboration_monitoring = AsyncMock()
		service._setup_collaboration_communication = AsyncMock()
		service._save_hybrid_collaboration = AsyncMock(side_effect=lambda x: x)
		service._send_collaboration_start_notifications = AsyncMock()
		
		result = await service.start_hybrid_collaboration(
			session_name="Product Planning Session",
			project_id="proj-001",
			human_participants=[employee_data["id"]],
			ai_participants=[ai_agent_data["id"]],
			tenant_id=tenant_data["id"],
			session_type="collaborative_work",
			planned_duration_minutes=90,
			created_by=employee_data["id"]
		)
		
		# Assertions
		assert result.session_name == "Product Planning Session"
		assert result.project_id == "proj-001"
		assert result.tenant_id == tenant_data["id"]
		assert len(result.human_participants) == 1
		assert len(result.ai_participants) == 1
		assert result.session_lead == employee_data["id"]
		assert result.planned_duration_minutes == 90
		
		# Verify setup methods were called
		service._initialize_collaboration_work_allocation.assert_called_once()
		service._setup_collaboration_monitoring.assert_called_once()
		service._setup_collaboration_communication.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_collaboration_invalid_participants(self, service, tenant_data):
		"""Test collaboration with invalid participants"""
		# Mock non-existent human participant
		service._get_employee_profile = AsyncMock(return_value=None)
		
		with pytest.raises(ValueError, match="Human participant .* not found"):
			await service.start_hybrid_collaboration(
				session_name="Invalid Session",
				project_id="proj-001",
				human_participants=["non-existent-human"],
				ai_participants=["ai-agent-001"],
				tenant_id=tenant_data["id"],
				created_by="admin"
			)


class TestPredictiveAnalytics:
	"""Test predictive analytics functionality"""
	
	@pytest.mark.asyncio
	async def test_generate_workforce_predictions(self, service, tenant_data):
		"""Test generating workforce predictions"""
		# Mock dependencies
		service._gather_historical_data = AsyncMock(return_value={"sample": "data"})
		service._initialize_prediction_models = AsyncMock()
		service._predict_staffing_requirements = AsyncMock(return_value={"predicted_staffing": "data"})
		service._predict_absence_patterns = AsyncMock(return_value={"absence_patterns": "data"})
		service._predict_overtime_costs = AsyncMock(return_value={"overtime_costs": "data"})
		service._analyze_productivity_trends = AsyncMock(return_value={"productivity_trends": "data"})
		service._identify_efficiency_opportunities = AsyncMock(return_value=["opportunity1", "opportunity2"])
		service._generate_cost_optimization = AsyncMock(return_value={"cost_optimization": "data"})
		service._analyze_compliance_risks = AsyncMock(return_value=["risk1", "risk2"])
		service._analyze_operational_risks = AsyncMock(return_value=["op_risk1", "op_risk2"])
		service._generate_actionable_insights = AsyncMock(return_value=["insight1", "insight2"])
		service._calculate_projected_savings = AsyncMock(return_value=Decimal("50000.00"))
		service._calculate_roi_estimates = AsyncMock(return_value={"roi": "3.5x"})
		service._save_analytics_report = AsyncMock(side_effect=lambda x: x)
		
		result = await service.generate_workforce_predictions(
			tenant_id=tenant_data["id"],
			prediction_period_days=30,
			departments=["dept-001", "dept-002"]
		)
		
		# Assertions
		assert result.tenant_id == tenant_data["id"]
		assert result.analysis_type == "workforce_optimization"
		assert result.model_confidence == 0.85
		assert result.projected_savings == Decimal("50000.00")
		
		# Verify all prediction methods were called
		service._gather_historical_data.assert_called_once()
		service._predict_staffing_requirements.assert_called_once()
		service._predict_absence_patterns.assert_called_once()
		service._predict_overtime_costs.assert_called_once()


class TestHelperMethods:
	"""Test service helper methods"""
	
	@pytest.mark.asyncio
	async def test_calculate_work_hours(self, service):
		"""Test work hours calculation"""
		# Create time entry with clock in/out times
		time_entry = TATimeEntry(
			tenant_id="test-tenant",
			employee_id="emp-001",
			entry_date=date.today(),
			clock_in=datetime.now().replace(hour=9, minute=0, second=0, microsecond=0),
			clock_out=datetime.now().replace(hour=17, minute=30, second=0, microsecond=0),
			created_by="emp-001"
		)
		
		# Mock config for break deduction
		service.config.compliance.break_auto_deduction = True
		service.config.compliance.minimum_break_minutes = 30
		service.config.compliance.daily_overtime_threshold_hours = 8.0
		
		await service._calculate_work_hours(time_entry)
		
		# Should be 8.0 hours (8.5 - 0.5 break) total
		# 8.0 regular, 0.0 overtime (within threshold)
		assert time_entry.total_hours == Decimal("8.0")
		assert time_entry.regular_hours == Decimal("8.0")
		assert time_entry.overtime_hours == Decimal("0.0")
		assert time_entry.break_minutes == 30
	
	@pytest.mark.asyncio
	async def test_calculate_work_hours_with_overtime(self, service):
		"""Test work hours calculation with overtime"""
		# Create time entry with overtime
		time_entry = TATimeEntry(
			tenant_id="test-tenant",
			employee_id="emp-001",
			entry_date=date.today(),
			clock_in=datetime.now().replace(hour=8, minute=0, second=0, microsecond=0),
			clock_out=datetime.now().replace(hour=19, minute=0, second=0, microsecond=0),
			created_by="emp-001"
		)
		
		# Mock config
		service.config.compliance.break_auto_deduction = True
		service.config.compliance.minimum_break_minutes = 60
		service.config.compliance.daily_overtime_threshold_hours = 8.0
		
		await service._calculate_work_hours(time_entry)
		
		# Should be 10.0 hours total (11 - 1 break)
		# 8.0 regular, 2.0 overtime
		assert time_entry.total_hours == Decimal("10.0")
		assert time_entry.regular_hours == Decimal("8.0")
		assert time_entry.overtime_hours == Decimal("2.0")
	
	@pytest.mark.asyncio
	async def test_requires_approval_logic(self, service):
		"""Test approval requirement logic"""
		# Create test time entry and employee
		employee = TAEmployee(
			tenant_id="test-tenant",
			employee_number="EMP001",
			first_name="Test",
			last_name="Employee",
			email="test@example.com",
			hire_date=date.today(),
			created_by="admin"
		)
		
		time_entry = TATimeEntry(
			tenant_id="test-tenant",
			employee_id=employee.id,
			entry_date=date.today(),
			anomaly_score=0.3,  # Low anomaly score
			total_hours=Decimal("8.0"),  # Normal hours
			fraud_indicators=[],
			created_by=employee.id
		)
		
		# Mock config for auto-approval
		service.config.workflow.auto_approval_enabled = True
		service.config.workflow.auto_approval_threshold_hours = 10.0
		
		requires_approval = await service._requires_approval(time_entry, employee)
		
		# Should not require approval (low anomaly, normal hours, no fraud)
		assert requires_approval == False
	
	@pytest.mark.asyncio
	async def test_requires_approval_high_anomaly(self, service):
		"""Test approval required for high anomaly score"""
		employee = TAEmployee(
			tenant_id="test-tenant",
			employee_number="EMP001",
			first_name="Test",
			last_name="Employee",
			email="test@example.com",
			hire_date=date.today(),
			created_by="admin"
		)
		
		time_entry = TATimeEntry(
			tenant_id="test-tenant",
			employee_id=employee.id,
			entry_date=date.today(),
			anomaly_score=0.8,  # High anomaly score
			total_hours=Decimal("8.0"),
			fraud_indicators=[],
			created_by=employee.id
		)
		
		service.config.workflow.auto_approval_enabled = True
		
		requires_approval = await service._requires_approval(time_entry, employee)
		
		# Should require approval due to high anomaly score
		assert requires_approval == True


if __name__ == "__main__":
	pytest.main([__file__, "-v"])