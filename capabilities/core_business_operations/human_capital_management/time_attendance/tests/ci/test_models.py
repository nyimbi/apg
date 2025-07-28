"""
Time & Attendance Models Tests

Comprehensive unit tests for Pydantic v2 models with validation,
computed fields, and business logic verification.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
from decimal import Decimal
from datetime import datetime, date, timedelta
from pydantic import ValidationError
from uuid import uuid4

from models import (
	TAEmployee, TATimeEntry, TASchedule, TALeaveRequest, TAFraudDetection,
	TABiometricAuthentication, TAPredictiveAnalytics, TAComplianceRule,
	TARemoteWorker, TAAIAgent, TAHybridCollaboration,
	TimeEntryStatus, TimeEntryType, AttendanceStatus, BiometricType,
	DeviceType, FraudType, LeaveType, ApprovalStatus, WorkforceType,
	WorkMode, AIAgentType, ProductivityMetric, RemoteWorkStatus,
	_validate_confidence_score, _validate_geolocation
)


class TestValidationFunctions:
	"""Test validation utility functions"""
	
	def test_validate_confidence_score_valid(self):
		"""Test confidence score validation with valid values"""
		assert _validate_confidence_score(0.0) == 0.0
		assert _validate_confidence_score(0.5) == 0.5
		assert _validate_confidence_score(1.0) == 1.0
		assert _validate_confidence_score(0.999) == 0.999
	
	def test_validate_confidence_score_invalid(self):
		"""Test confidence score validation with invalid values"""
		with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
			_validate_confidence_score(-0.1)
		
		with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
			_validate_confidence_score(1.1)
		
		with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
			_validate_confidence_score(2.0)
	
	def test_validate_geolocation_valid(self):
		"""Test geolocation validation with valid coordinates"""
		valid_location = {"latitude": 40.7128, "longitude": -74.0060}
		result = _validate_geolocation(valid_location)
		assert result == valid_location
		
		# Test edge cases
		edge_cases = [
			{"latitude": 90.0, "longitude": 180.0},
			{"latitude": -90.0, "longitude": -180.0},
			{"latitude": 0.0, "longitude": 0.0}
		]
		
		for location in edge_cases:
			assert _validate_geolocation(location) == location
	
	def test_validate_geolocation_invalid(self):
		"""Test geolocation validation with invalid coordinates"""
		invalid_locations = [
			{"latitude": 91.0, "longitude": 0.0},  # Invalid latitude
			{"latitude": -91.0, "longitude": 0.0},  # Invalid latitude
			{"latitude": 0.0, "longitude": 181.0},  # Invalid longitude
			{"latitude": 0.0, "longitude": -181.0},  # Invalid longitude
			{"latitude": "invalid", "longitude": 0.0},  # Non-numeric
			{"longitude": 0.0},  # Missing latitude
			{"latitude": 0.0},  # Missing longitude
			{}  # Empty dict
		]
		
		for location in invalid_locations:
			with pytest.raises(ValueError):
				_validate_geolocation(location)


class TestTAEmployee:
	"""Test TAEmployee model"""
	
	def test_create_valid_employee(self, tenant_data):
		"""Test creating valid employee"""
		employee = TAEmployee(
			tenant_id=tenant_data["id"],
			employee_number="EMP001",
			first_name="John",
			last_name="Doe",
			email="john.doe@testcorp.com",
			hire_date=date.today(),
			created_by="admin"
		)
		
		assert employee.tenant_id == tenant_data["id"]
		assert employee.employee_number == "EMP001"
		assert employee.full_name == "John Doe"
		assert employee.employment_status == "active"
		assert employee.workforce_type == WorkforceType.HUMAN
	
	def test_employee_computed_fields(self):
		"""Test computed fields"""
		employee = TAEmployee(
			tenant_id="test-tenant",
			employee_number="EMP001",
			first_name="Jane",
			last_name="Smith",
			email="jane.smith@testcorp.com",
			hire_date=date.today() - timedelta(days=365),
			created_by="admin"
		)
		
		assert employee.full_name == "Jane Smith"
		assert employee.years_of_service >= 1
	
	def test_employee_validation_errors(self):
		"""Test employee validation errors"""
		# Missing required fields
		with pytest.raises(ValidationError):
			TAEmployee()
		
		# Invalid email
		with pytest.raises(ValidationError):
			TAEmployee(
				tenant_id="test-tenant",
				employee_number="EMP001",
				first_name="John",
				last_name="Doe",
				email="invalid-email",
				hire_date=date.today(),
				created_by="admin"
			)
		
		# Future hire date
		with pytest.raises(ValidationError, match="Hire date cannot be in the future"):
			TAEmployee(
				tenant_id="test-tenant",
				employee_number="EMP001",
				first_name="John",
				last_name="Doe",
				email="john.doe@testcorp.com",
				hire_date=date.today() + timedelta(days=1),
				created_by="admin"
			)


class TestTATimeEntry:
	"""Test TATimeEntry model"""
	
	def test_create_valid_time_entry(self, employee_data, tenant_data):
		"""Test creating valid time entry"""
		time_entry = TATimeEntry(
			tenant_id=tenant_data["id"],
			employee_id=employee_data["id"],
			entry_date=date.today(),
			clock_in=datetime.now().replace(hour=9, minute=0),
			clock_out=datetime.now().replace(hour=17, minute=30),
			created_by=employee_data["id"]
		)
		
		assert time_entry.tenant_id == tenant_data["id"]
		assert time_entry.employee_id == employee_data["id"]
		assert time_entry.entry_type == TimeEntryType.REGULAR
		assert time_entry.status == TimeEntryStatus.DRAFT
	
	def test_time_entry_hours_calculation(self, employee_data, tenant_data):
		"""Test automatic hours calculation"""
		clock_in = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
		clock_out = datetime.now().replace(hour=17, minute=30, second=0, microsecond=0)
		
		time_entry = TATimeEntry(
			tenant_id=tenant_data["id"],
			employee_id=employee_data["id"],
			entry_date=date.today(),
			clock_in=clock_in,
			clock_out=clock_out,
			created_by=employee_data["id"]
		)
		
		# Should calculate 8.5 hours
		expected_hours = Decimal("8.5")
		assert time_entry.calculate_total_hours() == expected_hours
	
	def test_time_entry_validation_errors(self, employee_data, tenant_data):
		"""Test time entry validation errors"""
		# Clock out before clock in
		with pytest.raises(ValidationError, match="Clock out time must be after clock in time"):
			TATimeEntry(
				tenant_id=tenant_data["id"],
				employee_id=employee_data["id"],
				entry_date=date.today(),
				clock_in=datetime.now().replace(hour=17, minute=0),
				clock_out=datetime.now().replace(hour=9, minute=0),
				created_by=employee_data["id"]
			)
		
		# Invalid confidence scores
		with pytest.raises(ValidationError):
			TATimeEntry(
				tenant_id=tenant_data["id"],
				employee_id=employee_data["id"],
				entry_date=date.today(),
				verification_confidence=1.5,  # Invalid confidence score
				created_by=employee_data["id"]
			)


class TestTARemoteWorker:
	"""Test TARemoteWorker model"""
	
	def test_create_valid_remote_worker(self, employee_data, tenant_data):
		"""Test creating valid remote worker"""
		remote_worker = TARemoteWorker(
			tenant_id=tenant_data["id"],
			employee_id=employee_data["id"],
			workspace_id="ws-home-001",
			work_mode=WorkMode.REMOTE_ONLY,
			created_by=employee_data["id"]
		)
		
		assert remote_worker.tenant_id == tenant_data["id"]
		assert remote_worker.employee_id == employee_data["id"]
		assert remote_worker.work_mode == WorkMode.REMOTE_ONLY
		assert remote_worker.current_activity == RemoteWorkStatus.OFFLINE
		assert remote_worker.overall_productivity_score == 0.0
		assert remote_worker.work_life_balance_score == 0.8
	
	def test_remote_worker_computed_fields(self, employee_data, tenant_data):
		"""Test remote worker computed fields"""
		# Add some productivity metrics
		productivity_metrics = [
			{
				"timestamp": datetime.now().isoformat(),
				"metric_type": "task_completion",
				"score": 0.85,
				"data": {"tasks_completed": 5}
			},
			{
				"timestamp": datetime.now().isoformat(),
				"metric_type": "focus_time",
				"score": 0.90,
				"data": {"focus_minutes": 240}
			}
		]
		
		remote_worker = TARemoteWorker(
			tenant_id=tenant_data["id"],
			employee_id=employee_data["id"],
			workspace_id="ws-home-001",
			work_mode=WorkMode.HYBRID,
			productivity_metrics=productivity_metrics,
			created_by=employee_data["id"]
		)
		
		assert len(remote_worker.productivity_metrics) == 2
		assert remote_worker.work_mode == WorkMode.HYBRID


class TestTAAIAgent:
	"""Test TAAIAgent model"""
	
	def test_create_valid_ai_agent(self, tenant_data):
		"""Test creating valid AI agent"""
		ai_agent = TAAIAgent(
			tenant_id=tenant_data["id"],
			agent_name="Claude Assistant",
			agent_type=AIAgentType.CONVERSATIONAL_AI,
			capabilities=["nlp", "coding", "analysis"],
			configuration={
				"api_endpoints": {"chat": "/api/chat"},
				"resource_limits": {"max_tokens": 100000}
			},
			created_by="admin"
		)
		
		assert ai_agent.tenant_id == tenant_data["id"]
		assert ai_agent.agent_name == "Claude Assistant"
		assert ai_agent.agent_type == AIAgentType.CONVERSATIONAL_AI
		assert ai_agent.health_status == "healthy"
		assert ai_agent.tasks_completed == 0
		assert ai_agent.overall_performance_score == 0.0
	
	def test_ai_agent_computed_fields(self, tenant_data):
		"""Test AI agent computed fields"""
		ai_agent = TAAIAgent(
			tenant_id=tenant_data["id"],
			agent_name="Test Agent",
			agent_type=AIAgentType.AUTOMATION_BOT,
			capabilities=["automation"],
			configuration={
				"api_endpoints": {"execute": "/api/execute"},
				"resource_limits": {"max_operations": 1000}
			},
			tasks_completed=100,
			total_operational_cost=Decimal("50.00"),
			created_by="admin"
		)
		
		# Test cost per task calculation
		expected_cost_per_task = Decimal("0.5000")
		assert ai_agent.cost_per_task == expected_cost_per_task
	
	def test_ai_agent_validation_errors(self, tenant_data):
		"""Test AI agent validation errors"""
		# Missing required configuration fields
		with pytest.raises(ValidationError, match="configuration must contain 'api_endpoints' field"):
			TAAIAgent(
				tenant_id=tenant_data["id"],
				agent_name="Invalid Agent",
				agent_type=AIAgentType.CONVERSATIONAL_AI,
				capabilities=["nlp"],
				configuration={},  # Missing required fields
				created_by="admin"
			)
		
		# Empty capabilities
		with pytest.raises(ValidationError):
			TAAIAgent(
				tenant_id=tenant_data["id"],
				agent_name="Invalid Agent",
				agent_type=AIAgentType.CONVERSATIONAL_AI,
				capabilities=[],  # Empty capabilities
				configuration={
					"api_endpoints": {"chat": "/api/chat"},
					"resource_limits": {"max_tokens": 100000}
				},
				created_by="admin"
			)


class TestTAHybridCollaboration:
	"""Test TAHybridCollaboration model"""
	
	def test_create_valid_collaboration(self, employee_data, ai_agent_data, tenant_data):
		"""Test creating valid hybrid collaboration"""
		collaboration = TAHybridCollaboration(
			tenant_id=tenant_data["id"],
			session_name="Product Planning",
			project_id="proj-001",
			human_participants=[employee_data["id"]],
			ai_participants=[ai_agent_data["id"]],
			session_lead=employee_data["id"],
			start_time=datetime.now(),
			created_by=employee_data["id"]
		)
		
		assert collaboration.tenant_id == tenant_data["id"]
		assert collaboration.session_name == "Product Planning"
		assert len(collaboration.human_participants) == 1
		assert len(collaboration.ai_participants) == 1
		assert collaboration.session_lead == employee_data["id"]
		assert collaboration.planned_duration_minutes == 60
	
	def test_collaboration_computed_fields(self, employee_data, ai_agent_data, tenant_data):
		"""Test collaboration computed fields"""
		start_time = datetime.now()
		end_time = start_time + timedelta(minutes=90)
		
		collaboration = TAHybridCollaboration(
			tenant_id=tenant_data["id"],
			session_name="Extended Session",
			project_id="proj-002",
			human_participants=[employee_data["id"]],
			ai_participants=[ai_agent_data["id"]],
			session_lead=employee_data["id"],
			start_time=start_time,
			end_time=end_time,
			planned_duration_minutes=90,
			created_by=employee_data["id"]
		)
		
		# Test actual duration calculation
		assert collaboration.actual_duration_minutes == 90
	
	def test_collaboration_validation_errors(self, employee_data, ai_agent_data, tenant_data):
		"""Test collaboration validation errors"""
		# Empty participants
		with pytest.raises(ValidationError):
			TAHybridCollaboration(
				tenant_id=tenant_data["id"],
				session_name="Invalid Session",
				project_id="proj-001",
				human_participants=[],  # Empty
				ai_participants=[ai_agent_data["id"]],
				session_lead=employee_data["id"],
				start_time=datetime.now(),
				created_by=employee_data["id"]
			)
		
		# Invalid duration
		with pytest.raises(ValidationError):
			TAHybridCollaboration(
				tenant_id=tenant_data["id"],
				session_name="Invalid Duration",
				project_id="proj-001",
				human_participants=[employee_data["id"]],
				ai_participants=[ai_agent_data["id"]],
				session_lead=employee_data["id"],
				start_time=datetime.now(),
				planned_duration_minutes=0,  # Invalid duration
				created_by=employee_data["id"]
			)


class TestEnums:
	"""Test enum values and validation"""
	
	def test_time_entry_status_enum(self):
		"""Test TimeEntryStatus enum"""
		assert TimeEntryStatus.DRAFT.value == "draft"
		assert TimeEntryStatus.SUBMITTED.value == "submitted"
		assert TimeEntryStatus.APPROVED.value == "approved"
		assert TimeEntryStatus.REJECTED.value == "rejected"
		assert TimeEntryStatus.PROCESSING.value == "processing"
	
	def test_work_mode_enum(self):
		"""Test WorkMode enum"""
		assert WorkMode.REMOTE_ONLY.value == "remote_only"
		assert WorkMode.HYBRID.value == "hybrid"
		assert WorkMode.OFFICE_ONLY.value == "office_only"
		assert WorkMode.FLEXIBLE.value == "flexible"
	
	def test_ai_agent_type_enum(self):
		"""Test AIAgentType enum"""
		assert AIAgentType.CONVERSATIONAL_AI.value == "conversational_ai"
		assert AIAgentType.AUTOMATION_BOT.value == "automation_bot"
		assert AIAgentType.ANALYTICS_AGENT.value == "analytics_agent"
		assert AIAgentType.CONTENT_GENERATOR.value == "content_generator"
		assert AIAgentType.CODE_ASSISTANT.value == "code_assistant"
	
	def test_productivity_metric_enum(self):
		"""Test ProductivityMetric enum"""
		assert ProductivityMetric.TASK_COMPLETION.value == "task_completion"
		assert ProductivityMetric.FOCUS_TIME.value == "focus_time"
		assert ProductivityMetric.COLLABORATION_QUALITY.value == "collaboration_quality"
		assert ProductivityMetric.OUTPUT_QUALITY.value == "output_quality"
		assert ProductivityMetric.INNOVATION_INDEX.value == "innovation_index"


class TestModelIntegration:
	"""Test model integration and relationships"""
	
	def test_employee_time_entry_relationship(self, employee_data, tenant_data):
		"""Test employee and time entry relationship"""
		employee = TAEmployee(**employee_data, tenant_id=tenant_data["id"], created_by="admin")
		
		time_entry = TATimeEntry(
			tenant_id=tenant_data["id"],
			employee_id=employee.id,
			entry_date=date.today(),
			clock_in=datetime.now().replace(hour=9, minute=0),
			created_by=employee.id
		)
		
		assert time_entry.employee_id == employee.id
		assert time_entry.tenant_id == employee.tenant_id
	
	def test_remote_worker_employee_relationship(self, employee_data, tenant_data):
		"""Test remote worker and employee relationship"""
		employee = TAEmployee(**employee_data, tenant_id=tenant_data["id"], created_by="admin")
		
		remote_worker = TARemoteWorker(
			tenant_id=tenant_data["id"],
			employee_id=employee.id,
			workspace_id="ws-001",
			work_mode=WorkMode.REMOTE_ONLY,
			created_by=employee.id
		)
		
		assert remote_worker.employee_id == employee.id
		assert remote_worker.tenant_id == employee.tenant_id


if __name__ == "__main__":
	pytest.main([__file__, "-v"])