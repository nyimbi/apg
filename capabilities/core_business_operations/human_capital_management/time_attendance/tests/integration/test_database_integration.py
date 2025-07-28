"""
Time & Attendance Database Integration Tests

Integration tests for database operations, multi-tenant isolation,
and data consistency in the Time & Attendance capability.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from sqlalchemy import text

from database import DatabaseManager, get_db_session
from models import (
	TAEmployee, TATimeEntry, TARemoteWorker, TAAIAgent, TAHybridCollaboration,
	TimeEntryStatus, TimeEntryType, WorkMode, AIAgentType, RemoteWorkStatus
)


@pytest.mark.integration
class TestDatabaseConnection:
	"""Test database connection and basic operations"""
	
	@pytest.mark.asyncio
	async def test_database_initialization(self, test_config):
		"""Test database manager initialization"""
		db_manager = DatabaseManager(test_config)
		await db_manager.initialize()
		
		assert db_manager.engines['async'] is not None
		assert db_manager.sessions['async'] is not None
		
		await db_manager.cleanup()
	
	@pytest.mark.asyncio
	async def test_database_session_creation(self, db_manager):
		"""Test database session creation"""
		session = await db_manager.get_session()
		
		assert session is not None
		
		# Test simple query
		result = await session.execute(text("SELECT 1 as test"))
		row = result.fetchone()
		assert row[0] == 1
		
		await session.close()
	
	@pytest.mark.asyncio
	async def test_tenant_schema_creation(self, db_manager, tenant_data):
		"""Test creating tenant-specific schema"""
		schema_name = await db_manager.create_tenant_schema(
			tenant_data["id"],
			tenant_data["name"]
		)
		
		assert schema_name.startswith("ta_tenant_")
		
		# Verify schema was created
		tenant_schema = await db_manager.get_tenant_schema(tenant_data["id"])
		assert tenant_schema == schema_name


@pytest.mark.integration
class TestEmployeeOperations:
	"""Test employee database operations"""
	
	@pytest.mark.asyncio
	async def test_create_employee(self, db_session, employee_data, tenant_data):
		"""Test creating employee record"""
		employee = TAEmployee(
			**employee_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		
		db_session.add(employee)
		await db_session.commit()
		await db_session.refresh(employee)
		
		assert employee.id is not None
		assert employee.employee_number == employee_data["employee_number"]
		assert employee.email == employee_data["email"]
		assert employee.full_name == f"{employee_data['first_name']} {employee_data['last_name']}"
	
	@pytest.mark.asyncio
	async def test_employee_unique_constraints(self, db_session, employee_data, tenant_data):
		"""Test employee unique constraints"""
		# Create first employee
		employee1 = TAEmployee(
			**employee_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		db_session.add(employee1)
		await db_session.commit()
		
		# Try to create duplicate employee (same email)
		employee2_data = employee_data.copy()
		employee2_data["employee_number"] = "EMP002"
		employee2 = TAEmployee(
			**employee2_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		
		db_session.add(employee2)
		
		# Should raise integrity error due to unique email constraint
		with pytest.raises(Exception):  # SQLAlchemy will raise an integrity error
			await db_session.commit()
	
	@pytest.mark.asyncio
	async def test_employee_query_operations(self, db_session, generate_employees, tenant_data):
		"""Test employee query operations"""
		# Create multiple employees
		employees_data = generate_employees(5, tenant_data["id"])
		
		employees = []
		for emp_data in employees_data:
			employee = TAEmployee(**emp_data, created_by="admin")
			employees.append(employee)
			db_session.add(employee)
		
		await db_session.commit()
		
		# Test count query
		from sqlalchemy import select, func
		count_result = await db_session.execute(
			select(func.count(TAEmployee.id)).where(TAEmployee.tenant_id == tenant_data["id"])
		)
		count = count_result.scalar()
		assert count == 5
		
		# Test filter by department
		dept_result = await db_session.execute(
			select(TAEmployee).where(
				TAEmployee.tenant_id == tenant_data["id"],
				TAEmployee.department_id == "dept-001"
			)
		)
		dept_employees = dept_result.scalars().all()
		assert len(dept_employees) >= 1


@pytest.mark.integration
class TestTimeEntryOperations:
	"""Test time entry database operations"""
	
	@pytest.mark.asyncio
	async def test_create_time_entry(self, db_session, employee_data, tenant_data):
		"""Test creating time entry record"""
		# First create employee
		employee = TAEmployee(
			**employee_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		db_session.add(employee)
		await db_session.commit()
		await db_session.refresh(employee)
		
		# Create time entry
		time_entry = TATimeEntry(
			tenant_id=tenant_data["id"],
			employee_id=employee.id,
			entry_date=date.today(),
			clock_in=datetime.now().replace(hour=9, minute=0),
			clock_out=datetime.now().replace(hour=17, minute=30),
			entry_type=TimeEntryType.REGULAR,
			status=TimeEntryStatus.SUBMITTED,
			created_by=employee.id
		)
		
		db_session.add(time_entry)
		await db_session.commit()
		await db_session.refresh(time_entry)
		
		assert time_entry.id is not None
		assert time_entry.employee_id == employee.id
		assert time_entry.entry_date == date.today()
	
	@pytest.mark.asyncio
	async def test_time_entry_relationships(self, db_session, employee_data, tenant_data):
		"""Test time entry relationships with employee"""
		# Create employee
		employee = TAEmployee(
			**employee_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		db_session.add(employee)
		await db_session.commit()
		await db_session.refresh(employee)
		
		# Create multiple time entries
		time_entries = []
		for i in range(3):
			entry_date = date.today() - timedelta(days=i)
			time_entry = TATimeEntry(
				tenant_id=tenant_data["id"],
				employee_id=employee.id,
				entry_date=entry_date,
				clock_in=datetime.combine(entry_date, datetime.min.time().replace(hour=9)),
				clock_out=datetime.combine(entry_date, datetime.min.time().replace(hour=17, minute=30)),
				total_hours=Decimal("8.5"),
				regular_hours=Decimal("8.0"),
				overtime_hours=Decimal("0.5"),
				status=TimeEntryStatus.APPROVED,
				created_by=employee.id
			)
			time_entries.append(time_entry)
			db_session.add(time_entry)
		
		await db_session.commit()
		
		# Query time entries for employee
		from sqlalchemy import select
		result = await db_session.execute(
			select(TATimeEntry).where(TATimeEntry.employee_id == employee.id)
		)
		retrieved_entries = result.scalars().all()
		
		assert len(retrieved_entries) == 3
		for entry in retrieved_entries:
			assert entry.employee_id == employee.id
	
	@pytest.mark.asyncio
	async def test_time_entry_calculations(self, db_session, employee_data, tenant_data):
		"""Test time entry hour calculations"""
		# Create employee
		employee = TAEmployee(
			**employee_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		db_session.add(employee)
		await db_session.commit()
		await db_session.refresh(employee)
		
		# Create time entry with specific hours
		clock_in = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
		clock_out = datetime.now().replace(hour=18, minute=30, second=0, microsecond=0)
		
		time_entry = TATimeEntry(
			tenant_id=tenant_data["id"],
			employee_id=employee.id,
			entry_date=date.today(),
			clock_in=clock_in,
			clock_out=clock_out,
			created_by=employee.id
		)
		
		# Calculate total hours (should be 10.5 hours)
		calculated_hours = time_entry.calculate_total_hours()
		assert calculated_hours == Decimal("10.5")
		
		time_entry.total_hours = calculated_hours
		db_session.add(time_entry)
		await db_session.commit()
		
		# Verify saved correctly
		await db_session.refresh(time_entry)
		assert time_entry.total_hours == Decimal("10.5")


@pytest.mark.integration
class TestRemoteWorkerOperations:
	"""Test remote worker database operations"""
	
	@pytest.mark.asyncio
	async def test_create_remote_worker(self, db_session, employee_data, tenant_data):
		"""Test creating remote worker record"""
		# Create employee first
		employee = TAEmployee(
			**employee_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		db_session.add(employee)
		await db_session.commit()
		await db_session.refresh(employee)
		
		# Create remote worker
		remote_worker = TARemoteWorker(
			tenant_id=tenant_data["id"],
			employee_id=employee.id,
			workspace_id="ws-home-001",
			work_mode=WorkMode.REMOTE_ONLY,
			home_office_setup={
				"location": "Home Office",
				"equipment": {"computer": "MacBook Pro", "monitor": "4K Display"}
			},
			timezone="America/New_York",
			current_activity=RemoteWorkStatus.ACTIVE_WORKING,
			created_by=employee.id
		)
		
		db_session.add(remote_worker)
		await db_session.commit()
		await db_session.refresh(remote_worker)
		
		assert remote_worker.id is not None
		assert remote_worker.employee_id == employee.id
		assert remote_worker.work_mode == WorkMode.REMOTE_ONLY
		assert remote_worker.timezone == "America/New_York"
	
	@pytest.mark.asyncio
	async def test_remote_worker_productivity_tracking(self, db_session, employee_data, tenant_data):
		"""Test remote worker productivity metrics storage"""
		# Create employee and remote worker
		employee = TAEmployee(
			**employee_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		db_session.add(employee)
		await db_session.commit()
		await db_session.refresh(employee)
		
		remote_worker = TARemoteWorker(
			tenant_id=tenant_data["id"],
			employee_id=employee.id,
			workspace_id="ws-home-001",
			work_mode=WorkMode.HYBRID,
			created_by=employee.id
		)
		
		# Add productivity metrics
		productivity_metrics = [
			{
				"timestamp": datetime.now().isoformat(),
				"metric_type": "task_completion",
				"score": 0.85,
				"data": {"tasks_completed": 5, "tasks_planned": 6}
			},
			{
				"timestamp": datetime.now().isoformat(),
				"metric_type": "focus_time",
				"score": 0.90,
				"data": {"focus_minutes": 240, "total_minutes": 480}
			}
		]
		
		remote_worker.productivity_metrics = productivity_metrics
		remote_worker.overall_productivity_score = 0.875
		
		db_session.add(remote_worker)
		await db_session.commit()
		await db_session.refresh(remote_worker)
		
		assert len(remote_worker.productivity_metrics) == 2
		assert remote_worker.overall_productivity_score == 0.875
		assert remote_worker.productivity_metrics[0]["metric_type"] == "task_completion"


@pytest.mark.integration
class TestAIAgentOperations:
	"""Test AI agent database operations"""
	
	@pytest.mark.asyncio
	async def test_create_ai_agent(self, db_session, ai_agent_data):
		"""Test creating AI agent record"""
		ai_agent = TAAIAgent(**ai_agent_data, created_by="admin")
		
		db_session.add(ai_agent)
		await db_session.commit()
		await db_session.refresh(ai_agent)
		
		assert ai_agent.id is not None
		assert ai_agent.agent_name == ai_agent_data["agent_name"]
		assert ai_agent.agent_type == ai_agent_data["agent_type"]
		assert ai_agent.health_status == "healthy"
	
	@pytest.mark.asyncio
	async def test_ai_agent_resource_tracking(self, db_session, ai_agent_data):
		"""Test AI agent resource consumption tracking"""
		ai_agent = TAAIAgent(**ai_agent_data, created_by="admin")
		
		# Simulate work tracking
		ai_agent.tasks_completed = 100
		ai_agent.cpu_hours = Decimal("5.5")
		ai_agent.gpu_hours = Decimal("2.0")
		ai_agent.memory_usage_gb_hours = Decimal("10.0")
		ai_agent.api_calls_count = 1500
		ai_agent.total_operational_cost = Decimal("25.50")
		ai_agent.accuracy_score = 0.96
		ai_agent.error_rate = 0.02
		
		# Calculate derived metrics
		if ai_agent.tasks_completed > 0:
			ai_agent.cost_per_task = ai_agent.total_operational_cost / ai_agent.tasks_completed
		
		db_session.add(ai_agent)
		await db_session.commit()
		await db_session.refresh(ai_agent)
		
		assert ai_agent.tasks_completed == 100
		assert ai_agent.cost_per_task == Decimal("0.2550")
		assert ai_agent.accuracy_score == 0.96
		assert ai_agent.error_rate == 0.02


@pytest.mark.integration
class TestHybridCollaborationOperations:
	"""Test hybrid collaboration database operations"""
	
	@pytest.mark.asyncio
	async def test_create_hybrid_collaboration(self, db_session, employee_data, ai_agent_data, tenant_data):
		"""Test creating hybrid collaboration session"""
		# Create employee and AI agent first
		employee = TAEmployee(
			**employee_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		db_session.add(employee)
		
		ai_agent = TAAIAgent(**ai_agent_data, created_by="admin")
		db_session.add(ai_agent)
		
		await db_session.commit()
		await db_session.refresh(employee)
		await db_session.refresh(ai_agent)
		
		# Create hybrid collaboration
		collaboration = TAHybridCollaboration(
			tenant_id=tenant_data["id"],
			session_name="Product Planning Session",
			project_id="proj-001",
			session_type="collaborative_work",
			human_participants=[employee.id],
			ai_participants=[ai_agent.id],
			session_lead=employee.id,
			start_time=datetime.now(),
			planned_duration_minutes=90,
			created_by=employee.id
		)
		
		db_session.add(collaboration)
		await db_session.commit()
		await db_session.refresh(collaboration)
		
		assert collaboration.id is not None
		assert collaboration.session_name == "Product Planning Session"
		assert len(collaboration.human_participants) == 1
		assert len(collaboration.ai_participants) == 1
		assert collaboration.session_lead == employee.id
	
	@pytest.mark.asyncio
	async def test_collaboration_work_allocation(self, db_session, employee_data, ai_agent_data, tenant_data):
		"""Test collaboration work allocation tracking"""
		# Create dependencies
		employee = TAEmployee(
			**employee_data,
			tenant_id=tenant_data["id"],
			created_by="admin"
		)
		db_session.add(employee)
		
		ai_agent = TAAIAgent(**ai_agent_data, created_by="admin")
		db_session.add(ai_agent)
		
		await db_session.commit()
		await db_session.refresh(employee)
		await db_session.refresh(ai_agent)
		
		# Create collaboration with work allocation
		collaboration = TAHybridCollaboration(
			tenant_id=tenant_data["id"],
			session_name="Data Analysis Session",
			project_id="proj-002",
			human_participants=[employee.id],
			ai_participants=[ai_agent.id],
			session_lead=employee.id,
			start_time=datetime.now(),
			end_time=datetime.now() + timedelta(minutes=120),
			planned_duration_minutes=120,
			work_allocation={
				"human_tasks": {
					employee.id: ["strategy_definition", "final_review"]
				},
				"ai_tasks": {
					ai_agent.id: ["data_processing", "pattern_analysis"]
				},
				"collaborative_tasks": ["insight_generation", "recommendation_synthesis"]
			},
			collaboration_effectiveness=0.92,
			created_by=employee.id
		)
		
		db_session.add(collaboration)
		await db_session.commit()
		await db_session.refresh(collaboration)
		
		assert collaboration.actual_duration_minutes == 120
		assert collaboration.collaboration_effectiveness == 0.92
		assert "human_tasks" in collaboration.work_allocation
		assert "ai_tasks" in collaboration.work_allocation


@pytest.mark.integration
class TestMultiTenantIsolation:
	"""Test multi-tenant data isolation"""
	
	@pytest.mark.asyncio
	async def test_tenant_data_isolation(self, db_session):
		"""Test that tenant data is properly isolated"""
		# Create employees for different tenants
		tenant1_data = {"id": "tenant-001", "name": "Tenant 1"}
		tenant2_data = {"id": "tenant-002", "name": "Tenant 2"}
		
		employee1 = TAEmployee(
			tenant_id=tenant1_data["id"],
			employee_number="EMP001",
			first_name="John",
			last_name="Doe",
			email="john@tenant1.com",
			hire_date=date.today(),
			created_by="admin"
		)
		
		employee2 = TAEmployee(
			tenant_id=tenant2_data["id"],
			employee_number="EMP001",  # Same employee number, different tenant
			first_name="Jane",
			last_name="Smith",
			email="jane@tenant2.com",
			hire_date=date.today(),
			created_by="admin"
		)
		
		db_session.add_all([employee1, employee2])
		await db_session.commit()
		
		# Query employees for tenant 1 only
		from sqlalchemy import select
		tenant1_result = await db_session.execute(
			select(TAEmployee).where(TAEmployee.tenant_id == tenant1_data["id"])
		)
		tenant1_employees = tenant1_result.scalars().all()
		
		# Should only return tenant 1 employee
		assert len(tenant1_employees) == 1
		assert tenant1_employees[0].email == "john@tenant1.com"
		
		# Query employees for tenant 2 only
		tenant2_result = await db_session.execute(
			select(TAEmployee).where(TAEmployee.tenant_id == tenant2_data["id"])
		)
		tenant2_employees = tenant2_result.scalars().all()
		
		# Should only return tenant 2 employee
		assert len(tenant2_employees) == 1
		assert tenant2_employees[0].email == "jane@tenant2.com"


@pytest.mark.integration
class TestDatabasePerformance:
	"""Test database performance and indexing"""
	
	@pytest.mark.asyncio
	async def test_query_performance_with_indexes(self, db_session, generate_employees, generate_time_entries, tenant_data):
		"""Test query performance with proper indexing"""
		# Create multiple employees
		employees_data = generate_employees(20, tenant_data["id"])
		
		employees = []
		for emp_data in employees_data:
			employee = TAEmployee(**emp_data, created_by="admin")
			employees.append(employee)
			db_session.add(employee)
		
		await db_session.commit()
		
		# Create time entries for each employee
		for employee in employees[:5]:  # Limit to first 5 employees for test performance
			await db_session.refresh(employee)
			time_entries_data = generate_time_entries(employee.id, 30, tenant_data["id"])
			
			for entry_data in time_entries_data:
				time_entry = TATimeEntry(**entry_data, created_by=employee.id)
				db_session.add(time_entry)
		
		await db_session.commit()
		
		# Test indexed query performance
		import time
		from sqlalchemy import select
		
		start_time = time.time()
		
		# Query that should benefit from indexing
		result = await db_session.execute(
			select(TATimeEntry).where(
				TATimeEntry.tenant_id == tenant_data["id"],
				TATimeEntry.entry_date >= date.today() - timedelta(days=7)
			).limit(100)
		)
		entries = result.scalars().all()
		
		end_time = time.time()
		query_time = end_time - start_time
		
		# Query should complete quickly (under 1 second for this dataset)
		assert query_time < 1.0
		assert len(entries) > 0


if __name__ == "__main__":
	pytest.main([__file__, "-v"])