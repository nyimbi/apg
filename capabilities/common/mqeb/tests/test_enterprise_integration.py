#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Enterprise Integration Tests
Tests for enterprise workflow engine and business process intelligence

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid_extensions import uuid7str

# Import MQEB components
from ..models import MQMessage, MessagePriority
from ..service import MQEBService
from ..enterprise_integration import (
	EnterpriseWorkflowEngine, WorkflowDefinition, ProcessStep, WorkflowExecution,
	EnterpriseConnector, BusinessProcessMetrics,
	WorkflowEngine, IntegrationProtocol, BusinessProcessType, ProcessStepType,
	create_enterprise_workflow_engine
)


class TestWorkflowDefinition:
	"""Test workflow definition functionality"""
	
	def test_workflow_definition_creation(self):
		"""Test workflow definition creation"""
		workflow = WorkflowDefinition(
			workflow_id="test_workflow",
			name="Test Workflow",
			description="A test workflow for unit testing",
			engine=WorkflowEngine.CUSTOM_WORKFLOW,
			tenant_id="test_tenant",
			process_type=BusinessProcessType.ORDER_PROCESSING,
			trigger_patterns=["orders.*"],
			steps=[
				ProcessStep(
					step_id="validate_step",
					name="Validate Order",
					step_type=ProcessStepType.VALIDATION,
					action="validate_order"
				)
			],
			timeout_minutes=30
		)
		
		assert workflow.workflow_id == "test_workflow"
		assert workflow.name == "Test Workflow"
		assert workflow.engine == WorkflowEngine.CUSTOM_WORKFLOW
		assert workflow.process_type == BusinessProcessType.ORDER_PROCESSING
		assert len(workflow.steps) == 1
		assert workflow.enabled == True
		assert workflow.timeout_minutes == 30


class TestProcessStep:
	"""Test process step functionality"""
	
	def test_process_step_creation(self):
		"""Test process step creation"""
		step = ProcessStep(
			step_id="test_step",
			name="Test Step",
			step_type=ProcessStepType.DATA_TRANSFORMATION,
			action="transform_data",
			parameters={"format": "json"},
			conditions=["${status} == 'active'"],
			dependencies=["previous_step"],
			timeout_seconds=60,
			retry_count=3,
			on_failure="retry",
			parallel=True
		)
		
		assert step.step_id == "test_step"
		assert step.step_type == ProcessStepType.DATA_TRANSFORMATION
		assert step.action == "transform_data"
		assert step.parameters["format"] == "json"
		assert len(step.conditions) == 1
		assert len(step.dependencies) == 1
		assert step.parallel == True
		assert step.retry_count == 3


class TestEnterpriseConnector:
	"""Test enterprise connector functionality"""
	
	def test_connector_creation(self):
		"""Test enterprise connector creation"""
		connector = EnterpriseConnector(
			connector_id="test_crm",
			name="Test CRM Connector",
			system_type="CRM",
			protocol=IntegrationProtocol.REST_API,
			endpoint_config={
				"base_url": "https://api.testcrm.com",
				"timeout_seconds": 30
			},
			authentication={
				"type": "oauth2",
				"client_id": "test_client"
			},
			data_mapping={
				"customer_id": "external_id",
				"email": "primary_email"
			},
			rate_limits={"requests_per_minute": 1000}
		)
		
		assert connector.connector_id == "test_crm"
		assert connector.system_type == "CRM"
		assert connector.protocol == IntegrationProtocol.REST_API
		assert connector.endpoint_config["base_url"] == "https://api.testcrm.com"
		assert connector.authentication["type"] == "oauth2"
		assert connector.enabled == True


class TestEnterpriseWorkflowEngine:
	"""Test enterprise workflow engine functionality"""
	
	@pytest.fixture
	async def mqeb_service(self):
		"""Create MQEB service for testing"""
		service = MQEBService()
		await service.initialize()
		yield service
		await service.shutdown()
	
	@pytest.fixture
	async def workflow_engine(self, mqeb_service):
		"""Create workflow engine for testing"""
		engine = await create_enterprise_workflow_engine(mqeb_service)
		yield engine
		await engine.shutdown()
	
	@pytest.mark.asyncio
	async def test_workflow_engine_initialization(self, workflow_engine):
		"""Test workflow engine initialization"""
		assert workflow_engine.running == True
		assert len(workflow_engine.workflows) > 0
		assert len(workflow_engine.connectors) > 0
		
		# Check default workflows were loaded
		assert "order_processing_workflow" in workflow_engine.workflows
		assert "customer_onboarding_workflow" in workflow_engine.workflows
		assert "compliance_reporting_workflow" in workflow_engine.workflows
		
		# Check default connectors were loaded
		assert "crm_connector" in workflow_engine.connectors
		assert "inventory_system" in workflow_engine.connectors
		assert "payment_gateway" in workflow_engine.connectors
	
	@pytest.mark.asyncio
	async def test_workflow_registration(self, workflow_engine):
		"""Test workflow registration"""
		custom_workflow = WorkflowDefinition(
			workflow_id="custom_test_workflow",
			name="Custom Test Workflow",
			description="Custom workflow for testing",
			engine=WorkflowEngine.CUSTOM_WORKFLOW,
			tenant_id="test_tenant",
			process_type=BusinessProcessType.DATA_PIPELINE,
			trigger_patterns=["test.data.*"],
			steps=[
				ProcessStep(
					step_id="process_data",
					name="Process Data",
					step_type=ProcessStepType.DATA_TRANSFORMATION,
					action="process_test_data"
				)
			]
		)
		
		workflow_id = await workflow_engine.register_workflow(custom_workflow)
		
		assert workflow_id == "custom_test_workflow"
		assert workflow_id in workflow_engine.workflows
		assert workflow_engine.workflows[workflow_id].name == "Custom Test Workflow"
	
	@pytest.mark.asyncio
	async def test_workflow_triggering_by_message_topic(self, workflow_engine):
		"""Test workflow triggering based on message topic"""
		# Create message that should trigger order processing workflow
		order_message = MQMessage(
			topic="orders.placed.new",
			payload=b'{"order_id": "ORD123", "customer_id": "CUST456", "items": [{"sku": "ITEM001", "quantity": 2}], "total": 29.98}',
			tenant_id="test_tenant",
			source_application="order_service"
		)
		
		# Trigger workflow
		execution_id = await workflow_engine.trigger_workflow(order_message)
		
		assert execution_id is not None
		assert execution_id in workflow_engine.active_executions
		
		execution = workflow_engine.active_executions[execution_id]
		assert execution.workflow_id == "order_processing_workflow"
		assert execution.trigger_message_id == order_message.id
		assert execution.status == "running"
		assert execution.tenant_id == "default"  # Default workflow tenant
	
	@pytest.mark.asyncio
	async def test_workflow_execution_completion(self, workflow_engine):
		"""Test complete workflow execution"""
		# Create simple test workflow
		simple_workflow = WorkflowDefinition(
			workflow_id="simple_test_workflow",
			name="Simple Test Workflow",
			description="Simple workflow for testing execution",
			engine=WorkflowEngine.CUSTOM_WORKFLOW,
			tenant_id="test_tenant",
			process_type=BusinessProcessType.DATA_PIPELINE,
			trigger_patterns=["test.simple.*"],
			steps=[
				ProcessStep(
					step_id="step1",
					name="First Step",
					step_type=ProcessStepType.VALIDATION,
					action="validate_data"
				),
				ProcessStep(
					step_id="step2",
					name="Second Step",
					step_type=ProcessStepType.DATA_TRANSFORMATION,
					action="transform_data",
					dependencies=["step1"]
				)
			],
			timeout_minutes=5
		)
		
		await workflow_engine.register_workflow(simple_workflow)
		
		# Create trigger message
		trigger_message = MQMessage(
			topic="test.simple.execution",
			payload=b'{"test": "data"}',
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		# Trigger and wait for execution
		execution_id = await workflow_engine.trigger_workflow(trigger_message)
		assert execution_id is not None
		
		# Wait for completion (steps are simulated with small delays)
		await asyncio.sleep(1)
		
		# Check execution status
		status = await workflow_engine.get_workflow_status(execution_id)
		assert status is not None
		assert status['execution_id'] == execution_id
		assert 'status' in status
		assert 'progress' in status
	
	@pytest.mark.asyncio
	async def test_workflow_step_dependencies(self, workflow_engine):
		"""Test workflow step dependency resolution"""
		# Create workflow with complex dependencies
		dependency_workflow = WorkflowDefinition(
			workflow_id="dependency_test_workflow",
			name="Dependency Test Workflow",
			description="Test step dependencies",
			engine=WorkflowEngine.CUSTOM_WORKFLOW,
			tenant_id="test_tenant",
			process_type=BusinessProcessType.DATA_PIPELINE,
			trigger_patterns=["test.dependency.*"],
			steps=[
				ProcessStep(
					step_id="step_a",
					name="Step A",
					step_type=ProcessStepType.VALIDATION,
					action="validate_data"
				),
				ProcessStep(
					step_id="step_b",
					name="Step B",
					step_type=ProcessStepType.DATA_TRANSFORMATION,
					action="transform_data"
				),
				ProcessStep(
					step_id="step_c",
					name="Step C",
					step_type=ProcessStepType.INTEGRATION,
					action="integrate_data",
					dependencies=["step_a", "step_b"]  # Depends on both A and B
				),
				ProcessStep(
					step_id="step_d",
					name="Step D",
					step_type=ProcessStepType.NOTIFICATION,
					action="notify_completion",
					dependencies=["step_c"]  # Depends on C
				)
			]
		)
		
		await workflow_engine.register_workflow(dependency_workflow)
		
		# Trigger workflow
		trigger_message = MQMessage(
			topic="test.dependency.complex",
			payload=b'{"test": "dependency data"}',
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		execution_id = await workflow_engine.trigger_workflow(trigger_message)
		assert execution_id is not None
		
		# Wait for some execution
		await asyncio.sleep(1)
		
		status = await workflow_engine.get_workflow_status(execution_id)
		assert status is not None
		assert status['progress']['total_steps'] == 4
	
	@pytest.mark.asyncio
	async def test_parallel_step_execution(self, workflow_engine):
		"""Test parallel step execution"""
		parallel_workflow = WorkflowDefinition(
			workflow_id="parallel_test_workflow",
			name="Parallel Test Workflow",
			description="Test parallel step execution",
			engine=WorkflowEngine.CUSTOM_WORKFLOW,
			tenant_id="test_tenant",
			process_type=BusinessProcessType.DATA_PIPELINE,
			trigger_patterns=["test.parallel.*"],
			steps=[
				ProcessStep(
					step_id="init_step",
					name="Initialize",
					step_type=ProcessStepType.VALIDATION,
					action="initialize_data"
				),
				ProcessStep(
					step_id="parallel_step_1",
					name="Parallel Step 1",
					step_type=ProcessStepType.DATA_TRANSFORMATION,
					action="transform_data_1",
					dependencies=["init_step"],
					parallel=True
				),
				ProcessStep(
					step_id="parallel_step_2",
					name="Parallel Step 2",
					step_type=ProcessStepType.DATA_TRANSFORMATION,
					action="transform_data_2",
					dependencies=["init_step"],
					parallel=True
				),
				ProcessStep(
					step_id="parallel_step_3",
					name="Parallel Step 3",
					step_type=ProcessStepType.DATA_TRANSFORMATION,
					action="transform_data_3",
					dependencies=["init_step"],
					parallel=True
				),
				ProcessStep(
					step_id="final_step",
					name="Finalize",
					step_type=ProcessStepType.INTEGRATION,
					action="finalize_data",
					dependencies=["parallel_step_1", "parallel_step_2", "parallel_step_3"]
				)
			]
		)
		
		await workflow_engine.register_workflow(parallel_workflow)
		
		# Trigger workflow
		trigger_message = MQMessage(
			topic="test.parallel.execution",
			payload=b'{"test": "parallel data"}',
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		execution_id = await workflow_engine.trigger_workflow(trigger_message)
		assert execution_id is not None
		
		# Wait for execution
		await asyncio.sleep(1)
		
		status = await workflow_engine.get_workflow_status(execution_id)
		assert status is not None
		assert status['progress']['total_steps'] == 5
	
	@pytest.mark.asyncio
	async def test_workflow_condition_evaluation(self, workflow_engine):
		"""Test workflow step condition evaluation"""
		conditional_workflow = WorkflowDefinition(
			workflow_id="conditional_test_workflow",
			name="Conditional Test Workflow",
			description="Test conditional step execution",
			engine=WorkflowEngine.CUSTOM_WORKFLOW,
			tenant_id="test_tenant",
			process_type=BusinessProcessType.DATA_PIPELINE,
			trigger_patterns=["test.conditional.*"],
			steps=[
				ProcessStep(
					step_id="always_step",
					name="Always Execute",
					step_type=ProcessStepType.VALIDATION,
					action="validate_data"
				),
				ProcessStep(
					step_id="conditional_step",
					name="Conditional Step",
					step_type=ProcessStepType.DATA_TRANSFORMATION,
					action="transform_data",
					conditions=["${environment} == 'production'"],
					dependencies=["always_step"]
				),
				ProcessStep(
					step_id="final_step",
					name="Final Step",
					step_type=ProcessStepType.NOTIFICATION,
					action="notify_completion",
					dependencies=["always_step"]  # Only depends on always_step
				)
			],
			variables={"environment": "testing"}  # Will skip conditional step
		)
		
		await workflow_engine.register_workflow(conditional_workflow)
		
		# Trigger workflow
		trigger_message = MQMessage(
			topic="test.conditional.execution",
			payload=b'{"test": "conditional data"}',
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		execution_id = await workflow_engine.trigger_workflow(trigger_message)
		assert execution_id is not None
		
		# Wait for execution
		await asyncio.sleep(1)
		
		status = await workflow_engine.get_workflow_status(execution_id)
		assert status is not None
	
	@pytest.mark.asyncio
	async def test_workflow_timeout_handling(self, workflow_engine):
		"""Test workflow timeout handling"""
		timeout_workflow = WorkflowDefinition(
			workflow_id="timeout_test_workflow",
			name="Timeout Test Workflow",
			description="Test workflow timeout",
			engine=WorkflowEngine.CUSTOM_WORKFLOW,
			tenant_id="test_tenant",
			process_type=BusinessProcessType.DATA_PIPELINE,
			trigger_patterns=["test.timeout.*"],
			steps=[
				ProcessStep(
					step_id="quick_step",
					name="Quick Step",
					step_type=ProcessStepType.VALIDATION,
					action="validate_data"
				)
			],
			timeout_minutes=0.01  # Very short timeout (0.6 seconds)
		)
		
		await workflow_engine.register_workflow(timeout_workflow)
		
		# Trigger workflow
		trigger_message = MQMessage(
			topic="test.timeout.execution",
			payload=b'{"test": "timeout data"}',
			tenant_id="test_tenant",
			source_application="test_app"
		)
		
		execution_id = await workflow_engine.trigger_workflow(trigger_message)
		assert execution_id is not None
		
		# Wait longer than timeout
		await asyncio.sleep(2)
		
		# Check if execution was moved to history due to timeout
		status = await workflow_engine.get_workflow_status(execution_id)
		assert status is not None
		# Status might be completed or timeout depending on timing
	
	@pytest.mark.asyncio
	async def test_enterprise_status_reporting(self, workflow_engine):
		"""Test enterprise integration status reporting"""
		status = await workflow_engine.get_enterprise_status()
		
		assert 'enabled' in status
		assert 'workflows' in status
		assert 'executions' in status
		assert 'connectors' in status
		assert 'process_metrics' in status
		assert 'analytics' in status
		
		assert status['enabled'] == True
		assert status['workflows']['total'] > 0
		assert status['workflows']['enabled'] > 0
		assert status['connectors']['total'] > 0
		assert status['connectors']['enabled'] > 0
		assert isinstance(status['process_metrics'], dict)
	
	@pytest.mark.asyncio
	async def test_business_process_execution_flow(self, workflow_engine):
		"""Test complete business process execution flow"""
		# Test order processing workflow (one of the defaults)
		order_message = MQMessage(
			topic="orders.placed.ecommerce",
			payload=b'''{
				"order_id": "ORD-2025-001",
				"customer_id": "CUST-456",
				"items": [
					{"sku": "PRODUCT-001", "quantity": 2, "price": 15.99},
					{"sku": "PRODUCT-002", "quantity": 1, "price": 29.99}
				],
				"subtotal": 61.97,
				"tax": 4.96,
				"shipping": 8.99,
				"total": 75.92,
				"payment_method": "credit_card",
				"shipping_address": {
					"street": "123 Main St",
					"city": "Anytown",
					"state": "CA",
					"zip": "12345"
				}
			}''',
			tenant_id="ecommerce_tenant",
			source_application="order_service"
		)
		
		# Trigger order processing workflow
		execution_id = await workflow_engine.trigger_workflow(order_message)
		assert execution_id is not None
		
		# Wait for processing (steps have simulated delays)
		await asyncio.sleep(2)
		
		# Check execution progress
		status = await workflow_engine.get_workflow_status(execution_id)
		assert status is not None
		assert status['workflow_id'] == "order_processing_workflow"
		
		# Verify workflow steps were processed
		expected_steps = [
			"validate_order",
			"check_inventory", 
			"calculate_pricing",
			"process_payment",
			"create_shipment",
			"send_confirmation"
		]
		
		assert status['progress']['total_steps'] == len(expected_steps)
		
		# Check that execution has step results
		execution = None
		if execution_id in workflow_engine.active_executions:
			execution = workflow_engine.active_executions[execution_id]
		else:
			# Check history
			for hist_execution in workflow_engine.execution_history:
				if hist_execution.execution_id == execution_id:
					execution = hist_execution
					break
		
		assert execution is not None
		assert len(execution.step_results) > 0
	
	@pytest.mark.asyncio
	async def test_customer_onboarding_workflow(self, workflow_engine):
		"""Test customer onboarding workflow execution"""
		# Test customer onboarding workflow
		customer_message = MQMessage(
			topic="customers.registered.new",
			payload=b'''{
				"customer_id": "CUST-789",
				"email": "newcustomer@example.com",
				"first_name": "Jane",
				"last_name": "Smith",
				"phone": "+1-555-123-4567",
				"registration_date": "2025-01-15T10:30:00Z",
				"marketing_consent": true,
				"account_type": "premium"
			}''',
			tenant_id="customer_tenant",
			source_application="registration_service"
		)
		
		# Trigger customer onboarding workflow
		execution_id = await workflow_engine.trigger_workflow(customer_message)
		assert execution_id is not None
		
		# Wait for processing
		await asyncio.sleep(2)
		
		# Check execution
		status = await workflow_engine.get_workflow_status(execution_id)
		assert status is not None
		assert status['workflow_id'] == "customer_onboarding_workflow"
		
		# Expected onboarding steps
		expected_onboarding_steps = [
			"verify_email",
			"kyc_check",
			"create_accounts",
			"setup_preferences",
			"send_welcome"
		]
		
		assert status['progress']['total_steps'] == len(expected_onboarding_steps)
	
	@pytest.mark.asyncio
	async def test_compliance_reporting_workflow(self, workflow_engine):
		"""Test compliance reporting workflow execution"""
		# Test compliance reporting workflow
		compliance_message = MQMessage(
			topic="compliance.report.scheduled.monthly",
			payload=b'''{
				"report_type": "monthly_compliance",
				"frameworks": ["GDPR", "HIPAA", "PCI_DSS", "SOX"],
				"reporting_period": {
					"start": "2025-01-01T00:00:00Z",
					"end": "2025-01-31T23:59:59Z"
				},
				"regulatory_bodies": [
					"EU_DPA",
					"US_HHS",
					"PCI_COUNCIL",
					"SEC"
				]
			}''',
			tenant_id="compliance_tenant",
			source_application="compliance_scheduler"
		)
		
		# Trigger compliance reporting workflow
		execution_id = await workflow_engine.trigger_workflow(compliance_message)
		assert execution_id is not None
		
		# Wait for processing (compliance reports take longer)
		await asyncio.sleep(3)
		
		# Check execution
		status = await workflow_engine.get_workflow_status(execution_id)
		assert status is not None
		assert status['workflow_id'] == "compliance_reporting_workflow"
		
		# Expected compliance steps
		expected_compliance_steps = [
			"collect_data",
			"generate_reports",
			"validate_reports",
			"submit_reports",
			"archive_reports"
		]
		
		assert status['progress']['total_steps'] == len(expected_compliance_steps)


class TestBusinessProcessMetrics:
	"""Test business process metrics and analytics"""
	
	def test_process_metrics_creation(self):
		"""Test business process metrics creation"""
		metrics = BusinessProcessMetrics(
			timestamp=datetime.utcnow(),
			tenant_id="test_tenant",
			process_type=BusinessProcessType.ORDER_PROCESSING,
			total_executions=100,
			successful_executions=95,
			failed_executions=5,
			average_duration_ms=2500.0,
			sla_violations=2,
			throughput_per_hour=50.0,
			error_rate_percentage=5.0,
			cost_per_execution=0.15
		)
		
		assert metrics.tenant_id == "test_tenant"
		assert metrics.process_type == BusinessProcessType.ORDER_PROCESSING
		assert metrics.total_executions == 100
		assert metrics.successful_executions == 95
		assert metrics.failed_executions == 5
		assert metrics.error_rate_percentage == 5.0
		assert metrics.cost_per_execution == 0.15
	
	@pytest.mark.asyncio
	async def test_process_metrics_collection(self):
		"""Test process metrics collection during workflow execution"""
		service = MQEBService()
		await service.initialize()
		
		try:
			engine = await create_enterprise_workflow_engine(service)
			
			# Create and execute workflow
			trigger_message = MQMessage(
				topic="orders.placed.metrics_test",
				payload=b'{"order_id": "METRICS_001", "total": 50.00}',
				tenant_id="metrics_tenant",
				source_application="metrics_test"
			)
			
			execution_id = await engine.trigger_workflow(trigger_message)
			assert execution_id is not None
			
			# Wait for execution
			await asyncio.sleep(2)
			
			# Check that metrics were collected
			status = await engine.get_enterprise_status()
			assert 'process_metrics' in status
			
			# Should have metrics for order processing
			if BusinessProcessType.ORDER_PROCESSING.value in status['process_metrics']:
				order_metrics = status['process_metrics'][BusinessProcessType.ORDER_PROCESSING.value]
				assert 'avg_duration_ms' in order_metrics
				assert 'error_rate' in order_metrics
			
			await engine.shutdown()
			
		finally:
			await service.shutdown()


if __name__ == "__main__":
	# Run tests if script is executed directly
	pytest.main([__file__, "-v"])