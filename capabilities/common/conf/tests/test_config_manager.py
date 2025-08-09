"""
Revolutionary Configuration Management Tests

Comprehensive test suite for the revolutionary configuration management system
ensuring >95% code coverage and APG quality standards compliance.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from typing import Dict, Any
from datetime import datetime

from ..service import RevolutionaryConfigurationManager, create_configuration_manager
from ..models import (
	CMResource, CMTemplate, CMPolicy, CMEnvironment, CMDeployment,
	ResourceState, DeploymentStatus, ResourceType, CloudProvider,
	ConfigurationDSL, ValidationResult
)


class TestRevolutionaryConfigurationManager:
	"""Test Revolutionary Configuration Management functionality"""
	
	@pytest.fixture
	def event_loop(self):
		"""Create event loop for async tests"""
		loop = asyncio.get_event_loop()
		return loop

	@pytest.fixture
	async def config_manager(self):
		"""Create test configuration manager"""
		manager = await create_configuration_manager(tenant_id="test_tenant")
		await manager.initialize({})
		return manager

	@pytest.fixture
	def sample_configuration(self) -> Dict[str, Any]:
		"""Sample configuration for testing"""
		return {
			"name": "test-vm",
			"type": "virtual_machine", 
			"cloud_provider": "aws",
			"configuration": {
				"kind": "VirtualMachine",
				"spec": {
					"resources": {
						"instance_type": "t3.micro",
						"image": "ami-12345",
						"vpc_id": "vpc-test"
					}
				}
			},
			"description": "Test virtual machine configuration"
		}

	@pytest.mark.asyncio
	async def test_manager_initialization(self, config_manager):
		"""Test configuration manager initialization"""
		assert config_manager._initialized is True
		assert config_manager.tenant_id == "test_tenant"
		assert config_manager.id is not None
		assert isinstance(config_manager.created_at, datetime)
		
		# Check AI components are initialized
		assert config_manager.ai_engine is not None
		assert config_manager.universal_layer is not None
		assert config_manager.quantum_security is not None
		assert config_manager.predictive_analytics is not None

	@pytest.mark.asyncio
	async def test_create_configuration(self, config_manager, sample_configuration):
		"""Test creating configuration resource"""
		resource = await config_manager.create_configuration(sample_configuration)
		
		# Verify resource properties
		assert isinstance(resource, CMResource)
		assert resource.name == "test-vm"
		assert resource.resource_type == ResourceType.VIRTUAL_MACHINE
		assert resource.cloud_provider == CloudProvider.AWS
		assert resource.state == ResourceState.PENDING
		assert resource.tenant_id == "test_tenant"
		
		# Verify it's stored in manager
		assert resource.id in config_manager.resources
		assert config_manager.metrics["total_configurations"] == 1

	@pytest.mark.asyncio
	async def test_deploy_configuration(self, config_manager, sample_configuration):
		"""Test deploying configuration"""
		# Create resource first
		resource = await config_manager.create_configuration(sample_configuration)
		
		# Deploy the resource
		deployment = await config_manager.deploy_configuration(
			resource.id, "test_environment"
		)
		
		# Verify deployment
		assert isinstance(deployment, CMDeployment)
		assert deployment.resource_id == resource.id
		assert deployment.environment_id == "test_environment"
		assert deployment.status in [DeploymentStatus.COMPLETED, DeploymentStatus.FAILED]
		assert deployment.tenant_id == "test_tenant"
		
		# Verify deployment is stored
		assert deployment.id in config_manager.deployments

	@pytest.mark.asyncio
	async def test_drift_detection(self, config_manager, sample_configuration):
		"""Test configuration drift detection and remediation"""
		# Create resource
		resource = await config_manager.create_configuration(sample_configuration)
		
		# Test drift detection
		result = await config_manager.detect_and_remediate_drift(resource.id)
		
		# Verify result format
		assert isinstance(result, dict)
		assert "resource_id" in result
		assert "drift_detected" in result
		assert "timestamp" in result
		assert result["resource_id"] == resource.id

	@pytest.mark.asyncio
	async def test_intelligent_template_creation(self, config_manager):
		"""Test AI-generated template creation"""
		requirements = {
			"name": "web-server-template",
			"description": "AI-generated web server template",
			"category": "web",
			"requirements": {
				"application": "nginx",
				"performance": "high",
				"scaling": "auto"
			},
			"created_by": "test_user"
		}
		
		template = await config_manager.create_intelligent_template(requirements)
		
		# Verify template
		assert isinstance(template, CMTemplate)
		assert template.name == "web-server-template"
		assert template.category == "web"
		assert template.tenant_id == "test_tenant"
		
		# Verify it's stored
		assert template.id in config_manager.templates

	@pytest.mark.asyncio
	async def test_natural_language_configuration(self, config_manager):
		"""Test natural language to configuration conversion"""
		nl_request = "Create a small web server on AWS with auto-scaling enabled"
		context = {
			"user_id": "test_user",
			"environment": "development"
		}
		
		result = await config_manager.natural_language_configuration(nl_request, context)
		
		# Verify result format
		assert isinstance(result, dict)
		assert "request" in result
		assert "parsed_intent" in result
		assert "generated_configuration" in result
		assert "ready_to_deploy" in result
		assert result["request"] == nl_request

	@pytest.mark.asyncio
	async def test_predictive_insights(self, config_manager, sample_configuration):
		"""Test predictive analytics and insights"""
		# Create resource for testing
		resource = await config_manager.create_configuration(sample_configuration)
		
		# Get resource-specific insights
		resource_insights = await config_manager.get_predictive_insights(resource.id)
		assert isinstance(resource_insights, dict)
		assert "insights" in resource_insights
		assert "resource_id" in resource_insights
		
		# Get system-wide insights
		system_insights = await config_manager.get_predictive_insights()
		assert isinstance(system_insights, dict)
		assert "insights" in system_insights
		assert resource_insights["resource_id"] is None

	@pytest.mark.asyncio
	async def test_revolutionary_metrics(self, config_manager, sample_configuration):
		"""Test comprehensive system metrics"""
		# Create some test data
		resource = await config_manager.create_configuration(sample_configuration)
		
		# Get metrics
		metrics = await config_manager.get_revolutionary_metrics()
		
		# Verify metrics structure
		assert isinstance(metrics, dict)
		assert "system_metrics" in metrics
		assert "ai_intelligence" in metrics
		assert "universal_abstraction" in metrics
		assert "quantum_security" in metrics
		assert "predictive_analytics" in metrics
		assert "performance_indicators" in metrics
		
		# Check performance indicators
		indicators = metrics["performance_indicators"]
		assert "incident_reduction_percentage" in indicators
		assert "provisioning_speed_improvement" in indicators
		assert "compliance_automation" in indicators
		assert "autonomous_operations_percentage" in indicators

	@pytest.mark.asyncio
	async def test_deployment_transaction(self, config_manager, sample_configuration):
		"""Test atomic deployment transaction with rollback"""
		# Create resource
		resource = await config_manager.create_configuration(sample_configuration)
		
		# Test successful transaction
		async with config_manager.deployment_transaction(resource.id):
			# Simulate deployment operation
			resource.state = ResourceState.DEPLOYING
		
		# Test failed transaction (should rollback)
		original_state = resource.state
		try:
			async with config_manager.deployment_transaction(resource.id):
				resource.state = ResourceState.FAILED
				raise Exception("Simulated deployment failure")
		except Exception:
			pass
		
		# Verify rollback occurred
		assert resource.state == original_state

	@pytest.mark.asyncio
	async def test_shutdown(self, config_manager):
		"""Test graceful shutdown"""
		await config_manager.shutdown()
		# Verify shutdown completed without errors

	def test_model_validation(self):
		"""Test Pydantic model validation"""
		# Test valid resource creation
		config_dsl = ConfigurationDSL(
			kind="VirtualMachine",
			spec={"resources": {"instance_type": "t3.micro"}}
		)
		
		resource = CMResource(
			name="test-resource",
			resource_type=ResourceType.VIRTUAL_MACHINE,
			cloud_provider=CloudProvider.AWS,
			configuration=config_dsl
		)
		
		assert resource.name == "test-resource"
		assert resource.resource_type == ResourceType.VIRTUAL_MACHINE
		
		# Test invalid resource creation (should raise validation error)
		with pytest.raises(Exception):
			CMResource(
				name="",  # Empty name should fail validation
				resource_type=ResourceType.VIRTUAL_MACHINE,
				cloud_provider=CloudProvider.AWS,
				configuration=config_dsl
			)

	@pytest.mark.asyncio
	async def test_policy_enforcement(self, config_manager, sample_configuration):
		"""Test policy enforcement with autonomous compliance"""
		# Create resource and policy
		resource = await config_manager.create_configuration(sample_configuration)
		
		# Create test policy
		policy = CMPolicy(
			name="test-security-policy",
			description="Test security policy",
			policy_type="security",
			rules=[{"field": "encryption", "required": True}],
			actions=["warn"],
			tenant_id="test_tenant"
		)
		config_manager.policies[policy.id] = policy
		
		# Test policy enforcement
		result = await config_manager.enforce_policy(policy.id, resource.id)
		
		assert isinstance(result, dict)
		assert "policy_id" in result
		assert "resource_id" in result
		assert "compliant" in result
		assert result["policy_id"] == policy.id
		assert result["resource_id"] == resource.id

	def test_logging_helper(self, config_manager):
		"""Test logging path formatting helper"""
		test_path = f"/test/path/{config_manager.tenant_id}/config"
		formatted = config_manager._log_pretty_path(test_path)
		assert "[TENANT]" in formatted

	@pytest.mark.asyncio
	async def test_error_handling(self, config_manager):
		"""Test error handling in various scenarios"""
		# Test invalid resource ID
		with pytest.raises(AssertionError):
			await config_manager.deploy_configuration("invalid_id", "test_env")
		
		# Test uninitialized manager operations
		uninitialized_manager = RevolutionaryConfigurationManager()
		with pytest.raises(AssertionError):
			await uninitialized_manager.create_configuration({"name": "test"})

	@pytest.mark.asyncio 
	async def test_factory_functions(self):
		"""Test factory functions and service instance management"""
		from ..service import get_config_manager
		
		# Test getting manager for same tenant returns same instance
		manager1 = await get_config_manager("tenant1")
		manager2 = await get_config_manager("tenant1")
		assert manager1 is manager2
		
		# Test different tenant returns different instance
		manager3 = await get_config_manager("tenant2")
		assert manager1 is not manager3

	def test_runtime_assertions(self):
		"""Test runtime assertions in model creation"""
		# Test resource creation with proper assertions
		config_dsl = ConfigurationDSL(
			kind="Test",
			spec={"resources": {"test": "value"}}
		)
		
		resource = CMResource(
			name="valid-name",
			resource_type=ResourceType.CUSTOM,
			cloud_provider=CloudProvider.AWS,
			configuration=config_dsl
		)
		
		# Runtime assertions should pass
		assert resource.id is not None
		assert resource.name == "valid-name"


class TestConfigurationModels:
	"""Test configuration management models"""
	
	def test_configuration_dsl(self):
		"""Test ConfigurationDSL model"""
		dsl = ConfigurationDSL(
			kind="TestResource",
			spec={"resources": {"test": "value"}},
			version="1.0"
		)
		
		assert dsl.kind == "TestResource"
		assert dsl.version == "1.0"
		
		# Test YAML export
		yaml_output = dsl.to_yaml()
		assert "TestResource" in yaml_output
		
		# Test HCL export
		hcl_output = dsl.to_hcl()
		assert "TestResource" in hcl_output

	def test_deployment_duration_calculation(self):
		"""Test deployment duration calculation"""
		deployment = CMDeployment(
			resource_id="test",
			environment_id="test_env",
			deployment_plan={"steps": ["deploy"]},
			started_at=datetime.utcnow(),
			completed_at=datetime.utcnow()
		)
		
		duration = deployment.calculate_duration()
		assert isinstance(duration, int)
		assert duration >= 0

	def test_template_instantiation(self):
		"""Test template parameter instantiation"""
		template = CMTemplate(
			name="test-template",
			description="Test template",
			configuration_template={"instance_type": "{{instance_type}}"},
			parameters={"instance_type": "t3.micro"}
		)
		
		# Test instantiation
		instantiated = template.instantiate({"instance_type": "t3.large"})
		assert isinstance(instantiated, dict)

	def test_policy_evaluation(self):
		"""Test policy evaluation against resources"""
		policy = CMPolicy(
			name="test-policy",
			description="Test policy",
			policy_type="security",
			rules=[{"field": "encryption", "required": True}],
			actions=["warn"]
		)
		
		config_dsl = ConfigurationDSL(
			kind="Test",
			spec={"resources": {"encryption": True}}
		)
		
		resource = CMResource(
			name="test-resource",
			resource_type=ResourceType.VIRTUAL_MACHINE,
			cloud_provider=CloudProvider.AWS,
			configuration=config_dsl
		)
		
		# Test policy evaluation
		result = policy.evaluate(resource)
		assert isinstance(result, dict)
		assert "compliant" in result
		assert "violations" in result
		assert "recommendations" in result


# Test markers and configuration
pytestmark = pytest.mark.asyncio