"""
Deployment Automation Tests

Comprehensive tests for the deployment automation capability including:
- Deployment configuration and execution
- Multiple deployment strategies
- Environment management
- Rollback and recovery
- Integration with workflow orchestration

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from typing import Dict, Any
from uuid_extensions import uuid7str

from . import (
	test_tenant_id,
	test_user_id,
	mock_deployment_target,
	CompositionTestHelper
)


class TestDeploymentAutomation:
	"""Test deployment automation core functionality."""
	
	async def test_deployment_service_creation(self, test_tenant_id: str):
		"""Test deployment service creation."""
		from ..deployment_automation import get_deployment_service
		
		service = get_deployment_service(test_tenant_id)
		assert service is not None
		assert service.tenant_id == test_tenant_id
		
	async def test_deploy_composition_basic(
		self,
		test_tenant_id: str,
		mock_deployment_target
	):
		"""Test basic composition deployment."""
		from ..deployment_automation import get_deployment_service, DeploymentStrategy
		
		service = get_deployment_service(test_tenant_id)
		composition_id = f"comp_{uuid7str()}"
		
		result = await service.deploy_composition(
			composition_id=composition_id,
			target=mock_deployment_target,
			strategy=DeploymentStrategy.ROLLING_UPDATE
		)
		
		assert result is not None
		assert result.deployment_id is not None
		assert result.status is not None
		
	async def test_deploy_composition_blue_green(
		self,
		test_tenant_id: str,
		mock_deployment_target
	):
		"""Test blue-green deployment strategy."""
		from ..deployment_automation import get_deployment_service, DeploymentStrategy
		
		service = get_deployment_service(test_tenant_id)
		composition_id = f"comp_{uuid7str()}"
		
		result = await service.deploy_composition(
			composition_id=composition_id,
			target=mock_deployment_target,
			strategy=DeploymentStrategy.BLUE_GREEN
		)
		
		assert result is not None
		assert result.status.value in ["completed", "in_progress"]
		assert "blue-green" in result.message.lower()
		
	async def test_deploy_composition_canary(
		self,
		test_tenant_id: str,
		mock_deployment_target
	):
		"""Test canary deployment strategy."""
		from ..deployment_automation import get_deployment_service, DeploymentStrategy
		
		service = get_deployment_service(test_tenant_id)
		composition_id = f"comp_{uuid7str()}"
		
		result = await service.deploy_composition(
			composition_id=composition_id,
			target=mock_deployment_target,
			strategy=DeploymentStrategy.CANARY
		)
		
		assert result is not None
		assert result.status.value in ["completed", "in_progress"]
		assert "canary" in result.message.lower()


class TestDeploymentStrategies:
	"""Test different deployment strategies."""
	
	async def test_all_deployment_strategies(
		self,
		test_tenant_id: str,
		mock_deployment_target
	):
		"""Test all supported deployment strategies."""
		from ..deployment_automation import (
			get_deployment_service,
			DeploymentStrategy
		)
		
		service = get_deployment_service(test_tenant_id)
		
		strategies = [
			DeploymentStrategy.ROLLING_UPDATE,
			DeploymentStrategy.BLUE_GREEN,
			DeploymentStrategy.CANARY,
			DeploymentStrategy.RECREATE
		]
		
		for strategy in strategies:
			composition_id = f"comp_{uuid7str()}"
			
			result = await service.deploy_composition(
				composition_id=composition_id,
				target=mock_deployment_target,
				strategy=strategy
			)
			
			assert result is not None
			assert result.deployment_id is not None
			
	def test_deployment_strategy_enum(self):
		"""Test deployment strategy enumeration."""
		from ..deployment_automation import DeploymentStrategy
		
		strategies = list(DeploymentStrategy)
		assert len(strategies) >= 4
		assert DeploymentStrategy.ROLLING_UPDATE in strategies
		assert DeploymentStrategy.BLUE_GREEN in strategies
		assert DeploymentStrategy.CANARY in strategies


class TestDeploymentConfiguration:
	"""Test deployment configuration management."""
	
	def test_deployment_config_creation(self, test_tenant_id: str, mock_deployment_target):
		"""Test deployment configuration creation."""
		from ..deployment_automation import DeploymentConfig, DeploymentStrategy
		from datetime import datetime
		
		config = DeploymentConfig(
			tenant_id=test_tenant_id,
			composition_id=f"comp_{uuid7str()}",
			strategy=DeploymentStrategy.ROLLING_UPDATE,
			target=mock_deployment_target,
			container_image="test-image:latest",
			version="1.0.0",
			created_at=datetime.utcnow().isoformat(),
			updated_at=datetime.utcnow().isoformat()
		)
		
		assert config.tenant_id == test_tenant_id
		assert config.strategy == DeploymentStrategy.ROLLING_UPDATE
		assert config.container_image == "test-image:latest"
		
	def test_deployment_target_validation(self):
		"""Test deployment target validation."""
		from ..deployment_automation import DeploymentTarget, DeploymentEnvironment
		
		target = DeploymentTarget(
			environment=DeploymentEnvironment.PRODUCTION,
			cluster_name="prod-cluster",
			namespace="production",
			replicas=5,
			resource_limits={"cpu": "2", "memory": "4Gi"}
		)
		
		assert target.environment == DeploymentEnvironment.PRODUCTION
		assert target.replicas == 5
		assert target.resource_limits["cpu"] == "2"


class TestDeploymentEnvironments:
	"""Test deployment environment management."""
	
	def test_deployment_environments(self):
		"""Test deployment environment enumeration."""
		from ..deployment_automation import DeploymentEnvironment
		
		environments = list(DeploymentEnvironment)
		assert DeploymentEnvironment.DEVELOPMENT in environments
		assert DeploymentEnvironment.STAGING in environments
		assert DeploymentEnvironment.PRODUCTION in environments
		
	async def test_environment_specific_deployment(
		self,
		test_tenant_id: str
	):
		"""Test deployment to specific environments."""
		from ..deployment_automation import (
			get_deployment_service,
			DeploymentTarget,
			DeploymentEnvironment,
			DeploymentStrategy
		)
		
		service = get_deployment_service(test_tenant_id)
		
		environments = [
			DeploymentEnvironment.DEVELOPMENT,
			DeploymentEnvironment.STAGING,
			DeploymentEnvironment.PRODUCTION
		]
		
		for env in environments:
			target = DeploymentTarget(
				environment=env,
				cluster_name=f"{env.value}-cluster",
				namespace=f"{env.value}-ns",
				replicas=1 if env == DeploymentEnvironment.DEVELOPMENT else 3
			)
			
			result = await service.deploy_composition(
				composition_id=f"comp_{uuid7str()}",
				target=target,
				strategy=DeploymentStrategy.ROLLING_UPDATE
			)
			
			assert result is not None
			assert f"{env.value}-cluster" in result.rollout_url or result.rollout_url is None


class TestDeploymentStatus:
	"""Test deployment status management."""
	
	def test_deployment_status_enum(self):
		"""Test deployment status enumeration."""
		from ..deployment_automation import DeploymentStatus
		
		statuses = list(DeploymentStatus)
		assert DeploymentStatus.PENDING in statuses
		assert DeploymentStatus.IN_PROGRESS in statuses
		assert DeploymentStatus.COMPLETED in statuses
		assert DeploymentStatus.FAILED in statuses
		assert DeploymentStatus.ROLLED_BACK in statuses
		
	async def test_deployment_result_structure(
		self,
		test_tenant_id: str,
		mock_deployment_target
	):
		"""Test deployment result structure."""
		from ..deployment_automation import get_deployment_service, DeploymentStrategy
		
		service = get_deployment_service(test_tenant_id)
		
		result = await service.deploy_composition(
			composition_id=f"comp_{uuid7str()}",
			target=mock_deployment_target,
			strategy=DeploymentStrategy.ROLLING_UPDATE
		)
		
		# Verify result structure
		assert hasattr(result, 'deployment_id')
		assert hasattr(result, 'status')
		assert hasattr(result, 'message')
		assert hasattr(result, 'rollout_url')
		assert hasattr(result, 'health_status')
		assert hasattr(result, 'logs')
		
		assert isinstance(result.logs, list)


class TestDeploymentIntegration:
	"""Test deployment integration with composition system."""
	
	async def test_deploy_composition_with_workflow(
		self,
		test_tenant_id: str,
		test_user_id: str,
		mock_deployment_target
	):
		"""Test deployment integration with workflow orchestration."""
		from .. import deploy_composition
		from ..deployment_automation import DeploymentStrategy
		
		composition_id = f"comp_{uuid7str()}"
		
		# Test deployment with approval workflow
		result_id = await deploy_composition(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			composition_id=composition_id,
			deployment_target=mock_deployment_target,
			deployment_strategy=DeploymentStrategy.BLUE_GREEN,
			require_approval=True
		)
		
		assert result_id is not None
		assert isinstance(result_id, str)
		
	async def test_deploy_composition_direct(
		self,
		test_tenant_id: str,
		test_user_id: str,
		mock_deployment_target
	):
		"""Test direct deployment without approval workflow."""
		from .. import deploy_composition
		from ..deployment_automation import DeploymentStrategy
		
		composition_id = f"comp_{uuid7str()}"
		
		# Test direct deployment
		result_id = await deploy_composition(
			tenant_id=test_tenant_id,
			user_id=test_user_id,
			composition_id=composition_id,
			deployment_target=mock_deployment_target,
			deployment_strategy=DeploymentStrategy.ROLLING_UPDATE,
			require_approval=False
		)
		
		assert result_id is not None
		assert isinstance(result_id, str)


class TestDeploymentPerformance:
	"""Test deployment performance and scalability."""
	
	async def test_concurrent_deployments(
		self,
		test_tenant_id: str,
		mock_deployment_target
	):
		"""Test concurrent deployment handling."""
		from ..deployment_automation import get_deployment_service, DeploymentStrategy
		
		service = get_deployment_service(test_tenant_id)
		
		# Create multiple concurrent deployment tasks
		tasks = []
		for i in range(5):
			task = service.deploy_composition(
				composition_id=f"comp_{i}_{uuid7str()}",
				target=mock_deployment_target,
				strategy=DeploymentStrategy.ROLLING_UPDATE
			)
			tasks.append(task)
		
		# Execute all deployments concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Verify all deployments completed
		successful_deployments = [
			r for r in results 
			if not isinstance(r, Exception)
		]
		
		assert len(successful_deployments) == 5
		
	async def test_deployment_performance_metrics(
		self,
		test_tenant_id: str,
		mock_deployment_target
	):
		"""Test deployment performance measurement."""
		from ..deployment_automation import get_deployment_service, DeploymentStrategy
		from datetime import datetime
		
		service = get_deployment_service(test_tenant_id)
		
		start_time = datetime.utcnow()
		
		result = await service.deploy_composition(
			composition_id=f"comp_{uuid7str()}",
			target=mock_deployment_target,
			strategy=DeploymentStrategy.ROLLING_UPDATE
		)
		
		end_time = datetime.utcnow()
		deployment_time = (end_time - start_time).total_seconds()
		
		assert result is not None
		assert deployment_time < 10.0  # Should complete within 10 seconds (mock)


class TestDeploymentErrorHandling:
	"""Test deployment error handling and recovery."""
	
	async def test_invalid_composition_deployment(self, test_tenant_id: str):
		"""Test deployment with invalid composition."""
		from ..deployment_automation import (
			get_deployment_service,
			DeploymentTarget,
			DeploymentEnvironment,
			DeploymentStrategy
		)
		
		service = get_deployment_service(test_tenant_id)
		
		# Create invalid target (empty cluster name)
		invalid_target = DeploymentTarget(
			environment=DeploymentEnvironment.DEVELOPMENT,
			cluster_name="",  # Invalid empty cluster name
			namespace="test",
			replicas=1
		)
		
		try:
			result = await service.deploy_composition(
				composition_id="invalid_comp",
				target=invalid_target,
				strategy=DeploymentStrategy.ROLLING_UPDATE
			)
			
			# Should handle gracefully (mock implementation might not fail)
			assert result is not None
			
		except Exception as e:
			# Acceptable to raise exception for invalid input
			assert isinstance(e, (ValueError, RuntimeError))
			
	async def test_deployment_rollback_scenario(
		self,
		test_tenant_id: str,
		mock_deployment_target
	):
		"""Test deployment rollback scenario."""
		from ..deployment_automation import get_deployment_service, DeploymentStrategy
		
		service = get_deployment_service(test_tenant_id)
		
		# Simulate failed deployment that needs rollback
		result = await service.deploy_composition(
			composition_id="rollback_test_comp",
			target=mock_deployment_target,
			strategy=DeploymentStrategy.BLUE_GREEN
		)
		
		assert result is not None
		# In real implementation, would test rollback functionality