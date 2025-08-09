"""
APG Central Configuration - Capability Management System

Central orchestrator for managing all APG capabilities across distributed deployments.
Provides unified configuration management, service discovery, and cross-capability coordination.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
from pathlib import Path
import yaml

# APG Platform Integration
from .service import CentralConfigurationEngine
from .models import ConfigurationCreate, ConfigurationUpdate, SecurityLevel


class CapabilityStatus(Enum):
	"""Status of APG capabilities."""
	UNKNOWN = "unknown"
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	UNHEALTHY = "unhealthy"
	OFFLINE = "offline"
	STARTING = "starting"
	STOPPING = "stopping"


class DeploymentEnvironment(Enum):
	"""Deployment environments."""
	DEVELOPMENT = "development"
	STAGING = "staging"
	PRODUCTION = "production"
	TESTING = "testing"


class CloudProvider(Enum):
	"""Supported cloud providers."""
	AWS = "aws"
	AZURE = "azure"
	GCP = "gcp"
	KUBERNETES = "kubernetes"
	ON_PREMISES = "on_premises"
	DOCKER = "docker"


@dataclass
class CapabilityEndpoint:
	"""API endpoint for a capability."""
	url: str
	type: str  # rest, graphql, grpc, websocket
	authentication: Dict[str, Any]
	health_check_path: str
	timeout_seconds: int = 30


@dataclass
class CapabilityDeployment:
	"""Deployment information for a capability."""
	deployment_id: str
	environment: DeploymentEnvironment
	cloud_provider: CloudProvider
	region: str
	endpoint: CapabilityEndpoint
	resource_requirements: Dict[str, Any]
	scaling_config: Dict[str, Any]
	monitoring_config: Dict[str, Any]
	deployed_at: datetime
	last_updated: datetime


@dataclass
class CapabilitySpec:
	"""Specification for an APG capability."""
	capability_id: str
	name: str
	version: str
	description: str
	category: str
	tags: List[str]
	
	# Configuration schema
	config_schema: Dict[str, Any]
	default_config: Dict[str, Any]
	required_capabilities: List[str]
	provided_interfaces: List[str]
	
	# Deployment info
	deployments: List[CapabilityDeployment]
	
	# Metadata
	maintainer: str
	documentation_url: str
	repository_url: str
	created_at: datetime
	updated_at: datetime


@dataclass
class CapabilityHealth:
	"""Health status of a capability."""
	capability_id: str
	deployment_id: str
	status: CapabilityStatus
	response_time_ms: float
	last_check: datetime
	error_message: Optional[str]
	metrics: Dict[str, float]
	dependencies_status: Dict[str, CapabilityStatus]


@dataclass
class CrossCapabilityConfiguration:
	"""Configuration that spans multiple capabilities."""
	config_id: str
	name: str
	description: str
	affected_capabilities: List[str]
	configuration_data: Dict[str, Any]
	deployment_strategy: str  # simultaneous, sequential, rolling
	rollback_strategy: str
	created_by: str
	created_at: datetime


class APGCapabilityManager:
	"""Central manager for all APG capabilities."""
	
	def __init__(
		self,
		config_engine: CentralConfigurationEngine,
		api_mesh_url: str = "http://api-service-mesh:8000"
	):
		"""Initialize capability manager."""
		self.config_engine = config_engine
		self.api_mesh_url = api_mesh_url
		
		# Capability registry
		self.capabilities: Dict[str, CapabilitySpec] = {}
		self.capability_health: Dict[str, CapabilityHealth] = {}
		
		# Cross-capability configurations
		self.cross_capability_configs: Dict[str, CrossCapabilityConfiguration] = {}
		
		# Service discovery
		self.service_registry: Dict[str, Dict[str, Any]] = {}
		
		# HTTP client for API mesh communication
		self.http_client = httpx.AsyncClient(timeout=30.0)
		
		# Health monitoring
		self.health_check_interval = 60  # seconds
		self.health_check_task: Optional[asyncio.Task] = None
		
		# Initialize built-in capabilities
		asyncio.create_task(self._initialize_builtin_capabilities())
		
		# Start health monitoring
		asyncio.create_task(self._start_health_monitoring())
	
	# ==================== Capability Registration ====================
	
	async def register_capability(
		self,
		capability_spec: CapabilitySpec
	) -> bool:
		"""Register a new APG capability."""
		try:
			# Validate capability spec
			await self._validate_capability_spec(capability_spec)
			
			# Register in local registry
			self.capabilities[capability_spec.capability_id] = capability_spec
			
			# Create default configuration workspace
			await self._create_capability_workspace(capability_spec)
			
			# Register with API mesh
			await self._register_with_api_mesh(capability_spec)
			
			# Store capability metadata in configuration
			await self._store_capability_metadata(capability_spec)
			
			print(f"✅ Registered capability: {capability_spec.name} ({capability_spec.capability_id})")
			return True
			
		except Exception as e:
			print(f"❌ Failed to register capability {capability_spec.capability_id}: {e}")
			return False
	
	async def _validate_capability_spec(self, spec: CapabilitySpec):
		"""Validate capability specification."""
		if not spec.capability_id:
			raise ValueError("Capability ID is required")
		
		if not spec.name:
			raise ValueError("Capability name is required")
		
		if not spec.config_schema:
			raise ValueError("Configuration schema is required")
		
		# Validate deployment endpoints
		for deployment in spec.deployments:
			if not deployment.endpoint.url:
				raise ValueError(f"Endpoint URL required for deployment {deployment.deployment_id}")
	
	async def _create_capability_workspace(self, spec: CapabilitySpec):
		"""Create configuration workspace for capability."""
		workspace_name = f"capability_{spec.capability_id}"
		
		# Create workspace configuration
		workspace_config = ConfigurationCreate(
			name=f"{spec.name} Workspace",
			key_path=f"/apg/capabilities/{spec.capability_id}/workspace",
			value={
				"capability_id": spec.capability_id,
				"name": spec.name,
				"version": spec.version,
				"workspace_type": "capability",
				"created_at": datetime.now(timezone.utc).isoformat()
			},
			security_level=SecurityLevel.INTERNAL,
			tags=["capability", "workspace", spec.category]
		)
		
		# Store in configuration engine
		await self.config_engine.create_configuration(
			workspace_id="apg_capabilities",
			config_data=workspace_config
		)
	
	async def _register_with_api_mesh(self, spec: CapabilitySpec):
		"""Register capability with API service mesh."""
		try:
			# Register each deployment endpoint
			for deployment in spec.deployments:
				registration_data = {
					"service_id": f"{spec.capability_id}_{deployment.deployment_id}",
					"service_name": spec.name,
					"capability_id": spec.capability_id,
					"endpoint_url": deployment.endpoint.url,
					"endpoint_type": deployment.endpoint.type,
					"health_check_path": deployment.endpoint.health_check_path,
					"environment": deployment.environment.value,
					"cloud_provider": deployment.cloud_provider.value,
					"region": deployment.region,
					"metadata": {
						"version": spec.version,
						"category": spec.category,
						"tags": spec.tags
					}
				}
				
				# Register with API mesh
				response = await self.http_client.post(
					f"{self.api_mesh_url}/api/v1/services/register",
					json=registration_data
				)
				
				if response.status_code not in [200, 201]:
					print(f"⚠️ Failed to register with API mesh: {response.text}")
				else:
					print(f"✅ Registered {spec.name} with API mesh")
					
		except Exception as e:
			print(f"⚠️ API mesh registration failed: {e}")
	
	async def _store_capability_metadata(self, spec: CapabilitySpec):
		"""Store capability metadata in configuration."""
		metadata_config = ConfigurationCreate(
			name=f"{spec.name} Metadata",
			key_path=f"/apg/capabilities/{spec.capability_id}/metadata",
			value={
				"spec": asdict(spec),
				"registered_at": datetime.now(timezone.utc).isoformat(),
				"managed_by": "central_configuration"
			},
			security_level=SecurityLevel.INTERNAL,
			tags=["capability", "metadata", spec.category]
		)
		
		await self.config_engine.create_configuration(
			workspace_id="apg_capabilities",
			config_data=metadata_config
		)
	
	# ==================== Capability Configuration Management ====================
	
	async def update_capability_configuration(
		self,
		capability_id: str,
		deployment_id: str,
		configuration_updates: Dict[str, Any],
		apply_immediately: bool = True
	) -> bool:
		"""Update configuration for a specific capability deployment."""
		try:
			if capability_id not in self.capabilities:
				raise ValueError(f"Capability {capability_id} not registered")
			
			spec = self.capabilities[capability_id]
			deployment = None
			
			# Find the deployment
			for dep in spec.deployments:
				if dep.deployment_id == deployment_id:
					deployment = dep
					break
			
			if not deployment:
				raise ValueError(f"Deployment {deployment_id} not found for capability {capability_id}")
			
			# Validate configuration against schema
			await self._validate_configuration(spec.config_schema, configuration_updates)
			
			# Store configuration update
			config_key = f"/apg/capabilities/{capability_id}/deployments/{deployment_id}/config"
			config_update = ConfigurationUpdate(
				value=configuration_updates,
				metadata={
					"updated_by": "capability_manager",
					"deployment_id": deployment_id,
					"apply_immediately": apply_immediately
				}
			)
			
			# Update in configuration engine
			await self.config_engine.update_configuration(
				configuration_id=config_key,
				updates=config_update,
				change_reason="Capability configuration update"
			)
			
			# Apply configuration if requested
			if apply_immediately:
				await self._apply_configuration_to_deployment(capability_id, deployment_id, configuration_updates)
			
			print(f"✅ Updated configuration for {capability_id}/{deployment_id}")
			return True
			
		except Exception as e:
			print(f"❌ Failed to update capability configuration: {e}")
			return False
	
	async def _validate_configuration(self, schema: Dict[str, Any], config: Dict[str, Any]):
		"""Validate configuration against capability schema."""
		# Simplified JSON schema validation
		# In production, use jsonschema library
		required_fields = schema.get("required", [])
		
		for field in required_fields:
			if field not in config:
				raise ValueError(f"Required field '{field}' missing from configuration")
	
	async def _apply_configuration_to_deployment(
		self,
		capability_id: str,
		deployment_id: str,
		configuration: Dict[str, Any]
	):
		"""Apply configuration to a specific deployment."""
		spec = self.capabilities[capability_id]
		deployment = None
		
		# Find deployment
		for dep in spec.deployments:
			if dep.deployment_id == deployment_id:
				deployment = dep
				break
		
		if not deployment:
			return
		
		try:
			# Send configuration update to capability
			update_url = f"{deployment.endpoint.url}/api/v1/config/update"
			update_payload = {
				"configuration": configuration,
				"source": "central_configuration",
				"timestamp": datetime.now(timezone.utc).isoformat()
			}
			
			response = await self.http_client.post(
				update_url,
				json=update_payload,
				timeout=deployment.endpoint.timeout_seconds
			)
			
			if response.status_code == 200:
				print(f"✅ Configuration applied to {capability_id}/{deployment_id}")
			else:
				print(f"⚠️ Failed to apply configuration: {response.text}")
				
		except Exception as e:
			print(f"⚠️ Failed to apply configuration to deployment: {e}")
	
	# ==================== Cross-Capability Configuration ====================
	
	async def create_cross_capability_configuration(
		self,
		name: str,
		description: str,
		capability_configs: Dict[str, Dict[str, Any]],
		deployment_strategy: str = "sequential"
	) -> str:
		"""Create configuration that affects multiple capabilities."""
		config_id = f"cross_config_{uuid.uuid4().hex[:8]}"
		
		cross_config = CrossCapabilityConfiguration(
			config_id=config_id,
			name=name,
			description=description,
			affected_capabilities=list(capability_configs.keys()),
			configuration_data=capability_configs,
			deployment_strategy=deployment_strategy,
			rollback_strategy="automatic",
			created_by="system",
			created_at=datetime.now(timezone.utc)
		)
		
		# Store cross-capability configuration
		self.cross_capability_configs[config_id] = cross_config
		
		# Store in configuration engine
		config_data = ConfigurationCreate(
			name=f"Cross-Capability: {name}",
			key_path=f"/apg/cross_capability_configs/{config_id}",
			value=asdict(cross_config),
			security_level=SecurityLevel.INTERNAL,
			tags=["cross-capability", "orchestration"]
		)
		
		await self.config_engine.create_configuration(
			workspace_id="apg_orchestration",
			config_data=config_data
		)
		
		print(f"✅ Created cross-capability configuration: {name} ({config_id})")
		return config_id
	
	async def apply_cross_capability_configuration(
		self,
		config_id: str,
		dry_run: bool = False
	) -> Dict[str, Any]:
		"""Apply cross-capability configuration to all affected capabilities."""
		if config_id not in self.cross_capability_configs:
			raise ValueError(f"Cross-capability configuration {config_id} not found")
		
		cross_config = self.cross_capability_configs[config_id]
		results = {
			"config_id": config_id,
			"status": "success",
			"applied_to": [],
			"failed": [],
			"dry_run": dry_run
		}
		
		if dry_run:
			print(f"🔍 DRY RUN: Would apply cross-capability configuration {config_id}")
			results["message"] = "Dry run completed - no changes made"
			return results
		
		try:
			if cross_config.deployment_strategy == "simultaneous":
				# Apply to all capabilities simultaneously
				tasks = []
				for capability_id, config_data in cross_config.configuration_data.items():
					task = self._apply_config_to_capability(capability_id, config_data)
					tasks.append(task)
				
				results_list = await asyncio.gather(*tasks, return_exceptions=True)
				
				for i, result in enumerate(results_list):
					capability_id = list(cross_config.configuration_data.keys())[i]
					if isinstance(result, Exception):
						results["failed"].append({"capability_id": capability_id, "error": str(result)})
					else:
						results["applied_to"].append(capability_id)
			
			elif cross_config.deployment_strategy == "sequential":
				# Apply to capabilities one by one
				for capability_id, config_data in cross_config.configuration_data.items():
					try:
						await self._apply_config_to_capability(capability_id, config_data)
						results["applied_to"].append(capability_id)
						print(f"✅ Applied configuration to {capability_id}")
					except Exception as e:
						results["failed"].append({"capability_id": capability_id, "error": str(e)})
						print(f"❌ Failed to apply configuration to {capability_id}: {e}")
			
			# Update status
			if results["failed"]:
				results["status"] = "partial_success" if results["applied_to"] else "failed"
			
			print(f"✅ Cross-capability configuration applied: {len(results['applied_to'])} success, {len(results['failed'])} failed")
			
		except Exception as e:
			results["status"] = "failed"
			results["error"] = str(e)
			print(f"❌ Failed to apply cross-capability configuration: {e}")
		
		return results
	
	async def _apply_config_to_capability(
		self,
		capability_id: str,
		config_data: Dict[str, Any]
	):
		"""Apply configuration to all deployments of a capability."""
		if capability_id not in self.capabilities:
			raise ValueError(f"Capability {capability_id} not registered")
		
		spec = self.capabilities[capability_id]
		
		# Apply to all deployments
		for deployment in spec.deployments:
			await self._apply_configuration_to_deployment(
				capability_id,
				deployment.deployment_id,
				config_data
			)
	
	# ==================== Health Monitoring ====================
	
	async def _start_health_monitoring(self):
		"""Start health monitoring for all capabilities."""
		print("🏥 Starting capability health monitoring...")
		
		while True:
			try:
				await self._check_all_capabilities_health()
				await asyncio.sleep(self.health_check_interval)
			except Exception as e:
				print(f"❌ Health monitoring error: {e}")
				await asyncio.sleep(60)  # Wait longer on error
	
	async def _check_all_capabilities_health(self):
		"""Check health of all registered capabilities."""
		health_check_tasks = []
		
		for capability_id, spec in self.capabilities.items():
			for deployment in spec.deployments:
				task = self._check_deployment_health(capability_id, deployment)
				health_check_tasks.append(task)
		
		if health_check_tasks:
			await asyncio.gather(*health_check_tasks, return_exceptions=True)
	
	async def _check_deployment_health(
		self,
		capability_id: str,
		deployment: CapabilityDeployment
	):
		"""Check health of a specific deployment."""
		start_time = datetime.now()
		
		try:
			# Health check request
			health_url = f"{deployment.endpoint.url}{deployment.endpoint.health_check_path}"
			response = await self.http_client.get(
				health_url,
				timeout=deployment.endpoint.timeout_seconds
			)
			
			response_time = (datetime.now() - start_time).total_seconds() * 1000
			
			# Determine status
			if response.status_code == 200:
				status = CapabilityStatus.HEALTHY
				error_message = None
			elif response.status_code in [503, 504]:
				status = CapabilityStatus.DEGRADED
				error_message = f"HTTP {response.status_code}"
			else:
				status = CapabilityStatus.UNHEALTHY
				error_message = f"HTTP {response.status_code}: {response.text[:100]}"
			
			# Parse metrics from response
			metrics = {}
			try:
				health_data = response.json()
				if isinstance(health_data, dict):
					metrics = health_data.get("metrics", {})
			except:
				pass
			
		except Exception as e:
			status = CapabilityStatus.OFFLINE
			error_message = str(e)
			response_time = (datetime.now() - start_time).total_seconds() * 1000
			metrics = {}
		
		# Update health status
		health_key = f"{capability_id}_{deployment.deployment_id}"
		self.capability_health[health_key] = CapabilityHealth(
			capability_id=capability_id,
			deployment_id=deployment.deployment_id,
			status=status,
			response_time_ms=response_time,
			last_check=datetime.now(timezone.utc),
			error_message=error_message,
			metrics=metrics,
			dependencies_status={}
		)
	
	# ==================== Service Discovery ====================
	
	async def discover_capability_endpoints(
		self,
		capability_id: str,
		environment: Optional[DeploymentEnvironment] = None
	) -> List[CapabilityEndpoint]:
		"""Discover available endpoints for a capability."""
		endpoints = []
		
		if capability_id in self.capabilities:
			spec = self.capabilities[capability_id]
			
			for deployment in spec.deployments:
				if environment is None or deployment.environment == environment:
					# Check if deployment is healthy
					health_key = f"{capability_id}_{deployment.deployment_id}"
					health = self.capability_health.get(health_key)
					
					if health and health.status in [CapabilityStatus.HEALTHY, CapabilityStatus.DEGRADED]:
						endpoints.append(deployment.endpoint)
		
		return endpoints
	
	async def get_capability_by_interface(
		self,
		interface_name: str,
		environment: Optional[DeploymentEnvironment] = None
	) -> List[str]:
		"""Find capabilities that provide a specific interface."""
		matching_capabilities = []
		
		for capability_id, spec in self.capabilities.items():
			if interface_name in spec.provided_interfaces:
				# Check if any deployment is healthy
				has_healthy_deployment = False
				
				for deployment in spec.deployments:
					if environment is None or deployment.environment == environment:
						health_key = f"{capability_id}_{deployment.deployment_id}"
						health = self.capability_health.get(health_key)
						
						if health and health.status in [CapabilityStatus.HEALTHY, CapabilityStatus.DEGRADED]:
							has_healthy_deployment = True
							break
				
				if has_healthy_deployment:
					matching_capabilities.append(capability_id)
		
		return matching_capabilities
	
	# ==================== Built-in Capabilities ====================
	
	async def _initialize_builtin_capabilities(self):
		"""Initialize built-in APG capabilities."""
		await asyncio.sleep(1)  # Wait for initialization
		
		# Central Configuration (self-registration)
		await self._register_central_configuration()
		
		# API Service Mesh
		await self._register_api_service_mesh()
		
		# Real-time Collaboration
		await self._register_realtime_collaboration()
		
		print("✅ Built-in capabilities initialized")
	
	async def _register_central_configuration(self):
		"""Register central configuration capability."""
		spec = CapabilitySpec(
			capability_id="central_configuration",
			name="Central Configuration",
			version="1.0.0",
			description="Revolutionary AI-powered configuration management platform",
			category="platform",
			tags=["configuration", "ai", "automation"],
			config_schema={
				"type": "object",
				"properties": {
					"ai_enabled": {"type": "boolean", "default": True},
					"automation_enabled": {"type": "boolean", "default": True},
					"security_level": {"type": "string", "enum": ["standard", "high"], "default": "high"}
				},
				"required": ["ai_enabled"]
			},
			default_config={
				"ai_enabled": True,
				"automation_enabled": True,
				"security_level": "high"
			},
			required_capabilities=[],
			provided_interfaces=["configuration_management", "ai_optimization", "automation"],
			deployments=[
				CapabilityDeployment(
					deployment_id="primary",
					environment=DeploymentEnvironment.PRODUCTION,
					cloud_provider=CloudProvider.KUBERNETES,
					region="us-east-1",
					endpoint=CapabilityEndpoint(
						url="http://localhost:8000",
						type="rest",
						authentication={"type": "api_key"},
						health_check_path="/health"
					),
					resource_requirements={"cpu": "2", "memory": "4Gi"},
					scaling_config={"min_replicas": 2, "max_replicas": 10},
					monitoring_config={"metrics_enabled": True, "logging_level": "info"},
					deployed_at=datetime.now(timezone.utc),
					last_updated=datetime.now(timezone.utc)
				)
			],
			maintainer="platform-team",
			documentation_url="https://docs.apg.platform/central-configuration",
			repository_url="https://github.com/apg-platform/central-configuration",
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc)
		)
		
		self.capabilities[spec.capability_id] = spec
	
	async def _register_api_service_mesh(self):
		"""Register API service mesh capability."""
		spec = CapabilitySpec(
			capability_id="api_service_mesh",
			name="API Service Mesh",
			version="1.0.0",
			description="Intelligent API service mesh for capability communication",
			category="platform",
			tags=["api", "mesh", "networking"],
			config_schema={
				"type": "object",
				"properties": {
					"load_balancing": {"type": "string", "enum": ["round_robin", "least_connections"], "default": "round_robin"},
					"circuit_breaker_enabled": {"type": "boolean", "default": True},
					"rate_limiting": {"type": "object"}
				}
			},
			default_config={
				"load_balancing": "round_robin",
				"circuit_breaker_enabled": True,
				"rate_limiting": {"requests_per_second": 1000}
			},
			required_capabilities=[],
			provided_interfaces=["service_mesh", "load_balancing", "service_discovery"],
			deployments=[
				CapabilityDeployment(
					deployment_id="primary",
					environment=DeploymentEnvironment.PRODUCTION,
					cloud_provider=CloudProvider.KUBERNETES,
					region="us-east-1",
					endpoint=CapabilityEndpoint(
						url="http://api-service-mesh:8000",
						type="rest",
						authentication={"type": "api_key"},
						health_check_path="/health"
					),
					resource_requirements={"cpu": "1", "memory": "2Gi"},
					scaling_config={"min_replicas": 3, "max_replicas": 15},
					monitoring_config={"metrics_enabled": True, "tracing_enabled": True},
					deployed_at=datetime.now(timezone.utc),
					last_updated=datetime.now(timezone.utc)
				)
			],
			maintainer="platform-team",
			documentation_url="https://docs.apg.platform/api-service-mesh",
			repository_url="https://github.com/apg-platform/api-service-mesh",
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc)
		)
		
		self.capabilities[spec.capability_id] = spec
	
	async def _register_realtime_collaboration(self):
		"""Register real-time collaboration capability."""
		spec = CapabilitySpec(
			capability_id="realtime_collaboration",
			name="Real-time Collaboration",
			version="1.0.0",
			description="Real-time collaboration platform with multi-protocol support",
			category="collaboration",
			tags=["realtime", "collaboration", "websocket"],
			config_schema={
				"type": "object",
				"properties": {
					"max_concurrent_users": {"type": "integer", "default": 1000},
					"protocols_enabled": {"type": "array", "items": {"type": "string"}},
					"message_retention_hours": {"type": "integer", "default": 24}
				}
			},
			default_config={
				"max_concurrent_users": 1000,
				"protocols_enabled": ["websocket", "webrtc", "socketio"],
				"message_retention_hours": 24
			},
			required_capabilities=["central_configuration"],
			provided_interfaces=["real_time_messaging", "collaboration", "presence"],
			deployments=[
				CapabilityDeployment(
					deployment_id="primary",
					environment=DeploymentEnvironment.PRODUCTION,
					cloud_provider=CloudProvider.KUBERNETES,
					region="us-east-1",
					endpoint=CapabilityEndpoint(
						url="http://realtime-collaboration:8000",
						type="websocket",
						authentication={"type": "jwt"},
						health_check_path="/health"
					),
					resource_requirements={"cpu": "2", "memory": "4Gi"},
					scaling_config={"min_replicas": 2, "max_replicas": 20},
					monitoring_config={"metrics_enabled": True, "websocket_monitoring": True},
					deployed_at=datetime.now(timezone.utc),
					last_updated=datetime.now(timezone.utc)
				)
			],
			maintainer="collaboration-team",
			documentation_url="https://docs.apg.platform/realtime-collaboration",
			repository_url="https://github.com/apg-platform/realtime-collaboration",
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc)
		)
		
		self.capabilities[spec.capability_id] = spec
	
	# ==================== Public Interface ====================
	
	async def get_all_capabilities(self) -> Dict[str, CapabilitySpec]:
		"""Get all registered capabilities."""
		return self.capabilities.copy()
	
	async def get_capability_health_status(self) -> Dict[str, CapabilityHealth]:
		"""Get health status of all capabilities."""
		return self.capability_health.copy()
	
	async def get_capability_dashboard_data(self) -> Dict[str, Any]:
		"""Get dashboard data for capability overview."""
		total_capabilities = len(self.capabilities)
		healthy_deployments = len([h for h in self.capability_health.values() if h.status == CapabilityStatus.HEALTHY])
		total_deployments = len(self.capability_health)
		
		# Capability categories
		categories = {}
		for spec in self.capabilities.values():
			categories[spec.category] = categories.get(spec.category, 0) + 1
		
		return {
			"total_capabilities": total_capabilities,
			"total_deployments": total_deployments,
			"healthy_deployments": healthy_deployments,
			"health_percentage": (healthy_deployments / max(total_deployments, 1)) * 100,
			"categories": categories,
			"cross_capability_configs": len(self.cross_capability_configs),
			"last_updated": datetime.now(timezone.utc).isoformat()
		}
	
	async def close(self):
		"""Clean up capability manager resources."""
		if self.health_check_task:
			self.health_check_task.cancel()
		
		await self.http_client.aclose()
		print("🔄 Capability manager closed")


# ==================== Factory Functions ====================

async def create_capability_manager(
	config_engine: CentralConfigurationEngine,
	api_mesh_url: str = "http://api-service-mesh:8000"
) -> APGCapabilityManager:
	"""Create and initialize capability manager."""
	manager = APGCapabilityManager(config_engine, api_mesh_url)
	await asyncio.sleep(2)  # Allow initialization
	print("🎯 APG Capability Manager initialized")
	return manager