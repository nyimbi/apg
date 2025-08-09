"""
APG Capability Registry - APG Platform Integration

Complete integration with APG composition engine, discovery infrastructure,
and ecosystem-wide capability management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict

from .models import CRCapability, CRComposition, CRRegistry
from .service import CRService

# =============================================================================
# APG Integration Models
# =============================================================================

class APGCapabilityMetadata(BaseModel):
	"""APG-specific capability metadata."""
	model_config = ConfigDict(extra='forbid')
	
	apg_version: str = Field("2.0", description="APG platform version")
	capability_id: str = Field(..., description="Capability ID")
	capability_code: str = Field(..., description="Capability code")
	capability_name: str = Field(..., description="Display name")
	
	# APG Integration Points
	composition_engine: Dict[str, Any] = Field(default_factory=dict, description="Composition engine integration")
	discovery_metadata: Dict[str, Any] = Field(default_factory=dict, description="Discovery metadata")
	registry_hooks: List[str] = Field(default_factory=list, description="Registry hooks")
	
	# APG Ecosystem Integration
	provides_interfaces: List[str] = Field(default_factory=list, description="Provided interfaces")
	implements_protocols: List[str] = Field(default_factory=list, description="Implemented protocols")
	requires_capabilities: List[str] = Field(default_factory=list, description="Required capabilities")
	
	# APG Runtime Configuration
	runtime_config: Dict[str, Any] = Field(default_factory=dict, description="Runtime configuration")
	deployment_strategy: str = Field("standard", description="Deployment strategy")
	scaling_policies: Dict[str, Any] = Field(default_factory=dict, description="Scaling policies")

class APGCompositionConfig(BaseModel):
	"""APG composition engine configuration."""
	model_config = ConfigDict(extra='forbid')
	
	composition_id: str = Field(..., description="Composition ID")
	name: str = Field(..., description="Composition name")
	
	# APG Engine Settings
	engine_version: str = Field("2.0", description="APG engine version")
	orchestration_mode: str = Field("intelligent", description="Orchestration mode")
	dependency_resolution: str = Field("automatic", description="Dependency resolution")
	
	# Capability Integration
	capability_bindings: List[Dict[str, Any]] = Field(default_factory=list, description="Capability bindings")
	service_mappings: Dict[str, str] = Field(default_factory=dict, description="Service mappings")
	interface_contracts: List[Dict[str, Any]] = Field(default_factory=list, description="Interface contracts")
	
	# Runtime Behavior
	execution_order: List[str] = Field(default_factory=list, description="Execution order")
	parallel_execution: List[List[str]] = Field(default_factory=list, description="Parallel execution groups")
	failure_handling: Dict[str, Any] = Field(default_factory=dict, description="Failure handling")
	
	# Performance Configuration
	resource_allocation: Dict[str, Any] = Field(default_factory=dict, description="Resource allocation")
	monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Monitoring configuration")

class APGDiscoveryRegistration(BaseModel):
	"""APG discovery service registration."""
	model_config = ConfigDict(extra='forbid')
	
	registration_id: str = Field(default_factory=uuid7str, description="Registration ID")
	capability_id: str = Field(..., description="Capability ID")
	
	# Discovery Metadata
	discovery_tags: List[str] = Field(default_factory=list, description="Discovery tags")
	search_keywords: List[str] = Field(default_factory=list, description="Search keywords")
	category_hierarchy: List[str] = Field(default_factory=list, description="Category hierarchy")
	
	# APG Integration
	apg_tenant_id: str = Field(..., description="APG tenant ID")
	apg_namespace: str = Field("default", description="APG namespace")
	apg_environment: str = Field("production", description="APG environment")
	
	# Service Discovery
	service_endpoints: List[Dict[str, Any]] = Field(default_factory=list, description="Service endpoints")
	health_check_config: Dict[str, Any] = Field(default_factory=dict, description="Health check configuration")
	load_balancing: Dict[str, Any] = Field(default_factory=dict, description="Load balancing configuration")

# =============================================================================
# APG Integration Service
# =============================================================================

class APGIntegrationService:
	"""Service for APG platform integration."""
	
	def __init__(self, tenant_id: str = "default"):
		self.tenant_id = tenant_id
		self.registry_service: Optional[CRService] = None
		self.apg_config: Dict[str, Any] = {}
		self.registered_capabilities: Set[str] = set()
		self.active_compositions: Dict[str, APGCompositionConfig] = {}
		
		self._load_apg_config()
	
	def _load_apg_config(self):
		"""Load APG platform configuration."""
		# In production, would load from APG configuration service
		self.apg_config = {
			"platform_version": "2.0",
			"tenant_id": self.tenant_id,
			"composition_engine": {
				"enabled": True,
				"auto_discovery": True,
				"intelligent_routing": True,
				"dependency_injection": True
			},
			"discovery_service": {
				"enabled": True,
				"auto_registration": True,
				"health_monitoring": True,
				"load_balancing": True
			},
			"registry_integration": {
				"sync_interval": 300,  # 5 minutes
				"auto_sync": True,
				"conflict_resolution": "latest_wins"
			}
		}
	
	async def set_registry_service(self, service: CRService):
		"""Set the capability registry service."""
		self.registry_service = service
	
	# =========================================================================
	# APG Composition Engine Integration
	# =========================================================================
	
	async def register_with_composition_engine(
		self,
		capability_id: str
	) -> APGCapabilityMetadata:
		"""Register capability with APG composition engine."""
		if not self.registry_service:
			raise ValueError("Registry service not configured")
		
		# Get capability details
		capability = await self.registry_service.get_capability(capability_id)
		if not capability:
			raise ValueError(f"Capability not found: {capability_id}")
		
		# Create APG metadata
		apg_metadata = APGCapabilityMetadata(
			capability_id=capability_id,
			capability_code=capability.capability_code,
			capability_name=capability.capability_name,
			
			# Composition engine integration
			composition_engine={
				"auto_wiring": True,
				"dependency_injection": True,
				"service_discovery": True,
				"lifecycle_management": True
			},
			
			# Discovery metadata
			discovery_metadata={
				"tags": capability.composition_keywords or [],
				"category": capability.category,
				"subcategory": capability.subcategory,
				"business_value": capability.business_value,
				"target_users": capability.target_users or []
			},
			
			# Registry hooks
			registry_hooks=[
				"on_capability_updated",
				"on_capability_deprecated",
				"on_version_released"
			],
			
			# Interface definitions
			provides_interfaces=capability.api_endpoints or [],
			implements_protocols=self._extract_protocols(capability),
			requires_capabilities=self._extract_dependencies(capability),
			
			# Runtime configuration
			runtime_config={
				"multi_tenant": capability.multi_tenant,
				"audit_enabled": capability.audit_enabled,
				"security_integration": capability.security_integration,
				"performance_optimized": capability.performance_optimized,
				"ai_enhanced": capability.ai_enhanced
			},
			
			deployment_strategy=self._determine_deployment_strategy(capability),
			scaling_policies=self._generate_scaling_policies(capability)
		)
		
		# Register with APG composition engine
		await self._register_apg_capability(apg_metadata)
		
		self.registered_capabilities.add(capability_id)
		
		return apg_metadata
	
	async def create_apg_composition(
		self,
		composition_id: str,
		capability_ids: List[str]
	) -> APGCompositionConfig:
		"""Create APG composition configuration."""
		if not self.registry_service:
			raise ValueError("Registry service not configured")
		
		# Get composition details
		composition = await self.registry_service.get_composition(composition_id)
		if not composition:
			raise ValueError(f"Composition not found: {composition_id}")
		
		# Get capability details
		capabilities = []
		for cap_id in capability_ids:
			cap = await self.registry_service.get_capability(cap_id)
			if cap:
				capabilities.append(cap)
		
		# Analyze dependencies and create execution plan
		execution_plan = await self._analyze_execution_dependencies(capabilities)
		
		# Create APG composition config
		apg_config = APGCompositionConfig(
			composition_id=composition_id,
			name=composition.name,
			
			# Capability bindings
			capability_bindings=[
				{
					"capability_id": cap.capability_id,
					"capability_code": cap.capability_code,
					"binding_mode": "automatic",
					"interface_mapping": self._map_interfaces(cap),
					"dependency_injection": True
				}
				for cap in capabilities
			],
			
			# Service mappings
			service_mappings=self._create_service_mappings(capabilities),
			
			# Interface contracts
			interface_contracts=self._generate_interface_contracts(capabilities),
			
			# Execution configuration
			execution_order=execution_plan["sequential_order"],
			parallel_execution=execution_plan["parallel_groups"],
			
			# Failure handling
			failure_handling={
				"strategy": "graceful_degradation",
				"retry_policy": {
					"max_retries": 3,
					"backoff_strategy": "exponential",
					"timeout_ms": 30000
				},
				"circuit_breaker": {
					"enabled": True,
					"failure_threshold": 5,
					"recovery_timeout_ms": 60000
				}
			},
			
			# Resource allocation
			resource_allocation=self._calculate_resource_requirements(capabilities),
			
			# Monitoring
			monitoring_config={
				"metrics_enabled": True,
				"tracing_enabled": True,
				"logging_level": "INFO",
				"performance_monitoring": True,
				"health_checks": True
			}
		)
		
		# Register with APG composition engine
		await self._register_apg_composition(apg_config)
		
		self.active_compositions[composition_id] = apg_config
		
		return apg_config
	
	# =========================================================================
	# APG Discovery Service Integration
	# =========================================================================
	
	async def register_with_discovery_service(
		self,
		capability_id: str
	) -> APGDiscoveryRegistration:
		"""Register capability with APG discovery service."""
		if not self.registry_service:
			raise ValueError("Registry service not configured")
		
		capability = await self.registry_service.get_capability(capability_id)
		if not capability:
			raise ValueError(f"Capability not found: {capability_id}")
		
		# Create discovery registration
		discovery_registration = APGDiscoveryRegistration(
			capability_id=capability_id,
			
			# Discovery tags and keywords
			discovery_tags=self._generate_discovery_tags(capability),
			search_keywords=capability.composition_keywords or [],
			category_hierarchy=[
				capability.category,
				capability.subcategory or "general"
			],
			
			# APG integration
			apg_tenant_id=self.tenant_id,
			apg_namespace=self._determine_namespace(capability),
			apg_environment="production",
			
			# Service endpoints
			service_endpoints=self._extract_service_endpoints(capability),
			
			# Health check configuration
			health_check_config={
				"enabled": True,
				"endpoint": "/health",
				"interval_ms": 30000,
				"timeout_ms": 5000,
				"healthy_threshold": 2,
				"unhealthy_threshold": 3
			},
			
			# Load balancing
			load_balancing={
				"algorithm": "round_robin",
				"sticky_sessions": False,
				"health_check_enabled": True,
				"failover_enabled": True
			}
		)
		
		# Register with APG discovery service
		await self._register_apg_discovery(discovery_registration)
		
		return discovery_registration
	
	# =========================================================================
	# APG Ecosystem Synchronization
	# =========================================================================
	
	async def sync_with_apg_ecosystem(self) -> Dict[str, Any]:
		"""Synchronize registry with APG ecosystem."""
		if not self.registry_service:
			raise ValueError("Registry service not configured")
		
		sync_results = {
			"capabilities_synced": 0,
			"compositions_synced": 0,
			"discovery_updates": 0,
			"composition_updates": 0,
			"errors": []
		}
		
		try:
			# Sync capabilities with composition engine
			capabilities = await self.registry_service.list_capabilities()
			for capability in capabilities:
				try:
					if capability.capability_id not in self.registered_capabilities:
						await self.register_with_composition_engine(capability.capability_id)
						await self.register_with_discovery_service(capability.capability_id)
						sync_results["capabilities_synced"] += 1
				except Exception as e:
					sync_results["errors"].append(f"Capability sync failed {capability.capability_id}: {str(e)}")
			
			# Sync compositions
			compositions = await self.registry_service.list_compositions()
			for composition in compositions:
				try:
					if composition.composition_id not in self.active_compositions:
						await self.create_apg_composition(
							composition.composition_id,
							composition.capability_ids or []
						)
						sync_results["compositions_synced"] += 1
				except Exception as e:
					sync_results["errors"].append(f"Composition sync failed {composition.composition_id}: {str(e)}")
			
			# Update discovery metadata
			await self._sync_discovery_metadata()
			sync_results["discovery_updates"] = len(self.registered_capabilities)
			
			# Update composition configurations
			await self._sync_composition_configs()
			sync_results["composition_updates"] = len(self.active_compositions)
			
		except Exception as e:
			sync_results["errors"].append(f"General sync error: {str(e)}")
		
		return sync_results
	
	# =========================================================================
	# APG Integration Helpers
	# =========================================================================
	
	def _extract_protocols(self, capability: CRCapability) -> List[str]:
		"""Extract implemented protocols from capability."""
		protocols = []
		
		# Standard APG protocols
		if capability.multi_tenant:
			protocols.append("apg.multi_tenant")
		if capability.audit_enabled:
			protocols.append("apg.audit")
		if capability.security_integration:
			protocols.append("apg.security")
		if capability.ai_enhanced:
			protocols.append("apg.ai_enhanced")
		
		# Protocol detection from metadata
		if capability.metadata:
			protocols.extend(capability.metadata.get("protocols", []))
		
		return protocols
	
	def _extract_dependencies(self, capability: CRCapability) -> List[str]:
		"""Extract capability dependencies."""
		dependencies = []
		
		# Dependencies from metadata
		if capability.metadata:
			dependencies.extend(capability.metadata.get("dependencies", []))
		
		# Inferred dependencies
		if capability.security_integration:
			dependencies.append("auth_rbac")
		if capability.audit_enabled:
			dependencies.append("audit_compliance")
		
		return dependencies
	
	def _determine_deployment_strategy(self, capability: CRCapability) -> str:
		"""Determine deployment strategy for capability."""
		if capability.performance_optimized:
			return "high_performance"
		elif capability.ai_enhanced:
			return "ai_optimized"
		elif capability.multi_tenant:
			return "multi_tenant"
		else:
			return "standard"
	
	def _generate_scaling_policies(self, capability: CRCapability) -> Dict[str, Any]:
		"""Generate scaling policies for capability."""
		base_policy = {
			"auto_scaling": True,
			"min_instances": 1,
			"max_instances": 10,
			"cpu_threshold": 70,
			"memory_threshold": 80,
			"scale_up_cooldown": 300,
			"scale_down_cooldown": 600
		}
		
		if capability.performance_optimized:
			base_policy.update({
				"min_instances": 2,
				"max_instances": 20,
				"cpu_threshold": 60,
				"memory_threshold": 70
			})
		
		return base_policy
	
	async def _analyze_execution_dependencies(
		self,
		capabilities: List[CRCapability]
	) -> Dict[str, Any]:
		"""Analyze execution dependencies for composition."""
		# Build dependency graph
		dependency_graph = {}
		for cap in capabilities:
			dependencies = self._extract_dependencies(cap)
			dependency_graph[cap.capability_id] = [
				dep_id for dep_id in dependencies
				if any(c.capability_code == dep_id for c in capabilities)
			]
		
		# Topological sort for execution order
		sequential_order = self._topological_sort(dependency_graph)
		
		# Identify parallel execution groups
		parallel_groups = self._identify_parallel_groups(dependency_graph, sequential_order)
		
		return {
			"sequential_order": sequential_order,
			"parallel_groups": parallel_groups,
			"dependency_graph": dependency_graph
		}
	
	def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
		"""Perform topological sort on dependency graph."""
		# Simplified topological sort
		visited = set()
		order = []
		
		def visit(node):
			if node in visited:
				return
			visited.add(node)
			for dependency in graph.get(node, []):
				if dependency in graph:
					visit(dependency)
			order.append(node)
		
		for node in graph:
			visit(node)
		
		return order
	
	def _identify_parallel_groups(
		self,
		graph: Dict[str, List[str]],
		order: List[str]
	) -> List[List[str]]:
		"""Identify capabilities that can execute in parallel."""
		# Simplified parallel group identification
		groups = []
		processed = set()
		
		for node in order:
			if node in processed:
				continue
			
			# Find nodes with no dependencies on remaining nodes
			group = [node]
			for other_node in order:
				if (other_node not in processed and 
					other_node != node and
					not any(dep in [node] + group for dep in graph.get(other_node, []))):
					group.append(other_node)
			
			if len(group) > 1:
				groups.append(group)
			
			processed.update(group)
		
		return groups
	
	def _map_interfaces(self, capability: CRCapability) -> Dict[str, Any]:
		"""Map capability interfaces for APG integration."""
		return {
			"rest_endpoints": capability.api_endpoints or [],
			"data_models": capability.data_models or [],
			"service_contracts": capability.provides_services or [],
			"event_publishers": [],
			"event_subscribers": []
		}
	
	def _create_service_mappings(self, capabilities: List[CRCapability]) -> Dict[str, str]:
		"""Create service mappings for composition."""
		mappings = {}
		for cap in capabilities:
			service_name = cap.capability_code.lower()
			mappings[service_name] = f"apg.{self.tenant_id}.{service_name}"
		return mappings
	
	def _generate_interface_contracts(self, capabilities: List[CRCapability]) -> List[Dict[str, Any]]:
		"""Generate interface contracts for composition."""
		contracts = []
		for cap in capabilities:
			contracts.append({
				"capability_id": cap.capability_id,
				"interface_type": "rest_api",
				"contract_version": "1.0",
				"schema_definition": {
					"endpoints": cap.api_endpoints or [],
					"data_models": cap.data_models or []
				}
			})
		return contracts
	
	def _calculate_resource_requirements(self, capabilities: List[CRCapability]) -> Dict[str, Any]:
		"""Calculate resource requirements for composition."""
		base_cpu = 0.5
		base_memory = 512
		
		total_cpu = len(capabilities) * base_cpu
		total_memory = len(capabilities) * base_memory
		
		# Adjust for performance-optimized capabilities
		for cap in capabilities:
			if cap.performance_optimized:
				total_cpu += 1.0
				total_memory += 1024
			if cap.ai_enhanced:
				total_cpu += 2.0
				total_memory += 2048
		
		return {
			"cpu_cores": total_cpu,
			"memory_mb": total_memory,
			"storage_gb": 10,
			"network_bandwidth_mbps": 100
		}
	
	def _generate_discovery_tags(self, capability: CRCapability) -> List[str]:
		"""Generate discovery tags for capability."""
		tags = [
			capability.category,
			capability.status,
			f"quality_{int(capability.quality_score * 100)}"
		]
		
		if capability.subcategory:
			tags.append(capability.subcategory)
		
		if capability.multi_tenant:
			tags.append("multi_tenant")
		if capability.ai_enhanced:
			tags.append("ai_enhanced")
		if capability.performance_optimized:
			tags.append("performance_optimized")
		
		return tags
	
	def _determine_namespace(self, capability: CRCapability) -> str:
		"""Determine APG namespace for capability."""
		category_namespaces = {
			"foundation_infrastructure": "core",
			"business_operations": "business",
			"analytics_intelligence": "analytics",
			"manufacturing_operations": "manufacturing",
			"industry_verticals": "vertical"
		}
		return category_namespaces.get(capability.category, "default")
	
	def _extract_service_endpoints(self, capability: CRCapability) -> List[Dict[str, Any]]:
		"""Extract service endpoints for discovery."""
		endpoints = []
		for endpoint in capability.api_endpoints or []:
			endpoints.append({
				"path": endpoint,
				"protocol": "http",
				"port": 8080,
				"health_check": True
			})
		return endpoints
	
	async def _register_apg_capability(self, metadata: APGCapabilityMetadata):
		"""Register capability with APG composition engine."""
		# In production, would call APG composition engine API
		print(f"Registering capability with APG: {metadata.capability_code}")
	
	async def _register_apg_composition(self, config: APGCompositionConfig):
		"""Register composition with APG engine."""
		# In production, would call APG composition engine API
		print(f"Registering composition with APG: {config.name}")
	
	async def _register_apg_discovery(self, registration: APGDiscoveryRegistration):
		"""Register with APG discovery service."""
		# In production, would call APG discovery service API
		print(f"Registering with APG discovery: {registration.capability_id}")
	
	async def _sync_discovery_metadata(self):
		"""Sync discovery metadata with APG."""
		# In production, would sync with APG discovery service
		pass
	
	async def _sync_composition_configs(self):
		"""Sync composition configurations with APG."""
		# In production, would sync with APG composition engine
		pass

# =============================================================================
# APG Integration Factory
# =============================================================================

_apg_integration_instances: Dict[str, APGIntegrationService] = {}

async def get_apg_integration_service(tenant_id: str = "default") -> APGIntegrationService:
	"""Get or create APG integration service instance."""
	if tenant_id not in _apg_integration_instances:
		service = APGIntegrationService(tenant_id)
		_apg_integration_instances[tenant_id] = service
	
	return _apg_integration_instances[tenant_id]

# Export integration service
__all__ = [
	"APGIntegrationService",
	"APGCapabilityMetadata",
	"APGCompositionConfig", 
	"APGDiscoveryRegistration",
	"get_apg_integration_service"
]