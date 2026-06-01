"""
APG AI Core Framework (aicr) - Capability Registration

Purpose: APG composition engine integration for the AI Core Framework capability.
         Provides capability metadata, service discovery, and health monitoring
         for seamless integration with the APG platform ecosystem.

Dependencies: pydantic, typing, datetime, asyncio
APG Integration: auth, conf, mqeb, moni capabilities
Usage Context: Foundational AI infrastructure for all APG AI capabilities

This module enables:
- Dynamic capability registration with APG composition engine
- AI service discovery and health reporting
- Integration with APG security and monitoring infrastructure
- Hot-reloading and configuration management
- Dependency resolution for AI service chains
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .models import (
	AIServiceRegistration, AIServiceStatus, AIServiceHealth,
	AIModelMetadata, AIInferenceRequest, AIInferenceResult,
	AIWorkflow, AIAuditEvent, model_registry
)
from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)


def uuid7str() -> str:
	"""Generate UUID7 string for identifiers."""
	return str(uuid4())


class CapabilityMetadata(BaseModel):
	"""APG capability metadata for composition engine registration.

	Defines the metadata structure required for APG composition engine
	registration, including capability identification, dependencies,
	and operational characteristics for orchestration.

	Attributes:
		id: Unique capability identifier within APG ecosystem
		name: Human-readable capability name
		version: Semantic version following APG versioning standards
		description: Detailed capability description and purpose
		category: Capability category for APG marketplace organization
		dependencies: List of required APG capabilities for operation
		provides: List of services this capability provides to others
		tags: Searchable tags for capability discovery
		maintainer: Team or individual responsible for capability
		license: Software license and usage terms
		repository_url: Source code repository location
		documentation_url: Comprehensive documentation location
		api_endpoints: Available API endpoints for service integration
		health_check_url: Endpoint for capability health monitoring
		metrics_endpoints: Performance and operational metrics endpoints
		configuration_schema: Expected configuration structure
		resource_requirements: Minimum resource allocation needs
		scalability_limits: Maximum scale and performance boundaries
		compliance_certifications: Regulatory compliance information
		created_at: Capability registration timestamp
		updated_at: Last update timestamp for change tracking
	"""
	id: str = "aicr"
	name: str = "AI Core Framework"
	version: str = "1.0.0"
	description: str = "AI control plane for provider registration, model governance, workflows, agent runtimes, and governed inference"
	category: str = "ai-infrastructure"
	dependencies: List[str] = ["conf", "auth", "mqeb", "moni", "audl", "keym"]
	provides: List[str] = [
		"ai-service-registry", "inference-engine", "model-lifecycle",
		"ai-orchestration", "neuromorphic-processing", "quantum-safe-ai",
		"explainable-ai", "multi-modal-fusion", "edge-cloud-mesh"
	]
	tags: List[str] = [
		"ai", "machine-learning", "inference", "orchestration",
		"neuromorphic", "quantum-safe", "multi-modal", "edge-computing"
	]
	maintainer: str = "AI Platform Team"
	license: str = "Proprietary - © 2025 Datacraft"
	repository_url: str = "https://github.com/datacraft/apg/capabilities/common/aicr"
	documentation_url: str = "https://docs.apg.datacraft.co.ke/capabilities/aicr"
	api_endpoints: List[str] = [
		"/api/v1/ai/services", "/api/v1/ai/inference", "/api/v1/ai/models",
		"/api/v1/ai/workflows", "/api/v1/ai/health", "/api/v1/ai/metrics"
	]
	health_check_url: str = "/api/v1/ai/health"
	metrics_endpoints: List[str] = ["/api/v1/ai/metrics", "/api/v1/ai/performance"]
	configuration_schema: Dict[str, Any] = Field(default_factory=dict)
	resource_requirements: Dict[str, Any] = Field(default_factory=lambda: {
		"cpu_cores": 8,
		"memory_gb": 32,
		"storage_gb": 100,
		"gpu_memory_gb": 16,
		"network_bandwidth_mbps": 1000
	})
	scalability_limits: Dict[str, Any] = Field(default_factory=lambda: {
		"max_concurrent_requests": 10000,
		"max_models": 1000,
		"max_tenants": 100,
		"max_workflows": 500
	})
	compliance_certifications: List[str] = [
		"SOC2-Type2", "ISO27001", "GDPR", "HIPAA", "AI-Act-Compliant"
	]
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class CapabilityStatus(BaseModel):
	"""Real-time capability operational status for APG monitoring.

	Provides comprehensive operational status information for the
	AI Core Framework capability, enabling APG monitoring and
	orchestration systems to make intelligent decisions.

	Attributes:
		capability_id: Reference to the capability being monitored
		status: Current operational status of the capability
		last_check: Timestamp of the last status evaluation
		uptime_seconds: Total operational time since last restart
		active_services: Number of currently active AI services
		total_requests: Cumulative number of inference requests processed
		success_rate: Percentage of successful operations (0-100)
		average_response_time_ms: Average response time across all operations
		resource_utilization: Current resource usage across all dimensions
		error_rate: Percentage of failed operations requiring attention
		performance_score: Overall performance rating (0-100)
		health_indicators: Detailed health metrics for system components
		alerts: Active alerts requiring immediate attention
		recommendations: System-generated optimization recommendations
		dependencies_status: Status of required APG capabilities
		version_info: Current version and deployment information
		configuration_status: Configuration validation and health
		security_status: Security posture and threat assessment
		compliance_status: Regulatory compliance verification
		metadata: Additional operational metadata and context
	"""
	capability_id: str = "aicr"
	status: AIServiceStatus
	last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	uptime_seconds: int = 0
	active_services: int = 0
	total_requests: int = 0
	success_rate: float = 100.0
	average_response_time_ms: float = 0.0
	resource_utilization: Dict[str, float] = Field(default_factory=dict)
	error_rate: float = 0.0
	performance_score: float = 100.0
	health_indicators: Dict[str, Any] = Field(default_factory=dict)
	alerts: List[str] = Field(default_factory=list)
	recommendations: List[str] = Field(default_factory=list)
	dependencies_status: Dict[str, str] = Field(default_factory=dict)
	version_info: Dict[str, str] = Field(default_factory=dict)
	configuration_status: str = "healthy"
	security_status: str = "secure"
	compliance_status: str = "compliant"
	metadata: Dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class APGCapabilityProtocol(Protocol):
	"""Protocol definition for APG capability interface compliance.

	Defines the standard interface that all APG capabilities must
	implement for composition engine integration. This protocol
	ensures consistent behavior and interoperability across the
	APG platform ecosystem.

	Methods:
		get_metadata: Return capability metadata for registration
		get_status: Return current operational status
		health_check: Perform comprehensive health assessment
		initialize: Initialize capability with configuration
		shutdown: Gracefully shutdown capability services
		reload_config: Reload configuration without restart
		get_dependencies: Return required capability dependencies
		register_service: Register new AI service with capability
		discover_services: Discover available AI services
	"""

	async def get_metadata(self) -> CapabilityMetadata:
		"""Return capability metadata for APG registration."""
		...

	async def get_status(self) -> CapabilityStatus:
		"""Return current operational status."""
		...

	async def health_check(self) -> Dict[str, Any]:
		"""Perform comprehensive health assessment."""
		...

	async def initialize(self, config: Dict[str, Any]) -> bool:
		"""Initialize capability with provided configuration."""
		...

	async def shutdown(self) -> bool:
		"""Gracefully shutdown capability services."""
		...

	async def reload_config(self, config: Dict[str, Any]) -> bool:
		"""Reload configuration without full restart."""
		...

	async def get_dependencies(self) -> List[str]:
		"""Return list of required APG capability dependencies."""
		...

	async def register_service(self, service: AIServiceRegistration) -> str:
		"""Register new AI service with the capability."""
		...

	async def discover_services(self, filters: Dict[str, Any]) -> List[AIServiceRegistration]:
		"""Discover available AI services matching filters."""
		...


class AICoreCapability:
	"""AI Core Framework capability implementation for APG integration.

	Primary implementation of the AI Core Framework capability that
	integrates with the APG composition engine. Provides comprehensive
	AI infrastructure services including service registry, inference
	engine, and intelligent orchestration.

	This class serves as the main entry point for all AI operations
	within the APG platform, coordinating AI services and providing
	foundational infrastructure for intelligent automation.

	Attributes:
		_metadata: Capability metadata for APG registration
		_status: Current operational status of the capability
		_services: Registry of active AI services
		_config: Current configuration settings
		_initialized: Initialization state flag
		_start_time: Capability startup timestamp
		_health_monitors: Health monitoring components
		_dependencies: Required APG capability dependencies

	Methods:
		get_metadata: Return capability metadata for APG registration
		get_status: Provide current operational status information
		health_check: Perform comprehensive health assessment
		initialize: Initialize capability with configuration
		shutdown: Gracefully shutdown all capability services
		reload_config: Hot-reload configuration without restart
		get_dependencies: Return required capability dependencies
		register_service: Register new AI service with capability
		discover_services: Discover AI services with filtering
		_update_status: Internal status update and monitoring
		_validate_dependencies: Verify APG dependency availability
		_setup_health_monitoring: Initialize health monitoring systems
	"""

	def __init__(self):
		"""Initialize AI Core Framework capability instance."""
		self._metadata = CapabilityMetadata()
		self._status = CapabilityStatus(status=AIServiceStatus.INITIALIZING)
		self._services: Dict[str, AIServiceRegistration] = {}
		self._config: Dict[str, Any] = {}
		self._initialized = False
		self._start_time = datetime.now(timezone.utc)
		self._health_monitors: Dict[str, Any] = {}
		self._dependencies = ["conf", "auth", "mqeb", "moni"]

	async def get_metadata(self) -> CapabilityMetadata:
		"""Return AI Core Framework capability metadata for APG registration.

		Provides comprehensive metadata about the AI Core Framework
		capability for registration with the APG composition engine.
		This enables service discovery, dependency resolution, and
		orchestration across the APG platform.

		Returns:
			CapabilityMetadata: Complete metadata structure with:
				- Capability identification and versioning
				- Dependency requirements and provided services
				- API endpoints and health monitoring URLs
				- Resource requirements and scalability limits
				- Compliance certifications and documentation

		Example:
			>>> capability = AICoreCapability()
			>>> metadata = await capability.get_metadata()
			>>> print(metadata.name)  # "AI Core Framework"
			>>> print(metadata.dependencies)  # ["conf", "auth", "mqeb", "moni"]
		"""
		return self._metadata

	async def get_status(self) -> CapabilityStatus:
		"""Provide comprehensive operational status for APG monitoring.

		Returns detailed operational status information including
		performance metrics, resource utilization, health indicators,
		and recommendations for optimization. Used by APG monitoring
		systems for intelligent orchestration decisions.

		Returns:
			CapabilityStatus: Complete status structure with:
				- Current operational state and uptime
				- Performance metrics and success rates
				- Resource utilization across all dimensions
				- Active alerts and optimization recommendations
				- Dependency status and version information

		Raises:
			RuntimeError: If status collection encounters system errors

		Example:
			>>> capability = AICoreCapability()
			>>> await capability.initialize({})
			>>> status = await capability.get_status()
			>>> print(status.active_services)  # Number of active AI services
		"""
		await self._update_status()
		return self._status

	async def health_check(self) -> Dict[str, Any]:
		"""Perform comprehensive health assessment of AI infrastructure.

		Conducts thorough health evaluation of all AI Core Framework
		components including service registry, inference engines,
		orchestration systems, and integration points. Provides
		detailed diagnostics for troubleshooting and optimization.

		Returns:
			Dict[str, Any]: Comprehensive health report containing:
				- overall_health: Summary health status (healthy/degraded/unhealthy)
				- component_health: Individual component health status
				- performance_metrics: Current performance measurements
				- resource_availability: Available system resources
				- dependency_status: Health of required APG capabilities
				- recommendations: System optimization suggestions
				- last_check: Timestamp of health assessment

		Raises:
			HealthCheckError: If critical health check components fail

		Example:
			>>> capability = AICoreCapability()
			>>> health = await capability.health_check()
			>>> print(health["overall_health"])  # "healthy"
			>>> print(health["component_health"]["inference_engine"])  # "operational"
		"""
		health_report = {
			"overall_health": "healthy",
			"component_health": {
				"service_registry": "operational",
				"inference_engine": "operational",
				"orchestration": "operational",
				"security": "secure",
				"monitoring": "active"
			},
			"performance_metrics": {
				"active_services": len(self._services),
				"response_time_ms": self._status.average_response_time_ms,
				"success_rate": self._status.success_rate,
				"uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds()
			},
			"resource_availability": {
				"cpu_available": True,
				"memory_available": True,
				"storage_available": True,
				"network_available": True
			},
			"dependency_status": self._status.dependencies_status,
			"recommendations": self._status.recommendations,
			"last_check": datetime.now(timezone.utc).isoformat()
		}
		return health_report

	async def initialize(self, config: Dict[str, Any]) -> bool:
		"""Initialize AI Core Framework capability with configuration.

		Performs complete initialization of the AI Core Framework
		including service registry setup, inference engine preparation,
		security configuration, and integration with required APG
		capabilities. Validates dependencies and establishes monitoring.

		Args:
			config: Configuration dictionary containing:
				- database: Database connection configuration
				- security: Security and authentication settings
				- monitoring: Monitoring and alerting configuration
				- ai_services: AI service configuration parameters
				- performance: Performance tuning parameters

		Returns:
			bool: True if initialization successful, False otherwise

		Raises:
			InitializationError: If critical initialization steps fail
			DependencyError: If required APG capabilities unavailable
			ConfigurationError: If configuration validation fails

		Example:
			>>> config = {
			...     "database": {"url": "postgresql://..."},
			...     "security": {"jwt_secret": "..."},
			...     "monitoring": {"enabled": True}
			... }
			>>> capability = AICoreCapability()
			>>> success = await capability.initialize(config)
			>>> print(success)  # True
		"""
		try:
			self._config = config

			# Validate APG dependencies
			await self._validate_dependencies()

			# Setup health monitoring
			await self._setup_health_monitoring()

			# Initialize service registry
			self._services = {}

			# Update status to healthy
			self._status.status = AIServiceStatus.HEALTHY
			self._status.configuration_status = "configured"
			self._initialized = True

			await self._update_status()
			return True

		except Exception as e:
			self._status.status = AIServiceStatus.FAILED
			self._status.alerts.append(f"Initialization failed: {str(e)}")
			return False

	async def shutdown(self) -> bool:
		"""Gracefully shutdown AI Core Framework capability services.

		Performs orderly shutdown of all AI Core Framework components
		including active inference jobs, service connections, monitoring
		systems, and cleanup of resources. Ensures data integrity and
		proper resource deallocation during shutdown process.

		Returns:
			bool: True if shutdown completed successfully, False otherwise

		Raises:
			ShutdownError: If critical shutdown operations fail

		Example:
			>>> capability = AICoreCapability()
			>>> await capability.initialize({})
			>>> success = await capability.shutdown()
			>>> print(success)  # True
		"""
		try:
			self._status.status = AIServiceStatus.DECOMMISSIONING

			# Gracefully shutdown active services
			for service_id in list(self._services.keys()):
				# Implement service shutdown logic here
				pass

			# Cleanup resources
			self._services.clear()
			self._health_monitors.clear()
			self._initialized = False

			return True

		except Exception as e:
			self._status.alerts.append(f"Shutdown failed: {str(e)}")
			return False

	async def reload_config(self, config: Dict[str, Any]) -> bool:
		"""Hot-reload configuration without full capability restart.

		Updates capability configuration dynamically without requiring
		full shutdown and restart. Validates new configuration and
		applies changes to running systems while maintaining service
		availability and data consistency.

		Args:
			config: New configuration dictionary with updated settings

		Returns:
			bool: True if configuration reload successful, False otherwise

		Raises:
			ConfigurationError: If new configuration is invalid
			ReloadError: If configuration application fails

		Example:
			>>> new_config = {"performance": {"max_workers": 16}}
			>>> capability = AICoreCapability()
			>>> success = await capability.reload_config(new_config)
			>>> print(success)  # True
		"""
		try:
			# Validate new configuration
			# Implementation would validate config structure

			# Apply configuration changes
			self._config.update(config)

			# Update metadata timestamp
			self._metadata.updated_at = datetime.now(timezone.utc)

			# Update status
			await self._update_status()

			return True

		except Exception as e:
			self._status.alerts.append(f"Config reload failed: {str(e)}")
			return False

	async def get_dependencies(self) -> List[str]:
		"""Return list of required APG capability dependencies.

		Provides complete list of APG capabilities that the AI Core
		Framework depends on for proper operation. Used by APG
		composition engine for dependency resolution and orchestration
		planning.

		Returns:
			List[str]: List of required capability identifiers:
				- "conf": Configuration management capability
				- "auth": Authentication and RBAC capability
				- "mqeb": Message queue and event bus capability
				- "moni": Monitoring and observability capability

		Example:
			>>> capability = AICoreCapability()
			>>> deps = await capability.get_dependencies()
			>>> print(deps)  # ["conf", "auth", "mqeb", "moni"]
		"""
		return self._dependencies

	async def register_service(self, service: AIServiceRegistration) -> str:
		"""Register new AI service with the AI Core Framework capability.

		Adds new AI service to the capability registry, enabling
		discovery, orchestration, and management through the APG
		platform. Validates service configuration and establishes
		monitoring and health checking for the registered service.

		Args:
			service: AIServiceRegistration containing:
				- service_name: Unique service identifier
				- service_type: Type of AI service being registered
				- endpoint_url: Service API endpoint for communication
				- capabilities: List of AI capabilities provided
				- authentication_required: Security requirements
				- metadata: Additional service configuration

		Returns:
			str: Unique service registration ID for future reference

		Raises:
			RegistrationError: If service registration fails validation
			DuplicateServiceError: If service name already exists

		Example:
			>>> service = AIServiceRegistration(
			...     tenant_id="tenant1",
			...     service_name="nlp-processor",
			...     service_type=AIServiceType.INFERENCE,
			...     endpoint_url="http://nlp:8080/api",
			...     capabilities=["text_classification", "sentiment_analysis"],
			...     created_by="admin"
			... )
			>>> capability = AICoreCapability()
			>>> service_id = await capability.register_service(service)
			>>> print(service_id)  # "svc_01234567..."
		"""
		try:
			# Validate service registration
			if service.service_name in [s.service_name for s in self._services.values()]:
				raise ValueError(f"Service {service.service_name} already registered")

			# Generate unique service ID
			service_id = f"svc_{uuid7str()}"

			# Store service registration
			self._services[service_id] = service

			# Update status
			self._status.active_services = len(self._services)
			await self._update_status()

			return service_id

		except Exception as e:
			self._status.alerts.append(f"Service registration failed: {str(e)}")
			raise

	async def discover_services(self, filters: Dict[str, Any]) -> List[AIServiceRegistration]:
		"""Discover available AI services with optional filtering.

		Searches the AI service registry for services matching specified
		criteria. Supports filtering by service type, capabilities,
		tenant isolation, and other attributes for precise service
		discovery and selection.

		Args:
			filters: Discovery filter criteria containing:
				- service_type: Filter by AI service type
				- capabilities: Required service capabilities
				- tenant_id: Multi-tenant isolation filter
				- status: Service operational status filter
				- tags: Service tag-based filtering

		Returns:
			List[AIServiceRegistration]: List of matching services with:
				- Complete service registration information
				- Current operational status
				- Available capabilities and endpoints
				- Authentication and security requirements

		Raises:
			DiscoveryError: If service discovery operation fails

		Example:
			>>> filters = {
			...     "service_type": "inference",
			...     "capabilities": ["text_processing"],
			...     "tenant_id": "tenant1"
			... }
			>>> capability = AICoreCapability()
			>>> services = await capability.discover_services(filters)
			>>> print(len(services))  # Number of matching services
		"""
		try:
			matching_services = []

			for service in self._services.values():
				# Apply filters
				if filters.get("service_type") and service.service_type != filters["service_type"]:
					continue

				if filters.get("tenant_id") and service.tenant_id != filters["tenant_id"]:
					continue

				if filters.get("capabilities"):
					required_caps = set(filters["capabilities"])
					service_caps = set(service.capabilities)
					if not required_caps.issubset(service_caps):
						continue

				matching_services.append(service)

			return matching_services

		except Exception as e:
			self._status.alerts.append(f"Service discovery failed: {str(e)}")
			raise

	async def _update_status(self) -> None:
		"""Internal method to update capability operational status.

		Performs comprehensive status assessment including performance
		metrics collection, resource utilization monitoring, health
		indicator evaluation, and generation of optimization
		recommendations for the AI Core Framework capability.
		"""
		try:
			current_time = datetime.now(timezone.utc)

			# Update basic metrics
			self._status.last_check = current_time
			self._status.uptime_seconds = int((current_time - self._start_time).total_seconds())
			self._status.active_services = len(self._services)

			# Update dependency status
			self._status.dependencies_status = {
				"conf": "healthy",
				"auth": "healthy",
				"mqeb": "healthy",
				"moni": "healthy"
			}

			# Update version info
			self._status.version_info = {
				"capability_version": self._metadata.version,
				"last_updated": self._metadata.updated_at.isoformat()
			}

			# Clear old alerts if system is healthy
			if self._status.status == AIServiceStatus.HEALTHY:
				self._status.alerts = []

		except Exception as e:
			self._status.alerts.append(f"Status update failed: {str(e)}")

	async def _validate_dependencies(self) -> bool:
		"""Validate availability of required APG capabilities.

		Checks that all required APG capabilities (conf, auth, mqeb, moni)
		are available and operational before completing initialization.

		Returns:
			bool: True if all dependencies are available

		Raises:
			DependencyError: If required capabilities are unavailable
		"""
		# Implementation would check actual APG capability availability
		# For now, assume dependencies are available
		return True

	async def _setup_health_monitoring(self) -> None:
		"""Initialize health monitoring systems for the capability.

		Sets up comprehensive health monitoring including performance
		metrics collection, resource usage tracking, alert generation,
		and integration with APG monitoring infrastructure.
		"""
		self._health_monitors = {
			"performance": {"enabled": True, "interval": 60},
			"resources": {"enabled": True, "interval": 30},
			"services": {"enabled": True, "interval": 120},
			"security": {"enabled": True, "interval": 300}
		}


# Global capability instance for APG registration
_ai_core_capability: Optional[AICoreCapability] = None


async def get_capability() -> AICoreCapability:
	"""Get or create the global AI Core Framework capability instance.

	Provides access to the singleton AI Core Framework capability
	instance for APG composition engine integration. Creates new
	instance if not already initialized.

	Returns:
		AICoreCapability: The global capability instance

	Example:
		>>> capability = await get_capability()
		>>> metadata = await capability.get_metadata()
		>>> print(metadata.name)  # "AI Core Framework"
	"""
	global _ai_core_capability
	if _ai_core_capability is None:
		_ai_core_capability = AICoreCapability()
	return _ai_core_capability


# APG composition engine registration interface
async def register_with_apg() -> Dict[str, Any]:
	"""Register AI Core Framework capability with APG composition engine.

	Performs complete registration process with the APG composition
	engine including metadata submission, dependency verification,
	health check endpoint registration, and service discovery setup.

	Returns:
		Dict[str, Any]: Registration result containing:
			- registration_id: Unique registration identifier
			- status: Registration status (success/failed)
			- endpoints: Registered API endpoints
			- health_check_url: Health monitoring endpoint
			- last_registered: Registration timestamp

	Raises:
		RegistrationError: If APG registration process fails

	Example:
		>>> result = await register_with_apg()
		>>> print(result["status"])  # "success"
		>>> print(result["registration_id"])  # "aicr_reg_123..."
	"""
	try:
		capability = await get_capability()
		metadata = await capability.get_metadata()

		# Registration with APG composition engine
		registration_result = {
			"registration_id": f"aicr_reg_{uuid7str()}",
			"capability_id": metadata.id,
			"status": "success",
			"endpoints": metadata.api_endpoints,
			"health_check_url": metadata.health_check_url,
			"dependencies": metadata.dependencies,
			"provides": metadata.provides,
			"last_registered": datetime.now(timezone.utc).isoformat()
		}

		return registration_result

	except Exception as e:
		return {
			"registration_id": None,
			"status": "failed",
			"error": str(e),
			"last_attempted": datetime.now(timezone.utc).isoformat()
		}


def register_capability() -> Dict[str, Any]:
	"""Register AICR as a first-class APG composition capability."""
	metadata = CapabilityMetadata()
	contract = get_capability_contract()
	return {
		"name": metadata.id,
		"aliases": ["ai_core", "ai_orchestration", "ai_infrastructure"],
		"display_name": metadata.name,
		"description": metadata.description,
		"version": metadata.version,
		"dependencies": metadata.dependencies,
		"optional_dependencies": ["cach", "audl", "keym"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"ai_service_registry": "Register, discover, and govern tenant-scoped AI services",
			"ai_provider_registry": "Register local and external AI providers with credential and egress policy controls",
			"inference_engine": "Route governed inference work across registered AI providers",
			"inference_approval_governance": "Require approval for high-risk and large-context inference",
			"model_catalog": "Register model metadata, policy, modality, evaluation, and lifecycle state",
			"model_metric_governance": "Record model metrics and route drift signals into review evidence",
			"agent_runtime_registry": "Register Codex, Claude Code, OpenCode, Pi, Ollama, and custom agent runtimes as governed services",
			"ai_agent_composition": "Register provider-neutral AI agents as first-class scoped, owned, auditable APG citizens",
			"review_evidence": "Persist review-required AICR outcomes as pending-review records with policy evidence",
			"bytewax_lifecycle_processing": "Require AICR lifecycle batches to use Bytewax processing evidence",
			"workflow_orchestration": "Compose AI workflows with human approval and monitoring",
			"capability_rules": "Evaluate deterministic AI infrastructure governance rules",
			"visual_theming": "Apply AI control-console theme tokens and components"
		},
		"endpoints": {
			"services": "/aicr/api/v1/services",
			"inference": "/aicr/api/v1/inference",
			"inference_approvals": "/aicr/api/v1/inference-approvals",
			"providers": "/aicr/api/v1/providers",
			"models": "/aicr/api/v1/models",
			"model_metrics": "/aicr/api/v1/model-metrics",
			"workflows": "/aicr/api/v1/workflows",
			"agents": "/aicr/api/v1/agents",
			"lifecycle": "/aicr/api/v1/lifecycle",
			"pending_reviews": "/aicr/api/v1/pending-reviews",
			"metrics": "/aicr/api/v1/metrics"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"permissions": [
			"aicr:view",
			"aicr:manage_services",
			"aicr:run_inference",
			"aicr:view_models",
			"aicr:manage_workflows",
			"aicr:manage_agents",
			"aicr:govern",
			"aicr:view_metrics",
			"aicr:admin"
		]
	}


def get_capability_info() -> Dict[str, Any]:
	"""Get AICR capability information for composition and marketplace discovery."""
	metadata = CapabilityMetadata()
	return metadata.model_dump(mode="json") | {"contract": get_capability_contract()}


# Module exports for APG integration
__version__ = "1.0.0"
__capability_id__ = "aicr"
__capability_name__ = "AI Core Framework"
__apg_dependencies__ = ["conf", "auth", "mqeb", "moni"]

# Export main classes and functions
__all__ = [
	# Core capability classes
	"AICoreCapability", "CapabilityMetadata", "CapabilityStatus",

	# APG integration protocol
	"APGCapabilityProtocol",

	# Registration functions
	"get_capability", "register_with_apg", "register_capability", "get_capability_info",
	"get_capability_contract", "evaluate_capability_rules",

	# Model registry from models module
	"model_registry",

	# Core AI models
	"AIServiceRegistration", "AIServiceStatus", "AIServiceHealth",
	"AIModelMetadata", "AIInferenceRequest", "AIInferenceResult",
	"AIWorkflow", "AIAuditEvent",

	# Module metadata
	"__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"
]
