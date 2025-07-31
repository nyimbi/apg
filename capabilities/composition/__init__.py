"""
APG Capabilities Composition Engine

Modern composition system that enables APG programmers to compose custom ERP solutions by
selecting specific capabilities and sub-capabilities. This system provides:

- Auto-discovery of all available capabilities and sub-capabilities
- Advanced AI-powered composition validation and optimization
- Industry-specific templates for quick deployment
- Dynamic Flask application generation with enterprise features
- Intelligent dependency resolution and conflict detection
- Multi-tenant orchestration and deployment automation
- Real-time monitoring and performance analytics
- Enterprise-grade security and compliance integration

Core Components:
- CapabilityRegistry: Auto-discovery and metadata management
- IntelligentCompositionEngine: AI-powered composition optimization
- DeploymentAutomation: Automated deployment and scaling
- WorkflowOrchestration: Advanced workflow management
- CentralConfiguration: Unified configuration management
- AccessControlIntegration: Centralized access control
- APIServiceMesh: Service mesh and API management
- EventStreamingBus: Event-driven architecture support

The system makes the entire hierarchical capability architecture work together seamlessly
with enterprise-grade performance, security, and scalability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

# Import modern composition components
from .capability_registry import (
	# Core Services
	CRService,
	get_registry_service,
	APGIntegrationService,
	get_apg_integration_service,
	
	# Advanced Composition Engine
	IntelligentCompositionEngine,
	get_composition_engine,
	CompositionValidationResult,
	ConflictReport,
	CompositionRecommendation,
	PerformanceImpact,
	ConflictSeverity,
	RecommendationType,
	
	# Core Models
	CRCapability,
	CRDependency,
	CRComposition,
	CRVersion,
	CRMetadata,
	CRRegistry,
	
	# Enums
	CRCapabilityStatus,
	CRDependencyType,
	CRCompositionType,
	CRVersionConstraint,
	
	# APG Integration
	APGTenantContext,
	CRServiceResponse,
	CAPABILITY_METADATA as CR_CAPABILITY_METADATA,
)

from .deployment_automation import (
	DeploymentAutomationService,
	DeploymentStrategy,
	DeploymentEnvironment,
	DeploymentStatus,
	DeploymentTarget,
	DeploymentConfig,
	DeploymentResult,
	get_deployment_service
)

from .workflow_orchestration import (
	WorkflowEngine,
	WorkflowDefinition,
	WorkflowInstance,
	WorkflowStatus,
	TaskStatus,
	TaskType,
	WorkflowTask,
	get_workflow_engine,
	WORKFLOW_TEMPLATES
)

from .central_configuration import (
	CentralConfigurationManager,
	ConfigurationApplet,
	ConfigurationField,
	ConfigurationScope,
	ConfigurationDataType,
	get_configuration_manager,
	register_configuration_applet
)

from .access_control_integration import (
	AccessControlIntegration,
	TenantManager,
	CapabilityPermission,
	CompositionPermission,
	AccessLevel,
	PermissionScope,
	get_tenant_manager,
	get_access_control,
	COMPOSITION_ROLES
)

# Version and metadata
__version__ = "2.0.0"
__author__ = "APG Development Team"

# Modern composition types
class CompositionType(str, Enum):
	"""Types of capability compositions."""
	ERP_ENTERPRISE = "erp_enterprise"
	INDUSTRY_VERTICAL = "industry_vertical"
	DEPARTMENTAL = "departmental"
	MICROSERVICE = "microservice"
	HYBRID = "hybrid"
	CUSTOM = "custom"

# Sub-capability Registry
SUBCAPABILITIES = [
	"capability_registry",
	"workflow_orchestration", 
	"api_service_mesh",
	"event_streaming_bus",
	"deployment_automation",
	"central_configuration",
	"access_control_integration"
]

# Enhanced Composition Templates
COMPOSITION_TEMPLATES = {
	"complete_erp": {
		"description": "Complete enterprise resource planning solution",
		"capabilities": [
			"core_business_operations",
			"manufacturing_production",
			"general_cross_functional",
			"composition"
		],
		"deployment_strategy": DeploymentStrategy.MICROSERVICES,
		"estimated_deployment_time": "4-6 weeks"
	},
	"healthcare_enterprise": {
		"description": "Healthcare enterprise management platform",
		"capabilities": [
			"industry_vertical_solutions.healthcare_medical",
			"core_business_operations.financial_management",
			"core_business_operations.human_capital_management",
			"general_cross_functional.governance_risk_compliance"
		],
		"deployment_strategy": DeploymentStrategy.HYBRID_CLOUD,
		"compliance_requirements": ["HIPAA", "HITECH"],
		"estimated_deployment_time": "6-8 weeks"
	},
	"smart_manufacturing": {
		"description": "AI-powered smart manufacturing platform",
		"capabilities": [
			"manufacturing_production",
			"emerging_technologies.artificial_intelligence",
			"emerging_technologies.edge_computing_iot",
			"general_cross_functional.enterprise_asset_management"
		],
		"deployment_strategy": DeploymentStrategy.EDGE_DISTRIBUTED,
		"estimated_deployment_time": "8-12 weeks"
	},
	"digital_commerce": {
		"description": "Complete digital commerce and marketplace platform",
		"capabilities": [
			"platform_foundation",
			"general_cross_functional.customer_relationship_management",
			"core_business_operations.sales_revenue_management",
			"emerging_technologies.artificial_intelligence"
		],
		"deployment_strategy": DeploymentStrategy.SERVERLESS,
		"estimated_deployment_time": "3-5 weeks"
	}
}

# Modern composition interface
async def compose_application(
	tenant_id: str,
	user_id: str,
	capabilities: List[str],
	composition_type: CRCompositionType = CRCompositionType.ENTERPRISE,
	industry_focus: Optional[List[str]] = None,
	custom_config: Optional[Dict[str, Any]] = None
) -> CompositionValidationResult:
	"""
	Modern AI-powered composition with intelligent recommendations and optimization.
	
	Args:
		tenant_id: Unique tenant identifier
		user_id: User identifier for permissions and audit
		capabilities: List of capability IDs to compose
		composition_type: Type of composition being created
		industry_focus: Industry-specific requirements
		custom_config: Custom configuration overrides
		
	Returns:
		CompositionValidationResult with AI recommendations and analysis
	"""
	# Check permissions
	access_control = get_access_control(tenant_id)
	if access_control:
		can_create = await access_control.check_composition_permission(
			user_id=user_id,
			composition_type=composition_type.value,
			operation="create"
		)
		if not can_create:
			raise PermissionError("User does not have composition creation permissions")
	
	# Get composition engine with database session
	registry_service = get_registry_service()
	db_session = await registry_service.get_db_session(tenant_id)
	
	engine = IntelligentCompositionEngine(
		db_session=db_session,
		tenant_id=tenant_id,
		user_id=user_id
	)
	
	return await engine.validate_composition(
		capability_ids=capabilities,
		composition_type=composition_type,
		industry_focus=industry_focus,
		custom_config=custom_config
	)

async def discover_capabilities(
	tenant_id: str,
	user_id: str,
	filters: Optional[Dict[str, Any]] = None
) -> List[CRCapability]:
	"""Discover available capabilities with tenant and user context."""
	access_control = get_access_control(tenant_id)
	
	# Get all capabilities
	registry_service = get_registry_service()
	all_capabilities = await registry_service.discover_capabilities(tenant_id)
	
	# Filter by user permissions
	accessible_capabilities = []
	for capability in all_capabilities:
		if access_control:
			has_access = await access_control.check_capability_access(
				user_id=user_id,
				capability_id=capability.id,
				requested_access=AccessLevel.READ
			)
			if has_access:
				accessible_capabilities.append(capability)
		else:
			accessible_capabilities.append(capability)
	
	# Apply additional filters
	if filters:
		# Apply category, tags, status filters
		if "category" in filters:
			accessible_capabilities = [
				c for c in accessible_capabilities 
				if c.category == filters["category"]
			]
		if "status" in filters:
			accessible_capabilities = [
				c for c in accessible_capabilities 
				if c.status == filters["status"]
			]
	
	return accessible_capabilities

async def create_tenant(
	tenant_id: str,
	admin_user_id: str,
	tenant_name: str,
	enabled_capabilities: List[str],
	configuration: Optional[Dict[str, Any]] = None
) -> bool:
	"""Create a new tenant with composition capabilities."""
	tenant_manager = get_tenant_manager()
	
	success = tenant_manager.create_tenant(
		tenant_id=tenant_id,
		name=tenant_name,
		enabled_capabilities=enabled_capabilities,
		admin_user_id=admin_user_id,
		metadata=configuration or {}
	)
	
	if success:
		# Initialize configuration manager for tenant
		config_manager = get_configuration_manager()
		
		# Apply default configurations
		if configuration:
			for applet_id, config in configuration.items():
				await config_manager.update_configuration(
					tenant_id=tenant_id,
					applet_id=applet_id,
					user_id=admin_user_id,
					updates=config,
					reason="Initial tenant setup"
				)
	
	return success

async def deploy_composition(
	tenant_id: str,
	user_id: str,
	composition_id: str,
	deployment_target: DeploymentTarget,
	deployment_strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
	require_approval: bool = True
) -> str:
	"""Deploy composition with modern workflow integration."""
	# Check deployment permissions
	access_control = get_access_control(tenant_id)
	if access_control:
		can_deploy = await access_control.check_composition_permission(
			user_id=user_id,
			composition_type="deployment",
			operation="deploy"
		)
		if not can_deploy:
			raise PermissionError("User does not have deployment permissions")
	
	if require_approval:
		# Create deployment approval workflow
		workflow_engine = get_workflow_engine(tenant_id)
		
		workflow = WorkflowDefinition(
			name=f"Deploy Composition {composition_id}",
			tenant_id=tenant_id,
			created_by=user_id,
			category="deployment",
			tasks=[
				WorkflowTask(
					name="Security Review",
					task_type=TaskType.AUTOMATED,
					description="Automated security validation"
				),
				WorkflowTask(
					name="Deployment Approval",
					task_type=TaskType.APPROVAL,
					assigned_to="deployment_manager",
					dependencies=["security_review"],
					sla_hours=24
				),
				WorkflowTask(
					name="Execute Deployment",
					task_type=TaskType.AUTOMATED,
					dependencies=["deployment_approval"],
					actions=[{
						"type": "deploy_composition",
						"composition_id": composition_id,
						"target": deployment_target.__dict__,
						"strategy": deployment_strategy.value
					}]
				)
			]
		)
		
		workflow_id = workflow_engine.create_workflow(workflow)
		instance_id = await workflow_engine.start_workflow(
			workflow_id=workflow_id,
			started_by=user_id,
			initial_context={
				"composition_id": composition_id,
				"target": deployment_target.__dict__,
				"strategy": deployment_strategy.value
			}
		)
		
		return instance_id
	else:
		# Direct deployment
		deployment_service = get_deployment_service(tenant_id)
		result = await deployment_service.deploy_composition(
			composition_id=composition_id,
			target=deployment_target,
			strategy=deployment_strategy
		)
		return result.deployment_id

async def get_composition_recommendations(
	tenant_id: str,
	user_id: str,
	business_requirements: Dict[str, Any],
	industry: Optional[str] = None
) -> List[CompositionRecommendation]:
	"""Get AI-powered composition recommendations based on business needs."""
	registry_service = get_registry_service()
	db_session = await registry_service.get_db_session(tenant_id)
	
	engine = IntelligentCompositionEngine(
		db_session=db_session,
		tenant_id=tenant_id,
		user_id=user_id
	)
	
	return await engine.generate_business_recommendations(
		business_requirements=business_requirements,
		industry=industry
	)

async def configure_capability(
	tenant_id: str,
	capability_id: str,
	user_id: str,
	configuration: Dict[str, Any]
) -> bool:
	"""Configure a capability using central configuration management."""
	config_manager = get_configuration_manager()
	
	# Apply configuration
	errors = await config_manager.update_configuration(
		tenant_id=tenant_id,
		applet_id=capability_id,
		user_id=user_id,
		updates=configuration,
		reason="Capability configuration update"
	)
	
	return len(errors) == 0

# Utility functions
def get_capability_info() -> Dict[str, Any]:
	"""Get composition capability information."""
	return CR_CAPABILITY_METADATA

def list_subcapabilities() -> List[str]:
	"""List all available subcapabilities."""
	return SUBCAPABILITIES.copy()

def get_composition_templates() -> Dict[str, Any]:
	"""Get available composition templates."""
	return COMPOSITION_TEMPLATES.copy()

def get_deployment_strategies() -> List[DeploymentStrategy]:
	"""Get supported deployment strategies."""
	return list(DeploymentStrategy)

# Export modern interface
__all__ = [
	# Core Composition Functions
	"compose_application",
	"discover_capabilities",
	"create_tenant",
	"deploy_composition",
	"get_composition_recommendations",
	"configure_capability",
	
	# Core Services
	"CRService",
	"get_registry_service",
	"APGIntegrationService",
	"get_apg_integration_service",
	
	# AI-Powered Composition Engine
	"IntelligentCompositionEngine",
	"get_composition_engine",
	"CompositionValidationResult",
	"ConflictReport",
	"CompositionRecommendation",
	"PerformanceImpact",
	"ConflictSeverity",
	"RecommendationType",
	
	# Enterprise Models
	"CRCapability",
	"CRDependency",
	"CRComposition",
	"CRVersion",
	"CRMetadata",
	"CRRegistry",
	
	# Enterprise Enums
	"CRCapabilityStatus",
	"CRDependencyType",
	"CRCompositionType",
	"CRVersionConstraint",
	
	# Modern Types
	"CompositionType",
	
	# Deployment Automation
	"DeploymentAutomationService",
	"DeploymentStrategy",
	"DeploymentEnvironment",
	"DeploymentStatus",
	"DeploymentTarget",
	"DeploymentConfig",
	"DeploymentResult",
	"get_deployment_service",
	
	# Workflow Orchestration
	"WorkflowEngine",
	"WorkflowDefinition",
	"WorkflowInstance",
	"WorkflowStatus",
	"TaskStatus",
	"TaskType",
	"WorkflowTask",
	"get_workflow_engine",
	"WORKFLOW_TEMPLATES",
	
	# Central Configuration
	"CentralConfigurationManager",
	"ConfigurationApplet",
	"ConfigurationField",
	"ConfigurationScope",
	"ConfigurationDataType",
	"get_configuration_manager",
	"register_configuration_applet",
	
	# Access Control Integration
	"AccessControlIntegration",
	"TenantManager",
	"CapabilityPermission",
	"CompositionPermission",
	"AccessLevel",
	"PermissionScope",
	"get_tenant_manager",
	"get_access_control",
	"COMPOSITION_ROLES",
	
	# APG Integration
	"APGTenantContext",
	"CRServiceResponse",
	
	# Utility Functions
	"get_capability_info",
	"list_subcapabilities",
	"get_composition_templates",
	"get_deployment_strategies",
	
	# Template Data
	"SUBCAPABILITIES",
	"COMPOSITION_TEMPLATES",
]