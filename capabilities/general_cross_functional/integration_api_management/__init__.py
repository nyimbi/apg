"""
APG Integration API Management - Module Initialization

Comprehensive API gateway and management platform providing secure, scalable,
and monitored integration between APG capabilities and external systems.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from .models import (
	# SQLAlchemy Models
	AMAPI,
	AMEndpoint,
	AMPolicy,
	AMConsumer,
	AMAPIKey,
	AMSubscription,
	AMDeployment,
	AMAnalytics,
	AMUsageRecord,
	
	# Enums
	APIStatus,
	APIVersion,
	ProtocolType,
	AuthenticationType,
	PolicyType,
	DeploymentStrategy,
	ConsumerStatus,
	MetricType,
	LoadBalancingAlgorithm,
	
	# Pydantic Models
	APIConfig,
	EndpointConfig,
	PolicyConfig,
	ConsumerConfig,
	APIKeyConfig,
	SubscriptionConfig
)

from .service import (
	APILifecycleService,
	ConsumerManagementService,
	PolicyManagementService,
	AnalyticsService
)

from .views import (
	# Forms
	APIConfigForm,
	PolicyConfigForm,
	ConsumerRegistrationForm,
	
	# API Management Views
	APIManagementView,
	EndpointManagementView,
	PolicyManagementView,
	
	# Consumer Management Views
	ConsumerManagementView,
	APIKeyManagementView,
	
	# Analytics Views
	AnalyticsDashboardView,
	UsageRecordsView,
	
	# Developer Portal Views
	DeveloperPortalView,
	
	# Deployment Views
	DeploymentManagementView
)

from .api import (
	APIManagementApi,
	ConsumerManagementApi,
	AnalyticsApi,
	GatewayApi,
	register_api_endpoints
)

from .blueprint import (
	integration_api_management_bp,
	create_integration_api_management_blueprint,
	init_integration_api_management
)

from .discovery import (
	ServiceDiscovery,
	APGCapabilityInfo,
	APIDiscoveryInfo,
	ServiceHealth,
	CapabilityType
)

from .integration import (
	APGIntegrationManager,
	APGEvent,
	EventType,
	WorkflowStatus,
	CrossCapabilityWorkflow,
	WorkflowStep,
	PolicyRule
)

from .monitoring import (
	MetricsCollector,
	HealthMonitor,
	AlertManager,
	HealthCheck,
	Metric,
	HealthReport
)

from .gateway import (
	APIGateway,
	GatewayRouter,
	LoadBalancer,
	CircuitBreaker
)

from .config import (
	APIManagementSettings,
	create_configuration,
	ConfigurationManager
)

from .runner import (
	GatewayApplication,
	run_gateway,
	main
)

from .factory import (
	IntegrationAPIManagementCapability,
	create_integration_api_management_capability,
	create_standalone_capability,
	get_capability_metadata
)

# Capability metadata
__capability_info__ = {
	'capability_id': 'integration_api_management',
	'capability_name': 'Integration API Management',
	'capability_code': 'IAM',
	'version': '1.0.0',
	'category': 'general_cross_functional',
	'maturity_level': 'foundation_infrastructure',
	'criticality': 'CRITICAL',
	'description': 'Comprehensive API gateway and management platform for secure, scalable integration',
	'features': [
		'High-Performance API Gateway (100K+ RPS)',
		'OAuth 2.0/OIDC Authentication & JWT Management',
		'API Lifecycle Management with Versioning',
		'Developer Portal with Interactive Documentation',
		'Real-time Analytics & Performance Monitoring',
		'Policy Engine for Security & Rate Limiting',
		'Multi-tenant Isolation & Enterprise Security'
	],
	'dependencies': [
		'capability_registry',
		'event_streaming_bus'
	],
	'provides': [
		'api_gateway',
		'api_lifecycle_management',
		'consumer_management',
		'analytics_monitoring',
		'developer_portal'
	]
}

# Version information
__version__ = '1.0.0'
__author__ = 'Nyimbi Odero <nyimbi@gmail.com>'
__copyright__ = '© 2025 Datacraft. All rights reserved.'

# Export all public interfaces
__all__ = [
	# Core capability info
	'__capability_info__',
	'__version__',
	'__author__',
	'__copyright__',
	
	# SQLAlchemy Models
	'AMAPI',
	'AMEndpoint', 
	'AMPolicy',
	'AMConsumer',
	'AMAPIKey',
	'AMSubscription',
	'AMDeployment',
	'AMAnalytics',
	'AMUsageRecord',
	
	# Enums
	'APIStatus',
	'APIVersion',
	'ProtocolType',
	'AuthenticationType',
	'PolicyType',
	'DeploymentStrategy',
	'ConsumerStatus',
	'MetricType',
	'LoadBalancingAlgorithm',
	'ServiceHealth',
	'CapabilityType',
	'EventType',
	'WorkflowStatus',
	
	# Pydantic Models
	'APIConfig',
	'EndpointConfig',
	'PolicyConfig',
	'ConsumerConfig',
	'APIKeyConfig',
	'SubscriptionConfig',
	'APGCapabilityInfo',
	'APIDiscoveryInfo',
	'CrossCapabilityWorkflow',
	'WorkflowStep',
	'PolicyRule',
	
	# Services
	'APILifecycleService',
	'ConsumerManagementService',
	'PolicyManagementService',
	'AnalyticsService',
	
	# Forms
	'APIConfigForm',
	'PolicyConfigForm',
	'ConsumerRegistrationForm',
	
	# Views
	'APIManagementView',
	'EndpointManagementView',
	'PolicyManagementView',
	'ConsumerManagementView',
	'APIKeyManagementView',
	'AnalyticsDashboardView',
	'UsageRecordsView',
	'DeveloperPortalView',
	'DeploymentManagementView',
	
	# API Endpoints
	'APIManagementApi',
	'ConsumerManagementApi',
	'AnalyticsApi',
	'GatewayApi',
	'register_api_endpoints',
	
	# Blueprint
	'integration_api_management_bp',
	'create_integration_api_management_blueprint',
	'init_integration_api_management',
	
	# Discovery and Integration
	'ServiceDiscovery',
	'APGIntegrationManager',
	'APGEvent',
	
	# Monitoring and Gateway
	'MetricsCollector',
	'HealthMonitor',
	'AlertManager',
	'HealthCheck',
	'Metric',
	'HealthReport',
	'APIGateway',
	'GatewayRouter',
	'LoadBalancer',
	'CircuitBreaker',
	
	# Configuration and Runner
	'APIManagementSettings',
	'create_configuration',
	'ConfigurationManager',
	'GatewayApplication',
	'run_gateway',
	'main',
	
	# Factory
	'IntegrationAPIManagementCapability',
	'create_integration_api_management_capability',
	'create_standalone_capability',
	'get_capability_metadata'
]