"""
APG API Service Mesh - Intelligent Service Communication and Traffic Management

A comprehensive service mesh implementation providing service discovery, load balancing,
traffic management, and observability for the APG platform ecosystem.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@gmail.com"
__copyright__ = "© 2025 Datacraft. All rights reserved."

# Core capability metadata
CAPABILITY_INFO = {
	"capability_code": "ASM",
	"capability_name": "API Service Mesh",
	"category": "composition_orchestration",
	"subcategory": "service_mesh",
	"version": __version__,
	"description": "Intelligent API orchestration and service mesh networking",
	"long_description": (
		"Comprehensive service mesh providing service discovery, load balancing, "
		"traffic management, security, and observability for distributed systems"
	),
	"author": __author__,
	"copyright": __copyright__,
	"tags": ["service_mesh", "api_gateway", "load_balancing", "service_discovery", "traffic_management"],
	"requires_database": True,
	"requires_redis": True,
	"multi_tenant": True,
	"audit_enabled": True
}

# Import core components
from .models import (
	# Core service models
	SMService,
	SMEndpoint,
	SMRoute,
	SMLoadBalancer,
	SMPolicy,
	
	# Monitoring models
	SMMetrics,
	SMTrace,
	SMHealthCheck,
	SMAlert,
	SMTopology,
	
	# Configuration models
	SMConfiguration,
	SMCertificate,
	SMSecurityPolicy,
	SMRateLimiter,
	
	# Enums
	ServiceStatus,
	EndpointProtocol,
	LoadBalancerAlgorithm,
	HealthStatus,
	PolicyType,
	RouteMatchType
)

from .service import (
	ASMService,
	ServiceDiscoveryService,
	TrafficManagementService,
	LoadBalancerService,
	PolicyEngineService,
	HealthMonitoringService,
	MetricsCollectionService
)

from .api import api_app, router as api_router
from .apg_integration import APGServiceMeshIntegration

# Flask-AppBuilder components
from .blueprint import create_blueprint
from .views import (
	ASMServiceView,
	ASMEndpointView,
	ASMRouteView,
	ASMLoadBalancerView,
	ASMPolicyView,
	ASMMetricsView,
	ASMTopologyView
)

# Export main classes for external use
__all__ = [
	# Metadata
	"CAPABILITY_INFO",
	"__version__",
	
	# Models
	"SMService",
	"SMEndpoint", 
	"SMRoute",
	"SMLoadBalancer",
	"SMPolicy",
	"SMMetrics",
	"SMTrace",
	"SMHealthCheck",
	"SMAlert",
	"SMTopology",
	"SMConfiguration",
	"SMCertificate",
	"SMSecurityPolicy",
	"SMRateLimiter",
	
	# Enums
	"ServiceStatus",
	"EndpointProtocol",
	"LoadBalancerAlgorithm",
	"HealthStatus",
	"PolicyType",
	"RouteMatchType",
	
	# Services
	"ASMService",
	"ServiceDiscoveryService",
	"TrafficManagementService",
	"LoadBalancerService",
	"PolicyEngineService",
	"HealthMonitoringService",
	"MetricsCollectionService",
	
	# API and Integration
	"api_app",
	"api_router",
	"APGServiceMeshIntegration",
	
	# UI Components
	"create_blueprint",
	"ASMServiceView",
	"ASMEndpointView",
	"ASMRouteView",
	"ASMLoadBalancerView",
	"ASMPolicyView",
	"ASMMetricsView",
	"ASMTopologyView"
]

# Initialize logging
import logging
import sys

def setup_logging():
	"""Setup logging configuration for the service mesh."""
	logger = logging.getLogger(__name__)
	
	if not logger.handlers:
		handler = logging.StreamHandler(sys.stdout)
		formatter = logging.Formatter(
			'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
		)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		logger.setLevel(logging.INFO)
	
	return logger

# Initialize logger
logger = setup_logging()
logger.info(f"APG API Service Mesh v{__version__} initialized")

# Health check function
def health_check() -> dict[str, str]:
	"""Basic health check for the service mesh capability."""
	return {
		"status": "healthy",
		"capability": CAPABILITY_INFO["capability_name"],
		"version": __version__,
		"components": [
			"service_discovery",
			"traffic_management", 
			"load_balancing",
			"health_monitoring",
			"metrics_collection"
		]
	}

# Capability registration information for APG
def get_capability_info() -> dict[str, any]:
	"""Get capability information for APG platform registration."""
	return {
		**CAPABILITY_INFO,
		"health_check": health_check,
		"api_endpoints": [
			"/api/services",
			"/api/routes", 
			"/api/load-balancers",
			"/api/policies",
			"/api/metrics",
			"/api/health"
		],
		"ui_endpoints": [
			"/service-mesh/dashboard",
			"/service-mesh/services",
			"/service-mesh/topology",
			"/service-mesh/monitoring"
		],
		"dependencies": [
			"capability_registry",
			"event_streaming_bus"
		],
		"provides_services": [
			"service_discovery",
			"load_balancing",
			"traffic_routing",
			"health_monitoring",
			"metrics_collection",
			"security_policies"
		]
	}