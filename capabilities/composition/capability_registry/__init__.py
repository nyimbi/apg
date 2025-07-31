"""
APG Capability Registry - Foundation Infrastructure

Intelligent capability discovery, registration, and orchestration services
that enable APG's unique modular, composable architecture.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any, Optional, Set
from uuid_extensions import uuid7str

# Capability Registry Metadata
__version__ = "1.0.0"
__capability_code__ = "CAPABILITY_REGISTRY"
__capability_name__ = "APG Capability Registry"
__description__ = "Foundational infrastructure for APG capability discovery and orchestration"

# APG Composition Keywords for Integration
__composition_keywords__ = [
	"enables_capability_discovery",
	"provides_composition_engine",
	"manages_dependencies",
	"orchestrates_capabilities",
	"validates_compositions",
	"enables_marketplace_integration",
	"provides_metadata_management",
	"supports_versioning",
	"enables_intelligent_recommendations",
	"foundational_infrastructure"
]

# APG Integration Dependencies
__apg_dependencies__ = [
	"auth_rbac",  # For capability access control and user authentication
	"audit_compliance",  # For audit trails and compliance logging
	"notification_engine",  # For alerts and status notifications
]

# APG Service Integrations
__apg_integrations__ = [
	"ai_orchestration",  # For intelligent composition recommendations
	"real_time_collaboration",  # For collaborative composition design
	"multi_language_localization",  # For international metadata support
	"advanced_analytics_platform",  # For capability usage analytics
]

# APG Capability Metadata for Composition Engine Registration
CAPABILITY_METADATA = {
	"capability_id": "composition_orchestration.capability_registry",
	"capability_code": __capability_code__,
	"capability_name": __capability_name__,
	"version": __version__,
	"description": __description__,
	"category": "foundation_infrastructure",
	"priority": 0,  # Highest priority - foundational capability
	"dependencies": __apg_dependencies__,
	"integrations": __apg_integrations__,
	"provides_services": [
		"capability_discovery",
		"capability_registration",
		"metadata_management",
		"dependency_resolution",
		"composition_validation",
		"version_management",
		"marketplace_integration",
		"intelligent_recommendations"
	],
	"data_models": [
		"CRCapability",
		"CRDependency", 
		"CRComposition",
		"CRVersion",
		"CRMetadata",
		"CRRegistry"
	],
	"api_endpoints": [
		"/api/v1/capabilities",
		"/api/v1/compositions",
		"/api/v1/dependencies",
		"/api/v1/versions",
		"/api/v1/marketplace"
	],
	"composition_keywords": __composition_keywords__,
	"multi_tenant": True,
	"audit_enabled": True,
	"security_integration": True,
	"performance_optimized": True,
	"ai_enhanced": True
}

def get_capability_info() -> Dict[str, Any]:
	"""Get capability registry information for APG composition engine."""
	return CAPABILITY_METADATA

def get_capability_dependencies() -> List[str]:
	"""Get list of APG capability dependencies."""
	return __apg_dependencies__.copy()

def get_capability_integrations() -> List[str]:
	"""Get list of APG capability integrations."""
	return __apg_integrations__.copy()

def get_composition_keywords() -> List[str]:
	"""Get composition keywords for capability discovery."""
	return __composition_keywords__.copy()

def _log_capability_startup() -> str:
	"""Log capability registry startup information."""
	return f"APG Capability Registry v{__version__} - Foundation infrastructure initialized"

def _log_registry_operation(operation: str, details: str) -> str:
	"""Log registry operations for debugging and monitoring."""
	return f"Registry Operation: {operation} - {details}"

# APG Tenant Context Integration
class APGTenantContext:
	"""APG tenant context for multi-tenant operations."""
	
	def __init__(
		self,
		tenant_id: str,
		user_id: str,
		user_roles: List[str],
		permissions: List[str]
	):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.user_roles = user_roles
		self.permissions = permissions
		self.context_id = uuid7str()

# Registry Service Response Pattern
class CRServiceResponse:
	"""Standard service response for capability registry operations."""
	
	def __init__(
		self,
		success: bool,
		message: str,
		data: Optional[Dict[str, Any]] = None,
		errors: Optional[List[str]] = None
	):
		self.success = success
		self.message = message
		self.data = data or {}
		self.errors = errors or []
		self.response_id = uuid7str()

# Import core components
from .service import CRService, get_registry_service
from .apg_integration import APGIntegrationService, get_apg_integration_service
from .api import api_app
from .mobile_service import MobileOfflineService
from .models import (
	CRCapability, CRDependency, CRComposition, CRVersion, CRMetadata, CRRegistry,
	CRCapabilityStatus, CRDependencyType, CRCompositionType, CRVersionConstraint
)
from .composition_engine import (
	IntelligentCompositionEngine, get_composition_engine,
	CompositionValidationResult, ConflictReport, CompositionRecommendation,
	PerformanceImpact, ConflictSeverity, RecommendationType
)

__all__ = [
	# Service Components
	"CRService",
	"get_registry_service",
	"APGIntegrationService",
	"get_apg_integration_service",
	"MobileOfflineService",
	"api_app",
	
	# Composition Engine
	"IntelligentCompositionEngine",
	"get_composition_engine",
	"CompositionValidationResult",
	"ConflictReport",
	"CompositionRecommendation",
	"PerformanceImpact",
	"ConflictSeverity",
	"RecommendationType",
	
	# Core Models
	"CRCapability",
	"CRDependency",
	"CRComposition", 
	"CRVersion",
	"CRMetadata",
	"CRRegistry",
	
	# Enums
	"CRCapabilityStatus",
	"CRDependencyType",
	"CRCompositionType",
	"CRVersionConstraint",
	
	# APG Integration
	"APGTenantContext",
	"CRServiceResponse",
	"CAPABILITY_METADATA",
	"get_capability_info",
	"get_capability_dependencies", 
	"get_capability_integrations",
	"get_composition_keywords"
]

# APG Composition Engine Auto-Registration
async def register_with_apg_composition_engine(tenant_id: str = "default"):
	"""Register this capability with APG's composition engine."""
	apg_service = await get_apg_integration_service(tenant_id)
	registry_service = await get_registry_service(tenant_id)
	await apg_service.set_registry_service(registry_service)
	
	# Register the capability registry itself as a foundational capability
	capability_id = CAPABILITY_METADATA["capability_id"]
	await apg_service.register_with_composition_engine(capability_id)
	await apg_service.register_with_discovery_service(capability_id)
	
	return apg_service