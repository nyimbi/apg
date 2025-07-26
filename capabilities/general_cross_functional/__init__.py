"""
APG Enhanced General Cross-Functional Capabilities

Comprehensive cross-cutting capabilities that provide foundational services
across all business functions and industry verticals.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import List, Dict, Any

# Enhanced General Cross-Functional Metadata
__version__ = "2.0.0"
__category__ = "general_cross_functional"
__description__ = "Enhanced cross-cutting capabilities for enterprise-wide functionality"

# Enhanced Sub-capability Registry
SUBCAPABILITIES = [
	"customer_relationship_management",
	"enterprise_asset_management", 
	"workflow_business_process_mgmt",
	"document_content_management",
	"business_intelligence_analytics",
	"governance_risk_compliance",
	"geographical_location_services",
	"advanced_analytics_platform",
	"integration_api_management",
	"sustainability_esg_management",
	"knowledge_learning_management",
	"mobile_device_management",
	"multi_language_localization",
	"product_lifecycle_management"
]

# New Capabilities Overview
NEW_CAPABILITIES = {
	"geographical_location_services": {
		"description": "Comprehensive geofencing and location intelligence",
		"key_features": ["Geofencing", "Territory Management", "Location Analytics", "Compliance"],
		"integration_points": ["CRM", "Asset Management", "Workflow", "Analytics"]
	},
	"advanced_analytics_platform": {
		"description": "Self-service analytics and data science workbench",
		"key_features": ["Data Discovery", "ML Workbench", "Visual Analytics", "Predictive Models"],
		"integration_points": ["All Capabilities", "External Data Sources", "BI Tools"]
	},
	"integration_api_management": {
		"description": "Enterprise integration hub and API management",
		"key_features": ["API Gateway", "Integration Hub", "Connector Marketplace", "Event Streaming"],
		"integration_points": ["External Systems", "Legacy Applications", "Cloud Services"]
	},
	"sustainability_esg_management": {
		"description": "ESG reporting and sustainability management",
		"key_features": ["Carbon Tracking", "ESG Reporting", "Sustainability KPIs", "Compliance"],
		"integration_points": ["Asset Management", "Financial Management", "Reporting"]
	},
	"knowledge_learning_management": {
		"description": "Corporate knowledge base and learning platform",
		"key_features": ["Knowledge Base", "Learning Paths", "Expert Networks", "AI Search"],
		"integration_points": ["Document Management", "HR", "Collaboration"]
	},
	"mobile_device_management": {
		"description": "Enterprise mobile applications and device management",
		"key_features": ["Mobile Apps", "Device Policies", "Offline Sync", "Security"],
		"integration_points": ["All Capabilities", "Authentication", "Sync Services"]
	},
	"multi_language_localization": {
		"description": "Internationalization and localization management",
		"key_features": ["Translation Management", "Cultural Adaptation", "Regional Compliance"],
		"integration_points": ["All UI Components", "Documentation", "Workflows"]
	}
}

# APG Composition Engine Registration
CAPABILITY_METADATA = {
	"capability_id": "general_cross_functional",
	"version": __version__,
	"category": "cross_functional",
	"subcapabilities": SUBCAPABILITIES,
	"dependencies": [
		"auth_rbac",
		"audit_compliance",
		"notification_engine"
	],
	"provides_services": [
		"customer_relationship_services",
		"asset_management_services",
		"workflow_automation_services",
		"document_content_services",
		"business_intelligence_services", 
		"governance_compliance_services",
		"location_intelligence_services",
		"advanced_analytics_services",
		"integration_management_services",
		"sustainability_reporting_services",
		"knowledge_learning_services",
		"mobile_platform_services",
		"localization_services",
		"product_lifecycle_services"
	],
	"composition_priority": 1,  # High priority - foundational
	"new_capabilities": list(NEW_CAPABILITIES.keys())
}

def get_capability_info() -> Dict[str, Any]:
	"""Get enhanced general cross-functional capability information."""
	return CAPABILITY_METADATA

def list_subcapabilities() -> List[str]:
	"""List all available subcapabilities."""
	return SUBCAPABILITIES.copy()

def get_new_capabilities() -> Dict[str, Any]:
	"""Get information about new capabilities."""
	return NEW_CAPABILITIES.copy()

def list_legacy_capabilities() -> List[str]:
	"""List capabilities that existed in previous version."""
	return [
		"customer_relationship_management",
		"enterprise_asset_management",
		"workflow_business_process_mgmt", 
		"document_content_management",
		"business_intelligence_analytics",
		"governance_risk_compliance",
		"product_lifecycle_management"
	]

def list_enhanced_capabilities() -> List[str]:
	"""List capabilities that have been enhanced."""
	return [
		"document_content_management",  # Enhanced from document_management
		"business_intelligence_analytics"  # Enhanced with advanced features
	]

__all__ = [
	"SUBCAPABILITIES",
	"NEW_CAPABILITIES",
	"CAPABILITY_METADATA",
	"get_capability_info",
	"list_subcapabilities",
	"get_new_capabilities",
	"list_legacy_capabilities",
	"list_enhanced_capabilities"
]