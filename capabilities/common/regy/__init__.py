#!/usr/bin/env python3
"""
Registry (regy) - APG API/Service Registry
==========================================

A comprehensive API and service registry capability that provides intelligent service discovery,
dynamic registration, health monitoring, and real-time synchronization within the APG ecosystem.

Author: APG Platform Team
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from .models import *
from .service import *

__version__ = "1.0.0"
__capability_name__ = "Registry (regy)"
__capability_description__ = "API/Service Registry with intelligent service discovery"
__apg_capability_id__ = "regy"
__apg_capability_type__ = "integration"
__apg_dependencies__ = ["auth", "conf", "moni", "audl"]

# APG Capability Metadata for Composition Engine
APG_CAPABILITY_METADATA = {
	"id": "regy",
	"name": "Registry (regy)",
	"description": "API/Service Registry with intelligent service discovery and health monitoring",
	"version": __version__,
	"type": "integration",
	"category": "service_management",
	"dependencies": [
		"auth",      # Authentication and RBAC
		"conf",      # Configuration management
		"moni",      # Monitoring and observability
		"audl",      # Audit logging
		"apig"       # API Gateway integration
	],
	"provides": [
		"service_discovery",
		"service_registration",
		"health_monitoring",
		"load_balancing",
		"circuit_breaking",
		"api_versioning"
	],
	"integration_points": {
		"auth": ["service_authentication", "rbac_policies"],
		"conf": ["dynamic_configuration", "service_config"],
		"moni": ["service_metrics", "health_checks"],
		"audl": ["registration_events", "discovery_logs"],
		"apig": ["gateway_integration", "routing_updates"]
	},
	"endpoints": {
		"rest_api": "/api/regy/v1",
		"health": "/api/regy/v1/health",
		"metrics": "/api/regy/v1/metrics",
		"discovery": "/api/regy/v1/discover",
		"registration": "/api/regy/v1/register"
	},
	"ui_integration": {
		"menu_items": [
			{
				"name": "Service Registry",
				"url": "/regy/services",
				"icon": "fa-network-wired",
				"category": "Integration"
			},
			{
				"name": "Service Discovery",
				"url": "/regy/discovery",
				"icon": "fa-search",
				"category": "Integration"
			}
		],
		"dashboard_widgets": [
			{
				"name": "Service Health",
				"component": "ServiceHealthWidget",
				"size": "medium"
			}
		]
	}
}
