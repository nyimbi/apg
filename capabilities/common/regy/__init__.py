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
from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)

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


def register_capability() -> dict:
	"""Register REGY with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "regy",
		"aliases": ["service_registry", "api_registry", "service_discovery"],
		"display_name": "API/Service Registry",
		"description": __capability_description__,
		"version": __version__,
		"dependencies": APG_CAPABILITY_METADATA["dependencies"],
		"optional_dependencies": [],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"service_registration": "Register tenant-scoped services and API contracts",
			"service_discovery": "Discover healthy service instances for callers and gateway sync",
			"health_monitoring": "Track active health checks and registration status",
			"api_versioning": "Govern compatible and breaking service versions",
			"capability_rules": "Evaluate deterministic registry governance rules",
			"visual_theming": "Apply service-catalog theme tokens and components"
		},
		"endpoints": {
			"services": "/regy/api/v1/services",
			"registration": "/regy/api/v1/register",
			"discovery": "/regy/api/v1/discover",
			"health": "/regy/api/v1/health",
			"versions": "/regy/api/v1/versions",
			"gateway_sync": "/regy/api/v1/gateway-sync"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": [
			"regy:view",
			"regy:view_services",
			"regy:register_service",
			"regy:discover",
			"regy:view_health",
			"regy:manage_versions",
			"regy:sync_gateway",
			"regy:admin"
		]
	}


def get_capability_info() -> dict:
	"""Get REGY capability information for composition and marketplace discovery."""
	info = APG_CAPABILITY_METADATA.copy()
	info["contract"] = get_capability_contract()
	return info
