"""
APG Connection Management Capability

A revolutionary integration platform that transforms how enterprises connect,
synchronize, and orchestrate data across systems using locally hosted Singer.io
infrastructure with AI-driven automation.

Key Features:
- Local Singer.io tap ecosystem with 20+ connectors
- Zero-configuration AI-powered schema detection
- Real-time bi-directional data synchronization
- Visual flow designer with collaborative editing
- Self-healing connections with predictive analytics
- Enterprise security with end-to-end encryption
- Complete APG platform integration

Author: APG Platform Team
Version: 1.0.0
License: Proprietary - Datacraft © 2025
"""

from .models import (
	Connection,
	SingerTap,
	SingerTarget,
	DataFlow,
	TransformationRule,
	ConnectionHealth
)

from .service import (
	ConnectionManager,
	SingerRuntimeManager,
	FlowExecutor,
	IntelligentConnector
)

from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)

# APG Composition Engine Registration
def register_capability() -> dict:
	"""Register the connection management capability with APG composition engine."""
	contract = get_capability_contract()
	return {
		'name': 'conn',
		'display_name': 'Connection Management',
		'description': 'Enterprise integration platform with Singer.io ecosystem',
		'version': '1.0.0',
		'dependencies': ['apig', 'auth', 'encr', 'audl'],
		'configuration': contract['configuration'],
		'configuration_schema': contract['configuration_schema'],
		'rule_engine': contract['rule_engine'],
		'capabilities': {
			'connections': 'Manage data source connections',
			'transformations': 'Real-time data transformation',
			'monitoring': 'Connection health monitoring',
			'singer_taps': 'Local Singer.io tap management',
			'ai_mapping': 'AI-powered schema mapping',
			'visual_designer': 'Drag-and-drop flow design',
			'capability_rules': 'Capability-specific rule evaluation',
			'visual_theming': 'Tenant-aware UI theme tokens and components'
		},
		'endpoints': {
			'connections': '/api/v1/connections',
			'flows': '/api/v1/flows',
			'taps': '/api/v1/taps',
			'monitoring': '/api/v1/monitoring'
		},
		'ui_components': {
			'designer': '/conn/designer',
			'dashboard': '/conn/dashboard',
			'monitoring': '/conn/monitoring',
			'rules': '/conn/rules',
			'settings': '/conn/settings'
		},
		'ui_manifest': contract['ui'],
		'theme': contract['theme'],
		'permissions': [
			'conn:view',
			'conn:create',
			'conn:edit',
			'conn:delete',
			'conn:admin'
		]
	}

__all__ = [
	'Connection',
	'SingerTap',
	'SingerTarget',
	'DataFlow',
	'TransformationRule',
	'ConnectionHealth',
	'ConnectionManager',
	'SingerRuntimeManager',
	'FlowExecutor',
	'IntelligentConnector',
	'register_capability',
	'get_capability_contract',
	'evaluate_capability_rules'
]
