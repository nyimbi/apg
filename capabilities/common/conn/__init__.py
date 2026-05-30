"""APG Connection Management capability."""

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
from .conn_runtime import ConnService

# APG Composition Engine Registration
def register_capability() -> dict:
	"""Register the connection management capability with APG composition engine."""
	contract = get_capability_contract()
	return {
		'name': 'conn',
		'display_name': 'Connection Management',
		'description': 'Governed connector, connection, flow, and Singer tap lifecycle control plane',
		'version': '1.0.0',
		'dependencies': ['apig', 'auth', 'encr', 'audl', 'keym', 'moni', 'regy'],
		'configuration': contract['configuration'],
		'configuration_schema': contract['configuration_schema'],
		'rule_engine': contract['rule_engine'],
		'capabilities': {
			'connectors': 'Register local Singer, APG, HTTP, database, file, and stream connectors',
			'connections': 'Manage tenant-scoped data source and target connections',
			'flows': 'Compose governed data flows with mapping, lineage, and quality gates',
			'sync_runs': 'Track sync, replay, schedule, and batch lifecycle records',
			'monitoring': 'Connection health and sync monitoring adapter surface',
			'singer_taps': 'Local Singer.io tap management and registry integration',
			'visual_designer': 'Generated-app flow design and composition metadata',
			'capability_rules': 'Capability-specific rule evaluation',
			'visual_theming': 'Tenant-aware UI theme tokens and components'
		},
		'endpoints': {
			'connections': '/api/v1/connections',
			'flows': '/api/v1/flows',
			'taps': '/api/v1/taps',
			'monitoring': '/api/v1/monitoring',
			'lineage': '/api/v1/lineage'
		},
		'ui_components': {
			route['name']: route['path']
			for route in contract['ui']['routes']
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
	'ConnService',
	'register_capability',
	'get_capability_contract',
	'evaluate_capability_rules'
]
