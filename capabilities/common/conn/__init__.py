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
		'dependencies': contract['requires'],
		'optional_dependencies': ['keym', 'moni', 'regy', 'meta', 'dqol'],
		'provides': contract['provides'],
		'configuration': contract['configuration'],
		'configuration_schema': contract['configuration_schema'],
		'rule_engine': contract['rule_engine'],
		'agents': contract['agents'],
		'streaming': contract['streaming'],
		'review_evidence': contract['review_evidence'],
		'capabilities': {
			'connectors': 'Register local Singer, APG, HTTP, database, file, and stream connectors',
			'connections': 'Manage tenant-scoped data source and target connections',
			'flows': 'Compose governed data flows with mapping, lineage, and quality gates',
			'sync_runs': 'Track sync, replay, schedule, and batch lifecycle records',
			'monitoring': 'Connection health and sync monitoring adapter surface',
			'singer_taps': 'Local Singer.io tap management and registry integration',
			'visual_designer': 'Generated-app flow design and composition metadata',
			'capability_rules': 'Capability-specific rule evaluation',
			'connector_agent_composition': 'Govern AI and automation agents that review or mutate connector state',
			'bytewax_lifecycle_batches': 'Validate connector lifecycle mutation batches through Bytewax',
			'review_evidence': 'Preserve policy decisions, matched rules, review reasons, and review evidence for generated connector governance queues',
			'visual_theming': 'Tenant-aware UI theme tokens and components'
		},
		'endpoints': {
			'connections': '/api/v1/connections',
			'flows': '/api/v1/flows',
			'taps': '/api/v1/taps',
			'monitoring': '/api/v1/monitoring',
			'lineage': '/api/v1/lineage',
			'agents': '/api/v1/agents',
			'lifecycle': '/api/v1/lifecycle'
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
