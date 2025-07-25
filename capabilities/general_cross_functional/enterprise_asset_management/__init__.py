"""
Enterprise Asset Management (EAM) Sub-Capability

Manages the entire lifecycle of physical assets including maintenance,
repair, operations, and asset performance optimization.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Enterprise Asset Management (EAM)',
	'code': 'EA',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Manages the entire lifecycle of physical assets (maintenance, repair, operations, and asset performance)',
	'industry_focus': 'Manufacturing, Utilities, Transportation, Healthcare',
	'dependencies': [],
	'optional_dependencies': ['workflow_business_process_mgmt', 'business_intelligence_analytics'],
	'database_tables': [
		'gc_ea_asset',
		'gc_ea_asset_hierarchy',
		'gc_ea_maintenance_plan',
		'gc_ea_work_order',
		'gc_ea_asset_condition',
		'gc_ea_spare_parts',
		'gc_ea_failure_analysis',
		'gc_ea_asset_performance'
	],
	'configuration': {
		'enable_predictive_maintenance': True,
		'enable_iot_integration': True,
		'default_maintenance_strategy': 'preventive',
		'enable_mobile_work_orders': True,
		'auto_parts_replenishment': True,
		'condition_monitoring_frequency_hours': 24,
		'enable_asset_criticality_scoring': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META