"""
Field Service Management Sub-Capability

Manages mobile workforce, dispatching technicians, tracking service calls,
and managing parts/equipment in the field.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Field Service Management',
	'code': 'FS',
	'version': '1.0.0',
	'capability': 'service_specific',
	'description': 'Manages mobile workforce, dispatching technicians, tracking service calls, and managing parts/equipment in the field',
	'industry_focus': 'Field Services, Maintenance, Utilities, Telecommunications',
	'dependencies': [],
	'optional_dependencies': ['service_contract_management', 'resource_scheduling'],
	'database_tables': [
		'ss_fs_service_call',
		'ss_fs_technician',
		'ss_fs_dispatch',
		'ss_fs_work_order',
		'ss_fs_parts_inventory',
		'ss_fs_mobile_device',
		'ss_fs_service_report',
		'ss_fs_location_tracking'
	],
	'configuration': {
		'enable_gps_tracking': True,
		'enable_mobile_app': True,
		'auto_dispatch_optimization': True,
		'require_digital_signature': True,
		'enable_parts_management': True,
		'service_call_rating_enabled': True,
		'emergency_response_time_minutes': 120
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META