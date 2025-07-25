"""
Resource Scheduling Sub-Capability

Optimizes the allocation and scheduling of human and equipment resources
for projects and tasks across the organization.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Resource Scheduling',
	'code': 'RS',
	'version': '1.0.0',
	'capability': 'service_specific',
	'description': 'Optimizes the allocation and scheduling of human and equipment resources for projects and tasks',
	'industry_focus': 'Professional Services, Field Services, Consulting',
	'dependencies': [],
	'optional_dependencies': ['project_management', 'time_expense_tracking'],
	'database_tables': [
		'ss_rs_resource',
		'ss_rs_resource_skill',
		'ss_rs_schedule',
		'ss_rs_availability',
		'ss_rs_booking',
		'ss_rs_capacity_plan',
		'ss_rs_utilization_report'
	],
	'configuration': {
		'enable_skill_matching': True,
		'enable_automatic_scheduling': True,
		'default_booking_duration_hours': 8,
		'utilization_target_percentage': 85,
		'schedule_optimization_algorithm': 'genetic',
		'enable_overbooking_warnings': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META