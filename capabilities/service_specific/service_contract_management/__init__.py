"""
Service Contract Management Sub-Capability

Manages service agreements, warranties, and recurring service schedules
with automated renewal and compliance tracking.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Service Contract Management',
	'code': 'SC',
	'version': '1.0.0',
	'capability': 'service_specific',
	'description': 'Manages service agreements, warranties, and recurring service schedules',
	'industry_focus': 'Field Services, Maintenance Services, Support Services',
	'dependencies': [],
	'optional_dependencies': ['field_service_management', 'professional_services_automation'],
	'database_tables': [
		'ss_sc_service_contract',
		'ss_sc_contract_line_item',
		'ss_sc_service_level_agreement',
		'ss_sc_warranty',
		'ss_sc_renewal_schedule',
		'ss_sc_compliance_check',
		'ss_sc_contract_performance'
	],
	'configuration': {
		'enable_auto_renewal': True,
		'renewal_notification_days': 90,
		'enable_sla_monitoring': True,
		'default_contract_term_months': 12,
		'enable_contract_templates': True,
		'require_electronic_signature': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META