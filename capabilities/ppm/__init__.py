"""
Professional Services Automation (PSA) Sub-Capability

Integrated suite for professional service firms covering project management,
time/expense tracking, billing, and resource management.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Professional Services Automation (PSA)',
	'code': 'PS',
	'version': '1.0.0',
	'capability': 'service_specific',
	'description': 'Integrated suite for professional service firms covering project management, time/expense, billing, and resource management',
	'industry_focus': 'Professional Services, Consulting, Legal Services, Accounting',
	'dependencies': ['project_management', 'time_expense_tracking', 'resource_scheduling'],
	'optional_dependencies': ['service_contract_management'],
	'database_tables': [
		'ss_ps_client_engagement',
		'ss_ps_proposal',
		'ss_ps_billing_schedule',
		'ss_ps_invoice_generation',
		'ss_ps_profitability_analysis',
		'ss_ps_practice_area',
		'ss_ps_consultant_profile',
		'ss_ps_knowledge_base'
	],
	'configuration': {
		'enable_proposal_automation': True,
		'enable_automatic_billing': True,
		'profitability_calculation_method': 'activity_based',
		'enable_client_portal': True,
		'default_billing_frequency': 'monthly',
		'enable_knowledge_management': True,
		'consultant_utilization_target': 80
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# Check hard dependencies
	required_deps = ['project_management', 'time_expense_tracking', 'resource_scheduling']
	for dep in required_deps:
		if dep not in available_subcapabilities:
			errors.append(f"PSA requires {dep} sub-capability")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}