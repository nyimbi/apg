"""
Benefits Administration Sub-Capability

Manages employee benefits plans including health, retirement, leave, and other benefits.
Includes enrollment, eligibility management, and benefits reporting.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Benefits Administration',
	'code': 'BA',
	'version': '1.0.0',
	'capability': 'human_resources',
	'description': 'Manages employee benefits plans (health, retirement, leave, etc.)',
	'industry_focus': 'All',
	'dependencies': ['employee_data_management'],
	'optional_dependencies': ['payroll'],
	'database_tables': [
		'hr_ba_benefit_plan',
		'hr_ba_benefit_category',
		'hr_ba_employee_benefit',
		'hr_ba_enrollment_period',
		'hr_ba_benefit_election',
		'hr_ba_dependent',
		'hr_ba_beneficiary',
		'hr_ba_cobra_event',
		'hr_ba_claim',
		'hr_ba_premium_calculation',
		'hr_ba_carrier'
	],
	'api_endpoints': [
		'/api/human_resources/benefits/plans',
		'/api/human_resources/benefits/enrollments',
		'/api/human_resources/benefits/elections',
		'/api/human_resources/benefits/dependents',
		'/api/human_resources/benefits/claims',
		'/api/human_resources/benefits/reports'
	],
	'views': [
		'HRBenefitPlanModelView',
		'HREmployeeBenefitModelView',
		'HRBenefitElectionModelView',
		'HRDependentModelView',
		'HRClaimModelView',
		'HRBenefitsDashboardView'
	],
	'permissions': [
		'benefits.read',
		'benefits.write',
		'benefits.enroll',
		'benefits.approve',
		'benefits.manage_plans',
		'benefits.view_claims',
		'benefits.admin'
	],
	'configuration': {
		'enable_open_enrollment': True,
		'enable_life_events': True,
		'cobra_notification_days': 60,
		'enrollment_confirmation_required': True,
		'enable_dependent_verification': True,
		'auto_calculate_premiums': True,
		'enable_benefits_portal': True,
		'waiting_period_days': 90
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	if 'employee_data_management' not in available_subcapabilities:
		errors.append("Employee Data Management is required for Benefits Administration")
	
	if 'payroll' not in available_subcapabilities:
		warnings.append("Payroll integration not available - benefits deductions will need manual setup")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}