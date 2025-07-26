"""
Recruitment & Onboarding Sub-Capability

Manages the hiring process from job posting, applicant tracking, to new hire 
integration and documentation. Includes candidate management and onboarding workflows.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Recruitment & Onboarding',
	'code': 'RO',
	'version': '1.0.0',
	'capability': 'human_resources',
	'description': 'Manages the hiring process from job posting, applicant tracking, to new hire integration and documentation',
	'industry_focus': 'All',
	'dependencies': ['employee_data_management'],
	'optional_dependencies': ['performance_management'],
	'database_tables': [
		'hr_ro_job_posting',
		'hr_ro_candidate',
		'hr_ro_application',
		'hr_ro_interview',
		'hr_ro_interview_feedback',
		'hr_ro_job_offer',
		'hr_ro_onboarding_task',
		'hr_ro_onboarding_checklist',
		'hr_ro_document_template',
		'hr_ro_background_check'
	],
	'api_endpoints': [
		'/api/human_resources/recruitment/job_postings',
		'/api/human_resources/recruitment/candidates',
		'/api/human_resources/recruitment/applications',
		'/api/human_resources/recruitment/interviews',
		'/api/human_resources/onboarding/tasks',
		'/api/human_resources/onboarding/checklists'
	],
	'views': [
		'HRJobPostingModelView',
		'HRCandidateModelView',
		'HRApplicationModelView',
		'HRInterviewModelView',
		'HROnboardingTaskModelView',
		'HRRecruitmentDashboardView'
	],
	'permissions': [
		'recruitment.read',
		'recruitment.write',
		'recruitment.post_jobs',
		'recruitment.interview',
		'recruitment.make_offers',
		'onboarding.manage',
		'recruitment.admin'
	],
	'configuration': {
		'enable_job_board_integration': True,
		'require_background_checks': True,
		'auto_create_onboarding_tasks': True,
		'enable_candidate_portal': True,
		'interview_scheduling_buffer_hours': 24,
		'offer_expiry_days': 7,
		'onboarding_period_days': 90
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
		errors.append("Employee Data Management is required for Recruitment & Onboarding")
	
	if 'performance_management' not in available_subcapabilities:
		warnings.append("Performance Management integration not available - new hire review tracking unavailable")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}