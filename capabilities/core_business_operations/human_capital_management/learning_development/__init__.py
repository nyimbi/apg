"""
Learning & Development Sub-Capability

Manages training programs, certifications, and skill development for workforce enhancement.
Includes training catalog, learning paths, and competency tracking.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Learning & Development',
	'code': 'LD',
	'version': '1.0.0',
	'capability': 'human_resources',
	'description': 'Manages training programs, certifications, and skill development for workforce enhancement',
	'industry_focus': 'All',
	'dependencies': ['employee_data_management'],
	'optional_dependencies': ['performance_management'],
	'database_tables': [
		'hr_ld_training_program',
		'hr_ld_course',
		'hr_ld_learning_path',
		'hr_ld_enrollment',
		'hr_ld_completion',
		'hr_ld_instructor',
		'hr_ld_training_session',
		'hr_ld_assessment',
		'hr_ld_competency_framework',
		'hr_ld_learning_objective',
		'hr_ld_training_budget'
	],
	'api_endpoints': [
		'/api/human_resources/learning/programs',
		'/api/human_resources/learning/courses',
		'/api/human_resources/learning/enrollments',
		'/api/human_resources/learning/completions',
		'/api/human_resources/learning/assessments',
		'/api/human_resources/learning/reports'
	],
	'views': [
		'HRTrainingProgramModelView',
		'HRCourseModelView',
		'HREnrollmentModelView',
		'HRCompletionModelView',
		'HRInstructorModelView',
		'HRLearningDashboardView'
	],
	'permissions': [
		'learning.read',
		'learning.write',
		'learning.enroll',
		'learning.manage_programs',
		'learning.approve_training',
		'learning.view_reports',
		'learning.admin'
	],
	'configuration': {
		'enable_external_providers': True,
		'require_manager_approval': True,
		'enable_learning_paths': True,
		'auto_assign_mandatory_training': True,
		'certificate_expiry_tracking': True,
		'enable_mobile_learning': True,
		'training_budget_tracking': True,
		'completion_certificate_generation': True
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
		errors.append("Employee Data Management is required for Learning & Development")
	
	if 'performance_management' not in available_subcapabilities:
		warnings.append("Performance Management integration not available - development plan integration limited")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}