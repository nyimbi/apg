"""
Performance Management Sub-Capability

Tracks employee goals, conducts performance reviews, and manages appraisals 
for talent development. Includes goal setting, feedback, and development planning.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Performance Management',
	'code': 'PM',
	'version': '1.0.0',
	'capability': 'human_resources',
	'description': 'Tracks employee goals, conducts performance reviews, and manages appraisals for talent development',
	'industry_focus': 'All',
	'dependencies': ['employee_data_management'],
	'optional_dependencies': ['learning_development'],
	'database_tables': [
		'hr_pm_performance_review',
		'hr_pm_review_template',
		'hr_pm_review_question',
		'hr_pm_review_response',
		'hr_pm_goal',
		'hr_pm_goal_progress',
		'hr_pm_feedback',
		'hr_pm_development_plan',
		'hr_pm_competency',
		'hr_pm_competency_rating',
		'hr_pm_performance_rating'
	],
	'api_endpoints': [
		'/api/human_resources/performance/reviews',
		'/api/human_resources/performance/goals',
		'/api/human_resources/performance/feedback',
		'/api/human_resources/performance/development_plans',
		'/api/human_resources/performance/competencies',
		'/api/human_resources/performance/reports'
	],
	'views': [
		'HRPerformanceReviewModelView',
		'HRGoalModelView',
		'HRFeedbackModelView',
		'HRDevelopmentPlanModelView',
		'HRCompetencyModelView',
		'HRPerformanceDashboardView'
	],
	'permissions': [
		'performance.read',
		'performance.write',
		'performance.review',
		'performance.approve',
		'performance.view_all_reviews',
		'performance.manage_goals',
		'performance.admin'
	],
	'configuration': {
		'enable_360_feedback': True,
		'require_manager_review': True,
		'enable_self_assessment': True,
		'review_cycle_months': 12,
		'mid_year_review_enabled': True,
		'enable_continuous_feedback': True,
		'goal_weight_percentage': True,
		'auto_create_development_plans': True
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
		errors.append("Employee Data Management is required for Performance Management")
	
	if 'learning_development' not in available_subcapabilities:
		warnings.append("Learning & Development integration not available - development plan tracking limited")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}