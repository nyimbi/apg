"""
Project Management Sub-Capability

Plans, tracks, and manages projects from initiation to completion,
including tasks, timelines, resources, and deliverables.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Project Management',
	'code': 'PM',
	'version': '1.0.0',
	'capability': 'service_specific',
	'description': 'Plans, tracks, and manages projects from initiation to completion, including tasks, timelines, and resources',
	'industry_focus': 'Professional Services, Consulting, IT Services',
	'dependencies': [],
	'optional_dependencies': ['resource_scheduling', 'time_expense_tracking'],
	'database_tables': [
		'ss_pm_project',
		'ss_pm_task',
		'ss_pm_milestone',
		'ss_pm_deliverable',
		'ss_pm_resource_assignment',
		'ss_pm_project_template',
		'ss_pm_status_report',
		'ss_pm_risk_issue'
	],
	'api_endpoints': [
		'/api/services/project_management/projects',
		'/api/services/project_management/tasks',
		'/api/services/project_management/milestones',
		'/api/services/project_management/resources',
		'/api/services/project_management/reports'
	],
	'views': [
		'SSPMProjectModelView',
		'SSPMTaskModelView',
		'SSPMMilestoneModelView',
		'SSPMResourceAssignmentModelView',
		'SSPMDashboardView'
	],
	'permissions': [
		'project_management.read',
		'project_management.write',
		'project_management.manage',
		'project_management.report',
		'project_management.admin'
	],
	'menu_items': [
		{
			'name': 'Projects',
			'endpoint': 'SSPMProjectModelView.list',
			'icon': 'fa-project-diagram',
			'permission': 'project_management.read'
		},
		{
			'name': 'Tasks',
			'endpoint': 'SSPMTaskModelView.list',
			'icon': 'fa-tasks',
			'permission': 'project_management.read'
		},
		{
			'name': 'Milestones',
			'endpoint': 'SSPMMilestoneModelView.list',
			'icon': 'fa-flag',
			'permission': 'project_management.read'
		},
		{
			'name': 'Resources',
			'endpoint': 'SSPMResourceAssignmentModelView.list',
			'icon': 'fa-users',
			'permission': 'project_management.read'
		},
		{
			'name': 'PM Dashboard',
			'endpoint': 'SSPMDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'project_management.read'
		}
	],
	'configuration': {
		'enable_gantt_charts': True,
		'enable_agile_boards': True,
		'default_project_methodology': 'waterfall',
		'auto_task_numbering': True,
		'enable_time_tracking': True,
		'enable_budget_tracking': True,
		'milestone_notification_days': 7
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# No hard dependencies, but warn about useful integrations
	if 'resource_scheduling' not in available_subcapabilities:
		warnings.append("Resource Scheduling integration not available - resource allocation may be manual")
	
	if 'time_expense_tracking' not in available_subcapabilities:
		warnings.append("Time & Expense Tracking integration not available - project costing may be limited")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}