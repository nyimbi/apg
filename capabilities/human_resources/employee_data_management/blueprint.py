"""
Employee Data Management Blueprint

Blueprint registration for Employee Data Management sub-capability views and APIs.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, Any, List

from .views import (
	HREmployeeModelView, HRDepartmentModelView, HRPositionModelView,
	HRSkillModelView, HRCertificationModelView, HREmployeeSkillModelView,
	HREmployeeCertificationModelView, HREmployeeDashboardView,
	HREmploymentHistoryModelView
)


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Employee Data Management sub-capability views"""
	
	# Register model views
	appbuilder.add_view(
		HREmployeeModelView,
		"Employees",
		icon="fa-user",
		category="Human Resources",
		category_icon="fa-users"
	)
	
	appbuilder.add_view(
		HRDepartmentModelView,
		"Departments", 
		icon="fa-building",
		category="Human Resources"
	)
	
	appbuilder.add_view(
		HRPositionModelView,
		"Positions",
		icon="fa-briefcase",
		category="Human Resources"
	)
	
	appbuilder.add_view(
		HRSkillModelView,
		"Skills Catalog",
		icon="fa-cogs",
		category="Human Resources"
	)
	
	appbuilder.add_view(
		HRCertificationModelView,
		"Certifications Catalog",
		icon="fa-certificate",
		category="Human Resources"
	)
	
	appbuilder.add_view(
		HREmployeeSkillModelView,
		"Employee Skills",
		icon="fa-user-cog",
		category="Human Resources"
	)
	
	appbuilder.add_view(
		HREmployeeCertificationModelView,
		"Employee Certifications",
		icon="fa-award",
		category="Human Resources"
	)
	
	appbuilder.add_view(
		HREmploymentHistoryModelView,
		"Employment History",
		icon="fa-history",
		category="Human Resources"
	)
	
	# Register dashboard view
	appbuilder.add_view_no_menu(HREmployeeDashboardView())
	appbuilder.add_link(
		"Employee Dashboard",
		href="/hr/employee_dashboard/",
		icon="fa-dashboard",
		category="Human Resources"
	)
	
	appbuilder.add_link(
		"Organizational Chart",
		href="/hr/employee_dashboard/org_chart",
		icon="fa-sitemap",
		category="Human Resources"
	)
	
	# Register permissions
	register_permissions(appbuilder)


def register_permissions(appbuilder: AppBuilder):
	"""Register Employee Data Management permissions"""
	
	permissions = [
		# Employee permissions
		('can_list', 'HREmployeeModelView'),
		('can_show', 'HREmployeeModelView'),
		('can_add', 'HREmployeeModelView'),
		('can_edit', 'HREmployeeModelView'),
		('can_delete', 'HREmployeeModelView'),
		
		# Department permissions
		('can_list', 'HRDepartmentModelView'),
		('can_show', 'HRDepartmentModelView'),
		('can_add', 'HRDepartmentModelView'),
		('can_edit', 'HRDepartmentModelView'),
		('can_delete', 'HRDepartmentModelView'),
		
		# Position permissions
		('can_list', 'HRPositionModelView'),
		('can_show', 'HRPositionModelView'),
		('can_add', 'HRPositionModelView'),
		('can_edit', 'HRPositionModelView'),
		('can_delete', 'HRPositionModelView'),
		
		# Skills permissions
		('can_list', 'HRSkillModelView'),
		('can_show', 'HRSkillModelView'),
		('can_add', 'HRSkillModelView'),
		('can_edit', 'HRSkillModelView'),
		('can_delete', 'HRSkillModelView'),
		
		# Certifications permissions
		('can_list', 'HRCertificationModelView'),
		('can_show', 'HRCertificationModelView'),
		('can_add', 'HRCertificationModelView'),
		('can_edit', 'HRCertificationModelView'),
		('can_delete', 'HRCertificationModelView'),
		
		# Employee skills permissions
		('can_list', 'HREmployeeSkillModelView'),
		('can_show', 'HREmployeeSkillModelView'),
		('can_add', 'HREmployeeSkillModelView'),
		('can_edit', 'HREmployeeSkillModelView'),
		('can_delete', 'HREmployeeSkillModelView'),
		
		# Employee certifications permissions
		('can_list', 'HREmployeeCertificationModelView'),
		('can_show', 'HREmployeeCertificationModelView'),
		('can_add', 'HREmployeeCertificationModelView'),
		('can_edit', 'HREmployeeCertificationModelView'),
		('can_delete', 'HREmployeeCertificationModelView'),
		
		# Employment history permissions (read-only)
		('can_list', 'HREmploymentHistoryModelView'),
		('can_show', 'HREmploymentHistoryModelView'),
		
		# Dashboard permissions
		('can_index', 'HREmployeeDashboardView'),
		('can_org_chart', 'HREmployeeDashboardView'),
		
		# Custom permissions
		('can_view_sensitive_data', 'HREmployee'),
		('can_manage_org_structure', 'HREmployee'),
		('can_terminate_employee', 'HREmployee'),
		('can_view_salary_info', 'HREmployee'),
		('can_approve_changes', 'HREmployee')
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Employee Data Management"""
	
	return {
		'name': 'Employee Data Management',
		'items': [
			{
				'name': 'Employee Dashboard',
				'href': '/hr/employee_dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_index on HREmployeeDashboardView'
			},
			{
				'name': 'Employees',
				'href': '/hremployeemodelview/list/',
				'icon': 'fa-user',
				'permission': 'can_list on HREmployeeModelView'
			},
			{
				'name': 'Departments',
				'href': '/hrdepartmentmodelview/list/',
				'icon': 'fa-building',
				'permission': 'can_list on HRDepartmentModelView'
			},
			{
				'name': 'Positions',
				'href': '/hrpositionmodelview/list/',
				'icon': 'fa-briefcase',
				'permission': 'can_list on HRPositionModelView'
			},
			{
				'name': 'Organizational Chart',
				'href': '/hr/employee_dashboard/org_chart',
				'icon': 'fa-sitemap',
				'permission': 'can_org_chart on HREmployeeDashboardView'
			},
			{
				'name': 'Skills Management',
				'href': '/hrskillmodelview/list/',
				'icon': 'fa-cogs',
				'permission': 'can_list on HRSkillModelView'
			},
			{
				'name': 'Certifications',
				'href': '/hrcertificationmodelview/list/',
				'icon': 'fa-certificate',
				'permission': 'can_list on HRCertificationModelView'
			},
			{
				'name': 'Employment History',
				'href': '/hremploymenthistorymodelview/list/',
				'icon': 'fa-history',
				'permission': 'can_list on HREmploymentHistoryModelView'
			}
		]
	}


def create_subcapability_blueprint() -> Blueprint:
	"""Create Flask blueprint for Employee Data Management"""
	
	edm_bp = Blueprint(
		'employee_data_management',
		__name__,
		url_prefix='/hr/edm',
		template_folder='templates',
		static_folder='static'
	)
	
	return edm_bp


def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies for Employee Data Management"""
	
	from . import validate_dependencies
	return validate_dependencies(available_subcapabilities)


def get_subcapability_info() -> Dict[str, Any]:
	"""Get Employee Data Management sub-capability information"""
	
	from . import get_subcapability_info
	return get_subcapability_info()