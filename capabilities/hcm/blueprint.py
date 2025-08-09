"""
Human Resources Capability Blueprint

Main blueprint registration for Human Resources capability and all its sub-capabilities.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import List, Dict, Any

# Import sub-capability blueprint registration functions
from .employee_data_management.blueprint import init_subcapability as init_edm
from .employee_data_management.api import register_api_views as register_edm_api


def register_capability_views(appbuilder: AppBuilder, subcapabilities: List[str] = None):
	"""Register Human Resources views and sub-capabilities with Flask-AppBuilder"""
	
	# If no specific sub-capabilities requested, use all implemented
	if subcapabilities is None:
		subcapabilities = [
			'employee_data_management',
			'payroll',
			'time_attendance',
			'recruitment_onboarding',
			'performance_management',
			'benefits_administration',
			'learning_development'
		]
	
	# Register sub-capabilities
	for subcap in subcapabilities:
		if subcap == 'employee_data_management':
			init_edm(appbuilder)
			register_edm_api(appbuilder)
		elif subcap == 'payroll':
			from .payroll.blueprint import init_subcapability as init_pr
			from .payroll.api import register_api_views as register_pr_api
			init_pr(appbuilder)
			register_pr_api(appbuilder)
		elif subcap == 'time_attendance':
			from .time_attendance.blueprint import init_subcapability as init_ta
			from .time_attendance.api import register_api_views as register_ta_api
			init_ta(appbuilder)
			register_ta_api(appbuilder)
		elif subcap == 'recruitment_onboarding':
			from .recruitment_onboarding.blueprint import init_subcapability as init_ro
			from .recruitment_onboarding.api import register_api_views as register_ro_api
			init_ro(appbuilder)
			register_ro_api(appbuilder)
		elif subcap == 'performance_management':
			from .performance_management.blueprint import init_subcapability as init_pm
			from .performance_management.api import register_api_views as register_pm_api
			init_pm(appbuilder)
			register_pm_api(appbuilder)
		elif subcap == 'benefits_administration':
			from .benefits_administration.blueprint import init_subcapability as init_ba
			from .benefits_administration.api import register_api_views as register_ba_api
			init_ba(appbuilder)
			register_ba_api(appbuilder)
		elif subcap == 'learning_development':
			from .learning_development.blueprint import init_subcapability as init_ld
			from .learning_development.api import register_api_views as register_ld_api
			init_ld(appbuilder)
			register_ld_api(appbuilder)
	
	# Create main capability dashboard if needed
	create_capability_dashboard(appbuilder)


def create_capability_dashboard(appbuilder: AppBuilder):
	"""Create main Human Resources dashboard"""
	
	from flask_appbuilder import BaseView, expose, has_access
	
	class HumanResourcesDashboardView(BaseView):
		"""Main Human Resources Dashboard"""
		
		route_base = "/human_resources/dashboard"
		default_view = 'index'
		
		@expose('/')
		@has_access
		def index(self):
			"""Display Human Resources dashboard"""
			
			# Get summary data from all active sub-capabilities
			dashboard_data = self._get_dashboard_data()
			
			return self.render_template(
				'human_resources_dashboard.html',
				dashboard_data=dashboard_data,
				title="Human Resources Dashboard"
			)
		
		def _get_dashboard_data(self) -> Dict[str, Any]:
			"""Get dashboard data from all sub-capabilities"""
			
			data = {
				'subcapabilities': [],
				'summary': {}
			}
			
			# Employee Data Management summary
			try:
				from .employee_data_management.service import EmployeeDataManagementService
				edm_service = EmployeeDataManagementService(self.get_tenant_id())
				
				# Get key employee metrics
				total_employees = edm_service.get_employee_count()
				active_employees = edm_service.get_employee_count(active_only=True)
				new_hires = edm_service.get_new_hires_count(days=30)
				upcoming_reviews = edm_service.get_upcoming_reviews_count(days=30)
				
				data['subcapabilities'].append({
					'name': 'Employee Data Management',
					'status': 'active',
					'metrics': {
						'total_employees': total_employees,
						'active_employees': active_employees,
						'new_hires_30_days': new_hires,
						'upcoming_reviews': upcoming_reviews
					}
				})
				
			except Exception as e:
				print(f"Error getting EDM dashboard data: {e}")
			
			# Payroll summary
			try:
				from .payroll.service import PayrollService
				payroll_service = PayrollService(self.get_tenant_id())
				
				# Get key payroll metrics
				current_period = payroll_service.get_current_payroll_period()
				pending_payrolls = payroll_service.get_pending_payroll_count()
				total_payroll_ytd = payroll_service.get_total_payroll_ytd()
				
				data['subcapabilities'].append({
					'name': 'Payroll',
					'status': 'active', 
					'metrics': {
						'current_period': current_period.period_name if current_period else 'N/A',
						'pending_payrolls': pending_payrolls,
						'total_payroll_ytd': float(total_payroll_ytd) if total_payroll_ytd else 0.0
					}
				})
				
			except Exception as e:
				print(f"Error getting Payroll dashboard data: {e}")
			
			return data
		
		def get_tenant_id(self) -> str:
			"""Get current tenant ID"""
			# TODO: Implement tenant resolution
			return "default_tenant"
	
	# Register the dashboard view
	appbuilder.add_view_no_menu(HumanResourcesDashboardView())
	appbuilder.add_link(
		"Human Resources Dashboard",
		href="/human_resources/dashboard/",
		icon="fa-dashboard",
		category="Human Resources"
	)


def create_capability_blueprint() -> Blueprint:
	"""Create Flask blueprint for Human Resources capability"""
	
	hr_bp = Blueprint(
		'human_resources',
		__name__,
		url_prefix='/human_resources',
		template_folder='templates',
		static_folder='static'
	)
	
	return hr_bp


def register_capability_permissions(appbuilder: AppBuilder):
	"""Register Human Resources capability-level permissions"""
	
	permissions = [
		# Capability-level permissions
		('can_access', 'HumanResources'),
		('can_view_dashboard', 'HumanResources'),
		
		# Cross-sub-capability permissions
		('can_view_employee_reports', 'HumanResources'),
		('can_manage_employee_data', 'HumanResources'),
		('can_process_payroll', 'HumanResources'),
		('can_manage_benefits', 'HumanResources'),
		('can_conduct_reviews', 'HumanResources'),
		('can_manage_training', 'HumanResources'),
		('can_manage_recruitment', 'HumanResources'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_capability_menu_structure(subcapabilities: List[str] = None) -> Dict[str, Any]:
	"""Get complete menu structure for Human Resources capability"""
	
	if subcapabilities is None:
		subcapabilities = ['employee_data_management']
	
	menu = {
		'name': 'Human Resources',
		'icon': 'fa-users',
		'items': [
			{
				'name': 'Dashboard',
				'href': '/human_resources/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_view_dashboard on HumanResources'
			}
		]
	}
	
	# Add sub-capability menu items
	if 'employee_data_management' in subcapabilities:
		from .employee_data_management.blueprint import get_menu_structure as get_edm_menu
		edm_menu = get_edm_menu()
		menu['items'].extend(edm_menu['items'])
	
	# Add other sub-capabilities as they're implemented
	
	return menu


def validate_subcapability_dependencies(subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate that sub-capability dependencies are met"""
	
	from . import validate_composition
	return validate_composition(subcapabilities)


def init_capability(appbuilder: AppBuilder, subcapabilities: List[str] = None):
	"""Initialize Human Resources capability with specified sub-capabilities"""
	
	# Validate dependencies
	validation = validate_subcapability_dependencies(subcapabilities or ['employee_data_management'])
	
	if not validation['valid']:
		raise ValueError(f"Invalid sub-capability composition: {validation['errors']}")
	
	# Register views and permissions
	register_capability_views(appbuilder, subcapabilities)
	register_capability_permissions(appbuilder)
	
	# Log warnings if any
	if validation['warnings']:
		for warning in validation['warnings']:
			print(f"Warning: {warning}")
	
	print(f"Human Resources capability initialized with sub-capabilities: {subcapabilities}")


def get_capability_info() -> Dict[str, Any]:
	"""Get Human Resources capability information"""
	
	from . import get_capability_info
	return get_capability_info()


def get_available_subcapabilities() -> List[str]:
	"""Get list of available sub-capabilities"""
	
	from . import get_subcapabilities
	return get_subcapabilities()