"""
Budgeting & Forecasting Blueprint

Flask blueprint registration for the Budgeting & Forecasting sub-capability
including view registration and menu setup.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	CFBFBudgetScenarioModelView,
	CFBFTemplateModelView,
	CFBFDriversModelView,
	CFBFBudgetModelView,
	CFBFBudgetLineModelView,
	CFBFForecastModelView,
	CFBFForecastLineModelView,
	CFBFActualVsBudgetView,
	CFBFApprovalModelView,
	CFBFAllocationModelView,
	CFBFDashboardView,
	CFBFScenarioComparisonView
)

# Create blueprint
budgeting_forecasting_bp = Blueprint(
	'budgeting_forecasting',
	__name__,
	url_prefix='/core_financials/budgeting_forecasting',
	template_folder='templates',
	static_folder='static'
)


def register_views(appbuilder: AppBuilder):
	"""Register all views with Flask-AppBuilder"""
	
	# Dashboard Views (register first for menu ordering)
	appbuilder.add_view(
		CFBFDashboardView,
		"BF Dashboard",
		icon="fa-dashboard",
		category="Budgeting & Forecasting",
		category_icon="fa-calculator"
	)
	
	# Core Budget Views
	appbuilder.add_view(
		CFBFBudgetModelView,
		"Budgets",
		icon="fa-calculator",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFBudgetLineModelView,
		"Budget Lines",
		icon="fa-list",
		category="Budgeting & Forecasting"
	)
	
	# Scenario and Planning Views
	appbuilder.add_view(
		CFBFBudgetScenarioModelView,
		"Budget Scenarios",
		icon="fa-sitemap",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFScenarioComparisonView,
		"Scenario Comparison",
		icon="fa-balance-scale",
		category="Budgeting & Forecasting"
	)
	
	# Forecasting Views
	appbuilder.add_view(
		CFBFForecastModelView,
		"Forecasts",
		icon="fa-chart-line",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFForecastLineModelView,
		"Forecast Lines",
		icon="fa-chart-area",
		category="Budgeting & Forecasting"
	)
	
	# Analysis Views
	appbuilder.add_view(
		CFBFActualVsBudgetView,
		"Variance Analysis",
		icon="fa-chart-bar",
		category="Budgeting & Forecasting"
	)
	
	# Configuration Views
	appbuilder.add_view(
		CFBFTemplateModelView,
		"Budget Templates",
		icon="fa-copy",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFDriversModelView,
		"Budget Drivers",
		icon="fa-cogs",
		category="Budgeting & Forecasting"
	)
	
	appbuilder.add_view(
		CFBFAllocationModelView,
		"Budget Allocations",
		icon="fa-share-alt",
		category="Budgeting & Forecasting"
	)
	
	# Workflow Views
	appbuilder.add_view(
		CFBFApprovalModelView,
		"Budget Approvals",
		icon="fa-check-circle",
		category="Budgeting & Forecasting"
	)


def register_permissions(appbuilder: AppBuilder):
	"""Register permissions for the sub-capability"""
	
	# Base permissions are automatically created by Flask-AppBuilder
	# Custom permissions can be added here if needed
	
	permission_mappings = {
		# Budget permissions
		'can_create_budget': 'Create new budgets',
		'can_submit_budget': 'Submit budgets for approval',
		'can_approve_budget': 'Approve budgets',
		'can_lock_budget': 'Lock approved budgets',
		'can_copy_budget': 'Copy budgets to new periods',
		
		# Forecast permissions
		'can_create_forecast': 'Create new forecasts',
		'can_generate_forecast': 'Generate forecast calculations',
		'can_approve_forecast': 'Approve forecasts',
		
		# Analysis permissions
		'can_run_variance_analysis': 'Run variance analysis',
		'can_view_variance_trends': 'View variance trend analysis',
		'can_export_variance_reports': 'Export variance reports',
		
		# Configuration permissions
		'can_manage_scenarios': 'Manage budget scenarios',
		'can_manage_templates': 'Manage budget templates',
		'can_manage_drivers': 'Manage budget drivers',
		'can_manage_allocations': 'Manage budget allocations',
		
		# Administrative permissions
		'can_admin_budgets': 'Full budget administration',
		'can_view_all_budgets': 'View all tenant budgets',
		'can_delete_budgets': 'Delete budgets (admin only)'
	}
	
	# Register custom permissions
	for permission_name, description in permission_mappings.items():
		# Implementation would register these permissions with the security manager
		pass


def setup_menu_structure(appbuilder: AppBuilder):
	"""Setup custom menu structure for budgeting & forecasting"""
	
	# The menu structure is primarily handled by the view registration
	# Additional custom menu items can be added here if needed
	
	# Example of adding a custom menu separator
	# appbuilder.add_separator("Budgeting & Forecasting")
	
	# Example of adding a link to external budgeting resources
	# appbuilder.add_link(
	#     "Budget Help",
	#     href="/static/help/budgeting_guide.html",
	#     icon="fa-question-circle",
	#     category="Budgeting & Forecasting"
	# )
	
	pass


def register_api_routes():
	"""Register API routes for the sub-capability"""
	
	# API routes are handled in api.py
	# This function can be used for any additional route registration
	pass


def init_budgeting_forecasting(appbuilder: AppBuilder):
	"""Initialize the Budgeting & Forecasting sub-capability"""
	
	# Register all views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Setup menu structure
	setup_menu_structure(appbuilder)
	
	# Register API routes
	register_api_routes()
	
	# Log initialization
	print("Budgeting & Forecasting sub-capability initialized successfully")


# Blueprint factory function
def create_budgeting_forecasting_blueprint() -> Blueprint:
	"""Create and configure the budgeting & forecasting blueprint"""
	
	# Additional blueprint configuration can be added here
	
	return budgeting_forecasting_bp