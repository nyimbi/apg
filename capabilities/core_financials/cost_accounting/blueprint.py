"""
Cost Accounting Blueprint

Flask blueprint registration for Cost Accounting sub-capability.
Registers all views, API endpoints, and URL routes.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	CACostCenterModelView, CACostCategoryModelView, CACostDriverModelView,
	CACostAllocationModelView, CACostPoolModelView, CAActivityModelView,
	CAProductCostModelView, CAJobCostModelView, CAStandardCostModelView,
	CAVarianceAnalysisView, CADashboardView
)


def register_views(appbuilder: AppBuilder):
	"""Register Cost Accounting views with Flask-AppBuilder"""
	
	# Cost Center Management
	appbuilder.add_view(
		CACostCenterModelView,
		"Cost Centers",
		icon="fa-building-o",
		category="Cost Accounting",
		category_icon="fa-calculator"
	)
	
	appbuilder.add_view(
		CACostCategoryModelView,
		"Cost Categories",
		icon="fa-tags",
		category="Cost Accounting"
	)
	
	appbuilder.add_view(
		CACostDriverModelView,
		"Cost Drivers",
		icon="fa-tachometer",
		category="Cost Accounting"
	)
	
	# Cost Allocation
	appbuilder.add_view(
		CACostAllocationModelView,
		"Cost Allocation Rules",
		icon="fa-share-alt",
		category="Cost Accounting"
	)
	
	# Activity-Based Costing
	appbuilder.add_view(
		CACostPoolModelView,
		"Cost Pools",
		icon="fa-circle",
		category="Cost Accounting"
	)
	
	appbuilder.add_view(
		CAActivityModelView,
		"Activities (ABC)",
		icon="fa-cogs",
		category="Cost Accounting"
	)
	
	# Product and Job Costing
	appbuilder.add_view(
		CAProductCostModelView,
		"Product Costing",
		icon="fa-cube",
		category="Cost Accounting"
	)
	
	appbuilder.add_view(
		CAJobCostModelView,
		"Job Costing",
		icon="fa-tasks",
		category="Cost Accounting"
	)
	
	# Standard Costing and Variance Analysis
	appbuilder.add_view(
		CAStandardCostModelView,
		"Standard Costs",
		icon="fa-bar-chart",
		category="Cost Accounting"
	)
	
	# Custom Views (no menu items, accessed via links)
	appbuilder.add_view_no_menu(CAVarianceAnalysisView())
	appbuilder.add_link(
		"Variance Analysis",
		href="/ca/variance_analysis/",
		icon="fa-line-chart",
		category="Cost Accounting"
	)
	
	# Dashboard
	appbuilder.add_view_no_menu(CADashboardView())
	appbuilder.add_link(
		"CA Dashboard",
		href="/ca/dashboard/",
		icon="fa-dashboard",
		category="Cost Accounting"
	)


def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for Cost Accounting"""
	
	ca_bp = Blueprint(
		'cost_accounting',
		__name__,
		url_prefix='/ca',
		template_folder='templates',
		static_folder='static'
	)
	
	return ca_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register Cost Accounting permissions"""
	
	permissions = [
		# Cost Center permissions
		('can_list', 'CACostCenterModelView'),
		('can_show', 'CACostCenterModelView'),
		('can_add', 'CACostCenterModelView'),
		('can_edit', 'CACostCenterModelView'),
		('can_delete', 'CACostCenterModelView'),
		('can_hierarchy', 'CACostCenterModelView'),
		('can_budget_variance', 'CACostCenterModelView'),
		
		# Cost Category permissions
		('can_list', 'CACostCategoryModelView'),
		('can_show', 'CACostCategoryModelView'),
		('can_add', 'CACostCategoryModelView'),
		('can_edit', 'CACostCategoryModelView'),
		('can_delete', 'CACostCategoryModelView'),
		
		# Cost Driver permissions
		('can_list', 'CACostDriverModelView'),
		('can_show', 'CACostDriverModelView'),
		('can_add', 'CACostDriverModelView'),
		('can_edit', 'CACostDriverModelView'),
		('can_delete', 'CACostDriverModelView'),
		
		# Cost Allocation permissions
		('can_list', 'CACostAllocationModelView'),
		('can_show', 'CACostAllocationModelView'),
		('can_add', 'CACostAllocationModelView'),
		('can_edit', 'CACostAllocationModelView'),
		('can_delete', 'CACostAllocationModelView'),
		('can_execute_allocation', 'CACostAllocationModelView'),
		
		# Cost Pool permissions
		('can_list', 'CACostPoolModelView'),
		('can_show', 'CACostPoolModelView'),
		('can_add', 'CACostPoolModelView'),
		('can_edit', 'CACostPoolModelView'),
		('can_delete', 'CACostPoolModelView'),
		('can_calculate_rates', 'CACostPoolModelView'),
		
		# Activity permissions
		('can_list', 'CAActivityModelView'),
		('can_show', 'CAActivityModelView'),
		('can_add', 'CAActivityModelView'),
		('can_edit', 'CAActivityModelView'),
		('can_delete', 'CAActivityModelView'),
		
		# Product Cost permissions
		('can_list', 'CAProductCostModelView'),
		('can_show', 'CAProductCostModelView'),
		('can_add', 'CAProductCostModelView'),
		('can_edit', 'CAProductCostModelView'),
		('can_delete', 'CAProductCostModelView'),
		('can_cost_breakdown', 'CAProductCostModelView'),
		
		# Job Cost permissions
		('can_list', 'CAJobCostModelView'),
		('can_show', 'CAJobCostModelView'),
		('can_add', 'CAJobCostModelView'),
		('can_edit', 'CAJobCostModelView'),
		('can_delete', 'CAJobCostModelView'),
		('can_job_profitability', 'CAJobCostModelView'),
		('can_update_costs', 'CAJobCostModelView'),
		
		# Standard Cost permissions
		('can_list', 'CAStandardCostModelView'),
		('can_show', 'CAStandardCostModelView'),
		('can_add', 'CAStandardCostModelView'),
		('can_edit', 'CAStandardCostModelView'),
		('can_delete', 'CAStandardCostModelView'),
		
		# Variance Analysis permissions
		('can_index', 'CAVarianceAnalysisView'),
		('can_perform_analysis', 'CAVarianceAnalysisView'),
		('can_show_analysis', 'CAVarianceAnalysisView'),
		('can_period_report', 'CAVarianceAnalysisView'),
		
		# Dashboard permissions
		('can_index', 'CADashboardView'),
		('can_abc_analysis', 'CADashboardView'),
		('can_job_summary', 'CADashboardView'),
		('can_cost_center_performance', 'CADashboardView'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for Cost Accounting"""
	
	return {
		'name': 'Cost Accounting',
		'icon': 'fa-calculator',
		'items': [
			{
				'name': 'CA Dashboard',
				'href': '/ca/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_index on CADashboardView'
			},
			{
				'name': 'Cost Centers',
				'href': '/cacostcentermodelview/list/',
				'icon': 'fa-building-o',
				'permission': 'can_list on CACostCenterModelView'
			},
			{
				'name': 'Cost Categories',
				'href': '/cacostcategorymodelview/list/',
				'icon': 'fa-tags',
				'permission': 'can_list on CACostCategoryModelView'
			},
			{
				'name': 'Cost Drivers',
				'href': '/cacostdrivermodelview/list/',
				'icon': 'fa-tachometer',
				'permission': 'can_list on CACostDriverModelView'
			},
			{
				'name': 'Cost Allocation Rules',
				'href': '/cacostallocationmodelview/list/',
				'icon': 'fa-share-alt',
				'permission': 'can_list on CACostAllocationModelView'
			},
			{
				'name': 'Cost Pools',
				'href': '/cacostpoolmodelview/list/',
				'icon': 'fa-circle',
				'permission': 'can_list on CACostPoolModelView'
			},
			{
				'name': 'Activities (ABC)',
				'href': '/caactivitymodelview/list/',
				'icon': 'fa-cogs',
				'permission': 'can_list on CAActivityModelView'
			},
			{
				'name': 'Product Costing',
				'href': '/caproductcostmodelview/list/',
				'icon': 'fa-cube',
				'permission': 'can_list on CAProductCostModelView'
			},
			{
				'name': 'Job Costing',
				'href': '/cajobcostmodelview/list/',
				'icon': 'fa-tasks',
				'permission': 'can_list on CAJobCostModelView'
			},
			{
				'name': 'Standard Costs',
				'href': '/castandardcostmodelview/list/',
				'icon': 'fa-bar-chart',
				'permission': 'can_list on CAStandardCostModelView'
			},
			{
				'name': 'Variance Analysis',
				'href': '/ca/variance_analysis/',
				'icon': 'fa-line-chart',
				'permission': 'can_index on CAVarianceAnalysisView'
			}
		]
	}


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Cost Accounting sub-capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default Cost Accounting data if needed"""
	
	from .models import CFCACostCategory, CFCACostDriver, CFCAActivity
	from ...auth_rbac.models import db
	from . import get_default_cost_categories, get_default_cost_drivers, get_default_activities
	
	try:
		# Create default cost categories if they don't exist
		existing_categories = CFCACostCategory.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_categories == 0:
			default_categories = get_default_cost_categories()
			
			# Create parent categories first
			for cat_data in default_categories:
				if 'parent_category' not in cat_data:
					category = CFCACostCategory(
						tenant_id='default_tenant',
						category_code=cat_data['category_code'],
						category_name=cat_data['category_name'],
						description=cat_data['description'],
						cost_type=cat_data['cost_type'],
						cost_behavior='Variable' if cat_data['is_variable'] else 'Fixed',
						cost_nature=cat_data.get('cost_nature', 'General'),
						is_variable=cat_data['is_variable'],
						is_traceable=cat_data.get('is_traceable', True),
						is_controllable=cat_data.get('is_controllable', True),
						gl_account_code=cat_data['gl_account_code'],
						effective_date=datetime.now().date(),
						level=0
					)
					db.session.add(category)
			
			# Commit parent categories first
			db.session.commit()
			
			# Create child categories
			for cat_data in default_categories:
				if 'parent_category' in cat_data:
					parent = CFCACostCategory.query.filter_by(
						tenant_id='default_tenant',
						category_code=cat_data['parent_category']
					).first()
					
					if parent:
						category = CFCACostCategory(
							tenant_id='default_tenant',
							category_code=cat_data['category_code'],
							category_name=cat_data['category_name'],
							description=cat_data['description'],
							cost_type=cat_data['cost_type'],
							cost_behavior='Variable' if cat_data['is_variable'] else 'Fixed',
							cost_nature=cat_data.get('cost_nature', 'General'),
							is_variable=cat_data['is_variable'],
							is_traceable=cat_data.get('is_traceable', True),
							is_controllable=cat_data.get('is_controllable', True),
							gl_account_code=cat_data['gl_account_code'],
							parent_category_id=parent.category_id,
							effective_date=datetime.now().date(),
							level=parent.level + 1
						)
						db.session.add(category)
			
			db.session.commit()
			print("Default cost categories created")
		
		# Create default cost drivers if they don't exist
		existing_drivers = CFCACostDriver.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_drivers == 0:
			default_drivers = get_default_cost_drivers()
			
			for driver_data in default_drivers:
				driver = CFCACostDriver(
					tenant_id='default_tenant',
					driver_code=driver_data['driver_code'],
					driver_name=driver_data['driver_name'],
					description=driver_data['description'],
					unit_of_measure=driver_data['unit_of_measure'],
					driver_type=driver_data['driver_type'],
					is_volume_based=driver_data.get('driver_type') == 'Volume',
					is_activity_based=driver_data.get('driver_type') == 'Activity',
					requires_measurement=True,
					effective_date=datetime.now().date()
				)
				db.session.add(driver)
			
			db.session.commit()
			print("Default cost drivers created")
		
		# Create default activities if they don't exist
		existing_activities = CFCAActivity.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_activities == 0:
			default_activities = get_default_activities()
			
			for activity_data in default_activities:
				# Find the primary driver
				primary_driver = CFCACostDriver.query.filter_by(
					tenant_id='default_tenant',
					driver_code=activity_data['primary_driver']
				).first()
				
				activity = CFCAActivity(
					tenant_id='default_tenant',
					activity_code=activity_data['activity_code'],
					activity_name=activity_data['activity_name'],
					description=activity_data['description'],
					activity_type=activity_data['activity_type'],
					value_category='Value-Added',
					primary_driver_id=primary_driver.driver_id if primary_driver else None,
					effective_date=datetime.now().date(),
					is_value_added=True
				)
				db.session.add(activity)
			
			db.session.commit()
			print("Default activities created")
			
	except Exception as e:
		print(f"Error initializing default Cost Accounting data: {e}")
		db.session.rollback()


# Import datetime for default data initialization
from datetime import datetime