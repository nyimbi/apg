"""
Cost Accounting Sub-Capability

Tracks and analyzes costs associated with products, services, and operations for better 
profitability insights. Provides activity-based costing, job costing, standard costing,
and variance analysis capabilities.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Cost Accounting',
	'code': 'CA',
	'version': '1.0.0',
	'capability': 'core_financials',
	'description': 'Tracks and analyzes costs associated with products, services, and operations for better profitability insights.',
	'industry_focus': 'Manufacturing, Services, Project-based',
	'dependencies': ['general_ledger'],
	'optional_dependencies': ['budgeting_forecasting', 'fixed_asset_management'],
	'database_tables': [
		'cf_ca_cost_center',
		'cf_ca_cost_category',
		'cf_ca_cost_driver',
		'cf_ca_cost_allocation',
		'cf_ca_cost_pool',
		'cf_ca_activity',
		'cf_ca_activity_cost',
		'cf_ca_product_cost',
		'cf_ca_job_cost', 
		'cf_ca_standard_cost',
		'cf_ca_variance_analysis'
	],
	'api_endpoints': [
		'/api/core_financials/ca/cost_centers',
		'/api/core_financials/ca/cost_categories',
		'/api/core_financials/ca/cost_drivers',
		'/api/core_financials/ca/cost_allocations',
		'/api/core_financials/ca/cost_pools',
		'/api/core_financials/ca/activities',
		'/api/core_financials/ca/product_costs',
		'/api/core_financials/ca/job_costs',
		'/api/core_financials/ca/standard_costs',
		'/api/core_financials/ca/variance_analysis',
		'/api/core_financials/ca/reports'
	],
	'views': [
		'CACostCenterModelView',
		'CACostCategoryModelView',
		'CACostDriverModelView',
		'CACostAllocationModelView',
		'CACostPoolModelView',
		'CAActivityModelView',
		'CAProductCostModelView',
		'CAJobCostModelView',
		'CAStandardCostModelView',
		'CAVarianceAnalysisView',
		'CADashboardView'
	],
	'permissions': [
		'ca.read',
		'ca.write',
		'ca.cost_allocation',
		'ca.abc_setup',
		'ca.job_costing',
		'ca.standard_costing',
		'ca.variance_analysis',
		'ca.reports',
		'ca.admin'
	],
	'menu_items': [
		{
			'name': 'Cost Centers',
			'endpoint': 'CACostCenterModelView.list',
			'icon': 'fa-building-o',
			'permission': 'ca.read'
		},
		{
			'name': 'Cost Categories',
			'endpoint': 'CACostCategoryModelView.list',
			'icon': 'fa-tags',
			'permission': 'ca.read'
		},
		{
			'name': 'Cost Allocation',
			'endpoint': 'CACostAllocationModelView.list',
			'icon': 'fa-share-alt',
			'permission': 'ca.cost_allocation'
		},
		{
			'name': 'Activity-Based Costing',
			'endpoint': 'CAActivityModelView.list',
			'icon': 'fa-cogs',
			'permission': 'ca.abc_setup'
		},
		{
			'name': 'Product Costing',
			'endpoint': 'CAProductCostModelView.list',
			'icon': 'fa-cube',
			'permission': 'ca.read'
		},
		{
			'name': 'Job Costing',
			'endpoint': 'CAJobCostModelView.list',
			'icon': 'fa-tasks',
			'permission': 'ca.job_costing'
		},
		{
			'name': 'Standard Costs',
			'endpoint': 'CAStandardCostModelView.list',
			'icon': 'fa-bar-chart',
			'permission': 'ca.standard_costing'
		},
		{
			'name': 'Variance Analysis',
			'endpoint': 'CAVarianceAnalysisView.index',
			'icon': 'fa-line-chart',
			'permission': 'ca.variance_analysis'
		},
		{
			'name': 'CA Dashboard',
			'endpoint': 'CADashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'ca.read'
		}
	],
	'configuration': {
		'default_allocation_method': 'weighted_average',
		'enable_abc_costing': True,
		'enable_job_costing': True,
		'enable_standard_costing': True,
		'variance_threshold_percent': 5.0,
		'auto_allocate_overhead': False,
		'default_currency': 'USD',
		'cost_calculation_frequency': 'monthly'
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# Check required dependencies
	if 'general_ledger' not in available_subcapabilities:
		errors.append("General Ledger is required for cost accounting GL integration")
	
	# Check optional dependencies
	if 'budgeting_forecasting' not in available_subcapabilities:
		warnings.append("Budgeting & Forecasting integration not available - standard cost budgeting limited")
	
	if 'fixed_asset_management' not in available_subcapabilities:
		warnings.append("Fixed Asset Management integration not available - asset-related cost allocation limited")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_cost_categories() -> List[Dict[str, Any]]:
	"""Get default cost categories structure"""
	return [
		# Material Costs
		{
			'category_code': 'MAT',
			'category_name': 'Material Costs',
			'description': 'Raw materials, components, and supplies',
			'cost_type': 'Direct',
			'is_variable': True,
			'gl_account_code': '5100'
		},
		{
			'category_code': 'MAT-DIR',
			'category_name': 'Direct Materials',
			'description': 'Materials directly used in production',
			'cost_type': 'Direct',
			'is_variable': True,
			'parent_category': 'MAT',
			'gl_account_code': '5110'
		},
		{
			'category_code': 'MAT-IND',
			'category_name': 'Indirect Materials',
			'description': 'Materials not directly traceable to products',
			'cost_type': 'Indirect',
			'is_variable': True,
			'parent_category': 'MAT',
			'gl_account_code': '5120'
		},
		
		# Labor Costs
		{
			'category_code': 'LAB',
			'category_name': 'Labor Costs',
			'description': 'Direct and indirect labor costs',
			'cost_type': 'Direct',
			'is_variable': True,
			'gl_account_code': '5200'
		},
		{
			'category_code': 'LAB-DIR',
			'category_name': 'Direct Labor',
			'description': 'Labor directly involved in production',
			'cost_type': 'Direct',
			'is_variable': True,
			'parent_category': 'LAB',
			'gl_account_code': '5210'
		},
		{
			'category_code': 'LAB-IND',
			'category_name': 'Indirect Labor',
			'description': 'Supervisory and support labor',
			'cost_type': 'Indirect',
			'is_variable': False,
			'parent_category': 'LAB',
			'gl_account_code': '5220'
		},
		
		# Overhead Costs
		{
			'category_code': 'OH',
			'category_name': 'Manufacturing Overhead',
			'description': 'All other manufacturing costs',
			'cost_type': 'Indirect',
			'is_variable': False,
			'gl_account_code': '5300'
		},
		{
			'category_code': 'OH-VAR',
			'category_name': 'Variable Overhead',
			'description': 'Overhead costs that vary with production',
			'cost_type': 'Indirect',
			'is_variable': True,
			'parent_category': 'OH',
			'gl_account_code': '5310'
		},
		{
			'category_code': 'OH-FIX',
			'category_name': 'Fixed Overhead',
			'description': 'Overhead costs that remain constant',
			'cost_type': 'Indirect',
			'is_variable': False,
			'parent_category': 'OH',
			'gl_account_code': '5320'
		},
		
		# Administrative Costs
		{
			'category_code': 'ADM',
			'category_name': 'Administrative Costs',
			'description': 'General administrative expenses',
			'cost_type': 'Period',
			'is_variable': False,
			'gl_account_code': '5400'
		},
		
		# Selling Costs
		{
			'category_code': 'SEL',
			'category_name': 'Selling Costs',
			'description': 'Sales and marketing expenses',
			'cost_type': 'Period',
			'is_variable': True,
			'gl_account_code': '5500'
		}
	]

def get_default_cost_drivers() -> List[Dict[str, Any]]:
	"""Get default cost driver definitions"""
	return [
		{
			'driver_code': 'MACH_HRS',
			'driver_name': 'Machine Hours',
			'description': 'Hours of machine operation',
			'unit_of_measure': 'Hours',
			'driver_type': 'Volume',
			'is_active': True
		},
		{
			'driver_code': 'LAB_HRS',
			'driver_name': 'Labor Hours',
			'description': 'Direct labor hours worked',
			'unit_of_measure': 'Hours',
			'driver_type': 'Volume',
			'is_active': True
		},
		{
			'driver_code': 'UNITS_PROD',
			'driver_name': 'Units Produced',
			'description': 'Number of units produced',
			'unit_of_measure': 'Units',
			'driver_type': 'Volume',
			'is_active': True
		},
		{
			'driver_code': 'SETUPS',
			'driver_name': 'Number of Setups',
			'description': 'Machine and process setups',
			'unit_of_measure': 'Setups',
			'driver_type': 'Activity',
			'is_active': True
		},
		{
			'driver_code': 'INSPECTIONS',
			'driver_name': 'Quality Inspections',
			'description': 'Number of quality inspections',
			'unit_of_measure': 'Inspections',
			'driver_type': 'Activity',
			'is_active': True
		},
		{
			'driver_code': 'ORDERS',
			'driver_name': 'Number of Orders',
			'description': 'Customer orders processed',
			'unit_of_measure': 'Orders',
			'driver_type': 'Transaction',
			'is_active': True
		},
		{
			'driver_code': 'SQFT',
			'driver_name': 'Square Footage',
			'description': 'Floor space occupied',
			'unit_of_measure': 'Sq Ft',
			'driver_type': 'Facility',
			'is_active': True
		}
	]

def get_default_activities() -> List[Dict[str, Any]]:
	"""Get default activity definitions for ABC"""
	return [
		{
			'activity_code': 'PROD',
			'activity_name': 'Production',
			'description': 'Direct manufacturing activities',
			'activity_type': 'Primary',
			'cost_pool_type': 'Production',
			'primary_driver': 'MACH_HRS'
		},
		{
			'activity_code': 'SETUP',
			'activity_name': 'Machine Setup',
			'description': 'Setting up machines for production runs',
			'activity_type': 'Support',
			'cost_pool_type': 'Setup',
			'primary_driver': 'SETUPS'  
		},
		{
			'activity_code': 'QC',
			'activity_name': 'Quality Control',
			'description': 'Product quality testing and inspection',
			'activity_type': 'Support',
			'cost_pool_type': 'Quality',
			'primary_driver': 'INSPECTIONS'
		},
		{
			'activity_code': 'ORDER_PROC',
			'activity_name': 'Order Processing',
			'description': 'Processing customer orders',
			'activity_type': 'Support',
			'cost_pool_type': 'Administrative',
			'primary_driver': 'ORDERS'
		},
		{
			'activity_code': 'MAINT',
			'activity_name': 'Equipment Maintenance',
			'description': 'Maintaining production equipment',
			'activity_type': 'Support',
			'cost_pool_type': 'Maintenance',
			'primary_driver': 'MACH_HRS'
		},
		{
			'activity_code': 'FACILITY',
			'activity_name': 'Facility Management',
			'description': 'Managing building and utilities',
			'activity_type': 'Sustaining',
			'cost_pool_type': 'Facility',
			'primary_driver': 'SQFT'
		}
	]