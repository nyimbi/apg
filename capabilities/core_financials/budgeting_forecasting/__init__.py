"""
Budgeting & Forecasting Sub-Capability

Provides tools for financial planning, creating budgets, and predicting future financial performance.
Handles budget creation, scenario planning, variance analysis, and forecasting algorithms.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Budgeting & Forecasting',
	'code': 'BF',
	'version': '1.0.0',
	'capability': 'core_financials',
	'description': 'Provides tools for financial planning, creating budgets, and predicting future financial performance.',
	'industry_focus': 'All',
	'dependencies': ['general_ledger'],
	'optional_dependencies': ['accounts_payable', 'accounts_receivable', 'cash_management'],
	'database_tables': [
		'cf_bf_budget',
		'cf_bf_budget_line',
		'cf_bf_budget_scenario',
		'cf_bf_budget_version',
		'cf_bf_forecast',
		'cf_bf_forecast_line',
		'cf_bf_actual_vs_budget',
		'cf_bf_drivers',
		'cf_bf_template',
		'cf_bf_approval',
		'cf_bf_allocation'
	],
	'api_endpoints': [
		'/api/core_financials/bf/budgets',
		'/api/core_financials/bf/budget_lines',
		'/api/core_financials/bf/scenarios',
		'/api/core_financials/bf/versions',
		'/api/core_financials/bf/forecasts',
		'/api/core_financials/bf/forecast_lines',
		'/api/core_financials/bf/variance_analysis',
		'/api/core_financials/bf/drivers',
		'/api/core_financials/bf/templates',
		'/api/core_financials/bf/approvals',
		'/api/core_financials/bf/allocations',
		'/api/core_financials/bf/reports'
	],
	'views': [
		'CFBFBudgetModelView',
		'CFBFBudgetLineModelView',
		'CFBFBudgetScenarioModelView',
		'CFBFBudgetVersionModelView',
		'CFBFForecastModelView',
		'CFBFForecastLineModelView',
		'CFBFActualVsBudgetView',
		'CFBFDriversModelView',
		'CFBFTemplateModelView',
		'CFBFApprovalModelView',
		'CFBFAllocationModelView',
		'CFBFDashboardView',
		'CFBFVarianceAnalysisView',
		'CFBFScenarioComparisonView'
	],
	'permissions': [
		'bf.read',
		'bf.write',
		'bf.create_budget',
		'bf.approve_budget',
		'bf.submit_budget',
		'bf.create_forecast',
		'bf.approve_forecast',
		'bf.create_scenario',
		'bf.manage_templates',
		'bf.manage_drivers',
		'bf.variance_analysis',
		'bf.budget_admin',
		'bf.admin'
	],
	'menu_items': [
		{
			'name': 'BF Dashboard',
			'endpoint': 'CFBFDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'bf.read'
		},
		{
			'name': 'Budgets',
			'endpoint': 'CFBFBudgetModelView.list',
			'icon': 'fa-calculator',
			'permission': 'bf.read'
		},
		{
			'name': 'Budget Scenarios',
			'endpoint': 'CFBFBudgetScenarioModelView.list',
			'icon': 'fa-sitemap',
			'permission': 'bf.read'
		},
		{
			'name': 'Forecasts',
			'endpoint': 'CFBFForecastModelView.list',
			'icon': 'fa-chart-line',
			'permission': 'bf.read'
		},
		{
			'name': 'Variance Analysis',
			'endpoint': 'CFBFVarianceAnalysisView.index',
			'icon': 'fa-chart-bar',
			'permission': 'bf.variance_analysis'
		},
		{
			'name': 'Scenario Comparison',
			'endpoint': 'CFBFScenarioComparisonView.index',
			'icon': 'fa-balance-scale',
			'permission': 'bf.read'
		},
		{
			'name': 'Budget Templates',
			'endpoint': 'CFBFTemplateModelView.list',
			'icon': 'fa-copy',
			'permission': 'bf.manage_templates'
		},
		{
			'name': 'Budget Drivers',
			'endpoint': 'CFBFDriversModelView.list',
			'icon': 'fa-cogs',
			'permission': 'bf.manage_drivers'
		},
		{
			'name': 'Budget Approvals',
			'endpoint': 'CFBFApprovalModelView.list',
			'icon': 'fa-check-circle',
			'permission': 'bf.approve_budget'
		}
	],
	'configuration': {
		'auto_budget_numbering': True,
		'auto_forecast_numbering': True,
		'require_budget_approval': True,
		'require_forecast_approval': False,
		'allow_budget_revisions': True,
		'default_budget_period': 'Annual',  # Annual, Quarterly, Monthly
		'default_forecast_periods': 12,  # months
		'budget_scenarios_enabled': True,
		'driver_based_budgeting': True,
		'allocation_methods': ['Direct', 'Percentage', 'Driver-Based', 'Formula'],
		'variance_threshold_percent': 10.0,  # Alert threshold
		'variance_threshold_amount': 10000.00,  # Alert threshold
		'auto_variance_alerts': True,
		'rolling_forecasts': True,
		'budget_consolidation': True,
		'multi_currency_budgets': True,
		'budget_templates': True,
		'workflow_approvals': True,
		'budget_locking': True,
		'actuals_integration': True,
		'forecast_algorithms': ['Linear', 'Exponential', 'Seasonal', 'Regression'],
		'budget_periods': [
			{'code': 'ANNUAL', 'name': 'Annual', 'months': 12},
			{'code': 'QUARTERLY', 'name': 'Quarterly', 'months': 3},
			{'code': 'MONTHLY', 'name': 'Monthly', 'months': 1}
		]
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
		errors.append("General Ledger sub-capability is required for budget/actual comparisons")
	
	# Check optional dependencies
	if 'accounts_payable' not in available_subcapabilities:
		warnings.append("Accounts Payable integration not available - expense budgeting may be limited")
	
	if 'accounts_receivable' not in available_subcapabilities:
		warnings.append("Accounts Receivable integration not available - revenue forecasting may be limited")
		
	if 'cash_management' not in available_subcapabilities:
		warnings.append("Cash Management integration not available - cash flow forecasting not available")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_budget_categories() -> List[Dict[str, Any]]:
	"""Get default budget categories"""
	return [
		{'code': 'REVENUE', 'name': 'Revenue', 'description': 'Revenue and income budgets', 'type': 'Revenue'},
		{'code': 'SALARIES', 'name': 'Salaries & Wages', 'description': 'Personnel costs', 'type': 'Expense'},
		{'code': 'BENEFITS', 'name': 'Employee Benefits', 'description': 'Benefits and payroll taxes', 'type': 'Expense'},
		{'code': 'MARKETING', 'name': 'Marketing & Advertising', 'description': 'Marketing and promotional expenses', 'type': 'Expense'},
		{'code': 'OPERATIONS', 'name': 'Operations', 'description': 'Operational expenses', 'type': 'Expense'},
		{'code': 'FACILITIES', 'name': 'Facilities', 'description': 'Rent, utilities, maintenance', 'type': 'Expense'},
		{'code': 'TECHNOLOGY', 'name': 'Technology', 'description': 'IT expenses and software', 'type': 'Expense'},
		{'code': 'CAPEX', 'name': 'Capital Expenditure', 'description': 'Capital investments', 'type': 'Asset'},
		{'code': 'OTHER', 'name': 'Other', 'description': 'Other budget items', 'type': 'Other'}
	]

def get_default_budget_drivers() -> List[Dict[str, Any]]:
	"""Get default budget drivers"""
	return [
		{'code': 'HEADCOUNT', 'name': 'Headcount', 'description': 'Number of employees', 'unit': 'Count'},
		{'code': 'SALES_VOLUME', 'name': 'Sales Volume', 'description': 'Units sold', 'unit': 'Units'},
		{'code': 'CUSTOMERS', 'name': 'Customer Count', 'description': 'Number of customers', 'unit': 'Count'},
		{'code': 'SQ_FOOTAGE', 'name': 'Square Footage', 'description': 'Office/facility space', 'unit': 'Sq Ft'},
		{'code': 'INFLATION', 'name': 'Inflation Rate', 'description': 'Annual inflation rate', 'unit': 'Percent'},
		{'code': 'GROWTH_RATE', 'name': 'Growth Rate', 'description': 'Revenue growth rate', 'unit': 'Percent'},
		{'code': 'CAPACITY', 'name': 'Capacity Utilization', 'description': 'Production capacity usage', 'unit': 'Percent'},
		{'code': 'CONVERSION', 'name': 'Conversion Rate', 'description': 'Sales conversion rate', 'unit': 'Percent'}
	]

def get_default_budget_scenarios() -> List[Dict[str, Any]]:
	"""Get default budget scenarios"""
	return [
		{'code': 'BASE', 'name': 'Base Case', 'description': 'Most likely scenario', 'probability': 60.0},
		{'code': 'OPTIMISTIC', 'name': 'Optimistic', 'description': 'Best case scenario', 'probability': 20.0},
		{'code': 'PESSIMISTIC', 'name': 'Pessimistic', 'description': 'Worst case scenario', 'probability': 20.0},
		{'code': 'CONSERVATIVE', 'name': 'Conservative', 'description': 'Conservative estimates', 'probability': 0.0},
		{'code': 'AGGRESSIVE', 'name': 'Aggressive', 'description': 'Aggressive growth targets', 'probability': 0.0}
	]

def get_default_allocation_methods() -> List[Dict[str, Any]]:
	"""Get default allocation methods"""
	return [
		{
			'code': 'DIRECT',
			'name': 'Direct Allocation',
			'description': 'Direct assignment to cost centers',
			'formula': None
		},
		{
			'code': 'PERCENTAGE',
			'name': 'Percentage Allocation',
			'description': 'Allocate based on fixed percentages',
			'formula': 'amount * percentage / 100'
		},
		{
			'code': 'HEADCOUNT',
			'name': 'Headcount-Based',
			'description': 'Allocate based on headcount',
			'formula': 'amount * dept_headcount / total_headcount'
		},
		{
			'code': 'REVENUE',
			'name': 'Revenue-Based',
			'description': 'Allocate based on revenue',
			'formula': 'amount * dept_revenue / total_revenue'
		},
		{
			'code': 'SQUARE_FOOTAGE',
			'name': 'Square Footage',
			'description': 'Allocate based on space usage',
			'formula': 'amount * dept_sqft / total_sqft'
		}
	]

def get_default_gl_account_mappings() -> Dict[str, str]:
	"""Get default GL account mappings for budgeting"""
	return {
		'budget_variance': '5900',      # Budget Variance account
		'forecast_adjustment': '5910',  # Forecast Adjustment account
		'budget_reserve': '2900',       # Budget Reserve/Contingency
		'commitment_control': '2910',   # Budget Commitment Control
		'encumbrance': '2920'           # Budget Encumbrance
	}

def get_variance_analysis_rules() -> List[Dict[str, Any]]:
	"""Get default variance analysis rules"""
	return [
		{
			'name': 'Significant Variance',
			'description': 'Variance exceeds threshold amount or percentage',
			'condition': 'abs(variance_amount) > threshold_amount OR abs(variance_percent) > threshold_percent',
			'alert_level': 'Warning',
			'auto_notify': True
		},
		{
			'name': 'Unfavorable Revenue Variance',
			'description': 'Revenue is significantly below budget',
			'condition': 'account_type = "Revenue" AND variance_amount < -threshold_amount',
			'alert_level': 'Critical',
			'auto_notify': True
		},
		{
			'name': 'Unfavorable Expense Variance',
			'description': 'Expenses are significantly over budget',
			'condition': 'account_type = "Expense" AND variance_amount > threshold_amount',
			'alert_level': 'Warning',
			'auto_notify': True
		},
		{
			'name': 'Budget Overrun',
			'description': 'Cumulative spending exceeds annual budget',
			'condition': 'ytd_actual > annual_budget',
			'alert_level': 'Critical',
			'auto_notify': True
		}
	]