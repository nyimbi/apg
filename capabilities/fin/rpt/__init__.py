"""
Financial Reporting Sub-Capability

Generates comprehensive financial statements (Income Statement, Balance Sheet, 
Cash Flow) and analytical reports from ERP data. Provides consolidation,
regulatory reporting, and customizable financial analysis capabilities.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Financial Reporting',
	'code': 'FR',
	'version': '1.0.0',
	'capability': 'core_financials',
	'description': 'Generates comprehensive financial statements and analytical reports with consolidation capabilities.',
	'industry_focus': 'All',
	'dependencies': ['general_ledger'],
	'optional_dependencies': ['accounts_payable', 'accounts_receivable', 'cash_management', 'fixed_asset_management', 'budgeting_forecasting'],
	'database_tables': [
		'cf_fr_report_template',
		'cf_fr_report_definition',
		'cf_fr_report_line',
		'cf_fr_report_period',
		'cf_fr_report_generation',
		'cf_fr_financial_statement',
		'cf_fr_consolidation',
		'cf_fr_notes',
		'cf_fr_disclosure',
		'cf_fr_analytical_report',
		'cf_fr_report_distribution'
	],
	'api_endpoints': [
		'/api/core_financials/fr/templates',
		'/api/core_financials/fr/statements',
		'/api/core_financials/fr/consolidations',
		'/api/core_financials/fr/reports',
		'/api/core_financials/fr/distributions',
		'/api/core_financials/fr/generate'
	],
	'views': [
		'CFRFReportTemplateModelView',
		'CFRFFinancialStatementModelView',
		'CFRFConsolidationModelView',
		'CFRFAnalyticalReportModelView',
		'CFRFReportGenerationView',
		'CFRFFinancialDashboardView'
	],
	'permissions': [
		'fr.read',
		'fr.write',
		'fr.generate',
		'fr.consolidate',
		'fr.distribute',
		'fr.admin'
	],
	'menu_items': [
		{
			'name': 'Financial Statements',
			'endpoint': 'CFRFFinancialStatementModelView.list',
			'icon': 'fa-file-text-o',
			'permission': 'fr.read'
		},
		{
			'name': 'Report Templates',
			'endpoint': 'CFRFReportTemplateModelView.list',
			'icon': 'fa-file-o',
			'permission': 'fr.read'
		},
		{
			'name': 'Consolidations',
			'endpoint': 'CFRFConsolidationModelView.list',
			'icon': 'fa-sitemap',
			'permission': 'fr.consolidate'
		},
		{
			'name': 'Analytical Reports',
			'endpoint': 'CFRFAnalyticalReportModelView.list',
			'icon': 'fa-line-chart',
			'permission': 'fr.read'
		},
		{
			'name': 'Report Generation',
			'endpoint': 'CFRFReportGenerationView.index',
			'icon': 'fa-play-circle',
			'permission': 'fr.generate'
		},
		{
			'name': 'Financial Dashboard',
			'endpoint': 'CFRFFinancialDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'fr.read'
		}
	],
	'configuration': {
		'default_periods': ['monthly', 'quarterly', 'annually'],
		'consolidation_method': 'full',  # full, proportional, equity
		'currency_translation': 'current_rate',
		'rounding_precision': 2,
		'auto_generate_notes': True,
		'require_approval': True,
		'distribution_formats': ['pdf', 'excel', 'html'],
		'standard_statements': ['balance_sheet', 'income_statement', 'cash_flow', 'equity_changes']
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
		errors.append("General Ledger is required - Financial reporting depends on GL data")
	
	# Check useful optional dependencies
	optional_checks = [
		('accounts_payable', 'AP aging and liability reporting limited'),
		('accounts_receivable', 'AR aging and asset reporting limited'),
		('cash_management', 'Cash flow statement generation limited'),
		('fixed_asset_management', 'Asset depreciation and reporting limited'),
		('budgeting_forecasting', 'Budget vs actual reporting not available')
	]
	
	for subcap, warning in optional_checks:
		if subcap not in available_subcapabilities:
			warnings.append(warning)
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_standard_report_templates() -> List[Dict[str, Any]]:
	"""Get standard financial report templates"""
	return [
		{
			'template_code': 'BS_STANDARD',
			'template_name': 'Standard Balance Sheet',
			'statement_type': 'balance_sheet',
			'format_type': 'comparative',
			'currency_type': 'single',
			'is_system': True,
			'lines': [
				{'line_code': 'ASSETS', 'line_name': 'ASSETS', 'line_type': 'header', 'sort_order': 10},
				{'line_code': 'CURRENT_ASSETS', 'line_name': 'Current Assets', 'line_type': 'subtotal', 'sort_order': 20},
				{'line_code': 'CASH', 'line_name': 'Cash and Cash Equivalents', 'account_filter': '1110*', 'sort_order': 30},
				{'line_code': 'AR', 'line_name': 'Accounts Receivable', 'account_filter': '1120*', 'sort_order': 40},
				{'line_code': 'INVENTORY', 'line_name': 'Inventory', 'account_filter': '1130*', 'sort_order': 50},
				{'line_code': 'FIXED_ASSETS', 'line_name': 'Fixed Assets', 'line_type': 'subtotal', 'sort_order': 60},
				{'line_code': 'PPE', 'line_name': 'Property, Plant & Equipment', 'account_filter': '1200*', 'sort_order': 70},
				{'line_code': 'TOTAL_ASSETS', 'line_name': 'TOTAL ASSETS', 'line_type': 'total', 'sort_order': 100}
			]
		},
		{
			'template_code': 'IS_STANDARD',
			'template_name': 'Standard Income Statement',
			'statement_type': 'income_statement',
			'format_type': 'comparative',
			'currency_type': 'single',
			'is_system': True,
			'lines': [
				{'line_code': 'REVENUE', 'line_name': 'REVENUE', 'line_type': 'header', 'sort_order': 10},
				{'line_code': 'SALES', 'line_name': 'Sales Revenue', 'account_filter': '4100*', 'sort_order': 20},
				{'line_code': 'TOTAL_REV', 'line_name': 'Total Revenue', 'line_type': 'subtotal', 'sort_order': 30},
				{'line_code': 'EXPENSES', 'line_name': 'EXPENSES', 'line_type': 'header', 'sort_order': 40},
				{'line_code': 'COGS', 'line_name': 'Cost of Goods Sold', 'account_filter': '5100*', 'sort_order': 50},
				{'line_code': 'OPEX', 'line_name': 'Operating Expenses', 'account_filter': '5200*', 'sort_order': 60},
				{'line_code': 'NET_INCOME', 'line_name': 'NET INCOME', 'line_type': 'total', 'sort_order': 100}
			]
		},
		{
			'template_code': 'CF_STANDARD',
			'template_name': 'Standard Cash Flow Statement',
			'statement_type': 'cash_flow',
			'format_type': 'indirect',
			'currency_type': 'single',
			'is_system': True,
			'lines': [
				{'line_code': 'OPERATING', 'line_name': 'OPERATING ACTIVITIES', 'line_type': 'header', 'sort_order': 10},
				{'line_code': 'NET_INCOME_CF', 'line_name': 'Net Income', 'source_type': 'calculation', 'sort_order': 20},
				{'line_code': 'DEPRECIATION', 'line_name': 'Depreciation', 'source_type': 'calculation', 'sort_order': 30},
				{'line_code': 'AR_CHANGE', 'line_name': 'Change in Accounts Receivable', 'source_type': 'calculation', 'sort_order': 40},
				{'line_code': 'INVESTING', 'line_name': 'INVESTING ACTIVITIES', 'line_type': 'header', 'sort_order': 50},
				{'line_code': 'FINANCING', 'line_name': 'FINANCING ACTIVITIES', 'line_type': 'header', 'sort_order': 60}
			]
		}
	]

def get_consolidation_methods() -> List[Dict[str, Any]]:
	"""Get available consolidation methods"""
	return [
		{
			'method_code': 'FULL',
			'method_name': 'Full Consolidation',
			'description': 'Combine 100% of subsidiary accounts with elimination entries',
			'ownership_threshold': 50.1
		},
		{
			'method_code': 'PROPORTIONAL',
			'method_name': 'Proportional Consolidation',
			'description': 'Combine proportional share of subsidiary accounts',
			'ownership_threshold': 20.0
		},
		{
			'method_code': 'EQUITY',
			'method_name': 'Equity Method',
			'description': 'Single line investment with equity pickup',
			'ownership_threshold': 20.0
		}
	]