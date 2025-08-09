"""
General Ledger Sub-Capability

Central repository for all financial transactions, providing a complete financial picture.
Manages chart of accounts, journal entries, and general ledger reporting.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'General Ledger',
	'code': 'GL',
	'version': '1.0.0',
	'capability': 'core_financials',
	'description': 'Central repository for all financial transactions, providing a complete financial picture.',
	'industry_focus': 'All',
	'dependencies': [],
	'optional_dependencies': ['accounts_payable', 'accounts_receivable', 'cost_accounting'],
	'database_tables': [
		'cf_gl_account',
		'cf_gl_account_type', 
		'cf_gl_journal_entry',
		'cf_gl_journal_line',
		'cf_gl_period',
		'cf_gl_posting',
		'cf_gl_trial_balance'
	],
	'api_endpoints': [
		'/api/core_financials/gl/accounts',
		'/api/core_financials/gl/journals', 
		'/api/core_financials/gl/postings',
		'/api/core_financials/gl/trial_balance',
		'/api/core_financials/gl/reports'
	],
	'views': [
		'GLAccountModelView',
		'GLJournalEntryModelView', 
		'GLPostingModelView',
		'GLTrialBalanceView',
		'GLDashboardView'
	],
	'permissions': [
		'gl.read',
		'gl.write', 
		'gl.post',
		'gl.reverse',
		'gl.close_period',
		'gl.admin'
	],
	'menu_items': [
		{
			'name': 'Chart of Accounts',
			'endpoint': 'GLAccountModelView.list',
			'icon': 'fa-list-alt',
			'permission': 'gl.read'
		},
		{
			'name': 'Journal Entries', 
			'endpoint': 'GLJournalEntryModelView.list',
			'icon': 'fa-book',
			'permission': 'gl.read'
		},
		{
			'name': 'Trial Balance',
			'endpoint': 'GLTrialBalanceView.index', 
			'icon': 'fa-balance-scale',
			'permission': 'gl.read'
		},
		{
			'name': 'GL Dashboard',
			'endpoint': 'GLDashboardView.index',
			'icon': 'fa-dashboard', 
			'permission': 'gl.read'
		}
	],
	'configuration': {
		'auto_numbering': True,
		'require_approval': True,
		'allow_future_dates': False,
		'fiscal_year_start': 1,  # January
		'default_currency': 'USD',
		'multi_currency': False
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# General Ledger has no hard dependencies
	# But warn about useful optional dependencies
	if 'cash_management' not in available_subcapabilities:
		warnings.append("Cash Management integration not available - manual cash postings required")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_chart_of_accounts() -> List[Dict[str, Any]]:
	"""Get default chart of accounts structure"""
	return [
		# Assets
		{'code': '1000', 'name': 'Assets', 'type': 'Asset', 'parent': None},
		{'code': '1100', 'name': 'Current Assets', 'type': 'Asset', 'parent': '1000'},
		{'code': '1110', 'name': 'Cash and Cash Equivalents', 'type': 'Asset', 'parent': '1100'},
		{'code': '1120', 'name': 'Accounts Receivable', 'type': 'Asset', 'parent': '1100'},
		{'code': '1130', 'name': 'Inventory', 'type': 'Asset', 'parent': '1100'},
		{'code': '1200', 'name': 'Fixed Assets', 'type': 'Asset', 'parent': '1000'},
		
		# Liabilities  
		{'code': '2000', 'name': 'Liabilities', 'type': 'Liability', 'parent': None},
		{'code': '2100', 'name': 'Current Liabilities', 'type': 'Liability', 'parent': '2000'},
		{'code': '2110', 'name': 'Accounts Payable', 'type': 'Liability', 'parent': '2100'},
		{'code': '2120', 'name': 'Accrued Expenses', 'type': 'Liability', 'parent': '2100'},
		
		# Equity
		{'code': '3000', 'name': 'Equity', 'type': 'Equity', 'parent': None},
		{'code': '3100', 'name': 'Retained Earnings', 'type': 'Equity', 'parent': '3000'},
		
		# Revenue
		{'code': '4000', 'name': 'Revenue', 'type': 'Revenue', 'parent': None},
		{'code': '4100', 'name': 'Sales Revenue', 'type': 'Revenue', 'parent': '4000'},
		
		# Expenses
		{'code': '5000', 'name': 'Expenses', 'type': 'Expense', 'parent': None},
		{'code': '5100', 'name': 'Cost of Goods Sold', 'type': 'Expense', 'parent': '5000'},
		{'code': '5200', 'name': 'Operating Expenses', 'type': 'Expense', 'parent': '5000'}
	]