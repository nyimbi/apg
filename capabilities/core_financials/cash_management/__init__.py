"""
Cash Management Sub-Capability

Comprehensive cash management system for tracking cash flow, bank reconciliations,
and liquidity management to ensure financial stability and optimal cash utilization.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Cash Management',
	'code': 'CM',
	'version': '1.0.0',
	'capability': 'core_financials',
	'description': 'Comprehensive cash management system for tracking cash flow, bank reconciliations, and liquidity management.',
	'industry_focus': 'All',
	'dependencies': ['general_ledger'],
	'optional_dependencies': ['accounts_payable', 'accounts_receivable'],
	'database_tables': [
		'cf_cm_bank_account',
		'cf_cm_bank_transaction',
		'cf_cm_reconciliation',
		'cf_cm_reconciliation_item',
		'cf_cm_cash_forecast',
		'cf_cm_cash_position',
		'cf_cm_investment',
		'cf_cm_currency_rate',
		'cf_cm_cash_transfer',
		'cf_cm_deposit',
		'cf_cm_check_register'
	],
	'api_endpoints': [
		'/api/core_financials/cm/bank_accounts',
		'/api/core_financials/cm/transactions',
		'/api/core_financials/cm/reconciliations',
		'/api/core_financials/cm/cash_forecast',
		'/api/core_financials/cm/cash_position',
		'/api/core_financials/cm/investments',
		'/api/core_financials/cm/fx_rates',
		'/api/core_financials/cm/transfers',
		'/api/core_financials/cm/deposits',
		'/api/core_financials/cm/checks',
		'/api/core_financials/cm/reports'
	],
	'views': [
		'CFCMBankAccountModelView',
		'CFCMBankTransactionModelView',
		'CFCMReconciliationModelView',
		'CFCMCashForecastModelView',
		'CFCMCashPositionModelView',
		'CFCMInvestmentModelView',
		'CFCMCurrencyRateModelView',
		'CFCMCashTransferModelView',
		'CFCMDepositModelView',
		'CFCMCheckRegisterModelView',
		'CFCMDashboardView'
	],
	'permissions': [
		'cm.read',
		'cm.write',
		'cm.reconcile',
		'cm.approve_transfers',
		'cm.manage_investments',
		'cm.view_forecast',
		'cm.admin'
	],
	'menu_items': [
		{
			'name': 'Bank Accounts',
			'endpoint': 'CFCMBankAccountModelView.list',
			'icon': 'fa-university',
			'permission': 'cm.read'
		},
		{
			'name': 'Bank Transactions',
			'endpoint': 'CFCMBankTransactionModelView.list',
			'icon': 'fa-exchange',
			'permission': 'cm.read'
		},
		{
			'name': 'Bank Reconciliation',
			'endpoint': 'CFCMReconciliationModelView.list',
			'icon': 'fa-check-square',
			'permission': 'cm.reconcile'
		},
		{
			'name': 'Cash Forecast',
			'endpoint': 'CFCMCashForecastModelView.list',
			'icon': 'fa-line-chart',
			'permission': 'cm.view_forecast'
		},
		{
			'name': 'Cash Position',
			'endpoint': 'CFCMCashPositionModelView.list',
			'icon': 'fa-money',
			'permission': 'cm.read'
		},
		{
			'name': 'Investments',
			'endpoint': 'CFCMInvestmentModelView.list',
			'icon': 'fa-chart-line',
			'permission': 'cm.manage_investments'
		},
		{
			'name': 'Cash Transfers',
			'endpoint': 'CFCMCashTransferModelView.list',
			'icon': 'fa-arrows-h',
			'permission': 'cm.approve_transfers'
		},
		{
			'name': 'Deposits',
			'endpoint': 'CFCMDepositModelView.list',
			'icon': 'fa-download',
			'permission': 'cm.read'
		},
		{
			'name': 'Check Register',
			'endpoint': 'CFCMCheckRegisterModelView.list',
			'icon': 'fa-list',
			'permission': 'cm.read'
		},
		{
			'name': 'Cash Dashboard',
			'endpoint': 'CFCMDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'cm.read'
		}
	],
	'configuration': {
		'auto_reconciliation': True,
		'require_approval_transfers': True,
		'cash_forecast_days': 90,
		'default_currency': 'USD',
		'multi_currency': True,
		'investment_tracking': True,
		'check_void_period': 180,  # Days after which checks can be voided
		'bank_statement_import': True,
		'automatic_matching': True,
		'tolerance_amount': 5.00,  # Auto-match tolerance
		'fx_rate_sources': ['Central Bank', 'Reuters', 'Bloomberg']
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
		errors.append("General Ledger is required for Cash Management integration")
	
	# Check optional dependencies
	if 'accounts_payable' not in available_subcapabilities:
		warnings.append("Accounts Payable integration not available - cash forecasting will be limited")
	
	if 'accounts_receivable' not in available_subcapabilities:
		warnings.append("Accounts Receivable integration not available - cash forecasting will be limited")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_bank_account_types() -> List[Dict[str, Any]]:
	"""Get default bank account types"""
	return [
		{
			'code': 'CHECKING',
			'name': 'Checking Account',
			'description': 'Primary operating account for daily transactions',
			'requires_reconciliation': True,
			'allows_investments': False,
			'default_gl_account': '1110'  # Cash and Cash Equivalents
		},
		{
			'code': 'SAVINGS',
			'name': 'Savings Account',
			'description': 'Interest-bearing savings account',
			'requires_reconciliation': True,
			'allows_investments': False,
			'default_gl_account': '1110'
		},
		{
			'code': 'MONEY_MARKET',
			'name': 'Money Market Account',
			'description': 'High-yield money market account',
			'requires_reconciliation': True,
			'allows_investments': True,
			'default_gl_account': '1115'  # Short-term Investments
		},
		{
			'code': 'INVESTMENT',
			'name': 'Investment Account',
			'description': 'Investment account for securities',
			'requires_reconciliation': True,
			'allows_investments': True,
			'default_gl_account': '1120'  # Marketable Securities
		},
		{
			'code': 'PETTY_CASH',
			'name': 'Petty Cash',
			'description': 'Physical cash for small expenses',
			'requires_reconciliation': False,
			'allows_investments': False,
			'default_gl_account': '1105'  # Petty Cash
		},
		{
			'code': 'LOCKBOX',
			'name': 'Lockbox Account',
			'description': 'Lockbox for automated receivables processing',
			'requires_reconciliation': True,
			'allows_investments': False,
			'default_gl_account': '1110'
		}
	]

def get_default_transaction_types() -> List[Dict[str, Any]]:
	"""Get default transaction types"""
	return [
		{
			'code': 'DEPOSIT',
			'name': 'Deposit',
			'description': 'Incoming funds deposit',
			'is_inflow': True,
			'requires_approval': False
		},
		{
			'code': 'WITHDRAWAL',
			'name': 'Withdrawal',
			'description': 'Outgoing funds withdrawal',
			'is_inflow': False,
			'requires_approval': True
		},
		{
			'code': 'TRANSFER_IN',
			'name': 'Transfer In',
			'description': 'Incoming transfer from another account',
			'is_inflow': True,
			'requires_approval': False
		},
		{
			'code': 'TRANSFER_OUT',
			'name': 'Transfer Out',
			'description': 'Outgoing transfer to another account',
			'is_inflow': False,
			'requires_approval': True
		},
		{
			'code': 'CHECK',
			'name': 'Check Payment',
			'description': 'Check payment disbursement',
			'is_inflow': False,
			'requires_approval': True
		},
		{
			'code': 'ACH_IN',
			'name': 'ACH Credit',
			'description': 'ACH credit received',
			'is_inflow': True,
			'requires_approval': False
		},
		{
			'code': 'ACH_OUT',
			'name': 'ACH Debit',
			'description': 'ACH debit payment',
			'is_inflow': False,
			'requires_approval': True
		},
		{
			'code': 'WIRE_IN',
			'name': 'Wire Transfer In',
			'description': 'Incoming wire transfer',
			'is_inflow': True,
			'requires_approval': False
		},
		{
			'code': 'WIRE_OUT',
			'name': 'Wire Transfer Out',
			'description': 'Outgoing wire transfer',
			'is_inflow': False,
			'requires_approval': True
		},
		{
			'code': 'FEE',
			'name': 'Bank Fee',
			'description': 'Bank service fee or charge',
			'is_inflow': False,
			'requires_approval': False
		},
		{
			'code': 'INTEREST',
			'name': 'Interest Earned',
			'description': 'Interest income earned',
			'is_inflow': True,
			'requires_approval': False
		},
		{
			'code': 'NSF',
			'name': 'NSF Fee',
			'description': 'Non-sufficient funds fee',
			'is_inflow': False,
			'requires_approval': False
		}
	]

def get_cash_forecast_categories() -> List[Dict[str, Any]]:
	"""Get default cash forecast categories"""
	return [
		{
			'code': 'AR_COLLECTIONS',
			'name': 'A/R Collections',
			'description': 'Collections from accounts receivable',
			'category_type': 'INFLOW',
			'source': 'accounts_receivable',
			'forecast_method': 'aging_analysis'
		},
		{
			'code': 'AP_PAYMENTS',
			'name': 'A/P Payments',
			'description': 'Payments to accounts payable',
			'category_type': 'OUTFLOW',
			'source': 'accounts_payable',
			'forecast_method': 'due_date_analysis'
		},
		{
			'code': 'PAYROLL',
			'name': 'Payroll',
			'description': 'Employee payroll disbursements',
			'category_type': 'OUTFLOW',
			'source': 'payroll',
			'forecast_method': 'recurring_schedule'
		},
		{
			'code': 'SALES_RECEIPTS',
			'name': 'Sales Receipts',
			'description': 'Direct sales receipts',
			'category_type': 'INFLOW',
			'source': 'sales',
			'forecast_method': 'historical_trend'
		},
		{
			'code': 'LOAN_PAYMENTS',
			'name': 'Loan Payments',
			'description': 'Loan principal and interest payments',
			'category_type': 'OUTFLOW',
			'source': 'loans',
			'forecast_method': 'amortization_schedule'
		},
		{
			'code': 'TAX_PAYMENTS',
			'name': 'Tax Payments',
			'description': 'Tax liability payments',
			'category_type': 'OUTFLOW',
			'source': 'tax',
			'forecast_method': 'scheduled_dates'
		},
		{
			'code': 'INVESTMENT_INCOME',
			'name': 'Investment Income',
			'description': 'Returns from investments',
			'category_type': 'INFLOW',
			'source': 'investments',
			'forecast_method': 'yield_analysis'
		},
		{
			'code': 'CAPITAL_EXPENDITURE',
			'name': 'Capital Expenditure',
			'description': 'Capital asset purchases',
			'category_type': 'OUTFLOW',
			'source': 'capex',
			'forecast_method': 'budget_based'
		}
	]