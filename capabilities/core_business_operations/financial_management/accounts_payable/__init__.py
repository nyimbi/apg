"""
Accounts Payable Sub-Capability

Manages vendor invoices, payments, and expenses, optimizing cash outflow.
Handles vendor relationships, invoice processing, payment workflows, and expense reporting.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Accounts Payable',
	'code': 'AP',
	'version': '1.0.0',
	'capability': 'core_financials',
	'description': 'Manages vendor invoices, payments, and expenses, optimizing cash outflow.',
	'industry_focus': 'All',
	'dependencies': ['general_ledger'],
	'optional_dependencies': ['cash_management', 'procurement', 'expense_management'],
	'database_tables': [
		'cf_ap_vendor',
		'cf_ap_invoice',
		'cf_ap_invoice_line',
		'cf_ap_payment',
		'cf_ap_payment_line',
		'cf_ap_expense_report',
		'cf_ap_expense_line',
		'cf_ap_purchase_order',
		'cf_ap_tax_code',
		'cf_ap_aging'
	],
	'api_endpoints': [
		'/api/core_financials/ap/vendors',
		'/api/core_financials/ap/invoices',
		'/api/core_financials/ap/payments',
		'/api/core_financials/ap/expenses',
		'/api/core_financials/ap/purchase_orders',
		'/api/core_financials/ap/aging',
		'/api/core_financials/ap/reports'
	],
	'views': [
		'APVendorModelView',
		'APInvoiceModelView',
		'APPaymentModelView',
		'APExpenseReportModelView',
		'APPurchaseOrderModelView',
		'APAgingView',
		'APDashboardView'
	],
	'permissions': [
		'ap.read',
		'ap.write',
		'ap.approve_invoice',
		'ap.post_invoice',
		'ap.process_payment',
		'ap.approve_payment',
		'ap.vendor_admin',
		'ap.expense_admin',
		'ap.admin'
	],
	'menu_items': [
		{
			'name': 'AP Dashboard',
			'endpoint': 'APDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'ap.read'
		},
		{
			'name': 'Vendors',
			'endpoint': 'APVendorModelView.list',
			'icon': 'fa-building',
			'permission': 'ap.read'
		},
		{
			'name': 'Vendor Invoices',
			'endpoint': 'APInvoiceModelView.list',
			'icon': 'fa-file-invoice',
			'permission': 'ap.read'
		},
		{
			'name': 'Payments',
			'endpoint': 'APPaymentModelView.list',
			'icon': 'fa-credit-card',
			'permission': 'ap.read'
		},
		{
			'name': 'Expense Reports',
			'endpoint': 'APExpenseReportModelView.list',
			'icon': 'fa-receipt',
			'permission': 'ap.read'
		},
		{
			'name': 'Purchase Orders',
			'endpoint': 'APPurchaseOrderModelView.list',
			'icon': 'fa-shopping-cart',
			'permission': 'ap.read'
		},
		{
			'name': 'AP Aging',
			'endpoint': 'APAgingView.index',
			'icon': 'fa-clock',
			'permission': 'ap.read'
		}
	],
	'configuration': {
		'auto_invoice_numbering': True,
		'auto_payment_numbering': True,
		'require_invoice_approval': True,
		'require_payment_approval': True,
		'allow_partial_payments': True,
		'default_payment_terms': 30,  # days
		'default_currency': 'USD',
		'multi_currency': True,
		'three_way_matching': True,  # PO, Receipt, Invoice
		'expense_approval_workflow': True,
		'vendor_self_service': False,
		'electronic_payments': True,
		'check_printing': True,
		'tax_calculation': True,
		'aging_buckets': [30, 60, 90, 120],  # days
		'duplicate_invoice_check': True,
		'auto_gl_posting': True
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
		errors.append("General Ledger sub-capability is required for AP operations")
	
	# Check optional dependencies
	if 'cash_management' not in available_subcapabilities:
		warnings.append("Cash Management integration not available - manual cash reconciliation required")
	
	if 'procurement' not in available_subcapabilities:
		warnings.append("Procurement integration not available - manual PO creation required")
	
	if 'expense_management' not in available_subcapabilities:
		warnings.append("Expense Management integration not available - basic expense tracking only")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_vendor_types() -> List[Dict[str, Any]]:
	"""Get default vendor types"""
	return [
		{'code': 'SUPPLIER', 'name': 'Supplier', 'description': 'General suppliers of goods and services'},
		{'code': 'CONTRACTOR', 'name': 'Contractor', 'description': 'Independent contractors and service providers'},
		{'code': 'UTILITY', 'name': 'Utility', 'description': 'Utility companies (electric, gas, water, etc.)'},
		{'code': 'LANDLORD', 'name': 'Landlord', 'description': 'Property rental and lease providers'},
		{'code': 'EMPLOYEE', 'name': 'Employee', 'description': 'Employees for expense reimbursements'},
		{'code': 'GOVERNMENT', 'name': 'Government', 'description': 'Government agencies and tax authorities'},
		{'code': 'PROFESSIONAL', 'name': 'Professional Services', 'description': 'Legal, accounting, consulting services'},
		{'code': 'TECHNOLOGY', 'name': 'Technology', 'description': 'Software, hardware, and IT service providers'}
	]

def get_default_payment_terms() -> List[Dict[str, Any]]:
	"""Get default payment terms"""
	return [
		{'code': 'NET_30', 'name': 'Net 30', 'days': 30, 'discount_days': 0, 'discount_percent': 0.00},
		{'code': 'NET_15', 'name': 'Net 15', 'days': 15, 'discount_days': 0, 'discount_percent': 0.00},
		{'code': '2_10_NET_30', 'name': '2/10 Net 30', 'days': 30, 'discount_days': 10, 'discount_percent': 2.00},
		{'code': '1_10_NET_30', 'name': '1/10 Net 30', 'days': 30, 'discount_days': 10, 'discount_percent': 1.00},
		{'code': 'DUE_ON_RECEIPT', 'name': 'Due on Receipt', 'days': 0, 'discount_days': 0, 'discount_percent': 0.00},
		{'code': 'NET_60', 'name': 'Net 60', 'days': 60, 'discount_days': 0, 'discount_percent': 0.00},
		{'code': 'NET_90', 'name': 'Net 90', 'days': 90, 'discount_days': 0, 'discount_percent': 0.00}
	]

def get_default_tax_codes() -> List[Dict[str, Any]]:
	"""Get default tax codes for AP"""
	return [
		{
			'code': 'STANDARD',
			'name': 'Standard Tax',
			'description': 'Standard tax rate',
			'rate': 8.50,
			'is_active': True,
			'gl_account_code': '2300'  # Tax Payable
		},
		{
			'code': 'EXEMPT',
			'name': 'Tax Exempt',
			'description': 'Tax exempt purchases',
			'rate': 0.00,
			'is_active': True,
			'gl_account_code': None
		},
		{
			'code': 'REDUCED',
			'name': 'Reduced Tax',
			'description': 'Reduced tax rate',
			'rate': 5.00,
			'is_active': True,
			'gl_account_code': '2300'
		}
	]

def get_default_expense_categories() -> List[Dict[str, Any]]:
	"""Get default expense categories"""
	return [
		{'code': 'TRAVEL', 'name': 'Travel', 'description': 'Business travel expenses', 'gl_account_code': '5210'},
		{'code': 'MEALS', 'name': 'Meals & Entertainment', 'description': 'Business meals and entertainment', 'gl_account_code': '5220'},
		{'code': 'OFFICE', 'name': 'Office Supplies', 'description': 'Office supplies and materials', 'gl_account_code': '5230'},
		{'code': 'PHONE', 'name': 'Communications', 'description': 'Phone, internet, communication costs', 'gl_account_code': '5240'},
		{'code': 'TRAINING', 'name': 'Training & Development', 'description': 'Training, conferences, education', 'gl_account_code': '5250'},
		{'code': 'VEHICLE', 'name': 'Vehicle', 'description': 'Vehicle expenses, fuel, maintenance', 'gl_account_code': '5260'},
		{'code': 'OTHER', 'name': 'Other', 'description': 'Other business expenses', 'gl_account_code': '5290'}
	]

def get_default_gl_account_mappings() -> Dict[str, str]:
	"""Get default GL account mappings for AP transactions"""
	return {
		'accounts_payable': '2110',  # Accounts Payable control account
		'expense_accrual': '2120',   # Accrued Expenses
		'prepaid_expenses': '1140',  # Prepaid Expenses
		'cash_clearing': '1105',     # Cash Clearing account
		'discount_taken': '4200',    # Purchase Discounts (contra expense)
		'exchange_gain_loss': '4300',  # Foreign Exchange Gain/Loss
		'tax_payable': '2300',       # Tax Payable
		'withholding_tax': '2310'    # Withholding Tax Payable
	}