"""
Accounts Receivable Sub-Capability

Manages customer invoices, payments, and collections, accelerating cash inflow.
Provides comprehensive customer management, invoice generation, payment processing,
collections management, and aging analysis capabilities.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Accounts Receivable',
	'code': 'AR',
	'version': '1.0.0',
	'capability': 'core_financials',
	'description': 'Manages customer invoices, payments, and collections, accelerating cash inflow.',
	'industry_focus': 'All',
	'dependencies': ['general_ledger'],
	'optional_dependencies': ['inventory_management', 'sales_management', 'cash_management'],
	'database_tables': [
		'cf_ar_customer',
		'cf_ar_invoice',
		'cf_ar_invoice_line',
		'cf_ar_payment',
		'cf_ar_payment_line',
		'cf_ar_credit_memo',
		'cf_ar_credit_memo_line',
		'cf_ar_statement',
		'cf_ar_collection',
		'cf_ar_aging',
		'cf_ar_tax_code',
		'cf_ar_recurring_billing'
	],
	'api_endpoints': [
		'/api/core_financials/ar/customers',
		'/api/core_financials/ar/invoices',
		'/api/core_financials/ar/payments',
		'/api/core_financials/ar/credit_memos',
		'/api/core_financials/ar/collections',
		'/api/core_financials/ar/aging',
		'/api/core_financials/ar/statements',
		'/api/core_financials/ar/recurring_billing',
		'/api/core_financials/ar/reports'
	],
	'views': [
		'ARCustomerModelView',
		'ARInvoiceModelView',
		'ARPaymentModelView',
		'ARCreditMemoModelView',
		'ARStatementModelView',
		'ARCollectionModelView',
		'ARRecurringBillingModelView',
		'ARAgingView',
		'ARDashboardView'
	],
	'permissions': [
		'ar.read',
		'ar.write',
		'ar.post',
		'ar.collect',
		'ar.credit',
		'ar.statements',
		'ar.admin'
	],
	'menu_items': [
		{
			'name': 'AR Dashboard',
			'endpoint': 'ARDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'ar.read'
		},
		{
			'name': 'Customers',
			'endpoint': 'ARCustomerModelView.list',
			'icon': 'fa-users',
			'permission': 'ar.read'
		},
		{
			'name': 'Customer Invoices',
			'endpoint': 'ARInvoiceModelView.list',
			'icon': 'fa-file-invoice-dollar',
			'permission': 'ar.read'
		},
		{
			'name': 'Customer Payments',
			'endpoint': 'ARPaymentModelView.list',
			'icon': 'fa-money-check',
			'permission': 'ar.read'
		},
		{
			'name': 'Credit Memos',
			'endpoint': 'ARCreditMemoModelView.list',
			'icon': 'fa-undo',
			'permission': 'ar.read'
		},
		{
			'name': 'Collections',
			'endpoint': 'ARCollectionModelView.list',
			'icon': 'fa-phone',
			'permission': 'ar.read'
		},
		{
			'name': 'Customer Statements',
			'endpoint': 'ARStatementModelView.list',
			'icon': 'fa-file-alt',
			'permission': 'ar.read'
		},
		{
			'name': 'Recurring Billing',
			'endpoint': 'ARRecurringBillingModelView.list',
			'icon': 'fa-repeat',
			'permission': 'ar.read'
		},
		{
			'name': 'AR Aging',
			'endpoint': 'ARAgingView.index',
			'icon': 'fa-calendar-alt',
			'permission': 'ar.read'
		}
	],
	'configuration': {
		'auto_numbering': True,
		'require_approval': False,
		'allow_future_dates': True,
		'default_currency': 'USD',
		'multi_currency': True,
		'auto_apply_payments': True,
		'enable_collections': True,
		'dunning_enabled': True,
		'statement_cycle': 'MONTHLY',
		'aging_buckets': [30, 60, 90, 120]
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# Check required dependency
	if 'general_ledger' not in available_subcapabilities:
		errors.append("General Ledger is required for AR posting and financial reporting")
	
	# Check useful optional dependencies
	if 'inventory_management' not in available_subcapabilities:
		warnings.append("Inventory Management integration not available - manual item setup required")
	
	if 'sales_management' not in available_subcapabilities:
		warnings.append("Sales Management integration not available - limited sales order processing")
	
	if 'cash_management' not in available_subcapabilities:
		warnings.append("Cash Management integration not available - manual cash application required")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_customer_types() -> List[Dict[str, Any]]:
	"""Get default customer types"""
	return [
		{'code': 'RETAIL', 'name': 'Retail Customer', 'description': 'Individual retail customers'},
		{'code': 'WHOLESALE', 'name': 'Wholesale Customer', 'description': 'Wholesale/distributor customers'},
		{'code': 'CORPORATE', 'name': 'Corporate Customer', 'description': 'Large corporate accounts'},
		{'code': 'GOVERNMENT', 'name': 'Government Customer', 'description': 'Government agencies'},
		{'code': 'NONPROFIT', 'name': 'Non-Profit Customer', 'description': 'Non-profit organizations'},
		{'code': 'INTERNATIONAL', 'name': 'International Customer', 'description': 'International customers'},
		{'code': 'RESELLER', 'name': 'Reseller Partner', 'description': 'Authorized resellers'}
	]

def get_default_payment_terms() -> List[Dict[str, Any]]:
	"""Get default payment terms"""
	return [
		{'code': 'NET_10', 'name': 'Net 10', 'days': 10, 'discount_days': 0, 'discount_rate': 0.00},
		{'code': 'NET_15', 'name': 'Net 15', 'days': 15, 'discount_days': 0, 'discount_rate': 0.00},
		{'code': 'NET_30', 'name': 'Net 30', 'days': 30, 'discount_days': 0, 'discount_rate': 0.00},
		{'code': '2_10_NET_30', 'name': '2/10 Net 30', 'days': 30, 'discount_days': 10, 'discount_rate': 2.00},
		{'code': '1_10_NET_30', 'name': '1/10 Net 30', 'days': 30, 'discount_days': 10, 'discount_rate': 1.00},
		{'code': 'DUE_ON_RECEIPT', 'name': 'Due on Receipt', 'days': 0, 'discount_days': 0, 'discount_rate': 0.00},
		{'code': 'COD', 'name': 'Cash on Delivery', 'days': 0, 'discount_days': 0, 'discount_rate': 0.00},
		{'code': 'NET_60', 'name': 'Net 60', 'days': 60, 'discount_days': 0, 'discount_rate': 0.00}
	]

def get_default_tax_codes() -> List[Dict[str, Any]]:
	"""Get default AR tax codes"""
	return [
		{'code': 'SALES_TAX', 'name': 'Sales Tax', 'description': 'Standard sales tax', 'rate': 8.25, 'is_active': True},
		{'code': 'VAT_STANDARD', 'name': 'VAT Standard', 'description': 'Standard VAT rate', 'rate': 20.00, 'is_active': True},
		{'code': 'VAT_REDUCED', 'name': 'VAT Reduced', 'description': 'Reduced VAT rate', 'rate': 5.00, 'is_active': True},
		{'code': 'EXEMPT', 'name': 'Tax Exempt', 'description': 'Tax exempt transactions', 'rate': 0.00, 'is_active': True},
		{'code': 'GST', 'name': 'Goods & Services Tax', 'description': 'GST for applicable regions', 'rate': 10.00, 'is_active': True},
		{'code': 'PST', 'name': 'Provincial Sales Tax', 'description': 'Provincial sales tax', 'rate': 7.00, 'is_active': True},
		{'code': 'HST', 'name': 'Harmonized Sales Tax', 'description': 'Combined GST/PST', 'rate': 13.00, 'is_active': True}
	]

def get_default_gl_account_mappings() -> Dict[str, str]:
	"""Get default GL account mappings for AR"""
	return {
		'accounts_receivable': '1120',  # AR control account
		'sales_revenue': '4100',  # Sales revenue
		'sales_tax_payable': '2130',  # Sales tax liability
		'unearned_revenue': '2140',  # Deferred revenue
		'bad_debt_expense': '5250',  # Bad debt expense
		'allowance_doubtful_accounts': '1125',  # Allowance for doubtful accounts
		'cash': '1110',  # Cash account for payments
		'sales_discounts': '4200',  # Sales discounts/returns
		'finance_charges': '4300',  # Finance charges revenue
		'collection_costs': '5260'  # Collection costs expense
	}

def get_default_dunning_levels() -> List[Dict[str, Any]]:
	"""Get default dunning/collection levels"""
	return [
		{
			'level': 1,
			'name': 'Friendly Reminder',
			'days_past_due': 10,
			'message_template': 'This is a friendly reminder that your payment is past due.',
			'is_active': True,
			'finance_charge_rate': 0.00
		},
		{
			'level': 2,
			'name': 'First Notice',
			'days_past_due': 30,
			'message_template': 'Your account is now 30 days past due. Please remit payment immediately.',
			'is_active': True,
			'finance_charge_rate': 1.50
		},
		{
			'level': 3,
			'name': 'Second Notice',
			'days_past_due': 60,
			'message_template': 'Your account is seriously past due. Payment must be received within 10 days to avoid collection action.',
			'is_active': True,
			'finance_charge_rate': 1.50
		},
		{
			'level': 4,
			'name': 'Final Notice',
			'days_past_due': 90,
			'message_template': 'FINAL NOTICE: Your account will be forwarded to collections if payment is not received within 5 days.',
			'is_active': True,
			'finance_charge_rate': 2.00
		},
		{
			'level': 5,
			'name': 'Collections',
			'days_past_due': 120,
			'message_template': 'Your account has been forwarded to our collections department.',
			'is_active': True,
			'finance_charge_rate': 2.00
		}
	]

def get_default_statement_templates() -> List[Dict[str, Any]]:
	"""Get default statement templates"""
	return [
		{
			'template_name': 'Standard Statement',
			'template_type': 'MONTHLY',
			'description': 'Standard monthly customer statement',
			'include_aged_balance': True,
			'include_payment_terms': True,
			'include_remittance_slip': True,
			'is_default': True
		},
		{
			'template_name': 'Summary Statement',
			'template_type': 'MONTHLY',
			'description': 'Summary statement with totals only',
			'include_aged_balance': True,
			'include_payment_terms': False,
			'include_remittance_slip': False,
			'is_default': False
		},
		{
			'template_name': 'Detailed Statement',
			'template_type': 'MONTHLY',
			'description': 'Detailed statement with all transactions',
			'include_aged_balance': True,
			'include_payment_terms': True,
			'include_remittance_slip': True,
			'is_default': False
		}
	]

def get_invoice_numbering_formats() -> List[Dict[str, Any]]:
	"""Get default invoice numbering formats"""
	return [
		{
			'format_name': 'Sequential',
			'pattern': 'INV-{NNNNNN}',
			'description': 'Sequential numbering: INV-000001',
			'is_default': True
		},
		{
			'format_name': 'Year-Sequential',
			'pattern': 'INV-{YYYY}-{NNNNNN}',
			'description': 'Year-based: INV-2024-000001',
			'is_default': False
		},
		{
			'format_name': 'Customer-Sequential',
			'pattern': '{CUSTOMER}-{NNNNNN}',
			'description': 'Customer-based: CUST001-000001',
			'is_default': False
		},
		{
			'format_name': 'Date-Sequential',
			'pattern': 'INV-{YYYYMMDD}-{NNN}',
			'description': 'Date-based: INV-20240101-001',
			'is_default': False
		}
	]

def get_recurring_billing_frequencies() -> List[Dict[str, Any]]:
	"""Get recurring billing frequencies"""
	return [
		{'code': 'WEEKLY', 'name': 'Weekly', 'days': 7},
		{'code': 'BIWEEKLY', 'name': 'Bi-Weekly', 'days': 14},
		{'code': 'MONTHLY', 'name': 'Monthly', 'days': 30},
		{'code': 'QUARTERLY', 'name': 'Quarterly', 'days': 90},
		{'code': 'SEMIANNUAL', 'name': 'Semi-Annual', 'days': 180},
		{'code': 'ANNUAL', 'name': 'Annual', 'days': 365}
	]