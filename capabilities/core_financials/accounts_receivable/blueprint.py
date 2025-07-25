"""
Accounts Receivable Blueprint

Flask blueprint registration for Accounts Receivable sub-capability.
Registers all views, API endpoints, and URL routes for AR functionality.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	ARCustomerModelView, ARInvoiceModelView, ARPaymentModelView,
	ARCreditMemoModelView, ARStatementModelView, ARCollectionModelView,
	ARRecurringBillingModelView, ARTaxCodeModelView, ARAgingView, ARDashboardView
)
from .api import create_api_blueprint


def register_views(appbuilder: AppBuilder):
	"""Register Accounts Receivable views with Flask-AppBuilder"""
	
	# Dashboard
	appbuilder.add_view_no_menu(ARDashboardView())
	appbuilder.add_link(
		"AR Dashboard",
		href="/ar_dashboard/",
		icon="fa-dashboard",
		category="Accounts Receivable",
		category_icon="fa-money-check-alt"
	)
	
	# Customer Management
	appbuilder.add_view(
		ARCustomerModelView,
		"Customers",
		icon="fa-users",
		category="Accounts Receivable"
	)
	
	# Invoice Management
	appbuilder.add_view(
		ARInvoiceModelView,
		"Customer Invoices",
		icon="fa-file-invoice-dollar",
		category="Accounts Receivable"
	)
	
	# Payment Management
	appbuilder.add_view(
		ARPaymentModelView,
		"Customer Payments",
		icon="fa-money-check",
		category="Accounts Receivable"
	)
	
	# Credit Memo Management
	appbuilder.add_view(
		ARCreditMemoModelView,
		"Credit Memos",
		icon="fa-undo",
		category="Accounts Receivable"
	)
	
	# Statement Management
	appbuilder.add_view(
		ARStatementModelView,
		"Customer Statements",
		icon="fa-file-alt",
		category="Accounts Receivable"
	)
	
	# Collection Management
	appbuilder.add_view(
		ARCollectionModelView,
		"Collections",
		icon="fa-phone",
		category="Accounts Receivable"
	)
	
	# Recurring Billing
	appbuilder.add_view(
		ARRecurringBillingModelView,
		"Recurring Billing",
		icon="fa-repeat",
		category="Accounts Receivable"
	)
	
	# Tax Codes
	appbuilder.add_view(
		ARTaxCodeModelView,
		"AR Tax Codes",
		icon="fa-percentage",
		category="Accounts Receivable"
	)
	
	# AR Aging Report
	appbuilder.add_view_no_menu(ARAgingView())
	appbuilder.add_link(
		"AR Aging Report",
		href="/ar_aging/",
		icon="fa-calendar-alt",
		category="Accounts Receivable"
	)


def register_api_blueprint(app):
	"""Register API blueprint with Flask app"""
	api_bp = create_api_blueprint()
	app.register_blueprint(api_bp)


def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for Accounts Receivable"""
	
	ar_bp = Blueprint(
		'accounts_receivable',
		__name__,
		url_prefix='/ar',
		template_folder='templates',
		static_folder='static'
	)
	
	return ar_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register Accounts Receivable permissions"""
	
	permissions = [
		# Customer permissions
		('can_list', 'ARCustomerModelView'),
		('can_show', 'ARCustomerModelView'),
		('can_add', 'ARCustomerModelView'),
		('can_edit', 'ARCustomerModelView'),
		('can_delete', 'ARCustomerModelView'),
		('can_place_on_hold', 'ARCustomerModelView'),
		('can_release_hold', 'ARCustomerModelView'),
		('can_customer_summary', 'ARCustomerModelView'),
		
		# Invoice permissions
		('can_list', 'ARInvoiceModelView'),
		('can_show', 'ARInvoiceModelView'),
		('can_add', 'ARInvoiceModelView'),
		('can_edit', 'ARInvoiceModelView'),
		('can_delete', 'ARInvoiceModelView'),
		('can_post_invoice', 'ARInvoiceModelView'),
		('can_hold_invoice', 'ARInvoiceModelView'),
		('can_invoice_lines', 'ARInvoiceModelView'),
		
		# Payment permissions
		('can_list', 'ARPaymentModelView'),
		('can_show', 'ARPaymentModelView'),
		('can_add', 'ARPaymentModelView'),
		('can_edit', 'ARPaymentModelView'),
		('can_delete', 'ARPaymentModelView'),
		('can_post_payment', 'ARPaymentModelView'),
		('can_auto_apply', 'ARPaymentModelView'),
		('can_void_payment', 'ARPaymentModelView'),
		('can_payment_lines', 'ARPaymentModelView'),
		
		# Credit Memo permissions
		('can_list', 'ARCreditMemoModelView'),
		('can_show', 'ARCreditMemoModelView'),
		('can_add', 'ARCreditMemoModelView'),
		('can_edit', 'ARCreditMemoModelView'),
		('can_delete', 'ARCreditMemoModelView'),
		('can_post_credit_memo', 'ARCreditMemoModelView'),
		
		# Statement permissions
		('can_list', 'ARStatementModelView'),
		('can_show', 'ARStatementModelView'),
		('can_add', 'ARStatementModelView'),
		('can_edit', 'ARStatementModelView'),
		('can_delete', 'ARStatementModelView'),
		('can_generate_statements', 'ARStatementModelView'),
		
		# Collection permissions
		('can_list', 'ARCollectionModelView'),
		('can_show', 'ARCollectionModelView'),
		('can_add', 'ARCollectionModelView'),
		('can_edit', 'ARCollectionModelView'),
		('can_delete', 'ARCollectionModelView'),
		('can_mark_promise_kept', 'ARCollectionModelView'),
		('can_mark_promise_broken', 'ARCollectionModelView'),
		
		# Recurring Billing permissions
		('can_list', 'ARRecurringBillingModelView'),
		('can_show', 'ARRecurringBillingModelView'),
		('can_add', 'ARRecurringBillingModelView'),
		('can_edit', 'ARRecurringBillingModelView'),
		('can_delete', 'ARRecurringBillingModelView'),
		('can_pause_billing', 'ARRecurringBillingModelView'),
		('can_resume_billing', 'ARRecurringBillingModelView'),
		('can_process_billing', 'ARRecurringBillingModelView'),
		
		# Tax Code permissions
		('can_list', 'ARTaxCodeModelView'),
		('can_show', 'ARTaxCodeModelView'),
		('can_add', 'ARTaxCodeModelView'),
		('can_edit', 'ARTaxCodeModelView'),
		('can_delete', 'ARTaxCodeModelView'),
		
		# Report permissions
		('can_index', 'ARAgingView'),
		('can_export', 'ARAgingView'),
		('can_index', 'ARDashboardView'),
		('can_api_summary', 'ARDashboardView'),
		('can_api_cash_flow', 'ARDashboardView'),
		
		# API permissions
		('can_get_list', 'ARCustomerApi'),
		('can_get_customer', 'ARCustomerApi'),
		('can_create_customer', 'ARCustomerApi'),
		('can_update_customer', 'ARCustomerApi'),
		('can_get_customer_summary', 'ARCustomerApi'),
		('can_get_customer_invoices', 'ARCustomerApi'),
		('can_get_customer_payments', 'ARCustomerApi'),
		('can_place_customer_on_hold', 'ARCustomerApi'),
		('can_release_customer_hold', 'ARCustomerApi'),
		
		('can_get_list', 'ARInvoiceApi'),
		('can_get_invoice', 'ARInvoiceApi'),
		('can_create_invoice', 'ARInvoiceApi'),
		('can_update_invoice', 'ARInvoiceApi'),
		('can_post_invoice', 'ARInvoiceApi'),
		('can_get_invoice_lines', 'ARInvoiceApi'),
		('can_add_invoice_line', 'ARInvoiceApi'),
		('can_update_invoice_line', 'ARInvoiceApi'),
		('can_delete_invoice_line', 'ARInvoiceApi'),
		
		('can_get_list', 'ARPaymentApi'),
		('can_get_payment', 'ARPaymentApi'),
		('can_create_payment', 'ARPaymentApi'),
		('can_update_payment', 'ARPaymentApi'),
		('can_post_payment', 'ARPaymentApi'),
		('can_auto_apply_payment', 'ARPaymentApi'),
		('can_void_payment', 'ARPaymentApi'),
		('can_get_payment_lines', 'ARPaymentApi'),
		('can_add_payment_line', 'ARPaymentApi'),
		
		('can_get_list', 'ARCreditMemoApi'),
		('can_get_credit_memo', 'ARCreditMemoApi'),
		('can_create_credit_memo', 'ARCreditMemoApi'),
		('can_update_credit_memo', 'ARCreditMemoApi'),
		('can_post_credit_memo', 'ARCreditMemoApi'),
		('can_get_credit_memo_lines', 'ARCreditMemoApi'),
		
		('can_get_list', 'ARStatementApi'),
		('can_get_statement', 'ARStatementApi'),
		('can_generate_statement', 'ARStatementApi'),
		('can_generate_batch_statements', 'ARStatementApi'),
		
		('can_get_list', 'ARCollectionApi'),
		('can_get_collection', 'ARCollectionApi'),
		('can_create_collection', 'ARCollectionApi'),
		('can_update_collection', 'ARCollectionApi'),
		('can_get_customers_for_collections', 'ARCollectionApi'),
		('can_generate_dunning_letters', 'ARCollectionApi'),
		
		('can_get_list', 'ARRecurringBillingApi'),
		('can_get_recurring_billing', 'ARRecurringBillingApi'),
		('can_create_recurring_billing', 'ARRecurringBillingApi'),
		('can_update_recurring_billing', 'ARRecurringBillingApi'),
		('can_process_recurring_billing', 'ARRecurringBillingApi'),
		('can_pause_recurring_billing', 'ARRecurringBillingApi'),
		('can_resume_recurring_billing', 'ARRecurringBillingApi'),
		
		('can_get_list', 'ARTaxCodeApi'),
		('can_get_tax_code', 'ARTaxCodeApi'),
		('can_create_tax_code', 'ARTaxCodeApi'),
		('can_update_tax_code', 'ARTaxCodeApi'),
		
		('can_get_aging_report', 'ARAgingApi'),
		('can_get_aging_summary', 'ARAgingApi'),
		('can_generate_aging_report', 'ARAgingApi'),
		
		('can_get_dashboard_summary', 'ARDashboardApi'),
		('can_get_cash_flow_projection', 'ARDashboardApi'),
		('can_get_collection_dashboard', 'ARDashboardApi'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for Accounts Receivable"""
	
	return {
		'name': 'Accounts Receivable',
		'icon': 'fa-money-check-alt',
		'items': [
			{
				'name': 'AR Dashboard',
				'href': '/ar_dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_index on ARDashboardView'
			},
			{
				'name': 'Customers',
				'href': '/arcustomermodelview/list/',
				'icon': 'fa-users',
				'permission': 'can_list on ARCustomerModelView'
			},
			{
				'name': 'Customer Invoices',
				'href': '/arinvoicemodelview/list/',
				'icon': 'fa-file-invoice-dollar',
				'permission': 'can_list on ARInvoiceModelView'
			},
			{
				'name': 'Customer Payments',
				'href': '/arpaymentmodelview/list/',
				'icon': 'fa-money-check',
				'permission': 'can_list on ARPaymentModelView'
			},
			{
				'name': 'Credit Memos',
				'href': '/arcreditmemomodelview/list/',
				'icon': 'fa-undo',
				'permission': 'can_list on ARCreditMemoModelView'
			},
			{
				'name': 'Customer Statements',
				'href': '/arstatementmodelview/list/',
				'icon': 'fa-file-alt',
				'permission': 'can_list on ARStatementModelView'
			},
			{
				'name': 'Collections',
				'href': '/arcollectionmodelview/list/',
				'icon': 'fa-phone',
				'permission': 'can_list on ARCollectionModelView'
			},
			{
				'name': 'Recurring Billing',
				'href': '/arrecurringbillingmodelview/list/',
				'icon': 'fa-repeat',
				'permission': 'can_list on ARRecurringBillingModelView'
			},
			{
				'name': 'AR Aging Report',
				'href': '/ar_aging/',
				'icon': 'fa-calendar-alt',
				'permission': 'can_index on ARAgingView'
			},
			{
				'name': 'AR Tax Codes',
				'href': '/artaxcodemodelview/list/',
				'icon': 'fa-percentage',
				'permission': 'can_list on ARTaxCodeModelView'
			}
		]
	}


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Accounts Receivable sub-capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default AR data if needed"""
	
	from .models import CFARTaxCode
	from ...auth_rbac.models import db
	from . import get_default_tax_codes
	
	try:
		# Check if tax codes already exist (use a default tenant for now)
		existing_tax_codes = CFARTaxCode.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_tax_codes == 0:
			# Create default tax codes
			default_tax_codes = get_default_tax_codes()
			
			for tax_data in default_tax_codes:
				tax_code = CFARTaxCode(
					tenant_id='default_tenant',
					code=tax_data['code'],
					name=tax_data['name'],
					description=tax_data['description'],
					tax_rate=tax_data['rate'],
					is_active=tax_data['is_active']
				)
				db.session.add(tax_code)
			
			db.session.commit()
			print("Default AR tax codes created")
			
	except Exception as e:
		print(f"Error initializing default AR data: {e}")
		db.session.rollback()


def create_default_customers(tenant_id: str, appbuilder: AppBuilder):
	"""Create default customers for a tenant"""
	
	from .service import AccountsReceivableService
	from . import get_default_customer_types
	
	try:
		ar_service = AccountsReceivableService(tenant_id)
		
		# Check if customers already exist
		existing_customers = ar_service.get_customers()
		if len(existing_customers) > 0:
			return
		
		# Create sample customers for each type
		customer_types = get_default_customer_types()
		
		sample_customers = [
			{
				'customer_number': '001001',
				'customer_name': 'ABC Manufacturing Corp',
				'customer_type': 'CORPORATE',
				'contact_name': 'John Smith',
				'email': 'accounting@abcmanufacturing.com',
				'phone': '555-0101',
				'billing_address_line1': '123 Business Ave',
				'billing_city': 'Business City',
				'billing_state_province': 'BC',
				'billing_postal_code': '12345',
				'billing_country': 'USA',
				'payment_terms_code': 'NET_30',
				'payment_method': 'ACH',
				'credit_limit': 50000.00,
				'is_active': True
			},
			{
				'customer_number': '002001',
				'customer_name': 'XYZ Retail Stores',
				'customer_type': 'RETAIL',
				'contact_name': 'Jane Doe',
				'email': 'payments@xyzretail.com',
				'phone': '555-0202',
				'billing_address_line1': '456 Commerce St',
				'billing_city': 'Commerce City',
				'billing_state_province': 'CC',
				'billing_postal_code': '67890',
				'billing_country': 'USA',
				'payment_terms_code': '2_10_NET_30',
				'payment_method': 'CHECK',
				'credit_limit': 25000.00,
				'is_active': True
			},
			{
				'customer_number': '003001',
				'customer_name': 'Global Distributors Inc',
				'customer_type': 'WHOLESALE',
				'contact_name': 'Mike Johnson',
				'email': 'ap@globaldist.com',
				'phone': '555-0303',
				'billing_address_line1': '789 Distribution Blvd',
				'billing_city': 'Distribution Center',
				'billing_state_province': 'DC',
				'billing_postal_code': '13579',
				'billing_country': 'USA',
				'payment_terms_code': 'NET_60',
				'payment_method': 'WIRE',
				'credit_limit': 100000.00,
				'is_active': True
			},
			{
				'customer_number': '004001',
				'customer_name': 'City Government',
				'customer_type': 'GOVERNMENT',
				'contact_name': 'Sarah Wilson',
				'email': 'procurement@citygovt.gov',
				'phone': '555-0404',
				'billing_address_line1': '101 City Hall Plaza',
				'billing_city': 'Government City',
				'billing_state_province': 'GC',
				'billing_postal_code': '24680',
				'billing_country': 'USA',
				'payment_terms_code': 'NET_30',
				'payment_method': 'ACH',
				'credit_limit': 75000.00,
				'tax_exempt': True,
				'tax_exempt_number': 'EXEMPT-001',
				'is_active': True
			},
			{
				'customer_number': '005001',
				'customer_name': 'Community Foundation',
				'customer_type': 'NONPROFIT',
				'contact_name': 'David Brown',
				'email': 'finance@communityfoundation.org',
				'phone': '555-0505',
				'billing_address_line1': '202 Charity Lane',
				'billing_city': 'Foundation City',
				'billing_state_province': 'FC',
				'billing_postal_code': '35791',
				'billing_country': 'USA',
				'payment_terms_code': 'NET_30',
				'payment_method': 'CHECK',
				'credit_limit': 15000.00,
				'tax_exempt': True,
				'tax_exempt_number': 'NONPROFIT-501C3',
				'is_active': True
			}
		]
		
		for customer_data in sample_customers:
			ar_service.create_customer(customer_data)
		
		print(f"Default customers created for tenant {tenant_id}")
		
	except Exception as e:
		print(f"Error creating default customers: {e}")


def setup_ar_integration(appbuilder: AppBuilder):
	"""Set up AR integration with other modules"""
	
	try:
		# Set up GL integration
		from ..general_ledger.models import CFGLAccount
		from ...auth_rbac.models import db
		from . import get_default_gl_account_mappings
		
		# Ensure required GL accounts exist
		gl_mappings = get_default_gl_account_mappings()
		
		for account_type, account_code in gl_mappings.items():
			existing_account = CFGLAccount.query.filter_by(
				tenant_id='default_tenant',
				account_code=account_code
			).first()
			
			if not existing_account:
				print(f"Warning: GL account {account_code} for {account_type} not found")
		
		print("AR-GL integration check completed")
		
	except Exception as e:
		print(f"Error setting up AR integration: {e}")


def get_ar_configuration():
	"""Get AR configuration settings"""
	
	from . import SUBCAPABILITY_META
	
	return SUBCAPABILITY_META['configuration']


def validate_ar_setup(tenant_id: str) -> dict[str, Any]:
	"""Validate AR setup for a tenant"""
	
	from .service import AccountsReceivableService
	from ..general_ledger.models import CFGLAccount
	from . import get_default_gl_account_mappings
	
	validation_results = {
		'valid': True,
		'errors': [],
		'warnings': []
	}
	
	try:
		ar_service = AccountsReceivableService(tenant_id)
		
		# Check if required GL accounts exist
		gl_mappings = get_default_gl_account_mappings()
		missing_accounts = []
		
		for account_type, account_code in gl_mappings.items():
			account = CFGLAccount.query.filter_by(
				tenant_id=tenant_id,
				account_code=account_code
			).first()
			
			if not account:
				missing_accounts.append(f"{account_type} ({account_code})")
		
		if missing_accounts:
			validation_results['errors'].append(
				f"Missing required GL accounts: {', '.join(missing_accounts)}"
			)
			validation_results['valid'] = False
		
		# Check if customers exist
		customers = ar_service.get_customers()
		if len(customers) == 0:
			validation_results['warnings'].append("No customers configured")
		
		# Check tax codes
		from .models import CFARTaxCode
		tax_codes = CFARTaxCode.query.filter_by(tenant_id=tenant_id).count()
		if tax_codes == 0:
			validation_results['warnings'].append("No tax codes configured")
		
		# Check payment terms configuration
		payment_terms_count = len([c for c in customers if c.payment_terms_code])
		if payment_terms_count == 0 and len(customers) > 0:
			validation_results['warnings'].append("No payment terms configured for customers")
		
	except Exception as e:
		validation_results['errors'].append(f"Validation error: {str(e)}")
		validation_results['valid'] = False
	
	return validation_results


def get_ar_dashboard_widgets():
	"""Get dashboard widgets for AR"""
	
	return [
		{
			'name': 'Total AR Balance',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/total_balance',
			'icon': 'fa-dollar-sign',
			'color': 'primary'
		},
		{
			'name': 'Past Due Amount',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/past_due',
			'icon': 'fa-exclamation-triangle',
			'color': 'warning'
		},
		{
			'name': 'Collection Required',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/collection_required',
			'icon': 'fa-phone',
			'color': 'danger'
		},
		{
			'name': 'Current Month Sales',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/current_month_sales',
			'icon': 'fa-chart-line',
			'color': 'success'
		},
		{
			'name': 'AR Aging',
			'widget_type': 'chart',
			'chart_type': 'pie',
			'api_endpoint': '/api/ar/dashboard/aging_chart',
			'height': 300
		},
		{
			'name': 'Cash Flow Projection',
			'widget_type': 'chart',
			'chart_type': 'bar',
			'api_endpoint': '/api/ar/dashboard/cash_flow_chart',
			'height': 250
		}
	]


def get_ar_reports():
	"""Get available AR reports"""
	
	return [
		{
			'name': 'AR Aging Report',
			'description': 'Customer aging analysis by due date buckets',
			'endpoint': '/ar_aging/',
			'parameters': ['as_of_date'],
			'formats': ['HTML', 'PDF', 'Excel']
		},
		{
			'name': 'Customer Statement',
			'description': 'Individual customer account statement',
			'endpoint': '/api/ar/reports/customer_statement',
			'parameters': ['customer_id', 'statement_date'],
			'formats': ['PDF']
		},
		{
			'name': 'Sales Analysis',
			'description': 'Sales analysis by customer, product, territory',
			'endpoint': '/api/ar/reports/sales_analysis',
			'parameters': ['date_from', 'date_to', 'group_by'],
			'formats': ['HTML', 'Excel']
		},
		{
			'name': 'Collection Report',
			'description': 'Collection activities and effectiveness',
			'endpoint': '/api/ar/reports/collection_report',
			'parameters': ['date_from', 'date_to'],
			'formats': ['HTML', 'PDF']
		},
		{
			'name': 'Cash Receipts Journal',
			'description': 'Detailed cash receipts for a period',
			'endpoint': '/api/ar/reports/cash_receipts',
			'parameters': ['date_from', 'date_to'],
			'formats': ['HTML', 'Excel']
		},
		{
			'name': 'Invoice Register',
			'description': 'List of invoices for a period',
			'endpoint': '/api/ar/reports/invoice_register',
			'parameters': ['date_from', 'date_to'],
			'formats': ['HTML', 'Excel']
		}
	]