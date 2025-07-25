"""
Accounts Payable Blueprint

Flask blueprint registration for Accounts Payable sub-capability.
Registers all views, API endpoints, and URL routes for AP functionality.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	APVendorModelView, APInvoiceModelView, APPaymentModelView,
	APExpenseReportModelView, APPurchaseOrderModelView, APTaxCodeModelView,
	APAgingView, APDashboardView
)
from .api import create_api_blueprint


def register_views(appbuilder: AppBuilder):
	"""Register Accounts Payable views with Flask-AppBuilder"""
	
	# Dashboard
	appbuilder.add_view_no_menu(APDashboardView())
	appbuilder.add_link(
		"AP Dashboard",
		href="/ap/dashboard/",
		icon="fa-dashboard",
		category="Accounts Payable",
		category_icon="fa-credit-card"
	)
	
	# Vendor Management
	appbuilder.add_view(
		APVendorModelView,
		"Vendors",
		icon="fa-building",
		category="Accounts Payable"
	)
	
	# Invoice Management
	appbuilder.add_view(
		APInvoiceModelView,
		"Vendor Invoices",
		icon="fa-file-invoice",
		category="Accounts Payable"
	)
	
	# Payment Management
	appbuilder.add_view(
		APPaymentModelView,
		"Payments",
		icon="fa-credit-card",
		category="Accounts Payable"
	)
	
	# Expense Reports
	appbuilder.add_view(
		APExpenseReportModelView,
		"Expense Reports",
		icon="fa-receipt",
		category="Accounts Payable"
	)
	
	# Purchase Orders
	appbuilder.add_view(
		APPurchaseOrderModelView,
		"Purchase Orders",
		icon="fa-shopping-cart",
		category="Accounts Payable"
	)
	
	# Tax Codes
	appbuilder.add_view(
		APTaxCodeModelView,
		"Tax Codes",
		icon="fa-percentage",
		category="Accounts Payable"
	)
	
	# Aging Report
	appbuilder.add_view_no_menu(APAgingView())
	appbuilder.add_link(
		"AP Aging",
		href="/ap/aging/",
		icon="fa-clock",
		category="Accounts Payable"
	)


def register_api_blueprint(app):
	"""Register API blueprint with Flask app"""
	api_bp = create_api_blueprint()
	app.register_blueprint(api_bp)


def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for Accounts Payable"""
	
	ap_bp = Blueprint(
		'accounts_payable',
		__name__,
		url_prefix='/ap',
		template_folder='templates',
		static_folder='static'
	)
	
	return ap_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register Accounts Payable permissions"""
	
	permissions = [
		# Vendor permissions
		('can_list', 'APVendorModelView'),
		('can_show', 'APVendorModelView'),
		('can_add', 'APVendorModelView'),
		('can_edit', 'APVendorModelView'),
		('can_delete', 'APVendorModelView'),
		('can_hold_payment', 'APVendorModelView'),
		('can_release_hold', 'APVendorModelView'),
		('can_vendor_summary', 'APVendorModelView'),
		
		# Invoice permissions
		('can_list', 'APInvoiceModelView'),
		('can_show', 'APInvoiceModelView'),
		('can_add', 'APInvoiceModelView'),
		('can_edit', 'APInvoiceModelView'),
		('can_delete', 'APInvoiceModelView'),
		('can_approve_invoice', 'APInvoiceModelView'),
		('can_post_invoice', 'APInvoiceModelView'),
		('can_hold_invoice', 'APInvoiceModelView'),
		('can_invoice_lines', 'APInvoiceModelView'),
		
		# Payment permissions
		('can_list', 'APPaymentModelView'),
		('can_show', 'APPaymentModelView'),
		('can_add', 'APPaymentModelView'),
		('can_edit', 'APPaymentModelView'),
		('can_delete', 'APPaymentModelView'),
		('can_approve_payment', 'APPaymentModelView'),
		('can_post_payment', 'APPaymentModelView'),
		('can_void_payment', 'APPaymentModelView'),
		('can_payment_lines', 'APPaymentModelView'),
		
		# Expense Report permissions
		('can_list', 'APExpenseReportModelView'),
		('can_show', 'APExpenseReportModelView'),
		('can_add', 'APExpenseReportModelView'),
		('can_edit', 'APExpenseReportModelView'),
		('can_delete', 'APExpenseReportModelView'),
		('can_submit_report', 'APExpenseReportModelView'),
		('can_approve_report', 'APExpenseReportModelView'),
		('can_expense_lines', 'APExpenseReportModelView'),
		
		# Purchase Order permissions
		('can_list', 'APPurchaseOrderModelView'),
		('can_show', 'APPurchaseOrderModelView'),
		('can_add', 'APPurchaseOrderModelView'),
		('can_edit', 'APPurchaseOrderModelView'),
		('can_delete', 'APPurchaseOrderModelView'),
		
		# Tax Code permissions
		('can_list', 'APTaxCodeModelView'),
		('can_show', 'APTaxCodeModelView'),
		('can_add', 'APTaxCodeModelView'),
		('can_edit', 'APTaxCodeModelView'),
		('can_delete', 'APTaxCodeModelView'),
		
		# Report permissions
		('can_index', 'APAgingView'),
		('can_export', 'APAgingView'),
		('can_index', 'APDashboardView'),
		('can_api_summary', 'APDashboardView'),
		('can_api_cash_flow', 'APDashboardView'),
		
		# API permissions
		('can_get_list', 'APVendorApi'),
		('can_get_vendor', 'APVendorApi'),
		('can_create_vendor', 'APVendorApi'),
		('can_get_vendor_summary', 'APVendorApi'),
		
		('can_get_list', 'APInvoiceApi'),
		('can_get_invoice', 'APInvoiceApi'),
		('can_create_invoice', 'APInvoiceApi'),
		('can_approve_invoice', 'APInvoiceApi'),
		('can_post_invoice', 'APInvoiceApi'),
		('can_get_invoice_lines', 'APInvoiceApi'),
		('can_add_invoice_line', 'APInvoiceApi'),
		
		('can_get_list', 'APPaymentApi'),
		('can_get_payment', 'APPaymentApi'),
		('can_create_payment', 'APPaymentApi'),
		('can_approve_payment', 'APPaymentApi'),
		('can_post_payment', 'APPaymentApi'),
		
		('can_get_list', 'APExpenseReportApi'),
		('can_get_expense_report', 'APExpenseReportApi'),
		('can_create_expense_report', 'APExpenseReportApi'),
		('can_submit_expense_report', 'APExpenseReportApi'),
		('can_approve_expense_report', 'APExpenseReportApi'),
		
		('can_get_aging_report', 'APAgingApi'),
		('can_get_summary', 'APDashboardApi'),
		('can_get_cash_requirements', 'APDashboardApi'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for Accounts Payable"""
	
	return {
		'name': 'Accounts Payable',
		'icon': 'fa-credit-card',
		'items': [
			{
				'name': 'AP Dashboard',
				'href': '/ap/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_index on APDashboardView'
			},
			{
				'name': 'Vendors',
				'href': '/apvendomodelview/list/',
				'icon': 'fa-building',
				'permission': 'can_list on APVendorModelView'
			},
			{
				'name': 'Vendor Invoices',
				'href': '/apinvoicemodelview/list/',
				'icon': 'fa-file-invoice',
				'permission': 'can_list on APInvoiceModelView'
			},
			{
				'name': 'Payments',
				'href': '/appaymentmodelview/list/',
				'icon': 'fa-credit-card',
				'permission': 'can_list on APPaymentModelView'
			},
			{
				'name': 'Expense Reports',
				'href': '/apexpensereportmodelview/list/',
				'icon': 'fa-receipt',
				'permission': 'can_list on APExpenseReportModelView'
			},
			{
				'name': 'Purchase Orders',
				'href': '/appurchaseordermodelview/list/',
				'icon': 'fa-shopping-cart',
				'permission': 'can_list on APPurchaseOrderModelView'
			},
			{
				'name': 'AP Aging',
				'href': '/ap/aging/',
				'icon': 'fa-clock',
				'permission': 'can_index on APAgingView'
			},
			{
				'name': 'Tax Codes',
				'href': '/aptaxcodemodelview/list/',
				'icon': 'fa-percentage',
				'permission': 'can_list on APTaxCodeModelView'
			}
		]
	}


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Accounts Payable sub-capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default AP data if needed"""
	
	from .models import CFAPTaxCode
	from ...auth_rbac.models import db
	from . import get_default_tax_codes
	
	try:
		# Check if tax codes already exist (use a default tenant for now)
		existing_tax_codes = CFAPTaxCode.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_tax_codes == 0:
			# Create default tax codes
			default_tax_codes = get_default_tax_codes()
			
			for tax_data in default_tax_codes:
				tax_code = CFAPTaxCode(
					tenant_id='default_tenant',
					code=tax_data['code'],
					name=tax_data['name'],
					description=tax_data['description'],
					tax_rate=tax_data['rate'],
					is_active=tax_data['is_active']
				)
				db.session.add(tax_code)
			
			db.session.commit()
			print("Default AP tax codes created")
			
	except Exception as e:
		print(f"Error initializing default AP data: {e}")
		db.session.rollback()


def create_default_vendors(tenant_id: str, appbuilder: AppBuilder):
	"""Create default vendors for a tenant"""
	
	from .service import AccountsPayableService
	from . import get_default_vendor_types
	
	try:
		ap_service = AccountsPayableService(tenant_id)
		
		# Check if vendors already exist
		existing_vendors = ap_service.get_vendors()
		if len(existing_vendors) > 0:
			return
		
		# Create sample vendors for each type
		vendor_types = get_default_vendor_types()
		
		sample_vendors = [
			{
				'vendor_number': '001001',
				'vendor_name': 'ABC Office Supplies',
				'vendor_type': 'SUPPLIER',
				'contact_name': 'John Smith',
				'email': 'orders@abcoffice.com',
				'phone': '555-0101',
				'address_line1': '123 Business Ave',
				'city': 'Business City',
				'state_province': 'BC',
				'postal_code': '12345',
				'country': 'USA',
				'payment_terms_code': 'NET_30',
				'payment_method': 'CHECK',
				'is_active': True
			},
			{
				'vendor_number': '002001',
				'vendor_name': 'Tech Solutions Inc',
				'vendor_type': 'TECHNOLOGY',
				'contact_name': 'Jane Doe',
				'email': 'billing@techsolutions.com',
				'phone': '555-0202',
				'address_line1': '456 Tech Park',
				'city': 'Innovation City',
				'state_province': 'IC',
				'postal_code': '67890',
				'country': 'USA',
				'payment_terms_code': '2_10_NET_30',
				'payment_method': 'ACH',
				'is_active': True
			},
			{
				'vendor_number': '003001',
				'vendor_name': 'Electric Utility Co',
				'vendor_type': 'UTILITY',
				'contact_name': 'Service Department',
				'email': 'service@electricco.com',
				'phone': '555-0303',
				'address_line1': '789 Power St',
				'city': 'Utility Town',
				'state_province': 'UT',
				'postal_code': '13579',
				'country': 'USA',
				'payment_terms_code': 'DUE_ON_RECEIPT',
				'payment_method': 'ACH',
				'is_active': True
			}
		]
		
		for vendor_data in sample_vendors:
			ap_service.create_vendor(vendor_data)
		
		print(f"Default vendors created for tenant {tenant_id}")
		
	except Exception as e:
		print(f"Error creating default vendors: {e}")


def setup_ap_integration(appbuilder: AppBuilder):
	"""Set up AP integration with other modules"""
	
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
		
		print("AP-GL integration check completed")
		
	except Exception as e:
		print(f"Error setting up AP integration: {e}")


def get_ap_configuration():
	"""Get AP configuration settings"""
	
	from . import SUBCAPABILITY_META
	
	return SUBCAPABILITY_META['configuration']


def validate_ap_setup(tenant_id: str) -> Dict[str, Any]:
	"""Validate AP setup for a tenant"""
	
	from .service import AccountsPayableService
	from ..general_ledger.models import CFGLAccount
	from . import get_default_gl_account_mappings
	
	validation_results = {
		'valid': True,
		'errors': [],
		'warnings': []
	}
	
	try:
		ap_service = AccountsPayableService(tenant_id)
		
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
		
		# Check if vendors exist
		vendors = ap_service.get_vendors()
		if len(vendors) == 0:
			validation_results['warnings'].append("No vendors configured")
		
		# Check tax codes
		from .models import CFAPTaxCode
		tax_codes = CFAPTaxCode.query.filter_by(tenant_id=tenant_id).count()
		if tax_codes == 0:
			validation_results['warnings'].append("No tax codes configured")
		
	except Exception as e:
		validation_results['errors'].append(f"Validation error: {str(e)}")
		validation_results['valid'] = False
	
	return validation_results