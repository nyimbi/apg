"""
APG Financial Management General Ledger - Flask-AppBuilder Blueprint

Comprehensive blueprint registration for enterprise general ledger including:
- Chart of Accounts management with hierarchical visualization
- Journal Entry processing with approval workflows
- Financial Reporting with real-time analytics
- Period management and closing procedures
- Multi-currency transaction handling
- Audit trail and compliance monitoring

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder, BaseView
from flask_babel import lazy_gettext as _

from .views import (
	GLAccountTypeView, GLAccountView, ChartOfAccountsView,
	GLJournalEntryView, FinancialReportsView, GLDashboardView
)


def register_views(appbuilder: AppBuilder):
	"""Register comprehensive General Ledger views with Flask-AppBuilder"""
	
	# Dashboard - Primary entry point
	appbuilder.add_view_no_menu(GLDashboardView())
	appbuilder.add_link(
		_("GL Dashboard"),
		href="/gl/dashboard/",
		icon="fa-dashboard",
		category=_("General Ledger"),
		category_icon="fa-book"
	)
	
	# Chart of Accounts Management
	appbuilder.add_view(
		GLAccountTypeView,
		_("Account Types"),
		icon="fa-tags", 
		category=_("General Ledger")
	)
	
	appbuilder.add_view(
		GLAccountView,
		_("Chart of Accounts"),
		icon="fa-list-alt",
		category=_("General Ledger")
	)
	
	appbuilder.add_view_no_menu(ChartOfAccountsView())
	appbuilder.add_link(
		_("Hierarchical Chart"),
		href="/chartofaccountsview/",
		icon="fa-sitemap",
		category=_("General Ledger")
	)
	
	# Journal Entry Management
	appbuilder.add_view(
		GLJournalEntryView,
		_("Journal Entries"),
		icon="fa-book",
		category=_("General Ledger")
	)
	
	# Financial Reporting
	appbuilder.add_view_no_menu(FinancialReportsView())
	appbuilder.add_link(
		_("Financial Reports"),
		href="/financialreportsview/",
		icon="fa-bar-chart",
		category=_("General Ledger")
	)
	
	appbuilder.add_link(
		_("Trial Balance"),
		href="/financialreportsview/trial_balance",
		icon="fa-balance-scale",
		category=_("General Ledger")
	)
	
	appbuilder.add_link(
		_("Balance Sheet"),
		href="/financialreportsview/balance_sheet",
		icon="fa-building",
		category=_("General Ledger")
	)
	
	appbuilder.add_link(
		_("Income Statement"),
		href="/financialreportsview/income_statement",
		icon="fa-line-chart",
		category=_("General Ledger")
	)


def create_blueprint() -> Blueprint:
	"""Create comprehensive Flask blueprint for General Ledger"""
	
	gl_bp = Blueprint(
		'general_ledger',
		__name__,
		url_prefix='/gl',
		template_folder='templates',
		static_folder='static'
	)
	
	return gl_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register comprehensive General Ledger permissions"""
	
	permissions = [
		# Account Type permissions
		('can_list', 'GLAccountTypeView'),
		('can_show', 'GLAccountTypeView'),
		('can_add', 'GLAccountTypeView'),
		('can_edit', 'GLAccountTypeView'),
		('can_delete', 'GLAccountTypeView'),
		
		# Account permissions
		('can_list', 'GLAccountView'),
		('can_show', 'GLAccountView'), 
		('can_add', 'GLAccountView'),
		('can_edit', 'GLAccountView'),
		('can_delete', 'GLAccountView'),
		
		# Chart of Accounts permissions
		('can_index', 'ChartOfAccountsView'),
		('can_create_account', 'ChartOfAccountsView'),
		
		# Journal Entry permissions
		('can_list', 'GLJournalEntryView'),
		('can_show', 'GLJournalEntryView'),
		('can_add', 'GLJournalEntryView'),
		('can_edit', 'GLJournalEntryView'),
		('can_delete', 'GLJournalEntryView'),
		('can_post_entries', 'GLJournalEntryView'),
		
		# Financial Reports permissions
		('can_index', 'FinancialReportsView'),
		('can_trial_balance', 'FinancialReportsView'),
		('can_balance_sheet', 'FinancialReportsView'),
		('can_income_statement', 'FinancialReportsView'),
		('can_account_ledger', 'FinancialReportsView'),
		
		# Dashboard permissions
		('can_index', 'GLDashboardView'),
		('can_dashboard_metrics_api', 'GLDashboardView'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		try:
			perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
			if not perm:
				appbuilder.sm.add_permission_view_menu(permission_name, view_name)
		except Exception as e:
			print(f"Error creating permission {permission_name} for {view_name}: {e}")


def get_menu_structure():
	"""Get comprehensive menu structure for General Ledger"""
	
	return {
		'name': _('General Ledger'),
		'icon': 'fa-book',
		'items': [
			{
				'name': _('GL Dashboard'),
				'href': '/gl/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_index on GLDashboardView',
				'description': _('General Ledger overview and key metrics')
			},
			{
				'name': _('Chart of Accounts'),
				'icon': 'fa-list-alt',
				'submenu': [
					{
						'name': _('Account Types'),
						'href': '/glaccounttypeview/list/',
						'icon': 'fa-tags',
						'permission': 'can_list on GLAccountTypeView'
					},
					{
						'name': _('Chart of Accounts'),
						'href': '/glaccountview/list/',
						'icon': 'fa-list',
						'permission': 'can_list on GLAccountView'
					},
					{
						'name': _('Hierarchical View'),
						'href': '/chartofaccountsview/',
						'icon': 'fa-sitemap',
						'permission': 'can_index on ChartOfAccountsView'
					}
				]
			},
			{
				'name': _('Transactions'),
				'icon': 'fa-exchange',
				'submenu': [
					{
						'name': _('Journal Entries'),
						'href': '/gljournalentryview/list/',
						'icon': 'fa-book',
						'permission': 'can_list on GLJournalEntryView'
					}
				]
			},
			{
				'name': _('Financial Reports'),
				'icon': 'fa-bar-chart',
				'submenu': [
					{
						'name': _('Reports Dashboard'),
						'href': '/financialreportsview/',
						'icon': 'fa-dashboard',
						'permission': 'can_index on FinancialReportsView'
					},
					{
						'name': _('Trial Balance'),
						'href': '/financialreportsview/trial_balance',
						'icon': 'fa-balance-scale',
						'permission': 'can_trial_balance on FinancialReportsView'
					},
					{
						'name': _('Balance Sheet'),
						'href': '/financialreportsview/balance_sheet',
						'icon': 'fa-building',
						'permission': 'can_balance_sheet on FinancialReportsView'
					},
					{
						'name': _('Income Statement'),
						'href': '/financialreportsview/income_statement',
						'icon': 'fa-line-chart',
						'permission': 'can_income_statement on FinancialReportsView'
					}
				]
			}
		]
	}


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize comprehensive General Ledger sub-capability"""
	
	try:
		# Register views
		register_views(appbuilder)
		print("✓ General Ledger views registered")
		
		# Register permissions
		register_permissions(appbuilder)
		print("✓ General Ledger permissions registered")
		
		# Initialize default data if needed
		_init_default_data(appbuilder)
		print("✓ General Ledger default data initialized")
		
		print("✓ General Ledger sub-capability initialization completed")
		
	except Exception as e:
		print(f"✗ Error initializing General Ledger sub-capability: {e}")
		raise


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default GL data if needed"""
	
	try:
		from .models import GLAccountType, AccountTypeEnum, BalanceTypeEnum
		from .service import GeneralLedgerService
		
		# Use default tenant for initialization
		default_tenant_id = "default_tenant"
		gl_service = GeneralLedgerService(default_tenant_id)
		
		# Check if account types already exist
		session = gl_service.get_session()
		existing_types = session.query(GLAccountType).filter_by(
			tenant_id=default_tenant_id
		).count()
		
		if existing_types == 0:
			# Create default tenant setup
			tenant_data = {
				'tenant_code': 'DEFAULT',
				'tenant_name': 'Default Tenant',
				'base_currency': 'USD',
				'reporting_framework': 'GAAP',
				'country_code': 'US'
			}
			
			# Setup tenant with default account types and periods
			gl_service.setup_tenant(tenant_data)
			print("✓ Default General Ledger tenant and data created")
		else:
			print("✓ General Ledger data already exists")
			
	except Exception as e:
		print(f"✗ Error initializing default GL data: {e}")


def get_capability_info():
	"""Get General Ledger capability information"""
	
	return {
		'name': 'Financial Management - General Ledger',
		'version': '1.0.0',
		'description': 'Enterprise-grade general ledger with multi-currency support, real-time reporting, and compliance features',
		'author': 'Nyimbi Odero <nyimbi@gmail.com>',
		'company': 'Datacraft',
		'category': 'Core Business Operations',
		'subcategory': 'Financial Management',
		'dependencies': [
			'authentication_rbac',
			'event_streaming_bus',
			'integration_api_management'
		],
		'features': [
			'Multi-tenant Chart of Accounts',
			'Hierarchical Account Structure',
			'Multi-currency Journal Entries',
			'Real-time Financial Reporting',
			'Trial Balance Generation',
			'Balance Sheet & Income Statement',
			'Account Ledger Details',
			'Period Management',
			'Audit Trail & Compliance',
			'Advanced Analytics & Ratios',
			'Responsive Web Interface',
			'REST API Integration'
		],
		'ui_routes': [
			'/gl/dashboard/',
			'/chartofaccountsview/',
			'/gljournalentryview/',
			'/financialreportsview/'
		],
		'api_endpoints': [
			'/api/v1/gl/accounts',
			'/api/v1/gl/journal-entries',
			'/api/v1/gl/reports',
			'/api/v1/gl/periods'
		]
	}