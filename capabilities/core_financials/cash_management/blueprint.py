"""
Cash Management Blueprint

Flask blueprint registration for Cash Management sub-capability.
Registers all views, API endpoints, and URL routes.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	CFCMBankAccountModelView, CFCMBankTransactionModelView, CFCMReconciliationModelView,
	CFCMCashForecastModelView, CFCMCashPositionModelView, CFCMInvestmentModelView,
	CFCMCashTransferModelView, CFCMCheckRegisterModelView, CFCMDashboardView
)


def register_views(appbuilder: AppBuilder):
	"""Register Cash Management views with Flask-AppBuilder"""
	
	# Bank Account Management
	appbuilder.add_view(
		CFCMBankAccountModelView,
		"Bank Accounts",
		icon="fa-university",
		category="Cash Management",
		category_icon="fa-money"
	)
	
	# Transaction Management
	appbuilder.add_view(
		CFCMBankTransactionModelView,
		"Bank Transactions",
		icon="fa-exchange",
		category="Cash Management"
	)
	
	# Bank Reconciliation
	appbuilder.add_view(
		CFCMReconciliationModelView,
		"Bank Reconciliation",
		icon="fa-check-square",
		category="Cash Management"
	)
	
	# Cash Forecasting
	appbuilder.add_view(
		CFCMCashForecastModelView,
		"Cash Forecast",
		icon="fa-line-chart",
		category="Cash Management"
	)
	
	# Cash Positions
	appbuilder.add_view(
		CFCMCashPositionModelView,
		"Cash Position",
		icon="fa-money",
		category="Cash Management"
	)
	
	# Investment Management
	appbuilder.add_view(
		CFCMInvestmentModelView,
		"Investments",
		icon="fa-chart-line",
		category="Cash Management"
	)
	
	# Cash Transfers
	appbuilder.add_view(
		CFCMCashTransferModelView,
		"Cash Transfers",
		icon="fa-arrows-h",
		category="Cash Management"
	)
	
	# Check Register
	appbuilder.add_view(
		CFCMCheckRegisterModelView,
		"Check Register",
		icon="fa-list",
		category="Cash Management"
	)
	
	# Dashboard
	appbuilder.add_view_no_menu(CFCMDashboardView())
	appbuilder.add_link(
		"Cash Dashboard",
		href="/cash_management/dashboard/",
		icon="fa-dashboard",
		category="Cash Management"
	)


def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for Cash Management"""
	
	cm_bp = Blueprint(
		'cash_management',
		__name__,
		url_prefix='/cm',
		template_folder='templates',
		static_folder='static'
	)
	
	return cm_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register Cash Management permissions"""
	
	permissions = [
		# Bank Account permissions
		('can_list', 'CFCMBankAccountModelView'),
		('can_show', 'CFCMBankAccountModelView'),
		('can_add', 'CFCMBankAccountModelView'),
		('can_edit', 'CFCMBankAccountModelView'),
		('can_delete', 'CFCMBankAccountModelView'),
		('can_reconcile', 'CFCMBankAccountModelView'),
		('can_balance_summary', 'CFCMBankAccountModelView'),
		
		# Bank Transaction permissions
		('can_list', 'CFCMBankTransactionModelView'),
		('can_show', 'CFCMBankTransactionModelView'),
		('can_add', 'CFCMBankTransactionModelView'),
		('can_edit', 'CFCMBankTransactionModelView'),
		('can_delete', 'CFCMBankTransactionModelView'),
		('can_mark_reconciled', 'CFCMBankTransactionModelView'),
		('can_import_form', 'CFCMBankTransactionModelView'),
		
		# Bank Reconciliation permissions
		('can_list', 'CFCMReconciliationModelView'),
		('can_show', 'CFCMReconciliationModelView'),
		('can_add', 'CFCMReconciliationModelView'),
		('can_edit', 'CFCMReconciliationModelView'),
		('can_delete', 'CFCMReconciliationModelView'),
		('can_auto_match', 'CFCMReconciliationModelView'),
		('can_complete', 'CFCMReconciliationModelView'),
		('can_reconcile_interface', 'CFCMReconciliationModelView'),
		
		# Cash Forecast permissions
		('can_list', 'CFCMCashForecastModelView'),
		('can_show', 'CFCMCashForecastModelView'),
		('can_add', 'CFCMCashForecastModelView'),
		('can_edit', 'CFCMCashForecastModelView'),
		('can_delete', 'CFCMCashForecastModelView'),
		('can_generate_forecast', 'CFCMCashForecastModelView'),
		('can_forecast_accuracy', 'CFCMCashForecastModelView'),
		
		# Cash Position permissions
		('can_list', 'CFCMCashPositionModelView'),
		('can_show', 'CFCMCashPositionModelView'),
		('can_add', 'CFCMCashPositionModelView'),
		('can_edit', 'CFCMCashPositionModelView'),
		('can_position_summary', 'CFCMCashPositionModelView'),
		
		# Investment permissions
		('can_list', 'CFCMInvestmentModelView'),
		('can_show', 'CFCMInvestmentModelView'),
		('can_add', 'CFCMInvestmentModelView'),
		('can_edit', 'CFCMInvestmentModelView'),
		('can_delete', 'CFCMInvestmentModelView'),
		('can_maturing_investments', 'CFCMInvestmentModelView'),
		
		# Cash Transfer permissions
		('can_list', 'CFCMCashTransferModelView'),
		('can_show', 'CFCMCashTransferModelView'),
		('can_add', 'CFCMCashTransferModelView'),
		('can_edit', 'CFCMCashTransferModelView'),
		('can_delete', 'CFCMCashTransferModelView'),
		('can_approve', 'CFCMCashTransferModelView'),
		('can_submit', 'CFCMCashTransferModelView'),
		
		# Check Register permissions
		('can_list', 'CFCMCheckRegisterModelView'),
		('can_show', 'CFCMCheckRegisterModelView'),
		('can_add', 'CFCMCheckRegisterModelView'),
		('can_edit', 'CFCMCheckRegisterModelView'),
		('can_delete', 'CFCMCheckRegisterModelView'),
		('can_void', 'CFCMCheckRegisterModelView'),
		('can_stop_payment', 'CFCMCheckRegisterModelView'),
		('can_outstanding_checks', 'CFCMCheckRegisterModelView'),
		
		# Dashboard permissions
		('can_dashboard', 'CFCMDashboardView'),
		('can_cash_flow_chart', 'CFCMDashboardView'),
		('can_liquidity_analysis', 'CFCMDashboardView'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for Cash Management"""
	
	return {
		'name': 'Cash Management',
		'icon': 'fa-money',
		'items': [
			{
				'name': 'Cash Dashboard',
				'href': '/cash_management/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_dashboard on CFCMDashboardView'
			},
			{
				'name': 'Bank Accounts',
				'href': '/cfcmbankaccountmodelview/list/',
				'icon': 'fa-university',
				'permission': 'can_list on CFCMBankAccountModelView'
			},
			{
				'name': 'Bank Transactions',
				'href': '/cfcmbanktransactionmodelview/list/',
				'icon': 'fa-exchange',
				'permission': 'can_list on CFCMBankTransactionModelView'
			},
			{
				'name': 'Bank Reconciliation',
				'href': '/cfcmreconciliationmodelview/list/',
				'icon': 'fa-check-square',
				'permission': 'can_list on CFCMReconciliationModelView'
			},
			{
				'name': 'Cash Forecast',
				'href': '/cfcmcashforecastmodelview/list/',
				'icon': 'fa-line-chart',
				'permission': 'can_list on CFCMCashForecastModelView'
			},
			{
				'name': 'Cash Position',
				'href': '/cfcmcashpositionmodelview/list/',
				'icon': 'fa-money',
				'permission': 'can_list on CFCMCashPositionModelView'
			},
			{
				'name': 'Investments',
				'href': '/cfcminvestmentmodelview/list/',
				'icon': 'fa-chart-line',
				'permission': 'can_list on CFCMInvestmentModelView'
			},
			{
				'name': 'Cash Transfers',
				'href': '/cfcmcashtransfermodelview/list/',
				'icon': 'fa-arrows-h',
				'permission': 'can_list on CFCMCashTransferModelView'
			},
			{
				'name': 'Check Register',
				'href': '/cfcmcheckregistermodelview/list/',
				'icon': 'fa-list',
				'permission': 'can_list on CFCMCheckRegisterModelView'
			}
		]
	}


def register_roles(appbuilder: AppBuilder):
	"""Register Cash Management roles"""
	
	roles = [
		{
			'name': 'Cash Manager',
			'permissions': [
				# Full access to all cash management functions
				'can_list on CFCMBankAccountModelView',
				'can_show on CFCMBankAccountModelView',
				'can_add on CFCMBankAccountModelView',
				'can_edit on CFCMBankAccountModelView',
				'can_reconcile on CFCMBankAccountModelView',
				'can_list on CFCMBankTransactionModelView',
				'can_show on CFCMBankTransactionModelView',
				'can_add on CFCMBankTransactionModelView',
				'can_edit on CFCMBankTransactionModelView',
				'can_list on CFCMReconciliationModelView',
				'can_show on CFCMReconciliationModelView',
				'can_add on CFCMReconciliationModelView',
				'can_edit on CFCMReconciliationModelView',
				'can_complete on CFCMReconciliationModelView',
				'can_approve on CFCMCashTransferModelView',
				'can_submit on CFCMCashTransferModelView',
				'can_dashboard on CFCMDashboardView'
			]
		},
		{
			'name': 'Cash Clerk',
			'permissions': [
				# Limited access for data entry and basic operations
				'can_list on CFCMBankAccountModelView',
				'can_show on CFCMBankAccountModelView',
				'can_list on CFCMBankTransactionModelView',
				'can_show on CFCMBankTransactionModelView',
				'can_add on CFCMBankTransactionModelView',
				'can_list on CFCMReconciliationModelView',
				'can_show on CFCMReconciliationModelView',
				'can_add on CFCMReconciliationModelView',
				'can_list on CFCMCheckRegisterModelView',
				'can_show on CFCMCheckRegisterModelView',
				'can_add on CFCMCheckRegisterModelView',
				'can_dashboard on CFCMDashboardView'
			]
		},
		{
			'name': 'Treasury Analyst',
			'permissions': [
				# Focus on forecasting and investment management
				'can_list on CFCMBankAccountModelView',
				'can_show on CFCMBankAccountModelView',
				'can_list on CFCMCashForecastModelView',
				'can_show on CFCMCashForecastModelView',
				'can_add on CFCMCashForecastModelView',
				'can_edit on CFCMCashForecastModelView',
				'can_generate_forecast on CFCMCashForecastModelView',
				'can_list on CFCMInvestmentModelView',
				'can_show on CFCMInvestmentModelView',
				'can_add on CFCMInvestmentModelView',
				'can_edit on CFCMInvestmentModelView',
				'can_list on CFCMCashPositionModelView',
				'can_show on CFCMCashPositionModelView',
				'can_dashboard on CFCMDashboardView',
				'can_liquidity_analysis on CFCMDashboardView'
			]
		},
		{
			'name': 'Cash Viewer',
			'permissions': [
				# Read-only access for reporting
				'can_list on CFCMBankAccountModelView',
				'can_show on CFCMBankAccountModelView',
				'can_list on CFCMBankTransactionModelView',
				'can_show on CFCMBankTransactionModelView',
				'can_list on CFCMReconciliationModelView',
				'can_show on CFCMReconciliationModelView',
				'can_list on CFCMCashForecastModelView',
				'can_show on CFCMCashForecastModelView',
				'can_list on CFCMCashPositionModelView',
				'can_show on CFCMCashPositionModelView',
				'can_dashboard on CFCMDashboardView'
			]
		}
	]
	
	# Create roles and assign permissions
	for role_data in roles:
		role = appbuilder.sm.find_role(role_data['name'])
		if not role:
			role = appbuilder.sm.add_role(role_data['name'])
		
		# Clear existing permissions and add new ones
		role.permissions = []
		
		for perm_name in role_data['permissions']:
			if ' on ' in perm_name:
				permission_name, view_name = perm_name.split(' on ')
				perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
				if perm and perm not in role.permissions:
					role.permissions.append(perm)
		
		appbuilder.sm.get_session.commit()


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Cash Management sub-capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Register roles
	register_roles(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default Cash Management data if needed"""
	
	from .models import CFCMCurrencyRate
	from ...auth_rbac.models import db
	from datetime import date
	from decimal import Decimal
	
	# Create default currency rates if they don't exist
	default_rates = [
		{
			'from_currency': 'USD',
			'to_currency': 'EUR',
			'exchange_rate': Decimal('0.85'),
			'rate_source': 'System Default',
			'rate_date': date.today()
		},
		{
			'from_currency': 'EUR',
			'to_currency': 'USD',
			'exchange_rate': Decimal('1.18'),
			'rate_source': 'System Default',
			'rate_date': date.today()
		},
		{
			'from_currency': 'USD',
			'to_currency': 'GBP',
			'exchange_rate': Decimal('0.73'),
			'rate_source': 'System Default',
			'rate_date': date.today()
		},
		{
			'from_currency': 'GBP',
			'to_currency': 'USD',
			'exchange_rate': Decimal('1.37'),
			'rate_source': 'System Default',
			'rate_date': date.today()
		},
		{
			'from_currency': 'USD',
			'to_currency': 'CAD',
			'exchange_rate': Decimal('1.25'),
			'rate_source': 'System Default',
			'rate_date': date.today()
		},
		{
			'from_currency': 'CAD',
			'to_currency': 'USD',
			'exchange_rate': Decimal('0.80'),
			'rate_source': 'System Default',
			'rate_date': date.today()
		}
	]
	
	try:
		# Check if currency rates already exist (use a default tenant for now)
		existing_rates = CFCMCurrencyRate.query.filter_by(
			tenant_id='default_tenant',
			rate_date=date.today()
		).count()
		
		if existing_rates == 0:
			for rate_data in default_rates:
				rate = CFCMCurrencyRate(
					tenant_id='default_tenant',
					**rate_data
				)
				rate.calculate_inverse_rate()
				db.session.add(rate)
			
			db.session.commit()
			print("Default currency rates created")
			
	except Exception as e:
		print(f"Error initializing default Cash Management data: {e}")
		db.session.rollback()


def register_api_endpoints(api):
	"""Register API endpoints for Cash Management"""
	
	from .api import (
		BankAccountListAPI, BankAccountAPI, BankTransactionListAPI,
		ReconciliationListAPI, ReconciliationAPI, CashForecastListAPI,
		CashPositionListAPI, InvestmentListAPI, CashTransferListAPI,
		CheckRegisterListAPI, CashFlowReportAPI, DashboardAPI
	)
	
	# Bank Accounts
	api.add_resource(BankAccountListAPI, '/api/cm/bank-accounts')
	api.add_resource(BankAccountAPI, '/api/cm/bank-accounts/<account_id>')
	
	# Bank Transactions
	api.add_resource(BankTransactionListAPI, '/api/cm/transactions')
	
	# Reconciliations
	api.add_resource(ReconciliationListAPI, '/api/cm/reconciliations')
	api.add_resource(ReconciliationAPI, '/api/cm/reconciliations/<reconciliation_id>')
	
	# Cash Forecasting
	api.add_resource(CashForecastListAPI, '/api/cm/cash-forecast')
	
	# Cash Positions
	api.add_resource(CashPositionListAPI, '/api/cm/cash-position')
	
	# Investments
	api.add_resource(InvestmentListAPI, '/api/cm/investments')
	
	# Cash Transfers
	api.add_resource(CashTransferListAPI, '/api/cm/transfers')
	
	# Check Register
	api.add_resource(CheckRegisterListAPI, '/api/cm/checks')
	
	# Reports
	api.add_resource(CashFlowReportAPI, '/api/cm/reports/cash-flow')
	api.add_resource(DashboardAPI, '/api/cm/dashboard')


def get_api_documentation():
	"""Get API documentation for Cash Management"""
	
	return {
		'title': 'Cash Management API',
		'version': '1.0.0',
		'description': 'API endpoints for Cash Management operations',
		'endpoints': [
			{
				'path': '/api/cm/bank-accounts',
				'methods': ['GET', 'POST'],
				'description': 'Bank account management'
			},
			{
				'path': '/api/cm/transactions',
				'methods': ['GET', 'POST'],
				'description': 'Bank transaction management'
			},
			{
				'path': '/api/cm/reconciliations',
				'methods': ['GET', 'POST'],
				'description': 'Bank reconciliation management'
			},
			{
				'path': '/api/cm/cash-forecast',
				'methods': ['GET', 'POST'],
				'description': 'Cash flow forecasting'
			},
			{
				'path': '/api/cm/cash-position',
				'methods': ['GET'],
				'description': 'Cash position reporting'
			},
			{
				'path': '/api/cm/investments',
				'methods': ['GET', 'POST'],
				'description': 'Investment management'
			},
			{
				'path': '/api/cm/transfers',
				'methods': ['GET', 'POST'],
				'description': 'Cash transfer management'
			},
			{
				'path': '/api/cm/checks',
				'methods': ['GET', 'POST'],
				'description': 'Check register management'
			},
			{
				'path': '/api/cm/reports/cash-flow',
				'methods': ['GET'],
				'description': 'Cash flow reporting'
			},
			{
				'path': '/api/cm/dashboard',
				'methods': ['GET'],
				'description': 'Cash management dashboard data'
			}
		]
	}