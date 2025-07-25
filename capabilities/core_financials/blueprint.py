"""
Core Financials Capability Blueprint

Main blueprint registration for Core Financials capability and all its sub-capabilities.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import List, Dict, Any

# Import sub-capability blueprint registration functions
from .general_ledger.blueprint import init_subcapability as init_gl
from .general_ledger.api import register_api_views as register_gl_api


def register_capability_views(appbuilder: AppBuilder, subcapabilities: List[str] = None):
	"""Register Core Financials views and sub-capabilities with Flask-AppBuilder"""
	
	# If no specific sub-capabilities requested, use all implemented
	if subcapabilities is None:
		subcapabilities = [
			'general_ledger',
			'accounts_payable',
			'accounts_receivable',  
			'cash_management',
			'fixed_asset_management',
			'budgeting_forecasting',
			'financial_reporting',
			'cost_accounting'
		]
	
	# Register sub-capabilities
	for subcap in subcapabilities:
		if subcap == 'general_ledger':
			init_gl(appbuilder)
			register_gl_api(appbuilder)
		elif subcap == 'accounts_payable':
			from .accounts_payable.blueprint import init_subcapability as init_ap
			from .accounts_payable.api import register_api_views as register_ap_api
			init_ap(appbuilder)
			register_ap_api(appbuilder)
		elif subcap == 'accounts_receivable':
			from .accounts_receivable.blueprint import init_subcapability as init_ar
			from .accounts_receivable.api import register_api_views as register_ar_api
			init_ar(appbuilder)
			register_ar_api(appbuilder)
		elif subcap == 'cash_management':
			from .cash_management.blueprint import init_subcapability as init_cm
			from .cash_management.api import register_api_views as register_cm_api
			init_cm(appbuilder)
			register_cm_api(appbuilder)
		elif subcap == 'fixed_asset_management':
			from .fixed_asset_management.blueprint import init_subcapability as init_fam
			from .fixed_asset_management.api import register_api_views as register_fam_api
			init_fam(appbuilder)
			register_fam_api(appbuilder)
		elif subcap == 'budgeting_forecasting':
			from .budgeting_forecasting.blueprint import init_subcapability as init_bf
			from .budgeting_forecasting.api import register_api_views as register_bf_api
			init_bf(appbuilder)
			register_bf_api(appbuilder)
		elif subcap == 'financial_reporting':
			from .financial_reporting.blueprint import init_subcapability as init_fr
			from .financial_reporting.api import register_api_views as register_fr_api
			init_fr(appbuilder)
			register_fr_api(appbuilder)
		elif subcap == 'cost_accounting':
			from .cost_accounting.blueprint import init_subcapability as init_ca
			from .cost_accounting.api import register_api_views as register_ca_api
			init_ca(appbuilder)
			register_ca_api(appbuilder)
	
	# Create main capability dashboard if needed
	create_capability_dashboard(appbuilder)


def create_capability_dashboard(appbuilder: AppBuilder):
	"""Create main Core Financials dashboard"""
	
	from flask_appbuilder import BaseView, expose, has_access
	
	class CoreFinancialsDashboardView(BaseView):
		"""Main Core Financials Dashboard"""
		
		route_base = "/core_financials/dashboard"
		default_view = 'index'
		
		@expose('/')
		@has_access
		def index(self):
			"""Display Core Financials dashboard"""
			
			# Get summary data from all active sub-capabilities
			dashboard_data = self._get_dashboard_data()
			
			return self.render_template(
				'core_financials_dashboard.html',
				dashboard_data=dashboard_data,
				title="Core Financials Dashboard"
			)
		
		def _get_dashboard_data(self) -> Dict[str, Any]:
			"""Get dashboard data from all sub-capabilities"""
			
			data = {
				'subcapabilities': [],
				'summary': {}
			}
			
			# General Ledger summary
			try:
				from .general_ledger.service import GeneralLedgerService
				gl_service = GeneralLedgerService(self.get_tenant_id())
				
				# Get key GL metrics
				trial_balance = gl_service.generate_trial_balance()
				current_period = gl_service.get_current_period()
				unposted_count = len(gl_service.get_journal_entries(status='Draft', limit=100))
				
				data['subcapabilities'].append({
					'name': 'General Ledger',
					'status': 'active',
					'metrics': {
						'total_assets': sum(acc['debit_balance'] for acc in trial_balance['accounts'] 
										   if acc['account_type'] == 'Asset'),
						'total_liabilities': sum(acc['credit_balance'] for acc in trial_balance['accounts'] 
											    if acc['account_type'] == 'Liability'),
						'current_period': current_period.period_name if current_period else 'N/A',
						'unposted_journals': unposted_count
					}
				})
				
			except Exception as e:
				print(f"Error getting GL dashboard data: {e}")
			
			return data
		
		def get_tenant_id(self) -> str:
			"""Get current tenant ID"""
			# TODO: Implement tenant resolution
			return "default_tenant"
	
	# Register the dashboard view
	appbuilder.add_view_no_menu(CoreFinancialsDashboardView())
	appbuilder.add_link(
		"Core Financials Dashboard",
		href="/core_financials/dashboard/",
		icon="fa-dashboard",
		category="Core Financials"
	)


def create_capability_blueprint() -> Blueprint:
	"""Create Flask blueprint for Core Financials capability"""
	
	cf_bp = Blueprint(
		'core_financials',
		__name__,
		url_prefix='/core_financials',
		template_folder='templates',
		static_folder='static'
	)
	
	return cf_bp


def register_capability_permissions(appbuilder: AppBuilder):
	"""Register Core Financials capability-level permissions"""
	
	permissions = [
		# Capability-level permissions
		('can_access', 'CoreFinancials'),
		('can_view_dashboard', 'CoreFinancials'),
		
		# Cross-sub-capability permissions
		('can_view_financial_reports', 'CoreFinancials'),
		('can_manage_chart_of_accounts', 'CoreFinancials'),
		('can_post_transactions', 'CoreFinancials'),
		('can_close_periods', 'CoreFinancials'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_capability_menu_structure(subcapabilities: List[str] = None) -> Dict[str, Any]:
	"""Get complete menu structure for Core Financials capability"""
	
	if subcapabilities is None:
		subcapabilities = ['general_ledger']
	
	menu = {
		'name': 'Core Financials',
		'icon': 'fa-dollar-sign',
		'items': [
			{
				'name': 'Dashboard',
				'href': '/core_financials/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_view_dashboard on CoreFinancials'
			}
		]
	}
	
	# Add sub-capability menu items
	if 'general_ledger' in subcapabilities:
		from .general_ledger.blueprint import get_menu_structure as get_gl_menu
		gl_menu = get_gl_menu()
		menu['items'].extend(gl_menu['items'])
	
	# Add other sub-capabilities as they're implemented
	
	return menu


def validate_subcapability_dependencies(subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate that sub-capability dependencies are met"""
	
	from . import validate_composition
	return validate_composition(subcapabilities)


def init_capability(appbuilder: AppBuilder, subcapabilities: List[str] = None):
	"""Initialize Core Financials capability with specified sub-capabilities"""
	
	# Validate dependencies
	validation = validate_subcapability_dependencies(subcapabilities or ['general_ledger'])
	
	if not validation['valid']:
		raise ValueError(f"Invalid sub-capability composition: {validation['errors']}")
	
	# Register views and permissions
	register_capability_views(appbuilder, subcapabilities)
	register_capability_permissions(appbuilder)
	
	# Log warnings if any
	if validation['warnings']:
		for warning in validation['warnings']:
			print(f"Warning: {warning}")
	
	print(f"Core Financials capability initialized with sub-capabilities: {subcapabilities}")


def get_capability_info() -> Dict[str, Any]:
	"""Get Core Financials capability information"""
	
	from . import get_capability_info
	return get_capability_info()


def get_available_subcapabilities() -> List[str]:
	"""Get list of available sub-capabilities"""
	
	from . import get_subcapabilities
	return get_subcapabilities()