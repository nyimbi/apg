"""
Procurement & Purchasing Capability Blueprint

Main blueprint registration for Procurement & Purchasing capability and all its sub-capabilities.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import List, Dict, Any

# Import sub-capability blueprint registration functions
from .requisitioning.blueprint import init_subcapability as init_requisitioning
from .requisitioning.api import register_api_views as register_requisitioning_api

from .purchase_order_management.blueprint import init_subcapability as init_po_mgmt
from .purchase_order_management.api import register_api_views as register_po_mgmt_api

from .vendor_management.blueprint import init_subcapability as init_vendor_mgmt
from .vendor_management.api import register_api_views as register_vendor_mgmt_api

from .sourcing_supplier_selection.blueprint import init_subcapability as init_sourcing
from .sourcing_supplier_selection.api import register_api_views as register_sourcing_api

from .contract_management.blueprint import init_subcapability as init_contract_mgmt
from .contract_management.api import register_api_views as register_contract_mgmt_api


def register_capability_views(appbuilder: AppBuilder, subcapabilities: List[str] = None):
	"""Register Procurement & Purchasing views and sub-capabilities with Flask-AppBuilder"""
	
	# If no specific sub-capabilities requested, use all implemented
	if subcapabilities is None:
		subcapabilities = [
			'requisitioning',
			'purchase_order_management', 
			'vendor_management',
			'sourcing_supplier_selection',
			'contract_management'
		]
	
	# Register sub-capabilities
	for subcap in subcapabilities:
		if subcap == 'requisitioning':
			init_requisitioning(appbuilder)
			register_requisitioning_api(appbuilder)
		elif subcap == 'purchase_order_management':
			init_po_mgmt(appbuilder)
			register_po_mgmt_api(appbuilder)
		elif subcap == 'vendor_management':
			init_vendor_mgmt(appbuilder)
			register_vendor_mgmt_api(appbuilder)
		elif subcap == 'sourcing_supplier_selection':
			init_sourcing(appbuilder)
			register_sourcing_api(appbuilder)
		elif subcap == 'contract_management':
			init_contract_mgmt(appbuilder)
			register_contract_mgmt_api(appbuilder)
	
	# Create main capability dashboard
	create_capability_dashboard(appbuilder)


def create_capability_dashboard(appbuilder: AppBuilder):
	"""Create main Procurement & Purchasing dashboard"""
	
	from flask_appbuilder import BaseView, expose, has_access
	
	class ProcurementPurchasingDashboardView(BaseView):
		"""Main Procurement & Purchasing Dashboard"""
		
		route_base = "/procurement_purchasing/dashboard"
		default_view = 'index'
		
		@expose('/')
		@has_access
		def index(self):
			"""Display Procurement & Purchasing dashboard"""
			
			# Get summary data from all active sub-capabilities
			dashboard_data = self._get_dashboard_data()
			
			return self.render_template(
				'procurement_purchasing_dashboard.html',
				dashboard_data=dashboard_data,
				title="Procurement & Purchasing Dashboard"
			)
		
		def _get_dashboard_data(self) -> Dict[str, Any]:
			"""Get dashboard data from all sub-capabilities"""
			
			data = {
				'subcapabilities': [],
				'summary': {},
				'key_metrics': {},
				'alerts': [],
				'recent_activity': []
			}
			
			tenant_id = self.get_tenant_id()
			
			# Requisitioning summary
			try:
				from .requisitioning.service import RequisitioningService
				req_service = RequisitioningService(tenant_id)
				
				pending_reqs = req_service.get_requisitions_by_status('Pending')
				draft_reqs = req_service.get_requisitions_by_status('Draft')
				approved_reqs = req_service.get_requisitions_by_status('Approved')
				
				data['subcapabilities'].append({
					'name': 'Requisitioning',
					'status': 'active',
					'metrics': {
						'pending_requisitions': len(pending_reqs),
						'draft_requisitions': len(draft_reqs),
						'approved_this_month': len(approved_reqs),
						'avg_approval_time': req_service.get_avg_approval_time()
					}
				})
				
				# Add alerts for overdue approvals
				overdue = req_service.get_overdue_approvals()
				if overdue:
					data['alerts'].append({
						'type': 'warning',
						'message': f'{len(overdue)} requisitions have overdue approvals',
						'action_url': '/requisitioning/overdue_approvals'
					})
				
			except Exception as e:
				print(f"Error getting Requisitioning dashboard data: {e}")
			
			# Purchase Order Management summary
			try:
				from .purchase_order_management.service import PurchaseOrderService
				po_service = PurchaseOrderService(tenant_id)
				
				open_pos = po_service.get_purchase_orders_by_status('Open')
				pending_receipt = po_service.get_purchase_orders_needing_receipt()
				total_po_value = po_service.get_total_po_value_ytd()
				
				data['subcapabilities'].append({
					'name': 'Purchase Order Management',
					'status': 'active',
					'metrics': {
						'open_purchase_orders': len(open_pos),
						'pending_receipt': len(pending_receipt),
						'total_po_value_ytd': total_po_value,
						'avg_po_processing_time': po_service.get_avg_processing_time()
					}
				})
				
				# Add alerts for overdue receipts
				overdue_receipts = po_service.get_overdue_receipts()
				if overdue_receipts:
					data['alerts'].append({
						'type': 'error',
						'message': f'{len(overdue_receipts)} purchase orders have overdue receipts',
						'action_url': '/purchase_orders/overdue_receipts'
					})
				
			except Exception as e:
				print(f"Error getting Purchase Order dashboard data: {e}")
			
			# Vendor Management summary
			try:
				from .vendor_management.service import VendorManagementService
				vendor_service = VendorManagementService(tenant_id)
				
				active_vendors = vendor_service.get_active_vendor_count()
				top_vendors = vendor_service.get_top_vendors_by_spend(limit=5)
				avg_performance = vendor_service.get_average_performance_score()
				
				data['subcapabilities'].append({
					'name': 'Vendor Management',
					'status': 'active',
					'metrics': {
						'active_vendors': active_vendors,
						'top_vendor_spend': sum(v['spend'] for v in top_vendors),
						'avg_performance_score': avg_performance,
						'new_vendors_this_month': vendor_service.get_new_vendor_count()
					}
				})
				
				# Add alerts for poor performing vendors
				poor_performers = vendor_service.get_poor_performing_vendors()
				if poor_performers:
					data['alerts'].append({
						'type': 'warning',
						'message': f'{len(poor_performers)} vendors have performance issues',
						'action_url': '/vendors/performance_issues'
					})
				
			except Exception as e:
				print(f"Error getting Vendor Management dashboard data: {e}")
			
			# Contract Management summary
			try:
				from .contract_management.service import ContractManagementService
				contract_service = ContractManagementService(tenant_id)
				
				active_contracts = contract_service.get_active_contract_count()
				expiring_soon = contract_service.get_contracts_expiring_soon(days=30)
				total_contract_value = contract_service.get_total_contract_value()
				
				data['subcapabilities'].append({
					'name': 'Contract Management',
					'status': 'active',
					'metrics': {
						'active_contracts': active_contracts,
						'expiring_soon': len(expiring_soon),
						'total_contract_value': total_contract_value,
						'renewal_rate': contract_service.get_renewal_rate()
					}
				})
				
				# Add alerts for expiring contracts
				if expiring_soon:
					data['alerts'].append({
						'type': 'info',
						'message': f'{len(expiring_soon)} contracts expire within 30 days',
						'action_url': '/contracts/expiring_soon'
					})
				
			except Exception as e:
				print(f"Error getting Contract Management dashboard data: {e}")
			
			# Calculate summary metrics
			data['summary'] = {
				'total_spend_ytd': sum(
					sub.get('metrics', {}).get('total_po_value_ytd', 0) 
					for sub in data['subcapabilities']
				),
				'active_vendors': sum(
					sub.get('metrics', {}).get('active_vendors', 0) 
					for sub in data['subcapabilities']
				),
				'pending_approvals': sum(
					sub.get('metrics', {}).get('pending_requisitions', 0) 
					for sub in data['subcapabilities']
				),
				'open_pos': sum(
					sub.get('metrics', {}).get('open_purchase_orders', 0) 
					for sub in data['subcapabilities']
				)
			}
			
			return data
		
		def get_tenant_id(self) -> str:
			"""Get current tenant ID"""
			# TODO: Implement tenant resolution
			return "default_tenant"
	
	# Register the dashboard view
	appbuilder.add_view_no_menu(ProcurementPurchasingDashboardView())
	appbuilder.add_link(
		"Procurement & Purchasing Dashboard",
		href="/procurement_purchasing/dashboard/",
		icon="fa-shopping-cart",
		category="Procurement & Purchasing"
	)


def create_capability_blueprint() -> Blueprint:
	"""Create Flask blueprint for Procurement & Purchasing capability"""
	
	pp_bp = Blueprint(
		'procurement_purchasing',
		__name__,
		url_prefix='/procurement_purchasing',
		template_folder='templates',
		static_folder='static'
	)
	
	return pp_bp


def register_capability_permissions(appbuilder: AppBuilder):
	"""Register Procurement & Purchasing capability-level permissions"""
	
	permissions = [
		# Capability-level permissions
		('can_access', 'ProcurementPurchasing'),
		('can_view_dashboard', 'ProcurementPurchasing'),
		
		# Cross-sub-capability permissions
		('can_view_procurement_reports', 'ProcurementPurchasing'),
		('can_manage_approval_workflows', 'ProcurementPurchasing'),
		('can_override_approvals', 'ProcurementPurchasing'),
		('can_manage_vendor_relationships', 'ProcurementPurchasing'),
		('can_execute_strategic_sourcing', 'ProcurementPurchasing'),
		('can_manage_contracts', 'ProcurementPurchasing'),
		('can_view_spend_analytics', 'ProcurementPurchasing'),
		('can_manage_procurement_policies', 'ProcurementPurchasing'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_capability_menu_structure(subcapabilities: List[str] = None) -> Dict[str, Any]:
	"""Get complete menu structure for Procurement & Purchasing capability"""
	
	if subcapabilities is None:
		subcapabilities = [
			'requisitioning', 'purchase_order_management', 'vendor_management',
			'sourcing_supplier_selection', 'contract_management'
		]
	
	menu = {
		'name': 'Procurement & Purchasing',
		'icon': 'fa-shopping-cart',
		'items': [
			{
				'name': 'Dashboard',
				'href': '/procurement_purchasing/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_view_dashboard on ProcurementPurchasing'
			}
		]
	}
	
	# Add sub-capability menu items
	if 'requisitioning' in subcapabilities:
		from .requisitioning.blueprint import get_menu_structure as get_req_menu
		req_menu = get_req_menu()
		menu['items'].extend(req_menu['items'])
	
	if 'purchase_order_management' in subcapabilities:
		from .purchase_order_management.blueprint import get_menu_structure as get_po_menu
		po_menu = get_po_menu()
		menu['items'].extend(po_menu['items'])
	
	if 'vendor_management' in subcapabilities:
		from .vendor_management.blueprint import get_menu_structure as get_vendor_menu
		vendor_menu = get_vendor_menu()
		menu['items'].extend(vendor_menu['items'])
	
	if 'sourcing_supplier_selection' in subcapabilities:
		from .sourcing_supplier_selection.blueprint import get_menu_structure as get_sourcing_menu
		sourcing_menu = get_sourcing_menu()
		menu['items'].extend(sourcing_menu['items'])
	
	if 'contract_management' in subcapabilities:
		from .contract_management.blueprint import get_menu_structure as get_contract_menu
		contract_menu = get_contract_menu()
		menu['items'].extend(contract_menu['items'])
	
	return menu


def validate_subcapability_dependencies(subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate that sub-capability dependencies are met"""
	
	from . import validate_composition
	return validate_composition(subcapabilities)


def init_capability(appbuilder: AppBuilder, subcapabilities: List[str] = None):
	"""Initialize Procurement & Purchasing capability with specified sub-capabilities"""
	
	# Validate dependencies
	validation = validate_subcapability_dependencies(subcapabilities or ['vendor_management'])
	
	if not validation['valid']:
		raise ValueError(f"Invalid sub-capability composition: {validation['errors']}")
	
	# Register views and permissions
	register_capability_views(appbuilder, subcapabilities)
	register_capability_permissions(appbuilder)
	
	# Log warnings if any
	if validation['warnings']:
		for warning in validation['warnings']:
			print(f"Warning: {warning}")
	
	print(f"Procurement & Purchasing capability initialized with sub-capabilities: {subcapabilities}")


def get_capability_info() -> Dict[str, Any]:
	"""Get Procurement & Purchasing capability information"""
	
	from . import get_capability_info
	return get_capability_info()


def get_available_subcapabilities() -> List[str]:
	"""Get list of available sub-capabilities"""
	
	from . import get_subcapabilities
	return get_subcapabilities()