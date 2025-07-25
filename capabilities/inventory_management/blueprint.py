"""
Inventory Management Capability Blueprint

Main blueprint registration for Inventory Management capability and all its sub-capabilities.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import List, Dict, Any


def register_capability_views(appbuilder: AppBuilder, subcapabilities: List[str] = None):
	"""Register Inventory Management views and sub-capabilities with Flask-AppBuilder"""
	
	# If no specific sub-capabilities requested, use all implemented
	if subcapabilities is None:
		subcapabilities = [
			'stock_tracking_control',
			'replenishment_reordering',
			'batch_lot_tracking',
			'expiry_date_management'
		]
	
	# Register sub-capabilities
	for subcap in subcapabilities:
		if subcap == 'stock_tracking_control':
			from .stock_tracking_control.blueprint import init_subcapability as init_stc
			from .stock_tracking_control.api import register_api_views as register_stc_api
			init_stc(appbuilder)
			register_stc_api(appbuilder)
		elif subcap == 'replenishment_reordering':
			from .replenishment_reordering.blueprint import init_subcapability as init_rr
			from .replenishment_reordering.api import register_api_views as register_rr_api
			init_rr(appbuilder)
			register_rr_api(appbuilder)
		elif subcap == 'batch_lot_tracking':
			from .batch_lot_tracking.blueprint import init_subcapability as init_blt
			from .batch_lot_tracking.api import register_api_views as register_blt_api
			init_blt(appbuilder)
			register_blt_api(appbuilder)
		elif subcap == 'expiry_date_management':
			from .expiry_date_management.blueprint import init_subcapability as init_edm
			from .expiry_date_management.api import register_api_views as register_edm_api
			init_edm(appbuilder)
			register_edm_api(appbuilder)
	
	# Create main capability dashboard
	create_capability_dashboard(appbuilder)


def create_capability_dashboard(appbuilder: AppBuilder):
	"""Create main Inventory Management dashboard"""
	
	from flask_appbuilder import BaseView, expose, has_access
	from datetime import datetime, timedelta
	
	class InventoryManagementDashboardView(BaseView):
		"""Main Inventory Management Dashboard"""
		
		route_base = "/inventory_management/dashboard"
		default_view = 'index'
		
		@expose('/')
		@has_access
		def index(self):
			"""Display Inventory Management dashboard"""
			
			# Get summary data from all active sub-capabilities
			dashboard_data = self._get_dashboard_data()
			
			return self.render_template(
				'inventory_management_dashboard.html',
				dashboard_data=dashboard_data,
				title="Inventory Management Dashboard"
			)
		
		def _get_dashboard_data(self) -> Dict[str, Any]:
			"""Get dashboard data from all sub-capabilities"""
			
			data = {
				'subcapabilities': [],
				'summary': {},
				'alerts': [],
				'kpis': {}
			}
			
			# Stock Tracking & Control summary
			try:
				from .stock_tracking_control.service import StockTrackingService
				stc_service = StockTrackingService(self.get_tenant_id())
				
				# Get key inventory metrics
				total_items = stc_service.get_total_item_count()
				low_stock_items = stc_service.get_low_stock_items(limit=100)
				total_value = stc_service.get_total_inventory_value()
				locations_count = stc_service.get_locations_count()
				
				data['subcapabilities'].append({
					'name': 'Stock Tracking & Control',
					'status': 'active',
					'metrics': {
						'total_items': total_items,
						'low_stock_items': len(low_stock_items),
						'total_value': total_value,
						'locations': locations_count
					}
				})
				
				data['kpis']['inventory_turnover'] = stc_service.calculate_inventory_turnover()
				data['kpis']['stockout_rate'] = stc_service.calculate_stockout_rate()
				
				# Add low stock alerts
				for item in low_stock_items[:5]:  # Top 5 low stock alerts
					data['alerts'].append({
						'type': 'warning',
						'category': 'Low Stock',
						'message': f"{item['item_name']} is below minimum level ({item['current_stock']}/{item['min_level']})",
						'priority': 'high' if item['current_stock'] == 0 else 'medium'
					})
				
			except Exception as e:
				print(f"Error getting Stock Tracking dashboard data: {e}")
			
			# Replenishment & Reordering summary
			try:
				from .replenishment_reordering.service import ReplenishmentService
				rr_service = ReplenishmentService(self.get_tenant_id())
				
				pending_orders = rr_service.get_pending_purchase_orders_count()
				auto_reorder_items = rr_service.get_auto_reorder_items_count()
				overdue_orders = rr_service.get_overdue_orders_count()
				
				data['subcapabilities'].append({
					'name': 'Replenishment & Reordering',
					'status': 'active',
					'metrics': {
						'pending_orders': pending_orders,
						'auto_reorder_items': auto_reorder_items,
						'overdue_orders': overdue_orders
					}
				})
				
				# Add overdue order alerts
				if overdue_orders > 0:
					data['alerts'].append({
						'type': 'danger',
						'category': 'Overdue Orders',
						'message': f"{overdue_orders} purchase orders are overdue",
						'priority': 'high'
					})
				
			except Exception as e:
				print(f"Error getting Replenishment dashboard data: {e}")
			
			# Batch & Lot Tracking summary
			try:
				from .batch_lot_tracking.service import BatchLotService
				blt_service = BatchLotService(self.get_tenant_id())
				
				active_batches = blt_service.get_active_batches_count()
				quarantined_lots = blt_service.get_quarantined_lots_count()
				recall_eligible = blt_service.get_recall_eligible_lots_count()
				
				data['subcapabilities'].append({
					'name': 'Batch & Lot Tracking',
					'status': 'active',
					'metrics': {
						'active_batches': active_batches,
						'quarantined_lots': quarantined_lots,
						'recall_eligible': recall_eligible
					}
				})
				
				# Add quarantine alerts
				if quarantined_lots > 0:
					data['alerts'].append({
						'type': 'warning',
						'category': 'Quality Control',
						'message': f"{quarantined_lots} lots are currently quarantined",
						'priority': 'high'
					})
				
			except Exception as e:
				print(f"Error getting Batch & Lot dashboard data: {e}")
			
			# Expiry Date Management summary
			try:
				from .expiry_date_management.service import ExpiryDateService
				edm_service = ExpiryDateService(self.get_tenant_id())
				
				expiring_soon = edm_service.get_items_expiring_soon_count(days=30)
				expired_items = edm_service.get_expired_items_count()
				waste_value = edm_service.get_waste_value_current_month()
				
				data['subcapabilities'].append({
					'name': 'Expiry Date Management',
					'status': 'active',
					'metrics': {
						'expiring_soon': expiring_soon,
						'expired_items': expired_items,
						'waste_value': waste_value
					}
				})
				
				# Add expiry alerts
				if expired_items > 0:
					data['alerts'].append({
						'type': 'danger',
						'category': 'Expired Items',
						'message': f"{expired_items} items have expired and require disposal",
						'priority': 'high'
					})
				
				if expiring_soon > 0:
					data['alerts'].append({
						'type': 'warning',
						'category': 'Expiring Soon',
						'message': f"{expiring_soon} items expire within 30 days",
						'priority': 'medium'
					})
				
			except Exception as e:
				print(f"Error getting Expiry Date dashboard data: {e}")
			
			return data
		
		def get_tenant_id(self) -> str:
			"""Get current tenant ID"""
			# TODO: Implement tenant resolution
			return "default_tenant"
	
	# Register the dashboard view
	appbuilder.add_view_no_menu(InventoryManagementDashboardView())
	appbuilder.add_link(
		"Inventory Dashboard",
		href="/inventory_management/dashboard/",
		icon="fa-dashboard",
		category="Inventory Management"
	)


def create_capability_blueprint() -> Blueprint:
	"""Create Flask blueprint for Inventory Management capability"""
	
	im_bp = Blueprint(
		'inventory_management',
		__name__,
		url_prefix='/inventory_management',
		template_folder='templates',
		static_folder='static'
	)
	
	return im_bp


def register_capability_permissions(appbuilder: AppBuilder):
	"""Register Inventory Management capability-level permissions"""
	
	permissions = [
		# Capability-level permissions
		('can_access', 'InventoryManagement'),
		('can_view_dashboard', 'InventoryManagement'),
		
		# Cross-sub-capability permissions
		('can_view_inventory_reports', 'InventoryManagement'),
		('can_manage_inventory_settings', 'InventoryManagement'),
		('can_perform_stock_adjustments', 'InventoryManagement'),
		('can_initiate_recalls', 'InventoryManagement'),
		('can_override_safety_limits', 'InventoryManagement'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_capability_menu_structure(subcapabilities: List[str] = None) -> Dict[str, Any]:
	"""Get complete menu structure for Inventory Management capability"""
	
	if subcapabilities is None:
		subcapabilities = ['stock_tracking_control']
	
	menu = {
		'name': 'Inventory Management',
		'icon': 'fa-boxes',
		'items': [
			{
				'name': 'Dashboard',
				'href': '/inventory_management/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_view_dashboard on InventoryManagement'
			}
		]
	}
	
	# Add sub-capability menu items
	if 'stock_tracking_control' in subcapabilities:
		from .stock_tracking_control.blueprint import get_menu_structure as get_stc_menu
		stc_menu = get_stc_menu()
		menu['items'].extend(stc_menu['items'])
	
	if 'replenishment_reordering' in subcapabilities:
		from .replenishment_reordering.blueprint import get_menu_structure as get_rr_menu
		rr_menu = get_rr_menu()
		menu['items'].extend(rr_menu['items'])
	
	if 'batch_lot_tracking' in subcapabilities:
		from .batch_lot_tracking.blueprint import get_menu_structure as get_blt_menu
		blt_menu = get_blt_menu()
		menu['items'].extend(blt_menu['items'])
	
	if 'expiry_date_management' in subcapabilities:
		from .expiry_date_management.blueprint import get_menu_structure as get_edm_menu
		edm_menu = get_edm_menu()
		menu['items'].extend(edm_menu['items'])
	
	return menu


def validate_subcapability_dependencies(subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate that sub-capability dependencies are met"""
	
	from . import validate_composition
	return validate_composition(subcapabilities)


def init_capability(appbuilder: AppBuilder, subcapabilities: List[str] = None):
	"""Initialize Inventory Management capability with specified sub-capabilities"""
	
	# Validate dependencies
	validation = validate_subcapability_dependencies(subcapabilities or ['stock_tracking_control'])
	
	if not validation['valid']:
		raise ValueError(f"Invalid sub-capability composition: {validation['errors']}")
	
	# Register views and permissions
	register_capability_views(appbuilder, subcapabilities)
	register_capability_permissions(appbuilder)
	
	# Log warnings if any
	if validation['warnings']:
		for warning in validation['warnings']:
			print(f"Warning: {warning}")
	
	print(f"Inventory Management capability initialized with sub-capabilities: {subcapabilities}")


def get_capability_info() -> Dict[str, Any]:
	"""Get Inventory Management capability information"""
	
	from . import get_capability_info
	return get_capability_info()


def get_available_subcapabilities() -> List[str]:
	"""Get list of available sub-capabilities"""
	
	from . import get_subcapabilities
	return get_subcapabilities()