"""
Warehouse Management System (WMS) Sub-Capability

Manages warehouse operations including receiving, putaway, picking, packing,
shipping, and inventory accuracy tracking.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Warehouse Management System (WMS)',
	'code': 'WM',
	'version': '1.0.0',
	'capability': 'supply_chain_management',
	'description': 'Manages warehouse operations: receiving, putaway, picking, packing, shipping, and inventory accuracy',
	'industry_focus': 'Distribution, Manufacturing, Retail',
	'dependencies': [],
	'optional_dependencies': ['logistics_transportation', 'demand_planning'],
	'database_tables': [
		'sc_wm_warehouse',
		'sc_wm_zone',
		'sc_wm_location',
		'sc_wm_receipt',
		'sc_wm_pick_task',
		'sc_wm_put_away_task',
		'sc_wm_cycle_count',
		'sc_wm_inventory_movement'
	],
	'api_endpoints': [
		'/api/supply_chain/warehouse/warehouses',
		'/api/supply_chain/warehouse/receipts',
		'/api/supply_chain/warehouse/picking',
		'/api/supply_chain/warehouse/inventory',
		'/api/supply_chain/warehouse/cycle_counts'
	],
	'views': [
		'SCWMWarehouseModelView',
		'SCWMReceiptModelView',
		'SCWMPickTaskModelView',
		'SCWMCycleCountModelView',
		'SCWMDashboardView'
	],
	'permissions': [
		'warehouse.read',
		'warehouse.write',
		'warehouse.receive',
		'warehouse.pick',
		'warehouse.count',
		'warehouse.admin'
	],
	'menu_items': [
		{
			'name': 'Warehouses',
			'endpoint': 'SCWMWarehouseModelView.list',
			'icon': 'fa-warehouse',
			'permission': 'warehouse.read'
		},
		{
			'name': 'Receipts',
			'endpoint': 'SCWMReceiptModelView.list',
			'icon': 'fa-clipboard-check',
			'permission': 'warehouse.read'
		},
		{
			'name': 'Pick Tasks',
			'endpoint': 'SCWMPickTaskModelView.list',
			'icon': 'fa-hand-paper',
			'permission': 'warehouse.read'
		},
		{
			'name': 'Cycle Counts',
			'endpoint': 'SCWMCycleCountModelView.list',
			'icon': 'fa-calculator',
			'permission': 'warehouse.read'
		},
		{
			'name': 'WMS Dashboard',
			'endpoint': 'SCWMDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'warehouse.read'
		}
	],
	'configuration': {
		'enable_directed_putaway': True,
		'enable_wave_planning': True,
		'cycle_count_frequency_days': 90,
		'auto_pick_task_assignment': True,
		'enable_cross_docking': False,
		'require_lot_tracking': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# No hard dependencies, but warn about useful integrations
	if 'logistics_transportation' not in available_subcapabilities:
		warnings.append("Logistics integration not available - shipping coordination may be manual")
	
	if 'demand_planning' not in available_subcapabilities:
		warnings.append("Demand Planning integration not available - inventory optimization may be less effective")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}