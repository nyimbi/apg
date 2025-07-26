"""
Logistics & Transportation Management Sub-Capability

Plans and optimizes the movement of goods, including freight management,
route optimization, and carrier coordination.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Logistics & Transportation Management',
	'code': 'LT',
	'version': '1.0.0',
	'capability': 'supply_chain_management',
	'description': 'Plans and optimizes the movement of goods, including freight, routes, and carriers',
	'industry_focus': 'Manufacturing, Distribution, Logistics',
	'dependencies': [],
	'optional_dependencies': ['warehouse_management', 'demand_planning'],
	'database_tables': [
		'sc_lt_shipment',
		'sc_lt_carrier',
		'sc_lt_route',
		'sc_lt_freight_rate',
		'sc_lt_transportation_mode',
		'sc_lt_delivery_schedule',
		'sc_lt_tracking_event'
	],
	'api_endpoints': [
		'/api/supply_chain/logistics/shipments',
		'/api/supply_chain/logistics/carriers',
		'/api/supply_chain/logistics/routes',
		'/api/supply_chain/logistics/tracking',
		'/api/supply_chain/logistics/optimization'
	],
	'views': [
		'SCLTShipmentModelView',
		'SCLTCarrierModelView',
		'SCLTRouteModelView',
		'SCLTTrackingView',
		'SCLTDashboardView'
	],
	'permissions': [
		'logistics.read',
		'logistics.write',
		'logistics.optimize',
		'logistics.track',
		'logistics.admin'
	],
	'menu_items': [
		{
			'name': 'Shipments',
			'endpoint': 'SCLTShipmentModelView.list',
			'icon': 'fa-shipping-fast',
			'permission': 'logistics.read'
		},
		{
			'name': 'Carriers',
			'endpoint': 'SCLTCarrierModelView.list',
			'icon': 'fa-truck',
			'permission': 'logistics.read'
		},
		{
			'name': 'Routes',
			'endpoint': 'SCLTRouteModelView.list',
			'icon': 'fa-route',
			'permission': 'logistics.read'
		},
		{
			'name': 'Tracking',
			'endpoint': 'SCLTTrackingView.index',
			'icon': 'fa-map-marker-alt',
			'permission': 'logistics.read'
		},
		{
			'name': 'Logistics Dashboard',
			'endpoint': 'SCLTDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'logistics.read'
		}
	],
	'configuration': {
		'enable_route_optimization': True,
		'default_transportation_mode': 'ground',
		'tracking_update_frequency_minutes': 30,
		'auto_carrier_selection': True,
		'cost_optimization_priority': 0.7,  # vs speed priority
		'enable_real_time_tracking': True
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
	if 'warehouse_management' not in available_subcapabilities:
		warnings.append("Warehouse Management integration not available - shipment origins/destinations may be limited")
	
	if 'demand_planning' not in available_subcapabilities:
		warnings.append("Demand Planning integration not available - route optimization may be less effective")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}