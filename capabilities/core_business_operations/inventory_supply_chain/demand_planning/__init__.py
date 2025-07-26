"""
Demand Planning Sub-Capability

Forecasts customer demand for products to optimize inventory and production planning.
Uses statistical models and machine learning to predict future demand patterns.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Demand Planning',
	'code': 'DP',
	'version': '1.0.0',
	'capability': 'supply_chain_management',
	'description': 'Forecasts customer demand for products to optimize inventory and production planning',
	'industry_focus': 'Manufacturing, Retail, Distribution',
	'dependencies': [],
	'optional_dependencies': ['warehouse_management', 'supplier_relationship_management'],
	'database_tables': [
		'sc_dp_forecast',
		'sc_dp_forecast_model',
		'sc_dp_demand_history',
		'sc_dp_seasonal_pattern',
		'sc_dp_forecast_accuracy',
		'sc_dp_demand_driver',
		'sc_dp_forecast_scenario'
	],
	'api_endpoints': [
		'/api/supply_chain/demand_planning/forecasts',
		'/api/supply_chain/demand_planning/models',
		'/api/supply_chain/demand_planning/history',
		'/api/supply_chain/demand_planning/accuracy',
		'/api/supply_chain/demand_planning/scenarios'
	],
	'views': [
		'SCDPForecastModelView',
		'SCDPDemandHistoryModelView',
		'SCDPSeasonalPatternModelView',
		'SCDPForecastAccuracyView',
		'SCDPDashboardView'
	],
	'permissions': [
		'demand_planning.read',
		'demand_planning.write',
		'demand_planning.forecast',
		'demand_planning.model_manage',
		'demand_planning.admin'
	],
	'menu_items': [
		{
			'name': 'Demand Forecasts',
			'endpoint': 'SCDPForecastModelView.list',
			'icon': 'fa-chart-line',
			'permission': 'demand_planning.read'
		},
		{
			'name': 'Forecast Models',
			'endpoint': 'SCDPForecastModelView.list',
			'icon': 'fa-robot',
			'permission': 'demand_planning.read'
		},
		{
			'name': 'Demand History',
			'endpoint': 'SCDPDemandHistoryModelView.list',
			'icon': 'fa-history',
			'permission': 'demand_planning.read'
		},
		{
			'name': 'Forecast Accuracy',
			'endpoint': 'SCDPForecastAccuracyView.index',
			'icon': 'fa-bullseye',
			'permission': 'demand_planning.read'
		},
		{
			'name': 'DP Dashboard',
			'endpoint': 'SCDPDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'demand_planning.read'
		}
	],
	'configuration': {
		'forecast_horizon_days': 90,
		'history_lookback_days': 730,
		'min_data_points': 24,
		'auto_model_selection': True,
		'accuracy_threshold': 0.85,
		'outlier_detection': True
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
		warnings.append("Warehouse Management integration not available - inventory levels won't be optimized automatically")
	
	if 'supplier_relationship_management' not in available_subcapabilities:
		warnings.append("SRM integration not available - supplier lead times won't be considered in planning")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}