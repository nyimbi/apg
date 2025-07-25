"""
Replenishment & Reordering Sub-capability

Automates or assists in reordering stock based on demand forecasts and minimum levels.
Essential for all industries, especially manufacturing, pharmaceutical, and retail.
"""

SUBCAPABILITY_META = {
	'name': 'Replenishment & Reordering',
	'code': 'RR', 
	'description': 'Automated stock replenishment and reordering based on demand forecasts and min/max levels',
	'version': '1.0.0',
	'industry_focus': 'All industries, especially Manufacturing, Pharmaceutical, Retail',
	'parent_capability': 'inventory_management',
	'database_prefix': 'im_rr_',
	'api_prefix': '/api/v1/inventory/replenishment',
	'menu_category': 'Inventory Management',
	'menu_icon': 'fa-sync-alt'
}