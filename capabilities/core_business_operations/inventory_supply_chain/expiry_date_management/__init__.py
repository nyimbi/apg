"""
Expiry Date Management Sub-capability

Tracks product expiration dates to minimize waste and ensure product freshness/safety.
Essential for pharmaceutical, food & beverage, and retail industries.
"""

SUBCAPABILITY_META = {
	'name': 'Expiry Date Management',
	'code': 'EDM', 
	'description': 'Product expiration tracking, FEFO rotation, and waste minimization for safety and compliance',
	'version': '1.0.0',
	'industry_focus': 'Pharmaceutical, Food & Beverage, Retail',
	'parent_capability': 'inventory_management',
	'database_prefix': 'im_edm_',
	'api_prefix': '/api/v1/inventory/expiry-management',
	'menu_category': 'Inventory Management',
	'menu_icon': 'fa-calendar-times'
}