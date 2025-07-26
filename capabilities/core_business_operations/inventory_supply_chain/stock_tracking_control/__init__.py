"""
Stock Tracking & Control Sub-capability

Monitors inventory levels, locations, and movements in real-time.
Essential for manufacturing, pharmaceutical, and retail operations.
"""

SUBCAPABILITY_META = {
	'name': 'Stock Tracking & Control',
	'code': 'STC', 
	'description': 'Real-time inventory level monitoring, location tracking, and movement control',
	'version': '1.0.0',
	'industry_focus': 'All industries, especially Manufacturing, Pharmaceutical, Retail',
	'parent_capability': 'inventory_management',
	'database_prefix': 'im_stc_',
	'api_prefix': '/api/v1/inventory/stock-tracking',
	'menu_category': 'Inventory Management',
	'menu_icon': 'fa-warehouse'
}