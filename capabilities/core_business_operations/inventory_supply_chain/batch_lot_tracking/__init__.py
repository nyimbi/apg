"""
Batch & Lot Tracking Sub-capability

Manages specific batches or lots of products for traceability,
crucial for recalls and quality control in manufacturing, pharmaceutical, and food & beverage industries.
"""

SUBCAPABILITY_META = {
	'name': 'Batch & Lot Tracking',
	'code': 'BLT', 
	'description': 'Comprehensive batch and lot tracking for full product traceability and recall management',
	'version': '1.0.0',
	'industry_focus': 'Manufacturing, Pharmaceutical, Food & Beverage',
	'parent_capability': 'inventory_management',
	'database_prefix': 'im_blt_',
	'api_prefix': '/api/v1/inventory/batch-lot-tracking',
	'menu_category': 'Inventory Management',
	'menu_icon': 'fa-barcode'
}