"""
Product Catalog Management Sub-Capability

Centralized management of product information (SKUs, descriptions, images, pricing)
across multiple channels and sellers.
"""

SUBCAPABILITY_META = {
	'name': 'Product Catalog Management',
	'code': 'PSP',
	'description': 'Centralized product information management across channels and sellers',
	'models': [
		'PSProduct',
		'PSProductCategory',
		'PSProductAttribute',
		'PSProductVariant',
		'PSProductImage',
		'PSProductPricing',
		'PSProductInventory',
		'PSProductBundle',
		'PSProductRelation',
		'PSProductImport'
	]
}