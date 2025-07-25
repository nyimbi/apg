"""
E-commerce (B2B/B2C) Sub-Capability

Integrates online sales channels with back-end inventory, order,
and financial management for single-party operations.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'E-commerce (B2B/B2C)',
	'code': 'EC',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Integrates online sales channels with back-end inventory, order, and financial management (for single-party operations)',
	'industry_focus': 'Retail, Distribution, Manufacturing',
	'dependencies': [],
	'optional_dependencies': ['customer_relationship_management', 'business_intelligence_analytics'],
	'database_tables': [
		'gc_ec_online_store',
		'gc_ec_product_catalog',
		'gc_ec_shopping_cart',
		'gc_ec_online_order',
		'gc_ec_payment_method',
		'gc_ec_shipping_option',
		'gc_ec_customer_review',
		'gc_ec_promotion_campaign'
	],
	'configuration': {
		'enable_multi_channel': True,
		'enable_mobile_commerce': True,
		'enable_guest_checkout': True,
		'enable_wishlist': True,
		'enable_product_recommendations': True,
		'enable_social_commerce': True,
		'default_payment_gateway': 'stripe',
		'enable_inventory_sync': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META