"""
Platform Services Capability

Comprehensive e-commerce and marketplace platform functionality including
digital storefront, product catalog, customer management, payments, and multi-vendor operations.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Platform Services',
	'code': 'PS',
	'version': '1.0.0',
	'description': 'E-commerce and marketplace platform with multi-vendor support, payments, and comprehensive management',
	'industry_focus': 'E-commerce, Marketplaces, Retail',
	'subcapabilities': [
		'digital_storefront_management',
		'product_catalog_management',
		'customer_user_accounts',
		'payment_gateway_integration',
		'seller_vendor_management',
		'commission_management',
		'multi_vendor_order_fulfillment',
		'ratings_reviews_management',
		'dispute_resolution',
		'search_discovery_optimization',
		'advertising_promotion_management'
	],
	'implemented_subcapabilities': [
		'digital_storefront_management',
		'product_catalog_management',
		'customer_user_accounts',
		'payment_gateway_integration',
		'seller_vendor_management',
		'commission_management',
		'multi_vendor_order_fulfillment',
		'ratings_reviews_management',
		'dispute_resolution',
		'search_discovery_optimization',
		'advertising_promotion_management'
	],
	'database_prefix': 'ps_',
	'menu_category': 'Platform Services',
	'menu_icon': 'fa-shopping-cart'
}

# Import implemented sub-capabilities for discovery
from . import digital_storefront_management
from . import product_catalog_management
from . import customer_user_accounts
from . import payment_gateway_integration
from . import seller_vendor_management
from . import commission_management
from . import multi_vendor_order_fulfillment
from . import ratings_reviews_management
from . import dispute_resolution
from . import search_discovery_optimization
from . import advertising_promotion_management

def get_capability_info() -> Dict[str, Any]:
	"""Get capability information"""
	return CAPABILITY_META

def get_subcapabilities() -> List[str]:
	"""Get list of available sub-capabilities"""
	return CAPABILITY_META['subcapabilities']

def get_implemented_subcapabilities() -> List[str]:
	"""Get list of currently implemented sub-capabilities"""
	return CAPABILITY_META['implemented_subcapabilities']

def validate_composition(subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate a composition of sub-capabilities"""
	errors = []
	warnings = []
	
	# Check if requested sub-capabilities are implemented
	implemented = get_implemented_subcapabilities()
	for subcap in subcapabilities:
		if subcap not in CAPABILITY_META['subcapabilities']:
			errors.append(f"Unknown sub-capability: {subcap}")
		elif subcap not in implemented:
			warnings.append(f"Sub-capability '{subcap}' is not yet implemented")
	
	# Check for essential combinations
	if 'digital_storefront_management' in subcapabilities and 'product_catalog_management' not in subcapabilities:
		warnings.append("Product Catalog Management is recommended when using Digital Storefront")
	
	if 'seller_vendor_management' in subcapabilities and 'commission_management' not in subcapabilities:
		warnings.append("Commission Management is recommended for multi-vendor platforms")
	
	if 'multi_vendor_order_fulfillment' in subcapabilities and 'seller_vendor_management' not in subcapabilities:
		errors.append("Seller/Vendor Management is required for multi-vendor order fulfillment")
	
	if 'payment_gateway_integration' in subcapabilities and 'customer_user_accounts' not in subcapabilities:
		warnings.append("Customer User Accounts is recommended when using payments")
	
	if 'ratings_reviews_management' in subcapabilities and 'customer_user_accounts' not in subcapabilities:
		warnings.append("Customer User Accounts is recommended for ratings and reviews")
	
	if 'dispute_resolution' in subcapabilities and 'ratings_reviews_management' not in subcapabilities:
		warnings.append("Ratings & Reviews Management is recommended for dispute resolution")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize Platform Services capability with Flask-AppBuilder"""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)