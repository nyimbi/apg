"""
Platform Services Blueprint Registration

Registers all Platform Services sub-capability blueprints with Flask-AppBuilder.
"""

from typing import List
from . import CAPABILITY_META

def init_capability(appbuilder, subcapabilities: List[str]):
	"""Initialize Platform Services capability with sub-capabilities"""
	
	# Import and register blueprints for each requested sub-capability
	if 'digital_storefront_management' in subcapabilities:
		from .digital_storefront_management.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'product_catalog_management' in subcapabilities:
		from .product_catalog_management.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'customer_user_accounts' in subcapabilities:
		from .customer_user_accounts.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'payment_gateway_integration' in subcapabilities:
		from .payment_gateway_integration.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'seller_vendor_management' in subcapabilities:
		from .seller_vendor_management.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'commission_management' in subcapabilities:
		from .commission_management.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'multi_vendor_order_fulfillment' in subcapabilities:
		from .multi_vendor_order_fulfillment.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'ratings_reviews_management' in subcapabilities:
		from .ratings_reviews_management.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'dispute_resolution' in subcapabilities:
		from .dispute_resolution.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'search_discovery_optimization' in subcapabilities:
		from .search_discovery_optimization.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	if 'advertising_promotion_management' in subcapabilities:
		from .advertising_promotion_management.blueprint import init_subcapability
		init_subcapability(appbuilder)
	
	return True