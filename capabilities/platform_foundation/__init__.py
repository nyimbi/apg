"""
APG Platform Foundation Capabilities

Digital commerce, marketplace operations, payment services, and customer
engagement capabilities for modern business platforms.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import List, Dict, Any

# Platform Foundation Metadata
__version__ = "1.0.0"
__category__ = "platform_foundation"
__description__ = "Digital platform foundation for commerce and engagement"

# Sub-capability Registry
SUBCAPABILITIES = [
	"digital_commerce",
	"marketplace_operations",
	"payment_financial_services",
	"customer_engagement"
]

# APG Composition Engine Registration
CAPABILITY_METADATA = {
	"capability_id": "platform_foundation",
	"version": __version__,
	"category": "platform_services",
	"subcapabilities": SUBCAPABILITIES,
	"dependencies": [
		"core_business_operations.financial_management",
		"core_business_operations.sales_revenue_management",
		"general_cross_functional.customer_relationship_management",
		"auth_rbac",
		"audit_compliance"
	],
	"provides_services": [
		"ecommerce_platform",
		"marketplace_management",
		"payment_processing",
		"customer_experience_management"
	],
	"composition_priority": 3
}

def get_capability_info() -> Dict[str, Any]:
	"""Get platform foundation capability information."""
	return CAPABILITY_METADATA

def list_subcapabilities() -> List[str]:
	"""List all available subcapabilities."""
	return SUBCAPABILITIES.copy()

__all__ = [
	"SUBCAPABILITIES",
	"CAPABILITY_METADATA",
	"get_capability_info",
	"list_subcapabilities"
]