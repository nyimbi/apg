"""
APG Core Business Operations Capabilities

Comprehensive suite of fundamental business operation capabilities including
financial management, human capital, procurement, inventory/supply chain,
and sales/revenue management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import List, Dict, Any

# Core Business Operations Metadata
__version__ = "1.0.0"
__category__ = "core_business_operations"
__description__ = "Fundamental business operation capabilities for enterprise management"

# Sub-capability Registry
SUBCAPABILITIES = [
	"financial_management",
	"human_capital_management", 
	"procurement_sourcing",
	"inventory_supply_chain",
	"sales_revenue_management"
]

# APG Composition Engine Registration
CAPABILITY_METADATA = {
	"capability_id": "core_business_operations",
	"version": __version__,
	"category": "core_business",
	"subcapabilities": SUBCAPABILITIES,
	"dependencies": ["auth_rbac", "audit_compliance", "notification_engine"],
	"provides_services": [
		"financial_management_suite",
		"human_capital_operations",
		"procurement_sourcing_services",
		"supply_chain_optimization",
		"sales_revenue_tracking"
	],
	"composition_priority": 1  # Core capability - high priority
}

def get_capability_info() -> Dict[str, Any]:
	"""Get core business operations capability information."""
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