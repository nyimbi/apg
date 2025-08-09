"""
Retail & Wholesale (RTL) Industry Vertical

Retail and wholesale industry-specific capabilities including point of sale,
inventory management, loyalty programs, and omni-channel operations.
"""

__version__ = "1.0.0"

# Retail sub-modules with 3-character naming
SUBCAPABILITIES = [
	"pos",  # Point of Sale
	"sin",  # Store Inventory
	"prm",  # Promotion & Discount Management
	"loy",  # Loyalty Program Management
	"omc",  # Omni-Channel Fulfillment
]

__all__ = [
    "pos",  # Point of Sale
    "sin",  # Store Inventory
    "prm",  # Promotion & Discount Management
    "loy",  # Loyalty Program Management
    "omc",  # Omni-Channel Fulfillment
]