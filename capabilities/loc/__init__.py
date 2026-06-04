"""
Localization & Multi-Entity (LOC) Capabilities

Multi-currency, multi-language, and multi-company support.
"""

__version__ = "1.0.0"

# Capability IDs exported for registry discovery
CAPABILITY_IDS = [
	"loc_mco",  # Multi-Country Operations
	"loc_mcy",  # Multi-Currency Management
	"loc_mlg",  # Multi-Language & Localisation
]

__all__ = ["CAPABILITY_IDS"]
