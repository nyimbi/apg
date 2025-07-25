"""
Recipe & Formula Management Sub-Capability

Manages precise recipes and formulas for process manufacturing, ensuring consistency
and regulatory compliance for pharmaceutical, food & beverage, and chemical industries.

Key Features:
- Master Recipe Management
- Formula Version Control
- Ingredient Specifications
- Process Instructions
- Batch Record Templates
- Regulatory Compliance Tracking
"""

from uuid_extensions import uuid7str

SUB_CAPABILITY_INFO = {
	"id": uuid7str(),
	"name": "Recipe & Formula Management",
	"code": "MF_RFM",
	"parent_capability": "MF", 
	"version": "1.0.0",
	"description": "Manages precise recipes and formulas for process manufacturing with regulatory compliance",
	"features": [
		"Master Recipe Management",
		"Formula Version Control",
		"Ingredient Specifications",
		"Process Instructions",
		"Batch Record Templates",
		"Regulatory Compliance Tracking"
	]
}

__all__ = ["SUB_CAPABILITY_INFO"]