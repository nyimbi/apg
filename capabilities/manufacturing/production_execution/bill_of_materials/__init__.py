"""
Bill of Materials (BOM) Management Sub-Capability

Manages structured lists of raw materials, assemblies, and components required for each product.
Provides multi-level BOM management, version control, and engineering change management.

Key Features:
- Multi-Level BOM Structure
- BOM Version Control
- Engineering Change Management
- Component Substitution
- Where-Used Analysis
- Cost Roll-up Calculations
"""

from uuid_extensions import uuid7str

SUB_CAPABILITY_INFO = {
	"id": uuid7str(),
	"name": "Bill of Materials Management",
	"code": "MF_BOM",
	"parent_capability": "MF", 
	"version": "1.0.0",
	"description": "Manages structured lists of raw materials, assemblies, and components for each product",
	"features": [
		"Multi-Level BOM Structure",
		"BOM Version Control",
		"Engineering Change Management",
		"Component Substitution",
		"Where-Used Analysis", 
		"Cost Roll-up Calculations"
	]
}

__all__ = ["SUB_CAPABILITY_INFO"]