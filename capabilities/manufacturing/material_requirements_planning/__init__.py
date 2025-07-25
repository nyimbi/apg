"""
Material Requirements Planning (MRP) Sub-Capability

Calculates the exact materials and components needed for production based on the master schedule.
Provides automated material planning, inventory requirements calculation, and purchase recommendations.

Key Features:
- MRP Run Processing
- Material Requirements Calculation  
- Planned Orders Generation
- Inventory Availability Analysis
- Purchase Recommendations
- Exception Messages & Alerts
"""

from uuid_extensions import uuid7str

SUB_CAPABILITY_INFO = {
	"id": uuid7str(),
	"name": "Material Requirements Planning",
	"code": "MF_MRP", 
	"parent_capability": "MF",
	"version": "1.0.0",
	"description": "Calculates exact materials and components needed for production based on master schedule",
	"features": [
		"MRP Run Processing",
		"Material Requirements Calculation",
		"Planned Orders Generation", 
		"Inventory Availability Analysis",
		"Purchase Recommendations",
		"Exception Messages & Alerts"
	]
}

__all__ = ["SUB_CAPABILITY_INFO"]