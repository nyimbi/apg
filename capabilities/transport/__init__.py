"""
Transportation & Logistics (TRL) Industry Vertical

Transportation and logistics management including fleet operations,
route optimization, cargo tracking, and supply chain coordination.
"""

__version__ = "1.0.0"

CAPABILITY_IDS = [
	"transport_car",  # Cargo Management
	"transport_del",  # Delivery Management
	"transport_dis",  # Dispatch Operations
	"transport_fle",  # Fleet Management
	"transport_fue",  # Fuel Management
	"transport_mai",  # Vehicle Maintenance
	"transport_rou",  # Route Optimisation
	"transport_sch",  # Transport Scheduling
	"transport_tra",  # Asset Tracking
	"transport_war",  # Warehouse Operations
]

__all__ = ["CAPABILITY_IDS"]
