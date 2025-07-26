"""
Shop Floor Control Sub-Capability

Monitors and manages activities on the production floor, including work orders,
machine status, labor tracking, and real-time production monitoring.

Key Features:
- Work Order Dispatch & Tracking
- Machine & Equipment Status
- Labor Time & Attendance
- Real-time Production Monitoring
- Quality Control Integration
- Performance Analytics
"""

from uuid_extensions import uuid7str

SUB_CAPABILITY_INFO = {
	"id": uuid7str(),
	"name": "Shop Floor Control",
	"code": "MF_SFC",
	"parent_capability": "MF",
	"version": "1.0.0",
	"description": "Monitors and manages activities on production floor including work orders and machine status",
	"features": [
		"Work Order Dispatch & Tracking",
		"Machine & Equipment Status",
		"Labor Time & Attendance",
		"Real-time Production Monitoring",
		"Quality Control Integration",
		"Performance Analytics"
	]
}

__all__ = ["SUB_CAPABILITY_INFO"]