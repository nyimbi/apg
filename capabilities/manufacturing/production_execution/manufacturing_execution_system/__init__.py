"""
Manufacturing Execution System (MES) Sub-Capability

Real-time monitoring and control of manufacturing operations, bridging planning
and execution with comprehensive production visibility and control.

Key Features:
- Real-time Production Monitoring
- Work Order Execution
- Resource Scheduling & Dispatch
- Performance & Efficiency Tracking
- Material Genealogy & Traceability
- Integration with Plant Floor Systems
"""

from uuid_extensions import uuid7str

SUB_CAPABILITY_INFO = {
	"id": uuid7str(),
	"name": "Manufacturing Execution System",
	"code": "MF_MES",
	"parent_capability": "MF",
	"version": "1.0.0", 
	"description": "Real-time monitoring and control of manufacturing operations, bridging planning and execution",
	"features": [
		"Real-time Production Monitoring",
		"Work Order Execution",
		"Resource Scheduling & Dispatch",
		"Performance & Efficiency Tracking",
		"Material Genealogy & Traceability",
		"Integration with Plant Floor Systems"
	]
}

__all__ = ["SUB_CAPABILITY_INFO"]