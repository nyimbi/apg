"""
Production Planning Sub-Capability

Schedules and plans manufacturing activities, optimizing resource utilization
across facilities, production lines, and time horizons. Provides master production
scheduling, demand planning, and resource optimization capabilities.

Key Features:
- Master Production Schedule (MPS)
- Demand Planning & Forecasting
- Resource Capacity Planning
- Production Order Management
- Schedule Optimization
- What-If Scenario Analysis
"""

from uuid_extensions import uuid7str

SUB_CAPABILITY_INFO = {
	"id": uuid7str(),
	"name": "Production Planning", 
	"code": "MF_PP",
	"parent_capability": "MF",
	"version": "1.0.0",
	"description": "Schedules and plans manufacturing activities, optimizing resource utilization",
	"features": [
		"Master Production Schedule (MPS)",
		"Demand Planning & Forecasting", 
		"Resource Capacity Planning",
		"Production Order Management",
		"Schedule Optimization",
		"What-If Scenario Analysis"
	]
}

__all__ = ["SUB_CAPABILITY_INFO"]