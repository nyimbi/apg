"""
Capacity Planning Sub-Capability

Determines the production capacity needed to meet demand, considering machines,
labor, shifts, and resource constraints for optimal production planning.

Key Features:
- Resource Capacity Modeling
- Demand vs Capacity Analysis
- Bottleneck Identification
- Shift & Workforce Planning
- Equipment Utilization
- What-If Scenario Planning
"""

from uuid_extensions import uuid7str

SUB_CAPABILITY_INFO = {
	"id": uuid7str(),
	"name": "Capacity Planning",
	"code": "MF_CP",
	"parent_capability": "MF",
	"version": "1.0.0",
	"description": "Determines production capacity needed to meet demand, considering machines, labor, and shifts",
	"features": [
		"Resource Capacity Modeling",
		"Demand vs Capacity Analysis",
		"Bottleneck Identification",
		"Shift & Workforce Planning",
		"Equipment Utilization",
		"What-If Scenario Planning"
	]
}

__all__ = ["SUB_CAPABILITY_INFO"]