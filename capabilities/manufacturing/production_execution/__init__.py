"""
Manufacturing Capability Module

Comprehensive manufacturing capability providing complete production lifecycle management,
including production planning, materials management, shop floor control, quality assurance,
and manufacturing execution systems for enterprise environments.

Key Features:
- Production Planning & Scheduling
- Material Requirements Planning (MRP)
- Shop Floor Control & Monitoring
- Bill of Materials (BOM) Management
- Capacity Planning & Resource Optimization
- Quality Management (QA/QC)
- Recipe & Formula Management
- Manufacturing Execution System (MES)

Industries: Manufacturing, Pharmaceutical, Food & Beverage, Automotive, Electronics
"""

from uuid_extensions import uuid7str

# Manufacturing Capability Metadata
CAPABILITY_INFO = {
	"id": uuid7str(),
	"name": "Manufacturing",
	"code": "MF",
	"version": "1.0.0",
	"description": "Complete manufacturing capability with production planning, MRP, shop floor control, quality management, and MES functionality",
	"category": "Production",
	"status": "active",
	"dependencies": [
		"inventory_management",
		"procurement_purchasing", 
		"audit_compliance"
	],
	"sub_capabilities": [
		{
			"name": "Production Planning",
			"code": "MF_PP", 
			"description": "Schedules and plans manufacturing activities, optimizing resource utilization"
		},
		{
			"name": "Material Requirements Planning",
			"code": "MF_MRP",
			"description": "Calculates exact materials and components needed for production based on master schedule"
		},
		{
			"name": "Shop Floor Control", 
			"code": "MF_SFC",
			"description": "Monitors and manages activities on production floor, including work orders and machine status"
		},
		{
			"name": "Bill of Materials Management",
			"code": "MF_BOM", 
			"description": "Manages structured lists of raw materials, assemblies, and components for each product"
		},
		{
			"name": "Capacity Planning",
			"code": "MF_CP",
			"description": "Determines production capacity needed to meet demand, considering machines, labor, and shifts"
		},
		{
			"name": "Quality Management",
			"code": "MF_QM",
			"description": "Ensures products meet quality standards through inspections, testing, and corrective actions"
		},
		{
			"name": "Recipe & Formula Management", 
			"code": "MF_RFM",
			"description": "Manages precise recipes and formulas for process manufacturing with regulatory compliance"
		},
		{
			"name": "Manufacturing Execution System",
			"code": "MF_MES", 
			"description": "Real-time monitoring and control of manufacturing operations, bridging planning and execution"
		}
	],
	"supported_industries": [
		"Manufacturing",
		"Pharmaceutical", 
		"Food & Beverage",
		"Automotive",
		"Electronics",
		"Chemical Processing",
		"Aerospace",
		"Medical Devices"
	],
	"compliance_standards": [
		"ISO 9001",
		"ISO 13485", 
		"FDA 21 CFR Part 820",
		"GMP (Good Manufacturing Practice)",
		"HACCP",
		"SOP (Standard Operating Procedures)",
		"Lot Traceability",
		"Validation & Verification"
	]
}

__all__ = ["CAPABILITY_INFO"]