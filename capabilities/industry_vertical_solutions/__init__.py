"""
APG Industry Vertical Solutions

Comprehensive industry-specific capabilities covering healthcare, energy,
telecommunications, transportation, real estate, education, and more.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import List, Dict, Any

# Industry Vertical Solutions Metadata
__version__ = "1.0.0"
__category__ = "industry_vertical_solutions"
__description__ = "Industry-specific capabilities for rapid vertical deployment"

# Sub-capability Registry
SUBCAPABILITIES = [
	"pharmaceutical_life_sciences",
	"mining_resources",
	"healthcare_medical",
	"energy_utilities",
	"telecommunications",
	"transportation_logistics",
	"real_estate_facilities",
	"education_academic",
	"government_public_sector"
]

# Industry Templates
INDUSTRY_TEMPLATES = {
	"healthcare_hospital": {
		"required_capabilities": [
			"healthcare_medical",
			"core_business_operations.human_capital_management",
			"general_cross_functional.governance_risk_compliance"
		],
		"compliance_frameworks": ["HIPAA", "HITECH", "FDA"]
	},
	"energy_utility": {
		"required_capabilities": [
			"energy_utilities",
			"core_business_operations.financial_management",
			"general_cross_functional.enterprise_asset_management"
		],
		"compliance_frameworks": ["NERC", "FERC", "Environmental"]
	},
	"telecom_operator": {
		"required_capabilities": [
			"telecommunications",
			"core_business_operations.sales_revenue_management",
			"general_cross_functional.customer_relationship_management"
		],
		"compliance_frameworks": ["FCC", "GDPR", "Telecommunications"]
	}
}

# APG Composition Engine Registration
CAPABILITY_METADATA = {
	"capability_id": "industry_vertical_solutions",
	"version": __version__,
	"category": "industry_vertical",
	"subcapabilities": SUBCAPABILITIES,
	"dependencies": [
		"core_business_operations",
		"general_cross_functional",
		"auth_rbac",
		"audit_compliance"
	],
	"provides_services": [
		"industry_compliance_management",
		"vertical_workflow_automation",
		"industry_specific_analytics",
		"regulatory_reporting",
		"sector_integration_services"
	],
	"composition_priority": 4,
	"templates": INDUSTRY_TEMPLATES
}

def get_capability_info() -> Dict[str, Any]:
	"""Get industry vertical solutions capability information."""
	return CAPABILITY_METADATA

def list_subcapabilities() -> List[str]:
	"""List all available subcapabilities."""
	return SUBCAPABILITIES.copy()

def get_industry_templates() -> Dict[str, Any]:
	"""Get available industry templates."""
	return INDUSTRY_TEMPLATES.copy()

def get_template_requirements(template_name: str) -> Dict[str, Any]:
	"""Get requirements for specific industry template."""
	return INDUSTRY_TEMPLATES.get(template_name, {})

__all__ = [
	"SUBCAPABILITIES",
	"INDUSTRY_TEMPLATES",
	"CAPABILITY_METADATA",
	"get_capability_info",
	"list_subcapabilities",
	"get_industry_templates",
	"get_template_requirements"
]