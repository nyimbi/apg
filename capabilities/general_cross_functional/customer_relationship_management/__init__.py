"""
Customer Relationship Management (CRM) Sub-Capability

Manages customer interactions, sales pipeline, marketing campaigns,
and customer service across all touchpoints.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Customer Relationship Management (CRM)',
	'code': 'CR',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Manages customer interactions, sales pipeline, marketing campaigns, and customer service',
	'industry_focus': 'All Industries',
	'dependencies': [],
	'optional_dependencies': ['business_intelligence_analytics', 'document_management'],
	'database_tables': [
		'gc_cr_customer',
		'gc_cr_contact',
		'gc_cr_lead',
		'gc_cr_opportunity',
		'gc_cr_campaign',
		'gc_cr_activity',
		'gc_cr_service_case',
		'gc_cr_customer_segment'
	],
	'configuration': {
		'enable_lead_scoring': True,
		'enable_marketing_automation': True,
		'default_sales_stage_probability': {
			'prospect': 10, 'qualified': 25, 'proposal': 50,
			'negotiation': 75, 'closed_won': 100, 'closed_lost': 0
		},
		'enable_customer_portal': True,
		'auto_case_escalation_hours': 24,
		'enable_social_media_integration': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META