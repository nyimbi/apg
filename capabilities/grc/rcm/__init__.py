"""
Governance, Risk, & Compliance (GRC) Sub-Capability

Identifies, assesses, and mitigates business risks, and ensures compliance
with internal policies and external regulations.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Governance, Risk, & Compliance (GRC)',
	'code': 'GR',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Identifies, assesses, and mitigates business risks, and ensures compliance with internal policies and external regulations',
	'industry_focus': 'All Industries (especially regulated)',
	'dependencies': [],
	'optional_dependencies': ['document_management', 'workflow_business_process_mgmt', 'business_intelligence_analytics'],
	'database_tables': [
		'gc_gr_policy',
		'gc_gr_risk_register',
		'gc_gr_control',
		'gc_gr_compliance_requirement',
		'gc_gr_audit',
		'gc_gr_incident',
		'gc_gr_assessment',
		'gc_gr_remediation_plan'
	],
	'configuration': {
		'enable_risk_scoring': True,
		'enable_continuous_monitoring': True,
		'risk_assessment_frequency_days': 90,
		'enable_policy_attestation': True,
		'enable_audit_trails': True,
		'auto_compliance_reporting': True,
		'enable_third_party_risk_mgmt': True,
		'default_risk_appetite': 'moderate'
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META