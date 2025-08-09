"""
Regulatory Compliance Sub-Capability

Manages adherence to industry-specific regulations (e.g., FDA, GMP, GxP), 
including documentation, audits, and reporting for pharmaceutical companies.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Regulatory Compliance',
	'code': 'RC',
	'version': '1.0.0',
	'capability': 'pharmaceutical_specific',
	'description': 'Manages adherence to industry-specific regulations (e.g., FDA, GMP, GxP), including documentation, audits, and reporting',
	'industry_focus': 'Pharmaceutical',
	'regulatory_scope': ['FDA', 'EMA', 'GMP', 'GCP', 'GLP', '21 CFR Part 11', 'ICH', 'WHO'],
	'dependencies': ['audit_compliance', 'auth_rbac'],
	'optional_dependencies': ['document_management', 'workflow_business_process_mgmt'],
	'database_tables': [
		'ph_rc_regulatory_framework',
		'ph_rc_submission',
		'ph_rc_submission_document',
		'ph_rc_audit',
		'ph_rc_audit_finding',
		'ph_rc_compliance_control',
		'ph_rc_deviation',
		'ph_rc_corrective_action',
		'ph_rc_regulatory_contact',
		'ph_rc_inspection',
		'ph_rc_regulatory_report'
	],
	'api_endpoints': [
		'/api/pharmaceutical/regulatory/frameworks',
		'/api/pharmaceutical/regulatory/submissions',
		'/api/pharmaceutical/regulatory/audits',
		'/api/pharmaceutical/regulatory/deviations',
		'/api/pharmaceutical/regulatory/actions',
		'/api/pharmaceutical/regulatory/inspections',
		'/api/pharmaceutical/regulatory/reports'
	],
	'views': [
		'PHRCRegulatoryFrameworkModelView',
		'PHRCSubmissionModelView',
		'PHRCAuditModelView',
		'PHRCDeviationModelView',
		'PHRCCorrectiveActionModelView',
		'PHRCInspectionModelView',
		'PHRCComplianceDashboardView'
	],
	'permissions': [
		'ph.regulatory.read',
		'ph.regulatory.write',
		'ph.regulatory.submit',
		'ph.regulatory.approve',
		'ph.regulatory.audit',
		'ph.regulatory.deviation',
		'ph.regulatory.action',
		'ph.regulatory.inspect',
		'ph.regulatory.admin'
	],
	'menu_items': [
		{
			'name': 'Regulatory Dashboard',
			'endpoint': 'PHRCComplianceDashboardView.index',
			'icon': 'fa-shield-alt',
			'permission': 'ph.regulatory.read'
		},
		{
			'name': 'Regulatory Submissions',
			'endpoint': 'PHRCSubmissionModelView.list',
			'icon': 'fa-file-medical-alt',
			'permission': 'ph.regulatory.read'
		},
		{
			'name': 'Compliance Audits',
			'endpoint': 'PHRCAuditModelView.list',
			'icon': 'fa-search',
			'permission': 'ph.regulatory.audit'
		},
		{
			'name': 'Deviations & CAPA',
			'endpoint': 'PHRCDeviationModelView.list',
			'icon': 'fa-exclamation-triangle',
			'permission': 'ph.regulatory.deviation'
		},
		{
			'name': 'Regulatory Inspections',
			'endpoint': 'PHRCInspectionModelView.list',
			'icon': 'fa-clipboard-check',
			'permission': 'ph.regulatory.inspect'
		}
	],
	'configuration': {
		'electronic_signatures_required': True,
		'audit_trail_mandatory': True,
		'change_control_required': True,
		'deviation_auto_numbering': True,
		'submission_workflow': True,
		'inspection_readiness': True,
		'data_integrity_controls': True,
		'regulatory_intelligence': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get regulatory compliance sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# Check required dependencies
	if 'audit_compliance' not in available_subcapabilities:
		errors.append("Audit & Compliance capability required for regulatory compliance")
	
	if 'auth_rbac' not in available_subcapabilities:
		errors.append("Authentication & RBAC capability required for regulatory compliance")
	
	# Check optional dependencies
	if 'document_management' not in available_subcapabilities:
		warnings.append("Document Management integration not available - manual document handling required")
	
	if 'workflow_business_process_mgmt' not in available_subcapabilities:
		warnings.append("Workflow Management not available - manual approval processes required")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_regulatory_frameworks() -> List[Dict[str, Any]]:
	"""Get supported regulatory frameworks"""
	return [
		{
			'code': 'FDA',
			'name': 'Food and Drug Administration',
			'region': 'United States',
			'scope': ['Drugs', 'Biologics', 'Medical Devices'],
			'key_regulations': ['21 CFR Part 11', '21 CFR Part 210', '21 CFR Part 211']
		},
		{
			'code': 'EMA',
			'name': 'European Medicines Agency',
			'region': 'European Union',
			'scope': ['Medicinal Products', 'Veterinary Medicines'],
			'key_regulations': ['EU GMP', 'Clinical Trial Regulation', 'Pharmacovigilance']
		},
		{
			'code': 'GMP',
			'name': 'Good Manufacturing Practice',
			'region': 'Global',
			'scope': ['Manufacturing', 'Quality Control', 'Quality Assurance'],
			'key_regulations': ['ICH Q7', 'WHO GMP', 'PIC/S GMP']
		},
		{
			'code': 'GCP',
			'name': 'Good Clinical Practice',
			'region': 'Global',
			'scope': ['Clinical Trials', 'Research', 'Data Integrity'],
			'key_regulations': ['ICH E6', 'ISO 14155', 'Declaration of Helsinki']
		},
		{
			'code': 'GLP',
			'name': 'Good Laboratory Practice',
			'region': 'Global',
			'scope': ['Laboratory Studies', 'Non-clinical Safety'],
			'key_regulations': ['OECD GLP', 'FDA GLP', 'EPA GLP']
		}
	]

def get_submission_types() -> List[Dict[str, Any]]:
	"""Get regulatory submission types"""
	return [
		{
			'type': 'IND',
			'name': 'Investigational New Drug',
			'description': 'Application to begin clinical trials',
			'framework': 'FDA',
			'typical_timeline': '30 days review'
		},
		{
			'type': 'NDA',
			'name': 'New Drug Application',
			'description': 'Application for drug marketing approval',
			'framework': 'FDA',
			'typical_timeline': '6-12 months review'
		},
		{
			'type': 'BLA',
			'name': 'Biologics License Application',
			'description': 'Application for biologic product approval',
			'framework': 'FDA',
			'typical_timeline': '6-12 months review'
		},
		{
			'type': 'MAA',
			'name': 'Marketing Authorization Application',
			'description': 'EU application for marketing authorization',
			'framework': 'EMA',
			'typical_timeline': '210 days review'
		},
		{
			'type': 'ANDA',
			'name': 'Abbreviated New Drug Application',
			'description': 'Generic drug approval application',
			'framework': 'FDA',
			'typical_timeline': '10-15 months review'
		}
	]

def get_compliance_metrics() -> List[Dict[str, Any]]:
	"""Get regulatory compliance metrics"""
	return [
		{
			'metric': 'Submission Success Rate',
			'description': 'Percentage of submissions approved on first review',
			'target': '85%',
			'measurement': 'Monthly'
		},
		{
			'metric': 'Audit Finding Resolution Time',
			'description': 'Average time to resolve audit findings',
			'target': '30 days',
			'measurement': 'Per Finding'
		},
		{
			'metric': 'Deviation Response Time',
			'description': 'Time to initiate deviation investigation',
			'target': '24 hours',
			'measurement': 'Per Deviation'
		},
		{
			'metric': 'CAPA Effectiveness',
			'description': 'Percentage of CAPAs that prevent recurrence',
			'target': '95%',
			'measurement': 'Annual'
		},
		{
			'metric': 'Inspection Readiness',
			'description': 'Time to prepare for regulatory inspection',
			'target': '48 hours',
			'measurement': 'Per Inspection'
		}
	]