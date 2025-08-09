"""
Pharmaceutical Specific Capability

Industry-specific ERP functionality for pharmaceutical and life sciences companies.
Manages regulatory compliance, clinical trials, R&D, product serialization, and batch release processes.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Pharmaceutical Specific',
	'code': 'PH',
	'version': '1.0.0',
	'description': 'Industry-specific ERP functionality for pharmaceutical and life sciences companies',
	'industry_focus': 'Pharmaceutical',
	'regulatory_frameworks': ['FDA', 'EMA', 'GMP', 'GxP', '21 CFR Part 11', 'ICH'],
	'dependencies': [
		'core_financials',
		'inventory_management', 
		'manufacturing',
		'audit_compliance',
		'auth_rbac'
	],
	'optional_dependencies': [
		'supply_chain_management',
		'human_resources',
		'procurement_purchasing'
	],
	'subcapabilities': [
		'regulatory_compliance',
		'product_serialization_tracking', 
		'clinical_trials_management',
		'rd_management',
		'batch_release_management'
	],
	'database_tables_prefix': 'ph_',
	'api_prefix': '/api/pharmaceutical',
	'permissions_prefix': 'ph.',
	'configuration': {
		'strict_audit_trails': True,
		'electronic_signatures': True,
		'data_integrity_controls': True,
		'regulatory_reporting': True,
		'serialization_required': True,
		'batch_genealogy': True,
		'change_control': True
	}
}

def get_capability_info() -> Dict[str, Any]:
	"""Get pharmaceutical capability information"""
	return CAPABILITY_META

def validate_dependencies(available_capabilities: List[str]) -> Dict[str, Any]:
	"""Validate that required dependencies are available"""
	errors = []
	warnings = []
	
	# Check required dependencies
	required = CAPABILITY_META['dependencies']
	for dep in required:
		if dep not in available_capabilities:
			errors.append(f"Required dependency '{dep}' not available")
	
	# Check optional dependencies
	optional = CAPABILITY_META['optional_dependencies']
	for dep in optional:
		if dep not in available_capabilities:
			warnings.append(f"Optional dependency '{dep}' not available - some features may be limited")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_regulatory_requirements() -> Dict[str, Any]:
	"""Get pharmaceutical regulatory requirements"""
	return {
		'fda_requirements': {
			'electronic_records': True,
			'electronic_signatures': True,
			'audit_trails': True,
			'data_integrity': True,
			'change_control': True
		},
		'gmp_requirements': {
			'batch_records': True,
			'quality_control': True,
			'deviation_management': True,
			'corrective_actions': True,
			'preventive_actions': True
		},
		'serialization_requirements': {
			'unique_identifiers': True,
			'aggregation': True,
			'track_and_trace': True,
			'verification': True,
			'reporting': True
		},
		'clinical_requirements': {
			'protocol_management': True,
			'patient_safety': True,
			'data_integrity': True,
			'regulatory_submissions': True,
			'adverse_event_reporting': True
		}
	}

def get_compliance_controls() -> List[Dict[str, Any]]:
	"""Get pharmaceutical compliance controls"""
	return [
		{
			'control_id': 'PH-001',
			'name': 'Electronic Signature Validation',
			'description': 'Validates electronic signatures per 21 CFR Part 11',
			'category': 'Data Integrity',
			'severity': 'Critical'
		},
		{
			'control_id': 'PH-002', 
			'name': 'Audit Trail Completeness',
			'description': 'Ensures complete audit trails for all regulatory data',
			'category': 'Compliance',
			'severity': 'Critical'
		},
		{
			'control_id': 'PH-003',
			'name': 'Batch Genealogy Tracking',
			'description': 'Tracks complete batch genealogy from raw materials to finished goods',
			'category': 'Traceability',
			'severity': 'High'
		},
		{
			'control_id': 'PH-004',
			'name': 'Serialization Validation',
			'description': 'Validates product serialization and aggregation',
			'category': 'Anti-Counterfeiting',
			'severity': 'High'
		},
		{
			'control_id': 'PH-005',
			'name': 'Clinical Data Integrity',
			'description': 'Ensures integrity of clinical trial data',
			'category': 'Clinical',
			'severity': 'Critical'
		}
	]