"""
Document Management Sub-Capability

Manages and stores electronic documents, ensuring version control,
security, and accessibility across the organization.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Document Management',
	'code': 'DM',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Manages and stores electronic documents, ensuring version control, security, and accessibility',
	'industry_focus': 'All Industries',
	'dependencies': [],
	'optional_dependencies': ['workflow_business_process_mgmt', 'governance_risk_compliance'],
	'database_tables': [
		'gc_dm_document',
		'gc_dm_document_version',
		'gc_dm_folder',
		'gc_dm_document_type',
		'gc_dm_access_permission',
		'gc_dm_checkout_log',
		'gc_dm_search_index',
		'gc_dm_retention_policy'
	],
	'configuration': {
		'enable_version_control': True,
		'enable_full_text_search': True,
		'enable_document_workflow': True,
		'enable_electronic_signature': True,
		'max_file_size_mb': 100,
		'enable_document_templates': True,
		'auto_ocr_enabled': True,
		'enable_collaboration': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META