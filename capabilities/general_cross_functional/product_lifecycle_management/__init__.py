"""
Product Lifecycle Management (PLM) Sub-Capability

Manages a product's entire lifecycle from conception, design,
and manufacturing to service and disposal.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Product Lifecycle Management (PLM)',
	'code': 'PL',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Manages a product\'s entire lifecycle from conception, design, and manufacturing to service and disposal',
	'industry_focus': 'Manufacturing, Engineering, Product Development',
	'dependencies': [],
	'optional_dependencies': ['document_management', 'workflow_business_process_mgmt'],
	'database_tables': [
		'gc_pl_product',
		'gc_pl_product_version',
		'gc_pl_design_document',
		'gc_pl_bill_of_materials',
		'gc_pl_change_request',
		'gc_pl_compliance_record',
		'gc_pl_lifecycle_stage',
		'gc_pl_collaboration_workspace'
	],
	'configuration': {
		'enable_version_control': True,
		'enable_change_management': True,
		'require_approval_workflows': True,
		'enable_supplier_collaboration': True,
		'enable_compliance_tracking': True,
		'default_document_retention_years': 7,
		'enable_3d_visualization': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META