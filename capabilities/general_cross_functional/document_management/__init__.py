"""
Document Management Sub-Capability

Manages and stores electronic documents, ensuring version control, security, 
and accessibility across the enterprise.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Document Management',
	'code': 'DM',
	'version': '1.0.0',
	'capability': 'general_cross_functional',
	'description': 'Manages and stores electronic documents, ensuring version control, security, and accessibility',
	'industry_focus': 'All',
	'dependencies': [],
	'optional_dependencies': ['workflow_management', 'governance_risk_compliance'],
	'database_tables': [
		'gc_dm_document',
		'gc_dm_document_version',
		'gc_dm_document_category',
		'gc_dm_document_type',
		'gc_dm_folder',
		'gc_dm_permission',
		'gc_dm_checkout',
		'gc_dm_workflow',
		'gc_dm_review',
		'gc_dm_tag',
		'gc_dm_metadata',
		'gc_dm_audit_log',
		'gc_dm_retention_policy',
		'gc_dm_archive'
	],
	'api_endpoints': [
		'/api/general_cross_functional/dm/documents',
		'/api/general_cross_functional/dm/folders',
		'/api/general_cross_functional/dm/versions',
		'/api/general_cross_functional/dm/permissions',
		'/api/general_cross_functional/dm/search',
		'/api/general_cross_functional/dm/workflows',
		'/api/general_cross_functional/dm/reviews',
		'/api/general_cross_functional/dm/retention',
		'/api/general_cross_functional/dm/dashboard'
	],
	'views': [
		'GCDMDocumentModelView',
		'GCDMFolderModelView',
		'GCDMDocumentVersionModelView',
		'GCDMPermissionModelView',
		'GCDMWorkflowModelView',
		'GCDMReviewModelView',
		'GCDMRetentionPolicyModelView',
		'GCDMSearchView',
		'GCDMDashboardView'
	],
	'permissions': [
		'dm.read',
		'dm.write',
		'dm.delete',
		'dm.approve',
		'dm.admin',
		'dm.checkout',
		'dm.version',
		'dm.permission_manage',
		'dm.workflow_manage',
		'dm.retention_manage',
		'dm.audit_view'
	],
	'menu_items': [
		{
			'name': 'Documents',
			'endpoint': 'GCDMDocumentModelView.list',
			'icon': 'fa-file-text',
			'permission': 'dm.read'
		},
		{
			'name': 'Folders',
			'endpoint': 'GCDMFolderModelView.list',
			'icon': 'fa-folder',
			'permission': 'dm.read'
		},
		{
			'name': 'Document Search',
			'endpoint': 'GCDMSearchView.index',
			'icon': 'fa-search',
			'permission': 'dm.read'
		},
		{
			'name': 'Document Workflows',
			'endpoint': 'GCDMWorkflowModelView.list',
			'icon': 'fa-random',
			'permission': 'dm.workflow_manage'
		},
		{
			'name': 'Reviews & Approvals',
			'endpoint': 'GCDMReviewModelView.list',
			'icon': 'fa-check-square',
			'permission': 'dm.approve'
		},
		{
			'name': 'Retention Policies',
			'endpoint': 'GCDMRetentionPolicyModelView.list',
			'icon': 'fa-clock-o',
			'permission': 'dm.retention_manage'
		},
		{
			'name': 'DM Dashboard',
			'endpoint': 'GCDMDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'dm.read'
		}
	],
	'configuration': {
		'max_file_size_mb': 100,
		'allowed_file_types': [
			'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
			'txt', 'rtf', 'jpg', 'jpeg', 'png', 'gif', 'bmp',
			'tiff', 'zip', 'rar', '7z', 'mp4', 'avi', 'mov'
		],
		'virus_scan_enabled': True,
		'auto_backup_enabled': True,
		'version_control_enabled': True,
		'encryption_enabled': True,
		'watermark_enabled': False,
		'ocr_enabled': True,
		'full_text_search_enabled': True,
		'email_notifications_enabled': True,
		'retention_enforcement_enabled': True,
		'audit_logging_enabled': True,
		'collaboration_enabled': True,
		'mobile_access_enabled': True,
		'offline_sync_enabled': False,
		'digital_signature_enabled': True,
		'thumbnail_generation_enabled': True,
		'storage_quota_gb': 1000
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# No hard dependencies for Document Management
	
	# Check for useful optional dependencies
	if 'workflow_management' not in available_subcapabilities:
		warnings.append("Workflow Management integration not available - manual approval processes only")
	
	if 'governance_risk_compliance' not in available_subcapabilities:
		warnings.append("GRC integration not available - limited compliance tracking")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_document_types() -> List[Dict[str, Any]]:
	"""Get default document types"""
	return [
		{
			'name': 'Policy',
			'description': 'Company policies and procedures',
			'retention_years': 7,
			'requires_approval': True,
			'template_available': True
		},
		{
			'name': 'Contract',
			'description': 'Legal contracts and agreements',
			'retention_years': 10,
			'requires_approval': True,
			'template_available': False
		},
		{
			'name': 'Invoice',
			'description': 'Financial invoices and receipts',
			'retention_years': 7,
			'requires_approval': False,
			'template_available': False
		},
		{
			'name': 'Report',
			'description': 'Business reports and analysis',
			'retention_years': 3,
			'requires_approval': True,
			'template_available': True
		},
		{
			'name': 'Specification',
			'description': 'Technical specifications and requirements',
			'retention_years': 5,
			'requires_approval': True,
			'template_available': True
		},
		{
			'name': 'Manual',
			'description': 'User manuals and documentation',
			'retention_years': 5,
			'requires_approval': True,
			'template_available': True
		},
		{
			'name': 'Certificate',
			'description': 'Certificates and credentials',
			'retention_years': 10,
			'requires_approval': False,
			'template_available': False
		},
		{
			'name': 'Drawing',
			'description': 'Technical drawings and blueprints',
			'retention_years': 15,
			'requires_approval': True,
			'template_available': False
		}
	]

def get_default_folder_structure() -> List[Dict[str, Any]]:
	"""Get default folder structure"""
	return [
		{
			'name': 'Corporate',
			'description': 'Corporate documents and policies',
			'parent': None,
			'children': [
				'Policies',
				'Procedures',
				'Legal',
				'Compliance'
			]
		},
		{
			'name': 'Finance',
			'description': 'Financial documents and reports',
			'parent': None,
			'children': [
				'Invoices',
				'Contracts',
				'Reports',
				'Audits'
			]
		},
		{
			'name': 'HR',
			'description': 'Human resources documents',
			'parent': None,
			'children': [
				'Employee Files',
				'Policies',
				'Training',
				'Benefits'
			]
		},
		{
			'name': 'Operations',
			'description': 'Operational documents and procedures',
			'parent': None,
			'children': [
				'SOPs',
				'Work Instructions',
				'Quality',
				'Safety'
			]
		},
		{
			'name': 'IT',
			'description': 'IT documentation and specifications',
			'parent': None,
			'children': [
				'System Documentation',
				'User Manuals',
				'Security',
				'Infrastructure'
			]
		}
	]

def get_retention_policies() -> List[Dict[str, Any]]:
	"""Get default retention policies"""
	return [
		{
			'name': 'Financial Records',
			'description': 'Financial documents and records',
			'retention_years': 7,
			'auto_delete': False,
			'legal_hold_enabled': True
		},
		{
			'name': 'HR Records',
			'description': 'Employee and HR documents',
			'retention_years': 5,
			'auto_delete': False,
			'legal_hold_enabled': True
		},
		{
			'name': 'Contracts',
			'description': 'Legal contracts and agreements',
			'retention_years': 10,
			'auto_delete': False,
			'legal_hold_enabled': True
		},
		{
			'name': 'Technical Documentation',
			'description': 'Technical specifications and manuals',
			'retention_years': 5,
			'auto_delete': False,
			'legal_hold_enabled': False
		},
		{
			'name': 'Temporary Files',
			'description': 'Temporary and working documents',
			'retention_years': 1,
			'auto_delete': True,
			'legal_hold_enabled': False
		}
	]