"""
Document Management Blueprint

Flask blueprint for document management integration with Flask-AppBuilder.
Registers views, menu items, and provides initialization functions.
"""

from typing import List, Dict, Any, Optional
import logging

from flask import Blueprint
from flask_appbuilder import AppBuilder
from flask_babel import lazy_gettext as _

from .views import (
	GCDMDocumentModelView,
	GCDMFolderModelView,
	GCDMDocumentVersionModelView,
	GCDMPermissionModelView,
	GCDMWorkflowModelView,
	GCDMReviewModelView,
	GCDMRetentionPolicyModelView,
	GCDMSearchView,
	GCDMDashboardView
)

from .models import (
	GCDMDocument,
	GCDMDocumentVersion,
	GCDMDocumentCategory,
	GCDMDocumentType,
	GCDMFolder,
	GCDMPermission,
	GCDMCheckout,
	GCDMWorkflow,
	GCDMReview,
	GCDMTag,
	GCDMRetentionPolicy,
	GCDMArchive,
	GCDMAuditLog
)

from . import SUBCAPABILITY_META

logger = logging.getLogger(__name__)

# Create Flask blueprint
document_management_bp = Blueprint(
	'document_management',
	__name__,
	url_prefix='/document_management',
	template_folder='templates',
	static_folder='static'
)

def init_document_management(appbuilder: AppBuilder) -> None:
	"""
	Initialize document management sub-capability.
	
	Args:
		appbuilder: Flask-AppBuilder instance
	"""
	try:
		logger.info("Initializing Document Management sub-capability")
		
		# Register main views
		appbuilder.add_view(
			GCDMDocumentModelView,
			"Documents",
			icon="fa-file-text",
			category="Document Management",
			category_icon="fa-folder-open"
		)
		
		appbuilder.add_view(
			GCDMFolderModelView,
			"Folders",
			icon="fa-folder",
			category="Document Management"
		)
		
		appbuilder.add_view(
			GCDMDocumentVersionModelView,
			"Document Versions",
			icon="fa-code-fork",
			category="Document Management"
		)
		
		# Search and dashboard views
		appbuilder.add_view_no_menu(GCDMSearchView)
		appbuilder.add_view(
			GCDMSearchView,
			"Document Search",
			icon="fa-search",
			category="Document Management"
		)
		
		appbuilder.add_view_no_menu(GCDMDashboardView)
		appbuilder.add_view(
			GCDMDashboardView,
			"DM Dashboard",
			icon="fa-dashboard",
			category="Document Management"
		)
		
		# Administration views
		appbuilder.add_view(
			GCDMPermissionModelView,
			"Permissions",
			icon="fa-lock",
			category="Document Admin",
			category_icon="fa-cog"
		)
		
		appbuilder.add_view(
			GCDMWorkflowModelView,
			"Workflows",
			icon="fa-random",
			category="Document Admin"
		)
		
		appbuilder.add_view(
			GCDMReviewModelView,
			"Reviews & Approvals",
			icon="fa-check-square",
			category="Document Admin"
		)
		
		appbuilder.add_view(
			GCDMRetentionPolicyModelView,
			"Retention Policies",
			icon="fa-clock-o",
			category="Document Admin"
		)
		
		# Create default permissions
		_create_permissions(appbuilder)
		
		# Initialize default data
		_initialize_default_data(appbuilder)
		
		logger.info("Document Management sub-capability initialized successfully")
		
	except Exception as e:
		logger.error(f"Failed to initialize Document Management: {e}")
		raise

def _create_permissions(appbuilder: AppBuilder) -> None:
	"""Create default permissions for document management."""
	try:
		from flask_appbuilder.security.sqla.models import Permission, ViewMenu, PermissionView
		
		# Define document management permissions
		dm_permissions = [
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
		]
		
		# Create permissions if they don't exist
		for perm_name in dm_permissions:
			# Check if permission exists
			existing_perm = appbuilder.sm.find_permission(perm_name)
			if not existing_perm:
				# Create permission
				appbuilder.sm.add_permission(perm_name)
				logger.info(f"Created permission: {perm_name}")
		
		# Create view-menu permissions for Flask-AppBuilder views
		view_permissions = [
			('can_list', 'GCDMDocumentModelView'),
			('can_show', 'GCDMDocumentModelView'),
			('can_add', 'GCDMDocumentModelView'),
			('can_edit', 'GCDMDocumentModelView'),
			('can_delete', 'GCDMDocumentModelView'),
			('can_download', 'GCDMDocumentModelView'),
			('can_checkout', 'GCDMDocumentModelView'),
			('can_checkin', 'GCDMDocumentModelView'),
			('can_list', 'GCDMFolderModelView'),
			('can_show', 'GCDMFolderModelView'),
			('can_add', 'GCDMFolderModelView'),
			('can_edit', 'GCDMFolderModelView'),
			('can_delete', 'GCDMFolderModelView'),
			('can_search', 'GCDMSearchView'),
			('can_index', 'GCDMDashboardView')
		]
		
		for permission_name, view_name in view_permissions:
			pv = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
			if not pv:
				# Permission-view-menu will be created automatically by Flask-AppBuilder
				pass
		
		logger.info("Document management permissions created")
		
	except Exception as e:
		logger.error(f"Failed to create permissions: {e}")
		raise

def _initialize_default_data(appbuilder: AppBuilder) -> None:
	"""Initialize default document management data."""
	try:
		db_session = appbuilder.get_session
		
		# Create default document categories
		_create_default_categories(db_session)
		
		# Create default document types
		_create_default_document_types(db_session)
		
		# Create default folder structure
		_create_default_folders(db_session)
		
		# Create default retention policies
		_create_default_retention_policies(db_session)
		
		db_session.commit()
		logger.info("Default document management data initialized")
		
	except Exception as e:
		logger.error(f"Failed to initialize default data: {e}")
		db_session.rollback()
		raise

def _create_default_categories(db_session) -> None:
	"""Create default document categories."""
	from . import get_default_folder_structure
	
	# Check if categories already exist
	existing_count = db_session.query(GCDMDocumentCategory).count()
	if existing_count > 0:
		return
	
	default_categories = [
		{
			'name': 'Corporate Documents',
			'code': 'CORP',
			'description': 'Corporate policies, procedures, and governance documents',
			'color': '#007bff',
			'icon': 'fa-building'
		},
		{
			'name': 'Financial Documents',
			'code': 'FIN',
			'description': 'Financial reports, invoices, and accounting documents',
			'color': '#28a745',
			'icon': 'fa-dollar'
		},
		{
			'name': 'HR Documents',
			'code': 'HR',
			'description': 'Human resources policies, employee documents, and training materials',
			'color': '#17a2b8',
			'icon': 'fa-users'
		},
		{
			'name': 'Technical Documentation',
			'code': 'TECH',
			'description': 'Technical specifications, manuals, and system documentation',
			'color': '#6c757d',
			'icon': 'fa-cogs'
		},
		{
			'name': 'Legal Documents',
			'code': 'LEGAL',
			'description': 'Contracts, agreements, and legal correspondence',
			'color': '#dc3545',
			'icon': 'fa-gavel'
		}
	]
	
	for cat_data in default_categories:
		category = GCDMDocumentCategory(
			tenant_id='system',
			name=cat_data['name'],
			code=cat_data['code'],
			description=cat_data['description'],
			color=cat_data['color'],
			icon=cat_data['icon'],
			level=0,
			path=f"/{cat_data['code']}",
			is_active=True
		)
		db_session.add(category)
	
	logger.info("Default document categories created")

def _create_default_document_types(db_session) -> None:
	"""Create default document types."""
	from . import get_default_document_types
	
	# Check if document types already exist
	existing_count = db_session.query(GCDMDocumentType).count()
	if existing_count > 0:
		return
	
	default_types = get_default_document_types()
	
	for type_data in default_types:
		doc_type = GCDMDocumentType(
			tenant_id='system',
			name=type_data['name'],
			code=type_data['name'].upper().replace(' ', '_'),
			description=type_data['description'],
			default_retention_years=type_data['retention_years'],
			requires_approval=type_data['requires_approval'],
			has_template=type_data['template_available'],
			allowed_extensions='["pdf","doc","docx","xls","xlsx","ppt","pptx","txt"]',
			max_file_size_mb=100,
			requires_version_control=True,
			security_classification='internal',
			is_active=True
		)
		db_session.add(doc_type)
	
	logger.info("Default document types created")

def _create_default_folders(db_session) -> None:
	"""Create default folder structure."""
	from . import get_default_folder_structure
	
	# Check if folders already exist
	existing_count = db_session.query(GCDMFolder).count()
	if existing_count > 0:
		return
	
	folder_structure = get_default_folder_structure()
	
	for folder_data in folder_structure:
		# Create parent folder
		parent_folder = GCDMFolder(
			tenant_id='system',
			name=folder_data['name'],
			description=folder_data['description'],
			folder_path=f"/{folder_data['name']}",
			full_path=f"/{folder_data['name']}",
			level=0,
			document_count=0,
			total_size_mb=0,
			is_active=True
		)
		db_session.add(parent_folder)
		db_session.flush()  # Get parent folder ID
		
		# Create child folders
		for child_name in folder_data.get('children', []):
			child_folder = GCDMFolder(
				tenant_id='system',
				name=child_name,
				description=f"{child_name} documents",
				folder_path=f"/{folder_data['name']}/{child_name}",
				full_path=f"/{folder_data['name']}/{child_name}",
				parent_folder_id=parent_folder.id,
				level=1,
				document_count=0,
				total_size_mb=0,
				is_active=True
			)
			db_session.add(child_folder)
	
	logger.info("Default folder structure created")

def _create_default_retention_policies(db_session) -> None:
	"""Create default retention policies."""
	from . import get_retention_policies
	
	# Check if retention policies already exist
	existing_count = db_session.query(GCDMRetentionPolicy).count()
	if existing_count > 0:
		return
	
	retention_policies = get_retention_policies()
	
	for policy_data in retention_policies:
		policy = GCDMRetentionPolicy(
			tenant_id='system',
			name=policy_data['name'],
			description=policy_data['description'],
			policy_code=policy_data['name'].upper().replace(' ', '_'),
			retention_period_years=policy_data['retention_years'],
			retention_period_months=0,
			retention_period_days=0,
			trigger_event='creation_date',
			auto_delete_enabled=policy_data['auto_delete'],
			auto_archive_enabled=True,
			legal_hold_override=policy_data['legal_hold_enabled'],
			require_approval_for_deletion=True,
			notify_before_deletion_days=30,
			is_active=True,
			effective_date=appbuilder.get_session.execute('SELECT CURRENT_DATE').scalar()
		)
		db_session.add(policy)
	
	logger.info("Default retention policies created")

def register_views(appbuilder: AppBuilder, menu_items: List[Dict[str, Any]]) -> None:
	"""
	Register document management views with custom menu structure.
	
	Args:
		appbuilder: Flask-AppBuilder instance
		menu_items: List of menu items from metadata
	"""
	try:
		# Register views based on menu items configuration
		for menu_item in menu_items:
			view_name = menu_item['endpoint'].split('.')[0]
			
			# Map view names to view classes
			view_mapping = {
				'GCDMDocumentModelView': GCDMDocumentModelView,
				'GCDMFolderModelView': GCDMFolderModelView,
				'GCDMDocumentVersionModelView': GCDMDocumentVersionModelView,
				'GCDMPermissionModelView': GCDMPermissionModelView,
				'GCDMWorkflowModelView': GCDMWorkflowModelView,
				'GCDMReviewModelView': GCDMReviewModelView,
				'GCDMRetentionPolicyModelView': GCDMRetentionPolicyModelView,
				'GCDMSearchView': GCDMSearchView,
				'GCDMDashboardView': GCDMDashboardView
			}
			
			view_class = view_mapping.get(view_name)
			if view_class:
				appbuilder.add_view(
					view_class,
					menu_item['name'],
					icon=menu_item['icon'],
					category="Document Management"
				)
		
		logger.info("Document management views registered successfully")
		
	except Exception as e:
		logger.error(f"Failed to register views: {e}")
		raise

def get_api_endpoints() -> List[Dict[str, Any]]:
	"""
	Get API endpoint information for document management.
	
	Returns:
		List of API endpoint definitions
	"""
	return [
		{
			'path': '/api/general_cross_functional/dm/documents',
			'methods': ['GET', 'POST'],
			'description': 'Document CRUD operations',
			'view_function': 'document_api'
		},
		{
			'path': '/api/general_cross_functional/dm/documents/<id>',
			'methods': ['GET', 'PUT', 'DELETE'],
			'description': 'Individual document operations',
			'view_function': 'document_detail_api'
		},
		{
			'path': '/api/general_cross_functional/dm/documents/<id>/download',
			'methods': ['GET'],
			'description': 'Download document file',
			'view_function': 'document_download_api'
		},
		{
			'path': '/api/general_cross_functional/dm/documents/<id>/checkout',
			'methods': ['POST'],
			'description': 'Checkout document',
			'view_function': 'document_checkout_api'
		},
		{
			'path': '/api/general_cross_functional/dm/documents/<id>/checkin',
			'methods': ['POST'],
			'description': 'Checkin document',
			'view_function': 'document_checkin_api'
		},
		{
			'path': '/api/general_cross_functional/dm/documents/<id>/versions',
			'methods': ['GET', 'POST'],
			'description': 'Document version management',
			'view_function': 'document_versions_api'
		},
		{
			'path': '/api/general_cross_functional/dm/folders',
			'methods': ['GET', 'POST'],
			'description': 'Folder operations',
			'view_function': 'folder_api'
		},
		{
			'path': '/api/general_cross_functional/dm/search',
			'methods': ['GET', 'POST'],
			'description': 'Document search',
			'view_function': 'search_api'
		},
		{
			'path': '/api/general_cross_functional/dm/dashboard',
			'methods': ['GET'],
			'description': 'Dashboard statistics',
			'view_function': 'dashboard_api'
		}
	]

def get_database_models() -> List[str]:
	"""
	Get list of database model class names.
	
	Returns:
		List of model class names
	"""
	return [
		'GCDMDocument',
		'GCDMDocumentVersion', 
		'GCDMDocumentCategory',
		'GCDMDocumentType',
		'GCDMFolder',
		'GCDMPermission',
		'GCDMCheckout',
		'GCDMWorkflow',
		'GCDMReview',
		'GCDMTag',
		'GCDMDocumentTag',
		'GCDMMetadata',
		'GCDMRetentionPolicy',
		'GCDMArchive',
		'GCDMAuditLog'
	]

def get_subcapability_info() -> Dict[str, Any]:
	"""
	Get sub-capability information.
	
	Returns:
		Sub-capability metadata dictionary
	"""
	return SUBCAPABILITY_META

def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Validate document management configuration.
	
	Args:
		config: Configuration dictionary
		
	Returns:
		Validation result with errors and warnings
	"""
	errors = []
	warnings = []
	
	# Validate storage configuration
	storage_root = config.get('DOCUMENT_STORAGE_ROOT')
	if not storage_root:
		errors.append("DOCUMENT_STORAGE_ROOT configuration is required")
	
	# Validate file size limits
	max_file_size = config.get('MAX_DOCUMENT_SIZE_MB', 100)
	if max_file_size > 1000:
		warnings.append(f"Large file size limit ({max_file_size}MB) may impact performance")
	
	# Validate allowed extensions
	allowed_extensions = config.get('ALLOWED_DOCUMENT_EXTENSIONS', [])
	if not allowed_extensions:
		warnings.append("No file extensions specified - all file types will be allowed")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}