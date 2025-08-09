"""
Requisitioning Sub-capability Blueprint

Blueprint registration for requisitioning views, APIs, and menu structure.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder
from typing import Dict, Any

from .views import (
	RequisitionView, RequisitionLineView, ApprovalWorkflowView,
	RequisitionCommentView, RequisitionDashboardView, RequisitionChartView
)


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Requisitioning sub-capability views"""
	
	# Register main views
	appbuilder.add_view(
		RequisitionView,
		"Requisitions",
		icon="fa-list-alt",
		category="Requisitioning",
		category_icon="fa-list-alt"
	)
	
	appbuilder.add_view(
		RequisitionLineView, 
		"Requisition Lines",
		icon="fa-list",
		category="Requisitioning"
	)
	
	appbuilder.add_view(
		ApprovalWorkflowView,
		"Approval Workflows", 
		icon="fa-check-circle",
		category="Requisitioning"
	)
	
	appbuilder.add_view(
		RequisitionCommentView,
		"Requisition Comments",
		icon="fa-comments",
		category="Requisitioning"
	)
	
	# Register dashboard view
	appbuilder.add_view(
		RequisitionDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Requisitioning"
	)
	
	# Register chart view
	appbuilder.add_view(
		RequisitionChartView,
		"Status Chart",
		icon="fa-bar-chart",
		category="Requisitioning"
	)
	
	# Add menu links for common actions
	appbuilder.add_link(
		"My Requisitions",
		href="/requisitioning/dashboard/my_requisitions",
		icon="fa-user",
		category="Requisitioning"
	)
	
	appbuilder.add_link(
		"My Approvals", 
		href="/requisitioning/dashboard/my_approvals",
		icon="fa-gavel",
		category="Requisitioning"
	)
	
	appbuilder.add_link(
		"Create Requisition",
		href="/requisitionview/add",
		icon="fa-plus",
		category="Requisitioning"
	)


def register_permissions(appbuilder: AppBuilder):
	"""Register Requisitioning-specific permissions"""
	
	permissions = [
		# Requisition permissions
		('can_create_requisition', 'RequisitionView'),
		('can_edit_requisition', 'RequisitionView'),
		('can_approve_requisition', 'RequisitionView'),
		('can_reject_requisition', 'RequisitionView'),
		('can_cancel_requisition', 'RequisitionView'),
		('can_submit_requisition', 'RequisitionView'),
		('can_convert_to_po', 'RequisitionView'),
		('can_view_all_requisitions', 'RequisitionView'),
		('can_override_approvals', 'RequisitionView'),
		
		# Comment permissions
		('can_add_comment', 'RequisitionCommentView'),
		('can_edit_comment', 'RequisitionCommentView'),
		('can_delete_comment', 'RequisitionCommentView'),
		('can_view_internal_comments', 'RequisitionCommentView'),
		
		# Workflow permissions
		('can_view_workflow', 'ApprovalWorkflowView'),
		('can_delegate_approval', 'ApprovalWorkflowView'),
		('can_escalate_approval', 'ApprovalWorkflowView'),
		
		# Dashboard permissions
		('can_view_dashboard', 'RequisitionDashboardView'),
		('can_view_metrics', 'RequisitionDashboardView'),
		('can_export_data', 'RequisitionDashboardView'),
		
		# Administrative permissions
		('can_manage_approval_rules', 'RequisitionView'),
		('can_configure_workflows', 'RequisitionView'),
		('can_bulk_approve', 'RequisitionView'),
		('can_bulk_reject', 'RequisitionView')
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure() -> Dict[str, Any]:
	"""Get menu structure for Requisitioning sub-capability"""
	
	return {
		'name': 'Requisitioning',
		'icon': 'fa-list-alt',
		'items': [
			{
				'name': 'Dashboard',
				'href': '/requisitioning/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_view_dashboard on RequisitionDashboardView'
			},
			{
				'name': 'My Requisitions',
				'href': '/requisitioning/dashboard/my_requisitions',
				'icon': 'fa-user',
				'permission': 'can_list on RequisitionView'
			},
			{
				'name': 'My Approvals',
				'href': '/requisitioning/dashboard/my_approvals', 
				'icon': 'fa-gavel',
				'permission': 'can_approve_requisition on RequisitionView'
			},
			{
				'name': 'Create Requisition',
				'href': '/requisitionview/add',
				'icon': 'fa-plus',
				'permission': 'can_create_requisition on RequisitionView'
			},
			{
				'name': 'All Requisitions',
				'href': '/requisitionview/list/',
				'icon': 'fa-list',
				'permission': 'can_view_all_requisitions on RequisitionView'
			},
			{
				'name': 'Approval Workflows',
				'href': '/approvalworkflowview/list/',
				'icon': 'fa-check-circle',
				'permission': 'can_view_workflow on ApprovalWorkflowView'
			},
			{
				'name': 'Status Chart',
				'href': '/requisitionchartview/',
				'icon': 'fa-bar-chart',
				'permission': 'can_view_dashboard on RequisitionDashboardView'
			}
		]
	}


def create_subcapability_blueprint() -> Blueprint:
	"""Create Flask blueprint for Requisitioning sub-capability"""
	
	req_bp = Blueprint(
		'requisitioning',
		__name__,
		url_prefix='/requisitioning',
		template_folder='templates',
		static_folder='static'
	)
	
	return req_bp


def get_subcapability_info() -> Dict[str, Any]:
	"""Get Requisitioning sub-capability information"""
	
	from . import get_subcapability_info
	return get_subcapability_info()


def validate_prerequisites() -> Dict[str, Any]:
	"""Validate prerequisites for Requisitioning sub-capability"""
	
	validation = {
		'valid': True,
		'errors': [],
		'warnings': [],
		'missing_dependencies': []
	}
	
	# Check for required tables/models
	required_models = [
		'PPRRequisition', 'PPRRequisitionLine', 
		'PPRApprovalWorkflow', 'PPRRequisitionComment'
	]
	
	# Check for integration dependencies
	integration_dependencies = [
		'auth_rbac',  # For user management and permissions
		'core_financials.general_ledger'  # For GL account validation
	]
	
	# TODO: Add actual validation logic
	# This would check database tables, required services, etc.
	
	return validation


def get_default_configuration() -> Dict[str, Any]:
	"""Get default configuration for Requisitioning sub-capability"""
	
	return {
		'approval_workflows': {
			'default_workflow': 'amount_based',
			'escalation_hours': 48,
			'auto_approve_threshold': 0,  # Amount below which auto-approval occurs
			'require_manager_approval': 1000,  # Amount above which manager approval required
			'require_finance_approval': 25000,  # Amount above which finance approval required
			'require_executive_approval': 100000  # Amount above which executive approval required
		},
		'notifications': {
			'enabled': True,
			'email_notifications': True,
			'sms_notifications': False,
			'slack_notifications': False,
			'reminder_intervals': [24, 48, 72]  # Hours
		},
		'document_management': {
			'allow_attachments': True,
			'max_attachment_size_mb': 10,
			'allowed_file_types': ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.jpg', '.png'],
			'require_attachments': False
		},
		'budget_integration': {
			'enabled': True,
			'check_budget_availability': True,
			'create_encumbrances': True,
			'allow_budget_overruns': False
		},
		'mobile_support': {
			'enabled': True,
			'allow_mobile_submission': True,
			'allow_mobile_approval': True,
			'offline_support': False
		}
	}


def get_integration_points() -> Dict[str, Any]:
	"""Get integration points for Requisitioning sub-capability"""
	
	return {
		'inbound_integrations': {
			'user_management': {
				'description': 'User information for requestors and approvers',
				'required': True,
				'endpoints': ['get_user_info', 'get_manager_hierarchy']
			},
			'budget_system': {
				'description': 'Budget checking and encumbrance creation',
				'required': False,
				'endpoints': ['check_budget_availability', 'create_encumbrance']
			},
			'chart_of_accounts': {
				'description': 'GL account validation',
				'required': True,
				'endpoints': ['validate_gl_account', 'get_cost_centers']
			}
		},
		'outbound_integrations': {
			'purchase_order_management': {
				'description': 'Requisition to PO conversion',
				'required': False,
				'endpoints': ['create_po_from_requisition']
			},
			'notification_system': {
				'description': 'Approval notifications and alerts',
				'required': False,
				'endpoints': ['send_notification', 'create_alert']
			},
			'audit_logging': {
				'description': 'Activity and audit trail logging',
				'required': True,
				'endpoints': ['log_activity', 'create_audit_entry']
			}
		}
	}