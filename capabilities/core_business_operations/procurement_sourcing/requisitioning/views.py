"""
Requisitioning Views

Flask-AppBuilder views for requisition management including CRUD operations,
approval workflows, and dashboard functionality.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
from decimal import Decimal
from flask import request, redirect, url_for, flash, jsonify
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget
from wtforms import validators, widgets
from wtforms.fields import StringField, TextAreaField, SelectField, DateField, DecimalField, IntegerField

from .models import PPRRequisition, PPRRequisitionLine, PPRApprovalWorkflow, PPRRequisitionComment
from .service import RequisitioningService


class RequisitionView(ModelView):
	"""Main view for requisition management"""
	
	datamodel = SQLAInterface(PPRRequisition)
	
	# List view configuration
	list_columns = [
		'requisition_number', 'title', 'requestor_name', 'department',
		'request_date', 'required_date', 'status', 'priority', 'total_amount'
	]
	
	list_template = 'requisition_list.html'
	
	search_columns = [
		'requisition_number', 'title', 'requestor_name', 'department',
		'status', 'description', 'business_justification'
	]
	
	# Show view configuration
	show_columns = [
		'requisition_number', 'title', 'description', 'business_justification',
		'requestor_name', 'requestor_email', 'department', 'cost_center',
		'request_date', 'required_date', 'delivery_location',
		'status', 'workflow_status', 'priority',
		'currency_code', 'subtotal_amount', 'tax_amount', 'total_amount',
		'budget_year', 'budget_period', 'budget_account_id', 'budget_checked', 'budget_available',
		'approval_level', 'current_approver_name',
		'approved', 'approved_by', 'approved_date', 'approved_amount',
		'rejected', 'rejected_by', 'rejected_date', 'rejection_reason',
		'submitted', 'submitted_date',
		'converted_to_po', 'purchase_order_id', 'conversion_date',
		'project_name', 'activity_code',
		'rush_order', 'drop_ship', 'special_instructions',
		'attachment_count', 'notes', 'internal_notes',
		'created_date', 'modified_date'
	]
	
	show_template = 'requisition_show.html'
	
	# Edit view configuration
	edit_columns = [
		'title', 'description', 'business_justification',
		'required_date', 'delivery_location', 'priority',
		'budget_account_id', 'project_id', 'project_name', 'activity_code',
		'rush_order', 'drop_ship', 'special_instructions', 'notes'
	]
	
	edit_template = 'requisition_edit.html'
	
	# Add view configuration
	add_columns = [
		'title', 'description', 'business_justification',
		'requestor_name', 'requestor_email', 'department', 'cost_center',
		'required_date', 'delivery_location', 'priority',
		'budget_account_id', 'project_id', 'project_name', 'activity_code',
		'rush_order', 'drop_ship', 'special_instructions', 'notes'
	]
	
	add_template = 'requisition_add.html'
	
	# Formatters
	formatters_columns = {
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'subtotal_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'tax_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'approved_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'budget_available': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: self._format_status_badge(x),
		'priority': lambda x: self._format_priority_badge(x),
		'approved': lambda x: '✓' if x else '✗',
		'rejected': lambda x: '✓' if x else '✗',
		'converted_to_po': lambda x: '✓' if x else '✗'
	}
	
	# Permissions
	base_permissions = [
		'can_list', 'can_show', 'can_add', 'can_edit', 'can_delete'
	]
	
	# Custom methods
	
	@expose('/approve/<string:pk>')
	@has_access
	def approve(self, pk: str):
		"""Approve requisition"""
		
		service = RequisitioningService(self.get_tenant_id())
		
		try:
			requisition = service.approve_requisition(
				pk, 
				self.get_current_user_id(),
				request.form.get('comments')
			)
			flash(f'Requisition {requisition.requisition_number} approved successfully', 'success')
			
		except Exception as e:
			flash(f'Error approving requisition: {str(e)}', 'error')
		
		return redirect(url_for('RequisitionView.show', pk=pk))
	
	@expose('/reject/<string:pk>')
	@has_access
	def reject(self, pk: str):
		"""Reject requisition"""
		
		service = RequisitioningService(self.get_tenant_id())
		
		try:
			reason = request.form.get('reason')
			if not reason:
				flash('Rejection reason is required', 'error')
				return redirect(url_for('RequisitionView.show', pk=pk))
			
			requisition = service.reject_requisition(pk, self.get_current_user_id(), reason)
			flash(f'Requisition {requisition.requisition_number} rejected', 'warning')
			
		except Exception as e:
			flash(f'Error rejecting requisition: {str(e)}', 'error')
		
		return redirect(url_for('RequisitionView.show', pk=pk))
	
	@expose('/submit/<string:pk>')
	@has_access
	def submit(self, pk: str):
		"""Submit requisition for approval"""
		
		service = RequisitioningService(self.get_tenant_id())
		
		try:
			requisition = service.submit_requisition(pk, self.get_current_user_id())
			flash(f'Requisition {requisition.requisition_number} submitted for approval', 'success')
			
		except Exception as e:
			flash(f'Error submitting requisition: {str(e)}', 'error')
		
		return redirect(url_for('RequisitionView.show', pk=pk))
	
	@expose('/cancel/<string:pk>')
	@has_access
	def cancel(self, pk: str):
		"""Cancel requisition"""
		
		service = RequisitioningService(self.get_tenant_id())
		
		try:
			reason = request.form.get('reason', 'Cancelled by user')
			requisition = service.cancel_requisition(pk, self.get_current_user_id(), reason)
			flash(f'Requisition {requisition.requisition_number} cancelled', 'warning')
			
		except Exception as e:
			flash(f'Error cancelling requisition: {str(e)}', 'error')
		
		return redirect(url_for('RequisitionView.show', pk=pk))
	
	@expose('/convert_to_po/<string:pk>')
	@has_access
	def convert_to_po(self, pk: str):
		"""Convert requisition to purchase order"""
		
		# This would integrate with Purchase Order Management
		flash('Convert to PO functionality requires Purchase Order Management module', 'info')
		return redirect(url_for('RequisitionView.show', pk=pk))
	
	@expose('/add_comment/<string:pk>', methods=['POST'])
	@has_access
	def add_comment(self, pk: str):
		"""Add comment to requisition"""
		
		service = RequisitioningService(self.get_tenant_id())
		
		try:
			comment_text = request.form.get('comment_text')
			comment_type = request.form.get('comment_type', 'General')
			is_internal = request.form.get('is_internal') == 'true'
			
			if comment_text:
				service.add_requisition_comment(
					pk, comment_text, self.get_current_user_id(),
					comment_type, is_internal
				)
				flash('Comment added successfully', 'success')
			else:
				flash('Comment text is required', 'error')
				
		except Exception as e:
			flash(f'Error adding comment: {str(e)}', 'error')
		
		return redirect(url_for('RequisitionView.show', pk=pk))
	
	def _format_status_badge(self, status: str) -> str:
		"""Format status as colored badge"""
		color_map = {
			'Draft': 'secondary',
			'Submitted': 'info', 
			'Approved': 'success',
			'Rejected': 'danger',
			'Cancelled': 'warning',
			'Converted': 'primary'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'
	
	def _format_priority_badge(self, priority: str) -> str:
		"""Format priority as colored badge"""
		color_map = {
			'Low': 'light',
			'Normal': 'secondary',
			'High': 'warning', 
			'Urgent': 'danger'
		}
		color = color_map.get(priority, 'secondary')
		return f'<span class="badge badge-{color}">{priority}</span>'
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# TODO: Implement tenant resolution
		return "default_tenant"
	
	def get_current_user_id(self) -> str:
		"""Get current user ID"""
		# TODO: Get from Flask-Login or similar
		return "current_user"


class RequisitionLineView(ModelView):
	"""View for requisition line items"""
	
	datamodel = SQLAInterface(PPRRequisitionLine)
	
	# List view configuration
	list_columns = [
		'requisition.requisition_number', 'line_number', 'description',
		'item_code', 'quantity_requested', 'unit_of_measure',
		'unit_price', 'line_amount', 'tax_amount'
	]
	
	search_columns = [
		'description', 'item_code', 'item_description', 'manufacturer',
		'model_number', 'part_number'
	]
	
	# Show view configuration  
	show_columns = [
		'requisition.requisition_number', 'line_number', 'description', 'detailed_specification',
		'item_code', 'item_description', 'item_category', 'manufacturer', 'model_number', 'part_number',
		'quantity_requested', 'unit_of_measure', 'unit_price', 'line_amount',
		'tax_code', 'tax_rate', 'tax_amount', 'is_tax_inclusive',
		'gl_account_id', 'cost_center', 'department', 'project_id', 'activity_code',
		'required_date', 'delivery_location', 'special_instructions',
		'preferred_vendor_name', 'vendor_part_number',
		'is_asset', 'asset_category', 'useful_life_years',
		'is_service', 'service_period_start', 'service_period_end',
		'warranty_required', 'warranty_period', 'technical_specs',
		'line_status', 'notes'
	]
	
	# Edit/Add view configuration
	edit_columns = [
		'line_number', 'description', 'detailed_specification',
		'item_code', 'item_description', 'item_category', 'manufacturer', 'model_number', 'part_number',
		'quantity_requested', 'unit_of_measure', 'unit_price',
		'tax_code', 'tax_rate', 'is_tax_inclusive',
		'gl_account_id', 'cost_center', 'department', 'project_id', 'activity_code',
		'required_date', 'delivery_location', 'special_instructions',
		'preferred_vendor_id', 'preferred_vendor_name', 'vendor_part_number',
		'is_asset', 'asset_category', 'useful_life_years',
		'is_service', 'service_period_start', 'service_period_end',
		'warranty_required', 'warranty_period', 'technical_specs', 'notes'
	]
	
	add_columns = edit_columns
	
	# Formatters
	formatters_columns = {
		'line_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'tax_amount': lambda x: f"${x:,.2f}" if x else "$0.00", 
		'unit_price': lambda x: f"${x:,.4f}" if x else "$0.0000",
		'quantity_requested': lambda x: f"{x:,.4f}" if x else "0.0000",
		'is_asset': lambda x: '✓' if x else '✗',
		'is_service': lambda x: '✓' if x else '✗',
		'warranty_required': lambda x: '✓' if x else '✗',
		'is_tax_inclusive': lambda x: '✓' if x else '✗'
	}


class ApprovalWorkflowView(ModelView):
	"""View for approval workflow steps"""
	
	datamodel = SQLAInterface(PPRApprovalWorkflow)
	
	# List view configuration
	list_columns = [
		'requisition.requisition_number', 'step_order', 'step_name',
		'approver_name', 'status', 'assigned_date', 'due_date'
	]
	
	search_columns = [
		'step_name', 'approver_name', 'approver_role', 'status'
	]
	
	# Show view configuration
	show_columns = [
		'requisition.requisition_number', 'step_order', 'step_name', 'step_description',
		'approver_name', 'approver_email', 'approver_role',
		'required', 'approval_limit', 'parallel_approval',
		'status', 'assigned_date', 'due_date', 'completed',
		'approved', 'approved_date', 'rejected', 'rejected_date',
		'comments', 'rejection_reason',
		'delegated_to_name', 'delegated_date', 'delegation_reason',
		'escalated', 'escalated_date', 'escalation_reason',
		'notification_sent', 'reminder_count', 'last_reminder_date'
	]
	
	# Formatters
	formatters_columns = {
		'status': lambda x: self._format_workflow_status_badge(x),
		'approval_limit': lambda x: f"${x:,.2f}" if x else "No Limit",
		'required': lambda x: '✓' if x else '✗',
		'parallel_approval': lambda x: '✓' if x else '✗',
		'approved': lambda x: '✓' if x else '✗',
		'rejected': lambda x: '✓' if x else '✗',
		'escalated': lambda x: '✓' if x else '✗',
		'notification_sent': lambda x: '✓' if x else '✗'
	}
	
	def _format_workflow_status_badge(self, status: str) -> str:
		"""Format workflow status as colored badge"""
		color_map = {
			'Pending': 'warning',
			'Approved': 'success',
			'Rejected': 'danger',
			'Skipped': 'secondary'
		}
		color = color_map.get(status, 'secondary')
		return f'<span class="badge badge-{color}">{status}</span>'


class RequisitionCommentView(ModelView):
	"""View for requisition comments"""
	
	datamodel = SQLAInterface(PPRRequisitionComment)
	
	# List view configuration
	list_columns = [
		'requisition.requisition_number', 'comment_type', 'author_name',
		'comment_date', 'is_internal', 'requires_response'
	]
	
	search_columns = ['comment_text', 'author_name', 'comment_type']
	
	# Show view configuration
	show_columns = [
		'requisition.requisition_number', 'comment_text', 'comment_type',
		'is_internal', 'is_system_generated',
		'author_name', 'author_role', 'comment_date',
		'visible_to_requestor', 'visible_to_approvers', 'requires_response',
		'parent_comment_id', 'has_responses', 'response_count',
		'comment_status'
	]
	
	# Add/Edit view configuration
	add_columns = [
		'comment_text', 'comment_type', 'is_internal',
		'visible_to_requestor', 'visible_to_approvers', 'requires_response'
	]
	
	edit_columns = ['comment_text', 'comment_type', 'is_internal']
	
	# Formatters
	formatters_columns = {
		'comment_type': lambda x: self._format_comment_type_badge(x),
		'is_internal': lambda x: '✓' if x else '✗',
		'is_system_generated': lambda x: '✓' if x else '✗',
		'visible_to_requestor': lambda x: '✓' if x else '✗',
		'visible_to_approvers': lambda x: '✓' if x else '✗',
		'requires_response': lambda x: '✓' if x else '✗',
		'has_responses': lambda x: '✓' if x else '✗'
	}
	
	def _format_comment_type_badge(self, comment_type: str) -> str:
		"""Format comment type as colored badge"""
		color_map = {
			'General': 'secondary',
			'Approval': 'success',
			'Rejection': 'danger',
			'Question': 'info',
			'Answer': 'primary'
		}
		color = color_map.get(comment_type, 'secondary')
		return f'<span class="badge badge-{color}">{comment_type}</span>'


class RequisitionDashboardView(BaseView):
	"""Dashboard view for requisitioning metrics and analytics"""
	
	route_base = "/requisitioning/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard view"""
		
		service = RequisitioningService(self.get_tenant_id())
		
		# Get dashboard data
		dashboard_data = self._get_dashboard_data(service)
		
		return self.render_template(
			'requisitioning_dashboard.html',
			dashboard_data=dashboard_data,
			title="Requisitioning Dashboard"
		)
	
	@expose('/metrics')
	@has_access
	def metrics(self):
		"""Detailed metrics view"""
		
		service = RequisitioningService(self.get_tenant_id())
		
		# Get date range from request
		date_from = request.args.get('date_from')
		date_to = request.args.get('date_to')
		
		# Convert to date objects
		if date_from:
			date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
		if date_to:
			date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
		
		metrics = service.get_requisition_metrics(date_from, date_to)
		
		return jsonify(metrics)
	
	@expose('/my_approvals')
	@has_access
	def my_approvals(self):
		"""View for approvals pending for current user"""
		
		service = RequisitioningService(self.get_tenant_id())
		
		pending_approvals = service.get_requisitions_for_approval(self.get_current_user_id())
		
		return self.render_template(
			'my_approvals.html',
			approvals=pending_approvals,
			title="My Pending Approvals"
		)
	
	@expose('/my_requisitions')
	@has_access  
	def my_requisitions(self):
		"""View for current user's requisitions"""
		
		service = RequisitioningService(self.get_tenant_id())
		
		my_requisitions = service.get_requisitions_by_requestor(self.get_current_user_id())
		
		return self.render_template(
			'my_requisitions.html',
			requisitions=my_requisitions,
			title="My Requisitions"
		)
	
	def _get_dashboard_data(self, service: RequisitioningService) -> Dict[str, Any]:
		"""Get data for dashboard"""
		
		# Get basic metrics
		metrics = service.get_requisition_metrics()
		
		# Get status counts
		status_counts = {}
		for status in ['Draft', 'Submitted', 'Approved', 'Rejected', 'Cancelled']:
			count = len(service.get_requisitions_by_status(status, limit=1000))
			status_counts[status.lower()] = count
		
		# Get pending approvals for current user
		pending_approvals = service.get_requisitions_for_approval(self.get_current_user_id())
		
		# Get overdue approvals
		overdue_approvals = service.get_overdue_approvals()
		
		# Get user's recent requisitions
		my_recent = service.get_requisitions_by_requestor(self.get_current_user_id(), limit=5)
		
		return {
			'metrics': metrics,
			'status_counts': status_counts,
			'pending_approvals': len(pending_approvals),
			'overdue_approvals': len(overdue_approvals),
			'my_recent_requisitions': len(my_recent),
			'avg_approval_time': metrics.get('avg_approval_time_hours', 0),
			'approval_rate': metrics.get('approval_rate', 0),
			'recent_requisitions': my_recent[:3],  # Show top 3
			'urgent_approvals': [r for r in pending_approvals if r.priority == 'Urgent']
		}
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# TODO: Implement tenant resolution
		return "default_tenant"
	
	def get_current_user_id(self) -> str:
		"""Get current user ID"""
		# TODO: Get from Flask-Login or similar
		return "current_user"


class RequisitionChartView(DirectByChartView):
	"""Chart views for requisition analytics"""
	
	datamodel = SQLAInterface(PPRRequisition)
	chart_title = 'Requisition Status Distribution'
	chart_type = 'PieChart'
	direct_by_column = 'status'
	
	definitions = [
		{
			'group': 'status',
			'series': ['status']
		}
	]