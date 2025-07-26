"""
Document Management Views

Flask-AppBuilder views for document management user interface including
document listing, upload, version management, permissions, and dashboard.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
import json
import logging

from flask import redirect, flash, request, url_for, send_file, jsonify, render_template_string
from flask_appbuilder import ModelView, BaseView, has_access, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.actions import action
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget
from flask_appbuilder.forms import DynamicForm
from flask_babel import lazy_gettext as _
from wtforms import StringField, TextAreaField, SelectField, FileField, BooleanField
from wtforms.validators import DataRequired, Length, Optional as OptionalValidator
from wtforms.widgets import TextArea
from sqlalchemy import and_, or_, desc, func

from .models import (
	GCDMDocument, GCDMDocumentVersion, GCDMDocumentCategory, GCDMDocumentType,
	GCDMFolder, GCDMPermission, GCDMCheckout, GCDMWorkflow, GCDMReview,
	GCDMTag, GCDMRetentionPolicy, GCDMArchive, GCDMAuditLog
)
from .service import DocumentService

logger = logging.getLogger(__name__)

class DocumentUploadForm(DynamicForm):
	"""Form for document upload."""
	title = StringField(
		_('Title'),
		validators=[DataRequired(), Length(max=500)]
	)
	description = TextAreaField(
		_('Description'),
		validators=[OptionalValidator(), Length(max=2000)]
	)
	category_id = SelectField(
		_('Category'),
		validators=[DataRequired()]
	)
	document_type_id = SelectField(
		_('Document Type'),
		validators=[DataRequired()]
	)
	folder_id = SelectField(
		_('Folder'),
		validators=[DataRequired()]
	)
	file = FileField(
		_('File'),
		validators=[DataRequired()]
	)
	keywords = StringField(
		_('Keywords'),
		validators=[OptionalValidator(), Length(max=500)]
	)
	tags = StringField(
		_('Tags (comma-separated)'),
		validators=[OptionalValidator(), Length(max=200)]
	)

class DocumentEditForm(DynamicForm):
	"""Form for document editing."""
	title = StringField(
		_('Title'),
		validators=[DataRequired(), Length(max=500)]
	)
	description = TextAreaField(
		_('Description'),
		validators=[OptionalValidator(), Length(max=2000)]
	)
	category_id = SelectField(
		_('Category'),
		validators=[DataRequired()]
	)
	document_type_id = SelectField(
		_('Document Type'),
		validators=[DataRequired()]
	)
	keywords = StringField(
		_('Keywords'),
		validators=[OptionalValidator(), Length(max=500)]
	)

class GCDMDocumentModelView(ModelView):
	"""Document management main view."""
	
	datamodel = SQLAInterface(GCDMDocument)
	
	# List view configuration
	list_columns = [
		'document_number', 'title', 'category.name', 'document_type.name',
		'status', 'owner_user_id', 'document_date', 'file_size_bytes', 'version_number'
	]
	
	list_title = _('Documents')
	show_title = _('Document Details')
	add_title = _('Upload Document')
	edit_title = _('Edit Document')
	
	# Search configuration
	search_columns = ['title', 'description', 'keywords', 'document_number']
	search_form_extra_fields = {
		'category_filter': SelectField(_('Category')),
		'document_type_filter': SelectField(_('Document Type')),
		'status_filter': SelectField(_('Status')),
		'date_from': StringField(_('Date From (YYYY-MM-DD)')),
		'date_to': StringField(_('Date To (YYYY-MM-DD)'))
	}
	
	# Column configuration
	label_columns = {
		'document_number': _('Document Number'),
		'title': _('Title'),
		'description': _('Description'),
		'category.name': _('Category'),
		'document_type.name': _('Document Type'),
		'folder.name': _('Folder'),
		'status': _('Status'),
		'owner_user_id': _('Owner'),
		'document_date': _('Document Date'),
		'file_name': _('File Name'),
		'file_size_bytes': _('File Size'),
		'version_number': _('Version'),
		'view_count': _('Views'),
		'download_count': _('Downloads'),
		'is_checked_out': _('Checked Out'),
		'checked_out_by': _('Checked Out By'),
		'security_classification': _('Security Level'),
		'retention_date': _('Retention Date'),
		'created_on': _('Created On'),
		'changed_on': _('Last Modified')
	}
	
	# Show view configuration
	show_columns = [
		'document_number', 'title', 'description', 'category.name', 'document_type.name',
		'folder.name', 'status', 'owner_user_id', 'document_date', 'effective_date',
		'file_name', 'file_size_bytes', 'version_number', 'view_count', 'download_count',
		'is_checked_out', 'checked_out_by', 'checked_out_date', 'security_classification',
		'retention_date', 'created_on', 'changed_on', 'created_by', 'changed_by'
	]
	
	# Edit configuration
	edit_columns = ['title', 'description', 'category_id', 'document_type_id', 'keywords']
	
	# Add configuration - use custom form
	add_form = DocumentUploadForm
	edit_form = DocumentEditForm
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Custom formatters
	formatters_columns = {
		'file_size_bytes': lambda x: f"{x / 1024 / 1024:.2f} MB" if x else "0 MB",
		'status': lambda x: x.title() if x else "",
		'is_checked_out': lambda x: "Yes" if x else "No",
		'document_date': lambda x: x.strftime('%Y-%m-%d') if x else "",
		'created_on': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else "",
		'changed_on': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else ""
	}
	
	# Order
	base_order = ('created_on', 'desc')
	
	def pre_add(self, obj):
		"""Pre-process before adding document."""
		# Set tenant_id from current user context
		obj.tenant_id = self.get_current_tenant_id()
		obj.owner_user_id = self.get_current_user_id()
		
	def pre_update(self, obj):
		"""Pre-process before updating document."""
		# Ensure tenant_id is not changed
		obj.tenant_id = self.get_current_tenant_id()
	
	def pre_delete(self, obj):
		"""Pre-process before deleting document."""
		# Check if document can be deleted
		if obj.is_checked_out:
			flash(_('Cannot delete checked out document'), 'error')
			return False
		if obj.legal_hold:
			flash(_('Cannot delete document under legal hold'), 'error')
			return False
		return True
	
	@action('checkout', _('Checkout'), _('Checkout selected documents'), 'fa-lock')
	def checkout_documents(self, items):
		"""Checkout selected documents."""
		service = DocumentService(self.datamodel.session)
		user_id = self.get_current_user_id()
		
		success_count = 0
		for document in items:
			try:
				service.checkout_document(document.id, user_id)
				success_count += 1
			except Exception as e:
				flash(f"Failed to checkout {document.title}: {str(e)}", 'error')
		
		if success_count > 0:
			flash(f"Successfully checked out {success_count} document(s)", 'success')
		
		return redirect(self.get_redirect())
	
	@action('checkin', _('Checkin'), _('Checkin selected documents'), 'fa-unlock')
	def checkin_documents(self, items):
		"""Checkin selected documents."""
		service = DocumentService(self.datamodel.session)
		user_id = self.get_current_user_id()
		
		success_count = 0
		for document in items:
			try:
				if document.is_checked_out and document.checked_out_by == user_id:
					service.checkin_document(document.id, user_id)
					success_count += 1
			except Exception as e:
				flash(f"Failed to checkin {document.title}: {str(e)}", 'error')
		
		if success_count > 0:
			flash(f"Successfully checked in {success_count} document(s)", 'success')
		
		return redirect(self.get_redirect())
	
	@action('download', _('Download'), _('Download selected documents'), 'fa-download')
	def download_documents(self, items):
		"""Download selected documents."""
		if len(items) == 1:
			return redirect(url_for('GCDMDocumentModelView.download_file', pk=items[0].id))
		else:
			flash(_('Please select only one document to download'), 'warning')
			return redirect(self.get_redirect())
	
	@expose('/download/<pk>')
	@has_access
	def download_file(self, pk):
		"""Download document file."""
		service = DocumentService(self.datamodel.session)
		user_id = self.get_current_user_id()
		
		try:
			file_path, file_name = service.download_document(pk, user_id)
			return send_file(file_path, as_attachment=True, download_name=file_name)
		except Exception as e:
			flash(f"Download failed: {str(e)}", 'error')
			return redirect(self.get_redirect())
	
	@expose('/versions/<pk>')
	@has_access
	def view_versions(self, pk):
		"""View document versions."""
		service = DocumentService(self.datamodel.session)
		user_id = self.get_current_user_id()
		
		try:
			document = service.get_document(pk, user_id)
			versions = service.get_document_versions(pk, user_id)
			
			return render_template_string("""
			<h3>Document Versions: {{ document.title }}</h3>
			<table class="table table-striped">
				<thead>
					<tr>
						<th>Version</th>
						<th>Changed By</th>
						<th>Change Date</th>
						<th>Description</th>
						<th>Status</th>
						<th>Actions</th>
					</tr>
				</thead>
				<tbody>
					{% for version in versions %}
					<tr>
						<td>{{ version.version_number }}</td>
						<td>{{ version.changed_by }}</td>
						<td>{{ version.created_on.strftime('%Y-%m-%d %H:%M') }}</td>
						<td>{{ version.change_description }}</td>
						<td>
							{% if version.is_current %}
								<span class="label label-success">Current</span>
							{% else %}
								<span class="label label-default">{{ version.status }}</span>
							{% endif %}
						</td>
						<td>
							{% if not version.is_current %}
								<a href="{{ url_for('GCDMDocumentModelView.revert_version', pk=document.id, version_id=version.id) }}" 
								   class="btn btn-sm btn-warning"
								   onclick="return confirm('Are you sure you want to revert to this version?')">
									Revert
								</a>
							{% endif %}
						</td>
					</tr>
					{% endfor %}
				</tbody>
			</table>
			""", document=document, versions=versions)
			
		except Exception as e:
			flash(f"Failed to load versions: {str(e)}", 'error')
			return redirect(self.get_redirect())
	
	@expose('/revert/<pk>/<version_id>')
	@has_access
	def revert_version(self, pk, version_id):
		"""Revert document to specific version."""
		service = DocumentService(self.datamodel.session)
		user_id = self.get_current_user_id()
		
		try:
			service.revert_to_version(pk, version_id, user_id)
			flash(_('Document reverted successfully'), 'success')
		except Exception as e:
			flash(f"Revert failed: {str(e)}", 'error')
		
		return redirect(self.get_redirect())
	
	def get_current_user_id(self) -> str:
		"""Get current user ID."""
		# This should be implemented based on your authentication system
		return "current_user_id"  # Placeholder
	
	def get_current_tenant_id(self) -> str:
		"""Get current tenant ID."""
		# This should be implemented based on your multi-tenancy system
		return "current_tenant_id"  # Placeholder

class GCDMFolderModelView(ModelView):
	"""Folder management view."""
	
	datamodel = SQLAInterface(GCDMFolder)
	
	list_columns = ['name', 'folder_path', 'parent_folder.name', 'document_count', 'is_active']
	show_columns = [
		'name', 'description', 'folder_path', 'full_path', 'parent_folder.name',
		'document_count', 'total_size_mb', 'is_active', 'created_on', 'changed_on'
	]
	edit_columns = ['name', 'description', 'parent_folder_id']
	add_columns = ['name', 'description', 'parent_folder_id']
	
	list_title = _('Folders')
	show_title = _('Folder Details')
	add_title = _('Create Folder')
	edit_title = _('Edit Folder')
	
	label_columns = {
		'name': _('Name'),
		'description': _('Description'),
		'folder_path': _('Path'),
		'full_path': _('Full Path'),
		'parent_folder.name': _('Parent Folder'),
		'document_count': _('Documents'),
		'total_size_mb': _('Total Size (MB)'),
		'is_active': _('Active')
	}
	
	formatters_columns = {
		'total_size_mb': lambda x: f"{x:.2f}" if x else "0.00",
		'is_active': lambda x: "Yes" if x else "No"
	}
	
	base_order = ('folder_path', 'asc')

class GCDMDocumentVersionModelView(ModelView):
	"""Document version management view."""
	
	datamodel = SQLAInterface(GCDMDocumentVersion)
	
	list_columns = [
		'document.title', 'version_number', 'changed_by', 'change_description',
		'is_current', 'status', 'created_on'
	]
	show_columns = [
		'document.title', 'version_number', 'version_label', 'changed_by',
		'change_description', 'change_type', 'is_current', 'status',
		'file_name', 'file_size_bytes', 'created_on'
	]
	
	list_title = _('Document Versions')
	show_title = _('Version Details')
	
	label_columns = {
		'document.title': _('Document'),
		'version_number': _('Version'),
		'version_label': _('Label'),
		'changed_by': _('Changed By'),
		'change_description': _('Description'),
		'change_type': _('Change Type'),
		'is_current': _('Current'),
		'status': _('Status'),
		'file_name': _('File Name'),
		'file_size_bytes': _('File Size')
	}
	
	formatters_columns = {
		'file_size_bytes': lambda x: f"{x / 1024 / 1024:.2f} MB" if x else "0 MB",
		'is_current': lambda x: "Yes" if x else "No",
		'status': lambda x: x.title() if x else ""
	}
	
	base_order = ('created_on', 'desc')
	
	# Disable add/edit - versions are created through document service
	base_permissions = ['can_list', 'can_show']

class GCDMPermissionModelView(ModelView):
	"""Document permission management view."""
	
	datamodel = SQLAInterface(GCDMPermission)
	
	list_columns = [
		'document.title', 'folder.name', 'subject_type', 'subject_id',
		'permission_level', 'is_active', 'granted_by'
	]
	show_columns = [
		'document.title', 'folder.name', 'subject_type', 'subject_id',
		'permission_level', 'can_read', 'can_write', 'can_delete',
		'can_share', 'can_approve', 'applies_to_children', 'is_active',
		'granted_by', 'effective_date', 'expiry_date', 'created_on'
	]
	edit_columns = [
		'subject_type', 'subject_id', 'permission_level', 'can_read',
		'can_write', 'can_delete', 'can_share', 'can_approve',
		'applies_to_children', 'effective_date', 'expiry_date'
	]
	add_columns = [
		'document_id', 'folder_id', 'subject_type', 'subject_id',
		'permission_level', 'can_read', 'can_write', 'can_delete',
		'can_share', 'can_approve', 'applies_to_children',
		'effective_date', 'expiry_date'
	]
	
	list_title = _('Permissions')
	show_title = _('Permission Details')
	add_title = _('Grant Permission')
	edit_title = _('Edit Permission')
	
	label_columns = {
		'document.title': _('Document'),
		'folder.name': _('Folder'),
		'subject_type': _('Subject Type'),
		'subject_id': _('Subject ID'),
		'permission_level': _('Permission Level'),
		'can_read': _('Can Read'),
		'can_write': _('Can Write'),
		'can_delete': _('Can Delete'),
		'can_share': _('Can Share'),
		'can_approve': _('Can Approve'),
		'applies_to_children': _('Apply to Children'),
		'granted_by': _('Granted By'),
		'effective_date': _('Effective Date'),
		'expiry_date': _('Expiry Date')
	}

class GCDMWorkflowModelView(ModelView):
	"""Workflow management view."""
	
	datamodel = SQLAInterface(GCDMWorkflow)
	
	list_columns = [
		'name', 'workflow_type', 'document.title', 'status',
		'current_assignee', 'initiated_date', 'due_date'
	]
	show_columns = [
		'name', 'description', 'workflow_type', 'document.title',
		'status', 'current_step', 'current_assignee', 'initiated_by',
		'initiated_date', 'due_date', 'completed_date', 'priority'
	]
	edit_columns = [
		'name', 'description', 'workflow_type', 'current_assignee',
		'due_date', 'priority'
	]
	
	list_title = _('Workflows')
	show_title = _('Workflow Details')
	add_title = _('Create Workflow')
	edit_title = _('Edit Workflow')
	
	label_columns = {
		'name': _('Name'),
		'description': _('Description'),
		'workflow_type': _('Type'),
		'document.title': _('Document'),
		'status': _('Status'),
		'current_step': _('Current Step'),
		'current_assignee': _('Assigned To'),
		'initiated_by': _('Initiated By'),
		'initiated_date': _('Started'),
		'due_date': _('Due Date'),
		'completed_date': _('Completed'),
		'priority': _('Priority')
	}
	
	formatters_columns = {
		'status': lambda x: x.title() if x else "",
		'priority': lambda x: x.title() if x else ""
	}

class GCDMReviewModelView(ModelView):
	"""Review management view."""
	
	datamodel = SQLAInterface(GCDMReview)
	
	list_columns = [
		'document.title', 'review_type', 'reviewer_user_id', 'status',
		'assigned_date', 'due_date', 'decision'
	]
	show_columns = [
		'document.title', 'review_type', 'reviewer_user_id', 'reviewer_name',
		'status', 'decision', 'assigned_date', 'due_date', 'started_date',
		'completed_date', 'comments', 'overall_score'
	]
	edit_columns = [
		'status', 'decision', 'comments', 'detailed_feedback', 'overall_score'
	]
	
	list_title = _('Reviews')
	show_title = _('Review Details')
	edit_title = _('Update Review')
	
	label_columns = {
		'document.title': _('Document'),
		'review_type': _('Review Type'),
		'reviewer_user_id': _('Reviewer ID'),
		'reviewer_name': _('Reviewer'),
		'status': _('Status'),
		'decision': _('Decision'),
		'assigned_date': _('Assigned'),
		'due_date': _('Due Date'),
		'started_date': _('Started'),
		'completed_date': _('Completed'),
		'comments': _('Comments'),
		'overall_score': _('Score')
	}

class GCDMRetentionPolicyModelView(ModelView):
	"""Retention policy management view."""
	
	datamodel = SQLAInterface(GCDMRetentionPolicy)
	
	list_columns = [
		'name', 'policy_code', 'retention_period_years', 'auto_delete_enabled',
		'is_active', 'effective_date'
	]
	show_columns = [
		'name', 'description', 'policy_code', 'retention_period_years',
		'retention_period_months', 'retention_period_days', 'auto_delete_enabled',
		'auto_archive_enabled', 'legal_hold_override', 'is_active',
		'effective_date', 'expiry_date', 'regulatory_basis'
	]
	edit_columns = [
		'name', 'description', 'retention_period_years', 'retention_period_months',
		'retention_period_days', 'auto_delete_enabled', 'auto_archive_enabled',
		'legal_hold_override', 'effective_date', 'expiry_date'
	]
	add_columns = [
		'name', 'description', 'policy_code', 'retention_period_years',
		'retention_period_months', 'retention_period_days', 'auto_delete_enabled',
		'auto_archive_enabled', 'legal_hold_override', 'effective_date'
	]
	
	list_title = _('Retention Policies')
	show_title = _('Retention Policy Details')
	add_title = _('Create Retention Policy')
	edit_title = _('Edit Retention Policy')

class GCDMSearchView(BaseView):
	"""Advanced document search view."""
	
	route_base = '/document_search'
	default_view = 'search'
	
	@expose('/')
	@has_access
	def search(self):
		"""Advanced search interface."""
		# Get search parameters from request
		query = request.args.get('query', '')
		category_id = request.args.get('category_id', '')
		document_type_id = request.args.get('document_type_id', '')
		folder_id = request.args.get('folder_id', '')
		status = request.args.get('status', '')
		tags = request.args.get('tags', '')
		date_from = request.args.get('date_from', '')
		date_to = request.args.get('date_to', '')
		
		# Search results
		results = []
		total_count = 0
		
		if query or any([category_id, document_type_id, folder_id, status, tags, date_from, date_to]):
			service = DocumentService(self.appbuilder.get_session)
			user_id = self.get_current_user_id()
			tenant_id = self.get_current_tenant_id()
			
			# Convert date strings
			date_from_obj = datetime.strptime(date_from, '%Y-%m-%d').date() if date_from else None
			date_to_obj = datetime.strptime(date_to, '%Y-%m-%d').date() if date_to else None
			tag_list = [tag.strip() for tag in tags.split(',')] if tags else None
			
			try:
				results, total_count = service.search_documents(
					tenant_id=tenant_id,
					user_id=user_id,
					query=query,
					category_id=category_id if category_id else None,
					document_type_id=document_type_id if document_type_id else None,
					folder_id=folder_id if folder_id else None,
					status=status if status else None,
					tags=tag_list,
					date_from=date_from_obj,
					date_to=date_to_obj
				)
			except Exception as e:
				flash(f"Search failed: {str(e)}", 'error')
		
		# Get filter options
		categories = self.appbuilder.get_session.query(GCDMDocumentCategory).filter_by(is_active=True).all()
		document_types = self.appbuilder.get_session.query(GCDMDocumentType).filter_by(is_active=True).all()
		folders = self.appbuilder.get_session.query(GCDMFolder).filter_by(is_active=True).all()
		
		return self.render_template(
			'search.html',
			query=query,
			results=results,
			total_count=total_count,
			categories=categories,
			document_types=document_types,
			folders=folders,
			selected_category=category_id,
			selected_document_type=document_type_id,
			selected_folder=folder_id,
			selected_status=status,
			tags=tags,
			date_from=date_from,
			date_to=date_to
		)
	
	def get_current_user_id(self) -> str:
		"""Get current user ID."""
		return "current_user_id"  # Placeholder
	
	def get_current_tenant_id(self) -> str:
		"""Get current tenant ID."""
		return "current_tenant_id"  # Placeholder

class GCDMDashboardView(BaseView):
	"""Document management dashboard."""
	
	route_base = '/document_dashboard'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Dashboard main view."""
		session = self.appbuilder.get_session
		tenant_id = self.get_current_tenant_id()
		
		# Get dashboard statistics
		stats = self._get_dashboard_stats(session, tenant_id)
		
		# Get recent documents
		recent_documents = session.query(GCDMDocument).filter(
			GCDMDocument.tenant_id == tenant_id,
			GCDMDocument.is_active == True,
			GCDMDocument.is_deleted == False
		).order_by(desc(GCDMDocument.created_on)).limit(10).all()
		
		# Get documents requiring attention
		attention_documents = session.query(GCDMDocument).filter(
			GCDMDocument.tenant_id == tenant_id,
			GCDMDocument.is_active == True,
			GCDMDocument.is_deleted == False,
			or_(
				GCDMDocument.is_checked_out == True,
				GCDMDocument.status == 'review',
				GCDMDocument.retention_date < date.today()
			)
		).limit(10).all()
		
		return self.render_template(
			'dashboard.html',
			stats=stats,
			recent_documents=recent_documents,
			attention_documents=attention_documents
		)
	
	def _get_dashboard_stats(self, session, tenant_id) -> Dict[str, Any]:
		"""Get dashboard statistics."""
		# Total documents
		total_docs = session.query(func.count(GCDMDocument.id)).filter(
			GCDMDocument.tenant_id == tenant_id,
			GCDMDocument.is_active == True,
			GCDMDocument.is_deleted == False
		).scalar()
		
		# Documents by status
		status_stats = session.query(
			GCDMDocument.status,
			func.count(GCDMDocument.id)
		).filter(
			GCDMDocument.tenant_id == tenant_id,
			GCDMDocument.is_active == True,
			GCDMDocument.is_deleted == False
		).group_by(GCDMDocument.status).all()
		
		# Checked out documents
		checked_out = session.query(func.count(GCDMDocument.id)).filter(
			GCDMDocument.tenant_id == tenant_id,
			GCDMDocument.is_checked_out == True
		).scalar()
		
		# Total storage used
		total_storage = session.query(func.sum(GCDMDocument.file_size_bytes)).filter(
			GCDMDocument.tenant_id == tenant_id,
			GCDMDocument.is_active == True
		).scalar() or 0
		
		return {
			'total_documents': total_docs,
			'status_distribution': dict(status_stats),
			'checked_out_count': checked_out,
			'total_storage_mb': total_storage / (1024 * 1024),
			'average_file_size_mb': (total_storage / total_docs / (1024 * 1024)) if total_docs > 0 else 0
		}
	
	def get_current_tenant_id(self) -> str:
		"""Get current tenant ID."""
		return "current_tenant_id"  # Placeholder