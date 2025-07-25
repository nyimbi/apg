"""
Regulatory Compliance Views

Flask-AppBuilder views for pharmaceutical regulatory compliance management
including submissions, audits, deviations, and compliance monitoring.
"""

from datetime import datetime, date
from typing import Dict, List, Any
from flask import request, flash, redirect, url_for, jsonify
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from wtforms import SelectField, TextAreaField, DateField
from wtforms.validators import DataRequired

from ....auth_rbac.views import BaseSecureView
from .models import (
	PHRCRegulatoryFramework, PHRCSubmission, PHRCSubmissionDocument,
	PHRCAudit, PHRCAuditFinding, PHRCDeviation, PHRCCorrectiveAction,
	PHRCComplianceControl, PHRCRegulatoryContact, PHRCInspection,
	PHRCRegulatoryReport
)
from .service import RegulatoryComplianceService


class PHRCRegulatoryFrameworkModelView(BaseSecureView, ModelView):
	"""View for managing regulatory frameworks"""
	
	datamodel = SQLAInterface(PHRCRegulatoryFramework)
	
	# List view configuration
	list_columns = [
		'framework_code', 'framework_name', 'region', 
		'is_active', 'version', 'effective_date'
	]
	search_columns = ['framework_code', 'framework_name', 'region']
	order_columns = ['framework_name', 'region', 'effective_date']
	
	# Form configuration
	show_columns = [
		'framework_code', 'framework_name', 'region', 'description',
		'website_url', 'contact_info', 'key_regulations', 'submission_types',
		'is_active', 'version', 'effective_date'
	]
	edit_columns = [
		'framework_code', 'framework_name', 'region', 'description',
		'website_url', 'is_active', 'version', 'effective_date'
	]
	add_columns = edit_columns
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit']
	
	def pre_add(self, item):
		"""Set tenant_id before adding"""
		item.tenant_id = self.get_current_tenant_id()
	
	def pre_update(self, item):
		"""Validate before updating"""
		if not item.tenant_id:
			item.tenant_id = self.get_current_tenant_id()


class PHRCSubmissionModelView(BaseSecureView, ModelView):
	"""View for managing regulatory submissions"""
	
	datamodel = SQLAInterface(PHRCSubmission)
	
	# List view configuration
	list_columns = [
		'submission_number', 'submission_type', 'submission_title',
		'product_name', 'status', 'submission_date', 'target_response_date'
	]
	search_columns = [
		'submission_number', 'submission_title', 'product_name', 'indication'
	]
	order_columns = ['submission_date', 'target_response_date', 'status']
	
	# Form configuration
	show_columns = [
		'submission_number', 'submission_type', 'submission_title', 'description',
		'product_name', 'active_ingredient', 'therapeutic_area', 'indication',
		'framework', 'status', 'submission_date', 'target_response_date',
		'actual_response_date', 'review_division', 'reviewer_name',
		'priority_designation', 'fees_paid', 'currency'
	]
	edit_columns = [
		'submission_type', 'submission_title', 'description',
		'product_name', 'active_ingredient', 'therapeutic_area', 'indication',
		'framework_id', 'status', 'review_division', 'reviewer_name',
		'priority_designation', 'fees_paid'
	]
	add_columns = [
		'submission_type', 'submission_title', 'description',
		'product_name', 'active_ingredient', 'therapeutic_area', 'indication',
		'framework_id', 'priority_designation'
	]
	
	# Relationships
	related_views = [PHRCRegulatoryFrameworkModelView]
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_submit']
	
	@expose('/submit/<string:pk>')
	@has_access
	def submit_to_authority(self, pk):
		"""Submit application to regulatory authority"""
		service = RegulatoryComplianceService(self.get_current_tenant_id())
		
		try:
			success = service.submit_to_authority(pk)
			if success:
				flash('Submission filed successfully', 'success')
			else:
				flash('Submission not found', 'error')
		except ValueError as e:
			flash(f'Submission validation failed: {str(e)}', 'error')
		except Exception as e:
			flash(f'Error submitting: {str(e)}', 'error')
		
		return redirect(url_for('PHRCSubmissionModelView.show', pk=pk))
	
	def pre_add(self, item):
		"""Set tenant_id and generate submission number"""
		item.tenant_id = self.get_current_tenant_id()
		if not item.submission_number:
			service = RegulatoryComplianceService(self.get_current_tenant_id())
			item.submission_number = service._generate_submission_number(item.submission_type)


class PHRCAuditModelView(BaseSecureView, ModelView):
	"""View for managing regulatory audits"""
	
	datamodel = SQLAInterface(PHRCAudit)
	
	# List view configuration
	list_columns = [
		'audit_number', 'audit_title', 'audit_type', 'status',
		'planned_start_date', 'planned_end_date', 'overall_rating'
	]
	search_columns = ['audit_number', 'audit_title', 'audit_scope']
	order_columns = ['planned_start_date', 'status', 'overall_rating']
	
	# Form configuration
	show_columns = [
		'audit_number', 'audit_title', 'audit_type', 'audit_scope',
		'framework', 'planned_start_date', 'planned_end_date',
		'actual_start_date', 'actual_end_date', 'lead_auditor',
		'audit_team', 'auditee_contact', 'status', 'overall_rating',
		'executive_summary'
	]
	edit_columns = [
		'audit_title', 'audit_type', 'audit_scope', 'framework_id',
		'planned_start_date', 'planned_end_date', 'lead_auditor',
		'auditee_contact', 'status', 'overall_rating', 'executive_summary'
	]
	add_columns = [
		'audit_title', 'audit_type', 'audit_scope', 'framework_id',
		'planned_start_date', 'planned_end_date', 'lead_auditor', 'auditee_contact'
	]
	
	# Relationships
	related_views = [PHRCRegulatoryFrameworkModelView]
	
	@expose('/summary/<string:pk>')
	@has_access
	def audit_summary(self, pk):
		"""Get audit summary with findings"""
		service = RegulatoryComplianceService(self.get_current_tenant_id())
		summary = service.get_audit_summary(pk)
		
		return jsonify(summary)
	
	def pre_add(self, item):
		"""Set tenant_id and generate audit number"""
		item.tenant_id = self.get_current_tenant_id()
		if not item.audit_number:
			service = RegulatoryComplianceService(self.get_current_tenant_id())
			item.audit_number = service._generate_audit_number(item.audit_type)


class PHRCAuditFindingModelView(BaseSecureView, ModelView):
	"""View for managing audit findings"""
	
	datamodel = SQLAInterface(PHRCAuditFinding)
	
	# List view configuration
	list_columns = [
		'finding_number', 'finding_title', 'severity', 'category',
		'status', 'response_deadline', 'assigned_to'
	]
	search_columns = ['finding_number', 'finding_title', 'description']
	order_columns = ['severity', 'response_deadline', 'status']
	
	# Form configuration
	show_columns = [
		'finding_number', 'finding_title', 'description', 'severity',
		'category', 'regulation_reference', 'response_required',
		'response_deadline', 'assigned_to', 'status', 'closure_date',
		'evidence_documents'
	]
	edit_columns = [
		'finding_title', 'description', 'severity', 'category',
		'regulation_reference', 'response_required', 'response_deadline',
		'assigned_to', 'status'
	]
	add_columns = [
		'audit_id', 'finding_title', 'description', 'severity',
		'category', 'regulation_reference', 'response_deadline', 'assigned_to'
	]
	
	def pre_add(self, item):
		"""Set tenant_id and generate finding number"""
		item.tenant_id = self.get_current_tenant_id()
		if not item.finding_number:
			service = RegulatoryComplianceService(self.get_current_tenant_id())
			item.finding_number = service._generate_finding_number(item.audit_id)


class PHRCDeviationModelView(BaseSecureView, ModelView):
	"""View for managing quality deviations"""
	
	datamodel = SQLAInterface(PHRCDeviation)
	
	# List view configuration
	list_columns = [
		'deviation_number', 'deviation_title', 'severity', 'deviation_type',
		'discovered_date', 'status', 'assigned_investigator'
	]
	search_columns = ['deviation_number', 'deviation_title', 'description']
	order_columns = ['discovered_date', 'severity', 'status']
	
	# Form configuration
	show_columns = [
		'deviation_number', 'deviation_title', 'description', 'deviation_type',
		'severity', 'impact_assessment', 'process_area', 'product_affected',
		'batch_lot_affected', 'discovered_date', 'discovered_by',
		'discovery_method', 'investigation_required', 'investigation_deadline',
		'assigned_investigator', 'root_cause', 'status', 'closure_date'
	]
	edit_columns = [
		'deviation_title', 'description', 'deviation_type', 'severity',
		'impact_assessment', 'process_area', 'product_affected',
		'batch_lot_affected', 'discovery_method', 'investigation_deadline',
		'assigned_investigator', 'root_cause', 'status'
	]
	add_columns = [
		'deviation_title', 'description', 'deviation_type', 'severity',
		'process_area', 'product_affected', 'batch_lot_affected',
		'discovered_by', 'discovery_method'
	]
	
	# Custom actions
	@expose('/assign_investigation/<string:pk>')
	@has_access
	def assign_investigation(self, pk):
		"""Assign deviation investigation"""
		if request.method == 'POST':
			investigator_id = request.form.get('investigator_id')
			deadline = request.form.get('deadline')
			
			service = RegulatoryComplianceService(self.get_current_tenant_id())
			
			deadline_date = None
			if deadline:
				deadline_date = datetime.strptime(deadline, '%Y-%m-%d').date()
			
			success = service.assign_investigation(pk, investigator_id, deadline_date)
			
			if success:
				flash('Investigation assigned successfully', 'success')
			else:
				flash('Error assigning investigation', 'error')
			
			return redirect(url_for('PHRCDeviationModelView.show', pk=pk))
		
		# Return form for GET request
		return self.render_template('assign_investigation.html', pk=pk)
	
	@expose('/complete_investigation/<string:pk>')
	@has_access
	def complete_investigation(self, pk):
		"""Complete deviation investigation"""
		if request.method == 'POST':
			root_cause = request.form.get('root_cause')
			impact_assessment = request.form.get('impact_assessment')
			
			service = RegulatoryComplianceService(self.get_current_tenant_id())
			success = service.complete_investigation(pk, root_cause, impact_assessment)
			
			if success:
				flash('Investigation completed successfully', 'success')
			else:
				flash('Error completing investigation', 'error')
			
			return redirect(url_for('PHRCDeviationModelView.show', pk=pk))
		
		# Return form for GET request
		return self.render_template('complete_investigation.html', pk=pk)
	
	def pre_add(self, item):
		"""Set tenant_id and generate deviation number"""
		item.tenant_id = self.get_current_tenant_id()
		if not item.deviation_number:
			service = RegulatoryComplianceService(self.get_current_tenant_id())
			item.deviation_number = service._generate_deviation_number()


class PHRCCorrectiveActionModelView(BaseSecureView, ModelView):
	"""View for managing corrective and preventive actions (CAPA)"""
	
	datamodel = SQLAInterface(PHRCCorrectiveAction)
	
	# List view configuration
	list_columns = [
		'action_number', 'action_title', 'action_type', 'category',
		'planned_completion_date', 'status', 'assigned_to'
	]
	search_columns = ['action_number', 'action_title', 'description']
	order_columns = ['planned_completion_date', 'status', 'action_type']
	
	# Form configuration
	show_columns = [
		'action_number', 'action_title', 'description', 'source_type',
		'deviation', 'finding', 'action_type', 'category',
		'planned_start_date', 'planned_completion_date', 'assigned_to',
		'actual_start_date', 'actual_completion_date', 'implementation_notes',
		'status', 'effectiveness_check_required', 'effectiveness_check_date',
		'effectiveness_verified', 'effectiveness_notes'
	]
	edit_columns = [
		'action_title', 'description', 'action_type', 'category',
		'planned_start_date', 'planned_completion_date', 'assigned_to',
		'implementation_notes', 'status', 'effectiveness_check_required',
		'effectiveness_verified', 'effectiveness_notes'
	]
	add_columns = [
		'action_title', 'description', 'source_type', 'deviation_id',
		'finding_id', 'action_type', 'category', 'planned_completion_date',
		'assigned_to'
	]
	
	@expose('/complete/<string:pk>')
	@has_access
	def complete_action(self, pk):
		"""Complete corrective action"""
		if request.method == 'POST':
			completion_notes = request.form.get('completion_notes')
			
			service = RegulatoryComplianceService(self.get_current_tenant_id())
			success = service.complete_action(pk, completion_notes)
			
			if success:
				flash('Action completed successfully', 'success')
			else:
				flash('Error completing action', 'error')
			
			return redirect(url_for('PHRCCorrectiveActionModelView.show', pk=pk))
		
		return self.render_template('complete_action.html', pk=pk)
	
	@expose('/verify_effectiveness/<string:pk>')
	@has_access
	def verify_effectiveness(self, pk):
		"""Verify CAPA effectiveness"""
		if request.method == 'POST':
			is_effective = request.form.get('is_effective') == 'true'
			verification_notes = request.form.get('verification_notes')
			
			service = RegulatoryComplianceService(self.get_current_tenant_id())
			success = service.verify_effectiveness(pk, is_effective, verification_notes)
			
			if success:
				flash('Effectiveness verification completed', 'success')
			else:
				flash('Error verifying effectiveness', 'error')
			
			return redirect(url_for('PHRCCorrectiveActionModelView.show', pk=pk))
		
		return self.render_template('verify_effectiveness.html', pk=pk)
	
	def pre_add(self, item):
		"""Set tenant_id and generate action number"""
		item.tenant_id = self.get_current_tenant_id()
		if not item.action_number:
			service = RegulatoryComplianceService(self.get_current_tenant_id())
			item.action_number = service._generate_action_number()


class PHRCInspectionModelView(BaseSecureView, ModelView):
	"""View for managing regulatory inspections"""
	
	datamodel = SQLAInterface(PHRCInspection)
	
	# List view configuration
	list_columns = [
		'inspection_number', 'inspection_type', 'regulatory_authority',
		'planned_start_date', 'status', 'outcome'
	]
	search_columns = ['inspection_number', 'regulatory_authority', 'lead_inspector']
	order_columns = ['planned_start_date', 'status', 'outcome']
	
	# Form configuration
	show_columns = [
		'inspection_number', 'inspection_type', 'inspection_scope',
		'regulatory_authority', 'lead_inspector', 'inspection_team',
		'notification_date', 'planned_start_date', 'planned_end_date',
		'actual_start_date', 'actual_end_date', 'preparation_status',
		'preparation_checklist', 'responsible_team', 'status', 'outcome',
		'inspection_report_received', 'response_required', 'response_deadline'
	]
	edit_columns = [
		'inspection_type', 'inspection_scope', 'regulatory_authority',
		'lead_inspector', 'notification_date', 'planned_start_date',
		'planned_end_date', 'preparation_status', 'responsible_team',
		'status', 'outcome', 'inspection_report_received',
		'response_required', 'response_deadline'
	]
	add_columns = [
		'inspection_type', 'inspection_scope', 'regulatory_authority',
		'planned_start_date', 'planned_end_date'
	]
	
	def pre_add(self, item):
		"""Set tenant_id and generate inspection number"""
		item.tenant_id = self.get_current_tenant_id()
		if not item.inspection_number:
			# Generate inspection number
			from datetime import datetime
			year = datetime.now().year
			count = PHRCInspection.query.filter_by(tenant_id=self.get_current_tenant_id()).count()
			item.inspection_number = f"INSP-{year}-{count + 1:04d}"


class PHRCComplianceDashboardView(BaseSecureView, BaseView):
	"""Compliance dashboard with metrics and alerts"""
	
	route_base = '/pharmaceutical/regulatory/dashboard'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Compliance dashboard main page"""
		service = RegulatoryComplianceService(self.get_current_tenant_id())
		dashboard_data = service.get_compliance_dashboard()
		
		return self.render_template(
			'regulatory_compliance_dashboard.html',
			dashboard_data=dashboard_data
		)
	
	@expose('/api/metrics')
	@has_access
	def api_metrics(self):
		"""API endpoint for dashboard metrics"""
		service = RegulatoryComplianceService(self.get_current_tenant_id())
		dashboard_data = service.get_compliance_dashboard()
		
		return jsonify(dashboard_data)


class PHRCSubmissionStatusChartView(DirectByChartView):
	"""Chart showing submission status distribution"""
	
	datamodel = SQLAInterface(PHRCSubmission)
	chart_title = 'Submission Status Distribution'
	label_columns = {'status': 'Status'}
	group_by_columns = ['status']


class PHRCDeviationSeverityChartView(DirectByChartView):
	"""Chart showing deviation severity distribution"""
	
	datamodel = SQLAInterface(PHRCDeviation)
	chart_title = 'Deviation Severity Distribution'
	label_columns = {'severity': 'Severity'}
	group_by_columns = ['severity']


# Additional utility views for specific regulatory functions

class PHRCRegulatoryContactModelView(BaseSecureView, ModelView):
	"""View for managing regulatory contacts"""
	
	datamodel = SQLAInterface(PHRCRegulatoryContact)
	
	list_columns = [
		'contact_name', 'organization', 'department',
		'relationship_type', 'is_active'
	]
	search_columns = ['contact_name', 'organization', 'department']
	
	show_columns = [
		'contact_name', 'title', 'organization', 'department',
		'email', 'phone', 'address', 'expertise_areas',
		'product_types', 'relationship_type', 'preferred_communication',
		'notes', 'is_active'
	]
	edit_columns = [
		'contact_name', 'title', 'organization', 'department',
		'email', 'phone', 'expertise_areas', 'product_types',
		'relationship_type', 'preferred_communication', 'notes', 'is_active'
	]
	add_columns = edit_columns
	
	def pre_add(self, item):
		item.tenant_id = self.get_current_tenant_id()


class PHRCRegulatoryReportModelView(BaseSecureView, ModelView):
	"""View for managing regulatory reports"""
	
	datamodel = SQLAInterface(PHRCRegulatoryReport)
	
	list_columns = [
		'report_number', 'report_type', 'report_title',
		'reporting_period_end', 'due_date', 'status'
	]
	search_columns = ['report_number', 'report_title', 'report_type']
	order_columns = ['due_date', 'reporting_period_end', 'status']
	
	show_columns = [
		'report_number', 'report_type', 'report_title',
		'reporting_period_start', 'reporting_period_end',
		'due_date', 'submission_date', 'submitted_by',
		'report_content', 'attachments', 'status',
		'approved_by', 'approved_date'
	]
	edit_columns = [
		'report_type', 'report_title', 'reporting_period_start',
		'reporting_period_end', 'due_date', 'report_content', 'status'
	]
	add_columns = [
		'report_type', 'report_title', 'reporting_period_start',
		'reporting_period_end', 'due_date'
	]
	
	def pre_add(self, item):
		item.tenant_id = self.get_current_tenant_id()
		if not item.report_number:
			# Generate report number
			from datetime import datetime
			year = datetime.now().year
			count = PHRCRegulatoryReport.query.filter_by(tenant_id=self.get_current_tenant_id()).count()
			item.report_number = f"REP-{item.report_type[:3].upper()}-{year}-{count + 1:04d}"