"""
Audit & Compliance Views

Flask-AppBuilder views for comprehensive audit logging, compliance monitoring,
and regulatory reporting with tamper-proof storage and integrity verification.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .models import (
	ACAuditLog, ACComplianceRule, ACComplianceViolation, ACDataRetentionPolicy,
	ACComplianceReport, ACSystemConfiguration
)


class AuditComplianceBaseView(BaseView):
	"""Base view for audit and compliance functionality"""
	
	def __init__(self):
		super().__init__()
		self.default_view = 'dashboard'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID from security context"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def _format_compliance_score(self, score: float) -> str:
		"""Format compliance score for display"""
		return f"{score:.1f}%" if score is not None else "0.0%"


class ACAuditLogModelView(ModelView):
	"""Audit log management view"""
	
	datamodel = SQLAInterface(ACAuditLog)
	
	# List view configuration
	list_columns = [
		'event_type', 'event_category', 'action', 'user_id',
		'resource_type', 'severity', 'ip_address', 'created_on'
	]
	show_columns = [
		'log_id', 'event_type', 'event_category', 'event_source', 'severity',
		'user_id', 'session_id', 'action', 'resource_type', 'resource_id',
		'resource_name', 'ip_address', 'user_agent', 'old_values', 'new_values',
		'pii_accessed', 'sensitive_data', 'compliance_relevant', 'event_hash',
		'created_on'
	]
	
	# Make this view read-only for audit integrity
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	can_delete = False
	
	# Search and filtering
	search_columns = ['event_type', 'action', 'user_id', 'resource_type']
	base_filters = [['compliance_relevant', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('created_on', 'desc')
	
	# Custom labels
	label_columns = {
		'log_id': 'Log ID',
		'event_type': 'Event Type',
		'event_category': 'Event Category',
		'event_source': 'Event Source',
		'user_id': 'User ID',
		'session_id': 'Session ID',
		'resource_type': 'Resource Type',
		'resource_id': 'Resource ID',
		'resource_name': 'Resource Name',
		'ip_address': 'IP Address',
		'user_agent': 'User Agent',
		'old_values': 'Old Values',
		'new_values': 'New Values',
		'pii_accessed': 'PII Accessed',
		'sensitive_data': 'Sensitive Data',
		'compliance_relevant': 'Compliance Relevant',
		'event_hash': 'Event Hash'
	}
	
	@expose('/verify_integrity/<int:pk>')
	@has_access
	def verify_integrity(self, pk):
		"""Verify audit log entry integrity"""
		audit_log = self.datamodel.get(pk)
		if not audit_log:
			flash('Audit log entry not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			is_valid = audit_log.verify_integrity()
			if is_valid:
				flash(f'Audit log entry integrity verified successfully', 'success')
			else:
				flash(f'INTEGRITY VIOLATION: Audit log entry has been tampered with!', 'error')
		except Exception as e:
			flash(f'Error verifying integrity: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/export/')
	@has_access
	def export_audit_logs(self):
		"""Export audit logs for compliance reporting"""
		try:
			start_date = request.args.get('start_date')
			end_date = request.args.get('end_date')
			event_types = request.args.getlist('event_types')
			
			# Implementation would export filtered logs
			export_data = self._export_logs(start_date, end_date, event_types)
			
			flash(f'Exported {len(export_data)} audit log entries', 'success')
			return redirect(self.get_redirect())
		except Exception as e:
			flash(f'Error exporting audit logs: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def _export_logs(self, start_date: str, end_date: str, event_types: List[str]) -> List[Dict]:
		"""Export audit logs based on filters"""
		# Implementation would query and format logs for export
		return []


class ACComplianceRuleModelView(ModelView):
	"""Compliance rule management view"""
	
	datamodel = SQLAInterface(ACComplianceRule)
	
	# List view configuration
	list_columns = [
		'name', 'rule_type', 'compliance_framework', 'severity',
		'is_active', 'triggered_count', 'effectiveness_score', 'last_triggered'
	]
	show_columns = [
		'rule_id', 'name', 'description', 'rule_type', 'compliance_framework',
		'conditions', 'actions', 'severity', 'is_active', 'auto_remediate',
		'applicable_events', 'triggered_count', 'effectiveness_score',
		'false_positive_count', 'last_triggered'
	]
	edit_columns = [
		'name', 'description', 'rule_type', 'compliance_framework',
		'conditions', 'actions', 'severity', 'is_active', 'auto_remediate',
		'notification_enabled', 'applicable_events', 'excluded_users'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['name', 'description', 'rule_type']
	base_filters = [['is_active', lambda: True, lambda: True]]
	
	# Ordering
	base_order = ('effectiveness_score', 'desc')
	
	# Form validation
	validators_columns = {
		'name': [DataRequired(), Length(min=1, max=200)],
		'rule_type': [DataRequired()],
		'compliance_framework': [DataRequired()],
		'conditions': [DataRequired()],
		'severity': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'rule_id': 'Rule ID',
		'rule_type': 'Rule Type',
		'compliance_framework': 'Compliance Framework',
		'is_active': 'Active',
		'auto_remediate': 'Auto Remediate',
		'notification_enabled': 'Notification Enabled',
		'applicable_events': 'Applicable Events',
		'excluded_users': 'Excluded Users',
		'triggered_count': 'Triggered Count',
		'effectiveness_score': 'Effectiveness Score',
		'false_positive_count': 'False Positive Count',
		'last_triggered': 'Last Triggered'
	}
	
	@expose('/test_rule/<int:pk>')
	@has_access
	def test_rule(self, pk):
		"""Test compliance rule against sample data"""
		rule = self.datamodel.get(pk)
		if not rule:
			flash('Compliance rule not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would test rule with sample audit events
			test_results = self._test_rule_conditions(rule)
			
			return render_template('audit_compliance/rule_test_results.html',
								   rule=rule,
								   test_results=test_results,
								   page_title=f"Test Results: {rule.name}")
		except Exception as e:
			flash(f'Error testing rule: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/calculate_effectiveness/<int:pk>')
	@has_access
	def calculate_effectiveness(self, pk):
		"""Calculate rule effectiveness score"""
		rule = self.datamodel.get(pk)
		if not rule:
			flash('Compliance rule not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			rule.calculate_effectiveness()
			self.datamodel.edit(rule)
			flash(f'Effectiveness calculated for rule "{rule.name}": {rule.effectiveness_score:.1f}%', 'success')
		except Exception as e:
			flash(f'Error calculating effectiveness: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new compliance rule"""
		item.tenant_id = self._get_tenant_id()
	
	def _test_rule_conditions(self, rule: ACComplianceRule) -> Dict[str, Any]:
		"""Test rule conditions with sample data"""
		# Implementation would test rule logic
		return {
			'test_cases': [],
			'passed_tests': 0,
			'failed_tests': 0,
			'warnings': []
		}
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class ACComplianceViolationModelView(ModelView):
	"""Compliance violation management view"""
	
	datamodel = SQLAInterface(ACComplianceViolation)
	
	# List view configuration
	list_columns = [
		'rule', 'severity', 'description', 'status',
		'assigned_to', 'risk_score', 'created_on', 'resolved_at'
	]
	show_columns = [
		'violation_id', 'audit_log', 'rule', 'severity', 'description',
		'violation_type', 'status', 'assigned_to', 'resolution_notes',
		'risk_score', 'business_impact', 'regulatory_impact',
		'remediation_actions', 'remediation_deadline', 'created_on', 'resolved_at'
	]
	edit_columns = [
		'severity', 'description', 'status', 'assigned_to', 'resolution_notes',
		'business_impact', 'regulatory_impact', 'remediation_deadline'
	]
	add_columns = [
		'audit_log', 'rule', 'severity', 'description', 'violation_type',
		'assigned_to', 'business_impact', 'regulatory_impact'
	]
	
	# Search and filtering
	search_columns = ['description', 'violation_type']
	base_filters = [['status', lambda: 'open', lambda: True]]
	
	# Ordering
	base_order = ('risk_score', 'desc')
	
	# Form validation
	validators_columns = {
		'description': [DataRequired()],
		'severity': [DataRequired()],
		'risk_score': [NumberRange(min=0, max=100)]
	}
	
	# Custom labels
	label_columns = {
		'violation_id': 'Violation ID',
		'audit_log': 'Audit Log',
		'violation_type': 'Violation Type',
		'assigned_to': 'Assigned To',
		'resolution_notes': 'Resolution Notes',
		'risk_score': 'Risk Score',
		'business_impact': 'Business Impact',
		'regulatory_impact': 'Regulatory Impact',
		'remediation_actions': 'Remediation Actions',
		'remediation_deadline': 'Remediation Deadline',
		'resolved_at': 'Resolved At'
	}
	
	@expose('/resolve/<int:pk>')
	@has_access
	def resolve_violation(self, pk):
		"""Mark violation as resolved"""
		violation = self.datamodel.get(pk)
		if not violation:
			flash('Violation not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			violation.status = 'resolved'
			violation.resolved_at = datetime.utcnow()
			violation.resolved_by = self._get_current_user_id()
			
			self.datamodel.edit(violation)
			flash(f'Violation resolved successfully', 'success')
		except Exception as e:
			flash(f'Error resolving violation: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/escalate/<int:pk>')
	@has_access
	def escalate_violation(self, pk):
		"""Escalate violation to higher level"""
		violation = self.datamodel.get(pk)
		if not violation:
			flash('Violation not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			reason = request.args.get('reason', 'Manual escalation')
			violation.escalate(reason)
			
			self.datamodel.edit(violation)
			flash(f'Violation escalated to level {violation.escalation_level}', 'success')
		except Exception as e:
			flash(f'Error escalating violation: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/calculate_risk/<int:pk>')
	@has_access
	def calculate_risk(self, pk):
		"""Calculate violation risk score"""
		violation = self.datamodel.get(pk)
		if not violation:
			flash('Violation not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			violation.calculate_risk_score()
			self.datamodel.edit(violation)
			flash(f'Risk score calculated: {violation.risk_score:.1f}', 'success')
		except Exception as e:
			flash(f'Error calculating risk score: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new violation"""
		item.tenant_id = self._get_tenant_id()
		item.calculate_risk_score()
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class ACComplianceReportModelView(ModelView):
	"""Compliance report management view"""
	
	datamodel = SQLAInterface(ACComplianceReport)
	
	# List view configuration
	list_columns = [
		'name', 'report_type', 'compliance_framework', 'status',
		'overall_compliance_score', 'generated_at', 'generated_by'
	]
	show_columns = [
		'report_id', 'name', 'report_type', 'compliance_framework',
		'reporting_period_start', 'reporting_period_end', 'status',
		'overall_compliance_score', 'findings', 'recommendations',
		'generated_at', 'generated_by', 'file_path'
	]
	edit_columns = [
		'name', 'report_type', 'compliance_framework', 'reporting_period_start',
		'reporting_period_end', 'scope_description', 'included_systems'
	]
	add_columns = edit_columns
	
	# Search and filtering
	search_columns = ['name', 'report_type', 'compliance_framework']
	base_filters = [['status', lambda: 'completed', lambda: True]]
	
	# Ordering
	base_order = ('generated_at', 'desc')
	
	# Form validation
	validators_columns = {
		'name': [DataRequired(), Length(min=1, max=200)],
		'report_type': [DataRequired()],
		'compliance_framework': [DataRequired()],
		'reporting_period_start': [DataRequired()],
		'reporting_period_end': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'report_id': 'Report ID',
		'report_type': 'Report Type',
		'compliance_framework': 'Compliance Framework',
		'reporting_period_start': 'Period Start',
		'reporting_period_end': 'Period End',
		'scope_description': 'Scope Description',
		'included_systems': 'Included Systems',
		'overall_compliance_score': 'Compliance Score',
		'generated_at': 'Generated At',
		'generated_by': 'Generated By',
		'file_path': 'File Path'
	}
	
	@expose('/generate/<int:pk>')
	@has_access
	def generate_report(self, pk):
		"""Generate compliance report"""
		report = self.datamodel.get(pk)
		if not report:
			flash('Report not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Implementation would generate the actual report
			report.status = 'generating'
			report.generated_by = self._get_current_user_id()
			
			# Simulate report generation
			self._generate_compliance_report(report)
			
			report.status = 'completed'
			report.generated_at = datetime.utcnow()
			
			self.datamodel.edit(report)
			flash(f'Report "{report.name}" generated successfully', 'success')
		except Exception as e:
			report.status = 'failed'
			self.datamodel.edit(report)
			flash(f'Error generating report: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/view_report/<int:pk>')
	@has_access
	def view_report(self, pk):
		"""View generated compliance report"""
		report = self.datamodel.get(pk)
		if not report:
			flash('Report not found', 'error')
			return redirect(self.get_redirect())
		
		if report.status != 'completed':
			flash('Report is not ready for viewing', 'warning')
			return redirect(self.get_redirect())
		
		try:
			return render_template('audit_compliance/compliance_report.html',
								   report=report,
								   page_title=f"Report: {report.name}")
		except Exception as e:
			flash(f'Error viewing report: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new report"""
		item.tenant_id = self._get_tenant_id()
		item.status = 'draft'
	
	def _generate_compliance_report(self, report: ACComplianceReport):
		"""Generate compliance report content"""
		# Implementation would perform actual report generation
		report.calculate_compliance_score()
		
		# Add sample findings
		report.add_finding(
			'access_control',
			'medium',
			'Some users have excessive privileges',
			['user_management_system'],
			'Review and reduce user privileges'
		)
		
		# Add sample recommendations
		report.add_recommendation(
			'security',
			'high',
			'Implement multi-factor authentication',
			'2 weeks',
			'Immediate'
		)
		
		report.finalize_report()
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class AuditComplianceDashboardView(AuditComplianceBaseView):
	"""Audit and compliance dashboard"""
	
	route_base = "/audit_compliance_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Audit compliance dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('audit_compliance/dashboard.html',
								   metrics=metrics,
								   page_title="Audit & Compliance Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('audit_compliance/dashboard.html',
								   metrics={},
								   page_title="Audit & Compliance Dashboard")
	
	@expose('/compliance_status/')
	@has_access
	def compliance_status(self):
		"""Overall compliance status overview"""
		try:
			status_data = self._get_compliance_status()
			
			return render_template('audit_compliance/compliance_status.html',
								   status_data=status_data,
								   page_title="Compliance Status")
		except Exception as e:
			flash(f'Error loading compliance status: {str(e)}', 'error')
			return redirect(url_for('AuditComplianceDashboardView.index'))
	
	@expose('/violation_trends/')
	@has_access
	def violation_trends(self):
		"""Compliance violation trends analysis"""
		try:
			period_days = int(request.args.get('period', 30))
			trends_data = self._get_violation_trends(period_days)
			
			return render_template('audit_compliance/violation_trends.html',
								   trends_data=trends_data,
								   period_days=period_days,
								   page_title="Violation Trends")
		except Exception as e:
			flash(f'Error loading violation trends: {str(e)}', 'error')
			return redirect(url_for('AuditComplianceDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get audit compliance metrics for dashboard"""
		# Implementation would calculate real metrics from database
		return {
			'total_audit_events': 45230,
			'events_today': 1250,
			'compliance_relevant_events': 12450,
			'active_violations': 28,
			'resolved_violations': 142,
			'overall_compliance_score': 94.2,
			'active_rules': 85,
			'triggered_rules_today': 12,
			'frameworks': [
				{'name': 'GDPR', 'score': 96.8, 'violations': 3},
				{'name': 'SOX', 'score': 92.1, 'violations': 8},
				{'name': 'HIPAA', 'score': 98.2, 'violations': 1}
			],
			'recent_violations': [],
			'top_event_types': [
				{'type': 'data_access', 'count': 8420},
				{'type': 'login', 'count': 6750},
				{'type': 'data_change', 'count': 4230}
			]
		}
	
	def _get_compliance_status(self) -> Dict[str, Any]:
		"""Get overall compliance status"""
		return {
			'frameworks': {
				'GDPR': {'score': 96.8, 'status': 'compliant', 'violations': 3},
				'SOX': {'score': 92.1, 'status': 'compliant', 'violations': 8},
				'HIPAA': {'score': 98.2, 'status': 'compliant', 'violations': 1},
				'PCI_DSS': {'score': 89.5, 'status': 'at_risk', 'violations': 12}
			},
			'overall_score': 94.2,
			'trend': 'improving',
			'last_assessment': datetime.now() - timedelta(days=7)
		}
	
	def _get_violation_trends(self, period_days: int) -> Dict[str, Any]:
		"""Get violation trends data"""
		return {
			'period_days': period_days,
			'total_violations': 28,
			'new_violations': 12,
			'resolved_violations': 15,
			'by_severity': {
				'critical': 2,
				'high': 8,
				'medium': 12,
				'low': 6
			},
			'by_framework': {
				'GDPR': 3,
				'SOX': 8,
				'HIPAA': 1,
				'PCI_DSS': 12,
				'ISO27001': 4
			},
			'daily_trends': [],
			'resolution_time_avg': 4.2  # days
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all audit compliance views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		ACAuditLogModelView,
		"Audit Logs",
		icon="fa-list-alt",
		category="Audit & Compliance",
		category_icon="fa-shield"
	)
	
	appbuilder.add_view(
		ACComplianceRuleModelView,
		"Compliance Rules",
		icon="fa-gavel",
		category="Audit & Compliance"
	)
	
	appbuilder.add_view(
		ACComplianceViolationModelView,
		"Violations",
		icon="fa-exclamation-triangle",
		category="Audit & Compliance"
	)
	
	appbuilder.add_view(
		ACComplianceReportModelView,
		"Compliance Reports",
		icon="fa-file-text",
		category="Audit & Compliance"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(AuditComplianceDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"Compliance Dashboard",
		href="/audit_compliance_dashboard/",
		icon="fa-dashboard",
		category="Audit & Compliance"
	)
	
	appbuilder.add_link(
		"Compliance Status",
		href="/audit_compliance_dashboard/compliance_status/",
		icon="fa-check-circle",
		category="Audit & Compliance"
	)
	
	appbuilder.add_link(
		"Violation Trends",
		href="/audit_compliance_dashboard/violation_trends/",
		icon="fa-line-chart",
		category="Audit & Compliance"
	)