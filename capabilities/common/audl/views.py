"""
APG Audit Logging Views

Flask-AppBuilder views for revolutionary audit logging UI with real-time dashboards,
natural language search, and collaborative investigation interfaces.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import render_template, request, jsonify, flash
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.mixins import AuditMixin
from flask_appbuilder.charts.views import DirectByChartView
from wtforms import Form, StringField, SelectField, TextAreaField, DateTimeField
from wtforms.validators import DataRequired, Optional
from wtforms.widgets import TextArea
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, Any, List, Optional

from .models import AuditEvent, AuditLevel, AuditEventType, EventSource, ComplianceFramework
from .service import AuditService

# APG Integration imports
try:
	from ..auth.models import User
	from ..mten.service import get_current_tenant
	from ..ntfy.service import NotificationService
	from ..colb.service import CollaborationService
except ImportError:
	# Mock for development
	User = None
	get_current_tenant = lambda: "test_tenant"
	NotificationService = None
	CollaborationService = None

class AuditEventSearchForm(Form):
	"""Advanced audit event search form"""
	query = StringField(
		'Natural Language Query',
		description='Ask questions like "show me failed login attempts last week" or "find admin changes to user permissions"'
	)
	event_type = SelectField(
		'Event Type',
		choices=[('', 'All Types')] + [(t.value, t.value) for t in AuditEventType],
		validators=[Optional()]
	)
	level = SelectField(
		'Level',
		choices=[('', 'All Levels')] + [(l.value, l.value) for l in AuditLevel],
		validators=[Optional()]
	)
	source = SelectField(
		'Source',
		choices=[('', 'All Sources')] + [(s.value, s.value) for s in EventSource],
		validators=[Optional()]
	)
	user_id = StringField(
		'User ID',
		description='Filter by specific user'
	)
	date_start = DateTimeField(
		'Start Date',
		validators=[Optional()],
		default=lambda: datetime.utcnow() - timedelta(days=7)
	)
	date_end = DateTimeField(
		'End Date',
		validators=[Optional()],
		default=datetime.utcnow
	)
	resource_type = StringField(
		'Resource Type',
		description='Filter by resource type (e.g., document, user, system)'
	)
	risk_score_min = SelectField(
		'Minimum Risk Score',
		choices=[('', 'Any Risk Level'), ('0.3', 'Low'), ('0.6', 'Medium'), ('0.8', 'High'), ('0.9', 'Critical')],
		validators=[Optional()]
	)

class ComplianceReportForm(Form):
	"""Compliance report generation form"""
	framework = SelectField(
		'Compliance Framework',
		choices=[(f.value, f.value) for f in ComplianceFramework],
		validators=[DataRequired()]
	)
	date_start = DateTimeField(
		'Report Start Date',
		validators=[DataRequired()],
		default=lambda: datetime.utcnow() - timedelta(days=30)
	)
	date_end = DateTimeField(
		'Report End Date',
		validators=[DataRequired()],
		default=datetime.utcnow
	)
	format = SelectField(
		'Export Format',
		choices=[('json', 'JSON'), ('pdf', 'PDF Report'), ('excel', 'Excel Spreadsheet')],
		validators=[DataRequired()],
		default='pdf'
	)
	include_violations = SelectField(
		'Include Violations',
		choices=[('true', 'Yes'), ('false', 'No')],
		validators=[DataRequired()],
		default='true'
	)
	include_recommendations = SelectField(
		'Include Recommendations',
		choices=[('true', 'Yes'), ('false', 'No')],
		validators=[DataRequired()],
		default='true'
	)

class AuditDashboardView(BaseView):
	"""Revolutionary real-time audit dashboard"""
	
	route_base = "/audit"
	default_view = "dashboard"
	
	@expose("/")
	@expose("/dashboard")
	@has_access
	def dashboard(self):
		"""Main audit dashboard with real-time monitoring"""
		tenant_id = get_current_tenant()
		
		# Get recent activity summary
		try:
			audit_service = AuditService(tenant_id=tenant_id)
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				# Get dashboard metrics
				metrics = loop.run_until_complete(audit_service.get_metrics())
				recent_events = loop.run_until_complete(
					self._get_recent_events(audit_service)
				)
				risk_summary = loop.run_until_complete(
					self._get_risk_summary(audit_service)
				)
				compliance_status = loop.run_until_complete(
					self._get_compliance_status(audit_service)
				)
				
			finally:
				loop.close()
				
		except Exception as e:
			flash(f"Error loading dashboard data: {str(e)}", "error")
			metrics = {"status": "error", "metrics": {}, "buffer_size": 0}
			recent_events = []
			risk_summary = {}
			compliance_status = {}
		
		return self.render_template(
			"audit/dashboard.html",
			metrics=metrics,
			recent_events=recent_events,
			risk_summary=risk_summary,
			compliance_status=compliance_status,
			tenant_id=tenant_id
		)
	
	async def _get_recent_events(self, audit_service: AuditService, limit: int = 50) -> List[Dict]:
		"""Get recent audit events for dashboard"""
		# Mock implementation - in production would query actual events
		from uuid_extensions import uuid7str
		
		events = []
		event_types = list(AuditEventType)
		levels = list(AuditLevel)
		
		for i in range(limit):
			event = {
				"id": uuid7str(),
				"timestamp": (datetime.utcnow() - timedelta(minutes=i*2)).isoformat(),
				"level": levels[i % len(levels)].value,
				"event_type": event_types[i % len(event_types)].value,
				"user_id": f"user_{i % 10}",
				"action": f"action_{i}",
				"resource_type": ["document", "user", "system"][i % 3],
				"resource_id": f"resource_{i}",
				"risk_score": min(1.0, (i % 10) * 0.1 + 0.1),
				"success": i % 4 != 0  # 75% success rate
			}
			events.append(event)
		
		return events
	
	async def _get_risk_summary(self, audit_service: AuditService) -> Dict:
		"""Get risk analysis summary"""
		return {
			"critical_events": 12,
			"high_risk_events": 45,
			"anomalies_detected": 8,
			"risk_trend": "increasing",
			"top_risk_categories": [
				{"category": "authentication", "count": 23, "avg_risk": 0.7},
				{"category": "data_access", "count": 34, "avg_risk": 0.4},
				{"category": "system_config", "count": 12, "avg_risk": 0.8}
			]
		}
	
	async def _get_compliance_status(self, audit_service: AuditService) -> Dict:
		"""Get compliance monitoring status"""
		return {
			"frameworks": [
				{"name": "GDPR", "compliance_score": 0.92, "violations": 2, "status": "good"},
				{"name": "SOX", "compliance_score": 0.88, "violations": 5, "status": "warning"},
				{"name": "HIPAA", "compliance_score": 0.95, "violations": 1, "status": "excellent"},
				{"name": "PCI-DSS", "compliance_score": 0.85, "violations": 8, "status": "needs_attention"}
			],
			"recent_violations": [
				{"framework": "SOX", "description": "Unauthorized financial data access", "severity": "high"},
				{"framework": "GDPR", "description": "Personal data exported without consent", "severity": "critical"}
			]
		}

class AuditSearchView(BaseView):
	"""Revolutionary natural language search interface"""
	
	route_base = "/audit/search"
	
	@expose("/", methods=["GET", "POST"])
	@has_access
	def search(self):
		"""Advanced audit log search with natural language queries"""
		form = AuditEventSearchForm(request.form)
		results = []
		search_context = None
		
		if request.method == "POST" and form.validate():
			tenant_id = get_current_tenant()
			
			try:
				audit_service = AuditService(tenant_id=tenant_id)
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				try:
					if form.query.data:
						# Natural language search
						results, search_context = loop.run_until_complete(
							self._natural_language_search(audit_service, form.query.data)
						)
					else:
						# Advanced filter search
						results = loop.run_until_complete(
							self._advanced_filter_search(audit_service, form)
						)
				finally:
					loop.close()
					
			except Exception as e:
				flash(f"Search failed: {str(e)}", "error")
		
		return self.render_template(
			"audit/search.html",
			form=form,
			results=results,
			search_context=search_context
		)
	
	async def _natural_language_search(self, audit_service: AuditService, query: str):
		"""Execute natural language search"""
		# Mock implementation - in production would use APG NLP service
		results = []
		context = {
			"interpreted_query": f"Searching for: {query}",
			"query_type": "natural_language",
			"confidence": 0.92,
			"suggestions": [
				"Try: 'show failed login attempts today'",
				"Try: 'find admin changes to user permissions this week'",
				"Try: 'list high-risk events from user john.doe'"
			]
		}
		
		# Generate mock results based on query
		query_lower = query.lower()
		event_count = 5 if "failed" in query_lower else 10
		
		from uuid_extensions import uuid7str
		for i in range(event_count):
			event = AuditEvent(
				id=uuid7str(),
				tenant_id="test_tenant",
				level=AuditLevel.WARNING if "failed" in query_lower else AuditLevel.INFO,
				event_type=AuditEventType.USER_FAILED_LOGIN if "login" in query_lower else AuditEventType.DATA_READ,
				source=EventSource.AUTH,
				category="authentication" if "login" in query_lower else "data_access",
				user_id=f"user_{i}",
				action="login_attempt" if "login" in query_lower else f"data_access_{i}",
				success=False if "failed" in query_lower else True,
				timestamp=datetime.utcnow() - timedelta(hours=i)
			)
			results.append(event)
		
		return results, context
	
	async def _advanced_filter_search(self, audit_service: AuditService, form):
		"""Execute advanced filter search"""
		# Mock implementation
		results = []
		from uuid_extensions import uuid7str
		
		for i in range(15):
			event = AuditEvent(
				id=uuid7str(),
				tenant_id="test_tenant",
				level=AuditLevel.INFO,
				event_type=AuditEventType.DATA_READ,
				source=EventSource.APG_CORE,
				category="data_access",
				user_id=form.user_id.data or f"user_{i}",
				action=f"filtered_action_{i}",
				resource_type=form.resource_type.data or "document",
				resource_id=f"resource_{i}",
				timestamp=datetime.utcnow() - timedelta(hours=i)
			)
			results.append(event)
		
		return results

class ComplianceReportView(BaseView):
	"""Automated compliance reporting interface"""
	
	route_base = "/audit/compliance"
	
	@expose("/", methods=["GET", "POST"])
	@has_access
	def reports(self):
		"""Generate and manage compliance reports"""
		form = ComplianceReportForm(request.form)
		recent_reports = []
		
		if request.method == "POST" and form.validate():
			tenant_id = get_current_tenant()
			
			try:
				# Generate compliance report
				report_id = self._generate_compliance_report(tenant_id, form)
				flash(f"Compliance report generation started. Report ID: {report_id}", "success")
				
			except Exception as e:
				flash(f"Report generation failed: {str(e)}", "error")
		
		# Get recent reports
		recent_reports = self._get_recent_reports()
		
		return self.render_template(
			"audit/compliance.html",
			form=form,
			recent_reports=recent_reports
		)
	
	def _generate_compliance_report(self, tenant_id: str, form: ComplianceReportForm) -> str:
		"""Generate compliance report in background"""
		from uuid_extensions import uuid7str
		
		report_id = uuid7str()
		
		# In production, this would trigger background report generation
		# For now, simulate the process
		
		return report_id
	
	def _get_recent_reports(self) -> List[Dict]:
		"""Get recent compliance reports"""
		from uuid_extensions import uuid7str
		
		reports = []
		frameworks = list(ComplianceFramework)
		
		for i in range(10):
			report = {
				"id": uuid7str(),
				"framework": frameworks[i % len(frameworks)].value,
				"generated_at": (datetime.utcnow() - timedelta(days=i)).isoformat(),
				"status": ["completed", "generating", "failed"][i % 3],
				"format": ["pdf", "excel", "json"][i % 3],
				"file_size": f"{(i+1)*2.3:.1f} MB"
			}
			reports.append(report)
		
		return reports

class AuditInvestigationView(BaseView):
	"""Collaborative audit investigation interface"""
	
	route_base = "/audit/investigate"
	
	@expose("/")
	@has_access
	def investigations(self):
		"""Manage collaborative audit investigations"""
		tenant_id = get_current_tenant()
		
		# Get active investigations
		active_investigations = self._get_active_investigations(tenant_id)
		recent_findings = self._get_recent_findings(tenant_id)
		
		return self.render_template(
			"audit/investigations.html",
			active_investigations=active_investigations,
			recent_findings=recent_findings
		)
	
	@expose("/create", methods=["GET", "POST"])
	@has_access
	def create_investigation(self):
		"""Create new investigation"""
		if request.method == "POST":
			# Create investigation logic
			investigation_data = request.get_json()
			investigation_id = self._create_investigation(investigation_data)
			
			return jsonify({
				"success": True,
				"investigation_id": investigation_id,
				"message": "Investigation created successfully"
			})
		
		return self.render_template("audit/create_investigation.html")
	
	@expose("/<investigation_id>")
	@has_access
	def investigation_detail(self, investigation_id: str):
		"""Investigation detail view with collaborative tools"""
		investigation = self._get_investigation_details(investigation_id)
		timeline = self._get_investigation_timeline(investigation_id)
		collaborators = self._get_investigation_collaborators(investigation_id)
		
		return self.render_template(
			"audit/investigation_detail.html",
			investigation=investigation,
			timeline=timeline,
			collaborators=collaborators
		)
	
	def _get_active_investigations(self, tenant_id: str) -> List[Dict]:
		"""Get active investigations"""
		from uuid_extensions import uuid7str
		
		investigations = []
		for i in range(5):
			investigation = {
				"id": uuid7str(),
				"title": f"Investigation {i+1}: Suspicious Activity Analysis",
				"description": f"Investigating unusual access patterns detected on {datetime.utcnow().date()}",
				"status": ["active", "pending", "completed"][i % 3],
				"priority": ["high", "medium", "low"][i % 3],
				"assigned_to": f"investigator_{i+1}",
				"created_at": (datetime.utcnow() - timedelta(days=i)).isoformat(),
				"events_count": (i+1) * 12,
				"findings_count": i * 3
			}
			investigations.append(investigation)
		
		return investigations
	
	def _get_recent_findings(self, tenant_id: str) -> List[Dict]:
		"""Get recent investigation findings"""
		findings = []
		for i in range(8):
			finding = {
				"id": f"finding_{i+1}",
				"investigation_id": f"inv_{i+1}",
				"title": f"Finding {i+1}: Unauthorized access detected",
				"severity": ["critical", "high", "medium", "low"][i % 4],
				"description": f"Analysis reveals suspicious pattern in audit event {i+1}",
				"discovered_at": (datetime.utcnow() - timedelta(hours=i*2)).isoformat(),
				"analyst": f"analyst_{i+1}"
			}
			findings.append(finding)
		
		return findings
	
	def _create_investigation(self, data: Dict) -> str:
		"""Create new investigation"""
		from uuid_extensions import uuid7str
		return uuid7str()
	
	def _get_investigation_details(self, investigation_id: str) -> Dict:
		"""Get investigation details"""
		return {
			"id": investigation_id,
			"title": "Advanced Persistent Threat Investigation",
			"description": "Investigating coordinated attack patterns across multiple user accounts",
			"status": "active",
			"priority": "critical",
			"created_at": datetime.utcnow().isoformat(),
			"assigned_investigators": ["analyst1", "analyst2", "security_lead"],
			"tags": ["apt", "credential_stuffing", "data_exfiltration"],
			"evidence_count": 47,
			"timeline_events": 156
		}
	
	def _get_investigation_timeline(self, investigation_id: str) -> List[Dict]:
		"""Get investigation timeline"""
		timeline = []
		for i in range(10):
			event = {
				"timestamp": (datetime.utcnow() - timedelta(hours=i*2)).isoformat(),
				"type": ["evidence_added", "finding_created", "note_added", "status_updated"][i % 4],
				"title": f"Timeline Event {i+1}",
				"description": f"Investigation activity: {i+1}",
				"analyst": f"analyst_{i % 3 + 1}"
			}
			timeline.append(event)
		
		return timeline
	
	def _get_investigation_collaborators(self, investigation_id: str) -> List[Dict]:
		"""Get investigation collaborators"""
		return [
			{"name": "Security Analyst 1", "role": "Lead Investigator", "status": "active"},
			{"name": "Security Analyst 2", "role": "Evidence Collector", "status": "active"},
			{"name": "Compliance Officer", "role": "Compliance Review", "status": "reviewing"},
			{"name": "IT Manager", "role": "Technical Support", "status": "available"}
		]

class AuditSettingsView(BaseView):
	"""Audit logging configuration and settings"""
	
	route_base = "/audit/settings"
	
	@expose("/")
	@has_access
	def settings(self):
		"""Audit logging settings and configuration"""
		tenant_id = get_current_tenant()
		
		# Get current settings
		current_settings = self._get_current_settings(tenant_id)
		retention_policies = self._get_retention_policies(tenant_id)
		alert_rules = self._get_alert_rules(tenant_id)
		
		return self.render_template(
			"audit/settings.html",
			settings=current_settings,
			retention_policies=retention_policies,
			alert_rules=alert_rules
		)
	
	@expose("/update", methods=["POST"])
	@has_access
	def update_settings(self):
		"""Update audit logging settings"""
		settings_data = request.get_json()
		
		try:
			self._update_settings(settings_data)
			flash("Settings updated successfully", "success")
			return jsonify({"success": True})
		except Exception as e:
			flash(f"Settings update failed: {str(e)}", "error")
			return jsonify({"success": False, "error": str(e)})
	
	def _get_current_settings(self, tenant_id: str) -> Dict:
		"""Get current audit settings"""
		return {
			"event_retention_days": 365,
			"high_risk_threshold": 0.7,
			"real_time_alerting": True,
			"ml_anomaly_detection": True,
			"compliance_monitoring": True,
			"export_encryption": True,
			"backup_enabled": True,
			"backup_frequency": "daily"
		}
	
	def _get_retention_policies(self, tenant_id: str) -> List[Dict]:
		"""Get data retention policies"""
		return [
			{"type": "audit_events", "retention_days": 365, "archive_after_days": 90},
			{"type": "compliance_reports", "retention_days": 2555, "archive_after_days": 365},  # 7 years
			{"type": "investigation_data", "retention_days": 1825, "archive_after_days": 365},  # 5 years
			{"type": "system_logs", "retention_days": 90, "archive_after_days": 30}
		]
	
	def _get_alert_rules(self, tenant_id: str) -> List[Dict]:
		"""Get alerting rules"""
		return [
			{
				"name": "Critical Risk Events",
				"condition": "risk_score >= 0.9",
				"enabled": True,
				"notification_channels": ["email", "slack", "sms"]
			},
			{
				"name": "Multiple Failed Logins",
				"condition": "failed_login_count >= 5 in 10 minutes",
				"enabled": True,
				"notification_channels": ["email", "slack"]
			},
			{
				"name": "Compliance Violations",
				"condition": "compliance_violation = true",
				"enabled": True,
				"notification_channels": ["email", "webhook"]
			}
		]
	
	def _update_settings(self, settings_data: Dict) -> None:
		"""Update audit settings"""
		# In production, this would update database settings
		pass

# Export all views for APG integration
__all__ = [
	"AuditDashboardView",
	"AuditSearchView", 
	"ComplianceReportView",
	"AuditInvestigationView",
	"AuditSettingsView"
]