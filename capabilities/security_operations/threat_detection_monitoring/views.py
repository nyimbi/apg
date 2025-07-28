"""
APG Threat Detection & Monitoring - Flask-AppBuilder Views

Enterprise threat detection dashboard and views with comprehensive
security monitoring, incident management, and analytics capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from flask import Blueprint, flash, redirect, request, url_for
from flask_appbuilder import BaseView, ModelView, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import has_access
from wtforms import BooleanField, DateTimeField, DecimalField, SelectField, StringField, TextAreaField
from wtforms.validators import DataRequired, Length, NumberRange

from .models import (
	SecurityEvent, ThreatIndicator, SecurityIncident, BehavioralProfile,  
	ThreatIntelligence, SecurityRule, IncidentResponse, ThreatAnalysis,
	SecurityMetrics, ForensicEvidence, ThreatSeverity, ThreatStatus,
	EventType, AnalysisEngine, ResponseAction
)
from .service import ThreatDetectionService


class ThreatDetectionViews:
	"""Threat Detection Flask-AppBuilder Views"""
	
	def __init__(self, appbuilder):
		self.appbuilder = appbuilder
		self.blueprint = Blueprint(
			'threat_detection',
			__name__,
			template_folder='templates',
			static_folder='static'
		)
		self._setup_views()
	
	def _setup_views(self):
		"""Setup all threat detection views"""
		
		class SecurityEventView(ModelView):
			"""Security Event management view"""
			datamodel = SQLAInterface(SecurityEvent)
			
			list_title = "Security Events"
			show_title = "Security Event Details"
			add_title = "Add Security Event"
			edit_title = "Edit Security Event"
			
			list_columns = [
				'event_id', 'event_type', 'source_system', 'source_ip',
				'user_id', 'risk_score', 'confidence', 'timestamp'
			]
			
			show_columns = [
				'id', 'tenant_id', 'event_id', 'event_type', 'source_system',
				'source_ip', 'destination_ip', 'user_id', 'username', 'asset_id',
				'hostname', 'timestamp', 'raw_data', 'normalized_data',
				'geolocation', 'user_agent', 'process_name', 'command_line',
				'file_path', 'risk_score', 'confidence', 'created_at', 'updated_at'
			]
			
			search_columns = [
				'event_id', 'source_system', 'source_ip', 'user_id',
				'username', 'hostname'
			]
			
			label_columns = {
				'event_id': 'Event ID',
				'event_type': 'Event Type',
				'source_system': 'Source System',
				'source_ip': 'Source IP',
				'destination_ip': 'Destination IP',
				'user_id': 'User ID',
				'username': 'Username',
				'asset_id': 'Asset ID',
				'hostname': 'Hostname',
				'risk_score': 'Risk Score',
				'confidence': 'Confidence',
				'created_at': 'Created At',
				'updated_at': 'Updated At'
			}
			
			formatters_columns = {
				'risk_score': lambda x: f"{float(x):.1f}%" if x else "0.0%",
				'confidence': lambda x: f"{float(x):.1f}%" if x else "0.0%",
				'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else None
			}
			
			base_order = ('timestamp', 'desc')
			base_filters = [['timestamp', FilterTimeFunction, datetime.utcnow]]
		
		class SecurityIncidentView(ModelView):
			"""Security Incident management view"""
			datamodel = SQLAInterface(SecurityIncident)
			
			list_title = "Security Incidents"
			show_title = "Security Incident Details"
			add_title = "Create Security Incident"
			edit_title = "Edit Security Incident"
			
			list_columns = [
				'title', 'severity', 'status', 'incident_type',
				'assigned_to', 'first_detected', 'last_activity'
			]
			
			show_columns = [
				'id', 'tenant_id', 'title', 'description', 'severity', 'status',
				'incident_type', 'attack_vector', 'affected_systems', 'affected_users',
				'affected_assets', 'indicators', 'events', 'assigned_to', 'assignee_name',
				'first_detected', 'last_activity', 'resolved_at', 'timeline',
				'evidence', 'containment_actions', 'remediation_actions',
				'impact_assessment', 'root_cause', 'lessons_learned',
				'created_at', 'updated_at'
			]
			
			add_columns = [
				'title', 'description', 'severity', 'incident_type',
				'affected_systems', 'affected_users', 'affected_assets',  
				'assigned_to', 'assignee_name'
			]
			
			edit_columns = [
				'title', 'description', 'severity', 'status', 'incident_type',
				'attack_vector', 'affected_systems', 'affected_users', 'affected_assets',
				'assigned_to', 'assignee_name', 'impact_assessment', 'root_cause',
				'lessons_learned'
			]
			
			search_columns = [
				'title', 'incident_type', 'assigned_to', 'assignee_name'
			]
			
			label_columns = {
				'title': 'Incident Title',
				'description': 'Description',
				'severity': 'Severity',
				'status': 'Status',
				'incident_type': 'Incident Type',
				'attack_vector': 'Attack Vector',
				'affected_systems': 'Affected Systems',
				'affected_users': 'Affected Users',
				'affected_assets': 'Affected Assets',
				'assigned_to': 'Assigned To',
				'assignee_name': 'Assignee Name',
				'first_detected': 'First Detected',
				'last_activity': 'Last Activity',
				'resolved_at': 'Resolved At',
				'impact_assessment': 'Impact Assessment',
				'root_cause': 'Root Cause',
				'lessons_learned': 'Lessons Learned'
			}
			
			formatters_columns = {
				'severity': lambda x: f"🔴 {x.value.title()}" if x.value == 'critical' else f"🟡 {x.value.title()}",
				'status': lambda x: f"🟢 {x.value.title()}" if x.value == 'resolved' else f"🔴 {x.value.title()}",
				'first_detected': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else None,
				'last_activity': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else None
			}
			
			base_order = ('first_detected', 'desc')
		
		class ThreatIndicatorView(ModelView):
			"""Threat Indicator (IOC) management view"""
			datamodel = SQLAInterface(ThreatIndicator)
			
			list_title = "Threat Indicators (IOCs)"
			show_title = "Threat Indicator Details"
			add_title = "Add Threat Indicator"
			edit_title = "Edit Threat Indicator"
			
			list_columns = [
				'indicator_type', 'indicator_value', 'description',
				'severity', 'confidence', 'source', 'is_active', 'first_seen'
			]
			
			add_columns = [
				'indicator_type', 'indicator_value', 'description', 'severity',
				'confidence', 'source', 'tags', 'malware_families',
				'attack_techniques', 'expiry_date'
			]
			
			edit_columns = [
				'indicator_type', 'indicator_value', 'description', 'severity',
				'confidence', 'source', 'tags', 'malware_families',
				'attack_techniques', 'expiry_date', 'is_active'
			]
			
			search_columns = [
				'indicator_type', 'indicator_value', 'description', 'source'
			]
			
			label_columns = {
				'indicator_type': 'Indicator Type',
				'indicator_value': 'Indicator Value',
				'description': 'Description',
				'severity': 'Severity',
				'confidence': 'Confidence',
				'source': 'Source',
				'tags': 'Tags',
				'first_seen': 'First Seen',
				'last_seen': 'Last Seen',
				'expiry_date': 'Expiry Date',
				'malware_families': 'Malware Families',
				'attack_techniques': 'Attack Techniques',
				'is_active': 'Active'
			}
			
			formatters_columns = {
				'confidence': lambda x: f"{float(x):.1f}%" if x else "0.0%",
				'severity': lambda x: f"🔴 {x.value.title()}" if x.value == 'critical' else f"🟡 {x.value.title()}",
				'is_active': lambda x: "✅ Active" if x else "❌ Inactive"
			}
		
		class BehavioralProfileView(ModelView):
			"""Behavioral Profile management view"""
			datamodel = SQLAInterface(BehavioralProfile)
			
			list_title = "Behavioral Profiles"
			show_title = "Behavioral Profile Details"
			add_title = "Create Behavioral Profile"
			edit_title = "Edit Behavioral Profile"
			
			list_columns = [
				'entity_name', 'entity_type', 'baseline_established',
				'baseline_confidence', 'anomaly_score', 'risk_score', 'last_analyzed'
			]
			
			show_columns = [
				'id', 'tenant_id', 'entity_id', 'entity_type', 'entity_name',
				'profile_period_start', 'profile_period_end', 'baseline_established',
				'baseline_confidence', 'normal_login_hours', 'normal_login_locations',
				'normal_access_patterns', 'typical_systems_accessed', 'typical_data_accessed',
				'typical_operations', 'peer_group', 'peer_comparison_score',
				'anomaly_score', 'risk_score', 'recent_anomalies', 'behavioral_changes',
				'last_analyzed', 'created_at', 'updated_at'
			]
			
			search_columns = ['entity_name', 'entity_type', 'entity_id']
			
			label_columns = {
				'entity_id': 'Entity ID',
				'entity_type': 'Entity Type',
				'entity_name': 'Entity Name',
				'profile_period_start': 'Profile Period Start',
				'profile_period_end': 'Profile Period End',
				'baseline_established': 'Baseline Established',
				'baseline_confidence': 'Baseline Confidence',
				'normal_login_hours': 'Normal Login Hours',
				'normal_login_locations': 'Normal Login Locations',
				'typical_systems_accessed': 'Typical Systems Accessed',
				'anomaly_score': 'Anomaly Score',
				'risk_score': 'Risk Score',
				'last_analyzed': 'Last Analyzed'
			}
			
			formatters_columns = {
				'baseline_confidence': lambda x: f"{float(x):.1f}%" if x else "0.0%",
				'anomaly_score': lambda x: f"{float(x):.1f}%" if x else "0.0%",
				'risk_score': lambda x: f"{float(x):.1f}%" if x else "0.0%",
				'baseline_established': lambda x: "✅ Yes" if x else "❌ No"
			}
		
		class SecurityRuleView(ModelView):
			"""Security Rule management view"""
			datamodel = SQLAInterface(SecurityRule)
			
			list_title = "Security Detection Rules"
			show_title = "Security Rule Details"
			add_title = "Create Security Rule"
			edit_title = "Edit Security Rule"
			
			list_columns = [
				'name', 'rule_type', 'analysis_engine', 'severity',
				'confidence', 'is_active', 'last_triggered', 'trigger_count'
			]
			
			add_columns = [
				'name', 'description', 'rule_type', 'analysis_engine', 'query',
				'query_language', 'severity', 'confidence', 'event_types',
				'data_sources', 'mitre_techniques', 'tags', 'threshold_conditions',
				'response_actions', 'created_by'
			]
			
			edit_columns = [
				'name', 'description', 'rule_type', 'analysis_engine', 'query',
				'query_language', 'severity', 'confidence', 'event_types',
				'data_sources', 'mitre_techniques', 'tags', 'threshold_conditions',
				'response_actions', 'is_active'
			]
			
			search_columns = ['name', 'rule_type', 'created_by']
			
			label_columns = {
				'name': 'Rule Name',
				'description': 'Description',
				'rule_type': 'Rule Type',
				'analysis_engine': 'Analysis Engine',
				'query': 'Detection Query',
				'query_language': 'Query Language',
				'severity': 'Severity',
				'confidence': 'Confidence',
				'event_types': 'Event Types',
				'data_sources': 'Data Sources',
				'mitre_techniques': 'MITRE Techniques',
				'tags': 'Tags',
				'response_actions': 'Response Actions',
				'is_active': 'Active',
				'created_by': 'Created By',
				'last_triggered': 'Last Triggered',
				'trigger_count': 'Trigger Count'
			}
			
			formatters_columns = {
				'confidence': lambda x: f"{float(x):.1f}%" if x else "0.0%",
				'is_active': lambda x: "✅ Active" if x else "❌ Inactive",
				'trigger_count': lambda x: f"{x:,}" if x else "0"
			}
		
		class SecurityDashboardView(BaseView):
			"""Security Operations Dashboard"""
			
			route_base = "/threat-detection"
			default_view = "dashboard"
			
			@expose("/dashboard")
			@has_access
			def dashboard(self):
				"""Main security dashboard"""
				try:
					service = ThreatDetectionService(self.appbuilder.get_session, request.args.get('tenant_id', 'default'))
					
					dashboard_data = {
						'total_events_today': 0,
						'active_incidents': 0,
						'critical_alerts': 0,
						'threat_indicators': 0,
						'recent_incidents': [],
						'top_threats': [],
						'security_metrics': {},
						'threat_trends': []
					}
					
					dashboard_data['total_events_today'] = service.count_events_today()
					dashboard_data['active_incidents'] = service.count_active_incidents()
					dashboard_data['critical_alerts'] = service.count_critical_alerts()
					dashboard_data['threat_indicators'] = service.count_active_indicators()
					
					dashboard_data['recent_incidents'] = service.get_recent_incidents(limit=10)
					dashboard_data['top_threats'] = service.get_top_threats()
					dashboard_data['security_metrics'] = service.get_dashboard_metrics()
					dashboard_data['threat_trends'] = service.get_threat_trends(days=30)
					
					return self.render_template(
						'threat_detection/dashboard.html',
						**dashboard_data
					)
					
				except Exception as e:
					flash(f"Error loading dashboard: {str(e)}", "error")
					return self.render_template('threat_detection/dashboard.html')
			
			@expose("/incident-response/<incident_id>")
			@has_access
			def incident_response(self, incident_id):
				"""Incident response interface"""
				try:
					service = ThreatDetectionService(self.appbuilder.get_session, request.args.get('tenant_id', 'default'))
					
					incident = service.get_security_incident(incident_id)
					if not incident:
						flash("Incident not found", "error")
						return redirect(url_for('SecurityIncidentView.list'))
					
					response_options = service.get_response_playbooks()
					forensic_evidence = service.get_forensic_evidence(incident_id)
					
					return self.render_template(
						'threat_detection/incident_response.html',
						incident=incident,
						response_options=response_options,
						forensic_evidence=forensic_evidence
					)
					
				except Exception as e:
					flash(f"Error loading incident response: {str(e)}", "error")
					return redirect(url_for('SecurityIncidentView.list'))
			
			@expose("/threat-hunting")
			@has_access
			def threat_hunting(self):
				"""Threat hunting interface"""
				try:
					service = ThreatDetectionService(self.appbuilder.get_session, request.args.get('tenant_id', 'default'))
					
					hunt_templates = service.get_hunt_templates()
					recent_hunts = service.get_recent_hunts(limit=20)
					
					return self.render_template(
						'threat_detection/threat_hunting.html',
						hunt_templates=hunt_templates,
						recent_hunts=recent_hunts
					)
					
				except Exception as e:
					flash(f"Error loading threat hunting: {str(e)}", "error")
					return self.render_template('threat_detection/threat_hunting.html')
			
			@expose("/behavioral-analytics")
			@has_access  
			def behavioral_analytics(self):
				"""Behavioral analytics dashboard"""
				try:
					service = ThreatDetectionService(self.appbuilder.get_session, request.args.get('tenant_id', 'default'))
					
					user_profiles = service.get_user_profiles(limit=50)
					anomaly_summary = service.get_anomaly_summary()
					behavioral_trends = service.get_behavioral_trends(days=30)
					
					return self.render_template(
						'threat_detection/behavioral_analytics.html',
						user_profiles=user_profiles,
						anomaly_summary=anomaly_summary,
						behavioral_trends=behavioral_trends
					)
					
				except Exception as e:
					flash(f"Error loading behavioral analytics: {str(e)}", "error")
					return self.render_template('threat_detection/behavioral_analytics.html')
			
			@expose("/threat-intelligence")
			@has_access
			def threat_intelligence(self):
				"""Threat intelligence dashboard"""
				try:
					service = ThreatDetectionService(self.appbuilder.get_session, request.args.get('tenant_id', 'default'))
					
					intelligence_feeds = service.get_active_intelligence_feeds()
					recent_intelligence = service.get_recent_intelligence(limit=100)
					intelligence_stats = service.get_intelligence_statistics()
					
					return self.render_template(
						'threat_detection/threat_intelligence.html',
						intelligence_feeds=intelligence_feeds,
						recent_intelligence=recent_intelligence,
						intelligence_stats=intelligence_stats
					)
					
				except Exception as e:
					flash(f"Error loading threat intelligence: {str(e)}", "error")
					return self.render_template('threat_detection/threat_intelligence.html')
			
			@expose("/security-metrics")
			@has_access
			def security_metrics(self):
				"""Security metrics and reporting"""
				try:
					service = ThreatDetectionService(self.appbuilder.get_session, request.args.get('tenant_id', 'default'))
					
					period_days = int(request.args.get('period', 30))
					metrics = service.get_security_metrics(period_days)
					
					performance_trends = service.get_performance_trends(period_days)
					incident_trends = service.get_incident_trends(period_days)
					
					return self.render_template(
						'threat_detection/security_metrics.html',
						metrics=metrics,
						performance_trends=performance_trends,
						incident_trends=incident_trends,
						period_days=period_days
					)
					
				except Exception as e:
					flash(f"Error loading security metrics: {str(e)}", "error")
					return self.render_template('threat_detection/security_metrics.html')
		
		self.appbuilder.add_view(
			SecurityEventView,
			"Security Events",
			icon="fa-shield-alt",
			category="Threat Detection",
			category_icon="fa-shield-alt"
		)
		
		self.appbuilder.add_view(
			SecurityIncidentView,
			"Security Incidents",
			icon="fa-exclamation-triangle",
			category="Threat Detection"
		)
		
		self.appbuilder.add_view(
			ThreatIndicatorView,
			"Threat Indicators",
			icon="fa-search",
			category="Threat Detection"
		)
		
		self.appbuilder.add_view(
			BehavioralProfileView,
			"Behavioral Profiles",
			icon="fa-user-chart",
			category="Threat Detection"
		)
		
		self.appbuilder.add_view(
			SecurityRuleView,
			"Security Rules",
			icon="fa-cogs",
			category="Threat Detection"
		)
		
		self.appbuilder.add_view(
			SecurityDashboardView,
			"Security Dashboard",
			icon="fa-dashboard",
			category="Threat Detection"
		)
	
	def get_blueprint(self) -> Blueprint:
		"""Get the Flask blueprint"""
		return self.blueprint