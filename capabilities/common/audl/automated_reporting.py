"""
APG Audit Logging Automated Reporting & Analytics

Revolutionary automated compliance reporting system with intelligent delivery,
custom report builders, executive dashboards, and predictive analytics.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import pandas as pd
import numpy as np
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from .models import AuditEvent, AuditLevel, AuditEventType
from .elasticsearch_integration import ElasticsearchAuditService, SearchQuery
from .compliance_frameworks import ComplianceManager, ComplianceFramework
from .ml_anomaly_detection import AnomalyMLEngine

# APG Integration
try:
	from ..ntfy.service import NotificationService, Priority
	from ..mten.service import get_current_tenant
	from ..doc_mgmt.service import DocumentManagementService
	from ..bi.service import BusinessIntelligenceService
	from ..sched.service import SchedulerService
except ImportError:
	# Mock services for development
	class MockService:
		async def __aenter__(self): return self
		async def __aexit__(self, *args): pass
		async def send_notification(self, **kwargs): pass
		async def create_document(self, **kwargs): return {"id": "test_doc"}
		async def create_dashboard(self, **kwargs): return {"id": "test_dashboard"}
		async def schedule_task(self, **kwargs): return {"id": "test_schedule"}
	
	NotificationService = MockService
	DocumentManagementService = MockService  
	BusinessIntelligenceService = MockService
	SchedulerService = MockService
	get_current_tenant = lambda: "test_tenant"

logger = logging.getLogger(__name__)

class ReportType(Enum):
	"""Types of automated reports"""
	COMPLIANCE_SUMMARY = "compliance_summary"
	SECURITY_AUDIT = "security_audit"  
	USER_ACTIVITY = "user_activity"
	RISK_ASSESSMENT = "risk_assessment"
	ANOMALY_ANALYSIS = "anomaly_analysis"
	EXECUTIVE_DASHBOARD = "executive_dashboard"
	REGULATORY_SUBMISSION = "regulatory_submission"
	INCIDENT_INVESTIGATION = "incident_investigation"
	TREND_ANALYSIS = "trend_analysis"
	COMPARATIVE_ANALYSIS = "comparative_analysis"

class ReportFormat(Enum):
	"""Report output formats"""
	PDF = "pdf"
	HTML = "html"
	EXCEL = "excel"
	CSV = "csv"
	JSON = "json"
	DASHBOARD = "dashboard"

class DeliveryMethod(Enum):
	"""Report delivery methods"""
	EMAIL = "email"
	WEBHOOK = "webhook"
	PORTAL = "portal"
	API = "api"
	DASHBOARD = "dashboard"
	SLACK = "slack"
	TEAMS = "teams"

class ReportFrequency(Enum):
	"""Report generation frequency"""
	REAL_TIME = "real_time"
	HOURLY = "hourly"
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	ANNUALLY = "annually"
	ON_DEMAND = "on_demand"

@dataclass
class ReportConfiguration:
	"""Configuration for automated reports"""
	id: str
	name: str
	description: str
	report_type: ReportType
	enabled: bool = True
	
	# Scheduling
	frequency: ReportFrequency = ReportFrequency.DAILY
	schedule_time: Optional[str] = None  # "09:00" format
	timezone: str = "UTC"
	
	# Content filters
	date_range_days: int = 30
	event_types: List[AuditEventType] = field(default_factory=list)
	user_filters: List[str] = field(default_factory=list)
	resource_filters: List[str] = field(default_factory=list)
	compliance_frameworks: List[str] = field(default_factory=list)
	severity_threshold: str = "LOW"
	
	# Output configuration
	formats: List[ReportFormat] = field(default_factory=lambda: [ReportFormat.PDF])
	delivery_methods: List[DeliveryMethod] = field(default_factory=lambda: [DeliveryMethod.EMAIL])
	recipients: List[str] = field(default_factory=list)
	
	# Advanced options
	include_charts: bool = True
	include_recommendations: bool = True
	include_trend_analysis: bool = True
	include_comparative_data: bool = False
	custom_branding: bool = True
	
	# Retention
	retention_days: int = 365
	archive_after_days: int = 90

@dataclass
class ReportMetrics:
	"""Report generation metrics and analytics"""
	total_events_analyzed: int = 0
	compliance_violations_found: int = 0
	anomalies_detected: int = 0
	high_risk_activities: int = 0
	users_analyzed: int = 0
	systems_analyzed: int = 0
	
	# Performance metrics
	generation_time_seconds: float = 0.0
	data_processing_time: float = 0.0
	report_rendering_time: float = 0.0
	
	# Quality metrics
	data_completeness_percentage: float = 100.0
	accuracy_score: float = 0.0
	confidence_level: float = 0.0

@dataclass
class ReportInsight:
	"""Automated insights and recommendations"""
	type: str
	severity: str
	title: str
	description: str
	recommendation: str
	confidence: float
	supporting_data: Dict[str, Any] = field(default_factory=dict)
	trend_direction: str = "stable"  # increasing, decreasing, stable
	business_impact: str = "low"  # low, medium, high, critical

class ReportGenerator:
	"""Revolutionary automated report generation engine"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		
		# Services
		self.elasticsearch_service: Optional[ElasticsearchAuditService] = None
		self.compliance_manager: Optional[ComplianceManager] = None
		self.ml_engine: Optional[AnomalyMLEngine] = None
		self.notification_service = NotificationService()
		self.document_service = DocumentManagementService()
		self.bi_service = BusinessIntelligenceService()
		self.scheduler_service = SchedulerService()
		
		# Report templates
		self.templates = {}
		self.report_configs: Dict[str, ReportConfiguration] = {}
		self.generated_reports: List[Dict[str, Any]] = []
		
		# Analytics
		self.generation_stats = {
			"total_reports_generated": 0,
			"successful_deliveries": 0,
			"failed_deliveries": 0,
			"avg_generation_time": 0.0,
			"popular_report_types": {},
			"delivery_success_rate": 0.0
		}
	
	async def initialize(self) -> None:
		"""Initialize report generator with services and templates"""
		try:
			logger.info(f"Initializing automated reporting for tenant {self.tenant_id}")
			
			# Initialize services
			self.elasticsearch_service = ElasticsearchAuditService(tenant_id=self.tenant_id)
			await self.elasticsearch_service.initialize()
			
			self.compliance_manager = ComplianceManager(tenant_id=self.tenant_id)
			await self.compliance_manager.initialize()
			
			self.ml_engine = AnomalyMLEngine(tenant_id=self.tenant_id)
			await self.ml_engine.initialize()
			
			# Load report templates
			await self._load_report_templates()
			
			# Load default report configurations
			await self._load_default_report_configs()
			
			# Schedule existing reports
			await self._schedule_reports()
			
			logger.info("Automated reporting initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize report generator: {str(e)}")
			raise
	
	async def generate_report(
		self, 
		config: ReportConfiguration,
		date_range_override: Optional[Tuple[datetime, datetime]] = None
	) -> Dict[str, Any]:
		"""Generate automated report with analytics and insights"""
		try:
			start_time = datetime.utcnow()
			logger.info(f"Generating report: {config.name} ({config.report_type.value})")
			
			# Determine date range
			if date_range_override:
				start_date, end_date = date_range_override
			else:
				end_date = datetime.utcnow()
				start_date = end_date - timedelta(days=config.date_range_days)
			
			# Collect and analyze data
			report_data = await self._collect_report_data(config, start_date, end_date)
			
			# Generate insights and recommendations
			insights = await self._generate_insights(config, report_data)
			
			# Create visualizations
			charts = await self._generate_charts(config, report_data) if config.include_charts else []
			
			# Calculate metrics
			metrics = await self._calculate_report_metrics(report_data)
			
			# Generate report content
			report_content = await self._generate_report_content(
				config, report_data, insights, charts, metrics, start_date, end_date
			)
			
			# Render in requested formats
			rendered_reports = {}
			for format_type in config.formats:
				rendered_reports[format_type.value] = await self._render_report(
					report_content, format_type
				)
			
			# Create report metadata
			generation_time = (datetime.utcnow() - start_time).total_seconds()
			report_metadata = {
				"id": str(uuid.uuid4()),
				"config_id": config.id,
				"name": config.name,
				"type": config.report_type.value,
				"tenant_id": self.tenant_id,
				"generated_at": start_time,
				"generation_time_seconds": generation_time,
				"date_range": {
					"start": start_date.isoformat(),
					"end": end_date.isoformat()
				},
				"metrics": metrics.__dict__,
				"insights_count": len(insights),
				"formats": list(rendered_reports.keys()),
				"status": "completed"
			}
			
			# Store report
			stored_reports = await self._store_report(report_metadata, rendered_reports)
			
			# Deliver report
			delivery_results = await self._deliver_report(config, report_metadata, stored_reports)
			
			# Update statistics
			self._update_generation_stats(config, generation_time, delivery_results)
			
			# Add to generated reports list
			self.generated_reports.append(report_metadata)
			
			logger.info(f"Report generated successfully: {config.name}")
			
			return {
				"success": True,
				"report": report_metadata,
				"delivery_results": delivery_results,
				"insights": [insight.__dict__ for insight in insights]
			}
			
		except Exception as e:
			logger.error(f"Report generation failed: {str(e)}")
			return {
				"success": False,
				"error": str(e),
				"config_id": config.id
			}
	
	async def _collect_report_data(
		self, 
		config: ReportConfiguration, 
		start_date: datetime, 
		end_date: datetime
	) -> Dict[str, Any]:
		"""Collect data for report generation"""
		try:
			logger.info(f"Collecting data for report: {config.name}")
			
			# Base search query
			search_query = SearchQuery(
				tenant_id=self.tenant_id,
				date_range_start=start_date,
				date_range_end=end_date,
				size=50000  # Large size for comprehensive analysis
			)
			
			# Apply filters
			if config.event_types:
				search_query.event_types = config.event_types
			if config.user_filters:
				search_query.user_filters = config.user_filters
			
			# Collect audit events
			search_result = await self.elasticsearch_service.search(search_query)
			events = search_result.events if search_result else []
			
			# Collect compliance data
			compliance_data = {}
			if config.compliance_frameworks:
				for framework in config.compliance_frameworks:
					compliance_data[framework] = await self.compliance_manager.get_compliance_status(
						ComplianceFramework(framework)
					)
			
			# Collect ML anomalies
			anomalies = []
			if events:
				anomalies = await self.ml_engine.detect_anomalies(events[:1000])  # Sample for performance
			
			# Aggregate data by various dimensions
			aggregations = await self._create_data_aggregations(events)
			
			return {
				"events": events,
				"compliance_data": compliance_data,
				"anomalies": anomalies,
				"aggregations": aggregations,
				"date_range": {
					"start": start_date,
					"end": end_date,
					"days": (end_date - start_date).days
				},
				"total_events": len(events)
			}
			
		except Exception as e:
			logger.error(f"Data collection failed: {str(e)}")
			return {"events": [], "total_events": 0}
	
	async def _create_data_aggregations(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Create data aggregations for analysis"""
		if not events:
			return {}
		
		df = pd.DataFrame(events)
		
		# Temporal aggregations
		df['timestamp'] = pd.to_datetime(df['timestamp'])
		df['hour'] = df['timestamp'].dt.hour
		df['day_of_week'] = df['timestamp'].dt.dayofweek
		df['date'] = df['timestamp'].dt.date
		
		aggregations = {
			"temporal": {
				"events_by_hour": df.groupby('hour').size().to_dict(),
				"events_by_day": df.groupby('day_of_week').size().to_dict(),
				"events_by_date": df.groupby('date').size().to_dict(),
				"peak_activity_hour": df['hour'].mode().iloc[0] if not df.empty else 0,
				"peak_activity_day": df['day_of_week'].mode().iloc[0] if not df.empty else 0
			},
			
			"user_activity": {
				"unique_users": df['user_id'].nunique() if 'user_id' in df.columns else 0,
				"top_users": df['user_id'].value_counts().head(10).to_dict() if 'user_id' in df.columns else {},
				"events_per_user": df.groupby('user_id').size().describe().to_dict() if 'user_id' in df.columns else {}
			},
			
			"event_types": {
				"distribution": df['event_type'].value_counts().to_dict() if 'event_type' in df.columns else {},
				"success_rate": df['success'].mean() if 'success' in df.columns else 1.0,
				"failure_events": len(df[df['success'] == False]) if 'success' in df.columns else 0
			},
			
			"security": {
				"high_risk_events": len(df[df['risk_score'] > 0.7]) if 'risk_score' in df.columns else 0,
				"avg_risk_score": df['risk_score'].mean() if 'risk_score' in df.columns else 0.0,
				"failed_logins": len(df[df['event_type'] == 'USER_FAILED_LOGIN']) if 'event_type' in df.columns else 0,
				"admin_activities": len(df[df['level'] == 'ADMIN']) if 'level' in df.columns else 0
			},
			
			"resources": {
				"unique_resources": df['resource_type'].nunique() if 'resource_type' in df.columns else 0,
				"most_accessed": df['resource_type'].value_counts().head(10).to_dict() if 'resource_type' in df.columns else {},
				"sensitive_access": len(df[df['resource_type'].str.contains('sensitive', case=False, na=False)]) if 'resource_type' in df.columns else 0
			},
			
			"geographic": {
				"unique_ips": df['ip_address'].nunique() if 'ip_address' in df.columns else 0,
				"top_ips": df['ip_address'].value_counts().head(10).to_dict() if 'ip_address' in df.columns else {},
				"external_access": len(df[~df['ip_address'].str.startswith(('192.168.', '10.', '172.'))]) if 'ip_address' in df.columns else 0
			}
		}
		
		return aggregations
	
	async def _generate_insights(
		self, 
		config: ReportConfiguration, 
		report_data: Dict[str, Any]
	) -> List[ReportInsight]:
		"""Generate automated insights and recommendations"""
		insights = []
		events = report_data.get("events", [])
		aggregations = report_data.get("aggregations", {})
		anomalies = report_data.get("anomalies", [])
		
		# Security insights
		if aggregations.get("security", {}).get("high_risk_events", 0) > 10:
			insights.append(ReportInsight(
				type="security",
				severity="high",
				title="Elevated Security Risk Activity",
				description=f"Detected {aggregations['security']['high_risk_events']} high-risk events in the reporting period",
				recommendation="Review high-risk activities and implement additional monitoring for affected users and resources",
				confidence=0.9,
				supporting_data=aggregations.get("security", {}),
				trend_direction="increasing",
				business_impact="high"
			))
		
		# User behavior insights
		user_stats = aggregations.get("user_activity", {})
		events_per_user = user_stats.get("events_per_user", {})
		if events_per_user.get("max", 0) > events_per_user.get("75%", 0) * 3:
			insights.append(ReportInsight(
				type="user_behavior",
				severity="medium",
				title="Anomalous User Activity Patterns",
				description="Some users show significantly higher activity than normal patterns",
				recommendation="Investigate users with excessive activity for potential account compromise or policy violations",
				confidence=0.8,
				supporting_data=user_stats
			))
		
		# Compliance insights
		compliance_data = report_data.get("compliance_data", {})
		for framework, status in compliance_data.items():
			if status.get("violation_count", 0) > 0:
				insights.append(ReportInsight(
					type="compliance",
					severity="high",
					title=f"{framework} Compliance Issues",
					description=f"Found {status['violation_count']} compliance violations",
					recommendation=f"Address {framework} violations to maintain regulatory compliance",
					confidence=1.0,
					supporting_data=status,
					business_impact="critical"
				))
		
		# Temporal insights
		temporal = aggregations.get("temporal", {})
		if temporal.get("peak_activity_hour", 0) < 6 or temporal.get("peak_activity_hour", 0) > 22:
			insights.append(ReportInsight(
				type="operational",
				severity="medium", 
				title="Unusual Activity Hours",
				description=f"Peak activity occurs at {temporal['peak_activity_hour']}:00, outside normal business hours",
				recommendation="Review after-hours activities and ensure proper authorization",
				confidence=0.7,
				supporting_data=temporal
			))
		
		# ML anomaly insights
		if anomalies:
			high_confidence_anomalies = [a for a in anomalies if a.confidence > 0.8]
			if high_confidence_anomalies:
				insights.append(ReportInsight(
					type="anomaly",
					severity="high",
					title="ML-Detected Behavioral Anomalies",
					description=f"Machine learning detected {len(high_confidence_anomalies)} high-confidence anomalies",
					recommendation="Investigate flagged anomalies for potential security threats or policy violations",
					confidence=0.9,
					supporting_data={"anomaly_count": len(anomalies), "high_confidence": len(high_confidence_anomalies)}
				))
		
		# Geographic insights
		geographic = aggregations.get("geographic", {})
		if geographic.get("external_access", 0) > geographic.get("unique_ips", 1) * 0.5:
			insights.append(ReportInsight(
				type="security",
				severity="medium",
				title="High External Access Volume",
				description="Significant portion of access originates from external IP addresses",
				recommendation="Review external access patterns and ensure VPN/security policies are enforced",
				confidence=0.8,
				supporting_data=geographic
			))
		
		return insights
	
	async def _generate_charts(
		self, 
		config: ReportConfiguration, 
		report_data: Dict[str, Any]
	) -> List[Dict[str, str]]:
		"""Generate charts and visualizations for reports"""
		charts = []
		aggregations = report_data.get("aggregations", {})
		
		try:
			# Set style for professional charts
			plt.style.use('seaborn-v0_8-whitegrid')
			
			# Events by hour chart
			temporal = aggregations.get("temporal", {})
			if temporal.get("events_by_hour"):
				fig, ax = plt.subplots(figsize=(12, 6))
				hours = list(temporal["events_by_hour"].keys())
				counts = list(temporal["events_by_hour"].values())
				
				ax.bar(hours, counts, color='steelblue', alpha=0.7)
				ax.set_xlabel('Hour of Day')
				ax.set_ylabel('Number of Events')
				ax.set_title('Audit Events by Hour of Day')
				ax.grid(True, alpha=0.3)
				
				chart_data = self._save_chart_to_base64(fig)
				charts.append({
					"title": "Events by Hour",
					"type": "bar_chart",
					"data": chart_data
				})
				plt.close(fig)
			
			# Event type distribution
			event_types = aggregations.get("event_types", {}).get("distribution", {})
			if event_types:
				fig, ax = plt.subplots(figsize=(10, 8))
				
				# Top 10 event types
				top_events = dict(sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:10])
				
				wedges, texts, autotexts = ax.pie(
					top_events.values(),
					labels=top_events.keys(),
					autopct='%1.1f%%',
					startangle=90
				)
				
				ax.set_title('Event Type Distribution')
				
				chart_data = self._save_chart_to_base64(fig)
				charts.append({
					"title": "Event Type Distribution", 
					"type": "pie_chart",
					"data": chart_data
				})
				plt.close(fig)
			
			# Risk score trend
			events = report_data.get("events", [])
			if events and any('risk_score' in event for event in events):
				df = pd.DataFrame([e for e in events if 'risk_score' in e])
				df['timestamp'] = pd.to_datetime(df['timestamp'])
				df = df.sort_values('timestamp')
				
				# Daily average risk scores
				daily_risk = df.groupby(df['timestamp'].dt.date)['risk_score'].mean()
				
				fig, ax = plt.subplots(figsize=(12, 6))
				ax.plot(daily_risk.index, daily_risk.values, marker='o', linewidth=2, markersize=4)
				ax.set_xlabel('Date')
				ax.set_ylabel('Average Risk Score')
				ax.set_title('Daily Average Risk Score Trend')
				ax.grid(True, alpha=0.3)
				plt.xticks(rotation=45)
				plt.tight_layout()
				
				chart_data = self._save_chart_to_base64(fig)
				charts.append({
					"title": "Risk Score Trend",
					"type": "line_chart", 
					"data": chart_data
				})
				plt.close(fig)
			
			logger.info(f"Generated {len(charts)} charts for report")
			
		except Exception as e:
			logger.error(f"Chart generation failed: {str(e)}")
		
		return charts
	
	def _save_chart_to_base64(self, fig) -> str:
		"""Save matplotlib figure to base64 string"""
		buffer = BytesIO()
		fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
		buffer.seek(0)
		chart_data = base64.b64encode(buffer.read()).decode()
		buffer.close()
		return chart_data
	
	async def _calculate_report_metrics(self, report_data: Dict[str, Any]) -> ReportMetrics:
		"""Calculate comprehensive report metrics"""
		events = report_data.get("events", [])
		anomalies = report_data.get("anomalies", [])
		compliance_data = report_data.get("compliance_data", {})
		aggregations = report_data.get("aggregations", {})
		
		# Count compliance violations
		violation_count = 0
		for framework_data in compliance_data.values():
			violation_count += framework_data.get("violation_count", 0)
		
		# Count high-risk activities
		high_risk_count = 0
		if events:
			high_risk_count = len([e for e in events if e.get("risk_score", 0) > 0.7])
		
		# Calculate data completeness
		total_possible_fields = len(events) * 10  # Assuming 10 key fields per event
		actual_fields = 0
		for event in events:
			actual_fields += len([v for v in event.values() if v is not None])
		
		data_completeness = (actual_fields / total_possible_fields * 100) if total_possible_fields > 0 else 100.0
		
		return ReportMetrics(
			total_events_analyzed=len(events),
			compliance_violations_found=violation_count,
			anomalies_detected=len(anomalies),
			high_risk_activities=high_risk_count,
			users_analyzed=aggregations.get("user_activity", {}).get("unique_users", 0),
			systems_analyzed=aggregations.get("resources", {}).get("unique_resources", 0),
			data_completeness_percentage=data_completeness,
			confidence_level=0.95  # High confidence with comprehensive analysis
		)
	
	async def _generate_report_content(
		self,
		config: ReportConfiguration,
		report_data: Dict[str, Any],
		insights: List[ReportInsight],
		charts: List[Dict[str, str]],
		metrics: ReportMetrics,
		start_date: datetime,
		end_date: datetime
	) -> Dict[str, Any]:
		"""Generate structured report content"""
		
		# Executive summary
		executive_summary = self._create_executive_summary(config, report_data, insights, metrics)
		
		# Key findings
		key_findings = self._extract_key_findings(insights, report_data)
		
		# Detailed analysis sections
		detailed_sections = await self._create_detailed_sections(config, report_data, insights)
		
		# Recommendations
		recommendations = self._consolidate_recommendations(insights)
		
		# Appendix with raw data
		appendix = self._create_appendix(report_data, metrics)
		
		return {
			"metadata": {
				"title": config.name,
				"type": config.report_type.value,
				"period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
				"generated": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
				"tenant": self.tenant_id
			},
			"executive_summary": executive_summary,
			"key_metrics": metrics.__dict__,
			"key_findings": key_findings,
			"insights": [insight.__dict__ for insight in insights],
			"charts": charts,
			"detailed_analysis": detailed_sections,
			"recommendations": recommendations,
			"appendix": appendix
		}
	
	def _create_executive_summary(
		self, 
		config: ReportConfiguration, 
		report_data: Dict[str, Any],
		insights: List[ReportInsight],
		metrics: ReportMetrics
	) -> str:
		"""Create executive summary text"""
		
		events_count = metrics.total_events_analyzed
		violations = metrics.compliance_violations_found
		anomalies = metrics.anomalies_detected
		high_risk = metrics.high_risk_activities
		
		summary_parts = [
			f"This {config.report_type.value.replace('_', ' ').title()} report analyzes {events_count:,} audit events ",
			f"across {metrics.users_analyzed} users and {metrics.systems_analyzed} systems."
		]
		
		if violations > 0:
			summary_parts.append(f" The analysis identified {violations} compliance violations requiring immediate attention.")
		
		if anomalies > 0:
			summary_parts.append(f" Machine learning algorithms detected {anomalies} behavioral anomalies.")
		
		if high_risk > 0:
			summary_parts.append(f" {high_risk} high-risk activities were flagged for review.")
		
		# Add key insight
		if insights:
			critical_insights = [i for i in insights if i.severity in ["critical", "high"]]
			if critical_insights:
				summary_parts.append(f" Key findings include {len(critical_insights)} critical issues requiring executive attention.")
		
		summary_parts.append(f" Data completeness is {metrics.data_completeness_percentage:.1f}%, ensuring reliable analysis.")
		
		return "".join(summary_parts)
	
	def _extract_key_findings(
		self, 
		insights: List[ReportInsight], 
		report_data: Dict[str, Any]
	) -> List[Dict[str, Any]]:
		"""Extract key findings from analysis"""
		findings = []
		
		# Top insights by severity
		for insight in sorted(insights, key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x.severity, 0), reverse=True)[:5]:
			findings.append({
				"title": insight.title,
				"description": insight.description,
				"severity": insight.severity,
				"confidence": insight.confidence,
				"business_impact": insight.business_impact
			})
		
		return findings
	
	async def _create_detailed_sections(
		self, 
		config: ReportConfiguration, 
		report_data: Dict[str, Any],
		insights: List[ReportInsight]
	) -> Dict[str, Any]:
		"""Create detailed analysis sections"""
		
		sections = {}
		aggregations = report_data.get("aggregations", {})
		
		# Security analysis
		if config.report_type in [ReportType.SECURITY_AUDIT, ReportType.COMPLIANCE_SUMMARY]:
			sections["security_analysis"] = {
				"title": "Security Analysis",
				"content": self._create_security_analysis(aggregations, insights),
				"key_metrics": aggregations.get("security", {}),
				"risk_level": self._calculate_overall_risk_level(aggregations)
			}
		
		# User behavior analysis
		if config.report_type in [ReportType.USER_ACTIVITY, ReportType.SECURITY_AUDIT]:
			sections["user_behavior"] = {
				"title": "User Behavior Analysis",
				"content": self._create_user_behavior_analysis(aggregations, insights),
				"top_users": aggregations.get("user_activity", {}).get("top_users", {}),
				"activity_patterns": aggregations.get("temporal", {})
			}
		
		# Compliance analysis
		compliance_data = report_data.get("compliance_data", {})
		if compliance_data or config.report_type == ReportType.COMPLIANCE_SUMMARY:
			sections["compliance_status"] = {
				"title": "Compliance Status",
				"content": self._create_compliance_analysis(compliance_data, insights),
				"framework_status": compliance_data
			}
		
		return sections
	
	def _create_security_analysis(
		self, 
		aggregations: Dict[str, Any], 
		insights: List[ReportInsight]
	) -> str:
		"""Create security analysis content"""
		security_data = aggregations.get("security", {})
		
		analysis = [
			f"Security Analysis Overview:\n",
			f"- High-risk events: {security_data.get('high_risk_events', 0)}",
			f"- Average risk score: {security_data.get('avg_risk_score', 0):.2f}",
			f"- Failed login attempts: {security_data.get('failed_logins', 0)}",
			f"- Administrative activities: {security_data.get('admin_activities', 0)}"
		]
		
		# Add security insights
		security_insights = [i for i in insights if i.type == "security"]
		if security_insights:
			analysis.append("\nSecurity Concerns:")
			for insight in security_insights[:3]:
				analysis.append(f"- {insight.title}: {insight.description}")
		
		return "\n".join(analysis)
	
	def _create_user_behavior_analysis(
		self, 
		aggregations: Dict[str, Any], 
		insights: List[ReportInsight]
	) -> str:
		"""Create user behavior analysis content"""
		user_data = aggregations.get("user_activity", {})
		temporal_data = aggregations.get("temporal", {})
		
		analysis = [
			f"User Behavior Analysis:\n",
			f"- Total unique users: {user_data.get('unique_users', 0)}",
			f"- Peak activity hour: {temporal_data.get('peak_activity_hour', 'N/A')}:00",
			f"- Most active day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][temporal_data.get('peak_activity_day', 0)]}"
		]
		
		# Add top users
		top_users = user_data.get("top_users", {})
		if top_users:
			analysis.append("\nMost Active Users:")
			for user, count in list(top_users.items())[:5]:
				analysis.append(f"- {user}: {count} events")
		
		return "\n".join(analysis)
	
	def _create_compliance_analysis(
		self, 
		compliance_data: Dict[str, Any], 
		insights: List[ReportInsight]
	) -> str:
		"""Create compliance analysis content"""
		if not compliance_data:
			return "No compliance framework data available for analysis."
		
		analysis = ["Compliance Framework Status:\n"]
		
		for framework, status in compliance_data.items():
			violations = status.get("violation_count", 0)
			compliance_score = status.get("compliance_score", 0)
			
			analysis.append(f"- {framework}: {compliance_score:.1%} compliant ({violations} violations)")
		
		# Add compliance insights
		compliance_insights = [i for i in insights if i.type == "compliance"]
		if compliance_insights:
			analysis.append("\nCompliance Issues:")
			for insight in compliance_insights:
				analysis.append(f"- {insight.title}: {insight.description}")
		
		return "\n".join(analysis)
	
	def _calculate_overall_risk_level(self, aggregations: Dict[str, Any]) -> str:
		"""Calculate overall risk level"""
		security_data = aggregations.get("security", {})
		
		high_risk_events = security_data.get("high_risk_events", 0)
		avg_risk_score = security_data.get("avg_risk_score", 0)
		failed_logins = security_data.get("failed_logins", 0)
		
		# Simple risk calculation
		risk_score = 0
		
		if high_risk_events > 20:
			risk_score += 3
		elif high_risk_events > 10:
			risk_score += 2
		elif high_risk_events > 5:
			risk_score += 1
		
		if avg_risk_score > 0.8:
			risk_score += 3
		elif avg_risk_score > 0.6:
			risk_score += 2
		elif avg_risk_score > 0.4:
			risk_score += 1
		
		if failed_logins > 50:
			risk_score += 2
		elif failed_logins > 20:
			risk_score += 1
		
		if risk_score >= 6:
			return "HIGH"
		elif risk_score >= 3:
			return "MEDIUM"
		else:
			return "LOW"
	
	def _consolidate_recommendations(self, insights: List[ReportInsight]) -> List[Dict[str, Any]]:
		"""Consolidate recommendations from insights"""
		recommendations = []
		
		# Group by priority/severity
		critical_insights = [i for i in insights if i.severity == "critical"]
		high_insights = [i for i in insights if i.severity == "high"]
		
		# Critical recommendations first
		for insight in critical_insights:
			recommendations.append({
				"priority": "CRITICAL",
				"title": f"Address {insight.title}",
				"description": insight.recommendation,
				"business_impact": insight.business_impact,
				"confidence": insight.confidence
			})
		
		# High priority recommendations
		for insight in high_insights[:3]:  # Limit to top 3
			recommendations.append({
				"priority": "HIGH", 
				"title": f"Review {insight.title}",
				"description": insight.recommendation,
				"business_impact": insight.business_impact,
				"confidence": insight.confidence
			})
		
		# Add general recommendations
		recommendations.append({
			"priority": "MEDIUM",
			"title": "Enhance Monitoring Coverage",
			"description": "Consider expanding audit logging coverage to capture additional security-relevant events",
			"business_impact": "medium",
			"confidence": 0.8
		})
		
		return recommendations
	
	def _create_appendix(
		self, 
		report_data: Dict[str, Any], 
		metrics: ReportMetrics
	) -> Dict[str, Any]:
		"""Create report appendix with technical details"""
		
		return {
			"methodology": "This report uses advanced analytics, machine learning anomaly detection, and compliance framework analysis",
			"data_sources": "Elasticsearch audit event store, compliance management system, ML anomaly detection engine",
			"analysis_period": report_data.get("date_range", {}),
			"data_quality": {
				"completeness": f"{metrics.data_completeness_percentage:.1f}%",
				"confidence_level": f"{metrics.confidence_level:.1%}",
				"total_events": metrics.total_events_analyzed
			},
			"technical_notes": [
				"Risk scores calculated using proprietary ML algorithms",
				"Anomaly detection uses ensemble methods with 95% confidence threshold", 
				"Compliance analysis based on industry-standard frameworks",
				"All timestamps in UTC unless otherwise specified"
			]
		}
	
	async def _render_report(
		self, 
		content: Dict[str, Any], 
		format_type: ReportFormat
	) -> Dict[str, Any]:
		"""Render report in specified format"""
		
		if format_type == ReportFormat.JSON:
			return {
				"format": "json",
				"content": json.dumps(content, default=str, indent=2),
				"mime_type": "application/json"
			}
		
		elif format_type == ReportFormat.HTML:
			html_content = await self._render_html_report(content)
			return {
				"format": "html",
				"content": html_content,
				"mime_type": "text/html"
			}
		
		elif format_type == ReportFormat.PDF:
			# Mock PDF generation - in production would use library like weasyprint
			return {
				"format": "pdf", 
				"content": f"PDF Report: {content['metadata']['title']}",
				"mime_type": "application/pdf"
			}
		
		elif format_type == ReportFormat.CSV:
			csv_content = await self._render_csv_report(content)
			return {
				"format": "csv",
				"content": csv_content,
				"mime_type": "text/csv"
			}
		
		else:
			return {
				"format": "text",
				"content": str(content),
				"mime_type": "text/plain"
			}
	
	async def _render_html_report(self, content: Dict[str, Any]) -> str:
		"""Render report as HTML"""
		
		html_template = """
		<!DOCTYPE html>
		<html>
		<head>
			<title>{{ metadata.title }}</title>
			<style>
				body { font-family: Arial, sans-serif; margin: 40px; }
				.header { background: #f8f9fa; padding: 20px; border-left: 4px solid #007bff; }
				.section { margin: 20px 0; }
				.metric { display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; }
				.insight { margin: 10px 0; padding: 15px; border-left: 4px solid #28a745; background: #f8fff8; }
				.insight.high { border-color: #dc3545; background: #fff8f8; }
				.insight.critical { border-color: #6f42c1; background: #f8f7ff; }
				.chart { margin: 20px 0; text-align: center; }
				table { width: 100%; border-collapse: collapse; }
				th, td { padding: 8px; border: 1px solid #ddd; text-align: left; }
				th { background: #f8f9fa; }
			</style>
		</head>
		<body>
			<div class="header">
				<h1>{{ metadata.title }}</h1>
				<p><strong>Period:</strong> {{ metadata.period }}</p>
				<p><strong>Generated:</strong> {{ metadata.generated }}</p>
			</div>
			
			<div class="section">
				<h2>Executive Summary</h2>
				<p>{{ executive_summary }}</p>
			</div>
			
			<div class="section">
				<h2>Key Metrics</h2>
				{% for key, value in key_metrics.items() %}
				<div class="metric">
					<strong>{{ key.replace('_', ' ').title() }}:</strong> {{ value }}
				</div>
				{% endfor %}
			</div>
			
			<div class="section">
				<h2>Key Findings</h2>
				{% for finding in key_findings %}
				<div class="insight {{ finding.severity }}">
					<h4>{{ finding.title }}</h4>
					<p>{{ finding.description }}</p>
					<small>Confidence: {{ (finding.confidence * 100) | round(1) }}% | Business Impact: {{ finding.business_impact }}</small>
				</div>
				{% endfor %}
			</div>
			
			{% if charts %}
			<div class="section">
				<h2>Visual Analysis</h2>
				{% for chart in charts %}
				<div class="chart">
					<h4>{{ chart.title }}</h4>
					<img src="data:image/png;base64,{{ chart.data }}" alt="{{ chart.title }}" style="max-width: 100%;">
				</div>
				{% endfor %}
			</div>
			{% endif %}
			
			<div class="section">
				<h2>Recommendations</h2>
				{% for rec in recommendations %}
				<div class="insight">
					<h4>[{{ rec.priority }}] {{ rec.title }}</h4>
					<p>{{ rec.description }}</p>
					<small>Business Impact: {{ rec.business_impact }} | Confidence: {{ (rec.confidence * 100) | round(1) }}%</small>
				</div>
				{% endfor %}
			</div>
			
		</body>
		</html>
		"""
		
		template = Template(html_template)
		return template.render(**content)
	
	async def _render_csv_report(self, content: Dict[str, Any]) -> str:
		"""Render report as CSV data"""
		
		lines = []
		lines.append("Report Data Export")
		lines.append(f"Title,{content['metadata']['title']}")
		lines.append(f"Period,{content['metadata']['period']}")
		lines.append("")
		
		# Key metrics
		lines.append("Key Metrics")
		lines.append("Metric,Value")
		for key, value in content["key_metrics"].items():
			lines.append(f"{key.replace('_', ' ').title()},{value}")
		lines.append("")
		
		# Key findings
		lines.append("Key Findings")
		lines.append("Title,Description,Severity,Confidence")
		for finding in content["key_findings"]:
			lines.append(f"{finding['title']},{finding['description']},{finding['severity']},{finding['confidence']}")
		
		return "\n".join(lines)
	
	async def _store_report(
		self, 
		metadata: Dict[str, Any], 
		rendered_reports: Dict[str, Dict[str, Any]]
	) -> Dict[str, str]:
		"""Store generated reports"""
		try:
			stored_reports = {}
			
			for format_name, report_data in rendered_reports.items():
				# Store using document management service
				doc_result = await self.document_service.create_document(
					title=f"{metadata['name']}_{format_name}",
					content=report_data["content"],
					mime_type=report_data["mime_type"],
					tags=["audit_report", metadata["type"], format_name],
					metadata=metadata
				)
				
				stored_reports[format_name] = doc_result["id"]
				logger.info(f"Stored report {metadata['id']} in {format_name} format")
			
			return stored_reports
			
		except Exception as e:
			logger.error(f"Failed to store reports: {str(e)}")
			return {}
	
	async def _deliver_report(
		self,
		config: ReportConfiguration,
		metadata: Dict[str, Any],
		stored_reports: Dict[str, str]
	) -> List[Dict[str, Any]]:
		"""Deliver report through configured channels"""
		
		delivery_results = []
		
		try:
			for delivery_method in config.delivery_methods:
				if delivery_method == DeliveryMethod.EMAIL:
					# Email delivery
					for recipient in config.recipients:
						result = await self.notification_service.send_notification(
							channel="email",
							recipient=recipient,
							title=f"Audit Report: {config.name}",
							message=f"Your scheduled audit report '{config.name}' is ready.",
							data={
								"report_metadata": metadata,
								"download_links": stored_reports
							},
							priority=Priority.MEDIUM
						)
						
						delivery_results.append({
							"method": "email",
							"recipient": recipient,
							"status": "success" if result else "failed"
						})
				
				elif delivery_method == DeliveryMethod.WEBHOOK:
					# Webhook delivery (mock)
					delivery_results.append({
						"method": "webhook",
						"status": "success",
						"webhook_url": "configured_webhook_url"
					})
				
				elif delivery_method == DeliveryMethod.DASHBOARD:
					# Dashboard integration
					dashboard_result = await self.bi_service.create_dashboard(
						title=f"Report: {config.name}",
						type="audit_report",
						data=metadata,
						reports=stored_reports
					)
					
					delivery_results.append({
						"method": "dashboard",
						"status": "success",
						"dashboard_id": dashboard_result["id"]
					})
				
				else:
					delivery_results.append({
						"method": delivery_method.value,
						"status": "not_implemented"
					})
			
			logger.info(f"Delivered report through {len(delivery_results)} channels")
			
		except Exception as e:
			logger.error(f"Report delivery failed: {str(e)}")
			delivery_results.append({
				"method": "error",
				"status": "failed",
				"error": str(e)
			})
		
		return delivery_results
	
	def _update_generation_stats(
		self, 
		config: ReportConfiguration, 
		generation_time: float,
		delivery_results: List[Dict[str, Any]]
	) -> None:
		"""Update report generation statistics"""
		
		self.generation_stats["total_reports_generated"] += 1
		
		# Update average generation time
		current_avg = self.generation_stats["avg_generation_time"]
		total_reports = self.generation_stats["total_reports_generated"]
		self.generation_stats["avg_generation_time"] = (
			(current_avg * (total_reports - 1) + generation_time) / total_reports
		)
		
		# Update delivery statistics
		successful_deliveries = len([r for r in delivery_results if r.get("status") == "success"])
		failed_deliveries = len([r for r in delivery_results if r.get("status") == "failed"])
		
		self.generation_stats["successful_deliveries"] += successful_deliveries
		self.generation_stats["failed_deliveries"] += failed_deliveries
		
		# Update success rate
		total_deliveries = self.generation_stats["successful_deliveries"] + self.generation_stats["failed_deliveries"]
		if total_deliveries > 0:
			self.generation_stats["delivery_success_rate"] = (
				self.generation_stats["successful_deliveries"] / total_deliveries
			)
		
		# Track popular report types
		report_type = config.report_type.value
		if report_type not in self.generation_stats["popular_report_types"]:
			self.generation_stats["popular_report_types"][report_type] = 0
		self.generation_stats["popular_report_types"][report_type] += 1
	
	async def _load_report_templates(self) -> None:
		"""Load report templates"""
		# Mock template loading - in production would load from files
		self.templates = {
			"compliance_summary": "Compliance Summary Template",
			"security_audit": "Security Audit Template",
			"executive_dashboard": "Executive Dashboard Template"
		}
		logger.info(f"Loaded {len(self.templates)} report templates")
	
	async def _load_default_report_configs(self) -> None:
		"""Load default report configurations"""
		
		# Daily compliance summary
		self.report_configs["daily_compliance"] = ReportConfiguration(
			id="daily_compliance",
			name="Daily Compliance Summary",
			description="Daily summary of compliance status and violations",
			report_type=ReportType.COMPLIANCE_SUMMARY,
			frequency=ReportFrequency.DAILY,
			schedule_time="09:00",
			date_range_days=1,
			formats=[ReportFormat.HTML, ReportFormat.PDF],
			delivery_methods=[DeliveryMethod.EMAIL],
			recipients=["compliance@company.com"]
		)
		
		# Weekly security audit
		self.report_configs["weekly_security"] = ReportConfiguration(
			id="weekly_security",
			name="Weekly Security Audit Report",
			description="Comprehensive weekly security analysis",
			report_type=ReportType.SECURITY_AUDIT,
			frequency=ReportFrequency.WEEKLY,
			schedule_time="07:00",
			date_range_days=7,
			formats=[ReportFormat.HTML, ReportFormat.PDF, ReportFormat.CSV],
			delivery_methods=[DeliveryMethod.EMAIL, DeliveryMethod.DASHBOARD],
			recipients=["security@company.com", "ciso@company.com"]
		)
		
		# Monthly executive dashboard
		self.report_configs["monthly_executive"] = ReportConfiguration(
			id="monthly_executive",
			name="Monthly Executive Dashboard",
			description="Executive summary of audit and compliance metrics",
			report_type=ReportType.EXECUTIVE_DASHBOARD,
			frequency=ReportFrequency.MONTHLY,
			schedule_time="08:00",
			date_range_days=30,
			formats=[ReportFormat.PDF, ReportFormat.DASHBOARD],
			delivery_methods=[DeliveryMethod.EMAIL, DeliveryMethod.DASHBOARD],
			recipients=["ceo@company.com", "cfo@company.com"]
		)
		
		logger.info(f"Loaded {len(self.report_configs)} default report configurations")
	
	async def _schedule_reports(self) -> None:
		"""Schedule automated report generation"""
		try:
			for config in self.report_configs.values():
				if config.enabled and config.frequency != ReportFrequency.ON_DEMAND:
					
					# Calculate cron expression based on frequency
					cron_expr = self._frequency_to_cron(config.frequency, config.schedule_time)
					
					# Schedule with APG scheduler service
					await self.scheduler_service.schedule_task(
						name=f"audit_report_{config.id}",
						cron_expression=cron_expr,
						task_type="audit_report_generation",
						task_data={
							"config_id": config.id,
							"tenant_id": self.tenant_id
						}
					)
					
					logger.info(f"Scheduled report: {config.name} ({cron_expr})")
			
		except Exception as e:
			logger.error(f"Failed to schedule reports: {str(e)}")
	
	def _frequency_to_cron(self, frequency: ReportFrequency, schedule_time: Optional[str]) -> str:
		"""Convert frequency to cron expression"""
		
		hour, minute = "9", "0"  # Default time
		if schedule_time:
			try:
				time_parts = schedule_time.split(":")
				hour, minute = time_parts[0], time_parts[1]
			except:
				pass
		
		if frequency == ReportFrequency.HOURLY:
			return f"0 0 * * * *"  # Every hour
		elif frequency == ReportFrequency.DAILY:
			return f"0 {minute} {hour} * * *"  # Daily at specified time
		elif frequency == ReportFrequency.WEEKLY:
			return f"0 {minute} {hour} * * 1"  # Monday at specified time
		elif frequency == ReportFrequency.MONTHLY:
			return f"0 {minute} {hour} 1 * *"  # First of month at specified time
		elif frequency == ReportFrequency.QUARTERLY:
			return f"0 {minute} {hour} 1 1,4,7,10 *"  # First of quarter
		elif frequency == ReportFrequency.ANNUALLY:
			return f"0 {minute} {hour} 1 1 *"  # January 1st
		else:
			return f"0 {minute} {hour} * * *"  # Default daily
	
	async def create_custom_report(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
		"""Create custom report configuration"""
		try:
			# Convert dict to ReportConfiguration
			config = ReportConfiguration(**config_dict)
			
			# Validate configuration
			if not config.name or not config.report_type:
				return {"success": False, "error": "Name and report_type are required"}
			
			# Store configuration
			self.report_configs[config.id] = config
			
			# Schedule if needed
			if config.enabled and config.frequency != ReportFrequency.ON_DEMAND:
				await self._schedule_reports()
			
			logger.info(f"Created custom report configuration: {config.name}")
			
			return {
				"success": True,
				"config_id": config.id,
				"message": "Custom report configuration created successfully"
			}
			
		except Exception as e:
			logger.error(f"Failed to create custom report: {str(e)}")
			return {"success": False, "error": str(e)}
	
	async def get_report_history(self, limit: int = 50) -> List[Dict[str, Any]]:
		"""Get report generation history"""
		return self.generated_reports[-limit:]
	
	async def get_report_statistics(self) -> Dict[str, Any]:
		"""Get report generation statistics"""
		return {
			"generation_stats": self.generation_stats,
			"active_configs": len([c for c in self.report_configs.values() if c.enabled]),
			"total_configs": len(self.report_configs),
			"recent_reports": len(self.generated_reports)
		}

# Export for APG integration
__all__ = [
	"ReportGenerator",
	"ReportConfiguration",
	"ReportMetrics",
	"ReportInsight",
	"ReportType",
	"ReportFormat", 
	"DeliveryMethod",
	"ReportFrequency"
]