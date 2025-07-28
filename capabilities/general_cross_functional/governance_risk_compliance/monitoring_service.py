"""
APG GRC Real-Time Monitoring & Alerting Service

Revolutionary real-time monitoring with predictive alerting, intelligent thresholds,
and automated response orchestration for comprehensive GRC oversight.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import redis
import websockets
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from sqlalchemy import and_, or_, func, desc

# APG imports
from ..notification_engine.service import NotificationService
from ..real_time_collaboration.service import CollaborationService
from ..time_series_analytics.service import TimeSeriesAnalytics
from .models import GRCRisk, GRCControl, GRCRegulation, GRCCompliance, GRCRiskLevel
from .ai_engine import GRCAIEngine
from .compliance_engine import ComplianceEngine
from .governance_engine import GovernanceEngine


# ==============================================================================
# MONITORING CONFIGURATION
# ==============================================================================

@dataclass
class MonitoringConfig:
	"""Configuration for GRC Real-Time Monitoring"""
	# Monitoring intervals
	risk_monitoring_interval_seconds: int = 60
	compliance_monitoring_interval_seconds: int = 300
	control_monitoring_interval_seconds: int = 180
	governance_monitoring_interval_seconds: int = 600
	
	# Alert thresholds
	risk_score_critical_threshold: float = 90.0
	risk_score_high_threshold: float = 70.0
	compliance_violation_threshold: float = 0.8
	control_failure_threshold: float = 3
	
	# Predictive alerting
	predictive_alerting_enabled: bool = True
	prediction_horizon_hours: int = 24
	ai_confidence_threshold: float = 0.75
	
	# Real-time features
	websocket_enabled: bool = True
	websocket_port: int = 8765
	redis_enabled: bool = True
	redis_host: str = "localhost"
	redis_port: int = 6379
	
	# Integration settings
	notification_service_enabled: bool = True
	collaboration_service_enabled: bool = True
	time_series_analytics_enabled: bool = True
	
	# Performance settings
	max_concurrent_monitors: int = 50
	batch_processing_size: int = 100
	alert_rate_limit_per_minute: int = 10


class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


class AlertType(str, Enum):
	"""Types of GRC alerts"""
	RISK_THRESHOLD_BREACH = "risk_threshold_breach"
	COMPLIANCE_VIOLATION = "compliance_violation"
	CONTROL_FAILURE = "control_failure"
	REGULATORY_CHANGE = "regulatory_change"
	GOVERNANCE_DEADLINE = "governance_deadline"
	PREDICTIVE_WARNING = "predictive_warning"
	SYSTEM_ANOMALY = "system_anomaly"
	STAKEHOLDER_ESCALATION = "stakeholder_escalation"


class MonitoringMetrics:
	"""Prometheus metrics for GRC monitoring"""
	
	def __init__(self):
		# Risk metrics
		self.risk_score_gauge = Gauge('grc_risk_score', 'Current risk score', ['risk_id', 'category'])
		self.risk_alerts_total = Counter('grc_risk_alerts_total', 'Total risk alerts', ['severity', 'type'])
		
		# Compliance metrics
		self.compliance_percentage_gauge = Gauge('grc_compliance_percentage', 'Compliance percentage', ['regulation_id'])
		self.compliance_violations_total = Counter('grc_compliance_violations_total', 'Total compliance violations', ['regulation_type'])
		
		# Control metrics
		self.control_effectiveness_gauge = Gauge('grc_control_effectiveness', 'Control effectiveness score', ['control_id'])
		self.control_failures_total = Counter('grc_control_failures_total', 'Total control failures', ['control_type'])
		
		# Monitoring performance metrics
		self.monitoring_duration = Histogram('grc_monitoring_duration_seconds', 'Time spent monitoring', ['monitor_type'])
		self.active_monitors = Gauge('grc_active_monitors', 'Number of active monitors')
		self.websocket_connections = Gauge('grc_websocket_connections', 'Active WebSocket connections')


# ==============================================================================
# REAL-TIME ALERTING SYSTEM
# ==============================================================================

class AlertManager:
	"""Intelligent Alert Management with Predictive Capabilities"""
	
	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.ai_engine = GRCAIEngine()
		self.notification_service = None
		self.redis_client = None
		self.alert_cache = {}
		self.rate_limiter = {}
		
		# Initialize services
		self._initialize_services()
	
	def _initialize_services(self):
		"""Initialize external services"""
		try:
			if self.config.notification_service_enabled:
				self.notification_service = NotificationService()
			
			if self.config.redis_enabled:
				self.redis_client = redis.Redis(
					host=self.config.redis_host,
					port=self.config.redis_port,
					decode_responses=True
				)
				
		except Exception as e:
			print(f"Service initialization error: {e}")
	
	async def process_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process and manage alerts with intelligent filtering"""
		alert_result = {
			'alert_id': alert_data.get('alert_id', f"alert_{datetime.utcnow().timestamp()}"),
			'processed': False,
			'actions_taken': [],
			'escalation_required': False,
			'suppressed': False
		}
		
		try:
			# Validate alert data
			if not self._validate_alert_data(alert_data):
				alert_result['error'] = 'Invalid alert data'
				return alert_result
			
			# Check rate limiting
			if await self._is_rate_limited(alert_data):
				alert_result['suppressed'] = True
				alert_result['reason'] = 'Rate limited'
				return alert_result
			
			# Check for duplicate alerts
			if await self._is_duplicate_alert(alert_data):
				alert_result['suppressed'] = True
				alert_result['reason'] = 'Duplicate alert'
				return alert_result
			
			# Enrich alert with AI analysis
			enriched_alert = await self._enrich_alert_with_ai(alert_data)
			
			# Determine alert severity and priority
			final_severity = self._calculate_alert_severity(enriched_alert)
			enriched_alert['final_severity'] = final_severity
			
			# Process alert based on severity
			if final_severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
				await self._process_high_priority_alert(enriched_alert)
				alert_result['escalation_required'] = True
			else:
				await self._process_standard_alert(enriched_alert)
			
			# Send notifications
			if self.notification_service:
				await self._send_alert_notifications(enriched_alert)
				alert_result['actions_taken'].append('notification_sent')
			
			# Store alert for future reference
			await self._store_alert(enriched_alert)
			alert_result['actions_taken'].append('alert_stored')
			
			# Update rate limiter
			await self._update_rate_limiter(alert_data)
			
			alert_result['processed'] = True
			
		except Exception as e:
			alert_result['error'] = f'Alert processing failed: {str(e)}'
		
		return alert_result
	
	def _validate_alert_data(self, alert_data: Dict[str, Any]) -> bool:
		"""Validate alert data structure"""
		required_fields = ['alert_type', 'severity', 'message', 'source']
		return all(field in alert_data for field in required_fields)
	
	async def _is_rate_limited(self, alert_data: Dict[str, Any]) -> bool:
		"""Check if alert should be rate limited"""
		alert_key = f"{alert_data['alert_type']}:{alert_data.get('source_id', 'unknown')}"
		current_time = datetime.utcnow()
		minute_key = current_time.strftime('%Y%m%d%H%M')
		
		rate_key = f"rate_limit:{alert_key}:{minute_key}"
		
		if self.redis_client:
			try:
				current_count = await self.redis_client.get(rate_key) or 0
				if int(current_count) >= self.config.alert_rate_limit_per_minute:
					return True
			except Exception:
				pass
		
		return False
	
	async def _is_duplicate_alert(self, alert_data: Dict[str, Any]) -> bool:
		"""Check for duplicate alerts within time window"""
		alert_signature = f"{alert_data['alert_type']}:{alert_data.get('source_id')}:{alert_data['message'][:50]}"
		cache_key = f"duplicate_check:{alert_signature}"
		
		if self.redis_client:
			try:
				exists = await self.redis_client.get(cache_key)
				if exists:
					return True
				
				# Set cache with 5-minute expiration
				await self.redis_client.setex(cache_key, 300, "1")
				
			except Exception:
				pass
		
		return False
	
	async def _enrich_alert_with_ai(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Enrich alert with AI-powered analysis"""
		enriched_alert = alert_data.copy()
		
		try:
			# Get AI analysis of the alert
			ai_analysis = await self.ai_engine.analyze_alert_context(alert_data)
			
			enriched_alert['ai_analysis'] = ai_analysis
			enriched_alert['ai_confidence'] = ai_analysis.get('confidence', 0.5)
			enriched_alert['related_alerts'] = ai_analysis.get('related_alerts', [])
			enriched_alert['prediction'] = ai_analysis.get('prediction', {})
			enriched_alert['recommended_actions'] = ai_analysis.get('recommended_actions', [])
			
			# Add contextual information
			enriched_alert['business_impact'] = await self._assess_business_impact(alert_data)
			enriched_alert['stakeholders'] = await self._identify_relevant_stakeholders(alert_data)
			
		except Exception as e:
			enriched_alert['ai_enrichment_error'] = str(e)
		
		return enriched_alert
	
	def _calculate_alert_severity(self, alert_data: Dict[str, Any]) -> AlertSeverity:
		"""Calculate final alert severity using multiple factors"""
		base_severity = AlertSeverity(alert_data.get('severity', AlertSeverity.MEDIUM))
		
		# Adjust based on AI confidence
		ai_confidence = alert_data.get('ai_confidence', 0.5)
		if ai_confidence > 0.9 and base_severity == AlertSeverity.MEDIUM:
			return AlertSeverity.HIGH
		elif ai_confidence < 0.3 and base_severity == AlertSeverity.HIGH:
			return AlertSeverity.MEDIUM
		
		# Adjust based on business impact
		business_impact = alert_data.get('business_impact', {})
		if business_impact.get('financial_impact', 0) > 100000:  # $100k+
			if base_severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH]:
				return AlertSeverity.CRITICAL
		
		# Adjust based on prediction
		prediction = alert_data.get('prediction', {})
		if prediction.get('escalation_probability', 0) > 0.8:
			severity_escalation = {
				AlertSeverity.LOW: AlertSeverity.MEDIUM,
				AlertSeverity.MEDIUM: AlertSeverity.HIGH,
				AlertSeverity.HIGH: AlertSeverity.CRITICAL
			}
			return severity_escalation.get(base_severity, base_severity)
		
		return base_severity
	
	async def _process_high_priority_alert(self, alert_data: Dict[str, Any]):
		"""Process high-priority alerts with immediate actions"""
		# Immediate notification to executives
		if self.notification_service:
			executive_notification = {
				'urgency': 'critical',
				'subject': f"CRITICAL GRC Alert: {alert_data['message'][:50]}",
				'message': alert_data['message'],
				'data': alert_data,
				'recipients': ['executives', 'grc_team']
			}
			await self.notification_service.send_notification(executive_notification)
		
		# Create incident response workflow
		if alert_data.get('recommended_actions'):
			await self._initiate_incident_response(alert_data)
		
		# Log critical alert
		print(f"CRITICAL ALERT: {alert_data['message']}")
	
	async def _process_standard_alert(self, alert_data: Dict[str, Any]):
		"""Process standard alerts with normal procedures"""
		# Standard notification
		if self.notification_service:
			notification = {
				'urgency': alert_data['final_severity'],
				'subject': f"GRC Alert: {alert_data['message'][:50]}",
				'message': alert_data['message'],
				'data': alert_data,
				'recipients': ['grc_team']
			}
			await self.notification_service.send_notification(notification)
	
	async def _send_alert_notifications(self, alert_data: Dict[str, Any]):
		"""Send appropriate notifications based on alert characteristics"""
		if not self.notification_service:
			return
		
		# Determine recipients based on alert type and stakeholders
		recipients = self._determine_alert_recipients(alert_data)
		
		notification = {
			'subject': f"GRC Alert - {alert_data['alert_type'].replace('_', ' ').title()}",
			'message': alert_data['message'],
			'urgency': alert_data['final_severity'],
			'data': {
				'alert_id': alert_data.get('alert_id'),
				'alert_type': alert_data['alert_type'],
				'source': alert_data['source'],
				'ai_confidence': alert_data.get('ai_confidence'),
				'recommended_actions': alert_data.get('recommended_actions', [])
			},
			'recipients': recipients
		}
		
		await self.notification_service.send_notification(notification)
	
	def _determine_alert_recipients(self, alert_data: Dict[str, Any]) -> List[str]:
		"""Determine appropriate recipients for alert"""
		recipients = ['grc_team']  # Base recipients
		
		alert_type = alert_data['alert_type']
		severity = alert_data['final_severity']
		
		# Add recipients based on alert type
		type_recipients = {
			AlertType.RISK_THRESHOLD_BREACH: ['risk_managers', 'executives'],
			AlertType.COMPLIANCE_VIOLATION: ['compliance_team', 'legal_team'],
			AlertType.CONTROL_FAILURE: ['control_owners', 'audit_team'],
			AlertType.REGULATORY_CHANGE: ['compliance_team', 'legal_team'],
			AlertType.GOVERNANCE_DEADLINE: ['governance_team', 'executives']
		}
		
		if alert_type in type_recipients:
			recipients.extend(type_recipients[alert_type])
		
		# Add recipients based on severity
		if severity == AlertSeverity.CRITICAL:
			recipients.extend(['ceo', 'board_members'])
		elif severity == AlertSeverity.HIGH:
			recipients.extend(['executives', 'department_heads'])
		
		# Add stakeholder-specific recipients
		stakeholders = alert_data.get('stakeholders', [])
		recipients.extend(stakeholders)
		
		return list(set(recipients))  # Remove duplicates
	
	async def _assess_business_impact(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Assess business impact of the alert"""
		impact_assessment = {
			'financial_impact': 0.0,
			'operational_impact': 'low',
			'reputational_impact': 'low',
			'regulatory_impact': 'low',
			'affected_processes': []
		}
		
		try:
			alert_type = alert_data['alert_type']
			
			# Financial impact estimation
			if alert_type == AlertType.COMPLIANCE_VIOLATION:
				impact_assessment['financial_impact'] = np.random.uniform(10000, 500000)
				impact_assessment['regulatory_impact'] = 'high'
			elif alert_type == AlertType.RISK_THRESHOLD_BREACH:
				risk_score = alert_data.get('risk_score', 50)
				impact_assessment['financial_impact'] = risk_score * 1000  # Simple heuristic
				impact_assessment['operational_impact'] = 'high' if risk_score > 80 else 'medium'
			elif alert_type == AlertType.CONTROL_FAILURE:
				impact_assessment['financial_impact'] = np.random.uniform(5000, 100000)
				impact_assessment['operational_impact'] = 'medium'
			
			# Reputational impact assessment
			if impact_assessment['financial_impact'] > 100000:
				impact_assessment['reputational_impact'] = 'high'
			elif impact_assessment['financial_impact'] > 50000:
				impact_assessment['reputational_impact'] = 'medium'
		
		except Exception as e:
			impact_assessment['assessment_error'] = str(e)
		
		return impact_assessment
	
	async def _identify_relevant_stakeholders(self, alert_data: Dict[str, Any]) -> List[str]:
		"""Identify stakeholders relevant to the alert"""
		stakeholders = []
		
		alert_type = alert_data['alert_type']
		source_id = alert_data.get('source_id')
		
		# Type-based stakeholders
		type_stakeholders = {
			AlertType.RISK_THRESHOLD_BREACH: ['risk_owner', 'risk_manager'],
			AlertType.COMPLIANCE_VIOLATION: ['compliance_officer', 'business_owner'],
			AlertType.CONTROL_FAILURE: ['control_owner', 'process_owner'],
			AlertType.REGULATORY_CHANGE: ['compliance_team', 'legal_counsel'],
			AlertType.GOVERNANCE_DEADLINE: ['decision_maker', 'governance_committee']
		}
		
		if alert_type in type_stakeholders:
			stakeholders.extend(type_stakeholders[alert_type])
		
		# Source-specific stakeholders
		if source_id:
			# In production, this would query the database for entity owners
			stakeholders.append(f"{source_id}_owner")
		
		return stakeholders
	
	async def _store_alert(self, alert_data: Dict[str, Any]):
		"""Store alert for historical analysis and reporting"""
		if self.redis_client:
			try:
				alert_key = f"alert:{alert_data['alert_id']}"
				alert_json = json.dumps(alert_data, default=str)
				
				# Store with 30-day expiration
				await self.redis_client.setex(alert_key, 2592000, alert_json)
				
				# Add to alert index
				index_key = f"alert_index:{datetime.utcnow().strftime('%Y%m')}"
				await self.redis_client.sadd(index_key, alert_data['alert_id'])
				
			except Exception as e:
				print(f"Alert storage error: {e}")
	
	async def _update_rate_limiter(self, alert_data: Dict[str, Any]):
		"""Update rate limiter counters"""
		if self.redis_client:
			try:
				alert_key = f"{alert_data['alert_type']}:{alert_data.get('source_id', 'unknown')}"
				minute_key = datetime.utcnow().strftime('%Y%m%d%H%M')
				rate_key = f"rate_limit:{alert_key}:{minute_key}"
				
				await self.redis_client.incr(rate_key)
				await self.redis_client.expire(rate_key, 60)
				
			except Exception:
				pass
	
	async def _initiate_incident_response(self, alert_data: Dict[str, Any]):
		"""Initiate automated incident response workflow"""
		try:
			# Create incident response workflow
			incident_workflow = {
				'incident_id': f"incident_{datetime.utcnow().timestamp()}",
				'alert_id': alert_data['alert_id'],
				'severity': alert_data['final_severity'],
				'actions': alert_data.get('recommended_actions', []),
				'stakeholders': alert_data.get('stakeholders', []),
				'created_at': datetime.utcnow().isoformat()
			}
			
			# Store incident for tracking
			if self.redis_client:
				incident_key = f"incident:{incident_workflow['incident_id']}"
				await self.redis_client.setex(
					incident_key, 604800,  # 7 days
					json.dumps(incident_workflow, default=str)
				)
			
			print(f"Incident response initiated: {incident_workflow['incident_id']}")
			
		except Exception as e:
			print(f"Incident response initiation error: {e}")


# ==============================================================================
# REAL-TIME MONITORING ENGINE
# ==============================================================================

class RealTimeMonitor:
	"""Real-Time GRC Monitoring with Predictive Analytics"""
	
	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.metrics = MonitoringMetrics()
		self.alert_manager = AlertManager(config)
		self.ai_engine = GRCAIEngine()
		self.monitoring_tasks = {}
		self.websocket_server = None
		self.active_connections = set()
		
		# Initialize services
		self.time_series_analytics = None
		if config.time_series_analytics_enabled:
			self.time_series_analytics = TimeSeriesAnalytics()
	
	async def start_monitoring(self):
		"""Start all monitoring processes"""
		print("Starting GRC Real-Time Monitoring...")
		
		# Start monitoring tasks
		self.monitoring_tasks = {
			'risk_monitor': asyncio.create_task(self._monitor_risks()),
			'compliance_monitor': asyncio.create_task(self._monitor_compliance()),
			'control_monitor': asyncio.create_task(self._monitor_controls()),
			'governance_monitor': asyncio.create_task(self._monitor_governance()),
			'predictive_monitor': asyncio.create_task(self._monitor_predictive_alerts())
		}
		
		# Start WebSocket server for real-time updates
		if self.config.websocket_enabled:
			await self._start_websocket_server()
		
		print("GRC Real-Time Monitoring started successfully")
	
	async def stop_monitoring(self):
		"""Stop all monitoring processes"""
		print("Stopping GRC Real-Time Monitoring...")
		
		# Cancel monitoring tasks
		for task_name, task in self.monitoring_tasks.items():
			task.cancel()
			try:
				await task
			except asyncio.CancelledError:
				print(f"Cancelled {task_name}")
		
		# Close WebSocket server
		if self.websocket_server:
			self.websocket_server.close()
			await self.websocket_server.wait_closed()
		
		print("GRC Real-Time Monitoring stopped")
	
	async def _monitor_risks(self):
		"""Monitor risk metrics and thresholds"""
		while True:
			try:
				with self.metrics.monitoring_duration.labels('risk').time():
					# Get current risks (in production, this would query the database)
					risks = await self._get_current_risks()
					
					for risk in risks:
						# Update metrics
						self.metrics.risk_score_gauge.labels(
							risk_id=risk.get('risk_id', 'unknown'),
							category=risk.get('category', 'unknown')
						).set(risk.get('residual_risk_score', 0))
						
						# Check thresholds
						await self._check_risk_thresholds(risk)
						
						# Check for anomalies
						await self._check_risk_anomalies(risk)
				
				# Update active monitors metric
				self.metrics.active_monitors.set(len(self.monitoring_tasks))
				
				await asyncio.sleep(self.config.risk_monitoring_interval_seconds)
				
			except Exception as e:
				print(f"Risk monitoring error: {e}")
				await asyncio.sleep(30)  # Wait before retrying
	
	async def _monitor_compliance(self):
		"""Monitor compliance status and violations"""
		while True:
			try:
				with self.metrics.monitoring_duration.labels('compliance').time():
					# Get compliance status (in production, this would query the database)
					compliance_items = await self._get_compliance_status()
					
					for item in compliance_items:
						# Update metrics
						self.metrics.compliance_percentage_gauge.labels(
							regulation_id=item.get('regulation_id', 'unknown')
						).set(item.get('compliance_percentage', 0))
						
						# Check for violations
						await self._check_compliance_violations(item)
						
						# Monitor regulatory changes
						await self._monitor_regulatory_changes(item)
				
				await asyncio.sleep(self.config.compliance_monitoring_interval_seconds)
				
			except Exception as e:
				print(f"Compliance monitoring error: {e}")
				await asyncio.sleep(60)
	
	async def _monitor_controls(self):
		"""Monitor control effectiveness and failures"""
		while True:
			try:
				with self.metrics.monitoring_duration.labels('controls').time():
					# Get control status (in production, this would query the database)
					controls = await self._get_control_status()
					
					for control in controls:
						# Update metrics
						self.metrics.control_effectiveness_gauge.labels(
							control_id=control.get('control_id', 'unknown')
						).set(control.get('effectiveness_score', 0))
						
						# Check for control failures
						await self._check_control_failures(control)
						
						# Monitor self-testing results
						await self._monitor_control_self_tests(control)
				
				await asyncio.sleep(self.config.control_monitoring_interval_seconds)
				
			except Exception as e:
				print(f"Control monitoring error: {e}")
				await asyncio.sleep(60)
	
	async def _monitor_governance(self):
		"""Monitor governance processes and deadlines"""
		while True:
			try:
				with self.metrics.monitoring_duration.labels('governance').time():
					# Get governance items (in production, this would query the database)
					governance_items = await self._get_governance_status()
					
					for item in governance_items:
						# Check deadlines
						await self._check_governance_deadlines(item)
						
						# Monitor decision workflows
						await self._monitor_decision_workflows(item)
						
						# Check stakeholder engagement
						await self._check_stakeholder_engagement(item)
				
				await asyncio.sleep(self.config.governance_monitoring_interval_seconds)
				
			except Exception as e:
				print(f"Governance monitoring error: {e}")
				await asyncio.sleep(120)
	
	async def _monitor_predictive_alerts(self):
		"""Monitor for predictive alerts using AI"""
		if not self.config.predictive_alerting_enabled:
			return
		
		while True:
			try:
				with self.metrics.monitoring_duration.labels('predictive').time():
					# Generate predictive alerts using AI
					predictive_alerts = await self.ai_engine.generate_predictive_alerts()
					
					for alert in predictive_alerts:
						if alert.get('confidence', 0) >= self.config.ai_confidence_threshold:
							await self._process_predictive_alert(alert)
				
				await asyncio.sleep(3600)  # Check hourly for predictive alerts
				
			except Exception as e:
				print(f"Predictive monitoring error: {e}")
				await asyncio.sleep(1800)  # Wait 30 minutes before retrying
	
	async def _get_current_risks(self) -> List[Dict[str, Any]]:
		"""Get current risk data (mock implementation)"""
		# In production, this would query the database
		return [
			{
				'risk_id': 'risk_001',
				'category': 'operational',
				'residual_risk_score': np.random.uniform(30, 95),
				'risk_velocity': np.random.uniform(-0.1, 0.1),
				'last_assessment': datetime.utcnow() - timedelta(days=np.random.randint(1, 30))
			},
			{
				'risk_id': 'risk_002',
				'category': 'financial',
				'residual_risk_score': np.random.uniform(40, 85),
				'risk_velocity': np.random.uniform(-0.05, 0.15),
				'last_assessment': datetime.utcnow() - timedelta(days=np.random.randint(1, 20))
			}
		]
	
	async def _get_compliance_status(self) -> List[Dict[str, Any]]:
		"""Get compliance status data (mock implementation)"""
		return [
			{
				'regulation_id': 'reg_001',
				'regulation_type': 'gdpr',
				'compliance_percentage': np.random.uniform(70, 98),
				'last_assessment': datetime.utcnow() - timedelta(days=np.random.randint(1, 14))
			},
			{
				'regulation_id': 'reg_002',
				'regulation_type': 'sox',
				'compliance_percentage': np.random.uniform(80, 95),
				'last_assessment': datetime.utcnow() - timedelta(days=np.random.randint(1, 21))
			}
		]
	
	async def _get_control_status(self) -> List[Dict[str, Any]]:
		"""Get control status data (mock implementation)"""
		return [
			{
				'control_id': 'ctrl_001',
				'control_type': 'access_control',
				'effectiveness_score': np.random.uniform(60, 95),
				'failure_count': np.random.randint(0, 5),
				'last_test': datetime.utcnow() - timedelta(days=np.random.randint(1, 30))
			},
			{
				'control_id': 'ctrl_002',
				'control_type': 'data_validation',
				'effectiveness_score': np.random.uniform(70, 98),
				'failure_count': np.random.randint(0, 3),
				'last_test': datetime.utcnow() - timedelta(days=np.random.randint(1, 25))
			}
		]
	
	async def _get_governance_status(self) -> List[Dict[str, Any]]:
		"""Get governance status data (mock implementation)"""
		return [
			{
				'item_id': 'gov_001',
				'item_type': 'decision',
				'deadline': datetime.utcnow() + timedelta(days=np.random.randint(1, 30)),
				'status': 'in_progress',
				'stakeholder_count': np.random.randint(3, 15)
			},
			{
				'item_id': 'gov_002',
				'item_type': 'policy',
				'deadline': datetime.utcnow() + timedelta(days=np.random.randint(5, 60)),
				'status': 'review',
				'stakeholder_count': np.random.randint(2, 10)
			}
		]
	
	async def _check_risk_thresholds(self, risk: Dict[str, Any]):
		"""Check risk score thresholds and generate alerts"""
		risk_score = risk.get('residual_risk_score', 0)
		risk_id = risk.get('risk_id')
		
		if risk_score >= self.config.risk_score_critical_threshold:
			await self._generate_alert({
				'alert_type': AlertType.RISK_THRESHOLD_BREACH,
				'severity': AlertSeverity.CRITICAL,
				'message': f'Risk {risk_id} has reached critical threshold ({risk_score:.1f})',
				'source': 'risk_monitor',
				'source_id': risk_id,
				'risk_score': risk_score,
				'threshold': self.config.risk_score_critical_threshold
			})
			
			self.metrics.risk_alerts_total.labels(
				severity='critical', type='threshold_breach'
			).inc()
			
		elif risk_score >= self.config.risk_score_high_threshold:
			await self._generate_alert({
				'alert_type': AlertType.RISK_THRESHOLD_BREACH,
				'severity': AlertSeverity.HIGH,
				'message': f'Risk {risk_id} has reached high threshold ({risk_score:.1f})',
				'source': 'risk_monitor',
				'source_id': risk_id,
				'risk_score': risk_score,
				'threshold': self.config.risk_score_high_threshold
			})
			
			self.metrics.risk_alerts_total.labels(
				severity='high', type='threshold_breach'
			).inc()
	
	async def _check_risk_anomalies(self, risk: Dict[str, Any]):
		"""Check for risk anomalies using AI"""
		try:
			risk_velocity = risk.get('risk_velocity', 0)
			
			# Simple anomaly detection - in production, this would use sophisticated ML
			if abs(risk_velocity) > 0.1:  # High velocity change
				await self._generate_alert({
					'alert_type': AlertType.SYSTEM_ANOMALY,
					'severity': AlertSeverity.MEDIUM,
					'message': f'Unusual risk velocity detected for {risk["risk_id"]}: {risk_velocity:+.3f}',
					'source': 'risk_anomaly_detector',
					'source_id': risk['risk_id'],
					'anomaly_type': 'risk_velocity',
					'anomaly_value': risk_velocity
				})
		
		except Exception as e:
			print(f"Risk anomaly check error: {e}")
	
	async def _check_compliance_violations(self, item: Dict[str, Any]):
		"""Check for compliance violations"""
		compliance_percentage = item.get('compliance_percentage', 100)
		regulation_id = item.get('regulation_id')
		
		if compliance_percentage < (1 - self.config.compliance_violation_threshold) * 100:
			await self._generate_alert({
				'alert_type': AlertType.COMPLIANCE_VIOLATION,
				'severity': AlertSeverity.HIGH,
				'message': f'Compliance violation detected for {regulation_id} ({compliance_percentage:.1f}%)',
				'source': 'compliance_monitor',
				'source_id': regulation_id,
				'compliance_percentage': compliance_percentage,
				'violation_threshold': self.config.compliance_violation_threshold
			})
			
			self.metrics.compliance_violations_total.labels(
				regulation_type=item.get('regulation_type', 'unknown')
			).inc()
	
	async def _monitor_regulatory_changes(self, item: Dict[str, Any]):
		"""Monitor for regulatory changes"""
		# In production, this would integrate with the compliance engine
		# For now, simulate occasional regulatory changes
		if np.random.random() < 0.01:  # 1% chance of regulatory change
			await self._generate_alert({
				'alert_type': AlertType.REGULATORY_CHANGE,
				'severity': AlertSeverity.MEDIUM,
				'message': f'Regulatory change detected for {item["regulation_id"]}',
				'source': 'regulatory_monitor',
				'source_id': item['regulation_id'],
				'change_type': 'amendment'
			})
	
	async def _check_control_failures(self, control: Dict[str, Any]):
		"""Check for control failures"""
		failure_count = control.get('failure_count', 0)
		control_id = control.get('control_id')
		
		if failure_count >= self.config.control_failure_threshold:
			await self._generate_alert({
				'alert_type': AlertType.CONTROL_FAILURE,
				'severity': AlertSeverity.HIGH,
				'message': f'Control {control_id} has {failure_count} failures (threshold: {self.config.control_failure_threshold})',
				'source': 'control_monitor',
				'source_id': control_id,
				'failure_count': failure_count,
				'threshold': self.config.control_failure_threshold
			})
			
			self.metrics.control_failures_total.labels(
				control_type=control.get('control_type', 'unknown')
			).inc()
	
	async def _monitor_control_self_tests(self, control: Dict[str, Any]):
		"""Monitor control self-test results"""
		# In production, this would check actual self-test results
		# For now, simulate occasional self-test failures
		if np.random.random() < 0.05:  # 5% chance of self-test failure
			await self._generate_alert({
				'alert_type': AlertType.CONTROL_FAILURE,
				'severity': AlertSeverity.MEDIUM,
				'message': f'Self-test failure detected for control {control["control_id"]}',
				'source': 'control_self_test',
				'source_id': control['control_id'],
				'test_type': 'self_test'
			})
	
	async def _check_governance_deadlines(self, item: Dict[str, Any]):
		"""Check governance deadlines"""
		deadline = item.get('deadline')
		item_id = item.get('item_id')
		
		if deadline:
			days_until_deadline = (deadline - datetime.utcnow()).days
			
			if days_until_deadline <= 1:
				await self._generate_alert({
					'alert_type': AlertType.GOVERNANCE_DEADLINE,
					'severity': AlertSeverity.HIGH,
					'message': f'Governance item {item_id} deadline approaching ({days_until_deadline} days)',
					'source': 'governance_monitor',
					'source_id': item_id,
					'deadline': deadline.isoformat(),
					'days_remaining': days_until_deadline
				})
			elif days_until_deadline <= 7:
				await self._generate_alert({
					'alert_type': AlertType.GOVERNANCE_DEADLINE,
					'severity': AlertSeverity.MEDIUM,
					'message': f'Governance item {item_id} deadline in {days_until_deadline} days',
					'source': 'governance_monitor',
					'source_id': item_id,
					'deadline': deadline.isoformat(),
					'days_remaining': days_until_deadline
				})
	
	async def _monitor_decision_workflows(self, item: Dict[str, Any]):
		"""Monitor decision workflow progress"""
		# In production, this would check actual workflow status
		pass
	
	async def _check_stakeholder_engagement(self, item: Dict[str, Any]):
		"""Check stakeholder engagement levels"""
		stakeholder_count = item.get('stakeholder_count', 0)
		
		if stakeholder_count > 10:  # Large stakeholder group
			await self._generate_alert({
				'alert_type': AlertType.STAKEHOLDER_ESCALATION,
				'severity': AlertSeverity.LOW,
				'message': f'Large stakeholder group ({stakeholder_count}) for governance item {item["item_id"]}',
				'source': 'governance_monitor',
				'source_id': item['item_id'],
				'stakeholder_count': stakeholder_count
			})
	
	async def _process_predictive_alert(self, alert: Dict[str, Any]):
		"""Process predictive alerts from AI engine"""
		await self._generate_alert({
			'alert_type': AlertType.PREDICTIVE_WARNING,
			'severity': AlertSeverity.MEDIUM,
			'message': alert.get('message', 'Predictive alert generated'),
			'source': 'ai_predictor',
			'source_id': alert.get('entity_id'),
			'prediction': alert,
			'ai_confidence': alert.get('confidence', 0.0)
		})
	
	async def _generate_alert(self, alert_data: Dict[str, Any]):
		"""Generate and process alert"""
		alert_data['timestamp'] = datetime.utcnow().isoformat()
		alert_data['alert_id'] = f"alert_{datetime.utcnow().timestamp()}"
		
		# Process alert through alert manager
		result = await self.alert_manager.process_alert(alert_data)
		
		# Broadcast to WebSocket clients
		await self._broadcast_alert(alert_data)
		
		return result
	
	async def _start_websocket_server(self):
		"""Start WebSocket server for real-time updates"""
		async def handle_websocket(websocket, path):
			self.active_connections.add(websocket)
			self.metrics.websocket_connections.set(len(self.active_connections))
			
			try:
				# Send initial connection confirmation
				await websocket.send(json.dumps({
					'type': 'connection_established',
					'timestamp': datetime.utcnow().isoformat(),
					'monitoring_status': 'active'
				}))
				
				# Keep connection alive
				async for message in websocket:
					# Handle incoming messages if needed
					pass
					
			except websockets.exceptions.ConnectionClosed:
				pass
			finally:
				self.active_connections.discard(websocket)
				self.metrics.websocket_connections.set(len(self.active_connections))
		
		self.websocket_server = await websockets.serve(
			handle_websocket, "localhost", self.config.websocket_port
		)
		
		print(f"WebSocket server started on port {self.config.websocket_port}")
	
	async def _broadcast_alert(self, alert_data: Dict[str, Any]):
		"""Broadcast alert to all WebSocket connections"""
		if not self.active_connections:
			return
		
		message = json.dumps({
			'type': 'grc_alert',
			'data': alert_data
		}, default=str)
		
		# Send to all active connections
		disconnected = set()
		for websocket in self.active_connections:
			try:
				await websocket.send(message)
			except websockets.exceptions.ConnectionClosed:
				disconnected.add(websocket)
		
		# Remove disconnected connections
		self.active_connections -= disconnected
		self.metrics.websocket_connections.set(len(self.active_connections))
	
	def get_metrics(self) -> str:
		"""Get Prometheus metrics"""
		return generate_latest()


# ==============================================================================
# MONITORING SERVICE
# ==============================================================================

class GRCMonitoringService:
	"""Main GRC Monitoring Service"""
	
	def __init__(self, config: Optional[MonitoringConfig] = None):
		self.config = config or MonitoringConfig()
		self.monitor = RealTimeMonitor(self.config)
		self.running = False
	
	async def start(self):
		"""Start the monitoring service"""
		if self.running:
			print("Monitoring service is already running")
			return
		
		print("Starting GRC Monitoring Service...")
		self.running = True
		
		try:
			await self.monitor.start_monitoring()
			print("GRC Monitoring Service started successfully")
		except Exception as e:
			print(f"Failed to start monitoring service: {e}")
			self.running = False
			raise
	
	async def stop(self):
		"""Stop the monitoring service"""
		if not self.running:
			print("Monitoring service is not running")
			return
		
		print("Stopping GRC Monitoring Service...")
		self.running = False
		
		try:
			await self.monitor.stop_monitoring()
			print("GRC Monitoring Service stopped successfully")
		except Exception as e:
			print(f"Error stopping monitoring service: {e}")
	
	def is_running(self) -> bool:
		"""Check if monitoring service is running"""
		return self.running
	
	def get_status(self) -> Dict[str, Any]:
		"""Get monitoring service status"""
		return {
			'running': self.running,
			'config': {
				'risk_monitoring_interval': self.config.risk_monitoring_interval_seconds,
				'compliance_monitoring_interval': self.config.compliance_monitoring_interval_seconds,
				'predictive_alerting_enabled': self.config.predictive_alerting_enabled,
				'websocket_enabled': self.config.websocket_enabled,
				'websocket_port': self.config.websocket_port
			},
			'metrics_endpoint': '/metrics',
			'websocket_endpoint': f'ws://localhost:{self.config.websocket_port}'
		}
	
	def get_metrics(self) -> str:
		"""Get monitoring metrics"""
		return self.monitor.get_metrics()


# Export the monitoring service
__all__ = [
	'GRCMonitoringService', 'MonitoringConfig', 'RealTimeMonitor', 
	'AlertManager', 'AlertSeverity', 'AlertType'
]