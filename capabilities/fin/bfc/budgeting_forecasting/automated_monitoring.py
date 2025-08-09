"""
APG Budgeting & Forecasting - Automated Budget Monitoring System

Intelligent automated monitoring with smart alerts, real-time variance analysis,
anomaly detection, and proactive budget management notifications.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from enum import Enum
from datetime import datetime, date, timedelta, time
from decimal import Decimal
import asyncio
import logging
import json
from uuid_extensions import uuid7str

from .models import APGBaseModel, PositiveAmount, NonEmptyString
from .service import APGTenantContext, ServiceResponse, APGServiceBase


# =============================================================================
# Automated Monitoring Enumerations
# =============================================================================

class AlertType(str, Enum):
	"""Types of budget monitoring alerts."""
	VARIANCE_THRESHOLD = "variance_threshold"
	BUDGET_OVERRUN = "budget_overrun"
	TREND_ANOMALY = "trend_anomaly"
	SEASONAL_DEVIATION = "seasonal_deviation"
	APPROVAL_REQUIRED = "approval_required"
	DEADLINE_APPROACHING = "deadline_approaching"
	FORECAST_ACCURACY = "forecast_accuracy"
	COMPLIANCE_ISSUE = "compliance_issue"


class AlertSeverity(str, Enum):
	"""Alert severity levels."""
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"


class MonitoringFrequency(str, Enum):
	"""Monitoring check frequencies."""
	REAL_TIME = "real_time"
	HOURLY = "hourly"
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"


class NotificationChannel(str, Enum):
	"""Notification delivery channels."""
	EMAIL = "email"
	SMS = "sms"
	SLACK = "slack"
	TEAMS = "teams"
	WEBHOOK = "webhook"
	IN_APP = "in_app"
	DASHBOARD = "dashboard"


class MonitoringScope(str, Enum):
	"""Scope of monitoring rules."""
	GLOBAL = "global"
	TENANT = "tenant"
	DEPARTMENT = "department"
	PROJECT = "project"
	BUDGET = "budget"
	LINE_ITEM = "line_item"


class TriggerCondition(str, Enum):
	"""Alert trigger conditions."""
	GREATER_THAN = "greater_than"
	LESS_THAN = "less_than"
	EQUALS = "equals"
	NOT_EQUALS = "not_equals"
	BETWEEN = "between"
	OUTSIDE_RANGE = "outside_range"
	PERCENTAGE_CHANGE = "percentage_change"
	TREND_REVERSAL = "trend_reversal"


class AlertStatus(str, Enum):
	"""Alert status tracking."""
	ACTIVE = "active"
	ACKNOWLEDGED = "acknowledged"
	RESOLVED = "resolved"
	SUPPRESSED = "suppressed"
	EXPIRED = "expired"


# =============================================================================
# Automated Monitoring Models
# =============================================================================

class MonitoringRule(APGBaseModel):
	"""Automated monitoring rule configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	rule_id: str = Field(default_factory=uuid7str)
	rule_name: NonEmptyString = Field(description="Rule name")
	alert_type: AlertType = Field(description="Type of alert")
	
	# Rule Configuration
	description: str = Field(description="Rule description")
	scope: MonitoringScope = Field(description="Monitoring scope")
	target_entities: List[str] = Field(default_factory=list, description="Target entity IDs")
	
	# Trigger Configuration
	metric_name: str = Field(description="Metric to monitor")
	trigger_condition: TriggerCondition = Field(description="Trigger condition")
	threshold_value: Decimal = Field(description="Threshold value")
	threshold_percentage: Optional[Decimal] = Field(None, description="Threshold percentage")
	
	# Advanced Conditions
	time_window: Optional[str] = Field(None, description="Time window for evaluation")
	consecutive_violations: int = Field(default=1, description="Consecutive violations required")
	comparison_baseline: Optional[str] = Field(None, description="Baseline for comparison")
	
	# Alert Configuration
	severity: AlertSeverity = Field(description="Alert severity")
	frequency: MonitoringFrequency = Field(description="Check frequency")
	notification_channels: List[NotificationChannel] = Field(description="Notification channels")
	
	# Recipients
	recipients: List[str] = Field(default_factory=list, description="Alert recipients")
	escalation_recipients: List[str] = Field(default_factory=list, description="Escalation recipients")
	escalation_delay: Optional[int] = Field(None, description="Escalation delay in minutes")
	
	# Rule Status
	is_active: bool = Field(default=True, description="Rule is active")
	last_checked: Optional[datetime] = Field(None, description="Last check time")
	last_triggered: Optional[datetime] = Field(None, description="Last trigger time")
	trigger_count: int = Field(default=0, description="Number of times triggered")
	
	# Suppression
	suppression_start: Optional[datetime] = Field(None, description="Suppression start time")
	suppression_end: Optional[datetime] = Field(None, description="Suppression end time")
	suppression_reason: Optional[str] = Field(None, description="Suppression reason")


class BudgetAlert(APGBaseModel):
	"""Individual budget monitoring alert."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	alert_id: str = Field(default_factory=uuid7str)
	rule_id: str = Field(description="Monitoring rule that generated alert")
	alert_type: AlertType = Field(description="Type of alert")
	
	# Alert Details
	title: NonEmptyString = Field(description="Alert title")
	message: str = Field(description="Alert message")
	severity: AlertSeverity = Field(description="Alert severity")
	
	# Context Information
	entity_type: str = Field(description="Type of entity (budget, department, etc.)")
	entity_id: str = Field(description="Entity ID")
	entity_name: str = Field(description="Entity name")
	metric_name: str = Field(description="Metric that triggered alert")
	
	# Alert Data
	current_value: Decimal = Field(description="Current metric value")
	threshold_value: Decimal = Field(description="Threshold value")
	variance_amount: Optional[Decimal] = Field(None, description="Variance amount")
	variance_percentage: Optional[Decimal] = Field(None, description="Variance percentage")
	
	# Supporting Data
	historical_context: Dict[str, Any] = Field(default_factory=dict, description="Historical context")
	additional_metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics")
	
	# Alert Status
	status: AlertStatus = Field(default=AlertStatus.ACTIVE)
	triggered_date: datetime = Field(default_factory=datetime.utcnow)
	acknowledged_date: Optional[datetime] = Field(None)
	resolved_date: Optional[datetime] = Field(None)
	
	# User Interaction
	acknowledged_by: Optional[str] = Field(None, description="User who acknowledged alert")
	resolved_by: Optional[str] = Field(None, description="User who resolved alert")
	resolution_notes: Optional[str] = Field(None, description="Resolution notes")
	
	# Notification Tracking
	notifications_sent: List[Dict[str, Any]] = Field(default_factory=list, description="Notification log")
	escalation_level: int = Field(default=0, description="Current escalation level")


class MonitoringDashboard(APGBaseModel):
	"""Real-time monitoring dashboard configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	dashboard_id: str = Field(default_factory=uuid7str)
	dashboard_name: NonEmptyString = Field(description="Dashboard name")
	
	# Dashboard Configuration
	monitoring_rules: List[str] = Field(description="Monitored rule IDs")
	refresh_interval: int = Field(default=60, description="Refresh interval in seconds")
	
	# Alert Summary
	active_alerts: int = Field(default=0, description="Number of active alerts")
	critical_alerts: int = Field(default=0, description="Number of critical alerts")
	warning_alerts: int = Field(default=0, description="Number of warning alerts")
	
	# Performance Metrics
	monitoring_health: str = Field(default="healthy", description="Overall monitoring health")
	last_check_time: Optional[datetime] = Field(None, description="Last monitoring check")
	system_performance: Dict[str, Any] = Field(default_factory=dict, description="System performance metrics")
	
	# Alert Trends
	alert_trends: Dict[str, List[int]] = Field(default_factory=dict, description="Alert count trends")
	resolution_times: Dict[str, float] = Field(default_factory=dict, description="Average resolution times")


class AnomalyDetection(APGBaseModel):
	"""Anomaly detection configuration and results."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	detection_id: str = Field(default_factory=uuid7str)
	detection_name: NonEmptyString = Field(description="Detection name")
	
	# Detection Configuration
	metric_name: str = Field(description="Metric for anomaly detection")
	detection_method: str = Field(description="Anomaly detection method")
	sensitivity: Decimal = Field(default=Decimal("0.8"), description="Detection sensitivity")
	
	# Historical Analysis
	baseline_period: str = Field(description="Baseline period for comparison")
	seasonal_adjustment: bool = Field(default=True, description="Apply seasonal adjustment")
	
	# Detection Results
	anomalies_detected: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
	anomaly_score: Optional[Decimal] = Field(None, description="Overall anomaly score")
	confidence_level: Optional[Decimal] = Field(None, description="Detection confidence")
	
	# Analysis Period
	analysis_start: date = Field(description="Analysis start date")
	analysis_end: date = Field(description="Analysis end date")
	last_analysis: datetime = Field(default_factory=datetime.utcnow)


class AutomatedAction(APGBaseModel):
	"""Automated action triggered by monitoring rules."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	action_id: str = Field(default_factory=uuid7str)
	action_name: NonEmptyString = Field(description="Action name")
	trigger_rule_id: str = Field(description="Rule that triggered action")
	
	# Action Configuration
	action_type: str = Field(description="Type of automated action")
	action_parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
	
	# Execution Details
	execution_status: str = Field(default="pending", description="Execution status")
	execution_time: Optional[datetime] = Field(None, description="Execution time")
	execution_result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
	
	# Safety Constraints
	approval_required: bool = Field(default=False, description="Requires approval before execution")
	max_impact: Optional[Decimal] = Field(None, description="Maximum financial impact allowed")
	rollback_possible: bool = Field(default=False, description="Action can be rolled back")


# =============================================================================
# Automated Budget Monitoring Service
# =============================================================================

class AutomatedBudgetMonitoringService(APGServiceBase):
	"""
	Automated budget monitoring service providing intelligent alerts,
	anomaly detection, and proactive budget management.
	"""
	
	def __init__(self, context: APGTenantContext, config: Optional[Dict[str, Any]] = None):
		super().__init__(context, config)
		self.logger = logging.getLogger(__name__)
		self._monitoring_active = False
		self._monitoring_tasks: List[asyncio.Task] = []
	
	async def create_monitoring_rule(
		self, 
		rule_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Create new monitoring rule."""
		try:
			self.logger.info(f"Creating monitoring rule: {rule_config.get('rule_name')}")
			
			# Validate configuration
			required_fields = ['rule_name', 'alert_type', 'metric_name', 'trigger_condition', 'threshold_value']
			missing_fields = [field for field in required_fields if field not in rule_config]
			if missing_fields:
				return ServiceResponse(
					success=False,
					message=f"Missing required fields: {missing_fields}",
					errors=missing_fields
				)
			
			# Create monitoring rule
			rule = MonitoringRule(
				rule_name=rule_config['rule_name'],
				alert_type=rule_config['alert_type'],
				description=rule_config.get('description', ''),
				scope=rule_config.get('scope', MonitoringScope.BUDGET),
				target_entities=rule_config.get('target_entities', []),
				metric_name=rule_config['metric_name'],
				trigger_condition=rule_config['trigger_condition'],
				threshold_value=rule_config['threshold_value'],
				threshold_percentage=rule_config.get('threshold_percentage'),
				severity=rule_config.get('severity', AlertSeverity.WARNING),
				frequency=rule_config.get('frequency', MonitoringFrequency.DAILY),
				notification_channels=rule_config.get('notification_channels', [NotificationChannel.EMAIL]),
				recipients=rule_config.get('recipients', []),
				escalation_recipients=rule_config.get('escalation_recipients', []),
				escalation_delay=rule_config.get('escalation_delay'),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Validate rule logic
			validation_result = await self._validate_monitoring_rule(rule)
			if not validation_result['valid']:
				return ServiceResponse(
					success=False,
					message="Invalid monitoring rule configuration",
					errors=validation_result['errors']
				)
			
			self.logger.info(f"Monitoring rule created: {rule.rule_id}")
			
			return ServiceResponse(
				success=True,
				message="Monitoring rule created successfully",
				data=rule.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error creating monitoring rule: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to create monitoring rule: {str(e)}",
				errors=[str(e)]
			)
	
	async def start_automated_monitoring(self) -> ServiceResponse:
		"""Start automated monitoring processes."""
		try:
			self.logger.info("Starting automated budget monitoring")
			
			if self._monitoring_active:
				return ServiceResponse(
					success=False,
					message="Monitoring is already active",
					errors=["monitoring_already_active"]
				)
			
			# Start monitoring tasks
			self._monitoring_active = True
			
			# Real-time monitoring task
			real_time_task = asyncio.create_task(self._real_time_monitoring_loop())
			self._monitoring_tasks.append(real_time_task)
			
			# Scheduled monitoring task
			scheduled_task = asyncio.create_task(self._scheduled_monitoring_loop())
			self._monitoring_tasks.append(scheduled_task)
			
			# Anomaly detection task
			anomaly_task = asyncio.create_task(self._anomaly_detection_loop())
			self._monitoring_tasks.append(anomaly_task)
			
			self.logger.info("Automated monitoring started successfully")
			
			return ServiceResponse(
				success=True,
				message="Automated monitoring started successfully",
				data={
					'monitoring_active': True,
					'active_tasks': len(self._monitoring_tasks),
					'start_time': datetime.utcnow()
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error starting automated monitoring: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to start automated monitoring: {str(e)}",
				errors=[str(e)]
			)
	
	async def stop_automated_monitoring(self) -> ServiceResponse:
		"""Stop automated monitoring processes."""
		try:
			self.logger.info("Stopping automated budget monitoring")
			
			if not self._monitoring_active:
				return ServiceResponse(
					success=False,
					message="Monitoring is not active",
					errors=["monitoring_not_active"]
				)
			
			# Stop monitoring
			self._monitoring_active = False
			
			# Cancel all monitoring tasks
			for task in self._monitoring_tasks:
				task.cancel()
			
			# Wait for tasks to complete
			if self._monitoring_tasks:
				await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
			
			self._monitoring_tasks.clear()
			
			self.logger.info("Automated monitoring stopped successfully")
			
			return ServiceResponse(
				success=True,
				message="Automated monitoring stopped successfully",
				data={
					'monitoring_active': False,
					'stop_time': datetime.utcnow()
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error stopping automated monitoring: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to stop automated monitoring: {str(e)}",
				errors=[str(e)]
			)
	
	async def get_active_alerts(
		self, 
		filter_criteria: Optional[Dict[str, Any]] = None
	) -> ServiceResponse:
		"""Get active budget alerts."""
		try:
			self.logger.info("Getting active budget alerts")
			
			# Apply filters
			filters = filter_criteria or {}
			
			# Get active alerts (simulated)
			active_alerts = await self._get_filtered_alerts(filters)
			
			# Calculate alert statistics
			alert_stats = await self._calculate_alert_statistics(active_alerts)
			
			return ServiceResponse(
				success=True,
				message="Active alerts retrieved successfully",
				data={
					'alerts': active_alerts,
					'statistics': alert_stats,
					'filter_criteria': filters,
					'total_count': len(active_alerts)
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error getting active alerts: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to get active alerts: {str(e)}",
				errors=[str(e)]
			)
	
	async def acknowledge_alert(
		self, 
		alert_id: str, 
		acknowledgment_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Acknowledge a budget alert."""
		try:
			self.logger.info(f"Acknowledging alert {alert_id}")
			
			# Load alert
			alert = await self._load_alert(alert_id)
			
			if alert['status'] != AlertStatus.ACTIVE:
				return ServiceResponse(
					success=False,
					message="Only active alerts can be acknowledged",
					errors=["alert_not_active"]
				)
			
			# Update alert status
			alert['status'] = AlertStatus.ACKNOWLEDGED
			alert['acknowledged_date'] = datetime.utcnow()
			alert['acknowledged_by'] = self.context.user_id
			
			# Add acknowledgment notes
			if 'notes' in acknowledgment_config:
				alert['acknowledgment_notes'] = acknowledgment_config['notes']
			
			# Send acknowledgment notifications
			await self._send_acknowledgment_notifications(alert, acknowledgment_config)
			
			return ServiceResponse(
				success=True,
				message="Alert acknowledged successfully",
				data=alert
			)
			
		except Exception as e:
			self.logger.error(f"Error acknowledging alert: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to acknowledge alert: {str(e)}",
				errors=[str(e)]
			)
	
	async def resolve_alert(
		self, 
		alert_id: str, 
		resolution_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Resolve a budget alert."""
		try:
			self.logger.info(f"Resolving alert {alert_id}")
			
			# Load alert
			alert = await self._load_alert(alert_id)
			
			# Update alert status
			alert['status'] = AlertStatus.RESOLVED
			alert['resolved_date'] = datetime.utcnow()
			alert['resolved_by'] = self.context.user_id
			alert['resolution_notes'] = resolution_config.get('notes', '')
			
			# Perform automated resolution actions
			if resolution_config.get('auto_actions', False):
				await self._execute_resolution_actions(alert, resolution_config)
			
			# Send resolution notifications
			await self._send_resolution_notifications(alert, resolution_config)
			
			return ServiceResponse(
				success=True,
				message="Alert resolved successfully",
				data=alert
			)
			
		except Exception as e:
			self.logger.error(f"Error resolving alert: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to resolve alert: {str(e)}",
				errors=[str(e)]
			)
	
	async def perform_anomaly_detection(
		self, 
		detection_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Perform anomaly detection on budget data."""
		try:
			self.logger.info("Performing budget anomaly detection")
			
			# Create anomaly detection configuration
			detection = AnomalyDetection(
				detection_name=detection_config.get('detection_name', 'Budget Anomaly Detection'),
				metric_name=detection_config['metric_name'],
				detection_method=detection_config.get('detection_method', 'statistical'),
				sensitivity=detection_config.get('sensitivity', Decimal("0.8")),
				baseline_period=detection_config.get('baseline_period', 'last_6_months'),
				seasonal_adjustment=detection_config.get('seasonal_adjustment', True),
				analysis_start=detection_config['analysis_start'],
				analysis_end=detection_config['analysis_end'],
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Perform anomaly detection
			anomalies = await self._detect_anomalies(detection)
			detection.anomalies_detected = anomalies
			
			# Calculate anomaly scores
			await self._calculate_anomaly_scores(detection)
			
			# Generate alerts for significant anomalies
			generated_alerts = await self._generate_anomaly_alerts(detection)
			
			return ServiceResponse(
				success=True,
				message="Anomaly detection completed successfully",
				data={
					'detection': detection.model_dump(),
					'anomalies_count': len(anomalies),
					'alerts_generated': len(generated_alerts)
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error performing anomaly detection: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to perform anomaly detection: {str(e)}",
				errors=[str(e)]
			)
	
	async def get_monitoring_dashboard(self) -> ServiceResponse:
		"""Get real-time monitoring dashboard data."""
		try:
			self.logger.info("Getting monitoring dashboard data")
			
			# Create dashboard
			dashboard = MonitoringDashboard(
				dashboard_name="Budget Monitoring Dashboard",
				monitoring_rules=[],  # Would be populated with actual rule IDs
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Get current alert counts
			alert_summary = await self._get_alert_summary()
			dashboard.active_alerts = alert_summary['active']
			dashboard.critical_alerts = alert_summary['critical']
			dashboard.warning_alerts = alert_summary['warning']
			
			# Get monitoring health
			health_status = await self._get_monitoring_health()
			dashboard.monitoring_health = health_status['status']
			dashboard.last_check_time = health_status['last_check']
			dashboard.system_performance = health_status['performance']
			
			# Get alert trends
			trends = await self._get_alert_trends()
			dashboard.alert_trends = trends['trends']
			dashboard.resolution_times = trends['resolution_times']
			
			return ServiceResponse(
				success=True,
				message="Monitoring dashboard data retrieved successfully",
				data=dashboard.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error getting monitoring dashboard: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to get monitoring dashboard: {str(e)}",
				errors=[str(e)]
			)
	
	# =============================================================================
	# Private Helper Methods
	# =============================================================================
	
	async def _validate_monitoring_rule(self, rule: MonitoringRule) -> Dict[str, Any]:
		"""Validate monitoring rule configuration."""
		errors = []
		
		# Validate threshold values
		if rule.trigger_condition in [TriggerCondition.BETWEEN, TriggerCondition.OUTSIDE_RANGE]:
			if rule.threshold_percentage is None:
				errors.append("Threshold percentage required for range conditions")
		
		# Validate notification channels
		if not rule.notification_channels:
			errors.append("At least one notification channel is required")
		
		# Validate recipients
		if NotificationChannel.EMAIL in rule.notification_channels and not rule.recipients:
			errors.append("Email recipients required when email notification is enabled")
		
		return {
			'valid': len(errors) == 0,
			'errors': errors
		}
	
	async def _real_time_monitoring_loop(self) -> None:
		"""Real-time monitoring loop."""
		self.logger.info("Starting real-time monitoring loop")
		
		while self._monitoring_active:
			try:
				# Check real-time rules
				await self._check_real_time_rules()
				
				# Short sleep for real-time monitoring
				await asyncio.sleep(30)  # Check every 30 seconds
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Error in real-time monitoring loop: {e}")
				await asyncio.sleep(60)  # Longer sleep on error
	
	async def _scheduled_monitoring_loop(self) -> None:
		"""Scheduled monitoring loop."""
		self.logger.info("Starting scheduled monitoring loop")
		
		while self._monitoring_active:
			try:
				# Check scheduled rules
				await self._check_scheduled_rules()
				
				# Sleep until next check
				await asyncio.sleep(300)  # Check every 5 minutes
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Error in scheduled monitoring loop: {e}")
				await asyncio.sleep(600)  # Longer sleep on error
	
	async def _anomaly_detection_loop(self) -> None:
		"""Anomaly detection loop."""
		self.logger.info("Starting anomaly detection loop")
		
		while self._monitoring_active:
			try:
				# Run anomaly detection
				await self._run_periodic_anomaly_detection()
				
				# Sleep for longer period
				await asyncio.sleep(3600)  # Check every hour
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Error in anomaly detection loop: {e}")
				await asyncio.sleep(1800)  # Sleep 30 minutes on error
	
	async def _check_real_time_rules(self) -> None:
		"""Check real-time monitoring rules."""
		# Simulated real-time rule checking
		rules = await self._get_active_rules(MonitoringFrequency.REAL_TIME)
		
		for rule in rules:
			await self._evaluate_monitoring_rule(rule)
	
	async def _check_scheduled_rules(self) -> None:
		"""Check scheduled monitoring rules."""
		# Simulated scheduled rule checking
		rules = await self._get_active_rules(MonitoringFrequency.DAILY)
		
		for rule in rules:
			await self._evaluate_monitoring_rule(rule)
	
	async def _run_periodic_anomaly_detection(self) -> None:
		"""Run periodic anomaly detection."""
		# Simulated anomaly detection
		detection_config = {
			'metric_name': 'budget_variance',
			'analysis_start': date.today() - timedelta(days=30),
			'analysis_end': date.today()
		}
		
		await self.perform_anomaly_detection(detection_config)
	
	async def _get_active_rules(self, frequency: MonitoringFrequency) -> List[Dict[str, Any]]:
		"""Get active monitoring rules by frequency."""
		# Simulated rule retrieval
		return [
			{
				'rule_id': 'rule_1',
				'rule_name': 'Budget Variance Alert',
				'frequency': frequency,
				'threshold_value': 10000
			}
		]
	
	async def _evaluate_monitoring_rule(self, rule: Dict[str, Any]) -> None:
		"""Evaluate individual monitoring rule."""
		# Simulated rule evaluation
		current_value = await self._get_current_metric_value(rule['rule_id'])
		
		if current_value > rule['threshold_value']:
			await self._trigger_alert(rule, current_value)
	
	async def _get_current_metric_value(self, rule_id: str) -> Decimal:
		"""Get current value for monitored metric."""
		# Simulated metric retrieval
		return Decimal("12500")  # Example variance value
	
	async def _trigger_alert(self, rule: Dict[str, Any], current_value: Decimal) -> None:
		"""Trigger alert based on rule violation."""
		alert = BudgetAlert(
			rule_id=rule['rule_id'],
			alert_type=AlertType.VARIANCE_THRESHOLD,
			title=f"Budget Variance Alert - {rule['rule_name']}",
			message=f"Budget variance exceeds threshold: ${current_value:,.2f}",
			severity=AlertSeverity.WARNING,
			entity_type="budget",
			entity_id="budget_123",
			entity_name="Annual Budget 2025",
			metric_name="variance_amount",
			current_value=current_value,
			threshold_value=rule['threshold_value'],
			tenant_id=self.context.tenant_id,
			created_by=self.context.user_id
		)
		
		# Send notifications
		await self._send_alert_notifications(alert)
	
	async def _send_alert_notifications(self, alert: BudgetAlert) -> None:
		"""Send alert notifications."""
		# Simulated notification sending
		notification_log = {
			'channel': 'email',
			'sent_time': datetime.utcnow(),
			'recipients': ['admin@company.com'],
			'status': 'sent'
		}
		alert.notifications_sent.append(notification_log)
	
	async def _get_filtered_alerts(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Get filtered active alerts."""
		# Simulated alert retrieval
		alerts = [
			{
				'alert_id': 'alert_1',
				'title': 'Budget Variance Alert',
				'severity': 'warning',
				'status': 'active',
				'triggered_date': datetime.utcnow(),
				'entity_name': 'IT Department'
			},
			{
				'alert_id': 'alert_2',
				'title': 'Budget Overrun Alert',
				'severity': 'critical',
				'status': 'active',
				'triggered_date': datetime.utcnow(),
				'entity_name': 'Marketing Department'
			}
		]
		
		# Apply filters
		if 'severity' in filters:
			alerts = [a for a in alerts if a['severity'] == filters['severity']]
		
		return alerts
	
	async def _calculate_alert_statistics(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Calculate alert statistics."""
		total = len(alerts)
		by_severity = {}
		by_status = {}
		
		for alert in alerts:
			severity = alert.get('severity', 'unknown')
			status = alert.get('status', 'unknown')
			
			by_severity[severity] = by_severity.get(severity, 0) + 1
			by_status[status] = by_status.get(status, 0) + 1
		
		return {
			'total_alerts': total,
			'by_severity': by_severity,
			'by_status': by_status,
			'average_age_hours': 6.5  # Simulated
		}
	
	async def _load_alert(self, alert_id: str) -> Dict[str, Any]:
		"""Load alert by ID."""
		# Simulated alert loading
		return {
			'alert_id': alert_id,
			'title': 'Budget Variance Alert',
			'status': AlertStatus.ACTIVE,
			'severity': AlertSeverity.WARNING,
			'entity_name': 'IT Department'
		}
	
	async def _send_acknowledgment_notifications(
		self, 
		alert: Dict[str, Any], 
		config: Dict[str, Any]
	) -> None:
		"""Send acknowledgment notifications."""
		# Simulated acknowledgment notification
		pass
	
	async def _send_resolution_notifications(
		self, 
		alert: Dict[str, Any], 
		config: Dict[str, Any]
	) -> None:
		"""Send resolution notifications."""
		# Simulated resolution notification
		pass
	
	async def _execute_resolution_actions(
		self, 
		alert: Dict[str, Any], 
		config: Dict[str, Any]
	) -> None:
		"""Execute automated resolution actions."""
		# Simulated automated actions
		pass
	
	async def _detect_anomalies(self, detection: AnomalyDetection) -> List[Dict[str, Any]]:
		"""Detect anomalies in budget data."""
		# Simulated anomaly detection
		return [
			{
				'date': '2025-01-15',
				'metric_value': 15000,
				'expected_value': 12000,
				'anomaly_score': 0.75,
				'description': 'Unusual spike in IT spending'
			}
		]
	
	async def _calculate_anomaly_scores(self, detection: AnomalyDetection) -> None:
		"""Calculate overall anomaly scores."""
		if detection.anomalies_detected:
			scores = [a.get('anomaly_score', 0) for a in detection.anomalies_detected]
			detection.anomaly_score = Decimal(str(sum(scores) / len(scores)))
			detection.confidence_level = Decimal("0.85")
	
	async def _generate_anomaly_alerts(self, detection: AnomalyDetection) -> List[Dict[str, Any]]:
		"""Generate alerts for detected anomalies."""
		alerts = []
		
		for anomaly in detection.anomalies_detected:
			if anomaly.get('anomaly_score', 0) > 0.7:  # High confidence threshold
				alert = {
					'alert_type': AlertType.TREND_ANOMALY,
					'title': f"Anomaly Detected: {anomaly['description']}",
					'severity': AlertSeverity.WARNING,
					'anomaly_data': anomaly
				}
				alerts.append(alert)
		
		return alerts
	
	async def _get_alert_summary(self) -> Dict[str, int]:
		"""Get current alert summary."""
		return {
			'active': 5,
			'critical': 1,
			'warning': 3,
			'info': 1
		}
	
	async def _get_monitoring_health(self) -> Dict[str, Any]:
		"""Get monitoring system health."""
		return {
			'status': 'healthy',
			'last_check': datetime.utcnow(),
			'performance': {
				'check_latency': 250,  # milliseconds
				'success_rate': 0.99,
				'error_rate': 0.01
			}
		}
	
	async def _get_alert_trends(self) -> Dict[str, Any]:
		"""Get alert trends and resolution metrics."""
		return {
			'trends': {
				'daily': [2, 3, 1, 4, 2, 5, 3],  # Last 7 days
				'weekly': [15, 18, 12, 20, 14],   # Last 5 weeks
			},
			'resolution_times': {
				'critical': 2.5,    # hours
				'warning': 8.0,     # hours
				'info': 24.0        # hours
			}
		}


# =============================================================================
# Service Factory Functions
# =============================================================================

def create_automated_budget_monitoring_service(
	context: APGTenantContext, 
	config: Optional[Dict[str, Any]] = None
) -> AutomatedBudgetMonitoringService:
	"""Create automated budget monitoring service instance."""
	return AutomatedBudgetMonitoringService(context, config)


async def create_sample_monitoring_rules(
	service: AutomatedBudgetMonitoringService
) -> List[ServiceResponse]:
	"""Create sample monitoring rules for testing."""
	
	# Budget variance rule
	variance_rule = {
		'rule_name': 'Budget Variance Alert',
		'alert_type': AlertType.VARIANCE_THRESHOLD,
		'description': 'Alert when budget variance exceeds threshold',
		'metric_name': 'variance_amount',
		'trigger_condition': TriggerCondition.GREATER_THAN,
		'threshold_value': Decimal("10000"),
		'severity': AlertSeverity.WARNING,
		'notification_channels': [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
		'recipients': ['budget.manager@company.com']
	}
	
	# Budget overrun rule
	overrun_rule = {
		'rule_name': 'Budget Overrun Alert',
		'alert_type': AlertType.BUDGET_OVERRUN,
		'description': 'Critical alert for budget overruns',
		'metric_name': 'actual_amount',
		'trigger_condition': TriggerCondition.GREATER_THAN,
		'threshold_value': Decimal("1000000"),  # Budget limit
		'severity': AlertSeverity.CRITICAL,
		'notification_channels': [NotificationChannel.EMAIL, NotificationChannel.SMS],
		'recipients': ['cfo@company.com', 'finance.director@company.com'],
		'escalation_recipients': ['ceo@company.com'],
		'escalation_delay': 30  # 30 minutes
	}
	
	# Trend anomaly rule
	anomaly_rule = {
		'rule_name': 'Spending Trend Anomaly',
		'alert_type': AlertType.TREND_ANOMALY,
		'description': 'Detect unusual spending patterns',
		'metric_name': 'monthly_spending',
		'trigger_condition': TriggerCondition.PERCENTAGE_CHANGE,
		'threshold_value': Decimal("0"),
		'threshold_percentage': Decimal("25"),  # 25% change
		'severity': AlertSeverity.INFO,
		'notification_channels': [NotificationChannel.IN_APP],
		'recipients': ['analyst@company.com']
	}
	
	# Create all rules
	results = []
	for rule_config in [variance_rule, overrun_rule, anomaly_rule]:
		result = await service.create_monitoring_rule(rule_config)
		results.append(result)
	
	return results