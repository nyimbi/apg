"""
APG Workflow Orchestration - Alerting & Notifications
Comprehensive alerting system with failure alerts, performance alerts, SLA notifications, and escalation
"""

import asyncio
import json
import smtplib
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import aiohttp
import aioredis
import logging
from pydantic import BaseModel, Field, validator
from pydantic.config import ConfigDict

# APG Framework imports
from apg.base.service import APGBaseService
from apg.base.models import BaseModel as APGBaseModel
from apg.integrations.notifications import NotificationService
from apg.integrations.slack import SlackClient
from apg.integrations.teams import TeamsClient
from apg.integrations.pagerduty import PagerDutyClient
from apg.base.security import SecurityManager

from .models import WorkflowExecution, WorkflowInstance
from .monitoring import WorkflowMetrics, SystemMetrics, HealthStatus


class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


class AlertStatus(str, Enum):
	"""Alert status"""
	ACTIVE = "active"
	ACKNOWLEDGED = "acknowledged"
	RESOLVED = "resolved"
	SUPPRESSED = "suppressed"


class NotificationChannel(str, Enum):
	"""Notification channels"""
	EMAIL = "email"
	SLACK = "slack"
	TEAMS = "teams"
	WEBHOOK = "webhook"
	SMS = "sms"
	PAGERDUTY = "pagerduty"
	DISCORD = "discord"


class EscalationLevel(str, Enum):
	"""Escalation levels"""
	LEVEL_1 = "level_1"
	LEVEL_2 = "level_2"
	LEVEL_3 = "level_3"
	EXECUTIVE = "executive"


@dataclass
class AlertCondition:
	"""Alert condition definition"""
	metric_name: str
	operator: str  # gt, lt, gte, lte, eq, ne
	threshold: float
	duration_minutes: int
	comparison_window_minutes: Optional[int] = None


@dataclass
class EscalationRule:
	"""Escalation rule definition"""
	level: EscalationLevel
	delay_minutes: int
	channels: List[NotificationChannel]
	recipients: List[str]
	conditions: List[str]  # Conditions that must be met


class Alert(APGBaseModel):
	"""Alert model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Alert identifier")
	rule_id: str = Field(..., description="Alert rule identifier")
	tenant_id: str = Field(..., description="Tenant identifier")
	
	# Alert metadata
	title: str = Field(..., description="Alert title")
	description: str = Field(..., description="Alert description")
	severity: AlertSeverity = Field(..., description="Alert severity")
	status: AlertStatus = Field(AlertStatus.ACTIVE, description="Alert status")
	
	# Timing
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Alert created time")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Alert last updated time")
	acknowledged_at: Optional[datetime] = Field(None, description="Alert acknowledged time")
	resolved_at: Optional[datetime] = Field(None, description="Alert resolved time")
	
	# Context
	source: str = Field(..., description="Alert source (workflow_id, system, etc.)")
	source_type: str = Field(..., description="Source type (workflow, system, connector)")
	metric_name: str = Field(..., description="Metric that triggered the alert")
	current_value: float = Field(..., description="Current metric value")
	threshold_value: float = Field(..., description="Threshold that was exceeded")
	
	# Escalation
	escalation_level: EscalationLevel = Field(EscalationLevel.LEVEL_1, description="Current escalation level")
	escalated_at: Optional[datetime] = Field(None, description="Last escalation time")
	
	# Notifications
	notifications_sent: List[str] = Field(default_factory=list, description="Notification channels used")
	notification_count: int = Field(0, description="Number of notifications sent")
	
	# Additional data
	labels: Dict[str, str] = Field(default_factory=dict, description="Alert labels")
	annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")
	related_alerts: List[str] = Field(default_factory=list, description="Related alert IDs")


class AlertRule(APGBaseModel):
	"""Alert rule configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Rule identifier")
	tenant_id: str = Field(..., description="Tenant identifier")
	name: str = Field(..., description="Rule name")
	description: str = Field(..., description="Rule description")
	
	# Rule configuration
	enabled: bool = Field(True, description="Whether rule is enabled")
	conditions: List[AlertCondition] = Field(..., description="Alert conditions")
	severity: AlertSeverity = Field(..., description="Alert severity")
	
	# Notification settings
	notification_channels: List[NotificationChannel] = Field(..., description="Notification channels")
	recipients: List[str] = Field(..., description="Notification recipients")
	
	# Escalation configuration
	escalation_enabled: bool = Field(False, description="Whether escalation is enabled")
	escalation_rules: List[EscalationRule] = Field(default_factory=list, description="Escalation rules")
	
	# Rate limiting
	max_notifications_per_hour: int = Field(10, description="Maximum notifications per hour")
	suppress_duration_minutes: int = Field(60, description="Suppression duration after resolution")
	
	# Labels and filters
	labels: Dict[str, str] = Field(default_factory=dict, description="Rule labels")
	filters: Dict[str, str] = Field(default_factory=dict, description="Metric filters")
	
	# Timing
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class NotificationTemplate(APGBaseModel):
	"""Notification message template"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Template identifier")
	name: str = Field(..., description="Template name")
	channel: NotificationChannel = Field(..., description="Target channel")
	
	# Template content
	subject_template: str = Field(..., description="Subject template")
	body_template: str = Field(..., description="Body template")
	
	# Formatting
	format_type: str = Field("text", description="Format type (text, html, markdown)")
	variables: List[str] = Field(default_factory=list, description="Available template variables")
	
	# Configuration
	priority: str = Field("normal", description="Message priority")
	tags: List[str] = Field(default_factory=list, description="Message tags")


class AlertingConfig(APGBaseModel):
	"""Alerting configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Processing settings
	evaluation_interval_seconds: int = Field(60, description="Alert evaluation interval")
	batch_size: int = Field(100, description="Alert processing batch size")
	max_concurrent_notifications: int = Field(10, description="Max concurrent notifications")
	
	# Retention settings
	alert_retention_days: int = Field(90, description="Alert retention period")
	resolved_alert_retention_days: int = Field(30, description="Resolved alert retention period")
	
	# Rate limiting
	global_rate_limit_per_minute: int = Field(100, description="Global notification rate limit")
	per_channel_rate_limit_per_minute: int = Field(20, description="Per-channel rate limit")
	
	# Email settings
	smtp_host: str = Field("localhost", description="SMTP server host")
	smtp_port: int = Field(587, description="SMTP server port")
	smtp_username: Optional[str] = Field(None, description="SMTP username")
	smtp_password: Optional[str] = Field(None, description="SMTP password")
	smtp_use_tls: bool = Field(True, description="Use TLS for SMTP")
	default_sender: str = Field("noreply@workflow.local", description="Default sender email")
	
	# Webhook settings
	webhook_timeout_seconds: int = Field(30, description="Webhook timeout")
	webhook_retry_attempts: int = Field(3, description="Webhook retry attempts")
	
	# Integration settings
	slack_enabled: bool = Field(False, description="Enable Slack integration")
	teams_enabled: bool = Field(False, description="Enable Teams integration")
	pagerduty_enabled: bool = Field(False, description="Enable PagerDuty integration")
	discord_enabled: bool = Field(False, description="Enable Discord integration")


class AlertingService(APGBaseService):
	"""Main alerting and notification service"""
	
	def __init__(self, config: AlertingConfig):
		super().__init__()
		self.config = config
		
		# Storage
		self.redis_client: Optional[aioredis.Redis] = None
		
		# Components
		self.alert_processor = AlertProcessor(config)
		self.notification_dispatcher = NotificationDispatcher(config)
		self.escalation_manager = EscalationManager(config)
		self.template_manager = TemplateManager()
		
		# Integrations
		self.notification_service = NotificationService()
		self.slack_client = SlackClient() if config.slack_enabled else None
		self.teams_client = TeamsClient() if config.teams_enabled else None
		self.pagerduty_client = PagerDutyClient() if config.pagerduty_enabled else None
		
		# State
		self.active_alerts: Dict[str, Alert] = {}
		self.alert_rules: Dict[str, AlertRule] = {}
		self.notification_templates: Dict[str, NotificationTemplate] = {}
		
		# Background tasks
		self._alerting_tasks: List[asyncio.Task] = []
		self._shutdown_event = asyncio.Event()
		
		self._log_info("Alerting service initialized")
	
	async def initialize(self) -> None:
		"""Initialize alerting service"""
		try:
			# Initialize Redis connection
			self.redis_client = await aioredis.from_url(
				"redis://localhost:6379",
				encoding="utf-8",
				decode_responses=True
			)
			
			# Initialize components
			await self.alert_processor.initialize(self.redis_client)
			await self.notification_dispatcher.initialize(self.redis_client)
			await self.escalation_manager.initialize(self.redis_client)
			await self.template_manager.initialize()
			
			# Initialize integrations
			await self.notification_service.initialize()
			
			if self.slack_client:
				await self.slack_client.initialize()
			
			if self.teams_client:
				await self.teams_client.initialize()
			
			if self.pagerduty_client:
				await self.pagerduty_client.initialize()
			
			# Load configuration
			await self._load_alert_rules()
			await self._load_notification_templates()
			
			# Start background processing
			await self._start_alerting_tasks()
			
			self._log_info("Alerting service initialized successfully")
			
		except Exception as e:
			self._log_error(f"Failed to initialize alerting service: {e}")
			raise
	
	async def _start_alerting_tasks(self) -> None:
		"""Start background alerting tasks"""
		tasks = [
			self._alert_evaluation_task(),
			self._notification_processing_task(),
			self._escalation_processing_task(),
			self._alert_cleanup_task()
		]
		
		for task_coro in tasks:
			task = asyncio.create_task(task_coro)
			self._alerting_tasks.append(task)
		
		self._log_info(f"Started {len(self._alerting_tasks)} alerting tasks")
	
	async def _alert_evaluation_task(self) -> None:
		"""Background task for evaluating alert conditions"""
		while not self._shutdown_event.is_set():
			try:
				await self.alert_processor.evaluate_alert_rules(self.alert_rules)
				await asyncio.sleep(self.config.evaluation_interval_seconds)
			except Exception as e:
				self._log_error(f"Error in alert evaluation task: {e}")
				await asyncio.sleep(10)
	
	async def _notification_processing_task(self) -> None:
		"""Background task for processing notifications"""
		while not self._shutdown_event.is_set():
			try:
				await self.notification_dispatcher.process_notification_queue()
				await asyncio.sleep(5)
			except Exception as e:
				self._log_error(f"Error in notification processing task: {e}")
				await asyncio.sleep(10)
	
	async def _escalation_processing_task(self) -> None:
		"""Background task for processing escalations"""
		while not self._shutdown_event.is_set():
			try:
				await self.escalation_manager.process_escalations(self.active_alerts)
				await asyncio.sleep(60)  # Check escalations every minute
			except Exception as e:
				self._log_error(f"Error in escalation processing task: {e}")
				await asyncio.sleep(10)
	
	async def _alert_cleanup_task(self) -> None:
		"""Background task for cleaning up old alerts"""
		while not self._shutdown_event.is_set():
			try:
				await self._cleanup_old_alerts()
				await asyncio.sleep(3600)  # Run cleanup every hour
			except Exception as e:
				self._log_error(f"Error in alert cleanup task: {e}")
				await asyncio.sleep(10)
	
	async def create_alert_rule(self, rule: AlertRule) -> None:
		"""Create a new alert rule"""
		try:
			self.alert_rules[rule.id] = rule
			
			# Store in Redis
			await self._store_alert_rule(rule)
			
			self._log_info(f"Created alert rule: {rule.name} ({rule.id})")
			
		except Exception as e:
			self._log_error(f"Failed to create alert rule: {e}")
			raise
	
	async def update_alert_rule(self, rule_id: str, rule: AlertRule) -> None:
		"""Update an existing alert rule"""
		try:
			if rule_id not in self.alert_rules:
				raise ValueError(f"Alert rule {rule_id} not found")
			
			rule.updated_at = datetime.utcnow()
			self.alert_rules[rule_id] = rule
			
			# Store in Redis
			await self._store_alert_rule(rule)
			
			self._log_info(f"Updated alert rule: {rule.name} ({rule_id})")
			
		except Exception as e:
			self._log_error(f"Failed to update alert rule: {e}")
			raise
	
	async def delete_alert_rule(self, rule_id: str) -> None:
		"""Delete an alert rule"""
		try:
			if rule_id not in self.alert_rules:
				raise ValueError(f"Alert rule {rule_id} not found")
			
			del self.alert_rules[rule_id]
			
			# Remove from Redis
			if self.redis_client:
				await self.redis_client.delete(f"alert:rule:{rule_id}")
			
			self._log_info(f"Deleted alert rule: {rule_id}")
			
		except Exception as e:
			self._log_error(f"Failed to delete alert rule: {e}")
			raise
	
	async def fire_alert(self, alert: Alert) -> None:
		"""Fire a new alert"""
		try:
			# Check if alert already exists and is active
			existing_alert = self.active_alerts.get(alert.id)
			if existing_alert and existing_alert.status == AlertStatus.ACTIVE:
				# Update existing alert
				existing_alert.current_value = alert.current_value
				existing_alert.updated_at = datetime.utcnow()
				existing_alert.notification_count += 1
				alert = existing_alert
			else:
				# New alert
				self.active_alerts[alert.id] = alert
			
			# Store in Redis
			await self._store_alert(alert)
			
			# Queue notification
			await self.notification_dispatcher.queue_notification(alert)
			
			self._log_info(f"Fired alert: {alert.title} ({alert.id})")
			
		except Exception as e:
			self._log_error(f"Failed to fire alert: {e}")
			raise
	
	async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, note: Optional[str] = None) -> None:
		"""Acknowledge an alert"""
		try:
			alert = self.active_alerts.get(alert_id)
			if not alert:
				raise ValueError(f"Alert {alert_id} not found")
			
			alert.status = AlertStatus.ACKNOWLEDGED
			alert.acknowledged_at = datetime.utcnow()
			alert.updated_at = datetime.utcnow()
			
			if note:
				alert.annotations["acknowledgment_note"] = note
			alert.annotations["acknowledged_by"] = acknowledged_by
			
			# Store in Redis
			await self._store_alert(alert)
			
			# Send acknowledgment notification
			await self._send_acknowledgment_notification(alert, acknowledged_by, note)
			
			self._log_info(f"Acknowledged alert: {alert.title} ({alert_id}) by {acknowledged_by}")
			
		except Exception as e:
			self._log_error(f"Failed to acknowledge alert: {e}")
			raise
	
	async def resolve_alert(self, alert_id: str, resolved_by: str, note: Optional[str] = None) -> None:
		"""Resolve an alert"""
		try:
			alert = self.active_alerts.get(alert_id)
			if not alert:
				raise ValueError(f"Alert {alert_id} not found")
			
			alert.status = AlertStatus.RESOLVED
			alert.resolved_at = datetime.utcnow()
			alert.updated_at = datetime.utcnow()
			
			if note:
				alert.annotations["resolution_note"] = note
			alert.annotations["resolved_by"] = resolved_by
			
			# Store in Redis
			await self._store_alert(alert)
			
			# Remove from active alerts
			del self.active_alerts[alert_id]
			
			# Send resolution notification
			await self._send_resolution_notification(alert, resolved_by, note)
			
			self._log_info(f"Resolved alert: {alert.title} ({alert_id}) by {resolved_by}")
			
		except Exception as e:
			self._log_error(f"Failed to resolve alert: {e}")
			raise
	
	async def get_active_alerts(self, tenant_id: Optional[str] = None, 
							   severity: Optional[AlertSeverity] = None) -> List[Alert]:
		"""Get active alerts"""
		try:
			alerts = list(self.active_alerts.values())
			
			if tenant_id:
				alerts = [a for a in alerts if a.tenant_id == tenant_id]
			
			if severity:
				alerts = [a for a in alerts if a.severity == severity]
			
			return sorted(alerts, key=lambda x: x.created_at, reverse=True)
			
		except Exception as e:
			self._log_error(f"Failed to get active alerts: {e}")
			return []
	
	async def get_alert_history(self, tenant_id: Optional[str] = None,
							   start_time: Optional[datetime] = None,
							   end_time: Optional[datetime] = None,
							   limit: int = 100) -> List[Alert]:
		"""Get alert history"""
		try:
			# This would typically query a database or time-series store
			# For now, we'll return a simplified implementation
			alerts = []
			
			if self.redis_client:
				keys = await self.redis_client.keys("alert:history:*")
				for key in keys[:limit]:
					alert_data = await self.redis_client.hgetall(key)
					if alert_data:
						alert = self._deserialize_alert(alert_data)
						if tenant_id and alert.tenant_id != tenant_id:
							continue
						if start_time and alert.created_at < start_time:
							continue
						if end_time and alert.created_at > end_time:
							continue
						alerts.append(alert)
			
			return sorted(alerts, key=lambda x: x.created_at, reverse=True)
			
		except Exception as e:
			self._log_error(f"Failed to get alert history: {e}")
			return []
	
	async def _load_alert_rules(self) -> None:
		"""Load alert rules from storage"""
		try:
			if not self.redis_client:
				return
			
			keys = await self.redis_client.keys("alert:rule:*")
			for key in keys:
				rule_data = await self.redis_client.hgetall(key)
				if rule_data:
					rule = self._deserialize_alert_rule(rule_data)
					self.alert_rules[rule.id] = rule
			
			self._log_info(f"Loaded {len(self.alert_rules)} alert rules")
			
		except Exception as e:
			self._log_error(f"Failed to load alert rules: {e}")
	
	async def _load_notification_templates(self) -> None:
		"""Load notification templates"""
		try:
			# Load default templates
			default_templates = [
				NotificationTemplate(
					id="default_alert_email",
					name="Default Alert Email",
					channel=NotificationChannel.EMAIL,
					subject_template="[{severity}] {title}",
					body_template="""
Alert: {title}
Severity: {severity}
Source: {source}
Description: {description}
Current Value: {current_value}
Threshold: {threshold_value}
Time: {created_at}

View in dashboard: {dashboard_url}
					""".strip(),
					variables=["severity", "title", "source", "description", "current_value", "threshold_value", "created_at", "dashboard_url"]
				),
				NotificationTemplate(
					id="default_alert_slack",
					name="Default Alert Slack",
					channel=NotificationChannel.SLACK,
					subject_template="Alert: {title}",
					body_template="""
:warning: *{severity.upper()} Alert*

*{title}*
{description}

*Details:*
• Source: {source}
• Current Value: {current_value}
• Threshold: {threshold_value}
• Time: {created_at}
					""".strip(),
					format_type="markdown",
					variables=["severity", "title", "source", "description", "current_value", "threshold_value", "created_at"]
				)
			]
			
			for template in default_templates:
				self.notification_templates[template.id] = template
			
			self._log_info(f"Loaded {len(self.notification_templates)} notification templates")
			
		except Exception as e:
			self._log_error(f"Failed to load notification templates: {e}")
	
	async def _store_alert_rule(self, rule: AlertRule) -> None:
		"""Store alert rule in Redis"""
		try:
			if not self.redis_client:
				return
			
			key = f"alert:rule:{rule.id}"
			data = rule.model_dump()
			
			# Serialize complex fields
			serialized_data = {}
			for k, v in data.items():
				if isinstance(v, (list, dict)):
					serialized_data[k] = json.dumps(v, default=str)
				elif isinstance(v, datetime):
					serialized_data[k] = v.isoformat()
				else:
					serialized_data[k] = str(v)
			
			await self.redis_client.hset(key, mapping=serialized_data)
			
		except Exception as e:
			self._log_error(f"Failed to store alert rule: {e}")
	
	async def _store_alert(self, alert: Alert) -> None:
		"""Store alert in Redis"""
		try:
			if not self.redis_client:
				return
			
			# Store in active alerts
			key = f"alert:active:{alert.id}"
			data = alert.model_dump()
			
			# Serialize complex fields
			serialized_data = {}
			for k, v in data.items():
				if isinstance(v, (list, dict)):
					serialized_data[k] = json.dumps(v, default=str)
				elif isinstance(v, datetime):
					serialized_data[k] = v.isoformat()
				else:
					serialized_data[k] = str(v)
			
			await self.redis_client.hset(key, mapping=serialized_data)
			
			# Also store in history
			history_key = f"alert:history:{alert.id}:{int(alert.created_at.timestamp())}"
			await self.redis_client.hset(history_key, mapping=serialized_data)
			
			# Set expiration for history
			await self.redis_client.expire(history_key, self.config.alert_retention_days * 24 * 3600)
			
		except Exception as e:
			self._log_error(f"Failed to store alert: {e}")
	
	def _deserialize_alert_rule(self, data: Dict[str, str]) -> AlertRule:
		"""Deserialize alert rule from Redis data"""
		try:
			parsed_data = {}
			for k, v in data.items():
				if k in ['conditions', 'notification_channels', 'recipients', 'escalation_rules', 'labels', 'filters']:
					parsed_data[k] = json.loads(v)
				elif k in ['created_at', 'updated_at']:
					parsed_data[k] = datetime.fromisoformat(v)
				elif k == 'enabled':
					parsed_data[k] = v.lower() == 'true'
				else:
					parsed_data[k] = v
			
			return AlertRule(**parsed_data)
			
		except Exception as e:
			self._log_error(f"Failed to deserialize alert rule: {e}")
			raise
	
	def _deserialize_alert(self, data: Dict[str, str]) -> Alert:
		"""Deserialize alert from Redis data"""
		try:
			parsed_data = {}
			for k, v in data.items():
				if k in ['notifications_sent', 'labels', 'annotations', 'related_alerts']:
					parsed_data[k] = json.loads(v)
				elif k in ['created_at', 'updated_at', 'acknowledged_at', 'resolved_at', 'escalated_at']:
					parsed_data[k] = datetime.fromisoformat(v) if v and v != 'None' else None
				elif k in ['current_value', 'threshold_value']:
					parsed_data[k] = float(v)
				elif k == 'notification_count':
					parsed_data[k] = int(v)
				else:
					parsed_data[k] = v
			
			return Alert(**parsed_data)
			
		except Exception as e:
			self._log_error(f"Failed to deserialize alert: {e}")
			raise
	
	async def _send_acknowledgment_notification(self, alert: Alert, acknowledged_by: str, note: Optional[str]) -> None:
		"""Send acknowledgment notification"""
		try:
			message = f"Alert '{alert.title}' has been acknowledged by {acknowledged_by}"
			if note:
				message += f"\nNote: {note}"
			
			# Send to configured channels
			rule = self.alert_rules.get(alert.rule_id)
			if rule:
				for channel in rule.notification_channels:
					await self.notification_dispatcher.send_notification(
						channel, rule.recipients, "Alert Acknowledged", message
					)
		
		except Exception as e:
			self._log_error(f"Failed to send acknowledgment notification: {e}")
	
	async def _send_resolution_notification(self, alert: Alert, resolved_by: str, note: Optional[str]) -> None:
		"""Send resolution notification"""
		try:
			message = f"Alert '{alert.title}' has been resolved by {resolved_by}"
			if note:
				message += f"\nNote: {note}"
			
			# Send to configured channels
			rule = self.alert_rules.get(alert.rule_id)
			if rule:
				for channel in rule.notification_channels:
					await self.notification_dispatcher.send_notification(
						channel, rule.recipients, "Alert Resolved", message
					)
		
		except Exception as e:
			self._log_error(f"Failed to send resolution notification: {e}")
	
	async def _cleanup_old_alerts(self) -> None:
		"""Clean up old alerts"""
		try:
			if not self.redis_client:
				return
			
			# Clean up resolved alerts older than retention period
			cutoff_date = datetime.utcnow() - timedelta(days=self.config.resolved_alert_retention_days)
			
			# Clean up active alerts
			active_keys = await self.redis_client.keys("alert:active:*")
			for key in active_keys:
				alert_data = await self.redis_client.hgetall(key)
				if alert_data:
					try:
						resolved_at = alert_data.get('resolved_at')
						if resolved_at and resolved_at != 'None':
							resolved_time = datetime.fromisoformat(resolved_at)
							if resolved_time < cutoff_date:
								await self.redis_client.delete(key)
					except ValueError:
						continue
			
			# Clean up old history entries
			history_cutoff = datetime.utcnow() - timedelta(days=self.config.alert_retention_days)
			history_keys = await self.redis_client.keys("alert:history:*")
			
			deleted_count = 0
			for key in history_keys:
				try:
					# Extract timestamp from key
					timestamp_str = key.split(":")[-1]
					timestamp = datetime.fromtimestamp(int(timestamp_str))
					if timestamp < history_cutoff:
						await self.redis_client.delete(key)
						deleted_count += 1
				except (ValueError, IndexError):
					continue
			
			if deleted_count > 0:
				self._log_info(f"Cleaned up {deleted_count} old alert history entries")
			
		except Exception as e:
			self._log_error(f"Failed to cleanup old alerts: {e}")
	
	async def shutdown(self) -> None:
		"""Shutdown alerting service"""
		try:
			self._log_info("Shutting down alerting service...")
			
			# Signal shutdown to background tasks
			self._shutdown_event.set()
			
			# Wait for tasks to complete
			if self._alerting_tasks:
				await asyncio.gather(*self._alerting_tasks, return_exceptions=True)
			
			# Shutdown components
			await self.alert_processor.shutdown()
			await self.notification_dispatcher.shutdown()
			await self.escalation_manager.shutdown()
			await self.template_manager.shutdown()
			
			# Close connections
			if self.redis_client:
				await self.redis_client.close()
			
			await self.notification_service.shutdown()
			
			if self.slack_client:
				await self.slack_client.shutdown()
			
			if self.teams_client:
				await self.teams_client.shutdown()
			
			if self.pagerduty_client:
				await self.pagerduty_client.shutdown()
			
			self._log_info("Alerting service shutdown completed")
			
		except Exception as e:
			self._log_error(f"Error during alerting service shutdown: {e}")


class AlertProcessor:
	"""Processes alert conditions and triggers alerts"""
	
	def __init__(self, config: AlertingConfig):
		self.config = config
		self.redis_client: Optional[aioredis.Redis] = None
		self.logger = logging.getLogger(f"{__name__}.AlertProcessor")
	
	async def initialize(self, redis_client: aioredis.Redis) -> None:
		"""Initialize alert processor"""
		self.redis_client = redis_client
		self.logger.info("Alert processor initialized")
	
	async def evaluate_alert_rules(self, alert_rules: Dict[str, AlertRule]) -> None:
		"""Evaluate all alert rules"""
		try:
			for rule in alert_rules.values():
				if rule.enabled:
					await self._evaluate_rule(rule)
		except Exception as e:
			self.logger.error(f"Failed to evaluate alert rules: {e}")
	
	async def _evaluate_rule(self, rule: AlertRule) -> None:
		"""Evaluate a single alert rule"""
		try:
			# Get current metrics for this rule's query
			current_time = datetime.utcnow()
			
			# Execute the rule's metric query
			if hasattr(self, 'database') and self.database:
				# Parse the rule query - simplified implementation
				metric_value = await self._execute_rule_query(rule.query)
			else:
				# Fallback: use monitoring system
				from .monitoring import WorkflowMonitoring
				monitoring = WorkflowMonitoring(self.database, self.config, self.tenant_id)
				metric_value = await self._get_metric_from_monitoring(monitoring, rule)
			
			if metric_value is None:
				self.logger.warning(f"Could not get metric value for rule {rule.id}")
				return
			
			# Evaluate the threshold condition
			should_trigger = False
			if rule.condition == "greater_than" and metric_value > rule.threshold:
				should_trigger = True
			elif rule.condition == "less_than" and metric_value < rule.threshold:
				should_trigger = True
			elif rule.condition == "equal_to" and abs(metric_value - rule.threshold) < 0.001:
				should_trigger = True
			elif rule.condition == "not_equal_to" and abs(metric_value - rule.threshold) >= 0.001:
				should_trigger = True
			elif rule.condition == "greater_equal" and metric_value >= rule.threshold:
				should_trigger = True
			elif rule.condition == "less_equal" and metric_value <= rule.threshold:
				should_trigger = True
			
			# Check if we should trigger an alert
			if should_trigger:
				# Check if this alert is already active to avoid spam
				existing_alert = await self._get_active_alert(rule.id)
				
				if not existing_alert:
					# Create new alert
					alert = Alert(
						id=f"alert_{rule.id}_{int(current_time.timestamp())}",
						rule_id=rule.id,
						rule_name=rule.name,
						severity=rule.severity,
						status=AlertStatus.TRIGGERED,
						triggered_at=current_time,
						message=f"{rule.name}: Metric value {metric_value} {rule.condition} threshold {rule.threshold}",
						current_value=metric_value,
						threshold_value=rule.threshold,
						tenant_id=self.tenant_id
					)
					
					# Store the alert
					await self._store_alert(alert)
					
					# Send notifications
					await self._send_alert_notifications(alert)
					
					self.logger.info(f"Alert triggered for rule {rule.id}: {alert.message}")
				else:
					# Update existing alert timestamp
					existing_alert.last_triggered_at = current_time
					existing_alert.current_value = metric_value
					await self._update_alert(existing_alert)
			else:
				# Check if there's an active alert that should be resolved
				existing_alert = await self._get_active_alert(rule.id)
				if existing_alert and existing_alert.status == AlertStatus.TRIGGERED:
					existing_alert.status = AlertStatus.RESOLVED
					existing_alert.resolved_at = current_time
					await self._update_alert(existing_alert)
					
					self.logger.info(f"Alert resolved for rule {rule.id}")
			
		except Exception as e:
			self.logger.error(f"Failed to evaluate rule {rule.id}: {e}")
	
	async def _execute_rule_query(self, query: str) -> Optional[float]:
		"""Execute a rule query and return metric value"""
		try:
			# Simple query parsing for common metric queries
			if "workflow_instances" in query and "COUNT" in query.upper():
				# Count workflow instances
				count_query = """
				SELECT COUNT(*) as count
				FROM cr_workflow_instances 
				WHERE tenant_id = %s
				AND started_at >= NOW() - INTERVAL '1 hour'
				"""
				result = await self.database.fetch_one(count_query, (self.tenant_id,))
				return float(result['count']) if result else 0.0
				
			elif "avg_duration" in query:
				# Average workflow duration
				duration_query = """
				SELECT AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration
				FROM cr_workflow_instances 
				WHERE tenant_id = %s
				AND completed_at IS NOT NULL
				AND started_at >= NOW() - INTERVAL '1 hour'
				"""
				result = await self.database.fetch_one(duration_query, (self.tenant_id,))
				return float(result['avg_duration']) if result and result['avg_duration'] else 0.0
				
			elif "failed_workflows" in query:
				# Count failed workflows
				failed_query = """
				SELECT COUNT(*) as count
				FROM cr_workflow_instances 
				WHERE tenant_id = %s
				AND status = 'failed'
				AND started_at >= NOW() - INTERVAL '1 hour'
				"""
				result = await self.database.fetch_one(failed_query, (self.tenant_id,))
				return float(result['count']) if result else 0.0
			
			# Default: try to execute the query directly (advanced users)
			try:
				result = await self.database.fetch_one(query)
				if result:
					# Get the first numeric value from the result
					for value in result.values():
						if isinstance(value, (int, float)):
							return float(value)
			except Exception:
				pass  # Query failed, return None
			
			return None
			
		except Exception as e:
			self.logger.error(f"Failed to execute rule query: {e}")
			return None
	
	async def _get_metric_from_monitoring(self, monitoring, rule: AlertRule) -> Optional[float]:
		"""Get metric value from monitoring system"""
		try:
			# Get recent metrics and find matching ones
			metrics = await monitoring.get_recent_metrics(limit=100)
			
			# Simple matching based on rule name/query
			for metric in metrics:
				if rule.name.lower() in metric.get('name', '').lower():
					return float(metric.get('value', 0))
			
			return None
		except Exception as e:
			self.logger.error(f"Failed to get metric from monitoring: {e}")
			return None
	
	async def shutdown(self) -> None:
		"""Shutdown alert processor"""
		self.logger.info("Alert processor shutting down")


class NotificationDispatcher:
	"""Dispatches notifications through various channels"""
	
	def __init__(self, config: AlertingConfig):
		self.config = config
		self.redis_client: Optional[aioredis.Redis] = None
		self.logger = logging.getLogger(f"{__name__}.NotificationDispatcher")
		self.notification_queue: asyncio.Queue = asyncio.Queue()
	
	async def initialize(self, redis_client: aioredis.Redis) -> None:
		"""Initialize notification dispatcher"""
		self.redis_client = redis_client
		self.logger.info("Notification dispatcher initialized")
	
	async def queue_notification(self, alert: Alert) -> None:
		"""Queue a notification for processing"""
		try:
			await self.notification_queue.put(alert)
		except Exception as e:
			self.logger.error(f"Failed to queue notification: {e}")
	
	async def process_notification_queue(self) -> None:
		"""Process the notification queue"""
		try:
			while not self.notification_queue.empty():
				try:
					alert = await asyncio.wait_for(self.notification_queue.get(), timeout=1.0)
					await self._send_alert_notifications(alert)
				except asyncio.TimeoutError:
					break
				except Exception as e:
					self.logger.error(f"Failed to process notification: {e}")
		except Exception as e:
			self.logger.error(f"Error in notification queue processing: {e}")
	
	async def _send_alert_notifications(self, alert: Alert) -> None:
		"""Send notifications for an alert"""
		try:
			# Get notification channels for this alert's severity
			channels = await self._get_notification_channels(alert.severity)
			
			if not channels:
				self.logger.warning(f"No notification channels configured for severity {alert.severity}")
				return
			
			# Prepare notification content
			subject = f"[{alert.severity.value.upper()}] {alert.rule_name}"
			message = f"""
			Alert: {alert.rule_name}
			Severity: {alert.severity.value}
			Message: {alert.message}
			Current Value: {alert.current_value}
			Threshold: {alert.threshold_value}
			Triggered At: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
			Alert ID: {alert.id}
			"""
			
			# Send notifications through all configured channels
			for channel in channels:
				try:
					if channel.get('type') == "email":
						await self._send_email_notification(channel.get('recipients', []), subject, message)
					elif channel.get('type') == "slack":
						await self._send_slack_notification(channel.get('recipients', []), subject, message)
					elif channel.get('type') == "teams":
						await self._send_teams_notification(channel.get('recipients', []), subject, message)
					else:
						self.logger.warning(f"Unknown notification channel type: {channel.get('type')}")
				except Exception as channel_error:
					self.logger.error(f"Failed to send notification via {channel.get('type')}: {channel_error}")
					continue
			
			self.logger.info(f"Sent alert notifications for {alert.id} via {len(channels)} channels")
			
		except Exception as e:
			self.logger.error(f"Failed to send alert notifications: {e}")
	
	async def send_notification(self, channel: NotificationChannel, recipients: List[str], 
							   subject: str, message: str) -> None:
		"""Send a notification through a specific channel"""
		try:
			if channel == NotificationChannel.EMAIL:
				await self._send_email_notification(recipients, subject, message)
			elif channel == NotificationChannel.SLACK:
				await self._send_slack_notification(recipients, subject, message)
			elif channel == NotificationChannel.TEAMS:
				await self._send_teams_notification(recipients, subject, message)
			elif channel == NotificationChannel.WEBHOOK:
				await self._send_webhook_notification(recipients, subject, message)
			
		except Exception as e:
			self.logger.error(f"Failed to send {channel} notification: {e}")
	
	async def _send_email_notification(self, recipients: List[str], subject: str, message: str) -> None:
		"""Send email notification"""
		try:
			msg = MimeMultipart()
			msg['From'] = self.config.default_sender
			msg['To'] = ', '.join(recipients)
			msg['Subject'] = subject
			
			msg.attach(MimeText(message, 'plain'))
			
			context = ssl.create_default_context()
			with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
				if self.config.smtp_use_tls:
					server.starttls(context=context)
				if self.config.smtp_username and self.config.smtp_password:
					server.login(self.config.smtp_username, self.config.smtp_password)
				server.send_message(msg)
			
			self.logger.debug(f"Sent email notification to {len(recipients)} recipients")
			
		except Exception as e:
			self.logger.error(f"Failed to send email notification: {e}")
	
	async def _send_slack_notification(self, recipients: List[str], subject: str, message: str) -> None:
		"""Send Slack notification"""
		try:
			if not self.config.slack_webhook_url:
				self.logger.debug("Slack webhook URL not configured, skipping Slack notification")
				return
			
			import aiohttp
			
			# Format Slack message
			slack_payload = {
				"text": f"*{subject}*",
				"attachments": [{
					"color": "warning",
					"text": message,
					"footer": "Workflow Orchestration Alerts",
					"ts": int(datetime.utcnow().timestamp())
				}]
			}
			
			# Send to Slack webhook
			async with aiohttp.ClientSession() as session:
				async with session.post(self.config.slack_webhook_url, json=slack_payload) as response:
					if response.status == 200:
						self.logger.debug(f"Sent Slack notification to {len(recipients)} recipients")
					else:
						self.logger.error(f"Slack webhook returned status {response.status}")
			
		except Exception as e:
			self.logger.error(f"Failed to send Slack notification: {e}")
	
	async def _send_teams_notification(self, recipients: List[str], subject: str, message: str) -> None:
		"""Send Teams notification"""
		try:
			if not self.config.teams_webhook_url:
				self.logger.debug("Teams webhook URL not configured, skipping Teams notification")
				return
			
			import aiohttp
			
			# Format Teams message
			teams_payload = {
				"@type": "MessageCard",
				"@context": "https://schema.org/extensions",
				"summary": subject,
				"themeColor": "FFA500",  # Orange
				"sections": [{
					"activityTitle": subject,
					"activitySubtitle": "Workflow Orchestration Alert",
					"text": message,
					"markdown": True
				}]
			}
			
			# Send to Teams webhook
			async with aiohttp.ClientSession() as session:
				async with session.post(self.config.teams_webhook_url, json=teams_payload) as response:
					if response.status == 200:
						self.logger.debug(f"Sent Teams notification to {len(recipients)} recipients")
					else:
						self.logger.error(f"Teams webhook returned status {response.status}")
			
		except Exception as e:
			self.logger.error(f"Failed to send Teams notification: {e}")
	
	async def _send_webhook_notification(self, recipients: List[str], subject: str, message: str) -> None:
		"""Send webhook notification"""
		try:
			for webhook_url in recipients:
				payload = {
					"subject": subject,
					"message": message,
					"timestamp": datetime.utcnow().isoformat()
				}
				
				async with aiohttp.ClientSession() as session:
					async with session.post(
						webhook_url,
						json=payload,
						timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout_seconds)
					) as response:
						if response.status >= 400:
							self.logger.warning(f"Webhook notification failed: {response.status}")
			
		except Exception as e:
			self.logger.error(f"Failed to send webhook notification: {e}")
	
	async def shutdown(self) -> None:
		"""Shutdown notification dispatcher"""
		self.logger.info("Notification dispatcher shutting down")


class EscalationManager:
	"""Manages alert escalation"""
	
	def __init__(self, config: AlertingConfig):
		self.config = config
		self.redis_client: Optional[aioredis.Redis] = None
		self.logger = logging.getLogger(f"{__name__}.EscalationManager")
	
	async def initialize(self, redis_client: aioredis.Redis) -> None:
		"""Initialize escalation manager"""
		self.redis_client = redis_client
		self.logger.info("Escalation manager initialized")
	
	async def process_escalations(self, active_alerts: Dict[str, Alert]) -> None:
		"""Process escalations for active alerts"""
		try:
			for alert in active_alerts.values():
				if alert.status == AlertStatus.ACTIVE:
					await self._check_escalation(alert)
		except Exception as e:
			self.logger.error(f"Failed to process escalations: {e}")
	
	async def _check_escalation(self, alert: Alert) -> None:
		"""Check if an alert needs escalation"""
		try:
			current_time = datetime.utcnow()
			
			# Check if alert has been triggered long enough for escalation
			time_since_trigger = (current_time - alert.triggered_at).total_seconds()
			
			# Simple escalation logic based on severity and time
			escalation_thresholds = {
				AlertSeverity.CRITICAL: [5*60, 15*60, 30*60],  # 5, 15, 30 minutes
				AlertSeverity.WARNING: [15*60, 60*60],          # 15 minutes, 1 hour
				AlertSeverity.INFO: [60*60]                     # 1 hour
			}
			
			thresholds = escalation_thresholds.get(alert.severity, [])
			
			for level, threshold_seconds in enumerate(thresholds, 1):
				if time_since_trigger >= threshold_seconds:
					# Check if this escalation level has already been triggered
					escalation_key = f"escalation_{alert.id}_level_{level}"
					
					# Use Redis to track escalations
					if self.redis_client:
						escalation_sent = await self.redis_client.get(escalation_key)
						if not escalation_sent:
							# Send escalated notification
							escalated_subject = f"[ESCALATED L{level}] {alert.rule_name}"
							escalated_message = f"""
							ESCALATED ALERT - Level {level}
							
							Original Alert: {alert.rule_name}
							Severity: {alert.severity.value}
							Message: {alert.message}
							Current Value: {alert.current_value}
							Threshold: {alert.threshold_value}
							Originally Triggered: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
							Unresolved for: {int(time_since_trigger/60)} minutes
							Alert ID: {alert.id}
							
							This alert has been escalated due to lack of resolution.
							"""
							
							# Get escalation channels (could be different from regular channels)
							escalation_channels = await self._get_escalation_channels(alert.severity, level)
							
							for channel in escalation_channels:
								try:
									if channel.get('type') == "email":
										await self._send_email_notification(
											channel.get('recipients', []), 
											escalated_subject, 
											escalated_message
										)
									elif channel.get('type') == "slack":
										await self._send_slack_notification(
											channel.get('recipients', []), 
											escalated_subject, 
											escalated_message
										)
								except Exception as channel_error:
									self.logger.error(f"Failed to send escalation notification: {channel_error}")
							
							# Mark this escalation level as sent
							await self.redis_client.setex(escalation_key, 3600*24, "sent")  # 24 hour expiry
							
							self.logger.warning(
								f"Alert {alert.id} escalated to level {level} after {int(time_since_trigger/60)} minutes"
							)
			
		except Exception as e:
			self.logger.error(f"Failed to check escalation for alert {alert.id}: {e}")
	
	async def _get_notification_channels(self, severity: AlertSeverity) -> List[Dict[str, Any]]:
		"""Get notification channels for a given severity"""
		try:
			# Default notification channels - in production this would come from config/database
			default_email_recipients = ["alerts@company.com"]
			
			channels = []
			
			# Always send email for any severity
			channels.append({
				"type": "email",
				"recipients": default_email_recipients
			})
			
			# Add Slack for WARNING and CRITICAL
			if severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
				if self.config.slack_webhook_url:
					channels.append({
						"type": "slack",
						"recipients": ["#alerts"]
					})
			
			# Add Teams for CRITICAL
			if severity == AlertSeverity.CRITICAL:
				if self.config.teams_webhook_url:
					channels.append({
						"type": "teams",
						"recipients": ["Leadership Team"]
					})
			
			return channels
			
		except Exception as e:
			self.logger.error(f"Failed to get notification channels: {e}")
			return []
	
	async def _get_escalation_channels(self, severity: AlertSeverity, level: int) -> List[Dict[str, Any]]:
		"""Get escalation channels for a given severity and level"""
		try:
			# Escalation channels - typically more senior/urgent channels
			channels = []
			
			if level == 1:
				# Level 1: Add team leads
				channels.append({
					"type": "email",
					"recipients": ["team-leads@company.com"]
				})
			elif level == 2:
				# Level 2: Add management
				channels.append({
					"type": "email",
					"recipients": ["management@company.com"]
				})
			elif level >= 3:
				# Level 3+: Add executives
				channels.append({
					"type": "email",
					"recipients": ["executives@company.com"]
				})
			
			return channels
			
		except Exception as e:
			self.logger.error(f"Failed to get escalation channels: {e}")
			return []
	
	async def shutdown(self) -> None:
		"""Shutdown escalation manager"""
		self.logger.info("Escalation manager shutting down")


class TemplateManager:
	"""Manages notification templates"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.TemplateManager")
	
	async def initialize(self) -> None:
		"""Initialize template manager"""
		self.logger.info("Template manager initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown template manager"""
		self.logger.info("Template manager shutting down")