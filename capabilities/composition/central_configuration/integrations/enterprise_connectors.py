"""
APG Central Configuration - Enterprise Integration Connectors

Comprehensive enterprise integration connectors for popular platforms
including Slack, Microsoft Teams, ServiceNow, JIRA, and more.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import hmac
import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import httpx
from urllib.parse import urlencode

from ..service import CentralConfigurationEngine


class IntegrationType(Enum):
	"""Types of enterprise integrations."""
	MESSAGING = "messaging"
	TICKETING = "ticketing"
	MONITORING = "monitoring"
	CI_CD = "ci_cd"
	IDENTITY = "identity"
	COMPLIANCE = "compliance"
	WORKFLOW = "workflow"


class EventType(Enum):
	"""Types of events to send to integrations."""
	CONFIGURATION_CREATED = "configuration_created"
	CONFIGURATION_UPDATED = "configuration_updated"
	CONFIGURATION_DELETED = "configuration_deleted"
	SECURITY_ALERT = "security_alert"
	COMPLIANCE_VIOLATION = "compliance_violation"
	PERFORMANCE_ALERT = "performance_alert"
	SYSTEM_HEALTH = "system_health"
	DEPLOYMENT_STATUS = "deployment_status"
	BACKUP_STATUS = "backup_status"
	AI_INSIGHT = "ai_insight"


@dataclass
class IntegrationConfig:
	"""Configuration for an enterprise integration."""
	integration_id: str
	name: str
	integration_type: IntegrationType
	platform: str
	enabled: bool
	config: Dict[str, Any]
	event_filters: List[EventType]
	rate_limit_per_minute: int
	retry_config: Dict[str, Any]
	webhook_url: Optional[str]
	authentication: Dict[str, Any]
	created_at: datetime
	last_used: Optional[datetime]


@dataclass
class IntegrationEvent:
	"""Event to be sent to integrations."""
	event_id: str
	event_type: EventType
	timestamp: datetime
	source: str
	data: Dict[str, Any]
	metadata: Dict[str, Any]
	severity: str  # low, medium, high, critical


class EnterpriseIntegrationManager:
	"""Manager for enterprise integration connectors."""
	
	def __init__(self, config_engine: CentralConfigurationEngine):
		"""Initialize integration manager."""
		self.config_engine = config_engine
		self.integrations: Dict[str, IntegrationConfig] = {}
		self.event_queue: List[IntegrationEvent] = []
		self.connectors: Dict[str, Any] = {}
		
		# Rate limiting
		self.rate_limits: Dict[str, List[datetime]] = {}
		
		# Initialize connectors
		asyncio.create_task(self._initialize_connectors())
		asyncio.create_task(self._start_event_processor())
	
	# ==================== Initialization ====================
	
	async def _initialize_connectors(self):
		"""Initialize all enterprise connectors."""
		# Messaging connectors
		self.connectors["slack"] = SlackConnector()
		self.connectors["teams"] = MicrosoftTeamsConnector()
		self.connectors["discord"] = DiscordConnector()
		
		# Ticketing connectors
		self.connectors["jira"] = JiraConnector()
		self.connectors["servicenow"] = ServiceNowConnector()
		self.connectors["zendesk"] = ZendeskConnector()
		
		# Monitoring connectors
		self.connectors["datadog"] = DatadogConnector()
		self.connectors["newrelic"] = NewRelicConnector()
		self.connectors["splunk"] = SplunkConnector()
		
		# CI/CD connectors
		self.connectors["jenkins"] = JenkinsConnector()
		self.connectors["github"] = GitHubConnector()
		self.connectors["gitlab"] = GitLabConnector()
		
		# Identity connectors
		self.connectors["okta"] = OktaConnector()
		self.connectors["auth0"] = Auth0Connector()
		self.connectors["azure_ad"] = AzureADConnector()
		
		print(f"🔌 Initialized {len(self.connectors)} enterprise connectors")
	
	# ==================== Integration Management ====================
	
	async def add_integration(
		self,
		name: str,
		platform: str,
		integration_type: IntegrationType,
		config: Dict[str, Any],
		event_filters: Optional[List[EventType]] = None
	) -> str:
		"""Add a new enterprise integration."""
		integration_id = f"integration_{uuid.uuid4().hex[:8]}"
		
		integration = IntegrationConfig(
			integration_id=integration_id,
			name=name,
			integration_type=integration_type,
			platform=platform.lower(),
			enabled=True,
			config=config,
			event_filters=event_filters or list(EventType),
			rate_limit_per_minute=60,  # Default rate limit
			retry_config={
				"max_retries": 3,
				"retry_delay_seconds": 5,
				"exponential_backoff": True
			},
			webhook_url=config.get("webhook_url"),
			authentication=config.get("authentication", {}),
			created_at=datetime.now(timezone.utc),
			last_used=None
		)
		
		# Validate configuration
		if platform.lower() in self.connectors:
			connector = self.connectors[platform.lower()]
			if hasattr(connector, 'validate_config'):
				await connector.validate_config(config)
		
		self.integrations[integration_id] = integration
		
		print(f"✅ Added integration: {name} ({platform})")
		return integration_id
	
	async def remove_integration(self, integration_id: str) -> bool:
		"""Remove an enterprise integration."""
		if integration_id in self.integrations:
			del self.integrations[integration_id]
			print(f"🗑️ Removed integration: {integration_id}")
			return True
		return False
	
	async def update_integration(
		self,
		integration_id: str,
		updates: Dict[str, Any]
	) -> bool:
		"""Update an existing integration."""
		if integration_id not in self.integrations:
			return False
		
		integration = self.integrations[integration_id]
		
		# Update allowed fields
		if "enabled" in updates:
			integration.enabled = updates["enabled"]
		if "config" in updates:
			integration.config.update(updates["config"])
		if "event_filters" in updates:
			integration.event_filters = [EventType(et) for et in updates["event_filters"]]
		if "rate_limit_per_minute" in updates:
			integration.rate_limit_per_minute = updates["rate_limit_per_minute"]
		
		print(f"📝 Updated integration: {integration.name}")
		return True
	
	# ==================== Event Processing ====================
	
	async def send_event(
		self,
		event_type: EventType,
		source: str,
		data: Dict[str, Any],
		severity: str = "medium",
		metadata: Optional[Dict[str, Any]] = None
	):
		"""Send event to all relevant integrations."""
		event = IntegrationEvent(
			event_id=f"event_{uuid.uuid4().hex[:8]}",
			event_type=event_type,
			timestamp=datetime.now(timezone.utc),
			source=source,
			data=data,
			metadata=metadata or {},
			severity=severity
		)
		
		self.event_queue.append(event)
		print(f"📨 Queued event: {event_type.value} from {source}")
	
	async def _start_event_processor(self):
		"""Start processing events from the queue."""
		print("🔄 Starting event processor")
		
		while True:
			try:
				if self.event_queue:
					event = self.event_queue.pop(0)
					await self._process_event(event)
				
				await asyncio.sleep(1)  # Process events every second
				
			except Exception as e:
				print(f"❌ Event processing error: {e}")
				await asyncio.sleep(5)
	
	async def _process_event(self, event: IntegrationEvent):
		"""Process a single event by sending to relevant integrations."""
		relevant_integrations = []
		
		# Find integrations that should receive this event
		for integration in self.integrations.values():
			if (integration.enabled and 
				event.event_type in integration.event_filters):
				relevant_integrations.append(integration)
		
		if not relevant_integrations:
			return
		
		# Send to integrations
		send_tasks = []
		for integration in relevant_integrations:
			if await self._check_rate_limit(integration.integration_id):
				task = self._send_to_integration(integration, event)
				send_tasks.append(task)
		
		if send_tasks:
			await asyncio.gather(*send_tasks, return_exceptions=True)
	
	async def _send_to_integration(
		self,
		integration: IntegrationConfig,
		event: IntegrationEvent
	):
		"""Send event to a specific integration."""
		try:
			if integration.platform in self.connectors:
				connector = self.connectors[integration.platform]
				await connector.send_event(integration, event)
				
				integration.last_used = datetime.now(timezone.utc)
				print(f"✅ Sent event to {integration.name}: {event.event_type.value}")
			else:
				print(f"⚠️ No connector for platform: {integration.platform}")
		
		except Exception as e:
			print(f"❌ Failed to send event to {integration.name}: {e}")
			
			# Retry logic
			if integration.retry_config.get("max_retries", 0) > 0:
				await self._retry_send(integration, event, 1)
	
	async def _retry_send(
		self,
		integration: IntegrationConfig,
		event: IntegrationEvent,
		attempt: int
	):
		"""Retry sending event to integration."""
		max_retries = integration.retry_config.get("max_retries", 3)
		
		if attempt > max_retries:
			print(f"❌ Max retries exceeded for {integration.name}")
			return
		
		delay = integration.retry_config.get("retry_delay_seconds", 5)
		if integration.retry_config.get("exponential_backoff", False):
			delay = delay * (2 ** (attempt - 1))
		
		await asyncio.sleep(delay)
		
		try:
			connector = self.connectors[integration.platform]
			await connector.send_event(integration, event)
			print(f"✅ Retry successful for {integration.name} (attempt {attempt})")
		except Exception as e:
			print(f"❌ Retry {attempt} failed for {integration.name}: {e}")
			await self._retry_send(integration, event, attempt + 1)
	
	async def _check_rate_limit(self, integration_id: str) -> bool:
		"""Check if integration is within rate limits."""
		if integration_id not in self.integrations:
			return False
		
		integration = self.integrations[integration_id]
		now = datetime.now(timezone.utc)
		
		# Initialize rate limit tracking
		if integration_id not in self.rate_limits:
			self.rate_limits[integration_id] = []
		
		# Clean old timestamps (older than 1 minute)
		minute_ago = now - timedelta(minutes=1)
		self.rate_limits[integration_id] = [
			ts for ts in self.rate_limits[integration_id] if ts > minute_ago
		]
		
		# Check if under rate limit
		if len(self.rate_limits[integration_id]) < integration.rate_limit_per_minute:
			self.rate_limits[integration_id].append(now)
			return True
		
		return False


# ==================== Connector Classes ====================

class BaseConnector:
	"""Base class for enterprise connectors."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Send event to the platform. Override in subclasses."""
		raise NotImplementedError
	
	async def validate_config(self, config: Dict[str, Any]) -> bool:
		"""Validate integration configuration. Override in subclasses."""
		return True
	
	def _format_message(self, event: IntegrationEvent) -> str:
		"""Format event as human-readable message."""
		severity_emoji = {
			"low": "ℹ️",
			"medium": "⚠️",
			"high": "🚨",
			"critical": "🔥"
		}
		
		emoji = severity_emoji.get(event.severity, "📢")
		
		message = f"{emoji} **{event.event_type.value.replace('_', ' ').title()}**\n"
		message += f"**Source:** {event.source}\n"
		message += f"**Time:** {event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
		message += f"**Severity:** {event.severity.upper()}\n"
		
		if event.data:
			message += "\n**Details:**\n"
			for key, value in event.data.items():
				message += f"• {key.replace('_', ' ').title()}: {value}\n"
		
		return message


class SlackConnector(BaseConnector):
	"""Slack integration connector."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Send event to Slack."""
		webhook_url = integration.config.get("webhook_url")
		if not webhook_url:
			raise ValueError("Slack webhook_url not configured")
		
		# Format message for Slack
		message = self._format_slack_message(event)
		
		# Send to Slack
		async with httpx.AsyncClient() as client:
			response = await client.post(
				webhook_url,
				json=message,
				timeout=30.0
			)
			response.raise_for_status()
	
	def _format_slack_message(self, event: IntegrationEvent) -> Dict[str, Any]:
		"""Format event as Slack message."""
		color_map = {
			"low": "#36a64f",
			"medium": "#ff9500", 
			"high": "#ff0000",
			"critical": "#8B0000"
		}
		
		attachment = {
			"color": color_map.get(event.severity, "#36a64f"),
			"title": f"{event.event_type.value.replace('_', ' ').title()}",
			"text": f"Event from {event.source}",
			"fields": [
				{
					"title": "Severity",
					"value": event.severity.upper(),
					"short": True
				},
				{
					"title": "Timestamp",
					"value": event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
					"short": True
				}
			],
			"footer": "APG Central Configuration",
			"ts": int(event.timestamp.timestamp())
		}
		
		# Add event data as fields
		for key, value in event.data.items():
			attachment["fields"].append({
				"title": key.replace('_', ' ').title(),
				"value": str(value),
				"short": len(str(value)) < 50
			})
		
		return {"attachments": [attachment]}
	
	async def validate_config(self, config: Dict[str, Any]) -> bool:
		"""Validate Slack configuration."""
		if "webhook_url" not in config:
			raise ValueError("Slack webhook_url is required")
		
		webhook_url = config["webhook_url"]
		if not webhook_url.startswith("https://hooks.slack.com/"):
			raise ValueError("Invalid Slack webhook URL")
		
		return True


class MicrosoftTeamsConnector(BaseConnector):
	"""Microsoft Teams integration connector."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Send event to Microsoft Teams."""
		webhook_url = integration.config.get("webhook_url")
		if not webhook_url:
			raise ValueError("Teams webhook_url not configured")
		
		# Format message for Teams
		message = self._format_teams_message(event)
		
		# Send to Teams
		async with httpx.AsyncClient() as client:
			response = await client.post(
				webhook_url,
				json=message,
				timeout=30.0
			)
			response.raise_for_status()
	
	def _format_teams_message(self, event: IntegrationEvent) -> Dict[str, Any]:
		"""Format event as Teams message card."""
		color_map = {
			"low": "Good",
			"medium": "Warning",
			"high": "Attention", 
			"critical": "Attention"
		}
		
		facts = [
			{"name": "Source", "value": event.source},
			{"name": "Severity", "value": event.severity.upper()},
			{"name": "Time", "value": event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
		]
		
		# Add event data as facts
		for key, value in event.data.items():
			facts.append({
				"name": key.replace('_', ' ').title(),
				"value": str(value)
			})
		
		return {
			"@type": "MessageCard",
			"@context": "https://schema.org/extensions",
			"summary": f"APG Configuration Event: {event.event_type.value}",
			"themeColor": color_map.get(event.severity, "Good"),
			"sections": [{
				"activityTitle": f"APG Central Configuration",
				"activitySubtitle": f"{event.event_type.value.replace('_', ' ').title()}",
				"facts": facts
			}]
		}


class JiraConnector(BaseConnector):
	"""JIRA integration connector."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Send event to JIRA (create ticket for high/critical events)."""
		# Only create tickets for high/critical severity events
		if event.severity not in ["high", "critical"]:
			return
		
		base_url = integration.config.get("base_url")
		username = integration.config.get("username")
		api_token = integration.config.get("api_token")
		project_key = integration.config.get("project_key")
		
		if not all([base_url, username, api_token, project_key]):
			raise ValueError("JIRA configuration incomplete")
		
		# Create JIRA ticket
		ticket_data = {
			"fields": {
				"project": {"key": project_key},
				"summary": f"APG Config Alert: {event.event_type.value.replace('_', ' ').title()}",
				"description": self._format_message(event),
				"issuetype": {"name": "Task"},
				"priority": {"name": "High" if event.severity == "critical" else "Medium"},
				"labels": ["apg-central-config", f"severity-{event.severity}"]
			}
		}
		
		auth = (username, api_token)
		url = f"{base_url}/rest/api/2/issue"
		
		async with httpx.AsyncClient() as client:
			response = await client.post(
				url,
				json=ticket_data,
				auth=auth,
				timeout=30.0
			)
			response.raise_for_status()


class ServiceNowConnector(BaseConnector):
	"""ServiceNow integration connector."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Send event to ServiceNow."""
		base_url = integration.config.get("base_url")
		username = integration.config.get("username") 
		password = integration.config.get("password")
		
		if not all([base_url, username, password]):
			raise ValueError("ServiceNow configuration incomplete")
		
		# Create incident for high/critical events
		if event.severity in ["high", "critical"]:
			incident_data = {
				"short_description": f"APG Config Alert: {event.event_type.value.replace('_', ' ').title()}",
				"description": self._format_message(event),
				"urgency": "1" if event.severity == "critical" else "2",
				"impact": "2",
				"category": "Software",
				"subcategory": "Configuration Management",
				"caller_id": username
			}
			
			auth = (username, password)
			url = f"{base_url}/api/now/table/incident"
			
			async with httpx.AsyncClient() as client:
				response = await client.post(
					url,
					json=incident_data,
					auth=auth,
					timeout=30.0
				)
				response.raise_for_status()


class DatadogConnector(BaseConnector):
	"""Datadog integration connector."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Send event to Datadog."""
		api_key = integration.config.get("api_key")
		if not api_key:
			raise ValueError("Datadog API key not configured")
		
		# Send as Datadog event
		event_data = {
			"title": f"APG Config: {event.event_type.value.replace('_', ' ').title()}",
			"text": self._format_message(event),
			"priority": "normal" if event.severity in ["low", "medium"] else "high",
			"tags": [
				f"source:{event.source}",
				f"severity:{event.severity}",
				f"event_type:{event.event_type.value}",
				"service:apg-central-config"
			],
			"alert_type": "info" if event.severity == "low" else "warning" if event.severity == "medium" else "error"
		}
		
		headers = {"DD-API-KEY": api_key}
		url = "https://api.datadoghq.com/api/v1/events"
		
		async with httpx.AsyncClient() as client:
			response = await client.post(
				url,
				json=event_data,
				headers=headers,
				timeout=30.0
			)
			response.raise_for_status()


# ==================== Complete Enterprise Connector Implementations ====================

class DiscordConnector(BaseConnector):
	"""Complete Discord webhook integration."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Send event to Discord webhook with rich embeds."""
		webhook_url = integration.config.get("webhook_url")
		if not webhook_url:
			raise ValueError("Discord webhook_url not configured")
		
		# Create Discord embed
		embed = {
			"title": self._get_event_title(event),
			"description": event.message,
			"color": self._get_discord_color(event.severity),
			"timestamp": event.timestamp.isoformat(),
			"fields": [
				{"name": "Event Type", "value": event.event_type, "inline": True},
				{"name": "Severity", "value": event.severity.value, "inline": True},
				{"name": "Source", "value": event.source_service, "inline": True}
			],
			"footer": {"text": "APG Central Configuration"}
		}
		
		if event.metadata:
			for key, value in event.metadata.items():
				if len(embed["fields"]) < 25:  # Discord limit
					embed["fields"].append({
						"name": str(key).title(),
						"value": str(value)[:1024],  # Discord field value limit
						"inline": True
					})
		
		payload = {
			"embeds": [embed],
			"username": "APG Central Config",
			"avatar_url": integration.config.get("avatar_url", "")
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				webhook_url,
				json=payload,
				headers={"Content-Type": "application/json"}
			)
			response.raise_for_status()
	
	def _get_discord_color(self, severity: EventSeverity) -> int:
		"""Get Discord embed color based on severity."""
		color_map = {
			EventSeverity.INFO: 0x3498db,      # Blue
			EventSeverity.WARNING: 0xf39c12,   # Orange
			EventSeverity.ERROR: 0xe74c3c,     # Red
			EventSeverity.CRITICAL: 0x9b59b6   # Purple
		}
		return color_map.get(severity, 0x95a5a6)  # Gray default


class ZendeskConnector(BaseConnector):
	"""Complete Zendesk ticket integration."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Create Zendesk ticket for critical events."""
		base_url = integration.config.get("base_url")
		email = integration.config.get("email")
		api_token = integration.config.get("api_token")
		
		if not all([base_url, email, api_token]):
			raise ValueError("Zendesk base_url, email, and api_token required")
		
		# Only create tickets for warnings and above
		if event.severity not in [EventSeverity.WARNING, EventSeverity.ERROR, EventSeverity.CRITICAL]:
			return
		
		# Create ticket payload
		ticket_data = {
			"ticket": {
				"subject": f"APG Alert: {self._get_event_title(event)}",
				"description": self._format_zendesk_description(event),
				"priority": self._get_zendesk_priority(event.severity),
				"type": "incident",
				"status": "new",
				"tags": ["apg", "central-config", event.event_type.lower()],
				"custom_fields": [
					{"id": integration.config.get("severity_field_id", 123456), 
					 "value": event.severity.value},
					{"id": integration.config.get("source_field_id", 123457), 
					 "value": event.source_service}
				]
			}
		}
		
		auth = httpx.BasicAuth(f"{email}/token", api_token)
		url = f"{base_url}/api/v2/tickets.json"
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				url,
				json=ticket_data,
				auth=auth,
				headers={"Content-Type": "application/json"}
			)
			response.raise_for_status()
			
		ticket_info = response.json()
		event.metadata["zendesk_ticket_id"] = ticket_info["ticket"]["id"]
	
	def _format_zendesk_description(self, event: IntegrationEvent) -> str:
		"""Format event details for Zendesk ticket description."""
		description = f"""
APG Central Configuration Alert

Event Details:
- Type: {event.event_type}
- Severity: {event.severity.value}
- Source: {event.source_service}
- Timestamp: {event.timestamp.isoformat()}

Message:
{event.message}
"""
		
		if event.metadata:
			description += "\n\nAdditional Information:\n"
			for key, value in event.metadata.items():
				description += f"- {key}: {value}\n"
		
		return description.strip()
	
	def _get_zendesk_priority(self, severity: EventSeverity) -> str:
		"""Map event severity to Zendesk priority."""
		priority_map = {
			EventSeverity.INFO: "low",
			EventSeverity.WARNING: "normal",
			EventSeverity.ERROR: "high",
			EventSeverity.CRITICAL: "urgent"
		}
		return priority_map.get(severity, "normal")


class NewRelicConnector(BaseConnector):
	"""Complete New Relic custom events integration."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Send custom event to New Relic Insights."""
		account_id = integration.config.get("account_id")
		insert_key = integration.config.get("insert_key")
		
		if not all([account_id, insert_key]):
			raise ValueError("New Relic account_id and insert_key required")
		
		# Create New Relic event
		nr_event = {
			"eventType": "APGConfigurationEvent",
			"timestamp": int(event.timestamp.timestamp() * 1000),  # Milliseconds
			"severity": event.severity.value,
			"eventCategory": event.event_type,
			"source": event.source_service,
			"message": event.message[:4000],  # New Relic limit
			"apgVersion": "1.0.0"
		}
		
		# Add metadata as custom attributes
		if event.metadata:
			for key, value in event.metadata.items():
				# New Relic attribute naming rules
				clean_key = re.sub(r'[^a-zA-Z0-9_]', '_', str(key))
				if len(clean_key) <= 255:  # New Relic limit
					nr_event[clean_key] = str(value)[:4000]
		
		url = f"https://insights-collector.newrelic.com/v1/accounts/{account_id}/events"
		headers = {
			"Content-Type": "application/json",
			"X-Insert-Key": insert_key
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				url,
				json=[nr_event],  # New Relic expects array
				headers=headers
			)
			response.raise_for_status()


class SplunkConnector(BaseConnector):
	"""Complete Splunk HEC (HTTP Event Collector) integration."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Send event to Splunk via HTTP Event Collector."""
		hec_url = integration.config.get("hec_url")
		hec_token = integration.config.get("hec_token")
		index = integration.config.get("index", "main")
		
		if not all([hec_url, hec_token]):
			raise ValueError("Splunk hec_url and hec_token required")
		
		# Create Splunk event
		splunk_event = {
			"time": int(event.timestamp.timestamp()),
			"index": index,
			"source": "apg_central_configuration",
			"sourcetype": "apg:config:event",
			"host": event.source_service,
			"event": {
				"event_type": event.event_type,
				"severity": event.severity.value,
				"message": event.message,
				"timestamp_iso": event.timestamp.isoformat(),
				"source_service": event.source_service,
				"metadata": event.metadata or {}
			}
		}
		
		headers = {
			"Authorization": f"Splunk {hec_token}",
			"Content-Type": "application/json"
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"{hec_url}/services/collector/event",
				json=splunk_event,
				headers=headers
			)
			response.raise_for_status()


class JenkinsConnector(BaseConnector):
	"""Complete Jenkins build trigger integration."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Trigger Jenkins build for deployment events."""
		jenkins_url = integration.config.get("jenkins_url")
		username = integration.config.get("username")
		api_token = integration.config.get("api_token")
		job_name = integration.config.get("job_name")
		
		if not all([jenkins_url, username, api_token, job_name]):
			raise ValueError("Jenkins jenkins_url, username, api_token, and job_name required")
		
		# Only trigger builds for specific event types
		trigger_events = integration.config.get("trigger_events", ["configuration_deployed", "configuration_updated"])
		if event.event_type not in trigger_events:
			return
		
		# Build parameters
		parameters = {
			"APG_EVENT_TYPE": event.event_type,
			"APG_SEVERITY": event.severity.value,
			"APG_SOURCE": event.source_service,
			"APG_MESSAGE": event.message,
			"APG_TIMESTAMP": event.timestamp.isoformat()
		}
		
		# Add metadata as build parameters
		if event.metadata:
			for key, value in event.metadata.items():
				param_key = f"APG_{str(key).upper().replace(' ', '_')}"
				parameters[param_key] = str(value)
		
		# Build Jenkins URL
		auth = httpx.BasicAuth(username, api_token)
		build_url = f"{jenkins_url}/job/{job_name}/buildWithParameters"
		
		async with httpx.AsyncClient(timeout=60.0) as client:
			response = await client.post(
				build_url,
				params=parameters,
				auth=auth
			)
			response.raise_for_status()
			
		# Get queue item location from response
		if "Location" in response.headers:
			event.metadata["jenkins_queue_url"] = response.headers["Location"]


class GitHubConnector(BaseConnector):
	"""Complete GitHub API integration for issues and deployments."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Create GitHub issue or deployment status."""
		token = integration.config.get("token")
		repository = integration.config.get("repository")  # Format: owner/repo
		
		if not all([token, repository]):
			raise ValueError("GitHub token and repository required")
		
		headers = {
			"Authorization": f"token {token}",
			"Accept": "application/vnd.github.v3+json",
			"Content-Type": "application/json"
		}
		
		base_url = f"https://api.github.com/repos/{repository}"
		
		# Handle different event types
		if event.event_type in ["security_alert", "critical_error"] and event.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]:
			await self._create_github_issue(base_url, headers, event)
		elif event.event_type in ["configuration_deployed", "deployment_started"]:
			await self._create_deployment_status(base_url, headers, event)
	
	async def _create_github_issue(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Create GitHub issue for critical events."""
		issue_data = {
			"title": f"APG Alert: {self._get_event_title(event)}",
			"body": self._format_github_issue_body(event),
			"labels": ["apg", "alert", event.severity.value.lower()],
			"assignees": []  # Could be configured per integration
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"{base_url}/issues",
				json=issue_data,
				headers=headers
			)
			response.raise_for_status()
			
		issue_info = response.json()
		event.metadata["github_issue_number"] = issue_info["number"]
		event.metadata["github_issue_url"] = issue_info["html_url"]
	
	async def _create_deployment_status(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Update GitHub deployment status."""
		deployment_id = event.metadata.get("deployment_id")
		if not deployment_id:
			return  # No deployment to update
		
		state_map = {
			"deployment_started": "in_progress",
			"configuration_deployed": "success",
			"deployment_failed": "failure"
		}
		
		status_data = {
			"state": state_map.get(event.event_type, "pending"),
			"description": event.message[:140],  # GitHub limit
			"environment": event.metadata.get("environment", "production"),
			"auto_inactive": True
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"{base_url}/deployments/{deployment_id}/statuses",
				json=status_data,
				headers=headers
			)
			response.raise_for_status()
	
	def _format_github_issue_body(self, event: IntegrationEvent) -> str:
		"""Format event details for GitHub issue body."""
		body = f"""
## APG Central Configuration Alert

**Event Details:**
- **Type:** {event.event_type}
- **Severity:** {event.severity.value}
- **Source:** {event.source_service}
- **Timestamp:** {event.timestamp.isoformat()}

**Message:**
{event.message}
"""
		
		if event.metadata:
			body += "\n\n**Additional Information:**\n"
			for key, value in event.metadata.items():
				body += f"- **{key}:** {value}\n"
		
		body += "\n\n---\n*This issue was created automatically by APG Central Configuration*"
		return body.strip()


class GitLabConnector(BaseConnector):
	"""Complete GitLab API integration for issues and merge requests."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Create GitLab issue or update merge request."""
		token = integration.config.get("token")
		project_id = integration.config.get("project_id")
		gitlab_url = integration.config.get("gitlab_url", "https://gitlab.com")
		
		if not all([token, project_id]):
			raise ValueError("GitLab token and project_id required")
		
		headers = {
			"Authorization": f"Bearer {token}",
			"Content-Type": "application/json"
		}
		
		base_url = f"{gitlab_url}/api/v4/projects/{project_id}"
		
		# Create issue for critical events
		if event.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]:
			await self._create_gitlab_issue(base_url, headers, event)
	
	async def _create_gitlab_issue(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Create GitLab issue for critical events."""
		issue_data = {
			"title": f"APG Alert: {self._get_event_title(event)}",
			"description": self._format_gitlab_issue_description(event),
			"labels": f"apg,alert,{event.severity.value.lower()}",
			"issue_type": "incident" if event.severity == EventSeverity.CRITICAL else "issue"
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"{base_url}/issues",
				json=issue_data,
				headers=headers
			)
			response.raise_for_status()
			
		issue_info = response.json()
		event.metadata["gitlab_issue_iid"] = issue_info["iid"]
		event.metadata["gitlab_issue_url"] = issue_info["web_url"]
	
	def _format_gitlab_issue_description(self, event: IntegrationEvent) -> str:
		"""Format event details for GitLab issue description."""
		description = f"""
## APG Central Configuration Alert

**Event Details:**
- **Type:** {event.event_type}
- **Severity:** {event.severity.value}  
- **Source:** {event.source_service}
- **Timestamp:** {event.timestamp.isoformat()}

**Message:**
{event.message}
"""
		
		if event.metadata:
			description += "\n\n**Additional Information:**\n"
			for key, value in event.metadata.items():
				description += f"- **{key}:** {value}\n"
		
		description += "\n\n---\n*This issue was created automatically by APG Central Configuration*"
		return description.strip()


class OktaConnector(BaseConnector):
	"""Complete Okta API integration for user and group management."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Handle Okta user/group management events."""
		domain = integration.config.get("domain")
		api_token = integration.config.get("api_token")
		
		if not all([domain, api_token]):
			raise ValueError("Okta domain and api_token required")
		
		headers = {
			"Authorization": f"SSWS {api_token}",
			"Accept": "application/json",
			"Content-Type": "application/json"
		}
		
		base_url = f"https://{domain}/api/v1"
		
		# Handle user provisioning events
		if event.event_type == "user_provisioning_required":
			await self._provision_okta_user(base_url, headers, event)
		elif event.event_type == "user_deprovisioning_required":
			await self._deprovision_okta_user(base_url, headers, event)
		elif event.event_type == "group_membership_update":
			await self._update_group_membership(base_url, headers, event)
	
	async def _provision_okta_user(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Provision new user in Okta."""
		user_data = event.metadata.get("user_data", {})
		if not user_data:
			return
		
		okta_user = {
			"profile": {
				"firstName": user_data.get("first_name"),
				"lastName": user_data.get("last_name"),
				"email": user_data.get("email"),
				"login": user_data.get("username", user_data.get("email"))
			},
			"credentials": {
				"password": {"value": user_data.get("temporary_password", "TempPass123!")},
				"recovery_question": {
					"question": "What is your favorite APG feature?",
					"answer": "Central Configuration"
				}
			}
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"{base_url}/users",
				json=okta_user,
				headers=headers,
				params={"activate": "true"}
			)
			response.raise_for_status()
			
		user_info = response.json()
		event.metadata["okta_user_id"] = user_info["id"]
	
	async def _deprovision_okta_user(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Deactivate user in Okta."""
		user_id = event.metadata.get("okta_user_id")
		if not user_id:
			return
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"{base_url}/users/{user_id}/lifecycle/deactivate",
				headers=headers
			)
			response.raise_for_status()
	
	async def _update_group_membership(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Update user group membership in Okta."""
		user_id = event.metadata.get("okta_user_id")
		group_id = event.metadata.get("okta_group_id")
		action = event.metadata.get("action", "add")  # add or remove
		
		if not all([user_id, group_id]):
			return
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			if action == "add":
				response = await client.put(
					f"{base_url}/groups/{group_id}/users/{user_id}",
					headers=headers
				)
			else:  # remove
				response = await client.delete(
					f"{base_url}/groups/{group_id}/users/{user_id}",
					headers=headers
				)
			response.raise_for_status()


class Auth0Connector(BaseConnector):
	"""Complete Auth0 Management API integration."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Handle Auth0 user management events."""
		domain = integration.config.get("domain")
		client_id = integration.config.get("client_id")
		client_secret = integration.config.get("client_secret")
		
		if not all([domain, client_id, client_secret]):
			raise ValueError("Auth0 domain, client_id, and client_secret required")
		
		# Get management API token
		access_token = await self._get_auth0_token(domain, client_id, client_secret)
		
		headers = {
			"Authorization": f"Bearer {access_token}",
			"Content-Type": "application/json"
		}
		
		base_url = f"https://{domain}/api/v2"
		
		# Handle user management events
		if event.event_type == "user_provisioning_required":
			await self._create_auth0_user(base_url, headers, event)
		elif event.event_type == "user_role_update":
			await self._update_user_roles(base_url, headers, event)
	
	async def _get_auth0_token(self, domain: str, client_id: str, client_secret: str) -> str:
		"""Get Auth0 Management API access token."""
		token_data = {
			"client_id": client_id,
			"client_secret": client_secret,
			"audience": f"https://{domain}/api/v2/",
			"grant_type": "client_credentials"
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"https://{domain}/oauth/token",
				json=token_data,
				headers={"Content-Type": "application/json"}
			)
			response.raise_for_status()
			
		token_info = response.json()
		return token_info["access_token"]
	
	async def _create_auth0_user(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Create user in Auth0."""
		user_data = event.metadata.get("user_data", {})
		if not user_data:
			return
		
		auth0_user = {
			"email": user_data.get("email"),
			"username": user_data.get("username"),
			"password": user_data.get("password", "TempPass123!"),
			"name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
			"connection": integration.config.get("connection", "Username-Password-Authentication"),
			"email_verified": False,
			"verify_email": True,
			"app_metadata": {
				"apg_source": "central_configuration",
				"created_by": "apg_provisioning"
			}
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"{base_url}/users",
				json=auth0_user,
				headers=headers
			)
			response.raise_for_status()
			
		user_info = response.json()
		event.metadata["auth0_user_id"] = user_info["user_id"]
	
	async def _update_user_roles(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Update user roles in Auth0."""
		user_id = event.metadata.get("auth0_user_id")
		roles = event.metadata.get("roles", [])
		
		if not user_id or not roles:
			return
		
		role_data = {"roles": roles}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"{base_url}/users/{user_id}/roles",
				json=role_data,
				headers=headers
			)
			response.raise_for_status()


class AzureADConnector(BaseConnector):
	"""Complete Azure Active Directory Graph API integration."""
	
	async def send_event(self, integration: IntegrationConfig, event: IntegrationEvent):
		"""Handle Azure AD user management events."""
		tenant_id = integration.config.get("tenant_id")
		client_id = integration.config.get("client_id")
		client_secret = integration.config.get("client_secret")
		
		if not all([tenant_id, client_id, client_secret]):
			raise ValueError("Azure AD tenant_id, client_id, and client_secret required")
		
		# Get access token
		access_token = await self._get_azure_token(tenant_id, client_id, client_secret)
		
		headers = {
			"Authorization": f"Bearer {access_token}",
			"Content-Type": "application/json"
		}
		
		base_url = "https://graph.microsoft.com/v1.0"
		
		# Handle user management events
		if event.event_type == "user_provisioning_required":
			await self._create_azure_user(base_url, headers, event)
		elif event.event_type == "group_membership_update":
			await self._update_group_membership_azure(base_url, headers, event)
	
	async def _get_azure_token(self, tenant_id: str, client_id: str, client_secret: str) -> str:
		"""Get Azure AD access token."""
		token_data = {
			"client_id": client_id,
			"client_secret": client_secret,
			"scope": "https://graph.microsoft.com/.default",
			"grant_type": "client_credentials"
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
				data=token_data,
				headers={"Content-Type": "application/x-www-form-urlencoded"}
			)
			response.raise_for_status()
			
		token_info = response.json()
		return token_info["access_token"]
	
	async def _create_azure_user(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Create user in Azure AD."""
		user_data = event.metadata.get("user_data", {})
		if not user_data:
			return
		
		azure_user = {
			"accountEnabled": True,
			"displayName": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
			"mailNickname": user_data.get("username", user_data.get("email", "").split("@")[0]),
			"userPrincipalName": user_data.get("email"),
			"passwordProfile": {
				"forceChangePasswordNextSignIn": True,
				"password": user_data.get("password", "TempPass123!")
			},
			"givenName": user_data.get("first_name"),
			"surname": user_data.get("last_name"),
			"jobTitle": user_data.get("job_title", ""),
			"department": user_data.get("department", "")
		}
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			response = await client.post(
				f"{base_url}/users",
				json=azure_user,
				headers=headers
			)
			response.raise_for_status()
			
		user_info = response.json()
		event.metadata["azure_user_id"] = user_info["id"]
	
	async def _update_group_membership_azure(self, base_url: str, headers: Dict[str, str], event: IntegrationEvent):
		"""Update group membership in Azure AD."""
		user_id = event.metadata.get("azure_user_id")
		group_id = event.metadata.get("azure_group_id")
		action = event.metadata.get("action", "add")
		
		if not all([user_id, group_id]):
			return
		
		async with httpx.AsyncClient(timeout=30.0) as client:
			if action == "add":
				member_data = {
					"@odata.id": f"https://graph.microsoft.com/v1.0/users/{user_id}"
				}
				response = await client.post(
					f"{base_url}/groups/{group_id}/members/$ref",
					json=member_data,
					headers=headers
				)
			else:  # remove
				response = await client.delete(
					f"{base_url}/groups/{group_id}/members/{user_id}/$ref",
					headers=headers
				)
			response.raise_for_status()


# ==================== Factory Functions ====================

async def create_integration_manager(
	config_engine: CentralConfigurationEngine
) -> EnterpriseIntegrationManager:
	"""Create and initialize enterprise integration manager."""
	manager = EnterpriseIntegrationManager(config_engine)
	await asyncio.sleep(1)  # Allow initialization
	print("🔌 Enterprise Integration Manager initialized")
	return manager