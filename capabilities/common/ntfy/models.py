"""
Notification Engine Models

Database models for comprehensive multi-channel notification system
with template management, delivery tracking, and analytics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


def uuid7str():
	"""Generate UUID7 string for consistent ID generation"""
	from uuid_extensions import uuid7
	return str(uuid7())


class NENotification(Model, AuditMixin, BaseMixin):
	"""
	Core notification record with delivery tracking and analytics.
	
	This model represents individual notifications sent to users across
	multiple channels with comprehensive tracking and metadata.
	"""
	__tablename__ = 'ne_notification'
	
	# Identity
	notification_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Notification Content
	title = Column(String(500), nullable=False)
	message = Column(Text, nullable=False)
	template_id = Column(String(36), ForeignKey('ne_template.template_id'), nullable=True)
	template_variables = Column(JSON, default=dict)
	
	# Targeting
	recipient_id = Column(String(36), nullable=True, index=True)  # User ID
	recipient_email = Column(String(255), nullable=True)
	recipient_phone = Column(String(20), nullable=True)
	recipient_data = Column(JSON, default=dict)  # Additional recipient info
	
	# Delivery Configuration
	channels = Column(JSON, default=list)  # email, sms, push, in_app, webhook
	priority = Column(String(20), default='normal')  # low, normal, high, urgent
	delivery_method = Column(String(20), default='immediate')  # immediate, scheduled, batch
	scheduled_at = Column(DateTime, nullable=True)
	expires_at = Column(DateTime, nullable=True)
	
	# Status and Tracking
	status = Column(String(20), default='pending')  # pending, processing, sent, delivered, failed
	delivery_attempts = Column(Integer, default=0)
	last_attempt_at = Column(DateTime, nullable=True)
	delivered_at = Column(DateTime, nullable=True)
	read_at = Column(DateTime, nullable=True)
	clicked_at = Column(DateTime, nullable=True)
	
	# Metadata
	source_event = Column(String(100), nullable=True)  # Originating event
	campaign_id = Column(String(36), nullable=True)
	tags = Column(JSON, default=list)
	tracking_data = Column(JSON, default=dict)
	error_details = Column(JSON, default=dict)
	
	# Relationships
	template = relationship("NETemplate", back_populates="notifications")
	deliveries = relationship("NEDelivery", back_populates="notification", cascade="all, delete-orphan")
	interactions = relationship("NEInteraction", back_populates="notification", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<NENotification {self.notification_id}: {self.title[:50]}>"
	
	def is_delivered(self) -> bool:
		"""Check if notification has been delivered successfully"""
		return self.status == 'delivered' and self.delivered_at is not None
	
	def is_expired(self) -> bool:
		"""Check if notification has expired"""
		return self.expires_at is not None and datetime.utcnow() > self.expires_at
	
	def get_delivery_rate(self) -> float:
		"""Calculate delivery success rate across all channels"""
		if not self.deliveries:
			return 0.0
		
		successful_deliveries = sum(1 for d in self.deliveries if d.status == 'delivered')
		return (successful_deliveries / len(self.deliveries)) * 100
	
	def get_engagement_metrics(self) -> Dict[str, Any]:
		"""Get engagement metrics for this notification"""
		metrics = {
			'delivered': self.is_delivered(),
			'opened': self.read_at is not None,
			'clicked': self.clicked_at is not None,
			'interactions_count': len(self.interactions),
			'time_to_open': None,
			'time_to_click': None
		}
		
		if self.delivered_at and self.read_at:
			metrics['time_to_open'] = (self.read_at - self.delivered_at).total_seconds()
		
		if self.read_at and self.clicked_at:
			metrics['time_to_click'] = (self.clicked_at - self.read_at).total_seconds()
		
		return metrics


class NETemplate(Model, AuditMixin, BaseMixin):
	"""
	Notification templates with versioning and localization support.
	
	Templates define the content structure and format for notifications
	across different channels and languages.
	"""
	__tablename__ = 'ne_template'
	
	template_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Template Identity
	name = Column(String(200), nullable=False)
	code = Column(String(100), nullable=False, index=True)  # Unique template code
	version = Column(String(20), default='1.0.0')
	locale = Column(String(10), default='en-US')
	description = Column(Text, nullable=True)
	
	# Template Content
	subject_template = Column(Text, nullable=True)  # For email
	html_template = Column(Text, nullable=True)
	text_template = Column(Text, nullable=True)
	push_template = Column(Text, nullable=True)  # For push notifications
	sms_template = Column(Text, nullable=True)
	
	# Template Configuration
	template_engine = Column(String(20), default='mustache')  # mustache, jinja2, handlebars
	variables_schema = Column(JSON, default=dict)  # Expected variables with types
	default_variables = Column(JSON, default=dict)  # Default variable values
	
	# Channel Support
	supported_channels = Column(JSON, default=list)
	channel_specific_config = Column(JSON, default=dict)
	
	# Template Status
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)
	
	# A/B Testing
	ab_test_variant = Column(String(10), nullable=True)  # A, B, etc.
	parent_template_id = Column(String(36), nullable=True)  # For A/B test variants
	
	# Performance Metrics
	usage_count = Column(Integer, default=0)
	success_rate = Column(Float, default=0.0)
	average_engagement = Column(Float, default=0.0)
	
	# Relationships
	notifications = relationship("NENotification", back_populates="template")
	
	def __repr__(self):
		return f"<NETemplate {self.code} v{self.version} ({self.locale})>"
	
	def render(self, variables: Dict[str, Any], channel: str = 'email') -> Dict[str, str]:
		"""
		Render template with provided variables for specific channel.
		
		Args:
			variables: Template variables to substitute
			channel: Target channel (email, sms, push, etc.)
			
		Returns:
			Rendered content dictionary with subject, html, text, etc.
		"""
		# Merge default variables with provided variables
		render_vars = {**self.default_variables, **variables}
		
		rendered = {}
		
		# Render based on template engine
		if self.template_engine == 'mustache':
			rendered = self._render_mustache(render_vars, channel)
		elif self.template_engine == 'jinja2':
			rendered = self._render_jinja2(render_vars, channel)
		else:
			rendered = self._render_simple(render_vars, channel)
		
		return rendered
	
	def _render_mustache(self, variables: Dict[str, Any], channel: str) -> Dict[str, str]:
		"""Render using Mustache template engine"""
		import pystache
		
		rendered = {}
		
		if channel == 'email':
			if self.subject_template:
				rendered['subject'] = pystache.render(self.subject_template, variables)
			if self.html_template:
				rendered['html'] = pystache.render(self.html_template, variables)
			if self.text_template:
				rendered['text'] = pystache.render(self.text_template, variables)
		elif channel == 'sms' and self.sms_template:
			rendered['text'] = pystache.render(self.sms_template, variables)
		elif channel == 'push' and self.push_template:
			rendered['text'] = pystache.render(self.push_template, variables)
		
		return rendered
	
	def _render_jinja2(self, variables: Dict[str, Any], channel: str) -> Dict[str, str]:
		"""Render using Jinja2 template engine"""
		from jinja2 import Template, Environment, select_autoescape
		
		env = Environment(autoescape=select_autoescape(['html', 'xml']))
		rendered = {}
		
		if channel == 'email':
			if self.subject_template:
				template = env.from_string(self.subject_template)
				rendered['subject'] = template.render(variables)
			if self.html_template:
				template = env.from_string(self.html_template)
				rendered['html'] = template.render(variables)
			if self.text_template:
				template = env.from_string(self.text_template)
				rendered['text'] = template.render(variables)
		elif channel == 'sms' and self.sms_template:
			template = env.from_string(self.sms_template)
			rendered['text'] = template.render(variables)
		elif channel == 'push' and self.push_template:
			template = env.from_string(self.push_template)
			rendered['text'] = template.render(variables)
		
		return rendered
	
	def _render_simple(self, variables: Dict[str, Any], channel: str) -> Dict[str, str]:
		"""Simple variable substitution using string.format()"""
		rendered = {}
		
		try:
			if channel == 'email':
				if self.subject_template:
					rendered['subject'] = self.subject_template.format(**variables)
				if self.html_template:
					rendered['html'] = self.html_template.format(**variables)
				if self.text_template:
					rendered['text'] = self.text_template.format(**variables)
			elif channel == 'sms' and self.sms_template:
				rendered['text'] = self.sms_template.format(**variables)
			elif channel == 'push' and self.push_template:
				rendered['text'] = self.push_template.format(**variables)
		except KeyError as e:
			raise ValueError(f"Missing template variable: {e}")
		
		return rendered
	
	def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
		"""
		Validate provided variables against schema.
		
		Returns:
			List of validation errors (empty if valid)
		"""
		errors = []
		
		for var_name, var_config in self.variables_schema.items():
			if var_config.get('required', False) and var_name not in variables:
				errors.append(f"Required variable '{var_name}' is missing")
			
			if var_name in variables:
				expected_type = var_config.get('type', 'string')
				actual_value = variables[var_name]
				
				if expected_type == 'string' and not isinstance(actual_value, str):
					errors.append(f"Variable '{var_name}' must be a string")
				elif expected_type == 'number' and not isinstance(actual_value, (int, float)):
					errors.append(f"Variable '{var_name}' must be a number")
				elif expected_type == 'boolean' and not isinstance(actual_value, bool):
					errors.append(f"Variable '{var_name}' must be a boolean")
		
		return errors


class NEDelivery(Model, AuditMixin, BaseMixin):
	"""
	Individual delivery attempts per channel.
	
	Tracks channel-specific delivery attempts with provider-specific
	information and performance metrics.
	"""
	__tablename__ = 'ne_delivery'
	
	delivery_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	notification_id = Column(String(36), ForeignKey('ne_notification.notification_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Delivery Details
	channel = Column(String(20), nullable=False, index=True)  # email, sms, push, in_app, webhook
	provider = Column(String(50), nullable=True)  # sendgrid, twilio, firebase, etc.
	recipient_address = Column(String(500), nullable=False)  # Email, phone, device token
	
	# Delivery Status
	status = Column(String(20), default='pending', index=True)  # pending, sent, delivered, failed, bounced
	attempt_number = Column(Integer, default=1)
	sent_at = Column(DateTime, nullable=True)
	delivered_at = Column(DateTime, nullable=True)
	failed_at = Column(DateTime, nullable=True)
	
	# Provider Response
	provider_id = Column(String(200), nullable=True)  # Provider's message ID
	provider_response = Column(JSON, default=dict)
	error_code = Column(String(50), nullable=True)
	error_message = Column(Text, nullable=True)
	
	# Delivery Metrics
	delivery_time_ms = Column(Integer, nullable=True)
	cost = Column(Float, nullable=True)  # Delivery cost if applicable
	
	# Content Details
	subject = Column(String(500), nullable=True)  # For email
	content_length = Column(Integer, nullable=True)
	content_type = Column(String(50), nullable=True)  # html, text, json
	
	# Relationships
	notification = relationship("NENotification", back_populates="deliveries")
	
	def __repr__(self):
		return f"<NEDelivery {self.delivery_id}: {self.channel} -> {self.recipient_address[:20]}>"
	
	def is_successful(self) -> bool:
		"""Check if delivery was successful"""
		return self.status in ['sent', 'delivered']
	
	def get_delivery_duration(self) -> Optional[float]:
		"""Get delivery duration in seconds"""
		if self.sent_at and self.delivered_at:
			return (self.delivered_at - self.sent_at).total_seconds()
		return None
	
	def retry_delivery(self) -> None:
		"""Prepare delivery for retry"""
		self.attempt_number += 1
		self.status = 'pending'
		self.failed_at = None
		self.error_code = None
		self.error_message = None


class NEInteraction(Model, AuditMixin, BaseMixin):
	"""
	Track user interactions with notifications.
	
	Records user engagement events like opens, clicks, and other interactions
	across different channels with context information.
	"""
	__tablename__ = 'ne_interaction'
	
	interaction_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	notification_id = Column(String(36), ForeignKey('ne_notification.notification_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Interaction Details
	interaction_type = Column(String(20), nullable=False, index=True)  # open, click, dismiss, reply, forward
	channel = Column(String(20), nullable=False)
	timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	
	# Context Information
	user_agent = Column(Text, nullable=True)
	ip_address = Column(String(45), nullable=True)
	click_url = Column(String(1000), nullable=True)  # URL clicked in notification
	device_info = Column(JSON, default=dict)
	location_data = Column(JSON, default=dict)  # Geographic data if available
	
	# Interaction Metadata
	interaction_data = Column(JSON, default=dict)  # Additional interaction-specific data
	session_id = Column(String(64), nullable=True)  # User session ID
	referrer = Column(String(500), nullable=True)
	
	# Relationships
	notification = relationship("NENotification", back_populates="interactions")
	
	def __repr__(self):
		return f"<NEInteraction {self.interaction_type} on {self.notification_id}>"
	
	def get_time_since_delivery(self) -> Optional[float]:
		"""Get time elapsed since notification delivery in seconds"""
		if self.notification and self.notification.delivered_at:
			return (self.timestamp - self.notification.delivered_at).total_seconds()
		return None


class NECampaign(Model, AuditMixin, BaseMixin):
	"""
	Notification campaigns for coordinated messaging.
	
	Manages multi-step notification campaigns with triggers,
	scheduling, and performance tracking.
	"""
	__tablename__ = 'ne_campaign'
	
	campaign_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Campaign Identity
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	campaign_type = Column(String(50), default='drip')  # drip, blast, triggered, a_b_test
	
	# Campaign Configuration
	trigger_event = Column(String(100), nullable=True)  # Event that starts campaign
	trigger_conditions = Column(JSON, default=dict)  # Conditions for campaign start
	target_audience = Column(JSON, default=dict)  # Audience targeting criteria
	
	# Campaign Status
	status = Column(String(20), default='draft', index=True)  # draft, active, paused, completed, archived
	start_date = Column(DateTime, nullable=True)
	end_date = Column(DateTime, nullable=True)
	
	# Campaign Metrics
	total_recipients = Column(Integer, default=0)
	total_sent = Column(Integer, default=0)
	total_delivered = Column(Integer, default=0)
	total_opened = Column(Integer, default=0)
	total_clicked = Column(Integer, default=0)
	conversion_count = Column(Integer, default=0)
	
	# Performance Metrics
	delivery_rate = Column(Float, default=0.0)
	open_rate = Column(Float, default=0.0)
	click_rate = Column(Float, default=0.0)
	conversion_rate = Column(Float, default=0.0)
	unsubscribe_rate = Column(Float, default=0.0)
	
	# Relationships
	steps = relationship("NECampaignStep", back_populates="campaign", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<NECampaign {self.name} ({self.status})>"
	
	def calculate_metrics(self) -> None:
		"""Calculate campaign performance metrics"""
		if self.total_sent > 0:
			self.delivery_rate = (self.total_delivered / self.total_sent) * 100
		
		if self.total_delivered > 0:
			self.open_rate = (self.total_opened / self.total_delivered) * 100
		
		if self.total_opened > 0:
			self.click_rate = (self.total_clicked / self.total_opened) * 100
		
		if self.total_recipients > 0:
			self.conversion_rate = (self.conversion_count / self.total_recipients) * 100
	
	def is_active(self) -> bool:
		"""Check if campaign is currently active"""
		now = datetime.utcnow()
		return (self.status == 'active' and 
				(not self.start_date or now >= self.start_date) and
				(not self.end_date or now <= self.end_date))


class NECampaignStep(Model, AuditMixin, BaseMixin):
	"""
	Individual steps in a notification campaign.
	
	Defines the sequence and timing of notifications within
	a multi-step campaign workflow.
	"""
	__tablename__ = 'ne_campaign_step'
	
	step_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	campaign_id = Column(String(36), ForeignKey('ne_campaign.campaign_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Step Configuration
	step_number = Column(Integer, nullable=False)
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Timing Configuration
	delay_minutes = Column(Integer, default=0)  # Delay from campaign start or previous step
	delay_type = Column(String(20), default='fixed')  # fixed, dynamic, optimal
	
	# Content Configuration
	template_id = Column(String(36), nullable=False)
	template_variables = Column(JSON, default=dict)
	channels = Column(JSON, default=list)
	
	# Conditions
	send_conditions = Column(JSON, default=dict)  # Conditions to send this step
	skip_conditions = Column(JSON, default=dict)  # Conditions to skip this step
	
	# Step Metrics
	recipients_count = Column(Integer, default=0)
	sent_count = Column(Integer, default=0)
	delivered_count = Column(Integer, default=0)
	opened_count = Column(Integer, default=0)
	clicked_count = Column(Integer, default=0)
	skipped_count = Column(Integer, default=0)
	
	# Performance Metrics
	step_conversion_rate = Column(Float, default=0.0)
	engagement_score = Column(Float, default=0.0)
	
	# Relationships
	campaign = relationship("NECampaign", back_populates="steps")
	
	def __repr__(self):
		return f"<NECampaignStep {self.step_number}: {self.name}>"
	
	def calculate_step_metrics(self) -> None:
		"""Calculate step-specific performance metrics"""
		if self.sent_count > 0:
			self.step_conversion_rate = (self.clicked_count / self.sent_count) * 100
		
		# Calculate engagement score based on multiple factors
		if self.delivered_count > 0:
			open_weight = 0.4
			click_weight = 0.6
			
			open_rate = (self.opened_count / self.delivered_count)
			click_rate = (self.clicked_count / self.delivered_count)
			
			self.engagement_score = (open_rate * open_weight + click_rate * click_weight) * 100


class NEUserPreference(Model, AuditMixin, BaseMixin):
	"""
	User notification preferences and subscription management.
	
	Stores user-specific preferences for notification channels,
	frequency, timing, and content types.
	"""
	__tablename__ = 'ne_user_preference'
	
	preference_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	user_id = Column(String(36), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Channel Preferences
	email_enabled = Column(Boolean, default=True)
	sms_enabled = Column(Boolean, default=False)
	push_enabled = Column(Boolean, default=True)
	in_app_enabled = Column(Boolean, default=True)
	
	# Contact Information
	email_address = Column(String(255), nullable=True)
	phone_number = Column(String(20), nullable=True)
	push_tokens = Column(JSON, default=list)  # Device push tokens
	
	# Frequency Preferences
	frequency_settings = Column(JSON, default=dict)  # Per notification type frequency
	digest_enabled = Column(Boolean, default=False)
	digest_frequency = Column(String(20), default='daily')  # daily, weekly, monthly
	
	# Timing Preferences
	timezone = Column(String(50), default='UTC')
	quiet_hours_start = Column(String(5), nullable=True)  # HH:MM format
	quiet_hours_end = Column(String(5), nullable=True)    # HH:MM format
	preferred_send_times = Column(JSON, default=dict)  # Preferred times per channel
	
	# Content Preferences
	content_categories = Column(JSON, default=list)  # Subscribed categories
	language_preference = Column(String(10), default='en-US')
	personalization_enabled = Column(Boolean, default=True)
	
	# Subscription Status
	is_subscribed = Column(Boolean, default=True)
	unsubscribed_at = Column(DateTime, nullable=True)
	unsubscribe_reason = Column(String(200), nullable=True)
	
	# Engagement Data
	last_engagement = Column(DateTime, nullable=True)
	engagement_score = Column(Float, default=0.0)
	total_notifications_received = Column(Integer, default=0)
	total_notifications_opened = Column(Integer, default=0)
	
	def __repr__(self):
		return f"<NEUserPreference for user {self.user_id}>"
	
	def is_channel_enabled(self, channel: str) -> bool:
		"""Check if specific channel is enabled for user"""
		channel_mapping = {
			'email': self.email_enabled,
			'sms': self.sms_enabled,
			'push': self.push_enabled,
			'in_app': self.in_app_enabled
		}
		return channel_mapping.get(channel, False)
	
	def is_quiet_hours(self, check_time: datetime = None) -> bool:
		"""Check if current time is within user's quiet hours"""
		if not self.quiet_hours_start or not self.quiet_hours_end:
			return False
		
		if check_time is None:
			check_time = datetime.utcnow()
		
		# Convert to user's timezone
		import pytz
		user_tz = pytz.timezone(self.timezone)
		user_time = check_time.astimezone(user_tz).time()
		
		start_time = datetime.strptime(self.quiet_hours_start, '%H:%M').time()
		end_time = datetime.strptime(self.quiet_hours_end, '%H:%M').time()
		
		if start_time <= end_time:
			return start_time <= user_time <= end_time
		else:  # Quiet hours span midnight
			return user_time >= start_time or user_time <= end_time
	
	def update_engagement_score(self) -> None:
		"""Update user engagement score based on interaction history"""
		if self.total_notifications_received > 0:
			base_engagement = (self.total_notifications_opened / self.total_notifications_received)
			
			# Factor in recency of engagement
			recency_bonus = 1.0
			if self.last_engagement:
				days_since_engagement = (datetime.utcnow() - self.last_engagement).days
				recency_bonus = max(0.5, 1.0 - (days_since_engagement / 30))  # Decay over 30 days
			
			self.engagement_score = min(100.0, base_engagement * recency_bonus * 100)


class NEProvider(Model, AuditMixin, BaseMixin):
	"""
	Notification provider configuration and monitoring.
	
	Manages external service provider configurations for different
	notification channels with health monitoring and failover.
	"""
	__tablename__ = 'ne_provider'
	
	provider_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Provider Identity
	name = Column(String(100), nullable=False)
	provider_type = Column(String(50), nullable=False)  # email, sms, push, webhook
	provider_key = Column(String(50), nullable=False, unique=True)  # sendgrid, twilio, etc.
	
	# Configuration
	api_endpoint = Column(String(500), nullable=True)
	api_key = Column(String(500), nullable=True)  # Encrypted
	api_secret = Column(String(500), nullable=True)  # Encrypted
	configuration = Column(JSON, default=dict)  # Provider-specific config
	
	# Provider Status
	is_enabled = Column(Boolean, default=True)
	is_primary = Column(Boolean, default=False)  # Primary provider for channel
	priority = Column(Integer, default=0)  # Lower numbers = higher priority
	
	# Rate Limiting
	rate_limit_per_minute = Column(Integer, nullable=True)
	rate_limit_per_hour = Column(Integer, nullable=True)
	rate_limit_per_day = Column(Integer, nullable=True)
	daily_quota = Column(Integer, nullable=True)
	
	# Health Monitoring
	health_status = Column(String(20), default='unknown')  # healthy, degraded, unhealthy, unknown
	last_health_check = Column(DateTime, nullable=True)
	consecutive_failures = Column(Integer, default=0)
	
	# Performance Metrics
	total_sent = Column(Integer, default=0)
	total_delivered = Column(Integer, default=0)
	total_failed = Column(Integer, default=0)
	average_response_time = Column(Float, default=0.0)  # milliseconds
	success_rate = Column(Float, default=0.0)  # percentage
	
	# Cost Tracking
	cost_per_message = Column(Float, nullable=True)
	monthly_cost = Column(Float, default=0.0)
	
	def __repr__(self):
		return f"<NEProvider {self.name} ({self.provider_type})>"
	
	def is_healthy(self) -> bool:
		"""Check if provider is healthy and available"""
		return (self.is_enabled and 
				self.health_status == 'healthy' and
				self.consecutive_failures < 5)
	
	def update_performance_metrics(self) -> None:
		"""Update provider performance metrics"""
		total_attempts = self.total_sent + self.total_failed
		if total_attempts > 0:
			self.success_rate = (self.total_delivered / total_attempts) * 100
	
	def can_send(self, current_usage: Dict[str, int] = None) -> bool:
		"""Check if provider can send more messages based on rate limits"""
		if not self.is_healthy():
			return False
		
		if current_usage is None:
			return True
		
		# Check rate limits
		if (self.rate_limit_per_minute and 
			current_usage.get('per_minute', 0) >= self.rate_limit_per_minute):
			return False
		
		if (self.rate_limit_per_hour and 
			current_usage.get('per_hour', 0) >= self.rate_limit_per_hour):
			return False
		
		if (self.rate_limit_per_day and 
			current_usage.get('per_day', 0) >= self.rate_limit_per_day):
			return False
		
		if (self.daily_quota and 
			current_usage.get('daily_sent', 0) >= self.daily_quota):
			return False
		
		return True