"""
APG Customer Relationship Management - Email Integration Module

Advanced email integration system with comprehensive tracking, automation,
and analytics for revolutionary email-driven customer relationship management.

Copyright © 2025 Datacraft
Author: Nyimbi Odero  
Email: nyimbi@gmail.com
"""

import asyncio
import logging
import smtplib
import imaplib
import email
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from decimal import Decimal
from uuid_extensions import uuid7str
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
import html2text
import hashlib

from pydantic import BaseModel, Field, validator

from .database import DatabaseManager


logger = logging.getLogger(__name__)


class EmailProvider(str, Enum):
	"""Email service providers"""
	SMTP = "smtp"
	GMAIL = "gmail"
	OUTLOOK = "outlook"
	EXCHANGE = "exchange"
	SENDGRID = "sendgrid"
	MAILGUN = "mailgun"
	AWS_SES = "aws_ses"
	CUSTOM = "custom"


class EmailType(str, Enum):
	"""Types of emails"""
	OUTBOUND = "outbound"
	INBOUND = "inbound"
	TEMPLATE = "template"
	CAMPAIGN = "campaign"
	AUTOMATED = "automated"
	REPLY = "reply"
	FORWARD = "forward"


class EmailStatus(str, Enum):
	"""Email delivery status"""
	DRAFT = "draft"
	QUEUED = "queued"
	SENDING = "sending"
	SENT = "sent"
	DELIVERED = "delivered"
	OPENED = "opened"
	CLICKED = "clicked"
	REPLIED = "replied"
	BOUNCED = "bounced"
	FAILED = "failed"
	SPAM = "spam"
	UNSUBSCRIBED = "unsubscribed"


class EmailPriority(str, Enum):
	"""Email priority levels"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"


class EmailTemplate(BaseModel):
	"""Email template configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Template information
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	category: str = Field("general", description="Template category")
	tags: List[str] = Field(default_factory=list)
	
	# Template content
	subject: str = Field(..., description="Email subject template")
	html_content: Optional[str] = Field(None, description="HTML email content")
	text_content: Optional[str] = Field(None, description="Plain text content")
	
	# Template settings
	is_active: bool = Field(True, description="Whether template is active")
	is_default: bool = Field(False, description="Default template for category")
	
	# Personalization
	merge_fields: List[str] = Field(default_factory=list, description="Available merge fields")
	required_fields: List[str] = Field(default_factory=list, description="Required merge fields")
	dynamic_content: Dict[str, Any] = Field(default_factory=dict)
	
	# Tracking settings
	enable_tracking: bool = Field(True, description="Enable email tracking")
	track_opens: bool = Field(True, description="Track email opens")
	track_clicks: bool = Field(True, description="Track link clicks")
	track_downloads: bool = Field(False, description="Track file downloads")
	
	# Usage statistics
	usage_count: int = Field(0, description="Times template has been used")
	success_rate: float = Field(0.0, description="Template success rate")
	average_open_rate: float = Field(0.0, description="Average open rate")
	average_click_rate: float = Field(0.0, description="Average click rate")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class EmailMessage(BaseModel):
	"""Email message record"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Message identification
	message_id: Optional[str] = Field(None, description="Email message ID")
	thread_id: Optional[str] = Field(None, description="Email thread ID")
	conversation_id: Optional[str] = Field(None, description="Conversation identifier")
	
	# Email details
	email_type: EmailType
	subject: str = Field(..., description="Email subject")
	from_email: str = Field(..., description="Sender email address")
	from_name: Optional[str] = Field(None, description="Sender display name")
	to_emails: List[str] = Field(..., description="Recipient email addresses")
	cc_emails: List[str] = Field(default_factory=list)
	bcc_emails: List[str] = Field(default_factory=list)
	reply_to: Optional[str] = Field(None, description="Reply-to address")
	
	# Content
	html_content: Optional[str] = Field(None, description="HTML email content")
	text_content: Optional[str] = Field(None, description="Plain text content")
	attachments: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Status and tracking
	status: EmailStatus = EmailStatus.DRAFT
	priority: EmailPriority = EmailPriority.NORMAL
	scheduled_at: Optional[datetime] = Field(None, description="Scheduled send time")
	sent_at: Optional[datetime] = Field(None, description="Actual send time")
	delivered_at: Optional[datetime] = Field(None, description="Delivery time")
	
	# CRM relationships
	contact_id: Optional[str] = Field(None, description="Associated contact")
	account_id: Optional[str] = Field(None, description="Associated account")
	lead_id: Optional[str] = Field(None, description="Associated lead")
	opportunity_id: Optional[str] = Field(None, description="Associated opportunity")
	campaign_id: Optional[str] = Field(None, description="Associated campaign")
	
	# Template and automation
	template_id: Optional[str] = Field(None, description="Used template")
	workflow_id: Optional[str] = Field(None, description="Associated workflow")
	is_automated: bool = Field(False, description="Sent by automation")
	
	# Tracking data
	tracking_enabled: bool = Field(True, description="Tracking enabled for this email")
	tracking_pixel_url: Optional[str] = Field(None, description="Tracking pixel URL")
	click_tracking_urls: Dict[str, str] = Field(default_factory=dict)
	
	# Metrics
	open_count: int = Field(0, description="Times email was opened")
	click_count: int = Field(0, description="Times links were clicked")
	reply_count: int = Field(0, description="Number of replies received")
	forward_count: int = Field(0, description="Times email was forwarded")
	
	# Email provider details
	provider: EmailProvider = EmailProvider.SMTP
	provider_message_id: Optional[str] = Field(None, description="Provider's message ID")
	provider_metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Error handling
	send_attempts: int = Field(0, description="Number of send attempts")
	last_error: Optional[str] = Field(None, description="Last error message")
	error_details: Dict[str, Any] = Field(default_factory=dict)
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str


class EmailTracking(BaseModel):
	"""Email tracking event record"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	email_id: str
	
	# Event details
	event_type: str = Field(..., description="Type of tracking event")
	event_data: Dict[str, Any] = Field(default_factory=dict)
	
	# User information
	user_agent: Optional[str] = Field(None, description="User agent string")
	ip_address: Optional[str] = Field(None, description="IP address")
	location_data: Dict[str, Any] = Field(default_factory=dict)
	device_info: Dict[str, Any] = Field(default_factory=dict)
	
	# URL and link tracking
	clicked_url: Optional[str] = Field(None, description="Clicked URL")
	link_text: Optional[str] = Field(None, description="Link text")
	link_position: Optional[int] = Field(None, description="Link position in email")
	
	# Timing
	event_timestamp: datetime = Field(default_factory=datetime.utcnow)
	time_since_sent: Optional[int] = Field(None, description="Seconds since email sent")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailAnalytics(BaseModel):
	"""Comprehensive email analytics"""
	tenant_id: str
	analysis_period_start: datetime
	analysis_period_end: datetime
	
	# Overall metrics
	total_emails_sent: int = 0
	total_emails_delivered: int = 0
	total_emails_opened: int = 0
	total_emails_clicked: int = 0
	total_emails_replied: int = 0
	total_emails_bounced: int = 0
	
	# Rates
	delivery_rate: float = 0.0
	open_rate: float = 0.0
	click_rate: float = 0.0
	reply_rate: float = 0.0
	bounce_rate: float = 0.0
	unsubscribe_rate: float = 0.0
	
	# Engagement metrics
	unique_opens: int = 0
	unique_clicks: int = 0
	average_time_to_open: float = 0.0
	average_time_to_click: float = 0.0
	
	# Content performance
	top_performing_subjects: List[Dict[str, Any]] = Field(default_factory=list)
	top_clicked_links: List[Dict[str, Any]] = Field(default_factory=list)
	template_performance: Dict[str, Any] = Field(default_factory=dict)
	
	# Temporal analysis
	best_send_times: Dict[str, float] = Field(default_factory=dict)
	hourly_engagement: Dict[int, int] = Field(default_factory=dict)
	daily_engagement: Dict[str, int] = Field(default_factory=dict)
	
	# Segmentation analysis
	segment_performance: Dict[str, Any] = Field(default_factory=dict)
	geographic_performance: Dict[str, Any] = Field(default_factory=dict)
	device_performance: Dict[str, Any] = Field(default_factory=dict)
	
	# Trend analysis
	engagement_trends: List[Dict[str, Any]] = Field(default_factory=list)
	delivery_trends: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Analysis metadata
	analyzed_at: datetime = Field(default_factory=datetime.utcnow)
	analysis_version: str = "1.0"


class EmailIntegrationManager:
	"""
	Advanced email integration management system
	
	Provides comprehensive email sending, receiving, tracking,
	and analytics capabilities with multi-provider support.
	"""
	
	def __init__(self, db_manager: DatabaseManager):
		"""
		Initialize email integration manager
		
		Args:
			db_manager: Database manager instance
		"""
		self.db_manager = db_manager
		self._initialized = False
		self._smtp_connections = {}
		self._imap_connections = {}
		self._email_configs = {}
	
	async def initialize(self):
		"""Initialize the email integration manager"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Email Integration Manager...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		# Initialize text converter
		self._text_converter = html2text.HTML2Text()
		self._text_converter.ignore_links = False
		self._text_converter.ignore_images = False
		
		self._initialized = True
		logger.info("✅ Email Integration Manager initialized successfully")
	
	async def create_email_template(
		self,
		template_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> EmailTemplate:
		"""
		Create a new email template
		
		Args:
			template_data: Template configuration data
			tenant_id: Tenant identifier
			created_by: User creating the template
			
		Returns:
			Created email template
		"""
		try:
			logger.info(f"📧 Creating email template: {template_data.get('name')}")
			
			# Add required fields
			template_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Auto-generate text content from HTML if not provided
			if template_data.get('html_content') and not template_data.get('text_content'):
				template_data['text_content'] = self._html_to_text(template_data['html_content'])
			
			# Extract merge fields from content
			merge_fields = self._extract_merge_fields(
				template_data.get('subject', '') + ' ' + 
				(template_data.get('html_content') or '') + ' ' + 
				(template_data.get('text_content') or '')
			)
			template_data['merge_fields'] = merge_fields
			
			# Create template object
			template = EmailTemplate(**template_data)
			
			# Store template in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_email_templates (
						id, tenant_id, name, description, category, tags,
						subject, html_content, text_content, is_active, is_default,
						merge_fields, required_fields, dynamic_content,
						enable_tracking, track_opens, track_clicks, track_downloads,
						usage_count, success_rate, average_open_rate, average_click_rate,
						metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
						$15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28
					)
				""", 
				template.id, template.tenant_id, template.name, template.description,
				template.category, template.tags, template.subject, template.html_content,  
				template.text_content, template.is_active, template.is_default,
				template.merge_fields, template.required_fields, template.dynamic_content,
				template.enable_tracking, template.track_opens, template.track_clicks, template.track_downloads,
				template.usage_count, template.success_rate, template.average_open_rate, template.average_click_rate,
				template.metadata, template.created_at, template.updated_at,
				template.created_by, template.updated_by, template.version
				)
			
			logger.info(f"✅ Email template created successfully: {template.id}")
			return template
			
		except Exception as e:
			logger.error(f"Failed to create email template: {str(e)}", exc_info=True)
			raise
	
	async def send_email(
		self,
		email_data: Dict[str, Any],
		tenant_id: str,
		created_by: str,
		send_immediately: bool = True
	) -> EmailMessage:
		"""
		Send an email message
		
		Args:
			email_data: Email message data
			tenant_id: Tenant identifier
			created_by: User sending the email
			send_immediately: Whether to send immediately or queue
			
		Returns:
			Email message record
		"""
		try:
			logger.info(f"📤 Sending email: {email_data.get('subject')}")
			
			# Add required fields
			email_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by,
				'email_type': email_data.get('email_type', EmailType.OUTBOUND)
			})
			
			# Generate message ID if not provided
			if not email_data.get('message_id'):
				email_data['message_id'] = self._generate_message_id(tenant_id)
			
			# Auto-generate text content from HTML if not provided
			if email_data.get('html_content') and not email_data.get('text_content'):
				email_data['text_content'] = self._html_to_text(email_data['html_content'])
			
			# Setup tracking if enabled
			if email_data.get('tracking_enabled', True):
				email_data['tracking_pixel_url'] = self._generate_tracking_pixel_url(email_data['message_id'])
				if email_data.get('html_content'):
					email_data['html_content'], email_data['click_tracking_urls'] = self._add_click_tracking(
						email_data['html_content'], email_data['message_id']
					)
			
			# Create email message object
			email_message = EmailMessage(**email_data)
			
			# Store email in database
			await self._store_email_message(email_message)
			
			# Send immediately if requested
			if send_immediately:
				await self._send_email_via_provider(email_message)
			else:
				# Queue for later sending
				email_message.status = EmailStatus.QUEUED
				await self._update_email_status(email_message.id, EmailStatus.QUEUED, tenant_id)
			
			logger.info(f"✅ Email {'sent' if send_immediately else 'queued'} successfully: {email_message.id}")
			return email_message
			
		except Exception as e:
			logger.error(f"Failed to send email: {str(e)}", exc_info=True)
			raise
	
	async def send_template_email(
		self,
		template_id: str,
		recipient_data: Dict[str, Any],
		merge_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> EmailMessage:
		"""
		Send email using a template
		
		Args:
			template_id: Email template identifier
			recipient_data: Recipient information
			merge_data: Data for template merge fields
			tenant_id: Tenant identifier
			created_by: User sending the email
			
		Returns:
			Email message record
		"""
		try:
			logger.info(f"📧 Sending template email: {template_id}")
			
			# Get template
			template = await self._get_email_template(template_id, tenant_id)
			if not template:
				raise ValueError(f"Email template not found: {template_id}")
			
			# Merge template with data
			subject = self._merge_template_content(template.subject, merge_data)
			html_content = self._merge_template_content(template.html_content or '', merge_data)
			text_content = self._merge_template_content(template.text_content or '', merge_data)
			
			# Prepare email data
			email_data = {
				'subject': subject,
				'html_content': html_content,
				'text_content': text_content,
				'to_emails': [recipient_data.get('email')],
				'from_email': recipient_data.get('from_email'),
				'from_name': recipient_data.get('from_name'),
				'template_id': template_id,
				'tracking_enabled': template.enable_tracking,
				'contact_id': recipient_data.get('contact_id'),
				'account_id': recipient_data.get('account_id'),
				'lead_id': recipient_data.get('lead_id'),
				'opportunity_id': recipient_data.get('opportunity_id'),
				'campaign_id': recipient_data.get('campaign_id')
			}
			
			# Send email
			email_message = await self.send_email(email_data, tenant_id, created_by)
			
			# Update template usage statistics
			await self._update_template_usage(template_id, tenant_id)
			
			return email_message
			
		except Exception as e:
			logger.error(f"Failed to send template email: {str(e)}", exc_info=True)
			raise
	
	async def track_email_event(
		self,
		email_id: str,
		event_type: str,
		event_data: Dict[str, Any],
		tenant_id: str
	) -> EmailTracking:
		"""
		Track an email event (open, click, etc.)
		
		Args:
			email_id: Email message identifier
			event_type: Type of event (open, click, bounce, etc.)
			event_data: Event-specific data
			tenant_id: Tenant identifier
			
		Returns:
			Email tracking record
		"""
		try:
			logger.info(f"📊 Tracking email event: {event_type} for {email_id}")
			
			# Create tracking record
			tracking = EmailTracking(
				tenant_id=tenant_id,
				email_id=email_id,
				event_type=event_type,
				event_data=event_data,
				user_agent=event_data.get('user_agent'),
				ip_address=event_data.get('ip_address'),
				location_data=event_data.get('location_data', {}),
				device_info=event_data.get('device_info', {}),
				clicked_url=event_data.get('clicked_url'),
				link_text=event_data.get('link_text'),
				link_position=event_data.get('link_position'),
				time_since_sent=event_data.get('time_since_sent')
			)
			
			# Store tracking record
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_email_tracking (
						id, tenant_id, email_id, event_type, event_data,
						user_agent, ip_address, location_data, device_info,
						clicked_url, link_text, link_position, event_timestamp,
						time_since_sent, metadata
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
					)
				""", 
				tracking.id, tracking.tenant_id, tracking.email_id, tracking.event_type,
				tracking.event_data, tracking.user_agent, tracking.ip_address,
				tracking.location_data, tracking.device_info, tracking.clicked_url,
				tracking.link_text, tracking.link_position, tracking.event_timestamp,
				tracking.time_since_sent, tracking.metadata
				)
			
			# Update email metrics
			await self._update_email_metrics(email_id, event_type, tenant_id)
			
			# Update email status if needed
			if event_type == 'open' and await self._is_first_open(email_id, tenant_id):
				await self._update_email_status(email_id, EmailStatus.OPENED, tenant_id)
			elif event_type == 'click' and await self._is_first_click(email_id, tenant_id):
				await self._update_email_status(email_id, EmailStatus.CLICKED, tenant_id)
			
			return tracking
			
		except Exception as e:
			logger.error(f"Failed to track email event: {str(e)}", exc_info=True)
			raise
	
	async def get_email_analytics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime,
		filters: Optional[Dict[str, Any]] = None
	) -> EmailAnalytics:
		"""
		Get comprehensive email analytics
		
		Args:
			tenant_id: Tenant identifier
			start_date: Analysis period start
			end_date: Analysis period end
			filters: Additional filters
			
		Returns:
			Email analytics data
		"""
		try:
			logger.info(f"📈 Generating email analytics for tenant: {tenant_id}")
			
			analytics = EmailAnalytics(
				tenant_id=tenant_id,
				analysis_period_start=start_date,
				analysis_period_end=end_date
			)
			
			async with self.db_manager.get_connection() as conn:
				# Overall metrics
				overall_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_sent,
						COUNT(*) FILTER (WHERE status = 'delivered') as delivered,
						COUNT(*) FILTER (WHERE status = 'opened' OR open_count > 0) as opened,
						COUNT(*) FILTER (WHERE status = 'clicked' OR click_count > 0) as clicked,
						COUNT(*) FILTER (WHERE status = 'replied' OR reply_count > 0) as replied,
						COUNT(*) FILTER (WHERE status = 'bounced') as bounced
					FROM crm_email_messages
					WHERE tenant_id = $1 AND sent_at BETWEEN $2 AND $3
				""", tenant_id, start_date, end_date)
				
				if overall_stats:
					analytics.total_emails_sent = overall_stats['total_sent'] or 0
					analytics.total_emails_delivered = overall_stats['delivered'] or 0
					analytics.total_emails_opened = overall_stats['opened'] or 0
					analytics.total_emails_clicked = overall_stats['clicked'] or 0
					analytics.total_emails_replied = overall_stats['replied'] or 0
					analytics.total_emails_bounced = overall_stats['bounced'] or 0
					
					# Calculate rates
					if analytics.total_emails_sent > 0:
						analytics.delivery_rate = (analytics.total_emails_delivered / analytics.total_emails_sent) * 100
						analytics.bounce_rate = (analytics.total_emails_bounced / analytics.total_emails_sent) * 100
					
					if analytics.total_emails_delivered > 0:
						analytics.open_rate = (analytics.total_emails_opened / analytics.total_emails_delivered) * 100
						analytics.click_rate = (analytics.total_emails_clicked / analytics.total_emails_delivered) * 100
						analytics.reply_rate = (analytics.total_emails_replied / analytics.total_emails_delivered) * 100
				
				# Engagement timing analysis
				timing_stats = await conn.fetchrow("""
					SELECT 
						AVG(EXTRACT(EPOCH FROM (et.event_timestamp - em.sent_at))) as avg_time_to_open,
						AVG(EXTRACT(EPOCH FROM (et.event_timestamp - em.sent_at))) FILTER (WHERE et.event_type = 'click') as avg_time_to_click
					FROM crm_email_messages em
					JOIN crm_email_tracking et ON em.id = et.email_id
					WHERE em.tenant_id = $1 AND em.sent_at BETWEEN $2 AND $3
					AND et.event_type IN ('open', 'click')
				""", tenant_id, start_date, end_date)
				
				if timing_stats:
					analytics.average_time_to_open = timing_stats['avg_time_to_open'] or 0.0
					analytics.average_time_to_click = timing_stats['avg_time_to_click'] or 0.0
				
				# Top performing subjects
				subject_performance = await conn.fetch("""
					SELECT 
						subject,
						COUNT(*) as sent_count,
						COUNT(*) FILTER (WHERE open_count > 0) as opened_count,
						COUNT(*) FILTER (WHERE click_count > 0) as clicked_count,
						(COUNT(*) FILTER (WHERE open_count > 0)::FLOAT / COUNT(*)) * 100 as open_rate
					FROM crm_email_messages
					WHERE tenant_id = $1 AND sent_at BETWEEN $2 AND $3
					GROUP BY subject
					HAVING COUNT(*) >= 5
					ORDER BY open_rate DESC
					LIMIT 10
				""", tenant_id, start_date, end_date)
				
				analytics.top_performing_subjects = [
					{
						'subject': row['subject'],
						'sent_count': row['sent_count'],
						'open_rate': row['open_rate'],
						'click_count': row['clicked_count']
					}
					for row in subject_performance
				]
				
				# Hourly engagement analysis
				hourly_stats = await conn.fetch("""
					SELECT 
						EXTRACT(HOUR FROM event_timestamp) as hour,
						COUNT(*) as events
					FROM crm_email_tracking et
					JOIN crm_email_messages em ON et.email_id = em.id
					WHERE em.tenant_id = $1 AND em.sent_at BETWEEN $2 AND $3
					AND et.event_type IN ('open', 'click')
					GROUP BY EXTRACT(HOUR FROM event_timestamp)
					ORDER BY hour
				""", tenant_id, start_date, end_date)
				
				analytics.hourly_engagement = {
					int(row['hour']): row['events'] for row in hourly_stats
				}
			
			logger.info(f"✅ Generated analytics for {analytics.total_emails_sent} emails")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate email analytics: {str(e)}", exc_info=True)
			raise
	
	async def list_email_templates(
		self,
		tenant_id: str,
		category: Optional[str] = None,
		active_only: bool = True
	) -> List[EmailTemplate]:
		"""
		List email templates for tenant
		
		Args:
			tenant_id: Tenant identifier
			category: Template category filter
			active_only: Return only active templates
			
		Returns:
			List of email templates
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Build query
				where_clause = "WHERE tenant_id = $1"
				params = [tenant_id]
				
				if category:
					where_clause += " AND category = $2"
					params.append(category)
				
				if active_only:
					where_clause += f" AND is_active = true"
				
				# Get templates
				template_rows = await conn.fetch(f"""
					SELECT * FROM crm_email_templates
					{where_clause}
					ORDER BY is_default DESC, category, name
				""", *params)
				
				templates = [EmailTemplate(**dict(row)) for row in template_rows]
				return templates
				
		except Exception as e:
			logger.error(f"Failed to list email templates: {str(e)}", exc_info=True)
			raise
	
	# Helper methods
	
	def _html_to_text(self, html_content: str) -> str:
		"""Convert HTML content to plain text"""
		try:
			if not html_content:
				return ""
			return self._text_converter.handle(html_content).strip()
		except Exception:
			return ""
	
	def _extract_merge_fields(self, content: str) -> List[str]:
		"""Extract merge fields from template content"""
		try:
			# Find patterns like {{field_name}} or {field_name}
			pattern = r'\{\{?([a-zA-Z_][a-zA-Z0-9_]*)\}?\}'
			matches = re.findall(pattern, content)
			return list(set(matches))
		except Exception:
			return []
	
	def _generate_message_id(self, tenant_id: str) -> str:
		"""Generate unique message ID"""
		timestamp = int(datetime.utcnow().timestamp())
		return f"<{uuid7str()}.{timestamp}@{tenant_id}.crm>"
	
	def _generate_tracking_pixel_url(self, message_id: str) -> str:
		"""Generate tracking pixel URL"""
		return f"/api/email/track/open/{message_id}.png"
	
	def _add_click_tracking(self, html_content: str, message_id: str) -> Tuple[str, Dict[str, str]]:
		"""Add click tracking to HTML content"""
		try:
			# Simple implementation - find all <a> tags and wrap URLs
			click_tracking_urls = {}
			
			def replace_link(match):
				url = match.group(1)
				tracking_id = hashlib.md5(url.encode()).hexdigest()[:8]
				tracking_url = f"/api/email/track/click/{message_id}/{tracking_id}"
				click_tracking_urls[tracking_id] = url
				return f'href="{tracking_url}"'
			
			# Replace href attributes
			tracked_html = re.sub(r'href="([^"]*)"', replace_link, html_content)
			
			return tracked_html, click_tracking_urls
			
		except Exception:
			return html_content, {}
	
	def _merge_template_content(self, content: str, merge_data: Dict[str, Any]) -> str:
		"""Merge template content with data"""
		try:
			if not content:
				return ""
			
			# Simple merge - replace {{field}} with values
			for field, value in merge_data.items():
				placeholder = f"{{{{{field}}}}}"
				content = content.replace(placeholder, str(value))
			
			return content
			
		except Exception:
			return content
	
	async def _store_email_message(self, email_message: EmailMessage):
		"""Store email message in database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_email_messages (
						id, tenant_id, message_id, thread_id, conversation_id,
						email_type, subject, from_email, from_name, to_emails, cc_emails, bcc_emails, reply_to,
						html_content, text_content, attachments, status, priority, scheduled_at, sent_at, delivered_at,
						contact_id, account_id, lead_id, opportunity_id, campaign_id,
						template_id, workflow_id, is_automated, tracking_enabled, tracking_pixel_url, click_tracking_urls,
						open_count, click_count, reply_count, forward_count,
						provider, provider_message_id, provider_metadata,
						send_attempts, last_error, error_details, metadata,
						created_at, updated_at, created_by, updated_by
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
						$22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36,
						$37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47
					)
				""", 
				email_message.id, email_message.tenant_id, email_message.message_id, 
				email_message.thread_id, email_message.conversation_id,
				email_message.email_type.value, email_message.subject, email_message.from_email,
				email_message.from_name, email_message.to_emails, email_message.cc_emails, 
				email_message.bcc_emails, email_message.reply_to,
				email_message.html_content, email_message.text_content, email_message.attachments,
				email_message.status.value, email_message.priority.value, email_message.scheduled_at,
				email_message.sent_at, email_message.delivered_at,
				email_message.contact_id, email_message.account_id, email_message.lead_id,
				email_message.opportunity_id, email_message.campaign_id,
				email_message.template_id, email_message.workflow_id, email_message.is_automated,
				email_message.tracking_enabled, email_message.tracking_pixel_url, email_message.click_tracking_urls,
				email_message.open_count, email_message.click_count, email_message.reply_count, email_message.forward_count,
				email_message.provider.value, email_message.provider_message_id, email_message.provider_metadata,
				email_message.send_attempts, email_message.last_error, email_message.error_details, email_message.metadata,
				email_message.created_at, email_message.updated_at, email_message.created_by, email_message.updated_by
				)
		except Exception as e:
			logger.error(f"Failed to store email message: {str(e)}")
			raise
	
	async def _send_email_via_provider(self, email_message: EmailMessage):
		"""Send email via configured provider"""
		try:
			# Update status to sending
			email_message.status = EmailStatus.SENDING
			email_message.send_attempts += 1
			await self._update_email_status(email_message.id, EmailStatus.SENDING, email_message.tenant_id)
			
			# Simulate email sending (in real implementation, integrate with actual providers)
			logger.info(f"📤 Sending email via {email_message.provider}: {email_message.subject}")
			
			# Update to sent status
			email_message.status = EmailStatus.SENT
			email_message.sent_at = datetime.utcnow()
			await self._update_email_status(email_message.id, EmailStatus.SENT, email_message.tenant_id)
			
			# Simulate delivery (in real implementation, this would come from webhooks)
			await asyncio.sleep(0.1)  # Simulate network delay
			email_message.status = EmailStatus.DELIVERED
			email_message.delivered_at = datetime.utcnow()
			await self._update_email_status(email_message.id, EmailStatus.DELIVERED, email_message.tenant_id)
			
		except Exception as e:
			# Update to failed status
			email_message.status = EmailStatus.FAILED
			email_message.last_error = str(e)
			await self._update_email_status(email_message.id, EmailStatus.FAILED, email_message.tenant_id)
			raise
	
	async def _update_email_status(self, email_id: str, status: EmailStatus, tenant_id: str):
		"""Update email status in database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_email_messages 
					SET status = $3, updated_at = NOW()
					WHERE id = $1 AND tenant_id = $2
				""", email_id, tenant_id, status.value)
		except Exception as e:
			logger.error(f"Failed to update email status: {str(e)}")

	async def _update_email_metrics(self, email_id: str, event_type: str, tenant_id: str):
		"""Update email engagement metrics"""
		try:
			async with self.db_manager.get_connection() as conn:
				if event_type == 'open':
					await conn.execute("""
						UPDATE crm_email_messages 
						SET open_count = open_count + 1, updated_at = NOW()
						WHERE id = $1 AND tenant_id = $2
					""", email_id, tenant_id)
				elif event_type == 'click':
					await conn.execute("""
						UPDATE crm_email_messages 
						SET click_count = click_count + 1, updated_at = NOW()
						WHERE id = $1 AND tenant_id = $2
					""", email_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to update email metrics: {str(e)}")
	
	async def _get_email_template(self, template_id: str, tenant_id: str) -> Optional[EmailTemplate]:
		"""Get email template by ID"""
		try:
			async with self.db_manager.get_connection() as conn:
				template_row = await conn.fetchrow("""
					SELECT * FROM crm_email_templates
					WHERE id = $1 AND tenant_id = $2
				""", template_id, tenant_id)
				
				if template_row:
					return EmailTemplate(**dict(template_row))
				return None
				
		except Exception as e:
			logger.error(f"Failed to get email template: {str(e)}")
			return None
	
	async def _update_template_usage(self, template_id: str, tenant_id: str):
		"""Update template usage statistics"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_email_templates 
					SET usage_count = usage_count + 1, updated_at = NOW()
					WHERE id = $1 AND tenant_id = $2
				""", template_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to update template usage: {str(e)}")
	
	async def _is_first_open(self, email_id: str, tenant_id: str) -> bool:
		"""Check if this is the first open for an email"""
		try:
			async with self.db_manager.get_connection() as conn:
				count = await conn.fetchval("""
					SELECT COUNT(*) FROM crm_email_tracking
					WHERE email_id = $1 AND event_type = 'open'
				""", email_id)
				return count == 1
		except Exception:
			return False
	
	async def _is_first_click(self, email_id: str, tenant_id: str) -> bool:
		"""Check if this is the first click for an email"""
		try:
			async with self.db_manager.get_connection() as conn:
				count = await conn.fetchval("""
					SELECT COUNT(*) FROM crm_email_tracking
					WHERE email_id = $1 AND event_type = 'click'
				""", email_id)
				return count == 1
		except Exception:
			return False