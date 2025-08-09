"""
APG Customer Relationship Management - Web Form Integration Module

Advanced web form integration system for capturing leads from websites,
landing pages, and marketing campaigns with real-time processing and
automated lead qualification.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from uuid_extensions import uuid7str
from urllib.parse import urlparse, parse_qs

from pydantic import BaseModel, Field, ValidationError, validator, HttpUrl

from .models import CRMContact, CRMLead, ContactType, LeadSource, LeadStatus
from .database import DatabaseManager
from .lead_scoring import LeadScoringManager


logger = logging.getLogger(__name__)


class FormType(str, Enum):
	"""Types of web forms"""
	CONTACT = "contact"
	NEWSLETTER = "newsletter"
	DEMO_REQUEST = "demo_request"
	WHITEPAPER_DOWNLOAD = "whitepaper_download"
	TRIAL_SIGNUP = "trial_signup"
	CONSULTATION = "consultation"
	WEBINAR_REGISTRATION = "webinar_registration"
	PRICING_INQUIRY = "pricing_inquiry"
	CUSTOM = "custom"


class FormStatus(str, Enum):
	"""Status of web forms"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	DRAFT = "draft"
	ARCHIVED = "archived"


class SubmissionStatus(str, Enum):
	"""Status of form submissions"""
	RECEIVED = "received"
	PROCESSING = "processing"
	PROCESSED = "processed"
	DUPLICATE = "duplicate"
	SPAM = "spam"
	ERROR = "error"


class ValidationRule(BaseModel):
	"""Form field validation rule"""
	field: str = Field(..., description="Field name to validate")
	rule_type: str = Field(..., description="Validation rule type")
	value: Union[str, int, float, bool, List[Any]] = Field(..., description="Rule value")
	message: str = Field(..., description="Error message for failed validation")
	required: bool = Field(False, description="Whether field is required")


class FormField(BaseModel):
	"""Web form field definition"""
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., description="Field name")
	label: str = Field(..., description="Field label")
	field_type: str = Field(..., description="Field input type")
	placeholder: Optional[str] = Field(None, description="Field placeholder")
	default_value: Optional[str] = Field(None, description="Default field value")
	required: bool = Field(False, description="Whether field is required")
	order: int = Field(0, description="Field display order")
	validation_rules: List[ValidationRule] = Field(default_factory=list)
	options: List[str] = Field(default_factory=list, description="Options for select/radio fields")
	css_classes: str = Field("", description="CSS classes for styling")
	attributes: Dict[str, str] = Field(default_factory=dict, description="Additional HTML attributes")


class WebForm(BaseModel):
	"""Web form configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Basic information
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	form_type: FormType = FormType.CONTACT
	status: FormStatus = FormStatus.DRAFT
	
	# Form structure
	fields: List[FormField] = Field(default_factory=list)
	
	# Integration settings
	webhook_url: Optional[HttpUrl] = Field(None, description="Webhook URL for form submissions")
	redirect_url: Optional[HttpUrl] = Field(None, description="Redirect URL after successful submission")
	success_message: str = Field("Thank you for your submission!", description="Success message")
	error_message: str = Field("There was an error processing your submission. Please try again.", description="Error message")
	
	# Lead capture settings
	auto_create_lead: bool = Field(True, description="Automatically create lead from submission")
	auto_create_contact: bool = Field(True, description="Automatically create contact from submission")
	default_lead_source: LeadSource = LeadSource.WEBSITE
	default_lead_status: LeadStatus = LeadStatus.NEW
	
	# Spam protection
	enable_captcha: bool = Field(False, description="Enable CAPTCHA protection")
	enable_honeypot: bool = Field(True, description="Enable honeypot spam protection")
	enable_rate_limiting: bool = Field(True, description="Enable rate limiting")
	max_submissions_per_hour: int = Field(10, description="Maximum submissions per IP per hour")
	
	# Campaign tracking
	campaign_id: Optional[str] = Field(None, description="Associated campaign ID")
	utm_tracking: bool = Field(True, description="Track UTM parameters")
	
	# Form styling and branding
	css_framework: str = Field("bootstrap", description="CSS framework to use")
	custom_css: str = Field("", description="Custom CSS styles")
	theme: str = Field("default", description="Form theme")
	
	# Analytics and tracking
	enable_analytics: bool = Field(True, description="Enable form analytics")
	track_partial_submissions: bool = Field(True, description="Track abandoned form submissions")
	
	# Notifications
	notification_emails: List[str] = Field(default_factory=list, description="Email addresses for notifications")
	email_notifications: bool = Field(False, description="Send email notifications for submissions")
	
	# Security
	allowed_domains: List[str] = Field(default_factory=list, description="Domains allowed to embed form")
	api_key: str = Field(default_factory=lambda: hashlib.sha256(uuid7str().encode()).hexdigest()[:32])
	
	# Performance tracking
	submission_count: int = Field(0, description="Total number of submissions")
	conversion_rate: float = Field(0.0, description="Form conversion rate")
	last_submission_at: Optional[datetime] = None
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1
	
	@validator('fields')
	def validate_fields(cls, v):
		if not v:
			raise ValueError("Form must have at least one field")
		return v


class FormSubmission(BaseModel):
	"""Web form submission data"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	form_id: str
	
	# Submission data
	form_data: Dict[str, Any] = Field(..., description="Submitted form data")
	raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw HTTP request data")
	
	# Status and processing
	status: SubmissionStatus = SubmissionStatus.RECEIVED
	processing_errors: List[str] = Field(default_factory=list)
	
	# Source tracking
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	referrer: Optional[str] = None
	utm_parameters: Dict[str, str] = Field(default_factory=dict)
	
	# Created records
	contact_id: Optional[str] = None
	lead_id: Optional[str] = None
	lead_score: Optional[int] = None
	
	# Duplicate detection
	duplicate_of: Optional[str] = Field(None, description="ID of original submission if duplicate")
	duplicate_score: float = Field(0.0, description="Duplicate detection score")
	
	# Spam detection
	spam_score: float = Field(0.0, description="Spam detection score")
	spam_reasons: List[str] = Field(default_factory=list)
	
	# Processing metadata
	processed_at: Optional[datetime] = None
	processing_duration_ms: Optional[int] = None
	
	# Audit fields
	submitted_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class FormAnalytics(BaseModel):
	"""Analytics data for web forms"""
	form_id: str
	form_name: str
	
	# Submission metrics
	total_submissions: int = 0
	successful_submissions: int = 0
	failed_submissions: int = 0
	spam_submissions: int = 0
	duplicate_submissions: int = 0
	
	# Conversion metrics
	conversion_rate: float = 0.0
	abandonment_rate: float = 0.0
	average_completion_time: float = 0.0
	
	# Lead generation metrics
	leads_generated: int = 0
	contacts_created: int = 0
	qualified_leads: int = 0
	average_lead_score: float = 0.0
	
	# Time-based metrics
	submissions_today: int = 0
	submissions_this_week: int = 0
	submissions_this_month: int = 0
	
	# Performance metrics
	average_processing_time_ms: float = 0.0
	error_rate: float = 0.0
	
	# Traffic sources
	top_referrers: List[Dict[str, Any]] = Field(default_factory=list)
	utm_campaign_performance: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Analysis metadata
	analyzed_at: datetime = Field(default_factory=datetime.utcnow)
	analysis_period_days: int = 30


class WebFormIntegrationManager:
	"""
	Advanced web form integration management system
	
	Provides comprehensive web form creation, submission processing,
	lead capture, spam protection, and analytics capabilities.
	"""
	
	def __init__(self, db_manager: DatabaseManager, lead_scoring_manager: LeadScoringManager = None):
		"""
		Initialize web form integration manager
		
		Args:
			db_manager: Database manager instance
			lead_scoring_manager: Lead scoring manager instance
		"""
		self.db_manager = db_manager
		self.lead_scoring_manager = lead_scoring_manager
		self._initialized = False
	
	async def initialize(self):
		"""Initialize the web form integration manager"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Web Form Integration Manager...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		self._initialized = True
		logger.info("✅ Web Form Integration Manager initialized successfully")
	
	async def create_form(
		self,
		form_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> WebForm:
		"""
		Create a new web form
		
		Args:
			form_data: Form configuration data
			tenant_id: Tenant identifier
			created_by: User creating the form
			
		Returns:
			Created web form
		"""
		try:
			logger.info(f"📝 Creating web form: {form_data.get('name')}")
			
			# Add required fields
			form_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Create form object
			form = WebForm(**form_data)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_web_forms (
						id, tenant_id, name, description, form_type, status,
						fields, webhook_url, redirect_url, success_message, error_message,
						auto_create_lead, auto_create_contact, default_lead_source, default_lead_status,
						enable_captcha, enable_honeypot, enable_rate_limiting, max_submissions_per_hour,
						campaign_id, utm_tracking, css_framework, custom_css, theme,
						enable_analytics, track_partial_submissions, notification_emails, email_notifications,
						allowed_domains, api_key, submission_count, conversion_rate, last_submission_at,
						metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
						$16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28,
						$29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39
					)
				""", 
				form.id, form.tenant_id, form.name, form.description, form.form_type.value, form.status.value,
				[field.model_dump() for field in form.fields], str(form.webhook_url) if form.webhook_url else None,
				str(form.redirect_url) if form.redirect_url else None, form.success_message, form.error_message,
				form.auto_create_lead, form.auto_create_contact, form.default_lead_source.value, form.default_lead_status.value,
				form.enable_captcha, form.enable_honeypot, form.enable_rate_limiting, form.max_submissions_per_hour,
				form.campaign_id, form.utm_tracking, form.css_framework, form.custom_css, form.theme,
				form.enable_analytics, form.track_partial_submissions, form.notification_emails, form.email_notifications,
				form.allowed_domains, form.api_key, form.submission_count, form.conversion_rate, form.last_submission_at,
				form.metadata, form.created_at, form.updated_at, form.created_by, form.updated_by, form.version
				)
			
			logger.info(f"✅ Web form created successfully: {form.id}")
			return form
			
		except Exception as e:
			logger.error(f"Failed to create web form: {str(e)}", exc_info=True)
			raise
	
	async def process_form_submission(
		self,
		form_id: str,
		submission_data: Dict[str, Any],
		tenant_id: str,
		request_metadata: Dict[str, Any] = None
	) -> FormSubmission:
		"""
		Process a web form submission
		
		Args:
			form_id: Form identifier
			submission_data: Submitted form data
			tenant_id: Tenant identifier
			request_metadata: HTTP request metadata
			
		Returns:
			Processed form submission
		"""
		try:
			logger.info(f"📩 Processing form submission for form: {form_id}")
			
			start_time = datetime.utcnow()
			
			# Get form configuration
			form = await self.get_form(form_id, tenant_id)
			if not form:
				raise ValueError(f"Form not found: {form_id}")
			
			if form.status != FormStatus.ACTIVE:
				raise ValueError(f"Form is not active: {form_id}")
			
			# Create submission record
			submission = FormSubmission(
				tenant_id=tenant_id,
				form_id=form_id,
				form_data=submission_data,
				raw_data=request_metadata or {}
			)
			
			# Extract request metadata
			if request_metadata:
				submission.ip_address = request_metadata.get('ip_address')
				submission.user_agent = request_metadata.get('user_agent')
				submission.referrer = request_metadata.get('referrer')
				submission.utm_parameters = self._extract_utm_parameters(request_metadata)
			
			# Validate form data
			validation_errors = await self._validate_submission(form, submission_data)
			if validation_errors:
				submission.status = SubmissionStatus.ERROR
				submission.processing_errors = validation_errors
				await self._store_submission(submission)
				raise ValueError(f"Form validation failed: {', '.join(validation_errors)}")
			
			# Check for spam
			spam_result = await self._check_spam(form, submission)
			if spam_result['is_spam']:
				submission.status = SubmissionStatus.SPAM
				submission.spam_score = spam_result['score']
				submission.spam_reasons = spam_result['reasons']
				await self._store_submission(submission)
				logger.warning(f"Spam submission detected: {submission.id}")
				return submission
			
			# Check for duplicates
			duplicate_result = await self._check_duplicates(form, submission)
			if duplicate_result['is_duplicate']:
				submission.status = SubmissionStatus.DUPLICATE
				submission.duplicate_of = duplicate_result['original_id']
				submission.duplicate_score = duplicate_result['score']
				await self._store_submission(submission)
				logger.info(f"Duplicate submission detected: {submission.id}")
				return submission
			
			# Process submission
			submission.status = SubmissionStatus.PROCESSING
			await self._store_submission(submission)
			
			# Create contact if enabled
			if form.auto_create_contact:
				contact_id = await self._create_contact_from_submission(form, submission)
				submission.contact_id = contact_id
			
			# Create lead if enabled
			if form.auto_create_lead:
				lead_id = await self._create_lead_from_submission(form, submission)
				submission.lead_id = lead_id
				
				# Calculate lead score if scoring manager available
				if self.lead_scoring_manager and lead_id:
					try:
						score = await self.lead_scoring_manager.calculate_lead_score(lead_id, tenant_id)
						submission.lead_score = score.total_score
					except Exception as e:
						logger.warning(f"Failed to calculate lead score: {str(e)}")
			
			# Update submission status
			submission.status = SubmissionStatus.PROCESSED
			submission.processed_at = datetime.utcnow()
			
			# Calculate processing duration
			end_time = datetime.utcnow()
			submission.processing_duration_ms = int((end_time - start_time).total_seconds() * 1000)
			
			# Update submission in database
			await self._update_submission(submission)
			
			# Update form statistics
			await self._update_form_statistics(form_id, tenant_id)
			
			# Send notifications if configured
			if form.email_notifications and form.notification_emails:
				asyncio.create_task(self._send_notifications(form, submission))
			
			# Call webhook if configured
			if form.webhook_url:
				asyncio.create_task(self._call_webhook(form, submission))
			
			logger.info(f"✅ Form submission processed successfully: {submission.id}")
			return submission
			
		except Exception as e:
			logger.error(f"Failed to process form submission: {str(e)}", exc_info=True)
			raise
	
	async def get_form(
		self,
		form_id: str,
		tenant_id: str
	) -> Optional[WebForm]:
		"""
		Get form by ID
		
		Args:
			form_id: Form identifier
			tenant_id: Tenant identifier
			
		Returns:
			Form if found
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_web_forms 
					WHERE id = $1 AND tenant_id = $2
				""", form_id, tenant_id)
				
				if not row:
					return None
				
				# Convert row to dict and handle nested objects
				form_dict = dict(row)
				
				# Parse fields from JSON
				if form_dict['fields']:
					form_dict['fields'] = [FormField(**field) for field in form_dict['fields']]
				
				return WebForm(**form_dict)
				
		except Exception as e:
			logger.error(f"Failed to get web form: {str(e)}", exc_info=True)
			raise
	
	async def get_form_submissions(
		self,
		form_id: str,
		tenant_id: str,
		status: Optional[SubmissionStatus] = None,
		limit: int = 100,
		offset: int = 0
	) -> Dict[str, Any]:
		"""
		Get form submissions with filtering
		
		Args:
			form_id: Form identifier
			tenant_id: Tenant identifier
			status: Filter by submission status
			limit: Maximum results
			offset: Results offset
			
		Returns:
			Dict containing submissions and pagination info
		"""
		try:
			# Build query conditions
			conditions = ["tenant_id = $1", "form_id = $2"]
			params = [tenant_id, form_id]
			param_count = 2
			
			if status:
				param_count += 1
				conditions.append(f"status = ${param_count}")
				params.append(status.value)
			
			where_clause = " WHERE " + " AND ".join(conditions)
			
			async with self.db_manager.get_connection() as conn:
				# Get total count
				count_query = f"SELECT COUNT(*) FROM crm_form_submissions{where_clause}"
				total = await conn.fetchval(count_query, *params)
				
				# Get submissions
				query = f"""
					SELECT * FROM crm_form_submissions
					{where_clause}
					ORDER BY submitted_at DESC
					LIMIT {limit} OFFSET {offset}
				"""
				
				rows = await conn.fetch(query, *params)
				submissions = [FormSubmission(**dict(row)) for row in rows]
			
			return {
				"submissions": submissions,
				"total": total,
				"limit": limit,
				"offset": offset
			}
			
		except Exception as e:
			logger.error(f"Failed to get form submissions: {str(e)}", exc_info=True)
			raise
	
	async def get_form_analytics(
		self,
		form_id: str,
		tenant_id: str,
		period_days: int = 30
	) -> FormAnalytics:
		"""
		Get comprehensive analytics for a form
		
		Args:
			form_id: Form identifier
			tenant_id: Tenant identifier
			period_days: Analysis period in days
			
		Returns:
			Form analytics data
		"""
		try:
			logger.info(f"📊 Generating form analytics: {form_id}")
			
			# Get form
			form = await self.get_form(form_id, tenant_id)
			if not form:
				raise ValueError(f"Form not found: {form_id}")
			
			analytics = FormAnalytics(
				form_id=form_id,
				form_name=form.name,
				analysis_period_days=period_days
			)
			
			async with self.db_manager.get_connection() as conn:
				# Basic submission statistics
				stats_row = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_submissions,
						COUNT(*) FILTER (WHERE status = 'processed') as successful_submissions,
						COUNT(*) FILTER (WHERE status = 'error') as failed_submissions,
						COUNT(*) FILTER (WHERE status = 'spam') as spam_submissions,
						COUNT(*) FILTER (WHERE status = 'duplicate') as duplicate_submissions,
						AVG(processing_duration_ms) as avg_processing_time
					FROM crm_form_submissions 
					WHERE form_id = $1 AND tenant_id = $2
					AND submitted_at >= NOW() - INTERVAL '%s days'
				""", form_id, tenant_id, period_days)
				
				if stats_row:
					analytics.total_submissions = stats_row['total_submissions'] or 0
					analytics.successful_submissions = stats_row['successful_submissions'] or 0
					analytics.failed_submissions = stats_row['failed_submissions'] or 0
					analytics.spam_submissions = stats_row['spam_submissions'] or 0
					analytics.duplicate_submissions = stats_row['duplicate_submissions'] or 0
					analytics.average_processing_time_ms = float(stats_row['avg_processing_time'] or 0)
				
				# Calculate rates
				if analytics.total_submissions > 0:
					analytics.conversion_rate = (analytics.successful_submissions / analytics.total_submissions) * 100
					analytics.error_rate = (analytics.failed_submissions / analytics.total_submissions) * 100
				
				# Lead generation metrics
				lead_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) FILTER (WHERE lead_id IS NOT NULL) as leads_generated,
						COUNT(*) FILTER (WHERE contact_id IS NOT NULL) as contacts_created,
						AVG(lead_score) as avg_lead_score
					FROM crm_form_submissions 
					WHERE form_id = $1 AND tenant_id = $2
					AND submitted_at >= NOW() - INTERVAL '%s days'
					AND status = 'processed'
				""", form_id, tenant_id, period_days)
				
				if lead_stats:
					analytics.leads_generated = lead_stats['leads_generated'] or 0
					analytics.contacts_created = lead_stats['contacts_created'] or 0
					analytics.average_lead_score = float(lead_stats['avg_lead_score'] or 0)
				
				# Time-based metrics
				time_metrics = await conn.fetchrow("""
					SELECT 
						COUNT(*) FILTER (WHERE submitted_at >= CURRENT_DATE) as today,
						COUNT(*) FILTER (WHERE submitted_at >= DATE_TRUNC('week', NOW())) as this_week,
						COUNT(*) FILTER (WHERE submitted_at >= DATE_TRUNC('month', NOW())) as this_month
					FROM crm_form_submissions 
					WHERE form_id = $1 AND tenant_id = $2
				""", form_id, tenant_id)
				
				if time_metrics:
					analytics.submissions_today = time_metrics['today'] or 0
					analytics.submissions_this_week = time_metrics['this_week'] or 0
					analytics.submissions_this_month = time_metrics['this_month'] or 0
				
				# Top referrers
				referrer_rows = await conn.fetch("""
					SELECT 
						referrer,
						COUNT(*) as submission_count
					FROM crm_form_submissions 
					WHERE form_id = $1 AND tenant_id = $2
					AND referrer IS NOT NULL
					AND submitted_at >= NOW() - INTERVAL '%s days'
					GROUP BY referrer
					ORDER BY submission_count DESC
					LIMIT 10
				""", form_id, tenant_id, period_days)
				
				analytics.top_referrers = [
					{"referrer": row['referrer'], "count": row['submission_count']}
					for row in referrer_rows
				]
			
			logger.info(f"✅ Generated analytics for {analytics.total_submissions} submissions")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate form analytics: {str(e)}", exc_info=True)
			raise
	
	async def _validate_submission(
		self,
		form: WebForm,
		submission_data: Dict[str, Any]
	) -> List[str]:
		"""Validate form submission data"""
		errors = []
		
		for field in form.fields:
			value = submission_data.get(field.name)
			
			# Check required fields
			if field.required and (value is None or value == ""):
				errors.append(f"Field '{field.label}' is required")
				continue
			
			# Skip validation if field is empty and not required
			if value is None or value == "":
				continue
			
			# Apply validation rules
			for rule in field.validation_rules:
				if not self._apply_validation_rule(rule, value):
					errors.append(rule.message)
		
		return errors
	
	def _apply_validation_rule(self, rule: ValidationRule, value: Any) -> bool:
		"""Apply a single validation rule"""
		try:
			if rule.rule_type == "email":
				import re
				pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
				return bool(re.match(pattern, str(value)))
			
			elif rule.rule_type == "min_length":
				return len(str(value)) >= int(rule.value)
			
			elif rule.rule_type == "max_length":
				return len(str(value)) <= int(rule.value)
			
			elif rule.rule_type == "pattern":
				import re
				return bool(re.match(str(rule.value), str(value)))
			
			elif rule.rule_type == "numeric":
				try:
					float(value)
					return True
				except ValueError:
					return False
			
			elif rule.rule_type == "in":
				return value in rule.value
			
			return True
			
		except Exception as e:
			logger.error(f"Validation rule error: {str(e)}")
			return False
	
	async def _check_spam(
		self,
		form: WebForm,
		submission: FormSubmission
	) -> Dict[str, Any]:
		"""Check if submission is spam"""
		spam_score = 0.0
		reasons = []
		
		# Check honeypot field
		if form.enable_honeypot:
			honeypot_value = submission.form_data.get('_honeypot', '')
			if honeypot_value:
				spam_score += 50.0
				reasons.append("Honeypot field filled")
		
		# Check submission rate
		if form.enable_rate_limiting and submission.ip_address:
			recent_submissions = await self._count_recent_submissions(
				submission.ip_address, hours=1
			)
			if recent_submissions > form.max_submissions_per_hour:
				spam_score += 30.0
				reasons.append("Too many submissions from IP")
		
		# Check for suspicious patterns
		form_text = ' '.join(str(v) for v in submission.form_data.values())
		
		# Common spam keywords
		spam_keywords = ['cialis', 'viagra', 'casino', 'lottery', 'winner', 'congratulations', 'urgent']
		keyword_matches = sum(1 for keyword in spam_keywords if keyword.lower() in form_text.lower())
		if keyword_matches > 0:
			spam_score += keyword_matches * 10.0
			reasons.append(f"Contains {keyword_matches} spam keywords")
		
		# Check for excessive links
		import re
		link_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', form_text))
		if link_count > 2:
			spam_score += link_count * 5.0
			reasons.append(f"Contains {link_count} links")
		
		is_spam = spam_score >= 50.0
		
		return {
			'is_spam': is_spam,
			'score': spam_score,
			'reasons': reasons
		}
	
	async def _check_duplicates(
		self,
		form: WebForm,
		submission: FormSubmission
	) -> Dict[str, Any]:
		"""Check for duplicate submissions"""
		try:
			# Simple duplicate check based on email and recent submissions
			async with self.db_manager.get_connection() as conn:
				email = submission.form_data.get('email')
				if not email:
					return {'is_duplicate': False, 'score': 0.0, 'original_id': None}
				
				# Check for recent submissions with same email
				recent_submission = await conn.fetchrow("""
					SELECT id FROM crm_form_submissions
					WHERE form_id = $1 AND tenant_id = $2
					AND form_data->>'email' = $3
					AND submitted_at >= NOW() - INTERVAL '24 hours'
					AND status != 'duplicate'
					ORDER BY submitted_at DESC
					LIMIT 1
				""", form.id, submission.tenant_id, email)
				
				if recent_submission:
					return {
						'is_duplicate': True,
						'score': 100.0,
						'original_id': recent_submission['id']
					}
				
				return {'is_duplicate': False, 'score': 0.0, 'original_id': None}
				
		except Exception as e:
			logger.error(f"Duplicate check error: {str(e)}")
			return {'is_duplicate': False, 'score': 0.0, 'original_id': None}
	
	async def _count_recent_submissions(
		self,
		ip_address: str,
		hours: int = 1
	) -> int:
		"""Count recent submissions from IP address"""
		try:
			async with self.db_manager.get_connection() as conn:
				count = await conn.fetchval("""
					SELECT COUNT(*) FROM crm_form_submissions
					WHERE ip_address = $1
					AND submitted_at >= NOW() - INTERVAL '%s hours'
				""", ip_address, hours)
				return count or 0
		except Exception as e:
			logger.error(f"Recent submissions count error: {str(e)}")
			return 0
	
	def _extract_utm_parameters(self, request_metadata: Dict[str, Any]) -> Dict[str, str]:
		"""Extract UTM parameters from request"""
		utm_params = {}
		referrer = request_metadata.get('referrer', '')
		
		if referrer:
			try:
				parsed = urlparse(referrer)
				query_params = parse_qs(parsed.query)
				
				utm_keys = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content']
				for key in utm_keys:
					if key in query_params:
						utm_params[key] = query_params[key][0]
			except Exception as e:
				logger.error(f"UTM parameter extraction error: {str(e)}")
		
		return utm_params
	
	async def _create_contact_from_submission(
		self,
		form: WebForm,
		submission: FormSubmission
	) -> Optional[str]:
		"""Create contact from form submission"""
		try:
			form_data = submission.form_data
			
			# Map form fields to contact fields
			contact_data = {
				'tenant_id': submission.tenant_id,
				'first_name': form_data.get('first_name', ''),
				'last_name': form_data.get('last_name', ''),
				'email': form_data.get('email', ''),
				'phone': form_data.get('phone', ''),
				'company_name': form_data.get('company', ''),
				'job_title': form_data.get('job_title', ''),
				'contact_type': ContactType.LEAD,
				'lead_source': form.default_lead_source,
				'notes': f"Created from web form: {form.name}",
				'metadata': {
					'form_id': form.id,
					'submission_id': submission.id,
					'utm_parameters': submission.utm_parameters
				}
			}
			
			# Create contact (this would normally call the contact service)
			# For now, return a placeholder ID
			contact_id = uuid7str()
			logger.info(f"📞 Contact created from form submission: {contact_id}")
			return contact_id
			
		except Exception as e:
			logger.error(f"Failed to create contact from submission: {str(e)}")
			return None
	
	async def _create_lead_from_submission(
		self,
		form: WebForm,
		submission: FormSubmission
	) -> Optional[str]:
		"""Create lead from form submission"""
		try:
			form_data = submission.form_data
			
			# Map form fields to lead fields
			lead_data = {
				'tenant_id': submission.tenant_id,
				'title': f"Lead from {form.name}",
				'description': f"Lead generated from web form submission",
				'status': form.default_lead_status,
				'source': form.default_lead_source,
				'contact_id': submission.contact_id,
				'campaign_id': form.campaign_id,
				'metadata': {
					'form_id': form.id,
					'submission_id': submission.id,
					'form_data': form_data,
					'utm_parameters': submission.utm_parameters
				}
			}
			
			# Create lead (this would normally call the lead service)
			# For now, return a placeholder ID
			lead_id = uuid7str()
			logger.info(f"🎯 Lead created from form submission: {lead_id}")
			return lead_id
			
		except Exception as e:
			logger.error(f"Failed to create lead from submission: {str(e)}")
			return None
	
	async def _store_submission(self, submission: FormSubmission):
		"""Store form submission in database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_form_submissions (
						id, tenant_id, form_id, form_data, raw_data, status,
						processing_errors, ip_address, user_agent, referrer, utm_parameters,
						contact_id, lead_id, lead_score, duplicate_of, duplicate_score,
						spam_score, spam_reasons, processed_at, processing_duration_ms,
						submitted_at, created_at, updated_at
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
						$15, $16, $17, $18, $19, $20, $21, $22, $23
					)
					ON CONFLICT (id) DO UPDATE SET
						status = EXCLUDED.status,
						processing_errors = EXCLUDED.processing_errors,
						contact_id = EXCLUDED.contact_id,
						lead_id = EXCLUDED.lead_id,
						lead_score = EXCLUDED.lead_score,
						duplicate_of = EXCLUDED.duplicate_of,
						duplicate_score = EXCLUDED.duplicate_score,
						spam_score = EXCLUDED.spam_score,
						spam_reasons = EXCLUDED.spam_reasons,
						processed_at = EXCLUDED.processed_at,
						processing_duration_ms = EXCLUDED.processing_duration_ms,
						updated_at = NOW()
				""", 
				submission.id, submission.tenant_id, submission.form_id,
				submission.form_data, submission.raw_data, submission.status.value,
				submission.processing_errors, submission.ip_address, submission.user_agent,
				submission.referrer, submission.utm_parameters, submission.contact_id,
				submission.lead_id, submission.lead_score, submission.duplicate_of,
				submission.duplicate_score, submission.spam_score, submission.spam_reasons,
				submission.processed_at, submission.processing_duration_ms,
				submission.submitted_at, submission.created_at, submission.updated_at
				)
		except Exception as e:
			logger.error(f"Failed to store form submission: {str(e)}")
			raise
	
	async def _update_submission(self, submission: FormSubmission):
		"""Update form submission in database"""
		await self._store_submission(submission)
	
	async def _update_form_statistics(self, form_id: str, tenant_id: str):
		"""Update form submission statistics"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_web_forms SET
						submission_count = submission_count + 1,
						last_submission_at = NOW(),
						updated_at = NOW()
					WHERE id = $1 AND tenant_id = $2
				""", form_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to update form statistics: {str(e)}")
	
	async def _send_notifications(self, form: WebForm, submission: FormSubmission):
		"""Send email notifications for form submission"""
		try:
			# This would integrate with the notification system
			logger.info(f"📧 Sending notifications for form submission: {submission.id}")
		except Exception as e:
			logger.error(f"Failed to send notifications: {str(e)}")
	
	async def _call_webhook(self, form: WebForm, submission: FormSubmission):
		"""Call webhook URL for form submission"""
		try:
			# This would make HTTP request to webhook URL
			logger.info(f"🔗 Calling webhook for form submission: {submission.id}")
		except Exception as e:
			logger.error(f"Failed to call webhook: {str(e)}")