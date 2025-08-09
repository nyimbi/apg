"""
APG Notification Capability - Pydantic v2 API Models

API request/response models and validation schemas using Pydantic v2 with modern typing.
These models complement the SQLAlchemy ORM models for API interactions, validation,
and serialization while maintaining clean separation of concerns.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, validator, field_validator
from pydantic.types import EmailStr, HttpUrl
from uuid_extensions import uuid7str


# Configuration for all Pydantic models
model_config = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	validate_assignment=True
)


# Enum Types for Type Safety
class NotificationTemplateType(str, Enum):
	"""Template types for different notification purposes"""
	EMAIL = "email"
	SMS = "sms"
	PUSH = "push"
	IN_APP = "in_app"
	VOICE = "voice"
	SOCIAL = "social"
	MESSAGING = "messaging"
	IOT = "iot"
	ARVR = "arvr"
	GAMING = "gaming"
	AUTOMOTIVE = "automotive"
	LEGACY = "legacy"
	UNIVERSAL = "universal"


class DeliveryChannel(str, Enum):
	"""All supported delivery channels (25+)"""
	# Core Communication
	EMAIL = "email"
	SMS = "sms"
	VOICE = "voice"
	PUSH = "push"
	IN_APP = "in_app"
	
	# Social Media
	WHATSAPP = "whatsapp"
	TWITTER = "twitter"
	FACEBOOK = "facebook"
	LINKEDIN = "linkedin"
	INSTAGRAM = "instagram"
	
	# Messaging & Collaboration
	SLACK = "slack"
	TEAMS = "teams"
	DISCORD = "discord"
	TELEGRAM = "telegram"
	WECHAT = "wechat"
	
	# Mobile & Desktop
	NATIVE_MOBILE = "native_mobile"
	DESKTOP = "desktop"
	WEB_PUSH = "web_push"
	PWA = "pwa"
	
	# IoT & Smart Devices
	MQTT = "mqtt"
	ALEXA = "alexa"
	GOOGLE_ASSISTANT = "google_assistant"
	WEARABLES = "wearables"
	SMART_HOME = "smart_home"
	
	# AR/VR & Gaming
	ARKIT = "arkit"
	OCULUS = "oculus"
	STEAM = "steam"
	XBOX = "xbox"
	PLAYSTATION = "playstation"
	
	# Automotive
	ANDROID_AUTO = "android_auto"
	CARPLAY = "carplay"
	TESLA = "tesla"
	
	# Legacy & Specialized
	FAX = "fax"
	PRINT = "print"
	RSS = "rss"
	DIGITAL_SIGNAGE = "digital_signage"


class NotificationPriority(str, Enum):
	"""Notification priority levels"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"
	CRITICAL = "critical"


class CampaignType(str, Enum):
	"""Campaign types for different notification strategies"""
	TRANSACTIONAL = "transactional"
	MARKETING = "marketing"
	DRIP = "drip"
	BLAST = "blast"
	TRIGGERED = "triggered"
	AB_TEST = "ab_test"
	BEHAVIORAL = "behavioral"
	LIFECYCLE = "lifecycle"


class EngagementEvent(str, Enum):
	"""User engagement event types"""
	SENT = "sent"
	DELIVERED = "delivered"
	OPENED = "opened"
	CLICKED = "clicked"
	INTERACTED = "interacted"
	REPLIED = "replied"
	FORWARDED = "forwarded"
	DISMISSED = "dismissed"
	UNSUBSCRIBED = "unsubscribed"


class ConversionEvent(str, Enum):
	"""Conversion tracking event types"""
	PAGE_VIEW = "page_view"
	SIGNUP = "signup"
	PURCHASE = "purchase"
	DOWNLOAD = "download"
	FORM_SUBMIT = "form_submit"
	CUSTOM = "custom"


# Core Notification Models
class NotificationTemplateBase(BaseModel):
	"""Base template model with common fields"""
	model_config = model_config
	
	name: str = Field(..., min_length=1, max_length=200, description="Template name")
	description: Optional[str] = Field(None, max_length=1000, description="Template description")
	template_type: NotificationTemplateType = Field(..., description="Template type")
	supported_channels: List[DeliveryChannel] = Field(default_factory=list, description="Supported delivery channels")
	is_active: bool = Field(True, description="Whether template is active")


class NotificationTemplateCreate(NotificationTemplateBase):
	"""Template creation model"""
	content: Dict[str, Any] = Field(..., description="Template content for different channels")
	variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables schema")
	personalization_rules: List[Dict[str, Any]] = Field(default_factory=list, description="AI personalization rules")


class NotificationTemplateUpdate(BaseModel):
	"""Template update model"""
	model_config = model_config
	
	name: Optional[str] = Field(None, min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	content: Optional[Dict[str, Any]] = None
	variables: Optional[Dict[str, Any]] = None
	personalization_rules: Optional[List[Dict[str, Any]]] = None
	supported_channels: Optional[List[DeliveryChannel]] = None
	is_active: Optional[bool] = None


class UltimateNotificationTemplate(NotificationTemplateBase):
	"""Complete notification template with all features"""
	id: str = Field(default_factory=uuid7str, description="Unique template ID")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	version: str = Field("1.0.0", description="Template version")
	
	# Content and Configuration
	content: Dict[str, Any] = Field(..., description="Multi-channel template content")
	variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables schema")
	default_variables: Dict[str, Any] = Field(default_factory=dict, description="Default variable values")
	
	# Advanced Features
	personalization_rules: List[Dict[str, Any]] = Field(default_factory=list, description="AI personalization rules")
	rich_media_assets: List[Dict[str, Any]] = Field(default_factory=list, description="Rich media content")
	interactive_elements: List[Dict[str, Any]] = Field(default_factory=list, description="Interactive UI elements")
	geofencing_rules: Optional[Dict[str, Any]] = Field(None, description="Location-based targeting rules")
	
	# A/B Testing
	ab_test_variants: List[Dict[str, Any]] = Field(default_factory=list, description="A/B test variations")
	
	# Performance and Analytics
	usage_count: int = Field(0, description="Number of times used")
	success_rate: float = Field(0.0, ge=0.0, le=100.0, description="Template success rate percentage")
	engagement_score: float = Field(0.0, ge=0.0, le=100.0, description="Engagement score")
	
	# Timestamps
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	
	@field_validator('supported_channels')
	@classmethod
	def validate_channels(cls, v):
		if not v:
			raise ValueError("At least one delivery channel must be specified")
		return v


# Campaign Models
class CampaignBase(BaseModel):
	"""Base campaign model"""
	model_config = model_config
	
	name: str = Field(..., min_length=1, max_length=200, description="Campaign name")
	description: Optional[str] = Field(None, max_length=1000, description="Campaign description")
	campaign_type: CampaignType = Field(..., description="Campaign type")


class CampaignCreate(CampaignBase):
	"""Campaign creation model"""
	template_ids: List[str] = Field(..., min_items=1, description="Template IDs to use")
	audience_segments: List[Dict[str, Any]] = Field(..., min_items=1, description="Target audience segments")
	channels: List[DeliveryChannel] = Field(..., min_items=1, description="Delivery channels")
	scheduled_at: Optional[datetime] = Field(None, description="Scheduled execution time")
	priority: NotificationPriority = Field(NotificationPriority.NORMAL, description="Campaign priority")


class AdvancedCampaign(CampaignBase):
	"""Complete campaign model with orchestration capabilities"""
	id: str = Field(default_factory=uuid7str, description="Unique campaign ID")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	
	# Campaign Configuration
	template_ids: List[str] = Field(..., description="Notification template IDs")
	audience_segments: List[Dict[str, Any]] = Field(..., description="Target audience definitions")
	channels: List[DeliveryChannel] = Field(..., description="Enabled delivery channels")
	
	# Orchestration and Automation
	orchestration_rules: Dict[str, Any] = Field(default_factory=dict, description="Channel orchestration rules")
	automation_workflows: List[Dict[str, Any]] = Field(default_factory=list, description="Automated workflow steps")
	triggers: List[Dict[str, Any]] = Field(default_factory=list, description="Campaign trigger conditions")
	
	# Timing and Scheduling  
	priority: NotificationPriority = Field(NotificationPriority.NORMAL, description="Campaign priority")
	scheduled_at: Optional[datetime] = Field(None, description="Scheduled execution time")
	expires_at: Optional[datetime] = Field(None, description="Campaign expiration time")
	timezone: str = Field("UTC", description="Campaign timezone")
	
	# Collaboration and Approval
	collaboration_settings: Dict[str, Any] = Field(default_factory=dict, description="Real-time collaboration config")
	approval_workflow: List[Dict[str, Any]] = Field(default_factory=list, description="Approval workflow steps")
	stakeholders: List[str] = Field(default_factory=list, description="Campaign stakeholder user IDs")
	
	# Performance Tracking
	analytics_config: Dict[str, Any] = Field(default_factory=dict, description="Analytics configuration")
	tracking_enabled: bool = Field(True, description="Enable performance tracking")
	
	# Campaign Status
	status: str = Field("draft", description="Campaign status")
	execution_count: int = Field(0, description="Number of executions")
	total_recipients: int = Field(0, description="Total recipient count")
	
	# Timestamps
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	executed_at: Optional[datetime] = Field(None, description="Last execution time")


# User Preference Models
class ChannelPreference(BaseModel):
	"""Individual channel preference settings"""
	model_config = model_config
	
	enabled: bool = Field(True, description="Channel enabled for user")
	frequency: str = Field("normal", description="Notification frequency preference")
	quiet_hours: Optional[Dict[str, str]] = Field(None, description="Quiet hours configuration")
	address: Optional[str] = Field(None, description="Channel-specific address (email, phone, etc.)")


class UltimateUserPreferences(BaseModel):
	"""Comprehensive user notification preferences"""
	model_config = model_config
	
	user_id: str = Field(..., description="User ID")
	tenant_id: str = Field(..., description="Tenant ID")
	
	# Channel Preferences
	channel_preferences: Dict[DeliveryChannel, ChannelPreference] = Field(
		default_factory=dict, description="Per-channel preference settings"
	)
	
	# Personalization Settings
	personalization_enabled: bool = Field(True, description="Enable AI personalization")
	language_preference: str = Field("en-US", description="Preferred language")
	timezone: str = Field("UTC", description="User timezone")
	
	# Privacy and Consent
	privacy_preferences: Dict[str, Any] = Field(default_factory=dict, description="Privacy settings")
	consent_management: Dict[str, Any] = Field(default_factory=dict, description="Consent records")
	data_sharing_consent: bool = Field(False, description="Allow data sharing for personalization")
	
	# Advanced Preferences
	geolocation_enabled: bool = Field(False, description="Enable location-based notifications")
	rich_media_enabled: bool = Field(True, description="Enable rich media content")
	interactive_enabled: bool = Field(True, description="Enable interactive elements")
	
	# Frequency and Timing
	global_frequency_cap: Optional[int] = Field(None, description="Maximum notifications per day")
	optimal_send_times: Dict[str, List[str]] = Field(default_factory=dict, description="AI-learned optimal send times")
	
	# Engagement Profile
	engagement_score: float = Field(0.0, ge=0.0, le=100.0, description="User engagement score")
	last_engagement: Optional[datetime] = Field(None, description="Last engagement timestamp")
	
	# Timestamps
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# Delivery and Tracking Models
class DeliveryRequest(BaseModel):
	"""Notification delivery request"""
	model_config = model_config
	
	recipient_id: str = Field(..., description="Recipient user ID")
	template_id: str = Field(..., description="Template ID to use")
	channels: List[DeliveryChannel] = Field(..., min_items=1, description="Delivery channels")
	variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
	priority: NotificationPriority = Field(NotificationPriority.NORMAL, description="Delivery priority")
	scheduled_at: Optional[datetime] = Field(None, description="Scheduled delivery time")
	expires_at: Optional[datetime] = Field(None, description="Expiration time")
	
	# Advanced Options
	personalization_enabled: bool = Field(True, description="Enable AI personalization")
	tracking_enabled: bool = Field(True, description="Enable engagement tracking")
	rich_media_enabled: bool = Field(True, description="Enable rich media content")
	
	# Context Information
	context: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
	campaign_id: Optional[str] = Field(None, description="Associated campaign ID")
	tags: List[str] = Field(default_factory=list, description="Delivery tags")


class ComprehensiveDelivery(BaseModel):
	"""Complete delivery tracking with analytics"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique delivery ID")
	tenant_id: str = Field(..., description="Tenant ID")
	campaign_id: Optional[str] = Field(None, description="Associated campaign ID")
	
	# Recipient Information
	recipient_id: str = Field(..., description="Recipient user ID")
	recipient_data: Dict[str, Any] = Field(default_factory=dict, description="Recipient context data")
	
	# Delivery Configuration
	template_id: str = Field(..., description="Used template ID")
	channels: List[DeliveryChannel] = Field(..., description="Attempted delivery channels")
	successful_channels: List[DeliveryChannel] = Field(default_factory=list, description="Successfully delivered channels")
	failed_channels: List[DeliveryChannel] = Field(default_factory=list, description="Failed delivery channels")
	
	# Content and Personalization
	personalized_content: Dict[str, Any] = Field(default_factory=dict, description="Personalized content")
	rich_media_content: List[Dict[str, Any]] = Field(default_factory=list, description="Rich media assets")
	variables_used: Dict[str, Any] = Field(default_factory=dict, description="Template variables used")
	
	# Delivery Status and Metrics
	priority: NotificationPriority = Field(..., description="Delivery priority")
	status: str = Field("pending", description="Overall delivery status")
	delivery_attempts: int = Field(0, description="Number of delivery attempts")
	
	# Engagement Tracking
	engagement_events: List[Dict[str, Any]] = Field(default_factory=list, description="User engagement events")
	conversion_events: List[Dict[str, Any]] = Field(default_factory=list, description="Conversion tracking events")
	
	# Performance Data
	delivery_latency_ms: Optional[int] = Field(None, description="Delivery latency in milliseconds")
	processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
	cost: Optional[float] = Field(None, description="Delivery cost")
	
	# Geolocation and Device Info
	geolocation_data: Optional[Dict[str, Any]] = Field(None, description="User location data")
	device_information: Dict[str, Any] = Field(default_factory=dict, description="Device/client information")
	
	# Timestamps
	created_at: datetime = Field(default_factory=datetime.utcnow)
	scheduled_at: Optional[datetime] = Field(None, description="Scheduled delivery time")
	sent_at: Optional[datetime] = Field(None, description="Actual send time")
	delivered_at: Optional[datetime] = Field(None, description="Delivery confirmation time")
	first_opened_at: Optional[datetime] = Field(None, description="First open time")
	last_interaction_at: Optional[datetime] = Field(None, description="Last interaction time")


# Analytics Models
class EngagementMetrics(BaseModel):
	"""Engagement analytics metrics"""
	model_config = model_config
	
	total_sent: int = Field(0, description="Total notifications sent")
	total_delivered: int = Field(0, description="Total successfully delivered")
	total_opened: int = Field(0, description="Total opened notifications")
	total_clicked: int = Field(0, description="Total clicked notifications")
	total_converted: int = Field(0, description="Total conversions")
	
	# Calculated Rates
	delivery_rate: float = Field(0.0, ge=0.0, le=100.0, description="Delivery success rate")
	open_rate: float = Field(0.0, ge=0.0, le=100.0, description="Open rate percentage")
	click_rate: float = Field(0.0, ge=0.0, le=100.0, description="Click-through rate")
	conversion_rate: float = Field(0.0, ge=0.0, le=100.0, description="Conversion rate")
	
	# Advanced Metrics
	engagement_score: float = Field(0.0, ge=0.0, le=100.0, description="Overall engagement score")
	time_to_open_avg: Optional[float] = Field(None, description="Average time to open (seconds)")
	time_to_click_avg: Optional[float] = Field(None, description="Average time to click (seconds)")


class UltimateAnalytics(BaseModel):
	"""Comprehensive analytics with business intelligence"""
	model_config = model_config
	
	# Analysis Period
	period_start: datetime = Field(..., description="Analysis period start")
	period_end: datetime = Field(..., description="Analysis period end")
	generated_at: datetime = Field(default_factory=datetime.utcnow, description="Report generation time")
	
	# Basic Metrics
	engagement_metrics: EngagementMetrics = Field(..., description="Core engagement metrics")
	
	# Channel Performance
	channel_performance: Dict[DeliveryChannel, EngagementMetrics] = Field(
		default_factory=dict, description="Per-channel performance metrics"
	)
	
	# Campaign Analytics
	campaign_id: Optional[str] = Field(None, description="Campaign ID if campaign-specific")
	campaign_performance: Dict[str, Any] = Field(default_factory=dict, description="Campaign-specific metrics")
	
	# Advanced Analytics
	audience_insights: Dict[str, Any] = Field(default_factory=dict, description="Audience behavior insights")
	predictive_insights: Dict[str, Any] = Field(default_factory=dict, description="AI-generated predictions")
	attribution_data: Dict[str, Any] = Field(default_factory=dict, description="Multi-touch attribution")
	roi_metrics: Dict[str, Any] = Field(default_factory=dict, description="ROI and business impact")
	
	# Geographic and Temporal Analysis
	geographic_breakdown: Dict[str, Any] = Field(default_factory=dict, description="Geographic performance")
	temporal_patterns: Dict[str, Any] = Field(default_factory=dict, description="Time-based patterns")
	cohort_analysis: Dict[str, Any] = Field(default_factory=dict, description="User cohort analysis")
	
	# Optimization Recommendations
	optimization_suggestions: List[Dict[str, Any]] = Field(default_factory=list, description="AI optimization recommendations")
	ab_test_results: List[Dict[str, Any]] = Field(default_factory=list, description="A/B test outcomes")


# API Response Models
class ApiResponse(BaseModel):
	"""Standard API response wrapper"""
	model_config = model_config
	
	success: bool = Field(..., description="Request success status")
	message: str = Field(..., description="Response message")
	data: Optional[Any] = Field(None, description="Response data")
	errors: Optional[List[str]] = Field(None, description="Error messages if any")
	metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class PaginatedResponse(BaseModel):
	"""Paginated response wrapper"""
	model_config = model_config
	
	items: List[Any] = Field(..., description="Response items")
	total_count: int = Field(..., description="Total number of items")
	page: int = Field(..., description="Current page number")
	page_size: int = Field(..., description="Items per page")
	total_pages: int = Field(..., description="Total number of pages")
	has_next: bool = Field(..., description="Whether there are more pages")
	has_previous: bool = Field(..., description="Whether there are previous pages")


# Security and Compliance Models
class SecurityComplianceRecord(BaseModel):
	"""Security and compliance tracking"""
	model_config = model_config
	
	record_id: str = Field(default_factory=uuid7str, description="Unique record ID")
	user_id: str = Field(..., description="Associated user ID")
	tenant_id: str = Field(..., description="Tenant ID")
	
	# Data Classification
	data_classification: str = Field("internal", description="Data sensitivity classification")
	encryption_status: Dict[str, Any] = Field(default_factory=dict, description="Encryption information")
	
	# Consent Management
	consent_records: List[Dict[str, Any]] = Field(default_factory=list, description="User consent records")
	privacy_preferences: Dict[str, Any] = Field(default_factory=dict, description="Privacy settings")
	
	# Compliance Status
	gdpr_compliant: bool = Field(True, description="GDPR compliance status")
	ccpa_compliant: bool = Field(True, description="CCPA compliance status")
	hipaa_compliant: bool = Field(False, description="HIPAA compliance status")
	
	# Audit Trail
	audit_events: List[Dict[str, Any]] = Field(default_factory=list, description="Audit trail events")
	access_log: List[Dict[str, Any]] = Field(default_factory=list, description="Data access log")
	
	# Data Retention
	retention_policy: Dict[str, Any] = Field(default_factory=dict, description="Data retention policy")
	scheduled_deletion: Optional[datetime] = Field(None, description="Scheduled deletion date")
	
	# Timestamps
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# Advanced Feature Models
class GeofencingRules(BaseModel):
	"""Geofencing and location-based notification rules"""
	model_config = model_config
	
	enabled: bool = Field(False, description="Enable geofencing")
	locations: List[Dict[str, Any]] = Field(default_factory=list, description="Geofence locations")
	triggers: List[Dict[str, Any]] = Field(default_factory=list, description="Location trigger conditions")
	privacy_settings: Dict[str, Any] = Field(default_factory=dict, description="Location privacy settings")


class InteractiveElement(BaseModel):
	"""Interactive notification elements"""
	model_config = model_config
	
	element_type: str = Field(..., description="Element type (button, form, etc.)")
	configuration: Dict[str, Any] = Field(..., description="Element configuration")
	actions: List[Dict[str, Any]] = Field(default_factory=list, description="Available actions")
	analytics_enabled: bool = Field(True, description="Track interactions")


class RichMediaContent(BaseModel):
	"""Rich media content for notifications"""
	model_config = model_config
	
	media_type: str = Field(..., description="Media type (image, video, audio)")
	url: HttpUrl = Field(..., description="Media URL")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Media metadata")
	accessibility: Dict[str, Any] = Field(default_factory=dict, description="Accessibility information")


class ABTestVariant(BaseModel):
	"""A/B test variant configuration"""
	model_config = model_config
	
	variant_id: str = Field(default_factory=uuid7str, description="Unique variant ID")
	name: str = Field(..., description="Variant name")
	traffic_percentage: float = Field(..., ge=0.0, le=100.0, description="Traffic allocation percentage")
	configuration: Dict[str, Any] = Field(..., description="Variant configuration")
	performance_metrics: Optional[EngagementMetrics] = Field(None, description="Variant performance")


# Export all models for easy importing
__all__ = [
	# Enums
	"NotificationTemplateType", "DeliveryChannel", "NotificationPriority", 
	"CampaignType", "EngagementEvent", "ConversionEvent",
	
	# Core Models
	"NotificationTemplateBase", "NotificationTemplateCreate", "NotificationTemplateUpdate",
	"UltimateNotificationTemplate", "CampaignBase", "CampaignCreate", "AdvancedCampaign",
	"UltimateUserPreferences", "ChannelPreference",
	
	# Delivery Models
	"DeliveryRequest", "ComprehensiveDelivery",
	
	# Analytics Models
	"EngagementMetrics", "UltimateAnalytics",
	
	# API Models
	"ApiResponse", "PaginatedResponse",
	
	# Security & Compliance
	"SecurityComplianceRecord",
	
	# Advanced Features
	"GeofencingRules", "InteractiveElement", "RichMediaContent", "ABTestVariant"
]