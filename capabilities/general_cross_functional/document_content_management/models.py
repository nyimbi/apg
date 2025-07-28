"""
Document Content Management - Comprehensive Pydantic Models

Enterprise-grade document management, content collaboration, knowledge management,
and digital asset management supporting efficient content lifecycle management.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.config import ConfigDict
from pydantic.types import EmailStr, HttpUrl, Json
from uuid_extensions import uuid7str


class ConfigDict(ConfigDict):
	extra = 'forbid'
	validate_by_name = True
	validate_by_alias = True


# Base Document Management Model
class DCMBase(BaseModel):
	model_config = ConfigDict()
	
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(..., description="Multi-tenant organization identifier")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User ID who created the record")
	updated_by: str = Field(..., description="User ID who last updated the record")
	is_active: bool = Field(default=True, description="Active status flag")
	metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


# Enumeration Types
class DCMDocumentStatus(str, Enum):
	DRAFT = "draft"
	REVIEW = "review"
	APPROVED = "approved"
	PUBLISHED = "published"
	ARCHIVED = "archived"
	OBSOLETE = "obsolete"
	LOCKED = "locked"
	CHECKED_OUT = "checked_out"


class DCMDocumentType(str, Enum):
	TEXT_DOCUMENT = "text_document"
	SPREADSHEET = "spreadsheet"
	PRESENTATION = "presentation"
	PDF = "pdf"
	IMAGE = "image"
	VIDEO = "video"
	AUDIO = "audio"
	ARCHIVE = "archive"
	CAD_DRAWING = "cad_drawing"
	CONTRACT = "contract"
	INVOICE = "invoice"
	POLICY = "policy"
	PROCEDURE = "procedure"
	MANUAL = "manual"
	FORM = "form"
	TEMPLATE = "template"
	CUSTOM = "custom"


class DCMPermissionLevel(str, Enum):
	NONE = "none"
	READ = "read"
	WRITE = "write"
	DELETE = "delete"
	ADMIN = "admin"
	OWNER = "owner"


class DCMAccessType(str, Enum):
	PRIVATE = "private"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	PUBLIC = "public"
	RESTRICTED = "restricted"


class DCMWorkflowStatus(str, Enum):
	NOT_STARTED = "not_started"
	IN_PROGRESS = "in_progress"
	PENDING_REVIEW = "pending_review"
	PENDING_APPROVAL = "pending_approval"
	APPROVED = "approved"
	REJECTED = "rejected"
	COMPLETED = "completed"
	CANCELLED = "cancelled"


class DCMContentFormat(str, Enum):
	DOCX = "docx"
	PDF = "pdf"
	XLSX = "xlsx"
	PPTX = "pptx"
	TXT = "txt"
	HTML = "html"
	MD = "md"
	JSON = "json"
	XML = "xml"
	CSV = "csv"
	JPG = "jpg"
	PNG = "png"
	GIF = "gif"
	MP4 = "mp4"
	MP3 = "mp3"
	ZIP = "zip"
	OTHER = "other"


class DCMNotificationType(str, Enum):
	DOCUMENT_CREATED = "document_created"
	DOCUMENT_UPDATED = "document_updated"
	DOCUMENT_SHARED = "document_shared"
	COMMENT_ADDED = "comment_added"
	APPROVAL_REQUEST = "approval_request"
	APPROVAL_GRANTED = "approval_granted"
	APPROVAL_REJECTED = "approval_rejected"
	WORKFLOW_ASSIGNED = "workflow_assigned"
	DUE_DATE_REMINDER = "due_date_reminder"
	RETENTION_NOTICE = "retention_notice"


# Core Document Models
class DCMFolder(DCMBase):
	"""Hierarchical folder structure for document organization"""
	name: str = Field(..., description="Folder name")
	description: Optional[str] = Field(default=None, description="Folder description")
	parent_folder_id: Optional[str] = Field(default=None, description="Parent folder ID")
	folder_path: str = Field(..., description="Full folder path")
	level: int = Field(default=0, description="Hierarchy level (0 = root)")
	color: Optional[str] = Field(default=None, description="Folder color for UI")
	icon: Optional[str] = Field(default=None, description="Folder icon identifier")
	is_system: bool = Field(default=False, description="System-generated folder")
	is_template: bool = Field(default=False, description="Template folder")
	access_type: DCMAccessType = Field(default=DCMAccessType.INTERNAL, description="Default access type")
	inherited_permissions: bool = Field(default=True, description="Inherit permissions from parent")
	custom_properties: Dict[str, Any] = Field(default_factory=dict, description="Custom folder properties")
	sort_order: int = Field(default=0, description="Display sort order")
	document_count: int = Field(default=0, description="Number of documents in folder")
	subfolder_count: int = Field(default=0, description="Number of subfolders")


class DCMDocument(DCMBase):
	"""Core document entity with comprehensive metadata"""
	name: str = Field(..., description="Document name")
	title: str = Field(..., description="Document title")
	description: Optional[str] = Field(default=None, description="Document description")
	folder_id: Optional[str] = Field(default=None, description="Parent folder ID")
	document_type: DCMDocumentType = Field(..., description="Type of document")
	content_format: DCMContentFormat = Field(..., description="Content format/file type")
	status: DCMDocumentStatus = Field(default=DCMDocumentStatus.DRAFT, description="Current document status")
	access_type: DCMAccessType = Field(default=DCMAccessType.INTERNAL, description="Access classification")
	
	# File Information
	file_name: str = Field(..., description="Original file name")
	file_size: int = Field(..., description="File size in bytes")
	file_hash: str = Field(..., description="File content hash (SHA-256)")
	mime_type: str = Field(..., description="MIME type")
	storage_path: str = Field(..., description="Storage location path")
	
	# Version Control
	version_number: str = Field(default="1.0", description="Current version number")
	major_version: int = Field(default=1, description="Major version number")
	minor_version: int = Field(default=0, description="Minor version number")
	is_latest_version: bool = Field(default=True, description="Is this the latest version")
	
	# Content Metadata
	language: str = Field(default="en", description="Content language")
	keywords: List[str] = Field(default_factory=list, description="Document keywords/tags")
	categories: List[str] = Field(default_factory=list, description="Document categories")
	subject: Optional[str] = Field(default=None, description="Document subject")
	author: Optional[str] = Field(default=None, description="Document author")
	
	# Dates and Lifecycle
	published_date: Optional[datetime] = Field(default=None, description="Publication date")
	expiry_date: Optional[date] = Field(default=None, description="Document expiry date")
	review_date: Optional[date] = Field(default=None, description="Next review date")
	retention_date: Optional[date] = Field(default=None, description="Retention/deletion date")
	
	# Analytics and Usage
	view_count: int = Field(default=0, description="Number of times viewed")
	download_count: int = Field(default=0, description="Number of times downloaded")
	share_count: int = Field(default=0, description="Number of times shared")
	last_viewed_at: Optional[datetime] = Field(default=None, description="Last view timestamp")
	last_downloaded_at: Optional[datetime] = Field(default=None, description="Last download timestamp")
	
	# Collaboration
	is_locked: bool = Field(default=False, description="Document locked for editing")
	locked_by: Optional[str] = Field(default=None, description="User who locked the document")
	locked_at: Optional[datetime] = Field(default=None, description="Lock timestamp")
	checked_out_by: Optional[str] = Field(default=None, description="User who checked out document")
	checked_out_at: Optional[datetime] = Field(default=None, description="Check-out timestamp")
	
	# Content Analysis
	extracted_text: Optional[str] = Field(default=None, description="OCR/extracted text content")
	content_summary: Optional[str] = Field(default=None, description="AI-generated content summary")
	ai_tags: List[str] = Field(default_factory=list, description="AI-generated tags")
	sentiment_score: Optional[float] = Field(default=None, description="Content sentiment score")
	
	# Security and Compliance
	is_encrypted: bool = Field(default=False, description="Document encryption status")
	encryption_key_id: Optional[str] = Field(default=None, description="Encryption key identifier")
	classification_level: Optional[str] = Field(default=None, description="Security classification")
	compliance_tags: List[str] = Field(default_factory=list, description="Compliance requirement tags")
	
	# Custom Properties
	custom_properties: Dict[str, Any] = Field(default_factory=dict, description="Custom document properties")
	business_metadata: Dict[str, Any] = Field(default_factory=dict, description="Business-specific metadata")


class DCMDocumentVersion(DCMBase):
	"""Document version history and management"""
	document_id: str = Field(..., description="Parent document ID")
	version_number: str = Field(..., description="Version number (e.g., 1.0, 1.1, 2.0)")
	major_version: int = Field(..., description="Major version number")
	minor_version: int = Field(..., description="Minor version number")
	version_label: Optional[str] = Field(default=None, description="Version label/name")
	change_description: Optional[str] = Field(default=None, description="Description of changes")
	change_type: str = Field(..., description="Type of change (major, minor, patch)")
	
	# File Information
	file_name: str = Field(..., description="Version file name")
	file_size: int = Field(..., description="File size in bytes")
	file_hash: str = Field(..., description="File content hash")
	storage_path: str = Field(..., description="Version storage location")
	
	# Version Metadata
	is_current: bool = Field(default=False, description="Is this the current version")
	status: DCMDocumentStatus = Field(..., description="Version status")
	approved_by: Optional[str] = Field(default=None, description="User who approved this version")
	approved_at: Optional[datetime] = Field(default=None, description="Approval timestamp")
	
	# Content Changes
	content_diff: Optional[Dict[str, Any]] = Field(default=None, description="Content differences from previous version")
	word_count: Optional[int] = Field(default=None, description="Document word count")
	page_count: Optional[int] = Field(default=None, description="Document page count")
	
	# Lifecycle
	created_from_version: Optional[str] = Field(default=None, description="Source version ID")
	branched_from: Optional[str] = Field(default=None, description="Branch source version ID")
	merged_versions: List[str] = Field(default_factory=list, description="Merged version IDs")
	
	# Analytics
	download_count: int = Field(default=0, description="Version download count")
	restoration_count: int = Field(default=0, description="Times restored as current")


class DCMDocumentPermission(DCMBase):
	"""Document and folder access permissions"""
	resource_id: str = Field(..., description="Document or folder ID")
	resource_type: str = Field(..., description="Resource type (document/folder)")
	subject_id: str = Field(..., description="User, group, or role ID")
	subject_type: str = Field(..., description="Subject type (user/group/role)")
	permission_level: DCMPermissionLevel = Field(..., description="Permission level")
	
	# Permission Details
	can_read: bool = Field(default=False, description="Read permission")
	can_write: bool = Field(default=False, description="Write permission")
	can_delete: bool = Field(default=False, description="Delete permission")
	can_share: bool = Field(default=False, description="Share permission")
	can_download: bool = Field(default=False, description="Download permission")
	can_print: bool = Field(default=False, description="Print permission")
	can_export: bool = Field(default=False, description="Export permission")
	can_comment: bool = Field(default=False, description="Comment permission")
	can_approve: bool = Field(default=False, description="Approval permission")
	
	# Time-based Permissions
	valid_from: Optional[datetime] = Field(default=None, description="Permission start date")
	valid_until: Optional[datetime] = Field(default=None, description="Permission expiry date")
	
	# Conditional Permissions
	ip_restrictions: List[str] = Field(default_factory=list, description="IP address restrictions")
	device_restrictions: List[str] = Field(default_factory=list, description="Device restrictions")
	location_restrictions: List[str] = Field(default_factory=list, description="Geographic restrictions")
	
	# Delegation
	can_delegate: bool = Field(default=False, description="Can delegate permissions")
	delegated_by: Optional[str] = Field(default=None, description="Delegating user ID")
	delegation_level: int = Field(default=0, description="Delegation depth level")
	
	# Audit
	granted_by: str = Field(..., description="User who granted permission")
	granted_at: datetime = Field(default_factory=datetime.utcnow, description="Permission grant timestamp")
	last_accessed: Optional[datetime] = Field(default=None, description="Last access using this permission")
	access_count: int = Field(default=0, description="Number of accesses")


# Collaboration Models
class DCMComment(DCMBase):
	"""Document comments and annotations"""
	document_id: str = Field(..., description="Document ID")
	document_version: Optional[str] = Field(default=None, description="Specific document version")
	parent_comment_id: Optional[str] = Field(default=None, description="Parent comment for threading")
	comment_text: str = Field(..., description="Comment content")
	comment_type: str = Field(default="general", description="Comment type (general, review, approval)")
	
	# Positioning and Context
	page_number: Optional[int] = Field(default=None, description="Page number for annotation")
	position_data: Optional[Dict[str, Any]] = Field(default=None, description="Position/coordinate data")
	highlighted_text: Optional[str] = Field(default=None, description="Highlighted/selected text")
	context_snippet: Optional[str] = Field(default=None, description="Surrounding text context")
	
	# Status and Resolution
	is_resolved: bool = Field(default=False, description="Comment resolved status")
	resolved_by: Optional[str] = Field(default=None, description="User who resolved comment")
	resolved_at: Optional[datetime] = Field(default=None, description="Resolution timestamp")
	resolution_note: Optional[str] = Field(default=None, description="Resolution explanation")
	
	# Threading and Collaboration
	thread_id: str = Field(default_factory=uuid7str, description="Comment thread identifier")
	reply_count: int = Field(default=0, description="Number of replies")
	mention_users: List[str] = Field(default_factory=list, description="Mentioned user IDs")
	
	# Rich Content
	attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Comment attachments")
	formatting: Optional[Dict[str, Any]] = Field(default=None, description="Rich text formatting")
	
	# Visibility and Access
	is_private: bool = Field(default=False, description="Private comment visibility")
	visible_to: List[str] = Field(default_factory=list, description="Users with visibility")
	
	# Workflow Integration
	workflow_step_id: Optional[str] = Field(default=None, description="Associated workflow step")
	approval_decision: Optional[str] = Field(default=None, description="Approval decision (approve/reject)")


class DCMWorkflow(DCMBase):
	"""Document workflow and approval processes"""
	name: str = Field(..., description="Workflow name")
	description: Optional[str] = Field(default=None, description="Workflow description")
	workflow_type: str = Field(..., description="Type of workflow (review, approval, publication)")
	is_template: bool = Field(default=False, description="Template workflow flag")
	
	# Workflow Definition
	workflow_steps: List[Dict[str, Any]] = Field(..., description="Workflow step definitions")
	step_sequence: List[str] = Field(..., description="Step execution sequence")
	parallel_steps: List[List[str]] = Field(default_factory=list, description="Parallel step groups")
	
	# Conditions and Rules
	trigger_conditions: Dict[str, Any] = Field(default_factory=dict, description="Workflow trigger conditions")
	escalation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Escalation rules")
	timeout_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Timeout handling rules")
	
	# Participants
	default_assignees: List[str] = Field(default_factory=list, description="Default step assignees")
	escalation_contacts: List[str] = Field(default_factory=list, description="Escalation contacts")
	notification_recipients: List[str] = Field(default_factory=list, description="Notification recipients")
	
	# Settings
	auto_start: bool = Field(default=False, description="Automatically start workflow")
	allow_parallel: bool = Field(default=False, description="Allow parallel execution")
	require_completion: bool = Field(default=True, description="Require all steps to complete")
	
	# SLA and Timing
	sla_hours: Optional[int] = Field(default=None, description="SLA in hours")
	estimated_duration: Optional[int] = Field(default=None, description="Estimated duration in hours")
	business_hours_only: bool = Field(default=True, description="Count only business hours")
	
	# Status and Analytics
	is_enabled: bool = Field(default=True, description="Workflow enabled status")
	usage_count: int = Field(default=0, description="Number of times used")
	success_rate: float = Field(default=0.0, description="Completion success rate")
	average_duration: Optional[float] = Field(default=None, description="Average completion time")


class DCMWorkflowInstance(DCMBase):
	"""Active workflow instance for a specific document"""
	workflow_id: str = Field(..., description="Workflow template ID")
	document_id: str = Field(..., description="Document ID")
	instance_name: str = Field(..., description="Workflow instance name")
	status: DCMWorkflowStatus = Field(default=DCMWorkflowStatus.NOT_STARTED, description="Workflow status")
	
	# Execution Details
	started_by: str = Field(..., description="User who started workflow")
	started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")
	completed_at: Optional[datetime] = Field(default=None, description="Workflow completion time")
	duration_hours: Optional[float] = Field(default=None, description="Total duration in hours")
	
	# Current State
	current_step_id: Optional[str] = Field(default=None, description="Current workflow step")
	current_assignees: List[str] = Field(default_factory=list, description="Current step assignees")
	next_step_id: Optional[str] = Field(default=None, description="Next workflow step")
	
	# Progress Tracking
	completed_steps: List[str] = Field(default_factory=list, description="Completed step IDs")
	pending_steps: List[str] = Field(default_factory=list, description="Pending step IDs")
	skipped_steps: List[str] = Field(default_factory=list, description="Skipped step IDs")
	
	# Decisions and Outcomes
	final_decision: Optional[str] = Field(default=None, description="Final workflow decision")
	decision_rationale: Optional[str] = Field(default=None, description="Decision explanation")
	outcome_data: Dict[str, Any] = Field(default_factory=dict, description="Workflow outcome data")
	
	# Escalation and Delays
	escalation_count: int = Field(default=0, description="Number of escalations")
	last_escalated_at: Optional[datetime] = Field(default=None, description="Last escalation time")
	is_overdue: bool = Field(default=False, description="Workflow overdue status")
	delay_reason: Optional[str] = Field(default=None, description="Delay explanation")
	
	# Variables and Context
	workflow_variables: Dict[str, Any] = Field(default_factory=dict, description="Workflow variables")
	context_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
	
	# Notifications
	notification_log: List[Dict[str, Any]] = Field(default_factory=list, description="Notification history")
	reminder_count: int = Field(default=0, description="Number of reminders sent")
	last_reminder_at: Optional[datetime] = Field(default=None, description="Last reminder time")


class DCMWorkflowStep(DCMBase):
	"""Individual workflow step execution"""
	workflow_instance_id: str = Field(..., description="Workflow instance ID")
	step_id: str = Field(..., description="Step identifier")
	step_name: str = Field(..., description="Step name")
	step_type: str = Field(..., description="Step type (review, approval, notification)")
	
	# Assignment and Execution
	assigned_to: List[str] = Field(..., description="Assigned user IDs")
	assigned_at: datetime = Field(default_factory=datetime.utcnow, description="Assignment timestamp")
	started_at: Optional[datetime] = Field(default=None, description="Step start time")
	completed_at: Optional[datetime] = Field(default=None, description="Step completion time")
	
	# Status and Decision
	status: DCMWorkflowStatus = Field(default=DCMWorkflowStatus.NOT_STARTED, description="Step status")
	decision: Optional[str] = Field(default=None, description="Step decision")
	decision_by: Optional[str] = Field(default=None, description="User who made decision")
	decision_at: Optional[datetime] = Field(default=None, description="Decision timestamp")
	
	# Comments and Feedback
	comments: List[str] = Field(default_factory=list, description="Step comment IDs")
	feedback: Optional[str] = Field(default=None, description="Step feedback")
	attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Step attachments")
	
	# Timing and SLA
	due_date: Optional[datetime] = Field(default=None, description="Step due date")
	sla_hours: Optional[int] = Field(default=None, description="Step SLA in hours")
	is_overdue: bool = Field(default=False, description="Step overdue status")
	duration_hours: Optional[float] = Field(default=None, description="Step duration in hours")
	
	# Escalation and Delegation
	escalated_to: Optional[str] = Field(default=None, description="Escalated to user ID")
	escalated_at: Optional[datetime] = Field(default=None, description="Escalation timestamp")
	delegated_to: Optional[str] = Field(default=None, description="Delegated to user ID")
	delegated_by: Optional[str] = Field(default=None, description="Delegating user ID")
	
	# Step Configuration
	step_config: Dict[str, Any] = Field(default_factory=dict, description="Step configuration")
	required_approvers: int = Field(default=1, description="Number of required approvers")
	approval_threshold: Optional[float] = Field(default=None, description="Approval threshold percentage")


# Knowledge Management Models
class DCMKnowledgeBase(DCMBase):
	"""Knowledge base and wiki organization"""
	name: str = Field(..., description="Knowledge base name")
	description: Optional[str] = Field(default=None, description="Knowledge base description")
	category: str = Field(..., description="Knowledge base category")
	is_public: bool = Field(default=False, description="Public access flag")
	
	# Organization
	parent_kb_id: Optional[str] = Field(default=None, description="Parent knowledge base ID")
	knowledge_areas: List[str] = Field(default_factory=list, description="Knowledge area tags")
	topics: List[str] = Field(default_factory=list, description="Topic classifications")
	
	# Content Management
	article_count: int = Field(default=0, description="Number of articles")
	page_count: int = Field(default=0, description="Number of wiki pages")
	contributor_count: int = Field(default=0, description="Number of contributors")
	
	# Quality and Curation
	quality_score: float = Field(default=0.0, description="Overall quality score")
	last_reviewed_at: Optional[datetime] = Field(default=None, description="Last review timestamp")
	next_review_date: Optional[date] = Field(default=None, description="Next review date")
	curator_ids: List[str] = Field(default_factory=list, description="Content curator user IDs")
	
	# Usage Analytics
	view_count: int = Field(default=0, description="Total view count")
	search_count: int = Field(default=0, description="Number of searches")
	contribution_count: int = Field(default=0, description="Number of contributions")
	
	# Settings
	allow_comments: bool = Field(default=True, description="Allow comments on articles")
	require_approval: bool = Field(default=False, description="Require approval for changes")
	enable_versioning: bool = Field(default=True, description="Enable version control")
	auto_categorize: bool = Field(default=True, description="Automatic categorization")


class DCMKnowledgeArticle(DCMBase):
	"""Knowledge base article or wiki page"""
	knowledge_base_id: str = Field(..., description="Parent knowledge base ID")
	title: str = Field(..., description="Article title")
	slug: str = Field(..., description="URL-friendly slug")
	content: str = Field(..., description="Article content (markdown/HTML)")
	summary: Optional[str] = Field(default=None, description="Article summary")
	
	# Organization
	category: str = Field(..., description="Article category")
	tags: List[str] = Field(default_factory=list, description="Article tags")
	keywords: List[str] = Field(default_factory=list, description="SEO keywords")
	topics: List[str] = Field(default_factory=list, description="Topic classifications")
	
	# Authoring
	author_id: str = Field(..., description="Primary author user ID")
	contributors: List[str] = Field(default_factory=list, description="Contributor user IDs")
	last_editor_id: Optional[str] = Field(default=None, description="Last editor user ID")
	
	# Status and Lifecycle
	status: DCMDocumentStatus = Field(default=DCMDocumentStatus.DRAFT, description="Article status")
	published_at: Optional[datetime] = Field(default=None, description="Publication timestamp")
	last_reviewed_at: Optional[datetime] = Field(default=None, description="Last review timestamp")
	next_review_date: Optional[date] = Field(default=None, description="Next review date")
	
	# Content Metrics
	word_count: int = Field(default=0, description="Word count")
	reading_time_minutes: int = Field(default=0, description="Estimated reading time")
	complexity_score: Optional[float] = Field(default=None, description="Content complexity score")
	
	# Quality and Feedback
	quality_score: float = Field(default=0.0, description="Article quality score")
	usefulness_score: float = Field(default=0.0, description="User-rated usefulness")
	feedback_count: int = Field(default=0, description="Number of feedback items")
	
	# Usage Analytics
	view_count: int = Field(default=0, description="Article view count")
	like_count: int = Field(default=0, description="Number of likes")
	share_count: int = Field(default=0, description="Number of shares")
	bookmark_count: int = Field(default=0, description="Number of bookmarks")
	
	# SEO and Discovery
	meta_description: Optional[str] = Field(default=None, description="Meta description")
	featured_image: Optional[str] = Field(default=None, description="Featured image URL")
	is_featured: bool = Field(default=False, description="Featured article flag")
	
	# Related Content
	related_articles: List[str] = Field(default_factory=list, description="Related article IDs")
	referenced_documents: List[str] = Field(default_factory=list, description="Referenced document IDs")
	external_links: List[str] = Field(default_factory=list, description="External reference URLs")
	
	# Collaboration
	allow_comments: bool = Field(default=True, description="Allow comments")
	comment_count: int = Field(default=0, description="Number of comments")
	is_collaborative: bool = Field(default=False, description="Allow collaborative editing")


# Digital Asset Management Models
class DCMAsset(DCMBase):
	"""Digital asset management for rich media content"""
	name: str = Field(..., description="Asset name")
	title: str = Field(..., description="Asset title")
	description: Optional[str] = Field(default=None, description="Asset description")
	asset_type: str = Field(..., description="Asset type (image, video, audio, 3d_model)")
	category: str = Field(..., description="Asset category")
	
	# File Information
	file_name: str = Field(..., description="Original file name")
	file_size: int = Field(..., description="File size in bytes")
	file_format: DCMContentFormat = Field(..., description="File format")
	mime_type: str = Field(..., description="MIME type")
	file_hash: str = Field(..., description="File content hash")
	storage_path: str = Field(..., description="Storage location path")
	
	# Media Metadata
	dimensions: Optional[Dict[str, int]] = Field(default=None, description="Asset dimensions (width, height)")
	duration_seconds: Optional[float] = Field(default=None, description="Duration for time-based media")
	resolution: Optional[str] = Field(default=None, description="Media resolution")
	color_profile: Optional[str] = Field(default=None, description="Color profile")
	bit_rate: Optional[int] = Field(default=None, description="Bit rate for audio/video")
	frame_rate: Optional[float] = Field(default=None, description="Frame rate for video")
	
	# Brand and Rights Management
	brand_guidelines: bool = Field(default=False, description="Follows brand guidelines")
	usage_rights: str = Field(default="internal", description="Usage rights classification")
	license_type: Optional[str] = Field(default=None, description="License type")
	copyright_holder: Optional[str] = Field(default=None, description="Copyright holder")
	expiry_date: Optional[date] = Field(default=None, description="Rights expiry date")
	
	# Asset Processing
	preview_path: Optional[str] = Field(default=None, description="Preview/thumbnail path")
	thumbnails: List[Dict[str, Any]] = Field(default_factory=list, description="Generated thumbnails")
	processed_versions: Dict[str, str] = Field(default_factory=dict, description="Processed version paths")
	ai_analysis: Optional[Dict[str, Any]] = Field(default=None, description="AI-generated analysis")
	
	# Usage and Distribution
	download_count: int = Field(default=0, description="Download count")
	usage_count: int = Field(default=0, description="Usage count in documents")
	last_used_at: Optional[datetime] = Field(default=None, description="Last usage timestamp")
	
	# Metadata and Tagging
	keywords: List[str] = Field(default_factory=list, description="Asset keywords")
	tags: List[str] = Field(default_factory=list, description="User-defined tags")
	ai_tags: List[str] = Field(default_factory=list, description="AI-generated tags")
	color_palette: List[str] = Field(default_factory=list, description="Dominant colors")
	
	# Organization
	collections: List[str] = Field(default_factory=list, description="Asset collection IDs")
	campaigns: List[str] = Field(default_factory=list, description="Associated campaign IDs")
	projects: List[str] = Field(default_factory=list, description="Associated project IDs")
	
	# Quality and Approval
	quality_score: float = Field(default=0.0, description="Asset quality score")
	is_approved: bool = Field(default=False, description="Approval status")
	approved_by: Optional[str] = Field(default=None, description="Approving user ID")
	approved_at: Optional[datetime] = Field(default=None, description="Approval timestamp")


class DCMAssetCollection(DCMBase):
	"""Collections for organizing digital assets"""
	name: str = Field(..., description="Collection name")
	description: Optional[str] = Field(default=None, description="Collection description")
	collection_type: str = Field(..., description="Collection type (brand, campaign, project)")
	is_public: bool = Field(default=False, description="Public collection flag")
	
	# Organization
	parent_collection_id: Optional[str] = Field(default=None, description="Parent collection ID")
	tags: List[str] = Field(default_factory=list, description="Collection tags")
	categories: List[str] = Field(default_factory=list, description="Collection categories")
	
	# Content
	asset_count: int = Field(default=0, description="Number of assets in collection")
	featured_asset_id: Optional[str] = Field(default=None, description="Featured asset ID")
	cover_image: Optional[str] = Field(default=None, description="Collection cover image")
	
	# Access and Sharing
	access_type: DCMAccessType = Field(default=DCMAccessType.INTERNAL, description="Access level")
	shared_with: List[str] = Field(default_factory=list, description="Shared with user/group IDs")
	download_enabled: bool = Field(default=True, description="Enable asset downloads")
	
	# Usage and Analytics
	view_count: int = Field(default=0, description="Collection view count")
	download_count: int = Field(default=0, description="Total downloads from collection")
	
	# Settings
	auto_organize: bool = Field(default=False, description="Automatic asset organization")
	sort_order: str = Field(default="created_desc", description="Default sort order")


# Audit and Compliance Models
class DCMAuditLog(DCMBase):
	"""Comprehensive audit logging for all document activities"""
	resource_id: str = Field(..., description="Resource ID (document, folder, etc.)")
	resource_type: str = Field(..., description="Resource type")
	action: str = Field(..., description="Action performed")
	action_category: str = Field(..., description="Action category (access, modify, admin)")
	
	# User and Session
	user_id: str = Field(..., description="User who performed action")
	session_id: Optional[str] = Field(default=None, description="User session ID")
	impersonated_user: Optional[str] = Field(default=None, description="Impersonated user ID")
	
	# Context Information
	ip_address: Optional[str] = Field(default=None, description="Client IP address")
	user_agent: Optional[str] = Field(default=None, description="Client user agent")
	device_info: Optional[Dict[str, Any]] = Field(default=None, description="Device information")
	location: Optional[Dict[str, Any]] = Field(default=None, description="Geographic location")
	
	# Action Details
	old_values: Optional[Dict[str, Any]] = Field(default=None, description="Previous values")
	new_values: Optional[Dict[str, Any]] = Field(default=None, description="New values")
	affected_fields: List[str] = Field(default_factory=list, description="Modified fields")
	
	# Result and Status
	success: bool = Field(..., description="Action success status")
	error_message: Optional[str] = Field(default=None, description="Error message if failed")
	warning_message: Optional[str] = Field(default=None, description="Warning message")
	
	# Risk and Compliance
	risk_level: str = Field(default="low", description="Risk level (low, medium, high)")
	compliance_tags: List[str] = Field(default_factory=list, description="Compliance requirement tags")
	data_classification: Optional[str] = Field(default=None, description="Data classification level")
	
	# Additional Context
	business_context: Optional[str] = Field(default=None, description="Business context")
	technical_details: Dict[str, Any] = Field(default_factory=dict, description="Technical details")
	
	# Retention and Archival
	retention_category: str = Field(default="standard", description="Retention category")
	archive_date: Optional[date] = Field(default=None, description="Archive date")
	purge_date: Optional[date] = Field(default=None, description="Purge date")


class DCMRetentionPolicy(DCMBase):
	"""Document retention and lifecycle management policies"""
	name: str = Field(..., description="Policy name")
	description: Optional[str] = Field(default=None, description="Policy description")
	policy_type: str = Field(..., description="Policy type (retention, deletion, archival)")
	is_enabled: bool = Field(default=True, description="Policy enabled status")
	
	# Scope and Application
	applies_to: List[str] = Field(..., description="Resource types this policy applies to")
	folder_scope: List[str] = Field(default_factory=list, description="Folder IDs in scope")
	document_types: List[DCMDocumentType] = Field(default_factory=list, description="Document types in scope")
	
	# Retention Rules
	retention_period_days: int = Field(..., description="Retention period in days")
	trigger_event: str = Field(..., description="Event that starts retention period")
	grace_period_days: int = Field(default=0, description="Grace period before enforcement")
	
	# Actions
	retention_action: str = Field(..., description="Action to take (archive, delete, notify)")
	notification_before_days: int = Field(default=30, description="Notification period before action")
	escalation_contacts: List[str] = Field(default_factory=list, description="Escalation contact user IDs")
	
	# Exceptions and Overrides
	legal_hold_override: bool = Field(default=True, description="Legal hold overrides policy")
	manual_override_allowed: bool = Field(default=False, description="Allow manual overrides")
	override_approval_required: bool = Field(default=True, description="Require approval for overrides")
	
	# Compliance
	regulatory_basis: List[str] = Field(default_factory=list, description="Regulatory requirements")
	compliance_tags: List[str] = Field(default_factory=list, description="Compliance tags")
	audit_frequency: str = Field(default="quarterly", description="Audit frequency")
	
	# Execution
	last_executed: Optional[datetime] = Field(default=None, description="Last execution timestamp")
	next_execution: Optional[datetime] = Field(default=None, description="Next execution timestamp")
	execution_status: str = Field(default="pending", description="Execution status")
	
	# Statistics
	documents_affected: int = Field(default=0, description="Number of documents affected")
	actions_taken: int = Field(default=0, description="Number of actions taken")
	exceptions_granted: int = Field(default=0, description="Number of exceptions granted")


class DCMNotification(DCMBase):
	"""Notification system for document events"""
	notification_type: DCMNotificationType = Field(..., description="Type of notification")
	recipient_id: str = Field(..., description="Recipient user ID")
	sender_id: Optional[str] = Field(default=None, description="Sender user ID (if applicable)")
	
	# Content
	title: str = Field(..., description="Notification title")
	message: str = Field(..., description="Notification message")
	rich_content: Optional[Dict[str, Any]] = Field(default=None, description="Rich content data")
	
	# Related Resources
	document_id: Optional[str] = Field(default=None, description="Related document ID")
	workflow_id: Optional[str] = Field(default=None, description="Related workflow ID")
	comment_id: Optional[str] = Field(default=None, description="Related comment ID")
	
	# Delivery
	delivery_channels: List[str] = Field(..., description="Delivery channels (email, web, mobile)")
	priority: str = Field(default="normal", description="Notification priority")
	
	# Status
	is_read: bool = Field(default=False, description="Read status")
	read_at: Optional[datetime] = Field(default=None, description="Read timestamp")
	delivered_at: Optional[datetime] = Field(default=None, description="Delivery timestamp")
	
	# Actions
	action_required: bool = Field(default=False, description="Action required flag")
	action_url: Optional[str] = Field(default=None, description="Action URL")
	due_date: Optional[datetime] = Field(default=None, description="Action due date")
	
	# Grouping and Threading
	group_key: Optional[str] = Field(default=None, description="Notification group key")
	thread_id: Optional[str] = Field(default=None, description="Thread identifier")
	
	# Preferences
	dismissible: bool = Field(default=True, description="Can be dismissed by user")
	auto_expire_hours: Optional[int] = Field(default=None, description="Auto-expiry in hours")


# Search and Analytics Models
class DCMSearchIndex(DCMBase):
	"""Search index entries for full-text search"""
	document_id: str = Field(..., description="Indexed document ID")
	content_hash: str = Field(..., description="Content hash for change detection")
	
	# Indexed Content
	extracted_text: str = Field(..., description="Extracted text content")
	metadata_text: str = Field(..., description="Searchable metadata")
	title: str = Field(..., description="Document title")
	tags: List[str] = Field(default_factory=list, description="Document tags")
	
	# Index Metadata
	language: str = Field(default="en", description="Content language")
	word_count: int = Field(default=0, description="Word count")
	index_version: str = Field(default="1.0", description="Index version")
	
	# Search Optimization
	search_keywords: List[str] = Field(default_factory=list, description="Optimized search keywords")
	stemmed_terms: List[str] = Field(default_factory=list, description="Stemmed terms")
	phonetic_codes: List[str] = Field(default_factory=list, description="Phonetic codes for fuzzy search")
	
	# Status
	index_status: str = Field(default="indexed", description="Index status")
	last_indexed: datetime = Field(default_factory=datetime.utcnow, description="Last index timestamp")
	next_reindex: Optional[datetime] = Field(default=None, description="Next reindex timestamp")


class DCMAnalytics(DCMBase):
	"""Analytics and usage statistics"""
	resource_id: str = Field(..., description="Resource ID (document, folder, user)")
	resource_type: str = Field(..., description="Resource type")
	metric_name: str = Field(..., description="Metric name")
	metric_value: float = Field(..., description="Metric value")
	
	# Time and Context
	measurement_date: date = Field(..., description="Measurement date")
	time_period: str = Field(..., description="Time period (daily, weekly, monthly)")
	context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
	
	# Aggregation
	aggregation_level: str = Field(..., description="Aggregation level (individual, team, organization)")
	sample_size: Optional[int] = Field(default=None, description="Sample size for calculated metrics")
	
	# Metadata
	calculation_method: Optional[str] = Field(default=None, description="How metric was calculated")
	data_source: str = Field(default="system", description="Data source")
	confidence_level: Optional[float] = Field(default=None, description="Confidence level for estimates")


# Validation Methods
@validator('file_size')
def validate_file_size(cls, v):
	if v < 0:
		raise ValueError("File size cannot be negative")
	return v


@validator('version_number')
def validate_version_number(cls, v):
	if not v or not isinstance(v, str):
		raise ValueError("Version number must be a non-empty string")
	return v


@root_validator
def validate_date_consistency(cls, values):
	created_at = values.get('created_at')
	updated_at = values.get('updated_at')
	
	if created_at and updated_at and updated_at < created_at:
		raise ValueError("Updated time cannot be before creation time")
	
	return values


@validator('keywords', 'tags', 'categories', pre=True)
def validate_string_lists(cls, v):
	if v is None:
		return []
	if isinstance(v, str):
		return [tag.strip() for tag in v.split(',') if tag.strip()]
	return v