"""
Document Management Models

Comprehensive database models for document management system including documents,
versions, folders, permissions, workflows, reviews, retention policies, and audit trails.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Date, Text, JSON, DECIMAL, ForeignKey, Index, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from flask_appbuilder import Model
from flask_appbuilder.models.mixins import AuditMixin

from uuid_extensions import uuid7str

Base = declarative_base()

class DocumentStatus(Enum):
	"""Document status enumeration."""
	DRAFT = "draft"
	REVIEW = "review" 
	APPROVED = "approved"
	PUBLISHED = "published"
	ARCHIVED = "archived"
	OBSOLETE = "obsolete"

class DocumentType(Enum):
	"""Document type enumeration."""
	POLICY = "policy"
	PROCEDURE = "procedure"
	CONTRACT = "contract"
	INVOICE = "invoice"
	REPORT = "report"
	SPECIFICATION = "specification"
	MANUAL = "manual"
	CERTIFICATE = "certificate"
	DRAWING = "drawing"
	CORRESPONDENCE = "correspondence"
	FORM = "form"
	TEMPLATE = "template"

class PermissionLevel(Enum):
	"""Permission level enumeration."""
	NONE = "none"
	READ = "read"
	WRITE = "write"
	DELETE = "delete"
	ADMIN = "admin"

class WorkflowStatus(Enum):
	"""Workflow status enumeration."""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	REJECTED = "rejected"

class ReviewStatus(Enum):
	"""Review status enumeration."""
	PENDING = "pending"
	IN_REVIEW = "in_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	CHANGES_REQUESTED = "changes_requested"

class GCDMDocumentCategory(Model, AuditMixin):
	"""Document categories for organization and classification."""
	
	__tablename__ = 'gc_dm_document_category'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Tenant isolation
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Category details
	name: str = Column(String(100), nullable=False)
	description: str = Column(Text)
	code: str = Column(String(20), nullable=False)
	color: str = Column(String(7), default="#007bff")  # Hex color code
	icon: str = Column(String(50), default="fa-folder")
	
	# Hierarchy
	parent_category_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_document_category.id'))
	level: int = Column(Integer, default=0, nullable=False)
	path: str = Column(String(500))  # Hierarchical path
	
	# Configuration
	auto_apply_retention: bool = Column(Boolean, default=False)
	default_retention_years: Optional[int] = Column(Integer)
	requires_approval: bool = Column(Boolean, default=False)
	security_classification: str = Column(String(20), default="internal")
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	sort_order: int = Column(Integer, default=0)
	
	# Relationships
	parent_category = relationship("GCDMDocumentCategory", remote_side=[id])
	documents = relationship("GCDMDocument", back_populates="category")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_category_tenant_code', 'tenant_id', 'code'),
		Index('idx_gc_dm_category_parent', 'parent_category_id'),
		Index('idx_gc_dm_category_path', 'path'),
		UniqueConstraint('tenant_id', 'code', name='uq_gc_dm_category_tenant_code'),
	)

class GCDMDocumentType(Model, AuditMixin):
	"""Document types for classification and behavior."""
	
	__tablename__ = 'gc_dm_document_type'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Tenant isolation
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Type details
	name: str = Column(String(100), nullable=False)
	description: str = Column(Text)
	code: str = Column(String(20), nullable=False)
	
	# File restrictions
	allowed_extensions: str = Column(Text)  # JSON array of allowed file extensions
	max_file_size_mb: int = Column(Integer, default=100)
	requires_version_control: bool = Column(Boolean, default=True)
	
	# Workflow configuration
	requires_approval: bool = Column(Boolean, default=False)
	default_workflow_id: Optional[str] = Column(String(50))
	auto_archive_days: Optional[int] = Column(Integer)
	
	# Retention
	default_retention_years: int = Column(Integer, default=7)
	retention_policy_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_retention_policy.id'))
	
	# Security
	security_classification: str = Column(String(20), default="internal")
	requires_encryption: bool = Column(Boolean, default=False)
	watermark_enabled: bool = Column(Boolean, default=False)
	
	# Templates
	has_template: bool = Column(Boolean, default=False)
	template_file_path: Optional[str] = Column(String(500))
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	
	# Relationships
	documents = relationship("GCDMDocument", back_populates="document_type")
	retention_policy = relationship("GCDMRetentionPolicy")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_doctype_tenant_code', 'tenant_id', 'code'),
		UniqueConstraint('tenant_id', 'code', name='uq_gc_dm_doctype_tenant_code'),
	)

class GCDMFolder(Model, AuditMixin):
	"""Folder structure for document organization."""
	
	__tablename__ = 'gc_dm_folder'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Tenant isolation
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Folder details
	name: str = Column(String(200), nullable=False)
	description: str = Column(Text)
	folder_path: str = Column(String(1000), nullable=False)
	
	# Hierarchy
	parent_folder_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_folder.id'))
	level: int = Column(Integer, default=0, nullable=False)
	full_path: str = Column(String(1000), nullable=False)
	
	# Configuration
	is_system_folder: bool = Column(Boolean, default=False)
	auto_create_subfolders: bool = Column(Boolean, default=False)
	subfolder_template: Optional[str] = Column(Text)  # JSON template
	
	# Access control
	inherit_permissions: bool = Column(Boolean, default=True)
	public_read_access: bool = Column(Boolean, default=False)
	
	# Constraints
	max_documents: Optional[int] = Column(Integer)
	max_size_gb: Optional[Decimal] = Column(DECIMAL(10, 2))
	allowed_document_types: Optional[str] = Column(Text)  # JSON array
	
	# Statistics
	document_count: int = Column(Integer, default=0)
	total_size_mb: Decimal = Column(DECIMAL(15, 2), default=0)
	last_activity_date: Optional[datetime] = Column(DateTime)
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	is_locked: bool = Column(Boolean, default=False)
	
	# Relationships
	parent_folder = relationship("GCDMFolder", remote_side=[id])
	documents = relationship("GCDMDocument", back_populates="folder")
	permissions = relationship("GCDMPermission", back_populates="folder")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_folder_tenant_path', 'tenant_id', 'folder_path'),
		Index('idx_gc_dm_folder_parent', 'parent_folder_id'),
		Index('idx_gc_dm_folder_full_path', 'full_path'),
		UniqueConstraint('tenant_id', 'folder_path', name='uq_gc_dm_folder_tenant_path'),
	)

class GCDMDocument(Model, AuditMixin):
	"""Main document entity."""
	
	__tablename__ = 'gc_dm_document'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Tenant isolation
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Document identification
	document_number: str = Column(String(100), nullable=False)
	title: str = Column(String(500), nullable=False)
	description: str = Column(Text)
	
	# Classification
	category_id: str = Column(String(50), ForeignKey('gc_dm_document_category.id'), nullable=False)
	document_type_id: str = Column(String(50), ForeignKey('gc_dm_document_type.id'), nullable=False)
	folder_id: str = Column(String(50), ForeignKey('gc_dm_folder.id'), nullable=False)
	
	# File information
	file_name: str = Column(String(255), nullable=False)
	file_path: str = Column(String(1000), nullable=False)
	file_size_bytes: int = Column(Integer, default=0)
	file_extension: str = Column(String(10))
	mime_type: str = Column(String(100))
	file_hash: str = Column(String(64))  # SHA-256 hash
	
	# Content
	content_text: Optional[str] = Column(Text)  # Extracted text for search
	keywords: Optional[str] = Column(Text)  # Comma-separated keywords
	language: str = Column(String(10), default="en")
	
	# Version control
	version_number: str = Column(String(20), default="1.0")
	is_latest_version: bool = Column(Boolean, default=True, nullable=False)
	parent_document_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_document.id'))
	
	# Status and workflow
	status: str = Column(String(20), default=DocumentStatus.DRAFT.value, nullable=False)
	workflow_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_workflow.id'))
	
	# Ownership
	owner_user_id: str = Column(String(50), nullable=False)
	author: str = Column(String(200))
	department: Optional[str] = Column(String(100))
	business_unit: Optional[str] = Column(String(100))
	
	# Dates
	document_date: date = Column(Date, default=date.today, nullable=False)
	effective_date: Optional[date] = Column(Date)
	expiry_date: Optional[date] = Column(Date)
	review_date: Optional[date] = Column(Date)
	
	# Security
	security_classification: str = Column(String(20), default="internal")
	is_confidential: bool = Column(Boolean, default=False)
	is_encrypted: bool = Column(Boolean, default=False)
	encryption_key_id: Optional[str] = Column(String(100))
	
	# Retention
	retention_policy_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_retention_policy.id'))
	retention_date: Optional[date] = Column(Date)
	legal_hold: bool = Column(Boolean, default=False)
	
	# Access control
	is_public: bool = Column(Boolean, default=False)
	requires_checkout: bool = Column(Boolean, default=False)
	is_checked_out: bool = Column(Boolean, default=False)
	checked_out_by: Optional[str] = Column(String(50))
	checked_out_date: Optional[datetime] = Column(DateTime)
	
	# Digital signatures
	is_signed: bool = Column(Boolean, default=False)
	signature_required: bool = Column(Boolean, default=False)
	signatures_count: int = Column(Integer, default=0)
	
	# Statistics
	view_count: int = Column(Integer, default=0)
	download_count: int = Column(Integer, default=0)
	last_viewed_date: Optional[datetime] = Column(DateTime)
	
	# External references
	external_id: Optional[str] = Column(String(100))
	source_system: Optional[str] = Column(String(50))
	related_entity_type: Optional[str] = Column(String(50))
	related_entity_id: Optional[str] = Column(String(50))
	
	# Custom fields
	custom_fields: Optional[str] = Column(Text)  # JSON data
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	is_deleted: bool = Column(Boolean, default=False)
	deleted_date: Optional[datetime] = Column(DateTime)
	
	# Relationships
	category = relationship("GCDMDocumentCategory", back_populates="documents")
	document_type = relationship("GCDMDocumentType", back_populates="documents")
	folder = relationship("GCDMFolder", back_populates="documents")
	workflow = relationship("GCDMWorkflow", back_populates="documents")
	retention_policy = relationship("GCDMRetentionPolicy")
	parent_document = relationship("GCDMDocument", remote_side=[id])
	
	versions = relationship("GCDMDocumentVersion", back_populates="document")
	permissions = relationship("GCDMPermission", back_populates="document")
	checkouts = relationship("GCDMCheckout", back_populates="document")
	reviews = relationship("GCDMReview", back_populates="document")
	tags = relationship("GCDMTag", secondary="gc_dm_document_tag", back_populates="documents")
	metadata_entries = relationship("GCDMMetadata", back_populates="document")
	audit_logs = relationship("GCDMAuditLog", back_populates="document")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_document_tenant_number', 'tenant_id', 'document_number'),
		Index('idx_gc_dm_document_category', 'category_id'),
		Index('idx_gc_dm_document_type', 'document_type_id'),
		Index('idx_gc_dm_document_folder', 'folder_id'),
		Index('idx_gc_dm_document_status', 'status'),
		Index('idx_gc_dm_document_owner', 'owner_user_id'),
		Index('idx_gc_dm_document_dates', 'document_date', 'effective_date'),
		Index('idx_gc_dm_document_checkout', 'is_checked_out', 'checked_out_by'),
		Index('idx_gc_dm_document_search', 'title', 'keywords'),
		UniqueConstraint('tenant_id', 'document_number', name='uq_gc_dm_document_tenant_number'),
		CheckConstraint('file_size_bytes >= 0', name='ck_gc_dm_document_file_size'),
		CheckConstraint('view_count >= 0', name='ck_gc_dm_document_view_count'),
		CheckConstraint('download_count >= 0', name='ck_gc_dm_document_download_count'),
	)

class GCDMDocumentVersion(Model, AuditMixin):
	"""Document version history."""
	
	__tablename__ = 'gc_dm_document_version'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Document reference
	document_id: str = Column(String(50), ForeignKey('gc_dm_document.id'), nullable=False)
	
	# Version information
	version_number: str = Column(String(20), nullable=False)
	version_label: Optional[str] = Column(String(100))
	is_current: bool = Column(Boolean, default=False)
	is_published: bool = Column(Boolean, default=False)
	
	# File information
	file_name: str = Column(String(255), nullable=False)
	file_path: str = Column(String(1000), nullable=False)
	file_size_bytes: int = Column(Integer, default=0)
	file_hash: str = Column(String(64))
	
	# Change information
	change_description: str = Column(Text, nullable=False)
	change_type: str = Column(String(20), default="minor")  # major, minor, patch
	changed_by: str = Column(String(50), nullable=False)
	change_reason: Optional[str] = Column(Text)
	
	# Status
	status: str = Column(String(20), default="draft")
	
	# Approval
	approved_by: Optional[str] = Column(String(50))
	approved_date: Optional[datetime] = Column(DateTime)
	approval_comments: Optional[str] = Column(Text)
	
	# Relationships
	document = relationship("GCDMDocument", back_populates="versions")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_version_document', 'document_id'),
		Index('idx_gc_dm_version_number', 'document_id', 'version_number'),
		Index('idx_gc_dm_version_current', 'document_id', 'is_current'),
		UniqueConstraint('document_id', 'version_number', name='uq_gc_dm_version_document_number'),
	)

class GCDMPermission(Model, AuditMixin):
	"""Document and folder permissions."""
	
	__tablename__ = 'gc_dm_permission'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Tenant isolation
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Resource reference (document or folder)
	document_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_document.id'))
	folder_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_folder.id'))
	
	# Permission subject (user, role, or group)
	subject_type: str = Column(String(20), nullable=False)  # user, role, group
	subject_id: str = Column(String(50), nullable=False)
	
	# Permission level
	permission_level: str = Column(String(20), default=PermissionLevel.READ.value, nullable=False)
	
	# Specific permissions
	can_read: bool = Column(Boolean, default=True)
	can_write: bool = Column(Boolean, default=False)
	can_delete: bool = Column(Boolean, default=False)
	can_share: bool = Column(Boolean, default=False)
	can_approve: bool = Column(Boolean, default=False)
	can_administer: bool = Column(Boolean, default=False)
	
	# Permission scope
	applies_to_children: bool = Column(Boolean, default=True)
	inherited: bool = Column(Boolean, default=False)
	inherited_from_id: Optional[str] = Column(String(50))
	
	# Time restrictions
	effective_date: Optional[date] = Column(Date)
	expiry_date: Optional[date] = Column(Date)
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	
	# Granted by
	granted_by: str = Column(String(50), nullable=False)
	grant_reason: Optional[str] = Column(Text)
	
	# Relationships
	document = relationship("GCDMDocument", back_populates="permissions")
	folder = relationship("GCDMFolder", back_populates="permissions")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_permission_document', 'document_id'),
		Index('idx_gc_dm_permission_folder', 'folder_id'),
		Index('idx_gc_dm_permission_subject', 'subject_type', 'subject_id'),
		Index('idx_gc_dm_permission_level', 'permission_level'),
		CheckConstraint('(document_id IS NOT NULL) OR (folder_id IS NOT NULL)', name='ck_gc_dm_permission_resource'),
		CheckConstraint('(document_id IS NULL) OR (folder_id IS NULL)', name='ck_gc_dm_permission_single_resource'),
	)

class GCDMCheckout(Model, AuditMixin):
	"""Document checkout tracking."""
	
	__tablename__ = 'gc_dm_checkout'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Document reference
	document_id: str = Column(String(50), ForeignKey('gc_dm_document.id'), nullable=False)
	
	# Checkout information
	checked_out_by: str = Column(String(50), nullable=False)
	checkout_date: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
	expected_return_date: Optional[datetime] = Column(DateTime)
	
	# Checkout details
	checkout_reason: Optional[str] = Column(Text)
	checkout_type: str = Column(String(20), default="exclusive")  # exclusive, shared
	
	# Return information
	returned_date: Optional[datetime] = Column(DateTime)
	return_comments: Optional[str] = Column(Text)
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	is_overdue: bool = Column(Boolean, default=False)
	
	# Relationships
	document = relationship("GCDMDocument", back_populates="checkouts")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_checkout_document', 'document_id'),
		Index('idx_gc_dm_checkout_user', 'checked_out_by'),
		Index('idx_gc_dm_checkout_active', 'is_active'),
		Index('idx_gc_dm_checkout_overdue', 'is_overdue'),
	)

class GCDMWorkflow(Model, AuditMixin):
	"""Document workflow definitions and instances."""
	
	__tablename__ = 'gc_dm_workflow'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Tenant isolation
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Workflow definition
	name: str = Column(String(200), nullable=False)
	description: str = Column(Text)
	workflow_type: str = Column(String(50), default="approval")  # approval, review, routing
	
	# Workflow configuration
	definition: str = Column(Text, nullable=False)  # JSON workflow definition
	is_template: bool = Column(Boolean, default=False)
	template_id: Optional[str] = Column(String(50))
	
	# Instance information (if this is a workflow instance)
	document_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_document.id'))
	initiated_by: Optional[str] = Column(String(50))
	initiated_date: Optional[datetime] = Column(DateTime)
	
	# Status
	status: str = Column(String(20), default=WorkflowStatus.PENDING.value, nullable=False)
	current_step: Optional[str] = Column(String(100))
	current_assignee: Optional[str] = Column(String(50))
	
	# Completion
	completed_date: Optional[datetime] = Column(DateTime)
	completion_result: Optional[str] = Column(String(20))  # approved, rejected, cancelled
	completion_comments: Optional[str] = Column(Text)
	
	# Time tracking
	due_date: Optional[datetime] = Column(DateTime)
	estimated_duration_hours: Optional[int] = Column(Integer)
	actual_duration_hours: Optional[int] = Column(Integer)
	
	# Priority and escalation
	priority: str = Column(String(20), default="medium")
	escalation_rules: Optional[str] = Column(Text)  # JSON escalation configuration
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	
	# Relationships
	documents = relationship("GCDMDocument", back_populates="workflow")
	reviews = relationship("GCDMReview", back_populates="workflow")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_workflow_tenant', 'tenant_id'),
		Index('idx_gc_dm_workflow_document', 'document_id'),
		Index('idx_gc_dm_workflow_status', 'status'),
		Index('idx_gc_dm_workflow_assignee', 'current_assignee'),
		Index('idx_gc_dm_workflow_due', 'due_date'),
	)

class GCDMReview(Model, AuditMixin):
	"""Document review tracking."""
	
	__tablename__ = 'gc_dm_review'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Document and workflow reference
	document_id: str = Column(String(50), ForeignKey('gc_dm_document.id'), nullable=False)
	workflow_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_workflow.id'))
	
	# Review information
	review_type: str = Column(String(50), default="approval")  # approval, peer_review, quality_review
	reviewer_user_id: str = Column(String(50), nullable=False)
	reviewer_name: str = Column(String(200))
	reviewer_role: Optional[str] = Column(String(100))
	
	# Review details
	assigned_date: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
	due_date: Optional[datetime] = Column(DateTime)
	started_date: Optional[datetime] = Column(DateTime)
	completed_date: Optional[datetime] = Column(DateTime)
	
	# Review outcome
	status: str = Column(String(20), default=ReviewStatus.PENDING.value, nullable=False)
	decision: Optional[str] = Column(String(20))  # approved, rejected, changes_requested
	comments: Optional[str] = Column(Text)
	detailed_feedback: Optional[str] = Column(Text)
	
	# Review criteria
	review_criteria: Optional[str] = Column(Text)  # JSON criteria checklist
	criteria_scores: Optional[str] = Column(Text)  # JSON scores
	overall_score: Optional[Decimal] = Column(DECIMAL(5, 2))
	
	# Escalation
	is_escalated: bool = Column(Boolean, default=False)
	escalated_to: Optional[str] = Column(String(50))
	escalation_reason: Optional[str] = Column(Text)
	
	# Delegation
	delegated_to: Optional[str] = Column(String(50))
	delegation_reason: Optional[str] = Column(Text)
	
	# Relationships
	document = relationship("GCDMDocument", back_populates="reviews")
	workflow = relationship("GCDMWorkflow", back_populates="reviews")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_review_document', 'document_id'),
		Index('idx_gc_dm_review_workflow', 'workflow_id'),
		Index('idx_gc_dm_review_reviewer', 'reviewer_user_id'),
		Index('idx_gc_dm_review_status', 'status'),
		Index('idx_gc_dm_review_due', 'due_date'),
	)

class GCDMTag(Model, AuditMixin):
	"""Tags for document classification and search."""
	
	__tablename__ = 'gc_dm_tag'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Tenant isolation
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Tag details
	name: str = Column(String(100), nullable=False)
	description: Optional[str] = Column(Text)
	color: str = Column(String(7), default="#6c757d")  # Hex color code
	
	# Tag classification
	category: Optional[str] = Column(String(50))
	is_system_tag: bool = Column(Boolean, default=False)
	auto_apply_rules: Optional[str] = Column(Text)  # JSON rules for auto-tagging
	
	# Usage statistics
	usage_count: int = Column(Integer, default=0)
	last_used_date: Optional[datetime] = Column(DateTime)
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	
	# Relationships
	documents = relationship("GCDMDocument", secondary="gc_dm_document_tag", back_populates="tags")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_tag_tenant_name', 'tenant_id', 'name'),
		Index('idx_gc_dm_tag_category', 'category'),
		UniqueConstraint('tenant_id', 'name', name='uq_gc_dm_tag_tenant_name'),
	)

# Association table for document-tag many-to-many relationship
class GCDMDocumentTag(Model):
	"""Association table for document-tag relationships."""
	
	__tablename__ = 'gc_dm_document_tag'
	
	document_id: str = Column(String(50), ForeignKey('gc_dm_document.id'), primary_key=True)
	tag_id: str = Column(String(50), ForeignKey('gc_dm_tag.id'), primary_key=True)
	
	# Additional attributes
	applied_by: str = Column(String(50), nullable=False)
	applied_date: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
	is_auto_applied: bool = Column(Boolean, default=False)

class GCDMMetadata(Model, AuditMixin):
	"""Custom metadata for documents."""
	
	__tablename__ = 'gc_dm_metadata'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Document reference
	document_id: str = Column(String(50), ForeignKey('gc_dm_document.id'), nullable=False)
	
	# Metadata details
	metadata_key: str = Column(String(100), nullable=False)
	metadata_value: Optional[str] = Column(Text)
	data_type: str = Column(String(20), default="string")  # string, number, date, boolean, json
	
	# Metadata classification
	category: Optional[str] = Column(String(50))
	is_system_metadata: bool = Column(Boolean, default=False)
	is_searchable: bool = Column(Boolean, default=True)
	
	# Relationships
	document = relationship("GCDMDocument", back_populates="metadata_entries")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_metadata_document', 'document_id'),
		Index('idx_gc_dm_metadata_key', 'metadata_key'),
		Index('idx_gc_dm_metadata_category', 'category'),
		UniqueConstraint('document_id', 'metadata_key', name='uq_gc_dm_metadata_document_key'),
	)

class GCDMRetentionPolicy(Model, AuditMixin):
	"""Document retention policies."""
	
	__tablename__ = 'gc_dm_retention_policy'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Tenant isolation
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Policy details
	name: str = Column(String(200), nullable=False)
	description: str = Column(Text)
	policy_code: str = Column(String(50), nullable=False)
	
	# Retention rules
	retention_period_years: int = Column(Integer, nullable=False)
	retention_period_months: int = Column(Integer, default=0)
	retention_period_days: int = Column(Integer, default=0)
	
	# Policy triggers
	trigger_event: str = Column(String(50), default="creation_date")  # creation_date, effective_date, last_modified
	trigger_offset_days: int = Column(Integer, default=0)
	
	# Actions
	auto_delete_enabled: bool = Column(Boolean, default=False)
	auto_archive_enabled: bool = Column(Boolean, default=True)
	require_approval_for_deletion: bool = Column(Boolean, default=True)
	
	# Legal hold
	legal_hold_override: bool = Column(Boolean, default=True)
	litigation_hold_exemption: bool = Column(Boolean, default=False)
	
	# Notifications
	notify_before_deletion_days: int = Column(Integer, default=30)
	notification_recipients: Optional[str] = Column(Text)  # JSON array of emails/users
	
	# Compliance
	regulatory_basis: Optional[str] = Column(Text)
	compliance_framework: Optional[str] = Column(String(100))
	citation_reference: Optional[str] = Column(String(200))
	
	# Application rules
	applies_to_document_types: Optional[str] = Column(Text)  # JSON array
	applies_to_categories: Optional[str] = Column(Text)  # JSON array
	exclusion_rules: Optional[str] = Column(Text)  # JSON rules
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	effective_date: date = Column(Date, default=date.today, nullable=False)
	expiry_date: Optional[date] = Column(Date)
	
	# Relationships
	documents = relationship("GCDMDocument", back_populates="retention_policy")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_retention_tenant_code', 'tenant_id', 'policy_code'),
		Index('idx_gc_dm_retention_active', 'is_active'),
		Index('idx_gc_dm_retention_dates', 'effective_date', 'expiry_date'),
		UniqueConstraint('tenant_id', 'policy_code', name='uq_gc_dm_retention_tenant_code'),
	)

class GCDMArchive(Model, AuditMixin):
	"""Archived documents tracking."""
	
	__tablename__ = 'gc_dm_archive'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Document reference
	original_document_id: str = Column(String(50), nullable=False)
	document_number: str = Column(String(100), nullable=False)
	
	# Archive information
	archive_date: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
	archived_by: str = Column(String(50), nullable=False)
	archive_reason: str = Column(String(100), nullable=False)
	archive_location: str = Column(String(500), nullable=False)
	
	# Original document metadata (snapshot)
	original_metadata: str = Column(Text, nullable=False)  # JSON snapshot
	
	# Archive storage
	storage_type: str = Column(String(50), default="file_system")  # file_system, cloud, tape
	storage_path: str = Column(String(1000), nullable=False)
	storage_size_bytes: int = Column(Integer, default=0)
	storage_hash: str = Column(String(64))
	
	# Retrieval information
	retrieval_instructions: Optional[str] = Column(Text)
	estimated_retrieval_time_hours: Optional[int] = Column(Integer)
	retrieval_cost: Optional[Decimal] = Column(DECIMAL(10, 2))
	
	# Retention
	scheduled_destruction_date: Optional[date] = Column(Date)
	destruction_method: Optional[str] = Column(String(50))
	
	# Status
	is_retrievable: bool = Column(Boolean, default=True, nullable=False)
	is_destroyed: bool = Column(Boolean, default=False)
	destruction_date: Optional[datetime] = Column(DateTime)
	destruction_certificate: Optional[str] = Column(String(200))
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_archive_original_doc', 'original_document_id'),
		Index('idx_gc_dm_archive_number', 'document_number'),
		Index('idx_gc_dm_archive_date', 'archive_date'),
		Index('idx_gc_dm_archive_destruction', 'scheduled_destruction_date'),
		Index('idx_gc_dm_archive_retrievable', 'is_retrievable'),
	)

class GCDMAuditLog(Model, AuditMixin):
	"""Comprehensive audit log for document activities."""
	
	__tablename__ = 'gc_dm_audit_log'
	
	# Primary key
	id: str = Column(String(50), primary_key=True, default=uuid7str)
	
	# Tenant isolation
	tenant_id: str = Column(String(50), nullable=False, index=True)
	
	# Document reference
	document_id: Optional[str] = Column(String(50), ForeignKey('gc_dm_document.id'))
	document_number: Optional[str] = Column(String(100))
	
	# Activity details
	activity_type: str = Column(String(50), nullable=False)  # view, download, edit, delete, etc.
	activity_description: str = Column(Text, nullable=False)
	activity_category: str = Column(String(50), default="document_access")
	
	# User information
	user_id: str = Column(String(50), nullable=False)
	user_name: str = Column(String(200))
	user_ip_address: Optional[str] = Column(String(45))  # IPv6 compatible
	user_agent: Optional[str] = Column(Text)
	
	# Session information
	session_id: Optional[str] = Column(String(100))
	request_id: Optional[str] = Column(String(100))
	
	# Activity context
	before_values: Optional[str] = Column(Text)  # JSON before state
	after_values: Optional[str] = Column(Text)  # JSON after state
	affected_fields: Optional[str] = Column(Text)  # JSON array of changed fields
	
	# Technical details
	operation_result: str = Column(String(20), default="success")  # success, failure, warning
	error_message: Optional[str] = Column(Text)
	processing_time_ms: Optional[int] = Column(Integer)
	
	# Risk assessment
	risk_level: str = Column(String(20), default="low")  # low, medium, high, critical
	compliance_flag: bool = Column(Boolean, default=False)
	security_event: bool = Column(Boolean, default=False)
	
	# Additional context
	workflow_id: Optional[str] = Column(String(50))
	batch_id: Optional[str] = Column(String(50))
	external_reference: Optional[str] = Column(String(200))
	custom_data: Optional[str] = Column(Text)  # JSON additional data
	
	# Relationships
	document = relationship("GCDMDocument", back_populates="audit_logs")
	
	# Indexes
	__table_args__ = (
		Index('idx_gc_dm_audit_tenant_date', 'tenant_id', 'created_on'),
		Index('idx_gc_dm_audit_document', 'document_id'),
		Index('idx_gc_dm_audit_user', 'user_id'),
		Index('idx_gc_dm_audit_activity', 'activity_type'),
		Index('idx_gc_dm_audit_risk', 'risk_level'),
		Index('idx_gc_dm_audit_security', 'security_event'),
		Index('idx_gc_dm_audit_workflow', 'workflow_id'),
	)