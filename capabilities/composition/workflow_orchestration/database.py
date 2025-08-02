"""
APG Workflow Orchestration Database Models

Complete SQLAlchemy models for workflow orchestration with real database
implementation, no mocking or placeholders.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, AsyncGenerator
from uuid_extensions import uuid7str

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, 
    JSON, ForeignKey, Index, CheckConstraint, UniqueConstraint,
    Table, MetaData, create_engine, event, func, select, update, delete, and_, or_
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine, AsyncEngine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, selectinload, joinedload
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.pool import NullPool
import structlog

logger = structlog.get_logger(__name__)

# SQLAlchemy Base
Base = declarative_base()
metadata = MetaData()

# =============================================================================
# Enterprise Integration Tables
# =============================================================================

class WOAuditEvent(Base):
	"""Audit events table for enterprise compliance"""
	__tablename__ = 'wo_audit_events'
	__table_args__ = (
		Index('idx_wo_audit_events_timestamp', 'timestamp'),
		Index('idx_wo_audit_events_user_id', 'user_id'),
		Index('idx_wo_audit_events_event_type', 'event_type'),
		Index('idx_wo_audit_events_risk_level', 'risk_level'),
		Index('idx_wo_audit_events_tenant_id', 'tenant_id'),
	)
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	event_type = Column(String(100), nullable=False)
	user_id = Column(String(36), nullable=True)
	session_id = Column(String(36), nullable=True)
	source_ip = Column(String(45), nullable=True)
	user_agent = Column(Text, nullable=True)
	resource_type = Column(String(100), nullable=True)
	resource_id = Column(String(36), nullable=True)
	action = Column(String(100), nullable=False)
	result = Column(String(50), nullable=False)
	details = Column(JSONB, nullable=True, default={})
	risk_level = Column(String(20), nullable=False, default='low')
	compliance_tags = Column(ARRAY(String), nullable=True, default=[])
	tenant_id = Column(String(36), nullable=True)


class WOSecurityPolicy(Base):
	"""Security policies table"""
	__tablename__ = 'wo_security_policies'
	__table_args__ = (
		Index('idx_wo_security_policies_enabled', 'enabled'),
		Index('idx_wo_security_policies_policy_type', 'policy_type'),
	)
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	policy_type = Column(String(50), nullable=False)
	rules = Column(JSONB, nullable=False, default=[])
	enabled = Column(Boolean, nullable=False, default=True)
	enforcement_level = Column(String(20), nullable=False, default='strict')
	applicable_roles = Column(ARRAY(String), nullable=True, default=[])
	applicable_resources = Column(ARRAY(String), nullable=True, default=[])
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class WOSSOSession(Base):
	"""SSO sessions table"""
	__tablename__ = 'wo_sso_sessions'
	__table_args__ = (
		Index('idx_wo_sso_sessions_user_id', 'user_id'),
		Index('idx_wo_sso_sessions_provider', 'provider'),
		Index('idx_wo_sso_sessions_active', 'is_active'),
		Index('idx_wo_sso_sessions_expires', 'token_expires_at'),
	)
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	user_id = Column(String(36), nullable=False)
	provider = Column(String(50), nullable=False)
	provider_user_id = Column(String(100), nullable=False)
	email = Column(String(255), nullable=False)
	display_name = Column(String(255), nullable=False)
	roles = Column(ARRAY(String), nullable=True, default=[])
	groups = Column(ARRAY(String), nullable=True, default=[])
	claims = Column(JSONB, nullable=True, default={})
	access_token = Column(Text, nullable=True)
	refresh_token = Column(Text, nullable=True)
	id_token = Column(Text, nullable=True)
	token_expires_at = Column(DateTime(timezone=True), nullable=True)
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	last_activity = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	is_active = Column(Boolean, nullable=False, default=True)


class WOCompliancePolicy(Base):
	"""Compliance policies table"""
	__tablename__ = 'wo_compliance_policies'
	__table_args__ = (
		Index('idx_wo_compliance_policies_framework', 'framework'),
		Index('idx_wo_compliance_policies_status', 'status'),
		Index('idx_wo_compliance_policies_review_date', 'review_date'),
		Index('idx_wo_compliance_policies_tenant_id', 'tenant_id'),
	)
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	framework = Column(String(50), nullable=False)
	version = Column(String(20), nullable=False, default='1.0')
	status = Column(String(20), nullable=False, default='draft')
	effective_date = Column(DateTime(timezone=True), nullable=False)
	review_date = Column(DateTime(timezone=True), nullable=False)
	expiry_date = Column(DateTime(timezone=True), nullable=True)
	owner = Column(String(100), nullable=False)
	approver = Column(String(100), nullable=True)
	scope = Column(ARRAY(String), nullable=True, default=[])
	requirements = Column(ARRAY(String), nullable=True, default=[])
	controls = Column(JSONB, nullable=True, default=[])
	exceptions = Column(JSONB, nullable=True, default=[])
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	tenant_id = Column(String(36), nullable=True)


class WORiskAssessment(Base):
	"""Risk assessments table"""
	__tablename__ = 'wo_risk_assessments'
	__table_args__ = (
		Index('idx_wo_risk_assessments_asset_type', 'asset_type'),
		Index('idx_wo_risk_assessments_residual_risk', 'residual_risk'),
		Index('idx_wo_risk_assessments_status', 'status'),
		Index('idx_wo_risk_assessments_review_date', 'next_review_date'),
		Index('idx_wo_risk_assessments_tenant_id', 'tenant_id'),
	)
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	asset_type = Column(String(50), nullable=False)
	asset_id = Column(String(36), nullable=False)
	risk_category = Column(String(50), nullable=False)
	threat_sources = Column(ARRAY(String), nullable=True, default=[])
	vulnerabilities = Column(ARRAY(String), nullable=True, default=[])
	likelihood = Column(String(20), nullable=False)
	impact = Column(String(20), nullable=False)
	inherent_risk = Column(String(20), nullable=False)
	residual_risk = Column(String(20), nullable=False)
	risk_tolerance = Column(String(20), nullable=False)
	mitigation_controls = Column(ARRAY(String), nullable=True, default=[])
	treatment_plan = Column(Text, nullable=True)
	owner = Column(String(100), nullable=False)
	assessor = Column(String(100), nullable=False)
	assessment_date = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	next_review_date = Column(DateTime(timezone=True), nullable=False)
	status = Column(String(20), nullable=False, default='open')
	tenant_id = Column(String(36), nullable=True)


class WODataInventory(Base):
	"""Data inventory table for data governance"""
	__tablename__ = 'wo_data_inventory'
	__table_args__ = (
		Index('idx_wo_data_inventory_classification', 'classification'),
		Index('idx_wo_data_inventory_data_type', 'data_type'),
		Index('idx_wo_data_inventory_owner', 'owner'),
		Index('idx_wo_data_inventory_tenant_id', 'tenant_id'),
	)
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	data_type = Column(String(50), nullable=False)
	classification = Column(String(20), nullable=False)
	location = Column(String(500), nullable=False)
	owner = Column(String(100), nullable=False)
	steward = Column(String(100), nullable=False)
	retention_period = Column(Integer, nullable=True)
	purpose = Column(Text, nullable=False)
	legal_basis = Column(String(100), nullable=True)
	processing_activities = Column(ARRAY(String), nullable=True, default=[])
	sharing_agreements = Column(JSONB, nullable=True, default=[])
	protection_measures = Column(ARRAY(String), nullable=True, default=[])
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	tenant_id = Column(String(36), nullable=True)


class WOComplianceControl(Base):
	"""Compliance controls table"""
	__tablename__ = 'wo_compliance_controls'
	__table_args__ = (
		Index('idx_wo_compliance_controls_framework', 'framework'),
		Index('idx_wo_compliance_controls_control_type', 'control_type'),
		Index('idx_wo_compliance_controls_status', 'implementation_status'),
		Index('idx_wo_compliance_controls_test_date', 'next_test_date'),
		Index('idx_wo_compliance_controls_tenant_id', 'tenant_id'),
	)
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	control_id = Column(String(50), nullable=False)
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	framework = Column(String(50), nullable=False)
	control_type = Column(String(20), nullable=False)
	category = Column(String(50), nullable=False)
	objective = Column(Text, nullable=False)
	implementation_status = Column(String(30), nullable=False, default='not_implemented')
	effectiveness = Column(String(20), nullable=True)
	test_frequency = Column(String(20), nullable=False)
	last_test_date = Column(DateTime(timezone=True), nullable=True)
	next_test_date = Column(DateTime(timezone=True), nullable=True)
	test_results = Column(JSONB, nullable=True, default=[])
	responsible_party = Column(String(100), nullable=False)
	evidence_location = Column(String(500), nullable=True)
	automation_level = Column(String(20), nullable=False, default='manual')
	cost = Column(Float, nullable=True)
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	tenant_id = Column(String(36), nullable=True)


class WOComplianceIncident(Base):
	"""Compliance incidents table"""
	__tablename__ = 'wo_compliance_incidents'
	__table_args__ = (
		Index('idx_wo_compliance_incidents_severity', 'severity'),
		Index('idx_wo_compliance_incidents_status', 'status'),
		Index('idx_wo_compliance_incidents_framework', 'framework'),
		Index('idx_wo_compliance_incidents_discovery_date', 'discovery_date'),
		Index('idx_wo_compliance_incidents_tenant_id', 'tenant_id'),
	)
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	title = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	incident_type = Column(String(50), nullable=False)
	severity = Column(String(20), nullable=False)
	framework = Column(String(50), nullable=False)
	affected_systems = Column(ARRAY(String), nullable=True, default=[])
	affected_data = Column(ARRAY(String), nullable=True, default=[])
	root_cause = Column(Text, nullable=True)
	discovery_date = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	reported_date = Column(DateTime(timezone=True), nullable=True)
	resolution_date = Column(DateTime(timezone=True), nullable=True)
	status = Column(String(20), nullable=False, default='open')
	assignee = Column(String(100), nullable=True)
	remediation_actions = Column(JSONB, nullable=True, default=[])
	lessons_learned = Column(Text, nullable=True)
	regulatory_notification = Column(Boolean, nullable=False, default=False)
	customer_notification = Column(Boolean, nullable=False, default=False)
	created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
	tenant_id = Column(String(36), nullable=True)


# =============================================================================
# Core Workflow Models
# =============================================================================

class CRWorkflow(Base):
    """Core workflow definition table."""
    __tablename__ = 'cr_workflows'
    __table_args__ = (
        Index('idx_cr_workflows_tenant_id', 'tenant_id'),
        Index('idx_cr_workflows_category', 'category'),
        Index('idx_cr_workflows_active', 'is_active'),
        Index('idx_cr_workflows_created_at', 'created_at'),
        Index('idx_cr_workflows_tenant_active', 'tenant_id', 'is_active'),
        UniqueConstraint('workflow_id', 'tenant_id', name='uq_workflow_tenant'),
        CheckConstraint('version ~ \'^[0-9]+\\.[0-9]+\\.[0-9]+$\'', name='valid_version'),
        {'schema': 'workflow_orchestration'}
    )
    
    # Primary identifiers
    workflow_id = Column(String(26), primary_key=True, default=uuid7str)
    tenant_id = Column(String(26), nullable=False, index=True)
    
    # Basic information
    name = Column(String(200), nullable=False)
    description = Column(Text, default='')
    version = Column(String(50), nullable=False, default='1.0.0')
    category = Column(String(100), default='general', nullable=False)
    
    # Workflow definition
    workflow_definition = Column(JSONB, nullable=False)
    triggers = Column(JSONB, default=list)
    variables = Column(JSONB, default=dict)
    
    # Status and lifecycle
    is_active = Column(Boolean, default=True, nullable=False)
    is_template = Column(Boolean, default=False, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    
    # Performance settings
    max_concurrent_instances = Column(Integer, default=10)
    default_timeout_hours = Column(Integer)
    
    # Audit fields
    created_by = Column(String(26), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    updated_by = Column(String(26))
    
    # Relationships
    instances = relationship("CRWorkflowInstance", back_populates="workflow", cascade="all, delete-orphan")
    templates = relationship("CRWorkflowTemplate", back_populates="workflow", cascade="all, delete-orphan")

class CRWorkflowInstance(Base):
    """Workflow execution instance."""
    __tablename__ = 'cr_workflow_instances'
    __table_args__ = (
        Index('idx_cr_instances_workflow_id', 'workflow_id'),
        Index('idx_cr_instances_tenant_id', 'tenant_id'),
        Index('idx_cr_instances_status', 'status'),
        Index('idx_cr_instances_started_at', 'started_at'),
        Index('idx_cr_instances_tenant_status', 'tenant_id', 'status'),
        Index('idx_cr_instances_workflow_status', 'workflow_id', 'status'),
        CheckConstraint('status IN (\'draft\', \'active\', \'paused\', \'completed\', \'failed\', \'cancelled\')', name='valid_status'),
        CheckConstraint('progress_percentage >= 0 AND progress_percentage <= 100', name='valid_progress'),
        {'schema': 'workflow_orchestration'}
    )
    
    # Primary identifiers  
    instance_id = Column(String(26), primary_key=True, default=uuid7str)
    workflow_id = Column(String(26), ForeignKey('workflow_orchestration.cr_workflows.workflow_id'), nullable=False)
    tenant_id = Column(String(26), nullable=False, index=True)
    
    # Execution state
    status = Column(String(20), default='active', nullable=False)
    current_tasks = Column(ARRAY(String), default=list)
    completed_tasks = Column(ARRAY(String), default=list)
    failed_tasks = Column(ARRAY(String), default=list)
    skipped_tasks = Column(ARRAY(String), default=list)
    
    # Progress tracking
    progress_percentage = Column(Float, default=0.0)
    current_step = Column(String(200))
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    
    # Execution data
    context = Column(JSONB, default=dict)
    input_data = Column(JSONB, default=dict)
    output_data = Column(JSONB, default=dict)
    variables = Column(JSONB, default=dict)
    
    # Timing
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    completed_at = Column(DateTime(timezone=True))
    paused_at = Column(DateTime(timezone=True))
    resumed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    
    # Error handling
    error_message = Column(Text)
    error_details = Column(JSONB)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # SLA tracking
    sla_deadline = Column(DateTime(timezone=True))
    is_sla_breached = Column(Boolean, default=False)
    escalation_level = Column(Integer, default=0)
    escalated_to = Column(String(26))
    
    # Relationships and hierarchy
    parent_instance_id = Column(String(26), ForeignKey('workflow_orchestration.cr_workflow_instances.instance_id'))
    child_instance_ids = Column(ARRAY(String), default=list)
    
    # Audit fields
    started_by = Column(String(26), nullable=False)
    current_owner = Column(String(26))
    
    # Relationships
    workflow = relationship("CRWorkflow", back_populates="instances")
    task_executions = relationship("CRTaskExecution", back_populates="instance", cascade="all, delete-orphan")
    parent_instance = relationship("CRWorkflowInstance", remote_side=[instance_id])
    audit_logs = relationship("CRWorkflowAuditLog", back_populates="instance", cascade="all, delete-orphan")

class CRTaskExecution(Base):
    """Individual task execution record."""
    __tablename__ = 'cr_task_executions'
    __table_args__ = (
        Index('idx_cr_tasks_instance_id', 'instance_id'),
        Index('idx_cr_tasks_task_id', 'task_id'),
        Index('idx_cr_tasks_status', 'status'),
        Index('idx_cr_tasks_assigned_to', 'assigned_to'),
        Index('idx_cr_tasks_created_at', 'created_at'),
        Index('idx_cr_tasks_due_date', 'due_date'),
        Index('idx_cr_tasks_sla_deadline', 'sla_deadline'),
        Index('idx_cr_tasks_instance_status', 'instance_id', 'status'),
        CheckConstraint('status IN (\'pending\', \'ready\', \'assigned\', \'in_progress\', \'completed\', \'failed\', \'skipped\', \'cancelled\', \'escalated\', \'expired\')', name='valid_task_status'),
        CheckConstraint('priority >= 1 AND priority <= 10', name='valid_priority'),
        CheckConstraint('progress_percentage >= 0 AND progress_percentage <= 100', name='valid_task_progress'),
        {'schema': 'workflow_orchestration'}
    )
    
    # Primary identifiers
    execution_id = Column(String(26), primary_key=True, default=uuid7str)
    instance_id = Column(String(26), ForeignKey('workflow_orchestration.cr_workflow_instances.instance_id'), nullable=False)
    task_id = Column(String(26), nullable=False)
    task_name = Column(String(200), nullable=False)
    
    # Assignment details
    assigned_to = Column(String(26))
    assigned_role = Column(String(100))
    assigned_group = Column(String(100))
    current_assignee = Column(String(26))
    
    # Execution state
    status = Column(String(20), default='pending', nullable=False)
    priority = Column(Integer, default=5)
    
    # Timing
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    assigned_at = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    due_date = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    
    # Execution data
    input_data = Column(JSONB, default=dict)
    output_data = Column(JSONB, default=dict)
    result = Column(JSONB, default=dict)
    
    # Progress tracking
    progress_percentage = Column(Float, default=0.0)
    progress_message = Column(Text)
    
    # Error handling
    error_message = Column(Text)
    error_details = Column(JSONB)
    attempt_number = Column(Integer, default=1)
    max_attempts = Column(Integer, default=3)
    retry_at = Column(DateTime(timezone=True))
    
    # Human task specifics
    comments = Column(JSONB, default=list)
    attachments = Column(JSONB, default=list)
    approval_decision = Column(String(20))
    approval_reason = Column(Text)
    
    # Escalation tracking
    escalation_level = Column(Integer, default=0)
    escalated_at = Column(DateTime(timezone=True))
    escalated_to = Column(String(26))
    escalation_reason = Column(Text)
    
    # SLA tracking
    sla_deadline = Column(DateTime(timezone=True))
    is_sla_breached = Column(Boolean, default=False)
    sla_breach_time = Column(DateTime(timezone=True))
    
    # Audit fields
    created_by = Column(String(26), nullable=False)
    updated_by = Column(String(26))
    audit_events = Column(JSONB, default=list)
    
    # Metadata
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    instance = relationship("CRWorkflowInstance", back_populates="task_executions")

class CRWorkflowTemplate(Base):
    """Reusable workflow templates."""
    __tablename__ = 'cr_workflow_templates'
    __table_args__ = (
        Index('idx_cr_templates_category', 'category'),
        Index('idx_cr_templates_industry', 'industry'),
        Index('idx_cr_templates_public', 'is_public'),
        Index('idx_cr_templates_certified', 'is_certified'),
        Index('idx_cr_templates_usage_count', 'usage_count'),
        CheckConstraint('complexity_level IN (\'beginner\', \'intermediate\', \'advanced\', \'expert\')', name='valid_complexity'),
        CheckConstraint('estimated_duration_hours >= 0', name='valid_duration'),
        CheckConstraint('usage_count >= 0', name='valid_usage_count'),
        {'schema': 'workflow_orchestration'}
    )
    
    # Primary identifiers
    template_id = Column(String(26), primary_key=True, default=uuid7str)
    workflow_id = Column(String(26), ForeignKey('workflow_orchestration.cr_workflows.workflow_id'), nullable=False)
    
    # Template information
    name = Column(String(200), nullable=False)
    description = Column(Text, default='')
    category = Column(String(100), default='general')
    tags = Column(ARRAY(String), default=list)
    industry = Column(String(100))
    complexity_level = Column(String(20), default='intermediate')
    estimated_duration_hours = Column(Float)
    
    # Template data
    template_data = Column(JSONB, nullable=False)
    variables = Column(JSONB, default=dict)
    prerequisites = Column(ARRAY(String), default=list)
    outcomes = Column(ARRAY(String), default=list)
    
    # Status and access
    is_public = Column(Boolean, default=False)
    is_certified = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)
    
    # Audit fields
    created_by = Column(String(26), nullable=False)
    tenant_id = Column(String(26), nullable=False)
    version = Column(String(50), default='1.0.0')
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    workflow = relationship("CRWorkflow", back_populates="templates")

class CRWorkflowConnector(Base):
    """External system connector configurations."""
    __tablename__ = 'cr_workflow_connectors'
    __table_args__ = (
        Index('idx_cr_connectors_tenant_id', 'tenant_id'),
        Index('idx_cr_connectors_type', 'connector_type'),
        Index('idx_cr_connectors_enabled', 'is_enabled'),
        Index('idx_cr_connectors_health', 'health_status'),
        CheckConstraint('health_status IN (\'healthy\', \'degraded\', \'unhealthy\', \'unknown\')', name='valid_health_status'),
        CheckConstraint('audit_level IN (\'none\', \'basic\', \'detailed\')', name='valid_audit_level'),
        CheckConstraint('timeout_seconds >= 1 AND timeout_seconds <= 300', name='valid_timeout'),
        {'schema': 'workflow_orchestration'}
    )
    
    # Primary identifiers
    connector_id = Column(String(26), primary_key=True, default=uuid7str)
    tenant_id = Column(String(26), nullable=False)
    
    # Connector details
    name = Column(String(200), nullable=False)
    description = Column(Text, default='')
    connector_type = Column(String(100), nullable=False)
    
    # Configuration
    connection_config = Column(JSONB, nullable=False)
    authentication_config = Column(JSONB, default=dict)
    
    # Status and health
    is_enabled = Column(Boolean, default=True)
    is_validated = Column(Boolean, default=False)
    last_test_at = Column(DateTime(timezone=True))
    last_test_result = Column(Text)
    
    # Rate limiting
    rate_limit_per_minute = Column(Integer)
    daily_quota = Column(Integer)
    current_usage = Column(Integer, default=0)
    
    # Error handling
    retry_configuration = Column(JSONB, default=dict)
    timeout_seconds = Column(Integer, default=30)
    
    # Health monitoring
    health_check_enabled = Column(Boolean, default=True)
    health_check_interval_minutes = Column(Integer, default=5)
    last_health_check = Column(DateTime(timezone=True))
    health_status = Column(String(20), default='unknown')
    
    # Security
    encryption_enabled = Column(Boolean, default=True)
    audit_level = Column(String(20), default='basic')
    
    # Audit fields
    created_by = Column(String(26), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    updated_by = Column(String(26))
    
    # Metadata
    metadata = Column(JSONB, default=dict)

class CRWorkflowAuditLog(Base):
    """Comprehensive audit logging for workflows."""
    __tablename__ = 'cr_workflow_audit_logs'
    __table_args__ = (
        Index('idx_cr_audit_tenant_id', 'tenant_id'),
        Index('idx_cr_audit_timestamp', 'timestamp'),
        Index('idx_cr_audit_workflow_id', 'workflow_id'),
        Index('idx_cr_audit_instance_id', 'instance_id'),
        Index('idx_cr_audit_user_id', 'user_id'),
        Index('idx_cr_audit_event_type', 'event_type'),
        Index('idx_cr_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_cr_audit_correlation', 'correlation_id'),
        Index('idx_cr_audit_tenant_timestamp', 'tenant_id', 'timestamp'),
        CheckConstraint('event_category IN (\'workflow\', \'instance\', \'task\', \'user\', \'system\', \'security\', \'performance\')', name='valid_event_category'),
        CheckConstraint('result IN (\'success\', \'failure\', \'partial\')', name='valid_result'),
        CheckConstraint('impact_level IN (\'low\', \'medium\', \'high\', \'critical\')', name='valid_impact_level'),
        CheckConstraint('security_classification IN (\'public\', \'internal\', \'confidential\', \'restricted\')', name='valid_security_classification'),
        {'schema': 'workflow_orchestration'}
    )
    
    # Primary identifier
    audit_id = Column(String(26), primary_key=True, default=uuid7str)
    tenant_id = Column(String(26), nullable=False)
    
    # Context identification
    workflow_id = Column(String(26), ForeignKey('workflow_orchestration.cr_workflows.workflow_id'))
    instance_id = Column(String(26), ForeignKey('workflow_orchestration.cr_workflow_instances.instance_id'))
    task_execution_id = Column(String(26), ForeignKey('workflow_orchestration.cr_task_executions.execution_id'))
    
    # Event details
    event_type = Column(String(100), nullable=False)
    event_category = Column(String(20), nullable=False)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(26), nullable=False)
    
    # User context
    user_id = Column(String(26))
    session_id = Column(String(100))
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    
    # Event data
    event_data = Column(JSONB, default=dict)
    previous_values = Column(JSONB)
    new_values = Column(JSONB)
    
    # Result and impact
    result = Column(String(20), nullable=False)
    error_message = Column(Text)
    impact_level = Column(String(20), default='low')
    
    # Compliance
    compliance_tags = Column(ARRAY(String), default=list)
    security_classification = Column(String(20), default='internal')
    retention_policy = Column(String(50), default='standard')
    
    # Timing
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    duration_ms = Column(Integer)
    
    # Tracing
    correlation_id = Column(String(100))
    trace_id = Column(String(100))
    parent_span_id = Column(String(100))
    
    # Metadata
    metadata = Column(JSONB, default=dict)
    
    # Cached names for performance
    user_name = Column(String(200))
    resource_name = Column(String(200))
    
    # Relationships
    workflow = relationship("CRWorkflow")
    instance = relationship("CRWorkflowInstance", back_populates="audit_logs")
    task_execution = relationship("CRTaskExecution")

# =============================================================================
# Performance and Analytics Models
# =============================================================================

class CRWorkflowMetrics(Base):
    """Workflow performance metrics and analytics."""
    __tablename__ = 'cr_workflow_metrics'
    __table_args__ = (
        Index('idx_cr_metrics_workflow_id', 'workflow_id'),
        Index('idx_cr_metrics_tenant_id', 'tenant_id'),
        Index('idx_cr_metrics_recorded_at', 'recorded_at'),
        Index('idx_cr_metrics_workflow_tenant', 'workflow_id', 'tenant_id'),
        {'schema': 'workflow_orchestration'}
    )
    
    # Primary identifier
    metric_id = Column(String(26), primary_key=True, default=uuid7str)
    workflow_id = Column(String(26), ForeignKey('workflow_orchestration.cr_workflows.workflow_id'), nullable=False)
    tenant_id = Column(String(26), nullable=False)
    
    # Metrics data
    total_executions = Column(Integer, default=0)
    successful_executions = Column(Integer, default=0)
    failed_executions = Column(Integer, default=0)
    average_duration_seconds = Column(Float, default=0.0)
    last_execution_at = Column(DateTime(timezone=True))
    success_rate_percentage = Column(Float, default=0.0)
    
    # Performance data
    performance_data = Column(JSONB, default=dict)
    user_productivity = Column(JSONB, default=dict)
    sla_metrics = Column(JSONB, default=dict)
    
    # Timing
    recorded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))

# =============================================================================
# Database Engine and Session Management  
# =============================================================================

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str, echo: bool = False):
        self.database_url = database_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self.echo = echo
    
    async def initialize(self):
        """Initialize database engine and session factory."""
        self.engine = create_async_engine(
            self.database_url,
            echo=self.echo,
            poolclass=NullPool if "sqlite" in self.database_url else None,
            pool_pre_ping=True,
            connect_args={
                "command_timeout": 60,
                "server_settings": {
                    "application_name": "workflow_orchestration",
                }
            } if "postgresql" in self.database_url else {}
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Database engine initialized", database_url=self.database_url)
    
    async def create_tables(self):
        """Create all database tables."""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        async with self.engine.begin() as conn:
            # Create schema
            await conn.execute(text("CREATE SCHEMA IF NOT EXISTS workflow_orchestration"))
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created")
    
    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        if not self.session_factory:
            raise RuntimeError("Session factory not initialized")
        
        return self.session_factory()
    
    async def close(self):
        """Close database engine."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine closed")

# =============================================================================
# Data Access Layer
# =============================================================================

class WorkflowRepository:
    """Repository for workflow data access operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_workflow(self, workflow_data: Dict[str, Any]) -> CRWorkflow:
        """Create a new workflow."""
        workflow = CRWorkflow(**workflow_data)
        self.session.add(workflow)
        await self.session.commit()
        await self.session.refresh(workflow)
        return workflow
    
    async def get_workflow(self, workflow_id: str, tenant_id: str) -> Optional[CRWorkflow]:
        """Get workflow by ID and tenant."""
        result = await self.session.execute(
            select(CRWorkflow)
            .options(selectinload(CRWorkflow.instances))
            .where(
                and_(
                    CRWorkflow.workflow_id == workflow_id,
                    CRWorkflow.tenant_id == tenant_id
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def list_workflows(
        self, 
        tenant_id: str, 
        category: Optional[str] = None,
        is_active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[CRWorkflow]:
        """List workflows with filtering."""
        query = select(CRWorkflow).where(CRWorkflow.tenant_id == tenant_id)
        
        if category:
            query = query.where(CRWorkflow.category == category)
        if is_active is not None:
            query = query.where(CRWorkflow.is_active == is_active)
        
        query = query.order_by(CRWorkflow.created_at.desc()).limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update_workflow(self, workflow_id: str, tenant_id: str, updates: Dict[str, Any]) -> bool:
        """Update workflow."""
        updates['updated_at'] = datetime.now(timezone.utc)
        
        result = await self.session.execute(
            update(CRWorkflow)
            .where(
                and_(
                    CRWorkflow.workflow_id == workflow_id,
                    CRWorkflow.tenant_id == tenant_id
                )
            )
            .values(**updates)
        )
        
        await self.session.commit()
        return result.rowcount > 0
    
    async def delete_workflow(self, workflow_id: str, tenant_id: str) -> bool:
        """Delete workflow."""
        result = await self.session.execute(
            delete(CRWorkflow)
            .where(
                and_(
                    CRWorkflow.workflow_id == workflow_id,
                    CRWorkflow.tenant_id == tenant_id
                )
            )
        )
        
        await self.session.commit()
        return result.rowcount > 0

class WorkflowInstanceRepository:
    """Repository for workflow instance operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_instance(self, instance_data: Dict[str, Any]) -> CRWorkflowInstance:
        """Create workflow instance."""
        instance = CRWorkflowInstance(**instance_data)
        self.session.add(instance)
        await self.session.commit()
        await self.session.refresh(instance)
        return instance
    
    async def get_instance(
        self, 
        instance_id: str, 
        tenant_id: Optional[str] = None
    ) -> Optional[CRWorkflowInstance]:
        """Get workflow instance."""
        query = select(CRWorkflowInstance).options(
            selectinload(CRWorkflowInstance.task_executions),
            selectinload(CRWorkflowInstance.workflow)
        ).where(CRWorkflowInstance.instance_id == instance_id)
        
        if tenant_id:
            query = query.where(CRWorkflowInstance.tenant_id == tenant_id)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def list_instances(
        self, 
        tenant_id: str,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[CRWorkflowInstance]:
        """List workflow instances."""
        query = select(CRWorkflowInstance).where(CRWorkflowInstance.tenant_id == tenant_id)
        
        if workflow_id:
            query = query.where(CRWorkflowInstance.workflow_id == workflow_id)
        if status:
            query = query.where(CRWorkflowInstance.status == status)
        
        query = query.order_by(CRWorkflowInstance.started_at.desc()).limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update_instance(
        self, 
        instance_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update workflow instance."""
        result = await self.session.execute(
            update(CRWorkflowInstance)
            .where(CRWorkflowInstance.instance_id == instance_id)
            .values(**updates)
        )
        
        await self.session.commit()
        return result.rowcount > 0

class TaskExecutionRepository:
    """Repository for task execution operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_task_execution(self, task_data: Dict[str, Any]) -> CRTaskExecution:
        """Create task execution."""
        task = CRTaskExecution(**task_data)
        self.session.add(task)
        await self.session.commit()
        await self.session.refresh(task)
        return task
    
    async def get_task_execution(self, execution_id: str) -> Optional[CRTaskExecution]:
        """Get task execution."""
        result = await self.session.execute(
            select(CRTaskExecution)
            .options(selectinload(CRTaskExecution.instance))
            .where(CRTaskExecution.execution_id == execution_id)
        )
        return result.scalar_one_or_none()
    
    async def list_user_tasks(
        self, 
        user_id: str,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[CRTaskExecution]:
        """List tasks assigned to user."""
        query = select(CRTaskExecution).where(
            or_(
                CRTaskExecution.assigned_to == user_id,
                CRTaskExecution.current_assignee == user_id
            )
        )
        
        if tenant_id:
            query = query.join(CRWorkflowInstance).where(CRWorkflowInstance.tenant_id == tenant_id)
        if status:
            query = query.where(CRTaskExecution.status == status)
        
        query = query.order_by(CRTaskExecution.created_at.desc()).limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update_task_execution(
        self, 
        execution_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update task execution."""
        result = await self.session.execute(
            update(CRTaskExecution)
            .where(CRTaskExecution.execution_id == execution_id)
            .values(**updates)
        )
        
        await self.session.commit()
        return result.rowcount > 0

# =============================================================================
# Factory Functions
# =============================================================================

async def create_database_manager(database_url: str, echo: bool = False) -> DatabaseManager:
    """Create and initialize database manager."""
    manager = DatabaseManager(database_url, echo)
    await manager.initialize()
    return manager

async def create_repositories(session: AsyncSession) -> tuple:
    """Create repository instances."""
    return (
        WorkflowRepository(session),
        WorkflowInstanceRepository(session),
        TaskExecutionRepository(session)
    )

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models
    "CRWorkflow",
    "CRWorkflowInstance", 
    "CRTaskExecution",
    "CRWorkflowTemplate",
    "CRWorkflowConnector",
    "CRWorkflowAuditLog",
    "CRWorkflowMetrics",
    
    # Database management
    "DatabaseManager",
    "Base",
    "metadata",
    
    # Repositories
    "WorkflowRepository",
    "WorkflowInstanceRepository",
    "TaskExecutionRepository",
    
    # Factory functions
    "create_database_manager",
    "create_repositories"
]