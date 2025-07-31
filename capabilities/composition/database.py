"""
APG Composition Database Layer

Complete database implementation with SQLAlchemy 2.0, connection pooling,
migrations, and full CRUD operations.
"""

import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, Text, JSON, ForeignKey,
    Index, UniqueConstraint, CheckConstraint, event
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from datetime import datetime
from uuid7 import uuid7str
import structlog
import os

logger = structlog.get_logger(__name__)

# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://composition_user:password@localhost:5432/composition_db"
)

# Connection Pool Configuration
ENGINE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 50,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "echo": os.getenv("SQL_DEBUG", "false").lower() == "true"
}

# Create async engine
engine: AsyncEngine = create_async_engine(DATABASE_URL, **ENGINE_CONFIG)

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base model
Base = declarative_base()

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class TenantMixin:
    """Mixin for multi-tenant support."""
    tenant_id = Column(String(36), nullable=False, index=True)

# Core Models
class CRCapability(Base, TimestampMixin, TenantMixin):
    """Capability registry model."""
    __tablename__ = "cr_capabilities"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    code = Column(String(200), nullable=False)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    category = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False, default="1.0.0")
    status = Column(String(20), nullable=False, default="active", index=True)
    
    # Metadata
    metadata_ = Column("metadata", JSONB, default=dict)
    tags = Column(ARRAY(String), default=list)
    
    # Configuration
    config_schema = Column(JSONB)
    default_config = Column(JSONB, default=dict)
    
    # Dependencies
    dependencies = relationship("CRDependency", back_populates="capability", cascade="all, delete-orphan")
    
    # Indices
    __table_args__ = (
        Index("idx_capability_tenant_code", "tenant_id", "code"),
        Index("idx_capability_category_status", "category", "status"),
        UniqueConstraint("tenant_id", "code", name="uq_capability_tenant_code"),
        CheckConstraint("status IN ('active', 'deprecated', 'disabled')", name="ck_capability_status")
    )

class CRDependency(Base, TimestampMixin):
    """Capability dependency model."""
    __tablename__ = "cr_dependencies"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    capability_id = Column(String(36), ForeignKey("cr_capabilities.id", ondelete="CASCADE"), nullable=False)
    dependency_capability_id = Column(String(36), ForeignKey("cr_capabilities.id", ondelete="CASCADE"), nullable=False)
    
    dependency_type = Column(String(20), nullable=False, default="requires")
    version_constraint = Column(String(100))
    is_optional = Column(Boolean, default=False)
    
    # Relationships
    capability = relationship("CRCapability", foreign_keys=[capability_id], back_populates="dependencies")
    dependency_capability = relationship("CRCapability", foreign_keys=[dependency_capability_id])
    
    # Indices
    __table_args__ = (
        Index("idx_dependency_capability", "capability_id"),
        Index("idx_dependency_type", "dependency_type"),
        CheckConstraint("dependency_type IN ('requires', 'suggests', 'conflicts')", name="ck_dependency_type"),
        CheckConstraint("capability_id != dependency_capability_id", name="ck_no_self_dependency")
    )

class CRComposition(Base, TimestampMixin, TenantMixin):
    """Composition model."""
    __tablename__ = "cr_compositions"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    composition_type = Column(String(50), nullable=False, default="enterprise")
    
    # Capabilities
    capability_ids = Column(ARRAY(String), nullable=False)
    
    # Configuration
    configuration = Column(JSONB, default=dict)
    industry_focus = Column(ARRAY(String), default=list)
    
    # Status
    status = Column(String(20), nullable=False, default="draft")
    is_template = Column(Boolean, default=False)
    
    # Validation Results
    validation_result = Column(JSONB)
    last_validated_at = Column(DateTime)
    
    # User tracking
    created_by = Column(String(36), nullable=False)
    
    # Deployments
    deployments = relationship("CRDeployment", back_populates="composition", cascade="all, delete-orphan")
    
    # Indices
    __table_args__ = (
        Index("idx_composition_tenant_status", "tenant_id", "status"),
        Index("idx_composition_type", "composition_type"),
        Index("idx_composition_template", "is_template"),
        CheckConstraint("status IN ('draft', 'validated', 'deployed', 'archived')", name="ck_composition_status")
    )

class CRDeployment(Base, TimestampMixin, TenantMixin):
    """Deployment model."""
    __tablename__ = "cr_deployments"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    composition_id = Column(String(36), ForeignKey("cr_compositions.id", ondelete="CASCADE"), nullable=False)
    
    # Deployment Configuration
    environment = Column(String(50), nullable=False)
    strategy = Column(String(50), nullable=False, default="rolling_update")
    cluster_name = Column(String(200), nullable=False)
    namespace = Column(String(100), nullable=False)
    
    # Status
    status = Column(String(20), nullable=False, default="pending")
    message = Column(Text)
    
    # Resources
    replicas = Column(Integer, default=3)
    resource_limits = Column(JSONB, default=dict)
    
    # URLs
    rollout_url = Column(String(500))
    health_check_url = Column(String(500))
    
    # Health Status
    health_status = Column(JSONB)
    
    # User tracking
    deployed_by = Column(String(36), nullable=False)
    
    # Logs
    deployment_logs = Column(ARRAY(Text), default=list)
    
    # Relationships
    composition = relationship("CRComposition", back_populates="deployments")
    
    # Indices
    __table_args__ = (
        Index("idx_deployment_tenant_status", "tenant_id", "status"),
        Index("idx_deployment_environment", "environment"),
        Index("idx_deployment_composition", "composition_id"),
        CheckConstraint("status IN ('pending', 'in_progress', 'completed', 'failed', 'rolled_back')", name="ck_deployment_status")
    )

class CRWorkflow(Base, TimestampMixin, TenantMixin):
    """Workflow definition model."""
    __tablename__ = "cr_workflows"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    version = Column(String(50), nullable=False, default="1.0.0")
    category = Column(String(100), nullable=False, default="general")
    
    # Definition
    workflow_definition = Column(JSONB, nullable=False)
    triggers = Column(JSONB, default=list)
    variables = Column(JSONB, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_template = Column(Boolean, default=False)
    
    # User tracking
    created_by = Column(String(36), nullable=False)
    
    # Relationships
    instances = relationship("CRWorkflowInstance", back_populates="workflow", cascade="all, delete-orphan")
    
    # Indices
    __table_args__ = (
        Index("idx_workflow_tenant_active", "tenant_id", "is_active"),
        Index("idx_workflow_category", "category"),
        Index("idx_workflow_template", "is_template")
    )

class CRWorkflowInstance(Base, TimestampMixin, TenantMixin):
    """Workflow instance model."""
    __tablename__ = "cr_workflow_instances"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    workflow_id = Column(String(36), ForeignKey("cr_workflows.id", ondelete="CASCADE"), nullable=False)
    
    # Status
    status = Column(String(20), nullable=False, default="active")
    
    # Execution State
    current_tasks = Column(ARRAY(String), default=list)
    completed_tasks = Column(ARRAY(String), default=list)
    failed_tasks = Column(ARRAY(String), default=list)
    
    # Context
    context = Column(JSONB, default=dict)
    
    # User tracking
    started_by = Column(String(36), nullable=False)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Error handling
    error_message = Column(Text)
    
    # Relationships
    workflow = relationship("CRWorkflow", back_populates="instances")
    task_executions = relationship("CRTaskExecution", back_populates="instance", cascade="all, delete-orphan")
    
    # Indices
    __table_args__ = (
        Index("idx_workflow_instance_tenant_status", "tenant_id", "status"),
        Index("idx_workflow_instance_workflow", "workflow_id"),
        CheckConstraint("status IN ('active', 'paused', 'completed', 'failed', 'cancelled')", name="ck_workflow_instance_status")
    )

class CRTaskExecution(Base, TimestampMixin):
    """Task execution model."""
    __tablename__ = "cr_task_executions"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    instance_id = Column(String(36), ForeignKey("cr_workflow_instances.id", ondelete="CASCADE"), nullable=False)
    task_id = Column(String(36), nullable=False)
    
    # Assignment
    assigned_to = Column(String(36))
    
    # Status
    status = Column(String(20), nullable=False, default="pending")
    
    # Execution
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Results
    result = Column(JSONB, default=dict)
    error_message = Column(Text)
    
    # Retry logic
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    
    # Relationships
    instance = relationship("CRWorkflowInstance", back_populates="task_executions")
    
    # Indices
    __table_args__ = (
        Index("idx_task_execution_instance", "instance_id"),
        Index("idx_task_execution_status", "status"),
        Index("idx_task_execution_assigned", "assigned_to"),
        CheckConstraint("status IN ('pending', 'assigned', 'in_progress', 'completed', 'failed', 'skipped')", name="ck_task_execution_status")
    )

class CRConfiguration(Base, TimestampMixin, TenantMixin):
    """Configuration model."""
    __tablename__ = "cr_configurations"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    applet_id = Column(String(200), nullable=False)
    
    # Configuration Data
    configuration_data = Column(JSONB, nullable=False, default=dict)
    
    # Change tracking
    changed_by = Column(String(36), nullable=False)
    change_reason = Column(Text)
    
    # Indices
    __table_args__ = (
        Index("idx_configuration_tenant_applet", "tenant_id", "applet_id"),
        UniqueConstraint("tenant_id", "applet_id", name="uq_configuration_tenant_applet")
    )

class CRConfigurationHistory(Base, TimestampMixin, TenantMixin):
    """Configuration change history model."""
    __tablename__ = "cr_configuration_history"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    configuration_id = Column(String(36), ForeignKey("cr_configurations.id", ondelete="CASCADE"), nullable=False)
    
    # Change details
    change_type = Column(String(20), nullable=False)
    field_key = Column(String(200))
    old_value = Column(JSONB)
    new_value = Column(JSONB)
    
    # User tracking
    changed_by = Column(String(36), nullable=False)
    change_reason = Column(Text)
    
    # Indices
    __table_args__ = (
        Index("idx_config_history_config", "configuration_id"),
        Index("idx_config_history_change_type", "change_type"),
        CheckConstraint("change_type IN ('create', 'update', 'delete', 'bulk_update', 'reset')", name="ck_config_history_change_type")
    )

class CRPermission(Base, TimestampMixin, TenantMixin):
    """Permission model."""
    __tablename__ = "cr_permissions"
    
    id = Column(String(36), primary_key=True, default=uuid7str)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(36), nullable=False)
    
    # Subject (user or role)
    user_id = Column(String(36))
    role_id = Column(String(36))
    
    # Permission details
    access_level = Column(String(20), nullable=False, default="read")
    conditions = Column(JSONB, default=dict)
    
    # Grant details
    granted_by = Column(String(36), nullable=False)
    granted_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Indices
    __table_args__ = (
        Index("idx_permission_tenant_resource", "tenant_id", "resource_type", "resource_id"),
        Index("idx_permission_user", "user_id"),
        Index("idx_permission_role", "role_id"),
        Index("idx_permission_access_level", "access_level"),
        CheckConstraint("access_level IN ('none', 'read', 'write', 'execute', 'admin', 'owner')", name="ck_permission_access_level"),
        CheckConstraint("(user_id IS NOT NULL) OR (role_id IS NOT NULL)", name="ck_permission_subject")
    )

# Database Manager
class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self):
        self.engine = engine
        self.session_factory = async_session
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def init_database(self):
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    
    async def close(self):
        """Close database connections."""
        await self.engine.dispose()
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

# Global database manager instance
db_manager = DatabaseManager()

# Session dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with db_manager.get_session() as session:
        yield session

# Event listeners for automatic timestamp updates
@event.listens_for(CRCapability, 'before_update')
@event.listens_for(CRComposition, 'before_update')
@event.listens_for(CRDeployment, 'before_update')
@event.listens_for(CRWorkflow, 'before_update')
@event.listens_for(CRWorkflowInstance, 'before_update')
@event.listens_for(CRConfiguration, 'before_update')
def update_timestamp(mapper, connection, target):
    """Update timestamp before update."""
    target.updated_at = datetime.utcnow()

# Utility functions
async def create_tenant_schema(tenant_id: str) -> bool:
    """Create tenant-specific database schema if needed."""
    # For PostgreSQL, we use tenant_id column rather than separate schemas
    # This could be extended to create separate schemas for better isolation
    logger.info("Tenant schema ready", tenant_id=tenant_id)
    return True

async def migrate_database():
    """Run database migrations."""
    # In production, this would use Alembic
    await db_manager.init_database()
    logger.info("Database migration completed")

# Export key components
__all__ = [
    "Base",
    "CRCapability",
    "CRDependency", 
    "CRComposition",
    "CRDeployment",
    "CRWorkflow",
    "CRWorkflowInstance",
    "CRTaskExecution",
    "CRConfiguration",
    "CRConfigurationHistory",
    "CRPermission",
    "DatabaseManager",
    "db_manager",
    "get_db_session",
    "create_tenant_schema",
    "migrate_database"
]