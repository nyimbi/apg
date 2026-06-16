"""
APG Connection Management SQLAlchemy Models
PostgreSQL database models with 2-character prefix 'Cn' for Connection capability

Provides comprehensive data models for:
- Connection management (CnConnection)
- Data flows (CnDataFlow)
- Singer.io taps/targets (CnSingerTap, CnSingerTarget)
- Transformation rules (CnTransformationRule)
- Data lineage (CnLineageNode, CnLineageEdge)
- Health monitoring (CnHealthCheck)

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Enum as SQLEnum, Float, Index, UniqueConstraint
from sqlalchemy.types import CHAR, TypeDecorator
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB

JSONVariant = JSON().with_variant(JSONB, "postgresql")
from datetime import datetime, timezone
import uuid
import enum

Base = declarative_base()


class GUID(TypeDecorator):
    """Portable UUID type that accepts UUID objects and strings."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None or isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class GUIDString(GUID):
    """Portable UUID foreign key type that returns strings for legacy callers."""

    cache_ok = True

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return str(value)


class ConnectionStatus(str, enum.Enum):
    """Connection status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    TESTING = "testing"
    CONFIGURING = "configuring"


class ConnectionType(str, enum.Enum):
    """Connection type enumeration"""
    DATABASE = "database"
    API = "api"
    FILE = "file"
    STREAM = "stream"
    WEBHOOK = "webhook"
    QUEUE = "queue"


class SyncMode(str, enum.Enum):
    """Data synchronization mode"""
    FULL_TABLE = "full_refresh"
    FULL_REFRESH = "full_refresh"
    INCREMENTAL = "incremental"
    LOG_BASED = "log_based"
    CHANGE_DATA_CAPTURE = "change_data_capture"


class FlowStatus(str, enum.Enum):
    """Data flow status"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class ExecutionStatus(str, enum.Enum):
    """Flow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "error"
    ERROR = "error"
    CANCELLED = "cancelled"


class LineageNodeType(str, enum.Enum):
    """Data lineage node types"""
    CONNECTION = "connection"
    TABLE = "table"
    VIEW = "view"
    FIELD = "field"
    TRANSFORMATION = "transformation"
    FLOW = "flow"


class CnConnection(Base):
    """
    Core connection model for managing data source connections.
    Integrates with Singer.io ecosystem and APG security framework.
    """
    __tablename__ = 'cn_connections'

    # Primary Key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # APG Integration
    tenant_id = Column(String(100), nullable=False, index=True)

    # Basic Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)

    # Connection Details
    connection_type = Column(SQLEnum(ConnectionType), nullable=False, index=True)
    status = Column(SQLEnum(ConnectionStatus), default=ConnectionStatus.INACTIVE, index=True)

    # Singer.io Integration
    singer_tap = Column(String(100), index=True)
    singer_target = Column(String(100), index=True)
    tap_config = Column(JSONVariant, default={})
    target_config = Column(JSONVariant, default={})

    # Security & Authentication
    credentials_encrypted = Column(Boolean, default=True)
    credentials_key_id = Column(String(100))

    # Configuration
    sync_mode = Column(SQLEnum(SyncMode), default=SyncMode.FULL_TABLE)
    sync_frequency = Column(String(100))  # Cron expression
    batch_size = Column(Integer, default=1000)

    # Monitoring & Health
    enabled = Column(Boolean, default=True, index=True)
    last_sync = Column(DateTime(timezone=True))
    last_success = Column(DateTime(timezone=True))
    last_error = Column(Text)
    error_count = Column(Integer, default=0)
    records_processed = Column(Integer, default=0)

    # Metadata
    tags = Column(JSONVariant, default=[])
    meta_data = Column(JSONVariant, default={})

    # Audit Fields
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    updated_by = Column(String(100))

    # Relationships
    source_flows = relationship("CnDataFlow", foreign_keys="CnDataFlow.source_connection_id", back_populates="source_connection")
    target_flows = relationship("CnDataFlow", foreign_keys="CnDataFlow.target_connection_id", back_populates="target_connection")
    health_checks = relationship("CnHealthCheck", back_populates="connection")
    lineage_nodes = relationship("CnLineageNode", back_populates="connection")

    # Indexes
    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', name='uq_cn_connections_tenant_name'),
        Index('idx_cn_connections_tenant_status', 'tenant_id', 'status'),
        Index('idx_cn_connections_type_enabled', 'connection_type', 'enabled'),
        Index('idx_cn_connections_last_sync', 'last_sync'),
    )

    @validates('name')
    def validate_name(self, key, name):
        if not name or len(name.strip()) == 0:
            raise ValueError("Connection name cannot be empty")
        return name.strip()

    @validates('batch_size')
    def validate_batch_size(self, key, batch_size):
        if batch_size < 1 or batch_size > 100000:
            raise ValueError("Batch size must be between 1 and 100000")
        return batch_size

    def __repr__(self):
        return f"<CnConnection(name='{self.name}', type='{self.connection_type}', status='{self.status}')>"


class CnDataFlow(Base):
    """
    Data flow model for managing data pipelines between connections.
    Supports complex transformations, scheduling, and monitoring.
    """
    __tablename__ = 'cn_data_flows'

    # Primary Key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # APG Integration
    tenant_id = Column(String(100), nullable=False, index=True)

    # Basic Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)

    # Flow Configuration
    source_connection_id = Column(GUIDString(), ForeignKey('cn_connections.id'), nullable=False, index=True)
    target_connection_id = Column(GUIDString(), ForeignKey('cn_connections.id'), nullable=False, index=True)

    # Status and Control
    status = Column(SQLEnum(FlowStatus), default=FlowStatus.DRAFT, index=True)
    enabled = Column(Boolean, default=True, index=True)

    # Scheduling
    schedule_expression = Column(String(100))  # Cron expression
    schedule_timezone = Column(String(50), default='UTC')

    # Data Processing
    field_mappings = Column(JSONVariant, default={})
    transformation_config = Column(JSONVariant, default={})
    filter_config = Column(JSONVariant, default={})

    # State Management
    current_state = Column(JSONVariant, default={})
    last_state_update = Column(DateTime(timezone=True))

    # Execution Metrics
    last_execution = Column(DateTime(timezone=True))
    next_execution = Column(DateTime(timezone=True))
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    records_processed = Column(Integer, default=0)

    # Performance Metrics
    avg_execution_time_seconds = Column(Float, default=0.0)
    last_execution_time_seconds = Column(Float, default=0.0)

    # Metadata
    tags = Column(JSONVariant, default=[])
    meta_data = Column(JSONVariant, default={})

    # Audit Fields
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    updated_by = Column(String(100))

    # Relationships
    source_connection = relationship("CnConnection", foreign_keys=[source_connection_id], back_populates="source_flows")
    target_connection = relationship("CnConnection", foreign_keys=[target_connection_id], back_populates="target_flows")
    executions = relationship("CnFlowExecution", back_populates="flow")
    transformation_rules = relationship("CnTransformationRule", back_populates="flow")

    # Indexes
    __table_args__ = (
        Index('idx_cn_flows_tenant_status', 'tenant_id', 'status'),
        Index('idx_cn_flows_source_target', 'source_connection_id', 'target_connection_id'),
        Index('idx_cn_flows_next_execution', 'next_execution'),
        Index('idx_cn_flows_enabled_status', 'enabled', 'status'),
    )

    def __repr__(self):
        return f"<CnDataFlow(name='{self.name}', status='{self.status}', enabled={self.enabled})>"


class CnFlowExecution(Base):
    """
    Individual flow execution tracking for detailed monitoring and debugging.
    """
    __tablename__ = 'cn_flow_executions'

    # Primary Key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Flow Reference
    flow_id = Column(GUIDString(), ForeignKey('cn_data_flows.id'), nullable=False, index=True)
    tenant_id = Column(String(100), index=True)

    # Execution Details
    started_at = Column(DateTime(timezone=True), nullable=False, index=True)
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)

    # Results
    status = Column(SQLEnum(ExecutionStatus), nullable=False, index=True)
    records_processed = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)

    # State and Logs
    initial_state = Column(JSONVariant, default={})
    final_state = Column(JSONVariant, default={})
    execution_logs = Column(Text)
    error_message = Column(Text)

    # Metadata
    meta_data = Column(JSONVariant, default={})

    # Relationships
    flow = relationship("CnDataFlow", back_populates="executions")

    # Indexes
    __table_args__ = (
        Index('idx_cn_executions_flow_started', 'flow_id', 'started_at'),
        Index('idx_cn_executions_status_started', 'status', 'started_at'),
    )

    @validates('status')
    def validate_status(self, key, status):
        if isinstance(status, ExecutionStatus):
            return status
        return ExecutionStatus(status)

    @property
    def execution_details(self):
        return self.meta_data

    @execution_details.setter
    def execution_details(self, value):
        self.meta_data = value

    def get_duration(self):
        if not self.started_at or not self.completed_at:
            return None
        return self.completed_at - self.started_at

    def __repr__(self):
        return f"<CnFlowExecution(flow_id='{self.flow_id}', status='{self.status}', started='{self.started_at}')>"


class CnTransformationRule(Base):
    """
    Transformation rules for data processing within flows.
    """
    __tablename__ = 'cn_transformation_rules'

    # Primary Key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # APG Integration
    tenant_id = Column(String(100), nullable=False, index=True)

    # Basic Information
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Rule Configuration
    rule_type = Column(String(50), nullable=False, index=True)  # field_mapping, data_type_conversion, etc.
    source_field = Column(String(255), nullable=False)
    target_field = Column(String(255), nullable=False)
    transformation_expression = Column(Text)

    # Rule Logic
    rule_config = Column(JSONVariant, default={})
    conditions = Column(JSONVariant, default={})

    # Flow Association
    flow_id = Column(GUIDString(), ForeignKey('cn_data_flows.id'), index=True)
    execution_order = Column(Integer, default=0)

    # Status
    enabled = Column(Boolean, default=True, index=True)

    # Audit Fields
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))

    # Relationships
    flow = relationship("CnDataFlow", back_populates="transformation_rules")

    def __repr__(self):
        return f"<CnTransformationRule(name='{self.name}', type='{self.rule_type}')>"


class CnSingerTap(Base):
    """
    Singer.io tap registry and management.
    """
    __tablename__ = 'cn_singer_taps'

    # Primary Key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Tap Information
    tenant_id = Column(String(100), nullable=False, default='default', index=True)
    name = Column(String(100), nullable=False, index=True)
    package_name = Column(String(100), nullable=False)
    version = Column(String(50))
    description = Column(Text)

    # Installation
    installation_status = Column(String(50), default='available', index=True)  # available, installed, failed
    installation_path = Column(String(500))
    installation_date = Column(DateTime(timezone=True))

    # Configuration Schema
    config_schema = Column(JSONVariant, default={})
    supported_features = Column(JSONVariant, default=[])

    # Documentation
    documentation_url = Column(String(500))
    repository_url = Column(String(500))

    # Metadata
    meta_data = Column(JSONVariant, default={})

    # Audit Fields
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', name='uq_cn_singer_taps_tenant_name'),
    )

    @property
    def configuration_schema(self):
        return self.config_schema

    @configuration_schema.setter
    def configuration_schema(self, value):
        self.config_schema = value

    def __repr__(self):
        return f"<CnSingerTap(name='{self.name}', version='{self.version}', status='{self.installation_status}')>"


class CnSingerTarget(Base):
    """
    Singer.io target registry and management.
    """
    __tablename__ = 'cn_singer_targets'

    # Primary Key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Target Information
    tenant_id = Column(String(100), nullable=False, default='default', index=True)
    name = Column(String(100), nullable=False, index=True)
    package_name = Column(String(100), nullable=False)
    version = Column(String(50))
    description = Column(Text)

    # Installation
    installation_status = Column(String(50), default='available', index=True)
    installation_path = Column(String(500))
    installation_date = Column(DateTime(timezone=True))

    # Configuration Schema
    config_schema = Column(JSONVariant, default={})
    supported_features = Column(JSONVariant, default=[])

    # Documentation
    documentation_url = Column(String(500))
    repository_url = Column(String(500))

    # Metadata
    meta_data = Column(JSONVariant, default={})

    # Audit Fields
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', name='uq_cn_singer_targets_tenant_name'),
    )

    @property
    def configuration_schema(self):
        return self.config_schema

    @configuration_schema.setter
    def configuration_schema(self, value):
        self.config_schema = value

    def __repr__(self):
        return f"<CnSingerTarget(name='{self.name}', version='{self.version}', status='{self.installation_status}')>"


class CnLineageNode(Base):
    """
    Data lineage nodes representing entities in the data flow.
    """
    __tablename__ = 'cn_lineage_nodes'

    # Primary Key
    id = Column(String(255), primary_key=True)

    # APG Integration
    tenant_id = Column(String(100), nullable=False, index=True)

    # Node Information
    name = Column(String(255), nullable=False, index=True)
    node_type = Column(SQLEnum(LineageNodeType), nullable=False, index=True)

    # Associations
    connection_id = Column(String(255), ForeignKey('cn_connections.id'), index=True)
    external_id = Column(String(255), index=True)  # External system ID

    # Node Details
    schema_name = Column(String(100))
    table_name = Column(String(100))
    field_name = Column(String(100))

    # Metadata
    meta_data = Column(JSONVariant, default={})
    properties = Column(JSONVariant, default={})

    # Sensitive Data Classification
    sensitive = Column(Boolean, default=False, index=True)
    pii_classification = Column(String(50), index=True)  # none, low, medium, high

    # Audit Fields
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    connection = relationship("CnConnection", back_populates="lineage_nodes")
    source_edges = relationship("CnLineageEdge", foreign_keys="CnLineageEdge.source_node_id", back_populates="source_node")
    target_edges = relationship("CnLineageEdge", foreign_keys="CnLineageEdge.target_node_id", back_populates="target_node")

    # Indexes
    __table_args__ = (
        Index('idx_cn_lineage_nodes_tenant_type', 'tenant_id', 'node_type'),
        Index('idx_cn_lineage_nodes_connection', 'connection_id', 'node_type'),
        Index('idx_cn_lineage_nodes_sensitive', 'sensitive', 'pii_classification'),
    )

    def __repr__(self):
        return f"<CnLineageNode(name='{self.name}', type='{self.node_type}', sensitive={self.sensitive})>"


class CnLineageEdge(Base):
    """
    Data lineage edges representing relationships between nodes.
    """
    __tablename__ = 'cn_lineage_edges'

    # Primary Key
    id = Column(String(500), primary_key=True)

    # APG Integration
    tenant_id = Column(String(100), nullable=False, index=True)

    # Edge Information
    source_node_id = Column(String(255), ForeignKey('cn_lineage_nodes.id'), nullable=False, index=True)
    target_node_id = Column(String(255), ForeignKey('cn_lineage_nodes.id'), nullable=False, index=True)

    # Relationship Details
    relationship_type = Column(String(50), nullable=False, index=True)  # derives_from, transforms_to, contains, etc.
    transformation_logic = Column(Text)

    # Flow Association
    flow_id = Column(String(255), ForeignKey('cn_data_flows.id'), index=True)

    # Metadata
    meta_data = Column(JSONVariant, default={})
    properties = Column(JSONVariant, default={})

    # Quality and Trust
    confidence_score = Column(Float, default=1.0)  # 0.0 to 1.0

    # Audit Fields
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    source_node = relationship("CnLineageNode", foreign_keys=[source_node_id], back_populates="source_edges")
    target_node = relationship("CnLineageNode", foreign_keys=[target_node_id], back_populates="target_edges")

    # Indexes
    __table_args__ = (
        Index('idx_cn_lineage_edges_tenant', 'tenant_id'),
        Index('idx_cn_lineage_edges_source_target', 'source_node_id', 'target_node_id'),
        Index('idx_cn_lineage_edges_type', 'relationship_type'),
        Index('idx_cn_lineage_edges_flow', 'flow_id'),
    )

    def __repr__(self):
        return f"<CnLineageEdge(source='{self.source_node_id}', target='{self.target_node_id}', type='{self.relationship_type}')>"


class CnHealthCheck(Base):
    """
    Connection health monitoring and metrics.
    """
    __tablename__ = 'cn_health_checks'

    # Primary Key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Connection Reference
    connection_id = Column(GUIDString(), ForeignKey('cn_connections.id'), nullable=False, index=True)
    tenant_id = Column(String(100), index=True)

    # Health Check Details
    check_time = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    status = Column(SQLEnum(ConnectionStatus), nullable=False, index=True)

    # Performance Metrics
    latency_ms = Column(Float)
    throughput_records_per_sec = Column(Float)
    error_rate = Column(Float)
    cpu_usage_percent = Column(Float)
    memory_usage_percent = Column(Float)

    # Detailed Results
    check_results = Column(JSONVariant, default={})
    error_details = Column(Text)

    # Metadata
    meta_data = Column(JSONVariant, default={})

    # Relationships
    connection = relationship("CnConnection", back_populates="health_checks")

    # Indexes
    __table_args__ = (
        Index('idx_cn_health_checks_connection_time', 'connection_id', 'check_time'),
        Index('idx_cn_health_checks_status_time', 'status', 'check_time'),
    )

    @validates('status')
    def validate_status(self, key, status):
        if isinstance(status, ConnectionStatus):
            return status
        status_map = {
            'active': ConnectionStatus.ACTIVE,
            'healthy': ConnectionStatus.ACTIVE,
            'ok': ConnectionStatus.ACTIVE,
            'warning': ConnectionStatus.TESTING,
            'testing': ConnectionStatus.TESTING,
            'inactive': ConnectionStatus.INACTIVE,
            'error': ConnectionStatus.ERROR,
            'failed': ConnectionStatus.ERROR,
        }
        return status_map.get(str(status).lower(), ConnectionStatus.ERROR)

    @property
    def timestamp(self):
        return self.check_time

    @timestamp.setter
    def timestamp(self, value):
        self.check_time = value

    @property
    def cpu_usage(self):
        return self.cpu_usage_percent

    @cpu_usage.setter
    def cpu_usage(self, value):
        self.cpu_usage_percent = value

    @property
    def memory_usage(self):
        return self.memory_usage_percent

    @memory_usage.setter
    def memory_usage(self, value):
        self.memory_usage_percent = value

    @property
    def additional_metrics(self):
        return self.check_results

    @additional_metrics.setter
    def additional_metrics(self, value):
        self.check_results = value

    def is_healthy(self):
        status = self.status.value if hasattr(self.status, 'value') else self.status
        return (
            status in {ConnectionStatus.ACTIVE.value, 'active', 'healthy', 'ok'}
            and (self.error_rate is None or self.error_rate < 0.05)
            and (self.latency_ms is None or self.latency_ms < 3000)
        )

    def __repr__(self):
        return f"<CnHealthCheck(connection_id='{self.connection_id}', status='{self.status}', time='{self.check_time}')>"


# Compatibility aliases used by moved tests and legacy callers.
CnHealthMetric = CnHealthCheck
CnExecutionLog = CnFlowExecution


# Database schema generation helper
def generate_schema_sql():
    """Generate SQL DDL statements for PostgreSQL"""
    from sqlalchemy import create_engine
    from sqlalchemy.schema import CreateTable

    # Create a mock engine for SQL generation
    engine = create_engine('postgresql://user:pass@localhost/db', echo=False)

    sql_statements = []

    # Generate CREATE TABLE statements
    for table in Base.metadata.tables.values():
        create_table_sql = str(CreateTable(table).compile(engine))
        sql_statements.append(create_table_sql)

    return ";\n\n".join(sql_statements) + ";"


if __name__ == "__main__":
    # Generate and print schema SQL
    print("-- APG Connection Management Database Schema")
    print("-- Generated on:", datetime.now())
    print("-- Capability: Connection Management (Cn prefix)")
    print("-- Database: PostgreSQL")
    print()
    print(generate_schema_sql())
