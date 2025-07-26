"""
Initial WBPM Schema Creation

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-01-26 12:00:00.000000

APG Workflow & Business Process Management Migration
© 2025 Datacraft. All rights reserved.
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Revision identifiers
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial WBPM schema and tables."""
    
    # Create WBPM schema
    op.execute('CREATE SCHEMA IF NOT EXISTS wbpm')
    
    # Enable required extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_stat_statements"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "btree_gin"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "btree_gist"')
    
    # Create custom types
    op.execute("""
        CREATE TYPE wbpm.process_status AS ENUM (
            'draft', 'published', 'active', 'deprecated', 'archived'
        )
    """)
    
    op.execute("""
        CREATE TYPE wbpm.instance_status AS ENUM (
            'created', 'running', 'suspended', 'completed', 'failed', 'cancelled', 'terminated'
        )
    """)
    
    op.execute("""
        CREATE TYPE wbpm.task_status AS ENUM (
            'created', 'ready', 'reserved', 'in_progress', 'suspended', 'completed', 'failed', 'obsolete', 'exited'
        )
    """)
    
    op.execute("""
        CREATE TYPE wbpm.task_priority AS ENUM (
            'critical', 'high', 'medium', 'low'
        )
    """)
    
    op.execute("""
        CREATE TYPE wbpm.activity_type AS ENUM (
            'start_event', 'end_event', 'intermediate_event', 'user_task', 'service_task', 
            'script_task', 'business_rule_task', 'manual_task', 'receive_task', 'send_task',
            'exclusive_gateway', 'parallel_gateway', 'inclusive_gateway', 'event_gateway',
            'subprocess', 'call_activity'
        )
    """)
    
    op.execute("""
        CREATE TYPE wbpm.gateway_direction AS ENUM (
            'unspecified', 'converging', 'diverging', 'mixed'
        )
    """)
    
    op.execute("""
        CREATE TYPE wbpm.event_type AS ENUM (
            'none', 'message', 'timer', 'error', 'escalation', 'cancel', 'compensation',
            'conditional', 'link', 'signal', 'multiple', 'parallel_multiple', 'terminate'
        )
    """)
    
    op.execute("""
        CREATE TYPE wbpm.collaboration_role AS ENUM (
            'process_owner', 'process_contributor', 'process_reviewer', 'process_observer', 'task_collaborator'
        )
    """)
    
    op.execute("""
        CREATE TYPE wbpm.ai_service_type AS ENUM (
            'process_optimization', 'task_routing', 'bottleneck_detection', 'anomaly_detection',
            'performance_prediction', 'resource_optimization', 'decision_support'
        )
    """)
    
    # Create process_definition table
    op.create_table(
        'process_definition',
        sa.Column('process_id', sa.String(36), primary_key=True, default=sa.text('uuid_generate_v4()::text')),
        sa.Column('tenant_id', sa.String(36), nullable=False, index=True),
        sa.Column('process_key', sa.String(255), nullable=False),
        sa.Column('process_name', sa.String(255), nullable=False),
        sa.Column('process_description', sa.Text),
        sa.Column('process_version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('process_status', postgresql.ENUM('draft', 'published', 'active', 'deprecated', 'archived', name='process_status', schema='wbpm'), nullable=False, default='draft'),
        
        # BPMN Definition
        sa.Column('bpmn_xml', sa.Text, nullable=False),
        sa.Column('bpmn_json', postgresql.JSONB),
        sa.Column('process_variables', postgresql.JSONB, default=sa.text("'[]'::jsonb")),
        
        # Metadata
        sa.Column('category', sa.String(100)),
        sa.Column('tags', postgresql.ARRAY(sa.Text)),
        sa.Column('documentation_url', sa.String(500)),
        
        # Configuration
        sa.Column('is_executable', sa.Boolean, nullable=False, default=True),
        sa.Column('is_suspended', sa.Boolean, nullable=False, default=False),
        sa.Column('suspension_reason', sa.Text),
        
        # Version Control
        sa.Column('parent_version_id', sa.String(36)),
        sa.Column('version_notes', sa.Text),
        sa.Column('deployment_time', sa.TIMESTAMP(timezone=True)),
        
        # APG Standard Fields
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('created_by', sa.String(36), nullable=False),
        sa.Column('updated_by', sa.String(36), nullable=False),
        
        schema='wbpm'
    )
    
    # Create process_instance table
    op.create_table(
        'process_instance',
        sa.Column('instance_id', sa.String(36), primary_key=True, default=sa.text('uuid_generate_v4()::text')),
        sa.Column('tenant_id', sa.String(36), nullable=False, index=True),
        sa.Column('process_id', sa.String(36), nullable=False),
        
        # Instance Details
        sa.Column('business_key', sa.String(255)),
        sa.Column('instance_name', sa.String(255)),
        sa.Column('instance_status', postgresql.ENUM('created', 'running', 'suspended', 'completed', 'failed', 'cancelled', 'terminated', name='instance_status', schema='wbpm'), nullable=False, default='created'),
        
        # Execution Context
        sa.Column('process_variables', postgresql.JSONB, default=sa.text("'{}'::jsonb")),
        sa.Column('current_activities', postgresql.ARRAY(sa.Text)),
        sa.Column('suspended_activities', postgresql.ARRAY(sa.Text)),
        
        # Parent/Child Relationships
        sa.Column('parent_instance_id', sa.String(36)),
        sa.Column('root_instance_id', sa.String(36)),
        sa.Column('call_activity_id', sa.String(36)),
        
        # Timing
        sa.Column('start_time', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('end_time', sa.TIMESTAMP(timezone=True)),
        sa.Column('duration_ms', sa.BigInteger),
        
        # Initiator
        sa.Column('initiated_by', sa.String(36), nullable=False),
        
        # Error Handling
        sa.Column('last_error_message', sa.Text),
        sa.Column('error_count', sa.Integer, default=0),
        sa.Column('retry_count', sa.Integer, default=0),
        
        # Priority and SLA
        sa.Column('priority', postgresql.ENUM('critical', 'high', 'medium', 'low', name='task_priority', schema='wbpm'), default='medium'),
        sa.Column('due_date', sa.TIMESTAMP(timezone=True)),
        
        # APG Standard Fields
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('created_by', sa.String(36), nullable=False),
        sa.Column('updated_by', sa.String(36), nullable=False),
        
        schema='wbpm'
    )
    
    # Create process_activity table
    op.create_table(
        'process_activity',
        sa.Column('activity_id', sa.String(36), primary_key=True, default=sa.text('uuid_generate_v4()::text')),
        sa.Column('tenant_id', sa.String(36), nullable=False, index=True),
        sa.Column('process_id', sa.String(36), nullable=False),
        
        # BPMN Element Details
        sa.Column('element_id', sa.String(255), nullable=False),
        sa.Column('element_name', sa.String(255)),
        sa.Column('activity_type', postgresql.ENUM(
            'start_event', 'end_event', 'intermediate_event', 'user_task', 'service_task', 
            'script_task', 'business_rule_task', 'manual_task', 'receive_task', 'send_task',
            'exclusive_gateway', 'parallel_gateway', 'inclusive_gateway', 'event_gateway',
            'subprocess', 'call_activity', name='activity_type', schema='wbpm'
        ), nullable=False),
        
        # Configuration
        sa.Column('element_properties', postgresql.JSONB, default=sa.text("'{}'::jsonb")),
        sa.Column('execution_listeners', postgresql.JSONB, default=sa.text("'[]'::jsonb")),
        sa.Column('task_listeners', postgresql.JSONB, default=sa.text("'[]'::jsonb")),
        
        # Task-specific properties
        sa.Column('assignee', sa.String(36)),
        sa.Column('candidate_users', postgresql.ARRAY(sa.Text)),
        sa.Column('candidate_groups', postgresql.ARRAY(sa.Text)),
        sa.Column('form_key', sa.String(255)),
        
        # Service Task properties
        sa.Column('class_name', sa.String(255)),
        sa.Column('expression', sa.Text),
        sa.Column('delegate_expression', sa.Text),
        
        # Gateway properties
        sa.Column('gateway_direction', postgresql.ENUM('unspecified', 'converging', 'diverging', 'mixed', name='gateway_direction', schema='wbpm')),
        sa.Column('default_flow', sa.String(255)),
        
        # Event properties
        sa.Column('event_type', postgresql.ENUM(
            'none', 'message', 'timer', 'error', 'escalation', 'cancel', 'compensation',
            'conditional', 'link', 'signal', 'multiple', 'parallel_multiple', 'terminate',
            name='event_type', schema='wbpm'
        )),
        sa.Column('event_definition', postgresql.JSONB),
        
        # Timing and SLA
        sa.Column('due_date_expression', sa.Text),
        sa.Column('follow_up_date_expression', sa.Text),
        
        # APG Standard Fields
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('created_by', sa.String(36), nullable=False),
        sa.Column('updated_by', sa.String(36), nullable=False),
        
        schema='wbpm'
    )
    
    # Create process_flow table
    op.create_table(
        'process_flow',
        sa.Column('flow_id', sa.String(36), primary_key=True, default=sa.text('uuid_generate_v4()::text')),
        sa.Column('tenant_id', sa.String(36), nullable=False, index=True),
        sa.Column('process_id', sa.String(36), nullable=False),
        
        # Flow Definition
        sa.Column('element_id', sa.String(255), nullable=False),
        sa.Column('flow_name', sa.String(255)),
        sa.Column('source_activity_id', sa.String(36), nullable=False),
        sa.Column('target_activity_id', sa.String(36), nullable=False),
        
        # Flow Properties
        sa.Column('condition_expression', sa.Text),
        sa.Column('is_default_flow', sa.Boolean, default=False),
        sa.Column('flow_properties', postgresql.JSONB, default=sa.text("'{}'::jsonb")),
        
        # APG Standard Fields
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('created_by', sa.String(36), nullable=False),
        sa.Column('updated_by', sa.String(36), nullable=False),
        
        schema='wbpm'
    )
    
    # Create task table
    op.create_table(
        'task',
        sa.Column('task_id', sa.String(36), primary_key=True, default=sa.text('uuid_generate_v4()::text')),
        sa.Column('tenant_id', sa.String(36), nullable=False, index=True),
        sa.Column('process_instance_id', sa.String(36), nullable=False),
        sa.Column('activity_id', sa.String(36), nullable=False),
        
        # Task Details
        sa.Column('task_name', sa.String(255), nullable=False),
        sa.Column('task_description', sa.Text),
        sa.Column('task_status', postgresql.ENUM(
            'created', 'ready', 'reserved', 'in_progress', 'suspended', 'completed', 'failed', 'obsolete', 'exited',
            name='task_status', schema='wbpm'
        ), nullable=False, default='created'),
        
        # Assignment
        sa.Column('assignee', sa.String(36)),
        sa.Column('owner', sa.String(36)),
        sa.Column('delegation_state', sa.String(20)),
        
        # Candidate Assignment
        sa.Column('candidate_users', postgresql.ARRAY(sa.Text)),
        sa.Column('candidate_groups', postgresql.ARRAY(sa.Text)),
        
        # Task Data
        sa.Column('form_key', sa.String(255)),
        sa.Column('task_variables', postgresql.JSONB, default=sa.text("'{}'::jsonb")),
        sa.Column('local_variables', postgresql.JSONB, default=sa.text("'{}'::jsonb")),
        
        # Timing
        sa.Column('create_time', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('claim_time', sa.TIMESTAMP(timezone=True)),
        sa.Column('due_date', sa.TIMESTAMP(timezone=True)),
        sa.Column('follow_up_date', sa.TIMESTAMP(timezone=True)),
        sa.Column('completion_time', sa.TIMESTAMP(timezone=True)),
        
        # Priority and Effort
        sa.Column('priority', postgresql.ENUM('critical', 'high', 'medium', 'low', name='task_priority', schema='wbpm'), default='medium'),
        sa.Column('estimated_effort_hours', sa.DECIMAL(10, 2)),
        sa.Column('actual_effort_hours', sa.DECIMAL(10, 2)),
        
        # Parent Task
        sa.Column('parent_task_id', sa.String(36)),
        
        # Suspension
        sa.Column('suspension_state', sa.String(20)),
        sa.Column('suspension_reason', sa.Text),
        
        # APG Standard Fields
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, default=sa.func.now()),
        sa.Column('created_by', sa.String(36), nullable=False),
        sa.Column('updated_by', sa.String(36), nullable=False),
        
        schema='wbpm'
    )
    
    # Create indexes for performance
    
    # Process Definition indexes
    op.create_index('idx_process_definition_tenant', 'process_definition', ['tenant_id'], schema='wbpm')
    op.create_index('idx_process_definition_key', 'process_definition', ['process_key'], schema='wbpm')
    op.create_index('idx_process_definition_status', 'process_definition', ['process_status'], schema='wbpm')
    op.create_index('idx_process_definition_category', 'process_definition', ['category'], schema='wbpm')
    op.create_index('idx_process_definition_created', 'process_definition', ['created_at'], schema='wbpm')
    
    # Process Instance indexes
    op.create_index('idx_process_instance_tenant', 'process_instance', ['tenant_id'], schema='wbpm')
    op.create_index('idx_process_instance_process', 'process_instance', ['process_id'], schema='wbpm')
    op.create_index('idx_process_instance_status', 'process_instance', ['instance_status'], schema='wbpm')
    op.create_index('idx_process_instance_business_key', 'process_instance', ['business_key'], schema='wbpm')
    op.create_index('idx_process_instance_start_time', 'process_instance', ['start_time'], schema='wbpm')
    op.create_index('idx_process_instance_initiated_by', 'process_instance', ['initiated_by'], schema='wbpm')
    op.create_index('idx_process_instance_parent', 'process_instance', ['parent_instance_id'], schema='wbpm')
    
    # Task indexes
    op.create_index('idx_task_tenant', 'task', ['tenant_id'], schema='wbpm')
    op.create_index('idx_task_instance', 'task', ['process_instance_id'], schema='wbpm')
    op.create_index('idx_task_assignee', 'task', ['assignee'], schema='wbpm')
    op.create_index('idx_task_status', 'task', ['task_status'], schema='wbpm')
    op.create_index('idx_task_priority', 'task', ['priority'], schema='wbpm')
    op.create_index('idx_task_due_date', 'task', ['due_date'], schema='wbpm')
    op.create_index('idx_task_create_time', 'task', ['create_time'], schema='wbpm')
    
    # Activity indexes
    op.create_index('idx_activity_tenant', 'process_activity', ['tenant_id'], schema='wbpm')
    op.create_index('idx_activity_process', 'process_activity', ['process_id'], schema='wbpm')
    op.create_index('idx_activity_type', 'process_activity', ['activity_type'], schema='wbpm')
    op.create_index('idx_activity_element_id', 'process_activity', ['element_id'], schema='wbpm')
    
    # Flow indexes
    op.create_index('idx_flow_tenant', 'process_flow', ['tenant_id'], schema='wbpm')
    op.create_index('idx_flow_process', 'process_flow', ['process_id'], schema='wbpm')
    op.create_index('idx_flow_source', 'process_flow', ['source_activity_id'], schema='wbpm')
    op.create_index('idx_flow_target', 'process_flow', ['target_activity_id'], schema='wbpm')
    
    # Add foreign key constraints
    op.create_foreign_key(
        'fk_process_definition_parent', 'process_definition', 'process_definition',
        ['parent_version_id'], ['process_id'], 
        source_schema='wbpm', referent_schema='wbpm',
        ondelete='SET NULL'
    )
    
    op.create_foreign_key(
        'fk_process_instance_definition', 'process_instance', 'process_definition',
        ['process_id'], ['process_id'],
        source_schema='wbpm', referent_schema='wbpm',
        ondelete='RESTRICT'
    )
    
    op.create_foreign_key(
        'fk_process_instance_parent', 'process_instance', 'process_instance',
        ['parent_instance_id'], ['instance_id'],
        source_schema='wbpm', referent_schema='wbpm',
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_activity_process', 'process_activity', 'process_definition',
        ['process_id'], ['process_id'],
        source_schema='wbpm', referent_schema='wbpm',
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_flow_process', 'process_flow', 'process_definition',
        ['process_id'], ['process_id'],
        source_schema='wbmp', referent_schema='wbpm',
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_task_instance', 'task', 'process_instance',
        ['process_instance_id'], ['instance_id'],
        source_schema='wbpm', referent_schema='wbpm',
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_task_activity', 'task', 'process_activity',
        ['activity_id'], ['activity_id'],
        source_schema='wbpm', referent_schema='wbpm',
        ondelete='RESTRICT'
    )
    
    # Add unique constraints
    op.create_unique_constraint(
        'unique_process_key_version', 'process_definition',
        ['tenant_id', 'process_key', 'process_version'],
        schema='wbpm'
    )
    
    op.create_unique_constraint(
        'unique_element_per_process', 'process_activity',
        ['process_id', 'element_id'],
        schema='wbpm'
    )
    
    op.create_unique_constraint(
        'unique_element_per_process_flow', 'process_flow',
        ['process_id', 'element_id'],
        schema='wbpm'
    )


def downgrade() -> None:
    """Drop WBPM schema and all tables."""
    
    # Drop foreign key constraints first
    op.drop_constraint('fk_task_activity', 'task', schema='wbpm', type_='foreignkey')
    op.drop_constraint('fk_task_instance', 'task', schema='wbpm', type_='foreignkey')
    op.drop_constraint('fk_flow_process', 'process_flow', schema='wbpm', type_='foreignkey')
    op.drop_constraint('fk_activity_process', 'process_activity', schema='wbpm', type_='foreignkey')
    op.drop_constraint('fk_process_instance_parent', 'process_instance', schema='wbpm', type_='foreignkey')
    op.drop_constraint('fk_process_instance_definition', 'process_instance', schema='wbpm', type_='foreignkey')
    op.drop_constraint('fk_process_definition_parent', 'process_definition', schema='wbpm', type_='foreignkey')
    
    # Drop tables
    op.drop_table('task', schema='wbpm')
    op.drop_table('process_flow', schema='wbpm')
    op.drop_table('process_activity', schema='wbpm')
    op.drop_table('process_instance', schema='wbpm')
    op.drop_table('process_definition', schema='wbpm')
    
    # Drop custom types
    op.execute('DROP TYPE IF EXISTS wbpm.ai_service_type')
    op.execute('DROP TYPE IF EXISTS wbpm.collaboration_role')
    op.execute('DROP TYPE IF EXISTS wbpm.event_type')
    op.execute('DROP TYPE IF EXISTS wbpm.gateway_direction')
    op.execute('DROP TYPE IF EXISTS wbpm.activity_type')
    op.execute('DROP TYPE IF EXISTS wbpm.task_priority')
    op.execute('DROP TYPE IF EXISTS wbpm.task_status')
    op.execute('DROP TYPE IF EXISTS wbpm.instance_status')
    op.execute('DROP TYPE IF EXISTS wbpm.process_status')
    
    # Drop schema
    op.execute('DROP SCHEMA IF EXISTS wbpm CASCADE')