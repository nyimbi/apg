-- ============================================================================
-- APG Workflow & Business Process Management - Database Schema
-- Multi-Tenant PostgreSQL Schema with Row-Level Security and Performance Optimization
-- 
-- © 2025 Datacraft. All rights reserved.
-- Author: Nyimbi Odero <nyimbi@gmail.com>
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ============================================================================
-- SCHEMA MANAGEMENT
-- ============================================================================

-- Create main workflow schema
CREATE SCHEMA IF NOT EXISTS wbpm;
SET search_path = wbpm, public;

-- Grant usage permissions
GRANT USAGE ON SCHEMA wbpm TO application_role;
GRANT CREATE ON SCHEMA wbpm TO application_role;

-- ============================================================================
-- CORE ENUMERATIONS
-- ============================================================================

-- Process status enumeration
CREATE TYPE wbpm.process_status AS ENUM (
    'draft', 'published', 'active', 'deprecated', 'archived'
);

-- Process instance status enumeration
CREATE TYPE wbpm.instance_status AS ENUM (
    'created', 'running', 'suspended', 'completed', 'failed', 'cancelled', 'terminated'
);

-- Task status enumeration
CREATE TYPE wbpm.task_status AS ENUM (
    'created', 'ready', 'reserved', 'in_progress', 'suspended', 'completed', 'failed', 'obsolete', 'exited'
);

-- Task priority enumeration
CREATE TYPE wbpm.task_priority AS ENUM (
    'critical', 'high', 'medium', 'low'
);

-- Activity type enumeration
CREATE TYPE wbpm.activity_type AS ENUM (
    'start_event', 'end_event', 'intermediate_event', 'user_task', 'service_task', 
    'script_task', 'business_rule_task', 'manual_task', 'receive_task', 'send_task',
    'exclusive_gateway', 'parallel_gateway', 'inclusive_gateway', 'event_gateway',
    'subprocess', 'call_activity'
);

-- Gateway direction enumeration
CREATE TYPE wbpm.gateway_direction AS ENUM (
    'unspecified', 'converging', 'diverging', 'mixed'
);

-- Event type enumeration
CREATE TYPE wbpm.event_type AS ENUM (
    'none', 'message', 'timer', 'error', 'escalation', 'cancel', 'compensation',
    'conditional', 'link', 'signal', 'multiple', 'parallel_multiple', 'terminate'
);

-- Collaboration role enumeration
CREATE TYPE wbpm.collaboration_role AS ENUM (
    'process_owner', 'process_contributor', 'process_reviewer', 'process_observer', 'task_collaborator'
);

-- AI service type enumeration
CREATE TYPE wbpm.ai_service_type AS ENUM (
    'process_optimization', 'task_routing', 'bottleneck_detection', 'anomaly_detection',
    'performance_prediction', 'resource_optimization', 'decision_support'
);

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Process Definitions Table
CREATE TABLE wbpm.process_definition (
    process_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    process_key VARCHAR(255) NOT NULL,
    process_name VARCHAR(255) NOT NULL,
    process_description TEXT,
    process_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    process_status wbpm.process_status NOT NULL DEFAULT 'draft',
    
    -- BPMN Definition
    bpmn_xml TEXT NOT NULL,
    bpmn_json JSONB,
    process_variables JSONB DEFAULT '[]'::jsonb,
    
    -- Metadata
    category VARCHAR(100),
    tags TEXT[],
    documentation_url VARCHAR(500),
    
    -- Configuration
    is_executable BOOLEAN NOT NULL DEFAULT true,
    is_suspended BOOLEAN NOT NULL DEFAULT false,
    suspension_reason TEXT,
    
    -- Version Control
    parent_version_id VARCHAR(36),
    version_notes TEXT,
    deployment_time TIMESTAMP WITH TIME ZONE,
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_process_definition_parent FOREIGN KEY (parent_version_id) 
        REFERENCES wbpm.process_definition(process_id) ON DELETE SET NULL,
    CONSTRAINT unique_process_key_version UNIQUE (tenant_id, process_key, process_version),
    CONSTRAINT check_process_version CHECK (process_version ~ '^[0-9]+\.[0-9]+\.[0-9]+$')
);

-- Process Instances Table
CREATE TABLE wbpm.process_instance (
    instance_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    process_id VARCHAR(36) NOT NULL,
    
    -- Instance Details
    business_key VARCHAR(255),
    instance_name VARCHAR(255),
    instance_status wbpm.instance_status NOT NULL DEFAULT 'created',
    
    -- Execution Context
    process_variables JSONB DEFAULT '{}'::jsonb,
    current_activities TEXT[],
    suspended_activities TEXT[],
    
    -- Parent/Child Relationships
    parent_instance_id VARCHAR(36),
    root_instance_id VARCHAR(36),
    call_activity_id VARCHAR(36),
    
    -- Timing
    start_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms BIGINT,
    
    -- Initiator
    initiated_by VARCHAR(36) NOT NULL,
    
    -- Error Handling
    last_error_message TEXT,
    error_count INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    
    -- Priority and SLA
    priority wbpm.task_priority DEFAULT 'medium',
    due_date TIMESTAMP WITH TIME ZONE,
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_process_instance_definition FOREIGN KEY (process_id) 
        REFERENCES wbpm.process_definition(process_id) ON DELETE RESTRICT,
    CONSTRAINT fk_process_instance_parent FOREIGN KEY (parent_instance_id) 
        REFERENCES wbpm.process_instance(instance_id) ON DELETE CASCADE,
    CONSTRAINT fk_process_instance_root FOREIGN KEY (root_instance_id) 
        REFERENCES wbpm.process_instance(instance_id) ON DELETE CASCADE,
    CONSTRAINT check_end_time_after_start CHECK (end_time IS NULL OR end_time >= start_time),
    CONSTRAINT check_duration_calculation CHECK (
        (end_time IS NULL AND duration_ms IS NULL) OR 
        (end_time IS NOT NULL AND duration_ms = EXTRACT(EPOCH FROM (end_time - start_time)) * 1000)
    )
);

-- Process Activities Table (BPMN Elements)
CREATE TABLE wbpm.process_activity (
    activity_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    process_id VARCHAR(36) NOT NULL,
    
    -- BPMN Element Details
    element_id VARCHAR(255) NOT NULL, -- BPMN element ID
    element_name VARCHAR(255),
    activity_type wbpm.activity_type NOT NULL,
    
    -- Configuration
    element_properties JSONB DEFAULT '{}'::jsonb,
    execution_listeners JSONB DEFAULT '[]'::jsonb,
    task_listeners JSONB DEFAULT '[]'::jsonb,
    
    -- Task-specific properties
    assignee VARCHAR(36),
    candidate_users TEXT[],
    candidate_groups TEXT[],
    form_key VARCHAR(255),
    
    -- Service Task properties
    class_name VARCHAR(255),
    expression TEXT,
    delegate_expression TEXT,
    
    -- Gateway properties
    gateway_direction wbpm.gateway_direction,
    default_flow VARCHAR(255),
    
    -- Event properties
    event_type wbpm.event_type,
    event_definition JSONB,
    
    -- Timing and SLA
    due_date_expression TEXT,
    follow_up_date_expression TEXT,
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_activity_process FOREIGN KEY (process_id) 
        REFERENCES wbpm.process_definition(process_id) ON DELETE CASCADE,
    CONSTRAINT unique_element_per_process UNIQUE (process_id, element_id)
);

-- Process Flows Table (Sequence Flows)
CREATE TABLE wbpm.process_flow (
    flow_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    process_id VARCHAR(36) NOT NULL,
    
    -- Flow Definition
    element_id VARCHAR(255) NOT NULL, -- BPMN element ID
    flow_name VARCHAR(255),
    source_activity_id VARCHAR(36) NOT NULL,
    target_activity_id VARCHAR(36) NOT NULL,
    
    -- Flow Properties
    condition_expression TEXT,
    is_default_flow BOOLEAN DEFAULT false,
    flow_properties JSONB DEFAULT '{}'::jsonb,
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_flow_process FOREIGN KEY (process_id) 
        REFERENCES wbpm.process_definition(process_id) ON DELETE CASCADE,
    CONSTRAINT fk_flow_source FOREIGN KEY (source_activity_id) 
        REFERENCES wbpm.process_activity(activity_id) ON DELETE CASCADE,
    CONSTRAINT fk_flow_target FOREIGN KEY (target_activity_id) 
        REFERENCES wbpm.process_activity(activity_id) ON DELETE CASCADE,
    CONSTRAINT unique_element_per_process_flow UNIQUE (process_id, element_id),
    CONSTRAINT check_no_self_flow CHECK (source_activity_id != target_activity_id)
);

-- ============================================================================
-- TASK MANAGEMENT TABLES
-- ============================================================================

-- Tasks Table
CREATE TABLE wbpm.task (
    task_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    process_instance_id VARCHAR(36) NOT NULL,
    activity_id VARCHAR(36) NOT NULL,
    
    -- Task Details
    task_name VARCHAR(255) NOT NULL,
    task_description TEXT,
    task_status wbpm.task_status NOT NULL DEFAULT 'created',
    
    -- Assignment
    assignee VARCHAR(36),
    owner VARCHAR(36),
    delegation_state VARCHAR(20), -- pending, resolved
    
    -- Candidate Assignment
    candidate_users TEXT[],
    candidate_groups TEXT[],
    
    -- Task Data
    form_key VARCHAR(255),
    task_variables JSONB DEFAULT '{}'::jsonb,
    local_variables JSONB DEFAULT '{}'::jsonb,
    
    -- Timing
    create_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    claim_time TIMESTAMP WITH TIME ZONE,
    due_date TIMESTAMP WITH TIME ZONE,
    follow_up_date TIMESTAMP WITH TIME ZONE,
    completion_time TIMESTAMP WITH TIME ZONE,
    
    -- Priority and Effort
    priority wbpm.task_priority DEFAULT 'medium',
    estimated_effort_hours DECIMAL(10,2),
    actual_effort_hours DECIMAL(10,2),
    
    -- Parent Task (for subtasks)
    parent_task_id VARCHAR(36),
    
    -- Suspension
    suspension_state VARCHAR(20), -- active, suspended
    suspension_reason TEXT,
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_task_instance FOREIGN KEY (process_instance_id) 
        REFERENCES wbpm.process_instance(instance_id) ON DELETE CASCADE,
    CONSTRAINT fk_task_activity FOREIGN KEY (activity_id) 
        REFERENCES wbpm.process_activity(activity_id) ON DELETE RESTRICT,
    CONSTRAINT fk_task_parent FOREIGN KEY (parent_task_id) 
        REFERENCES wbpm.task(task_id) ON DELETE CASCADE,
    CONSTRAINT check_completion_time CHECK (
        (task_status != 'completed' AND completion_time IS NULL) OR
        (task_status = 'completed' AND completion_time IS NOT NULL)
    ),
    CONSTRAINT check_claim_time CHECK (
        (assignee IS NULL AND claim_time IS NULL) OR
        (assignee IS NOT NULL)
    ),
    CONSTRAINT check_effort_hours CHECK (
        estimated_effort_hours IS NULL OR estimated_effort_hours >= 0
    ),
    CONSTRAINT check_actual_effort_hours CHECK (
        actual_effort_hours IS NULL OR actual_effort_hours >= 0
    )
);

-- Task History Table (for audit trail)
CREATE TABLE wbpm.task_history (
    history_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    task_id VARCHAR(36) NOT NULL,
    
    -- History Details
    action_type VARCHAR(50) NOT NULL, -- created, assigned, completed, etc.
    action_description TEXT,
    
    -- State Changes
    old_value JSONB,
    new_value JSONB,
    
    -- Actor
    performed_by VARCHAR(36) NOT NULL,
    performed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Context
    action_context JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT fk_task_history_task FOREIGN KEY (task_id) 
        REFERENCES wbpm.task(task_id) ON DELETE CASCADE
);

-- Task Comments Table
CREATE TABLE wbpm.task_comment (
    comment_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    task_id VARCHAR(36) NOT NULL,
    
    -- Comment Details
    comment_text TEXT NOT NULL,
    comment_type VARCHAR(50) DEFAULT 'user', -- user, system, audit
    
    -- Reply Thread
    parent_comment_id VARCHAR(36),
    thread_level INTEGER DEFAULT 0,
    
    -- Attachments
    attachments JSONB DEFAULT '[]'::jsonb,
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_comment_task FOREIGN KEY (task_id) 
        REFERENCES wbpm.task(task_id) ON DELETE CASCADE,
    CONSTRAINT fk_comment_parent FOREIGN KEY (parent_comment_id) 
        REFERENCES wbpm.task_comment(comment_id) ON DELETE CASCADE,
    CONSTRAINT check_thread_level CHECK (thread_level >= 0 AND thread_level <= 10)
);

-- ============================================================================
-- PROCESS TEMPLATES AND COLLABORATION
-- ============================================================================

-- Process Templates Table
CREATE TABLE wbpm.process_template (
    template_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    
    -- Template Details
    template_name VARCHAR(255) NOT NULL,
    template_description TEXT,
    template_category VARCHAR(100),
    template_tags TEXT[],
    
    -- Template Content
    bpmn_template TEXT NOT NULL,
    template_variables JSONB DEFAULT '[]'::jsonb,
    configuration_schema JSONB DEFAULT '{}'::jsonb,
    
    -- Usage and Sharing
    is_public BOOLEAN DEFAULT false,
    usage_count INTEGER DEFAULT 0,
    rating_average DECIMAL(3,2) DEFAULT 0.0,
    rating_count INTEGER DEFAULT 0,
    
    -- Versioning
    template_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    parent_template_id VARCHAR(36),
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_template_parent FOREIGN KEY (parent_template_id) 
        REFERENCES wbpm.process_template(template_id) ON DELETE SET NULL,
    CONSTRAINT unique_template_name_version UNIQUE (tenant_id, template_name, template_version),
    CONSTRAINT check_rating_average CHECK (rating_average >= 0.0 AND rating_average <= 5.0),
    CONSTRAINT check_rating_count CHECK (rating_count >= 0)
);

-- Collaboration Sessions Table
CREATE TABLE wbpm.collaboration_session (
    session_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    
    -- Session Details
    session_name VARCHAR(255) NOT NULL,
    session_type VARCHAR(50) NOT NULL, -- design, execution, review, analysis
    target_process_id VARCHAR(36),
    target_instance_id VARCHAR(36),
    
    -- Session Configuration
    max_participants INTEGER DEFAULT 10,
    session_duration_minutes INTEGER,
    conflict_resolution_mode VARCHAR(50) DEFAULT 'last_writer_wins',
    
    -- Session State
    session_status VARCHAR(20) DEFAULT 'active', -- active, paused, completed
    start_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    
    -- Host Information
    session_host VARCHAR(36) NOT NULL,
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_session_process FOREIGN KEY (target_process_id) 
        REFERENCES wbpm.process_definition(process_id) ON DELETE SET NULL,
    CONSTRAINT fk_session_instance FOREIGN KEY (target_instance_id) 
        REFERENCES wbpm.process_instance(instance_id) ON DELETE SET NULL,
    CONSTRAINT check_session_duration CHECK (session_duration_minutes IS NULL OR session_duration_minutes > 0),
    CONSTRAINT check_max_participants CHECK (max_participants > 0 AND max_participants <= 100)
);

-- Collaboration Participants Table
CREATE TABLE wbpm.collaboration_participant (
    participant_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    session_id VARCHAR(36) NOT NULL,
    
    -- Participant Details
    user_id VARCHAR(36) NOT NULL,
    participant_name VARCHAR(255) NOT NULL,
    collaboration_role wbpm.collaboration_role NOT NULL,
    
    -- Participation State
    join_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    leave_time TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Visual Presence
    cursor_position JSONB,
    selected_elements TEXT[],
    participant_color VARCHAR(7), -- hex color code
    
    -- Permissions
    permissions TEXT[],
    
    -- Constraints
    CONSTRAINT fk_participant_session FOREIGN KEY (session_id) 
        REFERENCES wbpm.collaboration_session(session_id) ON DELETE CASCADE,
    CONSTRAINT unique_user_session UNIQUE (session_id, user_id),
    CONSTRAINT check_participant_color CHECK (participant_color ~ '^#[0-9A-Fa-f]{6}$')
);

-- ============================================================================
-- PROCESS ANALYTICS AND MONITORING
-- ============================================================================

-- Process Metrics Table
CREATE TABLE wbpm.process_metrics (
    metrics_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    process_id VARCHAR(36),
    instance_id VARCHAR(36),
    
    -- Metric Details
    metric_type VARCHAR(50) NOT NULL, -- cycle_time, wait_time, processing_time, cost, etc.
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50), -- seconds, minutes, hours, currency, etc.
    
    -- Context
    measurement_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    measurement_context JSONB DEFAULT '{}'::jsonb,
    
    -- Aggregation
    aggregation_period VARCHAR(50), -- hour, day, week, month
    aggregation_level VARCHAR(50), -- instance, process, tenant, global
    
    -- Constraints
    CONSTRAINT fk_metrics_process FOREIGN KEY (process_id) 
        REFERENCES wbpm.process_definition(process_id) ON DELETE CASCADE,
    CONSTRAINT fk_metrics_instance FOREIGN KEY (instance_id) 
        REFERENCES wbpm.process_instance(instance_id) ON DELETE CASCADE
);

-- Process Bottlenecks Table
CREATE TABLE wbpm.process_bottleneck (
    bottleneck_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    process_id VARCHAR(36) NOT NULL,
    
    -- Bottleneck Details
    bottleneck_activity VARCHAR(255) NOT NULL,
    bottleneck_type VARCHAR(50) NOT NULL, -- resource, time, queue, system
    severity VARCHAR(20) NOT NULL, -- critical, high, medium, low
    
    -- Impact Analysis
    impact_score DECIMAL(5,2) NOT NULL, -- 0.00 to 100.00
    affected_instances INTEGER DEFAULT 0,
    average_delay_minutes DECIMAL(10,2),
    
    -- Detection Details
    detection_method VARCHAR(50), -- statistical, ml, rule_based
    detection_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    confidence_score DECIMAL(3,2), -- 0.00 to 1.00
    
    -- Resolution
    recommendation TEXT,
    resolution_status VARCHAR(20) DEFAULT 'open', -- open, in_progress, resolved, dismissed
    resolution_notes TEXT,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(36),
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_bottleneck_process FOREIGN KEY (process_id) 
        REFERENCES wbpm.process_definition(process_id) ON DELETE CASCADE,
    CONSTRAINT check_impact_score CHECK (impact_score >= 0.0 AND impact_score <= 100.0),
    CONSTRAINT check_confidence_score CHECK (confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0))
);

-- ============================================================================
-- AI AND AUTOMATION TABLES
-- ============================================================================

-- AI Recommendations Table
CREATE TABLE wbpm.ai_recommendation (
    recommendation_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    
    -- Recommendation Details
    recommendation_type wbpm.ai_service_type NOT NULL,
    target_process_id VARCHAR(36),
    target_instance_id VARCHAR(36),
    
    -- Content
    recommendation_title VARCHAR(255) NOT NULL,
    recommendation_description TEXT NOT NULL,
    implementation_instructions TEXT,
    
    -- Analysis
    confidence_score DECIMAL(3,2) NOT NULL, -- 0.00 to 1.00
    impact_assessment JSONB DEFAULT '{}'::jsonb,
    implementation_effort VARCHAR(20), -- low, medium, high
    expected_benefit JSONB DEFAULT '{}'::jsonb,
    
    -- Lifecycle
    recommendation_status VARCHAR(20) DEFAULT 'pending', -- pending, accepted, rejected, implemented
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewed_by VARCHAR(36),
    review_notes TEXT,
    
    -- Implementation
    implementation_date TIMESTAMP WITH TIME ZONE,
    implementation_notes TEXT,
    success_metrics JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT fk_recommendation_process FOREIGN KEY (target_process_id) 
        REFERENCES wbpm.process_definition(process_id) ON DELETE CASCADE,
    CONSTRAINT fk_recommendation_instance FOREIGN KEY (target_instance_id) 
        REFERENCES wbpm.process_instance(instance_id) ON DELETE CASCADE,
    CONSTRAINT check_confidence_score_ai CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
);

-- Process Rules Table
CREATE TABLE wbpm.process_rule (
    rule_id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    tenant_id VARCHAR(36) NOT NULL,
    process_id VARCHAR(36),
    
    -- Rule Definition
    rule_name VARCHAR(255) NOT NULL,
    rule_description TEXT,
    rule_type VARCHAR(50) NOT NULL, -- business_rule, validation_rule, routing_rule, escalation_rule
    
    -- Rule Logic
    rule_condition TEXT NOT NULL, -- expression or condition
    rule_action TEXT NOT NULL, -- action to take when condition is met
    rule_priority INTEGER DEFAULT 100,
    
    -- Rule Context
    applies_to_activities TEXT[], -- specific activities this rule applies to
    rule_context JSONB DEFAULT '{}'::jsonb,
    
    -- Rule State
    is_active BOOLEAN DEFAULT true,
    activation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deactivation_date TIMESTAMP WITH TIME ZONE,
    
    -- Performance
    execution_count INTEGER DEFAULT 0,
    last_execution TIMESTAMP WITH TIME ZONE,
    average_execution_time_ms DECIMAL(10,3),
    
    -- APG Standard Fields
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(36) NOT NULL,
    updated_by VARCHAR(36) NOT NULL,
    
    -- Constraints
    CONSTRAINT fk_rule_process FOREIGN KEY (process_id) 
        REFERENCES wbpm.process_definition(process_id) ON DELETE CASCADE,
    CONSTRAINT check_rule_priority CHECK (rule_priority >= 1 AND rule_priority <= 1000),
    CONSTRAINT check_activation_dates CHECK (
        deactivation_date IS NULL OR deactivation_date > activation_date
    )
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Process Definition Indexes
CREATE INDEX idx_process_definition_tenant ON wbpm.process_definition(tenant_id);
CREATE INDEX idx_process_definition_key ON wbpm.process_definition(process_key);
CREATE INDEX idx_process_definition_status ON wbpm.process_definition(process_status);
CREATE INDEX idx_process_definition_category ON wbpm.process_definition(category);
CREATE INDEX idx_process_definition_created ON wbpm.process_definition(created_at);
CREATE INDEX idx_process_definition_tags ON wbpm.process_definition USING GIN(tags);

-- Process Instance Indexes
CREATE INDEX idx_process_instance_tenant ON wbpm.process_instance(tenant_id);
CREATE INDEX idx_process_instance_process ON wbpm.process_instance(process_id);
CREATE INDEX idx_process_instance_status ON wbpm.process_instance(instance_status);
CREATE INDEX idx_process_instance_business_key ON wbpm.process_instance(business_key);
CREATE INDEX idx_process_instance_start_time ON wbpm.process_instance(start_time);
CREATE INDEX idx_process_instance_initiated_by ON wbpm.process_instance(initiated_by);
CREATE INDEX idx_process_instance_parent ON wbpm.process_instance(parent_instance_id);
CREATE INDEX idx_process_instance_variables ON wbpm.process_instance USING GIN(process_variables);

-- Task Indexes
CREATE INDEX idx_task_tenant ON wbpm.task(tenant_id);
CREATE INDEX idx_task_instance ON wbpm.task(process_instance_id);
CREATE INDEX idx_task_assignee ON wbpm.task(assignee);
CREATE INDEX idx_task_status ON wbpm.task(task_status);
CREATE INDEX idx_task_priority ON wbpm.task(priority);
CREATE INDEX idx_task_due_date ON wbpm.task(due_date);
CREATE INDEX idx_task_create_time ON wbpm.task(create_time);
CREATE INDEX idx_task_candidate_users ON wbpm.task USING GIN(candidate_users);
CREATE INDEX idx_task_candidate_groups ON wbpm.task USING GIN(candidate_groups);
CREATE INDEX idx_task_variables ON wbpm.task USING GIN(task_variables);

-- Activity Indexes
CREATE INDEX idx_activity_tenant ON wbpm.process_activity(tenant_id);
CREATE INDEX idx_activity_process ON wbpm.process_activity(process_id);
CREATE INDEX idx_activity_type ON wbpm.process_activity(activity_type);
CREATE INDEX idx_activity_assignee ON wbpm.process_activity(assignee);
CREATE INDEX idx_activity_element_id ON wbpm.process_activity(element_id);

-- Flow Indexes
CREATE INDEX idx_flow_tenant ON wbpm.process_flow(tenant_id);
CREATE INDEX idx_flow_process ON wbpm.process_flow(process_id);
CREATE INDEX idx_flow_source ON wbpm.process_flow(source_activity_id);
CREATE INDEX idx_flow_target ON wbpm.process_flow(target_activity_id);

-- Metrics Indexes
CREATE INDEX idx_metrics_tenant ON wbpm.process_metrics(tenant_id);
CREATE INDEX idx_metrics_process ON wbpm.process_metrics(process_id);
CREATE INDEX idx_metrics_instance ON wbpm.process_metrics(instance_id);
CREATE INDEX idx_metrics_type ON wbpm.process_metrics(metric_type);
CREATE INDEX idx_metrics_timestamp ON wbpm.process_metrics(measurement_timestamp);

-- Collaboration Indexes
CREATE INDEX idx_collaboration_session_tenant ON wbpm.collaboration_session(tenant_id);
CREATE INDEX idx_collaboration_session_process ON wbpm.collaboration_session(target_process_id);
CREATE INDEX idx_collaboration_session_status ON wbpm.collaboration_session(session_status);
CREATE INDEX idx_collaboration_participant_session ON wbpm.collaboration_participant(session_id);
CREATE INDEX idx_collaboration_participant_user ON wbpm.collaboration_participant(user_id);

-- ============================================================================
-- PARTITIONING FOR LARGE TABLES
-- ============================================================================

-- Partition process instances by date (monthly partitions)
-- Note: This would be implemented based on specific requirements and data volume

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE wbpm.process_definition ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.process_instance ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.process_activity ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.process_flow ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.task ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.task_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.task_comment ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.process_template ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.collaboration_session ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.collaboration_participant ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.process_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.process_bottleneck ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.ai_recommendation ENABLE ROW LEVEL SECURITY;
ALTER TABLE wbpm.process_rule ENABLE ROW LEVEL SECURITY;

-- Create tenant isolation policies
CREATE POLICY tenant_isolation_process_definition ON wbpm.process_definition
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant'));

CREATE POLICY tenant_isolation_process_instance ON wbpm.process_instance
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant'));

CREATE POLICY tenant_isolation_task ON wbpm.task
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant'));

-- (Additional policies would be created for all other tables following the same pattern)

-- ============================================================================
-- TRIGGERS AND FUNCTIONS
-- ============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION wbpm.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers to all main tables
CREATE TRIGGER trigger_update_process_definition_updated_at
    BEFORE UPDATE ON wbpm.process_definition
    FOR EACH ROW EXECUTE FUNCTION wbpm.update_updated_at_column();

CREATE TRIGGER trigger_update_process_instance_updated_at
    BEFORE UPDATE ON wbpm.process_instance
    FOR EACH ROW EXECUTE FUNCTION wbpm.update_updated_at_column();

CREATE TRIGGER trigger_update_task_updated_at
    BEFORE UPDATE ON wbpm.task
    FOR EACH ROW EXECUTE FUNCTION wbpm.update_updated_at_column();

-- Function to calculate process instance duration
CREATE OR REPLACE FUNCTION wbpm.calculate_instance_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.end_time IS NOT NULL AND OLD.end_time IS NULL THEN
        NEW.duration_ms := EXTRACT(EPOCH FROM (NEW.end_time - NEW.start_time)) * 1000;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_calculate_instance_duration
    BEFORE UPDATE ON wbpm.process_instance
    FOR EACH ROW EXECUTE FUNCTION wbpm.calculate_instance_duration();

-- Function to automatically create task history entries
CREATE OR REPLACE FUNCTION wbpm.create_task_history()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO wbpm.task_history (
            tenant_id, task_id, action_type, action_description,
            new_value, performed_by, performed_at
        ) VALUES (
            NEW.tenant_id, NEW.task_id, 'created', 'Task created',
            to_jsonb(NEW), NEW.created_by, NEW.created_at
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.task_status != NEW.task_status THEN
            INSERT INTO wbpm.task_history (
                tenant_id, task_id, action_type, action_description,
                old_value, new_value, performed_by, performed_at
            ) VALUES (
                NEW.tenant_id, NEW.task_id, 'status_changed', 
                'Task status changed from ' || OLD.task_status || ' to ' || NEW.task_status,
                jsonb_build_object('status', OLD.task_status),
                jsonb_build_object('status', NEW.task_status),
                NEW.updated_by, NEW.updated_at
            );
        END IF;
        
        IF OLD.assignee IS DISTINCT FROM NEW.assignee THEN
            INSERT INTO wbpm.task_history (
                tenant_id, task_id, action_type, action_description,
                old_value, new_value, performed_by, performed_at
            ) VALUES (
                NEW.tenant_id, NEW.task_id, 'assignment_changed', 
                'Task assignment changed',
                jsonb_build_object('assignee', OLD.assignee),
                jsonb_build_object('assignee', NEW.assignee),
                NEW.updated_by, NEW.updated_at
            );
        END IF;
        
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_create_task_history
    AFTER INSERT OR UPDATE ON wbpm.task
    FOR EACH ROW EXECUTE FUNCTION wbpm.create_task_history();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Active Process Instances View
CREATE VIEW wbpm.v_active_process_instances AS
SELECT 
    pi.instance_id,
    pi.tenant_id,
    pi.business_key,
    pi.instance_name,
    pi.instance_status,
    pi.start_time,
    pi.duration_ms,
    pd.process_name,
    pd.process_version,
    pd.category,
    pi.initiated_by,
    COUNT(t.task_id) as active_task_count
FROM wbpm.process_instance pi
JOIN wbpm.process_definition pd ON pi.process_id = pd.process_id
LEFT JOIN wbpm.task t ON pi.instance_id = t.process_instance_id 
    AND t.task_status IN ('created', 'ready', 'reserved', 'in_progress')
WHERE pi.instance_status IN ('created', 'running', 'suspended')
GROUP BY pi.instance_id, pd.process_id;

-- Task Queue View
CREATE VIEW wbpm.v_task_queue AS
SELECT 
    t.task_id,
    t.tenant_id,
    t.task_name,
    t.task_status,
    t.priority,
    t.assignee,
    t.due_date,
    t.create_time,
    pi.instance_name,
    pi.business_key,
    pd.process_name,
    CASE 
        WHEN t.due_date < NOW() THEN true
        ELSE false
    END as is_overdue,
    EXTRACT(EPOCH FROM (NOW() - t.create_time))/3600 as age_hours
FROM wbpm.task t
JOIN wbpm.process_instance pi ON t.process_instance_id = pi.instance_id
JOIN wbpm.process_definition pd ON pi.process_id = pd.process_id
WHERE t.task_status IN ('created', 'ready', 'reserved', 'in_progress');

-- Process Performance Metrics View
CREATE VIEW wbpm.v_process_performance AS
SELECT 
    pd.process_id,
    pd.tenant_id,
    pd.process_name,
    pd.process_version,
    COUNT(pi.instance_id) as total_instances,
    COUNT(CASE WHEN pi.instance_status = 'completed' THEN 1 END) as completed_instances,
    COUNT(CASE WHEN pi.instance_status = 'failed' THEN 1 END) as failed_instances,
    COUNT(CASE WHEN pi.instance_status = 'running' THEN 1 END) as running_instances,
    AVG(pi.duration_ms) as avg_duration_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pi.duration_ms) as median_duration_ms,
    MIN(pi.start_time) as first_execution,
    MAX(pi.start_time) as last_execution
FROM wbpm.process_definition pd
LEFT JOIN wbpm.process_instance pi ON pd.process_id = pi.process_id
GROUP BY pd.process_id, pd.process_name, pd.process_version, pd.tenant_id;

-- ============================================================================
-- SAMPLE DATA AND TESTING
-- ============================================================================

-- Insert sample tenant for testing
-- INSERT INTO wbpm.process_definition (tenant_id, process_key, process_name, bpmn_xml, created_by, updated_by)
-- VALUES ('test_tenant_001', 'simple_approval', 'Simple Approval Process', '<bpmn:definitions>...</bpmn:definitions>', 'system', 'system');

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Grant permissions to application role
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA wbpm TO application_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA wbpm TO application_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA wbpm TO application_role;

-- ============================================================================
-- MAINTENANCE AND MONITORING
-- ============================================================================

-- Create indexes for monitoring and maintenance
CREATE INDEX idx_process_instance_performance ON wbpm.process_instance(start_time, end_time, duration_ms);
CREATE INDEX idx_task_performance ON wbpm.task(create_time, completion_time, priority);
CREATE INDEX idx_metrics_aggregation ON wbpm.process_metrics(aggregation_period, aggregation_level, measurement_timestamp);

-- Statistics collection for query optimization
SELECT 'Schema creation completed successfully. Remember to run ANALYZE on all tables after data population.' as status;

-- END OF SCHEMA