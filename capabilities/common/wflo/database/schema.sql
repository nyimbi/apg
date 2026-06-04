-- ============================================================================
-- APG Workflow Orchestration — PostgreSQL schema
-- © 2025 Datacraft  |  Author: Nyimbi Odero
--
-- Run:  psql $DATABASE_URL < database/schema.sql
--
-- Design principles:
--   • Tenant isolation on every table (row-level security ready)
--   • Soft deletes via is_deleted + deleted_at
--   • Append-only WorkflowHistory for forensics / SLA analysis
--   • Partial indexes for active-record queries
--   • JSONB for flexible metadata / variable payloads
--   • UUID-compatible TEXT primary keys (UUID7 strings)
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fast LIKE / trigram search on names
CREATE EXTENSION IF NOT EXISTS btree_gin; -- composite GIN indexes

-- ---------------------------------------------------------------------------
-- Helper: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION wflo_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


-- ============================================================================
-- 1. wflo_workflow_definition
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_workflow_definition (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    name                    TEXT        NOT NULL,
    description             TEXT        NOT NULL DEFAULT '',
    version                 INT         NOT NULL DEFAULT 1,
    status                  TEXT        NOT NULL DEFAULT 'draft',          -- DefinitionStatus
    trigger_type            TEXT        NOT NULL DEFAULT 'manual',         -- TriggerType
    trigger_config          JSONB       NOT NULL DEFAULT '{}',
    owner_ref               TEXT        NOT NULL,
    category                TEXT        NOT NULL DEFAULT 'general',
    tags                    TEXT[]      NOT NULL DEFAULT '{}',
    process_graph           JSONB       NOT NULL DEFAULT '{}',
    steps                   JSONB       NOT NULL DEFAULT '[]',
    retry_policy_ref        TEXT        NOT NULL DEFAULT '',
    compensation_ref        TEXT        NOT NULL DEFAULT '',
    sla_minutes             INT         NOT NULL DEFAULT 1440,
    max_runtime_minutes     INT         NOT NULL DEFAULT 1440,
    review_required         BOOLEAN     NOT NULL DEFAULT FALSE,
    publish_approval_ref    TEXT        NOT NULL DEFAULT '',
    published_at            TIMESTAMPTZ,
    published_by            TEXT,
    parent_definition_id    TEXT        REFERENCES wflo_workflow_definition(id) ON DELETE SET NULL,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    deleted_at              TIMESTAMPTZ,
    deleted_by              TEXT,

    PRIMARY KEY (id),
    CONSTRAINT wflo_def_name_version_tenant_uq UNIQUE (tenant_id, name, version),
    CONSTRAINT wflo_def_status_ck CHECK (status IN ('draft','review_required','published','deprecated','retired')),
    CONSTRAINT wflo_def_trigger_ck CHECK (trigger_type IN ('manual','scheduled','api','event','webhook','message')),
    CONSTRAINT wflo_def_sla_ck CHECK (sla_minutes >= 1),
    CONSTRAINT wflo_def_max_rt_ck CHECK (max_runtime_minutes >= 1)
);

CREATE INDEX IF NOT EXISTS idx_wflo_def_tenant_status
    ON wflo_workflow_definition (tenant_id, status)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_def_tenant_name
    ON wflo_workflow_definition USING GIN (tenant_id, name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_wflo_def_owner
    ON wflo_workflow_definition (tenant_id, owner_ref)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_def_parent
    ON wflo_workflow_definition (parent_definition_id)
    WHERE parent_definition_id IS NOT NULL;

CREATE TRIGGER wflo_def_updated_at
    BEFORE UPDATE ON wflo_workflow_definition
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 2. wflo_workflow_instance
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_workflow_instance (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    definition_id           TEXT        NOT NULL REFERENCES wflo_workflow_definition(id) ON DELETE RESTRICT,
    definition_version      INT         NOT NULL DEFAULT 1,
    correlation_id          TEXT        NOT NULL DEFAULT '',
    status                  TEXT        NOT NULL DEFAULT 'pending',        -- InstanceStatus
    current_node_id         TEXT,
    input_variables         JSONB       NOT NULL DEFAULT '{}',
    runtime_variables       JSONB       NOT NULL DEFAULT '{}',
    compensation_status     TEXT        NOT NULL DEFAULT 'not_required',   -- CompensationStatus
    compensation_log        JSONB       NOT NULL DEFAULT '[]',
    -- Lifecycle timestamps
    started_at              TIMESTAMPTZ,
    suspended_at            TIMESTAMPTZ,
    resumed_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    failed_at               TIMESTAMPTZ,
    cancelled_at            TIMESTAMPTZ,
    -- SLA
    due_at                  TIMESTAMPTZ,
    sla_breached            BOOLEAN     NOT NULL DEFAULT FALSE,
    -- Error / cancel context
    error_code              TEXT        NOT NULL DEFAULT '',
    error_message           TEXT        NOT NULL DEFAULT '',
    cancel_reason           TEXT        NOT NULL DEFAULT '',
    -- Migration lineage
    migrated_from_version   INT,
    parent_instance_id      TEXT        REFERENCES wflo_workflow_instance(id) ON DELETE SET NULL,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    deleted_at              TIMESTAMPTZ,
    deleted_by              TEXT,

    PRIMARY KEY (id),
    CONSTRAINT wflo_inst_status_ck CHECK (status IN (
        'pending','running','suspended','waiting_timer','waiting_approval',
        'waiting_signal','compensating','completed','failed','cancelled','migrated'
    )),
    CONSTRAINT wflo_inst_comp_ck CHECK (compensation_status IN (
        'not_required','pending','in_progress','completed','failed'
    ))
);

CREATE INDEX IF NOT EXISTS idx_wflo_inst_tenant_status
    ON wflo_workflow_instance (tenant_id, status)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_inst_definition
    ON wflo_workflow_instance (tenant_id, definition_id)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_inst_correlation
    ON wflo_workflow_instance (tenant_id, correlation_id)
    WHERE correlation_id <> '';

CREATE INDEX IF NOT EXISTS idx_wflo_inst_due_at
    ON wflo_workflow_instance (tenant_id, due_at)
    WHERE status NOT IN ('completed','cancelled','failed','migrated') AND is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_inst_sla_breached
    ON wflo_workflow_instance (tenant_id, sla_breached)
    WHERE sla_breached = TRUE AND is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_inst_parent
    ON wflo_workflow_instance (parent_instance_id)
    WHERE parent_instance_id IS NOT NULL;

CREATE TRIGGER wflo_inst_updated_at
    BEFORE UPDATE ON wflo_workflow_instance
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 3. wflo_task  (generic task node)
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_task (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    instance_id             TEXT        NOT NULL REFERENCES wflo_workflow_instance(id) ON DELETE CASCADE,
    definition_id           TEXT        NOT NULL,   -- denormalised for fast queries
    node_id                 TEXT        NOT NULL,
    task_type               TEXT        NOT NULL DEFAULT 'user',           -- TaskType
    status                  TEXT        NOT NULL DEFAULT 'created',        -- TaskStatus
    name                    TEXT        NOT NULL,
    description             TEXT        NOT NULL DEFAULT '',
    -- Assignment
    assignee_ref            TEXT        NOT NULL DEFAULT '',
    candidate_refs          TEXT[]      NOT NULL DEFAULT '{}',
    -- Timing
    ready_at                TIMESTAMPTZ,
    claimed_at              TIMESTAMPTZ,
    started_at              TIMESTAMPTZ,
    due_at                  TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    -- Outcomes
    outcome                 TEXT        NOT NULL DEFAULT '',
    output_variables        JSONB       NOT NULL DEFAULT '{}',
    -- Claim
    claimed_by              TEXT,
    completed_by            TEXT,
    -- Escalation
    escalated               BOOLEAN     NOT NULL DEFAULT FALSE,
    escalation_reason       TEXT        NOT NULL DEFAULT '',
    escalated_at            TIMESTAMPTZ,
    escalated_to            TEXT        NOT NULL DEFAULT '',
    -- Priority (0=lowest, 100=highest)
    priority                INT         NOT NULL DEFAULT 50,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    deleted_at              TIMESTAMPTZ,
    deleted_by              TEXT,

    PRIMARY KEY (id),
    CONSTRAINT wflo_task_status_ck CHECK (status IN (
        'created','ready','claimed','in_progress','completed','escalated','cancelled','timed_out'
    )),
    CONSTRAINT wflo_task_type_ck CHECK (task_type IN (
        'user','service','script','manual','receive','send','business_rule','call_activity'
    )),
    CONSTRAINT wflo_task_priority_ck CHECK (priority BETWEEN 0 AND 100)
);

CREATE INDEX IF NOT EXISTS idx_wflo_task_instance
    ON wflo_task (tenant_id, instance_id)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_task_assignee
    ON wflo_task (tenant_id, assignee_ref, status)
    WHERE is_deleted = FALSE AND status NOT IN ('completed','cancelled','timed_out');

CREATE INDEX IF NOT EXISTS idx_wflo_task_due
    ON wflo_task (tenant_id, due_at)
    WHERE due_at IS NOT NULL AND status NOT IN ('completed','cancelled','timed_out') AND is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_task_escalated
    ON wflo_task (tenant_id, escalated)
    WHERE escalated = TRUE AND is_deleted = FALSE;

CREATE TRIGGER wflo_task_updated_at
    BEFORE UPDATE ON wflo_task
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 4. wflo_user_task  (human-facing task with form schema)
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_user_task (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    task_id                 TEXT        NOT NULL REFERENCES wflo_task(id) ON DELETE CASCADE,
    instance_id             TEXT        NOT NULL REFERENCES wflo_workflow_instance(id) ON DELETE CASCADE,
    node_id                 TEXT        NOT NULL,
    form_schema             JSONB       NOT NULL DEFAULT '{}',
    form_data               JSONB       NOT NULL DEFAULT '{}',
    assignment_strategy     TEXT        NOT NULL DEFAULT 'direct',
    assignee_ref            TEXT        NOT NULL DEFAULT '',
    candidate_groups        TEXT[]      NOT NULL DEFAULT '{}',
    sla_minutes             INT         NOT NULL DEFAULT 480,
    reminder_minutes        INT         NOT NULL DEFAULT 60,
    status                  TEXT        NOT NULL DEFAULT 'created',
    completed_by            TEXT,
    completed_at            TIMESTAMPTZ,
    outcome                 TEXT        NOT NULL DEFAULT '',
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (id),
    CONSTRAINT wflo_utask_sla_ck CHECK (sla_minutes >= 1),
    CONSTRAINT wflo_utask_reminder_ck CHECK (reminder_minutes >= 1)
);

CREATE INDEX IF NOT EXISTS idx_wflo_utask_task
    ON wflo_user_task (task_id);

CREATE INDEX IF NOT EXISTS idx_wflo_utask_assignee
    ON wflo_user_task (tenant_id, assignee_ref)
    WHERE is_deleted = FALSE AND status NOT IN ('completed','cancelled');

CREATE TRIGGER wflo_utask_updated_at
    BEFORE UPDATE ON wflo_user_task
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 5. wflo_service_task  (automated service invocation)
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_service_task (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    task_id                 TEXT        NOT NULL REFERENCES wflo_task(id) ON DELETE CASCADE,
    instance_id             TEXT        NOT NULL REFERENCES wflo_workflow_instance(id) ON DELETE CASCADE,
    node_id                 TEXT        NOT NULL,
    service_ref             TEXT        NOT NULL,
    operation               TEXT        NOT NULL DEFAULT '',
    input_mapping           JSONB       NOT NULL DEFAULT '{}',
    output_mapping          JSONB       NOT NULL DEFAULT '{}',
    retry_count             INT         NOT NULL DEFAULT 0,
    max_retries             INT         NOT NULL DEFAULT 3,
    retry_backoff_seconds   INT         NOT NULL DEFAULT 5,
    timeout_seconds         INT         NOT NULL DEFAULT 30,
    status                  TEXT        NOT NULL DEFAULT 'created',
    last_error              TEXT        NOT NULL DEFAULT '',
    last_response           JSONB       NOT NULL DEFAULT '{}',
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (id),
    CONSTRAINT wflo_stask_retry_ck CHECK (retry_count >= 0 AND max_retries >= 0),
    CONSTRAINT wflo_stask_timeout_ck CHECK (timeout_seconds >= 1)
);

CREATE INDEX IF NOT EXISTS idx_wflo_stask_task
    ON wflo_service_task (task_id);

CREATE INDEX IF NOT EXISTS idx_wflo_stask_instance
    ON wflo_service_task (instance_id);

CREATE TRIGGER wflo_stask_updated_at
    BEFORE UPDATE ON wflo_service_task
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 6. wflo_timer
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_timer (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    instance_id             TEXT        NOT NULL REFERENCES wflo_workflow_instance(id) ON DELETE CASCADE,
    node_id                 TEXT        NOT NULL,
    timer_type              TEXT        NOT NULL DEFAULT 'duration',       -- TimerType
    fire_at                 TIMESTAMPTZ,
    duration_iso            TEXT        NOT NULL DEFAULT '',
    cycle_expression        TEXT        NOT NULL DEFAULT '',
    fired                   BOOLEAN     NOT NULL DEFAULT FALSE,
    fired_at                TIMESTAMPTZ,
    cancelled               BOOLEAN     NOT NULL DEFAULT FALSE,
    cancelled_at            TIMESTAMPTZ,
    fire_count              INT         NOT NULL DEFAULT 0,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (id),
    CONSTRAINT wflo_timer_type_ck CHECK (timer_type IN ('date','duration','cycle'))
);

-- Partial index for pending timers — critical for timer daemon polling
CREATE INDEX IF NOT EXISTS idx_wflo_timer_pending
    ON wflo_timer (tenant_id, fire_at)
    WHERE fired = FALSE AND cancelled = FALSE AND is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_timer_instance
    ON wflo_timer (instance_id)
    WHERE is_deleted = FALSE;

CREATE TRIGGER wflo_timer_updated_at
    BEFORE UPDATE ON wflo_timer
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 7. wflo_gateway
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_gateway (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    instance_id             TEXT        NOT NULL REFERENCES wflo_workflow_instance(id) ON DELETE CASCADE,
    node_id                 TEXT        NOT NULL,
    gateway_type            TEXT        NOT NULL DEFAULT 'exclusive',      -- GatewayType
    conditions              JSONB       NOT NULL DEFAULT '{}',
    incoming_branches       TEXT[]      NOT NULL DEFAULT '{}',
    completed_branches      TEXT[]      NOT NULL DEFAULT '{}',
    selected_paths          TEXT[]      NOT NULL DEFAULT '{}',
    evaluated_at            TIMESTAMPTZ,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (id),
    CONSTRAINT wflo_gw_type_ck CHECK (gateway_type IN (
        'exclusive','parallel','inclusive','event_based','complex'
    ))
);

CREATE INDEX IF NOT EXISTS idx_wflo_gw_instance
    ON wflo_gateway (instance_id)
    WHERE is_deleted = FALSE;

CREATE TRIGGER wflo_gw_updated_at
    BEFORE UPDATE ON wflo_gateway
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 8. wflo_boundary_event
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_boundary_event (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    instance_id             TEXT        NOT NULL REFERENCES wflo_workflow_instance(id) ON DELETE CASCADE,
    attached_to_task_id     TEXT        NOT NULL REFERENCES wflo_task(id) ON DELETE CASCADE,
    node_id                 TEXT        NOT NULL,
    event_type              TEXT        NOT NULL DEFAULT 'timer',          -- BoundaryEventType
    interrupting            BOOLEAN     NOT NULL DEFAULT TRUE,
    trigger_config          JSONB       NOT NULL DEFAULT '{}',
    triggered               BOOLEAN     NOT NULL DEFAULT FALSE,
    triggered_at            TIMESTAMPTZ,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (id),
    CONSTRAINT wflo_be_type_ck CHECK (event_type IN (
        'timer','error','escalation','compensation','signal','message','conditional'
    ))
);

CREATE INDEX IF NOT EXISTS idx_wflo_be_task
    ON wflo_boundary_event (attached_to_task_id)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_be_pending
    ON wflo_boundary_event (tenant_id, triggered)
    WHERE triggered = FALSE AND is_deleted = FALSE;

CREATE TRIGGER wflo_be_updated_at
    BEFORE UPDATE ON wflo_boundary_event
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 9. wflo_escalation
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_escalation (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    instance_id             TEXT        NOT NULL REFERENCES wflo_workflow_instance(id) ON DELETE CASCADE,
    task_id                 TEXT        REFERENCES wflo_task(id) ON DELETE SET NULL,
    escalated_from          TEXT        NOT NULL,
    escalated_to            TEXT        NOT NULL,
    reason                  TEXT        NOT NULL,
    status                  TEXT        NOT NULL DEFAULT 'active',         -- EscalationStatus
    level                   INT         NOT NULL DEFAULT 1,
    escalated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at             TIMESTAMPTZ,
    resolved_by             TEXT,
    resolution_note         TEXT        NOT NULL DEFAULT '',
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (id),
    CONSTRAINT wflo_esc_status_ck CHECK (status IN ('active','resolved','expired')),
    CONSTRAINT wflo_esc_level_ck CHECK (level >= 1)
);

CREATE INDEX IF NOT EXISTS idx_wflo_esc_instance
    ON wflo_escalation (tenant_id, instance_id)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_esc_active
    ON wflo_escalation (tenant_id, status)
    WHERE status = 'active' AND is_deleted = FALSE;

CREATE TRIGGER wflo_esc_updated_at
    BEFORE UPDATE ON wflo_escalation
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 10. wflo_compensation
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_compensation (
    id                          TEXT        NOT NULL,
    tenant_id                   TEXT        NOT NULL,
    instance_id                 TEXT        NOT NULL REFERENCES wflo_workflow_instance(id) ON DELETE CASCADE,
    compensation_node_id        TEXT        NOT NULL,
    compensates_task_id         TEXT        NOT NULL,  -- soft ref, task may be soft-deleted
    status                      TEXT        NOT NULL DEFAULT 'pending',    -- CompensationStatus
    triggered_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at                TIMESTAMPTZ,
    failed_at                   TIMESTAMPTZ,
    error_message               TEXT        NOT NULL DEFAULT '',
    compensation_data           JSONB       NOT NULL DEFAULT '{}',
    metadata                    JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by                  TEXT        NOT NULL DEFAULT 'system',
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (id),
    CONSTRAINT wflo_comp_status_ck CHECK (status IN (
        'not_required','pending','in_progress','completed','failed'
    ))
);

CREATE INDEX IF NOT EXISTS idx_wflo_comp_instance
    ON wflo_compensation (tenant_id, instance_id)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_wflo_comp_pending
    ON wflo_compensation (tenant_id, status)
    WHERE status IN ('pending','in_progress') AND is_deleted = FALSE;

CREATE TRIGGER wflo_comp_updated_at
    BEFORE UPDATE ON wflo_compensation
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 11. wflo_workflow_variable
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_workflow_variable (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    instance_id             TEXT        NOT NULL REFERENCES wflo_workflow_instance(id) ON DELETE CASCADE,
    scope                   TEXT        NOT NULL DEFAULT 'global',         -- VariableScope
    node_id                 TEXT,
    name                    TEXT        NOT NULL,
    value_type              TEXT        NOT NULL DEFAULT 'string',
    value                   JSONB,
    version                 INT         NOT NULL DEFAULT 1,
    mutated_by              TEXT        NOT NULL DEFAULT 'system',
    mutated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (id),
    CONSTRAINT wflo_var_scope_ck CHECK (scope IN ('global','local','process')),
    CONSTRAINT wflo_var_type_ck CHECK (value_type IN ('string','number','boolean','object','array')),
    CONSTRAINT wflo_var_local_needs_node CHECK (
        scope != 'local' OR node_id IS NOT NULL
    ),
    CONSTRAINT wflo_var_unique UNIQUE (instance_id, scope, name, COALESCE(node_id, ''))
);

CREATE INDEX IF NOT EXISTS idx_wflo_var_instance
    ON wflo_workflow_variable (instance_id, scope)
    WHERE is_deleted = FALSE;

CREATE TRIGGER wflo_var_updated_at
    BEFORE UPDATE ON wflo_workflow_variable
    FOR EACH ROW EXECUTE FUNCTION wflo_set_updated_at();


-- ============================================================================
-- 12. wflo_workflow_history  (immutable audit log — never updated)
-- ============================================================================
CREATE TABLE IF NOT EXISTS wflo_workflow_history (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    instance_id             TEXT        NOT NULL,   -- intentionally no FK CASCADE — keep history even if instance soft-deleted
    definition_id           TEXT        NOT NULL,
    event_type              TEXT        NOT NULL,
    node_id                 TEXT,
    task_id                 TEXT,
    actor_id                TEXT        NOT NULL DEFAULT 'system',
    from_status             TEXT,
    to_status               TEXT,
    variable_snapshot       JSONB       NOT NULL DEFAULT '{}',
    summary                 TEXT        NOT NULL DEFAULT '',
    details                 JSONB       NOT NULL DEFAULT '{}',
    sequence_number         INT         NOT NULL DEFAULT 0,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    -- Audit (no updated_at — append-only)
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,

    PRIMARY KEY (id)
);

-- Range partitioning hint: partition by RANGE(created_at) for large deployments
-- CREATE TABLE wflo_workflow_history_2025 PARTITION OF wflo_workflow_history
--     FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE INDEX IF NOT EXISTS idx_wflo_hist_instance
    ON wflo_workflow_history (tenant_id, instance_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_wflo_hist_definition
    ON wflo_workflow_history (tenant_id, definition_id, created_at);

CREATE INDEX IF NOT EXISTS idx_wflo_hist_event_type
    ON wflo_workflow_history (tenant_id, event_type, created_at);

CREATE INDEX IF NOT EXISTS idx_wflo_hist_actor
    ON wflo_workflow_history (tenant_id, actor_id, created_at);

-- GIN index for variable snapshot queries
CREATE INDEX IF NOT EXISTS idx_wflo_hist_snapshot
    ON wflo_workflow_history USING GIN (variable_snapshot);


-- ============================================================================
-- Convenience views
-- ============================================================================

-- Active instances with SLA health
CREATE OR REPLACE VIEW wflo_active_instance_sla AS
SELECT
    i.id,
    i.tenant_id,
    i.definition_id,
    i.status,
    i.due_at,
    i.sla_breached,
    CASE
        WHEN i.due_at IS NULL THEN 'ok'
        WHEN i.due_at < NOW() THEN 'breached'
        WHEN i.due_at - NOW() < INTERVAL '2 hours' THEN 'at_risk'
        ELSE 'ok'
    END AS sla_status,
    EXTRACT(EPOCH FROM (i.due_at - NOW())) / 60.0 AS remaining_minutes
FROM wflo_workflow_instance i
WHERE i.status NOT IN ('completed','cancelled','failed','migrated')
  AND i.is_deleted = FALSE;


-- Open tasks with overdue flag
CREATE OR REPLACE VIEW wflo_open_task_health AS
SELECT
    t.id,
    t.tenant_id,
    t.instance_id,
    t.assignee_ref,
    t.status,
    t.priority,
    t.due_at,
    t.escalated,
    CASE
        WHEN t.due_at IS NOT NULL AND t.due_at < NOW() THEN TRUE
        ELSE FALSE
    END AS overdue,
    EXTRACT(EPOCH FROM (t.due_at - NOW())) / 60.0 AS minutes_until_due
FROM wflo_task t
WHERE t.status NOT IN ('completed','cancelled','timed_out')
  AND t.is_deleted = FALSE;


-- Definition KPIs per tenant
CREATE OR REPLACE VIEW wflo_definition_kpi AS
SELECT
    d.tenant_id,
    COUNT(*) FILTER (WHERE d.status = 'published') AS published_count,
    COUNT(*) FILTER (WHERE d.status = 'draft') AS draft_count,
    COUNT(*) FILTER (WHERE d.status = 'review_required') AS review_count,
    COUNT(*) FILTER (WHERE d.status = 'retired') AS retired_count,
    COUNT(*) AS total_count
FROM wflo_workflow_definition d
WHERE d.is_deleted = FALSE
GROUP BY d.tenant_id;
