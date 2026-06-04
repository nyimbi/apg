-- APG Audit Logging — PostgreSQL schema
-- © 2025 Datacraft  www.datacraft.co.ke
--
-- Run:  psql $DATABASE_URL -f database/schema.sql
--
-- Prefix convention: AL_ = AuditLogging tables
-- All tables carry: id, tenant_id, created_at, updated_at, created_by, is_deleted
-- tenant_id is included in every index and enforced by the application service.
-- Legal-hold and retention enforcement happen at the application layer.
--
-- Partitioning: al_events is LIST-partitioned by tenant_id.
-- For high-cardinality tenants, sub-partition by RANGE on created_at.
-- ============================================================

-- ------------------------------------------------------------
-- Extensions
-- ------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";     -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "pg_trgm";      -- trigram full-text search

-- ============================================================
-- AL_EVENTS — immutable audit event log
-- ============================================================
CREATE TABLE IF NOT EXISTS al_events (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT,
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,

    -- Classification
    level               TEXT        NOT NULL,          -- AuditLevel
    event_type          TEXT        NOT NULL,          -- AuditEventType
    source              TEXT        NOT NULL,          -- EventSource
    category            TEXT        NOT NULL,
    subcategory         TEXT,

    -- Who
    actor_id            TEXT,
    actor_type          TEXT        NOT NULL DEFAULT 'user',
    actor_display_name  TEXT,
    session_id          TEXT,
    service_account     TEXT,

    -- What
    action              TEXT        NOT NULL,
    action_description  TEXT,
    operation_id        TEXT,

    -- On what
    resource_type       TEXT,
    resource_id         TEXT,
    resource_name       TEXT,
    resource_path       TEXT,
    parent_resource_id  TEXT,

    -- Where / how
    ip_address          TEXT,
    user_agent          TEXT,
    geographic_location TEXT,
    device_id           TEXT,
    request_id          TEXT,
    correlation_id      TEXT,

    -- Outcome
    success             BOOLEAN     NOT NULL DEFAULT TRUE,
    status_code         INTEGER,
    error_code          TEXT,
    error_message       TEXT,
    duration_ms         INTEGER,

    -- Risk / anomaly
    risk_score          NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    anomaly_score       NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    threat_indicators   TEXT[]       NOT NULL DEFAULT '{}',
    behavioral_tags     TEXT[]       NOT NULL DEFAULT '{}',

    -- Compliance
    compliance_tags     TEXT[]       NOT NULL DEFAULT '{}',
    data_classification TEXT,
    retention_days      INTEGER      NOT NULL DEFAULT 2555,
    legal_hold          BOOLEAN      NOT NULL DEFAULT FALSE,
    contains_pii        BOOLEAN      NOT NULL DEFAULT FALSE,

    -- Freeform
    details             JSONB        NOT NULL DEFAULT '{}',
    tags                JSONB        NOT NULL DEFAULT '{}',

    -- Integrity
    checksum            TEXT,
    chain_hash          TEXT,
    immutable           BOOLEAN      NOT NULL DEFAULT TRUE,

    PRIMARY KEY (tenant_id, id)
) PARTITION BY LIST (tenant_id);

-- Default partition for unrouted tenants
CREATE TABLE IF NOT EXISTS al_events_default
    PARTITION OF al_events DEFAULT;

-- Indexes on al_events
CREATE INDEX IF NOT EXISTS ix_al_events_created_at
    ON al_events (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_al_events_actor
    ON al_events (tenant_id, actor_id);

CREATE INDEX IF NOT EXISTS ix_al_events_resource
    ON al_events (tenant_id, resource_id);

CREATE INDEX IF NOT EXISTS ix_al_events_event_type
    ON al_events (tenant_id, event_type);

CREATE INDEX IF NOT EXISTS ix_al_events_risk_score
    ON al_events (tenant_id, risk_score DESC);

CREATE INDEX IF NOT EXISTS ix_al_events_legal_hold
    ON al_events (tenant_id, legal_hold)
    WHERE legal_hold = TRUE;

CREATE INDEX IF NOT EXISTS ix_al_events_correlation
    ON al_events (tenant_id, correlation_id)
    WHERE correlation_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_al_events_details_gin
    ON al_events USING gin (details);

CREATE INDEX IF NOT EXISTS ix_al_events_action_trgm
    ON al_events USING gin (action gin_trgm_ops);

-- ============================================================
-- AL_TRAILS — named audit trails grouping related events
-- ============================================================
CREATE TABLE IF NOT EXISTS al_trails (
    id          TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by  TEXT,
    is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,

    name        TEXT        NOT NULL,
    description TEXT,
    status      TEXT        NOT NULL DEFAULT 'active',   -- TrailStatus
    event_count INTEGER     NOT NULL DEFAULT 0,
    tags        JSONB        NOT NULL DEFAULT '{}',
    closed_at   TIMESTAMPTZ,
    closed_by   TEXT,

    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS ix_al_trails_tenant_status
    ON al_trails (tenant_id, status);

-- Junction table: trail ↔ event (many-to-many)
CREATE TABLE IF NOT EXISTS al_trail_events (
    tenant_id   TEXT        NOT NULL,
    trail_id    TEXT        NOT NULL,
    event_id    TEXT        NOT NULL,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    added_by    TEXT,
    PRIMARY KEY (tenant_id, trail_id, event_id),
    FOREIGN KEY (tenant_id, trail_id)
        REFERENCES al_trails (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, event_id)
        REFERENCES al_events (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_al_trail_events_event
    ON al_trail_events (tenant_id, event_id);

-- ============================================================
-- AL_COMPLIANCE_REPORTS
-- ============================================================
CREATE TABLE IF NOT EXISTS al_compliance_reports (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by              TEXT,
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,

    framework               TEXT        NOT NULL,   -- ComplianceFramework
    period_start            TIMESTAMPTZ NOT NULL,
    period_end              TIMESTAMPTZ NOT NULL,
    requested_by            TEXT        NOT NULL,
    status                  TEXT        NOT NULL DEFAULT 'pending',  -- ReportStatus
    include_violations      BOOLEAN     NOT NULL DEFAULT TRUE,
    include_recommendations BOOLEAN     NOT NULL DEFAULT TRUE,
    export_format           TEXT        NOT NULL DEFAULT 'json',
    violation_count         INTEGER     NOT NULL DEFAULT 0,
    summary                 JSONB        NOT NULL DEFAULT '{}',
    file_path               TEXT,
    completed_at            TIMESTAMPTZ,
    error_detail            TEXT,

    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS ix_al_compliance_reports_framework
    ON al_compliance_reports (tenant_id, framework, period_start DESC);

-- ============================================================
-- AL_RETENTION_POLICIES
-- ============================================================
CREATE TABLE IF NOT EXISTS al_retention_policies (
    id                   TEXT     NOT NULL,
    tenant_id            TEXT     NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by           TEXT,
    is_deleted           BOOLEAN  NOT NULL DEFAULT FALSE,

    name                 TEXT     NOT NULL,
    description          TEXT,
    event_types          TEXT[]   NOT NULL DEFAULT '{}',
    data_classifications TEXT[]   NOT NULL DEFAULT '{}',
    retain_days          INTEGER  NOT NULL,
    archive_after_days   INTEGER,
    action_on_expiry     TEXT     NOT NULL DEFAULT 'archive',  -- RetentionAction
    is_active            BOOLEAN  NOT NULL DEFAULT TRUE,
    last_enforced_at     TIMESTAMPTZ,
    events_affected      INTEGER  NOT NULL DEFAULT 0,

    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS ix_al_retention_policies_active
    ON al_retention_policies (tenant_id, is_active)
    WHERE is_active = TRUE;

-- ============================================================
-- AL_DATA_SUBJECT_REQUESTS (GDPR DSR)
-- ============================================================
CREATE TABLE IF NOT EXISTS al_data_subject_requests (
    id            TEXT        NOT NULL,
    tenant_id     TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by    TEXT,
    is_deleted    BOOLEAN     NOT NULL DEFAULT FALSE,

    dsr_type      TEXT        NOT NULL,   -- DSRType
    subject_id    TEXT        NOT NULL,
    requested_by  TEXT        NOT NULL,
    justification TEXT        NOT NULL,
    scope_details JSONB        NOT NULL DEFAULT '{}',
    status        TEXT        NOT NULL DEFAULT 'pending',  -- DSRStatus
    reviewer_id   TEXT,
    notes         TEXT,
    response_data JSONB        NOT NULL DEFAULT '{}',
    fulfilled_at  TIMESTAMPTZ,
    audit_impact  TEXT[]       NOT NULL DEFAULT '{}',

    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS ix_al_dsr_subject
    ON al_data_subject_requests (tenant_id, subject_id);

CREATE INDEX IF NOT EXISTS ix_al_dsr_status
    ON al_data_subject_requests (tenant_id, status);

-- ============================================================
-- AL_EVIDENCE_PACKAGES
-- ============================================================
CREATE TABLE IF NOT EXISTS al_evidence_packages (
    id               TEXT        NOT NULL,
    tenant_id        TEXT        NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by       TEXT,
    is_deleted       BOOLEAN     NOT NULL DEFAULT FALSE,

    name             TEXT        NOT NULL,
    description      TEXT,
    event_ids        TEXT[]       NOT NULL DEFAULT '{}',
    trail_ids        TEXT[]       NOT NULL DEFAULT '{}',
    requested_by     TEXT        NOT NULL,
    reason           TEXT        NOT NULL,
    legal_matter     TEXT,
    status           TEXT        NOT NULL DEFAULT 'assembling',  -- EvidencePackageStatus
    include_chain    BOOLEAN     NOT NULL DEFAULT TRUE,
    export_format    TEXT        NOT NULL DEFAULT 'zip',
    file_path        TEXT,
    file_checksum    TEXT,
    event_count      INTEGER     NOT NULL DEFAULT 0,
    sealed_at        TIMESTAMPTZ,
    sealed_by        TEXT,
    chain_of_custody JSONB        NOT NULL DEFAULT '[]',

    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS ix_al_evidence_packages_status
    ON al_evidence_packages (tenant_id, status);

CREATE INDEX IF NOT EXISTS ix_al_evidence_packages_legal_matter
    ON al_evidence_packages (tenant_id, legal_matter)
    WHERE legal_matter IS NOT NULL;

-- ============================================================
-- AL_TAMPER_DETECTION
-- ============================================================
CREATE TABLE IF NOT EXISTS al_tamper_detection (
    id               TEXT        NOT NULL,
    tenant_id        TEXT        NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by       TEXT,
    is_deleted       BOOLEAN     NOT NULL DEFAULT FALSE,

    scan_type        TEXT        NOT NULL,   -- "scheduled" | "on-demand" | "triggered"
    scanned_by       TEXT        NOT NULL,
    scope_filter     JSONB        NOT NULL DEFAULT '{}',
    status           TEXT        NOT NULL DEFAULT 'clean',   -- TamperStatus
    events_scanned   INTEGER     NOT NULL DEFAULT 0,
    events_suspect   INTEGER     NOT NULL DEFAULT 0,
    suspect_ids      TEXT[]       NOT NULL DEFAULT '{}',
    detail           JSONB        NOT NULL DEFAULT '{}',
    completed_at     TIMESTAMPTZ,

    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS ix_al_tamper_detection_status
    ON al_tamper_detection (tenant_id, status);

CREATE INDEX IF NOT EXISTS ix_al_tamper_detection_created_at
    ON al_tamper_detection (tenant_id, created_at DESC);

-- ============================================================
-- AL_AUDIT_QUERIES — persisted / scheduled queries
-- ============================================================
CREATE TABLE IF NOT EXISTS al_audit_queries (
    id               TEXT        NOT NULL,
    tenant_id        TEXT        NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by       TEXT,
    is_deleted       BOOLEAN     NOT NULL DEFAULT FALSE,

    name             TEXT,
    query_type       TEXT        NOT NULL DEFAULT 'structured',
    event_types      TEXT[]       NOT NULL DEFAULT '{}',
    actor_ids        TEXT[]       NOT NULL DEFAULT '{}',
    resource_ids     TEXT[]       NOT NULL DEFAULT '{}',
    sources          TEXT[]       NOT NULL DEFAULT '{}',
    date_start       TIMESTAMPTZ,
    date_end         TIMESTAMPTZ,
    risk_score_min   NUMERIC(5,4),
    risk_score_max   NUMERIC(5,4),
    compliance_tags  TEXT[]       NOT NULL DEFAULT '{}',
    success          BOOLEAN,
    full_text        TEXT,
    nlp_query        TEXT,
    raw_sql          TEXT,
    query_limit      INTEGER     NOT NULL DEFAULT 100,
    query_offset     INTEGER     NOT NULL DEFAULT 0,
    sort_by          TEXT        NOT NULL DEFAULT 'created_at',
    sort_desc        BOOLEAN     NOT NULL DEFAULT TRUE,
    requested_by     TEXT        NOT NULL,
    result_count     INTEGER     NOT NULL DEFAULT 0,
    executed_at      TIMESTAMPTZ,
    duration_ms      INTEGER,

    PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS ix_al_audit_queries_requested_by
    ON al_audit_queries (tenant_id, requested_by);

-- ============================================================
-- Trigger: keep updated_at current on all tables
-- ============================================================
CREATE OR REPLACE FUNCTION _audl_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'al_trails',
        'al_compliance_reports',
        'al_retention_policies',
        'al_data_subject_requests',
        'al_evidence_packages',
        'al_tamper_detection',
        'al_audit_queries'
    ]
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_%s_updated_at ON %s;
             CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %s
             FOR EACH ROW EXECUTE FUNCTION _audl_set_updated_at();',
            tbl, tbl, tbl, tbl
        );
    END LOOP;
END;
$$;

-- al_events: immutability enforced at app layer.
-- A trigger that blocks UPDATE/DELETE on al_events protects against
-- direct DB access bypassing the service.
CREATE OR REPLACE FUNCTION _audl_block_event_mutation()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.immutable THEN
        RAISE EXCEPTION
            'al_events row % is immutable — mutations are not permitted', OLD.id;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_al_events_immutable ON al_events;
CREATE TRIGGER trg_al_events_immutable
BEFORE UPDATE OR DELETE ON al_events
FOR EACH ROW EXECUTE FUNCTION _audl_block_event_mutation();
