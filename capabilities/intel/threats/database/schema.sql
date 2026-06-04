-- =============================================================================
-- APG Threat Intelligence — PostgreSQL Schema
-- © 2025 Datacraft
-- =============================================================================
-- Run: psql $DATABASE_URL -f database/schema.sql
-- Requires PostgreSQL 14+
-- =============================================================================

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ── Schema namespace ──────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS ti;
SET search_path TO ti, public;

-- =============================================================================
-- MITRE ATT&CK Techniques  (reference / lookup table, shared across tenants)
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_mitre_techniques (
    id                  TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id           TEXT        NOT NULL DEFAULT 'default',
    technique_id        TEXT        NOT NULL,          -- e.g. T1059.001
    name                TEXT        NOT NULL,
    tactic              TEXT        NOT NULL,          -- MitreTactic enum value
    description         TEXT,
    platforms           TEXT[]      NOT NULL DEFAULT '{}',
    data_sources        TEXT[]      NOT NULL DEFAULT '{}',
    detection_guidance  TEXT,
    mitigations         TEXT[]      NOT NULL DEFAULT '{}',
    sub_techniques      TEXT[]      NOT NULL DEFAULT '{}',
    url                 TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL,
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_ti_mitre_tenant_technique
    ON ti_mitre_techniques (tenant_id, technique_id)
    WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_ti_mitre_tenant     ON ti_mitre_techniques (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_mitre_tactic     ON ti_mitre_techniques (tactic);

-- =============================================================================
-- Kill-Chain Phases
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_kill_chain_phases (
    id                  TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id           TEXT        NOT NULL DEFAULT 'default',
    phase_name          TEXT        NOT NULL,          -- KillChainPhaseType enum
    kill_chain_name     TEXT        NOT NULL DEFAULT 'lockheed-martin-cyber-kill-chain',
    description         TEXT,
    "order"             INT         NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL,
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_ti_kcp_tenant_phase_chain
    ON ti_kill_chain_phases (tenant_id, phase_name, kill_chain_name)
    WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_ti_kcp_tenant ON ti_kill_chain_phases (tenant_id);

-- =============================================================================
-- Threat Feeds
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_feeds (
    id                      TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id               TEXT        NOT NULL DEFAULT 'default',
    name                    TEXT        NOT NULL,
    feed_type               TEXT        NOT NULL,      -- FeedType enum
    status                  TEXT        NOT NULL DEFAULT 'active',
    url                     TEXT,
    api_key                 TEXT,                      -- encrypted at rest
    collection_id           TEXT,
    poll_interval_seconds   INT         NOT NULL DEFAULT 3600,
    confidence_weight       NUMERIC(4,3) NOT NULL DEFAULT 0.8
                                CHECK (confidence_weight BETWEEN 0 AND 1),
    tlp_filter              TEXT,
    indicator_types         TEXT[]      NOT NULL DEFAULT '{}',
    description             TEXT,
    custodian_id            TEXT        NOT NULL,
    evidence_reference      TEXT        NOT NULL,
    last_polled_at          TIMESTAMPTZ,
    indicators_ingested     INT         NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by              TEXT        NOT NULL,
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_ti_feeds_tenant      ON ti_feeds (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_feeds_status      ON ti_feeds (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ti_feeds_type        ON ti_feeds (feed_type);

-- =============================================================================
-- Threat Actors
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_actors (
    id                  TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id           TEXT        NOT NULL DEFAULT 'default',
    name                TEXT        NOT NULL,
    actor_type          TEXT        NOT NULL,          -- ThreatActorType enum
    status              TEXT        NOT NULL DEFAULT 'suspected',
    aliases             TEXT[]      NOT NULL DEFAULT '{}',
    description         TEXT,
    motivation          TEXT,
    sophistication      TEXT,
    first_seen          TIMESTAMPTZ,
    last_seen           TIMESTAMPTZ,
    confidence_score    NUMERIC(5,4) NOT NULL
                            CHECK (confidence_score BETWEEN 0 AND 1),
    country_of_origin   TEXT,
    workspace_id        TEXT,
    evidence_reference  TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL,
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_ti_actors_tenant     ON ti_actors (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_actors_status     ON ti_actors (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ti_actors_type       ON ti_actors (actor_type);
CREATE INDEX IF NOT EXISTS idx_ti_actors_name_trgm  ON ti_actors USING gin (name gin_trgm_ops);

-- =============================================================================
-- Threat Campaigns
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_campaigns (
    id                  TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id           TEXT        NOT NULL DEFAULT 'default',
    name                TEXT        NOT NULL,
    campaign_type       TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'suspected',
    description         TEXT,
    actor_id            TEXT        NOT NULL REFERENCES ti_actors(id),
    first_seen          TIMESTAMPTZ,
    last_seen           TIMESTAMPTZ,
    risk_level          TEXT        NOT NULL,
    classification      TEXT        NOT NULL DEFAULT 'confidential',
    target_sectors      TEXT[]      NOT NULL DEFAULT '{}',
    target_countries    TEXT[]      NOT NULL DEFAULT '{}',
    mitre_technique_ids TEXT[]      NOT NULL DEFAULT '{}',
    kill_chain_phase_ids TEXT[]     NOT NULL DEFAULT '{}',
    indicator_ids       TEXT[]      NOT NULL DEFAULT '{}',
    evidence_reference  TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL,
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_ti_campaigns_tenant  ON ti_campaigns (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_campaigns_status  ON ti_campaigns (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ti_campaigns_actor   ON ti_campaigns (actor_id);
CREATE INDEX IF NOT EXISTS idx_ti_campaigns_risk    ON ti_campaigns (risk_level);
CREATE INDEX IF NOT EXISTS idx_ti_campaigns_sectors ON ti_campaigns USING gin (target_sectors);

-- =============================================================================
-- Threat Indicators (IOCs / TTPs)
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_indicators (
    id                  TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id           TEXT        NOT NULL DEFAULT 'default',
    indicator_type      TEXT        NOT NULL,          -- IndicatorType enum
    value               TEXT        NOT NULL,
    description         TEXT,
    status              TEXT        NOT NULL DEFAULT 'active',
    source_id           TEXT        NOT NULL,
    feed_id             TEXT        REFERENCES ti_feeds(id),
    confidence_score    NUMERIC(5,4) NOT NULL
                            CHECK (confidence_score BETWEEN 0 AND 1),
    valid_from          TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until         TIMESTAMPTZ,
    kill_chain_phase_ids TEXT[]     NOT NULL DEFAULT '{}',
    mitre_technique_ids TEXT[]      NOT NULL DEFAULT '{}',
    actor_ids           TEXT[]      NOT NULL DEFAULT '{}',
    tags                TEXT[]      NOT NULL DEFAULT '{}',
    tlp                 TEXT        NOT NULL DEFAULT 'green',
    evidence_reference  TEXT        NOT NULL,
    staleness_score     NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL,
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
-- Deduplicate indicator value within tenant
CREATE UNIQUE INDEX IF NOT EXISTS uq_ti_indicator_tenant_type_value
    ON ti_indicators (tenant_id, indicator_type, value)
    WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_ti_indicators_tenant  ON ti_indicators (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_indicators_status  ON ti_indicators (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ti_indicators_type    ON ti_indicators (indicator_type);
CREATE INDEX IF NOT EXISTS idx_ti_indicators_feed    ON ti_indicators (feed_id);
CREATE INDEX IF NOT EXISTS idx_ti_indicators_valid   ON ti_indicators (valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_ti_indicators_tlp     ON ti_indicators (tlp);
CREATE INDEX IF NOT EXISTS idx_ti_indicators_tags    ON ti_indicators USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_ti_indicators_actors  ON ti_indicators USING gin (actor_ids);
CREATE INDEX IF NOT EXISTS idx_ti_indicators_value_trgm ON ti_indicators USING gin (value gin_trgm_ops);

-- Partition hint: for high-volume deployments, range-partition on valid_from per month
-- PARTITION BY RANGE (valid_from);

-- =============================================================================
-- Attribution Evidence
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_attribution_evidence (
    id                  TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id           TEXT        NOT NULL DEFAULT 'default',
    actor_id            TEXT        NOT NULL REFERENCES ti_actors(id),
    evidence_type       TEXT        NOT NULL,          -- EvidenceType enum
    description         TEXT        NOT NULL,
    confidence_score    NUMERIC(5,4) NOT NULL
                            CHECK (confidence_score BETWEEN 0 AND 1),
    source_id           TEXT,
    indicator_ids       TEXT[]      NOT NULL DEFAULT '{}',
    collection_date     TIMESTAMPTZ,
    classification      TEXT        NOT NULL DEFAULT 'confidential',
    analyst_id          TEXT        NOT NULL,
    raw_reference       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL,
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_ti_attrevi_tenant    ON ti_attribution_evidence (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_attrevi_actor     ON ti_attribution_evidence (actor_id);
CREATE INDEX IF NOT EXISTS idx_ti_attrevi_type      ON ti_attribution_evidence (evidence_type);

-- =============================================================================
-- Threat Assessments
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_assessments (
    id                  TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id           TEXT        NOT NULL DEFAULT 'default',
    assessment_type     TEXT        NOT NULL,
    campaign_id         TEXT        NOT NULL REFERENCES ti_campaigns(id),
    analyst_id          TEXT        NOT NULL,
    risk_level          TEXT        NOT NULL,
    confidence_score    NUMERIC(5,4) NOT NULL
                            CHECK (confidence_score BETWEEN 0 AND 1),
    summary             TEXT        NOT NULL,
    findings            TEXT[]      NOT NULL DEFAULT '{}',
    recommendations     TEXT[]      NOT NULL DEFAULT '{}',
    mitre_technique_ids TEXT[]      NOT NULL DEFAULT '{}',
    evidence_reference  TEXT        NOT NULL,
    approved_by         TEXT,
    approved_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL,
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_ti_assess_tenant     ON ti_assessments (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_assess_campaign   ON ti_assessments (campaign_id);
CREATE INDEX IF NOT EXISTS idx_ti_assess_analyst    ON ti_assessments (analyst_id);
CREATE INDEX IF NOT EXISTS idx_ti_assess_risk       ON ti_assessments (risk_level);

-- =============================================================================
-- Threat Reports
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_reports (
    id                      TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id               TEXT        NOT NULL DEFAULT 'default',
    title                   TEXT        NOT NULL,
    report_type             TEXT        NOT NULL,
    status                  TEXT        NOT NULL DEFAULT 'draft',
    classification          TEXT        NOT NULL,
    summary                 TEXT        NOT NULL,
    body                    TEXT,
    assessment_id           TEXT        NOT NULL REFERENCES ti_assessments(id),
    author_id               TEXT        NOT NULL,
    analyst_ids             TEXT[]      NOT NULL DEFAULT '{}',
    related_actor_ids       TEXT[]      NOT NULL DEFAULT '{}',
    related_campaign_ids    TEXT[]      NOT NULL DEFAULT '{}',
    related_indicator_ids   TEXT[]      NOT NULL DEFAULT '{}',
    mitre_technique_ids     TEXT[]      NOT NULL DEFAULT '{}',
    tags                    TEXT[]      NOT NULL DEFAULT '{}',
    tlp                     TEXT        NOT NULL DEFAULT 'amber',
    approval_reference      TEXT        NOT NULL,
    evidence_reference      TEXT        NOT NULL,
    published_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by              TEXT        NOT NULL,
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_ti_reports_tenant        ON ti_reports (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_reports_status        ON ti_reports (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ti_reports_type          ON ti_reports (report_type);
CREATE INDEX IF NOT EXISTS idx_ti_reports_assessment    ON ti_reports (assessment_id);
CREATE INDEX IF NOT EXISTS idx_ti_reports_title_trgm    ON ti_reports USING gin (title gin_trgm_ops);

-- =============================================================================
-- Intel Requirements
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_requirements (
    id                      TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id               TEXT        NOT NULL DEFAULT 'default',
    title                   TEXT        NOT NULL,
    description             TEXT        NOT NULL,
    status                  TEXT        NOT NULL DEFAULT 'open',
    requestor_id            TEXT        NOT NULL,
    priority                TEXT        NOT NULL,
    due_date                TIMESTAMPTZ,
    assigned_analyst_id     TEXT,
    related_actor_ids       TEXT[]      NOT NULL DEFAULT '{}',
    related_campaign_ids    TEXT[]      NOT NULL DEFAULT '{}',
    tags                    TEXT[]      NOT NULL DEFAULT '{}',
    satisfying_report_ids   TEXT[]      NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by              TEXT        NOT NULL,
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_ti_req_tenant    ON ti_requirements (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_req_status    ON ti_requirements (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ti_req_priority  ON ti_requirements (priority);
CREATE INDEX IF NOT EXISTS idx_ti_req_due       ON ti_requirements (due_date) WHERE due_date IS NOT NULL;

-- =============================================================================
-- Audit / Event Log  (append-only, never soft-deleted)
-- =============================================================================
CREATE TABLE IF NOT EXISTS ti_audit_events (
    id              BIGSERIAL   NOT NULL,
    tenant_id       TEXT        NOT NULL,
    event_type      TEXT        NOT NULL,
    actor_id        TEXT        NOT NULL,
    resource_id     TEXT        NOT NULL,
    resource_type   TEXT        NOT NULL,
    payload         JSONB       NOT NULL DEFAULT '{}',
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_ti_audit_tenant      ON ti_audit_events (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ti_audit_event_type  ON ti_audit_events (event_type);
CREATE INDEX IF NOT EXISTS idx_ti_audit_resource    ON ti_audit_events (resource_id);
CREATE INDEX IF NOT EXISTS idx_ti_audit_occurred    ON ti_audit_events (occurred_at DESC);

-- =============================================================================
-- Updated-at trigger (applied to all mutable tables)
-- =============================================================================
CREATE OR REPLACE FUNCTION ti_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'ti_mitre_techniques', 'ti_kill_chain_phases', 'ti_feeds',
        'ti_actors', 'ti_campaigns', 'ti_indicators',
        'ti_attribution_evidence', 'ti_assessments', 'ti_reports',
        'ti_requirements'
    ] LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_%1$s_updated_at ON %1$s;
             CREATE TRIGGER trg_%1$s_updated_at
               BEFORE UPDATE ON %1$s
               FOR EACH ROW EXECUTE FUNCTION ti_set_updated_at();',
            t
        );
    END LOOP;
END;
$$;

-- =============================================================================
-- Seed: Standard Kill-Chain Phases (tenant=default, system account)
-- =============================================================================
INSERT INTO ti_kill_chain_phases (id, tenant_id, phase_name, kill_chain_name, "order", created_by)
VALUES
    ('kcp-recon',    'default', 'reconnaissance',       'lockheed-martin-cyber-kill-chain', 1, 'system'),
    ('kcp-weapon',   'default', 'weaponization',        'lockheed-martin-cyber-kill-chain', 2, 'system'),
    ('kcp-delivery', 'default', 'delivery',             'lockheed-martin-cyber-kill-chain', 3, 'system'),
    ('kcp-exploit',  'default', 'exploitation',         'lockheed-martin-cyber-kill-chain', 4, 'system'),
    ('kcp-install',  'default', 'installation',         'lockheed-martin-cyber-kill-chain', 5, 'system'),
    ('kcp-c2',       'default', 'command_and_control',  'lockheed-martin-cyber-kill-chain', 6, 'system'),
    ('kcp-actions',  'default', 'actions_on_objectives','lockheed-martin-cyber-kill-chain', 7, 'system')
ON CONFLICT DO NOTHING;
