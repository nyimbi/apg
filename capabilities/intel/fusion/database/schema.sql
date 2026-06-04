-- APG Intelligence Fusion — PostgreSQL schema
-- © 2025 Datacraft — Nyimbi Odero
--
-- Run: psql $DATABASE_URL < database/schema.sql
--
-- Tables are partitioned by tenant_id for large-scale deployments.
-- All FK references use TEXT IDs (UUID-7).

-- ─────────────────────────────────────────────────────────────────────────────
-- Generic JSONB record store (shared by all standalone capability modes)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS apg_records (
    id          TEXT        NOT NULL,
    collection  TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL DEFAULT 'default',
    data        JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, id)
);

CREATE INDEX IF NOT EXISTS idx_apg_records_tenant
    ON apg_records (collection, tenant_id);

CREATE INDEX IF NOT EXISTS idx_apg_records_data_gin
    ON apg_records USING gin (data);

CREATE OR REPLACE FUNCTION apg_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_apg_records_updated_at') THEN
        CREATE TRIGGER trg_apg_records_updated_at
            BEFORE UPDATE ON apg_records
            FOR EACH ROW EXECUTE FUNCTION apg_set_updated_at();
    END IF;
END $$;


-- ─────────────────────────────────────────────────────────────────────────────
-- Normalised intelligence fusion tables (production)
-- ─────────────────────────────────────────────────────────────────────────────

-- FusionWorkspace

CREATE TABLE IF NOT EXISTS fusion_workspaces (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    workspace_type      TEXT        NOT NULL,
    name                TEXT        NOT NULL,
    classification      TEXT        NOT NULL DEFAULT 'unclassified',
    authority_id        TEXT        NOT NULL,
    description         TEXT        NOT NULL DEFAULT '',
    tags                TEXT[]      NOT NULL DEFAULT '{}',
    status              TEXT        NOT NULL DEFAULT 'active',
    lead_analyst_id     TEXT        NOT NULL DEFAULT '',
    item_count          INT         NOT NULL DEFAULT 0,
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_fw_tenant_status
    ON fusion_workspaces (tenant_id, status)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_fw_tenant_type
    ON fusion_workspaces (tenant_id, workspace_type)
    WHERE is_deleted = FALSE;


-- IntelligenceItem

CREATE TABLE IF NOT EXISTS fusion_intel_items (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    workspace_id        TEXT        NOT NULL DEFAULT '',
    source_type         TEXT        NOT NULL,
    source_reference    TEXT        NOT NULL,
    content_summary     TEXT        NOT NULL DEFAULT '',
    content_fingerprint TEXT        NOT NULL,
    classification      TEXT        NOT NULL DEFAULT 'unclassified',
    tlp                 TEXT        NOT NULL DEFAULT 'TLP:AMBER',
    confidence_score    NUMERIC(5,4) NOT NULL DEFAULT 0.5,
    collected_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    custodian_id        TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'raw',
    metadata            JSONB       NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT fk_ii_workspace
        FOREIGN KEY (workspace_id) REFERENCES fusion_workspaces(id)
        ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS idx_ii_tenant_status
    ON fusion_intel_items (tenant_id, status)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_ii_tenant_source
    ON fusion_intel_items (tenant_id, source_type)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_ii_workspace
    ON fusion_intel_items (workspace_id)
    WHERE is_deleted = FALSE;

-- Deduplication by fingerprint within a tenant
CREATE UNIQUE INDEX IF NOT EXISTS idx_ii_fingerprint_tenant
    ON fusion_intel_items (tenant_id, content_fingerprint)
    WHERE is_deleted = FALSE;


-- CorrelationSet

CREATE TABLE IF NOT EXISTS fusion_correlations (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    workspace_id        TEXT        NOT NULL,
    correlation_type    TEXT        NOT NULL,
    item_ids            TEXT[]      NOT NULL DEFAULT '{}',
    analyst_id          TEXT        NOT NULL,
    confidence_score    NUMERIC(5,4) NOT NULL DEFAULT 0.5,
    rationale           TEXT        NOT NULL DEFAULT '',
    evidence_ids        TEXT[]      NOT NULL DEFAULT '{}',
    status              TEXT        NOT NULL DEFAULT 'open',
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT fk_cs_workspace
        FOREIGN KEY (workspace_id) REFERENCES fusion_workspaces(id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_cs_tenant_status
    ON fusion_correlations (tenant_id, status)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_cs_workspace
    ON fusion_correlations (workspace_id)
    WHERE is_deleted = FALSE;


-- AssessmentPicture

CREATE TABLE IF NOT EXISTS fusion_assessments (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    workspace_id        TEXT        NOT NULL,
    assessment_type     TEXT        NOT NULL,
    risk_level          TEXT        NOT NULL,
    summary             TEXT        NOT NULL DEFAULT '',
    analyst_id          TEXT        NOT NULL,
    confidence_score    NUMERIC(5,4) NOT NULL DEFAULT 0.5,
    hypothesis_ids      TEXT[]      NOT NULL DEFAULT '{}',
    correlation_ids     TEXT[]      NOT NULL DEFAULT '{}',
    evidence_ids        TEXT[]      NOT NULL DEFAULT '{}',
    approved_by         TEXT        NOT NULL DEFAULT '',
    approved_at         TIMESTAMPTZ,
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT fk_ap_workspace
        FOREIGN KEY (workspace_id) REFERENCES fusion_workspaces(id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_ap_tenant_risk
    ON fusion_assessments (tenant_id, risk_level)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_ap_workspace
    ON fusion_assessments (workspace_id)
    WHERE is_deleted = FALSE;


-- IntelligenceProduct

CREATE TABLE IF NOT EXISTS fusion_products (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    workspace_id        TEXT        NOT NULL,
    product_type        TEXT        NOT NULL,
    title               TEXT        NOT NULL,
    classification      TEXT        NOT NULL DEFAULT 'unclassified',
    tlp                 TEXT        NOT NULL DEFAULT 'TLP:AMBER',
    summary             TEXT        NOT NULL DEFAULT '',
    body_reference      TEXT        NOT NULL DEFAULT '',
    assessment_ids      TEXT[]      NOT NULL DEFAULT '{}',
    author_id           TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'draft',
    reviewer_id         TEXT        NOT NULL DEFAULT '',
    reviewed_at         TIMESTAMPTZ,
    released_at         TIMESTAMPTZ,
    dissemination_ids   TEXT[]      NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT fk_ip_workspace
        FOREIGN KEY (workspace_id) REFERENCES fusion_workspaces(id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_ip_tenant_status
    ON fusion_products (tenant_id, status)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_ip_tenant_tlp
    ON fusion_products (tenant_id, tlp)
    WHERE is_deleted = FALSE;


-- AnalyticalJudgement

CREATE TABLE IF NOT EXISTS fusion_judgements (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    workspace_id        TEXT        NOT NULL,
    judgement_type      TEXT        NOT NULL,
    statement           TEXT        NOT NULL,
    confidence_score    NUMERIC(5,4) NOT NULL DEFAULT 0.5,
    confidence_level    TEXT        NOT NULL DEFAULT 'likely',
    analyst_id          TEXT        NOT NULL,
    sat_method          TEXT,
    key_assumptions     TEXT[]      NOT NULL DEFAULT '{}',
    evidence_ids        TEXT[]      NOT NULL DEFAULT '{}',
    challenger_ids      TEXT[]      NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT fk_aj_workspace
        FOREIGN KEY (workspace_id) REFERENCES fusion_workspaces(id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_aj_tenant
    ON fusion_judgements (tenant_id, judgement_type)
    WHERE is_deleted = FALSE;


-- Evidence

CREATE TABLE IF NOT EXISTS fusion_evidence (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    workspace_id        TEXT        NOT NULL,
    evidence_type       TEXT        NOT NULL,
    source_reference    TEXT        NOT NULL,
    content_fingerprint TEXT        NOT NULL,
    classification      TEXT        NOT NULL DEFAULT 'unclassified',
    custodian_id        TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'pending',
    chain_of_custody    TEXT[]      NOT NULL DEFAULT '{}',
    metadata            JSONB       NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT fk_ev_workspace
        FOREIGN KEY (workspace_id) REFERENCES fusion_workspaces(id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_ev_tenant_status
    ON fusion_evidence (tenant_id, status)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_ev_workspace
    ON fusion_evidence (workspace_id)
    WHERE is_deleted = FALSE;

-- Deduplication by fingerprint
CREATE UNIQUE INDEX IF NOT EXISTS idx_ev_fingerprint_tenant
    ON fusion_evidence (tenant_id, content_fingerprint)
    WHERE is_deleted = FALSE;


-- HypothesisTest

CREATE TABLE IF NOT EXISTS fusion_hypotheses (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    workspace_id            TEXT        NOT NULL,
    statement               TEXT        NOT NULL,
    sat_method              TEXT        NOT NULL DEFAULT 'analysis_of_competing_hypotheses',
    analyst_id              TEXT        NOT NULL,
    alternative_hypotheses  TEXT[]      NOT NULL DEFAULT '{}',
    evidence_ids            TEXT[]      NOT NULL DEFAULT '{}',
    initial_confidence      NUMERIC(5,4) NOT NULL DEFAULT 0.5,
    final_confidence        NUMERIC(5,4),
    status                  TEXT        NOT NULL DEFAULT 'open',
    conclusion              TEXT        NOT NULL DEFAULT '',
    ach_matrix              JSONB       NOT NULL DEFAULT '{}',
    created_by              TEXT        NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT fk_ht_workspace
        FOREIGN KEY (workspace_id) REFERENCES fusion_workspaces(id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_ht_tenant_status
    ON fusion_hypotheses (tenant_id, status)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_ht_workspace
    ON fusion_hypotheses (workspace_id)
    WHERE is_deleted = FALSE;


-- DisseminationRecord

CREATE TABLE IF NOT EXISTS fusion_disseminations (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    product_id          TEXT        NOT NULL,
    audience            TEXT        NOT NULL,
    tlp                 TEXT        NOT NULL,
    approval_reference  TEXT        NOT NULL,
    disseminated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    disseminated_by     TEXT        NOT NULL,
    notes               TEXT        NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id),
    CONSTRAINT fk_dr_product
        FOREIGN KEY (product_id) REFERENCES fusion_products(id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_dr_tenant_product
    ON fusion_disseminations (tenant_id, product_id);


-- Domain events / audit log

CREATE TABLE IF NOT EXISTS fusion_events (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    event_type      TEXT        NOT NULL,
    actor_id        TEXT        NOT NULL,
    resource_id     TEXT        NOT NULL,
    resource_type   TEXT        NOT NULL,
    payload         JSONB       NOT NULL DEFAULT '{}',
    capability_id   TEXT        NOT NULL DEFAULT 'intel_fusion',
    stream          TEXT        NOT NULL DEFAULT 'apg.intel.fusion.lifecycle',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_fe_tenant_type
    ON fusion_events (tenant_id, event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_fe_resource
    ON fusion_events (resource_id, created_at DESC);
