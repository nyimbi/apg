-- APG Anti-Money Laundering — Full PostgreSQL schema
-- Run: psql $DATABASE_URL < database/schema.sql
-- All tables include: tenant_id isolation, audit columns, soft-delete, GIN indexes.

-- ---------------------------------------------------------------------------
-- Transaction Monitoring Rules
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_monitoring_rules (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    name                TEXT        NOT NULL,
    description         TEXT        NOT NULL DEFAULT '',
    rule_type           TEXT        NOT NULL,  -- threshold|velocity|pattern|network|watchlist|behavioral|geographic|peer_comparison
    conditions          JSONB       NOT NULL DEFAULT '[]',
    alert_type          TEXT        NOT NULL,
    severity            TEXT        NOT NULL,
    lookback_days       INT         NOT NULL DEFAULT 30,
    min_occurrences     INT         NOT NULL DEFAULT 1,
    score_weight        NUMERIC(5,2) NOT NULL DEFAULT 1.0,
    jurisdictions       TEXT[]      NOT NULL DEFAULT '{}',
    enabled             BOOLEAN     NOT NULL DEFAULT TRUE,
    status              TEXT        NOT NULL DEFAULT 'active',
    hit_count           INT         NOT NULL DEFAULT 0,
    false_positive_rate NUMERIC(6,4) NOT NULL DEFAULT 0.0,
    metadata            JSONB       NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_aml_rules_tenant   ON aml_monitoring_rules (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_rules_enabled  ON aml_monitoring_rules (tenant_id, enabled) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_rules_type     ON aml_monitoring_rules (tenant_id, rule_type) WHERE NOT is_deleted;


-- ---------------------------------------------------------------------------
-- AML Alerts
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_alerts (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    alert_type          TEXT        NOT NULL,
    severity            TEXT        NOT NULL,  -- low|medium|high|critical
    status              TEXT        NOT NULL DEFAULT 'open',
    subject_reference   TEXT        NOT NULL,
    kyc_profile_id      TEXT,
    rule_id             TEXT        REFERENCES aml_monitoring_rules(id) ON DELETE SET NULL,
    transaction_ids     TEXT[]      NOT NULL DEFAULT '{}',
    evidence_references TEXT[]      NOT NULL DEFAULT '{}',
    risk_score          INT         NOT NULL DEFAULT 0 CHECK (risk_score BETWEEN 0 AND 100),
    typology_codes      TEXT[]      NOT NULL DEFAULT '{}',
    amount              NUMERIC(20,4),
    currency            CHAR(3),
    narrative           TEXT        NOT NULL DEFAULT '',
    disposition         TEXT        NOT NULL DEFAULT '',
    reviewer_id         TEXT,
    case_id             TEXT,
    metadata            JSONB       NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_aml_alerts_tenant    ON aml_alerts (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_alerts_status    ON aml_alerts (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_alerts_severity  ON aml_alerts (tenant_id, severity) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_alerts_subject   ON aml_alerts (tenant_id, subject_reference) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_alerts_type      ON aml_alerts (tenant_id, alert_type) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_alerts_created   ON aml_alerts (tenant_id, created_at DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_alerts_meta      ON aml_alerts USING gin (metadata);
-- Partition hint: PARTITION BY RANGE (created_at) for high-volume deployments


-- ---------------------------------------------------------------------------
-- AML Cases
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_cases (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    alert_id            TEXT        NOT NULL REFERENCES aml_alerts(id) ON DELETE RESTRICT,
    case_type           TEXT        NOT NULL,
    investigator_id     TEXT        NOT NULL,
    subject_reference   TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'open',
    priority            INT         NOT NULL DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    evidence_references TEXT[]      NOT NULL DEFAULT '{}',
    notes               TEXT        NOT NULL DEFAULT '',
    due_date            TIMESTAMPTZ,
    sar_id              TEXT,
    ctr_id              TEXT,
    closed_at           TIMESTAMPTZ,
    closed_by           TEXT,
    metadata            JSONB       NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_aml_cases_tenant      ON aml_cases (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_cases_status      ON aml_cases (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_cases_investigator ON aml_cases (tenant_id, investigator_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_cases_subject     ON aml_cases (tenant_id, subject_reference) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_cases_due         ON aml_cases (tenant_id, due_date) WHERE NOT is_deleted AND due_date IS NOT NULL;


-- ---------------------------------------------------------------------------
-- Investigation Notes
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_investigation_notes (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    case_id         TEXT        NOT NULL REFERENCES aml_cases(id) ON DELETE CASCADE,
    body            TEXT        NOT NULL,
    is_privileged   BOOLEAN     NOT NULL DEFAULT FALSE,
    attachments     TEXT[]      NOT NULL DEFAULT '{}',
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_aml_notes_case   ON aml_investigation_notes (case_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_notes_tenant ON aml_investigation_notes (tenant_id) WHERE NOT is_deleted;


-- ---------------------------------------------------------------------------
-- Suspicious Activity Reports (SAR)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_sars (
    id                          TEXT        NOT NULL,
    tenant_id                   TEXT        NOT NULL,
    case_id                     TEXT        NOT NULL REFERENCES aml_cases(id) ON DELETE RESTRICT,
    subject_reference           TEXT        NOT NULL,
    subject_name                TEXT        NOT NULL,
    subject_dob                 TEXT,
    subject_tin                 TEXT,
    subject_address             TEXT        NOT NULL DEFAULT '',
    jurisdiction                TEXT        NOT NULL,
    filing_institution          TEXT        NOT NULL,
    narrative                   TEXT        NOT NULL,
    suspicious_activity_start   TIMESTAMPTZ NOT NULL,
    suspicious_activity_end     TIMESTAMPTZ NOT NULL,
    total_amount                NUMERIC(20,4) NOT NULL,
    currency                    CHAR(3)     NOT NULL,
    transaction_ids             TEXT[]      NOT NULL DEFAULT '{}',
    evidence_references         TEXT[]      NOT NULL DEFAULT '{}',
    typology_codes              TEXT[]      NOT NULL DEFAULT '{}',
    status                      TEXT        NOT NULL DEFAULT 'draft',  -- draft|pending_approval|approved|filed|rejected|amended
    approved_by                 TEXT,
    approved_at                 TIMESTAMPTZ,
    filing_reference            TEXT,
    filed_at                    TIMESTAMPTZ,
    rejection_reason            TEXT,
    metadata                    JSONB       NOT NULL DEFAULT '{}',
    created_by                  TEXT        NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_aml_sars_tenant      ON aml_sars (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_sars_status      ON aml_sars (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_sars_jurisdiction ON aml_sars (tenant_id, jurisdiction) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_sars_subject     ON aml_sars (tenant_id, subject_reference) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_sars_filed_at    ON aml_sars (tenant_id, filed_at DESC) WHERE NOT is_deleted AND filed_at IS NOT NULL;


-- ---------------------------------------------------------------------------
-- Currency Transaction Reports (CTR)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_ctrs (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    transaction_id      TEXT        NOT NULL,
    subject_reference   TEXT        NOT NULL,
    subject_name        TEXT        NOT NULL,
    subject_id_number   TEXT,
    amount              NUMERIC(20,4) NOT NULL,
    currency            CHAR(3)     NOT NULL,
    transaction_date    TIMESTAMPTZ NOT NULL,
    transaction_type    TEXT        NOT NULL,
    branch_id           TEXT,
    jurisdiction        TEXT        NOT NULL,
    filing_institution  TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'pending',  -- pending|filed|amended|exempt
    filing_reference    TEXT,
    filed_at            TIMESTAMPTZ,
    metadata            JSONB       NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_aml_ctrs_tenant       ON aml_ctrs (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_ctrs_status       ON aml_ctrs (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_ctrs_jurisdiction ON aml_ctrs (tenant_id, jurisdiction) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_ctrs_transaction  ON aml_ctrs (transaction_id);
-- High volume: PARTITION BY RANGE (transaction_date)


-- ---------------------------------------------------------------------------
-- Watchlist Matches
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_watchlist_matches (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    subject_reference   TEXT        NOT NULL,
    subject_name        TEXT        NOT NULL,
    list_name           TEXT        NOT NULL,  -- OFAC_SDN, UN_CONSOLIDATED, EU_CONSOLIDATED, PEP, etc.
    list_entry_id       TEXT        NOT NULL,
    match_score         NUMERIC(5,4) NOT NULL CHECK (match_score BETWEEN 0 AND 1),
    match_fields        TEXT[]      NOT NULL DEFAULT '{}',
    matched_name        TEXT        NOT NULL DEFAULT '',
    matched_dob         TEXT,
    matched_nationality TEXT,
    kyc_profile_id      TEXT,
    status              TEXT        NOT NULL DEFAULT 'pending',  -- pending|confirmed|false_positive|escalated
    reviewer_id         TEXT,
    reviewed_at         TIMESTAMPTZ,
    alert_id            TEXT        REFERENCES aml_alerts(id) ON DELETE SET NULL,
    metadata            JSONB       NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_aml_wl_tenant   ON aml_watchlist_matches (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_wl_status   ON aml_watchlist_matches (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_wl_subject  ON aml_watchlist_matches (tenant_id, subject_reference) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_wl_list     ON aml_watchlist_matches (list_name, match_score DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_wl_score    ON aml_watchlist_matches (tenant_id, match_score DESC) WHERE NOT is_deleted;


-- ---------------------------------------------------------------------------
-- Regulatory Filings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_regulatory_filings (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    filing_type             TEXT        NOT NULL,  -- SAR|CTR|STR|MLRO_REPORT
    jurisdiction            TEXT        NOT NULL,
    regulator               TEXT        NOT NULL,
    reference_id            TEXT        NOT NULL,
    period_start            TIMESTAMPTZ NOT NULL,
    period_end              TIMESTAMPTZ NOT NULL,
    filing_institution      TEXT        NOT NULL,
    status                  TEXT        NOT NULL DEFAULT 'pending',
    submission_reference    TEXT,
    submitted_at            TIMESTAMPTZ,
    acknowledged_at         TIMESTAMPTZ,
    rejection_reason        TEXT,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    created_by              TEXT        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_aml_filings_tenant       ON aml_regulatory_filings (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_filings_status       ON aml_regulatory_filings (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_filings_jurisdiction ON aml_regulatory_filings (tenant_id, jurisdiction) WHERE NOT is_deleted;


-- ---------------------------------------------------------------------------
-- Risk Segments
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_risk_segments (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    subject_reference       TEXT        NOT NULL,
    kyc_profile_id          TEXT,
    segment                 TEXT        NOT NULL,  -- low|medium|high|very_high|prohibited
    risk_score              INT         NOT NULL DEFAULT 0 CHECK (risk_score BETWEEN 0 AND 100),
    contributing_factors    TEXT[]      NOT NULL DEFAULT '{}',
    effective_date          TIMESTAMPTZ NOT NULL DEFAULT now(),
    review_date             TIMESTAMPTZ,
    previous_segment        TEXT,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    created_by              TEXT        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_aml_risk_tenant   ON aml_risk_segments (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_risk_subject  ON aml_risk_segments (tenant_id, subject_reference) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_risk_segment  ON aml_risk_segments (tenant_id, segment) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_aml_risk_review   ON aml_risk_segments (tenant_id, review_date) WHERE NOT is_deleted AND review_date IS NOT NULL;


-- ---------------------------------------------------------------------------
-- AML Domain Events (append-only audit log)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_events (
    id              BIGSERIAL   NOT NULL,
    tenant_id       TEXT        NOT NULL,
    event_type      TEXT        NOT NULL,
    capability_id   TEXT        NOT NULL DEFAULT 'fintech_aml',
    actor_id        TEXT        NOT NULL,
    reference_id    TEXT        NOT NULL,
    payload         JSONB       NOT NULL DEFAULT '{}',
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
) PARTITION BY RANGE (occurred_at);

-- Monthly partitions (create via cron or migration script)
CREATE TABLE IF NOT EXISTS aml_events_default PARTITION OF aml_events DEFAULT;

CREATE INDEX IF NOT EXISTS idx_aml_events_tenant  ON aml_events (tenant_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_aml_events_type    ON aml_events (event_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_aml_events_ref     ON aml_events (reference_id);
CREATE INDEX IF NOT EXISTS idx_aml_events_payload ON aml_events USING gin (payload);


-- ---------------------------------------------------------------------------
-- Shared JSONB store (backward compat with aml_runtime)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS apg_records (
    id          TEXT        NOT NULL,
    collection  TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL DEFAULT 'default',
    data        JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, id)
);
CREATE INDEX IF NOT EXISTS idx_apg_aml_tenant ON apg_records (collection, tenant_id);
CREATE INDEX IF NOT EXISTS idx_apg_aml_data   ON apg_records USING gin (data);


-- ---------------------------------------------------------------------------
-- updated_at trigger (attach to each table)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION aml_set_updated_at()
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
        'aml_monitoring_rules', 'aml_alerts', 'aml_cases',
        'aml_investigation_notes', 'aml_sars', 'aml_ctrs',
        'aml_watchlist_matches', 'aml_regulatory_filings', 'aml_risk_segments'
    ] LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_%I_updated_at ON %I;
             CREATE TRIGGER trg_%I_updated_at
             BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION aml_set_updated_at();',
            t, t, t, t
        );
    END LOOP;
END;
$$;
