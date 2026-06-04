-- APG Laboratory Information System — PostgreSQL Schema
-- © 2025 Datacraft — nyimbi@gmail.com
--
-- Normalized multi-tenant LIS schema.
-- All tables carry tenant_id for row-level isolation.
-- Run: psql $DATABASE_URL -f database/schema.sql
--
-- Naming convention: lb_ prefix (2-char capability code = lb)

-- ─── Extensions ───────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ─── Enumerations ─────────────────────────────────────────────────────────────

DO $$ BEGIN
    CREATE TYPE lb_order_status AS ENUM (
        'pending', 'collected', 'in_transit', 'received', 'processing',
        'resulted', 'verified', 'reported', 'cancelled', 'on_hold'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_specimen_status AS ENUM (
        'collected', 'in_transit', 'received', 'processing',
        'stored', 'rejected', 'disposed'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_result_status AS ENUM (
        'preliminary', 'final', 'corrected', 'cancelled', 'entered_in_error'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_qc_status AS ENUM (
        'passed', 'failed', 'pending_review', 'repeated', 'accepted', 'rejected'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_instrument_status AS ENUM (
        'online', 'offline', 'maintenance', 'calibrating', 'qc_hold', 'decommissioned'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_collection_priority AS ENUM (
        'routine', 'stat', 'asap', 'timed'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_referral_status AS ENUM (
        'pending', 'dispatched', 'received', 'resulted', 'cancelled'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_critical_severity AS ENUM (
        'critical_high', 'critical_low', 'panic_value'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_interface_protocol AS ENUM (
        'hl7_v2', 'hl7_fhir', 'astm_e1381', 'poct1_a', 'lis_bridge', 'rest_json'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_patient_sex AS ENUM ('M', 'F', 'O', 'U');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_abnormal_flag AS ENUM ('H', 'HH', 'L', 'LL', 'A', 'CH', 'CL');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE lb_custody_event_type AS ENUM (
        'collected', 'transferred', 'received', 'processed',
        'stored', 'aliquoted', 'disposed'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ─── lb_lab_test — Master test catalogue ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS lb_lab_test (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    test_code               TEXT            NOT NULL,
    test_name               TEXT            NOT NULL,
    category                TEXT            NOT NULL,
    specimen_types          TEXT[]          NOT NULL DEFAULT '{}',
    loinc_code              TEXT,
    cpt_code                TEXT,
    snomed_code             TEXT,
    turnaround_minutes      INTEGER         NOT NULL DEFAULT 120 CHECK (turnaround_minutes > 0),
    stat_turnaround_minutes INTEGER         NOT NULL DEFAULT 60  CHECK (stat_turnaround_minutes > 0),
    active                  BOOLEAN         NOT NULL DEFAULT TRUE,
    requires_fasting        BOOLEAN         NOT NULL DEFAULT FALSE,
    requires_consent        BOOLEAN         NOT NULL DEFAULT FALSE,
    price                   NUMERIC(12, 4),
    department              TEXT,
    instructions            TEXT,
    sample_volume_ml        NUMERIC(8, 3)   CHECK (sample_volume_ml > 0),
    container_type          TEXT,
    storage_temperature     TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by              TEXT            NOT NULL,
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id),
    UNIQUE (tenant_id, test_code)
);

CREATE INDEX IF NOT EXISTS idx_lb_lab_test_tenant       ON lb_lab_test (tenant_id);
CREATE INDEX IF NOT EXISTS idx_lb_lab_test_category     ON lb_lab_test (tenant_id, category);
CREATE INDEX IF NOT EXISTS idx_lb_lab_test_active       ON lb_lab_test (tenant_id, active) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_lb_lab_test_loinc        ON lb_lab_test (loinc_code) WHERE loinc_code IS NOT NULL;

-- ─── lb_lab_order — Lab test orders ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS lb_lab_order (
    id                  TEXT                    NOT NULL,
    tenant_id           TEXT                    NOT NULL,
    patient_id          TEXT                    NOT NULL,
    encounter_id        TEXT                    NOT NULL,
    test_code           TEXT                    NOT NULL,
    test_name           TEXT                    NOT NULL,
    test_category       TEXT                    NOT NULL,
    collection_priority lb_collection_priority  NOT NULL DEFAULT 'routine',
    ordered_by          TEXT                    NOT NULL,
    clinical_indication TEXT                    NOT NULL,
    specimen_type       TEXT                    NOT NULL,
    patient_age_years   NUMERIC(5, 2),
    patient_sex         lb_patient_sex,
    fasting             BOOLEAN                 NOT NULL DEFAULT FALSE,
    notes               TEXT,
    status              lb_order_status         NOT NULL DEFAULT 'pending',
    specimen_id         TEXT,
    result_id           TEXT,
    report_url          TEXT,
    ordered_at          TIMESTAMPTZ             NOT NULL DEFAULT now(),
    tat_due_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    cancelled_reason    TEXT,
    on_hold_reason      TEXT,
    referral_id         TEXT,
    created_at          TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by          TEXT                    NOT NULL,
    is_deleted          BOOLEAN                 NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_lb_order_tenant          ON lb_lab_order (tenant_id);
CREATE INDEX IF NOT EXISTS idx_lb_order_patient         ON lb_lab_order (tenant_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_lb_order_status          ON lb_lab_order (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_lb_order_priority        ON lb_lab_order (tenant_id, collection_priority, ordered_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_order_encounter       ON lb_lab_order (tenant_id, encounter_id);
CREATE INDEX IF NOT EXISTS idx_lb_order_ordered_at      ON lb_lab_order (tenant_id, ordered_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_order_tat_due         ON lb_lab_order (tenant_id, tat_due_at)
    WHERE status NOT IN ('reported', 'cancelled') AND tat_due_at IS NOT NULL;

-- ─── lb_specimen — Specimen collection and tracking ───────────────────────────

CREATE TABLE IF NOT EXISTS lb_specimen (
    id                      TEXT                NOT NULL,
    tenant_id               TEXT                NOT NULL,
    order_id                TEXT                NOT NULL REFERENCES lb_lab_order(id),
    patient_id              TEXT                NOT NULL,
    specimen_type           TEXT                NOT NULL,
    collected_by            TEXT                NOT NULL,
    collection_site         TEXT                NOT NULL,
    collection_volume_ml    NUMERIC(8, 3)       CHECK (collection_volume_ml > 0),
    tube_type               TEXT,
    barcode                 TEXT,
    status                  lb_specimen_status  NOT NULL DEFAULT 'collected',
    rejection_reason        TEXT,
    rejection_notes         TEXT,
    collected_at            TIMESTAMPTZ         NOT NULL DEFAULT now(),
    received_at             TIMESTAMPTZ,
    received_by             TEXT,
    processing_started_at   TIMESTAMPTZ,
    stored_at               TIMESTAMPTZ,
    storage_location        TEXT,
    aliquot_of              TEXT                REFERENCES lb_specimen(id),
    notes                   TEXT,
    created_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by              TEXT                NOT NULL,
    is_deleted              BOOLEAN             NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_lb_specimen_tenant       ON lb_specimen (tenant_id);
CREATE INDEX IF NOT EXISTS idx_lb_specimen_order        ON lb_specimen (tenant_id, order_id);
CREATE INDEX IF NOT EXISTS idx_lb_specimen_patient      ON lb_specimen (tenant_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_lb_specimen_status       ON lb_specimen (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_lb_specimen_barcode      ON lb_specimen (barcode) WHERE barcode IS NOT NULL;

-- ─── lb_custody_event — Immutable chain-of-custody log ────────────────────────

CREATE TABLE IF NOT EXISTS lb_custody_event (
    id              TEXT                    NOT NULL,
    tenant_id       TEXT                    NOT NULL,
    specimen_id     TEXT                    NOT NULL REFERENCES lb_specimen(id),
    event_type      lb_custody_event_type   NOT NULL,
    actor_id        TEXT                    NOT NULL,
    location        TEXT,
    notes           TEXT,
    occurred_at     TIMESTAMPTZ             NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
) PARTITION BY RANGE (occurred_at);

-- Monthly partitions for custody events (high volume)
CREATE TABLE IF NOT EXISTS lb_custody_event_y2025m01
    PARTITION OF lb_custody_event
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE IF NOT EXISTS lb_custody_event_y2025m12
    PARTITION OF lb_custody_event
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS lb_custody_event_y2026m01
    PARTITION OF lb_custody_event
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE TABLE IF NOT EXISTS lb_custody_event_y2026m12
    PARTITION OF lb_custody_event
    FOR VALUES FROM ('2026-12-01') TO ('2027-01-01');

CREATE INDEX IF NOT EXISTS idx_lb_custody_specimen      ON lb_custody_event (tenant_id, specimen_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_custody_event_type    ON lb_custody_event (tenant_id, event_type);

-- ─── lb_reference_range — Demographically stratified normal ranges ─────────────

CREATE TABLE IF NOT EXISTS lb_reference_range (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    test_code       TEXT            NOT NULL,
    analyte         TEXT            NOT NULL,
    unit            TEXT            NOT NULL,
    low             NUMERIC(18, 6),
    high            NUMERIC(18, 6),
    critical_low    NUMERIC(18, 6),
    critical_high   NUMERIC(18, 6),
    age_min_years   NUMERIC(5, 2),
    age_max_years   NUMERIC(5, 2),
    sex             lb_patient_sex,
    condition       TEXT,
    effective_date  TIMESTAMPTZ     NOT NULL DEFAULT now(),
    expiry_date     TIMESTAMPTZ,
    source          TEXT,
    active          BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      TEXT            NOT NULL,
    is_deleted      BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id),
    -- Logical constraint: high > low
    CONSTRAINT chk_lb_rr_high_gt_low CHECK (high IS NULL OR low IS NULL OR high > low),
    -- Critical limits must lie outside normal limits
    CONSTRAINT chk_lb_rr_crit_low_lt_low CHECK (critical_low IS NULL OR low IS NULL OR critical_low < low),
    CONSTRAINT chk_lb_rr_crit_high_gt_high CHECK (critical_high IS NULL OR high IS NULL OR critical_high > high)
);

CREATE INDEX IF NOT EXISTS idx_lb_rr_tenant_test        ON lb_reference_range (tenant_id, test_code, analyte);
CREATE INDEX IF NOT EXISTS idx_lb_rr_active             ON lb_reference_range (tenant_id, test_code, active, effective_date DESC)
    WHERE active AND NOT is_deleted;

-- ─── lb_lab_result — Test results with full audit trail ───────────────────────

CREATE TABLE IF NOT EXISTS lb_lab_result (
    id              TEXT                NOT NULL,
    tenant_id       TEXT                NOT NULL,
    order_id        TEXT                NOT NULL REFERENCES lb_lab_order(id),
    specimen_id     TEXT                NOT NULL REFERENCES lb_specimen(id),
    analyte         TEXT                NOT NULL,
    value_numeric   NUMERIC(18, 6),
    value_text      TEXT,
    unit            TEXT                NOT NULL,
    reference_low   NUMERIC(18, 6),
    reference_high  NUMERIC(18, 6),
    critical_low    NUMERIC(18, 6),
    critical_high   NUMERIC(18, 6),
    result_status   lb_result_status    NOT NULL DEFAULT 'preliminary',
    abnormal_flag   lb_abnormal_flag,
    critical_value  BOOLEAN             NOT NULL DEFAULT FALSE,
    delta_check_flag BOOLEAN            NOT NULL DEFAULT FALSE,
    previous_value_numeric  NUMERIC(18, 6),
    previous_value_text     TEXT,
    amendment_of    TEXT                REFERENCES lb_lab_result(id),
    instrument_id   TEXT,
    method          TEXT,
    dilution_factor NUMERIC(8, 4)       CHECK (dilution_factor > 0),
    notes           TEXT,
    performed_by    TEXT                NOT NULL,
    verified_by     TEXT,
    verified_at     TIMESTAMPTZ,
    released_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by      TEXT                NOT NULL,
    is_deleted      BOOLEAN             NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
) PARTITION BY RANGE (created_at);

CREATE TABLE IF NOT EXISTS lb_lab_result_y2025
    PARTITION OF lb_lab_result
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS lb_lab_result_y2026
    PARTITION OF lb_lab_result
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE INDEX IF NOT EXISTS idx_lb_result_tenant         ON lb_lab_result (tenant_id);
CREATE INDEX IF NOT EXISTS idx_lb_result_order          ON lb_lab_result (tenant_id, order_id);
CREATE INDEX IF NOT EXISTS idx_lb_result_specimen       ON lb_lab_result (tenant_id, specimen_id);
CREATE INDEX IF NOT EXISTS idx_lb_result_critical       ON lb_lab_result (tenant_id, critical_value, created_at DESC) WHERE critical_value;
CREATE INDEX IF NOT EXISTS idx_lb_result_status         ON lb_lab_result (tenant_id, result_status, created_at DESC);

-- ─── lb_analyser_interface — Analyser instrument registry ─────────────────────

CREATE TABLE IF NOT EXISTS lb_analyser_interface (
    id                          TEXT                    NOT NULL,
    tenant_id                   TEXT                    NOT NULL,
    name                        TEXT                    NOT NULL,
    model                       TEXT                    NOT NULL,
    serial_number               TEXT                    NOT NULL,
    manufacturer                TEXT                    NOT NULL,
    protocol                    lb_interface_protocol   NOT NULL DEFAULT 'hl7_v2',
    test_categories             TEXT[]                  NOT NULL DEFAULT '{}',
    location                    TEXT                    NOT NULL,
    ip_address                  TEXT,
    port                        INTEGER                 CHECK (port > 0 AND port < 65536),
    connection_string           TEXT,
    status                      lb_instrument_status    NOT NULL DEFAULT 'online',
    calibration_interval_days   INTEGER                 NOT NULL DEFAULT 90 CHECK (calibration_interval_days > 0),
    last_calibrated_at          TIMESTAMPTZ,
    calibration_due_at          TIMESTAMPTZ,
    last_qc_at                  TIMESTAMPTZ,
    last_message_at             TIMESTAMPTZ,
    message_count               BIGINT                  NOT NULL DEFAULT 0,
    created_at                  TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by                  TEXT                    NOT NULL,
    is_deleted                  BOOLEAN                 NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id),
    UNIQUE (tenant_id, serial_number)
);

CREATE INDEX IF NOT EXISTS idx_lb_analyser_tenant       ON lb_analyser_interface (tenant_id);
CREATE INDEX IF NOT EXISTS idx_lb_analyser_status       ON lb_analyser_interface (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_lb_analyser_cal_due      ON lb_analyser_interface (tenant_id, calibration_due_at)
    WHERE status != 'decommissioned' AND calibration_due_at IS NOT NULL;

-- ─── lb_qc_result — Quality control run records ───────────────────────────────

CREATE TABLE IF NOT EXISTS lb_qc_result (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    instrument_id       TEXT            NOT NULL REFERENCES lb_analyser_interface(id),
    test_code           TEXT            NOT NULL,
    lot_number          TEXT            NOT NULL,
    expiry_date         TIMESTAMPTZ,
    level               TEXT            NOT NULL,
    measured_value      NUMERIC(18, 6)  NOT NULL,
    target_value        NUMERIC(18, 6)  NOT NULL,
    sd                  NUMERIC(18, 6)  NOT NULL CHECK (sd > 0),
    z_score             NUMERIC(10, 4)  NOT NULL DEFAULT 0,
    cv_percent          NUMERIC(10, 4)  NOT NULL DEFAULT 0,
    status              lb_qc_status    NOT NULL DEFAULT 'pending_review',
    westgard_violations TEXT[]          NOT NULL DEFAULT '{}',
    reviewed_by         TEXT,
    reviewed_at         TIMESTAMPTZ,
    performed_by        TEXT            NOT NULL,
    notes               TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by          TEXT            NOT NULL,
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_lb_qc_tenant             ON lb_qc_result (tenant_id);
CREATE INDEX IF NOT EXISTS idx_lb_qc_instrument         ON lb_qc_result (tenant_id, instrument_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_qc_test               ON lb_qc_result (tenant_id, test_code, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_qc_status             ON lb_qc_result (tenant_id, status) WHERE NOT is_deleted;

-- ─── lb_critical_value — Critical value notification log ──────────────────────

CREATE TABLE IF NOT EXISTS lb_critical_value (
    id                      TEXT                    NOT NULL,
    tenant_id               TEXT                    NOT NULL,
    result_id               TEXT                    NOT NULL REFERENCES lb_lab_result(id),
    patient_id              TEXT                    NOT NULL,
    analyte                 TEXT                    NOT NULL,
    value_numeric           NUMERIC(18, 6),
    value_text              TEXT,
    unit                    TEXT                    NOT NULL,
    severity                lb_critical_severity    NOT NULL,
    notified_to             TEXT                    NOT NULL,
    notified_by             TEXT                    NOT NULL,
    notification_method     TEXT                    NOT NULL DEFAULT 'phone',
    read_back_confirmed     BOOLEAN                 NOT NULL DEFAULT FALSE,
    acknowledged_by         TEXT,
    acknowledged_at         TIMESTAMPTZ,
    escalated               BOOLEAN                 NOT NULL DEFAULT FALSE,
    escalated_to            TEXT,
    escalated_at            TIMESTAMPTZ,
    notes                   TEXT,
    created_at              TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by              TEXT                    NOT NULL,
    is_deleted              BOOLEAN                 NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_lb_cv_tenant             ON lb_critical_value (tenant_id);
CREATE INDEX IF NOT EXISTS idx_lb_cv_patient            ON lb_critical_value (tenant_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_lb_cv_unacked            ON lb_critical_value (tenant_id, created_at DESC)
    WHERE acknowledged_by IS NULL AND NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_lb_cv_result             ON lb_critical_value (tenant_id, result_id);

-- ─── lb_external_referral — External / reference lab referrals ────────────────

CREATE TABLE IF NOT EXISTS lb_external_referral (
    id                  TEXT                NOT NULL,
    tenant_id           TEXT                NOT NULL,
    order_id            TEXT                NOT NULL REFERENCES lb_lab_order(id),
    specimen_id         TEXT                NOT NULL REFERENCES lb_specimen(id),
    patient_id          TEXT                NOT NULL,
    reference_lab_name  TEXT                NOT NULL,
    reference_lab_code  TEXT                NOT NULL,
    test_code           TEXT                NOT NULL,
    test_name           TEXT                NOT NULL,
    clinical_notes      TEXT,
    expected_tat_hours  INTEGER             CHECK (expected_tat_hours > 0),
    dispatched_by       TEXT                NOT NULL,
    dispatched_at       TIMESTAMPTZ,
    tracking_number     TEXT,
    status              lb_referral_status  NOT NULL DEFAULT 'pending',
    received_at         TIMESTAMPTZ,
    external_result_id  TEXT,
    result_received_at  TIMESTAMPTZ,
    notes               TEXT,
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by          TEXT                NOT NULL,
    is_deleted          BOOLEAN             NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_lb_referral_tenant       ON lb_external_referral (tenant_id);
CREATE INDEX IF NOT EXISTS idx_lb_referral_status       ON lb_external_referral (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_lb_referral_specimen     ON lb_external_referral (tenant_id, specimen_id);
CREATE INDEX IF NOT EXISTS idx_lb_referral_tracking     ON lb_external_referral (tracking_number) WHERE tracking_number IS NOT NULL;

-- ─── lb_instrument_message — Raw analyser interface messages ──────────────────

CREATE TABLE IF NOT EXISTS lb_instrument_message (
    id              TEXT                    NOT NULL,
    tenant_id       TEXT                    NOT NULL,
    instrument_id   TEXT                    NOT NULL REFERENCES lb_analyser_interface(id),
    protocol        lb_interface_protocol   NOT NULL,
    message_type    TEXT                    NOT NULL,
    raw_payload     TEXT                    NOT NULL,
    parsed_results  JSONB                   NOT NULL DEFAULT '[]',
    received_at     TIMESTAMPTZ             NOT NULL DEFAULT now(),
    processed       BOOLEAN                 NOT NULL DEFAULT FALSE,
    error           TEXT,
    PRIMARY KEY (id)
) PARTITION BY RANGE (received_at);

CREATE TABLE IF NOT EXISTS lb_instrument_message_y2025
    PARTITION OF lb_instrument_message
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS lb_instrument_message_y2026
    PARTITION OF lb_instrument_message
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE INDEX IF NOT EXISTS idx_lb_msg_instrument        ON lb_instrument_message (tenant_id, instrument_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_msg_unprocessed       ON lb_instrument_message (tenant_id, received_at)
    WHERE NOT processed;

-- ─── lb_calibration — Instrument calibration records ─────────────────────────

CREATE TABLE IF NOT EXISTS lb_calibration (
    id                  TEXT        NOT NULL,
    tenant_id           TEXT        NOT NULL,
    instrument_id       TEXT        NOT NULL REFERENCES lb_analyser_interface(id),
    calibrated_by       TEXT        NOT NULL,
    calibration_date    TIMESTAMPTZ NOT NULL DEFAULT now(),
    next_due_date       TIMESTAMPTZ NOT NULL,
    notes               TEXT,
    pass_fail           BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT        NOT NULL,
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_lb_cal_instrument        ON lb_calibration (tenant_id, instrument_id, calibration_date DESC);

-- ─── lb_audit_log — Immutable audit trail ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS lb_audit_log (
    id          TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL,
    event_type  TEXT        NOT NULL,
    entity_id   TEXT        NOT NULL,
    actor_id    TEXT,
    details     JSONB       NOT NULL DEFAULT '{}',
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
) PARTITION BY RANGE (occurred_at);

CREATE TABLE IF NOT EXISTS lb_audit_log_y2025
    PARTITION OF lb_audit_log
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS lb_audit_log_y2026
    PARTITION OF lb_audit_log
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE INDEX IF NOT EXISTS idx_lb_audit_tenant          ON lb_audit_log (tenant_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_audit_entity          ON lb_audit_log (tenant_id, entity_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_audit_event           ON lb_audit_log (tenant_id, event_type, occurred_at DESC);

-- ─── updated_at auto-maintenance ──────────────────────────────────────────────

CREATE OR REPLACE FUNCTION lb_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DO $$ DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'lb_lab_test', 'lb_lab_order', 'lb_specimen', 'lb_reference_range',
        'lb_analyser_interface', 'lb_qc_result', 'lb_critical_value',
        'lb_external_referral', 'lb_calibration'
    ]
    LOOP
        EXECUTE format(
            'CREATE OR REPLACE TRIGGER trg_%I_updated_at
             BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION lb_set_updated_at()',
            tbl, tbl
        );
    END LOOP;
END $$;

-- ─── Row-level security helpers ────────────────────────────────────────────────

-- Uncomment and configure for RLS-based tenant isolation in production.
-- ALTER TABLE lb_lab_order ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY tenant_isolation ON lb_lab_order
--     USING (tenant_id = current_setting('app.tenant_id'));
