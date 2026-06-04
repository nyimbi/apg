-- =============================================================================
-- APG Electronic Medical Records — PostgreSQL Schema
-- FHIR R4 aligned, multi-tenant, audit-complete
-- © 2025 Datacraft
-- =============================================================================
-- Run: psql $DATABASE_URL -f database/schema.sql
-- Idempotent: all CREATE ... IF NOT EXISTS / DO $$ ... END $$ patterns.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- fuzzy name search

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS emr;
SET search_path TO emr, public;

-- ---------------------------------------------------------------------------
-- Audit helper — updated_at auto-maintenance
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION emr.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

-- macro to attach the trigger uniformly
CREATE OR REPLACE FUNCTION emr.attach_updated_at(tbl regclass)
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    EXECUTE format(
        'CREATE TRIGGER trg_%s_updated_at
         BEFORE UPDATE ON %s
         FOR EACH ROW EXECUTE FUNCTION emr.set_updated_at()',
        tbl::text, tbl
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END;
$$;

-- =============================================================================
-- 1. PATIENT
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.patient (
    id                  TEXT        PRIMARY KEY,
    tenant_id           TEXT        NOT NULL,
    -- FHIR R4 Patient.identifier[]
    identifiers         JSONB       NOT NULL DEFAULT '[]',
    -- FHIR R4 HumanName
    family_name         TEXT        NOT NULL,
    given_names         TEXT[]      NOT NULL DEFAULT '{}',
    prefix              TEXT[]      NOT NULL DEFAULT '{}',
    suffix              TEXT[]      NOT NULL DEFAULT '{}',
    -- demographics
    birth_date          DATE        NOT NULL,
    gender              TEXT        NOT NULL CHECK (gender IN ('male','female','other','unknown')),
    marital_status      TEXT        NOT NULL DEFAULT 'unknown',
    is_deceased         BOOLEAN     NOT NULL DEFAULT FALSE,
    deceased_date       DATE,
    -- contact
    telecom             JSONB       NOT NULL DEFAULT '[]',
    address             JSONB       NOT NULL DEFAULT '[]',
    language            TEXT        NOT NULL DEFAULT 'en',
    nationality         TEXT        NOT NULL DEFAULT '',
    blood_type          TEXT,
    -- sensitive flags
    mental_health_record BOOLEAN    NOT NULL DEFAULT FALSE,
    -- dedup
    biometric_hash      TEXT,
    -- relationships
    next_of_kin         JSONB       NOT NULL DEFAULT '[]',
    emergency_contact   JSONB,
    -- status
    status              TEXT        NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active','inactive','deceased','merged')),
    merged_into         TEXT        REFERENCES emr.patient(id) ON DELETE SET NULL,
    -- audit
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.patient');

CREATE INDEX IF NOT EXISTS idx_patient_tenant        ON emr.patient (tenant_id);
CREATE INDEX IF NOT EXISTS idx_patient_birth_date    ON emr.patient (birth_date);
CREATE INDEX IF NOT EXISTS idx_patient_biometric     ON emr.patient (biometric_hash) WHERE biometric_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_patient_family_trgm   ON emr.patient USING gin (family_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_patient_status        ON emr.patient (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_patient_identifiers   ON emr.patient USING gin (identifiers);

-- =============================================================================
-- 2. ENCOUNTER
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.encounter (
    id                  TEXT        PRIMARY KEY,
    tenant_id           TEXT        NOT NULL,
    patient_id          TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_type      TEXT        NOT NULL,
    provider_id         TEXT        NOT NULL,
    location_id         TEXT        NOT NULL,
    chief_complaint     TEXT        NOT NULL,
    reason_codes        TEXT[]      NOT NULL DEFAULT '{}',
    status              TEXT        NOT NULL DEFAULT 'in_progress'
                            CHECK (status IN ('planned','arrived','triaged','in_progress',
                                              'on_leave','finished','cancelled')),
    admit_time          TIMESTAMPTZ NOT NULL DEFAULT now(),
    discharge_time      TIMESTAMPTZ,
    discharge_summary_id TEXT,
    icd10_codes         TEXT[]      NOT NULL DEFAULT '{}',
    care_team           TEXT[]      NOT NULL DEFAULT '{}',
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.encounter');

CREATE INDEX IF NOT EXISTS idx_encounter_tenant     ON emr.encounter (tenant_id);
CREATE INDEX IF NOT EXISTS idx_encounter_patient    ON emr.encounter (patient_id);
CREATE INDEX IF NOT EXISTS idx_encounter_status     ON emr.encounter (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_encounter_admit_time ON emr.encounter (admit_time DESC);

-- =============================================================================
-- 3. DIAGNOSIS (per-encounter ICD-10)
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.diagnosis (
    id                  TEXT        PRIMARY KEY,
    tenant_id           TEXT        NOT NULL,
    patient_id          TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_id        TEXT        NOT NULL REFERENCES emr.encounter(id),
    icd10_code          TEXT        NOT NULL,
    description         TEXT        NOT NULL,
    certainty           TEXT        NOT NULL DEFAULT 'confirmed'
                            CHECK (certainty IN ('confirmed','differential','provisional','refuted')),
    is_primary          BOOLEAN     NOT NULL DEFAULT FALSE,
    onset_date          DATE,
    body_site           TEXT,
    laterality          TEXT,
    status              TEXT        NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active','inactive','resolved','chronic','episodic')),
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.diagnosis');

CREATE INDEX IF NOT EXISTS idx_diagnosis_patient    ON emr.diagnosis (patient_id);
CREATE INDEX IF NOT EXISTS idx_diagnosis_encounter  ON emr.diagnosis (encounter_id);
CREATE INDEX IF NOT EXISTS idx_diagnosis_icd10      ON emr.diagnosis (icd10_code);

-- =============================================================================
-- 4. PROBLEM LIST
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.problem (
    id                  TEXT        PRIMARY KEY,
    tenant_id           TEXT        NOT NULL,
    patient_id          TEXT        NOT NULL REFERENCES emr.patient(id),
    icd10_code          TEXT        NOT NULL,
    description         TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active','inactive','resolved','chronic','episodic')),
    onset_date          TIMESTAMPTZ,
    resolved_date       TIMESTAMPTZ,
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.problem');

CREATE INDEX IF NOT EXISTS idx_problem_patient      ON emr.problem (patient_id);
CREATE INDEX IF NOT EXISTS idx_problem_tenant       ON emr.problem (tenant_id);
CREATE INDEX IF NOT EXISTS idx_problem_icd10        ON emr.problem (icd10_code);
CREATE INDEX IF NOT EXISTS idx_problem_status       ON emr.problem (patient_id, status);

-- =============================================================================
-- 5. ALLERGY / INTOLERANCE
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.allergy (
    id                  TEXT        PRIMARY KEY,
    tenant_id           TEXT        NOT NULL,
    patient_id          TEXT        NOT NULL REFERENCES emr.patient(id),
    allergen            TEXT        NOT NULL,
    allergy_type        TEXT        NOT NULL
                            CHECK (allergy_type IN ('drug','food','environmental','contrast','latex','other')),
    severity            TEXT        NOT NULL
                            CHECK (severity IN ('mild','moderate','severe','life_threatening')),
    reaction            TEXT        NOT NULL,
    onset_date          DATE,
    notes               TEXT        NOT NULL DEFAULT '',
    status              TEXT        NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active','inactive','resolved','entered_in_error')),
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.allergy');

CREATE INDEX IF NOT EXISTS idx_allergy_patient      ON emr.allergy (patient_id);
CREATE INDEX IF NOT EXISTS idx_allergy_type         ON emr.allergy (patient_id, allergy_type);
CREATE INDEX IF NOT EXISTS idx_allergy_status       ON emr.allergy (patient_id, status);
CREATE INDEX IF NOT EXISTS idx_allergy_allergen_trgm ON emr.allergy USING gin (allergen gin_trgm_ops);

-- =============================================================================
-- 6. MEDICATION (active medication list)
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.medication (
    id                      TEXT        PRIMARY KEY,
    tenant_id               TEXT        NOT NULL,
    patient_id              TEXT        NOT NULL REFERENCES emr.patient(id),
    drug_name               TEXT        NOT NULL,
    ndc_code                TEXT,
    rxnorm_code             TEXT,
    dose                    TEXT        NOT NULL,
    route                   TEXT        NOT NULL,
    frequency               TEXT        NOT NULL,
    prescriber_id           TEXT        NOT NULL,
    indication_icd10        TEXT,
    status                  TEXT        NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active','discontinued','on_hold',
                                                  'completed','entered_in_error')),
    allergy_check_performed BOOLEAN     NOT NULL DEFAULT FALSE,
    interaction_check_performed BOOLEAN NOT NULL DEFAULT FALSE,
    start_date              TIMESTAMPTZ NOT NULL DEFAULT now(),
    end_date                TIMESTAMPTZ,
    created_by              TEXT        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.medication');

CREATE INDEX IF NOT EXISTS idx_medication_patient   ON emr.medication (patient_id);
CREATE INDEX IF NOT EXISTS idx_medication_status    ON emr.medication (patient_id, status);
CREATE INDEX IF NOT EXISTS idx_medication_drug_trgm ON emr.medication USING gin (drug_name gin_trgm_ops);

-- =============================================================================
-- 7. PRESCRIPTION (pharmacy workflow)
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.prescription (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    patient_id                  TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_id                TEXT        NOT NULL REFERENCES emr.encounter(id),
    prescriber_id               TEXT        NOT NULL,
    drug_name                   TEXT        NOT NULL,
    ndc_code                    TEXT,
    rxnorm_code                 TEXT,
    dose_quantity               NUMERIC     NOT NULL,
    dose_unit                   TEXT        NOT NULL,
    route                       TEXT        NOT NULL,
    frequency                   TEXT        NOT NULL,
    duration_days               INTEGER,
    refills_allowed             INTEGER     NOT NULL DEFAULT 0,
    refills_used                INTEGER     NOT NULL DEFAULT 0,
    dose_per_kg                 NUMERIC,
    is_controlled               BOOLEAN     NOT NULL DEFAULT FALSE,
    dea_schedule                TEXT,
    quantity_dispensed          NUMERIC,
    allergy_check_performed     BOOLEAN     NOT NULL DEFAULT FALSE,
    interaction_check_performed BOOLEAN     NOT NULL DEFAULT FALSE,
    pregnancy_check_performed   BOOLEAN     NOT NULL DEFAULT FALSE,
    renal_dose_adjusted         BOOLEAN     NOT NULL DEFAULT FALSE,
    hepatic_dose_adjusted       BOOLEAN     NOT NULL DEFAULT FALSE,
    indication_icd10            TEXT,
    patient_instructions        TEXT        NOT NULL DEFAULT '',
    pharmacist_notes            TEXT        NOT NULL DEFAULT '',
    status                      TEXT        NOT NULL DEFAULT 'draft'
                                    CHECK (status IN ('draft','active','on_hold',
                                                      'cancelled','completed','stopped','entered_in_error')),
    pharmacist_verified         BOOLEAN     NOT NULL DEFAULT FALSE,
    pharmacist_id               TEXT,
    verified_at                 TIMESTAMPTZ,
    lot_number                  TEXT,
    expiry_date                 TEXT,
    dispensed_at                TIMESTAMPTZ,
    dispensed_by                TEXT,
    last_refill_at              TIMESTAMPTZ,
    last_refilled_by            TEXT,
    created_by                  TEXT        NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.prescription');

CREATE INDEX IF NOT EXISTS idx_rx_patient           ON emr.prescription (patient_id);
CREATE INDEX IF NOT EXISTS idx_rx_encounter         ON emr.prescription (encounter_id);
CREATE INDEX IF NOT EXISTS idx_rx_status            ON emr.prescription (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_rx_controlled        ON emr.prescription (tenant_id, is_controlled) WHERE is_controlled = TRUE;
CREATE INDEX IF NOT EXISTS idx_rx_created           ON emr.prescription (created_at DESC);

-- =============================================================================
-- 8. VITAL SIGNS
-- Partitioned by month for high-volume tables
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.vital_sign (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    patient_id      TEXT        NOT NULL,  -- FK resolved at application layer (partition key)
    encounter_id    TEXT        NOT NULL,
    vital_type      TEXT        NOT NULL,
    value           NUMERIC     NOT NULL,
    value2          NUMERIC,               -- diastolic BP
    unit            TEXT        NOT NULL,
    recorded_by     TEXT        NOT NULL,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    method          TEXT        NOT NULL DEFAULT '',
    position        TEXT        NOT NULL DEFAULT '',
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id, recorded_at)
) PARTITION BY RANGE (recorded_at);

-- Current and next quarter partitions (extend as needed)
CREATE TABLE IF NOT EXISTS emr.vital_sign_2025_q1 PARTITION OF emr.vital_sign
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
CREATE TABLE IF NOT EXISTS emr.vital_sign_2025_q2 PARTITION OF emr.vital_sign
    FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
CREATE TABLE IF NOT EXISTS emr.vital_sign_2025_q3 PARTITION OF emr.vital_sign
    FOR VALUES FROM ('2025-07-01') TO ('2025-10-01');
CREATE TABLE IF NOT EXISTS emr.vital_sign_2025_q4 PARTITION OF emr.vital_sign
    FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');
CREATE TABLE IF NOT EXISTS emr.vital_sign_2026_q1 PARTITION OF emr.vital_sign
    FOR VALUES FROM ('2026-01-01') TO ('2026-04-01');
CREATE TABLE IF NOT EXISTS emr.vital_sign_2026_q2 PARTITION OF emr.vital_sign
    FOR VALUES FROM ('2026-04-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS emr.vital_sign_default PARTITION OF emr.vital_sign DEFAULT;

CREATE INDEX IF NOT EXISTS idx_vital_patient        ON emr.vital_sign (patient_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_vital_type           ON emr.vital_sign (patient_id, vital_type, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_vital_encounter      ON emr.vital_sign (encounter_id);

-- =============================================================================
-- 9. CLINICAL NOTE
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.clinical_note (
    id              TEXT        PRIMARY KEY,
    tenant_id       TEXT        NOT NULL,
    patient_id      TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_id    TEXT        NOT NULL REFERENCES emr.encounter(id),
    note_type       TEXT        NOT NULL,
    author_id       TEXT        NOT NULL,
    content         TEXT        NOT NULL,
    subjective      TEXT,
    objective       TEXT,
    assessment      TEXT,
    plan            TEXT,
    icd10_codes     TEXT[]      NOT NULL DEFAULT '{}',
    status          TEXT        NOT NULL DEFAULT 'draft'
                        CHECK (status IN ('draft','final','amended','entered_in_error')),
    is_sensitive    BOOLEAN     NOT NULL DEFAULT FALSE,
    amendment_of    TEXT        REFERENCES emr.clinical_note(id),
    cosigned_by     TEXT,
    finalized_at    TIMESTAMPTZ,
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.clinical_note');

CREATE INDEX IF NOT EXISTS idx_note_patient         ON emr.clinical_note (patient_id);
CREATE INDEX IF NOT EXISTS idx_note_encounter       ON emr.clinical_note (encounter_id);
CREATE INDEX IF NOT EXISTS idx_note_type            ON emr.clinical_note (note_type);
CREATE INDEX IF NOT EXISTS idx_note_status          ON emr.clinical_note (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_note_sensitive       ON emr.clinical_note (patient_id, is_sensitive) WHERE is_sensitive = TRUE;
-- Full-text search on clinical notes
CREATE INDEX IF NOT EXISTS idx_note_content_fts     ON emr.clinical_note USING gin (to_tsvector('english', content));

-- =============================================================================
-- 10. LAB ORDER
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.lab_order (
    id                      TEXT        PRIMARY KEY,
    tenant_id               TEXT        NOT NULL,
    patient_id              TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_id            TEXT        NOT NULL REFERENCES emr.encounter(id),
    ordering_provider_id    TEXT        NOT NULL,
    test_code               TEXT        NOT NULL,   -- LOINC
    test_name               TEXT        NOT NULL,
    specimen_type           TEXT        NOT NULL DEFAULT '',
    priority                TEXT        NOT NULL DEFAULT 'routine'
                                CHECK (priority IN ('routine','urgent','stat')),
    clinical_indication     TEXT        NOT NULL DEFAULT '',
    status                  TEXT        NOT NULL DEFAULT 'requested'
                                CHECK (status IN ('draft','requested','received','accepted',
                                                  'in_progress','completed','cancelled')),
    accession_number        TEXT,
    collection_time         TIMESTAMPTZ,
    received_time           TIMESTAMPTZ,
    created_by              TEXT        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.lab_order');

CREATE INDEX IF NOT EXISTS idx_lab_order_patient    ON emr.lab_order (patient_id);
CREATE INDEX IF NOT EXISTS idx_lab_order_status     ON emr.lab_order (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_lab_order_priority   ON emr.lab_order (tenant_id, priority);

-- =============================================================================
-- 11. LAB RESULT
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.lab_result (
    id                      TEXT        PRIMARY KEY,
    tenant_id               TEXT        NOT NULL,
    order_id                TEXT        NOT NULL REFERENCES emr.lab_order(id),
    patient_id              TEXT        NOT NULL REFERENCES emr.patient(id),
    test_code               TEXT        NOT NULL,
    test_name               TEXT        NOT NULL,
    value                   TEXT        NOT NULL,
    value_numeric           NUMERIC,
    unit                    TEXT        NOT NULL DEFAULT '',
    reference_range         TEXT        NOT NULL DEFAULT '',
    flag                    TEXT        NOT NULL DEFAULT 'normal'
                                CHECK (flag IN ('normal','low','high','critical_low',
                                               'critical_high','abnormal')),
    result_status           TEXT        NOT NULL DEFAULT 'final'
                                CHECK (result_status IN ('pending','preliminary','final',
                                                         'amended','corrected','cancelled')),
    performing_lab          TEXT        NOT NULL DEFAULT '',
    result_time             TIMESTAMPTZ NOT NULL DEFAULT now(),
    verified_by             TEXT        NOT NULL DEFAULT '',
    critical_notified       BOOLEAN     NOT NULL DEFAULT FALSE,
    critical_notified_at    TIMESTAMPTZ,
    critical_notified_to    TEXT,
    created_by              TEXT        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.lab_result');

CREATE INDEX IF NOT EXISTS idx_lab_result_patient   ON emr.lab_result (patient_id, result_time DESC);
CREATE INDEX IF NOT EXISTS idx_lab_result_order     ON emr.lab_result (order_id);
CREATE INDEX IF NOT EXISTS idx_lab_result_critical  ON emr.lab_result (tenant_id, critical_notified)
    WHERE flag IN ('critical_low','critical_high') AND critical_notified = FALSE;

-- =============================================================================
-- 12. IMAGING ORDER
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.imaging_order (
    id                      TEXT        PRIMARY KEY,
    tenant_id               TEXT        NOT NULL,
    patient_id              TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_id            TEXT        NOT NULL REFERENCES emr.encounter(id),
    ordering_provider_id    TEXT        NOT NULL,
    modality                TEXT        NOT NULL,   -- XR|CT|MRI|US|NM|PET|DEXA|FLUORO
    body_part               TEXT        NOT NULL,
    laterality              TEXT        NOT NULL DEFAULT '',
    cpt_code                TEXT        NOT NULL DEFAULT '',
    clinical_indication     TEXT        NOT NULL DEFAULT '',
    priority                TEXT        NOT NULL DEFAULT 'routine',
    contrast_required       BOOLEAN     NOT NULL DEFAULT FALSE,
    patient_instructions    TEXT        NOT NULL DEFAULT '',
    status                  TEXT        NOT NULL DEFAULT 'requested'
                                CHECK (status IN ('requested','scheduled','in_progress',
                                                  'completed','cancelled')),
    accession_number        TEXT,
    report_id               TEXT,
    reported_at             TIMESTAMPTZ,
    radiologist_id          TEXT,
    impression              TEXT,
    created_by              TEXT        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.imaging_order');

CREATE INDEX IF NOT EXISTS idx_imaging_patient      ON emr.imaging_order (patient_id);
CREATE INDEX IF NOT EXISTS idx_imaging_status       ON emr.imaging_order (tenant_id, status);

-- =============================================================================
-- 13. CARE PLAN
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.care_plan (
    id              TEXT        PRIMARY KEY,
    tenant_id       TEXT        NOT NULL,
    patient_id      TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_id    TEXT        REFERENCES emr.encounter(id),
    title           TEXT        NOT NULL,
    description     TEXT        NOT NULL DEFAULT '',
    goal            TEXT        NOT NULL DEFAULT '',
    icd10_codes     TEXT[]      NOT NULL DEFAULT '{}',
    activities      JSONB       NOT NULL DEFAULT '[]',
    start_date      DATE,
    end_date        DATE,
    care_team       TEXT[]      NOT NULL DEFAULT '{}',
    status          TEXT        NOT NULL DEFAULT 'draft'
                        CHECK (status IN ('draft','active','on_hold','completed','cancelled','revoked')),
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.care_plan');

CREATE INDEX IF NOT EXISTS idx_care_plan_patient    ON emr.care_plan (patient_id);
CREATE INDEX IF NOT EXISTS idx_care_plan_status     ON emr.care_plan (tenant_id, status);

-- =============================================================================
-- 14. REFERRAL
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.referral (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    patient_id                  TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_id                TEXT        NOT NULL REFERENCES emr.encounter(id),
    referring_provider_id       TEXT        NOT NULL,
    referred_to_provider_id     TEXT,
    referred_to_specialty       TEXT        NOT NULL,
    reason                      TEXT        NOT NULL,
    urgency                     TEXT        NOT NULL DEFAULT 'routine'
                                    CHECK (urgency IN ('routine','urgent','emergent')),
    icd10_code                  TEXT        NOT NULL DEFAULT '',
    notes                       TEXT        NOT NULL DEFAULT '',
    status                      TEXT        NOT NULL DEFAULT 'draft'
                                    CHECK (status IN ('draft','active','completed','cancelled','declined')),
    accepted_by                 TEXT,
    appointment_date            DATE,
    outcome_notes               TEXT        NOT NULL DEFAULT '',
    created_by                  TEXT        NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.referral');

CREATE INDEX IF NOT EXISTS idx_referral_patient     ON emr.referral (patient_id);
CREATE INDEX IF NOT EXISTS idx_referral_status      ON emr.referral (tenant_id, status);

-- =============================================================================
-- 15. CONSENT
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.consent (
    id              TEXT        PRIMARY KEY,
    tenant_id       TEXT        NOT NULL,
    patient_id      TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_id    TEXT        REFERENCES emr.encounter(id),
    scope           TEXT        NOT NULL,
    category        TEXT        NOT NULL DEFAULT '',
    policy_rule     TEXT        NOT NULL DEFAULT '',
    granted_to      TEXT[]      NOT NULL DEFAULT '{}',
    exceptions      TEXT[]      NOT NULL DEFAULT '{}',
    start_date      DATE,
    end_date        DATE,
    signed_by       TEXT        NOT NULL,
    witness_id      TEXT        NOT NULL DEFAULT '',
    notes           TEXT        NOT NULL DEFAULT '',
    status          TEXT        NOT NULL DEFAULT 'proposed'
                        CHECK (status IN ('active','inactive','entered_in_error','proposed','rejected')),
    override        BOOLEAN     NOT NULL DEFAULT FALSE,
    guardian_id     TEXT,
    relationship    TEXT,
    minor_consent   BOOLEAN     NOT NULL DEFAULT FALSE,
    valid_until     TEXT,
    verified_at     TIMESTAMPTZ,
    revoked_at      TIMESTAMPTZ,
    obtained_by     TEXT,
    obtained_at     TIMESTAMPTZ,
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.consent');

CREATE INDEX IF NOT EXISTS idx_consent_patient      ON emr.consent (patient_id);
CREATE INDEX IF NOT EXISTS idx_consent_type_status  ON emr.consent (patient_id, scope, status);
CREATE INDEX IF NOT EXISTS idx_consent_override     ON emr.consent (tenant_id, override) WHERE override = TRUE;

-- =============================================================================
-- 16. IMMUNISATION
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.immunisation (
    id                  TEXT        PRIMARY KEY,
    tenant_id           TEXT        NOT NULL,
    patient_id          TEXT        NOT NULL REFERENCES emr.patient(id),
    encounter_id        TEXT        REFERENCES emr.encounter(id),
    vaccine_code        TEXT        NOT NULL,   -- CVX code
    vaccine_name        TEXT        NOT NULL,
    dose_quantity       NUMERIC,
    dose_unit           TEXT        NOT NULL DEFAULT 'mL',
    route               TEXT        NOT NULL DEFAULT '',
    site                TEXT        NOT NULL DEFAULT '',
    lot_number          TEXT        NOT NULL DEFAULT '',
    manufacturer        TEXT        NOT NULL DEFAULT '',
    expiration_date     DATE,
    administered_date   DATE        NOT NULL,
    administered_by     TEXT        NOT NULL,
    notes               TEXT        NOT NULL DEFAULT '',
    status              TEXT        NOT NULL DEFAULT 'completed'
                            CHECK (status IN ('completed','entered_in_error','not_done')),
    created_by          TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.immunisation');

CREATE INDEX IF NOT EXISTS idx_immunisation_patient ON emr.immunisation (patient_id);
CREATE INDEX IF NOT EXISTS idx_immunisation_vaccine ON emr.immunisation (patient_id, vaccine_code);

-- =============================================================================
-- 17. FAMILY HISTORY
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.family_history (
    id              TEXT        PRIMARY KEY,
    tenant_id       TEXT        NOT NULL,
    patient_id      TEXT        NOT NULL REFERENCES emr.patient(id),
    relationship    TEXT        NOT NULL,
    deceased        BOOLEAN     NOT NULL DEFAULT FALSE,
    age_at_death    INTEGER,
    conditions      TEXT[]      NOT NULL DEFAULT '{}',   -- ICD-10 codes
    notes           TEXT        NOT NULL DEFAULT '',
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.family_history');

CREATE INDEX IF NOT EXISTS idx_fhx_patient          ON emr.family_history (patient_id);

-- =============================================================================
-- 18. AUDIT LOG (append-only)
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.audit_log (
    id          BIGSERIAL   PRIMARY KEY,
    tenant_id   TEXT        NOT NULL,
    actor_id    TEXT        NOT NULL,
    event       TEXT        NOT NULL,
    entity_id   TEXT        NOT NULL,
    entity_type TEXT,
    detail      JSONB,
    ip_address  TEXT,
    user_agent  TEXT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_tenant         ON emr.audit_log (tenant_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_entity         ON emr.audit_log (entity_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event          ON emr.audit_log (event, recorded_at DESC);
-- audit_log is append-only — no UPDATE trigger needed

-- =============================================================================
-- 19. CPT PROCEDURE
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.cpt_procedure (
    id              TEXT        PRIMARY KEY,
    tenant_id       TEXT        NOT NULL,
    encounter_id    TEXT        NOT NULL REFERENCES emr.encounter(id),
    patient_id      TEXT        NOT NULL REFERENCES emr.patient(id),
    cpt_code        TEXT        NOT NULL,
    description     TEXT        NOT NULL,
    units           INTEGER     NOT NULL DEFAULT 1 CHECK (units >= 1),
    modifier        TEXT,
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.cpt_procedure');

CREATE INDEX IF NOT EXISTS idx_cpt_encounter        ON emr.cpt_procedure (encounter_id);
CREATE INDEX IF NOT EXISTS idx_cpt_patient          ON emr.cpt_procedure (patient_id);

-- =============================================================================
-- 20. MEDICAL HISTORY (past medical / surgical history)
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.medical_history (
    id              TEXT        PRIMARY KEY,
    tenant_id       TEXT        NOT NULL,
    patient_id      TEXT        NOT NULL REFERENCES emr.patient(id),
    history_type    TEXT        NOT NULL DEFAULT 'medical'
                        CHECK (history_type IN ('medical','surgical','psychiatric','obstetric','social')),
    description     TEXT        NOT NULL,
    icd10_code      TEXT,
    snomed_code     TEXT,
    onset_date      DATE,
    resolved        BOOLEAN     NOT NULL DEFAULT FALSE,
    notes           TEXT        NOT NULL DEFAULT '',
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE
);

SELECT emr.attach_updated_at('emr.medical_history');

CREATE INDEX IF NOT EXISTS idx_mhx_patient          ON emr.medical_history (patient_id);

-- =============================================================================
-- 21. NARCOTICS REGISTER (Schedule II controlled substance log)
-- =============================================================================
CREATE TABLE IF NOT EXISTS emr.narcotics_register (
    id                  TEXT        PRIMARY KEY,
    tenant_id           TEXT        NOT NULL,
    prescription_id     TEXT        NOT NULL REFERENCES emr.prescription(id),
    patient_id          TEXT        NOT NULL REFERENCES emr.patient(id),
    drug_name           TEXT        NOT NULL,
    dea_schedule        TEXT        NOT NULL,
    quantity_dispensed  NUMERIC     NOT NULL,
    lot_number          TEXT        NOT NULL DEFAULT '',
    dispensed_by        TEXT        NOT NULL,
    dispensed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    witnessed_by        TEXT,
    running_balance     NUMERIC,
    notes               TEXT        NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
    -- append-only: no updated_at
);

CREATE INDEX IF NOT EXISTS idx_narcotics_tenant     ON emr.narcotics_register (tenant_id, dispensed_at DESC);
CREATE INDEX IF NOT EXISTS idx_narcotics_patient    ON emr.narcotics_register (patient_id);

-- =============================================================================
-- Views for common queries
-- =============================================================================

-- Active patient medications
CREATE OR REPLACE VIEW emr.v_active_medications AS
SELECT m.*, p.family_name, p.given_names
FROM emr.medication m
JOIN emr.patient p ON p.id = m.patient_id
WHERE m.status = 'active' AND m.is_deleted = FALSE;

-- Unnotified critical lab results
CREATE OR REPLACE VIEW emr.v_unnotified_critical_labs AS
SELECT r.*, lo.ordering_provider_id, lo.test_name AS ordered_test_name
FROM emr.lab_result r
JOIN emr.lab_order lo ON lo.id = r.order_id
WHERE r.flag IN ('critical_low','critical_high')
  AND r.critical_notified = FALSE
  AND r.is_deleted = FALSE;

-- Open encounters per tenant
CREATE OR REPLACE VIEW emr.v_open_encounters AS
SELECT e.*, p.family_name, p.given_names, p.birth_date
FROM emr.encounter e
JOIN emr.patient p ON p.id = e.patient_id
WHERE e.status IN ('in_progress','arrived','triaged')
  AND e.is_deleted = FALSE;

-- =============================================================================
-- Row-level security foundations (enable per deployment)
-- =============================================================================
-- ALTER TABLE emr.patient ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY tenant_isolation ON emr.patient
--     USING (tenant_id = current_setting('app.tenant_id', TRUE));
-- (repeat for each table)
