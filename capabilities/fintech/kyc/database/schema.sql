-- =============================================================================
-- APG Know Your Customer — PostgreSQL schema
-- © 2025 Datacraft · Author: Nyimbi Odero
--
-- Run: psql $DATABASE_URL -f database/schema.sql
--
-- Design notes:
--   • All tables carry tenant_id for multi-tenant isolation.
--   • Soft-delete via is_deleted boolean (never hard-delete KYC records).
--   • JSONB metadata column on every table for extensibility.
--   • Partial indexes on hot query paths (tenant+status, tenant+customer).
--   • Partitioning hints on high-volume tables (audit_events).
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Generic JSONB record store — used by InMemoryStore / PostgreSQLStore fallback
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

CREATE INDEX IF NOT EXISTS idx_apg_records_tenant
	ON apg_records (collection, tenant_id);

CREATE INDEX IF NOT EXISTS idx_apg_records_data_gin
	ON apg_records USING gin (data);

-- ---------------------------------------------------------------------------
-- KYC Applications
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_applications (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	customer_id         TEXT        NOT NULL,
	customer_type       TEXT        NOT NULL,   -- CustomerType enum
	country_code        TEXT        NOT NULL,   -- ISO-3166-1 alpha-2
	legal_name          TEXT        NOT NULL,
	consent_reference   TEXT        NOT NULL,
	kyc_tier            TEXT        NOT NULL DEFAULT 'standard',
	status              TEXT        NOT NULL DEFAULT 'draft',
	risk_score          INTEGER     NOT NULL DEFAULT 0,
	risk_band           TEXT        NOT NULL DEFAULT 'low',
	is_refugee          BOOLEAN     NOT NULL DEFAULT FALSE,
	is_informal_sector  BOOLEAN     NOT NULL DEFAULT FALSE,
	preferred_language  TEXT        NOT NULL DEFAULT 'en',
	expiry_date         DATE,
	last_verified_at    TIMESTAMPTZ,
	edd_triggered_at    TIMESTAMPTZ,
	dormant_since       TIMESTAMPTZ,
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	metadata            JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_app_tenant_status
	ON kc_applications (tenant_id, status)
	WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_kc_app_customer
	ON kc_applications (tenant_id, customer_id)
	WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_kc_app_expiry
	ON kc_applications (tenant_id, expiry_date)
	WHERE is_deleted = FALSE AND status = 'approved';

CREATE INDEX IF NOT EXISTS idx_kc_app_risk
	ON kc_applications (tenant_id, risk_band)
	WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- ID Documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_documents (
	id                      TEXT        PRIMARY KEY,
	tenant_id               TEXT        NOT NULL,
	application_id          TEXT        NOT NULL REFERENCES kc_applications(id),
	document_type           TEXT        NOT NULL,
	token_reference         TEXT        NOT NULL,   -- vault reference — never raw bytes
	document_number         TEXT        NOT NULL DEFAULT '',
	issuing_country         TEXT        NOT NULL DEFAULT '',
	issuing_authority       TEXT        NOT NULL DEFAULT '',
	issue_date              DATE,
	expiry_date             DATE,
	extracted_name          TEXT        NOT NULL DEFAULT '',
	extracted_dob           DATE,
	extracted_nationality   TEXT        NOT NULL DEFAULT '',
	name_script             TEXT        NOT NULL DEFAULT 'latin',
	name_transliterated     TEXT        NOT NULL DEFAULT '',
	confidence              NUMERIC(5,4) NOT NULL DEFAULT 0.0,
	status                  TEXT        NOT NULL DEFAULT 'pending',
	deceased_check_performed BOOLEAN    NOT NULL DEFAULT FALSE,
	synthetic_fraud_score   NUMERIC(5,4) NOT NULL DEFAULT 0.0,
	ocr_raw                 JSONB       NOT NULL DEFAULT '{}',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	metadata                JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_doc_application
	ON kc_documents (tenant_id, application_id)
	WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_kc_doc_type_status
	ON kc_documents (tenant_id, document_type, status)
	WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- Biometric Data
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_biometrics (
	id              TEXT        PRIMARY KEY,
	tenant_id       TEXT        NOT NULL,
	application_id  TEXT        NOT NULL REFERENCES kc_applications(id),
	biometric_type  TEXT        NOT NULL,
	token_reference TEXT        NOT NULL,
	liveness_score  NUMERIC(5,4) NOT NULL DEFAULT 0.0,
	match_score     NUMERIC(5,4) NOT NULL DEFAULT 0.0,
	spoof_score     NUMERIC(5,4) NOT NULL DEFAULT 0.0,
	capture_device  TEXT        NOT NULL DEFAULT '',
	status          TEXT        NOT NULL DEFAULT 'pending',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	metadata        JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_bio_application
	ON kc_biometrics (tenant_id, application_id)
	WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- Risk Profiles
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_risk_profiles (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	application_id              TEXT        NOT NULL REFERENCES kc_applications(id),
	customer_type               TEXT        NOT NULL,
	country_code                TEXT        NOT NULL,
	risk_score                  INTEGER     NOT NULL DEFAULT 0,
	risk_band                   TEXT        NOT NULL DEFAULT 'low',
	is_pep                      BOOLEAN     NOT NULL DEFAULT FALSE,
	is_sanctioned               BOOLEAN     NOT NULL DEFAULT FALSE,
	is_adverse_media            BOOLEAN     NOT NULL DEFAULT FALSE,
	high_risk_country           BOOLEAN     NOT NULL DEFAULT FALSE,
	high_risk_industry          BOOLEAN     NOT NULL DEFAULT FALSE,
	complex_ownership_structure BOOLEAN     NOT NULL DEFAULT FALSE,
	nominee_shareholders_present BOOLEAN   NOT NULL DEFAULT FALSE,
	score_breakdown             JSONB       NOT NULL DEFAULT '{}',
	edd_required                BOOLEAN     NOT NULL DEFAULT FALSE,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL DEFAULT 'system',
	metadata                    JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_risk_application
	ON kc_risk_profiles (tenant_id, application_id);

CREATE INDEX IF NOT EXISTS idx_kc_risk_band
	ON kc_risk_profiles (tenant_id, risk_band);

-- ---------------------------------------------------------------------------
-- PEP Checks
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_pep_checks (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	application_id      TEXT        NOT NULL,
	full_name           TEXT        NOT NULL,
	date_of_birth       DATE,
	nationality         TEXT        NOT NULL DEFAULT '',
	match_threshold     NUMERIC(4,3) NOT NULL DEFAULT 0.85,
	status              TEXT        NOT NULL DEFAULT 'pending',
	is_hit              BOOLEAN     NOT NULL DEFAULT FALSE,
	match_score         NUMERIC(5,4) NOT NULL DEFAULT 0.0,
	matched_name        TEXT        NOT NULL DEFAULT '',
	pep_category        TEXT        NOT NULL DEFAULT '',
	pep_level           TEXT        NOT NULL DEFAULT '',
	source_list         TEXT        NOT NULL DEFAULT '',
	false_positive_reason TEXT      NOT NULL DEFAULT '',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	metadata            JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_pep_application
	ON kc_pep_checks (tenant_id, application_id);

CREATE INDEX IF NOT EXISTS idx_kc_pep_hits
	ON kc_pep_checks (tenant_id, is_hit)
	WHERE is_hit = TRUE;

-- ---------------------------------------------------------------------------
-- Sanction Checks
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_sanction_checks (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	application_id      TEXT        NOT NULL,
	full_name           TEXT        NOT NULL,
	date_of_birth       DATE,
	nationality         TEXT        NOT NULL DEFAULT '',
	lists_screened      TEXT[]      NOT NULL DEFAULT '{}',
	match_threshold     NUMERIC(4,3) NOT NULL DEFAULT 0.85,
	status              TEXT        NOT NULL DEFAULT 'pending',
	is_hit              BOOLEAN     NOT NULL DEFAULT FALSE,
	matched_lists       TEXT[]      NOT NULL DEFAULT '{}',
	match_score         NUMERIC(5,4) NOT NULL DEFAULT 0.0,
	matched_name        TEXT        NOT NULL DEFAULT '',
	false_positive_reason TEXT      NOT NULL DEFAULT '',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	metadata            JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_sanction_application
	ON kc_sanction_checks (tenant_id, application_id);

CREATE INDEX IF NOT EXISTS idx_kc_sanction_hits
	ON kc_sanction_checks (tenant_id, is_hit)
	WHERE is_hit = TRUE;

-- ---------------------------------------------------------------------------
-- Adverse Media Checks
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_adverse_media (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	application_id      TEXT        NOT NULL,
	full_name           TEXT        NOT NULL,
	search_terms        TEXT[]      NOT NULL DEFAULT '{}',
	categories          TEXT[]      NOT NULL DEFAULT '{}',
	status              TEXT        NOT NULL DEFAULT 'pending',
	is_hit              BOOLEAN     NOT NULL DEFAULT FALSE,
	hit_categories      TEXT[]      NOT NULL DEFAULT '{}',
	article_count       INTEGER     NOT NULL DEFAULT 0,
	oldest_article_date DATE,
	newest_article_date DATE,
	summary             TEXT        NOT NULL DEFAULT '',
	false_positive_reason TEXT      NOT NULL DEFAULT '',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	metadata            JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_amedia_application
	ON kc_adverse_media (tenant_id, application_id);

-- ---------------------------------------------------------------------------
-- Business KYC
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_business_kyc (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	application_id              TEXT        NOT NULL REFERENCES kc_applications(id),
	registered_name             TEXT        NOT NULL,
	trading_name                TEXT        NOT NULL DEFAULT '',
	registration_number         TEXT        NOT NULL,
	registration_country        TEXT        NOT NULL,
	registration_date           DATE,
	industry_code               TEXT        NOT NULL DEFAULT '',
	annual_revenue_usd          NUMERIC(18,2),
	number_of_employees         INTEGER,
	website                     TEXT        NOT NULL DEFAULT '',
	primary_business_activity   TEXT        NOT NULL DEFAULT '',
	has_complex_structure       BOOLEAN     NOT NULL DEFAULT FALSE,
	has_nominee_shareholders    BOOLEAN     NOT NULL DEFAULT FALSE,
	ubo_count                   INTEGER     NOT NULL DEFAULT 0,
	status                      TEXT        NOT NULL DEFAULT 'draft',
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL DEFAULT 'system',
	metadata                    JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_bkyc_application
	ON kc_business_kyc (tenant_id, application_id);

CREATE INDEX IF NOT EXISTS idx_kc_bkyc_registration
	ON kc_business_kyc (tenant_id, registration_number, registration_country);

-- ---------------------------------------------------------------------------
-- UBO Declarations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_ubo_declarations (
	id                      TEXT        PRIMARY KEY,
	tenant_id               TEXT        NOT NULL,
	business_kyc_id         TEXT        NOT NULL REFERENCES kc_business_kyc(id),
	application_id          TEXT        NOT NULL,
	full_name               TEXT        NOT NULL,
	date_of_birth           DATE,
	nationality             TEXT        NOT NULL,
	country_of_residence    TEXT        NOT NULL DEFAULT '',
	ownership_percentage    NUMERIC(6,3) NOT NULL,
	ownership_type          TEXT        NOT NULL DEFAULT 'direct',
	is_nominee              BOOLEAN     NOT NULL DEFAULT FALSE,
	controlling_interest    BOOLEAN     NOT NULL DEFAULT FALSE,
	kyc_status              TEXT        NOT NULL DEFAULT 'draft',
	pep_check_id            TEXT,
	sanction_check_id       TEXT,
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	metadata                JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_ubo_business
	ON kc_ubo_declarations (tenant_id, business_kyc_id)
	WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_kc_ubo_application
	ON kc_ubo_declarations (tenant_id, application_id)
	WHERE is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- KYC Reviews
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_reviews (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	application_id      TEXT        NOT NULL REFERENCES kc_applications(id),
	review_type         TEXT        NOT NULL,
	status              TEXT        NOT NULL DEFAULT 'open',
	decision            TEXT        NOT NULL DEFAULT '',
	assigned_to         TEXT        NOT NULL DEFAULT '',
	notes               TEXT        NOT NULL DEFAULT '',
	opened_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
	completed_at        TIMESTAMPTZ,
	escalated_at        TIMESTAMPTZ,
	escalation_reason   TEXT        NOT NULL DEFAULT '',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	metadata            JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_review_application
	ON kc_reviews (tenant_id, application_id)
	WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_kc_review_open
	ON kc_reviews (tenant_id, assigned_to, status)
	WHERE status IN ('open', 'in_progress') AND is_deleted = FALSE;

-- ---------------------------------------------------------------------------
-- Onboarding Journeys
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_onboarding_journeys (
	id                      TEXT        PRIMARY KEY,
	tenant_id               TEXT        NOT NULL,
	application_id          TEXT        NOT NULL REFERENCES kc_applications(id),
	channel                 TEXT        NOT NULL DEFAULT 'web',
	customer_type           TEXT        NOT NULL,
	status                  TEXT        NOT NULL DEFAULT 'started',
	current_step            TEXT        NOT NULL DEFAULT 'identity',
	steps_completed         TEXT[]      NOT NULL DEFAULT '{}',
	steps_required          TEXT[]      NOT NULL DEFAULT '{}',
	started_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	completed_at            TIMESTAMPTZ,
	abandoned_at            TIMESTAMPTZ,
	time_to_complete_seconds INTEGER,
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	metadata                JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kc_journey_application
	ON kc_onboarding_journeys (tenant_id, application_id);

CREATE INDEX IF NOT EXISTS idx_kc_journey_status
	ON kc_onboarding_journeys (tenant_id, status, channel);

-- ---------------------------------------------------------------------------
-- KYC Audit Events
-- (RANGE-partitioned by month on created_at — hint for DBA to implement)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kc_audit_events (
	id              TEXT        NOT NULL,
	tenant_id       TEXT        NOT NULL,
	event_type      TEXT        NOT NULL,
	actor_id        TEXT        NOT NULL DEFAULT 'system',
	resource_id     TEXT        NOT NULL DEFAULT '',
	resource_type   TEXT        NOT NULL DEFAULT '',
	capability_id   TEXT        NOT NULL DEFAULT 'fintech_kyc',
	payload         JSONB       NOT NULL DEFAULT '{}',
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Default partition — DBA should create monthly partitions for production:
-- CREATE TABLE kc_audit_events_2026_06 PARTITION OF kc_audit_events
--   FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS kc_audit_events_default
	PARTITION OF kc_audit_events DEFAULT;

CREATE INDEX IF NOT EXISTS idx_kc_audit_tenant_type
	ON kc_audit_events (tenant_id, event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_kc_audit_resource
	ON kc_audit_events (tenant_id, resource_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- updated_at trigger (shared)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kc_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
	NEW.updated_at := now();
	RETURN NEW;
END;
$$;

DO $$ DECLARE
	t TEXT;
BEGIN
	FOREACH t IN ARRAY ARRAY[
		'kc_applications', 'kc_documents', 'kc_biometrics', 'kc_risk_profiles',
		'kc_pep_checks', 'kc_sanction_checks', 'kc_adverse_media',
		'kc_business_kyc', 'kc_ubo_declarations', 'kc_reviews',
		'kc_onboarding_journeys'
	] LOOP
		IF NOT EXISTS (
			SELECT 1 FROM pg_trigger
			WHERE tgname = 'trg_' || t || '_updated_at'
		) THEN
			EXECUTE format(
				'CREATE TRIGGER trg_%I_updated_at
				 BEFORE UPDATE ON %I
				 FOR EACH ROW EXECUTE FUNCTION kc_set_updated_at()',
				t, t
			);
		END IF;
	END LOOP;
END $$;
