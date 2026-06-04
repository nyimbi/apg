-- =============================================================================
-- APG Digital Lending — Complete PostgreSQL Schema
-- =============================================================================
-- Run:  psql $DATABASE_URL -f database/schema.sql
--
-- Prefix : ld_
-- Tenancy: every table has tenant_id for row-level isolation
-- Audit  : created_at, updated_at, created_by, is_deleted on every entity table
--
-- © 2025 Datacraft. All rights reserved.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ---------------------------------------------------------------------------
-- Enums
-- ---------------------------------------------------------------------------

DO $$ BEGIN CREATE TYPE ld_product_type AS ENUM (
	'term_loan','revolving','overdraft','microfinance','mortgage',
	'asset_finance','invoice_discounting','bnpl','salary_advance',
	'emergency','agri','sme','group'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_repayment_freq AS ENUM (
	'daily','weekly','biweekly','monthly','quarterly','bullet'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_schedule_type AS ENUM (
	'reducing_balance','flat_rate','bullet','interest_only'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_app_status AS ENUM (
	'draft','submitted','under_review','referred',
	'conditionally_approved','approved','declined',
	'withdrawn','disbursed','expired'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_loan_status AS ENUM (
	'active','settled','written_off','restructured','closed','cancelled'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_offer_status AS ENUM (
	'draft','issued','accepted','rejected','expired','withdrawn'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_repayment_status AS ENUM (
	'pending','partial','paid','overdue','waived','written_off'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_delinquency_status AS ENUM (
	'open','monitoring','collections','legal','resolved','written_off'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_restructure_type AS ENUM (
	'tenor_extension','rate_reduction','capitalise_arrears',
	'payment_holiday','full_restructure'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_collateral_type AS ENUM (
	'property','vehicle','cash','shares','inventory','machinery','land','other'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_collateral_status AS ENUM (
	'pledged','held','released','foreclosed'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_disbursement_rail AS ENUM (
	'bank_transfer','mobile_money','cash','cheque','internal'
); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN CREATE TYPE ld_provision_stage AS ENUM ('stage1','stage2','stage3');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------------
-- ld_loan_product
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_loan_product (
	id                       TEXT PRIMARY KEY,
	tenant_id                TEXT            NOT NULL,
	code                     TEXT            NOT NULL,
	name                     TEXT            NOT NULL,
	product_type             ld_product_type NOT NULL,
	currency                 CHAR(3)         NOT NULL DEFAULT 'KES',
	min_amount               NUMERIC(18,2)   NOT NULL CHECK (min_amount > 0),
	max_amount               NUMERIC(18,2)   NOT NULL CHECK (max_amount > 0),
	min_tenor_months         SMALLINT        NOT NULL CHECK (min_tenor_months >= 1),
	max_tenor_months         SMALLINT        NOT NULL CHECK (max_tenor_months >= 1),
	base_annual_rate         NUMERIC(6,4)    NOT NULL CHECK (base_annual_rate > 0 AND base_annual_rate <= 1),
	repayment_frequency      ld_repayment_freq NOT NULL DEFAULT 'monthly',
	schedule_type            ld_schedule_type  NOT NULL DEFAULT 'reducing_balance',
	processing_fee_pct       NUMERIC(5,4)    NOT NULL DEFAULT 0,
	insurance_fee_pct        NUMERIC(5,4)    NOT NULL DEFAULT 0,
	late_penalty_pct         NUMERIC(5,4)    NOT NULL DEFAULT 0.02,
	early_settlement_fee_pct NUMERIC(5,4)    NOT NULL DEFAULT 0.01,
	max_dsr                  NUMERIC(4,2)    NOT NULL DEFAULT 0.40,
	requires_collateral      BOOLEAN         NOT NULL DEFAULT FALSE,
	requires_guarantor       BOOLEAN         NOT NULL DEFAULT FALSE,
	min_credit_score         SMALLINT        NOT NULL DEFAULT 480,
	is_active                BOOLEAN         NOT NULL DEFAULT TRUE,
	created_at               TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at               TIMESTAMPTZ     NOT NULL DEFAULT now(),
	created_by               TEXT            NOT NULL DEFAULT 'system',
	is_deleted               BOOLEAN         NOT NULL DEFAULT FALSE,
	CONSTRAINT ld_product_tenant_code  UNIQUE (tenant_id, code),
	CONSTRAINT ld_product_amount_check CHECK (max_amount >= min_amount),
	CONSTRAINT ld_product_tenor_check  CHECK (max_tenor_months >= min_tenor_months)
);

CREATE INDEX IF NOT EXISTS idx_ld_product_tenant  ON ld_loan_product (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ld_product_active  ON ld_loan_product (tenant_id, is_active) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_ld_product_type    ON ld_loan_product (product_type);

-- ---------------------------------------------------------------------------
-- ld_borrower
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_borrower (
	id                 TEXT PRIMARY KEY,
	tenant_id          TEXT        NOT NULL,
	customer_reference TEXT        NOT NULL,
	kyc_profile_id     TEXT        NOT NULL,
	country            CHAR(2)     NOT NULL DEFAULT 'KE',
	income_evidence_id TEXT        NOT NULL DEFAULT '',
	consent_reference  TEXT        NOT NULL DEFAULT '',
	is_blacklisted     BOOLEAN     NOT NULL DEFAULT FALSE,
	blacklist_reason   TEXT,
	blacklisted_at     TIMESTAMPTZ,
	created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by         TEXT        NOT NULL DEFAULT 'system',
	is_deleted         BOOLEAN     NOT NULL DEFAULT FALSE,
	CONSTRAINT ld_borrower_tenant_ref UNIQUE (tenant_id, customer_reference)
);

CREATE INDEX IF NOT EXISTS idx_ld_borrower_tenant   ON ld_borrower (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ld_borrower_customer ON ld_borrower (customer_reference);
CREATE INDEX IF NOT EXISTS idx_ld_borrower_kyc      ON ld_borrower (kyc_profile_id);

-- ---------------------------------------------------------------------------
-- ld_credit_score
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_credit_score (
	id                     TEXT PRIMARY KEY,
	tenant_id              TEXT          NOT NULL,
	borrower_id            TEXT          NOT NULL REFERENCES ld_borrower(id),
	application_id         TEXT,
	composite_score        SMALLINT      NOT NULL CHECK (composite_score BETWEEN 300 AND 850),
	risk_grade             CHAR(1)       NOT NULL CHECK (risk_grade IN ('A','B','C','D','E','F')),
	probability_of_default NUMERIC(6,4)  NOT NULL,
	bureau_score           SMALLINT,
	bureau_name            TEXT,
	behavioural_score      SMALLINT,
	demographic_score      SMALLINT,
	payment_ratio          NUMERIC(5,4),
	utilisation_ratio      NUMERIC(5,4),
	defaults_count         SMALLINT      NOT NULL DEFAULT 0,
	fraud_flags            TEXT[]        NOT NULL DEFAULT '{}',
	income_verified        BOOLEAN       NOT NULL DEFAULT FALSE,
	components             JSONB         NOT NULL DEFAULT '{}',
	computed_at            DATE          NOT NULL DEFAULT CURRENT_DATE,
	created_at             TIMESTAMPTZ   NOT NULL DEFAULT now(),
	updated_at             TIMESTAMPTZ   NOT NULL DEFAULT now(),
	created_by             TEXT          NOT NULL DEFAULT 'system',
	is_deleted             BOOLEAN       NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_score_borrower ON ld_credit_score (borrower_id);
CREATE INDEX IF NOT EXISTS idx_ld_score_date     ON ld_credit_score (borrower_id, computed_at DESC);

-- ---------------------------------------------------------------------------
-- ld_loan_application
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_loan_application (
	id                     TEXT PRIMARY KEY,
	tenant_id              TEXT            NOT NULL,
	borrower_id            TEXT            NOT NULL REFERENCES ld_borrower(id),
	product_id             TEXT            NOT NULL REFERENCES ld_loan_product(id),
	requested_amount       NUMERIC(18,2)   NOT NULL CHECK (requested_amount > 0),
	requested_tenor_months SMALLINT        NOT NULL CHECK (requested_tenor_months >= 1),
	currency               CHAR(3)         NOT NULL DEFAULT 'KES',
	purpose                TEXT            NOT NULL,
	monthly_income         NUMERIC(18,2),
	bank_statement_ref     TEXT            NOT NULL DEFAULT '',
	payslip_ref            TEXT            NOT NULL DEFAULT '',
	kyc_ref                TEXT            NOT NULL,
	aml_ref                TEXT            NOT NULL DEFAULT '',
	fraud_ref              TEXT            NOT NULL DEFAULT '',
	status                 ld_app_status   NOT NULL DEFAULT 'submitted',
	notes                  TEXT            NOT NULL DEFAULT '',
	underwriter_id         TEXT,
	decision_date          DATE,
	decline_reason         TEXT,
	credit_score_id        TEXT            REFERENCES ld_credit_score(id),
	required_documents     TEXT[]          NOT NULL DEFAULT '{}',
	submitted_documents    TEXT[]          NOT NULL DEFAULT '{}',
	created_at             TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at             TIMESTAMPTZ     NOT NULL DEFAULT now(),
	created_by             TEXT            NOT NULL DEFAULT 'system',
	is_deleted             BOOLEAN         NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_app_tenant   ON ld_loan_application (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ld_app_borrower ON ld_loan_application (borrower_id);
CREATE INDEX IF NOT EXISTS idx_ld_app_product  ON ld_loan_application (product_id);
CREATE INDEX IF NOT EXISTS idx_ld_app_status   ON ld_loan_application (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_ld_app_created  ON ld_loan_application (tenant_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- ld_loan_offer
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_loan_offer (
	id                  TEXT PRIMARY KEY,
	tenant_id           TEXT              NOT NULL,
	application_id      TEXT              NOT NULL REFERENCES ld_loan_application(id),
	credit_score_id     TEXT              REFERENCES ld_credit_score(id),
	tier                TEXT              NOT NULL DEFAULT 'standard',
	offered_amount      NUMERIC(18,2)     NOT NULL CHECK (offered_amount > 0),
	currency            CHAR(3)           NOT NULL DEFAULT 'KES',
	annual_rate         NUMERIC(6,4)      NOT NULL,
	monthly_rate        NUMERIC(8,6)      NOT NULL,
	tenor_months        SMALLINT          NOT NULL,
	monthly_emi         NUMERIC(18,2)     NOT NULL,
	total_repayable     NUMERIC(18,2)     NOT NULL,
	total_interest      NUMERIC(18,2)     NOT NULL,
	processing_fee      NUMERIC(18,2)     NOT NULL DEFAULT 0,
	insurance_fee       NUMERIC(18,2)     NOT NULL DEFAULT 0,
	total_cost          NUMERIC(18,2)     NOT NULL,
	schedule_type       ld_schedule_type  NOT NULL DEFAULT 'reducing_balance',
	repayment_frequency ld_repayment_freq NOT NULL DEFAULT 'monthly',
	conditions          TEXT[]            NOT NULL DEFAULT '{}',
	status              ld_offer_status   NOT NULL DEFAULT 'issued',
	expiry_date         DATE              NOT NULL,
	accepted_at         TIMESTAMPTZ,
	created_at          TIMESTAMPTZ       NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ       NOT NULL DEFAULT now(),
	created_by          TEXT              NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN           NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_offer_application ON ld_loan_offer (application_id);
CREATE INDEX IF NOT EXISTS idx_ld_offer_status      ON ld_loan_offer (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ld_offer_expiry      ON ld_loan_offer (expiry_date) WHERE status = 'issued';

-- ---------------------------------------------------------------------------
-- ld_loan
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_loan (
	id                    TEXT PRIMARY KEY,
	tenant_id             TEXT              NOT NULL,
	application_id        TEXT              NOT NULL REFERENCES ld_loan_application(id),
	offer_id              TEXT              REFERENCES ld_loan_offer(id),
	borrower_id           TEXT              NOT NULL REFERENCES ld_borrower(id),
	product_id            TEXT              NOT NULL REFERENCES ld_loan_product(id),
	principal             NUMERIC(18,2)     NOT NULL,
	outstanding_principal NUMERIC(18,2)     NOT NULL,
	currency              CHAR(3)           NOT NULL DEFAULT 'KES',
	annual_rate           NUMERIC(6,4)      NOT NULL,
	tenor_months          SMALLINT          NOT NULL,
	schedule_type         ld_schedule_type  NOT NULL DEFAULT 'reducing_balance',
	repayment_frequency   ld_repayment_freq NOT NULL DEFAULT 'monthly',
	disbursement_date     DATE              NOT NULL,
	maturity_date         DATE,
	bank_account          TEXT              NOT NULL,
	disbursement_rail     ld_disbursement_rail NOT NULL DEFAULT 'bank_transfer',
	status                ld_loan_status    NOT NULL DEFAULT 'active',
	max_dpd               SMALLINT          NOT NULL DEFAULT 0,
	provision_stage       ld_provision_stage,
	total_repaid          NUMERIC(18,2)     NOT NULL DEFAULT 0,
	total_interest_paid   NUMERIC(18,2)     NOT NULL DEFAULT 0,
	total_fees_paid       NUMERIC(18,2)     NOT NULL DEFAULT 0,
	restructure_count     SMALLINT          NOT NULL DEFAULT 0,
	closure_reason        TEXT,
	closed_at             TIMESTAMPTZ,
	created_at            TIMESTAMPTZ       NOT NULL DEFAULT now(),
	updated_at            TIMESTAMPTZ       NOT NULL DEFAULT now(),
	created_by            TEXT              NOT NULL DEFAULT 'system',
	is_deleted            BOOLEAN           NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_loan_tenant   ON ld_loan (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ld_loan_borrower ON ld_loan (borrower_id);
CREATE INDEX IF NOT EXISTS idx_ld_loan_product  ON ld_loan (product_id);
CREATE INDEX IF NOT EXISTS idx_ld_loan_status   ON ld_loan (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_ld_loan_disburse ON ld_loan (disbursement_date);
CREATE INDEX IF NOT EXISTS idx_ld_loan_dpd      ON ld_loan (max_dpd DESC) WHERE status = 'active';

-- ---------------------------------------------------------------------------
-- ld_loan_schedule
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_loan_schedule (
	id                TEXT PRIMARY KEY,
	loan_id           TEXT                NOT NULL REFERENCES ld_loan(id) ON DELETE CASCADE,
	tenant_id         TEXT                NOT NULL,
	installment_no    SMALLINT            NOT NULL,
	due_date          DATE                NOT NULL,
	emi               NUMERIC(18,2)       NOT NULL,
	principal_portion NUMERIC(18,2)       NOT NULL,
	interest_portion  NUMERIC(18,2)       NOT NULL,
	opening_balance   NUMERIC(18,2)       NOT NULL,
	closing_balance   NUMERIC(18,2)       NOT NULL,
	status            ld_repayment_status NOT NULL DEFAULT 'pending',
	paid_amount       NUMERIC(18,2)       NOT NULL DEFAULT 0,
	paid_date         DATE,
	dpd               SMALLINT            NOT NULL DEFAULT 0,
	CONSTRAINT ld_schedule_loan_no UNIQUE (loan_id, installment_no)
);

CREATE INDEX IF NOT EXISTS idx_ld_sched_loan    ON ld_loan_schedule (loan_id);
CREATE INDEX IF NOT EXISTS idx_ld_sched_due     ON ld_loan_schedule (due_date) WHERE status IN ('pending','partial');
CREATE INDEX IF NOT EXISTS idx_ld_sched_overdue ON ld_loan_schedule (due_date, loan_id) WHERE status = 'overdue';

-- ---------------------------------------------------------------------------
-- ld_repayment_transaction
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_repayment_transaction (
	id                          TEXT PRIMARY KEY,
	tenant_id                   TEXT          NOT NULL,
	loan_id                     TEXT          NOT NULL REFERENCES ld_loan(id),
	amount                      NUMERIC(18,2) NOT NULL CHECK (amount > 0),
	payment_date                DATE          NOT NULL,
	payment_method              TEXT          NOT NULL,
	reference                   TEXT          NOT NULL,
	channel                     TEXT          NOT NULL DEFAULT 'branch',
	fees_cleared                NUMERIC(18,2) NOT NULL DEFAULT 0,
	interest_cleared            NUMERIC(18,2) NOT NULL DEFAULT 0,
	principal_cleared           NUMERIC(18,2) NOT NULL DEFAULT 0,
	overpayment                 NUMERIC(18,2) NOT NULL DEFAULT 0,
	outstanding_principal_after NUMERIC(18,2) NOT NULL,
	loan_status_after           TEXT          NOT NULL,
	allocations                 JSONB         NOT NULL DEFAULT '[]',
	notes                       TEXT          NOT NULL DEFAULT '',
	created_at                  TIMESTAMPTZ   NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ   NOT NULL DEFAULT now(),
	created_by                  TEXT          NOT NULL DEFAULT 'system',
	is_deleted                  BOOLEAN       NOT NULL DEFAULT FALSE,
	CONSTRAINT ld_repayment_ref_unique UNIQUE (tenant_id, reference)
);

CREATE INDEX IF NOT EXISTS idx_ld_repay_loan ON ld_repayment_transaction (loan_id);
CREATE INDEX IF NOT EXISTS idx_ld_repay_date ON ld_repayment_transaction (payment_date);
CREATE INDEX IF NOT EXISTS idx_ld_repay_ref  ON ld_repayment_transaction (reference);

-- ---------------------------------------------------------------------------
-- ld_delinquency
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_delinquency (
	id                    TEXT PRIMARY KEY,
	tenant_id             TEXT                  NOT NULL,
	loan_id               TEXT                  NOT NULL REFERENCES ld_loan(id),
	borrower_id           TEXT                  NOT NULL REFERENCES ld_borrower(id),
	dpd_days              SMALLINT              NOT NULL CHECK (dpd_days >= 0),
	delinquency_bucket    TEXT                  NOT NULL,
	overdue_amount        NUMERIC(18,2)         NOT NULL,
	currency              CHAR(3)               NOT NULL DEFAULT 'KES',
	status                ld_delinquency_status NOT NULL DEFAULT 'open',
	assigned_collector_id TEXT,
	assigned_lawyer_id    TEXT,
	collection_activities JSONB                 NOT NULL DEFAULT '[]',
	demand_notices        JSONB                 NOT NULL DEFAULT '[]',
	legal_actions         JSONB                 NOT NULL DEFAULT '[]',
	opened_at             DATE                  NOT NULL DEFAULT CURRENT_DATE,
	resolved_at           DATE,
	created_at            TIMESTAMPTZ           NOT NULL DEFAULT now(),
	updated_at            TIMESTAMPTZ           NOT NULL DEFAULT now(),
	created_by            TEXT                  NOT NULL DEFAULT 'system',
	is_deleted            BOOLEAN               NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_delinq_loan      ON ld_delinquency (loan_id);
CREATE INDEX IF NOT EXISTS idx_ld_delinq_borrower  ON ld_delinquency (borrower_id);
CREATE INDEX IF NOT EXISTS idx_ld_delinq_collector ON ld_delinquency (assigned_collector_id) WHERE status NOT IN ('resolved','written_off');
CREATE INDEX IF NOT EXISTS idx_ld_delinq_status    ON ld_delinquency (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ld_delinq_dpd       ON ld_delinquency (dpd_days DESC) WHERE status = 'open';

-- ---------------------------------------------------------------------------
-- ld_restructure
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_restructure (
	id                     TEXT PRIMARY KEY,
	tenant_id              TEXT                 NOT NULL,
	loan_id                TEXT                 NOT NULL REFERENCES ld_loan(id),
	restructure_type       ld_restructure_type  NOT NULL,
	old_annual_rate        NUMERIC(6,4)         NOT NULL,
	new_annual_rate        NUMERIC(6,4)         NOT NULL,
	old_tenor_months       SMALLINT             NOT NULL,
	new_tenor_months       SMALLINT             NOT NULL,
	old_outstanding        NUMERIC(18,2)        NOT NULL,
	new_outstanding        NUMERIC(18,2)        NOT NULL,
	capitalise_arrears     BOOLEAN              NOT NULL DEFAULT FALSE,
	arrears_capitalised    NUMERIC(18,2)        NOT NULL DEFAULT 0,
	payment_holiday_months SMALLINT             NOT NULL DEFAULT 0,
	new_monthly_emi        NUMERIC(18,2)        NOT NULL,
	reason                 TEXT                 NOT NULL,
	approved_by            TEXT                 NOT NULL,
	conditions             TEXT[]               NOT NULL DEFAULT '{}',
	effective_date         DATE                 NOT NULL,
	created_at             TIMESTAMPTZ          NOT NULL DEFAULT now(),
	updated_at             TIMESTAMPTZ          NOT NULL DEFAULT now(),
	created_by             TEXT                 NOT NULL DEFAULT 'system',
	is_deleted             BOOLEAN              NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_restructure_loan ON ld_restructure (loan_id);

-- ---------------------------------------------------------------------------
-- ld_write_off
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_write_off (
	id                TEXT PRIMARY KEY,
	tenant_id         TEXT          NOT NULL,
	loan_id           TEXT          NOT NULL REFERENCES ld_loan(id),
	write_off_amount  NUMERIC(18,2) NOT NULL,
	fees_written_off  NUMERIC(18,2) NOT NULL DEFAULT 0,
	total_written_off NUMERIC(18,2) NOT NULL,
	reason            TEXT          NOT NULL,
	write_off_date    DATE          NOT NULL,
	approved_by       TEXT          NOT NULL,
	recovery_prospect NUMERIC(4,2)  NOT NULL DEFAULT 0,
	recovered_amount  NUMERIC(18,2) NOT NULL DEFAULT 0,
	currency          CHAR(3)       NOT NULL DEFAULT 'KES',
	notes             TEXT          NOT NULL DEFAULT '',
	created_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
	updated_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
	created_by        TEXT          NOT NULL DEFAULT 'system',
	is_deleted        BOOLEAN       NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_writeoff_loan   ON ld_write_off (loan_id);
CREATE INDEX IF NOT EXISTS idx_ld_writeoff_date   ON ld_write_off (write_off_date);
CREATE INDEX IF NOT EXISTS idx_ld_writeoff_tenant ON ld_write_off (tenant_id);

-- ---------------------------------------------------------------------------
-- ld_collateral_item
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_collateral_item (
	id                   TEXT PRIMARY KEY,
	tenant_id            TEXT                   NOT NULL,
	loan_id              TEXT                   NOT NULL REFERENCES ld_loan(id),
	collateral_type      ld_collateral_type     NOT NULL,
	description          TEXT                   NOT NULL,
	market_value         NUMERIC(18,2)          NOT NULL CHECK (market_value > 0),
	forced_sale_value    NUMERIC(18,2)          NOT NULL,
	haircut_pct          NUMERIC(4,2)           NOT NULL,
	currency             CHAR(3)                NOT NULL DEFAULT 'KES',
	registration_number  TEXT                   NOT NULL DEFAULT '',
	valuation_date       DATE,
	valuer_name          TEXT                   NOT NULL DEFAULT '',
	location             TEXT                   NOT NULL DEFAULT '',
	insurance_policy_ref TEXT                   NOT NULL DEFAULT '',
	status               ld_collateral_status   NOT NULL DEFAULT 'held',
	released_by          TEXT,
	release_date         DATE,
	release_reason       TEXT,
	created_at           TIMESTAMPTZ            NOT NULL DEFAULT now(),
	updated_at           TIMESTAMPTZ            NOT NULL DEFAULT now(),
	created_by           TEXT                   NOT NULL DEFAULT 'system',
	is_deleted           BOOLEAN                NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_collateral_loan   ON ld_collateral_item (loan_id);
CREATE INDEX IF NOT EXISTS idx_ld_collateral_status ON ld_collateral_item (status);

-- ---------------------------------------------------------------------------
-- ld_guarantor
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_guarantor (
	id                TEXT PRIMARY KEY,
	tenant_id         TEXT          NOT NULL,
	loan_id           TEXT          NOT NULL REFERENCES ld_loan(id),
	guarantor_name    TEXT          NOT NULL,
	id_number         TEXT          NOT NULL,
	phone             TEXT          NOT NULL,
	email             TEXT          NOT NULL DEFAULT '',
	relationship      TEXT          NOT NULL,
	guaranteed_amount NUMERIC(18,2) NOT NULL,
	currency          CHAR(3)       NOT NULL DEFAULT 'KES',
	consent_ref       TEXT          NOT NULL,
	kyc_ref           TEXT          NOT NULL DEFAULT '',
	is_active         BOOLEAN       NOT NULL DEFAULT TRUE,
	created_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
	updated_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
	created_by        TEXT          NOT NULL DEFAULT 'system',
	is_deleted        BOOLEAN       NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_guarantor_loan ON ld_guarantor (loan_id);
CREATE INDEX IF NOT EXISTS idx_ld_guarantor_id   ON ld_guarantor (id_number);

-- ---------------------------------------------------------------------------
-- ld_loan_fee
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_loan_fee (
	id            TEXT PRIMARY KEY,
	tenant_id     TEXT          NOT NULL,
	loan_id       TEXT          NOT NULL REFERENCES ld_loan(id),
	fee_type      TEXT          NOT NULL,
	amount        NUMERIC(18,2) NOT NULL CHECK (amount > 0),
	paid          NUMERIC(18,2) NOT NULL DEFAULT 0,
	status        TEXT          NOT NULL DEFAULT 'outstanding',
	reason        TEXT          NOT NULL,
	waiver_reason TEXT,
	waived_by     TEXT,
	waived_at     TIMESTAMPTZ,
	charged_at    DATE          NOT NULL DEFAULT CURRENT_DATE,
	created_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
	updated_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
	created_by    TEXT          NOT NULL DEFAULT 'system',
	is_deleted    BOOLEAN       NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ld_fee_loan   ON ld_loan_fee (loan_id);
CREATE INDEX IF NOT EXISTS idx_ld_fee_status ON ld_loan_fee (loan_id, status);

-- ---------------------------------------------------------------------------
-- ld_portfolio_snapshot  (daily IFRS 9 / PAR snapshots for trending)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_portfolio_snapshot (
	id                  TEXT PRIMARY KEY,
	tenant_id           TEXT          NOT NULL,
	snapshot_date       DATE          NOT NULL,
	total_active_loans  INTEGER       NOT NULL DEFAULT 0,
	total_book          NUMERIC(18,2) NOT NULL DEFAULT 0,
	total_disbursed     NUMERIC(18,2) NOT NULL DEFAULT 0,
	portfolio_yield     NUMERIC(6,4)  NOT NULL DEFAULT 0,
	par_30              NUMERIC(6,4)  NOT NULL DEFAULT 0,
	par_60              NUMERIC(6,4)  NOT NULL DEFAULT 0,
	par_90              NUMERIC(6,4)  NOT NULL DEFAULT 0,
	npl_ratio           NUMERIC(6,4)  NOT NULL DEFAULT 0,
	npl_balance         NUMERIC(18,2) NOT NULL DEFAULT 0,
	written_off_total   NUMERIC(18,2) NOT NULL DEFAULT 0,
	stage1_ecl          NUMERIC(18,2) NOT NULL DEFAULT 0,
	stage2_ecl          NUMERIC(18,2) NOT NULL DEFAULT 0,
	stage3_ecl          NUMERIC(18,2) NOT NULL DEFAULT 0,
	total_ecl           NUMERIC(18,2) NOT NULL DEFAULT 0,
	provision_coverage  NUMERIC(6,4)  NOT NULL DEFAULT 0,
	currency            CHAR(3)       NOT NULL DEFAULT 'KES',
	created_at          TIMESTAMPTZ   NOT NULL DEFAULT now(),
	CONSTRAINT ld_snapshot_tenant_date UNIQUE (tenant_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_ld_snapshot_tenant ON ld_portfolio_snapshot (tenant_id, snapshot_date DESC);

-- ---------------------------------------------------------------------------
-- ld_audit_log
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ld_audit_log (
	id           TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
	tenant_id    TEXT        NOT NULL,
	event_type   TEXT        NOT NULL,
	entity_type  TEXT,
	entity_id    TEXT,
	actor_id     TEXT        NOT NULL DEFAULT 'system',
	before_state JSONB,
	after_state  JSONB,
	meta         JSONB       NOT NULL DEFAULT '{}',
	ip_address   INET,
	user_agent   TEXT,
	occurred_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ld_audit_tenant  ON ld_audit_log (tenant_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_ld_audit_entity  ON ld_audit_log (entity_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_ld_audit_event   ON ld_audit_log (event_type);

-- ---------------------------------------------------------------------------
-- updated_at trigger
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION ld_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
	NEW.updated_at = now();
	RETURN NEW;
END $$;

DO $$ DECLARE t TEXT; BEGIN
	FOREACH t IN ARRAY ARRAY[
		'ld_loan_product','ld_borrower','ld_credit_score',
		'ld_loan_application','ld_loan_offer','ld_restructure',
		'ld_write_off','ld_collateral_item','ld_guarantor',
		'ld_loan_fee','ld_repayment_transaction','ld_delinquency'
	]
	LOOP
		EXECUTE format(
			'CREATE OR REPLACE TRIGGER %I_updated_at
			 BEFORE UPDATE ON %I
			 FOR EACH ROW EXECUTE FUNCTION ld_set_updated_at()',
			t, t
		);
	END LOOP;
END $$;

-- ---------------------------------------------------------------------------
-- Useful views
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW ld_active_loan_book AS
SELECT
	l.id, l.tenant_id, l.borrower_id, l.product_id,
	p.name AS product_name, p.product_type,
	l.principal, l.outstanding_principal, l.currency,
	l.annual_rate, l.disbursement_date, l.max_dpd,
	l.provision_stage, l.total_repaid
FROM ld_loan l
JOIN ld_loan_product p ON p.id = l.product_id
WHERE l.status = 'active' AND NOT l.is_deleted;

CREATE OR REPLACE VIEW ld_par_summary AS
SELECT
	tenant_id,
	COUNT(*)                                                              AS total_loans,
	SUM(outstanding_principal)                                            AS total_book,
	SUM(CASE WHEN max_dpd > 30 THEN outstanding_principal ELSE 0 END)    AS par30_balance,
	SUM(CASE WHEN max_dpd > 60 THEN outstanding_principal ELSE 0 END)    AS par60_balance,
	SUM(CASE WHEN max_dpd > 90 THEN outstanding_principal ELSE 0 END)    AS par90_balance,
	ROUND(SUM(CASE WHEN max_dpd > 30 THEN outstanding_principal ELSE 0 END)
		/ NULLIF(SUM(outstanding_principal),0), 4)                        AS par30_ratio
FROM ld_loan
WHERE status = 'active' AND NOT is_deleted
GROUP BY tenant_id;
