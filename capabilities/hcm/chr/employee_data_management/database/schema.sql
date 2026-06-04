-- =============================================================================
-- APG HCM: Employee Data Management — PostgreSQL Schema
-- © 2025 Datacraft  |  Author: Nyimbi Odero
-- =============================================================================
-- Run:  psql $DATABASE_URL -f database/schema.sql
-- All tables use the EDM_ prefix (Employee Data Management).
-- Tenant isolation: every table carries tenant_id NOT NULL.
-- Audit columns: created_at, updated_at, created_by, is_deleted on every table.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "pg_trgm";    -- trigram full-text search

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS edm;

SET search_path TO edm, public;

-- ===========================================================================
-- JOB GRADES
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_job_grades (
	id              TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id       TEXT        NOT NULL,
	code            TEXT        NOT NULL,
	name            TEXT        NOT NULL,
	level           TEXT        NOT NULL,          -- G1–G10
	min_salary      NUMERIC(18,4) NOT NULL CHECK (min_salary >= 0),
	max_salary      NUMERIC(18,4) NOT NULL CHECK (max_salary >= min_salary),
	currency        CHAR(3)     NOT NULL DEFAULT 'KES',
	description     TEXT,
	is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, code)
);

CREATE INDEX IF NOT EXISTS idx_edm_job_grades_tenant ON edm_job_grades (tenant_id);

-- ===========================================================================
-- DEPARTMENTS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_departments (
	id              TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id       TEXT        NOT NULL,
	code            TEXT        NOT NULL,
	name            TEXT        NOT NULL,
	description     TEXT,
	parent_id       TEXT        REFERENCES edm_departments (id) ON DELETE SET NULL,
	manager_id      TEXT,                          -- FK to edm_employees, added after
	cost_center     TEXT,
	location        TEXT,
	is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
	headcount       INT         NOT NULL DEFAULT 0,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, code)
);

CREATE INDEX IF NOT EXISTS idx_edm_departments_tenant  ON edm_departments (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_departments_parent  ON edm_departments (parent_id);

-- ===========================================================================
-- POSITIONS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_positions (
	id                      TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id               TEXT        NOT NULL,
	code                    TEXT        NOT NULL,
	title                   TEXT        NOT NULL,
	department_id           TEXT        NOT NULL REFERENCES edm_departments (id),
	job_grade_id            TEXT        NOT NULL REFERENCES edm_job_grades (id),
	employment_type         TEXT        NOT NULL DEFAULT 'full_time',
	authorized_headcount    INT         NOT NULL DEFAULT 1 CHECK (authorized_headcount >= 1),
	current_headcount       INT         NOT NULL DEFAULT 0,
	reports_to_position_id  TEXT        REFERENCES edm_positions (id) ON DELETE SET NULL,
	description             TEXT,
	responsibilities        TEXT,
	requirements            TEXT,
	is_exempt               BOOLEAN     NOT NULL DEFAULT TRUE,
	is_active               BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, code)
);

CREATE INDEX IF NOT EXISTS idx_edm_positions_tenant    ON edm_positions (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_positions_dept      ON edm_positions (department_id);
CREATE INDEX IF NOT EXISTS idx_edm_positions_grade     ON edm_positions (job_grade_id);

-- ===========================================================================
-- EMPLOYEES  (core record)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_employees (
	id                  TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT        NOT NULL,
	employee_number     TEXT        NOT NULL,
	first_name          TEXT        NOT NULL,
	middle_name         TEXT,
	last_name           TEXT        NOT NULL,
	preferred_name      TEXT,
	full_name           TEXT        GENERATED ALWAYS AS (
		                TRIM(COALESCE(first_name,'') || ' ' ||
		                COALESCE(middle_name||' ','') ||
		                COALESCE(last_name,''))
		                ) STORED,
	work_email          TEXT        NOT NULL,
	personal_email      TEXT,
	phone_mobile        TEXT,
	phone_home          TEXT,
	phone_work          TEXT,
	gender              TEXT,
	date_of_birth       DATE,
	marital_status      TEXT,
	nationality         TEXT,
	country_of_work     CHAR(2)     NOT NULL DEFAULT 'KE',
	national_id         TEXT,
	address_line1       TEXT,
	address_line2       TEXT,
	city                TEXT,
	country             TEXT,
	department_id       TEXT        NOT NULL REFERENCES edm_departments (id),
	position_id         TEXT        NOT NULL REFERENCES edm_positions (id),
	job_grade_id        TEXT        NOT NULL REFERENCES edm_job_grades (id),
	manager_id          TEXT        REFERENCES edm_employees (id) ON DELETE SET NULL,
	hire_date           DATE        NOT NULL,
	start_date          DATE,
	probation_end_date  DATE,
	termination_date    DATE,
	employment_type     TEXT        NOT NULL DEFAULT 'full_time',
	employment_status   TEXT        NOT NULL DEFAULT 'probation',
	work_mode           TEXT        NOT NULL DEFAULT 'hybrid',
	base_salary         NUMERIC(18,4),
	currency            CHAR(3)     NOT NULL DEFAULT 'KES',
	pay_frequency       TEXT        NOT NULL DEFAULT 'monthly',
	photo_url           TEXT,
	badge_id            TEXT,
	is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, employee_number),
	UNIQUE (tenant_id, work_email)
);

-- Add FK from departments.manager_id after employees table exists
ALTER TABLE edm_departments
	ADD CONSTRAINT fk_dept_manager
	FOREIGN KEY (manager_id) REFERENCES edm_employees (id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_edm_employees_tenant      ON edm_employees (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_employees_dept        ON edm_employees (department_id);
CREATE INDEX IF NOT EXISTS idx_edm_employees_position    ON edm_employees (position_id);
CREATE INDEX IF NOT EXISTS idx_edm_employees_manager     ON edm_employees (manager_id);
CREATE INDEX IF NOT EXISTS idx_edm_employees_status      ON edm_employees (employment_status);
CREATE INDEX IF NOT EXISTS idx_edm_employees_name_trgm   ON edm_employees USING gin (full_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_edm_employees_hire_date   ON edm_employees (tenant_id, hire_date DESC);

-- ===========================================================================
-- QUALIFICATIONS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_qualifications (
	id                  TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT        NOT NULL,
	employee_id         TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	institution         TEXT        NOT NULL,
	qualification_name  TEXT        NOT NULL,
	field_of_study      TEXT,
	level               TEXT        NOT NULL,
	start_year          INT         NOT NULL CHECK (start_year BETWEEN 1950 AND 2100),
	end_year            INT         CHECK (end_year BETWEEN 1950 AND 2100),
	is_completed        BOOLEAN     NOT NULL DEFAULT TRUE,
	grade               TEXT,
	country             CHAR(2)     NOT NULL DEFAULT 'KE',
	document_ref        TEXT,
	verified            BOOLEAN     NOT NULL DEFAULT FALSE,
	verified_by         TEXT,
	verified_at         TIMESTAMPTZ,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_qualifications_employee ON edm_qualifications (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_qualifications_tenant   ON edm_qualifications (tenant_id);

-- ===========================================================================
-- TRAINING
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_training (
	id                  TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT        NOT NULL,
	employee_id         TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	title               TEXT        NOT NULL,
	provider            TEXT,
	training_type       TEXT        NOT NULL DEFAULT 'internal',
	status              TEXT        NOT NULL DEFAULT 'planned',
	start_date          DATE        NOT NULL,
	end_date            DATE,
	duration_hours      NUMERIC(8,2),
	cost                NUMERIC(18,4),
	currency            CHAR(3)     NOT NULL DEFAULT 'KES',
	location            TEXT,
	objectives          TEXT,
	score               NUMERIC(5,2),
	passed              BOOLEAN,
	certificate_ref     TEXT,
	facilitator_notes   TEXT,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_training_employee  ON edm_training (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_training_tenant    ON edm_training (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_training_status    ON edm_training (status);

-- ===========================================================================
-- PERFORMANCE REVIEWS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_performance_reviews (
	id                      TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id               TEXT        NOT NULL,
	employee_id             TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	reviewer_id             TEXT        NOT NULL REFERENCES edm_employees (id),
	review_period_start     DATE        NOT NULL,
	review_period_end       DATE        NOT NULL CHECK (review_period_end > review_period_start),
	review_type             TEXT        NOT NULL DEFAULT 'annual',
	status                  TEXT        NOT NULL DEFAULT 'draft',
	goals                   JSONB       NOT NULL DEFAULT '[]',
	self_rating             TEXT,
	manager_rating          TEXT,
	calibrated_rating       TEXT,
	overall_rating          TEXT,
	strengths               TEXT,
	development_areas       TEXT,
	goals_next_period       JSONB       NOT NULL DEFAULT '[]',
	approved_by             TEXT,
	acknowledged_at         TIMESTAMPTZ,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_perf_reviews_employee ON edm_performance_reviews (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_perf_reviews_tenant   ON edm_performance_reviews (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_perf_reviews_status   ON edm_performance_reviews (status);
CREATE INDEX IF NOT EXISTS idx_edm_perf_reviews_period   ON edm_performance_reviews (review_period_end DESC);

-- ===========================================================================
-- DISCIPLINARY
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_disciplinary (
	id                      TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id               TEXT        NOT NULL,
	employee_id             TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	initiated_by            TEXT        NOT NULL,
	disciplinary_type       TEXT        NOT NULL,
	status                  TEXT        NOT NULL DEFAULT 'initiated',
	incident_date           DATE        NOT NULL,
	incident_description    TEXT        NOT NULL,
	hearing_date            DATE,
	outcome                 TEXT,
	outcome_date            DATE,
	appeal_date             DATE,
	appeal_outcome          TEXT,
	closed_by               TEXT,
	closed_at               TIMESTAMPTZ,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_disciplinary_employee ON edm_disciplinary (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_disciplinary_tenant   ON edm_disciplinary (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_disciplinary_status   ON edm_disciplinary (status);

-- ===========================================================================
-- GRIEVANCES
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_grievances (
	id                      TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id               TEXT        NOT NULL,
	employee_id             TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	category                TEXT        NOT NULL,
	description             TEXT        NOT NULL,
	status                  TEXT        NOT NULL DEFAULT 'submitted',
	is_anonymous            BOOLEAN     NOT NULL DEFAULT FALSE,
	against_employee_id     TEXT        REFERENCES edm_employees (id) ON DELETE SET NULL,
	assigned_to             TEXT,
	investigation_notes     TEXT,
	resolution              TEXT,
	resolved_at             TIMESTAMPTZ,
	withdrawn_reason        TEXT,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_grievances_employee ON edm_grievances (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_grievances_tenant   ON edm_grievances (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_grievances_status   ON edm_grievances (status);

-- ===========================================================================
-- CONTRACTS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_contracts (
	id                      TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id               TEXT        NOT NULL,
	employee_id             TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	contract_type           TEXT        NOT NULL,
	status                  TEXT        NOT NULL DEFAULT 'draft',
	start_date              DATE        NOT NULL,
	end_date                DATE,
	probation_end_date      DATE,
	notice_period_days      INT         NOT NULL DEFAULT 30,
	base_salary             NUMERIC(18,4) NOT NULL,
	currency                CHAR(3)     NOT NULL DEFAULT 'KES',
	pay_frequency           TEXT        NOT NULL DEFAULT 'monthly',
	position_id             TEXT        NOT NULL REFERENCES edm_positions (id),
	job_grade_id            TEXT        NOT NULL REFERENCES edm_job_grades (id),
	document_ref            TEXT,
	signed_by_employee_at   TIMESTAMPTZ,
	signed_by_employer_at   TIMESTAMPTZ,
	terminated_at           TIMESTAMPTZ,
	termination_reason      TEXT,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_contracts_employee ON edm_contracts (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_contracts_tenant   ON edm_contracts (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_contracts_status   ON edm_contracts (status);
CREATE INDEX IF NOT EXISTS idx_edm_contracts_end_date ON edm_contracts (end_date);

-- ===========================================================================
-- BENEFIT ENROLLMENTS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_benefit_enrollments (
	id                      TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id               TEXT        NOT NULL,
	employee_id             TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	benefit_type            TEXT        NOT NULL,
	plan_name               TEXT        NOT NULL,
	provider                TEXT,
	status                  TEXT        NOT NULL DEFAULT 'eligible',
	coverage_start          DATE        NOT NULL,
	coverage_end            DATE,
	employee_contribution   NUMERIC(18,4) NOT NULL DEFAULT 0,
	employer_contribution   NUMERIC(18,4) NOT NULL DEFAULT 0,
	currency                CHAR(3)     NOT NULL DEFAULT 'KES',
	policy_number           TEXT,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_benefits_employee ON edm_benefit_enrollments (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_benefits_tenant   ON edm_benefit_enrollments (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_benefits_type     ON edm_benefit_enrollments (benefit_type);

-- ===========================================================================
-- DEPENDANTS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_dependants (
	id              TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id       TEXT        NOT NULL,
	employee_id     TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	first_name      TEXT        NOT NULL,
	last_name       TEXT        NOT NULL,
	relationship    TEXT        NOT NULL,
	date_of_birth   DATE,
	gender          TEXT,
	national_id     TEXT,
	is_beneficiary  BOOLEAN     NOT NULL DEFAULT FALSE,
	is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_dependants_employee ON edm_dependants (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_dependants_tenant   ON edm_dependants (tenant_id);

-- ===========================================================================
-- EMERGENCY CONTACTS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_emergency_contacts (
	id                  TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT        NOT NULL,
	employee_id         TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	name                TEXT        NOT NULL,
	relationship        TEXT        NOT NULL,
	phone_primary       TEXT        NOT NULL,
	phone_secondary     TEXT,
	email               TEXT,
	address             TEXT,
	is_primary          BOOLEAN     NOT NULL DEFAULT FALSE,
	is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_emergency_contacts_employee ON edm_emergency_contacts (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_emergency_contacts_tenant   ON edm_emergency_contacts (tenant_id);

-- ===========================================================================
-- WORK PERMITS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_work_permits (
	id                      TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id               TEXT        NOT NULL,
	employee_id             TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	nationality             CHAR(2)     NOT NULL,
	permit_type             TEXT        NOT NULL,
	status                  TEXT        NOT NULL DEFAULT 'applied',
	permit_number           TEXT,
	country_of_work         CHAR(2)     NOT NULL DEFAULT 'KE',
	issue_date              DATE,
	expiry_date             DATE,
	renewal_submitted_at    DATE,
	issuing_authority       TEXT,
	document_ref            TEXT,
	rejection_reason        TEXT,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_work_permits_employee ON edm_work_permits (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_work_permits_tenant   ON edm_work_permits (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_work_permits_expiry   ON edm_work_permits (expiry_date);

-- ===========================================================================
-- BACKGROUND CHECKS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_background_checks (
	id              TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id       TEXT        NOT NULL,
	employee_id     TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	check_type      TEXT        NOT NULL,
	provider        TEXT,
	initiated_by    TEXT        NOT NULL,
	status          TEXT        NOT NULL DEFAULT 'initiated',
	consent_given   BOOLEAN     NOT NULL DEFAULT FALSE,
	consent_date    DATE        NOT NULL,
	result_summary  TEXT,
	flags           JSONB       NOT NULL DEFAULT '[]',
	completed_at    TIMESTAMPTZ,
	expires_at      DATE,
	report_ref      TEXT,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_edm_bg_checks_employee ON edm_background_checks (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_bg_checks_tenant   ON edm_background_checks (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_bg_checks_expiry   ON edm_background_checks (expires_at);

-- ===========================================================================
-- EMPLOYMENT HISTORY  (immutable audit log)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_employment_history (
	id                  TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT        NOT NULL,
	employee_id         TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	event_type          TEXT        NOT NULL,
	effective_date      DATE        NOT NULL,
	reason              TEXT,
	prev_department_id  TEXT,
	new_department_id   TEXT,
	prev_position_id    TEXT,
	new_position_id     TEXT,
	prev_job_grade_id   TEXT,
	new_job_grade_id    TEXT,
	prev_manager_id     TEXT,
	new_manager_id      TEXT,
	prev_salary         NUMERIC(18,4),
	new_salary          NUMERIC(18,4),
	prev_status         TEXT,
	new_status          TEXT,
	approved_by         TEXT,
	approved_at         TIMESTAMPTZ,
	notes               TEXT,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
) PARTITION BY RANGE (effective_date);

-- Create initial partitions
CREATE TABLE IF NOT EXISTS edm_employment_history_2024
	PARTITION OF edm_employment_history
	FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS edm_employment_history_2025
	PARTITION OF edm_employment_history
	FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS edm_employment_history_2026
	PARTITION OF edm_employment_history
	FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE TABLE IF NOT EXISTS edm_employment_history_default
	PARTITION OF edm_employment_history DEFAULT;

CREATE INDEX IF NOT EXISTS idx_edm_history_employee ON edm_employment_history (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_history_tenant   ON edm_employment_history (tenant_id);
CREATE INDEX IF NOT EXISTS idx_edm_history_date     ON edm_employment_history (effective_date DESC);

-- ===========================================================================
-- ONBOARDING CHECKLISTS
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_onboarding_checklists (
	id              TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id       TEXT        NOT NULL,
	employee_id     TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	items           JSONB       NOT NULL DEFAULT '[]',
	completed_at    TIMESTAMPTZ,
	completion_pct  NUMERIC(5,2) NOT NULL DEFAULT 0.0,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, employee_id)
);

CREATE INDEX IF NOT EXISTS idx_edm_onboarding_employee ON edm_onboarding_checklists (employee_id);
CREATE INDEX IF NOT EXISTS idx_edm_onboarding_tenant   ON edm_onboarding_checklists (tenant_id);

-- ===========================================================================
-- SUCCESSION PLAN
-- ===========================================================================
CREATE TABLE IF NOT EXISTS edm_succession_plans (
	id                  TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT        NOT NULL,
	position_id         TEXT        NOT NULL REFERENCES edm_positions (id),
	candidate_id        TEXT        NOT NULL REFERENCES edm_employees (id) ON DELETE CASCADE,
	readiness           TEXT        NOT NULL DEFAULT 'development',
	readiness_score     NUMERIC(5,2) NOT NULL DEFAULT 0.0,
	retention_risk      NUMERIC(5,4) NOT NULL DEFAULT 0.0,
	gap_areas           JSONB       NOT NULL DEFAULT '[]',
	notes               TEXT,
	is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, position_id, candidate_id)
);

CREATE INDEX IF NOT EXISTS idx_edm_succession_position  ON edm_succession_plans (position_id);
CREATE INDEX IF NOT EXISTS idx_edm_succession_candidate ON edm_succession_plans (candidate_id);
CREATE INDEX IF NOT EXISTS idx_edm_succession_tenant    ON edm_succession_plans (tenant_id);

-- ===========================================================================
-- Auto-update updated_at trigger
-- ===========================================================================
CREATE OR REPLACE FUNCTION edm_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
	NEW.updated_at = NOW();
	RETURN NEW;
END;
$$;

DO $$
DECLARE tbl TEXT;
BEGIN
	FOREACH tbl IN ARRAY ARRAY[
		'edm_job_grades','edm_departments','edm_positions','edm_employees',
		'edm_qualifications','edm_training','edm_performance_reviews',
		'edm_disciplinary','edm_grievances','edm_contracts',
		'edm_benefit_enrollments','edm_dependants','edm_emergency_contacts',
		'edm_work_permits','edm_background_checks',
		'edm_onboarding_checklists','edm_succession_plans'
	]
	LOOP
		EXECUTE FORMAT(
			'CREATE TRIGGER trg_%s_updated_at
			 BEFORE UPDATE ON %s
			 FOR EACH ROW EXECUTE FUNCTION edm_set_updated_at()',
			tbl, tbl
		);
	END LOOP;
END;
$$;
