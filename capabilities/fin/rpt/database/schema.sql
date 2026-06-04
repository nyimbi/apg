-- =============================================================================
-- APG Financial Reporting — PostgreSQL Schema
-- © 2025 Datacraft. Author: Nyimbi Odero
-- =============================================================================

-- All tables use the prefix rpt_ and include tenant isolation,
-- soft-delete, and full audit columns.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- fast ILIKE on names

-- ─────────────────────────────────────────────────────────────────────────────
-- Enumerations
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TYPE rpt_report_status AS ENUM (
	'draft', 'pending', 'generating', 'generated', 'reviewing',
	'approved', 'published', 'distributed', 'archived', 'failed', 'cancelled'
);

CREATE TYPE rpt_statement_type AS ENUM (
	'balance_sheet', 'income_statement', 'cash_flow', 'equity_statement',
	'management_report', 'segment_report', 'consolidated', 'regulatory', 'xbrl', 'custom'
);

CREATE TYPE rpt_accounting_standard AS ENUM (
	'ifrs', 'us_gaap', 'local_gaap', 'management', 'regulatory'
);

CREATE TYPE rpt_consolidation_method AS ENUM (
	'full', 'proportional', 'equity', 'none'
);

CREATE TYPE rpt_output_format AS ENUM (
	'pdf', 'xlsx', 'html', 'json', 'xbrl', 'csv'
);

CREATE TYPE rpt_period_type AS ENUM (
	'daily', 'weekly', 'monthly', 'quarterly', 'semi_annual', 'annual', 'custom'
);

CREATE TYPE rpt_schedule_frequency AS ENUM (
	'daily', 'weekly', 'monthly', 'quarterly', 'annual', 'on_demand'
);

CREATE TYPE rpt_disclosure_type AS ENUM (
	'accounting_policy', 'significant_estimate', 'contingent_liability',
	'related_party', 'segment', 'regulatory', 'risk', 'other'
);

CREATE TYPE rpt_xbrl_taxonomy AS ENUM (
	'ifrs-full', 'us-gaap', 'glf', 'esrs', 'custom'
);

CREATE TYPE rpt_filing_jurisdiction AS ENUM (
	'sec', 'fca', 'esma', 'cma', 'nse', 'jse', 'custom'
);

CREATE TYPE rpt_agent_role AS ENUM (
	'statement_reviewer', 'consolidation_reviewer', 'disclosure_reviewer',
	'distribution_reviewer', 'variance_narrative_reviewer',
	'close_reporting_reviewer', 'xbrl_tagger', 'regulatory_preparer'
);

CREATE TYPE rpt_agent_runtime AS ENUM (
	'codex', 'claude_code', 'opencode', 'pi'
);

CREATE TYPE rpt_kpi_status AS ENUM ('ok', 'warning', 'alert');

CREATE TYPE rpt_narrative_significance AS ENUM ('low', 'medium', 'high');


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_report_definitions  (templates)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_report_definitions (
	id                  TEXT                     PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT                     NOT NULL,
	name                TEXT                     NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
	statement_type      rpt_statement_type       NOT NULL,
	accounting_standard rpt_accounting_standard  NOT NULL DEFAULT 'ifrs',
	description         TEXT,
	owner               TEXT                     NOT NULL,
	currency_code       CHAR(3)                  NOT NULL DEFAULT 'USD',
	comparative_periods SMALLINT                 NOT NULL DEFAULT 1 CHECK (comparative_periods BETWEEN 0 AND 5),
	line_count          INTEGER                  NOT NULL DEFAULT 0,
	status              rpt_report_status        NOT NULL DEFAULT 'draft',
	created_by          TEXT                     NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN                  NOT NULL DEFAULT FALSE,
	created_at          TIMESTAMPTZ              NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ              NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX rpt_defn_name_tenant_uidx
	ON rpt_report_definitions (tenant_id, lower(name))
	WHERE is_deleted = FALSE;

CREATE INDEX rpt_defn_tenant_idx          ON rpt_report_definitions (tenant_id);
CREATE INDEX rpt_defn_statement_type_idx  ON rpt_report_definitions (tenant_id, statement_type);
CREATE INDEX rpt_defn_status_idx          ON rpt_report_definitions (tenant_id, status);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_report_lines
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_report_lines (
	id              TEXT             PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id       TEXT             NOT NULL,
	definition_id   TEXT             NOT NULL REFERENCES rpt_report_definitions (id) ON DELETE CASCADE,
	line_code       TEXT             NOT NULL CHECK (char_length(line_code) BETWEEN 1 AND 50),
	label           TEXT             NOT NULL,
	account_mapping TEXT             NOT NULL,
	sort_order      INTEGER          NOT NULL,
	line_type       TEXT             NOT NULL DEFAULT 'detail',
	formula         TEXT,
	sign_reversal   BOOLEAN          NOT NULL DEFAULT FALSE,
	indent_level    SMALLINT         NOT NULL DEFAULT 0,
	bold            BOOLEAN          NOT NULL DEFAULT FALSE,
	note_reference  TEXT,
	created_by      TEXT             NOT NULL DEFAULT 'system',
	status          TEXT             NOT NULL DEFAULT 'active',
	is_deleted      BOOLEAN          NOT NULL DEFAULT FALSE,
	created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
	updated_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX rpt_line_code_defn_uidx
	ON rpt_report_lines (definition_id, line_code)
	WHERE is_deleted = FALSE;

CREATE INDEX rpt_line_definition_idx  ON rpt_report_lines (definition_id);
CREATE INDEX rpt_line_tenant_idx      ON rpt_report_lines (tenant_id);
CREATE INDEX rpt_line_sort_idx        ON rpt_report_lines (definition_id, sort_order);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_report_periods
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_report_periods (
	id           TEXT              PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id    TEXT              NOT NULL,
	period_code  TEXT              NOT NULL CHECK (char_length(period_code) BETWEEN 1 AND 50),
	name         TEXT              NOT NULL,
	period_type  rpt_period_type   NOT NULL,
	fiscal_year  SMALLINT          NOT NULL,
	start_date   DATE              NOT NULL,
	end_date     DATE              NOT NULL CHECK (end_date > start_date),
	is_current   BOOLEAN           NOT NULL DEFAULT FALSE,
	is_closed    BOOLEAN           NOT NULL DEFAULT FALSE,
	created_by   TEXT              NOT NULL DEFAULT 'system',
	is_deleted   BOOLEAN           NOT NULL DEFAULT FALSE,
	status       TEXT              NOT NULL DEFAULT 'open',
	created_at   TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
	updated_at   TIMESTAMPTZ       NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX rpt_period_code_tenant_uidx
	ON rpt_report_periods (tenant_id, period_code)
	WHERE is_deleted = FALSE;

CREATE INDEX rpt_period_tenant_idx      ON rpt_report_periods (tenant_id);
CREATE INDEX rpt_period_fiscal_year_idx ON rpt_report_periods (tenant_id, fiscal_year);
CREATE INDEX rpt_period_date_range_idx  ON rpt_report_periods (tenant_id, start_date, end_date);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_reports  (generation runs)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_reports (
	id                  TEXT                  PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT                  NOT NULL,
	definition_id       TEXT                  NOT NULL REFERENCES rpt_report_definitions (id),
	period_id           TEXT                  NOT NULL REFERENCES rpt_report_periods (id),
	output_format       rpt_output_format     NOT NULL DEFAULT 'pdf',
	data_quality_score  NUMERIC(5,4)          NOT NULL DEFAULT 1.0 CHECK (data_quality_score BETWEEN 0 AND 1),
	quality_reviewed_by TEXT,
	generation_type     TEXT                  NOT NULL DEFAULT 'standard',
	status              rpt_report_status     NOT NULL DEFAULT 'draft',
	warning_count       INTEGER               NOT NULL DEFAULT 0,
	error_count         INTEGER               NOT NULL DEFAULT 0,
	start_time          TIMESTAMPTZ,
	end_time            TIMESTAMPTZ,
	created_by          TEXT                  NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN               NOT NULL DEFAULT FALSE,
	created_at          TIMESTAMPTZ           NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ           NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_report_tenant_idx        ON rpt_reports (tenant_id);
CREATE INDEX rpt_report_definition_idx    ON rpt_reports (definition_id);
CREATE INDEX rpt_report_period_idx        ON rpt_reports (period_id);
CREATE INDEX rpt_report_status_idx        ON rpt_reports (tenant_id, status);
CREATE INDEX rpt_report_created_at_idx    ON rpt_reports (tenant_id, created_at DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_report_schedules
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_report_schedules (
	id             TEXT                     PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id      TEXT                     NOT NULL,
	definition_id  TEXT                     NOT NULL REFERENCES rpt_report_definitions (id),
	name           TEXT                     NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
	period_type    rpt_period_type          NOT NULL,
	frequency      rpt_schedule_frequency   NOT NULL DEFAULT 'monthly',
	output_format  rpt_output_format        NOT NULL DEFAULT 'pdf',
	recipients     TEXT[]                   NOT NULL DEFAULT '{}',
	auto_publish   BOOLEAN                  NOT NULL DEFAULT FALSE,
	enabled        BOOLEAN                  NOT NULL DEFAULT TRUE,
	last_run_at    TIMESTAMPTZ,
	next_run_at    TIMESTAMPTZ,
	status         rpt_report_status        NOT NULL DEFAULT 'draft',
	created_by     TEXT                     NOT NULL DEFAULT 'system',
	is_deleted     BOOLEAN                  NOT NULL DEFAULT FALSE,
	created_at     TIMESTAMPTZ              NOT NULL DEFAULT NOW(),
	updated_at     TIMESTAMPTZ              NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_schedule_tenant_idx      ON rpt_report_schedules (tenant_id);
CREATE INDEX rpt_schedule_definition_idx  ON rpt_report_schedules (definition_id);
CREATE INDEX rpt_schedule_enabled_idx     ON rpt_report_schedules (tenant_id, enabled, next_run_at);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_report_outputs
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_report_outputs (
	id               TEXT               PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id        TEXT               NOT NULL,
	generation_id    TEXT               NOT NULL REFERENCES rpt_reports (id) ON DELETE CASCADE,
	output_format    rpt_output_format  NOT NULL,
	file_name        TEXT               NOT NULL,
	file_path        TEXT,
	file_size_bytes  BIGINT,
	checksum_sha256  CHAR(64),
	status           TEXT               NOT NULL DEFAULT 'ready',
	created_by       TEXT               NOT NULL DEFAULT 'system',
	is_deleted       BOOLEAN            NOT NULL DEFAULT FALSE,
	created_at       TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
	updated_at       TIMESTAMPTZ        NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_output_generation_idx  ON rpt_report_outputs (generation_id);
CREATE INDEX rpt_output_tenant_idx      ON rpt_report_outputs (tenant_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_financial_statements
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_financial_statements (
	id                    TEXT                     PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id             TEXT                     NOT NULL,
	generation_id         TEXT                     NOT NULL REFERENCES rpt_reports (id),
	period_id             TEXT                     NOT NULL REFERENCES rpt_report_periods (id),
	statement_type        rpt_statement_type       NOT NULL,
	title                 TEXT                     NOT NULL CHECK (char_length(title) BETWEEN 1 AND 300),
	as_of_date            DATE                     NOT NULL,
	currency_code         CHAR(3)                  NOT NULL DEFAULT 'USD',
	reporting_entity      TEXT                     NOT NULL,
	accounting_standard   rpt_accounting_standard  NOT NULL DEFAULT 'ifrs',
	balance_check_passed  BOOLEAN                  NOT NULL DEFAULT TRUE,
	approved_by           TEXT                     NOT NULL,
	narrative_reviewed_by TEXT                     NOT NULL,
	is_final              BOOLEAN                  NOT NULL DEFAULT FALSE,
	is_published          BOOLEAN                  NOT NULL DEFAULT FALSE,
	total_assets          NUMERIC(20,4),
	total_liabilities     NUMERIC(20,4),
	total_equity          NUMERIC(20,4),
	total_revenue         NUMERIC(20,4),
	net_income            NUMERIC(20,4),
	statement_data        JSONB                    NOT NULL DEFAULT '{}',
	status                rpt_report_status        NOT NULL DEFAULT 'draft',
	created_by            TEXT                     NOT NULL DEFAULT 'system',
	is_deleted            BOOLEAN                  NOT NULL DEFAULT FALSE,
	created_at            TIMESTAMPTZ              NOT NULL DEFAULT NOW(),
	updated_at            TIMESTAMPTZ              NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_stmt_tenant_idx         ON rpt_financial_statements (tenant_id);
CREATE INDEX rpt_stmt_period_idx         ON rpt_financial_statements (period_id);
CREATE INDEX rpt_stmt_generation_idx     ON rpt_financial_statements (generation_id);
CREATE INDEX rpt_stmt_type_tenant_idx    ON rpt_financial_statements (tenant_id, statement_type);
CREATE INDEX rpt_stmt_status_idx         ON rpt_financial_statements (tenant_id, status);
CREATE INDEX rpt_stmt_as_of_date_idx     ON rpt_financial_statements (tenant_id, as_of_date DESC);
CREATE INDEX rpt_stmt_data_gin_idx       ON rpt_financial_statements USING gin (statement_data);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_consolidation_groups
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_consolidation_groups (
	id                       TEXT                      PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id                TEXT                      NOT NULL,
	parent_entity            TEXT                      NOT NULL,
	subsidiary_entity        TEXT                      NOT NULL,
	method                   rpt_consolidation_method  NOT NULL,
	ownership_percent        NUMERIC(6,3)              NOT NULL CHECK (ownership_percent BETWEEN 0 AND 100),
	functional_currency      CHAR(3)                   NOT NULL DEFAULT 'USD',
	reporting_currency       CHAR(3)                   NOT NULL DEFAULT 'USD',
	elimination_reviewed_by  TEXT,
	effective_from           DATE,
	effective_to             DATE,
	status                   TEXT                      NOT NULL DEFAULT 'reviewed',
	created_by               TEXT                      NOT NULL DEFAULT 'system',
	is_deleted               BOOLEAN                   NOT NULL DEFAULT FALSE,
	created_at               TIMESTAMPTZ               NOT NULL DEFAULT NOW(),
	updated_at               TIMESTAMPTZ               NOT NULL DEFAULT NOW(),
	CONSTRAINT cg_no_self_consolidation CHECK (lower(parent_entity) <> lower(subsidiary_entity))
);

CREATE INDEX rpt_cg_tenant_idx      ON rpt_consolidation_groups (tenant_id);
CREATE INDEX rpt_cg_parent_idx      ON rpt_consolidation_groups (tenant_id, parent_entity);
CREATE INDEX rpt_cg_subsidiary_idx  ON rpt_consolidation_groups (tenant_id, subsidiary_entity);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_segment_reports
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_segment_reports (
	id                   TEXT                     PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id            TEXT                     NOT NULL,
	generation_id        TEXT                     NOT NULL REFERENCES rpt_reports (id),
	segment_name         TEXT                     NOT NULL CHECK (char_length(segment_name) BETWEEN 1 AND 200),
	segment_code         TEXT                     NOT NULL,
	revenue              NUMERIC(20,4)            NOT NULL DEFAULT 0,
	operating_profit     NUMERIC(20,4)            NOT NULL DEFAULT 0,
	total_assets         NUMERIC(20,4)            NOT NULL DEFAULT 0,
	capital_expenditure  NUMERIC(20,4)            NOT NULL DEFAULT 0,
	depreciation         NUMERIC(20,4)            NOT NULL DEFAULT 0,
	employee_count       INTEGER                  NOT NULL DEFAULT 0,
	period_id            TEXT                     NOT NULL REFERENCES rpt_report_periods (id),
	accounting_standard  rpt_accounting_standard  NOT NULL DEFAULT 'ifrs',
	status               TEXT                     NOT NULL DEFAULT 'draft',
	created_by           TEXT                     NOT NULL DEFAULT 'system',
	is_deleted           BOOLEAN                  NOT NULL DEFAULT FALSE,
	created_at           TIMESTAMPTZ              NOT NULL DEFAULT NOW(),
	updated_at           TIMESTAMPTZ              NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX rpt_seg_code_generation_uidx
	ON rpt_segment_reports (generation_id, segment_code)
	WHERE is_deleted = FALSE;

CREATE INDEX rpt_seg_tenant_idx      ON rpt_segment_reports (tenant_id);
CREATE INDEX rpt_seg_generation_idx  ON rpt_segment_reports (generation_id);
CREATE INDEX rpt_seg_period_idx      ON rpt_segment_reports (period_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_xbrl_tags
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_xbrl_tags (
	id             TEXT               PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id      TEXT               NOT NULL,
	statement_id   TEXT               NOT NULL REFERENCES rpt_financial_statements (id) ON DELETE CASCADE,
	taxonomy       rpt_xbrl_taxonomy  NOT NULL,
	element_name   TEXT               NOT NULL CHECK (char_length(element_name) BETWEEN 1 AND 300),
	element_value  TEXT               NOT NULL,
	context_ref    TEXT               NOT NULL,
	unit_ref       TEXT,
	decimals       SMALLINT,
	period_start   DATE,
	period_end     DATE,
	instant_date   DATE,
	status         TEXT               NOT NULL DEFAULT 'tagged',
	created_by     TEXT               NOT NULL DEFAULT 'system',
	is_deleted     BOOLEAN            NOT NULL DEFAULT FALSE,
	created_at     TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
	updated_at     TIMESTAMPTZ        NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_xbrl_statement_idx  ON rpt_xbrl_tags (statement_id);
CREATE INDEX rpt_xbrl_tenant_idx     ON rpt_xbrl_tags (tenant_id);
CREATE INDEX rpt_xbrl_taxonomy_idx   ON rpt_xbrl_tags (tenant_id, taxonomy);
CREATE INDEX rpt_xbrl_element_idx    ON rpt_xbrl_tags (statement_id, element_name);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_regulatory_submissions
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_regulatory_submissions (
	id                    TEXT                    PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id             TEXT                    NOT NULL,
	statement_id          TEXT                    NOT NULL REFERENCES rpt_financial_statements (id),
	jurisdiction          rpt_filing_jurisdiction NOT NULL,
	form_type             TEXT                    NOT NULL CHECK (char_length(form_type) BETWEEN 1 AND 100),
	filing_deadline       DATE                    NOT NULL,
	prepared_by           TEXT                    NOT NULL,
	reviewed_by           TEXT,
	submission_reference  TEXT,
	submitted_at          TIMESTAMPTZ,
	notes                 TEXT,
	status                TEXT                    NOT NULL DEFAULT 'draft',
	created_by            TEXT                    NOT NULL DEFAULT 'system',
	is_deleted            BOOLEAN                 NOT NULL DEFAULT FALSE,
	created_at            TIMESTAMPTZ             NOT NULL DEFAULT NOW(),
	updated_at            TIMESTAMPTZ             NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_reg_tenant_idx        ON rpt_regulatory_submissions (tenant_id);
CREATE INDEX rpt_reg_statement_idx     ON rpt_regulatory_submissions (statement_id);
CREATE INDEX rpt_reg_jurisdiction_idx  ON rpt_regulatory_submissions (tenant_id, jurisdiction);
CREATE INDEX rpt_reg_deadline_idx      ON rpt_regulatory_submissions (tenant_id, filing_deadline);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_disclosures
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_disclosures (
	id                    TEXT                  PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id             TEXT                  NOT NULL,
	statement_id          TEXT                  NOT NULL REFERENCES rpt_financial_statements (id),
	disclosure_type       rpt_disclosure_type   NOT NULL DEFAULT 'other',
	title                 TEXT                  NOT NULL,
	content               TEXT                  NOT NULL,
	owner                 TEXT                  NOT NULL,
	reviewed_by           TEXT                  NOT NULL,
	regulation_framework  TEXT,
	risk_level            TEXT,
	status                TEXT                  NOT NULL DEFAULT 'reviewed',
	created_by            TEXT                  NOT NULL DEFAULT 'system',
	is_deleted            BOOLEAN               NOT NULL DEFAULT FALSE,
	created_at            TIMESTAMPTZ           NOT NULL DEFAULT NOW(),
	updated_at            TIMESTAMPTZ           NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_disc_tenant_idx     ON rpt_disclosures (tenant_id);
CREATE INDEX rpt_disc_statement_idx  ON rpt_disclosures (statement_id);
CREATE INDEX rpt_disc_type_idx       ON rpt_disclosures (tenant_id, disclosure_type);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_distributions
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_distributions (
	id               TEXT               PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id        TEXT               NOT NULL,
	statement_id     TEXT               NOT NULL REFERENCES rpt_financial_statements (id),
	recipients       TEXT[]             NOT NULL DEFAULT '{}',
	output_format    rpt_output_format  NOT NULL,
	delivery_method  TEXT               NOT NULL DEFAULT 'email',
	distributed_at   TIMESTAMPTZ,
	status           TEXT               NOT NULL DEFAULT 'distributed',
	created_by       TEXT               NOT NULL DEFAULT 'system',
	is_deleted       BOOLEAN            NOT NULL DEFAULT FALSE,
	created_at       TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
	updated_at       TIMESTAMPTZ        NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_dist_tenant_idx     ON rpt_distributions (tenant_id);
CREATE INDEX rpt_dist_statement_idx  ON rpt_distributions (statement_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_agents
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_agents (
	id            TEXT                NOT NULL PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id     TEXT                NOT NULL,
	name          TEXT                NOT NULL,
	runtime       rpt_agent_runtime   NOT NULL,
	role          rpt_agent_role      NOT NULL,
	instructions  TEXT                NOT NULL DEFAULT '',
	status        TEXT                NOT NULL DEFAULT 'active',
	created_by    TEXT                NOT NULL DEFAULT 'system',
	is_deleted    BOOLEAN             NOT NULL DEFAULT FALSE,
	created_at    TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
	updated_at    TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_agent_tenant_idx  ON rpt_agents (tenant_id);
CREATE INDEX rpt_agent_role_idx    ON rpt_agents (tenant_id, role);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_kpi_snapshots  (materialised KPI results for dashboards)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_kpi_snapshots (
	id             TEXT            PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id      TEXT            NOT NULL,
	statement_id   TEXT            REFERENCES rpt_financial_statements (id),
	period_id      TEXT            REFERENCES rpt_report_periods (id),
	as_of_date     DATE            NOT NULL,
	currency_code  CHAR(3)         NOT NULL DEFAULT 'USD',
	kpi_data       JSONB           NOT NULL DEFAULT '{}',
	generated_at   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
	is_deleted     BOOLEAN         NOT NULL DEFAULT FALSE,
	created_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
	updated_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_kpi_tenant_idx    ON rpt_kpi_snapshots (tenant_id);
CREATE INDEX rpt_kpi_stmt_idx      ON rpt_kpi_snapshots (statement_id);
CREATE INDEX rpt_kpi_date_idx      ON rpt_kpi_snapshots (tenant_id, as_of_date DESC);
CREATE INDEX rpt_kpi_data_gin_idx  ON rpt_kpi_snapshots USING gin (kpi_data);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_narrative_reports  (auto-commentary results)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_narrative_reports (
	id             TEXT                       PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id      TEXT                       NOT NULL,
	statement_id   TEXT                       NOT NULL REFERENCES rpt_financial_statements (id),
	period_id      TEXT                       NOT NULL REFERENCES rpt_report_periods (id),
	audience       TEXT                       NOT NULL DEFAULT 'board',
	sections       JSONB                      NOT NULL DEFAULT '[]',
	commentary     JSONB                      NOT NULL DEFAULT '[]',
	model_used     TEXT                       NOT NULL DEFAULT 'rule_based',
	generated_at   TIMESTAMPTZ                NOT NULL DEFAULT NOW(),
	is_deleted     BOOLEAN                    NOT NULL DEFAULT FALSE,
	created_at     TIMESTAMPTZ                NOT NULL DEFAULT NOW(),
	updated_at     TIMESTAMPTZ                NOT NULL DEFAULT NOW()
);

CREATE INDEX rpt_narrative_tenant_idx   ON rpt_narrative_reports (tenant_id);
CREATE INDEX rpt_narrative_stmt_idx     ON rpt_narrative_reports (statement_id);
CREATE INDEX rpt_narrative_period_idx   ON rpt_narrative_reports (period_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- rpt_audit_events  (immutable event log)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE rpt_audit_events (
	id           TEXT         PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
	tenant_id    TEXT         NOT NULL,
	event_type   TEXT         NOT NULL,
	record_id    TEXT,
	actor_id     TEXT,
	payload      JSONB        NOT NULL DEFAULT '{}',
	processor    TEXT         NOT NULL DEFAULT 'bytewax',
	stream       TEXT         NOT NULL DEFAULT 'apg.fin.rpt.lifecycle',
	created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
)
PARTITION BY RANGE (created_at);

-- Monthly partitions (extend as needed)
CREATE TABLE rpt_audit_events_2026_01 PARTITION OF rpt_audit_events
	FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE rpt_audit_events_2026_02 PARTITION OF rpt_audit_events
	FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE rpt_audit_events_2026_03 PARTITION OF rpt_audit_events
	FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE rpt_audit_events_2026_04 PARTITION OF rpt_audit_events
	FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE rpt_audit_events_2026_05 PARTITION OF rpt_audit_events
	FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE rpt_audit_events_2026_06 PARTITION OF rpt_audit_events
	FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE rpt_audit_events_default PARTITION OF rpt_audit_events DEFAULT;

CREATE INDEX rpt_audit_tenant_idx      ON rpt_audit_events (tenant_id, created_at DESC);
CREATE INDEX rpt_audit_event_type_idx  ON rpt_audit_events (tenant_id, event_type);
CREATE INDEX rpt_audit_record_idx      ON rpt_audit_events (record_id) WHERE record_id IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- Triggers — updated_at maintenance
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION rpt_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
	NEW.updated_at = NOW();
	RETURN NEW;
END;
$$;

DO $$
DECLARE
	tbl TEXT;
BEGIN
	FOREACH tbl IN ARRAY ARRAY[
		'rpt_report_definitions', 'rpt_report_lines', 'rpt_report_periods',
		'rpt_reports', 'rpt_report_schedules', 'rpt_report_outputs',
		'rpt_financial_statements', 'rpt_consolidation_groups', 'rpt_segment_reports',
		'rpt_xbrl_tags', 'rpt_regulatory_submissions', 'rpt_disclosures',
		'rpt_distributions', 'rpt_agents', 'rpt_kpi_snapshots', 'rpt_narrative_reports'
	]
	LOOP
		EXECUTE format(
			'CREATE TRIGGER %I BEFORE UPDATE ON %I FOR EACH ROW EXECUTE FUNCTION rpt_set_updated_at()',
			'rpt_updated_at_' || tbl, tbl
		);
	END LOOP;
END;
$$;
