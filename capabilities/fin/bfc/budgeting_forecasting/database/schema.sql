-- =============================================================================
-- APG Budgeting & Forecasting — PostgreSQL Schema
-- © 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
-- Run: psql $DATABASE_URL -f database/schema.sql
-- =============================================================================

-- Enable uuid-ossp for any uuid fallbacks
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for ILIKE acceleration

-- =============================================================================
-- ENUMERATIONS
-- =============================================================================

DO $$ BEGIN
    CREATE TYPE bfc_budget_type AS ENUM (
        'annual','quarterly','monthly','rolling','project','capital',
        'operational','zero_based'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_budget_status AS ENUM (
        'draft','submitted','under_review','approved','active',
        'locked','closed','cancelled'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_line_type AS ENUM (
        'revenue','expense','capital','transfer','allocation','contingency'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_distribution_method AS ENUM (
        'equal','top_down','bottom_up','zero_based','seasonal',
        'weighted','driver_based'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_forecast_type AS ENUM (
        'revenue','expense','cash_flow','demand','integrated','scenario'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_forecast_method AS ENUM (
        'statistical','ml','hybrid','judgmental','ensemble',
        'driver_based','rolling','ai'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_forecast_status AS ENUM (
        'draft','generating','completed','published','archived','failed'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_variance_type AS ENUM (
        'favorable','unfavorable','neutral'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_significance AS ENUM (
        'critical','high','medium','low','minimal'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_scenario_type AS ENUM (
        'base','optimistic','pessimistic','stress','what_if','monte_carlo'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_approval_status AS ENUM (
        'pending','approved','rejected','requires_revision','delegated'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_version_status AS ENUM (
        'working','baseline','archived'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE bfc_driver_type AS ENUM (
        'volume','price','headcount','exchange_rate','inflation','custom'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- =============================================================================
-- TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- bfc_budget_templates
-- Must be created before budgets (FK reference)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_budget_templates (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    name                TEXT            NOT NULL CHECK (length(trim(name)) > 0),
    description         TEXT,
    budget_type         bfc_budget_type NOT NULL,
    line_definitions    JSONB           NOT NULL DEFAULT '[]',
    distribution_rules  JSONB           NOT NULL DEFAULT '{}',
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    industry            TEXT,
    tags                TEXT[]          NOT NULL DEFAULT '{}',
    usage_count         INTEGER         NOT NULL DEFAULT 0,
    -- audit
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by          TEXT            NOT NULL,
    updated_by          TEXT            NOT NULL,
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT,
    PRIMARY KEY (id),
    UNIQUE (tenant_id, name)
);

CREATE INDEX IF NOT EXISTS idx_bfc_templates_tenant
    ON bfc_budget_templates (tenant_id) WHERE NOT is_deleted;

-- -----------------------------------------------------------------------------
-- bfc_budgets
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_budgets (
    id                  TEXT                NOT NULL,
    tenant_id           TEXT                NOT NULL,
    name                TEXT                NOT NULL CHECK (length(trim(name)) > 0),
    description         TEXT,
    budget_type         bfc_budget_type     NOT NULL DEFAULT 'annual',
    status              bfc_budget_status   NOT NULL DEFAULT 'draft',
    fiscal_year         INTEGER             NOT NULL,
    period_start        DATE                NOT NULL,
    period_end          DATE                NOT NULL,
    currency_code       CHAR(3)             NOT NULL DEFAULT 'USD',
    owner_id            TEXT                NOT NULL,
    department_id       TEXT,
    cost_center_id      TEXT,
    template_id         TEXT REFERENCES bfc_budget_templates (id) ON DELETE SET NULL,
    total_revenue       NUMERIC(20,4)       NOT NULL DEFAULT 0,
    total_expense       NUMERIC(20,4)       NOT NULL DEFAULT 0,
    net_amount          NUMERIC(20,4)       NOT NULL DEFAULT 0,
    version             INTEGER             NOT NULL DEFAULT 1,
    locked_at           TIMESTAMPTZ,
    approved_at         TIMESTAMPTZ,
    approved_by         TEXT,
    submitted_by        TEXT,
    submitted_at        TIMESTAMPTZ,
    notes               TEXT,
    tags                TEXT[]              NOT NULL DEFAULT '{}',
    metadata            JSONB               NOT NULL DEFAULT '{}',
    -- audit
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by          TEXT                NOT NULL,
    updated_by          TEXT                NOT NULL,
    is_deleted          BOOLEAN             NOT NULL DEFAULT FALSE,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT,
    -- constraints
    PRIMARY KEY (id),
    CHECK (period_end > period_start),
    CHECK (fiscal_year BETWEEN 1900 AND 2100)
);

CREATE INDEX IF NOT EXISTS idx_bfc_budgets_tenant_status
    ON bfc_budgets (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_budgets_fiscal_year
    ON bfc_budgets (tenant_id, fiscal_year) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_budgets_owner
    ON bfc_budgets (tenant_id, owner_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_budgets_period
    ON bfc_budgets (tenant_id, period_start, period_end) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_budgets_metadata
    ON bfc_budgets USING gin (metadata);

-- -----------------------------------------------------------------------------
-- bfc_budget_versions
-- Immutable snapshots — never updated after insert
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_budget_versions (
    id                  TEXT                NOT NULL,
    tenant_id           TEXT                NOT NULL,
    budget_id           TEXT                NOT NULL REFERENCES bfc_budgets (id) ON DELETE CASCADE,
    version_number      INTEGER             NOT NULL,
    version_label       TEXT                NOT NULL,
    status              bfc_version_status  NOT NULL DEFAULT 'working',
    snapshot_data       JSONB               NOT NULL DEFAULT '{}',
    total_revenue       NUMERIC(20,4)       NOT NULL DEFAULT 0,
    total_expense       NUMERIC(20,4)       NOT NULL DEFAULT 0,
    net_amount          NUMERIC(20,4)       NOT NULL DEFAULT 0,
    change_summary      TEXT,
    notes               TEXT,
    -- audit (no updated_* — versions are immutable)
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by          TEXT                NOT NULL,
    updated_by          TEXT                NOT NULL,
    is_deleted          BOOLEAN             NOT NULL DEFAULT FALSE,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT,
    PRIMARY KEY (id),
    UNIQUE (budget_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_bfc_versions_budget
    ON bfc_budget_versions (budget_id, version_number);

-- -----------------------------------------------------------------------------
-- bfc_budget_lines
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_budget_lines (
    id                      TEXT                        NOT NULL,
    tenant_id               TEXT                        NOT NULL,
    budget_id               TEXT                        NOT NULL REFERENCES bfc_budgets (id) ON DELETE CASCADE,
    line_number             INTEGER                     NOT NULL,
    description             TEXT                        NOT NULL,
    line_type               bfc_line_type               NOT NULL,
    account_code            TEXT                        NOT NULL,
    gl_account              TEXT,
    department_code         TEXT,
    cost_center_code        TEXT,
    project_code            TEXT,
    period_start            DATE                        NOT NULL,
    period_end              DATE                        NOT NULL,
    distribution_method     bfc_distribution_method     NOT NULL DEFAULT 'equal',
    budgeted_amount         NUMERIC(20,4)               NOT NULL CHECK (budgeted_amount >= 0),
    committed_amount        NUMERIC(20,4)               NOT NULL DEFAULT 0 CHECK (committed_amount >= 0),
    actual_amount           NUMERIC(20,4)               NOT NULL DEFAULT 0 CHECK (actual_amount >= 0),
    variance_amount         NUMERIC(20,4)               NOT NULL DEFAULT 0,
    variance_pct            NUMERIC(10,4)               NOT NULL DEFAULT 0,
    -- 12-slot monthly breakdown stored as JSONB array of strings (Decimal-safe)
    month_amounts           JSONB                       NOT NULL DEFAULT '["0","0","0","0","0","0","0","0","0","0","0","0"]',
    notes                   TEXT,
    tags                    TEXT[]                      NOT NULL DEFAULT '{}',
    metadata                JSONB                       NOT NULL DEFAULT '{}',
    -- audit
    created_at              TIMESTAMPTZ                 NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ                 NOT NULL DEFAULT now(),
    created_by              TEXT                        NOT NULL,
    updated_by              TEXT                        NOT NULL,
    is_deleted              BOOLEAN                     NOT NULL DEFAULT FALSE,
    deleted_at              TIMESTAMPTZ,
    deleted_by              TEXT,
    PRIMARY KEY (id),
    UNIQUE (budget_id, line_number),
    CHECK (period_end >= period_start)
);

CREATE INDEX IF NOT EXISTS idx_bfc_lines_budget
    ON bfc_budget_lines (budget_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_lines_account
    ON bfc_budget_lines (tenant_id, account_code) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_lines_type
    ON bfc_budget_lines (budget_id, line_type) WHERE NOT is_deleted;

-- -----------------------------------------------------------------------------
-- bfc_budget_approvals
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_budget_approvals (
    id                  TEXT                    NOT NULL,
    tenant_id           TEXT                    NOT NULL,
    budget_id           TEXT                    NOT NULL REFERENCES bfc_budgets (id) ON DELETE CASCADE,
    approver_id         TEXT                    NOT NULL,
    approver_name       TEXT                    NOT NULL,
    approver_role       TEXT                    NOT NULL,
    status              bfc_approval_status     NOT NULL DEFAULT 'pending',
    sequence            INTEGER                 NOT NULL DEFAULT 1,
    required_by         TIMESTAMPTZ,
    decided_at          TIMESTAMPTZ,
    comments            TEXT,
    conditions          TEXT[]                  NOT NULL DEFAULT '{}',
    delegated_to        TEXT,
    digital_signature   TEXT,
    -- audit
    created_at          TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by          TEXT                    NOT NULL,
    updated_by          TEXT                    NOT NULL,
    is_deleted          BOOLEAN                 NOT NULL DEFAULT FALSE,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_bfc_approvals_budget
    ON bfc_budget_approvals (budget_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_approvals_approver
    ON bfc_budget_approvals (approver_id, status) WHERE NOT is_deleted;

-- -----------------------------------------------------------------------------
-- bfc_forecast_models
-- Statistical / ML model configurations
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_forecast_models (
    id                      TEXT                    NOT NULL,
    tenant_id               TEXT                    NOT NULL,
    name                    TEXT                    NOT NULL,
    description             TEXT,
    method                  bfc_forecast_method     NOT NULL,
    horizon_periods         INTEGER                 NOT NULL CHECK (horizon_periods BETWEEN 1 AND 120),
    lookback_periods        INTEGER                 NOT NULL DEFAULT 24 CHECK (lookback_periods >= 1),
    seasonality             BOOLEAN                 NOT NULL DEFAULT TRUE,
    trend                   BOOLEAN                 NOT NULL DEFAULT TRUE,
    confidence_level        NUMERIC(5,2)            NOT NULL DEFAULT 95.0
                                CHECK (confidence_level BETWEEN 0 AND 100),
    hyperparameters         JSONB                   NOT NULL DEFAULT '{}',
    feature_columns         TEXT[]                  NOT NULL DEFAULT '{}',
    is_active               BOOLEAN                 NOT NULL DEFAULT TRUE,
    last_trained_at         TIMESTAMPTZ,
    model_metrics           JSONB                   NOT NULL DEFAULT '{}',
    training_data_start     DATE,
    training_data_end       DATE,
    -- audit
    created_at              TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by              TEXT                    NOT NULL,
    updated_by              TEXT                    NOT NULL,
    is_deleted              BOOLEAN                 NOT NULL DEFAULT FALSE,
    deleted_at              TIMESTAMPTZ,
    deleted_by              TEXT,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_bfc_forecast_models_tenant
    ON bfc_forecast_models (tenant_id, is_active) WHERE NOT is_deleted;

-- -----------------------------------------------------------------------------
-- bfc_forecasts
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_forecasts (
    id                  TEXT                    NOT NULL,
    tenant_id           TEXT                    NOT NULL,
    forecast_model_id   TEXT REFERENCES bfc_forecast_models (id) ON DELETE SET NULL,
    budget_id           TEXT REFERENCES bfc_budgets (id) ON DELETE SET NULL,
    name                TEXT                    NOT NULL,
    forecast_type       bfc_forecast_type       NOT NULL,
    status              bfc_forecast_status     NOT NULL DEFAULT 'draft',
    period_start        DATE                    NOT NULL,
    period_end          DATE                    NOT NULL,
    currency_code       CHAR(3)                 NOT NULL DEFAULT 'USD',
    total_forecasted    NUMERIC(20,4)           NOT NULL DEFAULT 0,
    confidence_lower    NUMERIC(20,4),
    confidence_upper    NUMERIC(20,4),
    mape                NUMERIC(10,6),
    rmse                NUMERIC(20,6),
    error_message       TEXT,
    generated_at        TIMESTAMPTZ,
    published_at        TIMESTAMPTZ,
    metadata            JSONB                   NOT NULL DEFAULT '{}',
    -- audit
    created_at          TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by          TEXT                    NOT NULL,
    updated_by          TEXT                    NOT NULL,
    is_deleted          BOOLEAN                 NOT NULL DEFAULT FALSE,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT,
    PRIMARY KEY (id),
    CHECK (period_end >= period_start)
);

CREATE INDEX IF NOT EXISTS idx_bfc_forecasts_tenant_status
    ON bfc_forecasts (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_forecasts_budget
    ON bfc_forecasts (budget_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_forecasts_period
    ON bfc_forecasts (tenant_id, period_start, period_end) WHERE NOT is_deleted;

-- -----------------------------------------------------------------------------
-- bfc_forecast_lines
-- Partitioned by period_date for large tenants
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_forecast_lines (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    forecast_id         TEXT            NOT NULL REFERENCES bfc_forecasts (id) ON DELETE CASCADE,
    period_date         DATE            NOT NULL,
    account_code        TEXT            NOT NULL,
    forecasted_value    NUMERIC(20,4)   NOT NULL,
    lower_bound         NUMERIC(20,4),
    upper_bound         NUMERIC(20,4),
    actual_value        NUMERIC(20,4),
    residual            NUMERIC(20,4),
    is_outlier          BOOLEAN         NOT NULL DEFAULT FALSE,
    driver_values       JSONB           NOT NULL DEFAULT '{}',
    -- audit
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by          TEXT            NOT NULL,
    updated_by          TEXT            NOT NULL,
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_bfc_forecast_lines_forecast
    ON bfc_forecast_lines (forecast_id, period_date) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_forecast_lines_account
    ON bfc_forecast_lines (tenant_id, account_code, period_date) WHERE NOT is_deleted;

-- -----------------------------------------------------------------------------
-- bfc_variance_reports
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_variance_reports (
    id                      TEXT                NOT NULL,
    tenant_id               TEXT                NOT NULL,
    budget_id               TEXT                NOT NULL REFERENCES bfc_budgets (id) ON DELETE CASCADE,
    report_period_start     DATE                NOT NULL,
    report_period_end       DATE                NOT NULL,
    total_budget            NUMERIC(20,4)       NOT NULL DEFAULT 0,
    total_actual            NUMERIC(20,4)       NOT NULL DEFAULT 0,
    total_variance          NUMERIC(20,4)       NOT NULL DEFAULT 0,
    variance_pct            NUMERIC(10,4)       NOT NULL DEFAULT 0,
    variance_type           bfc_variance_type   NOT NULL DEFAULT 'neutral',
    significance            bfc_significance    NOT NULL DEFAULT 'minimal',
    line_variances          JSONB               NOT NULL DEFAULT '[]',
    recommendations         TEXT[]              NOT NULL DEFAULT '{}',
    generated_at            TIMESTAMPTZ         NOT NULL DEFAULT now(),
    reviewed_by             TEXT,
    -- audit
    created_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by              TEXT                NOT NULL,
    updated_by              TEXT                NOT NULL,
    is_deleted              BOOLEAN             NOT NULL DEFAULT FALSE,
    deleted_at              TIMESTAMPTZ,
    deleted_by              TEXT,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_bfc_variance_budget
    ON bfc_variance_reports (budget_id, report_period_start) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_variance_tenant_sig
    ON bfc_variance_reports (tenant_id, significance) WHERE NOT is_deleted;

-- -----------------------------------------------------------------------------
-- bfc_scenarios
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_scenarios (
    id                  TEXT                NOT NULL,
    tenant_id           TEXT                NOT NULL,
    name                TEXT                NOT NULL,
    description         TEXT,
    scenario_type       bfc_scenario_type   NOT NULL,
    base_budget_id      TEXT REFERENCES bfc_budgets (id) ON DELETE SET NULL,
    base_forecast_id    TEXT REFERENCES bfc_forecasts (id) ON DELETE SET NULL,
    assumptions         JSONB               NOT NULL DEFAULT '{}',
    adjustments         JSONB               NOT NULL DEFAULT '[]',
    probability         NUMERIC(5,4)        NOT NULL DEFAULT 0.5
                            CHECK (probability BETWEEN 0 AND 1),
    is_active           BOOLEAN             NOT NULL DEFAULT TRUE,
    results             JSONB               NOT NULL DEFAULT '{}',
    ran_at              TIMESTAMPTZ,
    net_impact          NUMERIC(20,4),
    net_impact_pct      NUMERIC(10,4),
    -- audit
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by          TEXT                NOT NULL,
    updated_by          TEXT                NOT NULL,
    is_deleted          BOOLEAN             NOT NULL DEFAULT FALSE,
    deleted_at          TIMESTAMPTZ,
    deleted_by          TEXT,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_bfc_scenarios_tenant
    ON bfc_scenarios (tenant_id, is_active) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_scenarios_budget
    ON bfc_scenarios (base_budget_id) WHERE NOT is_deleted;

-- -----------------------------------------------------------------------------
-- bfc_driver_assumptions
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bfc_driver_assumptions (
    id                      TEXT                NOT NULL,
    tenant_id               TEXT                NOT NULL,
    name                    TEXT                NOT NULL,
    driver_type             bfc_driver_type     NOT NULL,
    value                   NUMERIC(20,6)       NOT NULL CHECK (value > 0),
    unit                    TEXT,
    period_start            DATE                NOT NULL,
    period_end              DATE                NOT NULL,
    growth_rate             NUMERIC(10,6),
    -- 12-element seasonality array stored as JSONB
    seasonality_factors     JSONB               NOT NULL DEFAULT '[1,1,1,1,1,1,1,1,1,1,1,1]',
    source                  TEXT,
    confidence              NUMERIC(5,2)        NOT NULL DEFAULT 80.0
                                CHECK (confidence BETWEEN 0 AND 100),
    scenario_id             TEXT REFERENCES bfc_scenarios (id) ON DELETE SET NULL,
    linked_accounts         TEXT[]              NOT NULL DEFAULT '{}',
    metadata                JSONB               NOT NULL DEFAULT '{}',
    -- audit
    created_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by              TEXT                NOT NULL,
    updated_by              TEXT                NOT NULL,
    is_deleted              BOOLEAN             NOT NULL DEFAULT FALSE,
    deleted_at              TIMESTAMPTZ,
    deleted_by              TEXT,
    PRIMARY KEY (id),
    CHECK (period_end >= period_start)
);

CREATE INDEX IF NOT EXISTS idx_bfc_drivers_tenant_type
    ON bfc_driver_assumptions (tenant_id, driver_type) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_bfc_drivers_scenario
    ON bfc_driver_assumptions (scenario_id) WHERE NOT is_deleted;

-- =============================================================================
-- AUDIT EVENTS TABLE
-- Append-only. No is_deleted or updates — events are immutable.
-- =============================================================================
CREATE TABLE IF NOT EXISTS bfc_audit_events (
    id              BIGSERIAL       PRIMARY KEY,
    tenant_id       TEXT            NOT NULL,
    actor_id        TEXT            NOT NULL,
    event_name      TEXT            NOT NULL,
    entity_id       TEXT            NOT NULL,
    stream          TEXT            NOT NULL DEFAULT 'apg.fin.bfc.lifecycle',
    payload         JSONB           NOT NULL DEFAULT '{}',
    occurred_at     TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bfc_audit_tenant
    ON bfc_audit_events (tenant_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_bfc_audit_entity
    ON bfc_audit_events (entity_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_bfc_audit_event
    ON bfc_audit_events (event_name, occurred_at DESC);

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

CREATE OR REPLACE VIEW bfc_v_budget_summary AS
    SELECT
        b.tenant_id,
        b.id,
        b.name,
        b.fiscal_year,
        b.budget_type,
        b.status,
        b.period_start,
        b.period_end,
        b.currency_code,
        b.total_revenue,
        b.total_expense,
        b.net_amount,
        b.owner_id,
        b.department_id,
        b.cost_center_id,
        COUNT(DISTINCT bl.id) AS line_count,
        COUNT(DISTINCT ba.id) FILTER (WHERE ba.status = 'pending') AS pending_approvals
    FROM bfc_budgets b
    LEFT JOIN bfc_budget_lines  bl ON bl.budget_id = b.id AND NOT bl.is_deleted
    LEFT JOIN bfc_budget_approvals ba ON ba.budget_id = b.id AND NOT ba.is_deleted
    WHERE NOT b.is_deleted
    GROUP BY b.tenant_id, b.id, b.name, b.fiscal_year, b.budget_type, b.status,
             b.period_start, b.period_end, b.currency_code, b.total_revenue,
             b.total_expense, b.net_amount, b.owner_id, b.department_id, b.cost_center_id;

CREATE OR REPLACE VIEW bfc_v_variance_summary AS
    SELECT
        vr.tenant_id,
        vr.budget_id,
        b.name AS budget_name,
        vr.report_period_start,
        vr.report_period_end,
        vr.total_budget,
        vr.total_actual,
        vr.total_variance,
        vr.variance_pct,
        vr.variance_type,
        vr.significance,
        vr.generated_at
    FROM bfc_variance_reports vr
    JOIN bfc_budgets b ON b.id = vr.budget_id
    WHERE NOT vr.is_deleted AND NOT b.is_deleted;

-- =============================================================================
-- update_updated_at trigger (shared across all BFC tables)
-- =============================================================================
CREATE OR REPLACE FUNCTION bfc_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DO $$ DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'bfc_budgets','bfc_budget_lines','bfc_budget_approvals',
        'bfc_budget_versions','bfc_budget_templates',
        'bfc_forecasts','bfc_forecast_lines','bfc_forecast_models',
        'bfc_variance_reports','bfc_scenarios','bfc_driver_assumptions'
    ]
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_%s_updated_at ON %s;
             CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %s
             FOR EACH ROW EXECUTE FUNCTION bfc_set_updated_at();',
            t, t, t, t
        );
    END LOOP;
END $$;
