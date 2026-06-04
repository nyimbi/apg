-- =============================================================================
-- General Ledger — PostgreSQL schema
-- © 2025 Datacraft.  Author: Nyimbi Odero <nyimbi@gmail.com>
--
-- Naming conventions:
--   Tables:  gl_<entity>          (prefix 'gl' for General Ledger)
--   Indexes: ix_<table>_<cols>
--   UQ:      uq_<table>_<cols>
--   FK:      fk_<table>_<col>
--   CK:      ck_<table>_<constraint>
--
-- All monetary amounts use NUMERIC(18,4).
-- All IDs are TEXT (UUID-7 strings).
-- Soft deletes: is_deleted BOOLEAN DEFAULT FALSE.
-- =============================================================================

-- ─────────────────────────────────────────────
-- Extensions
-- ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ─────────────────────────────────────────────
-- Tenant configuration
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_tenant (
    id                          TEXT        PRIMARY KEY,
    tenant_code                 TEXT        NOT NULL UNIQUE,
    tenant_name                 TEXT        NOT NULL,
    base_currency               TEXT        NOT NULL DEFAULT 'USD',
    functional_currency         TEXT        NOT NULL DEFAULT 'USD',
    reporting_currencies        JSONB       NOT NULL DEFAULT '[]',
    fiscal_year_start_month     SMALLINT    NOT NULL DEFAULT 1
                                    CHECK (fiscal_year_start_month BETWEEN 1 AND 12),
    period_type                 TEXT        NOT NULL DEFAULT 'MONTHLY'
                                    CHECK (period_type IN ('MONTHLY','QUARTERLY','13_PERIOD')),
    reporting_framework         TEXT        NOT NULL DEFAULT 'IFRS'
                                    CHECK (reporting_framework IN ('IFRS','GAAP','LOCAL_GAAP','TAX_BASIS')),
    country_code                TEXT        NOT NULL DEFAULT 'KE',
    timezone                    TEXT        NOT NULL DEFAULT 'Africa/Nairobi',
    sox_compliance              BOOLEAN     NOT NULL DEFAULT FALSE,
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by                  TEXT        NOT NULL DEFAULT 'system'
);

-- ─────────────────────────────────────────────
-- Chart of accounts
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_account (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    account_code                TEXT        NOT NULL,
    account_name                TEXT        NOT NULL,
    account_type                TEXT        NOT NULL
                                    CHECK (account_type IN ('asset','liability','equity','revenue','expense','contra')),
    normal_balance              TEXT        NOT NULL
                                    CHECK (normal_balance IN ('debit','credit')),
    currency                    TEXT        NOT NULL DEFAULT 'USD',
    allow_posting               BOOLEAN     NOT NULL DEFAULT TRUE,
    parent_account_id           TEXT        REFERENCES gl_account(id) ON DELETE RESTRICT,
    hierarchy_level             SMALLINT    NOT NULL DEFAULT 0 CHECK (hierarchy_level >= 0),
    hierarchy_path              TEXT,
    description                 TEXT,
    ifrs_mapping                TEXT,
    gaap_mapping                TEXT,
    tax_code                    TEXT,
    cost_center_required        BOOLEAN     NOT NULL DEFAULT FALSE,
    project_required            BOOLEAN     NOT NULL DEFAULT FALSE,
    is_reconciliation_account   BOOLEAN     NOT NULL DEFAULT FALSE,
    tags                        JSONB       NOT NULL DEFAULT '[]',
    status                      TEXT        NOT NULL DEFAULT 'active'
                                    CHECK (status IN ('active','inactive','archived')),
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by                  TEXT        NOT NULL DEFAULT 'system',

    CONSTRAINT uq_gl_account_code_tenant UNIQUE (tenant_id, account_code),
    CONSTRAINT fk_gl_account_tenant      FOREIGN KEY (tenant_id) REFERENCES gl_tenant(id) ON DELETE RESTRICT,
    CONSTRAINT ck_gl_account_no_self_parent CHECK (parent_account_id IS NULL OR parent_account_id <> id)
);

CREATE INDEX IF NOT EXISTS ix_gl_account_tenant       ON gl_account (tenant_id);
CREATE INDEX IF NOT EXISTS ix_gl_account_type         ON gl_account (tenant_id, account_type);
CREATE INDEX IF NOT EXISTS ix_gl_account_parent       ON gl_account (parent_account_id);
CREATE INDEX IF NOT EXISTS ix_gl_account_path         ON gl_account (hierarchy_path);
CREATE INDEX IF NOT EXISTS ix_gl_account_tags         ON gl_account USING gin (tags);
CREATE INDEX IF NOT EXISTS ix_gl_account_status       ON gl_account (tenant_id, status) WHERE is_deleted = FALSE;

-- ─────────────────────────────────────────────
-- Accounting periods
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_period (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    period_code                 TEXT        NOT NULL,
    fiscal_year                 SMALLINT    NOT NULL CHECK (fiscal_year BETWEEN 1900 AND 2200),
    period_number               SMALLINT    NOT NULL CHECK (period_number BETWEEN 1 AND 13),
    start_date                  DATE        NOT NULL,
    end_date                    DATE        NOT NULL,
    status                      TEXT        NOT NULL DEFAULT 'future'
                                    CHECK (status IN ('future','open','soft_closed','closed','locked')),
    allows_adjustments          BOOLEAN     NOT NULL DEFAULT FALSE,
    opened_by                   TEXT,
    opened_at                   TIMESTAMPTZ,
    closed_by                   TEXT,
    closed_at                   TIMESTAMPTZ,
    locked_by                   TEXT,
    locked_at                   TIMESTAMPTZ,
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by                  TEXT        NOT NULL DEFAULT 'system',

    CONSTRAINT uq_gl_period_code_tenant  UNIQUE (tenant_id, period_code),
    CONSTRAINT uq_gl_period_year_num     UNIQUE (tenant_id, fiscal_year, period_number),
    CONSTRAINT fk_gl_period_tenant       FOREIGN KEY (tenant_id) REFERENCES gl_tenant(id) ON DELETE RESTRICT,
    CONSTRAINT ck_gl_period_date_order   CHECK (start_date <= end_date)
);

CREATE INDEX IF NOT EXISTS ix_gl_period_tenant        ON gl_period (tenant_id);
CREATE INDEX IF NOT EXISTS ix_gl_period_dates         ON gl_period (tenant_id, start_date, end_date);
CREATE INDEX IF NOT EXISTS ix_gl_period_status        ON gl_period (tenant_id, status);
CREATE INDEX IF NOT EXISTS ix_gl_period_year          ON gl_period (tenant_id, fiscal_year);

-- ─────────────────────────────────────────────
-- Journal entry headers
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_journal_entry (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    journal_number              TEXT        NOT NULL,
    journal_date                DATE        NOT NULL,
    period_id                   TEXT        REFERENCES gl_period(id) ON DELETE RESTRICT,
    period_code                 TEXT,
    journal_type                TEXT        NOT NULL DEFAULT 'standard'
                                    CHECK (journal_type IN ('standard','adjustment','recurring','reversal',
                                                            'intercompany','accrual','import','manual')),
    description                 TEXT        NOT NULL,
    reference                   TEXT,
    status                      TEXT        NOT NULL DEFAULT 'draft'
                                    CHECK (status IN ('draft','balanced','pending_approval','approved',
                                                      'posted','reversed','cancelled')),
    total_debit                 NUMERIC(18,4) NOT NULL DEFAULT 0,
    total_credit                NUMERIC(18,4) NOT NULL DEFAULT 0,
    approval_status             TEXT        CHECK (approval_status IN ('pending','auto_approved','approved','rejected')),
    prepared_by                 TEXT        NOT NULL DEFAULT 'system',
    approved_by                 TEXT,
    approved_at                 TIMESTAMPTZ,
    posted_by                   TEXT,
    posted_at                   TIMESTAMPTZ,
    reversed_at                 TIMESTAMPTZ,
    reversal_journal_id         TEXT,
    attachments                 JSONB       NOT NULL DEFAULT '[]',
    source_system               TEXT,
    batch_id                    TEXT,
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by                  TEXT        NOT NULL DEFAULT 'system',

    CONSTRAINT uq_gl_journal_number_tenant   UNIQUE (tenant_id, journal_number),
    CONSTRAINT fk_gl_journal_tenant          FOREIGN KEY (tenant_id) REFERENCES gl_tenant(id) ON DELETE RESTRICT,
    CONSTRAINT ck_gl_journal_balanced        CHECK (ABS(total_debit - total_credit) < 0.0001 OR status = 'draft')
);

CREATE INDEX IF NOT EXISTS ix_gl_journal_tenant       ON gl_journal_entry (tenant_id);
CREATE INDEX IF NOT EXISTS ix_gl_journal_date         ON gl_journal_entry (tenant_id, journal_date);
CREATE INDEX IF NOT EXISTS ix_gl_journal_period       ON gl_journal_entry (period_id);
CREATE INDEX IF NOT EXISTS ix_gl_journal_status       ON gl_journal_entry (tenant_id, status);
CREATE INDEX IF NOT EXISTS ix_gl_journal_number       ON gl_journal_entry (tenant_id, journal_number);
CREATE INDEX IF NOT EXISTS ix_gl_journal_type         ON gl_journal_entry (tenant_id, journal_type);

-- ─────────────────────────────────────────────
-- Journal lines
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_journal_line (
    id                          TEXT        PRIMARY KEY,
    journal_id                  TEXT        NOT NULL,
    tenant_id                   TEXT        NOT NULL,
    line_number                 SMALLINT    NOT NULL CHECK (line_number >= 1),
    account_id                  TEXT        NOT NULL,
    debit                       NUMERIC(18,4) NOT NULL DEFAULT 0 CHECK (debit >= 0),
    credit                      NUMERIC(18,4) NOT NULL DEFAULT 0 CHECK (credit >= 0),
    functional_debit            NUMERIC(18,4) NOT NULL DEFAULT 0,
    functional_credit           NUMERIC(18,4) NOT NULL DEFAULT 0,
    currency                    TEXT        NOT NULL DEFAULT 'USD',
    exchange_rate               NUMERIC(12,6) NOT NULL DEFAULT 1,
    description                 TEXT,
    cost_center                 TEXT,
    project                     TEXT,
    entity                      TEXT,
    segment                     TEXT,
    tax_code                    TEXT,
    tax_amount                  NUMERIC(18,4) NOT NULL DEFAULT 0,
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_gl_line_journal   FOREIGN KEY (journal_id) REFERENCES gl_journal_entry(id) ON DELETE CASCADE,
    CONSTRAINT fk_gl_line_account   FOREIGN KEY (account_id) REFERENCES gl_account(id) ON DELETE RESTRICT,
    CONSTRAINT fk_gl_line_tenant    FOREIGN KEY (tenant_id)  REFERENCES gl_tenant(id) ON DELETE RESTRICT,
    CONSTRAINT uq_gl_line_number    UNIQUE (journal_id, line_number),
    CONSTRAINT ck_gl_line_one_side  CHECK (NOT (debit > 0 AND credit > 0))
);

CREATE INDEX IF NOT EXISTS ix_gl_line_journal      ON gl_journal_line (journal_id);
CREATE INDEX IF NOT EXISTS ix_gl_line_account      ON gl_journal_line (account_id);
CREATE INDEX IF NOT EXISTS ix_gl_line_tenant       ON gl_journal_line (tenant_id);
CREATE INDEX IF NOT EXISTS ix_gl_line_cost_center  ON gl_journal_line (cost_center) WHERE cost_center IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_gl_line_project      ON gl_journal_line (project) WHERE project IS NOT NULL;

-- ─────────────────────────────────────────────
-- Account balances (period snapshot)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_account_balance (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    account_id                  TEXT        NOT NULL,
    period_id                   TEXT        NOT NULL,
    period_code                 TEXT        NOT NULL,
    opening_balance             NUMERIC(18,4) NOT NULL DEFAULT 0,
    period_debits               NUMERIC(18,4) NOT NULL DEFAULT 0,
    period_credits              NUMERIC(18,4) NOT NULL DEFAULT 0,
    closing_balance             NUMERIC(18,4) NOT NULL DEFAULT 0,
    currency                    TEXT        NOT NULL DEFAULT 'USD',
    transaction_count           INTEGER     NOT NULL DEFAULT 0,
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_gl_balance_account_period_ccy UNIQUE (account_id, period_id, currency),
    CONSTRAINT fk_gl_balance_account FOREIGN KEY (account_id) REFERENCES gl_account(id) ON DELETE RESTRICT,
    CONSTRAINT fk_gl_balance_period  FOREIGN KEY (period_id)  REFERENCES gl_period(id)  ON DELETE RESTRICT,
    CONSTRAINT fk_gl_balance_tenant  FOREIGN KEY (tenant_id)  REFERENCES gl_tenant(id)  ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_gl_balance_account     ON gl_account_balance (account_id, period_code);
CREATE INDEX IF NOT EXISTS ix_gl_balance_tenant      ON gl_account_balance (tenant_id, period_code);

-- ─────────────────────────────────────────────
-- Budget
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_budget (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    budget_code                 TEXT        NOT NULL,
    fiscal_year                 SMALLINT    NOT NULL,
    budget_type                 TEXT        NOT NULL DEFAULT 'original'
                                    CHECK (budget_type IN ('original','revised','forecast')),
    account_code                TEXT        NOT NULL,
    period_code                 TEXT        NOT NULL,
    amount                      NUMERIC(18,4) NOT NULL,
    currency                    TEXT        NOT NULL DEFAULT 'USD',
    budget_version              TEXT        NOT NULL DEFAULT 'approved',
    notes                       TEXT,
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by                  TEXT        NOT NULL DEFAULT 'system',

    CONSTRAINT uq_gl_budget_key UNIQUE (tenant_id, account_code, period_code, budget_type, budget_version),
    CONSTRAINT fk_gl_budget_tenant FOREIGN KEY (tenant_id) REFERENCES gl_tenant(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_gl_budget_tenant       ON gl_budget (tenant_id, fiscal_year);
CREATE INDEX IF NOT EXISTS ix_gl_budget_account      ON gl_budget (tenant_id, account_code, period_code);

-- ─────────────────────────────────────────────
-- Reconciliation
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_reconciliation (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    account_code                TEXT        NOT NULL,
    period_code                 TEXT        NOT NULL,
    balance_per_gl              NUMERIC(18,4),
    balance_per_statement       NUMERIC(18,4),
    reconciling_items           JSONB       NOT NULL DEFAULT '[]',
    unreconciled_difference     NUMERIC(18,4),
    status                      TEXT        NOT NULL DEFAULT 'open'
                                    CHECK (status IN ('open','submitted','approved','rejected')),
    reconciled_by               TEXT,
    reconciled_at               TIMESTAMPTZ,
    approved_by                 TEXT,
    approved_at                 TIMESTAMPTZ,
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by                  TEXT        NOT NULL DEFAULT 'system',

    CONSTRAINT fk_gl_recon_tenant FOREIGN KEY (tenant_id) REFERENCES gl_tenant(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_gl_recon_tenant        ON gl_reconciliation (tenant_id, period_code);
CREATE INDEX IF NOT EXISTS ix_gl_recon_account       ON gl_reconciliation (tenant_id, account_code, period_code);
CREATE INDEX IF NOT EXISTS ix_gl_recon_status        ON gl_reconciliation (tenant_id, status);

-- ─────────────────────────────────────────────
-- Currency exchange rates
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_currency_rate (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    from_currency               TEXT        NOT NULL,
    to_currency                 TEXT        NOT NULL,
    rate_type                   TEXT        NOT NULL DEFAULT 'spot'
                                    CHECK (rate_type IN ('spot','average','budget','closing','historical')),
    effective_date              DATE        NOT NULL,
    expiry_date                 DATE,
    exchange_rate               NUMERIC(12,6) NOT NULL CHECK (exchange_rate > 0),
    inverse_rate                NUMERIC(12,6),
    rate_source                 TEXT        NOT NULL DEFAULT 'manual',
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by                  TEXT        NOT NULL DEFAULT 'system',

    CONSTRAINT uq_gl_rate_unique UNIQUE (tenant_id, from_currency, to_currency, rate_type, effective_date),
    CONSTRAINT fk_gl_rate_tenant FOREIGN KEY (tenant_id) REFERENCES gl_tenant(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_gl_rate_lookup ON gl_currency_rate (tenant_id, from_currency, to_currency, effective_date DESC);

-- ─────────────────────────────────────────────
-- Recurring journal templates
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_recurring_template (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    template_name               TEXT        NOT NULL,
    journal_type                TEXT        NOT NULL DEFAULT 'accrual',
    description                 TEXT,
    lines                       JSONB       NOT NULL DEFAULT '[]',
    amount_multiplier           NUMERIC(12,6) NOT NULL DEFAULT 1,
    owner                       TEXT        NOT NULL DEFAULT 'system',
    frequency                   TEXT        NOT NULL DEFAULT 'monthly'
                                    CHECK (frequency IN ('daily','weekly','monthly','quarterly','annually')),
    next_run_date               DATE,
    is_active                   BOOLEAN     NOT NULL DEFAULT TRUE,
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by                  TEXT        NOT NULL DEFAULT 'system',

    CONSTRAINT fk_gl_recur_tenant FOREIGN KEY (tenant_id) REFERENCES gl_tenant(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_gl_recur_tenant ON gl_recurring_template (tenant_id, is_active);

-- ─────────────────────────────────────────────
-- Intercompany journal log
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_intercompany_journal (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    counterpart_entity          TEXT        NOT NULL,
    amount                      NUMERIC(18,4) NOT NULL,
    currency                    TEXT        NOT NULL DEFAULT 'USD',
    entity_posting_id           TEXT,
    counterpart_posting_id      TEXT,
    status                      TEXT        NOT NULL DEFAULT 'posted',
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_gl_ic_tenant FOREIGN KEY (tenant_id) REFERENCES gl_tenant(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_gl_ic_tenant ON gl_intercompany_journal (tenant_id, counterpart_entity);

-- ─────────────────────────────────────────────
-- Fiscal year close records
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_fiscal_year_close (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    fiscal_year                 SMALLINT    NOT NULL,
    retained_earnings_account   TEXT        NOT NULL,
    net_to_retained_earnings    NUMERIC(18,4) NOT NULL,
    closing_journal_id          TEXT,
    status                      TEXT        NOT NULL DEFAULT 'closed',
    closed_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_gl_fy_close UNIQUE (tenant_id, fiscal_year),
    CONSTRAINT fk_gl_fy_tenant FOREIGN KEY (tenant_id) REFERENCES gl_tenant(id) ON DELETE RESTRICT
);

-- ─────────────────────────────────────────────
-- Approval workflows
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_approval_workflow (
    id                          TEXT        PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    journal_id                  TEXT        NOT NULL,
    journal_number              TEXT,
    amount                      NUMERIC(18,4) NOT NULL,
    threshold                   NUMERIC(18,4) NOT NULL,
    approver_id                 TEXT        NOT NULL,
    decision                    TEXT        NOT NULL DEFAULT 'pending'
                                    CHECK (decision IN ('pending','auto_approved','approved','rejected')),
    decision_at                 TIMESTAMPTZ,
    status                      TEXT        NOT NULL DEFAULT 'pending',
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_gl_wf_tenant  FOREIGN KEY (tenant_id)  REFERENCES gl_tenant(id)  ON DELETE RESTRICT,
    CONSTRAINT fk_gl_wf_journal FOREIGN KEY (journal_id) REFERENCES gl_journal_entry(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_gl_wf_tenant    ON gl_approval_workflow (tenant_id, status);
CREATE INDEX IF NOT EXISTS ix_gl_wf_journal   ON gl_approval_workflow (journal_id);

-- ─────────────────────────────────────────────
-- Audit log (immutable append-only)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gl_audit_log (
    id                          BIGSERIAL   PRIMARY KEY,
    tenant_id                   TEXT        NOT NULL,
    event_type                  TEXT        NOT NULL,
    record_id                   TEXT        NOT NULL,
    record_type                 TEXT        NOT NULL,
    actor                       TEXT,
    payload                     JSONB       NOT NULL DEFAULT '{}',
    emitted_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_gl_audit_tenant    ON gl_audit_log (tenant_id, emitted_at DESC);
CREATE INDEX IF NOT EXISTS ix_gl_audit_record    ON gl_audit_log (record_id, event_type);
CREATE INDEX IF NOT EXISTS ix_gl_audit_payload   ON gl_audit_log USING gin (payload);

-- ─────────────────────────────────────────────
-- Updated-at trigger helper
-- ─────────────────────────────────────────────
CREATE OR REPLACE FUNCTION gl_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'gl_tenant','gl_account','gl_period','gl_journal_entry',
        'gl_account_balance','gl_budget','gl_reconciliation',
        'gl_currency_rate','gl_recurring_template'
    ] LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_%1$s_updated_at ON %1$s;
             CREATE TRIGGER trg_%1$s_updated_at
             BEFORE UPDATE ON %1$s
             FOR EACH ROW EXECUTE FUNCTION gl_set_updated_at();',
            t
        );
    END LOOP;
END;
$$;
