-- =============================================================================
-- APG Point of Sale — PostgreSQL Schema
-- © 2025 Datacraft | www.datacraft.co.ke
--
-- Run:   psql $DATABASE_URL -f database/schema.sql
-- Undo:  psql $DATABASE_URL -f database/schema_drop.sql
--
-- Design principles:
--   • All tables are tenant-scoped (tenant_id column, NOT NULL)
--   • Soft-delete via is_deleted BOOLEAN NOT NULL DEFAULT FALSE
--   • Audit columns: created_at, updated_at, created_by
--   • UUID7 strings for primary keys (TEXT, not UUID type — preserves sort order)
--   • Monetary values stored as NUMERIC(18,4) — never FLOAT
--   • Large tables (transactions, payments, movements) partitioned by tenant+month
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pos;
SET search_path TO pos, public;

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- for ILIKE index support
CREATE EXTENSION IF NOT EXISTS btree_gin; -- for composite GIN indexes

-- ---------------------------------------------------------------------------
-- Enumerations
-- ---------------------------------------------------------------------------
DO $$ BEGIN
  CREATE TYPE pos.terminal_type    AS ENUM ('fixed_counter','mobile','self_service','kiosk','mpos');
  CREATE TYPE pos.terminal_status  AS ENUM ('offline','online','in_session','maintenance','suspended');
  CREATE TYPE pos.session_status   AS ENUM ('open','suspended','closed','reconciled','force_closed');
  CREATE TYPE pos.txn_type         AS ENUM ('sale','refund','exchange','void','layaway','layaway_pickup','no_sale');
  CREATE TYPE pos.txn_status       AS ENUM ('pending','authorised','partially_paid','completed','voided','suspended','refunded','partially_refunded');
  CREATE TYPE pos.payment_method   AS ENUM ('cash','card_credit','card_debit','mobile_money','loyalty_points','gift_card','store_credit','cheque','bank_transfer');
  CREATE TYPE pos.payment_status   AS ENUM ('pending','authorised','captured','declined','reversed','refunded');
  CREATE TYPE pos.discount_type    AS ENUM ('percentage','fixed_amount','buy_x_get_y','bundle','loyalty','coupon','staff','manager');
  CREATE TYPE pos.cash_event_type  AS ENUM ('opening_float','petty_cash_out','petty_cash_in','safe_drop','safe_pickup','till_loan','correction');
  CREATE TYPE pos.refund_reason    AS ENUM ('defective','wrong_item','customer_change_mind','overcharge','duplicate','not_as_described','other');
  CREATE TYPE pos.receipt_format   AS ENUM ('thermal','email','sms','digital','both');
  CREATE TYPE pos.override_reason  AS ENUM ('price_match','damage','clearance','customer_complaint','manager_discretion');
  CREATE TYPE pos.inv_movement_type AS ENUM ('sale','refund','adjustment','transfer','write_off');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- =============================================================================
-- TABLE: pos_terminals
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_terminals (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    store_id            TEXT            NOT NULL,
    terminal_code       TEXT            NOT NULL,
    terminal_type       pos.terminal_type NOT NULL DEFAULT 'fixed_counter',
    status              pos.terminal_status NOT NULL DEFAULT 'offline',
    serial_number       TEXT,
    hardware_model      TEXT,
    firmware_version    TEXT,
    offline_capable     BOOLEAN         NOT NULL DEFAULT TRUE,
    floor_limit         NUMERIC(18,4)   NOT NULL DEFAULT 5000.00,
    tax_profile_id      TEXT,
    default_currency    CHAR(3)         NOT NULL DEFAULT 'KES',
    current_session_id  TEXT,
    last_heartbeat_at   TIMESTAMPTZ,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by          TEXT            NOT NULL,
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_terminals_code
    ON pos.pos_terminals (tenant_id, store_id, terminal_code)
    WHERE NOT is_deleted;

CREATE INDEX IF NOT EXISTS idx_terminals_store
    ON pos.pos_terminals (tenant_id, store_id) WHERE NOT is_deleted;

CREATE INDEX IF NOT EXISTS idx_terminals_status
    ON pos.pos_terminals (tenant_id, status) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: pos_sessions
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_sessions (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    terminal_id         TEXT            NOT NULL REFERENCES pos.pos_terminals(id),
    store_id            TEXT            NOT NULL,
    cashier_id          TEXT            NOT NULL,
    supervisor_id       TEXT,
    session_number      TEXT            NOT NULL,
    status              pos.session_status NOT NULL DEFAULT 'open',
    opening_float       NUMERIC(18,4)   NOT NULL DEFAULT 0,
    closing_cash_counted NUMERIC(18,4),
    expected_cash       NUMERIC(18,4),
    variance            NUMERIC(18,4),
    transaction_count   INTEGER         NOT NULL DEFAULT 0,
    total_sales         NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_refunds       NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_cash_sales    NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_card_sales    NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_mobile_sales  NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_loyalty_sales NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_discounts     NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_tax           NUMERIC(18,4)   NOT NULL DEFAULT 0,
    opened_at           TIMESTAMPTZ     NOT NULL DEFAULT now(),
    closed_at           TIMESTAMPTZ,
    reconciled_at       TIMESTAMPTZ,
    notes               TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by          TEXT            NOT NULL,
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sessions_number
    ON pos.pos_sessions (tenant_id, session_number);

CREATE INDEX IF NOT EXISTS idx_sessions_terminal
    ON pos.pos_sessions (tenant_id, terminal_id, status);

CREATE INDEX IF NOT EXISTS idx_sessions_cashier
    ON pos.pos_sessions (tenant_id, cashier_id, opened_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_store_date
    ON pos.pos_sessions (tenant_id, store_id, opened_at DESC);

-- Enforce single open session per terminal
CREATE UNIQUE INDEX IF NOT EXISTS uq_sessions_open_terminal
    ON pos.pos_sessions (tenant_id, terminal_id)
    WHERE status = 'open' AND NOT is_deleted;

-- =============================================================================
-- TABLE: pos_transactions
-- Partitioned by tenant_id + year-month for scalability.
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_transactions (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    session_id              TEXT            NOT NULL REFERENCES pos.pos_sessions(id),
    terminal_id             TEXT            NOT NULL REFERENCES pos.pos_terminals(id),
    store_id                TEXT            NOT NULL,
    cashier_id              TEXT            NOT NULL,
    transaction_number      TEXT            NOT NULL,
    transaction_type        pos.txn_type    NOT NULL DEFAULT 'sale',
    status                  pos.txn_status  NOT NULL DEFAULT 'pending',
    customer_id             TEXT,
    original_transaction_id TEXT,           -- for refunds / exchanges
    subtotal                NUMERIC(18,4)   NOT NULL DEFAULT 0,
    discount_total          NUMERIC(18,4)   NOT NULL DEFAULT 0,
    tax_total               NUMERIC(18,4)   NOT NULL DEFAULT 0,
    grand_total             NUMERIC(18,4)   NOT NULL DEFAULT 0,
    amount_tendered         NUMERIC(18,4)   NOT NULL DEFAULT 0,
    change_due              NUMERIC(18,4)   NOT NULL DEFAULT 0,
    balance_due             NUMERIC(18,4)   NOT NULL DEFAULT 0,
    tax_exempt              BOOLEAN         NOT NULL DEFAULT FALSE,
    tax_exempt_ref          TEXT,
    offline_mode            BOOLEAN         NOT NULL DEFAULT FALSE,
    offline_synced          BOOLEAN         NOT NULL DEFAULT FALSE,
    offline_sync_seq        INTEGER,
    receipt_number          TEXT,
    signature_ref           TEXT,
    supervisor_override_id  TEXT,
    notes                   TEXT,
    posted_at               TIMESTAMPTZ,
    voided_at               TIMESTAMPTZ,
    refunded_at             TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by              TEXT            NOT NULL,
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
) PARTITION BY LIST (tenant_id);

-- Default partition catches all tenants without explicit partitions
CREATE TABLE IF NOT EXISTS pos.pos_transactions_default
    PARTITION OF pos.pos_transactions DEFAULT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_txn_number
    ON pos.pos_transactions (tenant_id, transaction_number);

CREATE INDEX IF NOT EXISTS idx_txn_session
    ON pos.pos_transactions (tenant_id, session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_txn_customer
    ON pos.pos_transactions (tenant_id, customer_id, created_at DESC)
    WHERE customer_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_txn_status
    ON pos.pos_transactions (tenant_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_txn_receipt
    ON pos.pos_transactions (tenant_id, receipt_number)
    WHERE receipt_number IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_txn_offline_sync
    ON pos.pos_transactions (tenant_id, terminal_id, offline_synced)
    WHERE offline_mode = TRUE;

-- =============================================================================
-- TABLE: pos_sale_items
-- Line items for each transaction.
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_sale_items (
    id              TEXT            NOT NULL,
    tenant_id       TEXT            NOT NULL,
    transaction_id  TEXT            NOT NULL REFERENCES pos.pos_transactions(id) ON DELETE CASCADE,
    line_number     SMALLINT        NOT NULL DEFAULT 1,
    sku             TEXT            NOT NULL,
    barcode         TEXT,
    description     TEXT            NOT NULL DEFAULT '',
    quantity        NUMERIC(18,4)   NOT NULL,
    unit_price      NUMERIC(18,4)   NOT NULL,
    original_price  NUMERIC(18,4),
    cost_price      NUMERIC(18,4),
    tax_code        TEXT,
    tax_rate        NUMERIC(6,4)    NOT NULL DEFAULT 0,
    tax_amount      NUMERIC(18,4)   NOT NULL DEFAULT 0,
    tax_inclusive   BOOLEAN         NOT NULL DEFAULT TRUE,
    discount_amount NUMERIC(18,4)   NOT NULL DEFAULT 0,
    discount_type   pos.discount_type,
    discount_ref    TEXT,
    line_total      NUMERIC(18,4)   NOT NULL DEFAULT 0,
    promotion_ids   TEXT[]          NOT NULL DEFAULT '{}',
    weight_item     BOOLEAN         NOT NULL DEFAULT FALSE,
    serialised      BOOLEAN         NOT NULL DEFAULT FALSE,
    serial_numbers  TEXT[]          NOT NULL DEFAULT '{}',
    department      TEXT,
    category        TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_items_transaction
    ON pos.pos_sale_items (transaction_id);

CREATE INDEX IF NOT EXISTS idx_items_sku
    ON pos.pos_sale_items (tenant_id, sku);

-- =============================================================================
-- TABLE: pos_payments
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_payments (
    id                  TEXT                NOT NULL,
    tenant_id           TEXT                NOT NULL,
    transaction_id      TEXT                NOT NULL REFERENCES pos.pos_transactions(id),
    session_id          TEXT                NOT NULL REFERENCES pos.pos_sessions(id),
    payment_method      pos.payment_method  NOT NULL,
    status              pos.payment_status  NOT NULL DEFAULT 'authorised',
    amount              NUMERIC(18,4)       NOT NULL,
    currency            CHAR(3)             NOT NULL DEFAULT 'KES',
    exchange_rate       NUMERIC(12,6)       NOT NULL DEFAULT 1,
    reference           TEXT,               -- card auth code, M-Pesa reference
    terminal_ref        TEXT,               -- PED / card terminal reference
    loyalty_points_used INTEGER,
    gift_card_number    TEXT,
    authorised_at       TIMESTAMPTZ         DEFAULT now(),
    gateway_response    JSONB,
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by          TEXT                NOT NULL,
    is_deleted          BOOLEAN             NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_payments_transaction
    ON pos.pos_payments (transaction_id);

CREATE INDEX IF NOT EXISTS idx_payments_session
    ON pos.pos_payments (tenant_id, session_id, payment_method);

CREATE INDEX IF NOT EXISTS idx_payments_method
    ON pos.pos_payments (tenant_id, payment_method, created_at DESC);

-- =============================================================================
-- TABLE: pos_refunds
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_refunds (
    id                      TEXT                NOT NULL,
    tenant_id               TEXT                NOT NULL,
    original_transaction_id TEXT                NOT NULL REFERENCES pos.pos_transactions(id),
    refund_transaction_id   TEXT                REFERENCES pos.pos_transactions(id),
    session_id              TEXT                NOT NULL REFERENCES pos.pos_sessions(id),
    terminal_id             TEXT                NOT NULL REFERENCES pos.pos_terminals(id),
    reason                  pos.refund_reason   NOT NULL,
    refund_amount           NUMERIC(18,4)       NOT NULL DEFAULT 0,
    refund_method           pos.payment_method,
    status                  pos.txn_status      NOT NULL DEFAULT 'pending',
    manager_auth_id         TEXT,
    notes                   TEXT,
    refunded_at             TIMESTAMPTZ         DEFAULT now(),
    created_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by              TEXT                NOT NULL,
    is_deleted              BOOLEAN             NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_refunds_original
    ON pos.pos_refunds (tenant_id, original_transaction_id);

-- =============================================================================
-- TABLE: pos_cash_floats
-- Cash events: opening float, safe drops, petty cash, till loans.
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_cash_floats (
    id              TEXT                    NOT NULL,
    tenant_id       TEXT                    NOT NULL,
    session_id      TEXT                    NOT NULL REFERENCES pos.pos_sessions(id),
    terminal_id     TEXT                    NOT NULL REFERENCES pos.pos_terminals(id),
    store_id        TEXT                    NOT NULL,
    cashier_id      TEXT                    NOT NULL,
    event_type      pos.cash_event_type     NOT NULL,
    amount          NUMERIC(18,4)           NOT NULL,
    balance_after   NUMERIC(18,4)           NOT NULL DEFAULT 0,
    reason          TEXT,
    authorised_by   TEXT,
    denominations   JSONB,                  -- {"1000": 2, "500": 5, ...}
    occurred_at     TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by      TEXT                    NOT NULL,
    is_deleted      BOOLEAN                 NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_cash_session
    ON pos.pos_cash_floats (tenant_id, session_id, occurred_at);

-- =============================================================================
-- TABLE: pos_discounts
-- Discount catalogue (promotions, coupons, staff discounts).
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_discounts (
    id              TEXT                NOT NULL,
    tenant_id       TEXT                NOT NULL,
    discount_type   pos.discount_type   NOT NULL,
    name            TEXT                NOT NULL,
    value           NUMERIC(10,4)       NOT NULL,   -- pct or fixed amount
    max_uses        INTEGER,
    times_used      INTEGER             NOT NULL DEFAULT 0,
    total_discount_given NUMERIC(18,4)  NOT NULL DEFAULT 0,
    min_purchase    NUMERIC(18,4),
    coupon_code     TEXT,
    valid_from      TIMESTAMPTZ,
    valid_until     TIMESTAMPTZ,
    product_skus    TEXT[]              NOT NULL DEFAULT '{}',
    category_ids    TEXT[]              NOT NULL DEFAULT '{}',
    requires_supervisor BOOLEAN         NOT NULL DEFAULT FALSE,
    is_active       BOOLEAN             NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by      TEXT                NOT NULL,
    is_deleted      BOOLEAN             NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_discount_coupon
    ON pos.pos_discounts (tenant_id, coupon_code)
    WHERE coupon_code IS NOT NULL AND NOT is_deleted;

CREATE INDEX IF NOT EXISTS idx_discounts_tenant_active
    ON pos.pos_discounts (tenant_id, is_active, valid_from, valid_until);

-- =============================================================================
-- TABLE: pos_price_overrides
-- Manager/supervisor price overrides — full audit trail.
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_price_overrides (
    id              TEXT                    NOT NULL,
    tenant_id       TEXT                    NOT NULL,
    transaction_id  TEXT                    NOT NULL REFERENCES pos.pos_transactions(id),
    session_id      TEXT                    NOT NULL REFERENCES pos.pos_sessions(id),
    sku             TEXT                    NOT NULL,
    original_price  NUMERIC(18,4)           NOT NULL,
    override_price  NUMERIC(18,4)           NOT NULL,
    variance        NUMERIC(18,4)           GENERATED ALWAYS AS (override_price - original_price) STORED,
    reason          pos.override_reason     NOT NULL,
    notes           TEXT,
    supervisor_id   TEXT                    NOT NULL,
    approved_at     TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by      TEXT                    NOT NULL,
    is_deleted      BOOLEAN                 NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_price_overrides_txn
    ON pos.pos_price_overrides (tenant_id, transaction_id);

CREATE INDEX IF NOT EXISTS idx_price_overrides_supervisor
    ON pos.pos_price_overrides (tenant_id, supervisor_id, approved_at DESC);

-- =============================================================================
-- TABLE: pos_inventory_movements
-- All stock changes originating from POS (sales, refunds, adjustments).
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_inventory_movements (
    id              TEXT                        NOT NULL,
    tenant_id       TEXT                        NOT NULL,
    store_id        TEXT                        NOT NULL,
    terminal_id     TEXT                        NOT NULL,
    transaction_id  TEXT                        NOT NULL,
    sku             TEXT                        NOT NULL,
    movement_type   pos.inv_movement_type       NOT NULL,
    quantity_delta  NUMERIC(18,4)               NOT NULL,   -- negative = stock out
    unit_cost       NUMERIC(18,4),
    stock_before    NUMERIC(18,4),
    stock_after     NUMERIC(18,4),
    notes           TEXT,
    occurred_at     TIMESTAMPTZ                 NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ                 NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ                 NOT NULL DEFAULT now(),
    created_by      TEXT                        NOT NULL,
    is_deleted      BOOLEAN                     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_movements_sku
    ON pos.pos_inventory_movements (tenant_id, store_id, sku, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_movements_transaction
    ON pos.pos_inventory_movements (transaction_id);

-- =============================================================================
-- TABLE: pos_receipts
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_receipts (
    id               TEXT               NOT NULL,
    tenant_id        TEXT               NOT NULL,
    transaction_id   TEXT               NOT NULL REFERENCES pos.pos_transactions(id),
    session_id       TEXT               NOT NULL,
    receipt_number   TEXT               NOT NULL,
    receipt_format   pos.receipt_format NOT NULL DEFAULT 'thermal',
    recipient_email  TEXT,
    recipient_mobile TEXT,
    header_lines     TEXT[]             NOT NULL DEFAULT '{}',
    footer_lines     TEXT[]             NOT NULL DEFAULT '{}',
    logo_url         TEXT,
    receipt_payload  JSONB              NOT NULL DEFAULT '{}',
    rendered_content TEXT,
    issued_at        TIMESTAMPTZ        NOT NULL DEFAULT now(),
    delivered_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ        NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ        NOT NULL DEFAULT now(),
    created_by       TEXT               NOT NULL,
    is_deleted       BOOLEAN            NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_receipt_number
    ON pos.pos_receipts (tenant_id, receipt_number);

CREATE INDEX IF NOT EXISTS idx_receipts_transaction
    ON pos.pos_receipts (transaction_id);

-- =============================================================================
-- TABLE: pos_loyalty_transactions
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_loyalty_transactions (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    customer_id             TEXT            NOT NULL,
    transaction_id          TEXT            NOT NULL REFERENCES pos.pos_transactions(id),
    points_earned           INTEGER         NOT NULL DEFAULT 0,
    points_redeemed         INTEGER         NOT NULL DEFAULT 0,
    points_balance_before   INTEGER         NOT NULL DEFAULT 0,
    points_balance_after    INTEGER         NOT NULL DEFAULT 0,
    earn_rate               NUMERIC(8,4)    NOT NULL DEFAULT 1.0,
    redeem_rate             NUMERIC(8,4)    NOT NULL DEFAULT 0.01,
    occurred_at             TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by              TEXT            NOT NULL,
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_loyalty_customer
    ON pos.pos_loyalty_transactions (tenant_id, customer_id, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_loyalty_transaction
    ON pos.pos_loyalty_transactions (transaction_id);

-- =============================================================================
-- TABLE: pos_supervisor_overrides
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_supervisor_overrides (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    session_id      TEXT        NOT NULL REFERENCES pos.pos_sessions(id),
    terminal_id     TEXT        NOT NULL REFERENCES pos.pos_terminals(id),
    supervisor_id   TEXT        NOT NULL,
    override_type   TEXT        NOT NULL,  -- 'price_override' | 'discount_override' | 'void' | 'refund' | 'close_session'
    target_id       TEXT,
    notes           TEXT,
    granted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      TEXT        NOT NULL,
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_overrides_session
    ON pos.pos_supervisor_overrides (tenant_id, session_id, granted_at DESC);

-- =============================================================================
-- TABLE: pos_eod_reports
-- End-of-day reconciliation reports.
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_eod_reports (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    store_id                TEXT            NOT NULL,
    business_date           DATE            NOT NULL,
    session_count           INTEGER         NOT NULL DEFAULT 0,
    transaction_count       INTEGER         NOT NULL DEFAULT 0,
    gross_sales             NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_refunds           NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_discounts         NUMERIC(18,4)   NOT NULL DEFAULT 0,
    total_tax               NUMERIC(18,4)   NOT NULL DEFAULT 0,
    net_sales               NUMERIC(18,4)   NOT NULL DEFAULT 0,
    cash_sales              NUMERIC(18,4)   NOT NULL DEFAULT 0,
    card_sales              NUMERIC(18,4)   NOT NULL DEFAULT 0,
    mobile_sales            NUMERIC(18,4)   NOT NULL DEFAULT 0,
    loyalty_sales           NUMERIC(18,4)   NOT NULL DEFAULT 0,
    other_sales             NUMERIC(18,4)   NOT NULL DEFAULT 0,
    opening_floats_total    NUMERIC(18,4)   NOT NULL DEFAULT 0,
    safe_drops_total        NUMERIC(18,4)   NOT NULL DEFAULT 0,
    variance_total          NUMERIC(18,4)   NOT NULL DEFAULT 0,
    hourly_breakdown        JSONB           NOT NULL DEFAULT '[]',
    top_selling_skus        JSONB           NOT NULL DEFAULT '[]',
    status                  TEXT            NOT NULL DEFAULT 'draft',   -- draft | approved
    generated_by            TEXT            NOT NULL,
    approved_by             TEXT,
    approved_at             TIMESTAMPTZ,
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by              TEXT            NOT NULL,
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_eod_store_date
    ON pos.pos_eod_reports (tenant_id, store_id, business_date)
    WHERE NOT is_deleted;

CREATE INDEX IF NOT EXISTS idx_eod_store
    ON pos.pos_eod_reports (tenant_id, store_id, business_date DESC);

-- =============================================================================
-- TABLE: pos_offline_sync_log
-- Tracks offline sync batches to detect gaps / replays.
-- =============================================================================
CREATE TABLE IF NOT EXISTS pos.pos_offline_sync_log (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    terminal_id     TEXT        NOT NULL REFERENCES pos.pos_terminals(id),
    session_id      TEXT        NOT NULL,
    sync_sequence   INTEGER     NOT NULL,
    batch_size      INTEGER     NOT NULL DEFAULT 0,
    accepted_count  INTEGER     NOT NULL DEFAULT 0,
    rejected_count  INTEGER     NOT NULL DEFAULT 0,
    checksum        TEXT,
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      TEXT        NOT NULL,
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_offline_sync_seq
    ON pos.pos_offline_sync_log (tenant_id, terminal_id, sync_sequence);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Session totals view
CREATE OR REPLACE VIEW pos.v_session_totals AS
SELECT
    s.id,
    s.tenant_id,
    s.store_id,
    s.terminal_id,
    s.cashier_id,
    s.status,
    s.opening_float,
    s.closing_cash_counted,
    s.variance,
    COUNT(DISTINCT t.id) FILTER (WHERE t.transaction_type = 'sale' AND t.status = 'completed') AS sale_count,
    COALESCE(SUM(t.grand_total) FILTER (WHERE t.transaction_type = 'sale' AND t.status = 'completed'), 0) AS gross_sales,
    COALESCE(SUM(t.grand_total) FILTER (WHERE t.transaction_type = 'refund'), 0) AS total_refunds,
    s.opened_at,
    s.closed_at
FROM pos.pos_sessions s
LEFT JOIN pos.pos_transactions t ON t.session_id = s.id AND NOT t.is_deleted
WHERE NOT s.is_deleted
GROUP BY s.id;

-- Daily sales summary view
CREATE OR REPLACE VIEW pos.v_daily_sales AS
SELECT
    tenant_id,
    store_id,
    DATE(created_at AT TIME ZONE 'Africa/Nairobi') AS business_date,
    COUNT(*) FILTER (WHERE transaction_type = 'sale' AND status = 'completed') AS sale_count,
    COALESCE(SUM(grand_total) FILTER (WHERE transaction_type = 'sale' AND status = 'completed'), 0) AS gross_sales,
    COALESCE(SUM(grand_total) FILTER (WHERE transaction_type = 'refund'), 0) AS total_refunds,
    COALESCE(SUM(discount_total) FILTER (WHERE transaction_type = 'sale' AND status = 'completed'), 0) AS total_discounts,
    COALESCE(SUM(tax_total) FILTER (WHERE transaction_type = 'sale' AND status = 'completed'), 0) AS total_tax
FROM pos.pos_transactions
WHERE NOT is_deleted
GROUP BY tenant_id, store_id, DATE(created_at AT TIME ZONE 'Africa/Nairobi');

-- =============================================================================
-- TRIGGERS: updated_at auto-maintenance
-- =============================================================================
CREATE OR REPLACE FUNCTION pos.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DO $$ DECLARE
    tbl TEXT;
BEGIN
    FOR tbl IN SELECT unnest(ARRAY[
        'pos_terminals','pos_sessions','pos_transactions','pos_payments',
        'pos_refunds','pos_cash_floats','pos_discounts','pos_price_overrides',
        'pos_inventory_movements','pos_receipts','pos_loyalty_transactions',
        'pos_supervisor_overrides','pos_eod_reports'
    ]) LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_updated_at ON pos.%I;
             CREATE TRIGGER trg_updated_at
             BEFORE UPDATE ON pos.%I
             FOR EACH ROW EXECUTE FUNCTION pos.set_updated_at();',
            tbl, tbl
        );
    END LOOP;
END $$;

-- =============================================================================
-- GRANT (adjust roles to match your deployment)
-- =============================================================================
-- GRANT USAGE ON SCHEMA pos TO apg_app;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pos TO apg_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA pos TO apg_app;
