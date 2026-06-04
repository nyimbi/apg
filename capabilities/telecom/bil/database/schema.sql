-- =============================================================================
-- APG Telecom Billing — Complete PostgreSQL Schema
-- © 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
--
-- Conventions:
--   - All PKs are UUID7 strings (TEXT)
--   - All tables carry: id, tenant_id, created_at, updated_at, created_by, is_deleted
--   - Soft-deletes: is_deleted = TRUE (never hard-delete billing records)
--   - Amounts: NUMERIC(20,6) for precision; never FLOAT
--   - Timestamps: TIMESTAMPTZ everywhere
--   - Large tables (cdr, usage_event, rating_result) use RANGE partitioning on recorded_at/occurred_at
--
-- Run: psql $DATABASE_URL -f database/schema.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- for LIKE/ILIKE index on msisdn etc.

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS bil;
SET search_path TO bil, public;

-- ---------------------------------------------------------------------------
-- Audit helper: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION bil.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- ---------------------------------------------------------------------------
-- 1. billing_account
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.billing_account (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    customer_id         TEXT            NOT NULL,
    account_type        TEXT            NOT NULL CHECK (account_type IN (
                            'postpaid','prepaid','hybrid','wholesale','mvno','roaming_partner')),
    status              TEXT            NOT NULL DEFAULT 'active' CHECK (status IN (
                            'active','suspended','closed','pending','barred')),
    currency            CHAR(3)         NOT NULL DEFAULT 'KES',
    billing_day         SMALLINT        NOT NULL DEFAULT 1 CHECK (billing_day BETWEEN 1 AND 28),
    payment_terms_days  SMALLINT        NOT NULL DEFAULT 30 CHECK (payment_terms_days >= 0),
    parent_account_id   TEXT            REFERENCES bil.billing_account(id) ON DELETE SET NULL,
    credit_limit_id     TEXT,
    tax_id              TEXT,
    contact_email       TEXT,
    outstanding_balance NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (outstanding_balance >= 0),
    last_invoice_id     TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by          TEXT            NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_ba_tenant       ON bil.billing_account (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_ba_customer     ON bil.billing_account (tenant_id, customer_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_ba_status       ON bil.billing_account (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_ba_parent       ON bil.billing_account (parent_account_id) WHERE parent_account_id IS NOT NULL;

CREATE TRIGGER trig_ba_updated_at
    BEFORE UPDATE ON bil.billing_account
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 2. credit_limit
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.credit_limit (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    account_id              TEXT            NOT NULL REFERENCES bil.billing_account(id),
    hard_limit              NUMERIC(20,6)   NOT NULL CHECK (hard_limit > 0),
    soft_limit              NUMERIC(20,6)   NOT NULL CHECK (soft_limit > 0),
    current_usage           NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (current_usage >= 0),
    currency                CHAR(3)         NOT NULL DEFAULT 'KES',
    approval_reference      TEXT            NOT NULL,
    review_date             TIMESTAMPTZ,
    auto_suspend_at_hard    BOOLEAN         NOT NULL DEFAULT TRUE,
    alert_at_soft           BOOLEAN         NOT NULL DEFAULT TRUE,
    suspended_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              TEXT            NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id),
    CONSTRAINT chk_soft_below_hard CHECK (soft_limit < hard_limit)
);

CREATE INDEX IF NOT EXISTS idx_cl_tenant   ON bil.credit_limit (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_cl_account  ON bil.credit_limit (account_id) WHERE NOT is_deleted;

CREATE TRIGGER trig_cl_updated_at
    BEFORE UPDATE ON bil.credit_limit
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 3. tariff_plan
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.tariff_plan (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    name                    TEXT            NOT NULL,
    plan_type               TEXT            NOT NULL CHECK (plan_type IN (
                                'flat_rate','tiered','volume','stepped',
                                'time_of_day','geo_based','contract','promotional','pay_as_you_go')),
    currency                CHAR(3)         NOT NULL DEFAULT 'KES',
    base_rate               NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (base_rate >= 0),
    rate_per_second         NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (rate_per_second >= 0),
    rate_per_kb             NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (rate_per_kb >= 0),
    rate_per_sms            NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (rate_per_sms >= 0),
    minimum_charge          NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (minimum_charge >= 0),
    tiers                   JSONB           NOT NULL DEFAULT '[]',
    time_bands              JSONB           NOT NULL DEFAULT '[]',
    valid_from              TIMESTAMPTZ     NOT NULL,
    valid_to                TIMESTAMPTZ,
    is_active               BOOLEAN         NOT NULL DEFAULT TRUE,
    applicable_cdr_types    TEXT[]          NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              TEXT            NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_tp_tenant    ON bil.tariff_plan (tenant_id, is_active) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tp_validity  ON bil.tariff_plan (valid_from, valid_to) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tp_tiers     ON bil.tariff_plan USING gin (tiers);

CREATE TRIGGER trig_tp_updated_at
    BEFORE UPDATE ON bil.tariff_plan
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 4. cdr  (partitioned by recorded_at — monthly)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.cdr (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    cdr_type                TEXT            NOT NULL CHECK (cdr_type IN (
                                'voice','sms','data','mms','video_call',
                                'roaming','interconnect','short_code')),
    direction               TEXT            NOT NULL CHECK (direction IN (
                                'originating','terminating','transit','forwarded')),
    source                  TEXT            NOT NULL,
    msisdn                  TEXT            NOT NULL,
    called_number           TEXT,
    imsi                    TEXT,
    imei                    TEXT,
    cell_id                 TEXT,
    duration_seconds        INTEGER         NOT NULL DEFAULT 0 CHECK (duration_seconds >= 0),
    data_volume_bytes       BIGINT          NOT NULL DEFAULT 0 CHECK (data_volume_bytes >= 0),
    sms_count               INTEGER         NOT NULL DEFAULT 0 CHECK (sms_count >= 0),
    recorded_at             TIMESTAMPTZ     NOT NULL,
    network_id              TEXT,
    roaming_network         TEXT,
    interconnect_carrier    TEXT,
    mediation_status        TEXT            NOT NULL DEFAULT 'raw' CHECK (mediation_status IN (
                                'raw','normalised','rated','aggregated','billed',
                                'rejected','held','duplicate')),
    rating_result_id        TEXT,
    duplicate_of            TEXT,
    raw_record              JSONB           NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              TEXT            NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id, recorded_at)
) PARTITION BY RANGE (recorded_at);

-- Monthly partitions for current year (extend as needed)
CREATE TABLE IF NOT EXISTS bil.cdr_2026_01 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_02 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_03 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_04 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_05 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_06 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_07 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_08 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_09 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_10 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_11 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-11-01') TO ('2026-12-01');
CREATE TABLE IF NOT EXISTS bil.cdr_2026_12 PARTITION OF bil.cdr
    FOR VALUES FROM ('2026-12-01') TO ('2027-01-01');
CREATE TABLE IF NOT EXISTS bil.cdr_default PARTITION OF bil.cdr DEFAULT;

CREATE INDEX IF NOT EXISTS idx_cdr_tenant_msisdn     ON bil.cdr (tenant_id, msisdn, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_cdr_status            ON bil.cdr (tenant_id, mediation_status, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_cdr_type              ON bil.cdr (tenant_id, cdr_type, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_cdr_msisdn_trgm       ON bil.cdr USING gin (msisdn gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_cdr_raw               ON bil.cdr USING gin (raw_record);

-- ---------------------------------------------------------------------------
-- 5. usage_event  (partitioned by occurred_at)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.usage_event (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    account_id          TEXT            NOT NULL,
    service_id          TEXT            NOT NULL,
    event_type          TEXT            NOT NULL,
    quantity            NUMERIC(20,6)   NOT NULL CHECK (quantity >= 0),
    unit                TEXT            NOT NULL,
    occurred_at         TIMESTAMPTZ     NOT NULL,
    session_id          TEXT,
    network_element     TEXT,
    rated               BOOLEAN         NOT NULL DEFAULT FALSE,
    rating_result_id    TEXT,
    metadata            JSONB           NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by          TEXT            NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id, occurred_at)
) PARTITION BY RANGE (occurred_at);

CREATE TABLE IF NOT EXISTS bil.usage_event_2026_01 PARTITION OF bil.usage_event
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE IF NOT EXISTS bil.usage_event_2026_06 PARTITION OF bil.usage_event
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS bil.usage_event_default PARTITION OF bil.usage_event DEFAULT;

CREATE INDEX IF NOT EXISTS idx_ue_tenant_account ON bil.usage_event (tenant_id, account_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_ue_rated          ON bil.usage_event (tenant_id, rated, occurred_at DESC);

-- ---------------------------------------------------------------------------
-- 6. rating_result
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.rating_result (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    cdr_id                  TEXT,
    usage_event_id          TEXT,
    account_id              TEXT            NOT NULL,
    tariff_plan_id          TEXT            NOT NULL REFERENCES bil.tariff_plan(id),
    rated_amount            NUMERIC(20,6)   NOT NULL CHECK (rated_amount >= 0),
    tax_amount              NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (tax_amount >= 0),
    currency                CHAR(3)         NOT NULL DEFAULT 'KES',
    rating_type             TEXT            NOT NULL,
    bundle_id               TEXT,
    bundle_consumed_units   NUMERIC(20,6)   NOT NULL DEFAULT 0,
    discount_id             TEXT,
    discount_amount         NUMERIC(20,6)   NOT NULL DEFAULT 0,
    rated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    breakdown               JSONB           NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              TEXT            NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_rr_tenant_account ON bil.rating_result (tenant_id, account_id, rated_at DESC);
CREATE INDEX IF NOT EXISTS idx_rr_cdr            ON bil.rating_result (cdr_id) WHERE cdr_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rr_event          ON bil.rating_result (usage_event_id) WHERE usage_event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rr_tariff         ON bil.rating_result (tariff_plan_id, rated_at DESC);

-- ---------------------------------------------------------------------------
-- 7. bundle
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.bundle (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    account_id          TEXT            NOT NULL REFERENCES bil.billing_account(id),
    bundle_type         TEXT            NOT NULL CHECK (bundle_type IN (
                            'voice','data','sms','combo','unlimited','family','corporate','roaming')),
    name                TEXT            NOT NULL,
    total_units         NUMERIC(20,6)   NOT NULL CHECK (total_units > 0),
    consumed_units      NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (consumed_units >= 0),
    unit                TEXT            NOT NULL,
    price               NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (price >= 0),
    currency            CHAR(3)         NOT NULL DEFAULT 'KES',
    valid_from          TIMESTAMPTZ     NOT NULL,
    valid_to            TIMESTAMPTZ     NOT NULL,
    status              TEXT            NOT NULL DEFAULT 'active' CHECK (status IN (
                            'active','exhausted','expired','suspended','pending')),
    rollover_allowed    BOOLEAN         NOT NULL DEFAULT FALSE,
    rollover_units      NUMERIC(20,6)   NOT NULL DEFAULT 0,
    shared              BOOLEAN         NOT NULL DEFAULT FALSE,
    shared_with         TEXT[]          NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by          TEXT            NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id),
    CONSTRAINT chk_consumed_lte_total CHECK (consumed_units <= total_units + rollover_units + 1)
);

CREATE INDEX IF NOT EXISTS idx_bundle_tenant_account ON bil.bundle (tenant_id, account_id, status);
CREATE INDEX IF NOT EXISTS idx_bundle_validity       ON bil.bundle (valid_from, valid_to) WHERE status = 'active';

CREATE TRIGGER trig_bundle_updated_at
    BEFORE UPDATE ON bil.bundle
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 8. discount
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.discount (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    account_id              TEXT            NOT NULL REFERENCES bil.billing_account(id),
    discount_type           TEXT            NOT NULL CHECK (discount_type IN (
                                'loyalty','promotional','bulk','bundle','retention',
                                'corporate','staff','seasonal','regulatory')),
    discount_pct            NUMERIC(5,2)    NOT NULL DEFAULT 0
                                CHECK (discount_pct >= 0 AND discount_pct <= 50),
    flat_amount             NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (flat_amount >= 0),
    currency                CHAR(3)         NOT NULL DEFAULT 'KES',
    approval_reference      TEXT            NOT NULL,
    valid_from              TIMESTAMPTZ     NOT NULL,
    valid_to                TIMESTAMPTZ     NOT NULL,
    applicable_charge_types TEXT[]          NOT NULL DEFAULT '{}',
    applications_count      INTEGER         NOT NULL DEFAULT 0,
    max_applications        INTEGER,
    is_active               BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              TEXT            NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_disc_tenant_account ON bil.discount (tenant_id, account_id, is_active);
CREATE INDEX IF NOT EXISTS idx_disc_validity       ON bil.discount (valid_from, valid_to) WHERE is_active;

CREATE TRIGGER trig_disc_updated_at
    BEFORE UPDATE ON bil.discount
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 9. promotion
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.promotion (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    name                    TEXT            NOT NULL,
    description             TEXT            NOT NULL DEFAULT '',
    status                  TEXT            NOT NULL DEFAULT 'draft' CHECK (status IN (
                                'draft','active','paused','expired','cancelled')),
    discount_pct            NUMERIC(5,2)    NOT NULL DEFAULT 0
                                CHECK (discount_pct >= 0 AND discount_pct <= 100),
    bonus_units             NUMERIC(20,6)   NOT NULL DEFAULT 0,
    bonus_unit_type         TEXT,
    eligible_account_types  TEXT[]          NOT NULL DEFAULT '{}',
    promo_code              TEXT UNIQUE,
    valid_from              TIMESTAMPTZ     NOT NULL,
    valid_to                TIMESTAMPTZ     NOT NULL,
    redemption_count        INTEGER         NOT NULL DEFAULT 0,
    max_redemptions         INTEGER,
    budget_cap              NUMERIC(20,6),
    budget_consumed         NUMERIC(20,6)   NOT NULL DEFAULT 0,
    currency                CHAR(3)         NOT NULL DEFAULT 'KES',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              TEXT            NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_promo_tenant  ON bil.promotion (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_promo_code    ON bil.promotion (promo_code) WHERE promo_code IS NOT NULL;

CREATE TRIGGER trig_promo_updated_at
    BEFORE UPDATE ON bil.promotion
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 10. bill_cycle
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.bill_cycle (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    cycle_type      TEXT        NOT NULL CHECK (cycle_type IN (
                        'monthly','bi_monthly','quarterly','weekly','daily','anniversary')),
    cutoff_date     DATE        NOT NULL,
    start_date      DATE        NOT NULL,
    end_date        DATE        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'active' CHECK (status IN ('active','closed','suspended')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by      TEXT        NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id),
    CONSTRAINT chk_cycle_dates CHECK (start_date <= end_date)
);

CREATE INDEX IF NOT EXISTS idx_cycle_tenant ON bil.bill_cycle (tenant_id, status);

CREATE TRIGGER trig_cycle_updated_at
    BEFORE UPDATE ON bil.bill_cycle
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 11. invoice
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.invoice (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    account_id          TEXT            NOT NULL REFERENCES bil.billing_account(id),
    cycle_id            TEXT            REFERENCES bil.bill_cycle(id),
    period_start        TIMESTAMPTZ     NOT NULL,
    period_end          TIMESTAMPTZ     NOT NULL,
    subtotal            NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (subtotal >= 0),
    tax_amount          NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (tax_amount >= 0),
    discount_amount     NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (discount_amount >= 0),
    total_amount        NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (total_amount >= 0),
    paid_amount         NUMERIC(20,6)   NOT NULL DEFAULT 0 CHECK (paid_amount >= 0),
    currency            CHAR(3)         NOT NULL DEFAULT 'KES',
    status              TEXT            NOT NULL DEFAULT 'draft' CHECK (status IN (
                            'draft','pending_approval','approved','sent','paid',
                            'partially_paid','overdue','disputed','cancelled','written_off')),
    due_date            TIMESTAMPTZ     NOT NULL,
    approval_reference  TEXT,
    approved_at         TIMESTAMPTZ,
    sent_at             TIMESTAMPTZ,
    line_items          JSONB           NOT NULL DEFAULT '[]',
    dunning_step        TEXT            CHECK (dunning_step IN (
                            'reminder_1','reminder_2','suspension_warning',
                            'service_suspended','legal_notice','collections','write_off')),
    last_dunning_at     TIMESTAMPTZ,
    notes               TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by          TEXT            NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_inv_tenant_account ON bil.invoice (tenant_id, account_id, status);
CREATE INDEX IF NOT EXISTS idx_inv_due_date       ON bil.invoice (tenant_id, due_date, status);
CREATE INDEX IF NOT EXISTS idx_inv_status         ON bil.invoice (status, tenant_id) WHERE status IN ('overdue','disputed','draft');
CREATE INDEX IF NOT EXISTS idx_inv_line_items     ON bil.invoice USING gin (line_items);

CREATE TRIGGER trig_inv_updated_at
    BEFORE UPDATE ON bil.invoice
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 12. payment_allocation
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.payment_allocation (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    account_id          TEXT            NOT NULL REFERENCES bil.billing_account(id),
    invoice_id          TEXT            NOT NULL REFERENCES bil.invoice(id),
    payment_method      TEXT            NOT NULL CHECK (payment_method IN (
                            'bank_transfer','mobile_money','credit_card','debit_card',
                            'direct_debit','cheque','cash','voucher','crypto')),
    amount              NUMERIC(20,6)   NOT NULL CHECK (amount > 0),
    currency            CHAR(3)         NOT NULL DEFAULT 'KES',
    reference           TEXT            NOT NULL,
    paid_at             TIMESTAMPTZ     NOT NULL,
    gateway_reference   TEXT,
    allocated           BOOLEAN         NOT NULL DEFAULT FALSE,
    allocated_at        TIMESTAMPTZ,
    metadata            JSONB           NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by          TEXT            NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_pay_tenant_account ON bil.payment_allocation (tenant_id, account_id, paid_at DESC);
CREATE INDEX IF NOT EXISTS idx_pay_invoice        ON bil.payment_allocation (invoice_id, allocated);
CREATE INDEX IF NOT EXISTS idx_pay_reference      ON bil.payment_allocation (reference);
CREATE INDEX IF NOT EXISTS idx_pay_method         ON bil.payment_allocation (payment_method, tenant_id);

CREATE TRIGGER trig_pay_updated_at
    BEFORE UPDATE ON bil.payment_allocation
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 13. roaming
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.roaming (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    account_id          TEXT            NOT NULL REFERENCES bil.billing_account(id),
    cdr_id              TEXT            NOT NULL,
    zone                TEXT            NOT NULL CHECK (zone IN (
                            'domestic','zone_a','zone_b','zone_c','premium','global')),
    visited_network     TEXT            NOT NULL,
    home_network        TEXT            NOT NULL,
    service_type        TEXT            NOT NULL CHECK (service_type IN (
                            'voice','sms','data','mms','video_call',
                            'roaming','interconnect','short_code')),
    duration_seconds    INTEGER         NOT NULL DEFAULT 0,
    data_volume_bytes   BIGINT          NOT NULL DEFAULT 0,
    base_charge         NUMERIC(20,6)   NOT NULL CHECK (base_charge >= 0),
    surcharge           NUMERIC(20,6)   NOT NULL DEFAULT 0,
    total_charge        NUMERIC(20,6)   NOT NULL DEFAULT 0,
    currency            CHAR(3)         NOT NULL DEFAULT 'KES',
    tap_file_reference  TEXT,
    settled             BOOLEAN         NOT NULL DEFAULT FALSE,
    settlement_id       TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by          TEXT            NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_roam_tenant_account ON bil.roaming (tenant_id, account_id);
CREATE INDEX IF NOT EXISTS idx_roam_zone           ON bil.roaming (zone, tenant_id);
CREATE INDEX IF NOT EXISTS idx_roam_unsettled      ON bil.roaming (settled, tenant_id) WHERE NOT settled;
CREATE INDEX IF NOT EXISTS idx_roam_cdr            ON bil.roaming (cdr_id);

CREATE TRIGGER trig_roam_updated_at
    BEFORE UPDATE ON bil.roaming
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 14. interconnect_settlement
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.interconnect_settlement (
    id                      TEXT            NOT NULL,
    tenant_id               TEXT            NOT NULL,
    carrier_id              TEXT            NOT NULL,
    carrier_name            TEXT            NOT NULL,
    period_start            TIMESTAMPTZ     NOT NULL,
    period_end              TIMESTAMPTZ     NOT NULL,
    originating_minutes     NUMERIC(20,6)   NOT NULL DEFAULT 0,
    terminating_minutes     NUMERIC(20,6)   NOT NULL DEFAULT 0,
    transit_minutes         NUMERIC(20,6)   NOT NULL DEFAULT 0,
    data_gb                 NUMERIC(20,6)   NOT NULL DEFAULT 0,
    receivable_amount       NUMERIC(20,6)   NOT NULL CHECK (receivable_amount >= 0),
    payable_amount          NUMERIC(20,6)   NOT NULL CHECK (payable_amount >= 0),
    net_amount              NUMERIC(20,6)   NOT NULL DEFAULT 0,
    currency                CHAR(3)         NOT NULL DEFAULT 'KES',
    status                  TEXT            NOT NULL DEFAULT 'draft' CHECK (status IN (
                                'draft','submitted','acknowledged','disputed','agreed','paid','overdue')),
    reference_number        TEXT            NOT NULL,
    dispute_reference       TEXT,
    paid_at                 TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              TEXT            NOT NULL DEFAULT 'system',
    is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id),
    CONSTRAINT chk_settlement_period CHECK (period_end > period_start)
);

CREATE INDEX IF NOT EXISTS idx_ic_tenant_carrier ON bil.interconnect_settlement (tenant_id, carrier_id, period_start DESC);
CREATE INDEX IF NOT EXISTS idx_ic_status         ON bil.interconnect_settlement (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ic_reference      ON bil.interconnect_settlement (reference_number);

CREATE TRIGGER trig_ic_updated_at
    BEFORE UPDATE ON bil.interconnect_settlement
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 15. dispute
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.dispute (
    id                  TEXT            NOT NULL,
    tenant_id           TEXT            NOT NULL,
    account_id          TEXT            NOT NULL REFERENCES bil.billing_account(id),
    invoice_id          TEXT            REFERENCES bil.invoice(id),
    cdr_id              TEXT,
    dispute_type        TEXT            NOT NULL CHECK (dispute_type IN (
                            'billing_error','service_quality','unauthorised_charge',
                            'roaming_dispute','interconnect_dispute','fraud','other')),
    disputed_amount     NUMERIC(20,6)   NOT NULL CHECK (disputed_amount >= 0),
    currency            CHAR(3)         NOT NULL DEFAULT 'KES',
    reason              TEXT            NOT NULL,
    status              TEXT            NOT NULL DEFAULT 'open' CHECK (status IN (
                            'open','under_review','evidence_requested','resolved_upheld',
                            'resolved_rejected','escalated','withdrawn','arbitration')),
    evidence_refs       TEXT[]          NOT NULL DEFAULT '{}',
    resolution_notes    TEXT,
    credit_amount       NUMERIC(20,6)   NOT NULL DEFAULT 0,
    resolver_id         TEXT,
    resolved_at         TIMESTAMPTZ,
    sla_deadline        TIMESTAMPTZ,
    raised_by           TEXT            NOT NULL DEFAULT 'system',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by          TEXT            NOT NULL DEFAULT 'system',
    is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_disp_tenant_account ON bil.dispute (tenant_id, account_id, status);
CREATE INDEX IF NOT EXISTS idx_disp_open           ON bil.dispute (tenant_id, status) WHERE status IN ('open','under_review');
CREATE INDEX IF NOT EXISTS idx_disp_invoice        ON bil.dispute (invoice_id) WHERE invoice_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_disp_sla            ON bil.dispute (sla_deadline) WHERE status NOT IN ('resolved_upheld','resolved_rejected','withdrawn');

CREATE TRIGGER trig_disp_updated_at
    BEFORE UPDATE ON bil.dispute
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 16. dunning_step  (audit trail of every dunning action)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.dunning_step (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    invoice_id      TEXT        NOT NULL REFERENCES bil.invoice(id),
    account_id      TEXT        NOT NULL REFERENCES bil.billing_account(id),
    step            TEXT        NOT NULL CHECK (step IN (
                        'reminder_1','reminder_2','suspension_warning',
                        'service_suspended','legal_notice','collections','write_off')),
    triggered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    next_step_date  TIMESTAMPTZ,
    channel         TEXT        CHECK (channel IN ('sms','email','push','post','legal')),
    message_ref     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by      TEXT        NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_dun_invoice ON bil.dunning_step (invoice_id, triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_dun_account ON bil.dunning_step (tenant_id, account_id, triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_dun_next    ON bil.dunning_step (next_step_date) WHERE next_step_date IS NOT NULL;

CREATE TRIGGER trig_dun_updated_at
    BEFORE UPDATE ON bil.dunning_step
    FOR EACH ROW EXECUTE FUNCTION bil.set_updated_at();

-- ---------------------------------------------------------------------------
-- 17. audit_event  (immutable event log — no UPDATE, no DELETE)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bil.audit_event (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    actor_id        TEXT        NOT NULL,
    event_type      TEXT        NOT NULL,
    reference_id    TEXT        NOT NULL,
    stream          TEXT        NOT NULL DEFAULT 'apg.telecom.bil.lifecycle',
    processor       TEXT        NOT NULL DEFAULT 'bytewax',
    payload         JSONB       NOT NULL DEFAULT '{}',
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id)
) PARTITION BY RANGE (occurred_at);

CREATE TABLE IF NOT EXISTS bil.audit_event_2026_h1 PARTITION OF bil.audit_event
    FOR VALUES FROM ('2026-01-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS bil.audit_event_2026_h2 PARTITION OF bil.audit_event
    FOR VALUES FROM ('2026-07-01') TO ('2027-01-01');
CREATE TABLE IF NOT EXISTS bil.audit_event_default PARTITION OF bil.audit_event DEFAULT;

CREATE INDEX IF NOT EXISTS idx_audit_tenant     ON bil.audit_event (tenant_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_ref        ON bil.audit_event (reference_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_type       ON bil.audit_event (event_type, tenant_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_payload    ON bil.audit_event USING gin (payload);

-- ---------------------------------------------------------------------------
-- 18. Materialized view: monthly revenue summary (refresh periodically)
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS bil.mv_monthly_revenue AS
SELECT
    i.tenant_id,
    DATE_TRUNC('month', i.period_start)   AS revenue_month,
    COUNT(*)                               AS invoice_count,
    COUNT(*) FILTER (WHERE i.status = 'paid')      AS paid_count,
    SUM(i.subtotal)                        AS total_subtotal,
    SUM(i.tax_amount)                      AS total_tax,
    SUM(i.discount_amount)                 AS total_discount,
    SUM(i.total_amount)                    AS total_invoiced,
    SUM(i.paid_amount)                     AS total_collected,
    SUM(i.total_amount) - SUM(i.paid_amount) AS outstanding,
    i.currency
FROM bil.invoice i
WHERE NOT i.is_deleted
GROUP BY i.tenant_id, DATE_TRUNC('month', i.period_start), i.currency
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_monthly_rev
    ON bil.mv_monthly_revenue (tenant_id, revenue_month, currency);

-- Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY bil.mv_monthly_revenue;

-- ---------------------------------------------------------------------------
-- 19. Materialized view: account balance snapshot
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS bil.mv_account_balance AS
SELECT
    ba.id               AS account_id,
    ba.tenant_id,
    ba.customer_id,
    ba.currency,
    COALESCE(SUM(i.total_amount), 0)    AS total_invoiced,
    COALESCE(SUM(i.paid_amount), 0)     AS total_paid,
    COALESCE(SUM(i.total_amount) - SUM(i.paid_amount), 0) AS balance_due,
    COUNT(i.*) FILTER (WHERE i.status = 'overdue')  AS overdue_invoice_count
FROM bil.billing_account ba
LEFT JOIN bil.invoice i ON i.account_id = ba.id AND NOT i.is_deleted
WHERE NOT ba.is_deleted
GROUP BY ba.id, ba.tenant_id, ba.customer_id, ba.currency
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_account_balance
    ON bil.mv_account_balance (account_id);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE bil.cdr                        IS 'Call Detail Records from network elements — partitioned monthly';
COMMENT ON TABLE bil.billing_account            IS 'Customer billing relationships (prepaid/postpaid/wholesale/MVNO)';
COMMENT ON TABLE bil.invoice                    IS 'Customer bills covering a billing period';
COMMENT ON TABLE bil.payment_allocation         IS 'Received payments allocated to invoices';
COMMENT ON TABLE bil.dispute                    IS 'Customer billing disputes with SLA tracking';
COMMENT ON TABLE bil.interconnect_settlement    IS 'Bilateral carrier interconnect settlements';
COMMENT ON TABLE bil.roaming                    IS 'Roaming charges from TAP/NRTRDE files or live events';
COMMENT ON TABLE bil.audit_event                IS 'Immutable billing event log — partitioned half-yearly';
COMMENT ON MATERIALIZED VIEW bil.mv_monthly_revenue   IS 'Pre-aggregated monthly revenue KPIs — refresh nightly';
COMMENT ON MATERIALIZED VIEW bil.mv_account_balance   IS 'Current balance snapshot per account — refresh hourly';
