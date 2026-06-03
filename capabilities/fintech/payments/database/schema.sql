-- APG Digital Payments — Production PostgreSQL Schema
-- Africa-first: M-Pesa, MTN MoMo, Airtel Money, Card, SWIFT, EFT
--
-- Run: psql $DATABASE_URL -f database/schema.sql
-- Requires: PostgreSQL 14+
-- Partitioning: payment_transactions partitioned by created_at (monthly)
--
-- © 2025 Datacraft. All rights reserved.

-- ============================================================
-- Extensions
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- for LIKE/ILIKE indexes on reference fields

-- ============================================================
-- Enumerations
-- ============================================================

DO $$ BEGIN
	CREATE TYPE payment_status AS ENUM (
		'pending', 'initiated', 'processing', 'completed',
		'failed', 'reversed', 'refunded', 'disputed', 'expired', 'captured', 'settled'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE payment_method AS ENUM (
		'mpesa_stk', 'mpesa_b2c', 'mpesa_b2b',
		'mtn_momo', 'airtel_money', 'tigo_pesa',
		'card_visa', 'card_mastercard',
		'bank_eft', 'swift', 'rtgs', 'pesalink',
		'cash', 'ussd', 'qr_code'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE transaction_type AS ENUM (
		'payment', 'refund', 'reversal', 'top_up', 'withdrawal',
		'transfer', 'settlement', 'charge', 'fee', 'adjustment'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE kyc_tier AS ENUM ('basic', 'standard', 'full_kyc', 'enhanced');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE dispute_status AS ENUM ('opened', 'under_review', 'resolved', 'closed');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE risk_level AS ENUM ('low', 'medium', 'high', 'blocked');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fx_rate_type AS ENUM ('spot', 'forward', 'cross', 'interbank', 'cbk', 'cbn', 'bou');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ============================================================
-- Generic audit columns function
-- ============================================================

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
	NEW.updated_at := now();
	RETURN NEW;
END;
$$;

-- ============================================================
-- 1. payment_orders
--    Top-level payment intent; created before any transaction.
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_orders (
	id                  TEXT            PRIMARY KEY,
	tenant_id           TEXT            NOT NULL,
	account_id          TEXT            NOT NULL,
	instrument_id       TEXT,
	amount              NUMERIC(20, 6)  NOT NULL CHECK (amount > 0),
	currency            CHAR(3)         NOT NULL,
	counterparty_ref    TEXT            NOT NULL DEFAULT '',
	purpose             TEXT            NOT NULL DEFAULT 'payment',
	status              payment_status  NOT NULL DEFAULT 'pending',
	kyc_tier            kyc_tier        NOT NULL DEFAULT 'basic',
	risk_level          risk_level      NOT NULL DEFAULT 'low',
	risk_score          NUMERIC(5, 2)   NOT NULL DEFAULT 0,
	idempotency_key     TEXT            NOT NULL DEFAULT '',
	metadata            JSONB           NOT NULL DEFAULT '{}',
	created_by          TEXT            NOT NULL DEFAULT '',
	created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_po_tenant         ON payment_orders (tenant_id);
CREATE INDEX IF NOT EXISTS idx_po_status         ON payment_orders (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_po_account        ON payment_orders (tenant_id, account_id);
CREATE INDEX IF NOT EXISTS idx_po_idempotency    ON payment_orders (tenant_id, idempotency_key)
	WHERE idempotency_key <> '';
CREATE INDEX IF NOT EXISTS idx_po_created        ON payment_orders (tenant_id, created_at DESC);

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_po_updated_at') THEN
		CREATE TRIGGER trg_po_updated_at
			BEFORE UPDATE ON payment_orders
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 2. payment_transactions
--    Individual transaction attempts; one order can have many
--    (retries, partial captures, refunds).
--    Partitioned monthly for performance at high volume.
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_transactions (
	id                  TEXT            NOT NULL,
	tenant_id           TEXT            NOT NULL,
	order_id            TEXT            NOT NULL REFERENCES payment_orders(id) ON DELETE RESTRICT,
	transaction_type    transaction_type NOT NULL DEFAULT 'payment',
	method              payment_method  NOT NULL,
	amount              NUMERIC(20, 6)  NOT NULL,
	currency            CHAR(3)         NOT NULL,
	status              payment_status  NOT NULL DEFAULT 'initiated',
	provider_ref        TEXT,
	provider_status     TEXT,
	recipient           TEXT            NOT NULL DEFAULT '',
	sender              TEXT            NOT NULL DEFAULT '',
	reference           TEXT            NOT NULL DEFAULT '',
	fee_amount          NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	excise_tax          NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	fx_rate             NUMERIC(20, 8),
	idempotency_key     TEXT            NOT NULL DEFAULT '',
	retry_count         SMALLINT        NOT NULL DEFAULT 0,
	created_by          TEXT            NOT NULL DEFAULT '',
	created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
	completed_at        TIMESTAMPTZ,
	metadata            JSONB           NOT NULL DEFAULT '{}',
	is_deleted          BOOLEAN         NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create partitions for current and next 12 months (extend as needed)
CREATE TABLE IF NOT EXISTS payment_transactions_2025_01 PARTITION OF payment_transactions
	FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE IF NOT EXISTS payment_transactions_2025_06 PARTITION OF payment_transactions
	FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE IF NOT EXISTS payment_transactions_2025_12 PARTITION OF payment_transactions
	FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');
CREATE TABLE IF NOT EXISTS payment_transactions_2026_01 PARTITION OF payment_transactions
	FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE IF NOT EXISTS payment_transactions_2026_06 PARTITION OF payment_transactions
	FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS payment_transactions_default PARTITION OF payment_transactions DEFAULT;

CREATE INDEX IF NOT EXISTS idx_txn_tenant        ON payment_transactions (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_txn_order         ON payment_transactions (order_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_txn_status        ON payment_transactions (tenant_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_txn_method        ON payment_transactions (tenant_id, method, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_txn_provider_ref  ON payment_transactions (provider_ref)
	WHERE provider_ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_txn_idempotency   ON payment_transactions (tenant_id, idempotency_key)
	WHERE idempotency_key <> '';
CREATE INDEX IF NOT EXISTS idx_txn_recipient_trgm ON payment_transactions
	USING gin (recipient gin_trgm_ops);

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_txn_updated_at') THEN
		CREATE TRIGGER trg_txn_updated_at
			BEFORE UPDATE ON payment_transactions
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 3. payment_legs
--    Marketplace split payments (one txn → N legs)
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_legs (
	id              TEXT            PRIMARY KEY,
	transaction_id  TEXT            NOT NULL,
	merchant_id     TEXT            NOT NULL,
	amount          NUMERIC(20, 6)  NOT NULL CHECK (amount > 0),
	currency        CHAR(3)         NOT NULL,
	percentage      NUMERIC(5, 2)   NOT NULL DEFAULT 0,
	purpose         TEXT            NOT NULL DEFAULT '',
	settled         BOOLEAN         NOT NULL DEFAULT FALSE,
	created_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_legs_txn       ON payment_legs (transaction_id);
CREATE INDEX IF NOT EXISTS idx_legs_merchant  ON payment_legs (merchant_id);
CREATE INDEX IF NOT EXISTS idx_legs_unsettled ON payment_legs (merchant_id, settled)
	WHERE settled = FALSE;

-- ============================================================
-- 4. mobile_money_transactions
--    M-Pesa STK/B2C/B2B, MTN MoMo, Airtel, Tigo specifics
-- ============================================================

CREATE TABLE IF NOT EXISTS mobile_money_transactions (
	id              TEXT            PRIMARY KEY,
	tenant_id       TEXT            NOT NULL,
	transaction_id  TEXT,           -- FK to payment_transactions.id
	provider        TEXT            NOT NULL,   -- mpesa / mtn_momo / airtel / tigo
	msisdn          TEXT            NOT NULL,
	amount          NUMERIC(20, 6)  NOT NULL,
	currency        CHAR(3)         NOT NULL,
	external_id     TEXT            NOT NULL DEFAULT '',
	status          payment_status  NOT NULL DEFAULT 'initiated',
	provider_ref    TEXT,
	checkout_request_id TEXT,
	callback_url    TEXT            NOT NULL DEFAULT '',
	narration       TEXT            NOT NULL DEFAULT '',
	float_balance   NUMERIC(20, 6),
	raw_callback    JSONB,
	created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mm_tenant      ON mobile_money_transactions (tenant_id);
CREATE INDEX IF NOT EXISTS idx_mm_msisdn      ON mobile_money_transactions (tenant_id, msisdn);
CREATE INDEX IF NOT EXISTS idx_mm_provider    ON mobile_money_transactions (provider, status);
CREATE INDEX IF NOT EXISTS idx_mm_external_id ON mobile_money_transactions (external_id)
	WHERE external_id <> '';
CREATE INDEX IF NOT EXISTS idx_mm_checkout    ON mobile_money_transactions (checkout_request_id)
	WHERE checkout_request_id IS NOT NULL;

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_mm_updated_at') THEN
		CREATE TRIGGER trg_mm_updated_at
			BEFORE UPDATE ON mobile_money_transactions
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 5. card_transactions
--    PCI-DSS compliant: no raw PAN stored, token only
-- ============================================================

CREATE TABLE IF NOT EXISTS card_transactions (
	id               TEXT            PRIMARY KEY,
	tenant_id        TEXT            NOT NULL,
	transaction_id   TEXT,
	card_token       TEXT            NOT NULL,   -- vault reference, never raw PAN
	amount           NUMERIC(20, 6)  NOT NULL,
	currency         CHAR(3)         NOT NULL,
	merchant_id      TEXT            NOT NULL,
	cvv_result       CHAR(1)         NOT NULL DEFAULT 'M',
	avs_result       CHAR(1)         NOT NULL DEFAULT 'Y',
	auth_code        TEXT,
	rrn              TEXT,                         -- retrieval reference number
	status           payment_status  NOT NULL DEFAULT 'initiated',
	three_ds_result  TEXT,
	card_type        TEXT            NOT NULL DEFAULT 'standard',
	interchange_fee  NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	created_at       TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at       TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_card_tenant      ON card_transactions (tenant_id);
CREATE INDEX IF NOT EXISTS idx_card_token       ON card_transactions (card_token);
CREATE INDEX IF NOT EXISTS idx_card_merchant    ON card_transactions (merchant_id, status);
CREATE INDEX IF NOT EXISTS idx_card_rrn         ON card_transactions (rrn) WHERE rrn IS NOT NULL;

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_card_updated_at') THEN
		CREATE TRIGGER trg_card_updated_at
			BEFORE UPDATE ON card_transactions
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 6. bank_transfers
--    EFT / RTGS / PesaLink domestic transfers
-- ============================================================

CREATE TABLE IF NOT EXISTS bank_transfers (
	id              TEXT            PRIMARY KEY,
	tenant_id       TEXT            NOT NULL,
	transaction_id  TEXT,
	from_account    TEXT            NOT NULL,
	to_account      TEXT            NOT NULL,
	bank_code       TEXT            NOT NULL DEFAULT '',
	amount          NUMERIC(20, 6)  NOT NULL,
	currency        CHAR(3)         NOT NULL,
	reference       TEXT            NOT NULL,
	narration       TEXT            NOT NULL DEFAULT '',
	clearing_type   TEXT            NOT NULL DEFAULT 'eft',
	status          payment_status  NOT NULL DEFAULT 'initiated',
	value_date      DATE,
	created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bt_tenant    ON bank_transfers (tenant_id);
CREATE INDEX IF NOT EXISTS idx_bt_accounts  ON bank_transfers (from_account, to_account);
CREATE INDEX IF NOT EXISTS idx_bt_ref       ON bank_transfers (reference);

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_bt_updated_at') THEN
		CREATE TRIGGER trg_bt_updated_at
			BEFORE UPDATE ON bank_transfers
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 7. swift_payments
--    SWIFT cross-border transfers (ISO 9362 BIC, IBAN)
-- ============================================================

CREATE TABLE IF NOT EXISTS swift_payments (
	id              TEXT            PRIMARY KEY,
	tenant_id       TEXT            NOT NULL,
	transaction_id  TEXT,
	sender_bic      CHAR(11)        NOT NULL,
	receiver_bic    CHAR(11)        NOT NULL,
	iban            TEXT            NOT NULL,
	amount          NUMERIC(20, 6)  NOT NULL,
	currency        CHAR(3)         NOT NULL,
	purpose_code    CHAR(3)         NOT NULL DEFAULT 'OTH',
	charges         CHAR(3)         NOT NULL DEFAULT 'SHA',
	uetr            TEXT            NOT NULL,   -- unique end-to-end transaction reference
	status          payment_status  NOT NULL DEFAULT 'initiated',
	bank_fee        NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_swift_tenant  ON swift_payments (tenant_id);
CREATE INDEX IF NOT EXISTS idx_swift_bics    ON swift_payments (sender_bic, receiver_bic);
CREATE INDEX IF NOT EXISTS idx_swift_uetr    ON swift_payments (uetr);

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_swift_updated_at') THEN
		CREATE TRIGGER trg_swift_updated_at
			BEFORE UPDATE ON swift_payments
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 8. payment_refunds
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_refunds (
	id                  TEXT            PRIMARY KEY,
	tenant_id           TEXT            NOT NULL,
	original_txn_id     TEXT            NOT NULL,
	amount              NUMERIC(20, 6)  NOT NULL CHECK (amount > 0),
	reason              TEXT            NOT NULL,
	refund_to_original  BOOLEAN         NOT NULL DEFAULT TRUE,
	status              payment_status  NOT NULL DEFAULT 'initiated',
	provider_ref        TEXT,
	approved_by         TEXT,
	initiated_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
	completed_at        TIMESTAMPTZ,
	created_by          TEXT            NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_refund_tenant  ON payment_refunds (tenant_id);
CREATE INDEX IF NOT EXISTS idx_refund_txn     ON payment_refunds (original_txn_id);
CREATE INDEX IF NOT EXISTS idx_refund_status  ON payment_refunds (tenant_id, status);

-- ============================================================
-- 9. payment_reversals
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_reversals (
	id               TEXT            PRIMARY KEY,
	tenant_id        TEXT            NOT NULL,
	original_txn_id  TEXT            NOT NULL,
	reason           TEXT            NOT NULL,
	reversal_code    TEXT            NOT NULL DEFAULT '',
	amount           NUMERIC(20, 6)  NOT NULL,
	status           payment_status  NOT NULL DEFAULT 'initiated',
	window_expires   TIMESTAMPTZ,
	created_at       TIMESTAMPTZ     NOT NULL DEFAULT now(),
	created_by       TEXT            NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_rev_tenant  ON payment_reversals (tenant_id);
CREATE INDEX IF NOT EXISTS idx_rev_txn     ON payment_reversals (original_txn_id);
CREATE INDEX IF NOT EXISTS idx_rev_window  ON payment_reversals (window_expires)
	WHERE window_expires IS NOT NULL;

-- ============================================================
-- 10. fx_conversions
-- ============================================================

CREATE TABLE IF NOT EXISTS fx_conversions (
	id              TEXT            PRIMARY KEY,
	tenant_id       TEXT            NOT NULL,
	from_currency   CHAR(3)         NOT NULL,
	to_currency     CHAR(3)         NOT NULL,
	from_amount     NUMERIC(20, 6)  NOT NULL,
	to_amount       NUMERIC(20, 6)  NOT NULL,
	rate            NUMERIC(20, 8)  NOT NULL,
	rate_type       fx_rate_type    NOT NULL DEFAULT 'spot',
	provider        TEXT            NOT NULL DEFAULT 'CBK',
	spread_bps      SMALLINT        NOT NULL DEFAULT 150,
	quoted_at       TIMESTAMPTZ     NOT NULL DEFAULT now(),
	executed_at     TIMESTAMPTZ,
	rate_expires    TIMESTAMPTZ,
	created_by      TEXT            NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_fx_tenant    ON fx_conversions (tenant_id);
CREATE INDEX IF NOT EXISTS idx_fx_pair      ON fx_conversions (from_currency, to_currency, quoted_at DESC);

-- ============================================================
-- 11. settlement_batches
-- ============================================================

CREATE TABLE IF NOT EXISTS settlement_batches (
	id                TEXT            PRIMARY KEY,
	tenant_id         TEXT            NOT NULL,
	settlement_date   DATE            NOT NULL,
	bank_account      TEXT            NOT NULL,
	total_amount      NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	currency          CHAR(3)         NOT NULL DEFAULT 'KES',
	transaction_ids   TEXT[]          NOT NULL DEFAULT '{}',
	status            TEXT            NOT NULL DEFAULT 'pending',
	variance_amount   NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	review_id         TEXT            NOT NULL DEFAULT '',
	created_by        TEXT            NOT NULL DEFAULT '',
	created_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
	completed_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_settle_tenant  ON settlement_batches (tenant_id);
CREATE INDEX IF NOT EXISTS idx_settle_date    ON settlement_batches (tenant_id, settlement_date);
CREATE INDEX IF NOT EXISTS idx_settle_status  ON settlement_batches (tenant_id, status);

-- ============================================================
-- 12. reconciliation_records
-- ============================================================

CREATE TABLE IF NOT EXISTS reconciliation_records (
	id                TEXT            PRIMARY KEY,
	tenant_id         TEXT            NOT NULL,
	settlement_id     TEXT            NOT NULL REFERENCES settlement_batches(id) ON DELETE CASCADE,
	transaction_id    TEXT            NOT NULL,
	expected_amount   NUMERIC(20, 6)  NOT NULL,
	actual_amount     NUMERIC(20, 6)  NOT NULL,
	variance          NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	status            TEXT            NOT NULL DEFAULT 'matched',
	note              TEXT            NOT NULL DEFAULT '',
	reconciled_at     TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recon_settlement  ON reconciliation_records (settlement_id);
CREATE INDEX IF NOT EXISTS idx_recon_tenant      ON reconciliation_records (tenant_id);
CREATE INDEX IF NOT EXISTS idx_recon_status      ON reconciliation_records (tenant_id, status);

-- ============================================================
-- 13. merchant_accounts
-- ============================================================

CREATE TABLE IF NOT EXISTS merchant_accounts (
	id                  TEXT            PRIMARY KEY,
	tenant_id           TEXT            NOT NULL,
	name                TEXT            NOT NULL,
	category_code       CHAR(4)         NOT NULL DEFAULT '7372',
	settlement_account  TEXT            NOT NULL,
	paybill_number      TEXT,
	till_number         TEXT,
	status              TEXT            NOT NULL DEFAULT 'active',
	daily_limit         NUMERIC(20, 6)  NOT NULL DEFAULT 5000000,
	created_by          TEXT            NOT NULL DEFAULT '',
	created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
	metadata            JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_merch_tenant     ON merchant_accounts (tenant_id);
CREATE INDEX IF NOT EXISTS idx_merch_paybill    ON merchant_accounts (paybill_number)
	WHERE paybill_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_merch_till       ON merchant_accounts (till_number)
	WHERE till_number IS NOT NULL;

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_merch_updated_at') THEN
		CREATE TRIGGER trg_merch_updated_at
			BEFORE UPDATE ON merchant_accounts
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 14. virtual_accounts
-- ============================================================

CREATE TABLE IF NOT EXISTS virtual_accounts (
	id          TEXT            PRIMARY KEY,
	tenant_id   TEXT            NOT NULL,
	owner_id    TEXT            NOT NULL,
	currency    CHAR(3)         NOT NULL,
	balance     NUMERIC(20, 6)  NOT NULL DEFAULT 0 CHECK (balance >= 0),
	reserved    NUMERIC(20, 6)  NOT NULL DEFAULT 0 CHECK (reserved >= 0),
	status      TEXT            NOT NULL DEFAULT 'active',
	created_at  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at  TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_va_tenant  ON virtual_accounts (tenant_id);
CREATE INDEX IF NOT EXISTS idx_va_owner   ON virtual_accounts (tenant_id, owner_id);

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_va_updated_at') THEN
		CREATE TRIGGER trg_va_updated_at
			BEFORE UPDATE ON virtual_accounts
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 15. payment_receipts
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_receipts (
	id              TEXT            PRIMARY KEY,
	tenant_id       TEXT            NOT NULL,
	transaction_id  TEXT            NOT NULL,
	amount          NUMERIC(20, 6)  NOT NULL,
	currency        CHAR(3)         NOT NULL,
	method          payment_method  NOT NULL,
	recipient       TEXT            NOT NULL,
	reference       TEXT            NOT NULL,
	status          payment_status  NOT NULL,
	fee_amount      NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	excise_tax      NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	issued_at       TIMESTAMPTZ     NOT NULL DEFAULT now(),
	sms_sent        BOOLEAN         NOT NULL DEFAULT FALSE,
	email_sent      BOOLEAN         NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_receipt_tenant  ON payment_receipts (tenant_id);
CREATE INDEX IF NOT EXISTS idx_receipt_txn     ON payment_receipts (transaction_id);

-- ============================================================
-- 16. payment_disputes
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_disputes (
	id              TEXT            PRIMARY KEY,
	tenant_id       TEXT            NOT NULL,
	transaction_id  TEXT            NOT NULL,
	raised_by       TEXT            NOT NULL,
	reason          TEXT            NOT NULL,
	evidence        JSONB           NOT NULL DEFAULT '{}',
	status          dispute_status  NOT NULL DEFAULT 'opened',
	amount          NUMERIC(20, 6)  NOT NULL,
	created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
	resolved_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_disp_tenant  ON payment_disputes (tenant_id);
CREATE INDEX IF NOT EXISTS idx_disp_txn     ON payment_disputes (transaction_id);
CREATE INDEX IF NOT EXISTS idx_disp_status  ON payment_disputes (tenant_id, status);

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_disp_updated_at') THEN
		CREATE TRIGGER trg_disp_updated_at
			BEFORE UPDATE ON payment_disputes
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 17. chargeback_cases
-- ============================================================

CREATE TABLE IF NOT EXISTS chargeback_cases (
	id              TEXT            PRIMARY KEY,
	tenant_id       TEXT            NOT NULL,
	dispute_id      TEXT            NOT NULL REFERENCES payment_disputes(id) ON DELETE RESTRICT,
	transaction_id  TEXT            NOT NULL,
	amount          NUMERIC(20, 6)  NOT NULL,
	decision        TEXT            NOT NULL DEFAULT '',
	settled_amount  NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	reason_code     TEXT            NOT NULL DEFAULT '',
	scheme_fee      NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
	resolved_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_cb_tenant    ON chargeback_cases (tenant_id);
CREATE INDEX IF NOT EXISTS idx_cb_dispute   ON chargeback_cases (dispute_id);
CREATE INDEX IF NOT EXISTS idx_cb_txn       ON chargeback_cases (transaction_id);

-- ============================================================
-- 18. bulk_payment_batches
-- ============================================================

CREATE TABLE IF NOT EXISTS bulk_payment_batches (
	id              TEXT            PRIMARY KEY,
	tenant_id       TEXT            NOT NULL,
	payment_date    DATE            NOT NULL,
	method          payment_method  NOT NULL,
	currency        CHAR(3)         NOT NULL,
	total_amount    NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	status          TEXT            NOT NULL DEFAULT 'queued',
	processed       INTEGER         NOT NULL DEFAULT 0,
	failed          INTEGER         NOT NULL DEFAULT 0,
	validation_errors JSONB         NOT NULL DEFAULT '[]',
	created_by      TEXT            NOT NULL DEFAULT '',
	created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
	completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_bulk_tenant  ON bulk_payment_batches (tenant_id);
CREATE INDEX IF NOT EXISTS idx_bulk_status  ON bulk_payment_batches (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_bulk_date    ON bulk_payment_batches (payment_date);

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_bulk_updated_at') THEN
		CREATE TRIGGER trg_bulk_updated_at
			BEFORE UPDATE ON bulk_payment_batches
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 19. webhook_registrations
-- ============================================================

CREATE TABLE IF NOT EXISTS webhook_registrations (
	id           TEXT        PRIMARY KEY,
	tenant_id    TEXT        NOT NULL,
	event_types  TEXT[]      NOT NULL DEFAULT '{}',
	url          TEXT        NOT NULL,
	secret       TEXT        NOT NULL,
	active       BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_wh_tenant  ON webhook_registrations (tenant_id, active);

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_wh_updated_at') THEN
		CREATE TRIGGER trg_wh_updated_at
			BEFORE UPDATE ON webhook_registrations
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;

-- ============================================================
-- 20. payment_notifications
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_notifications (
	id              TEXT        PRIMARY KEY,
	tenant_id       TEXT        NOT NULL,
	transaction_id  TEXT        NOT NULL,
	channel         TEXT        NOT NULL DEFAULT 'sms',
	recipient       TEXT        NOT NULL,
	message         TEXT        NOT NULL,
	sent            BOOLEAN     NOT NULL DEFAULT FALSE,
	sent_at         TIMESTAMPTZ,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_notif_tenant  ON payment_notifications (tenant_id);
CREATE INDEX IF NOT EXISTS idx_notif_txn     ON payment_notifications (transaction_id);
CREATE INDEX IF NOT EXISTS idx_notif_unsent  ON payment_notifications (created_at)
	WHERE sent = FALSE;

-- ============================================================
-- 21. payment_audit_events
--    Append-only immutable audit log
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_audit_events (
	id              BIGSERIAL       PRIMARY KEY,
	tenant_id       TEXT            NOT NULL,
	event_type      TEXT            NOT NULL,
	actor_id        TEXT            NOT NULL,
	resource_id     TEXT            NOT NULL,
	resource_type   TEXT            NOT NULL DEFAULT '',
	payload         JSONB           NOT NULL DEFAULT '{}',
	ip_address      INET,
	created_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_tenant    ON payment_audit_events (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_resource  ON payment_audit_events (tenant_id, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_event     ON payment_audit_events (event_type, created_at DESC);

-- Prevent updates and deletes on audit table
CREATE OR REPLACE RULE no_update_audit AS ON UPDATE TO payment_audit_events DO INSTEAD NOTHING;
CREATE OR REPLACE RULE no_delete_audit AS ON DELETE TO payment_audit_events DO INSTEAD NOTHING;

-- ============================================================
-- 22. payment_idempotency_keys
--    Separate table for fast dedup checks
-- ============================================================

CREATE TABLE IF NOT EXISTS payment_idempotency_keys (
	key             TEXT        NOT NULL,
	tenant_id       TEXT        NOT NULL,
	response_id     TEXT        NOT NULL,
	response_type   TEXT        NOT NULL DEFAULT 'transaction',
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	expires_at      TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '24 hours'),
	PRIMARY KEY (tenant_id, key)
);

CREATE INDEX IF NOT EXISTS idx_idem_expires  ON payment_idempotency_keys (expires_at);

-- Purge expired keys nightly via pg_cron or application scheduler:
-- DELETE FROM payment_idempotency_keys WHERE expires_at < now();

-- ============================================================
-- 23. customer_limits_usage
--    Rolling daily/monthly limit tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS customer_limits_usage (
	tenant_id       TEXT            NOT NULL,
	customer_id     TEXT            NOT NULL,
	kyc_tier        kyc_tier        NOT NULL DEFAULT 'basic',
	currency        CHAR(3)         NOT NULL,
	period_date     DATE            NOT NULL,      -- daily bucket
	period_month    CHAR(7)         NOT NULL,      -- monthly bucket YYYY-MM
	daily_total     NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	monthly_total   NUMERIC(20, 6)  NOT NULL DEFAULT 0,
	txn_count_day   INTEGER         NOT NULL DEFAULT 0,
	txn_count_month INTEGER         NOT NULL DEFAULT 0,
	updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
	PRIMARY KEY (tenant_id, customer_id, currency, period_date)
);

CREATE INDEX IF NOT EXISTS idx_limits_monthly  ON customer_limits_usage (tenant_id, customer_id, period_month);

-- ============================================================
-- Compatibility: generic JSONB store for standalone operation
-- ============================================================

CREATE TABLE IF NOT EXISTS apg_records (
	id          TEXT        NOT NULL,
	collection  TEXT        NOT NULL,
	tenant_id   TEXT        NOT NULL DEFAULT 'default',
	data        JSONB       NOT NULL,
	created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	PRIMARY KEY (collection, id)
);

CREATE INDEX IF NOT EXISTS idx_apg_payments_tenant ON apg_records (collection, tenant_id);
CREATE INDEX IF NOT EXISTS idx_apg_payments_data   ON apg_records USING gin (data);

DO $$ BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_apg_records_updated_at') THEN
		CREATE TRIGGER trg_apg_records_updated_at
			BEFORE UPDATE ON apg_records
			FOR EACH ROW EXECUTE FUNCTION set_updated_at();
	END IF;
END $$;
