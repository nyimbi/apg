-- =============================================================================
-- Cash Management — PostgreSQL Schema
-- APG Platform | fin.cbm.cash_management
-- © 2025 Datacraft. All rights reserved.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

CREATE SCHEMA IF NOT EXISTS cbm;
SET search_path TO cbm, public;

-- =============================================================================
-- Lookup / reference
-- =============================================================================

CREATE TABLE cbm_currency (
	code		CHAR(3) PRIMARY KEY,
	name		VARCHAR(60) NOT NULL,
	symbol		VARCHAR(5),
	decimal_places	SMALLINT NOT NULL DEFAULT 2
);

INSERT INTO cbm_currency VALUES
	('USD','US Dollar','$',2),
	('EUR','Euro','€',2),
	('GBP','Pound Sterling','£',2),
	('KES','Kenyan Shilling','KSh',2),
	('ZAR','South African Rand','R',2),
	('NGN','Nigerian Naira','₦',2),
	('GHS','Ghanaian Cedi','GH₵',2),
	('UGX','Ugandan Shilling','USh',0),
	('TZS','Tanzanian Shilling','TSh',0),
	('JPY','Japanese Yen','¥',0),
	('CHF','Swiss Franc','Fr',2),
	('CNY','Chinese Yuan','¥',2);

-- =============================================================================
-- Core entities
-- =============================================================================

CREATE TABLE cbm_bank_account (
	id				VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id			VARCHAR(36) NOT NULL,
	entity_id			VARCHAR(36) NOT NULL,
	bank_code			VARCHAR(20) NOT NULL,
	bank_name			VARCHAR(200) NOT NULL,
	account_number			VARCHAR(50) NOT NULL,
	account_name			VARCHAR(200) NOT NULL,
	account_type			VARCHAR(30) NOT NULL,
	currency			CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	status				VARCHAR(20) NOT NULL DEFAULT 'active',
	iban				VARCHAR(34),
	swift_bic			VARCHAR(11),
	routing_number			VARCHAR(30),
	current_balance			NUMERIC(20,4) NOT NULL DEFAULT 0,
	available_balance		NUMERIC(20,4) NOT NULL DEFAULT 0,
	ledger_balance			NUMERIC(20,4) NOT NULL DEFAULT 0,
	overdraft_limit			NUMERIC(20,4) NOT NULL DEFAULT 0,
	minimum_balance			NUMERIC(20,4) NOT NULL DEFAULT 0,
	revolving_credit_limit		NUMERIC(20,4) NOT NULL DEFAULT 0,
	revolving_credit_utilised	NUMERIC(20,4) NOT NULL DEFAULT 0,
	is_restricted			BOOLEAN NOT NULL DEFAULT FALSE,
	restriction_reason		TEXT,
	country_code			CHAR(2) NOT NULL DEFAULT 'US',
	branch_code			VARCHAR(20),
	last_statement_date		DATE,
	last_reconciled_date		DATE,
	open_banking_enabled		BOOLEAN NOT NULL DEFAULT FALSE,
	open_banking_provider		VARCHAR(50),
	open_banking_consent_expiry	DATE,
	notes				TEXT,
	created_at			TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at			TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by			VARCHAR(100) NOT NULL,
	is_deleted			BOOLEAN NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, account_number, bank_code)
);

CREATE TABLE cbm_bank_statement (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	account_id		VARCHAR(36) NOT NULL REFERENCES cbm_bank_account(id),
	statement_date		DATE NOT NULL,
	format			VARCHAR(20) NOT NULL,
	opening_balance		NUMERIC(20,4) NOT NULL,
	closing_balance		NUMERIC(20,4) NOT NULL,
	currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	transaction_count	INTEGER NOT NULL DEFAULT 0,
	status			VARCHAR(20) NOT NULL DEFAULT 'pending',
	raw_content		TEXT,
	file_name		VARCHAR(255),
	file_hash		VARCHAR(64),
	import_errors		JSONB DEFAULT '[]',
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, account_id, statement_date, format)
);

CREATE TABLE cbm_bank_transaction (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	account_id		VARCHAR(36) NOT NULL REFERENCES cbm_bank_account(id),
	statement_id		VARCHAR(36) REFERENCES cbm_bank_statement(id),
	transaction_date	DATE NOT NULL,
	value_date		DATE NOT NULL,
	posting_date		DATE,
	transaction_type	VARCHAR(20) NOT NULL,
	amount			NUMERIC(20,4) NOT NULL,
	currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	running_balance		NUMERIC(20,4),
	reference		VARCHAR(100),
	bank_reference		VARCHAR(100),
	description		TEXT,
	counterparty_name	VARCHAR(200),
	counterparty_account	VARCHAR(50),
	counterparty_bank	VARCHAR(50),
	status			VARCHAR(20) NOT NULL DEFAULT 'pending',
	swift_gpi_uetr		VARCHAR(36),
	is_same_day_value	BOOLEAN NOT NULL DEFAULT FALSE,
	float_days		SMALLINT NOT NULL DEFAULT 0,
	reconciliation_id	VARCHAR(36),
	gl_account		VARCHAR(30),
	cost_centre		VARCHAR(30),
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE cbm_cash_flow (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	account_id		VARCHAR(36) NOT NULL REFERENCES cbm_bank_account(id),
	flow_type		VARCHAR(20) NOT NULL,
	category		VARCHAR(50) NOT NULL,
	sub_category		VARCHAR(50),
	amount			NUMERIC(20,4) NOT NULL,
	currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	expected_date		DATE NOT NULL,
	actual_date		DATE,
	description		TEXT,
	reference		VARCHAR(100),
	status			VARCHAR(20) NOT NULL DEFAULT 'forecast',
	confidence_score	NUMERIC(4,3) DEFAULT 1.0,
	transaction_id		VARCHAR(36) REFERENCES cbm_bank_transaction(id),
	entity_id		VARCHAR(36),
	counterparty		VARCHAR(200),
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE cbm_cash_position (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	account_id		VARCHAR(36) NOT NULL REFERENCES cbm_bank_account(id),
	position_date		DATE NOT NULL,
	opening_balance		NUMERIC(20,4) NOT NULL DEFAULT 0,
	closing_balance		NUMERIC(20,4) NOT NULL DEFAULT 0,
	available_balance	NUMERIC(20,4) NOT NULL DEFAULT 0,
	ledger_balance		NUMERIC(20,4) NOT NULL DEFAULT 0,
	inflows			NUMERIC(20,4) NOT NULL DEFAULT 0,
	outflows		NUMERIC(20,4) NOT NULL DEFAULT 0,
	net_flow		NUMERIC(20,4) GENERATED ALWAYS AS (inflows - outflows) STORED,
	currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	status			VARCHAR(20) NOT NULL DEFAULT 'draft',
	reviewed_by		VARCHAR(100),
	reviewed_at		TIMESTAMPTZ,
	notes			TEXT,
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, account_id, position_date)
);

CREATE TABLE cbm_cash_forecast (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	forecast_name		VARCHAR(200) NOT NULL,
	horizon_days		INTEGER NOT NULL,
	scenario		VARCHAR(20) NOT NULL DEFAULT 'base',
	base_currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	confidence_score	NUMERIC(4,3),
	mape			NUMERIC(6,3),
	status			VARCHAR(20) NOT NULL DEFAULT 'draft',
	forecast_date		DATE NOT NULL,
	forecast_lines		JSONB NOT NULL DEFAULT '[]',
	model_metadata		JSONB DEFAULT '{}',
	reviewed_by		VARCHAR(100),
	reviewed_at		TIMESTAMPTZ,
	approved_by		VARCHAR(100),
	approved_at		TIMESTAMPTZ,
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE cbm_bank_reconciliation (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	account_id		VARCHAR(36) NOT NULL REFERENCES cbm_bank_account(id),
	statement_id		VARCHAR(36) REFERENCES cbm_bank_statement(id),
	reconciliation_date	DATE NOT NULL,
	bank_balance		NUMERIC(20,4) NOT NULL,
	book_balance		NUMERIC(20,4) NOT NULL,
	variance		NUMERIC(20,4) GENERATED ALWAYS AS (bank_balance - book_balance) STORED,
	outstanding_deposits	NUMERIC(20,4) NOT NULL DEFAULT 0,
	outstanding_payments	NUMERIC(20,4) NOT NULL DEFAULT 0,
	bank_errors		NUMERIC(20,4) NOT NULL DEFAULT 0,
	book_errors		NUMERIC(20,4) NOT NULL DEFAULT 0,
	adjusted_bank_balance	NUMERIC(20,4),
	matched_count		INTEGER NOT NULL DEFAULT 0,
	unmatched_count		INTEGER NOT NULL DEFAULT 0,
	exception_count		INTEGER NOT NULL DEFAULT 0,
	status			VARCHAR(30) NOT NULL DEFAULT 'unreconciled',
	variance_threshold	NUMERIC(20,4) NOT NULL DEFAULT 0.01,
	reviewed_by		VARCHAR(100),
	reviewed_at		TIMESTAMPTZ,
	approved_by		VARCHAR(100),
	approved_at		TIMESTAMPTZ,
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE cbm_liquidity_pool (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	pool_name		VARCHAR(200) NOT NULL,
	pooling_type		VARCHAR(20) NOT NULL,
	base_currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	header_account_id	VARCHAR(36) REFERENCES cbm_bank_account(id),
	notional_balance	NUMERIC(20,4) NOT NULL DEFAULT 0,
	effective_balance	NUMERIC(20,4) NOT NULL DEFAULT 0,
	interest_savings	NUMERIC(20,4) NOT NULL DEFAULT 0,
	status			VARCHAR(20) NOT NULL DEFAULT 'active',
	sweep_time		TIME,
	sweep_frequency		VARCHAR(20) DEFAULT 'daily',
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE cbm_liquidity_pool_member (
	id		VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	pool_id		VARCHAR(36) NOT NULL REFERENCES cbm_liquidity_pool(id),
	account_id	VARCHAR(36) NOT NULL REFERENCES cbm_bank_account(id),
	tenant_id	VARCHAR(36) NOT NULL,
	target_balance	NUMERIC(20,4) NOT NULL DEFAULT 0,
	sweep_direction	VARCHAR(10) NOT NULL DEFAULT 'both',
	priority	SMALLINT NOT NULL DEFAULT 1,
	created_at	TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	UNIQUE (pool_id, account_id)
);

CREATE TABLE cbm_intercompany_loan (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	lender_entity		VARCHAR(100) NOT NULL,
	borrower_entity		VARCHAR(100) NOT NULL,
	lender_account_id	VARCHAR(36) REFERENCES cbm_bank_account(id),
	borrower_account_id	VARCHAR(36) REFERENCES cbm_bank_account(id),
	principal		NUMERIC(20,4) NOT NULL,
	outstanding_balance	NUMERIC(20,4) NOT NULL,
	currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	interest_rate		NUMERIC(8,6) NOT NULL,
	start_date		DATE NOT NULL,
	maturity_date		DATE NOT NULL,
	day_count		SMALLINT NOT NULL DEFAULT 365,
	payment_frequency	VARCHAR(20) NOT NULL DEFAULT 'monthly',
	status			VARCHAR(20) NOT NULL DEFAULT 'proposed',
	approved_by		VARCHAR(100),
	approved_at		TIMESTAMPTZ,
	accrued_interest	NUMERIC(20,4) NOT NULL DEFAULT 0,
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE cbm_fx_position (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	account_id		VARCHAR(36) REFERENCES cbm_bank_account(id),
	base_currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	quote_currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	long_amount		NUMERIC(20,4) NOT NULL DEFAULT 0,
	short_amount		NUMERIC(20,4) NOT NULL DEFAULT 0,
	net_exposure		NUMERIC(20,4) GENERATED ALWAYS AS (long_amount - short_amount) STORED,
	spot_rate		NUMERIC(14,6),
	book_rate		NUMERIC(14,6),
	unrealised_pnl		NUMERIC(20,4),
	position_date		DATE NOT NULL,
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE cbm_hedge_instrument (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	hedge_type		VARCHAR(30) NOT NULL,
	base_currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	quote_currency		CHAR(3) NOT NULL REFERENCES cbm_currency(code),
	notional		NUMERIC(20,4) NOT NULL,
	contracted_rate		NUMERIC(14,6) NOT NULL,
	current_rate		NUMERIC(14,6),
	trade_date		DATE NOT NULL,
	maturity_date		DATE NOT NULL,
	counterparty		VARCHAR(200) NOT NULL,
	unrealised_pnl		NUMERIC(20,4),
	underlying_exposure_id	VARCHAR(36),
	hedge_ratio		NUMERIC(6,4),
	status			VARCHAR(20) NOT NULL DEFAULT 'pending',
	approved_by		VARCHAR(100),
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL,
	is_deleted		BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE cbm_cash_concentration (
	id			VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
	tenant_id		VARCHAR(36) NOT NULL,
	pool_id			VARCHAR(36) NOT NULL REFERENCES cbm_liquidity_pool(id),
	source_account_id	VARCHAR(36) NOT NULL REFERENCES cbm_bank_account(id),
	target_account_id	VARCHAR(36) NOT NULL REFERENCES cbm_bank_account(id),
	method			VARCHAR(30) NOT NULL DEFAULT 'zero_balance',
	sweep_amount		NUMERIC(20,4) NOT NULL,
	target_balance		NUMERIC(20,4) NOT NULL DEFAULT 0,
	executed_at		TIMESTAMPTZ,
	status			VARCHAR(20) NOT NULL DEFAULT 'pending',
	error_message		TEXT,
	created_at		TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by		VARCHAR(100) NOT NULL
);

-- =============================================================================
-- Audit table
-- =============================================================================

CREATE TABLE cbm_audit_log (
	id		BIGSERIAL PRIMARY KEY,
	tenant_id	VARCHAR(36) NOT NULL,
	entity_type	VARCHAR(50) NOT NULL,
	entity_id	VARCHAR(36) NOT NULL,
	action		VARCHAR(30) NOT NULL,
	actor_id	VARCHAR(100) NOT NULL,
	old_data	JSONB,
	new_data	JSONB,
	ip_address	INET,
	created_at	TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

CREATE TABLE cbm_audit_log_2025
	PARTITION OF cbm_audit_log FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
CREATE TABLE cbm_audit_log_2026
	PARTITION OF cbm_audit_log FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- =============================================================================
-- Indexes
-- =============================================================================

CREATE INDEX idx_bank_account_tenant		ON cbm_bank_account(tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_bank_account_entity		ON cbm_bank_account(tenant_id, entity_id) WHERE NOT is_deleted;
CREATE INDEX idx_bank_account_currency		ON cbm_bank_account(tenant_id, currency) WHERE NOT is_deleted;
CREATE INDEX idx_bank_account_status		ON cbm_bank_account(tenant_id, status) WHERE NOT is_deleted;

CREATE INDEX idx_bank_statement_account		ON cbm_bank_statement(account_id, statement_date DESC) WHERE NOT is_deleted;
CREATE INDEX idx_bank_statement_tenant_date	ON cbm_bank_statement(tenant_id, statement_date DESC) WHERE NOT is_deleted;

CREATE INDEX idx_bank_txn_account_date		ON cbm_bank_transaction(account_id, transaction_date DESC) WHERE NOT is_deleted;
CREATE INDEX idx_bank_txn_value_date		ON cbm_bank_transaction(account_id, value_date DESC) WHERE NOT is_deleted;
CREATE INDEX idx_bank_txn_reference		ON cbm_bank_transaction USING gin(reference gin_trgm_ops) WHERE NOT is_deleted;
CREATE INDEX idx_bank_txn_recon			ON cbm_bank_transaction(reconciliation_id) WHERE reconciliation_id IS NOT NULL;

CREATE INDEX idx_cash_flow_account_date		ON cbm_cash_flow(account_id, expected_date DESC) WHERE NOT is_deleted;
CREATE INDEX idx_cash_flow_tenant_date		ON cbm_cash_flow(tenant_id, expected_date DESC) WHERE NOT is_deleted;
CREATE INDEX idx_cash_flow_category		ON cbm_cash_flow(tenant_id, category) WHERE NOT is_deleted;

CREATE INDEX idx_cash_position_account_date	ON cbm_cash_position(account_id, position_date DESC) WHERE NOT is_deleted;
CREATE INDEX idx_cash_position_tenant_date	ON cbm_cash_position(tenant_id, position_date DESC) WHERE NOT is_deleted;

CREATE INDEX idx_cash_forecast_tenant		ON cbm_cash_forecast(tenant_id, forecast_date DESC) WHERE NOT is_deleted;

CREATE INDEX idx_recon_account_date		ON cbm_bank_reconciliation(account_id, reconciliation_date DESC) WHERE NOT is_deleted;
CREATE INDEX idx_recon_tenant_status		ON cbm_bank_reconciliation(tenant_id, status) WHERE NOT is_deleted;

CREATE INDEX idx_hedge_tenant_maturity		ON cbm_hedge_instrument(tenant_id, maturity_date) WHERE NOT is_deleted;
CREATE INDEX idx_interco_loan_tenant		ON cbm_intercompany_loan(tenant_id, maturity_date) WHERE NOT is_deleted;

CREATE INDEX idx_audit_entity			ON cbm_audit_log(tenant_id, entity_type, entity_id);
CREATE INDEX idx_audit_created			ON cbm_audit_log(created_at);

-- =============================================================================
-- updated_at triggers
-- =============================================================================

CREATE OR REPLACE FUNCTION cbm_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$;

DO $$
DECLARE t text;
BEGIN
	FOREACH t IN ARRAY ARRAY[
		'cbm_bank_account','cbm_bank_statement','cbm_bank_transaction',
		'cbm_cash_flow','cbm_cash_position','cbm_cash_forecast',
		'cbm_bank_reconciliation','cbm_liquidity_pool','cbm_intercompany_loan',
		'cbm_fx_position','cbm_hedge_instrument'
	]
	LOOP
		EXECUTE format(
			'CREATE TRIGGER trg_%s_updated_at BEFORE UPDATE ON %s
			 FOR EACH ROW EXECUTE FUNCTION cbm_set_updated_at()',
			t, t
		);
	END LOOP;
END $$;
