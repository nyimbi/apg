-- =============================================================================
-- APG Lease Management (realestate_lea) — PostgreSQL Schema
-- IFRS 16 / ASC 842 compliant normalized schema
-- © 2025 Datacraft · Author: Nyimbi Odero
-- =============================================================================
-- Run:   psql $DATABASE_URL -f database/schema.sql
-- Drop:  psql $DATABASE_URL -c "DROP SCHEMA lea CASCADE; CREATE SCHEMA lea;"
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS lea;
SET search_path TO lea, public;

-- ---------------------------------------------------------------------------
-- Extension dependencies
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- ---------------------------------------------------------------------------
-- ENUMERATIONS
-- ---------------------------------------------------------------------------

DO $$ BEGIN
	CREATE TYPE lea_lease_status AS ENUM (
		'draft','heads_of_terms','negotiating','signed','active',
		'holding_over','notice_served','expired','surrendered',
		'forfeited','assigned','terminated','renewed'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_lease_type AS ENUM (
		'commercial','retail','industrial','residential','ground_lease',
		'sublease','licence_to_occupy','peppercorn','assured_shorthold',
		'regulated','office'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_payment_frequency AS ENUM (
		'monthly','quarterly','semi_annual','annual','in_advance','in_arrears'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_ifrs16_category AS ENUM (
		'finance_lease','operating_lease','short_term_exemption','low_value_exemption',
		'finance','operating'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_escalation_type AS ENUM (
		'fixed_percentage','cpi_linked','open_market_review','ratchet',
		'turnover_linked','base_plus_variable','stepped'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_option_type AS ENUM (
		'break_option_tenant','break_option_landlord','renewal_option',
		'purchase_option','expansion_option','contraction_option','extension_option'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_option_status AS ENUM ('open','exercised','lapsed','waived');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_modification_trigger AS ENUM (
		'scope_increase','scope_decrease','term_extension','term_shortening',
		'payment_change','rate_change','combination'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_modification_status AS ENUM ('pending','approved','applied','rejected');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_payment_status AS ENUM (
		'scheduled','paid','overdue','partially_paid','waived','disputed'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_sublease_status AS ENUM ('active','expired','terminated','suspended');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_abstraction_status AS ENUM (
		'pending','in_progress','complete','verified','exception'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_rent_review_type AS ENUM (
		'upward_only','upward_downward','fixed','open_market','indexed'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_rent_review_status AS ENUM (
		'pending','in_negotiation','agreed','disputed','withdrawn'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_assignment_status AS ENUM (
		'pending','completed','rejected','withdrawn'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_expiry_action AS ENUM (
		'renew','surrender','holdover','negotiate','vacate'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_amortisation_method AS ENUM (
		'straight_line','declining_balance','units_of_production'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_accounting_standard AS ENUM ('ifrs16','asc842');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE lea_sublease_classification AS ENUM ('operating','finance');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------------
-- 1. lea_lease  — master lease record
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	property_id                 TEXT            NOT NULL,
	unit_id                     TEXT            NOT NULL,
	tenant_entity_id            TEXT            NOT NULL,
	lease_ref                   TEXT            NOT NULL,
	lease_type                  lea_lease_type  NOT NULL,
	status                      lea_lease_status NOT NULL DEFAULT 'draft',
	accounting_standard         lea_accounting_standard NOT NULL DEFAULT 'ifrs16',

	-- Dates
	commencement_date           DATE            NOT NULL,
	expiry_date                 DATE            NOT NULL,
	executed_at                 DATE,
	executed_by                 TEXT,
	successor_lease_id          TEXT,
	predecessor_lease_id        TEXT,

	-- Rent
	initial_rent                NUMERIC(18,4)   NOT NULL,
	current_rent                NUMERIC(18,4)   NOT NULL,
	rent_frequency              lea_payment_frequency NOT NULL DEFAULT 'monthly',
	currency                    CHAR(3)         NOT NULL DEFAULT 'KES',

	-- Area
	area                        NUMERIC(12,4),
	area_unit                   TEXT            NOT NULL DEFAULT 'sqm',

	-- Financials
	security_deposit            NUMERIC(18,4)   NOT NULL DEFAULT 0,
	total_payments_made         NUMERIC(18,4)   NOT NULL DEFAULT 0,

	-- IFRS 16 core
	ifrs16_category             lea_ifrs16_category,
	incremental_borrowing_rate  NUMERIC(8,6),       -- decimal fraction e.g. 0.085
	implicit_rate               NUMERIC(8,6),
	initial_direct_costs        NUMERIC(18,4)   NOT NULL DEFAULT 0,
	lease_incentives            NUMERIC(18,4)   NOT NULL DEFAULT 0,
	dismantling_costs           NUMERIC(18,4)   NOT NULL DEFAULT 0,
	residual_value_guarantee    NUMERIC(18,4)   NOT NULL DEFAULT 0,

	-- Variable/CPI
	variable_payment_indexed_to_cpi BOOLEAN     NOT NULL DEFAULT FALSE,
	cpi_base_index              NUMERIC(10,4),

	-- Sublease linkage
	is_sublease                 BOOLEAN         NOT NULL DEFAULT FALSE,
	parent_lease_id             TEXT,

	-- Abstraction
	abstraction_status          lea_abstraction_status NOT NULL DEFAULT 'pending',
	abstraction_verified        BOOLEAN         NOT NULL DEFAULT FALSE,

	-- Computed/cached balances (refreshed on each IFRS16 calculation)
	rou_asset                   NUMERIC(18,4),
	lease_liability             NUMERIC(18,4),

	-- Misc
	notes                       TEXT,
	options                     JSONB           NOT NULL DEFAULT '{}',

	-- Audit
	created_by                  TEXT            NOT NULL,
	updated_by                  TEXT,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_lease PRIMARY KEY (id),
	CONSTRAINT uq_lea_lease_ref UNIQUE (tenant_id, lease_ref),
	CONSTRAINT ck_lea_lease_dates CHECK (expiry_date > commencement_date),
	CONSTRAINT ck_lea_lease_rent CHECK (initial_rent >= 0),
	CONSTRAINT ck_lea_lease_deposit CHECK (security_deposit >= 0)
);

COMMENT ON TABLE lea_lease IS 'Master lease registry — IFRS 16 / ASC 842 compliant.';

CREATE INDEX idx_lea_lease_tenant        ON lea_lease (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_lease_status        ON lea_lease (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX idx_lea_lease_property      ON lea_lease (tenant_id, property_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_lease_expiry        ON lea_lease (tenant_id, expiry_date) WHERE NOT is_deleted AND status = 'active';
CREATE INDEX idx_lea_lease_type          ON lea_lease (tenant_id, lease_type) WHERE NOT is_deleted;
CREATE INDEX idx_lea_lease_ifrs16        ON lea_lease (tenant_id, ifrs16_category) WHERE NOT is_deleted;
CREATE INDEX idx_lea_lease_options_gin   ON lea_lease USING gin (options);

-- ---------------------------------------------------------------------------
-- 2. lea_lease_asset  — underlying physical / identified asset
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease_asset (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	asset_description           TEXT            NOT NULL,
	asset_class                 TEXT            NOT NULL,   -- building, vehicle, equipment
	asset_ref                   TEXT,
	fair_value_when_new         NUMERIC(18,4),
	useful_economic_life_months INTEGER,
	location                    TEXT,
	currency                    CHAR(3)         NOT NULL DEFAULT 'KES',
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_lease_asset PRIMARY KEY (id),
	CONSTRAINT fk_lea_lease_asset_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_lease_asset_lease ON lea_lease_asset (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_lease_asset_tenant ON lea_lease_asset (tenant_id) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 3. lea_rou_asset  — Right-of-Use asset
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_rou_asset (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	initial_measurement         NUMERIC(18,4)   NOT NULL,
	useful_life_months          INTEGER         NOT NULL,
	amortisation_method         lea_amortisation_method NOT NULL DEFAULT 'straight_line',
	accumulated_depreciation    NUMERIC(18,4)   NOT NULL DEFAULT 0,
	impairment_loss             NUMERIC(18,4)   NOT NULL DEFAULT 0,
	carrying_amount             NUMERIC(18,4)   NOT NULL DEFAULT 0,
	periods_amortised           INTEGER         NOT NULL DEFAULT 0,
	fully_amortised             BOOLEAN         NOT NULL DEFAULT FALSE,
	currency                    CHAR(3)         NOT NULL DEFAULT 'KES',
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_rou_asset PRIMARY KEY (id),
	CONSTRAINT fk_lea_rou_asset_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT,
	CONSTRAINT ck_lea_rou_initial_positive CHECK (initial_measurement > 0)
);

CREATE INDEX idx_lea_rou_asset_lease  ON lea_rou_asset (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_rou_asset_tenant ON lea_rou_asset (tenant_id) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 4. lea_lease_liability  — lease liability carrying amount
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease_liability (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	opening_balance             NUMERIC(18,4)   NOT NULL,
	current_balance             NUMERIC(18,4)   NOT NULL DEFAULT 0,
	interest_rate               NUMERIC(8,6)    NOT NULL,  -- decimal fraction
	cumulative_interest         NUMERIC(18,4)   NOT NULL DEFAULT 0,
	cumulative_principal        NUMERIC(18,4)   NOT NULL DEFAULT 0,
	current_portion             NUMERIC(18,4)   NOT NULL DEFAULT 0,
	non_current_portion         NUMERIC(18,4)   NOT NULL DEFAULT 0,
	currency                    CHAR(3)         NOT NULL DEFAULT 'KES',
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_lease_liability PRIMARY KEY (id),
	CONSTRAINT fk_lea_lease_liability_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT,
	CONSTRAINT ck_lea_liability_opening CHECK (opening_balance > 0)
);

CREATE INDEX idx_lea_liability_lease  ON lea_lease_liability (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_liability_tenant ON lea_lease_liability (tenant_id) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 5. lea_payment_schedule  — full amortisation/payment schedule
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_payment_schedule (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	period_number               INTEGER         NOT NULL,
	due_date                    DATE            NOT NULL,
	opening_balance             NUMERIC(18,4)   NOT NULL,
	payment_amount              NUMERIC(18,4)   NOT NULL,
	interest_portion            NUMERIC(18,4)   NOT NULL,
	principal_portion           NUMERIC(18,4)   NOT NULL,
	closing_balance             NUMERIC(18,4)   NOT NULL,
	cumulative_interest         NUMERIC(18,4)   NOT NULL,
	is_variable                 BOOLEAN         NOT NULL DEFAULT FALSE,
	variable_index              TEXT,
	escalation_applied          NUMERIC(8,4)    NOT NULL DEFAULT 0,
	currency                    CHAR(3)         NOT NULL DEFAULT 'KES',
	-- Payment tracking
	paid                        BOOLEAN         NOT NULL DEFAULT FALSE,
	paid_date                   DATE,
	paid_amount                 NUMERIC(18,4),
	variance                    NUMERIC(18,4)   NOT NULL DEFAULT 0,
	payment_status              lea_payment_status NOT NULL DEFAULT 'scheduled',
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_payment_schedule PRIMARY KEY (id),
	CONSTRAINT fk_lea_sched_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT,
	CONSTRAINT uq_lea_sched_period UNIQUE (lease_id, period_number)
);

-- Partition by tenant for very large schedules
CREATE INDEX idx_lea_sched_lease     ON lea_payment_schedule (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_sched_tenant    ON lea_payment_schedule (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_sched_due       ON lea_payment_schedule (due_date) WHERE NOT is_deleted AND NOT paid;
CREATE INDEX idx_lea_sched_overdue   ON lea_payment_schedule (tenant_id, due_date)
	WHERE NOT is_deleted AND payment_status = 'overdue';

-- ---------------------------------------------------------------------------
-- 6. lea_escalation_clause  — persistent escalation clause
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_escalation_clause (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	escalation_type             lea_escalation_type NOT NULL,
	fixed_rate                  NUMERIC(8,6),
	cpi_base_index              NUMERIC(10,4),
	review_frequency_months     INTEGER         NOT NULL DEFAULT 12,
	first_review_date           DATE,
	cap_rate                    NUMERIC(8,6),
	floor_rate                  NUMERIC(8,6),
	last_applied_date           DATE,
	last_applied_index          NUMERIC(10,4),
	applied_count               INTEGER         NOT NULL DEFAULT 0,
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_escalation_clause PRIMARY KEY (id),
	CONSTRAINT fk_lea_esc_clause_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_esc_clause_lease  ON lea_escalation_clause (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_esc_clause_tenant ON lea_escalation_clause (tenant_id) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 7. lea_rent_escalation  — escalation event log
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_rent_escalation (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	escalation_type             lea_escalation_type NOT NULL,
	effective_date              DATE            NOT NULL,
	old_rent                    NUMERIC(18,4)   NOT NULL DEFAULT 0,
	escalation_rate             NUMERIC(8,6),
	new_rent                    NUMERIC(18,4),
	computed_new_rent           NUMERIC(18,4),
	index_source                TEXT,
	cpi_current_index           NUMERIC(10,4),
	cpi_base_index              NUMERIC(10,4),
	applied                     BOOLEAN         NOT NULL DEFAULT FALSE,
	applied_at                  TIMESTAMPTZ,
	applied_by                  TEXT,
	remeasurement_required      BOOLEAN         NOT NULL DEFAULT FALSE,
	notes                       TEXT,
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_rent_escalation PRIMARY KEY (id),
	CONSTRAINT fk_lea_rent_esc_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_rent_esc_lease   ON lea_rent_escalation (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_rent_esc_tenant  ON lea_rent_escalation (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_rent_esc_pending ON lea_rent_escalation (tenant_id, effective_date)
	WHERE NOT is_deleted AND NOT applied;

-- ---------------------------------------------------------------------------
-- 8. lea_lease_option  — option tracker (renewal / break / purchase)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease_option (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	option_type                 lea_option_type NOT NULL,
	status                      lea_option_status NOT NULL DEFAULT 'open',
	exercise_from               DATE            NOT NULL,
	exercise_to                 DATE            NOT NULL,
	effective_date              DATE            NOT NULL,
	notice_required_days        INTEGER         NOT NULL DEFAULT 0,
	new_expiry                  DATE,
	extension_months            INTEGER,
	purchase_price              NUMERIC(18,4),
	reasonably_certain          BOOLEAN         NOT NULL DEFAULT FALSE,
	economic_incentive          BOOLEAN         NOT NULL DEFAULT FALSE,
	exercised_at                TIMESTAMPTZ,
	notice_served_at            TIMESTAMPTZ,
	last_assessed_date          DATE,
	assessment_changed          BOOLEAN         NOT NULL DEFAULT FALSE,
	notes                       TEXT,
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_option PRIMARY KEY (id),
	CONSTRAINT fk_lea_option_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT,
	CONSTRAINT ck_lea_option_window CHECK (exercise_to >= exercise_from)
);

CREATE INDEX idx_lea_option_lease   ON lea_lease_option (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_option_tenant  ON lea_lease_option (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_option_expiry  ON lea_lease_option (tenant_id, exercise_to)
	WHERE NOT is_deleted AND status = 'open';

-- ---------------------------------------------------------------------------
-- 9. lea_lease_modification  — modification and remeasurement events
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease_modification (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	modification_date           DATE            NOT NULL,
	trigger                     lea_modification_trigger NOT NULL,
	status                      lea_modification_status  NOT NULL DEFAULT 'pending',
	reason                      TEXT            NOT NULL,
	new_lease_term_months       INTEGER,
	new_base_payment            NUMERIC(18,4),
	new_rate                    NUMERIC(8,6),
	surrendered_proportion      NUMERIC(5,4),   -- 0.0000–0.9999
	new_commencement_date       DATE,
	creates_new_lease           BOOLEAN         NOT NULL DEFAULT FALSE,
	remeasured_liability        NUMERIC(18,4),
	remeasured_rou              NUMERIC(18,4),
	gain_loss_on_modification   NUMERIC(18,4)   NOT NULL DEFAULT 0,
	applied                     BOOLEAN         NOT NULL DEFAULT FALSE,
	applied_at                  TIMESTAMPTZ,
	approved_by                 TEXT,
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_modification PRIMARY KEY (id),
	CONSTRAINT fk_lea_mod_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT,
	CONSTRAINT ck_lea_mod_surrender CHECK (
		surrendered_proportion IS NULL OR (surrendered_proportion > 0 AND surrendered_proportion < 1)
	)
);

CREATE INDEX idx_lea_mod_lease   ON lea_lease_modification (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_mod_tenant  ON lea_lease_modification (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_mod_pending ON lea_lease_modification (tenant_id)
	WHERE NOT is_deleted AND status = 'pending';

-- ---------------------------------------------------------------------------
-- 10. lea_lease_amendment  — documentary amendments (redlines)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease_amendment (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	amendment_date              DATE            NOT NULL,
	description                 TEXT            NOT NULL,
	amended_clauses             TEXT[]          NOT NULL DEFAULT '{}',
	document_ids                TEXT[]          NOT NULL DEFAULT '{}',
	approved_by                 TEXT,
	approved_at                 TIMESTAMPTZ,
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_amendment PRIMARY KEY (id),
	CONSTRAINT fk_lea_amendment_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_amendment_lease  ON lea_lease_amendment (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_amendment_tenant ON lea_lease_amendment (tenant_id) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 11. lea_sublease  — sublease relationships
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_sublease (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	head_lease_id               TEXT            NOT NULL,
	sublessee_entity_id         TEXT            NOT NULL,
	commencement_date           DATE            NOT NULL,
	end_date                    DATE            NOT NULL,
	payment_amount              NUMERIC(18,4)   NOT NULL,
	payment_frequency           lea_payment_frequency NOT NULL DEFAULT 'monthly',
	sublease_classification     lea_sublease_classification NOT NULL DEFAULT 'operating',
	portion_sqm                 NUMERIC(12,4),
	implicit_rate               NUMERIC(8,6),
	status                      lea_sublease_status NOT NULL DEFAULT 'active',
	total_sublease_income       NUMERIC(18,4)   NOT NULL DEFAULT 0,
	net_investment_in_sublease  NUMERIC(18,4),
	currency                    CHAR(3)         NOT NULL DEFAULT 'KES',
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_sublease PRIMARY KEY (id),
	CONSTRAINT fk_lea_sublease_head FOREIGN KEY (head_lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT,
	CONSTRAINT ck_lea_sublease_dates CHECK (end_date > commencement_date),
	CONSTRAINT ck_lea_sublease_payment CHECK (payment_amount > 0)
);

CREATE INDEX idx_lea_sublease_head   ON lea_sublease (head_lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_sublease_tenant ON lea_sublease (tenant_id) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 12. lea_lease_expiry  — expiry pipeline tracker
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease_expiry (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	expiry_date                 DATE            NOT NULL,
	action_required             lea_expiry_action NOT NULL,
	assigned_to                 TEXT,
	days_ahead_flag             INTEGER         NOT NULL DEFAULT 180,
	days_to_expiry              INTEGER         NOT NULL DEFAULT 0,
	action_taken                TEXT,
	resolved                    BOOLEAN         NOT NULL DEFAULT FALSE,
	resolved_at                 TIMESTAMPTZ,
	notes                       TEXT,
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_lease_expiry PRIMARY KEY (id),
	CONSTRAINT fk_lea_expiry_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_expiry_lease   ON lea_lease_expiry (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_expiry_tenant  ON lea_lease_expiry (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_expiry_date    ON lea_lease_expiry (tenant_id, expiry_date)
	WHERE NOT is_deleted AND NOT resolved;

-- ---------------------------------------------------------------------------
-- 13. lea_rent_review  — formal rent review workflow
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_rent_review (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	review_type                 lea_rent_review_type NOT NULL,
	review_date                 DATE            NOT NULL,
	status                      lea_rent_review_status NOT NULL DEFAULT 'pending',
	proposed_rent               NUMERIC(18,4),
	agreed_rent                 NUMERIC(18,4),
	agreed_at                   TIMESTAMPTZ,
	backdating_authorised_by    TEXT,
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_rent_review PRIMARY KEY (id),
	CONSTRAINT fk_lea_rent_review_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_rent_review_lease  ON lea_rent_review (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_rent_review_tenant ON lea_rent_review (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_rent_review_due    ON lea_rent_review (tenant_id, review_date)
	WHERE NOT is_deleted AND status = 'pending';

-- ---------------------------------------------------------------------------
-- 14. lea_lease_abstraction  — AI-assisted lease data extraction
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease_abstraction (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	source_document_id          TEXT            NOT NULL,
	abstracted_by               TEXT            NOT NULL,
	status                      lea_abstraction_status NOT NULL DEFAULT 'pending',
	extracted_fields            JSONB           NOT NULL DEFAULT '{}',
	exceptions                  TEXT[]          NOT NULL DEFAULT '{}',
	verified_by                 TEXT,
	verified_at                 TIMESTAMPTZ,
	notes                       TEXT,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_abstraction PRIMARY KEY (id),
	CONSTRAINT fk_lea_abstraction_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_abstraction_lease  ON lea_lease_abstraction (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_abstraction_tenant ON lea_lease_abstraction (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_abstraction_fields ON lea_lease_abstraction USING gin (extracted_fields);

-- ---------------------------------------------------------------------------
-- 15. lea_ifrs16_schedule  — generated IFRS16 amortisation schedule (header)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_ifrs16_schedule (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	category                    lea_ifrs16_category NOT NULL,
	commencement_date           DATE            NOT NULL,
	expiry_date                 DATE            NOT NULL,
	annual_payment              NUMERIC(18,4)   NOT NULL,
	discount_rate               NUMERIC(8,6)    NOT NULL,   -- decimal fraction
	rou_asset                   NUMERIC(18,4)   NOT NULL DEFAULT 0,
	lease_liability             NUMERIC(18,4)   NOT NULL DEFAULT 0,
	auditor_approved            BOOLEAN         NOT NULL DEFAULT FALSE,
	auditor_approved_by         TEXT,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_ifrs16_schedule PRIMARY KEY (id),
	CONSTRAINT fk_lea_ifrs16_sched_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_ifrs16_sched_lease  ON lea_ifrs16_schedule (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_ifrs16_sched_tenant ON lea_ifrs16_schedule (tenant_id) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 16. lea_lease_assignment  — lease assignment records
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease_assignment (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	assignment_type             TEXT            NOT NULL,
	assignee_id                 TEXT            NOT NULL,
	effective_date              DATE            NOT NULL,
	landlord_consent_ref        TEXT,
	status                      lea_assignment_status NOT NULL DEFAULT 'pending',
	completed_at                TIMESTAMPTZ,
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_assignment PRIMARY KEY (id),
	CONSTRAINT fk_lea_assignment_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_assignment_lease  ON lea_lease_assignment (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_assignment_tenant ON lea_lease_assignment (tenant_id) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 17. lea_lease_payment  — actual payment receipts
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_lease_payment (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	schedule_id                 TEXT,
	payment_date                DATE            NOT NULL,
	amount_paid                 NUMERIC(18,4)   NOT NULL,
	payment_reference           TEXT,
	variance                    NUMERIC(18,4)   NOT NULL DEFAULT 0,
	overpayment                 BOOLEAN         NOT NULL DEFAULT FALSE,
	underpayment                BOOLEAN         NOT NULL DEFAULT FALSE,
	notes                       TEXT,
	created_by                  TEXT            NOT NULL,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_payment PRIMARY KEY (id),
	CONSTRAINT fk_lea_payment_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT,
	CONSTRAINT fk_lea_payment_schedule FOREIGN KEY (schedule_id)
		REFERENCES lea_payment_schedule (id) ON DELETE SET NULL,
	CONSTRAINT ck_lea_payment_positive CHECK (amount_paid > 0)
);

CREATE INDEX idx_lea_payment_lease   ON lea_lease_payment (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_payment_tenant  ON lea_lease_payment (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_payment_date    ON lea_lease_payment (tenant_id, payment_date) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 18. lea_journal_entry  — accounting journals generated by the system
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_journal_entry (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	period                      CHAR(7)         NOT NULL,  -- YYYY-MM
	narrative                   TEXT            NOT NULL,
	entries                     JSONB           NOT NULL DEFAULT '[]',
	posted                      BOOLEAN         NOT NULL DEFAULT FALSE,
	posted_at                   TIMESTAMPTZ,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_journal PRIMARY KEY (id),
	CONSTRAINT fk_lea_journal_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_journal_lease   ON lea_journal_entry (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_journal_tenant  ON lea_journal_entry (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_journal_period  ON lea_journal_entry (tenant_id, period) WHERE NOT is_deleted;

-- ---------------------------------------------------------------------------
-- 19. lea_rent_demand  — formal rent demands / invoices
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS lea_rent_demand (
	id                          TEXT            NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id                   TEXT            NOT NULL,
	lease_id                    TEXT            NOT NULL,
	period                      CHAR(7)         NOT NULL,  -- YYYY-MM or YYYY-QN
	amount_due                  NUMERIC(18,4)   NOT NULL,
	arrears_brought_forward     NUMERIC(18,4)   NOT NULL DEFAULT 0,
	total_due                   NUMERIC(18,4)   NOT NULL,
	due_date                    DATE            NOT NULL,
	currency                    CHAR(3)         NOT NULL DEFAULT 'KES',
	paid                        BOOLEAN         NOT NULL DEFAULT FALSE,
	paid_amount                 NUMERIC(18,4),
	paid_date                   DATE,
	created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN         NOT NULL DEFAULT FALSE,

	CONSTRAINT pk_lea_rent_demand PRIMARY KEY (id),
	CONSTRAINT fk_lea_demand_lease FOREIGN KEY (lease_id)
		REFERENCES lea_lease (id) ON DELETE RESTRICT
);

CREATE INDEX idx_lea_demand_lease   ON lea_rent_demand (lease_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_demand_tenant  ON lea_rent_demand (tenant_id) WHERE NOT is_deleted;
CREATE INDEX idx_lea_demand_unpaid  ON lea_rent_demand (tenant_id, due_date)
	WHERE NOT is_deleted AND NOT paid;

-- ---------------------------------------------------------------------------
-- VIEWS — commonly needed portfolio queries
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW lea_v_active_leases AS
SELECT
	l.id,
	l.tenant_id,
	l.property_id,
	l.unit_id,
	l.tenant_entity_id,
	l.lease_ref,
	l.lease_type,
	l.status,
	l.commencement_date,
	l.expiry_date,
	(l.expiry_date - CURRENT_DATE)                                AS days_to_expiry,
	l.current_rent,
	l.rent_frequency,
	l.currency,
	l.ifrs16_category,
	l.rou_asset,
	l.lease_liability,
	l.incremental_borrowing_rate,
	l.variable_payment_indexed_to_cpi,
	l.is_sublease,
	l.parent_lease_id
FROM lea_lease l
WHERE l.status = 'active'
  AND NOT l.is_deleted;

COMMENT ON VIEW lea_v_active_leases IS 'All active leases with days-to-expiry computed.';


CREATE OR REPLACE VIEW lea_v_expiry_pipeline AS
SELECT
	l.id                                                          AS lease_id,
	l.tenant_id,
	l.property_id,
	l.lease_ref,
	l.tenant_entity_id,
	l.expiry_date,
	(l.expiry_date - CURRENT_DATE)                                AS days_remaining,
	l.current_rent,
	l.currency,
	l.status,
	CASE
		WHEN (l.expiry_date - CURRENT_DATE) <= 30  THEN 'critical'
		WHEN (l.expiry_date - CURRENT_DATE) <= 90  THEN 'high'
		WHEN (l.expiry_date - CURRENT_DATE) <= 180 THEN 'medium'
		ELSE 'low'
	END                                                           AS urgency
FROM lea_lease l
WHERE l.status IN ('active', 'holding_over')
  AND l.expiry_date <= CURRENT_DATE + INTERVAL '180 days'
  AND NOT l.is_deleted
ORDER BY days_remaining;

COMMENT ON VIEW lea_v_expiry_pipeline IS 'Leases expiring within 180 days with urgency classification.';


CREATE OR REPLACE VIEW lea_v_ifrs16_portfolio AS
SELECT
	l.tenant_id,
	l.ifrs16_category,
	COUNT(*)                                                      AS lease_count,
	SUM(l.rou_asset)                                              AS total_rou_assets,
	SUM(l.lease_liability)                                        AS total_lease_liabilities,
	SUM(l.current_rent * 12)                                      AS total_annual_rent,
	AVG(l.incremental_borrowing_rate)                             AS avg_ibr
FROM lea_lease l
WHERE l.status = 'active'
  AND l.ifrs16_category IS NOT NULL
  AND NOT l.is_deleted
GROUP BY l.tenant_id, l.ifrs16_category;

COMMENT ON VIEW lea_v_ifrs16_portfolio IS 'IFRS 16 balance-sheet summary by category.';


CREATE OR REPLACE VIEW lea_v_overdue_schedule AS
SELECT
	ps.id,
	ps.tenant_id,
	ps.lease_id,
	ps.period_number,
	ps.due_date,
	ps.payment_amount,
	(CURRENT_DATE - ps.due_date)                                  AS days_overdue,
	CASE
		WHEN (CURRENT_DATE - ps.due_date) <= 30  THEN '0-30'
		WHEN (CURRENT_DATE - ps.due_date) <= 60  THEN '31-60'
		WHEN (CURRENT_DATE - ps.due_date) <= 90  THEN '61-90'
		ELSE '90+'
	END                                                           AS aging_bucket
FROM lea_payment_schedule ps
WHERE ps.payment_status = 'overdue'
  AND NOT ps.paid
  AND NOT ps.is_deleted
ORDER BY ps.due_date;

COMMENT ON VIEW lea_v_overdue_schedule IS 'Aged overdue payment schedule lines.';


-- ---------------------------------------------------------------------------
-- TRIGGERS — updated_at maintenance
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION lea_set_updated_at()
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
		'lea_lease','lea_lease_asset','lea_rou_asset','lea_lease_liability',
		'lea_payment_schedule','lea_escalation_clause','lea_rent_escalation',
		'lea_lease_option','lea_lease_modification','lea_lease_amendment',
		'lea_sublease','lea_lease_expiry','lea_rent_review','lea_lease_abstraction',
		'lea_ifrs16_schedule','lea_lease_assignment','lea_lease_payment',
		'lea_rent_demand'
	]
	LOOP
		EXECUTE format(
			'CREATE OR REPLACE TRIGGER trg_%s_updated_at
			 BEFORE UPDATE ON %s
			 FOR EACH ROW EXECUTE FUNCTION lea_set_updated_at();',
			t, t
		);
	END LOOP;
END $$;
