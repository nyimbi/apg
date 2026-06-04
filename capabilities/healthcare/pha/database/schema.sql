-- APG Pharmacy Management — PostgreSQL schema
-- © 2025 Datacraft — All rights reserved
-- Run: psql $DATABASE_URL < database/schema.sql
--
-- Tenant-isolated, HIPAA-aware schema.
-- All tables include: id, tenant_id, created_at, updated_at, created_by, is_deleted.
-- High-volume tables (dispense_orders, narcotics_register) use declarative partitioning.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ── Drug / Formulary ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_drugs (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	drug_name                   TEXT        NOT NULL,
	generic_name                TEXT        NOT NULL,
	ndc_code                    TEXT        NOT NULL,
	rxnorm_code                 TEXT,
	atc_code                    TEXT,
	drug_type                   TEXT        NOT NULL CHECK (drug_type IN ('brand','generic','biosimilar','otc','compounded','investigational','vaccine','blood_product')),
	drug_schedule               TEXT        NOT NULL CHECK (drug_schedule IN ('schedule_i','schedule_ii','schedule_iii','schedule_iv','schedule_v','non_controlled')),
	dosage_form                 TEXT        NOT NULL,
	strength                    TEXT        NOT NULL,
	unit                        TEXT        NOT NULL,
	route_of_administration     TEXT        NOT NULL DEFAULT 'oral',
	manufacturer                TEXT        NOT NULL,
	formulary_status            TEXT        NOT NULL DEFAULT 'preferred' CHECK (formulary_status IN ('preferred','non_preferred','non_formulary','prior_auth_required','step_therapy')),
	requires_refrigeration      BOOLEAN     NOT NULL DEFAULT FALSE,
	is_hazardous                BOOLEAN     NOT NULL DEFAULT FALSE,
	is_lasa                     BOOLEAN     NOT NULL DEFAULT FALSE,
	lasa_pair                   TEXT,
	lasa_alert_type             TEXT        CHECK (lasa_alert_type IN ('look_alike','sound_alike','look_and_sound_alike')),
	tall_man_name               TEXT,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_pha_drugs_ndc_tenant
	ON pha_drugs (tenant_id, ndc_code) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_drugs_tenant
	ON pha_drugs (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_drugs_schedule
	ON pha_drugs (tenant_id, drug_schedule) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_drugs_formulary
	ON pha_drugs (tenant_id, formulary_status) WHERE NOT is_deleted;

-- ── Prescription ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_prescriptions (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	patient_id                  TEXT        NOT NULL,
	prescriber_id               TEXT        NOT NULL,
	prescriber_npi              TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	drug_name                   TEXT        NOT NULL,
	dosage_form                 TEXT        NOT NULL,
	strength                    TEXT        NOT NULL,
	quantity                    NUMERIC     NOT NULL CHECK (quantity > 0),
	unit                        TEXT        NOT NULL,
	days_supply                 INTEGER     NOT NULL CHECK (days_supply > 0),
	sig                         TEXT        NOT NULL,
	refills_authorized          INTEGER     NOT NULL DEFAULT 0,
	refills_remaining           INTEGER     NOT NULL DEFAULT 0,
	diagnosis_icd10             TEXT,
	dea_number                  TEXT,
	is_controlled               BOOLEAN     NOT NULL DEFAULT FALSE,
	status                      TEXT        NOT NULL DEFAULT 'received' CHECK (status IN ('received','verified','in_progress','ready','dispensed','expired','cancelled')),
	formulary_override_reason   TEXT,
	dispensed_at                TIMESTAMPTZ,
	expires_at                  TIMESTAMPTZ,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_pha_rx_tenant_patient
	ON pha_prescriptions (tenant_id, patient_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_rx_status
	ON pha_prescriptions (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_rx_drug
	ON pha_prescriptions (tenant_id, drug_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_rx_expires
	ON pha_prescriptions (tenant_id, expires_at) WHERE NOT is_deleted;

-- ── Dispense Orders (hash-partitioned by tenant_id) ───────────────────────────
CREATE TABLE IF NOT EXISTS pha_dispense_orders (
	id                          TEXT        NOT NULL,
	tenant_id                   TEXT        NOT NULL,
	patient_id                  TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	prescription_id             TEXT        NOT NULL,
	inventory_item_id           TEXT,
	quantity                    NUMERIC     NOT NULL CHECK (quantity > 0),
	unit                        TEXT        NOT NULL,
	status                      TEXT        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','verified','dispensed','picked_up','returned','cancelled')),
	pharmacist_verified         BOOLEAN     NOT NULL DEFAULT FALSE,
	pharmacist_id               TEXT,
	verified_at                 TIMESTAMPTZ,
	dispensed_at                TIMESTAMPTZ,
	picked_up_at                TIMESTAMPTZ,
	counselling_completed       BOOLEAN     NOT NULL DEFAULT FALSE,
	label_printed               BOOLEAN     NOT NULL DEFAULT FALSE,
	barcode_scanned             BOOLEAN     NOT NULL DEFAULT FALSE,
	interaction_severity        TEXT        CHECK (interaction_severity IN ('contraindicated','major','moderate','minor','informational')),
	formulary_status            TEXT        NOT NULL DEFAULT 'preferred',
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id, tenant_id)
) PARTITION BY HASH (tenant_id);

CREATE TABLE IF NOT EXISTS pha_dispense_orders_p0
	PARTITION OF pha_dispense_orders FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE IF NOT EXISTS pha_dispense_orders_p1
	PARTITION OF pha_dispense_orders FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE IF NOT EXISTS pha_dispense_orders_p2
	PARTITION OF pha_dispense_orders FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE IF NOT EXISTS pha_dispense_orders_p3
	PARTITION OF pha_dispense_orders FOR VALUES WITH (MODULUS 4, REMAINDER 3);

CREATE INDEX IF NOT EXISTS idx_pha_do_tenant_patient
	ON pha_dispense_orders (tenant_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_pha_do_status
	ON pha_dispense_orders (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_pha_do_rx
	ON pha_dispense_orders (tenant_id, prescription_id);

-- ── Drug Interactions ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_drug_interactions (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	drug_a_id                   TEXT        NOT NULL,
	drug_b_id                   TEXT        NOT NULL,
	severity                    TEXT        NOT NULL CHECK (severity IN ('contraindicated','major','moderate','minor','informational')),
	mechanism                   TEXT        NOT NULL,
	clinical_effect             TEXT        NOT NULL,
	management                  TEXT        NOT NULL,
	evidence_source             TEXT        NOT NULL,
	onset                       TEXT,
	documentation_level         TEXT,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_pha_interaction_pair
	ON pha_drug_interactions (tenant_id, drug_a_id, drug_b_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_interaction_severity
	ON pha_drug_interactions (tenant_id, severity) WHERE NOT is_deleted;

-- ── Drug Inventory ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_inventory (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	lot_number                  TEXT        NOT NULL,
	quantity_on_hand            NUMERIC     NOT NULL DEFAULT 0 CHECK (quantity_on_hand >= 0),
	reorder_point               NUMERIC     NOT NULL DEFAULT 0,
	reorder_quantity            NUMERIC     NOT NULL DEFAULT 0,
	unit                        TEXT        NOT NULL,
	expiry_date                 TIMESTAMPTZ NOT NULL,
	location                    TEXT        NOT NULL,
	status                      TEXT        NOT NULL DEFAULT 'in_stock' CHECK (status IN ('in_stock','low_stock','out_of_stock','on_order','recalled','expired')),
	days_remaining              INTEGER     NOT NULL DEFAULT 0,
	storage_temperature_min_c   NUMERIC,
	storage_temperature_max_c   NUMERIC,
	supplier_id                 TEXT,
	purchase_price              NUMERIC,
	is_below_reorder_point      BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_pha_inv_tenant_drug
	ON pha_inventory (tenant_id, drug_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_inv_status
	ON pha_inventory (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_inv_expiry
	ON pha_inventory (tenant_id, expiry_date) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_inv_below_reorder
	ON pha_inventory (tenant_id, is_below_reorder_point)
	WHERE NOT is_deleted AND is_below_reorder_point = TRUE;

-- ── Narcotics Register (range-partitioned by created_at) ─────────────────────
-- Legally-mandated append-only ledger; do not UPDATE or DELETE rows.
CREATE TABLE IF NOT EXISTS pha_narcotics_register (
	id                          TEXT        NOT NULL,
	tenant_id                   TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	drug_name                   TEXT        NOT NULL,
	drug_schedule               TEXT        NOT NULL,
	action                      TEXT        NOT NULL CHECK (action IN ('receipt','dispense','waste','destroy','transfer','audit','discrepancy')),
	quantity                    NUMERIC     NOT NULL CHECK (quantity > 0),
	unit                        TEXT        NOT NULL,
	balance_before              NUMERIC     NOT NULL,
	balance_after               NUMERIC     NOT NULL,
	patient_id                  TEXT,
	prescription_id             TEXT,
	dispense_order_id           TEXT,
	performed_by                TEXT        NOT NULL,
	witness_id                  TEXT,
	witness_signature_ref       TEXT,
	notes                       TEXT        NOT NULL DEFAULT '',
	discrepancy_amount          NUMERIC,
	discrepancy_reason          TEXT,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

CREATE TABLE IF NOT EXISTS pha_narcotics_register_2025_q1
	PARTITION OF pha_narcotics_register FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
CREATE TABLE IF NOT EXISTS pha_narcotics_register_2025_q2
	PARTITION OF pha_narcotics_register FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
CREATE TABLE IF NOT EXISTS pha_narcotics_register_2025_q3
	PARTITION OF pha_narcotics_register FOR VALUES FROM ('2025-07-01') TO ('2025-10-01');
CREATE TABLE IF NOT EXISTS pha_narcotics_register_2025_q4
	PARTITION OF pha_narcotics_register FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');
CREATE TABLE IF NOT EXISTS pha_narcotics_register_2026_q1
	PARTITION OF pha_narcotics_register FOR VALUES FROM ('2026-01-01') TO ('2026-04-01');
CREATE TABLE IF NOT EXISTS pha_narcotics_register_2026_q2
	PARTITION OF pha_narcotics_register FOR VALUES FROM ('2026-04-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS pha_narcotics_register_2026_q3
	PARTITION OF pha_narcotics_register FOR VALUES FROM ('2026-07-01') TO ('2026-10-01');
CREATE TABLE IF NOT EXISTS pha_narcotics_register_2026_q4
	PARTITION OF pha_narcotics_register FOR VALUES FROM ('2026-10-01') TO ('2027-01-01');
CREATE TABLE IF NOT EXISTS pha_narcotics_register_default
	PARTITION OF pha_narcotics_register DEFAULT;

CREATE INDEX IF NOT EXISTS idx_pha_narc_tenant_drug
	ON pha_narcotics_register (tenant_id, drug_id);
CREATE INDEX IF NOT EXISTS idx_pha_narc_action
	ON pha_narcotics_register (tenant_id, action);
CREATE INDEX IF NOT EXISTS idx_pha_narc_discrepancy
	ON pha_narcotics_register (tenant_id)
	WHERE discrepancy_amount IS NOT NULL;

-- ── Cold Chain Records ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_cold_chain_records (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	inventory_item_id           TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	recorded_temperature_c      NUMERIC     NOT NULL,
	min_acceptable_c            NUMERIC     NOT NULL,
	max_acceptable_c            NUMERIC     NOT NULL,
	location                    TEXT        NOT NULL,
	sensor_id                   TEXT,
	status                      TEXT        NOT NULL DEFAULT 'compliant' CHECK (status IN ('compliant','excursion','critical','quarantined')),
	excursion_duration_minutes  INTEGER,
	corrective_action           TEXT,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_pha_cc_tenant
	ON pha_cold_chain_records (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_cc_status
	ON pha_cold_chain_records (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_cc_item
	ON pha_cold_chain_records (tenant_id, inventory_item_id) WHERE NOT is_deleted;

-- ── Expiry Tracking (view) ─────────────────────────────────────────────────────
CREATE OR REPLACE VIEW pha_expiry_alerts AS
SELECT
	i.tenant_id,
	i.id                                                         AS inventory_item_id,
	i.drug_id,
	d.drug_name,
	i.lot_number,
	i.quantity_on_hand,
	i.unit,
	i.expiry_date,
	EXTRACT(DAY FROM (i.expiry_date - now()))::INTEGER           AS days_remaining,
	CASE
		WHEN i.expiry_date <= now()                          THEN 'expired'
		WHEN i.expiry_date <= now() + INTERVAL '7 days'     THEN 'critical'
		WHEN i.expiry_date <= now() + INTERVAL '30 days'    THEN 'warning'
		WHEN i.expiry_date <= now() + INTERVAL '90 days'    THEN 'notice'
		ELSE 'ok'
	END                                                          AS alert_level,
	i.location
FROM  pha_inventory i
JOIN  pha_drugs     d ON d.id = i.drug_id
WHERE NOT i.is_deleted
  AND NOT d.is_deleted
  AND i.status NOT IN ('recalled');

-- ── Returned Medications ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_returned_medications (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	patient_id                  TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	dispense_order_id           TEXT        NOT NULL,
	prescription_id             TEXT        NOT NULL,
	quantity_returned           NUMERIC     NOT NULL CHECK (quantity_returned > 0),
	unit                        TEXT        NOT NULL,
	return_reason               TEXT        NOT NULL CHECK (return_reason IN ('adverse_reaction','patient_refused','wrong_medication','expired','dispensing_error')),
	condition                   TEXT        NOT NULL DEFAULT 'intact',
	return_disposition          TEXT        NOT NULL DEFAULT 'destroy',
	returned_by                 TEXT        NOT NULL,
	received_by                 TEXT        NOT NULL,
	notes                       TEXT        NOT NULL DEFAULT '',
	processed                   BOOLEAN     NOT NULL DEFAULT FALSE,
	processed_at                TIMESTAMPTZ,
	processed_by                TEXT,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_pha_ret_tenant_patient
	ON pha_returned_medications (tenant_id, patient_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_ret_reason
	ON pha_returned_medications (tenant_id, return_reason) WHERE NOT is_deleted;

-- ── Prior Authorisations ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_prior_auths (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	patient_id                  TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	prescription_id             TEXT        NOT NULL,
	insurance_id                TEXT        NOT NULL,
	diagnosis_icd10             TEXT        NOT NULL,
	requested_by                TEXT        NOT NULL,
	clinical_justification      TEXT        NOT NULL,
	supporting_documents        JSONB       NOT NULL DEFAULT '[]',
	status                      TEXT        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','denied','expired')),
	decision_by                 TEXT,
	decision_at                 TIMESTAMPTZ,
	denial_reason               TEXT,
	expires_at                  TIMESTAMPTZ,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_pha_pa_tenant_patient
	ON pha_prior_auths (tenant_id, patient_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_pa_status
	ON pha_prior_auths (tenant_id, status) WHERE NOT is_deleted;

-- ── Reorder Requests ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_reorder_requests (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	inventory_item_id           TEXT        NOT NULL,
	quantity_requested          NUMERIC     NOT NULL CHECK (quantity_requested > 0),
	unit                        TEXT        NOT NULL,
	supplier_id                 TEXT,
	urgency                     TEXT        NOT NULL DEFAULT 'routine' CHECK (urgency IN ('routine','urgent','stat')),
	triggered_by                TEXT        NOT NULL DEFAULT 'manual' CHECK (triggered_by IN ('manual','auto_reorder')),
	status                      TEXT        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','submitted','acknowledged','received','cancelled')),
	submitted_at                TIMESTAMPTZ,
	acknowledged_at             TIMESTAMPTZ,
	received_at                 TIMESTAMPTZ,
	quantity_received           NUMERIC,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_pha_reorder_tenant_drug
	ON pha_reorder_requests (tenant_id, drug_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_reorder_status
	ON pha_reorder_requests (tenant_id, status) WHERE NOT is_deleted;

-- ── Counselling Checklists ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_counselling_checklists (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	patient_id                  TEXT        NOT NULL,
	dispense_order_id           TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	indication_explained        BOOLEAN     NOT NULL DEFAULT FALSE,
	dosage_explained            BOOLEAN     NOT NULL DEFAULT FALSE,
	administration_explained    BOOLEAN     NOT NULL DEFAULT FALSE,
	side_effects_explained      BOOLEAN     NOT NULL DEFAULT FALSE,
	interactions_explained      BOOLEAN     NOT NULL DEFAULT FALSE,
	storage_explained           BOOLEAN     NOT NULL DEFAULT FALSE,
	missed_dose_explained       BOOLEAN     NOT NULL DEFAULT FALSE,
	patient_questions_addressed BOOLEAN     NOT NULL DEFAULT FALSE,
	patient_understood          BOOLEAN     NOT NULL DEFAULT FALSE,
	interpreter_used            BOOLEAN     NOT NULL DEFAULT FALSE,
	language                    TEXT        NOT NULL DEFAULT 'en',
	pharmacist_id               TEXT        NOT NULL,
	completion_score            NUMERIC     NOT NULL DEFAULT 0.0,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_pha_counsel_tenant_patient
	ON pha_counselling_checklists (tenant_id, patient_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_counsel_order
	ON pha_counselling_checklists (tenant_id, dispense_order_id) WHERE NOT is_deleted;

-- ── Controlled Substance Logs ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_controlled_substance_logs (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	drug_id                     TEXT        NOT NULL,
	drug_schedule               TEXT        NOT NULL,
	action                      TEXT        NOT NULL CHECK (action IN ('dispense','waste','destroy','count','transfer','receive')),
	quantity                    NUMERIC     NOT NULL CHECK (quantity > 0),
	unit                        TEXT        NOT NULL,
	patient_id                  TEXT,
	performed_by                TEXT        NOT NULL,
	witness_id                  TEXT,
	waste_amount                NUMERIC,
	notes                       TEXT        NOT NULL DEFAULT '',
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL,
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_pha_cs_tenant_drug
	ON pha_controlled_substance_logs (tenant_id, drug_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pha_cs_action
	ON pha_controlled_substance_logs (tenant_id, action) WHERE NOT is_deleted;

-- ── Audit Events ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pha_audit_events (
	id          BIGSERIAL   PRIMARY KEY,
	tenant_id   TEXT        NOT NULL,
	event_type  TEXT        NOT NULL,
	entity_id   TEXT        NOT NULL,
	actor_id    TEXT,
	payload     JSONB       NOT NULL DEFAULT '{}',
	occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pha_audit_tenant
	ON pha_audit_events (tenant_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_pha_audit_entity
	ON pha_audit_events (tenant_id, entity_id, occurred_at DESC);

-- ── updated_at trigger ────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION pha_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
	NEW.updated_at = now();
	RETURN NEW;
END;
$$;
