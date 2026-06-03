-- ============================================================
-- APG Fleet Management (transport_fle) — PostgreSQL Schema
-- ============================================================
-- Run: psql $DATABASE_URL -f database/schema.sql
-- Requires: PostgreSQL 14+
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- ────────────────────────────────────────────────────────────
-- Enumerations
-- ────────────────────────────────────────────────────────────

DO $$ BEGIN
	CREATE TYPE fle_vehicle_type AS ENUM (
		'rigid_truck','articulated_truck','van','pickup','tractor_unit',
		'trailer','tanker','refrigerated_vehicle','flatbed','tipper',
		'minibus','motorcycle','electric_vehicle','bus','crane_truck'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_vehicle_status AS ENUM (
		'active','inactive','in_maintenance','out_of_service','disposed',
		'on_hire','awaiting_inspection','breakdown','impounded'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_fuel_type AS ENUM (
		'diesel','petrol','cng','lng','electric','hybrid','hydrogen','biodiesel','hvo'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_ownership_type AS ENUM (
		'owned','leased','hired','contract_hire','finance_lease','hire_purchase'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_driver_status AS ENUM (
		'active','inactive','on_leave','suspended','training','probation','terminated'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_licence_class AS ENUM (
		'am','a1','a2','a','b','be','c1','c1e','c','ce','d1','d1e','d','de'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_trip_status AS ENUM (
		'planned','dispatched','in_progress','completed','cancelled','breakdown','delayed'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_maintenance_type AS ENUM (
		'scheduled','corrective','predictive','emergency'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_maintenance_status AS ENUM (
		'scheduled','in_progress','completed','overdue','cancelled','deferred'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_inspection_type AS ENUM (
		'pre_trip','post_trip','periodic','cof','roadside','annual'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_inspection_result AS ENUM (
		'pass','fail','advisory','conditional_pass'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_incident_severity AS ENUM (
		'minor','moderate','major','critical','fatal'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_incident_status AS ENUM (
		'reported','under_investigation','resolved','closed','disputed'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
	CREATE TYPE fle_tacho_mode AS ENUM (
		'driving','rest','other_work','availability','unknown'
	);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ────────────────────────────────────────────────────────────
-- Vehicles
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_vehicles (
	id                       TEXT               NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id                TEXT               NOT NULL,
	vehicle_type             fle_vehicle_type   NOT NULL,
	registration             TEXT               NOT NULL,
	vin                      TEXT               NOT NULL,
	make                     TEXT               NOT NULL DEFAULT '',
	model                    TEXT               NOT NULL DEFAULT '',
	year                     SMALLINT           NOT NULL,
	fuel_type                fle_fuel_type      NOT NULL DEFAULT 'diesel',
	ownership_type           fle_ownership_type NOT NULL DEFAULT 'owned',
	status                   fle_vehicle_status NOT NULL DEFAULT 'active',
	gross_vehicle_weight_kg  NUMERIC(10,2)      NOT NULL DEFAULT 0,
	payload_capacity_kg      NUMERIC(10,2)      NOT NULL DEFAULT 0,
	axle_count               SMALLINT           NOT NULL DEFAULT 2,
	odometer_km              NUMERIC(12,2)      NOT NULL DEFAULT 0,
	colour                   TEXT               NOT NULL DEFAULT '',
	depot_id                 TEXT,
	notes                    TEXT               NOT NULL DEFAULT '',
	created_at               TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
	updated_at               TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
	created_by               TEXT               NOT NULL DEFAULT 'system',
	is_deleted               BOOLEAN            NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_vehicles_pkey PRIMARY KEY (id),
	CONSTRAINT fle_vehicles_vin_tenant_uq UNIQUE (tenant_id, vin)
);

CREATE INDEX IF NOT EXISTS idx_fle_vehicles_tenant     ON fle_vehicles (tenant_id)           WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_vehicles_status     ON fle_vehicles (tenant_id, status)   WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_vehicles_reg        ON fle_vehicles (tenant_id, registration) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_vehicles_depot      ON fle_vehicles (tenant_id, depot_id) WHERE depot_id IS NOT NULL;

-- ────────────────────────────────────────────────────────────
-- Drivers
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_drivers (
	id                  TEXT               NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT               NOT NULL,
	name                TEXT               NOT NULL,
	employee_number     TEXT               NOT NULL DEFAULT '',
	licence_number      TEXT               NOT NULL,
	licence_class       fle_licence_class  NOT NULL,
	licence_expiry      TIMESTAMPTZ        NOT NULL,
	status              fle_driver_status  NOT NULL DEFAULT 'active',
	tacho_card_number   TEXT               NOT NULL DEFAULT '',
	cpc_expiry          TIMESTAMPTZ,
	medical_expiry      TIMESTAMPTZ,
	phone               TEXT               NOT NULL DEFAULT '',
	email               TEXT               NOT NULL DEFAULT '',
	depot_id            TEXT,
	notes               TEXT               NOT NULL DEFAULT '',
	created_at          TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
	created_by          TEXT               NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN            NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_drivers_pkey PRIMARY KEY (id),
	CONSTRAINT fle_drivers_licence_tenant_uq UNIQUE (tenant_id, licence_number)
);

CREATE INDEX IF NOT EXISTS idx_fle_drivers_tenant         ON fle_drivers (tenant_id)                   WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_drivers_status         ON fle_drivers (tenant_id, status)           WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_drivers_licence_expiry ON fle_drivers (tenant_id, licence_expiry)   WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_drivers_cpc            ON fle_drivers (tenant_id, cpc_expiry)       WHERE cpc_expiry IS NOT NULL;

-- ────────────────────────────────────────────────────────────
-- Vehicle Assignments
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_vehicle_assignments (
	id                  TEXT        NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT        NOT NULL,
	vehicle_id          TEXT        NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	driver_id           TEXT        NOT NULL REFERENCES fle_drivers(id)  ON DELETE RESTRICT,
	assigned_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	released_at         TIMESTAMPTZ,
	assignment_reason   TEXT        NOT NULL DEFAULT '',
	trip_id             TEXT,
	is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_assignments_pkey PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_fle_assign_vehicle ON fle_vehicle_assignments (tenant_id, vehicle_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_assign_driver  ON fle_vehicle_assignments (tenant_id, driver_id)  WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_assign_active  ON fle_vehicle_assignments (tenant_id, is_active)  WHERE is_active AND NOT is_deleted;

-- ────────────────────────────────────────────────────────────
-- Trips (partitioned by planned_departure)
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_trips (
	id                      TEXT             NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id               TEXT             NOT NULL,
	vehicle_id              TEXT             NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	driver_id               TEXT             NOT NULL REFERENCES fle_drivers(id)  ON DELETE RESTRICT,
	origin                  TEXT             NOT NULL,
	destination             TEXT             NOT NULL,
	origin_lat              DOUBLE PRECISION,
	origin_lon              DOUBLE PRECISION,
	dest_lat                DOUBLE PRECISION,
	dest_lon                DOUBLE PRECISION,
	planned_departure       TIMESTAMPTZ      NOT NULL,
	planned_arrival         TIMESTAMPTZ,
	actual_departure        TIMESTAMPTZ,
	actual_arrival          TIMESTAMPTZ,
	status                  fle_trip_status  NOT NULL DEFAULT 'planned',
	load_kg                 NUMERIC(10,2)    NOT NULL DEFAULT 0,
	load_description        TEXT             NOT NULL DEFAULT '',
	odometer_start_km       NUMERIC(12,2),
	odometer_end_km         NUMERIC(12,2),
	distance_km             NUMERIC(10,2),
	fuel_consumed_l         NUMERIC(10,3),
	route_id                TEXT,
	customs_required        BOOLEAN          NOT NULL DEFAULT FALSE,
	cross_border_countries  TEXT[]           NOT NULL DEFAULT '{}',
	delay_reason            TEXT             NOT NULL DEFAULT '',
	breakdown_at            TIMESTAMPTZ,
	notes                   TEXT             NOT NULL DEFAULT '',
	created_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
	updated_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
	created_by              TEXT             NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN          NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_trips_pkey PRIMARY KEY (id, planned_departure)
) PARTITION BY RANGE (planned_departure);

CREATE TABLE IF NOT EXISTS fle_trips_default PARTITION OF fle_trips DEFAULT;

CREATE INDEX IF NOT EXISTS idx_fle_trips_tenant   ON fle_trips (tenant_id, status)     WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_trips_vehicle  ON fle_trips (tenant_id, vehicle_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_trips_driver   ON fle_trips (tenant_id, driver_id)  WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_trips_depart   ON fle_trips (tenant_id, planned_departure) WHERE NOT is_deleted;

-- ────────────────────────────────────────────────────────────
-- Fuel Records
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_fuel_records (
	id              TEXT          NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id       TEXT          NOT NULL,
	vehicle_id      TEXT          NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	driver_id       TEXT          REFERENCES fle_drivers(id) ON DELETE SET NULL,
	trip_id         TEXT,
	fuelled_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
	litres          NUMERIC(10,3) NOT NULL,
	cost_per_litre  NUMERIC(10,4) NOT NULL,
	total_cost      NUMERIC(12,2) NOT NULL DEFAULT 0,
	currency        CHAR(3)       NOT NULL DEFAULT 'KES',
	station_name    TEXT          NOT NULL DEFAULT '',
	station_lat     DOUBLE PRECISION,
	station_lon     DOUBLE PRECISION,
	odometer_km     NUMERIC(12,2) NOT NULL,
	full_tank       BOOLEAN       NOT NULL DEFAULT TRUE,
	receipt_ref     TEXT          NOT NULL DEFAULT '',
	notes           TEXT          NOT NULL DEFAULT '',
	created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
	updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
	created_by      TEXT          NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN       NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_fuel_pkey PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_fle_fuel_vehicle ON fle_fuel_records (tenant_id, vehicle_id)          WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_fuel_date    ON fle_fuel_records (tenant_id, fuelled_at DESC)     WHERE NOT is_deleted;

-- ────────────────────────────────────────────────────────────
-- Maintenance
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_maintenance (
	id                       TEXT                   NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id                TEXT                   NOT NULL,
	vehicle_id               TEXT                   NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	maintenance_type         fle_maintenance_type   NOT NULL,
	description              TEXT                   NOT NULL,
	status                   fle_maintenance_status NOT NULL DEFAULT 'scheduled',
	scheduled_date           TIMESTAMPTZ            NOT NULL,
	completed_date           TIMESTAMPTZ,
	due_odometer_km          NUMERIC(12,2),
	odometer_at_service_km   NUMERIC(12,2),
	estimated_cost           NUMERIC(12,2)          NOT NULL DEFAULT 0,
	actual_cost              NUMERIC(12,2),
	currency                 CHAR(3)                NOT NULL DEFAULT 'KES',
	supplier_id              TEXT,
	work_order_ref           TEXT                   NOT NULL DEFAULT '',
	parts_replaced           TEXT[]                 NOT NULL DEFAULT '{}',
	next_service_date        TIMESTAMPTZ,
	next_service_odometer_km NUMERIC(12,2),
	notes                    TEXT                   NOT NULL DEFAULT '',
	created_at               TIMESTAMPTZ            NOT NULL DEFAULT NOW(),
	updated_at               TIMESTAMPTZ            NOT NULL DEFAULT NOW(),
	created_by               TEXT                   NOT NULL DEFAULT 'system',
	is_deleted               BOOLEAN                NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_maintenance_pkey PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_fle_maint_vehicle  ON fle_maintenance (tenant_id, vehicle_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_maint_status   ON fle_maintenance (tenant_id, status)     WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_maint_due      ON fle_maintenance (tenant_id, scheduled_date) WHERE status NOT IN ('completed','cancelled');

-- ────────────────────────────────────────────────────────────
-- Inspections
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_inspections (
	id                  TEXT                   NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id           TEXT                   NOT NULL,
	vehicle_id          TEXT                   NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	driver_id           TEXT                   REFERENCES fle_drivers(id) ON DELETE SET NULL,
	inspection_type     fle_inspection_type    NOT NULL,
	inspected_at        TIMESTAMPTZ            NOT NULL DEFAULT NOW(),
	inspected_by        TEXT                   NOT NULL DEFAULT '',
	result              fle_inspection_result  NOT NULL,
	defects             TEXT[]                 NOT NULL DEFAULT '{}',
	advisory_notes      TEXT[]                 NOT NULL DEFAULT '{}',
	odometer_km         NUMERIC(12,2),
	next_inspection_due TIMESTAMPTZ,
	certificate_ref     TEXT                   NOT NULL DEFAULT '',
	notes               TEXT                   NOT NULL DEFAULT '',
	created_at          TIMESTAMPTZ            NOT NULL DEFAULT NOW(),
	updated_at          TIMESTAMPTZ            NOT NULL DEFAULT NOW(),
	created_by          TEXT                   NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN                NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_inspections_pkey PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_fle_insp_vehicle ON fle_inspections (tenant_id, vehicle_id)         WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_insp_result  ON fle_inspections (tenant_id, result)             WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_insp_date    ON fle_inspections (tenant_id, inspected_at DESC)  WHERE NOT is_deleted;

-- ────────────────────────────────────────────────────────────
-- COF Inspections (Certificate of Fitness — East Africa)
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_cof_inspections (
	id                      TEXT                  NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id               TEXT                  NOT NULL,
	vehicle_id              TEXT                  NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	inspected_at            TIMESTAMPTZ           NOT NULL,
	inspection_station      TEXT                  NOT NULL DEFAULT '',
	inspector_id            TEXT                  NOT NULL DEFAULT '',
	result                  fle_inspection_result NOT NULL,
	cof_number              TEXT                  NOT NULL DEFAULT '',
	issued_at               TIMESTAMPTZ,
	expires_at              TIMESTAMPTZ,
	defects_found           TEXT[]                NOT NULL DEFAULT '{}',
	rectification_deadline  TIMESTAMPTZ,
	notes                   TEXT                  NOT NULL DEFAULT '',
	created_at              TIMESTAMPTZ           NOT NULL DEFAULT NOW(),
	updated_at              TIMESTAMPTZ           NOT NULL DEFAULT NOW(),
	created_by              TEXT                  NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN               NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_cof_pkey PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_fle_cof_vehicle ON fle_cof_inspections (tenant_id, vehicle_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_cof_expiry  ON fle_cof_inspections (tenant_id, expires_at) WHERE expires_at IS NOT NULL;

-- ────────────────────────────────────────────────────────────
-- Incidents
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_incidents (
	id                         TEXT                   NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id                  TEXT                   NOT NULL,
	vehicle_id                 TEXT                   NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	driver_id                  TEXT                   REFERENCES fle_drivers(id) ON DELETE SET NULL,
	trip_id                    TEXT,
	occurred_at                TIMESTAMPTZ            NOT NULL,
	severity                   fle_incident_severity  NOT NULL,
	status                     fle_incident_status    NOT NULL DEFAULT 'reported',
	description                TEXT                   NOT NULL,
	location                   TEXT                   NOT NULL DEFAULT '',
	lat                        DOUBLE PRECISION,
	lon                        DOUBLE PRECISION,
	injuries_count             SMALLINT               NOT NULL DEFAULT 0,
	fatalities_count           SMALLINT               NOT NULL DEFAULT 0,
	third_party_involved       BOOLEAN                NOT NULL DEFAULT FALSE,
	police_ref                 TEXT                   NOT NULL DEFAULT '',
	estimated_damage_cost      NUMERIC(12,2)          NOT NULL DEFAULT 0,
	actual_damage_cost         NUMERIC(12,2),
	currency                   CHAR(3)                NOT NULL DEFAULT 'KES',
	overloading_fine_allocated NUMERIC(12,2)          NOT NULL DEFAULT 0,
	notes                      TEXT                   NOT NULL DEFAULT '',
	created_at                 TIMESTAMPTZ            NOT NULL DEFAULT NOW(),
	updated_at                 TIMESTAMPTZ            NOT NULL DEFAULT NOW(),
	created_by                 TEXT                   NOT NULL DEFAULT 'system',
	is_deleted                 BOOLEAN                NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_incidents_pkey PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_fle_inc_vehicle  ON fle_incidents (tenant_id, vehicle_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_inc_severity ON fle_incidents (tenant_id, severity)   WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_inc_status   ON fle_incidents (tenant_id, status)     WHERE NOT is_deleted;

-- ────────────────────────────────────────────────────────────
-- Insurance Policies
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_insurance_policies (
	id            TEXT          NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id     TEXT          NOT NULL,
	vehicle_id    TEXT          NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	policy_number TEXT          NOT NULL,
	insurer       TEXT          NOT NULL,
	policy_type   TEXT          NOT NULL DEFAULT 'comprehensive',
	cover_start   TIMESTAMPTZ   NOT NULL,
	cover_end     TIMESTAMPTZ   NOT NULL,
	premium       NUMERIC(12,2) NOT NULL,
	currency      CHAR(3)       NOT NULL DEFAULT 'KES',
	excess        NUMERIC(12,2) NOT NULL DEFAULT 0,
	sum_insured   NUMERIC(14,2) NOT NULL DEFAULT 0,
	is_active     BOOLEAN       NOT NULL DEFAULT TRUE,
	notes         TEXT          NOT NULL DEFAULT '',
	created_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
	updated_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
	created_by    TEXT          NOT NULL DEFAULT 'system',
	is_deleted    BOOLEAN       NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_insurance_pkey              PRIMARY KEY (id),
	CONSTRAINT fle_insurance_policy_tenant_uq  UNIQUE (tenant_id, policy_number)
);

CREATE INDEX IF NOT EXISTS idx_fle_ins_vehicle ON fle_insurance_policies (tenant_id, vehicle_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_ins_expiry  ON fle_insurance_policies (tenant_id, cover_end)  WHERE is_active;

-- ────────────────────────────────────────────────────────────
-- Registrations
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_registrations (
	id                     TEXT        NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id              TEXT        NOT NULL,
	vehicle_id             TEXT        NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	registration_number    TEXT        NOT NULL,
	registration_authority TEXT        NOT NULL DEFAULT 'NTSA',
	issued_at              TIMESTAMPTZ NOT NULL,
	expires_at             TIMESTAMPTZ NOT NULL,
	certificate_ref        TEXT        NOT NULL DEFAULT '',
	road_worthiness_ref    TEXT        NOT NULL DEFAULT '',
	is_current             BOOLEAN     NOT NULL DEFAULT TRUE,
	notes                  TEXT        NOT NULL DEFAULT '',
	created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	created_by             TEXT        NOT NULL DEFAULT 'system',
	is_deleted             BOOLEAN     NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_reg_pkey PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_fle_reg_vehicle ON fle_registrations (tenant_id, vehicle_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_reg_expiry  ON fle_registrations (tenant_id, expires_at) WHERE is_current;

-- ────────────────────────────────────────────────────────────
-- Tachograph Records (EU EC 561/2006)
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_tachograph_records (
	id                 TEXT           NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id          TEXT           NOT NULL,
	vehicle_id         TEXT           NOT NULL REFERENCES fle_vehicles(id) ON DELETE RESTRICT,
	driver_id          TEXT           NOT NULL REFERENCES fle_drivers(id)  ON DELETE RESTRICT,
	trip_id            TEXT,
	period_start       TIMESTAMPTZ    NOT NULL,
	period_end         TIMESTAMPTZ    NOT NULL,
	mode               fle_tacho_mode NOT NULL DEFAULT 'driving',
	distance_km        NUMERIC(10,2)  NOT NULL DEFAULT 0,
	max_speed_kmh      NUMERIC(6,2)   NOT NULL DEFAULT 0,
	avg_speed_kmh      NUMERIC(6,2)   NOT NULL DEFAULT 0,
	driving_minutes    INTEGER        NOT NULL DEFAULT 0,
	break_minutes      INTEGER        NOT NULL DEFAULT 0,
	rest_minutes       INTEGER        NOT NULL DEFAULT 0,
	infringement_code  TEXT,
	notes              TEXT           NOT NULL DEFAULT '',
	created_at         TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
	updated_at         TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
	created_by         TEXT           NOT NULL DEFAULT 'system',
	is_deleted         BOOLEAN        NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_tacho_pkey PRIMARY KEY (id, period_start)
) PARTITION BY RANGE (period_start);

CREATE TABLE IF NOT EXISTS fle_tachograph_records_default PARTITION OF fle_tachograph_records DEFAULT;

CREATE INDEX IF NOT EXISTS idx_fle_tacho_driver       ON fle_tachograph_records (tenant_id, driver_id, period_start DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_tacho_vehicle      ON fle_tachograph_records (tenant_id, vehicle_id)                   WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_tacho_infringement ON fle_tachograph_records (tenant_id, infringement_code)            WHERE infringement_code IS NOT NULL;

-- ────────────────────────────────────────────────────────────
-- Telematics Events (high-volume, partitioned by occurred_at)
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_telematics_events (
	id             TEXT             NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id      TEXT             NOT NULL,
	vehicle_id     TEXT             NOT NULL,
	driver_id      TEXT,
	trip_id        TEXT,
	provider       TEXT             NOT NULL DEFAULT 'custom',
	event_type     TEXT             NOT NULL,
	occurred_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
	lat            DOUBLE PRECISION NOT NULL,
	lon            DOUBLE PRECISION NOT NULL,
	speed_kmh      DOUBLE PRECISION NOT NULL DEFAULT 0,
	heading_deg    DOUBLE PRECISION,
	altitude_m     DOUBLE PRECISION,
	odometer_km    NUMERIC(12,2),
	engine_on      BOOLEAN,
	fuel_level_pct DOUBLE PRECISION,
	payload        JSONB            NOT NULL DEFAULT '{}',
	created_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
	updated_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
	created_by     TEXT             NOT NULL DEFAULT 'system',
	is_deleted     BOOLEAN          NOT NULL DEFAULT FALSE,
	CONSTRAINT fle_telem_pkey PRIMARY KEY (id, occurred_at)
) PARTITION BY RANGE (occurred_at);

CREATE TABLE IF NOT EXISTS fle_telematics_events_default PARTITION OF fle_telematics_events DEFAULT;

CREATE INDEX IF NOT EXISTS idx_fle_telem_vehicle ON fle_telematics_events (tenant_id, vehicle_id, occurred_at DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_telem_type    ON fle_telematics_events (tenant_id, event_type, occurred_at DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_telem_driver  ON fle_telematics_events (tenant_id, driver_id)                   WHERE driver_id IS NOT NULL AND NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_fle_telem_payload ON fle_telematics_events USING GIN (payload)                      WHERE NOT is_deleted;

-- ────────────────────────────────────────────────────────────
-- Domain Events (immutable audit log)
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fle_domain_events (
	id          TEXT        NOT NULL DEFAULT gen_random_uuid()::TEXT,
	tenant_id   TEXT        NOT NULL,
	actor_id    TEXT        NOT NULL DEFAULT 'system',
	event_type  TEXT        NOT NULL,
	entity_id   TEXT        NOT NULL,
	occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	payload     JSONB       NOT NULL DEFAULT '{}',
	CONSTRAINT fle_events_pkey PRIMARY KEY (id, occurred_at)
) PARTITION BY RANGE (occurred_at);

CREATE TABLE IF NOT EXISTS fle_domain_events_default PARTITION OF fle_domain_events DEFAULT;

CREATE INDEX IF NOT EXISTS idx_fle_events_tenant ON fle_domain_events (tenant_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_fle_events_entity ON fle_domain_events (tenant_id, entity_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_fle_events_type   ON fle_domain_events (tenant_id, event_type);

-- ────────────────────────────────────────────────────────────
-- Auto-update updated_at trigger
-- ────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION fle_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
	NEW.updated_at := NOW();
	RETURN NEW;
END;
$$;

DO $$
DECLARE tbl TEXT;
BEGIN
	FOREACH tbl IN ARRAY ARRAY[
		'fle_vehicles','fle_drivers','fle_vehicle_assignments',
		'fle_fuel_records','fle_maintenance','fle_inspections',
		'fle_cof_inspections','fle_incidents','fle_insurance_policies',
		'fle_registrations'
	]
	LOOP
		EXECUTE format(
			'DROP TRIGGER IF EXISTS trg_%I_updated_at ON %I;
			 CREATE TRIGGER trg_%I_updated_at BEFORE UPDATE ON %I
			 FOR EACH ROW EXECUTE FUNCTION fle_set_updated_at()',
			tbl, tbl, tbl, tbl
		);
	END LOOP;
END;
$$;

COMMENT ON TABLE fle_vehicles             IS 'Fleet vehicles — core entity';
COMMENT ON TABLE fle_drivers              IS 'Licensed drivers assigned to fleet';
COMMENT ON TABLE fle_trips                IS 'Trip records (partitioned by planned_departure)';
COMMENT ON TABLE fle_telematics_events    IS 'High-frequency GPS/sensor stream (partitioned by occurred_at)';
COMMENT ON TABLE fle_tachograph_records   IS 'EU tachograph (EC 561/2006) records';
COMMENT ON TABLE fle_cof_inspections      IS 'East Africa Certificate of Fitness records';
COMMENT ON TABLE fle_domain_events        IS 'Immutable audit log of all state changes';
