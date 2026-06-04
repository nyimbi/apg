-- =============================================================================
-- Time & Attendance — PostgreSQL Schema
-- Copyright © 2025 Datacraft. Author: Nyimbi Odero
-- Prefix: tat_
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- =============================================================================
-- ENUMERATIONS
-- =============================================================================

DO $$ BEGIN
    CREATE TYPE tat_time_entry_status AS ENUM (
        'draft', 'submitted', 'approved', 'rejected', 'locked', 'processing'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE tat_time_entry_type AS ENUM (
        'regular', 'overtime', 'holiday', 'sick', 'vacation', 'personal',
        'training', 'travel', 'break', 'on_call', 'toil', 'comp_time'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE tat_attendance_status AS ENUM (
        'present', 'absent', 'late', 'early_departure', 'partial_day',
        'excused', 'remote'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE tat_leave_type AS ENUM (
        'vacation', 'sick', 'personal', 'maternity', 'paternity', 'parental',
        'bereavement', 'jury_duty', 'military', 'sabbatical', 'unpaid',
        'fmla', 'toil', 'comp_time', 'public_holiday', 'study'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE tat_approval_status AS ENUM (
        'pending', 'approved', 'rejected', 'escalated', 'auto_approved',
        'expired', 'withdrawn'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE tat_schedule_type AS ENUM (
        'fixed', 'flexible', 'rotating', 'compressed', 'remote',
        'annualised', 'zero_hours'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE tat_device_type AS ENUM (
        'mobile_app', 'web_browser', 'biometric_terminal', 'iot_sensor',
        'smart_watch', 'badge_reader', 'kiosk', 'api'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE tat_exception_type AS ENUM (
        'missing_clock_out', 'late_arrival', 'early_departure', 'overtime',
        'geofence_violation', 'biometric_failure', 'duplicate_entry',
        'schedule_violation', 'max_hours_exceeded'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE tat_shift_status AS ENUM (
        'planned', 'published', 'in_progress', 'completed', 'cancelled', 'swapped'
    );
EXCEPTION WHEN duplicate_object THEN null; END $$;

-- =============================================================================
-- BASE AUDIT COLUMNS (macro-like helper — applied via column list repetition)
-- =============================================================================

-- =============================================================================
-- TABLE: tat_time_policy
-- Governs overtime rules, workweek definition, and pay period configuration
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_time_policy (
    id                          VARCHAR(36)     PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id                   VARCHAR(36)     NOT NULL,
    name                        VARCHAR(100)    NOT NULL,
    description                 TEXT,
    timezone                    VARCHAR(50)     NOT NULL DEFAULT 'UTC',
    workweek                    JSONB           NOT NULL DEFAULT '["Mon","Tue","Wed","Thu","Fri"]',
    overtime_threshold_daily    NUMERIC(5,2)    NOT NULL DEFAULT 8.0,
    overtime_threshold_weekly   NUMERIC(5,2)    NOT NULL DEFAULT 40.0,
    double_time_threshold_daily NUMERIC(5,2)    NOT NULL DEFAULT 12.0,
    overtime_multiplier         NUMERIC(4,2)    NOT NULL DEFAULT 1.5,
    double_time_multiplier      NUMERIC(4,2)    NOT NULL DEFAULT 2.0,
    holiday_pay_multiplier      NUMERIC(4,2)    NOT NULL DEFAULT 2.0,
    min_rest_between_shifts_h   NUMERIC(4,1)    NOT NULL DEFAULT 11.0,
    max_consecutive_days        INTEGER         NOT NULL DEFAULT 6,
    max_weekly_hours            NUMERIC(5,2)    NOT NULL DEFAULT 48.0,
    break_rules                 JSONB           NOT NULL DEFAULT '{}',
    flexi_core_start            TIME,
    flexi_core_end              TIME,
    flexi_max_carry_hours       NUMERIC(5,2)    DEFAULT 16.0,
    toil_enabled                BOOLEAN         NOT NULL DEFAULT false,
    comp_time_enabled           BOOLEAN         NOT NULL DEFAULT false,
    comp_time_jurisdiction      VARCHAR(50),
    annualised_hours_enabled    BOOLEAN         NOT NULL DEFAULT false,
    contracted_annual_hours     NUMERIC(7,2),
    medical_cert_threshold_days INTEGER         NOT NULL DEFAULT 3,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    is_deleted                  BOOLEAN         NOT NULL DEFAULT false,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by                  VARCHAR(36)     NOT NULL,
    metadata                    JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tat_time_policy_tenant    ON tat_time_policy (tenant_id);
CREATE INDEX IF NOT EXISTS idx_tat_time_policy_active    ON tat_time_policy (tenant_id, is_active) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_public_holiday
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_public_holiday (
    id              VARCHAR(36)     PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)     NOT NULL,
    name            VARCHAR(100)    NOT NULL,
    holiday_date    DATE            NOT NULL,
    jurisdiction    VARCHAR(50)     NOT NULL,
    is_statutory    BOOLEAN         NOT NULL DEFAULT true,
    is_substituted  BOOLEAN         NOT NULL DEFAULT false,
    substitute_date DATE,
    timezone        VARCHAR(50)     NOT NULL DEFAULT 'UTC',
    is_deleted      BOOLEAN         NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_public_holiday_tenant     ON tat_public_holiday (tenant_id);
CREATE INDEX IF NOT EXISTS idx_tat_public_holiday_date       ON tat_public_holiday (tenant_id, holiday_date);
CREATE INDEX IF NOT EXISTS idx_tat_public_holiday_juris      ON tat_public_holiday (tenant_id, jurisdiction, holiday_date);
CREATE UNIQUE INDEX IF NOT EXISTS uq_tat_public_holiday ON tat_public_holiday (tenant_id, holiday_date, jurisdiction) WHERE NOT is_deleted AND NOT is_substituted;

-- =============================================================================
-- TABLE: tat_geofence_location
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_geofence_location (
    id              VARCHAR(36)     PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)     NOT NULL,
    name            VARCHAR(100)    NOT NULL,
    address         TEXT,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    radius_metres   NUMERIC(8,1)    NOT NULL DEFAULT 200.0,
    timezone        VARCHAR(50)     NOT NULL DEFAULT 'UTC',
    is_active       BOOLEAN         NOT NULL DEFAULT true,
    is_deleted      BOOLEAN         NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)     NOT NULL,
    metadata        JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tat_geofence_tenant ON tat_geofence_location (tenant_id) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_attendance_device
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_attendance_device (
    id                  VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id           VARCHAR(36)         NOT NULL,
    device_code         VARCHAR(50)         NOT NULL,
    device_name         VARCHAR(100)        NOT NULL,
    device_type         tat_device_type     NOT NULL,
    geofence_id         VARCHAR(36)         REFERENCES tat_geofence_location(id),
    serial_number       VARCHAR(100),
    firmware_version    VARCHAR(50),
    last_sync_at        TIMESTAMPTZ,
    biometric_enabled   BOOLEAN             NOT NULL DEFAULT false,
    is_active           BOOLEAN             NOT NULL DEFAULT true,
    is_deleted          BOOLEAN             NOT NULL DEFAULT false,
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by          VARCHAR(36)         NOT NULL,
    metadata            JSONB               NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tat_device_tenant     ON tat_attendance_device (tenant_id) WHERE NOT is_deleted;
CREATE UNIQUE INDEX IF NOT EXISTS uq_tat_device_code ON tat_attendance_device (tenant_id, device_code) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_shift_schedule
-- Template: defines recurring weekly/rotating patterns
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_shift_schedule (
    id                      VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id               VARCHAR(36)         NOT NULL,
    policy_id               VARCHAR(36)         NOT NULL REFERENCES tat_time_policy(id),
    schedule_name           VARCHAR(100)        NOT NULL,
    schedule_type           tat_schedule_type   NOT NULL DEFAULT 'fixed',
    description             TEXT,
    effective_date          DATE                NOT NULL,
    end_date                DATE,
    patterns                JSONB               NOT NULL DEFAULT '[]',
    -- [{"days_of_week":[0,1,2,3,4], "start_time":"09:00", "end_time":"17:00"}]
    assigned_employees      JSONB               NOT NULL DEFAULT '[]',
    department_id           VARCHAR(36),
    location_id             VARCHAR(36)         REFERENCES tat_geofence_location(id),
    min_employees_per_shift INTEGER             DEFAULT 1,
    max_employees_per_shift INTEGER,
    allow_overtime          BOOLEAN             NOT NULL DEFAULT true,
    allow_shift_swapping    BOOLEAN             NOT NULL DEFAULT true,
    status                  VARCHAR(20)         NOT NULL DEFAULT 'draft',
    version                 INTEGER             NOT NULL DEFAULT 1,
    parent_schedule_id      VARCHAR(36)         REFERENCES tat_shift_schedule(id),
    is_deleted              BOOLEAN             NOT NULL DEFAULT false,
    created_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by              VARCHAR(36)         NOT NULL,
    metadata                JSONB               NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tat_schedule_tenant      ON tat_shift_schedule (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_schedule_policy      ON tat_shift_schedule (policy_id);
CREATE INDEX IF NOT EXISTS idx_tat_schedule_dept        ON tat_shift_schedule (tenant_id, department_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_schedule_dates       ON tat_shift_schedule (tenant_id, effective_date, end_date) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_shift
-- Concrete instances of scheduled shifts for specific employees
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_shift (
    id                  VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id           VARCHAR(36)         NOT NULL,
    schedule_id         VARCHAR(36)         NOT NULL REFERENCES tat_shift_schedule(id),
    employee_id         VARCHAR(36)         NOT NULL,
    policy_id           VARCHAR(36)         NOT NULL REFERENCES tat_time_policy(id),
    shift_date          DATE                NOT NULL,
    planned_start       TIMESTAMPTZ         NOT NULL,
    planned_end         TIMESTAMPTZ         NOT NULL,
    actual_start        TIMESTAMPTZ,
    actual_end          TIMESTAMPTZ,
    location_id         VARCHAR(36)         REFERENCES tat_geofence_location(id),
    status              tat_shift_status    NOT NULL DEFAULT 'planned',
    is_night_shift      BOOLEAN             NOT NULL DEFAULT false,
    is_public_holiday   BOOLEAN             NOT NULL DEFAULT false,
    original_shift_id   VARCHAR(36)         REFERENCES tat_shift(id),  -- for swaps
    swapped_with_id     VARCHAR(36)         REFERENCES tat_shift(id),
    notes               TEXT,
    is_deleted          BOOLEAN             NOT NULL DEFAULT false,
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by          VARCHAR(36)         NOT NULL,
    metadata            JSONB               NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tat_shift_tenant         ON tat_shift (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_shift_employee       ON tat_shift (tenant_id, employee_id, shift_date) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_shift_date           ON tat_shift (tenant_id, shift_date) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_shift_schedule       ON tat_shift (schedule_id);

-- =============================================================================
-- TABLE: tat_roster
-- Published roster covering a date range for a department/location
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_roster (
    id              VARCHAR(36)     PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)     NOT NULL,
    name            VARCHAR(100)    NOT NULL,
    department_id   VARCHAR(36),
    location_id     VARCHAR(36)     REFERENCES tat_geofence_location(id),
    period_start    DATE            NOT NULL,
    period_end      DATE            NOT NULL,
    status          VARCHAR(20)     NOT NULL DEFAULT 'draft',
    published_at    TIMESTAMPTZ,
    published_by    VARCHAR(36),
    shift_ids       JSONB           NOT NULL DEFAULT '[]',
    constraints     JSONB           NOT NULL DEFAULT '{}',
    notes           TEXT,
    is_deleted      BOOLEAN         NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_roster_tenant   ON tat_roster (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_roster_period   ON tat_roster (tenant_id, period_start, period_end) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_time_entry
-- Individual clock-in/out records
-- Partitioned by entry_date for large volumes
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_time_entry (
    id                      VARCHAR(36)             PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id               VARCHAR(36)             NOT NULL,
    employee_id             VARCHAR(36)             NOT NULL,
    shift_id                VARCHAR(36)             REFERENCES tat_shift(id),
    policy_id               VARCHAR(36)             REFERENCES tat_time_policy(id),
    entry_date              DATE                    NOT NULL,
    clock_in                TIMESTAMPTZ             NOT NULL,
    clock_out               TIMESTAMPTZ,
    break_minutes           INTEGER                 NOT NULL DEFAULT 0,
    total_hours             NUMERIC(6,2),
    regular_hours           NUMERIC(6,2),
    overtime_hours          NUMERIC(6,2),
    double_time_hours       NUMERIC(6,2),
    holiday_hours           NUMERIC(6,2),
    entry_type              tat_time_entry_type     NOT NULL DEFAULT 'regular',
    entry_method            VARCHAR(30)             NOT NULL DEFAULT 'web',
    status                  tat_time_entry_status   NOT NULL DEFAULT 'draft',
    clock_in_lat            DOUBLE PRECISION,
    clock_in_lng            DOUBLE PRECISION,
    clock_out_lat           DOUBLE PRECISION,
    clock_out_lng           DOUBLE PRECISION,
    geofence_verified       BOOLEAN                 NOT NULL DEFAULT false,
    device_id               VARCHAR(36)             REFERENCES tat_attendance_device(id),
    ip_address              VARCHAR(45),
    biometric_confidence    NUMERIC(4,3),
    biometric_verified      BOOLEAN                 NOT NULL DEFAULT false,
    fraud_score             NUMERIC(4,3),
    requires_approval       BOOLEAN                 NOT NULL DEFAULT false,
    approved_by             VARCHAR(36),
    approved_at             TIMESTAMPTZ,
    rejection_reason        TEXT,
    reviewed_by             VARCHAR(36),
    reviewed_at             TIMESTAMPTZ,
    cost_center             VARCHAR(50),
    project_allocations     JSONB                   NOT NULL DEFAULT '[]',
    billable_hours          NUMERIC(6,2),
    notes                   TEXT,
    is_night_shift          BOOLEAN                 NOT NULL DEFAULT false,
    is_public_holiday       BOOLEAN                 NOT NULL DEFAULT false,
    is_deleted              BOOLEAN                 NOT NULL DEFAULT false,
    created_at              TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by              VARCHAR(36)             NOT NULL,
    metadata                JSONB                   NOT NULL DEFAULT '{}'
) PARTITION BY RANGE (entry_date);

-- Default partition (catch-all for current year)
CREATE TABLE IF NOT EXISTS tat_time_entry_default PARTITION OF tat_time_entry DEFAULT;

-- Create year partitions as needed; example for 2025 and 2026:
CREATE TABLE IF NOT EXISTS tat_time_entry_y2025 PARTITION OF tat_time_entry
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS tat_time_entry_y2026 PARTITION OF tat_time_entry
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE INDEX IF NOT EXISTS idx_tat_entry_tenant_emp  ON tat_time_entry (tenant_id, employee_id, entry_date);
CREATE INDEX IF NOT EXISTS idx_tat_entry_date        ON tat_time_entry (tenant_id, entry_date);
CREATE INDEX IF NOT EXISTS idx_tat_entry_status      ON tat_time_entry (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_entry_shift       ON tat_time_entry (shift_id) WHERE shift_id IS NOT NULL;

-- Prevent duplicate un-deleted entries for same employee on same date
CREATE UNIQUE INDEX IF NOT EXISTS uq_tat_entry_per_day
    ON tat_time_entry (tenant_id, employee_id, entry_date)
    WHERE NOT is_deleted AND entry_type NOT IN ('break', 'on_call');

-- =============================================================================
-- TABLE: tat_break
-- Break periods linked to a time entry
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_break (
    id              VARCHAR(36)     PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)     NOT NULL,
    time_entry_id   VARCHAR(36)     NOT NULL REFERENCES tat_time_entry(id) ON DELETE CASCADE,
    break_type      VARCHAR(30)     NOT NULL DEFAULT 'meal',
    break_start     TIMESTAMPTZ     NOT NULL,
    break_end       TIMESTAMPTZ     NOT NULL,
    duration_minutes INTEGER        GENERATED ALWAYS AS (
                        EXTRACT(EPOCH FROM (break_end - break_start))::INTEGER / 60
                    ) STORED,
    is_paid         BOOLEAN         NOT NULL DEFAULT false,
    is_deleted      BOOLEAN         NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_break_entry ON tat_break (time_entry_id) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_timesheet
-- Aggregated pay-period summaries for approval and payroll export
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_timesheet (
    id                  VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id           VARCHAR(36)         NOT NULL,
    employee_id         VARCHAR(36)         NOT NULL,
    period_start        DATE                NOT NULL,
    period_end          DATE                NOT NULL,
    total_hours         NUMERIC(7,2)        NOT NULL DEFAULT 0,
    regular_hours       NUMERIC(7,2)        NOT NULL DEFAULT 0,
    overtime_hours      NUMERIC(7,2)        NOT NULL DEFAULT 0,
    double_time_hours   NUMERIC(7,2)        NOT NULL DEFAULT 0,
    holiday_hours       NUMERIC(7,2)        NOT NULL DEFAULT 0,
    leave_hours         NUMERIC(7,2)        NOT NULL DEFAULT 0,
    gross_pay           NUMERIC(12,2),
    currency            VARCHAR(3)          NOT NULL DEFAULT 'USD',
    status              tat_approval_status NOT NULL DEFAULT 'pending',
    submitted_by        VARCHAR(36),
    submitted_at        TIMESTAMPTZ,
    approved_by         VARCHAR(36),
    approved_at         TIMESTAMPTZ,
    rejected_by         VARCHAR(36),
    rejected_at         TIMESTAMPTZ,
    rejection_reason    TEXT,
    entry_ids           JSONB               NOT NULL DEFAULT '[]',
    payroll_export_id   VARCHAR(36),
    notes               TEXT,
    is_deleted          BOOLEAN             NOT NULL DEFAULT false,
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by          VARCHAR(36)         NOT NULL,
    metadata            JSONB               NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tat_timesheet_tenant  ON tat_timesheet (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_timesheet_emp     ON tat_timesheet (tenant_id, employee_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_timesheet_period  ON tat_timesheet (tenant_id, period_start, period_end) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_timesheet_status  ON tat_timesheet (tenant_id, status) WHERE NOT is_deleted;

CREATE UNIQUE INDEX IF NOT EXISTS uq_tat_timesheet_period
    ON tat_timesheet (tenant_id, employee_id, period_start, period_end)
    WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_overtime_request
-- Pre-authorisation for overtime work
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_overtime_request (
    id              VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)         NOT NULL,
    employee_id     VARCHAR(36)         NOT NULL,
    shift_id        VARCHAR(36)         REFERENCES tat_shift(id),
    request_date    DATE                NOT NULL,
    requested_hours NUMERIC(5,2)        NOT NULL,
    reason          TEXT                NOT NULL,
    status          tat_approval_status NOT NULL DEFAULT 'pending',
    approved_by     VARCHAR(36),
    approved_at     TIMESTAMPTZ,
    rejected_by     VARCHAR(36),
    rejected_at     TIMESTAMPTZ,
    rejection_reason TEXT,
    is_deleted      BOOLEAN             NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)         NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_ot_request_tenant  ON tat_overtime_request (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_ot_request_emp     ON tat_overtime_request (tenant_id, employee_id) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_leave_policy
-- Leave entitlement rules per leave type
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_leave_policy (
    id                          VARCHAR(36)     PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id                   VARCHAR(36)     NOT NULL,
    name                        VARCHAR(100)    NOT NULL,
    leave_type                  tat_leave_type  NOT NULL,
    annual_days                 NUMERIC(5,1)    NOT NULL,
    accrual_method              VARCHAR(20)     NOT NULL DEFAULT 'prorated',
    carry_forward_days          NUMERIC(5,1)    NOT NULL DEFAULT 0,
    max_carry_forward_days      NUMERIC(5,1)    NOT NULL DEFAULT 0,
    waiting_days                INTEGER         NOT NULL DEFAULT 0,
    notice_days_required        INTEGER         NOT NULL DEFAULT 0,
    medical_cert_required_days  INTEGER         NOT NULL DEFAULT 3,
    min_service_months          INTEGER         NOT NULL DEFAULT 0,
    fte_prorated                BOOLEAN         NOT NULL DEFAULT true,
    paid                        BOOLEAN         NOT NULL DEFAULT true,
    jurisdiction                VARCHAR(50)     NOT NULL DEFAULT 'global',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    is_deleted                  BOOLEAN         NOT NULL DEFAULT false,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by                  VARCHAR(36)     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_leave_policy_tenant ON tat_leave_policy (tenant_id) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_leave_entitlement
-- Per-employee leave balance per leave type per year
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_leave_entitlement (
    id              VARCHAR(36)     PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)     NOT NULL,
    employee_id     VARCHAR(36)     NOT NULL,
    policy_id       VARCHAR(36)     NOT NULL REFERENCES tat_leave_policy(id),
    leave_type      tat_leave_type  NOT NULL,
    entitlement_year INTEGER        NOT NULL,
    annual_days     NUMERIC(5,1)    NOT NULL,
    carried_forward NUMERIC(5,1)    NOT NULL DEFAULT 0,
    accrued_to_date NUMERIC(5,1)    NOT NULL DEFAULT 0,
    used_days       NUMERIC(5,1)    NOT NULL DEFAULT 0,
    pending_days    NUMERIC(5,1)    NOT NULL DEFAULT 0,
    balance_days    NUMERIC(5,1)    GENERATED ALWAYS AS (accrued_to_date + carried_forward - used_days) STORED,
    available_days  NUMERIC(5,1)    GENERATED ALWAYS AS (accrued_to_date + carried_forward - used_days - pending_days) STORED,
    is_deleted      BOOLEAN         NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)     NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_tat_entitlement
    ON tat_leave_entitlement (tenant_id, employee_id, leave_type, entitlement_year)
    WHERE NOT is_deleted;

CREATE INDEX IF NOT EXISTS idx_tat_entitlement_emp ON tat_leave_entitlement (tenant_id, employee_id) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_leave_request
-- Employee leave requests with full approval workflow
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_leave_request (
    id                      VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id               VARCHAR(36)         NOT NULL,
    employee_id             VARCHAR(36)         NOT NULL,
    entitlement_id          VARCHAR(36)         REFERENCES tat_leave_entitlement(id),
    leave_type              tat_leave_type      NOT NULL,
    start_date              DATE                NOT NULL,
    end_date                DATE                NOT NULL,
    total_days              NUMERIC(5,1)        NOT NULL,
    total_hours             NUMERIC(6,2),
    is_half_day             BOOLEAN             NOT NULL DEFAULT false,
    half_day_portion        VARCHAR(10),        -- 'morning' | 'afternoon'
    is_emergency            BOOLEAN             NOT NULL DEFAULT false,
    reason                  TEXT,
    status                  tat_approval_status NOT NULL DEFAULT 'pending',
    current_approver        VARCHAR(36),
    approval_chain          JSONB               NOT NULL DEFAULT '[]',
    approved_by             VARCHAR(36),
    approved_at             TIMESTAMPTZ,
    rejected_by             VARCHAR(36),
    rejected_at             TIMESTAMPTZ,
    rejection_reason        TEXT,
    medical_cert_attached   BOOLEAN             NOT NULL DEFAULT false,
    attachments             JSONB               NOT NULL DEFAULT '[]',
    manager_notes           TEXT,
    hr_notes                TEXT,
    -- FMLA / statutory leave tracking
    is_statutory            BOOLEAN             NOT NULL DEFAULT false,
    statutory_type          VARCHAR(50),        -- 'FMLA', 'maternity', 'paternity', etc.
    statutory_jurisdiction  VARCHAR(50),
    is_deleted              BOOLEAN             NOT NULL DEFAULT false,
    created_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by              VARCHAR(36)         NOT NULL,
    metadata                JSONB               NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tat_leave_req_tenant  ON tat_leave_request (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_leave_req_emp     ON tat_leave_request (tenant_id, employee_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_leave_req_dates   ON tat_leave_request (tenant_id, start_date, end_date) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_leave_req_status  ON tat_leave_request (tenant_id, status) WHERE NOT is_deleted;

-- Prevent overlapping approved/pending leaves for same employee (GiST exclusion)
CREATE EXTENSION IF NOT EXISTS btree_gist;
ALTER TABLE tat_leave_request ADD CONSTRAINT no_overlapping_active_leaves
    EXCLUDE USING gist (
        tenant_id WITH =,
        employee_id WITH =,
        daterange(start_date, end_date, '[]') WITH &&
    ) WHERE (NOT is_deleted AND status IN ('pending', 'approved'));

-- =============================================================================
-- TABLE: tat_absence
-- Recorded unplanned absences
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_absence (
    id              VARCHAR(36)             NOT NULL DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)             NOT NULL,
    employee_id     VARCHAR(36)             NOT NULL,
    absence_date    DATE                    NOT NULL,
    absence_type    VARCHAR(30)             NOT NULL DEFAULT 'unplanned',
    status          tat_attendance_status   NOT NULL DEFAULT 'absent',
    leave_request_id VARCHAR(36)            REFERENCES tat_leave_request(id),
    reason          TEXT,
    bradford_score  INTEGER,
    is_deleted      BOOLEAN                 NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)             NOT NULL,
    PRIMARY KEY (id, absence_date)
) PARTITION BY RANGE (absence_date);

CREATE TABLE IF NOT EXISTS tat_absence_default PARTITION OF tat_absence DEFAULT;
CREATE TABLE IF NOT EXISTS tat_absence_y2025 PARTITION OF tat_absence FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
CREATE TABLE IF NOT EXISTS tat_absence_y2026 PARTITION OF tat_absence FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE INDEX IF NOT EXISTS idx_tat_absence_emp  ON tat_absence (tenant_id, employee_id, absence_date);

-- =============================================================================
-- TABLE: tat_comp_time
-- Compensatory time earned and used
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_comp_time (
    id              VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)         NOT NULL,
    employee_id     VARCHAR(36)         NOT NULL,
    time_entry_id   VARCHAR(36)         REFERENCES tat_time_entry(id),
    transaction_type VARCHAR(10)        NOT NULL DEFAULT 'earn',   -- 'earn' | 'use'
    hours           NUMERIC(6,2)        NOT NULL,
    balance_after   NUMERIC(7,2)        NOT NULL,
    effective_date  DATE                NOT NULL,
    expiry_date     DATE,
    reason          TEXT,
    approved_by     VARCHAR(36),
    is_deleted      BOOLEAN             NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)         NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_comp_time_emp ON tat_comp_time (tenant_id, employee_id, effective_date) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_attendance_exception
-- Anomalies and violations requiring investigation
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_attendance_exception (
    id              VARCHAR(36)             PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)             NOT NULL,
    employee_id     VARCHAR(36)             NOT NULL,
    time_entry_id   VARCHAR(36)             REFERENCES tat_time_entry(id),
    exception_type  tat_exception_type      NOT NULL,
    severity        VARCHAR(10)             NOT NULL DEFAULT 'medium',
    description     TEXT                    NOT NULL,
    detected_at     TIMESTAMPTZ             NOT NULL DEFAULT now(),
    owner_id        VARCHAR(36),
    resolved_at     TIMESTAMPTZ,
    resolution_notes TEXT,
    status          VARCHAR(20)             NOT NULL DEFAULT 'open',
    is_deleted      BOOLEAN                 NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ             NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ             NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)             NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_exception_tenant   ON tat_attendance_exception (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_exception_emp      ON tat_attendance_exception (tenant_id, employee_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_exception_status   ON tat_attendance_exception (tenant_id, status) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_payroll_export
-- Approved timesheet bundles sent to payroll
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_payroll_export (
    id              VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)         NOT NULL,
    period_start    DATE                NOT NULL,
    period_end      DATE                NOT NULL,
    timesheet_ids   JSONB               NOT NULL DEFAULT '[]',
    total_employees INTEGER             NOT NULL DEFAULT 0,
    total_hours     NUMERIC(10,2)       NOT NULL DEFAULT 0,
    total_gross_pay NUMERIC(14,2),
    currency        VARCHAR(3)          NOT NULL DEFAULT 'USD',
    status          VARCHAR(20)         NOT NULL DEFAULT 'draft',
    exported_at     TIMESTAMPTZ,
    exported_by     VARCHAR(36),
    event_stream    VARCHAR(100),
    processor       VARCHAR(30)         NOT NULL DEFAULT 'bytewax',
    notes           TEXT,
    is_deleted      BOOLEAN             NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)         NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_payroll_export_tenant  ON tat_payroll_export (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_tat_payroll_export_period  ON tat_payroll_export (tenant_id, period_start, period_end) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_flexitime_balance
-- Running flexi-time credit/debit ledger per employee
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_flexitime_balance (
    id              VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)         NOT NULL,
    employee_id     VARCHAR(36)         NOT NULL,
    balance_date    DATE                NOT NULL,
    delta_hours     NUMERIC(6,2)        NOT NULL,   -- positive = credit, negative = debit
    balance_hours   NUMERIC(7,2)        NOT NULL,   -- running total
    time_entry_id   VARCHAR(36)         REFERENCES tat_time_entry(id),
    notes           TEXT,
    is_deleted      BOOLEAN             NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by      VARCHAR(36)         NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_flexi_emp   ON tat_flexitime_balance (tenant_id, employee_id, balance_date) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_shift_swap_request
-- Employee-initiated shift swap workflow
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_shift_swap_request (
    id                  VARCHAR(36)         PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id           VARCHAR(36)         NOT NULL,
    requester_id        VARCHAR(36)         NOT NULL,
    requester_shift_id  VARCHAR(36)         NOT NULL REFERENCES tat_shift(id),
    target_id           VARCHAR(36),
    target_shift_id     VARCHAR(36)         REFERENCES tat_shift(id),
    reason              TEXT,
    status              tat_approval_status NOT NULL DEFAULT 'pending',
    approved_by         VARCHAR(36),
    approved_at         TIMESTAMPTZ,
    rejected_by         VARCHAR(36),
    rejected_at         TIMESTAMPTZ,
    rejection_reason    TEXT,
    is_deleted          BOOLEAN             NOT NULL DEFAULT false,
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ         NOT NULL DEFAULT now(),
    created_by          VARCHAR(36)         NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_swap_tenant ON tat_shift_swap_request (tenant_id) WHERE NOT is_deleted;

-- =============================================================================
-- TABLE: tat_biometric_sync_log
-- Audit trail for biometric device synchronisation
-- =============================================================================
CREATE TABLE IF NOT EXISTS tat_biometric_sync_log (
    id              VARCHAR(36)     PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id       VARCHAR(36)     NOT NULL,
    device_id       VARCHAR(36)     NOT NULL REFERENCES tat_attendance_device(id),
    sync_started_at TIMESTAMPTZ     NOT NULL DEFAULT now(),
    sync_ended_at   TIMESTAMPTZ,
    records_pulled  INTEGER         NOT NULL DEFAULT 0,
    records_created INTEGER         NOT NULL DEFAULT 0,
    records_skipped INTEGER         NOT NULL DEFAULT 0,
    errors          JSONB           NOT NULL DEFAULT '[]',
    status          VARCHAR(20)     NOT NULL DEFAULT 'running',
    initiated_by    VARCHAR(36)     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tat_sync_log_device ON tat_biometric_sync_log (device_id, sync_started_at DESC);

-- =============================================================================
-- TRIGGERS: updated_at auto-maintenance
-- =============================================================================
CREATE OR REPLACE FUNCTION tat_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DO $$ DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'tat_time_policy','tat_public_holiday','tat_geofence_location',
        'tat_attendance_device','tat_shift_schedule','tat_shift','tat_roster',
        'tat_overtime_request','tat_leave_policy','tat_leave_entitlement',
        'tat_leave_request','tat_comp_time','tat_attendance_exception',
        'tat_payroll_export','tat_flexitime_balance','tat_shift_swap_request'
    ] LOOP
        EXECUTE format(
            'CREATE TRIGGER trg_%s_updated_at BEFORE UPDATE ON %s '
            'FOR EACH ROW EXECUTE FUNCTION tat_set_updated_at()',
            tbl, tbl
        );
    END LOOP;
END $$;

-- =============================================================================
-- VIEWS
-- =============================================================================

CREATE OR REPLACE VIEW tat_v_active_shifts AS
SELECT
    s.id, s.tenant_id, s.employee_id, s.shift_date,
    s.planned_start, s.planned_end, s.actual_start, s.actual_end,
    s.status, s.is_night_shift, s.is_public_holiday,
    sc.schedule_name, sc.schedule_type,
    p.name AS policy_name, p.timezone
FROM tat_shift s
JOIN tat_shift_schedule sc ON sc.id = s.schedule_id
JOIN tat_time_policy p     ON p.id  = s.policy_id
WHERE NOT s.is_deleted AND s.status NOT IN ('cancelled');

COMMENT ON VIEW tat_v_active_shifts IS 'All non-cancelled shifts with schedule and policy context';

CREATE OR REPLACE VIEW tat_v_timesheet_summary AS
SELECT
    ts.id, ts.tenant_id, ts.employee_id,
    ts.period_start, ts.period_end,
    ts.total_hours, ts.regular_hours, ts.overtime_hours,
    ts.double_time_hours, ts.holiday_hours, ts.leave_hours,
    ts.gross_pay, ts.currency, ts.status,
    ts.submitted_at, ts.approved_at
FROM tat_timesheet ts
WHERE NOT ts.is_deleted;

COMMENT ON VIEW tat_v_timesheet_summary IS 'Timesheet KPIs for dashboard and reporting';

CREATE OR REPLACE VIEW tat_v_leave_balance AS
SELECT
    e.tenant_id, e.employee_id, e.leave_type, e.entitlement_year,
    e.annual_days, e.carried_forward, e.accrued_to_date,
    e.used_days, e.pending_days, e.balance_days, e.available_days,
    p.name AS policy_name, p.paid, p.jurisdiction
FROM tat_leave_entitlement e
JOIN tat_leave_policy p ON p.id = e.policy_id
WHERE NOT e.is_deleted;

COMMENT ON VIEW tat_v_leave_balance IS 'Employee leave balance with policy context';
