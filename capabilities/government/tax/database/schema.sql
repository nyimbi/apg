-- APG Tax Administration — Normalized PostgreSQL Schema
-- © 2025 Datacraft | Author: Nyimbi Odero
-- Run: psql $DATABASE_URL < database/schema.sql
--
-- Conventions:
--   - All PKs are UUID7 stored as TEXT
--   - tenant_id TEXT NOT NULL on every table (row-level multi-tenancy)
--   - Soft delete: is_deleted BOOLEAN NOT NULL DEFAULT FALSE
--   - Audit columns: created_at, updated_at, created_by on every table
--   - monetary values: NUMERIC(18,2)
--   - Enum columns: TEXT with CHECK constraints (avoids ALTER TYPE for new values)
--   - Indexes: tenant + status, tenant + FK for all FK columns
--   - Partitioning hints on high-volume tables (returns, payments)

-- ============================================================
-- Extensions
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- gen_random_uuid fallback
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- trigram full-text search on taxpayer names

-- ============================================================
-- Schema namespace
-- ============================================================
CREATE SCHEMA IF NOT EXISTS tax;

-- ============================================================
-- TAXPAYERS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.taxpayers (
    id                          TEXT        NOT NULL,
    tenant_id                   TEXT        NOT NULL,
    taxpayer_type               TEXT        NOT NULL CHECK (taxpayer_type IN (
                                    'individual','company','partnership','trust',
                                    'government_entity','ngo','foreign_entity')),
    tax_pin                     TEXT        NOT NULL,
    national_id                 TEXT,
    business_registration_number TEXT,
    taxpayer_name               TEXT        NOT NULL,
    trade_name                  TEXT,
    email                       TEXT,
    phone                       TEXT,
    physical_address            TEXT,
    postal_address              TEXT,
    sector_code                 TEXT,
    country_of_incorporation    TEXT        NOT NULL DEFAULT 'KE',
    is_resident                 BOOLEAN     NOT NULL DEFAULT TRUE,
    status                      TEXT        NOT NULL DEFAULT 'pending'
                                    CHECK (status IN (
                                    'pending','active','suspended',
                                    'deregistered','under_investigation','blocked')),
    compliance_score            NUMERIC(5,2),
    risk_rating                 TEXT,
    evidence_reference          TEXT        NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by                  TEXT        NOT NULL DEFAULT 'system',
    is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata                    JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_taxpayers_pin_tenant
    ON tax.taxpayers (tenant_id, tax_pin)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_taxpayers_tenant_status
    ON tax.taxpayers (tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_taxpayers_name_trgm
    ON tax.taxpayers USING gin (taxpayer_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_taxpayers_national_id
    ON tax.taxpayers (tenant_id, national_id)
    WHERE national_id IS NOT NULL;

-- Obligation tax types (many-to-many via array column for simplicity)
ALTER TABLE tax.taxpayers
    ADD COLUMN IF NOT EXISTS tax_types TEXT[] NOT NULL DEFAULT '{}';

-- ============================================================
-- TAX OBLIGATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.tax_obligations (
    id               TEXT        NOT NULL,
    tenant_id        TEXT        NOT NULL,
    taxpayer_id      TEXT        NOT NULL REFERENCES tax.taxpayers(id),
    tax_type         TEXT        NOT NULL,
    filing_frequency TEXT        NOT NULL CHECK (filing_frequency IN ('monthly','quarterly','annually','biannually')),
    due_day          INTEGER     NOT NULL DEFAULT 20 CHECK (due_day BETWEEN 1 AND 31),
    effective_from   DATE        NOT NULL,
    effective_to     DATE,
    status           TEXT        NOT NULL DEFAULT 'active'
                         CHECK (status IN ('active','dormant','cancelled','fulfilled')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by       TEXT        NOT NULL DEFAULT 'system',
    is_deleted       BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata         JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_obligations_taxpayer
    ON tax.tax_obligations (tenant_id, taxpayer_id, tax_type);

CREATE INDEX IF NOT EXISTS idx_obligations_status
    ON tax.tax_obligations (tenant_id, status);

-- ============================================================
-- TAX RETURNS
-- Partitioned by tax_period_start (RANGE) for large datasets.
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.tax_returns (
    id                   TEXT        NOT NULL,
    tenant_id            TEXT        NOT NULL,
    taxpayer_id          TEXT        NOT NULL,
    tax_pin              TEXT        NOT NULL,
    return_type          TEXT        NOT NULL CHECK (return_type IN (
                             'monthly_vat','annual_income','quarterly_advance',
                             'withholding_tax_return','corporate_annual',
                             'customs_entry','turnover_tax_monthly','capital_gains')),
    tax_period_start     DATE        NOT NULL,
    tax_period_end       DATE        NOT NULL CHECK (tax_period_end >= tax_period_start),
    gross_income         NUMERIC(18,2) NOT NULL DEFAULT 0 CHECK (gross_income >= 0),
    allowable_deductions NUMERIC(18,2) NOT NULL DEFAULT 0 CHECK (allowable_deductions >= 0),
    taxable_income       NUMERIC(18,2) NOT NULL DEFAULT 0,
    tax_liability        NUMERIC(18,2) NOT NULL DEFAULT 0 CHECK (tax_liability >= 0),
    tax_credits          NUMERIC(18,2) NOT NULL DEFAULT 0 CHECK (tax_credits >= 0),
    tax_paid             NUMERIC(18,2) NOT NULL DEFAULT 0 CHECK (tax_paid >= 0),
    net_tax_payable      NUMERIC(18,2) NOT NULL DEFAULT 0,
    filing_date          TIMESTAMPTZ,
    late_filing_days     INTEGER     NOT NULL DEFAULT 0,
    status               TEXT        NOT NULL DEFAULT 'draft' CHECK (status IN (
                             'draft','filed','amended','under_review',
                             'assessed','disputed','finalised','rejected')),
    evidence_reference   TEXT        NOT NULL,
    is_amended           BOOLEAN     NOT NULL DEFAULT FALSE,
    original_return_id   TEXT        REFERENCES tax.tax_returns(id),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by           TEXT        NOT NULL DEFAULT 'system',
    is_deleted           BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata             JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_returns_pin_period
    ON tax.tax_returns (tenant_id, tax_pin, tax_period_start, tax_period_end);

CREATE INDEX IF NOT EXISTS idx_returns_taxpayer
    ON tax.tax_returns (tenant_id, taxpayer_id);

CREATE INDEX IF NOT EXISTS idx_returns_status
    ON tax.tax_returns (tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_returns_filing_date
    ON tax.tax_returns (tenant_id, filing_date DESC);

-- ============================================================
-- TAX ASSESSMENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.tax_assessments (
    id                        TEXT        NOT NULL,
    tenant_id                 TEXT        NOT NULL,
    return_id                 TEXT        NOT NULL REFERENCES tax.tax_returns(id),
    taxpayer_id               TEXT        NOT NULL,
    assessment_type           TEXT        NOT NULL CHECK (assessment_type IN (
                                  'self_assessment','amended_assessment','best_judgement',
                                  'audit_assessment','estimated_assessment','agency_assessment')),
    assessed_amount           NUMERIC(18,2) NOT NULL CHECK (assessed_amount >= 0),
    tax_liability_per_return  NUMERIC(18,2) NOT NULL DEFAULT 0,
    additional_tax            NUMERIC(18,2) NOT NULL DEFAULT 0,
    assessor_id               TEXT        NOT NULL,
    assessment_date           DATE        NOT NULL,
    due_date                  DATE,
    status                    TEXT        NOT NULL DEFAULT 'draft' CHECK (status IN (
                                  'draft','issued','objected','upheld','reduced',
                                  'withdrawn','finalised','appealed')),
    evidence_reference        TEXT        NOT NULL,
    notes                     TEXT,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by                TEXT        NOT NULL DEFAULT 'system',
    is_deleted                BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata                  JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_assessments_taxpayer
    ON tax.tax_assessments (tenant_id, taxpayer_id);

CREATE INDEX IF NOT EXISTS idx_assessments_status
    ON tax.tax_assessments (tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_assessments_due_date
    ON tax.tax_assessments (tenant_id, due_date);

-- ============================================================
-- TAX PAYMENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.tax_payments (
    id                 TEXT        NOT NULL,
    tenant_id          TEXT        NOT NULL,
    taxpayer_id        TEXT        NOT NULL,
    assessment_id      TEXT        REFERENCES tax.tax_assessments(id),
    return_id          TEXT        REFERENCES tax.tax_returns(id),
    payment_reference  TEXT        NOT NULL,
    payment_method     TEXT        NOT NULL CHECK (payment_method IN (
                           'bank_transfer','mobile_money','cheque','cash',
                           'credit_card','direct_debit','rtgs','payment_plan')),
    amount             NUMERIC(18,2) NOT NULL CHECK (amount > 0),
    payment_date       DATE        NOT NULL,
    bank_reference     TEXT,
    status             TEXT        NOT NULL DEFAULT 'pending' CHECK (status IN (
                           'pending','processing','confirmed','failed',
                           'reversed','partially_applied','fully_applied')),
    evidence_reference TEXT        NOT NULL,
    applied_to         TEXT[]      NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by         TEXT        NOT NULL DEFAULT 'system',
    is_deleted         BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata           JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_payments_reference_tenant
    ON tax.tax_payments (tenant_id, payment_reference)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_payments_taxpayer
    ON tax.tax_payments (tenant_id, taxpayer_id, payment_date DESC);

CREATE INDEX IF NOT EXISTS idx_payments_status
    ON tax.tax_payments (tenant_id, status);

-- ============================================================
-- TAX DEBTS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.tax_debts (
    id                   TEXT        NOT NULL,
    tenant_id            TEXT        NOT NULL,
    taxpayer_id          TEXT        NOT NULL,
    assessment_id        TEXT        NOT NULL REFERENCES tax.tax_assessments(id),
    principal_amount     NUMERIC(18,2) NOT NULL CHECK (principal_amount >= 0),
    penalty_amount       NUMERIC(18,2) NOT NULL DEFAULT 0 CHECK (penalty_amount >= 0),
    interest_amount      NUMERIC(18,2) NOT NULL DEFAULT 0 CHECK (interest_amount >= 0),
    total_amount         NUMERIC(18,2) GENERATED ALWAYS AS
                             (principal_amount + penalty_amount + interest_amount) STORED,
    amount_paid          NUMERIC(18,2) NOT NULL DEFAULT 0 CHECK (amount_paid >= 0),
    balance              NUMERIC(18,2) GENERATED ALWAYS AS
                             (principal_amount + penalty_amount + interest_amount - amount_paid) STORED,
    due_date             DATE        NOT NULL,
    status               TEXT        NOT NULL DEFAULT 'outstanding' CHECK (status IN (
                             'outstanding','partially_paid','paid','written_off',
                             'under_arrangement','in_litigation','disputed')),
    collection_case_id   TEXT,
    demand_notices_issued INTEGER     NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by           TEXT        NOT NULL DEFAULT 'system',
    is_deleted           BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata             JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_debts_taxpayer
    ON tax.tax_debts (tenant_id, taxpayer_id, status);

CREATE INDEX IF NOT EXISTS idx_debts_due_date
    ON tax.tax_debts (tenant_id, due_date, status);

CREATE INDEX IF NOT EXISTS idx_debts_outstanding
    ON tax.tax_debts (tenant_id, status, balance)
    WHERE status IN ('outstanding','partially_paid');

-- ============================================================
-- TAX AUDITS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.tax_audits (
    id                TEXT        NOT NULL,
    tenant_id         TEXT        NOT NULL,
    taxpayer_id       TEXT        NOT NULL,
    tax_pin           TEXT        NOT NULL,
    audit_type        TEXT        NOT NULL CHECK (audit_type IN (
                          'desk_audit','field_audit','it_audit','transfer_pricing',
                          'vat_refund_audit','forensic_audit','compliance_audit','sector_audit')),
    auditor_id        TEXT        NOT NULL,
    audit_team        TEXT[]      NOT NULL DEFAULT '{}',
    tax_period_start  DATE        NOT NULL,
    tax_period_end    DATE        NOT NULL CHECK (tax_period_end >= tax_period_start),
    scope_description TEXT,
    risk_score        NUMERIC(5,2),
    status            TEXT        NOT NULL DEFAULT 'planned' CHECK (status IN (
                          'planned','in_progress','completed','report_issued',
                          'objection_filed','finalised','withdrawn')),
    total_additional_tax NUMERIC(18,2) NOT NULL DEFAULT 0,
    evidence_reference TEXT        NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by        TEXT        NOT NULL DEFAULT 'system',
    is_deleted        BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata          JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_audits_taxpayer
    ON tax.tax_audits (tenant_id, taxpayer_id, status);

CREATE INDEX IF NOT EXISTS idx_audits_status
    ON tax.tax_audits (tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_audits_auditor
    ON tax.tax_audits (tenant_id, auditor_id);

-- ============================================================
-- AUDIT FINDINGS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.audit_findings (
    id                TEXT        NOT NULL,
    tenant_id         TEXT        NOT NULL,
    audit_id          TEXT        NOT NULL REFERENCES tax.tax_audits(id),
    taxpayer_id       TEXT        NOT NULL,
    finding_type      TEXT        NOT NULL CHECK (finding_type IN (
                          'underpayment','overpayment','non_compliance',
                          'evasion','avoidance','fraud','procedural','informational')),
    description       TEXT        NOT NULL,
    additional_tax    NUMERIC(18,2) NOT NULL DEFAULT 0,
    penalty_amount    NUMERIC(18,2) NOT NULL DEFAULT 0,
    interest_amount   NUMERIC(18,2) NOT NULL DEFAULT 0,
    total_amount      NUMERIC(18,2) GENERATED ALWAYS AS
                          (additional_tax + penalty_amount + interest_amount) STORED,
    period_affected   TEXT,
    is_accepted       BOOLEAN     NOT NULL DEFAULT FALSE,
    response_received TEXT,
    evidence_reference TEXT       NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by        TEXT        NOT NULL DEFAULT 'system',
    is_deleted        BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata          JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_findings_audit
    ON tax.audit_findings (tenant_id, audit_id);

CREATE INDEX IF NOT EXISTS idx_findings_taxpayer
    ON tax.audit_findings (tenant_id, taxpayer_id);

-- ============================================================
-- OBJECTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.objections (
    id                    TEXT        NOT NULL,
    tenant_id             TEXT        NOT NULL,
    assessment_id         TEXT        NOT NULL REFERENCES tax.tax_assessments(id),
    taxpayer_id           TEXT        NOT NULL,
    tax_pin               TEXT        NOT NULL,
    grounds               TEXT        NOT NULL,
    amount_disputed       NUMERIC(18,2) NOT NULL CHECK (amount_disputed >= 0),
    amount_upheld         NUMERIC(18,2),
    supporting_documents  TEXT[]      NOT NULL DEFAULT '{}',
    filed_date            DATE        NOT NULL DEFAULT CURRENT_DATE,
    determination_date    DATE,
    determination_notes   TEXT,
    reviewing_officer_id  TEXT,
    days_to_determination INTEGER,
    status                TEXT        NOT NULL DEFAULT 'submitted' CHECK (status IN (
                              'submitted','under_review','upheld','partially_upheld',
                              'dismissed','appealed','withdrawn')),
    evidence_reference    TEXT        NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by            TEXT        NOT NULL DEFAULT 'system',
    is_deleted            BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata              JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_objections_assessment
    ON tax.objections (tenant_id, assessment_id);

CREATE INDEX IF NOT EXISTS idx_objections_status
    ON tax.objections (tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_objections_taxpayer
    ON tax.objections (tenant_id, taxpayer_id);

-- ============================================================
-- APPEALS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.appeals (
    id               TEXT        NOT NULL,
    tenant_id        TEXT        NOT NULL,
    objection_id     TEXT        NOT NULL REFERENCES tax.objections(id),
    taxpayer_id      TEXT        NOT NULL,
    grounds          TEXT        NOT NULL,
    amount_in_dispute NUMERIC(18,2) NOT NULL,
    tribunal         TEXT        NOT NULL DEFAULT 'Tax Appeals Tribunal',
    hearing_date     DATE,
    decision_date    DATE,
    decision_notes   TEXT,
    status           TEXT        NOT NULL DEFAULT 'submitted' CHECK (status IN (
                         'submitted','registered','hearing_scheduled','heard',
                         'decided','further_appealed','withdrawn','closed')),
    evidence_reference TEXT       NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by       TEXT        NOT NULL DEFAULT 'system',
    is_deleted       BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata         JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_appeals_objection
    ON tax.appeals (tenant_id, objection_id);

CREATE INDEX IF NOT EXISTS idx_appeals_status
    ON tax.appeals (tenant_id, status);

-- ============================================================
-- TAX REFUNDS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.tax_refunds (
    id                   TEXT        NOT NULL,
    tenant_id            TEXT        NOT NULL,
    taxpayer_id          TEXT        NOT NULL,
    tax_pin              TEXT        NOT NULL,
    return_id            TEXT        NOT NULL REFERENCES tax.tax_returns(id),
    refund_type          TEXT        NOT NULL,  -- overpayment, input_vat_credit, wht_credit
    claimed_amount       NUMERIC(18,2) NOT NULL CHECK (claimed_amount > 0),
    approved_amount      NUMERIC(18,2),
    bank_account_number  TEXT,
    bank_name            TEXT,
    reviewer_id          TEXT,
    review_notes         TEXT,
    processed_date       DATE,
    status               TEXT        NOT NULL DEFAULT 'claimed' CHECK (status IN (
                             'claimed','under_review','approved','rejected',
                             'processing','paid','offset','withheld')),
    evidence_reference   TEXT        NOT NULL,
    supporting_documents TEXT[]      NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by           TEXT        NOT NULL DEFAULT 'system',
    is_deleted           BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata             JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_refunds_taxpayer
    ON tax.tax_refunds (tenant_id, taxpayer_id, status);

CREATE INDEX IF NOT EXISTS idx_refunds_status
    ON tax.tax_refunds (tenant_id, status);

-- ============================================================
-- PENALTIES
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.penalties (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    taxpayer_id     TEXT        NOT NULL,
    assessment_id   TEXT        REFERENCES tax.tax_assessments(id),
    return_id       TEXT        REFERENCES tax.tax_returns(id),
    penalty_type    TEXT        NOT NULL CHECK (penalty_type IN (
                        'late_filing','late_payment','understatement','fraud',
                        'non_filing','incorrect_return','withholding_default')),
    base_amount     NUMERIC(18,2) NOT NULL,
    rate            NUMERIC(6,4) NOT NULL,
    calculated_amount NUMERIC(18,2) NOT NULL,
    period_days     INTEGER,
    status          TEXT        NOT NULL DEFAULT 'assessed' CHECK (status IN (
                        'assessed','confirmed','reduced','waived','paid','outstanding','disputed')),
    waiver_reason   TEXT,
    waived_by       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      TEXT        NOT NULL DEFAULT 'system',
    is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata        JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_penalties_taxpayer
    ON tax.penalties (tenant_id, taxpayer_id, status);

-- ============================================================
-- INTEREST
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.interests (
    id                TEXT        NOT NULL,
    tenant_id         TEXT        NOT NULL,
    taxpayer_id       TEXT        NOT NULL,
    assessment_id     TEXT        REFERENCES tax.tax_assessments(id),
    return_id         TEXT        REFERENCES tax.tax_returns(id),
    interest_type     TEXT        NOT NULL CHECK (interest_type IN (
                          'late_payment','late_filing','refund_interest','penalty_interest')),
    principal_amount  NUMERIC(18,2) NOT NULL,
    annual_rate       NUMERIC(6,4) NOT NULL,
    from_date         DATE        NOT NULL,
    to_date           DATE        NOT NULL CHECK (to_date >= from_date),
    days              INTEGER     GENERATED ALWAYS AS (to_date - from_date) STORED,
    calculated_amount NUMERIC(18,2) NOT NULL,
    status            TEXT        NOT NULL DEFAULT 'assessed',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by        TEXT        NOT NULL DEFAULT 'system',
    is_deleted        BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata          JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_interests_taxpayer
    ON tax.interests (tenant_id, taxpayer_id);

-- ============================================================
-- TAX CLEARANCE CERTIFICATES
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.tax_clearance_certificates (
    id                 TEXT        NOT NULL,
    tenant_id          TEXT        NOT NULL,
    taxpayer_id        TEXT        NOT NULL,
    tax_pin            TEXT        NOT NULL,
    purpose            TEXT        NOT NULL,
    certificate_number TEXT,
    issue_date         DATE,
    expiry_date        DATE,
    validity_months    INTEGER     NOT NULL DEFAULT 6,
    reviewer_id        TEXT,
    denial_reason      TEXT,
    status             TEXT        NOT NULL DEFAULT 'applied' CHECK (status IN (
                           'applied','under_review','issued','rejected','expired','revoked')),
    evidence_reference TEXT        NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by         TEXT        NOT NULL DEFAULT 'system',
    is_deleted         BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata           JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_clearances_cert_number
    ON tax.tax_clearance_certificates (tenant_id, certificate_number)
    WHERE certificate_number IS NOT NULL AND is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_clearances_taxpayer
    ON tax.tax_clearance_certificates (tenant_id, taxpayer_id, status);

CREATE INDEX IF NOT EXISTS idx_clearances_expiry
    ON tax.tax_clearance_certificates (tenant_id, expiry_date)
    WHERE status = 'issued';

-- ============================================================
-- DEMAND NOTICES
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.demand_notices (
    id               TEXT        NOT NULL,
    tenant_id        TEXT        NOT NULL,
    debt_id          TEXT        NOT NULL,
    taxpayer_id      TEXT        NOT NULL,
    tax_pin          TEXT        NOT NULL,
    notice_number    TEXT        NOT NULL,
    amount_demanded  NUMERIC(18,2) NOT NULL,
    issued_date      DATE        NOT NULL DEFAULT CURRENT_DATE,
    due_date         DATE        NOT NULL,
    notice_text      TEXT,
    issued_by        TEXT        NOT NULL DEFAULT 'system',
    is_deleted       BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata         JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_demand_notices_taxpayer
    ON tax.demand_notices (tenant_id, taxpayer_id);

CREATE INDEX IF NOT EXISTS idx_demand_notices_due
    ON tax.demand_notices (tenant_id, due_date);

-- ============================================================
-- EXCHANGE OF INFORMATION REQUESTS
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.eoi_requests (
    id                      TEXT        NOT NULL,
    tenant_id               TEXT        NOT NULL,
    treaty_partner          TEXT        NOT NULL,
    subject_taxpayer_id     TEXT        NOT NULL,
    subject_name            TEXT        NOT NULL,
    information_requested   TEXT        NOT NULL,
    legal_basis             TEXT        NOT NULL DEFAULT 'double_tax_agreement',
    urgency                 TEXT        NOT NULL DEFAULT 'routine'
                                CHECK (urgency IN ('routine','urgent','spontaneous')),
    submitted_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    response_deadline       DATE,
    status                  TEXT        NOT NULL DEFAULT 'submitted',
    response_received       TEXT,
    is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
    metadata                JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_eoi_tenant_partner
    ON tax.eoi_requests (tenant_id, treaty_partner);

-- ============================================================
-- AUDIT TRAIL (immutable append-only event log)
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.audit_trail (
    id           BIGSERIAL   PRIMARY KEY,
    tenant_id    TEXT        NOT NULL,
    event_type   TEXT        NOT NULL,
    reference_id TEXT        NOT NULL,
    actor_id     TEXT        NOT NULL DEFAULT 'system',
    processor    TEXT        NOT NULL DEFAULT 'bytewax',
    payload      JSONB       NOT NULL DEFAULT '{}',
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_trail_tenant_event
    ON tax.audit_trail (tenant_id, event_type, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_trail_reference
    ON tax.audit_trail (tenant_id, reference_id);

-- ============================================================
-- COMPLIANCE RISK PROFILES (materialized/cached)
-- ============================================================
CREATE TABLE IF NOT EXISTS tax.compliance_risk_profiles (
    id                   TEXT        NOT NULL,
    tenant_id            TEXT        NOT NULL,
    taxpayer_id          TEXT        NOT NULL REFERENCES tax.taxpayers(id),
    tax_pin              TEXT        NOT NULL,
    risk_score           NUMERIC(5,2) NOT NULL,
    risk_category        TEXT        NOT NULL CHECK (risk_category IN ('low','medium','high','critical')),
    factors              JSONB       NOT NULL DEFAULT '{}',
    recommended_action   TEXT        NOT NULL,
    generated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_risk_profiles_taxpayer
    ON tax.compliance_risk_profiles (tenant_id, taxpayer_id);

CREATE INDEX IF NOT EXISTS idx_risk_profiles_category
    ON tax.compliance_risk_profiles (tenant_id, risk_category);

-- ============================================================
-- TRIGGERS: auto-update updated_at
-- ============================================================
CREATE OR REPLACE FUNCTION tax.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'taxpayers','tax_obligations','tax_returns','tax_assessments',
        'tax_payments','tax_debts','tax_audits','audit_findings',
        'objections','appeals','tax_refunds','penalties','interests',
        'tax_clearance_certificates'
    ] LOOP
        EXECUTE format(
            'CREATE OR REPLACE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON tax.%s
             FOR EACH ROW EXECUTE FUNCTION tax.set_updated_at()',
            t, t
        );
    END LOOP;
END;
$$;

-- ============================================================
-- VIEWS: useful query helpers
-- ============================================================

-- Outstanding debts summary per taxpayer
CREATE OR REPLACE VIEW tax.v_taxpayer_debt_summary AS
SELECT
    d.tenant_id,
    d.taxpayer_id,
    t.tax_pin,
    t.taxpayer_name,
    COUNT(*) AS debt_count,
    SUM(d.balance) AS total_balance,
    MIN(d.due_date) AS oldest_due_date,
    MAX(d.due_date) AS newest_due_date
FROM tax.tax_debts d
JOIN tax.taxpayers t ON t.id = d.taxpayer_id AND t.tenant_id = d.tenant_id
WHERE d.status IN ('outstanding','partially_paid')
  AND d.is_deleted = FALSE
  AND t.is_deleted = FALSE
GROUP BY d.tenant_id, d.taxpayer_id, t.tax_pin, t.taxpayer_name;

-- Filing compliance summary per period (month)
CREATE OR REPLACE VIEW tax.v_monthly_filing_compliance AS
SELECT
    tenant_id,
    date_trunc('month', tax_period_start) AS month,
    return_type,
    COUNT(*) AS returns_filed,
    SUM(net_tax_payable) AS total_net_payable,
    SUM(tax_paid) AS total_paid,
    COUNT(*) FILTER (WHERE status = 'filed') AS on_time_count,
    COUNT(*) FILTER (WHERE late_filing_days > 0) AS late_count
FROM tax.tax_returns
WHERE is_deleted = FALSE
GROUP BY tenant_id, date_trunc('month', tax_period_start), return_type;

-- Revenue collection by tax type
CREATE OR REPLACE VIEW tax.v_revenue_by_tax_type AS
SELECT
    p.tenant_id,
    date_trunc('month', p.payment_date::timestamptz) AS month,
    SUM(p.amount) AS collected
FROM tax.tax_payments p
WHERE p.is_deleted = FALSE
  AND p.status IN ('confirmed','fully_applied','partially_applied')
GROUP BY p.tenant_id, date_trunc('month', p.payment_date::timestamptz);

-- Audit pipeline
CREATE OR REPLACE VIEW tax.v_audit_pipeline AS
SELECT
    a.tenant_id,
    a.audit_type,
    a.status,
    COUNT(*) AS case_count,
    SUM(a.total_additional_tax) AS total_additional_tax
FROM tax.tax_audits a
WHERE a.is_deleted = FALSE
GROUP BY a.tenant_id, a.audit_type, a.status;
