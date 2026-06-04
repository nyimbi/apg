-- APG Workflow Automation — PostgreSQL schema
-- Run: psql $DATABASE_URL < database/schema.sql
--
-- Uses the shared apg_records JSONB store.
-- For production, consider migrating to capability-specific normalized tables.

CREATE TABLE IF NOT EXISTS apg_records (
    id          TEXT        NOT NULL,
    collection  TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL DEFAULT 'default',
    data        JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, id)
);

CREATE INDEX IF NOT EXISTS idx_apg_wfa_tenant
    ON apg_records (collection, tenant_id);

CREATE INDEX IF NOT EXISTS idx_apg_wfa_data
    ON apg_records USING gin (data);
