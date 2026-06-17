-- APG shared JSONB record store — apply once per database.
-- All APG capabilities use this single table, distinguished by 'collection'.
--
-- Run: psql $DATABASE_URL -f schema.sql
-- Or:  python -c "from capabilities.common.db import SCHEMA_SQL; print(SCHEMA_SQL)" | psql $DATABASE_URL

CREATE TABLE IF NOT EXISTS apg_records (
    id          TEXT        NOT NULL,
    collection  TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL DEFAULT 'default',
    data        JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, id)
);

CREATE INDEX IF NOT EXISTS idx_apg_records_tenant
    ON apg_records (collection, tenant_id);

CREATE INDEX IF NOT EXISTS idx_apg_records_data_gin
    ON apg_records USING gin (data);

CREATE OR REPLACE FUNCTION apg_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at := now(); RETURN NEW; END;
$$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_apg_records_updated_at'
    ) THEN
        CREATE TRIGGER trg_apg_records_updated_at
            BEFORE UPDATE ON apg_records
            FOR EACH ROW EXECUTE FUNCTION apg_set_updated_at();
    END IF;
END $$;
