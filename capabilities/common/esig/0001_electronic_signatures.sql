-- FDA 21 CFR Part 11 / GxP electronic signature table
-- Migration: 0001_electronic_signatures
--
-- Append-only store for qualified electronic signatures.
-- Immutability enforced via PostgreSQL rules (same pattern as apg_audit_events).
-- 10-year retention required by 21 CFR Part 11 for pharma records.

CREATE TABLE IF NOT EXISTS apg_electronic_signatures (
    id                    TEXT        NOT NULL,
    tenant_id             TEXT        NOT NULL,
    document_id           TEXT        NOT NULL,
    signer_id             TEXT        NOT NULL,
    signer_display_name   TEXT        NOT NULL DEFAULT '',
    meaning               TEXT        NOT NULL,          -- 21 CFR §11.50(a)(3)
    timestamp             TIMESTAMPTZ NOT NULL DEFAULT now(),  -- §11.50(a)(2)
    document_hash         TEXT        NOT NULL DEFAULT '',
    signature_hash        TEXT        NOT NULL,          -- SHA-256 binding
    additional_context    JSONB,
    is_valid              BOOLEAN     NOT NULL DEFAULT TRUE,
    CONSTRAINT apg_esig_pkey PRIMARY KEY (id)
);

-- Append-only: no modification after signing (21 CFR Part 11 tamper-evidence)
CREATE OR REPLACE RULE apg_esig_no_update
    AS ON UPDATE TO apg_electronic_signatures DO INSTEAD NOTHING;

CREATE OR REPLACE RULE apg_esig_no_delete
    AS ON DELETE TO apg_electronic_signatures DO INSTEAD NOTHING;

-- Lookup by document (for approval chain queries)
CREATE INDEX IF NOT EXISTS idx_apg_esig_document
    ON apg_electronic_signatures (tenant_id, document_id, timestamp DESC);

-- Lookup by signer (for audit of who signed what)
CREATE INDEX IF NOT EXISTS idx_apg_esig_signer
    ON apg_electronic_signatures (tenant_id, signer_id, timestamp DESC);
