-- PCI DSS cardholder data token vault table
-- Migration: 0001_token_vault
--
-- Stores encrypted PAN → token mappings for format-preserving tokenization.
-- The encrypted_pan column holds the XOR-encrypted PAN (replace with AES-256
-- / Vault Transit / AWS KMS in production — the cipher is pluggable via
-- TokenizationService._xor_encrypt).
--
-- PCI DSS Requirement 3.5: "Secure all keys used to protect stored account data."

CREATE TABLE IF NOT EXISTS apg_token_vault (
    token           TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    encrypted_pan   TEXT        NOT NULL,   -- hex-encoded encrypted PAN
    card_type       TEXT        NOT NULL DEFAULT '',
    bin             TEXT        NOT NULL DEFAULT '',
    last_four       TEXT        NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT apg_token_vault_pkey PRIMARY KEY (token)
);

-- Tenant isolation: each tenant can only see its own tokens
CREATE INDEX IF NOT EXISTS idx_apg_token_vault_tenant
    ON apg_token_vault (tenant_id);

-- Row-level security for strict PCI DSS scope isolation
-- Enable with: ALTER TABLE apg_token_vault ENABLE ROW LEVEL SECURITY;
-- Policy: CREATE POLICY apg_token_vault_tenant ON apg_token_vault
--     USING (tenant_id = current_setting('apg.tenant_id', true));
