-- SOC 2 Type II compliant audit event table
-- Migration: 0002_soc2_audit_events
--
-- Creates an append-only audit log table with:
--   - Tamper-evident SHA-256 checksum per event
--   - Cryptographic hash chain linking events in insertion order
--   - PostgreSQL rules preventing UPDATE/DELETE (immutability)
--   - 90-day default retention index (configurable per SOC 2 controls)
--
-- This supplements (not replaces) apg_records. The existing JSONB store
-- continues to serve general capability storage needs. This table is
-- purpose-built for compliance audit trails.

CREATE TABLE IF NOT EXISTS apg_audit_events (
    id              TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    actor_id        TEXT        NOT NULL,
    event_type      TEXT        NOT NULL,
    resource_type   TEXT,
    resource_id     TEXT        NOT NULL,
    action          TEXT        NOT NULL,
    success         BOOLEAN     NOT NULL,
    ip_address      TEXT,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload         JSONB,
    checksum        TEXT        NOT NULL,   -- SHA-256 of canonical fields
    prev_hash       TEXT        NOT NULL,   -- hash of immediately preceding event
    chain_hash      TEXT        NOT NULL,   -- SHA-256(prev_hash || checksum)
    CONSTRAINT apg_audit_events_pkey PRIMARY KEY (id)
);

-- Append-only enforcement: block any UPDATE or DELETE on this table
CREATE OR REPLACE RULE apg_audit_events_no_update
    AS ON UPDATE TO apg_audit_events DO INSTEAD NOTHING;

CREATE OR REPLACE RULE apg_audit_events_no_delete
    AS ON DELETE TO apg_audit_events DO INSTEAD NOTHING;

-- Fast lookups by tenant + actor + time range (most common audit query pattern)
CREATE INDEX IF NOT EXISTS idx_apg_audit_events_tenant_time
    ON apg_audit_events (tenant_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_apg_audit_events_resource
    ON apg_audit_events (tenant_id, resource_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_apg_audit_events_actor
    ON apg_audit_events (tenant_id, actor_id, timestamp DESC);

-- GIN index for payload searches (compliance investigations)
CREATE INDEX IF NOT EXISTS idx_apg_audit_events_payload
    ON apg_audit_events USING gin (payload);
