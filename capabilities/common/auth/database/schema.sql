-- =============================================================================
-- APG Authentication & RBAC — PostgreSQL Schema
-- © 2025 Datacraft — www.datacraft.co.ke
-- Run: psql $DATABASE_URL < database/schema.sql
-- =============================================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "citext";

-- Shared audit trigger function
CREATE OR REPLACE FUNCTION auth_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
	NEW.updated_at = now();
	RETURN NEW;
END;
$$;

-- =============================================================================
-- USERS
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_users (
	id                      TEXT            NOT NULL,
	tenant_id               TEXT            NOT NULL,
	email                   CITEXT          NOT NULL,
	display_name            TEXT            NOT NULL,
	password_hash           TEXT,
	status                  TEXT            NOT NULL DEFAULT 'active'
		CHECK (status IN ('active','inactive','locked','suspended',
		                  'pending_verification','password_reset_required')),
	mfa_enabled             BOOLEAN         NOT NULL DEFAULT FALSE,
	failed_login_count      INT             NOT NULL DEFAULT 0,
	locked_until            TIMESTAMPTZ,
	last_login_at           TIMESTAMPTZ,
	password_changed_at     TIMESTAMPTZ,
	metadata                JSONB           NOT NULL DEFAULT '{}',
	created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
	created_by              TEXT            NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN         NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_auth_users_tenant_email
	ON auth_users (tenant_id, email) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_users_tenant
	ON auth_users (tenant_id) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_users_status
	ON auth_users (tenant_id, status) WHERE is_deleted = FALSE;

CREATE TRIGGER trg_auth_users_updated_at
	BEFORE UPDATE ON auth_users
	FOR EACH ROW EXECUTE FUNCTION auth_set_updated_at();

-- Password history (prevent reuse)
CREATE TABLE IF NOT EXISTS auth_password_history (
	id              TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id       TEXT        NOT NULL,
	user_id         TEXT        NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
	password_hash   TEXT        NOT NULL,
	changed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_pwd_history_user
	ON auth_password_history (user_id, changed_at DESC);

-- =============================================================================
-- GROUPS
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_groups (
	id              TEXT        NOT NULL,
	tenant_id       TEXT        NOT NULL,
	name            TEXT        NOT NULL,
	description     TEXT        NOT NULL DEFAULT '',
	parent_group_id TEXT        REFERENCES auth_groups(id) ON DELETE SET NULL,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_auth_groups_tenant_name
	ON auth_groups (tenant_id, name) WHERE is_deleted = FALSE;

CREATE TABLE IF NOT EXISTS auth_group_members (
	id          TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id   TEXT        NOT NULL,
	group_id    TEXT        NOT NULL REFERENCES auth_groups(id) ON DELETE CASCADE,
	user_id     TEXT        NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
	added_by    TEXT        NOT NULL DEFAULT 'system',
	added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by  TEXT        NOT NULL DEFAULT 'system',
	is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id),
	UNIQUE (group_id, user_id)
);

-- =============================================================================
-- PERMISSIONS
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_permissions (
	id          TEXT        NOT NULL,
	tenant_id   TEXT        NOT NULL,
	name        TEXT        NOT NULL,        -- e.g. 'auth:manage_users'
	resource    TEXT        NOT NULL,        -- e.g. 'users'
	action      TEXT        NOT NULL,        -- e.g. 'manage'
	effect      TEXT        NOT NULL DEFAULT 'allow'
		CHECK (effect IN ('allow','deny')),
	description TEXT        NOT NULL DEFAULT '',
	created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by  TEXT        NOT NULL DEFAULT 'system',
	is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_auth_perms_tenant_name
	ON auth_permissions (tenant_id, name) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_perms_resource_action
	ON auth_permissions (tenant_id, resource, action) WHERE is_deleted = FALSE;

-- =============================================================================
-- ROLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_roles (
	id          TEXT        NOT NULL,
	tenant_id   TEXT        NOT NULL,
	name        TEXT        NOT NULL,
	tier        TEXT        NOT NULL DEFAULT 'standard'
		CHECK (tier IN ('standard','elevated','privileged','admin','super_admin')),
	status      TEXT        NOT NULL DEFAULT 'active'
		CHECK (status IN ('active','inactive','deprecated')),
	description TEXT        NOT NULL DEFAULT '',
	created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by  TEXT        NOT NULL DEFAULT 'system',
	is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_auth_roles_tenant_name
	ON auth_roles (tenant_id, name) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_roles_tier
	ON auth_roles (tenant_id, tier) WHERE is_deleted = FALSE;

CREATE TABLE IF NOT EXISTS auth_role_permissions (
	id              TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id       TEXT        NOT NULL,
	role_id         TEXT        NOT NULL REFERENCES auth_roles(id) ON DELETE CASCADE,
	permission_id   TEXT        NOT NULL REFERENCES auth_permissions(id) ON DELETE CASCADE,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id),
	UNIQUE (role_id, permission_id)
);

CREATE TABLE IF NOT EXISTS auth_role_assignments (
	id              TEXT        NOT NULL,
	tenant_id       TEXT        NOT NULL,
	user_id         TEXT        NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
	role_id         TEXT        NOT NULL REFERENCES auth_roles(id) ON DELETE CASCADE,
	assigned_by     TEXT        NOT NULL,
	expires_at      TIMESTAMPTZ,
	justification   TEXT        NOT NULL DEFAULT '',
	is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_role_asgn_user
	ON auth_role_assignments (tenant_id, user_id) WHERE is_active = TRUE AND is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_role_asgn_role
	ON auth_role_assignments (tenant_id, role_id) WHERE is_active = TRUE AND is_deleted = FALSE;
-- Prevent duplicate active assignment of same role to same user
CREATE UNIQUE INDEX IF NOT EXISTS uidx_auth_role_asgn_active
	ON auth_role_assignments (tenant_id, user_id, role_id)
	WHERE is_active = TRUE AND is_deleted = FALSE;

-- =============================================================================
-- ABAC POLICIES
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_policies (
	id                      TEXT        NOT NULL,
	tenant_id               TEXT        NOT NULL,
	name                    TEXT        NOT NULL,
	effect                  TEXT        NOT NULL DEFAULT 'allow'
		CHECK (effect IN ('allow','deny')),
	priority                INT         NOT NULL DEFAULT 100,
	subject_conditions      JSONB       NOT NULL DEFAULT '[]',
	resource_conditions     JSONB       NOT NULL DEFAULT '[]',
	action_conditions       JSONB       NOT NULL DEFAULT '[]',
	environment_conditions  JSONB       NOT NULL DEFAULT '[]',
	description             TEXT        NOT NULL DEFAULT '',
	is_active               BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_auth_policies_tenant_name
	ON auth_policies (tenant_id, name) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_policies_priority
	ON auth_policies (tenant_id, priority) WHERE is_active = TRUE AND is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_policies_conditions
	ON auth_policies USING gin (subject_conditions, resource_conditions);

-- =============================================================================
-- SESSIONS
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_sessions (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	user_id             TEXT        NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
	device_id           TEXT        NOT NULL,
	ip_address          INET        NOT NULL,
	user_agent          TEXT        NOT NULL DEFAULT '',
	auth_method         TEXT        NOT NULL DEFAULT 'password',
	session_type        TEXT        NOT NULL DEFAULT 'interactive'
		CHECK (session_type IN ('interactive','service','delegated','impersonated')),
	status              TEXT        NOT NULL DEFAULT 'active'
		CHECK (status IN ('active','expired','revoked','invalidated')),
	mfa_verified        BOOLEAN     NOT NULL DEFAULT FALSE,
	step_up_completed   BOOLEAN     NOT NULL DEFAULT FALSE,
	risk_level          TEXT        NOT NULL DEFAULT 'low'
		CHECK (risk_level IN ('low','medium','high','critical')),
	trust_score         FLOAT       NOT NULL DEFAULT 1.0,
	expires_at          TIMESTAMPTZ,
	last_activity_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
	impersonator_id     TEXT        REFERENCES auth_users(id),
	delegation_id       TEXT,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
	ON auth_sessions (tenant_id, user_id, status) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry
	ON auth_sessions (expires_at) WHERE status = 'active';
-- Partitioning hint: in high-volume production, partition by created_at (monthly)

-- =============================================================================
-- MFA DEVICES
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_mfa_devices (
	id              TEXT        NOT NULL,
	tenant_id       TEXT        NOT NULL,
	user_id         TEXT        NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
	method          TEXT        NOT NULL
		CHECK (method IN ('totp','sms','email','hardware_key','passkey','backup_code')),
	device_name     TEXT        NOT NULL DEFAULT '',
	-- TOTP secret stored encrypted (application-level encryption)
	totp_secret_enc TEXT,
	-- SMS/email destination (hashed for privacy)
	phone_hash      TEXT,
	status          TEXT        NOT NULL DEFAULT 'pending_verification'
		CHECK (status IN ('active','pending_verification','revoked')),
	last_used_at    TIMESTAMPTZ,
	use_count       INT         NOT NULL DEFAULT 0,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_mfa_user
	ON auth_mfa_devices (tenant_id, user_id, method)
	WHERE status = 'active' AND is_deleted = FALSE;

-- Backup codes (hashed, one-time)
CREATE TABLE IF NOT EXISTS auth_backup_codes (
	id          TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id   TEXT        NOT NULL,
	user_id     TEXT        NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
	code_hash   TEXT        NOT NULL,
	used        BOOLEAN     NOT NULL DEFAULT FALSE,
	used_at     TIMESTAMPTZ,
	created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_backup_codes_user
	ON auth_backup_codes (user_id) WHERE used = FALSE;

-- =============================================================================
-- LOGIN ATTEMPTS (audit + brute-force detection)
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_login_attempts (
	id              TEXT        NOT NULL,
	tenant_id       TEXT        NOT NULL,
	user_id         TEXT        REFERENCES auth_users(id) ON DELETE SET NULL,
	email           CITEXT      NOT NULL,
	ip_address      INET        NOT NULL,
	user_agent      TEXT        NOT NULL DEFAULT '',
	outcome         TEXT        NOT NULL
		CHECK (outcome IN ('success','failed_credentials','failed_mfa',
		                   'blocked_lockout','blocked_ip','blocked_suspicious')),
	risk_score      FLOAT       NOT NULL DEFAULT 0.0,
	risk_factors    JSONB       NOT NULL DEFAULT '[]',
	geo_country     TEXT        NOT NULL DEFAULT '',
	geo_city        TEXT        NOT NULL DEFAULT '',
	device_id       TEXT,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_attempts_ip_time
	ON auth_login_attempts (ip_address, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_email_time
	ON auth_login_attempts (tenant_id, email, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_user_failed
	ON auth_login_attempts (user_id, outcome, created_at DESC)
	WHERE outcome != 'success';
-- Partition hint: partition by created_at (weekly) for high-traffic tenants

-- =============================================================================
-- PASSWORD POLICIES
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_password_policies (
	id                          TEXT        NOT NULL,
	tenant_id                   TEXT        NOT NULL,
	name                        TEXT        NOT NULL,
	min_length                  INT         NOT NULL DEFAULT 12,
	require_uppercase           BOOLEAN     NOT NULL DEFAULT TRUE,
	require_lowercase           BOOLEAN     NOT NULL DEFAULT TRUE,
	require_digits              BOOLEAN     NOT NULL DEFAULT TRUE,
	require_special             BOOLEAN     NOT NULL DEFAULT TRUE,
	max_age_days                INT         NOT NULL DEFAULT 90,
	history_count               INT         NOT NULL DEFAULT 10,
	max_failed_attempts         INT         NOT NULL DEFAULT 5,
	lockout_duration_minutes    INT         NOT NULL DEFAULT 30,
	breach_check_enabled        BOOLEAN     NOT NULL DEFAULT TRUE,
	is_default                  BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL DEFAULT 'system',
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_auth_pwd_policy_tenant_default
	ON auth_password_policies (tenant_id) WHERE is_default = TRUE AND is_deleted = FALSE;

-- =============================================================================
-- IP ALLOWLIST
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_ip_allowlist (
	id          TEXT        NOT NULL,
	tenant_id   TEXT        NOT NULL,
	cidr        CIDR        NOT NULL,
	label       TEXT        NOT NULL DEFAULT '',
	applies_to  TEXT        NOT NULL DEFAULT 'all',
	is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by  TEXT        NOT NULL DEFAULT 'system',
	is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_ip_allowlist_tenant
	ON auth_ip_allowlist (tenant_id) WHERE is_active = TRUE AND is_deleted = FALSE;

-- =============================================================================
-- OAUTH2 CLIENTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_oauth_clients (
	id                      TEXT        NOT NULL,
	tenant_id               TEXT        NOT NULL,
	name                    TEXT        NOT NULL,
	client_id               TEXT        NOT NULL UNIQUE,
	client_secret_hash      TEXT,           -- NULL for public clients
	redirect_uris           JSONB       NOT NULL DEFAULT '[]',
	allowed_grant_types     JSONB       NOT NULL DEFAULT '[]',
	allowed_scopes          JSONB       NOT NULL DEFAULT '[]',
	is_public               BOOLEAN     NOT NULL DEFAULT FALSE,
	is_active               BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

-- Authorization codes (short-lived, single-use)
CREATE TABLE IF NOT EXISTS auth_oauth_codes (
	id                      TEXT        NOT NULL DEFAULT gen_random_uuid()::text,
	tenant_id               TEXT        NOT NULL,
	client_id               TEXT        NOT NULL,
	user_id                 TEXT        REFERENCES auth_users(id) ON DELETE CASCADE,
	code_hash               TEXT        NOT NULL UNIQUE,
	redirect_uri            TEXT        NOT NULL,
	scope                   TEXT        NOT NULL,
	state                   TEXT        NOT NULL DEFAULT '',
	code_challenge          TEXT,
	code_challenge_method   TEXT        NOT NULL DEFAULT 'S256',
	used                    BOOLEAN     NOT NULL DEFAULT FALSE,
	expires_at              TIMESTAMPTZ NOT NULL,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_oauth_codes_expiry
	ON auth_oauth_codes (expires_at) WHERE used = FALSE;

-- Issued tokens (access + refresh)
CREATE TABLE IF NOT EXISTS auth_oauth_tokens (
	id                      TEXT        NOT NULL,
	tenant_id               TEXT        NOT NULL,
	client_id               TEXT        NOT NULL,
	user_id                 TEXT        REFERENCES auth_users(id) ON DELETE CASCADE,
	access_token_hash       TEXT        NOT NULL,
	refresh_token_hash      TEXT,
	scopes                  JSONB       NOT NULL DEFAULT '[]',
	status                  TEXT        NOT NULL DEFAULT 'active'
		CHECK (status IN ('active','expired','revoked')),
	expires_at              TIMESTAMPTZ NOT NULL,
	token_type              TEXT        NOT NULL DEFAULT 'Bearer',
	-- refresh token rotation: track rotated JTIs to detect reuse
	rotated_from_jti        TEXT,
	family_id               TEXT,           -- all tokens in a refresh-token chain
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_tokens_user
	ON auth_oauth_tokens (tenant_id, user_id, status) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_tokens_family
	ON auth_oauth_tokens (family_id) WHERE status = 'active';

-- =============================================================================
-- API KEYS
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_api_keys (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	user_id             TEXT        REFERENCES auth_users(id) ON DELETE CASCADE,
	service_account_id  TEXT,
	name                TEXT        NOT NULL,
	key_prefix          TEXT        NOT NULL,   -- first 8 chars for display
	key_hash            TEXT        NOT NULL,
	key_salt            TEXT        NOT NULL,
	scopes              JSONB       NOT NULL DEFAULT '[]',
	status              TEXT        NOT NULL DEFAULT 'active'
		CHECK (status IN ('active','inactive','expired','revoked')),
	expires_at          TIMESTAMPTZ,
	last_used_at        TIMESTAMPTZ,
	use_count           INT         NOT NULL DEFAULT 0,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_api_keys_user
	ON auth_api_keys (tenant_id, user_id, status) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_api_keys_prefix
	ON auth_api_keys (key_prefix) WHERE status = 'active' AND is_deleted = FALSE;

-- =============================================================================
-- SERVICE ACCOUNTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_service_accounts (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	name                TEXT        NOT NULL,
	description         TEXT        NOT NULL DEFAULT '',
	status              TEXT        NOT NULL DEFAULT 'active'
		CHECK (status IN ('active','inactive','rotating','decommissioned')),
	role_ids            JSONB       NOT NULL DEFAULT '[]',
	key_rotation_days   INT         NOT NULL DEFAULT 90,
	last_rotated_at     TIMESTAMPTZ,
	next_rotation_at    TIMESTAMPTZ,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_auth_svc_accts_tenant_name
	ON auth_service_accounts (tenant_id, name) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_svc_accts_rotation
	ON auth_service_accounts (next_rotation_at) WHERE status = 'active';

-- =============================================================================
-- DELEGATIONS
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_delegations (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	delegator_id        TEXT        NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
	delegate_id         TEXT        NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
	permission_ids      JSONB       NOT NULL DEFAULT '[]',
	status              TEXT        NOT NULL DEFAULT 'active'
		CHECK (status IN ('active','expired','revoked')),
	expires_at          TIMESTAMPTZ NOT NULL,
	justification       TEXT        NOT NULL DEFAULT '',
	requires_mfa        BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_delegations_delegate
	ON auth_delegations (tenant_id, delegate_id, status) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_auth_delegations_expiry
	ON auth_delegations (expires_at) WHERE status = 'active';

-- =============================================================================
-- AUDIT EVENTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_audit_events (
	id              TEXT        NOT NULL,
	tenant_id       TEXT        NOT NULL,
	event_type      TEXT        NOT NULL,
	actor_id        TEXT        NOT NULL,
	actor_ip        INET,
	target_id       TEXT        NOT NULL DEFAULT '',
	target_type     TEXT        NOT NULL DEFAULT '',
	outcome         TEXT        NOT NULL CHECK (outcome IN ('success','failure','denied')),
	risk_level      TEXT        NOT NULL DEFAULT 'low'
		CHECK (risk_level IN ('low','medium','high','critical')),
	session_id      TEXT,
	details         JSONB       NOT NULL DEFAULT '{}',
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_auth_audit_tenant_time
	ON auth_audit_events (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_audit_actor
	ON auth_audit_events (tenant_id, actor_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_audit_type
	ON auth_audit_events (tenant_id, event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_audit_details
	ON auth_audit_events USING gin (details);
-- Partition hint: partition by created_at (monthly) — audit tables grow unbounded

-- =============================================================================
-- JWT BLACKLIST
-- =============================================================================

CREATE TABLE IF NOT EXISTS auth_token_blacklist (
	jti         TEXT        NOT NULL,
	tenant_id   TEXT        NOT NULL,
	reason      TEXT        NOT NULL DEFAULT 'explicit_revocation',
	expires_at  TIMESTAMPTZ NOT NULL,   -- auto-cleanup when past expiry
	created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	PRIMARY KEY (jti)
);

CREATE INDEX IF NOT EXISTS idx_auth_blacklist_expiry
	ON auth_token_blacklist (expires_at);

-- =============================================================================
-- VIEWS
-- =============================================================================

CREATE OR REPLACE VIEW auth_active_sessions AS
	SELECT s.*, u.email, u.display_name
	FROM auth_sessions s
	JOIN auth_users u ON u.id = s.user_id
	WHERE s.status = 'active'
	  AND (s.expires_at IS NULL OR s.expires_at > now())
	  AND s.is_deleted = FALSE;

CREATE OR REPLACE VIEW auth_user_permissions AS
	SELECT
		ra.tenant_id,
		ra.user_id,
		p.id   AS permission_id,
		p.name AS permission_name,
		p.resource,
		p.action,
		p.effect,
		r.id   AS role_id,
		r.name AS role_name,
		r.tier AS role_tier
	FROM auth_role_assignments ra
	JOIN auth_roles r              ON r.id = ra.role_id
	JOIN auth_role_permissions rp  ON rp.role_id = r.id
	JOIN auth_permissions p        ON p.id = rp.permission_id
	WHERE ra.is_active = TRUE
	  AND ra.is_deleted = FALSE
	  AND (ra.expires_at IS NULL OR ra.expires_at > now())
	  AND r.status = 'active'
	  AND r.is_deleted = FALSE
	  AND rp.is_deleted = FALSE
	  AND p.is_deleted = FALSE;

-- =============================================================================
-- CLEANUP FUNCTION (call from a cron job)
-- =============================================================================

CREATE OR REPLACE FUNCTION auth_cleanup_expired()
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
	-- Expire sessions past their expiry time
	UPDATE auth_sessions
	SET status = 'expired', updated_at = now()
	WHERE status = 'active'
	  AND expires_at IS NOT NULL
	  AND expires_at < now();

	-- Expire delegations
	UPDATE auth_delegations
	SET status = 'expired', updated_at = now()
	WHERE status = 'active'
	  AND expires_at < now();

	-- Expire OAuth tokens
	UPDATE auth_oauth_tokens
	SET status = 'expired', updated_at = now()
	WHERE status = 'active'
	  AND expires_at < now();

	-- Purge JWT blacklist entries whose tokens have already expired
	DELETE FROM auth_token_blacklist WHERE expires_at < now();

	-- Mark OAuth codes expired
	UPDATE auth_oauth_codes
	SET used = TRUE
	WHERE used = FALSE AND expires_at < now();
END;
$$;
