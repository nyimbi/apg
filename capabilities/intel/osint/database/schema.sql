-- APG Open Source Intelligence — PostgreSQL schema v2
-- Run: psql $DATABASE_URL -f database/schema.sql
--
-- All tables include:
--   id          TEXT PRIMARY KEY  (UUID7 string)
--   tenant_id   TEXT NOT NULL     (tenant isolation, always indexed)
--   created_at  TIMESTAMPTZ
--   updated_at  TIMESTAMPTZ
--   created_by  TEXT
--   is_deleted  BOOLEAN DEFAULT FALSE
--
-- Partitioning hints for large-volume tables are noted inline.

-- ============================================================
-- Extensions
-- ============================================================
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- Sources
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_sources (
	id                      TEXT        NOT NULL,
	tenant_id               TEXT        NOT NULL,
	name                    TEXT        NOT NULL,
	source_type             TEXT        NOT NULL,
	url                     TEXT,
	description             TEXT,
	owner_id                TEXT        NOT NULL,
	terms_review_reference  TEXT        NOT NULL,
	risk_tier               TEXT        NOT NULL DEFAULT 'low',
	collection_method       TEXT        NOT NULL,
	status                  TEXT        NOT NULL DEFAULT 'active',
	requires_auth           BOOLEAN     NOT NULL DEFAULT FALSE,
	auth_reference          TEXT,
	rate_limit_rps          NUMERIC(10,4),
	credibility_baseline    NUMERIC(5,4) NOT NULL DEFAULT 0.5,
	tags                    TEXT[]      NOT NULL DEFAULT '{}',
	evidence_reference      TEXT        NOT NULL,
	last_collected_at       TIMESTAMPTZ,
	total_items_collected   BIGINT      NOT NULL DEFAULT 0,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_sources_tenant    ON osint_sources (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_sources_type      ON osint_sources (tenant_id, source_type) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_sources_risk      ON osint_sources (tenant_id, risk_tier) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_sources_tags      ON osint_sources USING gin (tags);

-- ============================================================
-- Collection tasks
-- PARTITION BY RANGE (created_at) recommended for high volume
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_tasks (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	source_id           TEXT        NOT NULL REFERENCES osint_sources(id) ON DELETE RESTRICT,
	task_type           TEXT        NOT NULL,
	status              TEXT        NOT NULL DEFAULT 'pending',
	parameters          JSONB       NOT NULL DEFAULT '{}',
	priority            TEXT        NOT NULL DEFAULT 'medium',
	scheduled_at        TIMESTAMPTZ,
	started_at          TIMESTAMPTZ,
	completed_at        TIMESTAMPTZ,
	max_depth           SMALLINT    NOT NULL DEFAULT 2,
	max_items           INT,
	keywords            TEXT[]      NOT NULL DEFAULT '{}',
	items_collected     INT         NOT NULL DEFAULT 0,
	error_message       TEXT,
	approval_reference  TEXT,
	evidence_reference  TEXT        NOT NULL,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_tasks_tenant     ON osint_tasks (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_tasks_source     ON osint_tasks (tenant_id, source_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_tasks_status     ON osint_tasks (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_tasks_type       ON osint_tasks (tenant_id, task_type) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_tasks_created    ON osint_tasks (tenant_id, created_at DESC) WHERE NOT is_deleted;

-- ============================================================
-- Raw intelligence
-- PARTITION BY RANGE (captured_at) recommended for high volume
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_raw_intel (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	task_id             TEXT        NOT NULL REFERENCES osint_tasks(id) ON DELETE RESTRICT,
	source_id           TEXT        NOT NULL,
	content_reference   TEXT        NOT NULL,
	content_type        TEXT        NOT NULL,
	url                 TEXT,
	fingerprint         TEXT        NOT NULL,
	confidence_score    NUMERIC(5,4) NOT NULL,
	language            TEXT,
	captured_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
	status              TEXT        NOT NULL DEFAULT 'raw',
	triage_decision     TEXT,
	analyst_id          TEXT,
	notes               TEXT,
	evidence_reference  TEXT        NOT NULL,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id),
	UNIQUE (tenant_id, fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_osint_raw_tenant       ON osint_raw_intel (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_raw_task         ON osint_raw_intel (tenant_id, task_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_raw_status       ON osint_raw_intel (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_raw_triage       ON osint_raw_intel (tenant_id, triage_decision) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_raw_captured     ON osint_raw_intel (tenant_id, captured_at DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_raw_fingerprint  ON osint_raw_intel (tenant_id, fingerprint);

-- ============================================================
-- Processed intelligence
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_processed_intel (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	raw_intel_id        TEXT        NOT NULL REFERENCES osint_raw_intel(id) ON DELETE RESTRICT,
	requirement_id      TEXT,
	assessment_type     TEXT        NOT NULL,
	summary             TEXT        NOT NULL,
	key_findings        TEXT[]      NOT NULL DEFAULT '{}',
	confidence_score    NUMERIC(5,4) NOT NULL,
	confidence_level    TEXT        NOT NULL DEFAULT 'possible',
	classification      TEXT        NOT NULL DEFAULT 'unclassified',
	tlp                 TEXT        NOT NULL DEFAULT 'amber',
	status              TEXT        NOT NULL DEFAULT 'processed',
	analyst_id          TEXT        NOT NULL,
	tags                TEXT[]      NOT NULL DEFAULT '{}',
	evidence_reference  TEXT        NOT NULL,
	entity_ids          TEXT[]      NOT NULL DEFAULT '{}',
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_proc_tenant      ON osint_processed_intel (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_proc_type        ON osint_processed_intel (tenant_id, assessment_type) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_proc_analyst     ON osint_processed_intel (tenant_id, analyst_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_proc_status      ON osint_processed_intel (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_proc_tags        ON osint_processed_intel USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_osint_proc_entities    ON osint_processed_intel USING gin (entity_ids);

-- ============================================================
-- Entities
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_entities (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	entity_type         TEXT        NOT NULL,
	name                TEXT        NOT NULL,
	aliases             TEXT[]      NOT NULL DEFAULT '{}',
	description         TEXT,
	attributes          JSONB       NOT NULL DEFAULT '{}',
	confidence_score    NUMERIC(5,4) NOT NULL,
	confidence_level    TEXT        NOT NULL DEFAULT 'possible',
	classification      TEXT        NOT NULL DEFAULT 'unclassified',
	source_intel_ids    TEXT[]      NOT NULL DEFAULT '{}',
	relationship_ids    TEXT[]      NOT NULL DEFAULT '{}',
	tags                TEXT[]      NOT NULL DEFAULT '{}',
	evidence_reference  TEXT        NOT NULL,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_entity_tenant    ON osint_entities (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_entity_type      ON osint_entities (tenant_id, entity_type) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_entity_name      ON osint_entities USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_osint_entity_aliases   ON osint_entities USING gin (aliases);
CREATE INDEX IF NOT EXISTS idx_osint_entity_tags      ON osint_entities USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_osint_entity_conf      ON osint_entities (tenant_id, confidence_score DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_entity_attrs     ON osint_entities USING gin (attributes);

-- ============================================================
-- Entity relationships
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_entity_relationships (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	source_entity_id    TEXT        NOT NULL REFERENCES osint_entities(id) ON DELETE CASCADE,
	target_entity_id    TEXT        NOT NULL REFERENCES osint_entities(id) ON DELETE CASCADE,
	relationship_type   TEXT        NOT NULL,
	description         TEXT,
	strength            NUMERIC(5,4) NOT NULL DEFAULT 0.5,
	confidence_score    NUMERIC(5,4) NOT NULL,
	first_seen          TIMESTAMPTZ,
	last_seen           TIMESTAMPTZ,
	attributes          JSONB       NOT NULL DEFAULT '{}',
	evidence_reference  TEXT        NOT NULL,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id),
	CHECK (source_entity_id <> target_entity_id)
);

CREATE INDEX IF NOT EXISTS idx_osint_rel_tenant       ON osint_entity_relationships (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_rel_source       ON osint_entity_relationships (tenant_id, source_entity_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_rel_target       ON osint_entity_relationships (tenant_id, target_entity_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_rel_type         ON osint_entity_relationships (tenant_id, relationship_type) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_rel_conf         ON osint_entity_relationships (tenant_id, confidence_score DESC) WHERE NOT is_deleted;

-- ============================================================
-- Social media profiles
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_social_profiles (
	id                      TEXT        NOT NULL,
	tenant_id               TEXT        NOT NULL,
	entity_id               TEXT        REFERENCES osint_entities(id) ON DELETE SET NULL,
	platform                TEXT        NOT NULL,
	handle                  TEXT        NOT NULL,
	profile_url             TEXT,
	display_name            TEXT,
	bio                     TEXT,
	followers_count         INT,
	following_count         INT,
	post_count              INT,
	verified                BOOLEAN     NOT NULL DEFAULT FALSE,
	is_active               BOOLEAN     NOT NULL DEFAULT TRUE,
	created_platform_at     TIMESTAMPTZ,
	attributes              JSONB       NOT NULL DEFAULT '{}',
	keywords_monitored      TEXT[]      NOT NULL DEFAULT '{}',
	evidence_reference      TEXT        NOT NULL,
	last_scraped_at         TIMESTAMPTZ,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_social_tenant    ON osint_social_profiles (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_social_platform  ON osint_social_profiles (tenant_id, platform) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_social_handle    ON osint_social_profiles (tenant_id, handle) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_social_entity    ON osint_social_profiles (tenant_id, entity_id) WHERE NOT is_deleted AND entity_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_osint_social_keywords  ON osint_social_profiles USING gin (keywords_monitored);

-- ============================================================
-- Web content
-- PARTITION BY RANGE (scraped_at) recommended
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_web_content (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	task_id             TEXT        NOT NULL REFERENCES osint_tasks(id) ON DELETE RESTRICT,
	url                 TEXT        NOT NULL,
	title               TEXT,
	content_hash        TEXT        NOT NULL,
	content_reference   TEXT        NOT NULL,
	mime_type           TEXT        NOT NULL DEFAULT 'text/html',
	language            TEXT,
	depth               SMALLINT    NOT NULL DEFAULT 0,
	parent_url          TEXT,
	links_extracted     TEXT[]      NOT NULL DEFAULT '{}',
	metadata            JSONB       NOT NULL DEFAULT '{}',
	scraped_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	evidence_reference  TEXT        NOT NULL,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_web_tenant       ON osint_web_content (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_web_task         ON osint_web_content (tenant_id, task_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_web_url          ON osint_web_content USING gin (url gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_osint_web_scraped      ON osint_web_content (tenant_id, scraped_at DESC) WHERE NOT is_deleted;

-- ============================================================
-- Domain records
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_domain_records (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	domain              TEXT        NOT NULL,
	registrar           TEXT,
	registrant_name     TEXT,
	registrant_email    TEXT,
	registrant_org      TEXT,
	registrant_country  TEXT,
	created_date        TIMESTAMPTZ,
	updated_date        TIMESTAMPTZ,
	expiry_date         TIMESTAMPTZ,
	name_servers        TEXT[]      NOT NULL DEFAULT '{}',
	a_records           TEXT[]      NOT NULL DEFAULT '{}',
	mx_records          TEXT[]      NOT NULL DEFAULT '{}',
	txt_records         TEXT[]      NOT NULL DEFAULT '{}',
	ssl_issuer          TEXT,
	ssl_expiry          TIMESTAMPTZ,
	ssl_san             TEXT[]      NOT NULL DEFAULT '{}',
	raw_whois           TEXT,
	attributes          JSONB       NOT NULL DEFAULT '{}',
	queried_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_domain_tenant    ON osint_domain_records (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_domain_name      ON osint_domain_records (tenant_id, domain) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_domain_email     ON osint_domain_records (tenant_id, registrant_email) WHERE NOT is_deleted AND registrant_email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_osint_domain_queried   ON osint_domain_records (tenant_id, queried_at DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_domain_attrs     ON osint_domain_records USING gin (attributes);

-- ============================================================
-- IP intelligence
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_ip_intel (
	id                      TEXT        NOT NULL,
	tenant_id               TEXT        NOT NULL,
	ip_address              TEXT        NOT NULL,
	ip_version              SMALLINT    NOT NULL DEFAULT 4,
	asn                     TEXT,
	asn_org                 TEXT,
	isp                     TEXT,
	country_code            CHAR(2),
	country_name            TEXT,
	region                  TEXT,
	city                    TEXT,
	latitude                DOUBLE PRECISION,
	longitude               DOUBLE PRECISION,
	is_tor                  BOOLEAN     NOT NULL DEFAULT FALSE,
	is_vpn                  BOOLEAN     NOT NULL DEFAULT FALSE,
	is_proxy                BOOLEAN     NOT NULL DEFAULT FALSE,
	is_datacenter           BOOLEAN     NOT NULL DEFAULT FALSE,
	abuse_confidence_score  NUMERIC(5,4) NOT NULL DEFAULT 0.0,
	threat_types            TEXT[]      NOT NULL DEFAULT '{}',
	open_ports              INT[]       NOT NULL DEFAULT '{}',
	reverse_dns             TEXT,
	attributes              JSONB       NOT NULL DEFAULT '{}',
	queried_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	evidence_reference      TEXT        NOT NULL,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_ip_tenant        ON osint_ip_intel (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_ip_address       ON osint_ip_intel (tenant_id, ip_address) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_ip_country       ON osint_ip_intel (tenant_id, country_code) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_ip_tor           ON osint_ip_intel (tenant_id, is_tor) WHERE NOT is_deleted AND is_tor;
CREATE INDEX IF NOT EXISTS idx_osint_ip_abuse         ON osint_ip_intel (tenant_id, abuse_confidence_score DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_ip_threats       ON osint_ip_intel USING gin (threat_types);

-- ============================================================
-- Document analysis
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_document_analyses (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	raw_intel_id        TEXT        NOT NULL REFERENCES osint_raw_intel(id) ON DELETE RESTRICT,
	language            TEXT,
	sentiment_score     NUMERIC(5,4),
	entities_extracted  JSONB       NOT NULL DEFAULT '[]',
	keywords            TEXT[]      NOT NULL DEFAULT '{}',
	topics              TEXT[]      NOT NULL DEFAULT '{}',
	summary             TEXT,
	threat_indicators   TEXT[]      NOT NULL DEFAULT '{}',
	location_mentions   JSONB       NOT NULL DEFAULT '[]',
	person_mentions     TEXT[]      NOT NULL DEFAULT '{}',
	org_mentions        TEXT[]      NOT NULL DEFAULT '{}',
	date_mentions       TEXT[]      NOT NULL DEFAULT '{}',
	model_used          TEXT,
	processing_time_ms  INT,
	evidence_reference  TEXT        NOT NULL,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_docana_tenant    ON osint_document_analyses (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_docana_raw       ON osint_document_analyses (tenant_id, raw_intel_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_docana_keywords  ON osint_document_analyses USING gin (keywords);
CREATE INDEX IF NOT EXISTS idx_osint_docana_threats   ON osint_document_analyses USING gin (threat_indicators);
CREATE INDEX IF NOT EXISTS idx_osint_docana_persons   ON osint_document_analyses USING gin (person_mentions);
CREATE INDEX IF NOT EXISTS idx_osint_docana_orgs      ON osint_document_analyses USING gin (org_mentions);

-- ============================================================
-- Credibility scores
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_credibility_scores (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	reference_id        TEXT        NOT NULL,
	reference_type      TEXT        NOT NULL,
	score               NUMERIC(5,4) NOT NULL,
	factors             JSONB       NOT NULL DEFAULT '{}',
	analyst_id          TEXT        NOT NULL,
	rationale           TEXT,
	evidence_reference  TEXT        NOT NULL,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_cred_tenant      ON osint_credibility_scores (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_cred_ref         ON osint_credibility_scores (tenant_id, reference_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_cred_type        ON osint_credibility_scores (tenant_id, reference_type) WHERE NOT is_deleted;

-- ============================================================
-- Dissemination packages
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_dissemination_packages (
	id                      TEXT        NOT NULL,
	tenant_id               TEXT        NOT NULL,
	processed_intel_ids     TEXT[]      NOT NULL DEFAULT '{}',
	audience                TEXT        NOT NULL,
	release_marking         TEXT        NOT NULL,
	classification          TEXT        NOT NULL DEFAULT 'unclassified',
	title                   TEXT        NOT NULL,
	executive_summary       TEXT        NOT NULL,
	approval_reference      TEXT        NOT NULL,
	evidence_reference      TEXT        NOT NULL,
	disseminated_at         TIMESTAMPTZ,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_diss_tenant      ON osint_dissemination_packages (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_diss_marking     ON osint_dissemination_packages (tenant_id, release_marking) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_diss_intel       ON osint_dissemination_packages USING gin (processed_intel_ids);

-- ============================================================
-- Reviews
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_reviews (
	id                  TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL,
	reference_id        TEXT        NOT NULL,
	reference_type      TEXT        NOT NULL,
	reviewer_id         TEXT        NOT NULL,
	status              TEXT        NOT NULL,
	notes               TEXT,
	evidence_reference  TEXT        NOT NULL,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_review_tenant    ON osint_reviews (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_review_ref       ON osint_reviews (tenant_id, reference_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_review_reviewer  ON osint_reviews (tenant_id, reviewer_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_review_status    ON osint_reviews (tenant_id, status) WHERE NOT is_deleted;

-- ============================================================
-- OSINT agents
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_agents (
	id              TEXT        NOT NULL,
	tenant_id       TEXT        NOT NULL,
	name            TEXT        NOT NULL,
	runtime         TEXT        NOT NULL,
	role            TEXT        NOT NULL,
	scope           TEXT        NOT NULL,
	capabilities    TEXT[]      NOT NULL DEFAULT '{}',
	is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_agent_tenant     ON osint_agents (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_agent_runtime    ON osint_agents (tenant_id, runtime) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_osint_agent_role       ON osint_agents (tenant_id, role) WHERE NOT is_deleted;

-- ============================================================
-- Audit events
-- PARTITION BY RANGE (occurred_at) strongly recommended
-- ============================================================
CREATE TABLE IF NOT EXISTS osint_audit_events (
	id              TEXT        NOT NULL DEFAULT gen_random_uuid(),
	tenant_id       TEXT        NOT NULL,
	event_type      TEXT        NOT NULL,
	reference_id    TEXT,
	actor_id        TEXT,
	payload         JSONB       NOT NULL DEFAULT '{}',
	occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
	stream          TEXT        NOT NULL DEFAULT 'apg.intel.osint.lifecycle',
	PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_osint_audit_tenant     ON osint_audit_events (tenant_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_osint_audit_type       ON osint_audit_events (tenant_id, event_type);
CREATE INDEX IF NOT EXISTS idx_osint_audit_ref        ON osint_audit_events (tenant_id, reference_id) WHERE reference_id IS NOT NULL;
