-- APG Monitoring and Observability (MONI) — PostgreSQL Schema
-- Author: Nyimbi Odero  |  Copyright: © 2025 Datacraft
-- Run: psql $DATABASE_URL -f database/schema.sql
-- All tables use tenant_id + soft-delete for multi-tenant isolation.
-- Partition hints are noted where cardinality warrants it.

-- ─── Extensions ──────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ─── Signal sources ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_signal_sources (
	source_record_id    TEXT        NOT NULL,
	tenant_id           TEXT        NOT NULL DEFAULT 'default',
	source_id           TEXT        NOT NULL,
	service_name        TEXT        NOT NULL,
	environment         TEXT        NOT NULL,
	owner               TEXT        NOT NULL,
	allowed_signal_types TEXT[]     NOT NULL DEFAULT ARRAY['metric','log','trace'],
	notification_route  TEXT,
	status              TEXT        NOT NULL DEFAULT 'active'
	                    CHECK (status IN ('active','disabled','retiring')),
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (tenant_id, source_id)
);

CREATE INDEX IF NOT EXISTS idx_mn_sources_tenant
	ON mn_signal_sources (tenant_id) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_sources_service
	ON mn_signal_sources (tenant_id, service_name) WHERE is_deleted = FALSE;

-- ─── Signals (metrics, logs, traces metadata) ────────────────────────────────
-- High-write table. Partition by created_at (monthly) in production.

CREATE TABLE IF NOT EXISTS mn_signals (
	signal_id               TEXT        NOT NULL,
	tenant_id               TEXT        NOT NULL,
	source_id               TEXT        NOT NULL,
	signal_type             TEXT        NOT NULL
	                        CHECK (signal_type IN ('metric','log','trace')),
	name                    TEXT        NOT NULL,
	value                   DOUBLE PRECISION,
	labels                  JSONB       NOT NULL DEFAULT '{}',
	severity                TEXT        NOT NULL DEFAULT 'info',
	trace_id                TEXT,
	service_name            TEXT,
	cardinality             INTEGER     NOT NULL DEFAULT 0,
	contains_pii            BOOLEAN     NOT NULL DEFAULT FALSE,
	pii_redacted            BOOLEAN     NOT NULL DEFAULT TRUE,
	matched_rules           TEXT[]      NOT NULL DEFAULT '{}',
	policy_decision         TEXT        NOT NULL DEFAULT 'allow',
	decision                TEXT        NOT NULL DEFAULT 'allow',
	status                  TEXT        NOT NULL DEFAULT 'accepted',
	review_reasons          TEXT[]      NOT NULL DEFAULT '{}',
	review_evidence         JSONB       NOT NULL DEFAULT '{}',
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (signal_id)
);

CREATE INDEX IF NOT EXISTS idx_mn_signals_tenant_type
	ON mn_signals (tenant_id, signal_type, created_at DESC) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_signals_source
	ON mn_signals (tenant_id, source_id) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_signals_trace
	ON mn_signals (trace_id) WHERE trace_id IS NOT NULL AND is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_signals_labels
	ON mn_signals USING gin (labels);

-- ─── SLOs ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_slos (
	slo_id              TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	service_name        TEXT        NOT NULL,
	objective           TEXT        NOT NULL,
	threshold           DOUBLE PRECISION NOT NULL,
	window_minutes      INTEGER     NOT NULL,
	owner               TEXT        NOT NULL,
	notification_route  TEXT,
	status              TEXT        NOT NULL DEFAULT 'active'
	                    CHECK (status IN ('active','breached','paused','retired')),
	current_compliance  DOUBLE PRECISION NOT NULL DEFAULT 100.0,
	error_budget_remaining_pct DOUBLE PRECISION NOT NULL DEFAULT 100.0,
	burn_rate           DOUBLE PRECISION NOT NULL DEFAULT 0.0,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_slos_tenant
	ON mn_slos (tenant_id) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_slos_service
	ON mn_slos (tenant_id, service_name) WHERE is_deleted = FALSE;

-- ─── Alerts ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_alerts (
	alert_id            TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	source_id           TEXT        NOT NULL,
	severity            TEXT        NOT NULL
	                    CHECK (severity IN ('info','low','medium','high','critical')),
	title               TEXT        NOT NULL,
	status              TEXT        NOT NULL DEFAULT 'open'
	                    CHECK (status IN ('open','acknowledged','resolved','denied','suppressed')),
	notification_route  TEXT,
	owner               TEXT,
	incident_id         TEXT,
	matched_rules       TEXT[]      NOT NULL DEFAULT '{}',
	policy_decision     TEXT        NOT NULL DEFAULT 'allow',
	decision            TEXT        NOT NULL DEFAULT 'allow',
	review_reasons      TEXT[]      NOT NULL DEFAULT '{}',
	review_evidence     JSONB       NOT NULL DEFAULT '{}',
	acknowledged_at     TIMESTAMPTZ,
	resolved_at         TIMESTAMPTZ,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_alerts_tenant_status
	ON mn_alerts (tenant_id, status, severity) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_alerts_incident
	ON mn_alerts (incident_id) WHERE incident_id IS NOT NULL AND is_deleted = FALSE;

-- ─── Incidents ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_incidents (
	incident_id         TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	title               TEXT        NOT NULL,
	severity            TEXT        NOT NULL
	                    CHECK (severity IN ('info','low','medium','high','critical')),
	owner               TEXT,
	notification_route  TEXT,
	status              TEXT        NOT NULL DEFAULT 'open'
	                    CHECK (status IN ('open','investigating','resolved','denied','closed')),
	alert_ids           TEXT[]      NOT NULL DEFAULT '{}',
	matched_rules       TEXT[]      NOT NULL DEFAULT '{}',
	policy_decision     TEXT        NOT NULL DEFAULT 'allow',
	review_reasons      TEXT[]      NOT NULL DEFAULT '{}',
	review_evidence     JSONB       NOT NULL DEFAULT '{}',
	resolved_at         TIMESTAMPTZ,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_incidents_tenant_status
	ON mn_incidents (tenant_id, status) WHERE is_deleted = FALSE;

-- ─── Remediation requests ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_remediation_requests (
	request_id          TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	incident_id         TEXT        NOT NULL REFERENCES mn_incidents (incident_id),
	requester           TEXT        NOT NULL,
	environment         TEXT        NOT NULL,
	runbook_id          TEXT        NOT NULL,
	runbook_approved    BOOLEAN     NOT NULL DEFAULT FALSE,
	proposed_action     TEXT        NOT NULL,
	reason              TEXT        NOT NULL,
	decision            TEXT        NOT NULL DEFAULT 'pending',
	status              TEXT        NOT NULL DEFAULT 'pending_review'
	                    CHECK (status IN ('pending_review','approved','rejected','denied','review_denied')),
	reviewer            TEXT,
	review_notes        TEXT,
	matched_rules       TEXT[]      NOT NULL DEFAULT '{}',
	policy_decision     TEXT        NOT NULL DEFAULT 'require_review',
	review_reasons      TEXT[]      NOT NULL DEFAULT '{}',
	review_evidence     JSONB       NOT NULL DEFAULT '{}',
	decided_at          TIMESTAMPTZ,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_remediation_tenant_status
	ON mn_remediation_requests (tenant_id, status) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_remediation_incident
	ON mn_remediation_requests (incident_id);

-- ─── Monitoring agents ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_monitoring_agents (
	agent_id                TEXT        NOT NULL,
	tenant_id               TEXT        NOT NULL,
	name                    TEXT        NOT NULL,
	runtime                 TEXT        NOT NULL,
	role                    TEXT        NOT NULL,
	scope                   TEXT        NOT NULL,
	owner                   TEXT        NOT NULL,
	purpose                 TEXT        NOT NULL,
	contribution_disclosed  BOOLEAN     NOT NULL DEFAULT TRUE,
	human_approval_required BOOLEAN     NOT NULL DEFAULT FALSE,
	status                  TEXT        NOT NULL DEFAULT 'active'
	                        CHECK (status IN ('active','pending_review','suspended','retired')),
	policy_decision         TEXT        NOT NULL DEFAULT 'allow',
	matched_rules           TEXT[]      NOT NULL DEFAULT '{}',
	review_reasons          TEXT[]      NOT NULL DEFAULT '{}',
	review_evidence         JSONB       NOT NULL DEFAULT '{}',
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	PRIMARY KEY (tenant_id, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_mn_agents_tenant_status
	ON mn_monitoring_agents (tenant_id, status) WHERE is_deleted = FALSE;

-- ─── Lifecycle batches ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_lifecycle_batches (
	batch_id            TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	event_stream        TEXT        NOT NULL,
	mutation_count      INTEGER     NOT NULL,
	accepted            BOOLEAN     NOT NULL DEFAULT TRUE,
	decision            TEXT        NOT NULL DEFAULT 'allow',
	required_processor  TEXT        NOT NULL DEFAULT 'bytewax',
	status              TEXT        NOT NULL DEFAULT 'accepted',
	matched_rules       TEXT[]      NOT NULL DEFAULT '{}',
	policy_decision     TEXT        NOT NULL DEFAULT 'allow',
	review_reasons      TEXT[]      NOT NULL DEFAULT '{}',
	review_evidence     JSONB       NOT NULL DEFAULT '{}',
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_batches_tenant
	ON mn_lifecycle_batches (tenant_id, created_at DESC) WHERE is_deleted = FALSE;

-- ─── Audit events ─────────────────────────────────────────────────────────────
-- Partition by created_at (monthly) in production. Never deleted (is_deleted unused).

CREATE TABLE IF NOT EXISTS mn_audit_events (
	event_id            TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	event_type          TEXT        NOT NULL,
	subject             TEXT        NOT NULL,
	actor               TEXT        NOT NULL,
	decision            TEXT        NOT NULL DEFAULT 'allow',
	matched_rules       TEXT[]      NOT NULL DEFAULT '{}',
	policy_decision     TEXT        NOT NULL DEFAULT 'allow',
	review_reasons      TEXT[]      NOT NULL DEFAULT '{}',
	review_evidence     JSONB       NOT NULL DEFAULT '{}',
	details             JSONB       NOT NULL DEFAULT '{}',
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mn_audit_tenant_time
	ON mn_audit_events (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mn_audit_subject
	ON mn_audit_events (tenant_id, subject, event_type);

-- ─── Metric series (TSDB metadata / label catalog) ───────────────────────────

CREATE TABLE IF NOT EXISTS mn_metric_series (
	series_id           TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	name                TEXT        NOT NULL,
	metric_type         TEXT        NOT NULL DEFAULT 'gauge',
	unit                TEXT,
	labels              JSONB       NOT NULL DEFAULT '{}',
	source              TEXT        NOT NULL,
	source_type         TEXT        NOT NULL DEFAULT 'unknown',
	retention_policy    TEXT        NOT NULL DEFAULT 'medium_term',
	capability_name     TEXT,
	cardinality         INTEGER     NOT NULL DEFAULT 0,
	last_value          DOUBLE PRECISION,
	last_seen           TIMESTAMPTZ,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_mn_series_unique
	ON mn_metric_series (tenant_id, name, (labels::text)) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_series_labels
	ON mn_metric_series USING gin (labels);

-- ─── Alert rules ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_alert_rules (
	rule_id                     TEXT        NOT NULL PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	name                        TEXT        NOT NULL,
	description                 TEXT        NOT NULL DEFAULT '',
	enabled                     BOOLEAN     NOT NULL DEFAULT TRUE,
	condition                   TEXT        NOT NULL,
	condition_type              TEXT        NOT NULL DEFAULT 'threshold',
	metric_name                 TEXT        NOT NULL,
	metric_labels               JSONB       NOT NULL DEFAULT '{}',
	scope                       TEXT        NOT NULL DEFAULT 'tenant',
	threshold_value             DOUBLE PRECISION,
	threshold_operator          TEXT        NOT NULL DEFAULT 'gt',
	evaluation_window_minutes   INTEGER     NOT NULL DEFAULT 5,
	evaluation_interval_seconds INTEGER     NOT NULL DEFAULT 60,
	severity                    TEXT        NOT NULL DEFAULT 'medium',
	alert_message               TEXT        NOT NULL,
	alert_summary               TEXT        NOT NULL DEFAULT '',
	runbook_url                 TEXT,
	escalation_enabled          BOOLEAN     NOT NULL DEFAULT TRUE,
	escalation_interval_minutes INTEGER     NOT NULL DEFAULT 30,
	max_escalation_level        INTEGER     NOT NULL DEFAULT 3,
	anomaly_detection_enabled   BOOLEAN     NOT NULL DEFAULT FALSE,
	anomaly_sensitivity         DOUBLE PRECISION NOT NULL DEFAULT 0.8,
	baseline_period_days        INTEGER     NOT NULL DEFAULT 7,
	trigger_count               INTEGER     NOT NULL DEFAULT 0,
	false_positive_rate         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
	effectiveness_score         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
	created_by                  TEXT        NOT NULL,
	last_triggered              TIMESTAMPTZ,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_rules_tenant_enabled
	ON mn_alert_rules (tenant_id, enabled) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_rules_metric
	ON mn_alert_rules (tenant_id, metric_name) WHERE is_deleted = FALSE;

-- ─── Health checks ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_health_checks (
	check_id                TEXT        NOT NULL PRIMARY KEY,
	tenant_id               TEXT        NOT NULL,
	name                    TEXT        NOT NULL,
	service_name            TEXT        NOT NULL,
	endpoint                TEXT        NOT NULL,
	method                  TEXT        NOT NULL DEFAULT 'GET',
	expected_status         INTEGER     NOT NULL DEFAULT 200,
	timeout_seconds         INTEGER     NOT NULL DEFAULT 5,
	interval_seconds        INTEGER     NOT NULL DEFAULT 30,
	healthy                 BOOLEAN     NOT NULL DEFAULT TRUE,
	last_checked            TIMESTAMPTZ,
	last_response_ms        DOUBLE PRECISION,
	consecutive_failures    INTEGER     NOT NULL DEFAULT 0,
	labels                  JSONB       NOT NULL DEFAULT '{}',
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_health_checks_tenant
	ON mn_health_checks (tenant_id) WHERE is_deleted = FALSE;

-- ─── Trace spans ──────────────────────────────────────────────────────────────
-- Partition by created_at (daily) in production.

CREATE TABLE IF NOT EXISTS mn_trace_spans (
	span_id             TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	trace_id            TEXT        NOT NULL,
	parent_span_id      TEXT,
	service_name        TEXT        NOT NULL,
	operation_name      TEXT        NOT NULL,
	start_time          TIMESTAMPTZ NOT NULL,
	end_time            TIMESTAMPTZ,
	duration_ms         DOUBLE PRECISION,
	status              TEXT        NOT NULL DEFAULT 'ok',
	error               BOOLEAN     NOT NULL DEFAULT FALSE,
	error_message       TEXT,
	tags                JSONB       NOT NULL DEFAULT '{}',
	logs                JSONB       NOT NULL DEFAULT '[]',
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_spans_trace
	ON mn_trace_spans (tenant_id, trace_id) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_spans_service
	ON mn_trace_spans (tenant_id, service_name, start_time DESC) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_spans_tags
	ON mn_trace_spans USING gin (tags);

-- ─── Log entries ──────────────────────────────────────────────────────────────
-- Partition by created_at (daily) in production.

CREATE TABLE IF NOT EXISTS mn_log_entries (
	log_id              TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	source_id           TEXT        NOT NULL,
	service_name        TEXT        NOT NULL,
	level               TEXT        NOT NULL DEFAULT 'info',
	message             TEXT        NOT NULL,
	timestamp           TIMESTAMPTZ NOT NULL DEFAULT now(),
	trace_id            TEXT,
	span_id             TEXT,
	labels              JSONB       NOT NULL DEFAULT '{}',
	contains_pii        BOOLEAN     NOT NULL DEFAULT FALSE,
	pii_redacted        BOOLEAN     NOT NULL DEFAULT TRUE,
	structured_data     JSONB       NOT NULL DEFAULT '{}',
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_logs_tenant_time
	ON mn_log_entries (tenant_id, timestamp DESC) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_logs_trace
	ON mn_log_entries (trace_id) WHERE trace_id IS NOT NULL AND is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_logs_message_trgm
	ON mn_log_entries USING gin (message gin_trgm_ops);

-- ─── Anomaly detections ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_anomaly_detections (
	anomaly_id          TEXT        NOT NULL PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	metric_name         TEXT        NOT NULL,
	source_id           TEXT        NOT NULL,
	detected_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
	anomaly_score       DOUBLE PRECISION NOT NULL,
	sensitivity         DOUBLE PRECISION NOT NULL DEFAULT 0.8,
	algorithm           TEXT        NOT NULL DEFAULT 'z_score',
	observed_value      DOUBLE PRECISION NOT NULL,
	expected_value      DOUBLE PRECISION NOT NULL,
	baseline_mean       DOUBLE PRECISION NOT NULL,
	baseline_std        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
	is_true_positive    BOOLEAN,
	feedback_note       TEXT,
	labels              JSONB       NOT NULL DEFAULT '{}',
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_anomalies_tenant_time
	ON mn_anomaly_detections (tenant_id, detected_at DESC) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_anomalies_metric
	ON mn_anomaly_detections (tenant_id, metric_name) WHERE is_deleted = FALSE;

-- ─── Dashboards ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_dashboards (
	dashboard_id            TEXT        NOT NULL PRIMARY KEY,
	tenant_id               TEXT        NOT NULL,
	name                    TEXT        NOT NULL,
	description             TEXT        NOT NULL DEFAULT '',
	dashboard_type          TEXT        NOT NULL DEFAULT 'operational',
	scope                   TEXT        NOT NULL DEFAULT 'tenant',
	auto_refresh            BOOLEAN     NOT NULL DEFAULT TRUE,
	refresh_interval_seconds INTEGER    NOT NULL DEFAULT 30,
	layout                  JSONB       NOT NULL DEFAULT '{}',
	widgets                 JSONB       NOT NULL DEFAULT '[]',
	widget_count            INTEGER     NOT NULL DEFAULT 0,
	public                  BOOLEAN     NOT NULL DEFAULT FALSE,
	shared_with             TEXT[]      NOT NULL DEFAULT '{}',
	view_count              INTEGER     NOT NULL DEFAULT 0,
	avg_load_time_ms        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
	created_by              TEXT        NOT NULL,
	last_viewed             TIMESTAMPTZ,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_dashboards_tenant
	ON mn_dashboards (tenant_id) WHERE is_deleted = FALSE;

-- ─── Error budgets ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mn_error_budgets (
	budget_id               TEXT        NOT NULL PRIMARY KEY,
	tenant_id               TEXT        NOT NULL,
	slo_id                  TEXT        NOT NULL REFERENCES mn_slos (slo_id),
	window_start            TIMESTAMPTZ NOT NULL,
	window_end              TIMESTAMPTZ NOT NULL,
	total_budget_minutes    DOUBLE PRECISION NOT NULL,
	consumed_minutes        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
	remaining_minutes       DOUBLE PRECISION NOT NULL,
	burn_rate               DOUBLE PRECISION NOT NULL DEFAULT 0.0,
	compliance_percent      DOUBLE PRECISION NOT NULL DEFAULT 100.0,
	computed_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_mn_budgets_slo
	ON mn_error_budgets (slo_id, window_start DESC) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mn_budgets_tenant
	ON mn_error_budgets (tenant_id) WHERE is_deleted = FALSE;

-- ─── Audit trigger: auto-update updated_at ───────────────────────────────────

CREATE OR REPLACE FUNCTION mn_set_updated_at()
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
		'mn_signal_sources', 'mn_slos', 'mn_alerts', 'mn_incidents',
		'mn_remediation_requests', 'mn_monitoring_agents', 'mn_metric_series',
		'mn_alert_rules', 'mn_health_checks', 'mn_dashboards', 'mn_error_budgets'
	] LOOP
		EXECUTE format(
			'CREATE TRIGGER trg_%1$s_updated_at
			 BEFORE UPDATE ON %1$s
			 FOR EACH ROW EXECUTE FUNCTION mn_set_updated_at()',
			t
		);
	END LOOP;
EXCEPTION WHEN duplicate_object THEN NULL;
END;
$$;
