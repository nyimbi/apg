"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 08: capability contract with configuration_schema and rule_engine.\n// Features: configuration_schema with required fields, standalone rule_engine\n//           block, provides, requires, master_data\n\nmodule capability_contract version 1.0.0 {\n    description: "Capability contract with schema validation and rule engine";\n}\n\ncapability AuditLog {\n    contract: {\n        id: audit_log,\n        provides: [audit_events, audit_reports, compliance_trail],\n        requires: [],\n        configuration: {\n            tenant_id: "default",\n            retention_days: 2555,\n            max_event_size_kb: 64,\n            compress_after_days: 90,\n            encrypt_at_rest: true\n        },\n        configuration_schema: {\n            required: ["tenant_id", "retention_days"]\n        },\n        rule_engine: {\n            type: deterministic,\n            default_decision: allow,\n            rules: [\n                {name: "tenant_required",  when: "tenant_id missing",                    action: deny,           priority: 100},\n                {name: "size_limit",       when: "event_size_kb > max_event_size_kb",    action: deny,           priority: 90},\n                {name: "sensitive_data",   when: "contains_pii == true",                 action: require_review, priority: 80}\n            ]\n        },\n        ui: {\n            shell: python,\n            routes: [\n                {name: "Audit Log",  path: "/audit",            component: "AuditLogViewer",  permission: "audit:view"},\n                {name: "Compliance", path: "/audit/compliance", component: "ComplianceView",  permission: "audit:compliance"}\n            ]\n        },\n        theme: {name: audit_theme, tokens: {accent: "#5C6BC0"}}\n    };\n\n    master_data: {entities: [audit_event, audit_session, compliance_rule]};\n}\n\ncapability NotificationEngine {\n    contract: {\n        id: notification_engine,\n        provides: [notifications, alert_dispatch, delivery_tracking],\n        requires: [audit_events],\n        configuration: {\n            tenant_id: "default",\n            default_channel: "email",\n            retry_attempts: 3,\n            retry_delay_seconds: 60\n        },\n        configuration_schema: {\n            required: ["tenant_id", "default_channel"]\n        },\n        rule_engine: {\n            type: policy,\n            default_decision: allow,\n            rules: [\n                {name: "rate_limit",      when: "notifications_per_hour > 1000",                   action: deny},\n                {name: "valid_channel",   when: "channel in [email, sms, push, webhook]",           action: allow},\n                {name: "invalid_channel", when: "channel not in [email, sms, push, webhook]",       action: deny}\n            ]\n        }\n    };\n}\n\napp CapabilityContractApp {\n    description: "Audit and notification capabilities";\n    capabilities: [AuditLog, NotificationEngine];\n    routes: ["/audit", "/notifications"];\n}\n'
APG_MODULE_NAME = 'capability_contract'
_GENERATED_TEST_ENV_KEYS = (
	'APG_API_KEY',
	'APG_AUTH_USERS',
	'APG_AUTO_MIGRATE',
	'APG_DATABASE_URL',
	'APG_DATA_FILE',
	'APG_DATA_PATH',
	'APG_DB_PATH',
	'APG_ENV',
	'APG_JWT_SECRET',
	'APG_PG_URL',
	'APG_PRODUCTION',
	'APG_SESSION_SECRET',
	'APG_SQLITE_PATH',
	'DATABASE_URL',
)


@pytest.fixture()
def generated_app_client(monkeypatch):
	for key in _GENERATED_TEST_ENV_KEYS:
		monkeypatch.delenv(key, raising=False)
	result = APGCompiler().compile_string(APG_SOURCE, APG_MODULE_NAME)
	assert result.success, result.errors
	namespace = {"__file__": "generated_app.py"}
	exec(compile(result.generated_files["app.py"], "generated_app.py", "exec"), namespace)
	app = namespace["_flask_app"]
	app.config["TESTING"] = True
	return app.test_client()
