"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 19: multi-capability composition with dependency chain.\n// Features: requires chain across 4 capabilities, rule_engine block,\n//           app with agent_teams, full composability pattern\n\nmodule platform_suite version 1.0.0 {\n    description: "Multi-capability platform demonstrating dependency chain composition";\n}\n\n// Layer 1: Foundation\ncapability Authentication {\n    contract: {\n        id: auth,\n        provides: [identity_verification, session_management, role_assignment, jwt_tokens],\n        configuration: {tenant_id: "default", token_ttl_seconds: 3600, mfa_required: false},\n        rule_engine: {\n            type: deterministic,\n            default_decision: deny,\n            rules: [\n                {name: "valid_session",    when: "session_token not missing",  action: allow,  priority: 100},\n                {name: "mfa_required",     when: "mfa_token missing",          action: deny,   priority: 90},\n                {name: "locked_account",   when: "account_locked == true",     action: deny,   priority: 80},\n                {name: "rate_limited",     when: "login_attempts > 5",         action: deny,   priority: 70}\n            ]\n        }\n    };\n}\n\n// Layer 2: Observability (depends on auth)\ncapability AuditTrail {\n    contract: {\n        id: audl,\n        provides: [audit_events, compliance_trail, change_history],\n        requires: [identity_verification],\n        configuration: {tenant_id: "default", retention_days: 2555, immutable: true},\n        rules: [\n            {name: "actor_required",    when: "actor_id missing",   action: deny},\n            {name: "event_type_valid",  when: "event_type missing", action: deny}\n        ]\n    };\n}\n\n// Layer 3: Core business (depends on auth + audit)\ncapability CustomerMaster {\n    contract: {\n        id: customer_master,\n        provides: [customer_records, credit_profiles, kyc_status],\n        requires: [identity_verification, audit_events],\n        configuration: {\n            tenant_id: "default",\n            kyc_required: true,\n            credit_check_provider: "internal"\n        },\n        rules: [\n            {name: "kyc_required",     when: "kyc_status == pending",   action: deny},\n            {name: "duplicate_email",  when: "email_exists == true",    action: require_review},\n            {name: "credit_limit",     when: "credit_score < 300",      action: require_review}\n        ],\n        rule_engine: {type: policy, default_decision: allow},\n        ui: {shell: python, routes: [\n            {name: "Customers", path: "/customers", component: "CustomerList", permission: "customers:view"}\n        ]},\n        theme: {name: customer_theme, tokens: {accent: "#00695C"}}\n    };\n    master_data: {entities: [customer, credit_profile, kyc_document]};\n}\n\n// Layer 4: Application (depends on all three)\ncapability SalesPortal {\n    contract: {\n        id: sales_portal,\n        provides: [sales_dashboard, order_management, customer_engagement],\n        requires: [identity_verification, audit_events, customer_records, credit_profiles],\n        configuration: {tenant_id: "default", max_order_value: 5000000},\n        rules: [\n            {name: "auth_required",       when: "session_token missing",  action: deny},\n            {name: "kyc_gate",            when: "kyc_status != verified", action: deny},\n            {name: "credit_gate",         when: "available_credit < 0",   action: deny},\n            {name: "large_order",         when: "order_total > 1000000",  action: require_review},\n            {name: "approved_region",     when: "customer_region == KE",  action: allow},\n            {name: "unsupported_region",  when: "customer_region missing", action: deny}\n        ],\n        ui: {shell: python, routes: [\n            {name: "Dashboard", path: "/sales",        component: "SalesDashboard", permission: "sales:view"},\n            {name: "Orders",    path: "/sales/orders", component: "OrderList",      permission: "sales:orders"}\n        ]},\n        theme: {name: sales_theme, tokens: {accent: "#1565C0"}}\n    };\n\n    screens: {\n        SalesDashboard: {\n            route: "/sales",\n            layout: dashboard,\n            contains: [RevenueChart, TopCustomers, RecentOrders],\n            binds: [sales.summary, customers.top, orders.recent],\n            actions: [create_order, refresh, export]\n        }\n    };\n    streaming: {processor: bytewax, state: sales_event_state};\n}\n\nagent SalesAdvisor {\n    role: "sales advisor";\n    model: "openai:gpt-4.1-mini";\n    runtime: codex;\n    system: "Advise sales team on customer engagement, upsell opportunities, and deal risk.";\n    capabilities: [customer_records, sales_dashboard, order_management];\n    tools: [customer.lookup, order.history, credit.check];\n    memory: vector sales_memory;\n    configuration: {temperature: 0.2, max_turns: 8};\n}\n\nagent ComplianceAdvisor {\n    role: "compliance advisor";\n    model: "ollama:llama3.3";\n    runtime: ollama;\n    system: "Review transactions for AML and KYC compliance. Flag any concerns.";\n    capabilities: [audit_events, kyc_status];\n    tools: [kyc.verify, aml.screen, compliance.report];\n    configuration: {temperature: 0.0, max_turns: 4};\n}\n\nagent_team SalesComplianceCrew {\n    agents: [SalesAdvisor, ComplianceAdvisor];\n    flow: SalesAdvisor -> ComplianceAdvisor [condition: compliance_check_needed];\n    capabilities: [customer_records, audit_events];\n}\n\napp PlatformSuite {\n    description: "Multi-capability platform with full dependency chain";\n    capabilities: [Authentication, AuditTrail, CustomerMaster, SalesPortal];\n    agent_teams: [SalesComplianceCrew];\n    routes: ["/", "/customers", "/sales"];\n    components: {\n        auth_service:     {capability: identity_verification, route: "/auth"},\n        customer_portal:  {capability: customer_records,      route: "/customers"},\n        sales_portal:     {capability: sales_dashboard,       route: "/sales"}\n    };\n    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};\n    deployments: {default: local, container: docker, cloud: kubernetes};\n}\n'
APG_MODULE_NAME = 'platform_suite'
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
