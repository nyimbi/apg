"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 09: comprehensive rule condition coverage.\n// Features: ==, !=, >, >=, <, <=, missing, not missing,\n//           in [...], not in [...], and, or, parenthesised combinations\n\nmodule credit_control version 1.0.0 {\n    description: "Credit control with comprehensive rule condition coverage";\n}\n\ncapability CreditControl {\n    contract: {\n        id: credit_control,\n        name: "Credit Control",\n        version: "1.0.0",\n        provides: [credit_limit_checks, hold_release, credit_scoring],\n        requires: [audit_events, customer_master],\n        configuration: {\n            default_limit: 50000,\n            currency: "KES",\n            high_risk_threshold: 0.7,\n            review_threshold: 0.5,\n            max_days_overdue: 30\n        },\n        configuration_schema: {\n            required: ["default_limit", "currency"]\n        },\n        rules: [\n            // Existence checks\n            {name: "tenant_required",         when: "tenant_id missing",                                                   action: deny,           priority: 100},\n            {name: "customer_present",        when: "customer_id not missing",                                             action: allow,          priority: 99},\n\n            // Comparison operators: >, <, >=, <=, ==, !=\n            {name: "over_limit",              when: "order_total > credit_limit",                                          action: deny,           priority: 90},\n            {name: "under_minimum",           when: "order_total < 100",                                                   action: deny,           priority: 89},\n            {name: "at_limit",                when: "order_total >= credit_limit",                                         action: require_review, priority: 88},\n            {name: "safe_order",              when: "order_total <= 1000",                                                 action: allow,          priority: 87},\n            {name: "premium_customer",        when: "customer_tier == premium",                                            action: allow,          priority: 86},\n            {name: "non_standard_currency",   when: "currency != KES",                                                    action: require_review, priority: 85},\n\n            // Set membership: in, not in\n            {name: "high_risk_region",        when: "country in [NG, SD, LY]",                                            action: require_review, priority: 80},\n            {name: "approved_currency",       when: "currency in [KES, UGX, TZS, USD, EUR]",                             action: allow,          priority: 79},\n            {name: "blocked_category",        when: "product_category not in [food, medicine, education]",                action: require_review, priority: 78},\n\n            // AND combinations\n            {name: "high_value_new_customer", when: "order_total > 100000 and customer_age_days < 90",                    action: require_review, priority: 70},\n            {name: "risk_and_overdue",        when: "risk_score > high_risk_threshold and days_overdue > max_days_overdue", action: deny,          priority: 69},\n            {name: "fraud_pattern",           when: "velocity_score > 0.8 and geo_anomaly == true",                       action: deny,           priority: 68},\n\n            // OR combinations\n            {name: "urgent_review",           when: "amount > 500000 or customer_tier == vip",                            action: require_review, priority: 60},\n            {name: "any_block_reason",        when: "is_blacklisted == true or is_frozen == true",                        action: deny,           priority: 59},\n\n            // Parenthesised combinations\n            {name: "complex_approval",        when: "(amount > 50000 or is_international == true) and risk_score < review_threshold",            action: require_review, priority: 50},\n            {name: "multi_factor_deny",       when: "(fraud_score > 0.9 or velocity_flag == true) and customer_age_days < 30",                   action: deny,           priority: 49},\n\n            // High-risk shortcut\n            {name: "high_risk_score",         when: "risk_score >= high_risk_threshold",                                  action: require_review, priority: 40}\n        ],\n        rule_engine: {type: deterministic, default_decision: allow}\n    };\n\n    approvals: {levels: 2, approvers: [credit_manager, finance_controller]};\n    master_data: {entities: [customer_credit_profile, credit_limit, payment_history]};\n}\n\napp CreditControlApp {\n    description: "Credit control and risk management";\n    capabilities: [CreditControl];\n    routes: ["/credit", "/credit/checks", "/credit/approvals"];\n}\n'
APG_MODULE_NAME = 'credit_control'
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
