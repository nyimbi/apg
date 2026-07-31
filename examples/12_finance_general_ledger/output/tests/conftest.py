"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 12: finance general ledger with comprehensive ERP metadata.\n// Features: erp_modules, approvals (levels/thresholds/approvers/segregation),\n//           master_data with ownership and deduplication, business_rules\n\nmodule finance_gl version 1.0.0 {\n    description: "General ledger with full ERP metadata coverage";\n}\n\ntable Account {\n    account_code: str;\n    account_name: str;\n    account_type: str;      // asset | liability | equity | revenue | expense\n    parent_code: str?;\n    currency: str = "KES";\n    is_active: bool = true;\n    is_control: bool = false;\n    normal_balance: str;    // debit | credit\n}\n\ntable JournalEntry {\n    journal_id: str;\n    reference: str;\n    period: str;            // YYYY-MM\n    entry_date: date;\n    description: str;\n    status: str = "draft";  // draft | posted | reversed\n    total_debit: decimal = 0.0;\n    total_credit: decimal = 0.0;\n    posted_by: str?;\n    posted_at: datetime?;\n}\n\ntable JournalLine {\n    line_id: str;\n    journal_id: str;\n    account_code: str;\n    debit: decimal = 0.0;\n    credit: decimal = 0.0;\n    cost_centre: str?;\n    project: str?;\n    memo: str?;\n}\n\ncapability GeneralLedger {\n    contract: {\n        id: general_ledger,\n        provides: [chart_of_accounts, journal_entries, period_close, trial_balance, financial_statements],\n        requires: [audit_events, auth],\n        configuration: {\n            tenant_id: "default",\n            fiscal_year_start: "01-01",\n            base_currency: "KES",\n            supported_currencies: ["KES", "UGX", "TZS", "USD", "EUR"],\n            decimal_places: 2\n        },\n        configuration_schema: {\n            required: ["tenant_id", "base_currency"]\n        },\n        rules: [\n            {name: "journal_balanced",     when: "total_debit != total_credit",                            action: deny},\n            {name: "open_period_required", when: "period_status != open",                                  action: deny},\n            {name: "posting_authorised",   when: "user_role in [accountant, controller, cfo]",             action: allow},\n            {name: "large_entry_review",   when: "total_debit > 1000000",                                  action: require_review},\n            {name: "period_close_check",   when: "reconciled == false and period_close_initiated == true", action: deny}\n        ],\n        ui: {\n            shell: python,\n            routes: [\n                {name: "Chart of Accounts", path: "/gl/accounts",     component: "AccountList",  permission: "gl:accounts"},\n                {name: "Journal Entries",   path: "/gl/journals",     component: "JournalList",  permission: "gl:journals"},\n                {name: "Trial Balance",     path: "/gl/trial",        component: "TrialBalance", permission: "gl:reports"},\n                {name: "Period Close",      path: "/gl/period-close", component: "PeriodClose",  permission: "gl:close"}\n            ]\n        },\n        theme: {name: finance_theme, tokens: {accent: "#1A237E", "color.primary": "#283593"}}\n    };\n\n    // Full ERP module coverage\n    erp_modules: [\n        finance, general_ledger, accounts_payable, accounts_receivable,\n        fixed_assets, project_accounting, reporting\n    ];\n\n    // Full approvals structure\n    approvals: {\n        levels: 3,\n        thresholds: {\n            level1: 100000,\n            level2: 500000,\n            level3: 1000000\n        },\n        approvers: [finance_manager, controller, cfo],\n        segregation_of_duties: true,\n        escalation: "finance_director"\n    };\n\n    // Master data with ownership and governance\n    master_data: {\n        entities: [\n            account, cost_centre, financial_period, currency, exchange_rate,\n            budget, budget_line, project, department\n        ],\n        ownership: {\n            account:          finance,\n            cost_centre:      operations,\n            financial_period: finance,\n            currency:         finance\n        },\n        deduplication: account_code,\n        governance: {\n            type: deterministic,\n            rules: [{name: "unique_code", when: "account_code exists", action: deny}]\n        }\n    };\n\n    // Business rules (separate from contract rules)\n    business_rules: [\n        {name: "balanced_entry",  when: "debits != credits",          action: deny},\n        {name: "valid_period",    when: "period_status == closed",    action: deny},\n        {name: "active_account",  when: "account_is_active == false", action: deny},\n        {name: "budget_exceeded", when: "commitment > budget_line",   action: require_review}\n    ];\n\n    i18n: {supported_languages: [en, sw, fr], default_language: en, fallback_language: en};\n    streaming: {processor: bytewax, input: gl_events, state: gl_state};\n}\n\napp FinanceGL {\n    description: "General ledger financial application";\n    capabilities: [GeneralLedger];\n    routes: ["/gl"];\n}\n'
APG_MODULE_NAME = 'finance_gl'
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
