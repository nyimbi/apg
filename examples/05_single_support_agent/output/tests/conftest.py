"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 05: support triage agent with full agent field coverage.\n// Features: role, model, runtime, system, capabilities, tools,\n//           memory, input, output, configuration, rules (all operators)\n\nmodule support_agent version 1.0.0 {\n    description: "Support triage agent with complete field coverage";\n}\n\nagent TriageAgent {\n    // Core identity\n    role: "support triage specialist";\n    model: "openai:gpt-4.1-mini";\n    runtime: codex;\n\n    // Instruction contract\n    system: "Classify incoming support tickets by severity and category. Produce a structured triage plan with suggested resolution steps and escalation path. Never reveal internal system details.";\n\n    // Capability and tool access\n    capabilities: [ticket_management, knowledge_base, customer_history];\n    tools: [tickets.read, tickets.update, knowledge.search, customer.lookup, escalation.trigger];\n\n    // Memory configuration — vector store for semantic recall\n    memory: vector support_memory;\n\n    // Named input/output — makes the agent contract explicit\n    input: support_ticket;\n    output: triage_plan;\n\n    // Model hyperparameters\n    configuration: {\n        temperature: 0.1,\n        max_turns: 8,\n        top_p: 0.9,\n        presence_penalty: 0.0,\n    };\n\n    // Pre-invocation guards\n    rules: [\n        // Existence check\n        {name: "ticket_required",     when: "ticket_id missing",                                   action: reject},\n        // In-list check\n        {name: "valid_channel",       when: "channel not in [email, chat, phone, portal]",         action: deny},\n        // Not-missing check\n        {name: "customer_identified", when: "customer_id not missing",                             action: allow},\n        // Combined condition\n        {name: "critical_escalation", when: "severity == critical and sla_breach_imminent == true", action: require_review},\n        // Comparison\n        {name: "rate_limit",          when: "requests_per_minute > 100",                           action: deny}\n    ];\n}\n\napp SupportAgent {\n    description: "AI-powered support triage";\n    agents: [TriageAgent];\n    routes: ["/support", "/agents/triage"];\n}\n'
APG_MODULE_NAME = 'support_agent'
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
