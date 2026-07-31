"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 06: multi-agent support team with full agent_team coverage.\n// Features: agent_team, conditional flow, team capabilities, team rules\n\nmodule support_team version 1.0.0 {\n    description: "Planner + writer support team with conditional handoffs";\n}\n\nagent Planner {\n    role: "support planner";\n    model: "openai:gpt-4.1-mini";\n    runtime: codex;\n    system: "Break the customer\'s support request into a concrete resolution plan.";\n    tools: [tickets.read, docs.search, product.lookup];\n    memory: vector planner_memory;\n    configuration: {temperature: 0.1, max_turns: 5};\n}\n\nagent Writer {\n    role: "support writer";\n    model: "openai:gpt-4.1-mini";\n    runtime: codex;\n    system: "Write concise, empathetic customer-facing replies based on the resolution plan.";\n    tools: [tickets.update, templates.fetch];\n    configuration: {temperature: 0.3, max_turns: 4};\n}\n\nagent Reviewer {\n    role: "quality reviewer";\n    model: "openai:gpt-4.1-mini";\n    runtime: codex;\n    system: "Review the draft reply for accuracy, tone, and completeness. Flag any issues.";\n    tools: [knowledge.verify, compliance.check];\n    configuration: {temperature: 0.0, max_turns: 3};\n}\n\nagent_team SupportCrew {\n    // All three agents participate\n    agents: [Planner, Writer, Reviewer];\n\n    // Handoff flow: Planner -> Writer always; Writer -> Reviewer when draft_ready\n    flow: Planner -> Writer, Writer -> Reviewer [condition: draft_ready];\n\n    // Team-level capability access\n    capabilities: [support_response, ticket_management];\n\n    // Team configuration\n    configuration: {\n        handoff_mode: sequential,\n        max_team_turns: 15,\n        escalation_threshold: 0.5\n    };\n\n    // Team rules applied before any agent is invoked\n    rules: [\n        {name: "ticket_required",   when: "ticket_id missing",     action: deny},\n        {name: "low_confidence",    when: "confidence < 0.6",      action: require_review},\n        {name: "sensitive_content", when: "is_sensitive == true",  action: require_review}\n    ];\n}\n\napp SupportTeamApp {\n    description: "Multi-agent support response platform";\n    agent_teams: [SupportCrew];\n    routes: ["/support", "/agent-teams/support"];\n}\n'
APG_MODULE_NAME = 'support_team'
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
