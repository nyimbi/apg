"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 01: typed customer record with optional and union fields.\n// Features: str, int, float, decimal, bool, datetime, date, str?, str | None,\n//           default values, List[str], Dict[str, Any]\n\nmodule customer_records version 1.0.0 {\n    description: "Customer data model with full type coverage";\n}\n\ntable Customer {\n    // Required scalar fields\n    customer_number: str;\n    legal_name: str;\n    email: str;\n    phone: str;\n    // Optional field — may be absent (null)\n    secondary_email: str?;\n    // Union type — explicit null alternative\n    company_name: str | None;\n    // Numeric types\n    credit_limit: decimal = 50000.0;\n    loyalty_points: int = 0;\n    discount_rate: float = 0.0;\n    // Boolean with default\n    is_active: bool = true;\n    is_verified: bool = false;\n    // Date/time types\n    registered_at: datetime;\n    date_of_birth: date?;\n    // Collection types\n    tags: List[str];\n    preferences: Dict[str, str];\n    // Status with a default\n    status: str = "prospect";\n    // Arbitrary metadata\n    metadata: Dict[str, Any];\n}\n\napp CustomerRecords {\n    description: "Customer records application";\n    capabilities: [];\n    routes: ["/customers"];\n}\n'
APG_MODULE_NAME = 'customer_records'
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
