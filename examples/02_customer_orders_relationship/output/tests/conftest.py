"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 02: customer and order entities with relationship convention.\n// Features: multiple tables, List[str], Dict[str,str], Dict[str,Any],\n//           explicit str | None, default values, decimal\n\nmodule customer_orders version 1.0.0 {\n    description: "Customer and order data model with relationships";\n}\n\ntable Customer {\n    customer_number: str;\n    legal_name: str;\n    email: str;\n    phone: str?;\n    segment: str = "standard";\n    credit_limit: decimal = 10000.0;\n    is_active: bool = true;\n    tags: List[str];\n    attributes: Dict[str, str];\n}\n\ntable Order {\n    order_number: str;\n    customer_id: str;          // foreign-key to Customer by convention\n    order_date: date;\n    delivery_date: date?;\n    status: str = "draft";\n    currency: str = "KES";\n    subtotal: decimal = 0.0;\n    tax: decimal = 0.0;\n    total: decimal = 0.0;\n    notes: str | None;\n    line_items: List[str];     // list of line item IDs\n    metadata: Dict[str, Any];\n}\n\ntable OrderLine {\n    line_id: str;\n    order_id: str;             // foreign-key to Order\n    product_code: str;\n    description: str;\n    quantity: int = 1;\n    unit_price: decimal;\n    discount_pct: float = 0.0;\n    line_total: decimal = 0.0;\n    is_taxable: bool = true;\n}\n\napp CustomerOrders {\n    description: "Customer and order management";\n    routes: ["/customers", "/orders"];\n}\n'
APG_MODULE_NAME = 'customer_orders'
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
