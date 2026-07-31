"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 03: inventory item with comprehensive type coverage.\n// Features: all scalar types (str, int, float, decimal, bool, bytes,\n//           datetime, date, time, Any), optional variants\n\nmodule inventory_records version 1.0.0 {\n    description: "Inventory item record demonstrating all scalar types";\n}\n\ntable InventoryItem {\n    // String types\n    sku: str;\n    description: str;\n    barcode: str?;\n    notes: str | None;\n\n    // Numeric types\n    quantity_on_hand: int = 0;\n    quantity_reserved: int = 0;\n    quantity_on_order: int = 0;\n    unit_price: decimal;\n    cost_price: decimal;\n    weight_kg: float = 0.0;\n    reorder_point: int = 10;\n    reorder_quantity: int = 50;\n\n    // Boolean flags\n    is_active: bool = true;\n    is_serialised: bool = false;\n    is_hazardous: bool = false;\n    requires_refrigeration: bool = false;\n\n    // Date/time types\n    created_at: datetime;\n    last_counted_at: datetime?;\n    best_before: date?;\n    reorder_time: time?;\n\n    // Binary data — e.g. product image thumbnail\n    image_thumbnail: bytes?;\n\n    // Collections\n    categories: List[str];\n    attributes: Dict[str, str];\n    dimensions: Dict[str, float];\n    metadata: Dict[str, Any];\n\n    // Status\n    status: str = "active";\n    warehouse_location: str?;\n}\n\ntable StockMovement {\n    movement_id: str;\n    item_id: str;\n    movement_type: str;      // receipt | issue | adjustment | transfer\n    quantity: int;\n    reference: str?;\n    notes: str | None;\n    movement_date: datetime;\n    performed_by: str;\n}\n\napp InventoryRecords {\n    description: "Inventory management";\n    routes: ["/inventory", "/stock-movements"];\n}\n'
APG_MODULE_NAME = 'inventory_records'
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
