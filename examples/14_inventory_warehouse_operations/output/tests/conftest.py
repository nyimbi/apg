"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 14: inventory warehouse with form and standalone screen entities.\n// Features: form entity type, standalone screen entity\n\nmodule inventory_warehouse version 1.0.0 {\n    description: "Warehouse operations with form and screen entity coverage";\n}\n\ntable InventoryItem {\n    sku: str;\n    description: str;\n    unit_of_measure: str = "EA";\n    quantity_on_hand: int = 0;\n    quantity_reserved: int = 0;\n    reorder_point: int = 10;\n    cost_price: decimal;\n    status: str = "active";\n    bin_location: str?;\n}\n\ntable BinLocation {\n    bin_id: str;\n    warehouse_id: str;\n    aisle: str;\n    shelf: str;\n    position: str;\n    max_capacity: int;\n    current_count: int = 0;\n    is_active: bool = true;\n}\n\ntable StockMovement {\n    movement_id: str;\n    item_sku: str;\n    from_bin: str?;\n    to_bin: str?;\n    movement_type: str;\n    quantity: int;\n    reference: str?;\n    performed_by: str;\n    performed_at: datetime;\n}\n\n// Standalone form entity — data-entry form for stock receipts\nform StockReceiptForm {\n    title: "Stock Receipt";\n    table: InventoryItem;\n    fields: [sku, quantity_on_hand, cost_price, bin_location];\n    actions: [save, save_and_add, cancel];\n}\n\n// Standalone screen entity\nscreen WarehouseMap {\n    route: "/warehouse/map";\n    title: "Warehouse Map";\n    layout: grid;\n    contains: [BinGrid, ItemSearch, MovementLog];\n    binds: [bins.all, movements.recent];\n    actions: [transfer, receive, count, adjust];\n    relationships: [\n        BinGrid -> MovementLog,\n        {from: ItemSearch, to: BinGrid, via: sku_filter}\n    ];\n}\n\ncapability WarehouseOps {\n    contract: {\n        id: warehouse_ops,\n        provides: [stock_movements, bin_management, cycle_counting],\n        requires: [inventory_items, audit_events],\n        configuration: {tenant_id: "default", warehouse_id: "WH-001"},\n        rules: [\n            {name: "bin_capacity",   when: "current_count >= max_capacity",                           action: deny},\n            {name: "valid_movement", when: "movement_type in [receipt, issue, transfer, adjustment]", action: allow},\n            {name: "sku_required",   when: "item_sku missing",                                        action: deny}\n        ],\n        ui: {\n            shell: python,\n            routes: [\n                {name: "Movements", path: "/warehouse/movements", component: "MovementLog",  permission: "wh:movements"},\n                {name: "Bins",      path: "/warehouse/bins",      component: "BinManager",   permission: "wh:bins"},\n                {name: "Map",       path: "/warehouse/map",       component: "WarehouseMap", permission: "wh:map"}\n            ]\n        },\n        theme: {name: warehouse_theme, tokens: {accent: "#546E7A"}}\n    };\n}\n\nworkflow StockReceipt {\n    steps: str = "initiated -> inspection -> put_away -> confirmed";\n    human_tasks: [inspection, put_away];\n    assignments: {inspection: quality_controller, put_away: warehouse_operator};\n    guards: {\n        put_away:  "inspection_passed == true",\n        confirmed: "bin_location not missing"\n    };\n}\n\napp WarehouseOpsApp {\n    description: "Warehouse operations management";\n    capabilities: [WarehouseOps];\n    routes: ["/warehouse"];\n}\n'
APG_MODULE_NAME = 'inventory_warehouse'
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
