"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 10: complete theme, i18n, and streaming configuration.\n// Features: theme with full token set, i18n with African language codes,\n//           streaming with all fields (processor, input, output, state, window)\n\nmodule localised_inventory version 1.0.0 {\n    description: "Inventory capability with full i18n and streaming";\n}\n\ncapability LocalisedInventory {\n    contract: {\n        id: localised_inventory,\n        provides: [stock_balances, inventory_events, reorder_alerts],\n        requires: [audit_events],\n        configuration: {\n            tenant_id: "default",\n            default_warehouse: "NBO-01",\n            low_stock_threshold: 10,\n            auto_reorder: true\n        },\n        rules: [\n            {name: "no_negative_stock", when: "on_hand - reserved < 0",        action: deny},\n            {name: "reorder_required",  when: "on_hand <= low_stock_threshold", action: warn}\n        ],\n        ui: {\n            shell: python,\n            requires_theme: true,\n            routes: [\n                {name: "Stock",     path: "/inventory",          component: "StockDashboard", permission: "inventory:view"},\n                {name: "Reorders",  path: "/inventory/reorder",  component: "ReorderQueue",   permission: "inventory:reorder"},\n                {name: "Movements", path: "/inventory/movements",component: "MovementLog",    permission: "inventory:movements"}\n            ]\n        },\n        theme: {\n            name: inventory_theme,\n            tokens: {\n                "color.primary":      "#1B5E20",\n                "color.accent":       "#F57F17",\n                "color.surface":      "#FFFFFF",\n                "color.background":   "#F5F5F5",\n                "color.error":        "#C62828",\n                "color.warning":      "#E65100",\n                "color.success":      "#2E7D32",\n                "color.text.primary": "#212121",\n                "color.text.muted":   "#757575",\n                "border.radius":      "4px",\n                "shadow.card":        "0 2px 4px rgba(0,0,0,0.12)",\n                "density":            "comfortable",\n                "font.family":        "Inter, sans-serif",\n                "font.size.base":     "14px"\n            },\n            components: {\n                stock_level_badge: {variant: "filled",       color_scheme: "status"},\n                movement_table:    {stripe: true,            compact: false},\n                reorder_alert:     {icon: "alert-circle",    position: "top-right"}\n            },\n            allow_tenant_overrides: true\n        },\n        runtime: {\n            target: python,\n            streaming: {processor: bytewax}\n        }\n    };\n\n    // Full streaming configuration\n    streaming: {\n        processor: bytewax,\n        input: inventory_event_bus,\n        output: reorder_alert_stream,\n        state: inventory_state,\n        window: 15min\n    };\n\n    // Comprehensive i18n — English plus major African languages\n    i18n: {\n        supported_languages: [\n            en, sw, fr, pt, ar,\n            rw, rn, yo, ig, ha,\n            tw, ak, ee, gaa, fon,\n            zu, xh, st, tn, ss, ve, nr, nso, ts,\n            am, ti, so, om,\n            ln, lu, kg, sg\n        ],\n        default_language: en,\n        fallback_language: en\n    };\n\n    master_data: {entities: [warehouse, bin_location, item_category, unit_of_measure]};\n}\n\napp LocalisedInventoryApp {\n    description: "Localised inventory management with streaming";\n    capabilities: [LocalisedInventory];\n    routes: ["/inventory"];\n}\n'
APG_MODULE_NAME = 'localised_inventory'
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
