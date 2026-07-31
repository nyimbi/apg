"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 04: order fulfillment with comprehensive workflow coverage.\n// Features: workflow with steps, human_tasks, assignments, guards,\n//           timers, waits, retry_policy, compensation; multiple workflows\n\nmodule order_fulfillment version 1.0.0 {\n    description: "Order fulfilment with full workflow coverage";\n}\n\ntable Order {\n    order_number: str;\n    customer_id: str;\n    status: str = "pending";\n    total: decimal;\n    priority: str = "standard";\n}\n\ntable Shipment {\n    shipment_id: str;\n    order_id: str;\n    carrier: str;\n    tracking_number: str?;\n    status: str = "pending";\n    dispatched_at: datetime?;\n    delivered_at: datetime?;\n}\n\ntable Return {\n    return_id: str;\n    order_id: str;\n    reason: str;\n    status: str = "requested";\n    refund_amount: decimal = 0.0;\n}\n\ntable PaymentRetry {\n    retry_id: str;\n    order_id: str;\n    attempt: int = 1;\n    status: str = "pending";\n}\n\nworkflow OrderFulfilment {\n    // Full state machine: new order → picked → packed → dispatched → delivered\n    steps: str = "received -> payment_authorised -> picking -> packing -> dispatched -> delivered";\n    human_tasks: [picking, packing];\n    assignments: {\n        picking: warehouse_operator,\n        packing: packing_station,\n    };\n    guards: {\n        payment_authorised: "payment_status == approved and fraud_score < 0.7",\n        dispatched: "all_items_packed and carrier_confirmed",\n    };\n    // SLA timers\n    timers: {\n        picking: "PT4H",\n        packing: "PT2H",\n        dispatched: "PT24H",\n    };\n    // External event waits\n    waits: {\n        payment_authorised: payment_confirmed,\n        delivered: delivery_confirmation_received,\n    };\n    // Retry policies for automated steps\n    retry_policy: {\n        payment_authorised: "3",\n        dispatched: "2",\n    };\n    // Compensation actions on cancellation\n    compensation: {\n        payment_authorised: void_payment_authorisation,\n        picking: return_items_to_stock,\n        dispatched: initiate_carrier_recall,\n    };\n}\n\nworkflow ReturnProcess {\n    steps: str = "requested -> approved -> received -> inspected -> refunded";\n    human_tasks: [approved, inspected];\n    assignments: {\n        approved: returns_manager,\n        inspected: quality_inspector,\n    };\n    guards: {\n        approved: "return_reason in [defective, wrong_item, not_as_described]",\n        refunded: "inspection_passed == true",\n    };\n    timers: {\n        approved: "PT48H",\n        inspected: "PT24H",\n    };\n    compensation: {\n        refunded: reverse_refund,\n    };\n}\n\napp FulfilmentApp {\n    description: "Order fulfilment and returns management";\n    routes: ["/orders", "/shipments", "/returns"];\n}\n'
APG_MODULE_NAME = 'order_fulfillment'
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
