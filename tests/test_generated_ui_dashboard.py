"""Generated dashboard and chart regressions."""

from __future__ import annotations

import json
import re

from compiler.compiler import compile_apg_file


def _json_scripts(html: str) -> list[dict]:
	specs: list[dict] = []
	for match in re.finditer(r'<script id="[^"]+" type="application/json">(.*?)</script>', html, re.DOTALL):
		specs.append(json.loads(match.group(1)))
	return specs


def test_generated_dashboard_chart_specs_are_valid_for_example_20():
	result = compile_apg_file("examples/20_enterprise_erp_platform/main.apg")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	status, html = namespace["_ui_payload"]("/ui")
	assert status == 200
	specs = _json_scripts(html)
	assert specs
	assert all("type" in spec and "data" in spec for spec in specs)

	status_entities = []
	for entity in namespace["ENTITIES"]:
		fields = entity.get("fields") or []
		if any(str(field.get("name", "")).lower() in {"status", "state", "stage", "phase"} for field in fields):
			status_entities.append(str(entity["name"]))
	assert status_entities
	assert any(spec["type"] == "donut" for spec in specs)

	status, analytics_html = namespace["_ui_payload"](
		f"/ui/entities/{status_entities[0]}",
		{"view": ["analytics"]},
	)
	assert status == 200
	analytics_specs = _json_scripts(analytics_html)
	assert {spec["type"] for spec in analytics_specs} >= {"line", "donut"}


def test_generated_home_dashboard_prioritizes_workspace_actions():
	result = compile_apg_file("examples/20_enterprise_erp_platform/main.apg")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	status, html = namespace["_ui_payload"]("/ui")

	assert status == 200
	assert 'aria-label="Workspace shortcuts"' in html
	assert "Start with Vendor" in html
	assert "Create the first record" in html
	assert "Open workflows" in html
	assert "Open agent console" in html
	assert "2 agent(s), 1 team(s)" in html
	assert "2</a>\n      <span class=\"apg-stat-label\">Capabilities</span>" in html
	assert "ERPAnalyst" in html
	assert "PrivacyAgent" in html
	assert "ERPAdvisors" in html


def test_generated_entity_list_exposes_saved_views_and_filter_state():
	result = compile_apg_file("examples/20_enterprise_erp_platform/main.apg")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	create_status, create_payload = namespace["create_record"](
		"Vendor",
		{
			"vendor_number": "V-001",
			"legal_name": "Acme Components",
			"tax_id": "KE-123",
			"payment_terms": "net_30",
			"currency": "KES",
			"bank_account": "100200",
			"status": "active",
			"country": "KE",
		},
	)
	assert create_status == 201, create_payload

	status, html = namespace["_ui_payload"]("/ui/entities/Vendor")
	assert status == 200
	assert 'aria-label="Saved views"' in html
	assert "All records" in html
	assert "Recently added" in html
	assert "Active" in html
	assert "Developer exports" in html
	assert "API JSON" in html
	assert "Record JSON" not in html
	assert "apg-table-wrap" in html
	assert "Acme Components" in html

	status, filtered = namespace["_ui_payload"](
		"/ui/entities/Vendor",
		{"filter.status": ["active"], "sort": ["id"], "dir": ["desc"], "q": ["Acme"]},
	)
	assert status == 200
	assert "Status: active" in filtered
	assert "Search: Acme" in filtered
	assert "Sort: id desc" in filtered
	assert "filter.status=active" in filtered
	assert "q=Acme" in filtered
	assert "dir=asc&amp;filter.status=active&amp;q=Acme&amp;sort=id" in filtered


def test_generated_entity_analytics_uses_record_dates_and_drilldowns():
	result = compile_apg_file("examples/20_enterprise_erp_platform/main.apg")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	for record in [
		{
			"vendor_number": "V-001",
			"legal_name": "Acme Components",
			"tax_id": "KE-123",
			"payment_terms": "net_30",
			"currency": "KES",
			"bank_account": "100200",
			"status": "active",
			"country": "KE",
			"created_at": "2026-07-01",
		},
		{
			"vendor_number": "V-002",
			"legal_name": "Nairobi Supplies",
			"tax_id": "KE-456",
			"payment_terms": "net_15",
			"currency": "KES",
			"bank_account": "100201",
			"status": "suspended",
			"country": "KE",
			"created_at": "2026-07-02",
		},
		{
			"vendor_number": "V-003",
			"legal_name": "Lagos Parts",
			"tax_id": "NG-789",
			"payment_terms": "net_30",
			"currency": "NGN",
			"bank_account": "100202",
			"status": "active",
			"country": "NG",
			"created_at": "2026-07-03",
		},
	]:
		create_status, create_payload = namespace["create_record"]("Vendor", record)
		assert create_status == 201, create_payload

	status, html = namespace["_ui_payload"]("/ui/entities/Vendor", {"view": ["analytics"]})
	assert status == 200
	assert 'aria-label="Vendor analytics summary"' in html
	assert "Grouped by created_at" in html
	assert "Largest segment" in html
	assert "View active records" in html
	assert "filter.status=active" in html
	assert "apg-status-row" in html

	specs = _json_scripts(html)
	line_spec = next(spec for spec in specs if spec["type"] == "line")
	donut_spec = next(spec for spec in specs if spec["type"] == "donut")
	assert len(line_spec["data"]) == 30
	assert [point for point in line_spec["data"] if point["y"] == 1][-3:] == [
		{"x": "2026-07-01", "y": 1},
		{"x": "2026-07-02", "y": 1},
		{"x": "2026-07-03", "y": 1},
	]
	assert donut_spec["data"] == [
		{"label": "active", "value": 2},
		{"label": "suspended", "value": 1},
	]


def test_generated_kanban_renders_board_and_keyboard_move_controls():
	result = compile_apg_file("examples/20_enterprise_erp_platform/main.apg")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	for index, status_value in enumerate(["active", "active", "active", "active", "suspended"], start=1):
		create_status, create_payload = namespace["create_record"](
			"Vendor",
			{
				"vendor_number": f"V-{index:03d}",
				"legal_name": f"Vendor {index}",
				"tax_id": f"KE-{index}",
				"payment_terms": "net_30",
				"currency": "KES",
				"bank_account": str(1000 + index),
				"status": status_value,
				"country": "KE",
			},
		)
		assert create_status == 201, create_payload

	status, html = namespace["_ui_payload"]("/ui/entities/Vendor", {"view": ["kanban"]})
	assert status == 200
	assert "Grouped by status" in html
	assert 'aria-label="Board summary"' in html
	assert "WIP guide" in html
	assert "Above WIP guide" in html
	assert "apg-kanban-card" in html
	assert "apg-kanban-move" in html
	assert 'name="return_view" value="kanban"' in html
	assert "filter.status=active" in html
	assert "This application requires Jinja2" not in html
	assert 'aria-label="Vendor table controls"' not in html

	move_status, move_payload = namespace["_ui_post_payload"](
		"/ui/entities/Vendor/records/1",
		{"record": {"status": "suspended", "return_view": "kanban"}},
	)
	assert move_status == 303
	assert move_payload["location"] == "/ui/entities/Vendor?view=kanban"


def test_generated_record_detail_renders_related_activity_and_navigation():
	result = compile_apg_file("examples/02_customer_orders_relationship/main.apg")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	for index in [1, 2]:
		create_status, create_payload = namespace["create_record"](
			"Customer",
			{
				"customer_number": f"C-{index:03d}",
				"legal_name": f"Customer {index}",
				"email": f"c{index}@example.com",
				"phone": "+254700000001",
				"segment": "standard",
				"credit_limit": 10000,
				"is_active": True,
				"tags": ["retail"],
				"attributes": {"region": "EA"},
			},
		)
		assert create_status == 201, create_payload

	order_status, order_payload = namespace["create_record"](
		"Order",
		{
			"order_number": "O-001",
			"customer_id": "1",
			"order_date": "2026-07-01",
			"delivery_date": "2026-07-05",
			"status": "open",
			"currency": "KES",
			"subtotal": 1000,
			"tax": 250,
			"total": 1250,
			"notes": "first order",
			"line_items": [{"sku": "SKU-1"}],
			"metadata": {"channel": "direct"},
		},
	)
	assert order_status == 201, order_payload

	status, html = namespace["_ui_record_detail_html"]("Customer", "1")
	assert status == 200
	assert "Record Details" in html
	assert "Customer 1" in html
	assert "Related" in html
	assert "Activity" in html
	assert "Copy link" in html
	assert "Next" in html
	assert "Order" in html
	assert "View filtered" in html
	assert "filter.customer_id=1" in html
	assert "Save Note" in html
	assert "<pre>" not in html


def test_generated_forms_use_native_validation_and_contextual_errors():
	result = compile_apg_file("examples/02_customer_orders_relationship/main.apg")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	status, html = namespace["_ui_payload"]("/ui/entities/Customer")
	assert status == 200
	assert 'id="apg-create-form"' in html
	assert 'name="email" type="email"' in html
	assert 'name="tags" rows="3"' in html
	assert "Required JSON value" in html
	assert "Discard this draft?" in html
	assert "window.confirm" not in html

	error_status, error_payload = namespace["_ui_post_payload"](
		"/ui/entities/Customer/records",
		{"record": {"customer_number": "C-001"}},
	)
	assert error_status == 422
	assert "html" in error_payload
	assert "Customer table controls" in error_payload["html"]
	assert "legal_name is required" in error_payload["html"]
	assert "attributes is required" in error_payload["html"]

	create_status, create_payload = namespace["create_record"](
		"Customer",
		{
			"customer_number": "C-001",
			"legal_name": "Asha Retail",
			"email": "asha@example.com",
			"phone": "+254700000001",
			"segment": "standard",
			"credit_limit": 10000,
			"is_active": True,
			"tags": ["retail"],
			"attributes": {"region": "EA"},
		},
	)
	assert create_status == 201, create_payload

	status, numeric_fragment = namespace["_ui_field_edit_html"]("Customer", "1", "credit_limit")
	assert status == 200
	assert 'type="number"' in numeric_fragment
	assert 'step="any"' in numeric_fragment

	status, json_fragment = namespace["_ui_field_edit_html"]("Customer", "1", "tags")
	assert status == 200
	assert "<textarea" in json_fragment


def test_generated_workflow_wizard_advances_sequentially_and_records_runs():
	result = compile_apg_file("examples/01_minimal_customer_records/main.apg")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	status, html = namespace["_ui_payload"]("/ui/workflows/Customer/create_customer")
	assert status == 200
	assert 'action="/ui/workflows/Customer/create_customer/step/0"' in html
	assert "Step 1 of 6: Identity" in html

	status, payload = namespace["_ui_post_payload"](
		"/ui/workflows/Customer/create_customer/step/0",
		{
			"customer_number": "WF-1001",
			"email": "workflow@example.com",
			"phone": "+254711111111",
			"status": "active",
		},
	)
	assert status == 200
	assert "Step 2 of 6: Core Details" in payload["html"]
	assert "Step 3 of 6" not in payload["html"]

	final_status, final_payload = namespace["_ui_post_payload"](
		"/ui/workflows/Customer/create_customer/step/5",
		{
			"__acc_customer_number": "WF-1001",
			"__acc_email": "workflow@example.com",
			"__acc_phone": "+254711111111",
			"__acc_status": "active",
			"__acc_credit_limit": "12",
			"__acc_loyalty_points": "12",
			"__acc_discount_rate": "12",
			"__acc_registered_at": "2026-07-04",
			"__acc_legal_name": "Workflow Co",
			"__acc_secondary_email": "ops@example.com",
			"__acc_company_name": "Workflow Co",
			"__acc_is_active": "true",
			"__acc_is_verified": "true",
			"date_of_birth": "2026-07-04",
			"tags": "[\"workflow\"]",
			"preferences": "{}",
			"metadata": "{}",
		},
	)
	assert final_status == 200
	assert "Recorded run" in final_payload["html"]
	assert "Inspect run" in final_payload["html"]
	assert "Open created record" in final_payload["html"]

	runs = namespace["list_workflow_runs"]()
	assert len(runs) == 1
	assert runs[0]["id"] == "workflow-run-1"
	assert runs[0]["status"] == "completed"
	assert runs[0]["entity"] == "Customer"
	assert len(runs[0]["trace"]) == 6
	assert runs[0]["record"]["tags"] == ["workflow"]
	assert runs[0]["record"]["preferences"] == {}

	status, list_html = namespace["_ui_payload"]("/ui/workflows")
	assert status == 200
	assert "1 recorded runs" in list_html
	assert "Recent runs" in list_html
	assert "workflow-run-1" in list_html
