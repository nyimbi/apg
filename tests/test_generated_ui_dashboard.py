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
