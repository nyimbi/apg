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
