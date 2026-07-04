"""Template coverage for generated UI routes."""

from __future__ import annotations

from compiler.compiler import compile_apg_file


EXAMPLE_20 = "examples/20_enterprise_erp_platform/main.apg"
REQUIRED_TEMPLATES = {
	"workflow_list.html.j2",
	"workflow_wizard.html.j2",
	"database_catalog.html.j2",
	"agent_console.html.j2",
	"capability_console.html.j2",
	"debug_console.html.j2",
}


def _generated_namespace() -> dict[str, object]:
	result = compile_apg_file(EXAMPLE_20)
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)
	return namespace


def test_generated_ui_routes_resolve_required_templates():
	namespace = _generated_namespace()
	templates = namespace["APG_UI_TEMPLATES"]
	assert REQUIRED_TEMPLATES <= set(templates)

	workflows = namespace["APP_WORKFLOWS"]
	entity_name = next(iter(workflows))
	workflow_id = workflows[entity_name][0]["id"]

	cases = [
		namespace["_ui_workflow_list_html"](),
		namespace["_ui_workflow_wizard_html"](entity_name, workflow_id),
		namespace["_ui_database_catalog_html"](),
		namespace["_ui_debug_html"](),
	]

	for status, html in cases:
		assert status in {200, 422}
		assert "This application requires Jinja2" not in html
		assert "<main" in html

	for template_name in ("agent_console.html.j2", "capability_console.html.j2"):
		rendered = namespace["_render_template"](
			template_name,
			name="Probe",
			team=False,
			action="/ui/agents/Probe/invoke",
			safe_name="Probe",
			description_json="{}",
			result=None,
			result_items=[],
			result_json="",
			error="",
		)
		assert rendered is not None
		assert "This application requires Jinja2" not in rendered
