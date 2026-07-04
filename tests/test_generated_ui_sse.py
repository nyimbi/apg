"""Generated live UI and SSE regressions."""

from __future__ import annotations

from compiler.compiler import compile_apg_file


def _generated_namespace() -> dict[str, object]:
	result = compile_apg_file("examples/06_support_agent_team/main.apg")
	assert result.success, result.errors
	assert "static/apg-sse.js" in result.generated_files
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)
	return namespace


def test_generated_events_endpoint_streams_sse_ready_event():
	namespace = _generated_namespace()
	app = namespace["_flask_app"]

	with app.test_client() as client:
		response = client.get(
			"/events?topics=events",
			headers={"Accept": "text/event-stream"},
			buffered=False,
		)
		assert response.status_code == 200
		assert response.content_type.startswith("text/event-stream")
		chunks = []
		iterator = iter(response.response)
		for _ in range(2):
			chunks.append(next(iterator).decode("utf-8"))

	assert ": connected" in "".join(chunks)
	assert "event: apg-ready" in "".join(chunks)


def test_generated_agent_output_sanitizer_escapes_hostile_html():
	namespace = _generated_namespace()
	render = namespace["_sanitize_agent_markdown"]

	html = render('<img src=x onerror=alert(1)> **safe** `code`\n- item')

	assert "<img" not in html
	assert "&lt;img" in html
	assert "<strong>safe</strong>" in html
	assert "<code>code</code>" in html
	assert "<li>item</li>" in html
