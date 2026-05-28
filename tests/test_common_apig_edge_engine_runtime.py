"""Executable APIG edge-engine checks for local upstream and WASM transforms."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


APIG_DIR = Path(__file__).resolve().parent.parent / "capabilities" / "common" / "apig"
sys.path.insert(0, str(APIG_DIR))

from edge_engine import APGEdgeEngine  # noqa: E402
from models import AgHttpRequest, AgHttpResponse, AgWasmModule, HttpMethod  # noqa: E402


def test_edge_engine_routes_to_registered_upstream_after_wasm_transform():
	async def scenario() -> None:
		engine = APGEdgeEngine("tenant_edge")
		await engine.initialize()

		seen: dict[str, object] = {}

		async def upstream(request: AgHttpRequest, analysis: dict[str, object]) -> AgHttpResponse:
			seen["path"] = request.path
			seen["headers"] = dict(request.headers)
			seen["traffic_class"] = analysis["traffic_class"]
			return AgHttpResponse(
				request_id=request.id,
				status_code=202,
				headers={"Content-Type": "application/json"},
				body=json.dumps({"handled": True, "path": request.path}).encode(),
			)

		engine.register_upstream_handler("orders", upstream, path_prefix="/edge")
		module = AgWasmModule(
			name="tenant request transform",
			wasm_binary_path="/tmp/nonexistent-transform.wasm",
			tenant_id="tenant_edge",
			created_by="test",
			configuration={
				"request_transform": {
					"path_prefix": "/edge",
					"headers": {"X-Wasm-Transform": "applied"},
				}
			},
		)
		assert await engine.load_wasm_module(module) is True

		request = AgHttpRequest(
			method=HttpMethod.GET,
			path="/orders/123",
			headers={"Accept": "application/json"},
			client_ip="203.0.113.10",
			user_agent="APGTest/1.0",
			tenant_id="tenant_edge",
		)
		response = await engine.process_request(request, wasm_module_id=module.id)

		assert response.status_code == 202
		assert response.headers["X-Upstream-Service"] == "orders"
		assert response.headers["X-Edge-Processed"] == "true"
		assert seen["path"] == "/edge/orders/123"
		assert seen["headers"]["X-Wasm-Transform"] == "applied"
		assert seen["traffic_class"] in {"read_api", "web_request"}

	asyncio.run(scenario())


def test_edge_engine_blocks_executable_security_threats():
	async def scenario() -> None:
		engine = APGEdgeEngine("tenant_edge")
		await engine.initialize()
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path="/api/search",
			query_string="q=DROP TABLE users",
			headers={},
			client_ip="203.0.113.20",
			tenant_id="tenant_edge",
		)

		response = await engine.process_request(request)

		assert response.status_code == 403
		assert "Destructive SQL pattern detected" in response.headers["X-Blocked-Reason"]

	asyncio.run(scenario())


def test_edge_engine_returns_502_when_no_upstream_handler_exists():
	async def scenario() -> None:
		engine = APGEdgeEngine("tenant_edge")
		await engine.initialize()
		request = AgHttpRequest(
			method=HttpMethod.GET,
			path="/api/missing",
			headers={},
			client_ip="203.0.113.30",
			tenant_id="tenant_edge",
		)

		response = await engine.process_request(request)
		body = json.loads((response.body or b"{}").decode())

		assert response.status_code == 502
		assert response.headers["X-Upstream-Service"] == "none"
		assert body["error"] == "No upstream handler registered for request"

	asyncio.run(scenario())
