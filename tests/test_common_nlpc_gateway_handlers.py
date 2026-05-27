"""Regressions for executable NLPC gateway service handlers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NLPC_PATH = REPO_ROOT / "capabilities" / "common" / "nlpc"
API_GATEWAY_PATH = NLPC_PATH / "api_gateway.py"


def _load_api_gateway(monkeypatch: pytest.MonkeyPatch) -> Any:
	production_operations = types.SimpleNamespace(
		ProductionOperationsManager=object,
		get_operations_manager=lambda: None,
	)
	monkeypatch.setitem(sys.modules, "production_operations", production_operations)
	monkeypatch.syspath_prepend(str(NLPC_PATH))
	sys.modules.pop("api_gateway", None)
	return importlib.import_module("api_gateway")


@pytest.mark.asyncio
async def test_registered_nlpc_gateway_service_handler_executes_async_route(monkeypatch: pytest.MonkeyPatch):
	api_gateway = _load_api_gateway(monkeypatch)
	monkeypatch.setattr(api_gateway.APIGateway, "_start_background_tasks", lambda self: None)
	gateway = api_gateway.APIGateway("tenant-a")

	async def compose_agent(request: Any, endpoint: Any, service_instance: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
		assert endpoint.handler_function == "compose_agent"
		assert service_instance["host"] == "local"
		return (
			202,
			{
				"agent_id": request.body["agent_id"],
				"provider": request.body["provider"],
				"theme": request.body["theme"],
				"tenant_id": gateway.tenant_id,
			},
			{"X-APG-Service": "agent-composer"},
		)

	gateway.register_service("agent_composer", {"handlers": {"compose_agent": compose_agent}})
	gateway.register_endpoint(api_gateway.APIEndpoint(
		path="/agents/compose",
		method="POST",
		version=api_gateway.APIVersion.V1,
		handler_function="compose_agent",
		service_name="agent_composer",
		auth_required=False,
		rate_limit_enabled=False,
	))

	response = await gateway.process_request(
		"POST",
		"/agents/compose",
		body={
			"agent_id": "codex-reviewer",
			"provider": "codex",
			"theme": "dark-console",
		},
	)

	assert response.status_code == 202
	assert response.service_used == "agent_composer"
	assert response.headers == {"X-APG-Service": "agent-composer"}
	assert response.body == {
		"agent_id": "codex-reviewer",
		"provider": "codex",
		"theme": "dark-console",
		"tenant_id": "tenant-a",
	}


@pytest.mark.asyncio
async def test_registered_nlpc_gateway_default_handler_normalizes_response(monkeypatch: pytest.MonkeyPatch):
	api_gateway = _load_api_gateway(monkeypatch)
	monkeypatch.setattr(api_gateway.APIGateway, "_start_background_tasks", lambda self: None)
	gateway = api_gateway.APIGateway("tenant-a")

	def list_capabilities(request: Any, endpoint: Any, service_instance: dict[str, Any]) -> list[str]:
		assert request.path == "/capabilities"
		assert service_instance["health"] == "healthy"
		return ["nlpc", "agent_composition"]

	gateway.register_service_handler("capability_catalog", "*", list_capabilities)
	gateway.register_endpoint(api_gateway.APIEndpoint(
		path="/capabilities",
		method="GET",
		version=api_gateway.APIVersion.V1,
		handler_function="list",
		service_name="capability_catalog",
		auth_required=False,
		rate_limit_enabled=False,
	))

	response = await gateway.process_request("GET", "/capabilities")

	assert response.status_code == 200
	assert response.service_used == "capability_catalog"
	assert response.body == {"items": ["nlpc", "agent_composition"]}


def test_nlpc_gateway_no_longer_contains_service_handler_placeholder():
	source = API_GATEWAY_PATH.read_text(encoding="utf-8")
	assert "Service handler not implemented" not in source
	assert "register_service_handler" in source
