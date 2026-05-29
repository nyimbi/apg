"""API helpers for APG AI agent composition."""

from __future__ import annotations

from typing import Any

from .service import AgntService


SERVICE = AgntService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.composition_summary(tenant_id)


def register_runtime(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_runtime(
		name=str(payload["name"]),
		kind=str(payload.get("kind") or "local"),
		approved=bool(payload.get("approved", True)),
		workspace_runtime=bool(payload.get("workspace_runtime", False)),
		external_runtime=bool(payload.get("external_runtime", False)),
		sandbox_policy=payload.get("sandbox_policy", "workspace-read"),
		capabilities=tuple(payload.get("capabilities") or ()),
		cost_limit=payload.get("cost_limit"),
		tenant_id=str(payload.get("tenant_id") or "default"),
	)


def request_runtime_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_runtime_approval(
		request_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		runtime_name=str(payload["runtime_name"]),
		requested_by=str(payload["requested_by"]),
		kind=str(payload.get("kind") or "external"),
		workspace_runtime=bool(payload.get("workspace_runtime", False)),
		sandbox_policy=payload.get("sandbox_policy", "workspace-read"),
		capabilities=tuple(payload.get("capabilities") or ()),
		cost_limit=payload.get("cost_limit"),
	)


def decide_runtime_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_runtime_approval(
		request_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def register_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		model=str(payload.get("model") or ""),
		runtime=str(payload.get("runtime") or "local"),
		system_prompt=str(payload.get("system_prompt") or ""),
		tool_allowlist=tuple(payload.get("tool_allowlist") or ()),
		input_contract=dict(payload.get("input_contract") or {}),
		output_contract=dict(payload.get("output_contract") or {}),
		memory_policy=dict(payload.get("memory_policy") or {}),
		status=str(payload.get("status") or "active"),
	)


def register_team(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_team(
		team_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		agent_ids=tuple(payload.get("agent_ids") or ()),
		handoffs=tuple(payload.get("handoffs") or ()),
		execution_mode=str(payload.get("execution_mode") or "sequential"),
		parallel_execution_enabled=bool(payload.get("parallel_execution_enabled", False)),
	)


def plan_execution(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.plan_execution(
		team_id=str(payload["team_id"]),
		objective=str(payload.get("objective") or ""),
		tenant_id=payload.get("tenant_id"),
	)


def list_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_agents(tenant_id)


def list_teams(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_teams(tenant_id)


def list_runtimes() -> list[dict[str, Any]]:
	return SERVICE.list_runtimes()


def list_runtime_approvals(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_runtime_approvals(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)
