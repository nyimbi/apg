"""Async service layer for APG Computer-Aided Manufacturing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import evaluate_capability_rules
	from .models import MfCamNcProgram, MfCamTool
except ImportError:
	from capability_contract import evaluate_capability_rules  # type: ignore
	from models import MfCamNcProgram, MfCamTool  # type: ignore

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgCamService:
	"""Computer-Aided Manufacturing service — async, in-memory."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._programs: dict[str, MfCamNcProgram] = {}
		self._tools: dict[str, MfCamTool] = {}

	async def create_program(self, program_number: str, program_name: str, machine_type: str, nc_code: str = "", item_id: str | None = None, item_code: str | None = None, created_by: str = "system", metadata: dict[str, Any] | None = None) -> MfCamNcProgram:
		prog = MfCamNcProgram(tenant_id=self._tenant_id, program_number=program_number, program_name=program_name, machine_type=machine_type, nc_code=nc_code, item_id=item_id, item_code=item_code, created_by=created_by, metadata=metadata or {})
		self._programs[prog.id] = prog
		return prog

	async def approve_program(self, program_id: str, approved_by: str) -> MfCamNcProgram:
		prog = self._programs.get(program_id)
		if not prog:
			raise KeyError(f"NC Program not found: {program_id}")
		prog.status = "approved"
		prog.approved_by = approved_by
		prog.approved_at = _now()
		return prog

	async def release_program(self, program_id: str) -> MfCamNcProgram:
		prog = self._programs.get(program_id)
		if not prog:
			raise KeyError(f"NC Program not found: {program_id}")
		ctx = {"tenant_context_present": True, "operation": "release_program", "approval_present": prog.status == "approved"}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Program release denied: {decision['actions']}")
		prog.status = "released"
		prog.released_at = _now()
		return prog

	async def get_program(self, program_id: str) -> MfCamNcProgram:
		if program_id not in self._programs:
			raise KeyError(f"NC Program not found: {program_id}")
		return self._programs[program_id]

	async def list_programs(self, status: str | None = None, machine_type: str | None = None) -> list[MfCamNcProgram]:
		progs = list(self._programs.values())
		if status:
			progs = [p for p in progs if p.status == status]
		if machine_type:
			progs = [p for p in progs if p.machine_type == machine_type]
		return progs

	async def create_tool(self, tool_number: str, tool_name: str, tool_type: str, diameter_mm: float | None = None, tool_life_minutes: float | None = None, metadata: dict[str, Any] | None = None) -> MfCamTool:
		tool = MfCamTool(tenant_id=self._tenant_id, tool_number=tool_number, tool_name=tool_name, tool_type=tool_type, diameter_mm=diameter_mm, tool_life_minutes=tool_life_minutes, metadata=metadata or {})
		self._tools[tool.id] = tool
		return tool

	async def log_tool_usage(self, tool_id: str, minutes_used: float) -> MfCamTool:
		tool = self._tools.get(tool_id)
		if not tool:
			raise KeyError(f"Tool not found: {tool_id}")
		tool.used_minutes += minutes_used
		if tool.tool_life_minutes and tool.used_minutes >= tool.tool_life_minutes:
			tool.status = "expired"
		return tool

	async def list_tools(self, status: str | None = None, tool_type: str | None = None) -> list[MfCamTool]:
		tools = list(self._tools.values())
		if status:
			tools = [t for t in tools if t.status == status]
		if tool_type:
			tools = [t for t in tools if t.tool_type == tool_type]
		return tools

	async def get_dashboard_summary(self) -> dict[str, Any]:
		progs = list(self._programs.values())
		tools = list(self._tools.values())
		return {
			"tenant_id": self._tenant_id,
			"programs": {"total": len(progs), "draft": sum(1 for p in progs if p.status == "draft"), "released": sum(1 for p in progs if p.status == "released")},
			"tools": {"total": len(tools), "active": sum(1 for t in tools if t.status == "active"), "expired": sum(1 for t in tools if t.status == "expired")},
		}
