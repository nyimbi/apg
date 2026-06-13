"""APG Enterprise Service Bus — integration flow management."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

from .models import EsbFlow, EsbFlowRun, EsbDeadLetter, EsbFlowStep, FlowStatus, FlowRunStatus, StepType

_log = logging.getLogger(__name__)


class EsbService:
	"""ESB integration flow management: create, activate, execute, and monitor flows."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._flows: dict[str, EsbFlow] = {}
		self._runs: dict[str, EsbFlowRun] = {}
		self._dead_letters: list[EsbDeadLetter] = []

	async def create_flow(
		self,
		name: str,
		description: str = "",
		trigger: dict[str, Any] | None = None,
		steps: list[dict[str, Any]] | None = None,
		tenant_id: str | None = None,
	) -> EsbFlow:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		guard_non_empty_string(name, "name")
		flow_steps = [EsbFlowStep(**s) for s in (steps or [])]
		flow = EsbFlow(
			tenant_id=tid, name=name, description=description,
			trigger=trigger or {}, steps=flow_steps,
		)
		self._flows[flow.id] = flow
		_log.info("Created flow '%s' (%s)", name, flow.id)
		return flow

	async def activate_flow(self, flow_id: str, tenant_id: str | None = None) -> EsbFlow:
		flow = self._require_flow(flow_id, tenant_id)
		flow.status = FlowStatus.ACTIVE
		flow.updated_at = datetime.now(timezone.utc)
		_log.info("Activated flow %s", flow_id)
		return flow

	async def pause_flow(self, flow_id: str, tenant_id: str | None = None) -> EsbFlow:
		flow = self._require_flow(flow_id, tenant_id)
		flow.status = FlowStatus.PAUSED
		flow.updated_at = datetime.now(timezone.utc)
		return flow

	async def execute_flow(
		self,
		flow_id: str,
		payload: dict[str, Any],
		tenant_id: str | None = None,
	) -> EsbFlowRun:
		"""Execute a flow synchronously with the given payload."""
		flow = self._require_flow(flow_id, tenant_id)
		assert flow.status == FlowStatus.ACTIVE, f"Flow {flow_id} is not active (status={flow.status})"
		run = EsbFlowRun(
			tenant_id=flow.tenant_id, flow_id=flow_id, trigger_payload=payload,
		)
		self._runs[run.id] = run
		try:
			result_payload = dict(payload)
			for step in flow.steps:
				result_payload = await self._execute_step(step, result_payload, run)
				run.step_results[step.id] = result_payload
			run.status = FlowRunStatus.COMPLETED
			run.completed_at = datetime.now(timezone.utc)
			if run.started_at and run.completed_at:
				run.duration_ms = int((run.completed_at - run.started_at).total_seconds() * 1000)
			_log.info("Flow %s run %s completed", flow_id, run.id)
		except Exception as exc:
			run.status = FlowRunStatus.FAILED
			run.error_message = str(exc)
			run.completed_at = datetime.now(timezone.utc)
			_log.error("Flow %s run %s failed: %s", flow_id, run.id, exc)
			if run.attempt_number >= flow.retry_attempts:
				dead = EsbDeadLetter(
					tenant_id=flow.tenant_id, flow_id=flow_id, flow_run_id=run.id,
					subject=str(flow.trigger.get("subject", "unknown")),
					payload=payload, error_message=str(exc), attempts=run.attempt_number,
				)
				self._dead_letters.append(dead)
				run.status = FlowRunStatus.DEAD_LETTERED
		return run

	async def list_flows(self, tenant_id: str | None = None) -> list[EsbFlow]:
		tid = tenant_id or self._tenant_id
		return [f for f in self._flows.values() if f.tenant_id == tid]

	async def list_runs(self, flow_id: str, tenant_id: str | None = None) -> list[EsbFlowRun]:
		tid = tenant_id or self._tenant_id
		return [r for r in self._runs.values() if r.flow_id == flow_id and r.tenant_id == tid]

	async def list_dead_letters(self, tenant_id: str | None = None) -> list[EsbDeadLetter]:
		tid = tenant_id or self._tenant_id
		return [d for d in self._dead_letters if d.tenant_id == tid and not d.resolved]

	async def resolve_dead_letter(self, dead_letter_id: str, tenant_id: str | None = None) -> EsbDeadLetter:
		dl = next((d for d in self._dead_letters if d.id == dead_letter_id), None)
		assert dl is not None, f"Dead letter {dead_letter_id} not found"
		dl.resolved = True
		dl.resolved_at = datetime.now(timezone.utc)
		return dl

	async def get_flow_stats(self, flow_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		runs = await self.list_runs(flow_id, tenant_id)
		total = len(runs)
		if total == 0:
			return {"flow_id": flow_id, "total_runs": 0, "success_rate_pct": 0, "avg_duration_ms": 0}
		completed = sum(1 for r in runs if r.status == FlowRunStatus.COMPLETED)
		failed = sum(1 for r in runs if r.status in (FlowRunStatus.FAILED, FlowRunStatus.DEAD_LETTERED))
		durations = [r.duration_ms for r in runs if r.duration_ms is not None]
		return {
			"flow_id": flow_id,
			"total_runs": total,
			"completed": completed,
			"failed": failed,
			"success_rate_pct": round(completed / total * 100, 1),
			"avg_duration_ms": round(sum(durations) / len(durations)) if durations else 0,
		}

	async def _execute_step(
		self,
		step: EsbFlowStep,
		payload: dict[str, Any],
		run: EsbFlowRun,
	) -> dict[str, Any]:
		"""Execute a single flow step. Real impl would call connectors, apply transforms."""
		if step.step_type == StepType.TRANSFORM and step.transformation:
			try:
				import jmespath  # type: ignore[import]
				result = jmespath.search(step.transformation, payload)
				return result if isinstance(result, dict) else {"result": result}
			except ImportError:
				pass
		elif step.step_type == StepType.FILTER:
			condition = step.config.get("condition", "")
			# Simple key=value filter
			key, _, val = condition.partition("=")
			if str(payload.get(key.strip())) != val.strip():
				return {}
		elif step.step_type == StepType.ROUTER:
			rules = step.config.get("rules", [])
			for rule in rules:
				if payload.get(rule.get("key")) == rule.get("value"):
					return {**payload, "_route": rule.get("target")}
		return payload

	def _require_flow(self, flow_id: str, tenant_id: str | None) -> EsbFlow:
		flow = self._flows.get(flow_id)
		assert flow is not None, f"Flow {flow_id} not found"
		return flow
