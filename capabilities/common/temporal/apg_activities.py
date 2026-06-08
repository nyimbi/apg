"""Temporal activity definitions for APG workflows.

Activities are the units of work that execute inside APG state machine
workflows. Each activity has at-least-once execution semantics with
configurable retries — they must be idempotent.

Activity functions are defined as plain async functions first (no temporalio
dependency), then decorated with @activity.defn when temporalio is available.
This keeps the logic testable without a Temporal server.
"""
from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger(__name__)


async def evaluate_guard(
	guard_expression: str,
	payload: dict[str, Any],
	tenant_id: str,
) -> bool:
	"""Evaluate an APG guard condition against the workflow payload.

	Guards are simple expressions like "all_payslips_generated == true"
	or "total_debit != total_credit". Evaluated safely against payload.
	"""
	try:
		ctx = {k: v for k, v in payload.items() if not k.startswith("_")}
		expr = guard_expression.strip()
		# Normalize APG lowercase boolean idioms to Python literals
		expr = expr.replace(" == true", " == True").replace(" == false", " == False")
		expr = expr.replace(" != true", " != True").replace(" != false", " != False")
		result = eval(expr, {"__builtins__": {"True": True, "False": False, "None": None}}, ctx)  # noqa: S307
		return bool(result)
	except Exception as exc:
		_log.warning("Guard evaluation failed for '%s': %s", guard_expression, exc)
		return True  # Non-blocking on evaluation error


async def escalate_human_task(
	workflow_id: str,
	state: str,
	tenant_id: str,
) -> None:
	"""Escalate a human task that has breached its SLA timer."""
	_log.warning(
		"Workflow %s: SLA breach at state '%s' for tenant %s",
		workflow_id, state, tenant_id,
	)
	try:
		if os.environ.get("NATS_URL"):
			from capabilities.common.nats.nats_adapter import NATSConnector
			connector = NATSConnector("ckm_wfa")
			await connector.connect()
			await connector.publish(
				"task_escalated",
				tenant_id,
				{"workflow_id": workflow_id, "state": state},
			)
	except Exception as exc:
		_log.error("Failed to publish escalation event: %s", exc)


async def execute_capability_action(
	capability_id: str,
	action: str,
	payload: dict[str, Any],
	tenant_id: str,
) -> dict[str, Any]:
	"""Execute an action on an APG capability service."""
	_log.info("Executing %s.%s for tenant %s", capability_id, action, tenant_id)
	try:
		from capabilities.capability_contract_registry import get_capability_service
		svc = get_capability_service(capability_id, tenant_id)
		handler = getattr(svc, action, None)
		if handler is None:
			raise AttributeError(f"{capability_id} has no action '{action}'")
		result = await handler(**payload)
		return {"ok": True, "result": result}
	except ImportError:
		_log.warning("Capability registry not available — action %s skipped", action)
		return {"ok": True, "result": None}
	except Exception as exc:
		_log.error("Capability action failed: %s", exc)
		return {"ok": False, "error": str(exc)}


# Apply Temporal @activity.defn decorators when temporalio is installed
try:
	from temporalio import activity as _activity

	evaluate_guard = _activity.defn(name="evaluate_guard")(evaluate_guard)  # type: ignore[assignment]
	escalate_human_task = _activity.defn(name="escalate_human_task")(escalate_human_task)  # type: ignore[assignment]
	execute_capability_action = _activity.defn(name="execute_capability_action")(execute_capability_action)  # type: ignore[assignment]
except ImportError:
	pass  # temporalio not installed — functions remain plain async, fully testable


class APGActivities:
	"""Bundle of all APG activity functions for Temporal worker registration."""
	evaluate_guard = staticmethod(evaluate_guard)
	escalate_human_task = staticmethod(escalate_human_task)
	execute_capability_action = staticmethod(execute_capability_action)
