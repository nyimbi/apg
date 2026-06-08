"""Temporal WorkflowAdapter implementation for APG.

Implements the WorkflowAdapter Protocol from capabilities/ckm/wfa/domain/adapters.py.
Drop-in replacement for NullWorkflowAdapter with crash-resilient durable execution.

Activated when TEMPORAL_HOST env var is set. The get_workflow_adapter() factory
in domain/adapters.py routes to this adapter automatically.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import timedelta
from typing import Any

_log = logging.getLogger(__name__)

TASK_QUEUE = "apg-workflows"
WORKFLOW_NAME = "APGStateMachineWorkflow"


class TemporalWorkflowAdapter:
	"""WorkflowAdapter backed by Temporal durable execution.

	Converts APG WorkflowDeclaration data into Temporal workflow starts
	and task completions via gRPC calls to the Temporal server.
	"""

	def __init__(self, host: str = "localhost:7233", namespace: str = "default") -> None:
		self._host = host
		self._namespace = namespace
		self._client: Any = None

	async def _get_client(self) -> Any:
		if self._client is not None:
			return self._client
		try:
			from temporalio.client import Client
			self._client = await Client.connect(
				self._host,
				namespace=self._namespace,
			)
			_log.info("Connected to Temporal at %s", self._host)
		except Exception as exc:
			_log.error("Failed to connect to Temporal: %s", exc)
			raise
		return self._client

	async def start_workflow(
		self,
		definition_id: str,
		payload: dict[str, Any],
	) -> dict[str, Any]:
		"""Start a durable APG workflow instance on Temporal.

		The definition_id maps to a registered APG workflow entity name.
		Payload should include tenant_id and any initial data.

		Returns: {"instance_id": str, "status": "running", "run_id": str}
		"""
		client = await self._get_client()
		tenant_id = payload.get("tenant_id", "default")
		actor_id = payload.get("actor_id", "system")
		workflow_id = f"{tenant_id}-{definition_id}-{uuid.uuid4().hex[:8]}"

		# Build APGWorkflowInput from the definition registered in the payload
		wf_def = payload.get("_workflow_declaration", {})
		from .apg_workflow import APGWorkflowInput
		wf_input = APGWorkflowInput(
			workflow_id=workflow_id,
			definition_id=definition_id,
			tenant_id=tenant_id,
			actor_id=actor_id,
			initial_state=wf_def.get("initial_state") or (wf_def.get("states") or ["start"])[0],
			states=wf_def.get("states", []),
			transitions=wf_def.get("transitions", []),
			guards=wf_def.get("guards", {}),
			human_tasks=wf_def.get("human_tasks", []),
			timers=wf_def.get("timers", {}),
			assignments=wf_def.get("assignments", {}),
			payload={k: v for k, v in payload.items() if not k.startswith("_")},
		)

		handle = await client.start_workflow(
			WORKFLOW_NAME,
			wf_input,
			id=workflow_id,
			task_queue=TASK_QUEUE,
			execution_timeout=timedelta(days=365),  # APG workflows can run for months
		)
		_log.info("Started Temporal workflow %s (run_id=%s)", workflow_id, handle.result_run_id)
		return {
			"instance_id": workflow_id,
			"status": "running",
			"run_id": handle.result_run_id or "",
		}

	async def complete_task(
		self,
		task_id: str,
		outcome: str,
		variables: dict[str, Any],
	) -> None:
		"""Signal a human task completion to a running workflow.

		task_id format: {workflow_instance_id}::{state_name}
		outcome: "approved" | "rejected"
		"""
		if "::" not in task_id:
			_log.error("Invalid task_id format (expected 'instance_id::state'): %s", task_id)
			return

		workflow_instance_id, state_name = task_id.split("::", 1)
		client = await self._get_client()

		handle = client.get_workflow_handle(workflow_instance_id)
		await handle.signal("complete_human_task", outcome, variables)
		_log.info(
			"Signaled task completion: workflow=%s state=%s outcome=%s",
			workflow_instance_id, state_name, outcome,
		)

	async def get_workflow_status(self, instance_id: str) -> dict[str, Any]:
		"""Return the current status of a workflow instance."""
		client = await self._get_client()
		try:
			handle = client.get_workflow_handle(instance_id)
			desc = await handle.describe()
			return {
				"instance_id": instance_id,
				"status": str(desc.status.name).lower() if desc.status else "unknown",
				"workflow_type": desc.workflow_type or "",
				"start_time": desc.start_time.isoformat() if desc.start_time else None,
			}
		except Exception as exc:
			_log.error("Failed to get workflow status for %s: %s", instance_id, exc)
			return {"instance_id": instance_id, "status": "unknown", "error": str(exc)}

	async def cancel_workflow(self, instance_id: str) -> bool:
		"""Cancel a running workflow instance."""
		client = await self._get_client()
		try:
			handle = client.get_workflow_handle(instance_id)
			await handle.cancel()
			_log.info("Cancelled workflow %s", instance_id)
			return True
		except Exception as exc:
			_log.error("Failed to cancel workflow %s: %s", instance_id, exc)
			return False


def get_temporal_workflow_adapter() -> TemporalWorkflowAdapter | None:
	"""Return TemporalWorkflowAdapter if TEMPORAL_HOST is configured, else None."""
	host = os.environ.get("TEMPORAL_HOST")
	if not host:
		return None
	namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
	return TemporalWorkflowAdapter(host=host, namespace=namespace)
