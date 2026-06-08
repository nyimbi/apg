"""APG Temporal worker entrypoint.

Runs a Temporal worker that executes APG workflow and activity definitions.
Start this alongside the APG platform services.

Usage::

    python -m capabilities.common.temporal.temporal_worker
    # or
    from capabilities.common.temporal import start_worker
    await start_worker()

Environment variables:
    TEMPORAL_HOST       Temporal gRPC endpoint (default: localhost:7233)
    TEMPORAL_NAMESPACE  Temporal namespace (default: default)
    TEMPORAL_TASK_QUEUE Worker task queue (default: apg-workflows)
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal

_log = logging.getLogger(__name__)


async def start_worker(
	host: str | None = None,
	namespace: str | None = None,
	task_queue: str | None = None,
) -> None:
	"""Start the APG Temporal worker.

	Registers APGStateMachineWorkflow and all APGActivities, then polls
	the task queue until interrupted.
	"""
	from temporalio.client import Client
	from temporalio.worker import Worker

	from .apg_workflow import APGStateMachineWorkflow
	from .apg_activities import (
		evaluate_guard,
		escalate_human_task,
		execute_capability_action,
	)

	_host = host or os.environ.get("TEMPORAL_HOST", "localhost:7233")
	_namespace = namespace or os.environ.get("TEMPORAL_NAMESPACE", "default")
	_task_queue = task_queue or os.environ.get("TEMPORAL_TASK_QUEUE", "apg-workflows")

	client = await Client.connect(_host, namespace=_namespace)
	_log.info(
		"Starting APG Temporal worker — host=%s ns=%s queue=%s",
		_host, _namespace, _task_queue,
	)

	worker = Worker(
		client,
		task_queue=_task_queue,
		workflows=[APGStateMachineWorkflow],
		activities=[evaluate_guard, escalate_human_task, execute_capability_action],
	)

	# Graceful shutdown on SIGTERM/SIGINT
	loop = asyncio.get_event_loop()
	shutdown_event = asyncio.Event()

	def _handle_signal(*_: object) -> None:
		_log.info("APG Temporal worker shutdown requested")
		shutdown_event.set()

	for sig in (signal.SIGTERM, signal.SIGINT):
		loop.add_signal_handler(sig, _handle_signal)

	async with worker:
		_log.info("APG Temporal worker running on queue '%s'", _task_queue)
		await shutdown_event.wait()
		_log.info("APG Temporal worker stopped")


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	asyncio.run(start_worker())
