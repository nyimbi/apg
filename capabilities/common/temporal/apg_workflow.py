"""APG Temporal workflow definition — executes APG state machine declarations.

Each APG `workflow` entity becomes an instance of APGStateMachineWorkflow.
The WorkflowDeclaration (states, transitions, guards, human_tasks, timers)
is passed as APGWorkflowInput and drives the Temporal workflow execution.

Durable execution guarantees:
  - Workflow state survives worker restart (Temporal event sourcing)
  - Human tasks block execution until explicitly approved/rejected
  - Guard conditions re-evaluated on each transition attempt
  - Timer-based escalation via Temporal durable timers
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class APGWorkflowInput:
	"""Input to an APG workflow instance."""
	workflow_id: str
	definition_id: str          # APG workflow entity name (e.g. "PayRunProcess")
	tenant_id: str
	actor_id: str
	initial_state: str
	states: list[str] = field(default_factory=list)
	transitions: list[dict[str, Any]] = field(default_factory=list)
	guards: dict[str, str] = field(default_factory=dict)
	human_tasks: list[str] = field(default_factory=list)
	timers: dict[str, str] = field(default_factory=dict)    # state -> ISO-8601 duration
	assignments: dict[str, str] = field(default_factory=dict)  # state -> role
	payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class APGWorkflowOutput:
	"""Result of a completed APG workflow instance."""
	workflow_id: str
	final_state: str
	completed: bool
	cancelled: bool = False
	error: str | None = None
	history: list[dict[str, Any]] = field(default_factory=list)


try:
	from temporalio import workflow, activity
	from temporalio.common import RetryPolicy

	@workflow.defn(name="APGStateMachineWorkflow")
	class APGStateMachineWorkflow:
		"""Temporal workflow that executes an APG state machine declaration.

		Supports:
		  - Sequential and conditional state transitions
		  - Human approval tasks (blocks at human_tasks states)
		  - Guard condition evaluation via activities
		  - Durable ISO-8601 timers for SLA escalation
		  - Signal-based task completion from external systems
		"""

		def __init__(self) -> None:
			self._current_state: str = ""
			self._approved: bool = False
			self._approval_signal: str = ""
			self._history: list[dict[str, Any]] = []

		@workflow.signal
		async def complete_human_task(self, outcome: str, variables: dict[str, Any]) -> None:
			"""Signal sent when a human task is approved or rejected."""
			self._approved = outcome == "approved"
			self._approval_signal = outcome

		@workflow.run
		async def run(self, wf_input: APGWorkflowInput) -> APGWorkflowOutput:
			self._current_state = wf_input.initial_state
			terminal_states = self._find_terminal_states(wf_input)

			while self._current_state not in terminal_states:
				self._history.append({
					"state": self._current_state,
					"timestamp": str(workflow.now()),
				})

				# Human task: block until signal received
				if self._current_state in wf_input.human_tasks:
					self._approved = False
					# Use durable timer for SLA escalation if configured
					timer_duration = wf_input.timers.get(self._current_state)
					if timer_duration:
						td = self._parse_iso_duration(timer_duration)
						try:
							await workflow.wait_condition(
								lambda: self._approval_signal != "",
								timeout=td,
							)
						except TimeoutError:
							# Timer fired — escalate and continue
							await workflow.execute_activity(
								"escalate_human_task",
								args=[wf_input.workflow_id, self._current_state, wf_input.tenant_id],
								start_to_close_timeout=timedelta(seconds=30),
								retry_policy=RetryPolicy(maximum_attempts=3),
							)
					else:
						await workflow.wait_condition(lambda: self._approval_signal != "")

					if not self._approved:
						return APGWorkflowOutput(
							workflow_id=wf_input.workflow_id,
							final_state=self._current_state,
							completed=False,
							error=f"Task rejected at state: {self._current_state}",
							history=self._history,
						)
					self._approval_signal = ""

				# Find next valid transition
				next_state = await self._advance(wf_input)
				if next_state is None:
					break
				self._current_state = next_state

			return APGWorkflowOutput(
				workflow_id=wf_input.workflow_id,
				final_state=self._current_state,
				completed=True,
				history=self._history,
			)

		async def _advance(self, wf_input: APGWorkflowInput) -> str | None:
			"""Find the next state by evaluating transitions from current state."""
			for t in wf_input.transitions:
				if t.get("source") != self._current_state:
					continue
				guard_expr = wf_input.guards.get(self._current_state)
				if guard_expr:
					passes = await workflow.execute_activity(
						"evaluate_guard",
						args=[guard_expr, wf_input.payload, wf_input.tenant_id],
						start_to_close_timeout=timedelta(seconds=10),
						retry_policy=RetryPolicy(maximum_attempts=2),
					)
					if not passes:
						continue
				return t.get("target")
			return None

		@staticmethod
		def _find_terminal_states(wf_input: APGWorkflowInput) -> set[str]:
			"""States with no outgoing transitions are terminal."""
			sources = {t.get("source") for t in wf_input.transitions}
			return {s for s in wf_input.states if s not in sources}

		@staticmethod
		def _parse_iso_duration(duration: str) -> timedelta:
			"""Parse ISO-8601 duration string to timedelta (PT24H, P1D, etc.)."""
			import re
			m = re.match(
				r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
				duration,
			)
			if not m:
				return timedelta(hours=24)
			days, hours, minutes, seconds = (int(g or 0) for g in m.groups())
			return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

except ImportError:
	# temporalio not installed — APGWorkflowInput/Output are pure dataclasses above and remain usable
	class APGStateMachineWorkflow:  # type: ignore[no-redef]
		"""Stub when temporalio is not installed."""
