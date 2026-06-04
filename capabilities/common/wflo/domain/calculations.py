"""Domain calculations for Workflow Orchestration.

Pure functions — no I/O, no side effects, fully type-safe.

Covers:
- SLA and throughput metrics
- Duration parsing (ISO 8601)
- Instance lifecycle timing
- Gateway condition evaluation
- Task workload and queue depth
- Analytics aggregations
- Bottleneck detection
"""
from __future__ import annotations

import re
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# ISO 8601 duration parsing
# ─────────────────────────────────────────────────────────────────────────────

_DURATION_RE = re.compile(
	r"^P"
	r"(?:(\d+)Y)?"
	r"(?:(\d+)M)?"
	r"(?:(\d+)W)?"
	r"(?:(\d+)D)?"
	r"(?:T"
	r"(?:(\d+)H)?"
	r"(?:(\d+)M)?"
	r"(?:(\d+(?:\.\d+)?)S)?"
	r")?$"
)


def parse_iso_duration(duration: str) -> timedelta:
	"""Parse an ISO 8601 duration string into a timedelta.

	Handles years (365d), months (30d), weeks, days, hours, minutes, seconds.

	Args:
		duration: e.g. "PT1H30M", "P1DT2H", "P1Y2M3DT4H5M6S"

	Returns:
		timedelta representation (year=365d, month=30d approximation)

	Raises:
		ValueError: If the string is not valid ISO 8601 duration.
	"""
	m = _DURATION_RE.match(duration.strip())
	if not m:
		raise ValueError(f"Invalid ISO 8601 duration: '{duration}'")
	years = int(m.group(1) or 0)
	months = int(m.group(2) or 0)
	weeks = int(m.group(3) or 0)
	days = int(m.group(4) or 0)
	hours = int(m.group(5) or 0)
	minutes = int(m.group(6) or 0)
	seconds = float(m.group(7) or 0)
	total_days = (years * 365) + (months * 30) + (weeks * 7) + days
	return timedelta(days=total_days, hours=hours, minutes=minutes, seconds=seconds)


def duration_to_minutes(duration: str) -> float:
	"""Convert ISO 8601 duration to total minutes (float)."""
	td = parse_iso_duration(duration)
	return td.total_seconds() / 60.0


# ─────────────────────────────────────────────────────────────────────────────
# SLA calculations
# ─────────────────────────────────────────────────────────────────────────────

def calculate_due_at(started_at: datetime, sla_minutes: int) -> datetime:
	"""Compute absolute SLA deadline from start time."""
	if sla_minutes < 1:
		raise ValueError(f"sla_minutes must be >= 1, got {sla_minutes}")
	return started_at + timedelta(minutes=sla_minutes)


def calculate_remaining_minutes(due_at: datetime, now: datetime | None = None) -> float:
	"""Minutes remaining until SLA deadline. Negative if breached."""
	ref = now or datetime.now(timezone.utc)
	return (due_at - ref).total_seconds() / 60.0


def calculate_sla_health_pct(started_at: datetime, due_at: datetime, now: datetime | None = None) -> float:
	"""Return fraction of SLA window remaining (1.0 = just started, 0.0 = deadline, <0 = breached)."""
	ref = now or datetime.now(timezone.utc)
	total = (due_at - started_at).total_seconds()
	if total <= 0:
		return 0.0
	remaining = (due_at - ref).total_seconds()
	return remaining / total


def calculate_sla_breach_rate(total: int, breached: int) -> float:
	"""Breach rate as a fraction in [0, 1]."""
	if total <= 0:
		return 0.0
	return min(1.0, breached / total)


# ─────────────────────────────────────────────────────────────────────────────
# Instance timing
# ─────────────────────────────────────────────────────────────────────────────

def calculate_elapsed_minutes(started_at: datetime, ended_at: datetime | None = None) -> float:
	"""Total elapsed time in minutes since start."""
	ref = ended_at or datetime.now(timezone.utc)
	delta = ref - started_at
	return max(0.0, delta.total_seconds() / 60.0)


def calculate_throughput(
	completed_instances: list[dict[str, Any]],
	window_minutes: int = 60,
	now: datetime | None = None,
) -> float:
	"""Instances completed per hour within the last window_minutes."""
	ref = now or datetime.now(timezone.utc)
	cutoff = ref - timedelta(minutes=window_minutes)
	count = 0
	for inst in completed_instances:
		completed_at = inst.get("completed_at")
		if completed_at is None:
			continue
		if isinstance(completed_at, str):
			completed_at = datetime.fromisoformat(completed_at)
		if completed_at >= cutoff:
			count += 1
	# Normalise to per-hour rate
	return count * (60.0 / window_minutes)


# ─────────────────────────────────────────────────────────────────────────────
# Duration distribution (for analytics)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_duration_percentiles(
	durations_minutes: list[float],
) -> dict[str, float]:
	"""Return p50, p75, p90, p95, p99 percentile durations.

	Args:
		durations_minutes: List of completed instance durations in minutes.

	Returns:
		Dict with keys p50, p75, p90, p95, p99 (all floats, minutes).
	"""
	if not durations_minutes:
		return {"p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
	sorted_d = sorted(durations_minutes)
	n = len(sorted_d)

	def _pct(p: float) -> float:
		idx = int(p / 100 * (n - 1))
		return sorted_d[idx]

	return {
		"p50": _pct(50),
		"p75": _pct(75),
		"p90": _pct(90),
		"p95": _pct(95),
		"p99": _pct(99),
	}


def calculate_mean_duration(durations_minutes: list[float]) -> float:
	"""Arithmetic mean duration in minutes."""
	if not durations_minutes:
		return 0.0
	return statistics.mean(durations_minutes)


def calculate_stddev_duration(durations_minutes: list[float]) -> float:
	"""Sample standard deviation of durations in minutes."""
	if len(durations_minutes) < 2:
		return 0.0
	return statistics.stdev(durations_minutes)


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck detection
# ─────────────────────────────────────────────────────────────────────────────

def identify_bottleneck_node(
	history_events: list[dict[str, Any]],
) -> tuple[str | None, float]:
	"""Find the BPMN node with the highest average dwell time.

	Dwell time = time between task.created and task.completed for the same node_id.

	Args:
		history_events: WorkflowHistory dicts with keys: node_id, event_type, created_at.

	Returns:
		(bottleneck_node_id, avg_minutes) — (None, 0.0) if insufficient data.
	"""
	# Group start/end pairs by node_id
	starts: dict[str, list[datetime]] = {}
	ends: dict[str, list[datetime]] = {}
	for evt in history_events:
		nid = evt.get("node_id")
		if not nid:
			continue
		ts = evt.get("created_at")
		if not ts:
			continue
		if isinstance(ts, str):
			ts = datetime.fromisoformat(ts)
		et = evt.get("event_type", "")
		if "created" in et or "started" in et:
			starts.setdefault(nid, []).append(ts)
		elif "completed" in et or "finished" in et:
			ends.setdefault(nid, []).append(ts)

	dwell_avgs: dict[str, float] = {}
	for nid, start_times in starts.items():
		end_times = ends.get(nid, [])
		if not end_times:
			continue
		pairs = min(len(start_times), len(end_times))
		durations = [
			(end_times[i] - start_times[i]).total_seconds() / 60.0
			for i in range(pairs)
			if end_times[i] > start_times[i]
		]
		if durations:
			dwell_avgs[nid] = statistics.mean(durations)

	if not dwell_avgs:
		return None, 0.0
	bottleneck = max(dwell_avgs, key=lambda k: dwell_avgs[k])
	return bottleneck, dwell_avgs[bottleneck]


# ─────────────────────────────────────────────────────────────────────────────
# Task queue metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_queue_depth_by_assignee(
	tasks: list[dict[str, Any]],
	open_statuses: set[str] | None = None,
) -> dict[str, int]:
	"""Count open tasks per assignee_ref.

	Args:
		tasks: Task dicts with keys: assignee_ref, status.
		open_statuses: Statuses considered open (default: created, ready, claimed, in_progress).

	Returns:
		Dict of assignee_ref → open task count.
	"""
	if open_statuses is None:
		open_statuses = {"created", "ready", "claimed", "in_progress"}
	depth: dict[str, int] = {}
	for t in tasks:
		if t.get("status") not in open_statuses:
			continue
		assignee = t.get("assignee_ref") or "unassigned"
		depth[assignee] = depth.get(assignee, 0) + 1
	return depth


def calculate_overdue_tasks(
	tasks: list[dict[str, Any]],
	now: datetime | None = None,
) -> list[str]:
	"""Return IDs of tasks that are past their due_at and not yet completed.

	Args:
		tasks: Task dicts with keys: id, due_at, status.
		now: Current time override for testing.
	"""
	ref = now or datetime.now(timezone.utc)
	terminal = {"completed", "cancelled", "timed_out"}
	overdue: list[str] = []
	for t in tasks:
		if t.get("status") in terminal:
			continue
		due = t.get("due_at")
		if not due:
			continue
		if isinstance(due, str):
			due = datetime.fromisoformat(due)
		if ref > due:
			overdue.append(t["id"])
	return overdue


def calculate_claim_lag_minutes(
	tasks: list[dict[str, Any]],
) -> float:
	"""Average minutes between task creation and claim for completed tasks."""
	lags: list[float] = []
	for t in tasks:
		created = t.get("created_at")
		claimed = t.get("claimed_at")
		if not created or not claimed:
			continue
		if isinstance(created, str):
			created = datetime.fromisoformat(created)
		if isinstance(claimed, str):
			claimed = datetime.fromisoformat(claimed)
		lag = (claimed - created).total_seconds() / 60.0
		if lag >= 0:
			lags.append(lag)
	return statistics.mean(lags) if lags else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Gateway condition evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_exclusive_gateway(
	conditions: dict[str, str],
	variables: dict[str, Any],
) -> str:
	"""Evaluate XOR gateway — return the first matching outgoing node ID.

	Conditions are simple equality expressions: "key == value" or "key > value".
	A condition of "" or "*" is treated as default (always true).

	Args:
		conditions: {target_node_id: expression_string}
		variables: Current workflow variables.

	Returns:
		Winning target_node_id.

	Raises:
		ValueError: If no condition matches and there is no default.
	"""
	default_path: str | None = None
	for target, expr in conditions.items():
		stripped = expr.strip()
		if not stripped or stripped == "*":
			default_path = target
			continue
		if _eval_simple_condition(stripped, variables):
			return target
	if default_path is not None:
		return default_path
	raise ValueError(f"Exclusive gateway: no condition matched. variables={variables!r}")


def evaluate_inclusive_gateway(
	conditions: dict[str, str],
	variables: dict[str, Any],
) -> list[str]:
	"""Evaluate OR gateway — return all matching outgoing node IDs."""
	selected: list[str] = []
	for target, expr in conditions.items():
		stripped = expr.strip()
		if not stripped or stripped == "*" or _eval_simple_condition(stripped, variables):
			selected.append(target)
	return selected


def _eval_simple_condition(expr: str, variables: dict[str, Any]) -> bool:
	"""Evaluate a simple "key op value" condition against variables.

	Supported operators: ==, !=, >, >=, <, <=
	Values are coerced to appropriate types for comparison.
	"""
	for op in (">=", "<=", "!=", ">", "<", "=="):
		if op in expr:
			left, right = expr.split(op, 1)
			left = left.strip()
			right = right.strip()
			lval = _resolve_value(left, variables)
			rval = _coerce(right, lval)
			if op == "==":
				return lval == rval
			if op == "!=":
				return lval != rval
			if op == ">":
				return float(lval) > float(rval)  # type: ignore[arg-type]
			if op == ">=":
				return float(lval) >= float(rval)  # type: ignore[arg-type]
			if op == "<":
				return float(lval) < float(rval)  # type: ignore[arg-type]
			if op == "<=":
				return float(lval) <= float(rval)  # type: ignore[arg-type]
	# Bare variable name — truthy check
	left = expr.strip()
	return bool(_resolve_value(left, variables))


def _resolve_value(token: str, variables: dict[str, Any]) -> Any:
	"""Resolve a token — variable reference or literal."""
	# Strip BPMN ${...} notation
	if token.startswith("${") and token.endswith("}"):
		token = token[2:-1].strip()
	return variables.get(token, token)


def _coerce(token: str, reference: Any) -> Any:
	"""Coerce a string token to match the type of reference."""
	if isinstance(reference, bool):
		return token.lower() in {"true", "1", "yes"}
	if isinstance(reference, int):
		try:
			return int(token)
		except ValueError:
			return token
	if isinstance(reference, float):
		try:
			return float(token)
		except ValueError:
			return token
	return token


# ─────────────────────────────────────────────────────────────────────────────
# Analytics roll-up
# ─────────────────────────────────────────────────────────────────────────────

def build_analytics_summary(
	instances: list[dict[str, Any]],
	tasks: list[dict[str, Any]],
	history: list[dict[str, Any]],
	now: datetime | None = None,
) -> dict[str, Any]:
	"""Build a full analytics summary dict for a single definition.

	Args:
		instances: WorkflowInstance dicts.
		tasks: Task dicts for those instances.
		history: WorkflowHistory dicts for those instances.
		now: Reference time (defaults to utc now).

	Returns:
		Dict suitable for populating WorkflowAnalytics.
	"""
	ref = now or datetime.now(timezone.utc)
	terminal_ok = {"completed"}
	terminal_fail = {"failed"}
	terminal_cancel = {"cancelled"}

	completed = [i for i in instances if i.get("status") in terminal_ok]
	failed = [i for i in instances if i.get("status") in terminal_fail]
	cancelled = [i for i in instances if i.get("status") in terminal_cancel]
	active = [
		i for i in instances
		if i.get("status") not in terminal_ok | terminal_fail | terminal_cancel | {"migrated"}
	]

	# Duration distribution (completed only)
	durations: list[float] = []
	for inst in completed:
		started = inst.get("started_at")
		ended = inst.get("completed_at")
		if started and ended:
			if isinstance(started, str):
				started = datetime.fromisoformat(started)
			if isinstance(ended, str):
				ended = datetime.fromisoformat(ended)
			durations.append((ended - started).total_seconds() / 60.0)

	pcts = calculate_duration_percentiles(durations)
	mean_dur = calculate_mean_duration(durations)

	# SLA breaches
	breached = sum(1 for i in instances if i.get("sla_breached"))

	# Task metrics
	total_tasks = len(tasks)
	claim_lag = calculate_claim_lag_minutes(tasks)
	overdue = calculate_overdue_tasks(tasks, now=ref)
	escalations = [t for t in tasks if t.get("escalated")]

	# Bottleneck
	bottleneck_nid, bottleneck_avg = identify_bottleneck_node(history)

	return {
		"total_instances": len(instances),
		"completed_instances": len(completed),
		"failed_instances": len(failed),
		"cancelled_instances": len(cancelled),
		"active_instances": len(active),
		"avg_duration_minutes": mean_dur,
		"p50_duration_minutes": pcts["p50"],
		"p95_duration_minutes": pcts["p95"],
		"p99_duration_minutes": pcts["p99"],
		"sla_breach_count": breached,
		"sla_breach_rate": calculate_sla_breach_rate(len(instances), breached),
		"total_tasks": total_tasks,
		"avg_task_claim_minutes": claim_lag,
		"avg_task_completion_minutes": 0.0,  # caller may fill from task timing
		"escalation_count": len(escalations),
		"escalation_rate": len(escalations) / total_tasks if total_tasks else 0.0,
		"bottleneck_node_id": bottleneck_nid,
		"bottleneck_avg_minutes": bottleneck_avg,
	}


__all__ = [
	"parse_iso_duration",
	"duration_to_minutes",
	"calculate_due_at",
	"calculate_remaining_minutes",
	"calculate_sla_health_pct",
	"calculate_sla_breach_rate",
	"calculate_elapsed_minutes",
	"calculate_throughput",
	"calculate_duration_percentiles",
	"calculate_mean_duration",
	"calculate_stddev_duration",
	"identify_bottleneck_node",
	"calculate_queue_depth_by_assignee",
	"calculate_overdue_tasks",
	"calculate_claim_lag_minutes",
	"evaluate_exclusive_gateway",
	"evaluate_inclusive_gateway",
	"build_analytics_summary",
]
