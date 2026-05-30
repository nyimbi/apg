"""Domain service for APG budgeting and forecasting."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_BFC_AGENT_ROLES,
		SUPPORTED_BFC_AGENT_RUNTIMES,
		SUPPORTED_FORECAST_METHODS,
		SUPPORTED_LINE_TYPES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (
		SUPPORTED_BFC_AGENT_ROLES,
		SUPPORTED_BFC_AGENT_RUNTIMES,
		SUPPORTED_FORECAST_METHODS,
		SUPPORTED_LINE_TYPES,
		evaluate_capability_rules,
		streaming_manifest,
	)


class BudgetingForecastingService:
	"""Tenant-scoped budget, forecast, scenario, variance, collaboration, and agent coordinator."""

	def __init__(self) -> None:
		self._budgets: dict[str, dict[str, Any]] = {}
		self._budget_lines: dict[str, dict[str, Any]] = {}
		self._forecasts: dict[str, dict[str, Any]] = {}
		self._forecast_points: dict[str, dict[str, Any]] = {}
		self._scenarios: dict[str, dict[str, Any]] = {}
		self._variances: dict[str, dict[str, Any]] = {}
		self._collaboration_sessions: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def create_budget(self, budget_id: str, tenant_id: str, name: str, owner: str, fiscal_year: int | None, currency: str, period_start: str, period_end: str) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_budget",
			"budget_owner_assigned": bool(owner),
			"fiscal_year_present": fiscal_year is not None,
			"currency_present": bool(currency),
			"period_dates_present": bool(period_start) and bool(period_end),
			"period_range_valid": self._period_range_valid(period_start, period_end),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("bfc_budget", budget_id),
			"budget_id": budget_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"fiscal_year": fiscal_year,
			"currency": currency,
			"period_start": period_start,
			"period_end": period_end,
			"status": "draft",
			"submitted_by": None,
			"approved_by": None,
			"line_count": 0,
			"total_amount": 0.0,
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._budgets[record["id"]] = record
		self._emit("budget_created", tenant_id, record["id"], {"budget_id": budget_id, "fiscal_year": fiscal_year})
		return deepcopy(record)

	def add_budget_line(self, line_id: str, tenant_id: str, budget_record_id: str, account_id: str, line_type: str, amount: float, period: str, cost_center: str | None = None) -> dict[str, Any]:
		budget = self._require_budget(budget_record_id, tenant_id) if budget_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "add_budget_line",
			"budget_present": budget is not None,
			"account_present": bool(account_id),
			"line_type_supported": line_type in SUPPORTED_LINE_TYPES,
			"line_amount": amount,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("bfc_budget_line", line_id),
			"line_id": line_id,
			"tenant_id": tenant_id,
			"budget_record_id": budget["id"],
			"account_id": account_id,
			"line_type": line_type,
			"amount": float(amount),
			"period": period,
			"cost_center": cost_center,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._budget_lines[record["id"]] = record
		budget["line_count"] = len(self.list_budget_lines(tenant_id, budget["id"]))
		budget["total_amount"] = round(sum(line["amount"] for line in self.list_budget_lines(tenant_id, budget["id"])), 2)
		budget["updated_at"] = self._now()
		self._emit("budget_line_added", tenant_id, record["id"], {"budget_id": budget["budget_id"], "amount": amount})
		return deepcopy(record)

	def submit_budget(self, tenant_id: str, budget_record_id: str, submitted_by: str) -> dict[str, Any]:
		budget = self._require_budget(budget_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "submit_budget",
			"line_count": budget["line_count"],
		}
		self._enforce(context)
		budget["status"] = "submitted"
		budget["submitted_by"] = submitted_by
		budget["updated_at"] = self._now()
		self._emit("budget_submitted", tenant_id, budget["id"], {"submitted_by": submitted_by, "line_count": budget["line_count"]})
		return deepcopy(budget)

	def approve_budget(self, tenant_id: str, budget_record_id: str, approved_by: str, approval_recorded: bool = True, high_value_reviewed_by: str | None = None) -> dict[str, Any]:
		budget = self._require_budget(budget_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_budget",
			"budget_submitted": budget["status"] == "submitted",
			"approval_recorded": approval_recorded,
			"separation_of_duties_passed": bool(approved_by) and approved_by != budget.get("submitted_by"),
			"budget_total": budget["total_amount"],
			"high_value_review_recorded": bool(high_value_reviewed_by),
		}
		self._enforce(context)
		budget["status"] = "approved"
		budget["approved_by"] = approved_by
		budget["high_value_reviewed_by"] = high_value_reviewed_by
		budget["updated_at"] = self._now()
		self._emit("budget_approved", tenant_id, budget["id"], {"approved_by": approved_by, "total_amount": budget["total_amount"]})
		return deepcopy(budget)

	def create_forecast(self, forecast_id: str, tenant_id: str, name: str, method: str, horizon_months: int, confidence: float = 80, base_budget_record_id: str | None = None) -> dict[str, Any]:
		if base_budget_record_id:
			self._require_budget(base_budget_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_forecast",
			"forecast_method_supported": method in SUPPORTED_FORECAST_METHODS,
			"horizon_months": horizon_months,
			"confidence_out_of_bounds": confidence < 0 or confidence > 100,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("bfc_forecast", forecast_id),
			"forecast_id": forecast_id,
			"tenant_id": tenant_id,
			"name": name,
			"method": method,
			"horizon_months": int(horizon_months),
			"confidence": float(confidence),
			"base_budget_record_id": base_budget_record_id,
			"status": "forecasted",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._forecasts[record["id"]] = record
		self._emit("forecast_created", tenant_id, record["id"], {"method": method, "horizon_months": horizon_months})
		return deepcopy(record)

	def record_forecast_point(self, point_id: str, tenant_id: str, forecast_record_id: str, period: str, value: float) -> dict[str, Any]:
		forecast = self._require_forecast(forecast_record_id, tenant_id) if forecast_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_forecast_point",
			"forecast_present": forecast is not None,
			"period_present": bool(period),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("bfc_forecast_point", point_id),
			"point_id": point_id,
			"tenant_id": tenant_id,
			"forecast_record_id": forecast["id"],
			"period": period,
			"value": float(value),
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._forecast_points[record["id"]] = record
		self._emit("forecast_point_recorded", tenant_id, record["id"], {"forecast_id": forecast["forecast_id"], "period": period, "value": value})
		return deepcopy(record)

	def create_scenario(self, scenario_id: str, tenant_id: str, name: str, probability: float, drivers: list[dict[str, Any]]) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_scenario",
			"scenario_name_present": bool(name),
			"probability_out_of_bounds": probability < 0 or probability > 100,
			"driver_count": len(drivers),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("bfc_scenario", scenario_id),
			"scenario_id": scenario_id,
			"tenant_id": tenant_id,
			"name": name,
			"probability": float(probability),
			"drivers": deepcopy(drivers),
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._scenarios[record["id"]] = record
		self._emit("scenario_created", tenant_id, record["id"], {"probability": probability, "driver_count": len(drivers)})
		return deepcopy(record)

	def record_variance(self, variance_id: str, tenant_id: str, budget_record_id: str, account_id: str, budget_amount: float, actual_amount: float | None, reviewed_by: str | None = None) -> dict[str, Any]:
		budget = self._require_budget(budget_record_id, tenant_id) if budget_record_id else None
		variance_amount = float(actual_amount or 0) - float(budget_amount)
		variance_percent = 0 if budget_amount == 0 else round((variance_amount / float(budget_amount)) * 100, 2)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_variance",
			"budget_present": budget is not None,
			"actual_amount_present": actual_amount is not None,
			"variance_percent_abs": abs(variance_percent),
			"variance_review_recorded": bool(reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("bfc_variance", variance_id),
			"variance_id": variance_id,
			"tenant_id": tenant_id,
			"budget_record_id": budget["id"],
			"account_id": account_id,
			"budget_amount": float(budget_amount),
			"actual_amount": float(actual_amount),
			"variance_amount": round(variance_amount, 2),
			"variance_percent": variance_percent,
			"reviewed_by": reviewed_by,
			"status": "reviewed" if reviewed_by else "recorded",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._variances[record["id"]] = record
		self._emit("variance_recorded", tenant_id, record["id"], {"variance_percent": variance_percent, "reviewed": bool(reviewed_by)})
		return deepcopy(record)

	def start_collaboration_session(self, session_id: str, tenant_id: str, budget_record_id: str, participants: list[str]) -> dict[str, Any]:
		budget = self._require_budget(budget_record_id, tenant_id) if budget_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_collaboration_session",
			"budget_present": budget is not None,
			"participant_count": len(participants),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("bfc_collaboration", session_id),
			"session_id": session_id,
			"tenant_id": tenant_id,
			"budget_record_id": budget["id"],
			"participants": list(participants),
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._collaboration_sessions[record["id"]] = record
		self._emit("collaboration_session_started", tenant_id, record["id"], {"participant_count": len(participants)})
		return deepcopy(record)

	def register_bfc_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_bfc_agent",
			"agent_runtime_supported": runtime in SUPPORTED_BFC_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_BFC_AGENT_ROLES,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("bfc_agent", name),
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"instructions": instructions,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._agents[record["id"]] = record
		self._emit("bfc_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_bfc_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown BFC agent: {agent_id}")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "agent_bfc_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		}
		return evaluate_capability_rules(context)

	def validate_batch(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation": "bfc_batch", "event_stream": event_stream}
		result = evaluate_capability_rules(context)
		return {"processor": "bytewax", "record_count": record_count, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"budget_count": len(self.list_budgets(tenant_id)),
			"approved_budget_count": len([item for item in self.list_budgets(tenant_id) if item["status"] == "approved"]),
			"budget_line_count": len(self.list_budget_lines(tenant_id)),
			"forecast_count": len(self.list_forecasts(tenant_id)),
			"forecast_point_count": len(self.list_forecast_points(tenant_id)),
			"scenario_count": len(self.list_scenarios(tenant_id)),
			"variance_count": len(self.list_variances(tenant_id)),
			"collaboration_session_count": len(self.list_collaboration_sessions(tenant_id)),
			"bfc_agent_count": len(self.list_bfc_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def forecast_summary(self, tenant_id: str) -> dict[str, Any]:
		forecasts = self.list_forecasts(tenant_id)
		return {
			"tenant_id": tenant_id,
			"forecast_count": len(forecasts),
			"average_confidence": round(sum(item["confidence"] for item in forecasts) / len(forecasts), 2) if forecasts else 0,
			"max_horizon_months": max([item["horizon_months"] for item in forecasts], default=0),
		}

	def variance_summary(self, tenant_id: str) -> dict[str, Any]:
		variances = self.list_variances(tenant_id)
		return {
			"tenant_id": tenant_id,
			"variance_count": len(variances),
			"material_variance_count": len([item for item in variances if abs(item["variance_percent"]) > 10]),
			"net_variance_amount": round(sum(item["variance_amount"] for item in variances), 2),
		}

	def list_budgets(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._budgets, tenant_id)

	def list_budget_lines(self, tenant_id: str, budget_record_id: str | None = None) -> list[dict[str, Any]]:
		records = self._tenant_records(self._budget_lines, tenant_id)
		if budget_record_id:
			records = [record for record in records if record["budget_record_id"] == budget_record_id]
		return records

	def list_forecasts(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._forecasts, tenant_id)

	def list_forecast_points(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._forecast_points, tenant_id)

	def list_scenarios(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._scenarios, tenant_id)

	def list_variances(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._variances, tenant_id)

	def list_collaboration_sessions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._collaboration_sessions, tenant_id)

	def list_bfc_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._agents, tenant_id)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		return self.create_budget(
			data.get("budget_id", data.get("id", "budget")),
			data.get("tenant_id", "default"),
			data.get("name", "Budget"),
			data.get("owner", "finance"),
			data.get("fiscal_year", 2026),
			data.get("currency", "USD"),
			data.get("period_start", "2026-01-01"),
			data.get("period_end", "2026-12-31"),
		)

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_budgets(tenant_id)

	def _require_budget(self, budget_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._budgets, budget_id, tenant_id, "budget", "budget_id")

	def _require_forecast(self, forecast_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._forecasts, forecast_id, tenant_id, "forecast", "forecast_id")

	def _require_record(self, records: dict[str, dict[str, Any]], record_id: str, tenant_id: str, label: str, public_key: str) -> dict[str, Any]:
		for record in records.values():
			if record["tenant_id"] == tenant_id and (record["id"] == record_id or record[public_key] == record_id):
				return record
		raise KeyError(f"Unknown {label}: {record_id}")

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review":
			raise PermissionError(",".join(result["matched_rules"]))

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(record) for record in records.values() if record["tenant_id"] == tenant_id]

	def _emit(self, event_name: str, tenant_id: str, record_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"event": event_name,
			"tenant_id": tenant_id,
			"record_id": record_id,
			"payload": deepcopy(payload),
			"processor": "bytewax",
			"stream": streaming_manifest()["stream"],
			"created_at": self._now(),
		})

	def _record_id(self, prefix: str, value: str) -> str:
		slug = "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")
		return f"{prefix}_{slug or 'record'}"

	def _period_range_valid(self, start: str, end: str) -> bool:
		if not start or not end:
			return False
		return str(end) > str(start)

	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()


BFCService = BudgetingForecastingService
