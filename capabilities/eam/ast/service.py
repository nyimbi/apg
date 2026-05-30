"""Domain service for APG enterprise asset management."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_EAM_AGENT_ROLES,
		SUPPORTED_EAM_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (
		SUPPORTED_EAM_AGENT_ROLES,
		SUPPORTED_EAM_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)


class EnterpriseAssetManagementService:
	"""Tenant-scoped asset, maintenance, inspection, inventory, and agent coordinator."""

	def __init__(self) -> None:
		self._locations: dict[str, dict[str, Any]] = {}
		self._assets: dict[str, dict[str, Any]] = {}
		self._maintenance_plans: dict[str, dict[str, Any]] = {}
		self._work_orders: dict[str, dict[str, Any]] = {}
		self._inspections: dict[str, dict[str, Any]] = {}
		self._condition_readings: dict[str, dict[str, Any]] = {}
		self._inventory_reservations: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def register_location(self, location_id: str, tenant_id: str, name: str, location_type: str, parent_location_id: str | None = None) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_location",
			"location_type_present": bool(location_type),
		}
		self._enforce(context)
		if parent_location_id:
			self._require_location(parent_location_id, tenant_id)
		record = {
			"id": self._record_id("eam_location", location_id),
			"location_id": location_id,
			"tenant_id": tenant_id,
			"name": name,
			"location_type": location_type,
			"parent_location_id": parent_location_id,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._locations[record["id"]] = record
		self._emit("location_registered", tenant_id, record["id"], {"location_id": location_id, "location_type": location_type})
		return deepcopy(record)

	def register_asset(
		self,
		asset_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		category: str,
		location_id: str,
		criticality: str,
		health_score: float = 100,
		capitalized: bool = False,
		fixed_asset_ref: str | None = None,
	) -> dict[str, Any]:
		location = self._find_location(location_id, tenant_id) if location_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_asset",
			"asset_owner_assigned": bool(owner),
			"asset_category_present": bool(category),
			"asset_location_present": location is not None,
			"criticality_present": bool(criticality),
			"capitalized": capitalized,
			"fixed_asset_ref_present": bool(fixed_asset_ref),
			"health_score": health_score,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("eam_asset", asset_id),
			"asset_id": asset_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"category": category,
			"location_id": location_id,
			"criticality": criticality,
			"health_score": float(health_score),
			"capitalized": capitalized,
			"fixed_asset_ref": fixed_asset_ref,
			"status": "in_service",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._assets[record["id"]] = record
		self._emit("asset_registered", tenant_id, record["id"], {"asset_id": asset_id, "criticality": criticality})
		return deepcopy(record)

	def create_maintenance_plan(self, plan_id: str, tenant_id: str, asset_record_id: str, strategy: str, interval_days: int, condition_source: str | None = None) -> dict[str, Any]:
		asset = self._require_asset(asset_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_maintenance_plan",
			"maintenance_strategy_present": bool(strategy),
			"interval_present": interval_days is not None,
			"interval_days": interval_days,
			"predictive_plan": strategy == "predictive",
			"condition_source_present": bool(condition_source),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("eam_maintenance_plan", plan_id),
			"plan_id": plan_id,
			"tenant_id": tenant_id,
			"asset_record_id": asset["id"],
			"asset_id": asset["asset_id"],
			"strategy": strategy,
			"interval_days": interval_days,
			"condition_source": condition_source,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._maintenance_plans[record["id"]] = record
		self._emit("maintenance_plan_created", tenant_id, record["id"], {"asset_id": asset["asset_id"], "strategy": strategy})
		return deepcopy(record)

	def open_work_order(self, work_order_id: str, tenant_id: str, asset_record_id: str, title: str, priority: str, safety_plan: str, approved_by: str | None = None) -> dict[str, Any]:
		asset = self._require_asset(asset_record_id, tenant_id) if asset_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_work_order",
			"asset_present": asset is not None,
			"priority_present": bool(priority),
			"safety_plan_present": bool(safety_plan),
			"critical_asset": asset["criticality"] == "critical" if asset else False,
			"approved": bool(approved_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("eam_work_order", work_order_id),
			"work_order_id": work_order_id,
			"tenant_id": tenant_id,
			"asset_record_id": asset["id"],
			"asset_id": asset["asset_id"],
			"title": title,
			"priority": priority,
			"safety_plan": safety_plan,
			"approved_by": approved_by,
			"status": "work_open",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._work_orders[record["id"]] = record
		self._emit("work_order_opened", tenant_id, record["id"], {"asset_id": asset["asset_id"], "priority": priority})
		return deepcopy(record)

	def create_work_order(self, work_order_id: str, tenant_id: str, asset_id: str, title: str, priority: str, safety_plan: str = "standard", approved_by: str | None = None) -> dict[str, Any]:
		return self.open_work_order(work_order_id, tenant_id, asset_id, title, priority, safety_plan, approved_by)

	def complete_work_order(self, tenant_id: str, work_order_record_id: str, outcome: str, completed_by: str) -> dict[str, Any]:
		work_order = self._require_work_order(work_order_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "complete_work_order",
			"outcome_present": bool(outcome),
		}
		self._enforce(context)
		work_order["outcome"] = outcome
		work_order["completed_by"] = completed_by
		work_order["status"] = "work_complete"
		work_order["updated_at"] = self._now()
		self._emit("work_order_completed", tenant_id, work_order_record_id, {"outcome": outcome, "completed_by": completed_by})
		return deepcopy(work_order)

	def record_inspection(self, inspection_id: str, tenant_id: str, asset_record_id: str, result: str, inspector: str, condition_score: float | None = None) -> dict[str, Any]:
		asset = self._require_asset(asset_record_id, tenant_id) if asset_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_inspection",
			"asset_present": asset is not None,
			"inspection_result_present": bool(result),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("eam_inspection", inspection_id),
			"inspection_id": inspection_id,
			"tenant_id": tenant_id,
			"asset_record_id": asset["id"],
			"asset_id": asset["asset_id"],
			"result": result,
			"inspector": inspector,
			"condition_score": condition_score,
			"status": "recorded",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._inspections[record["id"]] = record
		if condition_score is not None:
			asset["health_score"] = float(condition_score)
			asset["status"] = "degraded" if condition_score < 50 else "in_service"
			asset["updated_at"] = self._now()
		self._emit("inspection_recorded", tenant_id, record["id"], {"asset_id": asset["asset_id"], "result": result})
		return deepcopy(record)

	def record_condition_reading(
		self,
		reading_id: str,
		tenant_id: str,
		asset_record_id: str,
		metric: str,
		value: float | None,
		unit: str,
		review_recorded: bool = False,
		alert_threshold: float | None = None,
	) -> dict[str, Any]:
		asset = self._require_asset(asset_record_id, tenant_id)
		condition_alert = alert_threshold is not None and value is not None and value > alert_threshold
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_condition_reading",
			"metric_present": bool(metric),
			"value_present": value is not None,
			"condition_alert": condition_alert,
			"review_recorded": review_recorded,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("eam_condition", reading_id),
			"reading_id": reading_id,
			"tenant_id": tenant_id,
			"asset_record_id": asset["id"],
			"asset_id": asset["asset_id"],
			"metric": metric,
			"value": float(value),
			"unit": unit,
			"alert_threshold": alert_threshold,
			"status": "degraded" if condition_alert else "normal",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._condition_readings[record["id"]] = record
		if condition_alert:
			asset["status"] = "degraded"
			asset["updated_at"] = self._now()
		self._emit("condition_reading_recorded", tenant_id, record["id"], {"metric": metric, "status": record["status"]})
		return deepcopy(record)

	def record_condition(self, reading_id: str, tenant_id: str, asset_id: str, reading_type: str, value: float, threshold: float) -> dict[str, Any]:
		return self.record_condition_reading(reading_id, tenant_id, asset_id, reading_type, value, "unit", review_recorded=True, alert_threshold=threshold)

	def reserve_inventory(self, reservation_id: str, tenant_id: str, part_id: str, quantity: int, work_order_record_id: str | None = None) -> dict[str, Any]:
		if work_order_record_id:
			self._require_work_order(work_order_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "reserve_inventory",
			"part_present": bool(part_id),
			"quantity_present": quantity is not None,
			"quantity": quantity,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("eam_inventory_reservation", reservation_id),
			"reservation_id": reservation_id,
			"tenant_id": tenant_id,
			"part_id": part_id,
			"quantity": quantity,
			"work_order_record_id": work_order_record_id,
			"status": "reserved",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._inventory_reservations[record["id"]] = record
		self._emit("inventory_reservation_created", tenant_id, record["id"], {"part_id": part_id, "quantity": quantity})
		return deepcopy(record)

	def register_eam_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_eam_agent",
			"agent_runtime_supported": runtime in SUPPORTED_EAM_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_EAM_AGENT_ROLES,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("eam_agent", name),
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
		self._emit("eam_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_eam_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown EAM agent: {agent_id}")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "agent_eam_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		}
		return evaluate_capability_rules(context)

	def validate_batch_import(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation": "eam_batch_import", "event_stream": event_stream}
		result = evaluate_capability_rules(context)
		return {"processor": "bytewax", "record_count": record_count, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def validate_lifecycle_event(self, tenant_id: str, event_stream: str = "bytewax") -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation": "eam_event", "event_stream": event_stream}
		return evaluate_capability_rules(context)

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"location_count": len(self.list_locations(tenant_id)),
			"asset_count": len(self.list_assets(tenant_id)),
			"maintenance_plan_count": len(self.list_maintenance_plans(tenant_id)),
			"open_work_order_count": len([item for item in self.list_work_orders(tenant_id) if item["status"] != "work_complete"]),
			"inspection_count": len(self.list_inspections(tenant_id)),
			"degraded_condition_count": len([item for item in self.list_condition_readings(tenant_id) if item["status"] == "degraded"]),
			"inventory_reservation_count": len(self.list_inventory_reservations(tenant_id)),
			"eam_agent_count": len(self.list_eam_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def reliability_summary(self, tenant_id: str) -> dict[str, Any]:
		assets = self.list_assets(tenant_id)
		critical_assets = [asset for asset in assets if asset["criticality"] == "critical"]
		degraded_assets = [asset for asset in assets if asset["status"] == "degraded"]
		average_health = round(sum(asset["health_score"] for asset in assets) / len(assets), 2) if assets else 0
		return {
			"tenant_id": tenant_id,
			"asset_count": len(assets),
			"critical_asset_count": len(critical_assets),
			"degraded_asset_count": len(degraded_assets),
			"average_health_score": average_health,
			"condition_reading_count": len(self.list_condition_readings(tenant_id)),
		}

	def list_locations(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._locations, tenant_id)

	def list_assets(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._assets, tenant_id)

	def list_maintenance_plans(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._maintenance_plans, tenant_id)

	def list_work_orders(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._work_orders, tenant_id)

	def list_inspections(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._inspections, tenant_id)

	def list_condition_readings(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._condition_readings, tenant_id)

	def list_conditions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self.list_condition_readings(tenant_id)

	def list_inventory_reservations(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._inventory_reservations, tenant_id)

	def list_eam_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._agents, tenant_id)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		tenant_id = data.get("tenant_id", "default")
		location_id = data.get("location_id", "main-site")
		if not self._find_location(location_id, tenant_id):
			self.register_location(location_id, tenant_id, data.get("location_name", "Main Site"), data.get("location_type", "site"))
		return self.register_asset(
			data.get("asset_id", data.get("id", "asset")),
			tenant_id,
			data.get("name", "Asset"),
			data.get("owner", "owner"),
			data.get("category", "equipment"),
			location_id,
			data.get("criticality", "medium"),
			data.get("health_score", 100),
			data.get("capitalized", False),
			data.get("fixed_asset_ref"),
		)

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_assets(tenant_id)

	def _require_location(self, location_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._find_location(location_id, tenant_id)
		if record is None:
			raise KeyError(f"Unknown location: {location_id}")
		return record

	def _find_location(self, location_id: str, tenant_id: str) -> dict[str, Any] | None:
		for record in self._locations.values():
			if record["tenant_id"] == tenant_id and (record["location_id"] == location_id or record["id"] == location_id):
				return record
		return None

	def _require_asset(self, asset_id: str, tenant_id: str) -> dict[str, Any]:
		for record in self._assets.values():
			if record["tenant_id"] == tenant_id and (record["asset_id"] == asset_id or record["id"] == asset_id):
				return record
		raise KeyError(f"Unknown asset: {asset_id}")

	def _require_work_order(self, work_order_record_id: str, tenant_id: str) -> dict[str, Any]:
		for record in self._work_orders.values():
			if record["tenant_id"] == tenant_id and (record["work_order_id"] == work_order_record_id or record["id"] == work_order_record_id):
				return record
		raise KeyError(f"Unknown work order: {work_order_record_id}")

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

	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()


EAMAssetService = EnterpriseAssetManagementService
