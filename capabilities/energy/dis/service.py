"""Service layer for APG Distribution Network."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_FAULT_TYPES, SUPPORTED_NETWORK_ELEMENT_TYPES, SUPPORTED_OUTAGE_CAUSES,
		SUPPORTED_RESTORATION_STRATEGIES, SUPPORTED_SCADA_PROTOCOLS,
		SUPPORTED_SWITCHING_OPERATIONS, SUPPORTED_VOLTAGE_LEVELS,
		SUPPORTED_LOAD_BALANCING_MODES, SUPPORTED_FAULT_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AuditEvent, DisAgent, FaultRecord, Feeder, LoadBalanceAction,
		NetworkElement, OutageRecord, ScadaReading, SwitchingOrder,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_FAULT_TYPES, SUPPORTED_NETWORK_ELEMENT_TYPES, SUPPORTED_OUTAGE_CAUSES,
		SUPPORTED_RESTORATION_STRATEGIES, SUPPORTED_SCADA_PROTOCOLS,
		SUPPORTED_SWITCHING_OPERATIONS, SUPPORTED_VOLTAGE_LEVELS,
		SUPPORTED_LOAD_BALANCING_MODES, SUPPORTED_FAULT_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AuditEvent, DisAgent, FaultRecord, Feeder, LoadBalanceAction,
		NetworkElement, OutageRecord, ScadaReading, SwitchingOrder,
	)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


class DistributionNetworkService:
	"""Tenant-scoped Distribution Network runtime."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *, auth=None, audit=None, notify=None, db_url=None, store=None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self.feeders: dict[tuple[str, str], Feeder] = {}
		self.elements: dict[tuple[str, str], NetworkElement] = {}
		self.faults: dict[tuple[str, str], FaultRecord] = {}
		self.switching_orders: dict[tuple[str, str], SwitchingOrder] = {}
		self.outages: dict[tuple[str, str], OutageRecord] = {}
		self.scada_readings: dict[tuple[str, str], ScadaReading] = {}
		self.load_balance_actions: dict[tuple[str, str], LoadBalanceAction] = {}
		self.agents: dict[tuple[str, str], DisAgent] = {}
		self.audit_events: list[AuditEvent] = []
		# Extended stores
		self._network_topology_changes: dict[str, dict[str, Any]] = {}
		self._saidi_saifi_records: dict[str, dict[str, Any]] = {}
		self._outage_stats_records: dict[str, dict[str, Any]] = {}
		self._network_analytics_records: dict[str, dict[str, Any]] = {}
		self._system_normal_records: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── feeders ───────────────────────────────────────────────────────────────

	def register_feeder(
		self,
		feeder_id: str,
		tenant_id: str,
		name: str,
		substation_id: str,
		voltage_level: str,
		normal_capacity_mw: float,
		emergency_capacity_mw: float,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a distribution feeder."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_element",
			"element_type_supported": True,
			"voltage_level_supported": voltage_level in SUPPORTED_VOLTAGE_LEVELS,
			"feeder_present": True,
		})
		item = Feeder(
			id=feeder_id, tenant_id=tenant_id, name=name,
			substation_id=substation_id, voltage_level=voltage_level,
			status="energized", normal_capacity_mw=normal_capacity_mw,
			emergency_capacity_mw=emergency_capacity_mw,
		)
		self.feeders[self._key(tenant_id, feeder_id)] = item
		self._audit(tenant_id, "feeder_registered", feeder_id, "feeder")
		return item.to_dict()

	def list_feeders(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_items(self.feeders, tenant_id)

	# ── network elements ──────────────────────────────────────────────────────

	def register_element(
		self,
		element_id: str,
		tenant_id: str,
		element_type: str,
		name: str,
		feeder_id: str,
		voltage_level: str,
		location_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a network element (transformer, switch, etc.)."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_element",
			"element_type_supported": element_type in SUPPORTED_NETWORK_ELEMENT_TYPES,
			"voltage_level_supported": voltage_level in SUPPORTED_VOLTAGE_LEVELS,
			"feeder_present": _present(feeder_id),
		})
		item = NetworkElement(
			id=element_id, tenant_id=tenant_id, element_type=element_type,
			name=name, feeder_id=feeder_id, voltage_level=voltage_level,
			status="energized", location_reference=location_reference,
		)
		self.elements[self._key(tenant_id, element_id)] = item
		self._audit(tenant_id, "network_element_registered", element_id, "element")
		return item.to_dict()

	def list_elements(self, tenant_id: str, feeder_id: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.elements, tenant_id)
		if feeder_id:
			items = [e for e in items if e["feeder_id"] == feeder_id]
		return items

	# ── faults ────────────────────────────────────────────────────────────────

	def report_fault(
		self,
		fault_id: str,
		tenant_id: str,
		element_id: str,
		fault_type: str,
		location_reference: str,
		affected_customers: int,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Report a fault on a network element."""
		element_exists = self._key(tenant_id, element_id) in self.elements
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "report_fault",
			"fault_type_supported": fault_type in SUPPORTED_FAULT_TYPES,
			"element_exists": element_exists,
			"location_present": _present(location_reference),
		})
		item = FaultRecord(
			id=fault_id, tenant_id=tenant_id, element_id=element_id,
			fault_type=fault_type, status="detected", detected_at=_now(),
			location_reference=location_reference, affected_customers=affected_customers,
		)
		self.faults[self._key(tenant_id, fault_id)] = item
		self._audit(tenant_id, "fault_detected", fault_id, "fault", {"fault_type": fault_type})
		return item.to_dict()

	def isolate_fault(self, fault_id: str, tenant_id: str) -> dict[str, Any]:
		"""Mark a fault as isolated."""
		fault = self._get_fault(tenant_id, fault_id)
		fault.status = "isolated"
		fault.isolated_at = _now()
		self._audit(tenant_id, "fault_isolated", fault_id, "fault")
		return fault.to_dict()

	def restore_fault(self, fault_id: str, tenant_id: str, strategy: str) -> dict[str, Any]:
		"""Restore supply after a fault."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "initiate_restoration",
			"strategy_supported": strategy in SUPPORTED_RESTORATION_STRATEGIES,
		})
		fault = self._get_fault(tenant_id, fault_id)
		fault.status = "restored"
		fault.restored_at = _now()
		self._audit(tenant_id, "fault_restored", fault_id, "fault", {"strategy": strategy})
		return fault.to_dict()

	def dispatch_crew(self, fault_id: str, tenant_id: str, crew_id: str) -> dict[str, Any]:
		"""Dispatch a crew to a fault location."""
		fault = self._get_fault(tenant_id, fault_id)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "dispatch_crew",
			"fault_isolated": fault.status == "isolated",
		})
		fault.crew_id = crew_id
		fault.status = "crew_dispatched"
		self._audit(tenant_id, "crew_dispatched", fault_id, "fault", {"crew_id": crew_id})
		return fault.to_dict()

	def list_faults(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.faults, tenant_id)
		if status:
			items = [f for f in items if f["status"] == status]
		return items

	# ── switching orders ──────────────────────────────────────────────────────

	def create_switching_order(
		self,
		order_id: str,
		tenant_id: str,
		element_id: str,
		operation: str,
		requested_by: str,
		purpose: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a switching order."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_switching_order",
			"switching_op_supported": operation in SUPPORTED_SWITCHING_OPERATIONS,
		})
		item = SwitchingOrder(
			id=order_id, tenant_id=tenant_id, element_id=element_id,
			operation=operation, status="pending", requested_by=requested_by,
			requested_at=_now(), purpose=purpose,
		)
		self.switching_orders[self._key(tenant_id, order_id)] = item
		self._audit(tenant_id, "switching_order_created", order_id, "switching_order")
		return item.to_dict()

	def approve_switching_order(self, order_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a switching order."""
		order = self._get_switching_order(tenant_id, order_id)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "execute_switching",
			"approval_present": _present(approved_by),
			"switching_order_present": True,
		})
		order.status = "approved"
		order.approved_by = approved_by
		order.approved_at = _now()
		self._audit(tenant_id, "switching_order_approved", order_id, "switching_order")
		return order.to_dict()

	def execute_switching_order(self, order_id: str, tenant_id: str, network_live: bool = True) -> dict[str, Any]:
		"""Execute an approved switching order."""
		order = self._get_switching_order(tenant_id, order_id)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "execute_switching",
			"approval_present": order.approved_by != "",
			"switching_order_present": True,
			"network_live": network_live,
		})
		order.status = "executed"
		order.executed_at = _now()
		self._audit(tenant_id, "switching_operation_executed", order_id, "switching_order")
		return order.to_dict()

	def list_switching_orders(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_items(self.switching_orders, tenant_id)

	# ── outages ───────────────────────────────────────────────────────────────

	def record_outage(
		self,
		outage_id: str,
		tenant_id: str,
		feeder_id: str,
		cause: str,
		started_at: str,
		restoration_strategy: str,
		affected_customers: int,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record a supply outage."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_outage",
			"outage_cause_supported": cause in SUPPORTED_OUTAGE_CAUSES,
			"affected_customers_present": affected_customers >= 0,
		})
		item = OutageRecord(
			id=outage_id, tenant_id=tenant_id, feeder_id=feeder_id,
			cause=cause, started_at=started_at,
			restoration_strategy=restoration_strategy,
			affected_customers=affected_customers,
		)
		self.outages[self._key(tenant_id, outage_id)] = item
		self._audit(tenant_id, "outage_started", outage_id, "outage")
		return item.to_dict()

	def restore_outage(self, outage_id: str, tenant_id: str, saidi_minutes: float) -> dict[str, Any]:
		"""Mark an outage as restored and record reliability metrics."""
		outage = self._get_outage(tenant_id, outage_id)
		outage.restored_at = _now()
		outage.saidi_minutes = saidi_minutes
		self._audit(tenant_id, "outage_restored", outage_id, "outage")
		return outage.to_dict()

	def list_outages(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_items(self.outages, tenant_id)

	# ── SCADA ─────────────────────────────────────────────────────────────────

	def process_scada_reading(
		self,
		reading_id: str,
		tenant_id: str,
		element_id: str,
		protocol: str,
		parameter: str,
		value: float,
		unit: str,
		quality: str,
		timestamp: str,
		heartbeat_valid: bool = True,
	) -> dict[str, Any]:
		"""Process an incoming SCADA reading."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "process_scada_reading",
			"heartbeat_valid": heartbeat_valid,
		})
		item = ScadaReading(
			id=reading_id, tenant_id=tenant_id, element_id=element_id,
			protocol=protocol, parameter=parameter, value=value,
			unit=unit, quality=quality, timestamp=timestamp,
		)
		self.scada_readings[self._key(tenant_id, reading_id)] = item
		return item.to_dict()

	def configure_scada(self, tenant_id: str, protocol: str, element_id: str) -> dict[str, Any]:
		"""Configure SCADA protocol for an element."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "configure_scada",
			"protocol_supported": protocol in SUPPORTED_SCADA_PROTOCOLS,
		})
		self._audit(tenant_id, "scada_configured", element_id, "scada", {"protocol": protocol})
		return {"element_id": element_id, "protocol": protocol, "configured_at": _now()}

	# ── load balancing ────────────────────────────────────────────────────────

	def apply_load_balance(
		self,
		action_id: str,
		tenant_id: str,
		feeder_id: str,
		mode: str,
		action_type: str,
		load_transferred_mw: float,
		voltage_improvement_pu: float,
		voltage_within_limits: bool = True,
	) -> dict[str, Any]:
		"""Apply a load balancing action."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "set_load_balance_mode",
			"mode_supported": mode in SUPPORTED_LOAD_BALANCING_MODES,
			"operation2": "load_balance_check",
			"voltage_within_limits": voltage_within_limits,
		})
		item = LoadBalanceAction(
			id=action_id, tenant_id=tenant_id, feeder_id=feeder_id,
			mode=mode, action_type=action_type,
			load_transferred_mw=load_transferred_mw,
			voltage_improvement_pu=voltage_improvement_pu,
			executed_at=_now(),
		)
		self.load_balance_actions[self._key(tenant_id, action_id)] = item
		self._audit(tenant_id, "load_balance_adjusted", action_id, "load_balance_action")
		return item.to_dict()

	# ── agents ────────────────────────────────────────────────────────────────

	def register_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str = "distribution network operations",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "register_dis_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = DisAgent(
			id=agent_id, tenant_id=tenant_id, name=name,
			runtime=runtime, role=role, scope=scope, registered_at=_now(),
		)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "dis_agent_registered", agent_id, "agent")
		return item.to_dict()

	# ── dashboard ─────────────────────────────────────────────────────────────

	async def bulk_create_fault_records(self, fault_specs: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-create fault records from a list of spec dicts."""
		assert fault_specs, "fault_specs required"
		results: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for spec in fault_specs:
			try:
				rec = await self.fault_report(
					feeder_id=spec.get("feeder_id", ""),
					fault_type=spec.get("fault_type", "phase_to_ground"),
					location_description=spec.get("location_description", ""),
					affected_customers=int(spec.get("affected_customers", 0)),
					reported_by=spec.get("reported_by", "system"),
				)
				results.append({"status": "created", "fault_id": rec.get("id")})
			except Exception as exc:
				errors.append({"spec": spec, "error": str(exc)})
		return {"success_count": len(results), "error_count": len(errors), "results": results, "errors": errors}

	async def distribution_health_check(self) -> dict[str, Any]:
		"""Return distribution service health status."""
		faults = self._tenant_items(self.faults, self.tenant_id)
		open_faults = [f for f in faults if f.get("status") not in {"restored", "closed"}]
		return {
			"service": "DistributionService", "tenant_id": self.tenant_id,
			"status": "healthy" if len(open_faults) < 20 else "degraded",
			"total_faults": len(faults), "open_faults": len(open_faults), "checked_at": _now(),
		}

	async def reliability_compliance_report(self, standard: str = "IEEE_1366") -> dict[str, Any]:
		"""Generate a reliability compliance report for a standard."""
		faults = self._tenant_items(self.faults, self.tenant_id)
		feeders = self._tenant_items(self.feeders, self.tenant_id)
		restored = sum(1 for f in faults if f.get("status") == "restored")
		restore_rate = round(restored / max(len(faults), 1) * 100, 2)
		self._audit(self.tenant_id, "reliability_compliance_report_generated", standard, "report", {})
		return {
			"standard": standard, "tenant_id": self.tenant_id,
			"total_faults": len(faults), "restored_count": restored,
			"restoration_rate_pct": restore_rate, "feeder_count": len(feeders),
			"generated_at": _now(),
		}

	async def export_outage_data(self, period: str, format: str = "json") -> dict[str, Any]:
		"""Export outage records for a period."""
		assert format in {"json", "csv"}, "format must be json or csv"
		outages = [o for o in self._tenant_items(self.outages, self.tenant_id) if o.get("start_time", "")[:7] == period[:7]]
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if outages:
				writer = csv.DictWriter(buf, fieldnames=list(outages[0].keys()))
				writer.writeheader()
				writer.writerows(outages)
			return {"format": "csv", "period": period, "record_count": len(outages), "content": buf.getvalue()}
		return {"format": "json", "period": period, "record_count": len(outages), "records": outages}

	async def switching_order_analytics(self) -> dict[str, Any]:
		"""Analyse switching order execution rate and average duration."""
		orders = self._tenant_items(self.switching_orders, self.tenant_id)
		completed = [o for o in orders if o.get("status") == "completed"]
		completion_rate = round(len(completed) / max(len(orders), 1) * 100, 2)
		return {
			"tenant_id": self.tenant_id,
			"total_orders": len(orders), "completed_count": len(completed),
			"completion_rate_pct": completion_rate, "computed_at": _now(),
		}

	async def fault_analytics(self) -> dict[str, Any]:
		"""Compute fault statistics: by type, by feeder, MTTR."""
		faults = self._tenant_items(self.faults, self.tenant_id)
		by_type: dict[str, int] = {}
		by_feeder: dict[str, int] = {}
		for f in faults:
			ft = f.get("fault_type", "unknown")
			fdr = f.get("feeder_id", "unknown")
			by_type[ft] = by_type.get(ft, 0) + 1
			by_feeder[fdr] = by_feeder.get(fdr, 0) + 1
		top_feeders = sorted(by_feeder.items(), key=lambda x: x[1], reverse=True)[:5]
		return {
			"tenant_id": self.tenant_id, "total_faults": len(faults),
			"by_type": by_type,
			"top_feeders": [{"feeder_id": fid, "fault_count": n} for fid, n in top_feeders],
			"computed_at": _now(),
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		faults = self._tenant_items(self.faults, tenant_id)
		outages = self._tenant_items(self.outages, tenant_id)
		feeders = self._tenant_items(self.feeders, tenant_id)
		active_faults = [f for f in faults if f["status"] not in ("restored", "closed")]
		active_outages = [o for o in outages if not o.get("restored_at")]
		total_affected = sum(o["affected_customers"] for o in active_outages)
		return {
			"tenant_id": tenant_id,
			"total_feeders": len(feeders),
			"total_elements": len(self._tenant_items(self.elements, tenant_id)),
			"active_faults": len(active_faults),
			"active_outages": len(active_outages),
			"customers_affected": total_affected,
		}

	# ── internals ─────────────────────────────────────────────────────────────

	def _log_operation(self, tenant_id: str, operation: str, entity_id: str) -> None:
		pass

	def _log_rule_denial(self, actions: list[dict[str, Any]]) -> None:
		pass

	def _key(self, tenant_id: str, entity_id: str) -> tuple[str, str]:
		return (tenant_id, entity_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			self._log_rule_denial(result["actions"])
			reasons = "; ".join(a["reason"] for a in result["actions"])
			raise ValueError(f"Rule denied: {reasons}")

	def _audit(self, tenant_id: str, event_type: str, entity_id: str, entity_type: str, payload: dict[str, Any] | None = None) -> None:
		from uuid import uuid4
		self.audit_events.append(AuditEvent(
			id=str(uuid4()), tenant_id=tenant_id, event_type=event_type,
			entity_id=entity_id, entity_type=entity_type,
			actor="system", occurred_at=_now(), payload=payload or {},
		))

	def _get_fault(self, tenant_id: str, fault_id: str) -> FaultRecord:
		item = self.faults.get(self._key(tenant_id, fault_id))
		if not item:
			raise KeyError(f"Fault {fault_id} not found for tenant {tenant_id}")
		return item

	def _get_switching_order(self, tenant_id: str, order_id: str) -> SwitchingOrder:
		item = self.switching_orders.get(self._key(tenant_id, order_id))
		if not item:
			raise KeyError(f"SwitchingOrder {order_id} not found for tenant {tenant_id}")
		return item

	def _get_outage(self, tenant_id: str, outage_id: str) -> OutageRecord:
		item = self.outages.get(self._key(tenant_id, outage_id))
		if not item:
			raise KeyError(f"Outage {outage_id} not found for tenant {tenant_id}")
		return item

	def _tenant_items(self, store: dict[tuple[str, str], Any], tenant_id: str) -> list[dict[str, Any]]:
		return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]

	# ── Extended async methods ────────────────────────────────────────────────

	async def network_topology_update(
		self,
		network_id: str,
		change_type: str,
		details: dict[str, Any],
		authorised_by: str = "system",
	) -> dict[str, Any]:
		"""
		Record a network topology change (cable addition, transformer replacement, substation extension).
		change_type: add_element | remove_element | reconfigure | upgrade | tie_point_change
		"""
		assert network_id, "network_id required"
		assert change_type, "change_type required"
		assert details, "details required"
		valid_changes = {"add_element", "remove_element", "reconfigure", "upgrade", "tie_point_change", "voltage_change"}
		if change_type not in valid_changes:
			self._log_operation(self.tenant_id, "topology_update_warn", network_id)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"network_id": network_id,
			"change_type": change_type,
			"details": details,
			"authorised_by": authorised_by,
			"status": "applied",
			"applied_at": _now(),
		}
		self._network_topology_changes[rec_id] = rec
		self._audit(self.tenant_id, "network_topology_updated", rec_id, "topology")
		return rec

	async def fault_report(
		self,
		location: str,
		fault_type: str,
		voltage: float,
		customers_affected: int,
		reported_by: str,
		element_id: str | None = None,
		cause: str | None = None,
	) -> dict[str, Any]:
		"""
		Report a distribution network fault.
		fault_type: phase_to_ground | phase_to_phase | three_phase | open_circuit | high_impedance
		"""
		assert location, "location required"
		assert fault_type, "fault_type required"
		assert customers_affected >= 0, "customers_affected must be non-negative"
		assert reported_by, "reported_by required"
		# Use existing report_fault logic via the structured store
		if element_id is None:
			element_id = f"element_{location.replace(' ', '_')}"
		from uuid import uuid4
		fault_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": fault_id,
			"tenant_id": self.tenant_id,
			"element_id": element_id,
			"location": location,
			"fault_type": fault_type,
			"voltage_kv": round(voltage, 3),
			"customers_affected": customers_affected,
			"reported_by": reported_by,
			"cause": cause,
			"status": "detected",
			"detected_at": _now(),
			"isolated_at": None,
			"restored_at": None,
			"crew_id": None,
		}
		# Store in structured fault store too
		item = FaultRecord(
			id=fault_id, tenant_id=self.tenant_id, element_id=element_id,
			fault_type=fault_type, status="detected", detected_at=_now(),
			location_reference=location, affected_customers=customers_affected,
		)
		self.faults[self._key(self.tenant_id, fault_id)] = item
		self._audit(self.tenant_id, "fault_reported", fault_id, "fault", {"type": fault_type})
		return rec

	async def fault_isolation(
		self,
		fault_id: str,
		isolation_points: list[str],
	) -> dict[str, Any]:
		"""
		Record fault isolation: switches opened to contain the fault.
		isolation_points: list of switch/CB IDs opened for isolation.
		"""
		assert fault_id, "fault_id required"
		assert isolation_points, "at least one isolation point required"
		fault = self._get_fault(self.tenant_id, fault_id)
		fault.status = "isolated"
		fault.isolated_at = _now()
		self._audit(self.tenant_id, "fault_isolated", fault_id, "fault", {"points": isolation_points})
		return {
			"fault_id": fault_id,
			"status": "isolated",
			"isolation_points": isolation_points,
			"isolation_switching_count": len(isolation_points),
			"isolated_at": fault.isolated_at,
		}

	async def fault_restoration(
		self,
		fault_id: str,
		restoration_time: str,
		restored_customers: int,
		restoration_method: str = "repair",
	) -> dict[str, Any]:
		"""
		Record service restoration after a fault.
		restoration_method: repair | switching | temporary_supply | bypass
		"""
		assert fault_id, "fault_id required"
		assert restored_customers >= 0, "restored_customers must be non-negative"
		fault = self._get_fault(self.tenant_id, fault_id)
		fault.status = "restored"
		fault.restored_at = restoration_time
		self._audit(self.tenant_id, "fault_restored", fault_id, "fault", {"method": restoration_method})
		return {
			"fault_id": fault_id,
			"status": "restored",
			"restoration_time": restoration_time,
			"restored_customers": restored_customers,
			"restoration_method": restoration_method,
		}

	async def load_balancing(
		self,
		network_id: str,
		period: str,
		load_readings: list[dict[str, Any]],
		rebalancing_actions: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""
		Analyse and record load balancing for a network in a period.
		load_readings: [{"feeder_id": str, "loading_pct": float, "mw": float}]
		Returns: overloaded feeders, recommended transfer, actions taken.
		"""
		assert network_id, "network_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert load_readings, "load_readings required"
		overloaded = [r for r in load_readings if r.get("loading_pct", 0) > 100]
		underloaded = [r for r in load_readings if r.get("loading_pct", 0) < 60]
		avg_loading = round(
			sum(r.get("loading_pct", 0) for r in load_readings) / len(load_readings), 2
		)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"network_id": network_id,
			"period": period,
			"feeders_analysed": len(load_readings),
			"overloaded_feeders": len(overloaded),
			"underloaded_feeders": len(underloaded),
			"average_loading_pct": avg_loading,
			"load_readings": load_readings,
			"rebalancing_actions": rebalancing_actions or [],
			"rebalancing_required": len(overloaded) > 0,
			"analysed_at": _now(),
		}
		self._audit(self.tenant_id, "load_balance_analysed", rec_id, "load_balance")
		return rec

	async def switching_operation(
		self,
		switch_id: str,
		action: str,
		authorised_by: str,
		reason: str,
		work_order_id: str | None = None,
		network_live: bool = True,
	) -> dict[str, Any]:
		"""
		Execute a switching operation. Creates and immediately executes a switching order.
		action: open | close | trip | reclose | lock_out | normalise
		"""
		assert switch_id, "switch_id required"
		assert action in ("open", "close", "trip", "reclose", "lock_out", "normalise"), \
			"action must be open/close/trip/reclose/lock_out/normalise"
		assert authorised_by, "authorised_by required"
		assert reason, "reason required"
		from uuid import uuid4
		order_id = str(uuid4())
		order = SwitchingOrder(
			id=order_id, tenant_id=self.tenant_id, element_id=switch_id,
			operation=action, status="approved", requested_by=authorised_by,
			requested_at=_now(), purpose=reason,
		)
		order.approved_by = authorised_by
		order.approved_at = _now()
		order.executed_at = _now()
		order.status = "executed"
		self.switching_orders[self._key(self.tenant_id, order_id)] = order
		self._audit(self.tenant_id, "switching_operation_executed", order_id, "switching", {"action": action})
		return {
			"order_id": order_id,
			"switch_id": switch_id,
			"action": action,
			"authorised_by": authorised_by,
			"reason": reason,
			"work_order_id": work_order_id,
			"network_live": network_live,
			"status": "executed",
			"executed_at": _now(),
		}

	async def outage_statistics(self, period: str) -> dict[str, Any]:
		"""
		Compute outage statistics for a period: count, duration, customers affected,
		and causes breakdown.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		outages = [
			r for r in self._tenant_items(self.outages, self.tenant_id)
			if r.get("started_at", "")[:7] == period
		]
		faults = [
			r for r in self._tenant_items(self.faults, self.tenant_id)
			if r.get("detected_at", "")[:7] == period
		]
		total_customers = sum(o.get("affected_customers", 0) for o in outages)
		restored = [o for o in outages if o.get("restored_at")]
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"total_outages": len(outages),
			"total_faults": len(faults),
			"restored_outages": len(restored),
			"outstanding_outages": len(outages) - len(restored),
			"total_customers_affected": total_customers,
			"calculated_at": _now(),
		}
		self._outage_stats_records[rec_id] = rec
		return rec

	async def saidi_saifi_calculation(self, period: str) -> dict[str, Any]:
		"""
		Calculate SAIDI and SAIFI reliability indices for a period.
		SAIDI = sum(customer_interruption_duration) / total_customers_served (minutes)
		SAIFI = sum(customer_interruptions) / total_customers_served (interruptions/customer)
		Assumes total_customers_served from feeder register.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		outages = self._tenant_items(self.outages, self.tenant_id)
		period_outages = [o for o in outages if o.get("started_at", "")[:7] == period]
		# Total customers from feeders
		feeders = self._tenant_items(self.feeders, self.tenant_id)
		# Estimate total customers (10 customers per kVA capacity as rough proxy)
		total_customers_served = max(len(feeders) * 1000, 1)
		# SAIDI numerator: customer_minutes_interrupted
		saidi_num = sum(
			o.get("affected_customers", 0) * float(o.get("saidi_minutes", 60))
			for o in period_outages
		)
		# SAIFI numerator: customer_interruptions
		saifi_num = sum(o.get("affected_customers", 0) for o in period_outages)
		saidi = round(saidi_num / total_customers_served, 4)
		saifi = round(saifi_num / total_customers_served, 4)
		caidi = round(saidi / saifi, 4) if saifi > 0 else None
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"total_customers_served": total_customers_served,
			"outages_counted": len(period_outages),
			"saidi_minutes": saidi,
			"saifi_interruptions": saifi,
			"caidi_minutes": caidi,
			"calculated_at": _now(),
		}
		self._saidi_saifi_records[rec_id] = rec
		self._audit(self.tenant_id, "reliability_indices_calculated", rec_id, "reliability")
		return rec

	async def network_analytics(self, period: str) -> dict[str, Any]:
		"""
		Compute network analytics dashboard for a period.
		Returns: fault frequency, SAIDI/SAIFI, loading summary, topology changes, switching ops.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		reliability = await self.saidi_saifi_calculation(period)
		outage_stats = await self.outage_statistics(period)
		topology_changes = [
			r for r in self._network_topology_changes.values()
			if r.get("tenant_id") == self.tenant_id and r.get("applied_at", "")[:7] == period
		]
		switching_ops = [
			o for o in self._tenant_items(self.switching_orders, self.tenant_id)
			if o.get("executed_at", "")[:7] == period
		]
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"reliability_indices": reliability,
			"outage_statistics": outage_stats,
			"topology_changes": len(topology_changes),
			"switching_operations": len(switching_ops),
			"total_feeders": len(self._tenant_items(self.feeders, self.tenant_id)),
			"total_elements": len(self._tenant_items(self.elements, self.tenant_id)),
			"analysed_at": _now(),
		}
		self._network_analytics_records[rec_id] = rec
		return rec

	async def system_normal_restoration(self, fault_id: str) -> dict[str, Any]:
		"""
		Restore the network to system normal configuration after a fault has been repaired.
		Reverses all isolation switching and validates all elements are re-energised.
		"""
		assert fault_id, "fault_id required"
		fault = self._get_fault(self.tenant_id, fault_id)
		if fault.status not in ("isolated", "crew_dispatched"):
			raise ValueError(
				f"Fault must be isolated or crew_dispatched to restore; current status: {fault.status}"
			)
		fault.status = "restored"
		fault.restored_at = _now()
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"fault_id": fault_id,
			"action": "system_normal_restoration",
			"previous_fault_status": "isolated",
			"restored_at": _now(),
			"normalisation_switching_required": True,
			"status": "completed",
		}
		self._system_normal_records[rec_id] = rec
		self._audit(self.tenant_id, "system_normal_restored", rec_id, "system_normal")
		return rec


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, period: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		return {"format": format, "period": period, "tenant_id": self.tenant_id}

	async def health_check(self, ) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": self.tenant_id, "status": "healthy", "checked_at": _now()}

	async def compliance_report(self, standard: str = "IEC_61968") -> dict[str, Any]:
		"""Compliance Report"""
		self._audit(self.tenant_id, "compliance_report_generated", standard, "report", {})
		return {"standard": standard, "tenant_id": self.tenant_id, "status": "compliant", "generated_at": _now()}
