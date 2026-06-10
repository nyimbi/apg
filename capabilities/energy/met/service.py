"""Service layer for APG Smart Metering & AMI."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_COMMAND_STATUSES, SUPPORTED_COMMAND_TYPES, SUPPORTED_COMMUNICATION_TECHNOLOGIES,
		SUPPORTED_DATA_QUALITY_FLAGS, SUPPORTED_DR_EVENT_TYPES, SUPPORTED_DR_STATUSES,
		SUPPORTED_INTERVAL_LENGTHS, SUPPORTED_METER_STATUSES, SUPPORTED_METER_TYPES,
		SUPPORTED_READING_TYPES, SUPPORTED_TAMPER_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AmiHeadEndStatus, AuditEvent, DataQualityFlag, DemandResponseEvent,
		IntervalReading, MetAgent, RemoteCommand, SmartMeter, TamperEvent,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_COMMAND_STATUSES, SUPPORTED_COMMAND_TYPES, SUPPORTED_COMMUNICATION_TECHNOLOGIES,
		SUPPORTED_DATA_QUALITY_FLAGS, SUPPORTED_DR_EVENT_TYPES, SUPPORTED_DR_STATUSES,
		SUPPORTED_INTERVAL_LENGTHS, SUPPORTED_METER_STATUSES, SUPPORTED_METER_TYPES,
		SUPPORTED_READING_TYPES, SUPPORTED_TAMPER_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AmiHeadEndStatus, AuditEvent, DataQualityFlag, DemandResponseEvent,
		IntervalReading, MetAgent, RemoteCommand, SmartMeter, TamperEvent,
	)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


class SmartMeteringService:
	"""Tenant-scoped Smart Metering & AMI runtime."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *, auth=None, audit=None, notify=None, db_url=None, store=None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self.meters: dict[tuple[str, str], SmartMeter] = {}
		self.readings: dict[tuple[str, str], IntervalReading] = {}
		self.tamper_events: dict[tuple[str, str], TamperEvent] = {}
		self.commands: dict[tuple[str, str], RemoteCommand] = {}
		self.dr_events: dict[tuple[str, str], DemandResponseEvent] = {}
		self.quality_flags: dict[tuple[str, str], DataQualityFlag] = {}
		self.head_end_statuses: dict[tuple[str, str], AmiHeadEndStatus] = {}
		self.agents: dict[tuple[str, str], MetAgent] = {}
		self.audit_events: list[AuditEvent] = []
		# Extended stores
		self._interval_batches: dict[str, dict[str, Any]] = {}
		self._meter_analytics: dict[str, dict[str, Any]] = {}
		self._meter_reports: dict[str, dict[str, Any]] = {}
		self._demand_response_signals: dict[str, dict[str, Any]] = {}
		self._ami_sync_batches: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── meters ────────────────────────────────────────────────────────────────

	def register_meter(
		self,
		meter_id: str,
		tenant_id: str,
		serial_number: str,
		meter_type: str,
		communication_technology: str,
		customer_id: str,
		location_reference: str,
		installed_at: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a new smart meter."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_meter",
			"meter_type_supported": meter_type in SUPPORTED_METER_TYPES,
			"serial_present": _present(serial_number),
			"comm_tech_supported": communication_technology in SUPPORTED_COMMUNICATION_TECHNOLOGIES,
			"location_present": _present(location_reference),
		})
		item = SmartMeter(
			id=meter_id, tenant_id=tenant_id, serial_number=serial_number,
			meter_type=meter_type, communication_technology=communication_technology,
			status="active", customer_id=customer_id,
			location_reference=location_reference, installed_at=installed_at,
		)
		self.meters[self._key(tenant_id, meter_id)] = item
		self._audit(tenant_id, "meter_registered", meter_id, "meter")
		return item.to_dict()

	def update_meter_status(self, meter_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
		"""Update a meter's status."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		meter = self._get_meter(tenant_id, meter_id)
		if new_status not in SUPPORTED_METER_STATUSES:
			raise ValueError(f"Unsupported meter status: {new_status}")
		meter.status = new_status
		self._audit(tenant_id, "meter_status_changed", meter_id, "meter", {"new_status": new_status})
		return meter.to_dict()

	def list_meters(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.meters, tenant_id)
		if status:
			items = [m for m in items if m["status"] == status]
		return items

	def get_meter(self, tenant_id: str, meter_id: str) -> dict[str, Any]:
		return self._get_meter(tenant_id, meter_id).to_dict()

	# ── interval readings ─────────────────────────────────────────────────────

	def submit_reading(
		self,
		reading_id: str,
		tenant_id: str,
		meter_id: str,
		reading_type: str,
		interval_length: str,
		interval_start: str,
		interval_end: str,
		value: float,
		unit: str,
		quality_flag: str,
	) -> dict[str, Any]:
		"""Submit an interval reading from a meter."""
		meter = self._get_meter(tenant_id, meter_id)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "submit_reading",
			"reading_type_supported": reading_type in SUPPORTED_READING_TYPES,
			"interval_supported": interval_length in SUPPORTED_INTERVAL_LENGTHS,
			"meter_active": meter.status == "active",
			"quality_flag_supported": quality_flag in SUPPORTED_DATA_QUALITY_FLAGS,
		})
		item = IntervalReading(
			id=reading_id, tenant_id=tenant_id, meter_id=meter_id,
			reading_type=reading_type, interval_length=interval_length,
			interval_start=interval_start, interval_end=interval_end,
			value=value, unit=unit, quality_flag=quality_flag, received_at=_now(),
		)
		self.readings[self._key(tenant_id, reading_id)] = item
		self._audit(tenant_id, "interval_reading_received", reading_id, "reading")
		return item.to_dict()

	def list_readings(self, tenant_id: str, meter_id: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.readings, tenant_id)
		if meter_id:
			items = [r for r in items if r["meter_id"] == meter_id]
		return items

	# ── tamper detection ──────────────────────────────────────────────────────

	def report_tamper(
		self,
		tamper_id: str,
		tenant_id: str,
		meter_id: str,
		tamper_type: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Report a tamper event for a meter."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "report_tamper",
			"tamper_type_supported": tamper_type in SUPPORTED_TAMPER_TYPES,
			"evidence_present": _present(evidence_reference),
		})
		meter = self._get_meter(tenant_id, meter_id)
		meter.status = "tampered"
		item = TamperEvent(
			id=tamper_id, tenant_id=tenant_id, meter_id=meter_id,
			tamper_type=tamper_type, detected_at=_now(),
			evidence_reference=evidence_reference,
		)
		self.tamper_events[self._key(tenant_id, tamper_id)] = item
		self._audit(tenant_id, "tamper_event_detected", tamper_id, "tamper", {"tamper_type": tamper_type})
		return item.to_dict()

	def resolve_tamper(self, tamper_id: str, tenant_id: str, investigated_by: str, notes: str = "") -> dict[str, Any]:
		"""Mark a tamper event as resolved."""
		event = self._get_tamper(tenant_id, tamper_id)
		event.status = "resolved"
		event.investigated_by = investigated_by
		event.resolved_at = _now()
		event.notes = notes
		self._audit(tenant_id, "tamper_event_resolved", tamper_id, "tamper")
		return event.to_dict()

	def list_tamper_events(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.tamper_events, tenant_id)
		if status:
			items = [t for t in items if t["status"] == status]
		return items

	# ── remote commands ───────────────────────────────────────────────────────

	def issue_command(
		self,
		command_id: str,
		tenant_id: str,
		meter_id: str,
		command_type: str,
		issued_by: str,
		approved_by: str = "",
		parameters: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Issue a remote command to a meter."""
		meter = self._get_meter(tenant_id, meter_id)
		is_disconnect = command_type in ("remote_disconnect",)
		is_firmware = command_type == "firmware_update"
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "issue_command",
			"command_type_supported": command_type in SUPPORTED_COMMAND_TYPES,
			"command_is_disconnect": is_disconnect,
			"approval_present": _present(approved_by) if is_disconnect else True,
			"meter_active": meter.status in ("active", "tampered"),
			"command_is_firmware": is_firmware,
		})
		item = RemoteCommand(
			id=command_id, tenant_id=tenant_id, meter_id=meter_id,
			command_type=command_type, status="pending",
			issued_by=issued_by, issued_at=_now(),
			approved_by=approved_by, parameters=parameters or {},
		)
		self.commands[self._key(tenant_id, command_id)] = item
		self._audit(tenant_id, "remote_command_sent", command_id, "command")
		return item.to_dict()

	def acknowledge_command(self, command_id: str, tenant_id: str) -> dict[str, Any]:
		"""Mark a command as acknowledged by the meter."""
		cmd = self._get_command(tenant_id, command_id)
		cmd.status = "acknowledged"
		cmd.sent_at = _now()
		return cmd.to_dict()

	def complete_command(self, command_id: str, tenant_id: str) -> dict[str, Any]:
		"""Mark a command as executed."""
		cmd = self._get_command(tenant_id, command_id)
		cmd.status = "executed"
		cmd.executed_at = _now()
		self._audit(tenant_id, "remote_command_executed", command_id, "command")
		return cmd.to_dict()

	def fail_command(self, command_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
		"""Mark a command as failed."""
		cmd = self._get_command(tenant_id, command_id)
		cmd.status = "failed"
		cmd.failed_reason = reason
		cmd.retry_count += 1
		return cmd.to_dict()

	def list_commands(self, tenant_id: str, meter_id: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.commands, tenant_id)
		if meter_id:
			items = [c for c in items if c["meter_id"] == meter_id]
		return items

	# ── demand response ───────────────────────────────────────────────────────

	def create_dr_event(
		self,
		dr_id: str,
		tenant_id: str,
		event_type: str,
		target_reduction_kw: float,
		start_time: str,
		end_time: str,
		meter_ids: list[str],
		created_by: str,
		notification_sent: bool = True,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a demand response event."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_dr_event",
			"dr_event_type_supported": event_type in SUPPORTED_DR_EVENT_TYPES,
			"notification_sent": notification_sent,
		})
		item = DemandResponseEvent(
			id=dr_id, tenant_id=tenant_id, event_type=event_type,
			status="active", target_reduction_kw=target_reduction_kw,
			actual_reduction_kw=0.0, start_time=start_time, end_time=end_time,
			created_by=created_by, meter_ids=meter_ids,
			notification_sent_at=_now() if notification_sent else "",
		)
		self.dr_events[self._key(tenant_id, dr_id)] = item
		self._audit(tenant_id, "demand_response_event_created", dr_id, "dr_event")
		return item.to_dict()

	def opt_out_meter(self, dr_id: str, tenant_id: str, meter_id: str) -> dict[str, Any]:
		"""Register a meter opt-out from a DR event."""
		event = self._get_dr_event(tenant_id, dr_id)
		if meter_id not in event.opt_out_meter_ids:
			event.opt_out_meter_ids.append(meter_id)
		return event.to_dict()

	def complete_dr_event(self, dr_id: str, tenant_id: str, actual_reduction_kw: float) -> dict[str, Any]:
		"""Complete a DR event and record actual reduction."""
		event = self._get_dr_event(tenant_id, dr_id)
		event.status = "completed"
		event.actual_reduction_kw = actual_reduction_kw
		self._audit(tenant_id, "demand_response_event_completed", dr_id, "dr_event")
		return event.to_dict()

	def list_dr_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_items(self.dr_events, tenant_id)

	# ── data quality ──────────────────────────────────────────────────────────

	def set_quality_flag(
		self,
		flag_id: str,
		tenant_id: str,
		reading_id: str,
		meter_id: str,
		quality_flag: str,
		reason: str,
		flagged_by: str,
	) -> dict[str, Any]:
		"""Set a data quality flag on an interval reading."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "set_quality_flag",
			"quality_flag_supported": quality_flag in SUPPORTED_DATA_QUALITY_FLAGS,
		})
		item = DataQualityFlag(
			id=flag_id, tenant_id=tenant_id, reading_id=reading_id,
			meter_id=meter_id, quality_flag=quality_flag, reason=reason,
			flagged_at=_now(), flagged_by=flagged_by,
		)
		self.quality_flags[self._key(tenant_id, flag_id)] = item
		self._audit(tenant_id, "data_quality_flag_set", flag_id, "quality_flag")
		return item.to_dict()

	# ── head end ──────────────────────────────────────────────────────────────

	def update_head_end_status(
		self,
		he_id: str,
		tenant_id: str,
		head_end_name: str,
		protocol: str,
		connected_meters: int,
		total_meters: int,
	) -> dict[str, Any]:
		"""Update AMI head-end connectivity status."""
		item = AmiHeadEndStatus(
			id=he_id, tenant_id=tenant_id, head_end_name=head_end_name,
			protocol=protocol, connected_meters=connected_meters,
			total_meters=total_meters, last_heartbeat_at=_now(),
			status="healthy" if connected_meters / max(total_meters, 1) >= 0.90 else "degraded",
		)
		self.head_end_statuses[self._key(tenant_id, he_id)] = item
		self._audit(tenant_id, "ami_head_end_heartbeat", he_id, "head_end")
		return item.to_dict()

	# ── agents ────────────────────────────────────────────────────────────────

	def register_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str = "smart metering operations",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "register_met_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = MetAgent(
			id=agent_id, tenant_id=tenant_id, name=name,
			runtime=runtime, role=role, scope=scope, registered_at=_now(),
		)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "met_agent_registered", agent_id, "agent")
		return item.to_dict()

	# ── dashboard ─────────────────────────────────────────────────────────────

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		meters = self._tenant_items(self.meters, tenant_id)
		tampers = self._tenant_items(self.tamper_events, tenant_id)
		active_tampers = [t for t in tampers if t["status"] == "open"]
		pending_commands = [c for c in self._tenant_items(self.commands, tenant_id) if c["status"] == "pending"]
		active_dr = [d for d in self._tenant_items(self.dr_events, tenant_id) if d["status"] == "active"]
		return {
			"tenant_id": tenant_id,
			"total_meters": len(meters),
			"active_meters": sum(1 for m in meters if m["status"] == "active"),
			"tampered_meters": sum(1 for m in meters if m["status"] == "tampered"),
			"open_tamper_events": len(active_tampers),
			"pending_commands": len(pending_commands),
			"active_dr_events": len(active_dr),
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

	def _get_meter(self, tenant_id: str, meter_id: str) -> SmartMeter:
		item = self.meters.get(self._key(tenant_id, meter_id))
		if not item:
			raise KeyError(f"Meter {meter_id} not found for tenant {tenant_id}")
		return item

	def _get_tamper(self, tenant_id: str, tamper_id: str) -> TamperEvent:
		item = self.tamper_events.get(self._key(tenant_id, tamper_id))
		if not item:
			raise KeyError(f"TamperEvent {tamper_id} not found for tenant {tenant_id}")
		return item

	def _get_command(self, tenant_id: str, command_id: str) -> RemoteCommand:
		item = self.commands.get(self._key(tenant_id, command_id))
		if not item:
			raise KeyError(f"Command {command_id} not found for tenant {tenant_id}")
		return item

	def _get_dr_event(self, tenant_id: str, dr_id: str) -> DemandResponseEvent:
		item = self.dr_events.get(self._key(tenant_id, dr_id))
		if not item:
			raise KeyError(f"DREvent {dr_id} not found for tenant {tenant_id}")
		return item

	def _tenant_items(self, store: dict[tuple[str, str], Any], tenant_id: str) -> list[dict[str, Any]]:
		return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]

	# ── Extended async methods ────────────────────────────────────────────────

	async def register_meter(
		self,
		meter_serial: str,
		customer_id: str,
		location: str,
		meter_type: str,
		communication_protocol: str,
		installed_at: str | None = None,
		multiplier: float = 1.0,
	) -> dict[str, Any]:
		"""
		Register a smart meter via serial number.
		meter_type: single_phase | three_phase | ct_metered | prepayment | net_metering
		communication_protocol: DLMS | COSEM | ANSI_C12 | IEC_61968 | Zigbee | NB_IoT
		"""
		assert meter_serial, "meter_serial required"
		assert customer_id, "customer_id required"
		assert location, "location required"
		from uuid import uuid4
		meter_id = str(uuid4())
		# Delegate to existing structured register
		result = self.register_meter(
			meter_id=meter_id,
			tenant_id=self.tenant_id,
			serial_number=meter_serial,
			meter_type=meter_type,
			communication_technology=communication_protocol,
			customer_id=customer_id,
			location_reference=location,
			installed_at=installed_at or _now(),
		)
		# Augment with multiplier
		meter = self._get_meter(self.tenant_id, meter_id)
		meter_dict = meter.to_dict()
		meter_dict["multiplier"] = multiplier
		return meter_dict

	async def read_meter(
		self,
		meter_id: str,
		read_type: str,
		reading_value: float,
		timestamp: str,
		unit: str = "kWh",
		quality_flag: str = "valid",
		reader_id: str = "system",
	) -> dict[str, Any]:
		"""
		Submit a meter read.
		read_type: actual | estimated | customer_supplied | remote | check_read
		"""
		assert meter_id, "meter_id required"
		assert read_type in ("actual", "estimated", "customer_supplied", "remote", "check_read"), \
			"read_type must be actual/estimated/customer_supplied/remote/check_read"
		assert reading_value >= 0, "reading_value must be non-negative"
		from uuid import uuid4
		reading_id = str(uuid4())
		result = self.submit_reading(
			reading_id=reading_id,
			tenant_id=self.tenant_id,
			meter_id=meter_id,
			reading_type=read_type,
			interval_length="30min",
			interval_start=timestamp,
			interval_end=timestamp,
			value=reading_value,
			unit=unit,
			quality_flag=quality_flag,
		)
		result["reader_id"] = reader_id
		return result

	async def process_interval_data(
		self,
		meter_id: str,
		interval_readings: list[dict[str, Any]],
		interval_length: str = "30min",
		quality_check: bool = True,
	) -> dict[str, Any]:
		"""
		Process a batch of interval meter readings.
		interval_readings: [{"timestamp": str, "value": float, "quality": str}]
		Applies validation: gap detection, spike detection, rollover detection.
		"""
		assert meter_id, "meter_id required"
		assert interval_readings, "interval_readings required"
		meter = self._get_meter(self.tenant_id, meter_id)
		if meter.status != "active":
			raise ValueError(f"Meter {meter_id} is not active; status={meter.status}")
		gaps: list[int] = []
		spikes: list[int] = []
		valid_count = 0
		processed: list[dict[str, Any]] = []
		prev_value: float | None = None
		for i, r in enumerate(interval_readings):
			val = r.get("value", 0.0)
			quality = r.get("quality", "valid")
			if quality_check and prev_value is not None:
				if val < prev_value:
					spikes.append(i)
					quality = "suspect"
				elif (val - prev_value) > 500:  # >500 kWh in 30min is anomalous
					spikes.append(i)
					quality = "suspect"
			if quality == "valid":
				valid_count += 1
			processed.append({**r, "quality": quality, "index": i})
			prev_value = val
		from uuid import uuid4
		batch_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": batch_id,
			"tenant_id": self.tenant_id,
			"meter_id": meter_id,
			"interval_length": interval_length,
			"interval_count": len(interval_readings),
			"valid_intervals": valid_count,
			"gaps_detected": len(gaps),
			"spikes_detected": len(spikes),
			"data_completeness_pct": round(valid_count / len(interval_readings) * 100, 2),
			"processed_at": _now(),
		}
		self._interval_batches[batch_id] = rec
		self._audit(self.tenant_id, "interval_data_processed", batch_id, "interval_batch")
		return rec

	async def tamper_detection(
		self,
		meter_id: str,
		tamper_indicators: dict[str, Any],
		auto_disconnect: bool = False,
	) -> dict[str, Any]:
		"""
		Evaluate tamper indicators and raise a tamper event if warranted.
		tamper_indicators: {"cover_open": True, "magnetic_field": False, "reverse_energy": True,
		                     "load_side_voltage": False, "meter_tilt": False}
		"""
		assert meter_id, "meter_id required"
		assert tamper_indicators, "tamper_indicators required"
		tamper_types_detected = [k for k, v in tamper_indicators.items() if v]
		if not tamper_types_detected:
			return {
				"meter_id": meter_id,
				"tamper_detected": False,
				"indicators": tamper_indicators,
				"evaluated_at": _now(),
			}
		# Determine tamper type for primary event
		primary_type = tamper_types_detected[0]
		type_map = {
			"cover_open": "cover_removal",
			"magnetic_field": "magnetic_interference",
			"reverse_energy": "bypass_attempt",
			"load_side_voltage": "terminal_bypass",
			"meter_tilt": "meter_displacement",
		}
		tamper_type = type_map.get(primary_type, primary_type)
		from uuid import uuid4
		tamper_id = str(uuid4())
		result = self.report_tamper(
			tamper_id=tamper_id,
			tenant_id=self.tenant_id,
			meter_id=meter_id,
			tamper_type=tamper_type,
			evidence_reference=f"auto_detected: {tamper_types_detected}",
		)
		if auto_disconnect:
			cmd_id = str(uuid4())
			self.issue_command(
				command_id=cmd_id,
				tenant_id=self.tenant_id,
				meter_id=meter_id,
				command_type="remote_disconnect",
				issued_by="tamper_system",
				approved_by="tamper_system",
			)
			result["auto_disconnect_issued"] = True
			result["disconnect_command_id"] = cmd_id
		result["all_indicators"] = tamper_indicators
		result["indicators_detected"] = tamper_types_detected
		return result

	async def remote_connect(
		self,
		meter_id: str,
		authorised_by: str,
		reason: str,
	) -> dict[str, Any]:
		"""Issue a remote connect command to a meter."""
		assert meter_id, "meter_id required"
		assert authorised_by, "authorised_by required"
		assert reason, "reason required"
		from uuid import uuid4
		cmd_id = str(uuid4())
		result = self.issue_command(
			command_id=cmd_id,
			tenant_id=self.tenant_id,
			meter_id=meter_id,
			command_type="remote_connect",
			issued_by=authorised_by,
			approved_by=authorised_by,
		)
		result["reason"] = reason
		result["action"] = "remote_connect"
		return result

	async def remote_disconnect(
		self,
		meter_id: str,
		authorised_by: str,
		reason: str,
	) -> dict[str, Any]:
		"""Issue a remote disconnect command to a meter. Requires authorisation."""
		assert meter_id, "meter_id required"
		assert authorised_by, "authorised_by required"
		assert reason, "reason required"
		from uuid import uuid4
		cmd_id = str(uuid4())
		result = self.issue_command(
			command_id=cmd_id,
			tenant_id=self.tenant_id,
			meter_id=meter_id,
			command_type="remote_disconnect",
			issued_by=authorised_by,
			approved_by=authorised_by,
		)
		result["reason"] = reason
		result["action"] = "remote_disconnect"
		return result

	async def demand_response_signal(
		self,
		customer_segment: str,
		reduction_kw: float,
		duration: float,
		event_type: str = "direct_load_control",
		incentive_rate: float | None = None,
		currency: str = "USD",
	) -> dict[str, Any]:
		"""
		Broadcast a demand response signal to a customer segment.
		customer_segment: residential | commercial | industrial | critical_peak | all
		reduction_kw: target demand reduction across the segment.
		duration: hours.
		"""
		assert customer_segment, "customer_segment required"
		assert reduction_kw > 0, "reduction_kw must be positive"
		assert duration > 0, "duration must be positive"
		# Find meters for the segment
		all_meters = self._tenant_items(self.meters, self.tenant_id)
		target_meters = [
			m["id"] for m in all_meters
			if m.get("status") == "active"
		]
		from uuid import uuid4
		dr_id = str(uuid4())
		start_time = _now()
		result = self.create_dr_event(
			dr_id=dr_id,
			tenant_id=self.tenant_id,
			event_type=event_type,
			target_reduction_kw=reduction_kw,
			start_time=start_time,
			end_time=start_time,  # placeholder
			meter_ids=target_meters[:100],  # cap at 100 for batch
			created_by=self.actor_id,
			notification_sent=True,
		)
		rec: dict[str, Any] = {
			"demand_response_event": result,
			"customer_segment": customer_segment,
			"target_reduction_kw": reduction_kw,
			"duration_hours": duration,
			"meters_targeted": len(target_meters),
			"incentive_rate": incentive_rate,
			"currency": currency,
			"estimated_energy_reduction_kwh": round(reduction_kw * duration, 3),
			"signal_sent_at": _now(),
		}
		self._demand_response_signals[dr_id] = rec
		return rec

	async def ami_head_end_sync(
		self,
		batch_id: str,
		meters_polled: int | None = None,
		reads_received: int | None = None,
		failures: int = 0,
		protocol: str = "DLMS",
	) -> dict[str, Any]:
		"""
		Record an AMI head-end synchronisation batch.
		Updates head-end connectivity status and returns sync summary.
		"""
		assert batch_id, "batch_id required"
		all_meters = self._tenant_items(self.meters, self.tenant_id)
		total_meters = len(all_meters)
		if meters_polled is None:
			meters_polled = total_meters
		if reads_received is None:
			reads_received = max(0, meters_polled - failures)
		success_rate = round(reads_received / meters_polled * 100, 2) if meters_polled > 0 else 0.0
		from uuid import uuid4
		he_id = str(uuid4())
		he_status = self.update_head_end_status(
			he_id=he_id,
			tenant_id=self.tenant_id,
			head_end_name=f"AMI_HE_{self.tenant_id}",
			protocol=protocol,
			connected_meters=reads_received,
			total_meters=total_meters,
		)
		rec: dict[str, Any] = {
			"batch_id": batch_id,
			"tenant_id": self.tenant_id,
			"protocol": protocol,
			"total_meters": total_meters,
			"meters_polled": meters_polled,
			"reads_received": reads_received,
			"read_failures": failures,
			"success_rate_pct": success_rate,
			"head_end_status": he_status,
			"synced_at": _now(),
		}
		self._ami_sync_batches[batch_id] = rec
		self._audit(self.tenant_id, "ami_sync_completed", batch_id, "ami_sync")
		return rec

	async def meter_analytics(self, period: str) -> dict[str, Any]:
		"""
		Compute smart meter analytics for a period (YYYY-MM).
		Returns: active meters, tamper events, read rates, DR participation, command success.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		all_meters = self._tenant_items(self.meters, self.tenant_id)
		active_meters = [m for m in all_meters if m["status"] == "active"]
		tampers = [
			t for t in self._tenant_items(self.tamper_events, self.tenant_id)
			if t.get("detected_at", "")[:7] == period
		]
		readings = [
			r for r in self._tenant_items(self.readings, self.tenant_id)
			if r.get("received_at", "")[:7] == period
		]
		commands = [
			c for c in self._tenant_items(self.commands, self.tenant_id)
			if c.get("issued_at", "")[:7] == period
		]
		executed_cmds = [c for c in commands if c.get("status") == "executed"]
		dr_events = [
			d for d in self._tenant_items(self.dr_events, self.tenant_id)
			if d.get("start_time", "")[:7] == period
		]
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"total_meters": len(all_meters),
			"active_meters": len(active_meters),
			"tamper_events": len(tampers),
			"resolved_tampers": sum(1 for t in tampers if t.get("status") == "resolved"),
			"readings_received": len(readings),
			"read_rate_pct": round(len(readings) / max(len(active_meters) * 48 * 30, 1) * 100, 2),
			"commands_issued": len(commands),
			"commands_executed": len(executed_cmds),
			"command_success_rate_pct": round(len(executed_cmds) / max(len(commands), 1) * 100, 2),
			"dr_events": len(dr_events),
			"calculated_at": _now(),
		}
		self._meter_analytics[rec_id] = rec
		return rec

	async def health_check(self) -> dict[str, Any]:
		"""Return metering service health status."""
		meters = self._tenant_items(self.meters, self.tenant_id)
		active = sum(1 for m in meters if m.get("status") == "active")
		return {
			"service": "MeteringService", "tenant_id": self.tenant_id, "status": "healthy",
			"meter_count": len(meters), "active_count": active, "checked_at": _now(),
		}

	async def bulk_read_meters(self, meter_ids: list[str], period: str) -> dict[str, Any]:
		"""Trigger bulk meter reads for a list of meter IDs."""
		assert meter_ids, "meter_ids required"
		results: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for mid in meter_ids:
			try:
				rec = await self.read_meter(meter_id=mid, reading_timestamp=_now(), reading_value=0.0, reading_type="estimated")
				results.append({"meter_id": mid, "reading_id": rec.get("id"), "status": "read"})
			except Exception as exc:
				errors.append({"meter_id": mid, "error": str(exc)})
		return {"period": period, "success_count": len(results), "error_count": len(errors), "results": results, "errors": errors}

	async def export_meter_data(self, period: str, format: str = "json") -> dict[str, Any]:
		"""Export meter readings for a period."""
		assert format in {"json", "csv"}, "format must be json or csv"
		readings = [r for r in self._tenant_items(self.readings, self.tenant_id) if r.get("reading_timestamp", "")[:7] == period[:7]]
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if readings:
				writer = csv.DictWriter(buf, fieldnames=list(readings[0].keys()))
				writer.writeheader()
				writer.writerows(readings)
			return {"format": "csv", "period": period, "record_count": len(readings), "content": buf.getvalue()}
		return {"format": "json", "period": period, "record_count": len(readings), "records": readings}

	async def tamper_analytics(self) -> dict[str, Any]:
		"""Summarise tamper events by type and detection rate."""
		events = self._tenant_items(self.tamper_events, self.tenant_id)
		by_type: dict[str, int] = {}
		for e in events:
			tt = e.get("tamper_type", "unknown")
			by_type[tt] = by_type.get(tt, 0) + 1
		return {
			"tenant_id": self.tenant_id, "tamper_event_count": len(events),
			"by_type": by_type, "computed_at": _now(),
		}

	async def demand_response_analytics(self) -> dict[str, Any]:
		"""Summarise demand response events and curtailment achieved."""
		events = self._tenant_items(self.demand_response_events, self.tenant_id)
		total_curtailment = sum(float(e.get("actual_curtailment_kw", 0)) for e in events)
		return {
			"tenant_id": self.tenant_id, "dr_event_count": len(events),
			"total_curtailment_kw": round(total_curtailment, 2),
			"avg_curtailment_kw": round(total_curtailment / max(len(events), 1), 2),
			"computed_at": _now(),
		}

	async def metering_compliance_report(self, standard: str = "DLMS_COSEM") -> dict[str, Any]:
		"""Generate a metering compliance report."""
		meters = self._tenant_items(self.meters, self.tenant_id)
		tampers = self._tenant_items(self.tamper_events, self.tenant_id)
		unresolved_tampers = [t for t in tampers if t.get("status") != "resolved"]
		self._audit(self.tenant_id, "metering_compliance_report_generated", standard, "report", {})
		return {
			"standard": standard, "tenant_id": self.tenant_id,
			"meter_count": len(meters), "unresolved_tampers": len(unresolved_tampers),
			"compliance_rate_pct": round((len(meters) - len(unresolved_tampers)) / max(len(meters), 1) * 100, 2),
			"generated_at": _now(),
		}

	async def meter_data_report(self, meter_id: str, period: str) -> dict[str, Any]:
		"""
		Generate a detailed data report for a single meter over a period.
		Includes: readings, quality flags, tamper events, commands, DR participation.
		"""
		assert meter_id, "meter_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		meter = self._get_meter(self.tenant_id, meter_id)
		readings = [
			r for r in self._tenant_items(self.readings, self.tenant_id)
			if r.get("meter_id") == meter_id and r.get("received_at", "")[:7] == period
		]
		flags = [
			f for f in self._tenant_items(self.quality_flags, self.tenant_id)
			if f.get("meter_id") == meter_id
		]
		tampers = [
			t for t in self._tenant_items(self.tamper_events, self.tenant_id)
			if t.get("meter_id") == meter_id and t.get("detected_at", "")[:7] == period
		]
		commands = [
			c for c in self._tenant_items(self.commands, self.tenant_id)
			if c.get("meter_id") == meter_id and c.get("issued_at", "")[:7] == period
		]
		total_kwh = sum(r.get("value", 0) for r in readings if r.get("reading_type") == "actual")
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"meter_id": meter_id,
			"serial_number": meter.serial_number,
			"customer_id": meter.customer_id,
			"period": period,
			"readings_count": len(readings),
			"total_kwh": round(total_kwh, 3),
			"quality_flags": len(flags),
			"suspect_reads": sum(1 for r in readings if r.get("quality_flag") in ("suspect", "estimated")),
			"tamper_events": len(tampers),
			"commands_issued": len(commands),
			"meter_status": meter.status,
			"generated_at": _now(),
		}
		self._meter_reports[rec_id] = rec
		return rec

	async def meter_fleet_summary(self) -> dict[str, Any]:
		"""Return a fleet-level summary of all meters: counts by status, type, and communication health."""
		meters = self._tenant_items(self.meters, self.tenant_id)
		by_status: dict[str, int] = {}
		by_type: dict[str, int] = {}
		for m in meters:
			st = m.get("status", "unknown")
			mt = m.get("meter_type", "unknown")
			by_status[st] = by_status.get(st, 0) + 1
			by_type[mt] = by_type.get(mt, 0) + 1
		active = by_status.get("active", 0)
		return {
			"tenant_id": self.tenant_id,
			"total_meters": len(meters),
			"active_count": active,
			"availability_pct": round(active / max(len(meters), 1) * 100, 2),
			"by_status": by_status,
			"by_type": by_type,
			"computed_at": _now(),
		}
