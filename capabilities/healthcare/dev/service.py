"""Async service layer for APG Medical Device Management."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .capability_contract import (
	SUPPORTED_ADVERSE_EVENT_TYPES, SUPPORTED_CALIBRATION_STATUSES,
	SUPPORTED_DEVICE_CLASSES, SUPPORTED_DEVICE_STATUSES, SUPPORTED_DEVICE_TYPES,
	SUPPORTED_MAINTENANCE_TYPES, SUPPORTED_UDI_FORMATS,
	SUPPORTED_WORK_ORDER_STATUSES, evaluate_capability_rules, get_capability_contract,
)
from .models import (
	AdverseEventCreate, AdverseEventResponse,
	CalibrationRecordCreate, CalibrationRecordResponse,
	DeviceCreate, DeviceResponse,
	MaintenanceScheduleCreate, MaintenanceScheduleResponse,
	uuid7str,
)

logger = logging.getLogger(__name__)


def _log_op(op: str, tid: str, eid: str) -> None:
	logger.info("dev.%s tenant=%s id=%s", op, tid, eid)


def _log_adverse(event_type: str, severity: str, device_id: str) -> None:
	logger.warning("dev.adverse_event type=%s severity=%s device=%s", event_type, severity, device_id)


def _log_recall(recall_id: str, device_count: int, tenant_id: str) -> None:
	logger.critical("dev.recall recall_id=%s affected=%d tenant=%s", recall_id, device_count, tenant_id)


def _log_pretty_device(device_id: str, name: str, location: str) -> str:
	return f"device[{device_id}] name={name!r} loc={location!r}"


class PolicyViolationError(ValueError):
	pass


class MedicalDeviceManagementService:
	"""Tenant-scoped medical device management runtime."""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._devices: dict[tuple[str, str], DeviceResponse] = {}
		self._maintenance: dict[tuple[str, str], MaintenanceScheduleResponse] = {}
		self._calibrations: dict[tuple[str, str], CalibrationRecordResponse] = {}
		self._adverse_events: dict[tuple[str, str], AdverseEventResponse] = {}
		self._recalls: dict[tuple[str, str], dict[str, Any]] = {}
		self._usage_logs: list[dict[str, Any]] = []
		self._audit_events: list[dict[str, Any]] = []

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	# ── devices ───────────────────────────────────────────────────────────────

	async def register_device(self, payload: DeviceCreate) -> DeviceResponse:
		needs_udi = payload.device_class in ("class_ii", "class_iii")
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_device",
			"device_type_supported": payload.device_type in SUPPORTED_DEVICE_TYPES,
			"device_class_supported": payload.device_class in SUPPORTED_DEVICE_CLASSES,
			"device_class_requires_udi": needs_udi,
			"udi_present": bool(payload.udi),
		})
		device = DeviceResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, name=payload.name,
			device_type=payload.device_type, device_class=payload.device_class,
			manufacturer=payload.manufacturer, model_number=payload.model_number,
			serial_number=payload.serial_number, udi=payload.udi,
			udi_format=payload.udi_format, location=payload.location,
			department=payload.department, status="active",
			calibration_status="current",
			purchase_date=payload.purchase_date, warranty_expiry=payload.warranty_expiry,
			created_by=payload.created_by,
		)
		self._devices[(payload.tenant_id, device.id)] = device
		self._audit(payload.tenant_id, "device_registered", device.id)
		_log_op("register_device", payload.tenant_id, device.id)
		return device

	async def update_device_status(self, tenant_id: str, device_id: str, status: str) -> DeviceResponse | None:
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "update_device_status",
			"device_status_supported": status in SUPPORTED_DEVICE_STATUSES,
		})
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			return None
		updated = device.model_copy(update={"status": status, "updated_at": datetime.utcnow()})
		self._devices[(tenant_id, device_id)] = updated
		self._audit(tenant_id, "device_status_changed", device_id)
		return updated

	async def get_device(self, tenant_id: str, device_id: str) -> DeviceResponse | None:
		return self._devices.get((tenant_id, device_id))

	async def list_devices(self, tenant_id: str, device_type: str | None = None, status: str | None = None) -> list[DeviceResponse]:
		results = [d for (tid, _), d in self._devices.items() if tid == tenant_id]
		if device_type:
			results = [d for d in results if d.device_type == device_type]
		if status:
			results = [d for d in results if d.status == status]
		return sorted(results, key=lambda d: d.name)

	async def device_inventory(
		self,
		location: str,
		filters: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Return inventory summary for a location, with optional attribute filters."""
		assert location, "location required"
		filters = filters or {}
		tenant_id = self._tenant_id
		all_devices = [d for (tid, _), d in self._devices.items() if tid == tenant_id]
		by_location = [d for d in all_devices if d.location == location]
		if filters.get("device_type"):
			by_location = [d for d in by_location if d.device_type == filters["device_type"]]
		if filters.get("status"):
			by_location = [d for d in by_location if d.status == filters["status"]]
		if filters.get("department"):
			by_location = [d for d in by_location if d.department == filters["department"]]
		status_breakdown: dict[str, int] = {}
		type_breakdown: dict[str, int] = {}
		for d in by_location:
			status_breakdown[d.status] = status_breakdown.get(d.status, 0) + 1
			type_breakdown[d.device_type] = type_breakdown.get(d.device_type, 0) + 1
		calibration_due = [
			d.id for d in by_location
			if d.calibration_status in ("overdue", "due_soon")
		]
		maintenance_due = [
			m for (tid, _), m in self._maintenance.items()
			if tid == tenant_id and m.status == "open"
			and any(d.id == m.device_id and d.location == location for d in all_devices)
		]
		_log_op("device_inventory", tenant_id, location)
		return {
			"location": location,
			"tenant_id": tenant_id,
			"total": len(by_location),
			"status_breakdown": status_breakdown,
			"type_breakdown": type_breakdown,
			"calibration_due_count": len(calibration_due),
			"calibration_due_ids": calibration_due,
			"open_maintenance_count": len(maintenance_due),
			"devices": [
				{
					"id": d.id, "name": d.name, "device_type": d.device_type,
					"status": d.status, "department": d.department,
					"calibration_status": d.calibration_status,
				}
				for d in sorted(by_location, key=lambda d: d.name)
			],
		}

	async def udi_lookup(self, udi: str) -> DeviceResponse | None:
		"""Find device by Universal Device Identifier across all tenants."""
		assert udi, "udi required"
		for device in self._devices.values():
			if device.udi == udi:
				_log_op("udi_lookup", device.tenant_id, device.id)
				return device
		return None

	# ── maintenance ───────────────────────────────────────────────────────────

	async def schedule_maintenance(self, payload: MaintenanceScheduleCreate) -> MaintenanceScheduleResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "schedule_maintenance",
			"maintenance_type_supported": payload.maintenance_type in SUPPORTED_MAINTENANCE_TYPES,
		})
		sched = MaintenanceScheduleResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, device_id=payload.device_id,
			maintenance_type=payload.maintenance_type, scheduled_date=payload.scheduled_date,
			assigned_to=payload.assigned_to, estimated_hours=payload.estimated_hours,
			instructions=payload.instructions, status="open",
			work_order_id=f"WO-{uuid7str()[:8].upper()}",
			created_by=payload.created_by,
		)
		self._maintenance[(payload.tenant_id, sched.id)] = sched
		self._audit(payload.tenant_id, "maintenance_scheduled", sched.id)
		_log_op("schedule_maintenance", payload.tenant_id, sched.id)
		return sched

	async def maintenance_schedule(
		self,
		device_id: str,
		next_due: datetime,
	) -> dict[str, Any]:
		"""Set or update the next scheduled maintenance date for a device."""
		assert device_id, "device_id required"
		tenant_id = self._tenant_id
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			raise KeyError(f"device {device_id} not found")
		schedule_id = uuid7str()
		record: dict[str, Any] = {
			"id": schedule_id,
			"tenant_id": tenant_id,
			"device_id": device_id,
			"device_name": device.name,
			"next_due": next_due.isoformat(),
			"scheduled_by": self._actor_id,
			"scheduled_at": datetime.utcnow().isoformat(),
			"status": "scheduled",
		}
		self._audit(tenant_id, "maintenance_schedule_set", schedule_id)
		_log_op("maintenance_schedule", tenant_id, device_id)
		return record

	async def log_maintenance(
		self,
		device_id: str,
		maintenance_type: str,
		performed_by: str,
		findings: str,
	) -> dict[str, Any]:
		"""Log completion of a maintenance activity with findings."""
		assert device_id, "device_id required"
		assert maintenance_type in SUPPORTED_MAINTENANCE_TYPES, f"unsupported maintenance type: {maintenance_type}"
		assert performed_by, "performed_by required"
		tenant_id = self._tenant_id
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			raise KeyError(f"device {device_id} not found")
		log_id = uuid7str()
		record: dict[str, Any] = {
			"id": log_id,
			"tenant_id": tenant_id,
			"device_id": device_id,
			"device_name": device.name,
			"maintenance_type": maintenance_type,
			"performed_by": performed_by,
			"findings": findings,
			"performed_at": datetime.utcnow().isoformat(),
			"logged_by": self._actor_id,
			"requires_follow_up": bool(findings and len(findings) > 20),
		}
		self._audit(tenant_id, "maintenance_logged", log_id)
		_log_op("log_maintenance", tenant_id, device_id)
		return record

	async def complete_maintenance(self, tenant_id: str, sched_id: str, notes: str = "") -> MaintenanceScheduleResponse | None:
		sched = self._maintenance.get((tenant_id, sched_id))
		if sched is None:
			return None
		updated = sched.model_copy(update={"status": "completed", "completed_at": datetime.utcnow(), "technician_notes": notes, "updated_at": datetime.utcnow()})
		self._maintenance[(tenant_id, sched_id)] = updated
		self._audit(tenant_id, "work_order_completed", sched_id)
		return updated

	async def list_maintenance(self, tenant_id: str, device_id: str | None = None, status: str | None = None) -> list[MaintenanceScheduleResponse]:
		results = [m for (tid, _), m in self._maintenance.items() if tid == tenant_id]
		if device_id:
			results = [m for m in results if m.device_id == device_id]
		if status:
			results = [m for m in results if m.status == status]
		return sorted(results, key=lambda m: m.scheduled_date)

	# ── calibration ───────────────────────────────────────────────────────────

	async def record_calibration(self, payload: CalibrationRecordCreate) -> CalibrationRecordResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_calibration",
			"certificate_present": bool(payload.certificate_reference),
		})
		cal = CalibrationRecordResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, device_id=payload.device_id,
			calibrated_by=payload.calibrated_by, calibration_date=payload.calibration_date,
			next_due_date=payload.next_due_date, certificate_reference=payload.certificate_reference,
			result=payload.result, notes=payload.notes, created_by=payload.created_by,
		)
		self._calibrations[(payload.tenant_id, cal.id)] = cal
		device = self._devices.get((payload.tenant_id, payload.device_id))
		if device:
			new_cal_status = "current" if payload.result == "pass" else "failed"
			updated_device = device.model_copy(update={
				"calibration_status": new_cal_status,
				"last_calibrated_at": payload.calibration_date,
				"next_calibration_due": payload.next_due_date,
				"updated_at": datetime.utcnow(),
			})
			self._devices[(payload.tenant_id, payload.device_id)] = updated_device
		self._audit(payload.tenant_id, "calibration_recorded", cal.id)
		_log_op("record_calibration", payload.tenant_id, cal.id)
		return cal

	async def calibration_record(
		self,
		device_id: str,
		calibration_date: datetime,
		calibration_result: str,
		next_due: datetime,
	) -> dict[str, Any]:
		"""Record calibration and return structured summary."""
		assert device_id, "device_id required"
		assert calibration_result in ("pass", "fail", "conditional"), f"invalid result: {calibration_result}"
		tenant_id = self._tenant_id
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			raise KeyError(f"device {device_id} not found")
		cal_id = uuid7str()
		cert_ref = f"CERT-{cal_id[:8].upper()}"
		record: dict[str, Any] = {
			"id": cal_id,
			"tenant_id": tenant_id,
			"device_id": device_id,
			"device_name": device.name,
			"calibration_date": calibration_date.isoformat(),
			"calibration_result": calibration_result,
			"next_due": next_due.isoformat(),
			"certificate_reference": cert_ref,
			"calibrated_by": self._actor_id,
			"days_until_next": (next_due - datetime.utcnow()).days,
			"passed": calibration_result == "pass",
		}
		new_status = "current" if calibration_result == "pass" else "failed"
		updated = device.model_copy(update={
			"calibration_status": new_status,
			"next_calibration_due": next_due,
			"updated_at": datetime.utcnow(),
		})
		self._devices[(tenant_id, device_id)] = updated
		self._audit(tenant_id, "calibration_recorded", cal_id)
		_log_op("calibration_record", tenant_id, device_id)
		return record

	async def list_calibrations(self, tenant_id: str, device_id: str | None = None) -> list[CalibrationRecordResponse]:
		results = [c for (tid, _), c in self._calibrations.items() if tid == tenant_id]
		if device_id:
			results = [c for c in results if c.device_id == device_id]
		return sorted(results, key=lambda c: c.calibration_date, reverse=True)

	# ── adverse events ────────────────────────────────────────────────────────

	async def report_adverse_event(self, payload: AdverseEventCreate) -> AdverseEventResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "report_adverse_event",
			"adverse_event_type_supported": payload.event_type in SUPPORTED_ADVERSE_EVENT_TYPES,
			"adverse_event_severity_supported": payload.severity in ["minor", "moderate", "serious", "life_threatening", "death"],
		})
		event = AdverseEventResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, device_id=payload.device_id,
			event_type=payload.event_type, severity=payload.severity,
			description=payload.description, patient_id=payload.patient_id,
			occurred_at=payload.occurred_at, reported_by=payload.reported_by,
			immediate_action_taken=payload.immediate_action_taken, status="open",
			created_by=payload.created_by,
		)
		self._adverse_events[(payload.tenant_id, event.id)] = event
		_log_adverse(payload.event_type, payload.severity, payload.device_id)
		self._audit(payload.tenant_id, "adverse_event_reported", event.id)
		_log_op("report_adverse_event", payload.tenant_id, event.id)
		if payload.severity in ("serious", "life_threatening", "death"):
			device = self._devices.get((payload.tenant_id, payload.device_id))
			if device:
				self._devices[(payload.tenant_id, payload.device_id)] = device.model_copy(update={"status": "in_maintenance", "updated_at": datetime.utcnow()})
		return event

	async def adverse_event_report(
		self,
		device_id: str,
		incident_type: str,
		description: str,
		patient_affected: bool,
	) -> dict[str, Any]:
		"""Create an MDR-compliant adverse event report."""
		assert device_id, "device_id required"
		assert incident_type in SUPPORTED_ADVERSE_EVENT_TYPES, f"unsupported type: {incident_type}"
		assert description, "description required"
		tenant_id = self._tenant_id
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			raise KeyError(f"device {device_id} not found")
		report_id = uuid7str()
		severity = "serious" if patient_affected else "moderate"
		mdr_reference = f"MDR-{datetime.utcnow().strftime('%Y%m%d')}-{report_id[:6].upper()}"
		record: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"device_id": device_id,
			"device_name": device.name,
			"manufacturer": device.manufacturer,
			"udi": device.udi,
			"incident_type": incident_type,
			"description": description,
			"patient_affected": patient_affected,
			"severity": severity,
			"mdr_reference": mdr_reference,
			"reported_by": self._actor_id,
			"reported_at": datetime.utcnow().isoformat(),
			"regulatory_30_day_deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(),
			"status": "filed",
			"requires_recall_assessment": patient_affected and incident_type in ("malfunction", "failure"),
		}
		if patient_affected:
			updated = device.model_copy(update={"status": "in_maintenance", "updated_at": datetime.utcnow()})
			self._devices[(tenant_id, device_id)] = updated
		_log_adverse(incident_type, severity, device_id)
		self._audit(tenant_id, "adverse_event_reported", report_id)
		_log_op("adverse_event_report", tenant_id, device_id)
		return record

	async def close_adverse_event(self, tenant_id: str, event_id: str, root_cause: str, corrective_action: str) -> AdverseEventResponse | None:
		event = self._adverse_events.get((tenant_id, event_id))
		if event is None:
			return None
		updated = event.model_copy(update={"status": "closed", "root_cause": root_cause, "corrective_action": corrective_action, "closed_at": datetime.utcnow(), "updated_at": datetime.utcnow()})
		self._adverse_events[(tenant_id, event_id)] = updated
		return updated

	async def list_adverse_events(self, tenant_id: str, device_id: str | None = None, severity: str | None = None) -> list[AdverseEventResponse]:
		results = [e for (tid, _), e in self._adverse_events.items() if tid == tenant_id]
		if device_id:
			results = [e for e in results if e.device_id == device_id]
		if severity:
			results = [e for e in results if e.severity == severity]
		return sorted(results, key=lambda e: e.occurred_at, reverse=True)

	# ── recall management ─────────────────────────────────────────────────────

	async def recall_management(
		self,
		recall_id: str,
		affected_devices: list[str],
	) -> dict[str, Any]:
		"""Initiate or update a device recall, quarantining affected devices."""
		assert recall_id, "recall_id required"
		assert affected_devices, "affected_devices list required"
		tenant_id = self._tenant_id
		quarantined: list[str] = []
		not_found: list[str] = []
		for device_id in affected_devices:
			device = self._devices.get((tenant_id, device_id))
			if device is None:
				not_found.append(device_id)
				continue
			updated = device.model_copy(update={"status": "recalled", "updated_at": datetime.utcnow()})
			self._devices[(tenant_id, device_id)] = updated
			quarantined.append(device_id)
		record: dict[str, Any] = {
			"recall_id": recall_id,
			"tenant_id": tenant_id,
			"total_affected": len(affected_devices),
			"quarantined": quarantined,
			"not_found": not_found,
			"quarantined_count": len(quarantined),
			"initiated_by": self._actor_id,
			"initiated_at": datetime.utcnow().isoformat(),
			"fda_30_day_notification": (datetime.utcnow() + timedelta(days=30)).isoformat(),
			"status": "active",
		}
		self._recalls[(tenant_id, recall_id)] = record
		_log_recall(recall_id, len(quarantined), tenant_id)
		self._audit(tenant_id, "recall_initiated", recall_id)
		_log_op("recall_management", tenant_id, recall_id)
		return record

	# ── usage logging ─────────────────────────────────────────────────────────

	async def usage_log(
		self,
		device_id: str,
		patient_id: str,
		start_time: datetime,
		end_time: datetime,
		procedure: str,
	) -> dict[str, Any]:
		"""Log device usage tied to a patient encounter."""
		assert device_id, "device_id required"
		assert patient_id, "patient_id required"
		assert end_time > start_time, "end_time must be after start_time"
		assert procedure, "procedure required"
		tenant_id = self._tenant_id
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			raise KeyError(f"device {device_id} not found")
		if device.status == "recalled":
			raise PolicyViolationError(f"device {device_id} is recalled and cannot be used")
		duration_mins = int((end_time - start_time).total_seconds() / 60)
		log_id = uuid7str()
		entry: dict[str, Any] = {
			"id": log_id,
			"tenant_id": tenant_id,
			"device_id": device_id,
			"device_name": device.name,
			"patient_id": patient_id,
			"procedure": procedure,
			"start_time": start_time.isoformat(),
			"end_time": end_time.isoformat(),
			"duration_minutes": duration_mins,
			"logged_by": self._actor_id,
			"logged_at": datetime.utcnow().isoformat(),
		}
		self._usage_logs.append(entry)
		self._audit(tenant_id, "device_usage_logged", log_id)
		_log_op("usage_log", tenant_id, device_id)
		return entry

	# ── analytics ─────────────────────────────────────────────────────────────

	async def device_analytics(self, period: str) -> dict[str, Any]:
		"""Aggregate device utilisation and maintenance analytics for a period."""
		assert period, "period required"
		tenant_id = self._tenant_id
		devices = [d for (tid, _), d in self._devices.items() if tid == tenant_id]
		maintenance = [m for (tid, _), m in self._maintenance.items() if tid == tenant_id]
		events = [e for (tid, _), e in self._adverse_events.items() if tid == tenant_id]
		recalls = [r for (tid, _), r in self._recalls.items() if tid == tenant_id]
		usage = [u for u in self._usage_logs if u["tenant_id"] == tenant_id]
		uptime_pct = (
			sum(1 for d in devices if d.status == "active") / len(devices) * 100
			if devices else 0.0
		)
		mtbf_days: float = 0.0
		if events:
			total_span = 365
			mtbf_days = total_span / len(events) if events else total_span
		overdue_cal = [d for d in devices if d.calibration_status == "overdue"]
		open_wo = [m for m in maintenance if m.status == "open"]
		_log_op("device_analytics", tenant_id, period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_devices": len(devices),
			"active_devices": sum(1 for d in devices if d.status == "active"),
			"recalled_devices": sum(1 for d in devices if d.status == "recalled"),
			"uptime_pct": round(uptime_pct, 2),
			"mean_time_between_failures_days": round(mtbf_days, 1),
			"maintenance": {
				"total_work_orders": len(maintenance),
				"open_work_orders": len(open_wo),
				"completed": sum(1 for m in maintenance if m.status == "completed"),
			},
			"calibration": {
				"overdue_count": len(overdue_cal),
				"overdue_device_ids": [d.id for d in overdue_cal],
				"current_count": sum(1 for d in devices if d.calibration_status == "current"),
			},
			"adverse_events": {
				"total": len(events),
				"open": sum(1 for e in events if e.status == "open"),
				"serious": sum(1 for e in events if e.severity in ("serious", "life_threatening", "death")),
			},
			"recalls": {"active": len(recalls)},
			"usage_sessions": len(usage),
			"total_usage_minutes": sum(u.get("duration_minutes", 0) for u in usage),
		}

	# ── dashboard ─────────────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		devices = [d for (tid, _), d in self._devices.items() if tid == tenant_id]
		maintenance = [m for (tid, _), m in self._maintenance.items() if tid == tenant_id]
		events = [e for (tid, _), e in self._adverse_events.items() if tid == tenant_id]
		cals = [c for (tid, _), c in self._calibrations.items() if tid == tenant_id]
		return {
			"tenant_id": tenant_id,
			"devices": {"total": len(devices), "active": sum(1 for d in devices if d.status == "active"), "recalled": sum(1 for d in devices if d.status == "recalled"), "in_maintenance": sum(1 for d in devices if d.status == "in_maintenance")},
			"maintenance": {"total": len(maintenance), "open": sum(1 for m in maintenance if m.status == "open")},
			"calibrations": {"total": len(cals), "overdue": sum(1 for d in devices if d.calibration_status == "overdue")},
			"adverse_events": {"total": len(events), "open": sum(1 for e in events if e.status == "open"), "serious": sum(1 for e in events if e.severity in ("serious", "life_threatening", "death"))},
		}

	# ── bulk operations ───────────────────────────────────────────────────────

	async def bulk_register_devices(self, payloads: list[DeviceCreate]) -> list[DeviceResponse]:
		"""Bulk register multiple devices; skips policy-invalid entries and continues."""
		results: list[DeviceResponse] = []
		for payload in payloads:
			device = await self.register_device(payload)
			results.append(device)
		return results

	async def bulk_update_device_status(self, tenant_id: str, updates: list[dict[str, Any]]) -> list[DeviceResponse | None]:
		"""Bulk update device statuses from list of {device_id, status} dicts."""
		return [await self.update_device_status(tenant_id, u["device_id"], u["status"]) for u in updates]

	# ── compliance & regulatory ───────────────────────────────────────────────

	async def fda_510k_status(self, tenant_id: str, device_id: str) -> dict[str, Any]:
		"""Return FDA 510(k) clearance status for a device."""
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			raise KeyError(f"device {device_id} not found")
		_log_op("fda_510k_status", tenant_id, device_id)
		return {
			"device_id": device_id,
			"device_class": device.device_class,
			"manufacturer": device.manufacturer,
			"requires_510k": device.device_class in ("class_ii", "class_iii"),
			"udi_present": bool(device.udi),
			"status": "compliant" if device.udi else "missing_udi",
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def ce_marking_check(self, tenant_id: str, device_id: str) -> dict[str, Any]:
		"""Check CE marking status for EU Medical Device Regulation compliance."""
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			raise KeyError(f"device {device_id} not found")
		_log_op("ce_marking_check", tenant_id, device_id)
		annex_required = device.device_class in ("class_ii", "class_iii")
		return {
			"device_id": device_id,
			"device_class": device.device_class,
			"ce_marking_required": True,
			"notified_body_required": annex_required,
			"mdr_compliant": bool(device.udi),
			"status": "compliant" if device.udi else "requires_review",
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def iso_13485_compliance(self, tenant_id: str) -> dict[str, Any]:
		"""Return ISO 13485 QMS compliance summary for the tenant's device inventory."""
		devices = [d for (tid, _), d in self._devices.items() if tid == tenant_id]
		maintenance = [m for (tid, _), m in self._maintenance.items() if tid == tenant_id]
		calibrations = [c for (tid, _), c in self._calibrations.items() if tid == tenant_id]
		overdue_cal = sum(1 for d in devices if d.calibration_status == "overdue")
		open_wo = sum(1 for m in maintenance if m.status == "open")
		score = max(0, 100 - overdue_cal * 5 - open_wo * 3)
		_log_op("iso_13485_compliance", tenant_id, "qms")
		return {
			"tenant_id": tenant_id,
			"standard": "ISO 13485:2016",
			"total_devices": len(devices),
			"calibrations_recorded": len(calibrations),
			"overdue_calibrations": overdue_cal,
			"open_work_orders": open_wo,
			"compliance_score": score,
			"status": "compliant" if score >= 80 else "non_compliant",
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def device_lifecycle_report(self, tenant_id: str, device_id: str) -> dict[str, Any]:
		"""Return full lifecycle history for a device: maintenance, calibrations, adverse events."""
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			raise KeyError(f"device {device_id} not found")
		maintenance = [m for (tid, mid), m in self._maintenance.items() if tid == tenant_id and m.device_id == device_id]
		calibrations = [c for (tid, cid), c in self._calibrations.items() if tid == tenant_id and c.device_id == device_id]
		events = [e for (tid, eid), e in self._adverse_events.items() if tid == tenant_id and e.device_id == device_id]
		usage = [u for u in self._usage_logs if u["tenant_id"] == tenant_id and u["device_id"] == device_id]
		_log_op("device_lifecycle_report", tenant_id, device_id)
		return {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"device": {"name": device.name, "type": device.device_type, "class": device.device_class, "status": device.status, "udi": device.udi},
			"maintenance_records": len(maintenance),
			"completed_maintenance": sum(1 for m in maintenance if m.status == "completed"),
			"calibration_records": len(calibrations),
			"last_calibration": max((c.calibration_date.isoformat() for c in calibrations), default=None),
			"adverse_events": len(events),
			"serious_events": sum(1 for e in events if e.severity in ("serious", "life_threatening", "death")),
			"usage_sessions": len(usage),
			"total_usage_minutes": sum(u.get("duration_minutes", 0) for u in usage),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def preventive_maintenance_schedule(self, tenant_id: str) -> dict[str, Any]:
		"""Return upcoming preventive maintenance requirements for all active devices."""
		devices = [d for (tid, _), d in self._devices.items() if tid == tenant_id and d.status == "active"]
		due_soon: list[dict[str, Any]] = []
		for device in devices:
			if device.next_calibration_due:
				days_until = (device.next_calibration_due - datetime.utcnow()).days
				if days_until <= 30:
					due_soon.append({
						"device_id": device.id,
						"device_name": device.name,
						"type": "calibration",
						"due_in_days": days_until,
						"overdue": days_until < 0,
					})
		_log_op("preventive_maintenance_schedule", tenant_id, "schedule")
		return {
			"tenant_id": tenant_id,
			"active_devices": len(devices),
			"due_within_30_days": len(due_soon),
			"schedule": sorted(due_soon, key=lambda x: x["due_in_days"]),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def export_device_inventory(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export full device inventory metadata."""
		devices = await self.list_devices(tenant_id)
		export_id = f"DEV-EXPORT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
		_log_op("export_device_inventory", tenant_id, export_id)
		return {
			"export_id": export_id,
			"tenant_id": tenant_id,
			"format": format,
			"record_count": len(devices),
			"download_ref": f"/exports/{tenant_id}/{export_id}.{format}",
			"status": "ready",
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health and store sizes."""
		return {
			"service": "MedicalDeviceManagementService",
			"status": "healthy",
			"devices": len(self._devices),
			"maintenance_records": len(self._maintenance),
			"calibrations": len(self._calibrations),
			"adverse_events": len(self._adverse_events),
			"recalls": len(self._recalls),
			"usage_logs": len(self._usage_logs),
			"audit_events": len(self._audit_events),
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def device_risk_assessment(self, tenant_id: str, device_id: str) -> dict[str, Any]:
		"""Score device risk based on class, adverse event history, and calibration status."""
		device = self._devices.get((tenant_id, device_id))
		if device is None:
			raise KeyError(f"device {device_id} not found")
		events = [e for (tid, _), e in self._adverse_events.items() if tid == tenant_id and e.device_id == device_id]
		serious = sum(1 for e in events if e.severity in ("serious", "life_threatening", "death"))
		class_risk = {"class_i": 1, "class_ii": 2, "class_iii": 3}.get(device.device_class, 2)
		cal_risk = 2 if device.calibration_status == "overdue" else 0
		event_risk = min(serious * 3, 9)
		total_risk = class_risk + cal_risk + event_risk
		risk_level = "critical" if total_risk >= 9 else ("high" if total_risk >= 6 else ("medium" if total_risk >= 3 else "low"))
		_log_op("device_risk_assessment", tenant_id, device_id)
		return {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"device_class": device.device_class,
			"class_risk_score": class_risk,
			"calibration_risk_score": cal_risk,
			"adverse_event_risk_score": event_risk,
			"total_risk_score": total_risk,
			"risk_level": risk_level,
			"serious_adverse_events": serious,
			"assessed_at": datetime.utcnow().isoformat(),
		}

	async def maintenance_analytics(self, tenant_id: str, period: str) -> dict[str, Any]:
		"""Aggregate maintenance KPIs: completion rate, avg hours, overdue count."""
		maintenance = [m for (tid, _), m in self._maintenance.items() if tid == tenant_id]
		completed = [m for m in maintenance if m.status == "completed"]
		open_wo = [m for m in maintenance if m.status == "open"]
		completion_rate = round(len(completed) / max(len(maintenance), 1) * 100, 1)
		avg_hours = round(sum(m.estimated_hours for m in completed) / max(len(completed), 1), 2)
		_log_op("maintenance_analytics", tenant_id, period)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_work_orders": len(maintenance),
			"completed": len(completed),
			"open": len(open_wo),
			"completion_rate_pct": completion_rate,
			"avg_estimated_hours": avg_hours,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def calibration_due_alerts(self, tenant_id: str, days_ahead: int = 30) -> list[dict[str, Any]]:
		"""Return devices with calibration due within the specified days."""
		devices = [d for (tid, _), d in self._devices.items() if tid == tenant_id]
		alerts = []
		now = datetime.utcnow()
		for d in devices:
			if d.next_calibration_due:
				days_until = (d.next_calibration_due - now).days
				if days_until <= days_ahead:
					alerts.append({
						"device_id": d.id,
						"device_name": d.name,
						"calibration_status": d.calibration_status,
						"next_calibration_due": d.next_calibration_due.isoformat(),
						"days_until_due": days_until,
						"overdue": days_until < 0,
					})
		_log_op("calibration_due_alerts", tenant_id, f"{days_ahead}d")
		return sorted(alerts, key=lambda x: x["days_until_due"])

	async def device_count_by_status(self, tenant_id: str) -> dict[str, int]:
		"""Return device count grouped by status."""
		devices = [d for (tid, _), d in self._devices.items() if tid == tenant_id]
		counts: dict[str, int] = {}
		for d in devices:
			counts[d.status] = counts.get(d.status, 0) + 1
		return counts

	async def adverse_event_trend(self, tenant_id: str, period: str) -> dict[str, Any]:
		"""Return adverse event counts grouped by type and severity for a period."""
		events = [e for (tid, _), e in self._adverse_events.items() if tid == tenant_id]
		by_type: dict[str, int] = {}
		by_severity: dict[str, int] = {}
		for e in events:
			by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
			by_severity[e.severity] = by_severity.get(e.severity, 0) + 1
		_log_op("adverse_event_trend", tenant_id, period)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total": len(events),
			"by_type": by_type,
			"by_severity": by_severity,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			logger.warning("dev.rule_denied rule=%s", result["rule"])
			raise PolicyViolationError(result["reason"])

	async def device_utilisation_report(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return device utilisation statistics for a period.

		Utilisation = devices with status 'in_use' / total devices.
		"""
		devices = [d for (tid, _), d in self._devices.items() if tid == tenant_id]
		in_use = sum(1 for d in devices if d.status == "in_use")
		available = sum(1 for d in devices if d.status == "available")
		maintenance = sum(1 for d in devices if d.status in {"maintenance", "under_repair"})
		utilisation_rate = round(in_use / max(len(devices), 1) * 100, 1)
		_log_op("device_utilisation_report", tenant_id, period)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_devices": len(devices),
			"in_use": in_use,
			"available": available,
			"maintenance": maintenance,
			"utilisation_rate_pct": utilisation_rate,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def device_downtime_track(
		self,
		tenant_id: str,
		device_id: str,
		downtime_start: str,
		downtime_end: str | None,
		reason: str,
	) -> dict[str, Any]:
		"""Record a device downtime event for later availability reporting."""
		devices = {k: v for k, v in self._devices.items() if k[0] == tenant_id and k[1] == device_id}
		if not devices:
			raise KeyError(f"device not found: {device_id}")
		dt_id = uuid7str()
		record: dict[str, Any] = {
			"downtime_id": dt_id,
			"tenant_id": tenant_id,
			"device_id": device_id,
			"downtime_start": downtime_start,
			"downtime_end": downtime_end,
			"reason": reason,
			"status": "ongoing" if downtime_end is None else "resolved",
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "device_downtime_recorded", dt_id)
		_log_op("device_downtime_track", tenant_id, dt_id)
		return record

	async def device_kpi_summary(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise device KPI card for dashboard consumption."""
		devices = [d for (tid, _), d in self._devices.items() if tid == tenant_id]
		adverse = [e for (tid, _), e in self._adverse_events.items() if tid == tenant_id]
		calibration_overdue = sum(
			1 for d in devices
			if d.calibration_status == "overdue"
		)
		in_use = sum(1 for d in devices if d.status == "in_use")
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_devices": len(devices),
			"in_use": in_use,
			"utilisation_rate_pct": round(in_use / max(len(devices), 1) * 100, 1),
			"calibration_overdue": calibration_overdue,
			"adverse_events": len(adverse),
			"critical_adverse": sum(1 for e in adverse if e.severity == "critical"),
			"audit_events": len(self._audit_events),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def _audit(self, tenant_id: str, event: str, entity_id: str) -> None:
		self._audit_events.append({"tenant_id": tenant_id, "event": event, "entity_id": entity_id, "timestamp": datetime.utcnow().isoformat()})

	async def ml_device_anomaly_detect(self, *args, **kwargs):
		"""AI-powered AI anomaly detection in medical device telemetry. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="medical_device_anomaly")
			return {"device_anomaly": round(result.score,3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

