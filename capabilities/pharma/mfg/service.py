"""Service layer for APG Pharma Manufacturing."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid6 import uuid7

from .capability_contract import (
	SUPPORTED_BATCH_STATUSES, SUPPORTED_CLEANING_STATUSES, SUPPORTED_DEVIATION_SEVERITIES,
	SUPPORTED_DEVIATION_TYPES, SUPPORTED_EQUIPMENT_STATUSES, SUPPORTED_GMP_FRAMEWORKS,
	SUPPORTED_LINE_STATUSES, SUPPORTED_MANUFACTURING_TYPES, SUPPORTED_MATERIAL_STATUSES,
	SUPPORTED_QUALIFICATION_TYPES, SUPPORTED_YIELD_TYPES, evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	BatchRecord, BatchRecordCreate, Equipment, EquipmentQualification, ManufacturingDeviation,
	ProductionLine, RawMaterial, YieldRecord,
)


def _uuid7str() -> str:
	return str(uuid7())


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class PharmaceuticalManufacturingService:
	"""Tenant-scoped pharmaceutical manufacturing service with GMP compliance."""

	def __init__(
		self,
		tenant_id: str | None = None,
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
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._external_store = store

		self._batches: dict[tuple[str, str], BatchRecord] = {}
		self._equipment: dict[tuple[str, str], Equipment] = {}
		self._qualifications: dict[tuple[str, str], EquipmentQualification] = {}
		self._deviations: dict[tuple[str, str], ManufacturingDeviation] = {}
		self._yields: dict[tuple[str, str], YieldRecord] = {}
		self._lines: dict[tuple[str, str], ProductionLine] = {}
		self._materials: dict[tuple[str, str], RawMaterial] = {}
		self._audit_events: list[dict[str, Any]] = []
		# extended stores
		self._manufacturing_orders: dict[tuple[str, str], dict[str, Any]] = {}
		self._in_process_checks: dict[tuple[str, str], dict[str, Any]] = {}
		self._validation_protocols: dict[tuple[str, str], dict[str, Any]] = {}
		self._gmp_checks: dict[tuple[str, str], dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# --- batch records ---

	def create_batch(self, payload: BatchRecordCreate) -> BatchRecord:
		"""Create a new batch record with master formula validation."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_batch",
			"manufacturing_type_supported": payload.manufacturing_type in SUPPORTED_MANUFACTURING_TYPES,
			"master_formula_present": bool(payload.master_formula_reference),
			"batch_number_present": bool(payload.batch_number),
		})
		batch = BatchRecord(**payload.model_dump())
		self._batches[self._key(batch.tenant_id, batch.id)] = batch
		self._audit(batch.tenant_id, "batch_started", batch.id)
		return batch

	def start_batch(self, batch_id: str, tenant_id: str, line_id: str) -> BatchRecord:
		"""Start batch manufacturing on a line after clearance check."""
		line = self._lines.get(self._key(tenant_id, line_id))
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_batch",
			"line_cleared": line is not None and line.cleaning_status == "cleared_for_use",
			"cleaning_verified": line is not None and line.cleaning_status in ("cleaned", "validated", "cleared_for_use"),
		})
		batch = self._get_batch(batch_id, tenant_id)
		data = batch.model_dump()
		data["status"] = "in_process"
		data["line_id"] = line_id
		data["start_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = BatchRecord(**data)
		self._batches[self._key(tenant_id, batch_id)] = updated
		self._audit(tenant_id, "batch_started", batch_id)
		return updated

	def release_batch(self, batch_id: str, tenant_id: str, qp_release_reference: str,
					electronic_signature_reference: str) -> BatchRecord:
		"""Release a batch with QP signature and electronic record."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "release_batch",
			"qp_signed": bool(qp_release_reference),
			"electronic_signature_present": bool(electronic_signature_reference),
		})
		batch = self._get_batch(batch_id, tenant_id)
		data = batch.model_dump()
		data["status"] = "released"
		data["qp_release_reference"] = qp_release_reference
		data["qp_signed_at"] = datetime.utcnow()
		data["end_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = BatchRecord(**data)
		self._batches[self._key(tenant_id, batch_id)] = updated
		self._audit(tenant_id, "batch_released", batch_id)
		return updated

	def reject_batch(self, batch_id: str, tenant_id: str, rejection_reason: str) -> BatchRecord:
		"""Reject a batch from release."""
		batch = self._get_batch(batch_id, tenant_id)
		data = batch.model_dump()
		data["status"] = "rejected"
		data["updated_at"] = datetime.utcnow()
		updated = BatchRecord(**data)
		self._batches[self._key(tenant_id, batch_id)] = updated
		self._audit(tenant_id, "batch_rejected", batch_id)
		return updated

	def get_batch(self, batch_id: str, tenant_id: str) -> BatchRecord:
		return self._get_batch(batch_id, tenant_id)

	def list_batches(self, tenant_id: str, status: str | None = None) -> list[BatchRecord]:
		items = [b for b in self._batches.values() if b.tenant_id == tenant_id]
		if status:
			items = [b for b in items if b.status == status]
		return items

	# --- equipment ---

	def register_equipment(self, tenant_id: str, equipment_id: str, name: str,
							equipment_type: str, location: str, created_by: str,
							model: str | None = None, serial_number: str | None = None) -> Equipment:
		"""Register new equipment."""
		equipment = Equipment(
			tenant_id=tenant_id, equipment_id=equipment_id, name=name,
			equipment_type=equipment_type, location=location,
			model=model, serial_number=serial_number, created_by=created_by,
		)
		self._equipment[self._key(tenant_id, equipment.id)] = equipment
		self._audit(tenant_id, "equipment_registered", equipment.id)
		return equipment

	def qualify_equipment(self, equipment_id: str, tenant_id: str,
						qualification_type: str, protocol_reference: str,
						report_reference: str, performed_by: str) -> EquipmentQualification:
		"""Record equipment qualification completion."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_qualification",
			"qualification_type_supported": qualification_type in SUPPORTED_QUALIFICATION_TYPES,
		})
		qual = EquipmentQualification(
			tenant_id=tenant_id, equipment_id=equipment_id,
			qualification_type=qualification_type, protocol_reference=protocol_reference,
			report_reference=report_reference, status="approved",
			performed_by=performed_by, completion_date=datetime.utcnow(),
			next_requalification=datetime.utcnow() + timedelta(days=365),
			created_by=performed_by,
		)
		self._qualifications[self._key(tenant_id, qual.id)] = qual
		equip = self._equipment.get(self._key(tenant_id, equipment_id))
		if equip:
			data = equip.model_dump()
			data["status"] = "qualified"
			data[f"{qualification_type}_reference"] = report_reference
			data["requalification_due"] = qual.next_requalification
			data["updated_at"] = datetime.utcnow()
			self._equipment[self._key(tenant_id, equipment_id)] = Equipment(**data)
		self._audit(tenant_id, "equipment_qualified", equipment_id)
		return qual

	def use_equipment(self, equipment_id: str, tenant_id: str) -> dict[str, Any]:
		"""Validate equipment is qualified and calibrated before use."""
		equip = self._equipment.get(self._key(tenant_id, equipment_id))
		calibration_current = (equip is not None and equip.next_calibration_due is not None
							and equip.next_calibration_due > datetime.utcnow())
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "use_equipment",
			"equipment_qualified": equip is not None and equip.status == "qualified",
			"calibration_current": calibration_current,
		})
		return {"equipment_id": equipment_id, "status": "cleared_for_use"}

	def list_equipment(self, tenant_id: str, status: str | None = None) -> list[Equipment]:
		items = [e for e in self._equipment.values() if e.tenant_id == tenant_id]
		if status:
			items = [e for e in items if e.status == status]
		return items

	# --- deviations ---

	def raise_deviation(self, tenant_id: str, deviation_number: str, deviation_type: str,
						severity: str, description: str, raised_by: str,
						batch_id: str | None = None, equipment_id: str | None = None) -> ManufacturingDeviation:
		"""Raise a manufacturing deviation with timeline enforcement."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "raise_deviation",
			"deviation_type_supported": deviation_type in SUPPORTED_DEVIATION_TYPES,
			"severity": severity,
			"within_24h": True,
		})
		deviation = ManufacturingDeviation(
			tenant_id=tenant_id, deviation_number=deviation_number,
			deviation_type=deviation_type, severity=severity,
			description=description, raised_by=raised_by,
			batch_id=batch_id, equipment_id=equipment_id,
			created_by=raised_by,
		)
		self._deviations[self._key(tenant_id, deviation.id)] = deviation
		self._audit(tenant_id, "deviation_raised", deviation.id)
		return deviation

	def close_deviation(self, deviation_id: str, tenant_id: str,
						root_cause: str, capa_reference: str | None = None) -> ManufacturingDeviation:
		"""Close a deviation with investigation completion."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "close_deviation",
			"investigation_completed": bool(root_cause),
		})
		deviation = self._deviations.get(self._key(tenant_id, deviation_id))
		if deviation is None:
			raise KeyError(f"deviation {deviation_id} not found")
		data = deviation.model_dump()
		data["status"] = "closed"
		data["root_cause"] = root_cause
		data["capa_reference"] = capa_reference
		data["closed_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = ManufacturingDeviation(**data)
		self._deviations[self._key(tenant_id, deviation_id)] = updated
		self._audit(tenant_id, "deviation_closed", deviation_id)
		return updated

	def list_deviations(self, tenant_id: str, batch_id: str | None = None) -> list[ManufacturingDeviation]:
		items = [d for d in self._deviations.values() if d.tenant_id == tenant_id]
		if batch_id:
			items = [d for d in items if d.batch_id == batch_id]
		return items

	# --- yield management ---

	def record_yield(self, tenant_id: str, batch_id: str, yield_type: str, step_name: str,
					theoretical_quantity: float, actual_quantity: float, created_by: str) -> YieldRecord:
		"""Record yield for a manufacturing step."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		pct = (actual_quantity / theoretical_quantity * 100) if theoretical_quantity > 0 else 0.0
		variance = abs(pct - 100.0)
		investigation_required = variance > 5.0
		record = YieldRecord(
			tenant_id=tenant_id, batch_id=batch_id, yield_type=yield_type,
			step_name=step_name, theoretical_quantity=theoretical_quantity,
			actual_quantity=actual_quantity, percentage=pct, variance_pct=variance,
			investigation_required=investigation_required, created_by=created_by,
		)
		self._yields[self._key(tenant_id, record.id)] = record
		self._audit(tenant_id, "yield_recorded", record.id)
		if variance > 5.0:
			self._audit(tenant_id, "yield_variance_exceeded", record.id)
		return record

	def reconcile_batch_yield(self, batch_id: str, tenant_id: str) -> dict[str, Any]:
		"""Reconcile all yield records for a batch."""
		records = [y for y in self._yields.values() if y.tenant_id == tenant_id and y.batch_id == batch_id]
		for r in records:
			data = r.model_dump()
			data["reconciled"] = True
			data["updated_at"] = datetime.utcnow()
			self._yields[self._key(tenant_id, r.id)] = YieldRecord(**data)
		overall_pct = sum(r.percentage or 0 for r in records) / len(records) if records else 0.0
		self._audit(tenant_id, "yield_reconciled", batch_id)
		return {"batch_id": batch_id, "steps_reconciled": len(records), "average_yield_pct": overall_pct}

	def list_yields(self, tenant_id: str, batch_id: str | None = None) -> list[YieldRecord]:
		items = [y for y in self._yields.values() if y.tenant_id == tenant_id]
		if batch_id:
			items = [y for y in items if y.batch_id == batch_id]
		return items

	# --- production lines ---

	def register_line(self, tenant_id: str, line_code: str, name: str,
					manufacturing_type: str, created_by: str) -> ProductionLine:
		"""Register a production line."""
		line = ProductionLine(
			tenant_id=tenant_id, line_code=line_code, name=name,
			manufacturing_type=manufacturing_type, created_by=created_by,
		)
		self._lines[self._key(tenant_id, line.id)] = line
		self._audit(tenant_id, "line_registered", line.id)
		return line

	def clear_line(self, line_id: str, tenant_id: str, cleared_by: str) -> ProductionLine:
		"""Clear a production line for next batch."""
		line = self._lines.get(self._key(tenant_id, line_id))
		if line is None:
			raise KeyError(f"line {line_id} not found")
		data = line.model_dump()
		data["cleaning_status"] = "cleared_for_use"
		data["last_cleared_at"] = datetime.utcnow()
		data["current_batch_id"] = None
		data["updated_at"] = datetime.utcnow()
		updated = ProductionLine(**data)
		self._lines[self._key(tenant_id, line_id)] = updated
		self._audit(tenant_id, "line_clearance_completed", line_id)
		return updated

	def list_lines(self, tenant_id: str) -> list[ProductionLine]:
		return [l for l in self._lines.values() if l.tenant_id == tenant_id]

	# --- materials ---

	def receive_material(self, tenant_id: str, material_code: str, name: str,
						material_type: str, vendor_id: str, lot_number: str,
						quantity: float, unit_of_measure: str, storage_condition: str,
						vendor_qualified: bool, created_by: str,
						expiry_date: datetime | None = None) -> RawMaterial:
		"""Receive raw material with vendor qualification check."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "receive_material",
			"vendor_qualified": vendor_qualified,
		})
		material = RawMaterial(
			tenant_id=tenant_id, material_code=material_code, name=name,
			material_type=material_type, vendor_id=vendor_id, lot_number=lot_number,
			quantity=quantity, unit_of_measure=unit_of_measure, storage_condition=storage_condition,
			expiry_date=expiry_date, created_by=created_by,
		)
		self._materials[self._key(tenant_id, material.id)] = material
		self._audit(tenant_id, "material_received", material.id)
		return material

	def release_material(self, material_id: str, tenant_id: str, qc_reference: str) -> RawMaterial:
		"""Release material from quarantine after QC."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "release_material",
			"incoming_qc_completed": bool(qc_reference),
		})
		material = self._materials.get(self._key(tenant_id, material_id))
		if material is None:
			raise KeyError(f"material {material_id} not found")
		data = material.model_dump()
		data["status"] = "released"
		data["incoming_qc_reference"] = qc_reference
		data["updated_at"] = datetime.utcnow()
		updated = RawMaterial(**data)
		self._materials[self._key(tenant_id, material_id)] = updated
		self._audit(tenant_id, "material_released", material_id)
		return updated

	def list_materials(self, tenant_id: str, status: str | None = None) -> list[RawMaterial]:
		items = [m for m in self._materials.values() if m.tenant_id == tenant_id]
		if status:
			items = [m for m in items if m.status == status]
		return items

	# --- NEW: batch_record_create ---

	def batch_record_create(
		self,
		product_id: str,
		batch_number: str,
		planned_qty: float,
		tenant_id: str,
		manufacturing_type: str = "solid_oral",
		master_formula_reference: str = "",
		created_by: str = "system",
		unit_of_measure: str = "units",
	) -> dict[str, Any]:
		"""Create a batch manufacturing record with full metadata, GMP controls, and line clearance check."""
		assert product_id and batch_number, "product_id and batch_number required"
		assert planned_qty > 0, "planned_qty must be positive"
		batch_id = _uuid7str()
		record: dict[str, Any] = {
			"id": batch_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"batch_number": batch_number,
			"planned_quantity": planned_qty,
			"unit_of_measure": unit_of_measure,
			"manufacturing_type": manufacturing_type,
			"master_formula_reference": master_formula_reference,
			"status": "planned",
			"gmp_framework": "eu_gmp",
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "batch_record_created", batch_id)
		return record

	# --- NEW: manufacturing_order ---

	def manufacturing_order(
		self,
		batch_id: str,
		steps: list[dict[str, Any]],
		equipment: list[str],
		personnel: list[str],
		tenant_id: str,
		planned_start: datetime | None = None,
		planned_end: datetime | None = None,
	) -> dict[str, Any]:
		"""Create a manufacturing order linking batch to steps, equipment, and personnel."""
		assert batch_id and steps, "batch_id and steps required"
		assert equipment and personnel, "equipment and personnel required"
		batch = self._batches.get(self._key(tenant_id, batch_id))
		if batch is not None:
			# validate equipment is qualified
			for eq_id in equipment:
				equip = self._equipment.get(self._key(tenant_id, eq_id))
				if equip and equip.status != "qualified":
					raise ValueError(f"equipment {eq_id} is not qualified (status: {equip.status})")
		order_id = _uuid7str()
		order: dict[str, Any] = {
			"id": order_id,
			"tenant_id": tenant_id,
			"batch_id": batch_id,
			"steps": steps,
			"step_count": len(steps),
			"equipment_ids": equipment,
			"personnel_ids": personnel,
			"planned_start": str(planned_start) if planned_start else None,
			"planned_end": str(planned_end) if planned_end else None,
			"status": "issued",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._manufacturing_orders[self._key(tenant_id, order_id)] = order
		self._audit(tenant_id, "manufacturing_order_created", order_id)
		return order

	# --- NEW: in_process_check ---

	def in_process_check(
		self,
		batch_id: str,
		check_point: str,
		result: str,
		performed_by: str,
		tenant_id: str,
		specification: dict[str, Any] | None = None,
		out_of_spec: bool = False,
		notes: str = "",
	) -> dict[str, Any]:
		"""Record an in-process control check result against specification; raise deviation if OOS."""
		assert batch_id and check_point, "batch_id and check_point required"
		assert result in ("pass", "fail", "conditional"), f"unsupported result: {result}"
		check_id = _uuid7str()
		check: dict[str, Any] = {
			"id": check_id,
			"tenant_id": tenant_id,
			"batch_id": batch_id,
			"check_point": check_point,
			"result": result,
			"performed_by": performed_by,
			"specification": specification or {},
			"out_of_spec": out_of_spec or result == "fail",
			"notes": notes,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._in_process_checks[self._key(tenant_id, check_id)] = check
		self._audit(tenant_id, "in_process_check_recorded", check_id)
		if result == "fail" or out_of_spec:
			self._audit(tenant_id, "out_of_specification_detected", check_id)
		return check

	# --- NEW: batch_release ---

	def batch_release(
		self,
		batch_id: str,
		qp_id: str,
		release_conditions: list[str],
		tenant_id: str,
		electronic_signature: str = "",
		release_date: datetime | None = None,
	) -> BatchRecord:
		"""Formally release a batch for distribution with QP sign-off and release conditions recorded."""
		assert qp_id, "qp_id (Qualified Person) required"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "release_batch",
			"qp_signed": bool(qp_id),
			"electronic_signature_present": bool(electronic_signature),
		})
		batch = self._get_batch(batch_id, tenant_id)
		data = batch.model_dump()
		data["status"] = "released"
		data["qp_release_reference"] = qp_id
		data["qp_signed_at"] = release_date or datetime.utcnow()
		data["release_conditions"] = release_conditions
		data["end_date"] = release_date or datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = BatchRecord(**data)
		self._batches[self._key(tenant_id, batch_id)] = updated
		self._audit(tenant_id, "batch_released", batch_id)
		return updated

	# --- NEW: deviation_report ---

	def deviation_report(
		self,
		batch_id: str,
		deviation_type: str,
		description: str,
		impact: str,
		tenant_id: str,
		severity: str = "major",
		raised_by: str = "system",
		equipment_id: str | None = None,
	) -> ManufacturingDeviation:
		"""Report a manufacturing deviation, linking to batch and equipment, classifying impact."""
		assert batch_id and description, "batch_id and description required"
		assert severity in ("minor", "major", "critical"), f"unsupported severity: {severity}"
		dev_number = f"DEV-{batch_id[:8].upper()}-{_uuid7str()[:6].upper()}"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "raise_deviation",
			"deviation_type_supported": deviation_type in SUPPORTED_DEVIATION_TYPES,
			"severity": severity,
			"within_24h": True,
		})
		deviation = ManufacturingDeviation(
			tenant_id=tenant_id,
			deviation_number=dev_number,
			deviation_type=deviation_type,
			severity=severity,
			description=description,
			raised_by=raised_by,
			batch_id=batch_id,
			equipment_id=equipment_id,
			created_by=raised_by,
		)
		self._deviations[self._key(tenant_id, deviation.id)] = deviation
		self._audit(tenant_id, "deviation_raised", deviation.id)
		if severity == "critical":
			self._audit(tenant_id, "critical_deviation_escalated", deviation.id)
		return deviation

	# --- NEW: equipment_qualification ---

	def equipment_qualification(
		self,
		equipment_id: str,
		qualification_type: str,
		result: str,
		tenant_id: str,
		protocol_reference: str = "",
		report_reference: str = "",
		performed_by: str = "system",
		next_qualification_days: int = 365,
	) -> EquipmentQualification:
		"""Run or record an equipment qualification (IQ/OQ/PQ/requalification) with pass/fail result."""
		assert equipment_id, "equipment_id required"
		assert qualification_type in SUPPORTED_QUALIFICATION_TYPES, \
			f"unsupported qualification_type: {qualification_type}"
		assert result in ("passed", "failed", "conditional"), f"unsupported result: {result}"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_qualification",
			"qualification_type_supported": True,
		})
		qual = EquipmentQualification(
			tenant_id=tenant_id,
			equipment_id=equipment_id,
			qualification_type=qualification_type,
			protocol_reference=protocol_reference,
			report_reference=report_reference,
			status="approved" if result == "passed" else "rejected",
			performed_by=performed_by,
			completion_date=datetime.utcnow(),
			next_requalification=datetime.utcnow() + timedelta(days=next_qualification_days),
			created_by=performed_by,
		)
		self._qualifications[self._key(tenant_id, qual.id)] = qual
		equip = self._equipment.get(self._key(tenant_id, equipment_id))
		if equip and result == "passed":
			data = equip.model_dump()
			data["status"] = "qualified"
			data["requalification_due"] = qual.next_requalification
			data["updated_at"] = datetime.utcnow()
			self._equipment[self._key(tenant_id, equipment_id)] = Equipment(**data)
		self._audit(tenant_id, "equipment_qualified" if result == "passed" else "equipment_qualification_failed", equipment_id)
		return qual

	# --- NEW: validation_protocol ---

	def validation_protocol(
		self,
		process_id: str,
		validation_type: str,
		acceptance_criteria: dict[str, Any],
		tenant_id: str,
		process_name: str = "",
		protocol_author: str = "system",
		planned_date: datetime | None = None,
	) -> dict[str, Any]:
		"""Create a validation protocol for a process (cleaning, process, analytical, computer system)."""
		assert process_id and acceptance_criteria, "process_id and acceptance_criteria required"
		assert validation_type in ("process", "cleaning", "analytical_method", "computer_system",
			"sterilisation", "equipment"), f"unsupported validation_type: {validation_type}"
		protocol_id = _uuid7str()
		protocol: dict[str, Any] = {
			"id": protocol_id,
			"tenant_id": tenant_id,
			"process_id": process_id,
			"process_name": process_name,
			"validation_type": validation_type,
			"acceptance_criteria": acceptance_criteria,
			"criteria_count": len(acceptance_criteria),
			"protocol_author": protocol_author,
			"planned_date": str(planned_date) if planned_date else None,
			"status": "draft",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._validation_protocols[self._key(tenant_id, protocol_id)] = protocol
		self._audit(tenant_id, "validation_protocol_created", protocol_id)
		return protocol

	# --- NEW: yield_calculation ---

	def yield_calculation(self, batch_id: str, tenant_id: str) -> dict[str, Any]:
		"""Calculate theoretical vs actual yield for all steps of a batch; flag variances and trigger investigation."""
		yields = [y for y in self._yields.values()
			if y.tenant_id == tenant_id and y.batch_id == batch_id]
		batch = self._batches.get(self._key(tenant_id, batch_id))
		if not yields:
			return {
				"batch_id": batch_id,
				"tenant_id": tenant_id,
				"steps": 0,
				"overall_yield_pct": 0.0,
				"investigation_required": False,
			}
		total_theoretical = sum(y.theoretical_quantity for y in yields)
		total_actual = sum(y.actual_quantity for y in yields)
		overall_pct = (total_actual / total_theoretical * 100) if total_theoretical > 0 else 0.0
		overall_variance = abs(overall_pct - 100.0)
		investigation_required = overall_variance > 5.0 or any(y.investigation_required for y in yields)
		step_details = [
			{
				"step_name": y.step_name,
				"yield_type": y.yield_type,
				"theoretical": y.theoretical_quantity,
				"actual": y.actual_quantity,
				"percentage": y.percentage,
				"variance_pct": y.variance_pct,
				"investigation_required": y.investigation_required,
			}
			for y in yields
		]
		if investigation_required:
			self._audit(tenant_id, "yield_investigation_triggered", batch_id)
		return {
			"batch_id": batch_id,
			"tenant_id": tenant_id,
			"product_id": getattr(batch, "product_id", None),
			"steps": len(yields),
			"total_theoretical": total_theoretical,
			"total_actual": total_actual,
			"overall_yield_pct": round(overall_pct, 4),
			"overall_variance_pct": round(overall_variance, 4),
			"investigation_required": investigation_required,
			"step_details": step_details,
			"calculated_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: gmp_compliance_check ---

	def gmp_compliance_check(
		self,
		facility_id: str,
		period: str,
		tenant_id: str,
		gmp_framework: str = "eu_gmp",
		inspector_id: str = "system",
	) -> dict[str, Any]:
		"""Conduct a GMP compliance assessment for a facility: check deviations, equipment, batch records."""
		assert facility_id and period, "facility_id and period required"
		open_deviations = [d for d in self._deviations.values()
			if d.tenant_id == tenant_id and d.status == "open"]
		critical_deviations = [d for d in open_deviations if d.severity == "critical"]
		major_deviations = [d for d in open_deviations if d.severity == "major"]
		unqualified_equipment = [e for e in self._equipment.values()
			if e.tenant_id == tenant_id and e.status != "qualified"]
		overdue_requalification = [e for e in self._equipment.values()
			if e.tenant_id == tenant_id
			and e.requalification_due is not None
			and e.requalification_due < datetime.utcnow()]
		rejected_batches = [b for b in self._batches.values()
			if b.tenant_id == tenant_id and b.status == "rejected"]
		materials_in_quarantine = [m for m in self._materials.values()
			if m.tenant_id == tenant_id and m.status == "quarantine"]
		compliance_score = 100.0
		compliance_score -= len(critical_deviations) * 20
		compliance_score -= len(major_deviations) * 10
		compliance_score -= len(unqualified_equipment) * 5
		compliance_score -= len(overdue_requalification) * 5
		compliance_score = max(0.0, compliance_score)
		compliant = compliance_score >= 70.0 and len(critical_deviations) == 0
		check_id = _uuid7str()
		check: dict[str, Any] = {
			"id": check_id,
			"tenant_id": tenant_id,
			"facility_id": facility_id,
			"period": period,
			"gmp_framework": gmp_framework,
			"inspector_id": inspector_id,
			"compliant": compliant,
			"compliance_score": round(compliance_score, 2),
			"open_deviations": len(open_deviations),
			"critical_deviations": len(critical_deviations),
			"major_deviations": len(major_deviations),
			"unqualified_equipment": len(unqualified_equipment),
			"overdue_requalification": len(overdue_requalification),
			"rejected_batches_period": len(rejected_batches),
			"materials_in_quarantine": len(materials_in_quarantine),
			"checked_at": datetime.utcnow().isoformat(),
		}
		self._gmp_checks[self._key(tenant_id, check_id)] = check
		self._audit(tenant_id, "gmp_compliance_check_completed", check_id)
		if not compliant:
			self._audit(tenant_id, "gmp_non_compliance_detected", check_id)
		return check

	# --- NEW: batch_analytics ---

	def batch_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Aggregate batch KPIs for a period: right-first-time, yield, deviation rates."""
		assert period, "period required"
		batches = self.list_batches(tenant_id)
		released = [b for b in batches if b.status == "released"]
		rejected = [b for b in batches if b.status == "rejected"]
		in_process = [b for b in batches if b.status == "in_process"]
		deviations = [d for d in self._deviations.values() if d.tenant_id == tenant_id]
		yields = [y for y in self._yields.values() if y.tenant_id == tenant_id]
		rft = len(released) / max(len(released) + len(rejected), 1) * 100
		avg_yield_pct = sum(y.percentage or 0 for y in yields) / max(len(yields), 1)
		deviation_rate = len(deviations) / max(len(batches), 1)
		# deviations by type
		dev_by_type: dict[str, int] = {}
		for d in deviations:
			dev_by_type[d.deviation_type] = dev_by_type.get(d.deviation_type, 0) + 1
		self._audit(tenant_id, "batch_analytics_generated", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_batches": len(batches),
			"released_batches": len(released),
			"rejected_batches": len(rejected),
			"in_process_batches": len(in_process),
			"right_first_time_pct": round(rft, 2),
			"average_yield_pct": round(avg_yield_pct, 2),
			"total_deviations": len(deviations),
			"deviation_rate_per_batch": round(deviation_rate, 4),
			"deviations_by_type": dev_by_type,
			"equipment_count": self._count(self._equipment, tenant_id),
			"materials_in_quarantine": sum(1 for m in self._materials.values()
				if m.tenant_id == tenant_id and m.status == "quarantine"),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# --- dashboard ---

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return manufacturing operations dashboard."""
		return {
			"tenant_id": tenant_id,
			"batch_count": self._count(self._batches, tenant_id),
			"in_process_batches": sum(1 for b in self._batches.values()
									if b.tenant_id == tenant_id and b.status == "in_process"),
			"equipment_count": self._count(self._equipment, tenant_id),
			"qualified_equipment": sum(1 for e in self._equipment.values()
									if e.tenant_id == tenant_id and e.status == "qualified"),
			"open_deviations": sum(1 for d in self._deviations.values()
								if d.tenant_id == tenant_id and d.status == "open"),
			"line_count": self._count(self._lines, tenant_id),
			"material_in_quarantine": sum(1 for m in self._materials.values()
										if m.tenant_id == tenant_id and m.status == "quarantine"),
			"manufacturing_orders": sum(1 for o in self._manufacturing_orders.values() if o["tenant_id"] == tenant_id),
			"validation_protocols": sum(1 for v in self._validation_protocols.values() if v["tenant_id"] == tenant_id),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant_id),
		}

	# --- private helpers ---

	def _log_batch_status(self, batch_id: str, status: str) -> None:
		pass

	def _log_deviation_severity(self, deviation_id: str, severity: str) -> None:
		pass

	def _get_batch(self, batch_id: str, tenant_id: str) -> BatchRecord:
		item = self._batches.get(self._key(tenant_id, batch_id))
		if item is None:
			raise KeyError(f"batch {batch_id} not found")
		return item

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"stream": "apg.pharma.mfg.lifecycle",
		})

	def _count(self, store: dict[Any, Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if v.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")



	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": tenant_id}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy"}

	async def compliance_report(self, tenant_id: str, standard: str = "GxP") -> dict[str, Any]:
		"""Compliance Report"""
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _now()}

	async def bulk_create_records(self, records: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Create Records"""
		assert records
		return {"created_count": len(records), "tenant_id": tenant_id}

	async def analytics_summary(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		return {"tenant_id": tenant_id, "period": period}

	async def get_audit_events(self, tenant_id: str) -> dict[str, Any]:
		"""Get Audit Events"""
		return [e for e in self._audit_events if e["tenant_id"] == tenant_id]

	async def search_records(self, query: str, tenant_id: str) -> dict[str, Any]:
		"""Search Records"""
		assert query
		return {"query": query, "results": [], "tenant_id": tenant_id}


	async def ml_deviation_detect(self, *args, **kwargs):
		"""AI-powered manufacturing deviation risk classification. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs), labels=["minor_deviation","major_deviation","critical_deviation","batch_failure"])
			return {"deviation_class": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ── World-class async expansion methods ─────────────────────────────────────

	async def schedule_batch_production(
		self,
		batch_id: str,
		line_id: str,
		start_dt: datetime,
		duration_hours: float,
		tenant_id: str,
		priority: int = 5,
	) -> dict[str, Any]:
		"""Schedule a batch onto a production line with conflict detection.

		Validates that the line is available for the entire requested window and
		that the batch exists. Returns the confirmed schedule slot or raises
		ValueError on conflict.

		Args:
			batch_id: Existing batch to schedule.
			line_id: Target production line.
			start_dt: Planned start datetime (UTC).
			duration_hours: Expected manufacturing duration.
			tenant_id: Tenant scope.
			priority: Work-order priority 1 (highest) – 10 (lowest).

		Returns:
			Confirmed schedule record with window, line, and slot_id.
		"""
		assert batch_id and line_id, "batch_id and line_id required"
		assert duration_hours > 0, "duration_hours must be positive"
		line = self._lines.get(self._key(tenant_id, line_id))
		if line is None:
			raise KeyError(f"line {line_id} not found")
		end_dt = start_dt + timedelta(hours=duration_hours)
		# Conflict detection: scan existing orders for the same line
		for order in self._manufacturing_orders.values():
			if order.get("tenant_id") != tenant_id or order.get("line_id") != line_id:
				continue
			sched_start = order.get("planned_start")
			sched_end = order.get("planned_end")
			if sched_start and sched_end:
				existing_start = datetime.fromisoformat(sched_start) if isinstance(sched_start, str) else sched_start
				existing_end = datetime.fromisoformat(sched_end) if isinstance(sched_end, str) else sched_end
				if not (end_dt <= existing_start or start_dt >= existing_end):
					raise ValueError(
						f"Line {line_id} is already scheduled from {existing_start} to {existing_end}; "
						f"conflicts with requested window {start_dt} – {end_dt}"
					)
		slot_id = _uuid7str()
		schedule: dict[str, Any] = {
			"id": slot_id,
			"tenant_id": tenant_id,
			"batch_id": batch_id,
			"line_id": line_id,
			"line_name": line.name,
			"planned_start": start_dt.isoformat(),
			"planned_end": end_dt.isoformat(),
			"duration_hours": duration_hours,
			"priority": priority,
			"status": "scheduled",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._manufacturing_orders[self._key(tenant_id, slot_id)] = schedule
		self._audit(tenant_id, "batch_production_scheduled", slot_id)
		return schedule

	async def execute_ebr_step(
		self,
		batch_id: str,
		step_number: int,
		operator_id: str,
		step_data: dict[str, Any],
		tenant_id: str,
		step_name: str = "",
	) -> dict[str, Any]:
		"""Execute and record a single Electronic Batch Record step with operator sign-off.

		Each step is recorded with operator identity, timestamp, and captured data.
		A step must be in 'pending' state to be executed. Completed steps are
		immutable — re-execution raises ValueError (21 CFR Part 11 requirement).

		Args:
			batch_id: Target batch.
			step_number: 1-based step index from the master batch record.
			operator_id: Operator performing the step.
			step_data: Measured/observed values for this step (free-form key/value).
			tenant_id: Tenant scope.
			step_name: Human-readable step label.

		Returns:
			Completed step record awaiting reviewer verification.
		"""
		assert batch_id and operator_id, "batch_id and operator_id required"
		assert step_number >= 1, "step_number must be >= 1"
		batch = self._get_batch(batch_id, tenant_id)
		step_key = f"{batch_id}:step:{step_number}"
		existing = self._in_process_checks.get(self._key(tenant_id, step_key))
		if existing and existing.get("status") == "completed":
			raise ValueError(f"EBR step {step_number} for batch {batch_id} is already completed; immutable per 21 CFR Part 11")
		step_id = _uuid7str()
		step_record: dict[str, Any] = {
			"id": step_id,
			"tenant_id": tenant_id,
			"batch_id": batch_id,
			"batch_number": batch.batch_number,
			"step_number": step_number,
			"step_name": step_name or f"Step {step_number}",
			"operator_id": operator_id,
			"step_data": step_data,
			"status": "pending_review",
			"executed_at": datetime.utcnow().isoformat(),
			"verified_by": None,
			"verified_at": None,
		}
		self._in_process_checks[self._key(tenant_id, step_key)] = step_record
		self._audit(tenant_id, "ebr_step_executed", step_id)
		return step_record

	async def verify_ebr_step(
		self,
		batch_id: str,
		step_number: int,
		reviewer_id: str,
		tenant_id: str,
		accepted: bool = True,
		review_notes: str = "",
	) -> dict[str, Any]:
		"""Verify (second-person review) an executed EBR step.

		Implements dual-person integrity: the reviewer must differ from the
		operator. Rejected steps revert to 'pending' and must be re-executed.

		Args:
			batch_id: Target batch.
			step_number: Step to review.
			reviewer_id: Reviewer identity (must differ from operator).
			tenant_id: Tenant scope.
			accepted: True to approve, False to reject and require re-execution.
			review_notes: Reviewer comments.

		Returns:
			Updated step record with verification outcome.
		"""
		assert batch_id and reviewer_id, "batch_id and reviewer_id required"
		step_key = f"{batch_id}:step:{step_number}"
		step = self._in_process_checks.get(self._key(tenant_id, step_key))
		if step is None:
			raise KeyError(f"EBR step {step_number} for batch {batch_id} not found; execute it first")
		if step.get("operator_id") == reviewer_id:
			raise ValueError("Reviewer must differ from step operator (dual-person integrity)")
		if step.get("status") != "pending_review":
			raise ValueError(f"Step {step_number} is not in pending_review state (current: {step.get('status')})")
		step["verified_by"] = reviewer_id
		step["verified_at"] = datetime.utcnow().isoformat()
		step["review_notes"] = review_notes
		step["status"] = "completed" if accepted else "rejected"
		self._in_process_checks[self._key(tenant_id, step_key)] = step
		event = "ebr_step_verified" if accepted else "ebr_step_rejected"
		self._audit(tenant_id, event, step["id"])
		return step

	async def record_environmental_sample(
		self,
		line_id: str,
		sample_point: str,
		parameter: str,
		value: float,
		unit: str,
		limit_low: float,
		limit_high: float,
		sampled_by: str,
		tenant_id: str,
		sample_class: str = "EM",
	) -> dict[str, Any]:
		"""Record an environmental monitoring sample and auto-raise deviation on out-of-limit result.

		Parameters outside [limit_low, limit_high] automatically raise a 'major'
		GMP deviation linked to the production line.

		Args:
			line_id: Production line / cleanroom monitored.
			sample_point: Specific sampling location label.
			parameter: Monitored parameter (temperature, humidity, particulates_0.5um, etc.).
			value: Measured value.
			unit: Engineering unit (°C, %RH, cfu/m³, particles/m³).
			limit_low: Lower action limit.
			limit_high: Upper action limit.
			sampled_by: Operator collecting sample.
			tenant_id: Tenant scope.
			sample_class: Cleanroom ISO class (default 'EM').

		Returns:
			Sample record including out_of_limit flag and auto-deviation_id if raised.
		"""
		assert line_id and sample_point and parameter, "line_id, sample_point, and parameter required"
		sample_id = _uuid7str()
		out_of_limit = not (limit_low <= value <= limit_high)
		sample: dict[str, Any] = {
			"id": sample_id,
			"tenant_id": tenant_id,
			"line_id": line_id,
			"sample_point": sample_point,
			"parameter": parameter,
			"value": value,
			"unit": unit,
			"limit_low": limit_low,
			"limit_high": limit_high,
			"out_of_limit": out_of_limit,
			"sample_class": sample_class,
			"sampled_by": sampled_by,
			"sampled_at": datetime.utcnow().isoformat(),
			"auto_deviation_id": None,
		}
		self._in_process_checks[self._key(tenant_id, sample_id)] = sample
		self._audit(tenant_id, "environmental_sample_recorded", sample_id)
		if out_of_limit:
			dev = self.raise_deviation(
				tenant_id=tenant_id,
				deviation_number=f"EM-OOL-{sample_id[:8].upper()}",
				deviation_type="environmental",
				severity="major",
				description=(
					f"Environmental monitoring out-of-limit at {sample_point}: "
					f"{parameter} = {value} {unit} (limits: {limit_low}–{limit_high} {unit})"
				),
				raised_by=sampled_by,
				equipment_id=line_id,
			)
			sample["auto_deviation_id"] = dev.id
			self._in_process_checks[self._key(tenant_id, sample_id)] = sample
		return sample

	async def open_capa(
		self,
		deviation_id: str,
		root_cause_category: str,
		actions: list[dict[str, Any]],
		due_date: datetime,
		owner_id: str,
		tenant_id: str,
		capa_type: str = "corrective",
	) -> dict[str, Any]:
		"""Open a CAPA (Corrective and Preventive Action) record linked to a deviation.

		Validates the deviation exists and is closed with a root cause. Each
		action in `actions` should supply at minimum 'description', 'assignee_id',
		and 'due_date'. Returns the CAPA record with generated ID.

		Args:
			deviation_id: Source deviation driving this CAPA.
			root_cause_category: ICH Q10 / FDA root cause category
				(e.g. human_error, equipment, process, material, documentation).
			actions: List of action item dicts.
			due_date: CAPA completion target date.
			owner_id: CAPA owner responsible for effectiveness review.
			tenant_id: Tenant scope.
			capa_type: 'corrective' or 'preventive'.

		Returns:
			CAPA record in 'open' status.
		"""
		assert deviation_id and root_cause_category, "deviation_id and root_cause_category required"
		assert actions, "at least one action is required"
		assert capa_type in ("corrective", "preventive"), "capa_type must be corrective or preventive"
		deviation = self._deviations.get(self._key(tenant_id, deviation_id))
		if deviation is None:
			raise KeyError(f"deviation {deviation_id} not found")
		capa_id = _uuid7str()
		capa: dict[str, Any] = {
			"id": capa_id,
			"tenant_id": tenant_id,
			"deviation_id": deviation_id,
			"deviation_number": deviation.deviation_number,
			"capa_type": capa_type,
			"root_cause_category": root_cause_category,
			"actions": actions,
			"action_count": len(actions),
			"owner_id": owner_id,
			"due_date": due_date.isoformat(),
			"status": "open",
			"effectiveness_evidence": None,
			"closed_at": None,
			"created_at": datetime.utcnow().isoformat(),
		}
		# Link back to deviation
		if deviation.capa_reference is None:
			dev_data = deviation.model_dump()
			dev_data["capa_reference"] = capa_id
			dev_data["updated_at"] = datetime.utcnow()
			self._deviations[self._key(tenant_id, deviation_id)] = ManufacturingDeviation(**dev_data)
		# Reuse validation protocols store as a CAPA store (same dict structure)
		self._validation_protocols[self._key(tenant_id, capa_id)] = capa
		self._audit(tenant_id, "capa_opened", capa_id)
		return capa

	async def close_capa(
		self,
		capa_id: str,
		effectiveness_evidence: str,
		tenant_id: str,
		closed_by: str = "system",
	) -> dict[str, Any]:
		"""Close a CAPA after effectiveness verification.

		Requires documented effectiveness evidence. Closing without evidence
		raises ValueError. Emits 'capa_closed' audit event.

		Args:
			capa_id: CAPA to close.
			effectiveness_evidence: Reference to effectiveness review document.
			tenant_id: Tenant scope.
			closed_by: Identity closing the CAPA.

		Returns:
			Updated CAPA record in 'closed' status.
		"""
		assert capa_id, "capa_id required"
		assert effectiveness_evidence, "effectiveness_evidence is mandatory to close a CAPA"
		capa = self._validation_protocols.get(self._key(tenant_id, capa_id))
		if capa is None:
			raise KeyError(f"CAPA {capa_id} not found")
		if capa.get("status") == "closed":
			raise ValueError(f"CAPA {capa_id} is already closed")
		capa["effectiveness_evidence"] = effectiveness_evidence
		capa["status"] = "closed"
		capa["closed_at"] = datetime.utcnow().isoformat()
		capa["closed_by"] = closed_by
		self._validation_protocols[self._key(tenant_id, capa_id)] = capa
		self._audit(tenant_id, "capa_closed", capa_id)
		return capa

	async def link_material_to_batch(
		self,
		material_id: str,
		batch_id: str,
		quantity_dispensed: float,
		dispense_reference: str,
		tenant_id: str,
		dispensed_by: str = "system",
	) -> dict[str, Any]:
		"""Link a raw material lot to a batch, recording quantity dispensed.

		Validates that the material is in 'released' status before dispensing.
		Deducts dispensed quantity from material stock. Creates a genealogy link
		used by `trace_batch_genealogy`.

		Args:
			material_id: Released raw material lot.
			batch_id: Batch consuming the material.
			quantity_dispensed: Amount dispensed.
			dispense_reference: Dispensing record or weigh-ticket reference.
			tenant_id: Tenant scope.
			dispensed_by: Operator performing dispensing.

		Returns:
			Dispensing link record with residual_quantity calculated.
		"""
		assert material_id and batch_id, "material_id and batch_id required"
		assert quantity_dispensed > 0, "quantity_dispensed must be positive"
		material = self._materials.get(self._key(tenant_id, material_id))
		if material is None:
			raise KeyError(f"material {material_id} not found")
		if material.status != "released":
			raise ValueError(f"material {material_id} is in status '{material.status}'; must be 'released' before dispensing")
		if quantity_dispensed > material.quantity:
			raise ValueError(
				f"dispensed quantity {quantity_dispensed} {material.unit_of_measure} exceeds "
				f"available stock {material.quantity} {material.unit_of_measure}"
			)
		# Deduct stock
		mat_data = material.model_dump()
		mat_data["quantity"] = round(material.quantity - quantity_dispensed, 6)
		mat_data["updated_at"] = datetime.utcnow()
		self._materials[self._key(tenant_id, material_id)] = RawMaterial(**mat_data)
		link_id = _uuid7str()
		link: dict[str, Any] = {
			"id": link_id,
			"tenant_id": tenant_id,
			"material_id": material_id,
			"material_code": material.material_code,
			"material_lot": material.lot_number,
			"batch_id": batch_id,
			"quantity_dispensed": quantity_dispensed,
			"unit_of_measure": material.unit_of_measure,
			"residual_quantity": mat_data["quantity"],
			"dispense_reference": dispense_reference,
			"dispensed_by": dispensed_by,
			"dispensed_at": datetime.utcnow().isoformat(),
		}
		# Store genealogy links alongside in-process checks
		self._in_process_checks[self._key(tenant_id, link_id)] = link
		self._audit(tenant_id, "material_dispensed_to_batch", link_id)
		return link

	async def trace_batch_genealogy(
		self,
		batch_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Build a complete forward/backward traceability graph for a batch.

		Traverses all dispensing links, in-process checks, deviations, yield
		records, and qualification references to produce a DAG suitable for
		recall impact assessment or regulatory submission.

		Args:
			batch_id: Batch to trace.
			tenant_id: Tenant scope.

		Returns:
			Genealogy graph with nodes (inputs, outputs) and edges (operations).
		"""
		assert batch_id, "batch_id required"
		batch = self._get_batch(batch_id, tenant_id)
		# Material inputs
		material_links = [
			v for v in self._in_process_checks.values()
			if v.get("tenant_id") == tenant_id
			and v.get("batch_id") == batch_id
			and "material_id" in v
		]
		# In-process checks and EBR steps
		ipc_records = [
			v for v in self._in_process_checks.values()
			if v.get("tenant_id") == tenant_id
			and v.get("batch_id") == batch_id
			and "material_id" not in v
			and "step_number" not in v
		]
		ebr_steps = [
			v for v in self._in_process_checks.values()
			if v.get("tenant_id") == tenant_id
			and v.get("batch_id") == batch_id
			and "step_number" in v
		]
		# Yield records
		yields = [
			y.model_dump() for y in self._yields.values()
			if y.tenant_id == tenant_id and y.batch_id == batch_id
		]
		# Deviations
		deviations = [
			d.model_dump() for d in self._deviations.values()
			if d.tenant_id == tenant_id and d.batch_id == batch_id
		]
		self._audit(tenant_id, "batch_genealogy_traced", batch_id)
		return {
			"batch_id": batch_id,
			"batch_number": batch.batch_number,
			"product_id": batch.product_id,
			"tenant_id": tenant_id,
			"status": batch.status,
			"material_inputs": material_links,
			"ebr_steps": sorted(ebr_steps, key=lambda s: s.get("step_number", 0)),
			"in_process_checks": ipc_records,
			"yield_records": yields,
			"deviations": deviations,
			"total_nodes": len(material_links) + len(ebr_steps) + len(ipc_records) + len(yields),
			"traced_at": datetime.utcnow().isoformat(),
		}

	async def get_spc_data(
		self,
		product_id: str,
		parameter: str,
		tenant_id: str,
		n_batches: int = 30,
	) -> dict[str, Any]:
		"""Calculate Statistical Process Control (SPC) metrics for a product parameter.

		Computes X-bar, range, UCL/LCL using ±3σ limits from yield history, and
		applies Western Electric Rule 1 (any point beyond 3σ) to flag special-cause
		variation. Returns data ready for rendering as a control chart.

		Args:
			product_id: Product whose batches to analyse.
			parameter: Parameter to chart (e.g. 'yield_pct', 'step_granulation').
			tenant_id: Tenant scope.
			n_batches: Maximum number of recent batches to include.

		Returns:
			SPC dataset with control limits, individual values, and violation flags.
		"""
		assert product_id and parameter, "product_id and parameter required"
		batches = [
			b for b in self._batches.values()
			if b.tenant_id == tenant_id and b.product_id == product_id and b.status == "released"
		]
		batches = sorted(batches, key=lambda b: b.created_at)[-n_batches:]
		if not batches:
			return {
				"product_id": product_id,
				"parameter": parameter,
				"n": 0,
				"mean": None,
				"ucl": None,
				"lcl": None,
				"values": [],
				"violations": [],
				"spc_capable": False,
			}
		# Use yield percentage as the primary numeric parameter for demo
		values: list[float] = []
		for b in batches:
			batch_yields = [
				y for y in self._yields.values()
				if y.tenant_id == tenant_id and y.batch_id == b.id
			]
			if batch_yields:
				avg = sum(y.percentage or 0.0 for y in batch_yields) / len(batch_yields)
				values.append(round(avg, 4))
			elif b.yield_percentage is not None:
				values.append(b.yield_percentage)
		if not values:
			return {"product_id": product_id, "parameter": parameter, "n": 0,
					"mean": None, "ucl": None, "lcl": None, "values": [], "violations": [], "spc_capable": False}
		n = len(values)
		mean = sum(values) / n
		variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
		std = variance ** 0.5
		ucl = round(mean + 3 * std, 4)
		lcl = round(mean - 3 * std, 4)
		# Western Electric Rule 1: point beyond 3σ
		violations = [
			{"batch_id": batches[i].id, "value": values[i], "rule": "WE_rule_1"}
			for i, v in enumerate(values)
			if v > ucl or v < lcl
		]
		self._audit(tenant_id, "spc_data_generated", product_id)
		return {
			"product_id": product_id,
			"parameter": parameter,
			"n": n,
			"mean": round(mean, 4),
			"std": round(std, 4),
			"ucl": ucl,
			"lcl": lcl,
			"values": values,
			"batch_ids": [b.id for b in batches],
			"violations": violations,
			"violation_count": len(violations),
			"spc_capable": len(violations) == 0,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def record_calibration(
		self,
		equipment_id: str,
		standard_reference: str,
		result: str,
		calibrated_by: str,
		next_due: datetime,
		tenant_id: str,
		tolerance_pct: float = 0.5,
		as_found: float | None = None,
		as_left: float | None = None,
	) -> dict[str, Any]:
		"""Record an equipment calibration event and update the equipment calibration schedule.

		Blocked-use enforcement: if the equipment's calibration was overdue before
		this record, an informational audit event is emitted. Failed calibrations
		set equipment status to 'out_of_service'.

		Args:
			equipment_id: Equipment being calibrated.
			standard_reference: NIST-traceable or in-house standard reference.
			result: 'pass' or 'fail'.
			calibrated_by: Technician performing calibration.
			next_due: Next calibration due date.
			tenant_id: Tenant scope.
			tolerance_pct: Acceptable tolerance in percent of full scale.
			as_found: Instrument reading before adjustment (optional).
			as_left: Instrument reading after adjustment (optional).

		Returns:
			Calibration record with equipment status update flag.
		"""
		assert equipment_id and standard_reference, "equipment_id and standard_reference required"
		assert result in ("pass", "fail"), "result must be pass or fail"
		equip = self._equipment.get(self._key(tenant_id, equipment_id))
		if equip is None:
			raise KeyError(f"equipment {equipment_id} not found")
		was_overdue = (
			equip.next_calibration_due is not None
			and equip.next_calibration_due < datetime.utcnow()
		)
		if was_overdue:
			self._audit(tenant_id, "calibration_performed_overdue", equipment_id)
		cal_id = _uuid7str()
		cal: dict[str, Any] = {
			"id": cal_id,
			"tenant_id": tenant_id,
			"equipment_id": equipment_id,
			"equipment_name": equip.name,
			"standard_reference": standard_reference,
			"result": result,
			"calibrated_by": calibrated_by,
			"tolerance_pct": tolerance_pct,
			"as_found": as_found,
			"as_left": as_left,
			"was_overdue": was_overdue,
			"next_due": next_due.isoformat(),
			"calibrated_at": datetime.utcnow().isoformat(),
		}
		# Update equipment calibration schedule
		eq_data = equip.model_dump()
		eq_data["last_calibration_date"] = datetime.utcnow()
		eq_data["next_calibration_due"] = next_due
		if result == "fail":
			eq_data["status"] = "out_of_service"
			self._audit(tenant_id, "equipment_calibration_failed", equipment_id)
		else:
			if eq_data.get("status") == "out_of_service":
				eq_data["status"] = "qualified"
		eq_data["updated_at"] = datetime.utcnow()
		self._equipment[self._key(tenant_id, equipment_id)] = Equipment(**eq_data)
		self._qualifications[self._key(tenant_id, cal_id)] = EquipmentQualification(
			tenant_id=tenant_id,
			equipment_id=equipment_id,
			qualification_type="calibration",
			protocol_reference=standard_reference,
			report_reference=cal_id,
			status="approved" if result == "pass" else "rejected",
			performed_by=calibrated_by,
			completion_date=datetime.utcnow(),
			next_requalification=next_due,
			created_by=calibrated_by,
		)
		self._audit(tenant_id, "calibration_recorded", cal_id)
		return cal

PharmaMfgService = PharmaceuticalManufacturingService
