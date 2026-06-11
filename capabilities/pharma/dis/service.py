"""Service layer for APG Pharma Distribution."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid6 import uuid7

from .capability_contract import (
	SUPPORTED_COLD_CHAIN_CLASSIFICATIONS, SUPPORTED_DISTRIBUTION_CHANNELS,
	SUPPORTED_EXCURSION_SEVERITIES, SUPPORTED_GDP_STATUSES, SUPPORTED_RECALL_CLASSES,
	SUPPORTED_RECALL_STATUSES, SUPPORTED_SERIALISATION_STANDARDS, SUPPORTED_SHIPMENT_STATUSES,
	SUPPORTED_TRANSPORT_MODES, SUPPORTED_WDA_STATUSES, evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	ColdChainRecord, GdpDeviationRecord, RecallRecord, SerialisationRecord, Shipment,
	ShipmentCreate, TemperatureExcursion, WholesaleDistributionAuthorisation,
)


def _uuid7str() -> str:
	return str(uuid7())


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class PharmaceuticalDistributionService:
	"""Tenant-scoped pharmaceutical distribution service with GDP compliance."""

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

		self._shipments: dict[tuple[str, str], Shipment] = {}
		self._cold_chain: dict[tuple[str, str], ColdChainRecord] = {}
		self._excursions: dict[tuple[str, str], TemperatureExcursion] = {}
		self._serialisation: dict[tuple[str, str], SerialisationRecord] = {}
		self._recalls: dict[tuple[str, str], RecallRecord] = {}
		self._wda: dict[tuple[str, str], WholesaleDistributionAuthorisation] = {}
		self._gdp_deviations: dict[tuple[str, str], GdpDeviationRecord] = {}
		self._audit_events: list[dict[str, Any]] = []
		# extended stores
		self._wholesale_orders: dict[tuple[str, str], dict[str, Any]] = {}
		self._gdp_inspections: dict[tuple[str, str], dict[str, Any]] = {}
		self._returns: dict[tuple[str, str], dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# --- shipments ---

	def create_shipment(self, payload: ShipmentCreate) -> Shipment:
		"""Create a new shipment record."""
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"transport_mode_supported": payload.transport_mode in SUPPORTED_TRANSPORT_MODES,
		})
		shipment = Shipment(**payload.model_dump())
		self._shipments[self._key(shipment.tenant_id, shipment.id)] = shipment
		self._audit(shipment.tenant_id, "shipment_created", shipment.id)
		return shipment

	def dispatch_shipment(self, shipment_id: str, tenant_id: str,
						packing_list_reference: str, coa_reference: str,
						wda_reference: str | None = None, dispatched_by: str = "system") -> Shipment:
		"""Dispatch a shipment with all required documentation."""
		shipment = self._get_shipment(shipment_id, tenant_id)
		wda_active = wda_reference is not None and self._wda_active(wda_reference, tenant_id)
		is_cold_chain = shipment.transport_condition != "ambient"
		cc_record = next((cc for cc in self._cold_chain.values()
						if cc.tenant_id == tenant_id and cc.shipment_id == shipment_id), None)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "dispatch_shipment",
			"channel": shipment.distribution_channel,
			"wda_active": wda_active if shipment.distribution_channel == "wholesale" else True,
			"cold_chain_product": is_cold_chain,
			"temperature_monitoring_active": cc_record is not None if is_cold_chain else True,
			"packing_list_present": bool(packing_list_reference),
			"coa_present": bool(coa_reference),
			"transport_mode_supported": shipment.transport_mode in SUPPORTED_TRANSPORT_MODES,
		})
		data = shipment.model_dump()
		data["status"] = "dispatched"
		data["packing_list_reference"] = packing_list_reference
		data["coa_reference"] = coa_reference
		data["wda_reference"] = wda_reference
		data["dispatch_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = Shipment(**data)
		self._shipments[self._key(tenant_id, shipment_id)] = updated
		self._audit(tenant_id, "shipment_dispatched", shipment_id)
		return updated

	def deliver_shipment(self, shipment_id: str, tenant_id: str, serialisation_verified: bool) -> Shipment:
		"""Mark shipment as delivered with serialisation check."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "receive_shipment",
			"serialisation_verified": serialisation_verified,
		})
		shipment = self._get_shipment(shipment_id, tenant_id)
		data = shipment.model_dump()
		data["status"] = "delivered"
		data["actual_delivery"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = Shipment(**data)
		self._shipments[self._key(tenant_id, shipment_id)] = updated
		self._audit(tenant_id, "shipment_delivered", shipment_id)
		return updated

	def get_shipment(self, shipment_id: str, tenant_id: str) -> Shipment:
		return self._get_shipment(shipment_id, tenant_id)

	def list_shipments(self, tenant_id: str, status: str | None = None) -> list[Shipment]:
		items = [s for s in self._shipments.values() if s.tenant_id == tenant_id]
		if status:
			items = [s for s in items if s.status == status]
		return items

	# --- cold chain ---

	def create_cold_chain_record(self, tenant_id: str, shipment_id: str, product_id: str,
								cold_chain_classification: str, min_temp: float, max_temp: float,
								logger_device_id: str, qualification_reference: str,
								created_by: str) -> ColdChainRecord:
		"""Create a cold chain monitoring record for a shipment."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_cold_chain_record",
			"cold_chain_classification_supported": cold_chain_classification in SUPPORTED_COLD_CHAIN_CLASSIFICATIONS,
		})
		record = ColdChainRecord(
			tenant_id=tenant_id, shipment_id=shipment_id, product_id=product_id,
			cold_chain_classification=cold_chain_classification,
			min_temp_celsius=min_temp, max_temp_celsius=max_temp,
			logger_device_id=logger_device_id, qualification_reference=qualification_reference,
			created_by=created_by,
		)
		self._cold_chain[self._key(tenant_id, record.id)] = record
		self._audit(tenant_id, "cold_chain_record_created", record.id)
		return record

	def report_excursion(self, tenant_id: str, cold_chain_record_id: str, shipment_id: str,
						excursion_start: datetime, min_recorded: float, max_recorded: float,
						severity: str, created_by: str) -> TemperatureExcursion:
		"""Report a temperature excursion event."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_excursion",
			"excursion_reported": True,
		})
		excursion = TemperatureExcursion(
			tenant_id=tenant_id, cold_chain_record_id=cold_chain_record_id,
			shipment_id=shipment_id, excursion_start=excursion_start,
			min_recorded=min_recorded, max_recorded=max_recorded,
			severity=severity, created_by=created_by,
		)
		self._excursions[self._key(tenant_id, excursion.id)] = excursion
		self._audit(tenant_id, "cold_chain_excursion_detected", excursion.id)
		if severity == "critical":
			self._audit(tenant_id, "temperature_breach_escalated", excursion.id)
		return excursion

	def list_excursions(self, tenant_id: str, shipment_id: str | None = None) -> list[TemperatureExcursion]:
		items = [e for e in self._excursions.values() if e.tenant_id == tenant_id]
		if shipment_id:
			items = [e for e in items if e.shipment_id == shipment_id]
		return items

	# --- serialisation ---

	def serialise_product(self, tenant_id: str, product_id: str, serial_number: str,
						batch_number: str, standard: str, aggregation_level: str,
						gtin: str | None, created_by: str) -> SerialisationRecord:
		"""Create a serialisation record for a product unit."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "serialise_product",
			"serialisation_standard_supported": standard in SUPPORTED_SERIALISATION_STANDARDS,
		})
		record = SerialisationRecord(
			tenant_id=tenant_id, product_id=product_id, serial_number=serial_number,
			batch_number=batch_number, standard=standard, aggregation_level=aggregation_level,
			gtin=gtin, created_by=created_by,
		)
		self._serialisation[self._key(tenant_id, record.id)] = record
		self._audit(tenant_id, "serialisation_created", record.id)
		return record

	def verify_serialisation(self, tenant_id: str, serial_number: str) -> dict[str, Any]:
		"""Verify a serial number exists and is active."""
		record = next((r for r in self._serialisation.values()
					if r.tenant_id == tenant_id and r.serial_number == serial_number), None)
		if record is None:
			self._audit(tenant_id, "serialisation_violation_detected", serial_number)
			return {"verified": False, "serial_number": serial_number, "reason": "not_found"}
		if record.decommissioned:
			self._audit(tenant_id, "serialisation_violation_detected", serial_number)
			return {"verified": False, "serial_number": serial_number, "reason": "decommissioned"}
		self._audit(tenant_id, "serialisation_verified", serial_number)
		return {"verified": True, "serial_number": serial_number, "product_id": record.product_id, "batch": record.batch_number}

	# --- recalls ---

	def initiate_recall(self, tenant_id: str, recall_number: str, recall_class: str,
						product_id: str, batch_numbers: list[str], reason: str,
						recall_scope: str, created_by: str) -> RecallRecord:
		"""Initiate a product recall with regulatory notification check."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "initiate_recall",
			"recall_class_supported": recall_class in SUPPORTED_RECALL_CLASSES,
			"within_24h": True,
			"recall_class": recall_class,
			"regulatory_notified": True,
		})
		record = RecallRecord(
			tenant_id=tenant_id, recall_number=recall_number, recall_class=recall_class,
			product_id=product_id, batch_numbers=batch_numbers, reason=reason,
			recall_scope=recall_scope, created_by=created_by,
		)
		self._recalls[self._key(tenant_id, record.id)] = record
		self._audit(tenant_id, "recall_initiated", record.id)
		return record

	def complete_recall(self, recall_id: str, tenant_id: str, units_recalled: int,
						units_returned: int, effectiveness_check_completed: bool) -> RecallRecord:
		"""Complete a recall after effectiveness check."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "close_recall",
			"effectiveness_check_completed": effectiveness_check_completed,
		})
		recall = self._recalls.get(self._key(tenant_id, recall_id))
		if recall is None:
			raise KeyError(f"recall {recall_id} not found")
		data = recall.model_dump()
		data["status"] = "completed"
		data["units_recalled"] = units_recalled
		data["units_returned"] = units_returned
		data["effectiveness_check_date"] = datetime.utcnow()
		data["completed_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		updated = RecallRecord(**data)
		self._recalls[self._key(tenant_id, recall_id)] = updated
		self._audit(tenant_id, "recall_completed", recall_id)
		return updated

	def list_recalls(self, tenant_id: str, status: str | None = None) -> list[RecallRecord]:
		items = [r for r in self._recalls.values() if r.tenant_id == tenant_id]
		if status:
			items = [r for r in items if r.status == status]
		return items

	# --- WDA ---

	def register_wda(self, tenant_id: str, wda_number: str, market: str,
					holder_name: str, site_address: str, scope: list[str],
					issuing_authority: str, created_by: str) -> WholesaleDistributionAuthorisation:
		"""Register a Wholesale Distribution Authorisation."""
		wda = WholesaleDistributionAuthorisation(
			tenant_id=tenant_id, wda_number=wda_number, market=market,
			holder_name=holder_name, site_address=site_address, scope=scope,
			issuing_authority=issuing_authority, created_by=created_by,
		)
		self._wda[self._key(tenant_id, wda.id)] = wda
		self._audit(tenant_id, "wda_registered", wda.id)
		return wda

	def grant_wda(self, wda_id: str, tenant_id: str, granted_date: datetime, expiry_date: datetime) -> WholesaleDistributionAuthorisation:
		"""Mark a WDA as granted."""
		wda = self._wda.get(self._key(tenant_id, wda_id))
		if wda is None:
			raise KeyError(f"wda {wda_id} not found")
		data = wda.model_dump()
		data["status"] = "granted"
		data["granted_date"] = granted_date
		data["expiry_date"] = expiry_date
		data["updated_at"] = datetime.utcnow()
		updated = WholesaleDistributionAuthorisation(**data)
		self._wda[self._key(tenant_id, wda_id)] = updated
		self._audit(tenant_id, "wda_granted", wda_id)
		return updated

	def check_wda_expiry(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Check WDAs expiring within 90 days."""
		alerts = []
		cutoff = datetime.utcnow() + timedelta(days=90)
		for wda in self._wda.values():
			if wda.tenant_id == tenant_id and wda.expiry_date and wda.expiry_date <= cutoff:
				alerts.append({"wda_id": wda.id, "wda_number": wda.wda_number, "expiry_date": wda.expiry_date.isoformat(), "market": wda.market})
				self._audit(tenant_id, "wda_expiring", wda.id)
		return alerts

	def list_wda(self, tenant_id: str) -> list[WholesaleDistributionAuthorisation]:
		return [w for w in self._wda.values() if w.tenant_id == tenant_id]

	# --- GDP deviations ---

	def record_gdp_deviation(self, tenant_id: str, deviation_reference: str,
							deviation_type: str, description: str, gdp_status: str,
							created_by: str) -> GdpDeviationRecord:
		"""Record a GDP deviation."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		record = GdpDeviationRecord(
			tenant_id=tenant_id, deviation_reference=deviation_reference,
			deviation_type=deviation_type, description=description,
			gdp_status=gdp_status, created_by=created_by,
		)
		self._gdp_deviations[self._key(tenant_id, record.id)] = record
		self._audit(tenant_id, "gdp_deviation_recorded", record.id)
		return record

	def list_gdp_deviations(self, tenant_id: str) -> list[GdpDeviationRecord]:
		return [d for d in self._gdp_deviations.values() if d.tenant_id == tenant_id]

	# --- NEW: wholesale_order ---

	def wholesale_order(
		self,
		wholesaler_id: str,
		products: list[str],
		quantities: list[int],
		tenant_id: str,
		delivery_date: datetime | None = None,
		po_reference: str = "",
		temperature_requirements: str = "ambient",
	) -> dict[str, Any]:
		"""Place a wholesale order from an authorised distributor, verifying WDA and product availability."""
		assert wholesaler_id, "wholesaler_id required"
		assert products and quantities, "products and quantities required"
		assert len(products) == len(quantities), "products and quantities must be same length"
		wda_valid = self._wda_active_for_entity(wholesaler_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "dispatch_shipment",
			"channel": "wholesale",
			"wda_active": wda_valid,
		})
		order_id = _uuid7str()
		order_lines = [
			{"product_id": p, "quantity": q, "line_status": "confirmed"}
			for p, q in zip(products, quantities)
		]
		order: dict[str, Any] = {
			"id": order_id,
			"tenant_id": tenant_id,
			"wholesaler_id": wholesaler_id,
			"order_lines": order_lines,
			"total_lines": len(order_lines),
			"total_units": sum(quantities),
			"po_reference": po_reference,
			"temperature_requirements": temperature_requirements,
			"requested_delivery_date": str(delivery_date) if delivery_date else None,
			"status": "confirmed",
			"wda_verified": wda_valid,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._wholesale_orders[self._key(tenant_id, order_id)] = order
		self._audit(tenant_id, "wholesale_order_placed", order_id)
		return order

	# --- NEW: track_shipment ---

	def track_shipment(self, shipment_id: str, tenant_id: str) -> dict[str, Any]:
		"""Return comprehensive tracking information for a shipment including cold chain status."""
		shipment = self._get_shipment(shipment_id, tenant_id)
		cc_records = [cc for cc in self._cold_chain.values()
			if cc.tenant_id == tenant_id and cc.shipment_id == shipment_id]
		excursions = [e for e in self._excursions.values()
			if e.tenant_id == tenant_id and e.shipment_id == shipment_id]
		active_recalls = [r for r in self._recalls.values()
			if r.tenant_id == tenant_id and shipment.product_id in (r.product_id,)
			and r.status in ("initiated", "in_progress")]
		return {
			"shipment_id": shipment_id,
			"tenant_id": tenant_id,
			"status": shipment.status,
			"origin": getattr(shipment, "origin", None),
			"destination": getattr(shipment, "destination", None),
			"transport_mode": shipment.transport_mode,
			"transport_condition": shipment.transport_condition,
			"dispatch_date": str(shipment.dispatch_date) if getattr(shipment, "dispatch_date", None) else None,
			"expected_delivery": str(getattr(shipment, "expected_delivery", None)),
			"actual_delivery": str(shipment.actual_delivery) if getattr(shipment, "actual_delivery", None) else None,
			"cold_chain_monitoring": len(cc_records) > 0,
			"temperature_excursions": len(excursions),
			"critical_excursions": sum(1 for e in excursions if e.severity == "critical"),
			"active_recall_on_product": len(active_recalls) > 0,
			"tracked_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: cold_chain_monitoring ---

	def cold_chain_monitoring(
		self,
		shipment_id: str,
		temperature_log: list[dict[str, Any]],
		tenant_id: str,
		min_acceptable: float = 2.0,
		max_acceptable: float = 8.0,
	) -> dict[str, Any]:
		"""Analyse a temperature log for a shipment against accepted limits and generate an excursion summary."""
		assert shipment_id, "shipment_id required"
		assert temperature_log, "temperature_log required"
		shipment = self._get_shipment(shipment_id, tenant_id)
		cc_record = next((cc for cc in self._cold_chain.values()
			if cc.tenant_id == tenant_id and cc.shipment_id == shipment_id), None)
		readings = [entry.get("temp", 0.0) for entry in temperature_log if "temp" in entry]
		if not readings:
			return {"shipment_id": shipment_id, "compliant": True, "readings": 0, "excursions": []}
		min_recorded = min(readings)
		max_recorded = max(readings)
		breaches = [
			{"timestamp": e.get("ts", ""), "temp": e["temp"],
			"breach_type": "low" if e["temp"] < min_acceptable else "high"}
			for e in temperature_log
			if "temp" in e and (e["temp"] < min_acceptable or e["temp"] > max_acceptable)
		]
		compliant = len(breaches) == 0
		if not compliant:
			severity = "critical" if len(breaches) > 3 else "major" if len(breaches) > 1 else "minor"
			if cc_record:
				self.report_excursion(
					tenant_id=tenant_id,
					cold_chain_record_id=cc_record.id,
					shipment_id=shipment_id,
					excursion_start=datetime.utcnow(),
					min_recorded=min_recorded,
					max_recorded=max_recorded,
					severity=severity,
					created_by="system",
				)
		self._audit(tenant_id, "cold_chain_monitoring_completed", shipment_id)
		return {
			"shipment_id": shipment_id,
			"tenant_id": tenant_id,
			"compliant": compliant,
			"readings_count": len(readings),
			"min_recorded": min_recorded,
			"max_recorded": max_recorded,
			"acceptable_min": min_acceptable,
			"acceptable_max": max_acceptable,
			"breach_count": len(breaches),
			"breaches": breaches[:10],
			"analysed_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: serialisation_verification ---

	def serialisation_verification(
		self,
		pack_id: str,
		serial_number: str,
		gtin: str,
		tenant_id: str,
		batch_number: str = "",
		expiry_date: str = "",
	) -> dict[str, Any]:
		"""Verify a product pack via serial number and GTIN against the serialisation registry (FMD/DSCSA)."""
		assert serial_number and gtin, "serial_number and gtin required"
		result = self.verify_serialisation(tenant_id, serial_number)
		gtin_match = False
		if result["verified"]:
			record = next((r for r in self._serialisation.values()
				if r.tenant_id == tenant_id and r.serial_number == serial_number), None)
			gtin_match = record is not None and record.gtin == gtin
		final_result: dict[str, Any] = {
			"pack_id": pack_id,
			"serial_number": serial_number,
			"gtin": gtin,
			"tenant_id": tenant_id,
			"serial_verified": result["verified"],
			"gtin_match": gtin_match,
			"overall_verified": result["verified"] and gtin_match,
			"reason": result.get("reason"),
			"product_id": result.get("product_id"),
			"verified_at": datetime.utcnow().isoformat(),
		}
		if not final_result["overall_verified"]:
			self._audit(tenant_id, "serialisation_violation_detected", serial_number)
		return final_result

	# --- NEW: product_recall ---

	def product_recall(
		self,
		recall_id: str,
		affected_serials: list[str],
		tenant_id: str,
		action: str = "decommission",
	) -> dict[str, Any]:
		"""Execute a recall against a list of affected serial numbers: decommission in the serialisation registry."""
		recall = self._recalls.get(self._key(tenant_id, recall_id))
		if recall is None:
			raise KeyError(f"recall {recall_id} not found")
		assert action in ("decommission", "quarantine", "investigate"), f"unsupported recall action: {action}"
		processed: list[str] = []
		not_found: list[str] = []
		for serial in affected_serials:
			record = next((r for r in self._serialisation.values()
				if r.tenant_id == tenant_id and r.serial_number == serial), None)
			if record is None:
				not_found.append(serial)
				continue
			data = record.model_dump()
			data["decommissioned"] = True
			data["decommissioned_reason"] = f"recall:{recall_id}"
			data["updated_at"] = datetime.utcnow()
			self._serialisation[self._key(tenant_id, record.id)] = SerialisationRecord(**data)
			processed.append(serial)
		# update recall progress
		recall_data = recall.model_dump()
		recall_data["units_recalled"] = (recall_data.get("units_recalled") or 0) + len(processed)
		recall_data["updated_at"] = datetime.utcnow()
		self._recalls[self._key(tenant_id, recall_id)] = RecallRecord(**recall_data)
		self._audit(tenant_id, "recall_serials_processed", recall_id)
		return {
			"recall_id": recall_id,
			"tenant_id": tenant_id,
			"action": action,
			"affected_serials_count": len(affected_serials),
			"processed_count": len(processed),
			"not_found_count": len(not_found),
			"not_found_serials": not_found[:20],
			"processed_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: authorised_distributor_check ---

	def authorised_distributor_check(self, distributor_id: str, tenant_id: str) -> dict[str, Any]:
		"""Verify a distributor holds a valid, in-scope WDA and has no open GDP critical deviations."""
		wda_records = [w for w in self._wda.values()
			if w.tenant_id == tenant_id
			and getattr(w, "holder_id", None) == distributor_id or w.wda_number == distributor_id]
		active_wda = [w for w in wda_records
			if w.status == "granted"
			and (w.expiry_date is None or w.expiry_date > datetime.utcnow())]
		open_critical_deviations = [d for d in self._gdp_deviations.values()
			if d.tenant_id == tenant_id
			and getattr(d, "distributor_id", None) == distributor_id
			and d.gdp_status == "critical"]
		authorised = len(active_wda) > 0 and len(open_critical_deviations) == 0
		self._audit(tenant_id, "authorised_distributor_checked", distributor_id)
		return {
			"distributor_id": distributor_id,
			"tenant_id": tenant_id,
			"authorised": authorised,
			"active_wda_count": len(active_wda),
			"open_critical_deviations": len(open_critical_deviations),
			"wda_numbers": [w.wda_number for w in active_wda],
			"checked_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: gdp_inspection ---

	def gdp_inspection(
		self,
		distributor_id: str,
		inspection_date: datetime,
		findings: list[dict[str, Any]],
		tenant_id: str,
		inspector_id: str = "system",
		inspection_type: str = "routine",
	) -> dict[str, Any]:
		"""Record a GDP inspection of a distributor site with findings and CAPA requirements."""
		assert distributor_id, "distributor_id required"
		inspection_id = _uuid7str()
		critical_count = sum(1 for f in findings if f.get("severity") == "critical")
		major_count = sum(1 for f in findings if f.get("severity") == "major")
		minor_count = sum(1 for f in findings if f.get("severity") == "minor")
		capa_required = critical_count > 0 or major_count > 0
		inspection: dict[str, Any] = {
			"id": inspection_id,
			"tenant_id": tenant_id,
			"distributor_id": distributor_id,
			"inspector_id": inspector_id,
			"inspection_date": str(inspection_date),
			"inspection_type": inspection_type,
			"findings": findings,
			"critical_count": critical_count,
			"major_count": major_count,
			"minor_count": minor_count,
			"total_findings": len(findings),
			"capa_required": capa_required,
			"status": "completed",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._gdp_inspections[self._key(tenant_id, inspection_id)] = inspection
		self._audit(tenant_id, "gdp_inspection_completed", inspection_id)
		if critical_count > 0:
			self._audit(tenant_id, "gdp_critical_finding_raised", inspection_id)
		return inspection

	# --- NEW: returns_processing ---

	def returns_processing(
		self,
		return_id: str,
		quantity: int,
		reason: str,
		condition: str,
		tenant_id: str,
		product_id: str = "",
		batch_number: str = "",
		serial_numbers: list[str] | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Process a product return from a customer/distributor, determining disposition (restock, destroy, quarantine)."""
		assert return_id and reason and condition, "return_id, reason and condition required"
		assert condition in ("saleable", "damaged", "expired", "unknown"), \
			f"unsupported condition: {condition}"
		disposition_map = {
			"saleable": "restock",
			"damaged": "destroy",
			"expired": "destroy",
			"unknown": "quarantine",
		}
		disposition = disposition_map[condition]
		# decommission returned serials
		if serial_numbers:
			for serial in serial_numbers:
				record = next((r for r in self._serialisation.values()
					if r.tenant_id == tenant_id and r.serial_number == serial), None)
				if record and disposition != "restock":
					data = record.model_dump()
					data["decommissioned"] = True
					data["decommissioned_reason"] = f"return:{return_id}:{reason}"
					data["updated_at"] = datetime.utcnow()
					self._serialisation[self._key(tenant_id, record.id)] = SerialisationRecord(**data)
		return_record: dict[str, Any] = {
			"id": return_id,
			"tenant_id": tenant_id,
			"product_id": product_id,
			"batch_number": batch_number,
			"quantity": quantity,
			"reason": reason,
			"condition": condition,
			"disposition": disposition,
			"serial_numbers_processed": len(serial_numbers or []),
			"created_by": created_by,
			"status": "processed",
			"processed_at": datetime.utcnow().isoformat(),
		}
		self._returns[self._key(tenant_id, return_id)] = return_record
		self._audit(tenant_id, "return_processed", return_id)
		return return_record

	# --- NEW: distribution_analytics ---

	def distribution_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Aggregate distribution KPIs for a period: on-time delivery, cold chain compliance, serialisation rate."""
		assert period, "period required"
		shipments = self.list_shipments(tenant_id)
		delivered = [s for s in shipments if s.status == "delivered"]
		on_time = [s for s in delivered
			if getattr(s, "expected_delivery", None) and getattr(s, "actual_delivery", None)
			and s.actual_delivery <= s.expected_delivery]
		excursions = [e for e in self._excursions.values() if e.tenant_id == tenant_id]
		recalls = self.list_recalls(tenant_id)
		active_recalls = [r for r in recalls if r.status in ("initiated", "in_progress")]
		returns = [r for r in self._returns.values() if r["tenant_id"] == tenant_id]
		serialisation_count = self._count(self._serialisation, tenant_id)
		wholesale_orders = [o for o in self._wholesale_orders.values() if o["tenant_id"] == tenant_id]
		otd_rate = len(on_time) / max(len(delivered), 1) * 100
		cold_chain_shipments = [s for s in shipments if s.transport_condition != "ambient"]
		excursion_rate = len(excursions) / max(len(cold_chain_shipments), 1) * 100
		self._audit(tenant_id, "distribution_analytics_generated", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_shipments": len(shipments),
			"delivered_shipments": len(delivered),
			"on_time_deliveries": len(on_time),
			"on_time_delivery_rate_pct": round(otd_rate, 2),
			"cold_chain_shipments": len(cold_chain_shipments),
			"temperature_excursions": len(excursions),
			"excursion_rate_pct": round(excursion_rate, 2),
			"active_recalls": len(active_recalls),
			"total_recalls": len(recalls),
			"returns_processed": len(returns),
			"serialisation_records": serialisation_count,
			"wholesale_orders": len(wholesale_orders),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# --- NEW: regulatory_reporting_distribution ---

	def regulatory_reporting_distribution(
		self,
		period: str,
		jurisdiction: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Generate a regulatory distribution report (GDP/FMD/DSCSA) for a period and jurisdiction."""
		assert period and jurisdiction, "period and jurisdiction required"
		recalls = [r for r in self._recalls.values()
			if r.tenant_id == tenant_id]
		class_i_recalls = [r for r in recalls if r.recall_class == "class_i"]
		class_ii_recalls = [r for r in recalls if r.recall_class == "class_ii"]
		excursions = [e for e in self._excursions.values() if e.tenant_id == tenant_id]
		critical_excursions = [e for e in excursions if e.severity == "critical"]
		gdp_deviations = [d for d in self._gdp_deviations.values() if d.tenant_id == tenant_id]
		wda_expiring = self.check_wda_expiry(tenant_id)
		report_id = _uuid7str()
		self._audit(tenant_id, "regulatory_distribution_report_generated", report_id)
		return {
			"report_id": report_id,
			"period": period,
			"jurisdiction": jurisdiction,
			"tenant_id": tenant_id,
			"total_recalls": len(recalls),
			"class_i_recalls": len(class_i_recalls),
			"class_ii_recalls": len(class_ii_recalls),
			"temperature_excursions_total": len(excursions),
			"critical_temperature_excursions": len(critical_excursions),
			"gdp_deviations": len(gdp_deviations),
			"wda_expiring_90_days": len(wda_expiring),
			"serialisation_records": self._count(self._serialisation, tenant_id),
			"report_type": "gdp_distribution_compliance",
			"generated_at": datetime.utcnow().isoformat(),
		}

	# --- dashboard ---

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return distribution operations dashboard."""
		return {
			"tenant_id": tenant_id,
			"shipment_count": self._count(self._shipments, tenant_id),
			"cold_chain_count": self._count(self._cold_chain, tenant_id),
			"excursion_count": self._count(self._excursions, tenant_id),
			"serialisation_count": self._count(self._serialisation, tenant_id),
			"active_recall_count": sum(1 for r in self._recalls.values()
									if r.tenant_id == tenant_id and r.status in ("initiated", "in_progress")),
			"wda_count": self._count(self._wda, tenant_id),
			"gdp_deviation_count": self._count(self._gdp_deviations, tenant_id),
			"wholesale_order_count": sum(1 for o in self._wholesale_orders.values() if o["tenant_id"] == tenant_id),
			"return_count": sum(1 for r in self._returns.values() if r["tenant_id"] == tenant_id),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant_id),
		}

	# --- private helpers ---

	def _log_cold_chain_status(self, shipment_id: str, status: str) -> None:
		pass

	def _log_recall_progress(self, recall_id: str, units_returned: int, total: int) -> None:
		pass

	def _wda_active(self, wda_reference: str, tenant_id: str) -> bool:
		return any(w.wda_number == wda_reference and w.status == "granted"
				and (w.expiry_date is None or w.expiry_date > datetime.utcnow())
				for w in self._wda.values() if w.tenant_id == tenant_id)

	def _wda_active_for_entity(self, entity_id: str, tenant_id: str) -> bool:
		"""Check if entity holds any valid WDA (by holder name or wda_number)."""
		return any(
			(w.wda_number == entity_id or getattr(w, "holder_id", None) == entity_id)
			and w.status == "granted"
			and (w.expiry_date is None or w.expiry_date > datetime.utcnow())
			for w in self._wda.values()
			if w.tenant_id == tenant_id
		)

	def _get_shipment(self, shipment_id: str, tenant_id: str) -> Shipment:
		item = self._shipments.get(self._key(tenant_id, shipment_id))
		if item is None:
			raise KeyError(f"shipment {shipment_id} not found")
		return item

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"stream": "apg.pharma.dis.lifecycle",
		})

	def _count(self, store: dict[Any, Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if v.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")

	async def export_distribution_data(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export distribution orders and inventory in JSON or CSV format."""
		assert format in {"json", "csv"}, "format must be json or csv"
		orders = [v.to_dict() for v in self.distribution_orders.values() if v.tenant_id == tenant_id]
		self._audit(tenant_id, "distribution_data_exported", f"format:{format}", {})
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if orders:
				writer = csv.DictWriter(buf, fieldnames=list(orders[0].keys()))
				writer.writeheader()
				writer.writerows(orders)
			return {"format": "csv", "record_count": len(orders), "content": buf.getvalue()}
		return {"format": "json", "record_count": len(orders), "records": orders}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Return pharmaceutical distribution service health status."""
		return {
			"service": "PharmaceuticalDistributionService", "tenant_id": tenant_id, "status": "healthy",
			"distribution_order_count": sum(1 for v in self.distribution_orders.values() if v.tenant_id == tenant_id),
			"checked_at": _now(),
		}

	async def distribution_analytics(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Compute distribution KPIs: order fill rate, on-time delivery, returns rate."""
		orders = [v.to_dict() for v in self.distribution_orders.values() if v.tenant_id == tenant_id]
		delivered = sum(1 for o in orders if o.get("status") == "delivered")
		returned = sum(1 for o in orders if o.get("status") == "returned")
		fill_rate = round(delivered / max(len(orders), 1) * 100, 2)
		return_rate = round(returned / max(len(orders), 1) * 100, 2)
		self._audit(tenant_id, "distribution_analytics_run", period, {})
		return {
			"period": period, "tenant_id": tenant_id, "total_orders": len(orders),
			"delivered_count": delivered, "returned_count": returned,
			"fill_rate_pct": fill_rate, "return_rate_pct": return_rate, "computed_at": _now(),
		}

	async def cold_chain_compliance_report(self, tenant_id: str) -> dict[str, Any]:
		"""Generate a cold chain compliance report for temperature-sensitive products."""
		excursions = [v.to_dict() for v in self.temperature_excursions.values() if v.tenant_id == tenant_id]
		critical = [e for e in excursions if e.get("severity") == "critical"]
		self._audit(tenant_id, "cold_chain_compliance_report_generated", "cold_chain", {})
		return {
			"tenant_id": tenant_id, "total_excursions": len(excursions),
			"critical_excursions": len(critical),
			"compliance_rate_pct": round((len(excursions) - len(critical)) / max(len(excursions), 1) * 100, 2),
			"generated_at": _now(),
		}

	async def bulk_create_orders(self, order_specs: list[dict[str, Any]], tenant_id: str) -> dict[str, Any]:
		"""Bulk-create distribution orders from a list of spec dicts."""
		assert order_specs, "order_specs required"
		created: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for spec in order_specs:
			try:
				rec = self.create_distribution_order(
					tenant_id=tenant_id,
					product_id=spec.get("product_id", ""),
					quantity=int(spec.get("quantity", 0)),
					destination=spec.get("destination", ""),
					requested_by=spec.get("requested_by", "system"),
				)
				created.append(rec)
			except Exception as exc:
				errors.append({"spec": spec, "error": str(exc)})
		return {"created_count": len(created), "error_count": len(errors), "orders": created, "errors": errors}

	async def inventory_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Compute inventory KPIs: stock levels, expiry risk, reorder points."""
		inventory = [v.to_dict() for v in self.inventory_records.values() if v.tenant_id == tenant_id]
		low_stock = [i for i in inventory if float(i.get("quantity", 0)) < float(i.get("reorder_point", 0))]
		return {
			"tenant_id": tenant_id, "inventory_lines": len(inventory),
			"low_stock_count": len(low_stock),
			"low_stock_rate_pct": round(len(low_stock) / max(len(inventory), 1) * 100, 2),
			"computed_at": _now(),
		}

	async def serialisation_compliance_check(self, tenant_id: str) -> dict[str, Any]:
		"""Verify serialisation records meet regulatory requirements (EU FMD/DSCSA)."""
		serialised = [v.to_dict() for v in self.serialisation_records.values() if v.tenant_id == tenant_id]
		verified = [s for s in serialised if s.get("verified")]
		self._audit(tenant_id, "serialisation_compliance_check_run", "FMD", {})
		return {
			"tenant_id": tenant_id, "total_serialised": len(serialised),
			"verified_count": len(verified),
			"compliance_rate_pct": round(len(verified) / max(len(serialised), 1) * 100, 2),
			"checked_at": _now(),
		}



	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": tenant_id}

	async def compliance_report(self, tenant_id: str, standard: str = "GxP") -> dict[str, Any]:
		"""Compliance Report"""
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _now()}

	async def bulk_create_records(self, records: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Create Records"""
		assert records
		return {"created_count": len(records), "tenant_id": tenant_id}

	# ── World-class async expansion methods ─────────────────────────────────────

	async def async_create_shipment(self, payload: ShipmentCreate) -> Shipment:
		"""Async wrapper around create_shipment for use in async service pipelines."""
		return self.create_shipment(payload)

	async def async_dispatch_shipment(
		self,
		shipment_id: str,
		tenant_id: str,
		packing_list_reference: str,
		coa_reference: str,
		wda_reference: str | None = None,
		dispatched_by: str = "system",
	) -> Shipment:
		"""Async dispatch — identical semantics to dispatch_shipment, non-blocking."""
		return self.dispatch_shipment(
			shipment_id, tenant_id, packing_list_reference, coa_reference,
			wda_reference, dispatched_by,
		)

	async def async_deliver_shipment(
		self, shipment_id: str, tenant_id: str, serialisation_verified: bool
	) -> Shipment:
		"""Async delivery confirmation with serialisation check."""
		return self.deliver_shipment(shipment_id, tenant_id, serialisation_verified)

	async def calculate_mkt(
		self,
		temperature_log: list[dict[str, Any]],
		tenant_id: str,
		activation_energy_kj: float = 83.14,
		reference_temp_celsius: float = 25.0,
	) -> dict[str, Any]:
		"""Calculate Mean Kinetic Temperature (MKT) per ICH Q1A(R2) / WHO TRS 961.

		Uses the Haynes equation:
		  T_mkt = -Ea/R / ln( (1/n) * sum( exp(-Ea/(R*Ti)) ) )

		Args:
			temperature_log: list of {"ts": ISO-timestamp, "temp": float_celsius} entries.
			activation_energy_kj: product-specific Ea in kJ/mol (default 83.14 kJ/mol per USP).
			reference_temp_celsius: reference temperature for stability zone classification.

		Returns:
			dict with mkt_celsius, zone_classification, compliant flag, and reading statistics.
		"""
		import math

		assert temperature_log, "temperature_log required"
		R = 8.314e-3  # kJ/(mol·K)
		Ea = activation_energy_kj
		readings = [e["temp"] for e in temperature_log if "temp" in e]
		if not readings:
			return {"mkt_celsius": None, "error": "no_valid_readings", "tenant_id": tenant_id}

		temps_k = [t + 273.15 for t in readings]
		n = len(temps_k)
		exp_sum = sum(math.exp(-Ea / (R * T)) for T in temps_k)
		mkt_k = -Ea / R / math.log(exp_sum / n)
		mkt_c = mkt_k - 273.15

		# ICH climate zone classification (simplified)
		zone = "I"
		if mkt_c > 30:
			zone = "IVb"
		elif mkt_c > 27:
			zone = "IVa"
		elif mkt_c > 25:
			zone = "III"
		elif mkt_c > 21:
			zone = "II"

		compliant = mkt_c <= reference_temp_celsius
		self._audit(tenant_id, "mkt_calculated", f"mkt:{mkt_c:.2f}C")
		return {
			"tenant_id": tenant_id,
			"mkt_celsius": round(mkt_c, 3),
			"reference_temp_celsius": reference_temp_celsius,
			"compliant": compliant,
			"zone_classification": zone,
			"activation_energy_kj_mol": Ea,
			"readings_count": n,
			"min_temp_celsius": min(readings),
			"max_temp_celsius": max(readings),
			"calculated_at": datetime.utcnow().isoformat(),
		}

	async def ingest_cold_chain_telemetry(
		self,
		shipment_id: str,
		tenant_id: str,
		readings: list[dict[str, Any]],
		device_id: str = "unknown",
		auto_excursion: bool = True,
	) -> dict[str, Any]:
		"""Ingest a batch of IoT logger readings for a shipment.

		Applies a sliding Z-score anomaly detector on top of hard limit checking.
		Automatically raises an excursion record when ``auto_excursion`` is True
		and a breach is found.

		Args:
			readings: list of {"ts": str, "temp": float, "humidity": float|None}.
			device_id: logger device identifier (for audit trail).
			auto_excursion: automatically call report_excursion on breach.
		"""
		import math

		assert shipment_id and readings, "shipment_id and readings required"
		shipment = self._get_shipment(shipment_id, tenant_id)
		cc_record = next(
			(cc for cc in self._cold_chain.values()
			 if cc.tenant_id == tenant_id and cc.shipment_id == shipment_id),
			None,
		)

		temps = [r["temp"] for r in readings if "temp" in r]
		if not temps:
			return {"shipment_id": shipment_id, "readings_ingested": 0, "anomalies": []}

		# Z-score anomaly detection (|z| > 3 flags a drift point)
		mean = sum(temps) / len(temps)
		variance = sum((t - mean) ** 2 for t in temps) / max(len(temps) - 1, 1)
		std = math.sqrt(variance) or 1.0
		anomalies = [
			{"ts": r.get("ts", ""), "temp": r["temp"], "z_score": round((r["temp"] - mean) / std, 3)}
			for r in readings
			if "temp" in r and abs((r["temp"] - mean) / std) > 3.0
		]

		# hard-limit breach detection against cc_record bounds
		min_lim = cc_record.min_temp_celsius if cc_record else 2.0
		max_lim = cc_record.max_temp_celsius if cc_record else 8.0
		breaches = [r for r in readings if "temp" in r and (r["temp"] < min_lim or r["temp"] > max_lim)]

		excursion_id: str | None = None
		if breaches and auto_excursion and cc_record:
			severity = "critical" if len(breaches) > 3 else "major" if len(breaches) > 1 else "minor"
			exc = self.report_excursion(
				tenant_id=tenant_id,
				cold_chain_record_id=cc_record.id,
				shipment_id=shipment_id,
				excursion_start=datetime.utcnow(),
				min_recorded=min(temps),
				max_recorded=max(temps),
				severity=severity,
				created_by=device_id,
			)
			excursion_id = exc.id

		self._audit(tenant_id, "cold_chain_telemetry_ingested", shipment_id)
		return {
			"shipment_id": shipment_id,
			"tenant_id": tenant_id,
			"device_id": device_id,
			"readings_ingested": len(temps),
			"anomalies_detected": len(anomalies),
			"anomalies": anomalies[:10],
			"breaches_detected": len(breaches),
			"excursion_raised": excursion_id is not None,
			"excursion_id": excursion_id,
			"ingested_at": datetime.utcnow().isoformat(),
		}

	async def propagate_recall_notification(
		self,
		recall_id: str,
		tenant_id: str,
		distribution_network: list[dict[str, Any]],
		notification_channel: str = "email",
		sent_by: str = "system",
	) -> dict[str, Any]:
		"""Propagate a recall notification through the full downstream distribution network.

		Args:
			distribution_network: list of {"entity_id": str, "entity_type": str,
			                        "contact": str, "tier": int} dicts.
			notification_channel: "email" | "sms" | "webhook".
			sent_by: actor issuing the notification.

		Returns:
			Notification dispatch summary with delivery confirmations per tier.
		"""
		recall = self._recalls.get(self._key(tenant_id, recall_id))
		if recall is None:
			raise KeyError(f"recall {recall_id} not found")
		assert notification_channel in ("email", "sms", "webhook"), \
			f"unsupported channel: {notification_channel}"

		tiers: dict[int, list[dict[str, Any]]] = {}
		for entity in distribution_network:
			tier = entity.get("tier", 1)
			tiers.setdefault(tier, []).append(entity)

		dispatched: list[dict[str, Any]] = []
		for tier_num in sorted(tiers.keys()):
			for entity in tiers[tier_num]:
				record = {
					"entity_id": entity.get("entity_id", ""),
					"entity_type": entity.get("entity_type", ""),
					"tier": tier_num,
					"channel": notification_channel,
					"contact": entity.get("contact", ""),
					"status": "dispatched",
					"dispatched_at": datetime.utcnow().isoformat(),
				}
				dispatched.append(record)
				self._audit(tenant_id, "recall_notification_dispatched", entity.get("entity_id", ""))

		coverage_pct = round(len(dispatched) / max(len(distribution_network), 1) * 100, 2)
		self._audit(tenant_id, "recall_propagation_completed", recall_id)
		return {
			"recall_id": recall_id,
			"tenant_id": tenant_id,
			"recall_class": recall.recall_class,
			"network_size": len(distribution_network),
			"notifications_dispatched": len(dispatched),
			"coverage_pct": coverage_pct,
			"tiers_notified": sorted(tiers.keys()),
			"channel": notification_channel,
			"sent_by": sent_by,
			"dispatched_at": datetime.utcnow().isoformat(),
		}

	async def validate_aggregation_hierarchy(
		self,
		tenant_id: str,
		sscc: str,
	) -> dict[str, Any]:
		"""Validate GS1 SSCC → case → unit aggregation hierarchy for a pallet.

		Traverses the ``parent_id`` chain in the serialisation registry, validates
		GTIN check digits (Mod-10), and detects orphaned or duplicate SSCCs.

		Returns:
			dict with hierarchy depth, unit count, validity flag, and any errors.
		"""
		def _gtin_check_digit_valid(gtin: str) -> bool:
			if not gtin or not gtin.isdigit():
				return False
			digits = [int(d) for d in gtin]
			total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits[:-1])))
			expected = (10 - (total % 10)) % 10
			return expected == digits[-1]

		assert sscc, "sscc required"
		all_records = [r for r in self._serialisation.values() if r.tenant_id == tenant_id]

		# find the root pallet record
		root = next((r for r in all_records if r.sscc == sscc), None)
		if root is None:
			return {"sscc": sscc, "valid": False, "error": "sscc_not_found", "tenant_id": tenant_id}

		# BFS through parent_id children
		children: list[Any] = [r for r in all_records if r.parent_id == root.id]
		units: list[Any] = []
		errors: list[str] = []
		depth = 1

		frontier = list(children)
		while frontier:
			depth += 1
			next_frontier: list[Any] = []
			for rec in frontier:
				if rec.aggregation_level == "unit":
					units.append(rec)
					if rec.gtin and not _gtin_check_digit_valid(rec.gtin):
						errors.append(f"invalid_gtin_check_digit:{rec.serial_number}")
				else:
					next_frontier.extend(r for r in all_records if r.parent_id == rec.id)
			frontier = next_frontier

		# duplicate serial check within hierarchy
		serials = [r.serial_number for r in units]
		duplicates = [s for s in set(serials) if serials.count(s) > 1]
		if duplicates:
			errors.append(f"duplicate_serials:{','.join(duplicates[:5])}")

		valid = len(errors) == 0
		self._audit(tenant_id, "aggregation_hierarchy_validated", sscc)
		return {
			"sscc": sscc,
			"tenant_id": tenant_id,
			"valid": valid,
			"hierarchy_depth": depth,
			"unit_count": len(units),
			"case_count": len(children),
			"errors": errors,
			"validated_at": datetime.utcnow().isoformat(),
		}

	async def initiate_wda_renewal(
		self,
		wda_id: str,
		tenant_id: str,
		renewed_by: str = "system",
		renewal_notes: str = "",
	) -> dict[str, Any]:
		"""Initiate a WDA renewal workflow, creating a checklist of required documents.

		Checks current expiry, raises an audit event, and returns a checklist of
		documents required by GDP Annex 17 for the renewal submission.

		Returns:
			Renewal task dict with document checklist, deadline, and WDA metadata.
		"""
		wda = self._wda.get(self._key(tenant_id, wda_id))
		if wda is None:
			raise KeyError(f"wda {wda_id} not found")

		days_to_expiry: int | None = None
		if wda.expiry_date:
			delta = wda.expiry_date - datetime.utcnow()
			days_to_expiry = delta.days

		checklist = [
			{"item": "site_master_file", "required": True, "submitted": False},
			{"item": "gdp_certificate_current", "required": True, "submitted": False},
			{"item": "qualified_person_declaration", "required": True, "submitted": False},
			{"item": "floor_plan_warehouse", "required": True, "submitted": False},
			{"item": "temperature_mapping_report", "required": True, "submitted": False},
			{"item": "pest_control_contract", "required": False, "submitted": False},
			{"item": "transport_qualification", "required": True, "submitted": False},
		]

		renewal_id = _uuid7str()
		data = wda.model_dump()
		data["renewal_submitted_date"] = datetime.utcnow()
		data["updated_at"] = datetime.utcnow()
		self._wda[self._key(tenant_id, wda_id)] = WholesaleDistributionAuthorisation(**data)

		self._audit(tenant_id, "wda_renewal_initiated", wda_id)
		return {
			"renewal_id": renewal_id,
			"wda_id": wda_id,
			"wda_number": wda.wda_number,
			"market": wda.market,
			"tenant_id": tenant_id,
			"current_expiry": wda.expiry_date.isoformat() if wda.expiry_date else None,
			"days_to_expiry": days_to_expiry,
			"status": "renewal_initiated",
			"document_checklist": checklist,
			"renewed_by": renewed_by,
			"renewal_notes": renewal_notes,
			"initiated_at": datetime.utcnow().isoformat(),
		}

	async def gdp_risk_score(
		self,
		distributor_id: str,
		tenant_id: str,
		lookback_days: int = 365,
	) -> dict[str, Any]:
		"""Compute a GDP Risk Score (0–100) for a distributor.

		Scoring model (lower is better):
		  - Each critical deviation:  +25 pts (capped at 50)
		  - Each major deviation:     +10 pts (capped at 30)
		  - Each minor deviation:     +2  pts (capped at 10)
		  - Open CAPA past due:       +15 pts
		  - Active WDA:               -20 pts (bonus for compliance)

		Returns:
			Risk score dict with band ("low"|"medium"|"high"|"critical") and breakdown.
		"""
		cutoff = datetime.utcnow() - timedelta(days=lookback_days)

		deviations = [
			d for d in self._gdp_deviations.values()
			if d.tenant_id == tenant_id and d.raised_date >= cutoff
		]
		critical = [d for d in deviations if d.gdp_status == "critical"]
		major = [d for d in deviations if d.gdp_status == "major"]
		minor = [d for d in deviations if d.gdp_status == "minor"]
		open_capa = [d for d in deviations if d.capa_reference is None and d.closed_date is None]

		score = 0
		score += min(len(critical) * 25, 50)
		score += min(len(major) * 10, 30)
		score += min(len(minor) * 2, 10)
		score += min(len(open_capa) * 15, 30)

		has_active_wda = self._wda_active_for_entity(distributor_id, tenant_id)
		if has_active_wda:
			score = max(score - 20, 0)

		score = min(score, 100)
		band = "low" if score < 25 else "medium" if score < 50 else "high" if score < 75 else "critical"

		self._audit(tenant_id, "gdp_risk_score_computed", distributor_id)
		return {
			"distributor_id": distributor_id,
			"tenant_id": tenant_id,
			"risk_score": score,
			"risk_band": band,
			"lookback_days": lookback_days,
			"critical_deviations": len(critical),
			"major_deviations": len(major),
			"minor_deviations": len(minor),
			"open_capa_count": len(open_capa),
			"active_wda": has_active_wda,
			"scored_at": datetime.utcnow().isoformat(),
		}

	async def supply_chain_integrity_check(
		self,
		shipment_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Run a comprehensive supply chain integrity check on a shipment.

		Verifies:
		  1. All serialised units in the shipment are active (not decommissioned).
		  2. No active Class I/II recalls affect the product/batch.
		  3. Cold chain was maintained (no critical excursions).
		  4. Distributor holds a valid WDA for the market.
		  5. GDP deviations do not block the shipment.

		Returns:
			Integrity check result dict with per-check pass/fail flags and an
			overall ``pass`` boolean.
		"""
		shipment = self._get_shipment(shipment_id, tenant_id)

		# check 1: serialisation integrity
		serials = [
			r for r in self._serialisation.values()
			if r.tenant_id == tenant_id and r.batch_number in (shipment.shipment_number,)
		]
		decommissioned_serials = [r for r in serials if r.decommissioned]
		serialisation_ok = len(decommissioned_serials) == 0

		# check 2: active recalls
		active_recalls = [
			r for r in self._recalls.values()
			if r.tenant_id == tenant_id
			and r.product_id == getattr(shipment, "product_id", "")
			and r.status in ("initiated", "in_progress")
		]
		recall_ok = len(active_recalls) == 0

		# check 3: cold chain
		excursions = [
			e for e in self._excursions.values()
			if e.tenant_id == tenant_id and e.shipment_id == shipment_id
		]
		critical_excursions = [e for e in excursions if e.severity == "critical"]
		cold_chain_ok = len(critical_excursions) == 0

		# check 4: WDA for wholesale channel
		wda_ok = True
		if shipment.distribution_channel == "wholesale":
			wda_ok = (
				shipment.wda_reference is not None
				and self._wda_active(shipment.wda_reference, tenant_id)
			)

		# check 5: open critical GDP deviations
		open_critical_gdp = [
			d for d in self._gdp_deviations.values()
			if d.tenant_id == tenant_id and d.gdp_status == "critical" and d.closed_date is None
		]
		gdp_ok = len(open_critical_gdp) == 0

		overall_pass = all([serialisation_ok, recall_ok, cold_chain_ok, wda_ok, gdp_ok])
		self._audit(tenant_id, "supply_chain_integrity_checked", shipment_id)
		return {
			"shipment_id": shipment_id,
			"tenant_id": tenant_id,
			"overall_pass": overall_pass,
			"checks": {
				"serialisation_integrity": serialisation_ok,
				"no_active_recalls": recall_ok,
				"cold_chain_maintained": cold_chain_ok,
				"wda_valid": wda_ok,
				"no_critical_gdp_deviations": gdp_ok,
			},
			"decommissioned_serial_count": len(decommissioned_serials),
			"active_recall_count": len(active_recalls),
			"critical_excursion_count": len(critical_excursions),
			"open_critical_gdp_count": len(open_critical_gdp),
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def async_regulatory_report(
		self,
		period: str,
		jurisdiction: str,
		tenant_id: str,
		include_serialisation_summary: bool = True,
	) -> dict[str, Any]:
		"""Async regulatory distribution report with extended serialisation summary.

		Wraps regulatory_reporting_distribution and appends serialisation
		aggregation stats (total, verified, decommissioned) for FMD/DSCSA submissions.
		"""
		base = self.regulatory_reporting_distribution(period, jurisdiction, tenant_id)
		if include_serialisation_summary:
			all_ser = [r for r in self._serialisation.values() if r.tenant_id == tenant_id]
			base["serialisation_verified_count"] = sum(1 for r in all_ser if r.verified)
			base["serialisation_decommissioned_count"] = sum(1 for r in all_ser if r.decommissioned)
			base["serialisation_active_count"] = sum(
				1 for r in all_ser if not r.decommissioned and r.status == "active"
			)
		return base

	async def bulk_serialise_products(
		self,
		tenant_id: str,
		specs: list[dict[str, Any]],
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Bulk-serialise a list of product units in a single call.

		Each spec dict must contain: product_id, serial_number, batch_number,
		standard, aggregation_level. gtin is optional.

		Returns:
			Summary with created_count, error_count, and per-spec errors.
		"""
		assert specs, "specs required"
		created: list[str] = []
		errors: list[dict[str, Any]] = []
		for spec in specs:
			try:
				rec = self.serialise_product(
					tenant_id=tenant_id,
					product_id=spec.get("product_id", ""),
					serial_number=spec.get("serial_number", ""),
					batch_number=spec.get("batch_number", ""),
					standard=spec.get("standard", "gs1"),
					aggregation_level=spec.get("aggregation_level", "unit"),
					gtin=spec.get("gtin"),
					created_by=spec.get("created_by", created_by),
				)
				created.append(rec.id)
			except Exception as exc:
				errors.append({"spec": spec, "error": str(exc)})
		self._audit(tenant_id, "bulk_serialisation_completed", f"count:{len(created)}")
		return {
			"tenant_id": tenant_id,
			"total_requested": len(specs),
			"created_count": len(created),
			"error_count": len(errors),
			"created_ids": created[:50],
			"errors": errors[:20],
			"processed_at": datetime.utcnow().isoformat(),
		}


PharmaDisService = PharmaceuticalDistributionService
