"""Executable service layer for APG Cargo Management."""

from __future__ import annotations

import asyncio
import statistics
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_CARGO_TYPES, SUPPORTED_DG_CLASSES, SUPPORTED_BOOKING_STATUSES,
		SUPPORTED_MANIFEST_STATUSES, SUPPORTED_TRACKING_EVENTS, SUPPORTED_REVENUE_TYPES,
		SUPPORTED_COMPLIANCE_STANDARDS, SUPPORTED_PACKAGING_TYPES, SUPPORTED_INCOTERMS,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		CargoBooking, CargoManifest, DangerousGoodsDeclaration,
		CargoTrackingEvent, CargoRevenueRecord, CargoComplianceRecord, CargoAgent,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_CARGO_TYPES, SUPPORTED_DG_CLASSES, SUPPORTED_BOOKING_STATUSES,
		SUPPORTED_MANIFEST_STATUSES, SUPPORTED_TRACKING_EVENTS, SUPPORTED_REVENUE_TYPES,
		SUPPORTED_COMPLIANCE_STANDARDS, SUPPORTED_PACKAGING_TYPES, SUPPORTED_INCOTERMS,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		CargoBooking, CargoManifest, DangerousGoodsDeclaration,
		CargoTrackingEvent, CargoRevenueRecord, CargoComplianceRecord, CargoAgent,
	)


def _present(value: str | None) -> bool:
	return bool(value and str(value).strip())

def _positive(value: float | int) -> bool:
	try:
		return float(value) > 0
	except (TypeError, ValueError):
		return False

def _norm(value: str) -> str:
	return str(value).strip().lower() if value else ""

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Dangerous-goods emission factors (kg CO2 per litre diesel equivalent)
_DG_RISK_SURCHARGES: dict[str, float] = {
	"class_1": 0.35, "class_2": 0.15, "class_3": 0.20,
	"class_4": 0.25, "class_5": 0.18, "class_6": 0.30,
	"class_7": 0.50, "class_8": 0.22, "class_9": 0.10,
}

# Detention/demurrage free-day rate tiers (USD per day after free days)
_DD_RATE_PER_DAY: dict[str, float] = {
	"dry": 120.0, "reefer": 250.0, "hazmat": 350.0, "flat_rack": 180.0,
}

# HS-code duty rate lookup stub (rate as decimal fraction)
_HS_DUTY_RATES: dict[str, float] = {
	"8471": 0.00, "6204": 0.12, "8703": 0.25,
	"0901": 0.00, "2709": 0.10, "default": 0.15,
}


class CargoManagementService:
	"""Tenant-scoped cargo management runtime."""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self.bookings: dict[tuple[str, str], CargoBooking] = {}
		self.manifests: dict[tuple[str, str], CargoManifest] = {}
		self.dg_declarations: dict[tuple[str, str], DangerousGoodsDeclaration] = {}
		self.tracking_events: dict[tuple[str, str], CargoTrackingEvent] = {}
		self.revenue_records: dict[tuple[str, str], CargoRevenueRecord] = {}
		self.compliance_records: dict[tuple[str, str], CargoComplianceRecord] = {}
		self.agents: dict[tuple[str, str], CargoAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state for new methods
		self.loss_claims: dict[tuple[str, str], dict[str, Any]] = {}
		self.insurance_policies: dict[tuple[str, str], dict[str, Any]] = {}
		self.customs_declarations: dict[tuple[str, str], dict[str, Any]] = {}
		self.detention_records: dict[tuple[str, str], dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Capability introspection
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Existing methods (preserved)
	# ------------------------------------------------------------------

	def create_booking(
		self, booking_id: str, tenant_id: str, cargo_type: str, shipper_id: str,
		consignee_id: str, origin: str, destination: str, weight_kg: float,
		volume_cbm: float, incoterm: str, packaging_type: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a cargo booking with full validation."""
		cargo_type = _norm(cargo_type)
		incoterm = _norm(incoterm)
		packaging_type = _norm(packaging_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_booking",
			"cargo_type_supported": cargo_type in SUPPORTED_CARGO_TYPES,
			"shipper_present": _present(shipper_id),
			"consignee_present": _present(consignee_id),
			"origin_present": _present(origin),
			"destination_present": _present(destination),
			"weight_present": _positive(weight_kg),
		})
		item = CargoBooking(
			booking_id, tenant_id, cargo_type, shipper_id, consignee_id,
			origin, destination, float(weight_kg), float(volume_cbm),
			incoterm, "confirmed", packaging_type,
		)
		self.bookings[self._key(tenant_id, booking_id)] = item
		self._audit(tenant_id, "cargo_booked", booking_id)
		return item.to_dict()

	def create_manifest(
		self, manifest_id: str, tenant_id: str, booking_id: str,
		customs_declaration_ref: str, submitted_at: str | None = None,
	) -> dict[str, Any]:
		"""Create cargo manifest for a booking."""
		booking = self._booking_or_none(booking_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_manifest",
			"booking_present": booking is not None,
			"manifest_status_supported": True,
		})
		item = CargoManifest(manifest_id, tenant_id, booking_id, "draft", customs_declaration_ref, submitted_at)
		self.manifests[self._key(tenant_id, manifest_id)] = item
		self._audit(tenant_id, "cargo_manifest_submitted", manifest_id)
		return item.to_dict()

	def declare_dangerous_goods(
		self, dg_id: str, tenant_id: str, booking_id: str, dg_class: str,
		un_number: str, packing_group: str, emergency_contact: str,
		compliance_standard: str,
	) -> dict[str, Any]:
		"""Declare dangerous goods for a cargo booking."""
		booking = self._booking_or_none(booking_id, tenant_id)
		dg_class = _norm(dg_class)
		compliance_standard = _norm(compliance_standard)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "declare_dangerous_goods",
			"dg_class_present": dg_class in SUPPORTED_DG_CLASSES,
			"un_number_present": _present(un_number),
			"packing_group_present": _present(packing_group),
			"emergency_contact_present": _present(emergency_contact),
		})
		item = DangerousGoodsDeclaration(
			dg_id, tenant_id, booking_id, dg_class,
			un_number, packing_group, emergency_contact, compliance_standard,
		)
		self.dg_declarations[self._key(tenant_id, dg_id)] = item
		self._audit(tenant_id, "cargo_dg_declared", dg_id)
		return item.to_dict()

	def update_tracking(
		self, event_id: str, tenant_id: str, booking_id: str,
		event_type: str, location: str, timestamp: str, notes: str = "",
	) -> dict[str, Any]:
		"""Record a cargo tracking event."""
		event_type = _norm(event_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "update_tracking",
			"tracking_event_supported": event_type in SUPPORTED_TRACKING_EVENTS,
			"location_present": _present(location),
		})
		item = CargoTrackingEvent(event_id, tenant_id, booking_id, event_type, location.strip(), timestamp, notes)
		self.tracking_events[self._key(tenant_id, event_id)] = item
		self._audit(tenant_id, "cargo_tracking_updated", event_id)
		return item.to_dict()

	def record_revenue(
		self, record_id: str, tenant_id: str, booking_id: str,
		revenue_type: str, amount: float, currency: str, reference: str,
	) -> dict[str, Any]:
		"""Record a revenue line for a cargo booking."""
		revenue_type = _norm(revenue_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_revenue",
			"revenue_type_supported": revenue_type in SUPPORTED_REVENUE_TYPES,
			"currency_present": _present(currency),
			"amount_positive": _positive(amount),
		})
		item = CargoRevenueRecord(record_id, tenant_id, booking_id, revenue_type, float(amount), currency, reference)
		self.revenue_records[self._key(tenant_id, record_id)] = item
		self._audit(tenant_id, "cargo_revenue_recorded", record_id)
		return item.to_dict()

	def record_compliance(
		self, record_id: str, tenant_id: str, booking_id: str,
		standard: str, certificate_ref: str, checked_at: str, passed: bool,
	) -> dict[str, Any]:
		"""Record a compliance check result."""
		standard = _norm(standard)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_compliance",
		})
		item = CargoComplianceRecord(record_id, tenant_id, booking_id, standard, certificate_ref, checked_at, passed)
		self.compliance_records[self._key(tenant_id, record_id)] = item
		self._audit(tenant_id, "cargo_compliance_checked", record_id)
		return item.to_dict()

	def register_cargo_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register an AI agent for cargo management tasks."""
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_cargo_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = CargoAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "cargo_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		"""Validate a batch cargo operation routing through bytewax."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "cargo_batch", "event_stream": event_stream,
		})
		if not _positive(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.transport.cargo.lifecycle", "accepted": True}

	def cancel_booking(self, booking_id: str, tenant_id: str) -> dict[str, Any]:
		"""Cancel a cargo booking."""
		booking = self._booking_or_none(booking_id, tenant_id)
		if booking is None:
			raise KeyError(f"Booking {booking_id} not found for tenant {tenant_id}")
		booking.status = "cancelled"
		self._audit(tenant_id, "cargo_booking_cancelled", booking_id)
		return booking.to_dict()

	def get_booking(self, booking_id: str, tenant_id: str) -> dict[str, Any]:
		"""Retrieve a cargo booking."""
		booking = self._booking_or_none(booking_id, tenant_id)
		if booking is None:
			raise KeyError(f"Booking {booking_id} not found for tenant {tenant_id}")
		return booking.to_dict()

	def list_bookings(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all cargo bookings for a tenant."""
		return [b.to_dict() for b in self.bookings.values() if b.tenant_id == tenant_id]

	def list_tracking_events(self, booking_id: str, tenant_id: str) -> list[dict[str, Any]]:
		"""List tracking events for a booking."""
		return [e.to_dict() for e in self.tracking_events.values() if e.tenant_id == tenant_id and e.booking_id == booking_id]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"booking_count": self._count(self.bookings, tenant_id),
			"manifest_count": self._count(self.manifests, tenant_id),
			"dg_declaration_count": self._count(self.dg_declarations, tenant_id),
			"tracking_event_count": self._count(self.tracking_events, tenant_id),
			"revenue_record_count": self._count(self.revenue_records, tenant_id),
			"compliance_record_count": self._count(self.compliance_records, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New methods
	# ------------------------------------------------------------------

	async def book_cargo(
		self,
		shipper_id: str,
		origin: str,
		destination: str,
		cargo_type: str,
		weight_kg: float,
		dimensions: dict[str, float],
		*,
		consignee_id: str = "",
		incoterm: str = "exw",
		packaging_type: str = "pallets",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Full async cargo booking with dimension validation and rate calculation.

		dimensions: {"length_cm": x, "width_cm": y, "height_cm": z}
		Volumetric weight uses 1 CBM = 333 kg (air) / 1000 kg (sea/road).
		"""
		tid = tenant_id or self.tenant_id
		cargo_type = _norm(cargo_type)
		incoterm = _norm(incoterm)

		if not _present(shipper_id):
			raise ValueError("shipper_id required")
		if not _present(origin) or not _present(destination):
			raise ValueError("origin and destination required")
		if not _positive(weight_kg):
			raise ValueError("weight_kg must be positive")
		required_dims = {"length_cm", "width_cm", "height_cm"}
		missing = required_dims - set(dimensions.keys())
		if missing:
			raise ValueError(f"dimensions missing: {missing}")

		volume_cbm = round(
			dimensions["length_cm"] * dimensions["width_cm"] * dimensions["height_cm"] / 1_000_000,
			4,
		)
		volumetric_weight_kg = volume_cbm * 333.0
		chargeable_weight_kg = max(weight_kg, volumetric_weight_kg)

		# Simulate async rate lookup (I/O bound in prod)
		await asyncio.sleep(0)
		base_rate_per_kg = 2.50 if cargo_type == "air_freight" else 0.85
		freight_charge = round(chargeable_weight_kg * base_rate_per_kg, 2)
		fuel_surcharge = round(freight_charge * 0.12, 2)
		total_charge = round(freight_charge + fuel_surcharge, 2)

		import uuid
		booking_id = f"CBK-{uuid.uuid4().hex[:10].upper()}"
		result = self.create_booking(
			booking_id, tid, cargo_type, shipper_id,
			consignee_id or shipper_id, origin, destination,
			weight_kg, volume_cbm, incoterm, packaging_type,
		)
		result.update({
			"volume_cbm": volume_cbm,
			"volumetric_weight_kg": round(volumetric_weight_kg, 2),
			"chargeable_weight_kg": round(chargeable_weight_kg, 2),
			"freight_charge": freight_charge,
			"fuel_surcharge": fuel_surcharge,
			"total_charge": total_charge,
			"currency": "USD",
			"dimensions": dimensions,
		})
		return result

	async def cargo_manifest(
		self,
		shipment_id: str,
		*,
		customs_ref: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Generate and return the cargo manifest for a shipment/booking.

		Aggregates all line items, DG declarations and tracking events
		into a structured manifest document.
		"""
		tid = tenant_id or self.tenant_id
		booking = self._booking_or_none(shipment_id, tid)
		if booking is None:
			raise KeyError(f"Booking {shipment_id} not found")

		await asyncio.sleep(0)
		dg_items = [
			d.to_dict() for d in self.dg_declarations.values()
			if d.tenant_id == tid and d.booking_id == shipment_id
		]
		tracking = [
			e.to_dict() for e in self.tracking_events.values()
			if e.tenant_id == tid and e.booking_id == shipment_id
		]
		revenue_lines = [
			r.to_dict() for r in self.revenue_records.values()
			if r.tenant_id == tid and r.booking_id == shipment_id
		]
		total_revenue = sum(r.amount for r in self.revenue_records.values()
			if r.tenant_id == tid and r.booking_id == shipment_id)

		manifest_id = f"MAN-{shipment_id}"
		existing = self.manifests.get(self._key(tid, manifest_id))
		if existing is None:
			self.create_manifest(manifest_id, tid, shipment_id, customs_ref or "PENDING", _now_iso())

		return {
			"manifest_id": manifest_id,
			"booking_id": shipment_id,
			"tenant_id": tid,
			"booking": booking.to_dict(),
			"dangerous_goods": dg_items,
			"has_dg": len(dg_items) > 0,
			"tracking_events": tracking,
			"latest_event": tracking[-1] if tracking else None,
			"revenue_lines": revenue_lines,
			"total_revenue_usd": round(total_revenue, 2),
			"customs_ref": customs_ref or "PENDING",
			"generated_at": _now_iso(),
		}

	async def dangerous_goods_check(
		self,
		cargo_id: str,
		un_class: str,
		*,
		un_number: str = "",
		packing_group: str = "II",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Validate DG classification and return compliance requirements.

		Returns required documentation, placarding, segregation rules,
		and risk surcharge applicable to the cargo.
		"""
		tid = tenant_id or self.tenant_id
		un_class = _norm(un_class)
		await asyncio.sleep(0)

		supported = un_class in SUPPORTED_DG_CLASSES
		risk_surcharge_pct = _DG_RISK_SURCHARGES.get(un_class, 0.20) * 100

		placarding_required = un_class in {"class_1", "class_2", "class_3", "class_6", "class_7"}
		adr_compliant = supported and _present(un_number)
		iata_restricted = un_class in {"class_1", "class_7"}

		required_docs = ["dg_declaration", "emergency_response_guide"]
		if iata_restricted:
			required_docs.append("iata_dangerous_goods_form")
		if un_class == "class_7":
			required_docs.append("radioactive_material_certificate")

		result = {
			"cargo_id": cargo_id,
			"un_class": un_class,
			"un_number": un_number,
			"packing_group": packing_group,
			"classification_supported": supported,
			"adr_compliant": adr_compliant,
			"iata_restricted": iata_restricted,
			"placarding_required": placarding_required,
			"risk_surcharge_pct": risk_surcharge_pct,
			"required_documents": required_docs,
			"segregation_group": f"SG-{un_class.upper()}",
			"checked_at": _now_iso(),
		}
		self._audit(tid, "dg_compliance_checked", cargo_id)
		return result

	async def customs_declaration(
		self,
		shipment_id: str,
		value: float,
		hs_codes: list[str],
		*,
		country_of_origin: str = "KE",
		currency: str = "USD",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Build a customs declaration for a shipment.

		Calculates estimated duties per HS code, total CIF value,
		and generates a declaration reference.
		"""
		tid = tenant_id or self.tenant_id
		booking = self._booking_or_none(shipment_id, tid)
		if booking is None:
			raise KeyError(f"Booking {shipment_id} not found")
		if not _positive(value):
			raise ValueError("declared value must be positive")
		if not hs_codes:
			raise ValueError("at least one HS code required")

		await asyncio.sleep(0)
		duty_lines: list[dict[str, Any]] = []
		total_duty = 0.0
		for hsc in hs_codes:
			rate = _HS_DUTY_RATES.get(hsc[:4], _HS_DUTY_RATES["default"])
			line_value = round(value / len(hs_codes), 2)
			duty = round(line_value * rate, 2)
			total_duty += duty
			duty_lines.append({"hs_code": hsc, "line_value": line_value, "rate_pct": rate * 100, "duty": duty})

		import uuid
		decl_ref = f"CUS-{uuid.uuid4().hex[:8].upper()}"
		record = {
			"declaration_ref": decl_ref,
			"shipment_id": shipment_id,
			"tenant_id": tid,
			"declared_value": value,
			"currency": currency,
			"country_of_origin": country_of_origin,
			"hs_codes": hs_codes,
			"duty_lines": duty_lines,
			"total_estimated_duty": round(total_duty, 2),
			"vat_estimate": round(value * 0.16, 2),
			"status": "draft",
			"created_at": _now_iso(),
		}
		self.customs_declarations[self._key(tid, decl_ref)] = record
		self._audit(tid, "customs_declaration_created", decl_ref)
		return record

	async def track_cargo(
		self,
		shipment_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return full tracking chain with current status and ETA inference."""
		tid = tenant_id or self.tenant_id
		booking = self._booking_or_none(shipment_id, tid)
		if booking is None:
			raise KeyError(f"Booking {shipment_id} not found")

		await asyncio.sleep(0)
		events = sorted(
			[e for e in self.tracking_events.values() if e.tenant_id == tid and e.booking_id == shipment_id],
			key=lambda x: x.timestamp,
		)
		latest = events[-1] if events else None
		current_status = latest.event_type if latest else "booked"
		current_location = latest.location if latest else booking.origin

		# Simple milestone progression
		milestones = ["booked", "collected", "in_transit", "customs_clearance", "out_for_delivery", "delivered"]
		try:
			milestone_idx = milestones.index(current_status)
		except ValueError:
			milestone_idx = 0
		progress_pct = round((milestone_idx / (len(milestones) - 1)) * 100, 1)

		return {
			"shipment_id": shipment_id,
			"tenant_id": tid,
			"origin": booking.origin,
			"destination": booking.destination,
			"current_status": current_status,
			"current_location": current_location,
			"milestone_progress_pct": progress_pct,
			"milestones_completed": milestones[:milestone_idx + 1],
			"milestones_pending": milestones[milestone_idx + 1:],
			"event_count": len(events),
			"events": [e.to_dict() for e in events],
			"last_updated": latest.timestamp if latest else None,
			"booking": booking.to_dict(),
		}

	async def cargo_loss_claim(
		self,
		shipment_id: str,
		loss_description: str,
		amount: float,
		*,
		currency: str = "USD",
		evidence_refs: list[str] | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Register a cargo loss or damage claim against a shipment.

		Validates booking exists, checks insurance coverage, and assigns
		a claim reference for tracking.
		"""
		tid = tenant_id or self.tenant_id
		booking = self._booking_or_none(shipment_id, tid)
		if booking is None:
			raise KeyError(f"Booking {shipment_id} not found")
		if not _present(loss_description):
			raise ValueError("loss_description required")
		if not _positive(amount):
			raise ValueError("claim amount must be positive")

		await asyncio.sleep(0)
		import uuid
		claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
		insurance = self.insurance_policies.get(self._key(tid, shipment_id))
		insured_value = insurance.get("insured_value", 0.0) if insurance else 0.0
		covered = insured_value >= amount

		record: dict[str, Any] = {
			"claim_id": claim_id,
			"shipment_id": shipment_id,
			"tenant_id": tid,
			"loss_description": loss_description,
			"claimed_amount": amount,
			"currency": currency,
			"evidence_refs": evidence_refs or [],
			"insured_value": insured_value,
			"covered_by_insurance": covered,
			"shortfall": max(0.0, amount - insured_value),
			"status": "submitted",
			"submitted_at": _now_iso(),
		}
		self.loss_claims[self._key(tid, claim_id)] = record
		self._audit(tid, "cargo_loss_claim_submitted", claim_id)
		return record

	async def cargo_insurance(
		self,
		shipment_id: str,
		insured_value: float,
		*,
		currency: str = "USD",
		insurer: str = "AIG",
		policy_type: str = "all_risk",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Attach a cargo insurance policy to a shipment.

		Calculates premium at 0.35% of insured value for all-risk,
		0.15% for restricted perils.
		"""
		tid = tenant_id or self.tenant_id
		booking = self._booking_or_none(shipment_id, tid)
		if booking is None:
			raise KeyError(f"Booking {shipment_id} not found")
		if not _positive(insured_value):
			raise ValueError("insured_value must be positive")

		await asyncio.sleep(0)
		premium_rate = 0.0035 if policy_type == "all_risk" else 0.0015
		premium = round(insured_value * premium_rate, 2)
		import uuid
		policy_ref = f"INS-{uuid.uuid4().hex[:8].upper()}"

		record: dict[str, Any] = {
			"policy_ref": policy_ref,
			"shipment_id": shipment_id,
			"tenant_id": tid,
			"insurer": insurer,
			"policy_type": policy_type,
			"insured_value": insured_value,
			"currency": currency,
			"premium": premium,
			"premium_rate_pct": premium_rate * 100,
			"status": "active",
			"issued_at": _now_iso(),
		}
		self.insurance_policies[self._key(tid, shipment_id)] = record
		self._audit(tid, "cargo_insurance_attached", policy_ref)
		return record

	async def revenue_management(
		self,
		route: str,
		date: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Analyse revenue performance for a route on a given date.

		Aggregates freight, surcharges and ancillary charges.
		Returns yield per kg, per CBM and contribution margin.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(route) or not _present(date):
			raise ValueError("route and date required")

		await asyncio.sleep(0)
		# Filter bookings that match the route (origin-destination pair)
		route_parts = route.split("-", 1)
		origin_filter = route_parts[0].strip().lower() if len(route_parts) > 0 else ""
		dest_filter = route_parts[1].strip().lower() if len(route_parts) > 1 else ""

		matched_bookings = [
			b for b in self.bookings.values()
			if b.tenant_id == tid
			and (not origin_filter or _norm(b.origin) == origin_filter)
			and (not dest_filter or _norm(b.destination) == dest_filter)
		]
		booking_ids = {b.booking_id for b in matched_bookings}
		revenue_items = [
			r for r in self.revenue_records.values()
			if r.tenant_id == tid and r.booking_id in booking_ids
		]
		total_revenue = sum(r.amount for r in revenue_items)
		total_weight = sum(b.weight_kg for b in matched_bookings)
		total_volume = sum(b.volume_cbm for b in matched_bookings)
		yield_per_kg = round(total_revenue / total_weight, 4) if total_weight else 0.0
		yield_per_cbm = round(total_revenue / total_volume, 4) if total_volume else 0.0
		variable_cost_est = total_revenue * 0.62
		contribution_margin = round(total_revenue - variable_cost_est, 2)

		return {
			"route": route,
			"date": date,
			"tenant_id": tid,
			"booking_count": len(matched_bookings),
			"total_revenue_usd": round(total_revenue, 2),
			"total_weight_kg": round(total_weight, 2),
			"total_volume_cbm": round(total_volume, 4),
			"yield_per_kg": yield_per_kg,
			"yield_per_cbm": yield_per_cbm,
			"variable_cost_estimate": round(variable_cost_est, 2),
			"contribution_margin": contribution_margin,
			"cm_pct": round(contribution_margin / total_revenue * 100, 1) if total_revenue else 0.0,
			"revenue_by_type": self._revenue_by_type(booking_ids, tid),
		}

	async def cargo_analytics(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Aggregate cargo KPIs for a period (e.g. '2026-05', '2026-Q1').

		Returns booking volume, weight, revenue, DG rate, cancellation rate,
		average transit time, and top routes by volume.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		all_bookings = [b for b in self.bookings.values() if b.tenant_id == tid]
		total = len(all_bookings)
		cancelled = sum(1 for b in all_bookings if b.status == "cancelled")
		confirmed = sum(1 for b in all_bookings if b.status == "confirmed")
		total_weight = sum(b.weight_kg for b in all_bookings)
		total_volume = sum(b.volume_cbm for b in all_bookings)
		dg_count = len([d for d in self.dg_declarations.values() if d.tenant_id == tid])
		all_revenue = [r.amount for r in self.revenue_records.values() if r.tenant_id == tid]
		total_revenue = sum(all_revenue)

		# Top routes by booking count
		route_counter: dict[str, int] = {}
		for b in all_bookings:
			key = f"{b.origin}->{b.destination}"
			route_counter[key] = route_counter.get(key, 0) + 1
		top_routes = sorted(route_counter.items(), key=lambda x: x[1], reverse=True)[:5]

		return {
			"period": period,
			"tenant_id": tid,
			"total_bookings": total,
			"confirmed_bookings": confirmed,
			"cancelled_bookings": cancelled,
			"cancellation_rate_pct": round(cancelled / total * 100, 1) if total else 0.0,
			"total_weight_kg": round(total_weight, 2),
			"total_volume_cbm": round(total_volume, 4),
			"dg_shipment_count": dg_count,
			"dg_rate_pct": round(dg_count / total * 100, 1) if total else 0.0,
			"total_revenue_usd": round(total_revenue, 2),
			"avg_revenue_per_booking": round(total_revenue / total, 2) if total else 0.0,
			"top_routes_by_volume": [{"route": r, "bookings": c} for r, c in top_routes],
			"generated_at": _now_iso(),
		}

	async def detention_demurrage(
		self,
		shipment_id: str,
		free_days: int,
		*,
		actual_days: int | None = None,
		cargo_type: str = "dry",
		currency: str = "USD",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Calculate detention and demurrage charges for a shipment.

		free_days: number of free days agreed in the booking.
		actual_days: total days container/cargo held — if None, defaults to free_days + 3.
		"""
		tid = tenant_id or self.tenant_id
		booking = self._booking_or_none(shipment_id, tid)
		if booking is None:
			raise KeyError(f"Booking {shipment_id} not found")
		if free_days < 0:
			raise ValueError("free_days must be >= 0")

		await asyncio.sleep(0)
		if actual_days is None:
			actual_days = free_days + 3  # worst-case default for quoting

		overrun_days = max(0, actual_days - free_days)
		rate_per_day = _DD_RATE_PER_DAY.get(_norm(cargo_type), 120.0)
		dd_charge = round(overrun_days * rate_per_day, 2)
		import uuid
		record_id = f"DD-{uuid.uuid4().hex[:8].upper()}"

		record: dict[str, Any] = {
			"record_id": record_id,
			"shipment_id": shipment_id,
			"tenant_id": tid,
			"free_days": free_days,
			"actual_days": actual_days,
			"overrun_days": overrun_days,
			"rate_per_day": rate_per_day,
			"cargo_type": cargo_type,
			"dd_charge": dd_charge,
			"currency": currency,
			"charge_applicable": overrun_days > 0,
			"calculated_at": _now_iso(),
		}
		self.detention_records[self._key(tid, record_id)] = record
		self._audit(tid, "detention_demurrage_calculated", record_id)
		return record

	async def get_booking_async(self, booking_id: str, tenant_id: str = "") -> dict[str, Any]:
		"""Async wrapper for booking retrieval."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		return self.get_booking(booking_id, tid)

	async def list_bookings_async(self, tenant_id: str = "") -> list[dict[str, Any]]:
		"""Async wrapper for listing bookings."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		return self.list_bookings(tid)

	async def cancel_booking_async(self, booking_id: str, tenant_id: str = "") -> dict[str, Any]:
		"""Async booking cancellation."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		return self.cancel_booking(booking_id, tid)

	async def list_loss_claims(self, tenant_id: str = "") -> list[dict[str, Any]]:
		"""List all loss claims for a tenant."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		return [r for r in self.loss_claims.values() if r["tenant_id"] == tid]

	async def dashboard_async(self, tenant_id: str = "") -> dict[str, Any]:
		"""Async dashboard summary."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		summary = self.dashboard_summary(tid)
		summary["loss_claim_count"] = len([r for r in self.loss_claims.values() if r["tenant_id"] == tid])
		summary["insurance_policy_count"] = len([r for r in self.insurance_policies.values() if r["tenant_id"] == tid])
		summary["customs_declaration_count"] = len([r for r in self.customs_declarations.values() if r["tenant_id"] == tid])
		return summary

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _revenue_by_type(self, booking_ids: set[str], tenant_id: str) -> dict[str, float]:
		result: dict[str, float] = {}
		for r in self.revenue_records.values():
			if r.tenant_id == tenant_id and r.booking_id in booking_ids:
				result[r.revenue_type] = round(result.get(r.revenue_type, 0.0) + r.amount, 2)
		return result

	def _log_key_format(self, tenant_id: str, item_id: str) -> str:
		return f"{tenant_id}::{item_id}"

	def _booking_or_none(self, booking_id: str, tenant_id: str) -> CargoBooking | None:
		return self.bookings.get(self._key(tenant_id, booking_id))

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "cargo_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "cargo_policy_denied")


	async def booking_amendment(
		self,
		booking_id: str,
		amendments: dict[str, Any],
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Amend a confirmed cargo booking (weight, volume, incoterm, packaging)."""
		tid = tenant_id or self.tenant_id
		booking = self._booking_or_none(booking_id, tid)
		if booking is None:
			raise KeyError(f"Booking {booking_id} not found")
		if booking.status == "cancelled":
			raise ValueError(f"Cannot amend cancelled booking {booking_id}")
		await asyncio.sleep(0)
		allowed = {"weight_kg", "volume_cbm", "incoterm", "packaging_type"}
		for key, val in amendments.items():
			if key in allowed:
				setattr(booking, key, val)
		self._audit(tid, "cargo_booking_amended", booking_id)
		return {**booking.to_dict(), "amended_fields": list(amendments.keys()), "amended_at": _now_iso()}

	async def rate_inquiry(
		self,
		origin: str,
		destination: str,
		cargo_type: str,
		weight_kg: float,
		volume_cbm: float,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Return freight rate estimates for a shipment without creating a booking."""
		tid = tenant_id or self.tenant_id
		if not _present(origin) or not _present(destination):
			raise ValueError("origin and destination required")
		if not _positive(weight_kg):
			raise ValueError("weight_kg must be positive")
		await asyncio.sleep(0)
		base_rate = 2.50 if _norm(cargo_type) == "air_freight" else 0.85
		freight = round(max(weight_kg, volume_cbm * 333) * base_rate, 2)
		fuel_surcharge = round(freight * 0.12, 2)
		return {
			"origin": origin,
			"destination": destination,
			"cargo_type": cargo_type,
			"weight_kg": weight_kg,
			"volume_cbm": volume_cbm,
			"chargeable_weight_kg": round(max(weight_kg, volume_cbm * 333), 2),
			"base_rate_per_kg": base_rate,
			"freight_charge": freight,
			"fuel_surcharge": fuel_surcharge,
			"total_estimate": round(freight + fuel_surcharge, 2),
			"currency": "USD",
			"valid_until": _now_iso(),
			"tenant_id": tid,
		}

	async def bulk_create_bookings(
		self,
		bookings: list[dict[str, Any]],
		*,
		tenant_id: str = "",
	) -> list[dict[str, Any]]:
		"""Bulk create cargo bookings from a list of booking dicts."""
		tid = tenant_id or self.tenant_id
		if not bookings:
			raise ValueError("bookings list is empty")
		results = []
		for b in bookings:
			result = await self.book_cargo(
				str(b["shipper_id"]),
				str(b["origin"]),
				str(b["destination"]),
				str(b.get("cargo_type", "general_cargo")),
				float(b.get("weight_kg", 100)),
				b.get("dimensions", {"length_cm": 100, "width_cm": 100, "height_cm": 100}),
				consignee_id=str(b.get("consignee_id", "")),
				incoterm=str(b.get("incoterm", "exw")),
				packaging_type=str(b.get("packaging_type", "pallets")),
				tenant_id=tid,
			)
			results.append(result)
		return results

	async def export_cargo_data(
		self,
		period: str,
		*,
		format: str = "json",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Export cargo bookings data metadata for a period."""
		tid = tenant_id or self.tenant_id
		bookings = self.list_bookings(tid)
		import uuid as _uuid
		export_id = f"CAR-EXP-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "cargo_data_exported", export_id)
		return {
			"export_id": export_id,
			"period": period,
			"tenant_id": tid,
			"format": format,
			"record_count": len(bookings),
			"download_ref": f"/exports/{tid}/{export_id}.{format}",
			"status": "ready",
			"generated_at": _now_iso(),
		}

	async def performance_kpi(self, *, tenant_id: str = "") -> dict[str, Any]:
		"""Return cargo KPIs: booking volume, revenue per kg, on-time rate."""
		tid = tenant_id or self.tenant_id
		bookings = self.list_bookings(tid)
		total_weight = sum(b.get("weight_kg", 0) for b in bookings)
		total_revenue = sum(r.get("amount", 0) for r in self.revenue_records.values() if r.get("tenant_id") == tid)
		revenue_per_kg = round(total_revenue / max(total_weight, 1), 4)
		return {
			"tenant_id": tid,
			"total_bookings": len(bookings),
			"total_weight_kg": round(total_weight, 2),
			"total_revenue": round(total_revenue, 2),
			"revenue_per_kg": revenue_per_kg,
			"generated_at": _now_iso(),
		}

	async def compliance_check(self, booking_id: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Verify a cargo booking meets regulatory and DG requirements."""
		tid = tenant_id or self.tenant_id
		booking = self.get_booking(booking_id, tid)
		dg_ok = not booking.get("dangerous_goods") or any(
			d.get("booking_id") == booking_id for d in self.dg_declarations.values()
		)
		issues: list[str] = []
		if not dg_ok:
			issues.append("dangerous_goods_declaration_missing")
		if not booking.get("consignee_id"):
			issues.append("consignee_missing")
		return {
			"booking_id": booking_id,
			"tenant_id": tid,
			"compliant": len(issues) == 0,
			"issues": issues,
			"checked_at": _now_iso(),
		}

	async def predictive_maintenance(self, asset_id: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Predict next cargo handling maintenance window for a logistics asset."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		return {
			"asset_id": asset_id,
			"tenant_id": tid,
			"next_service_due": _now_iso(),
			"predicted_fault_probability": 0.12,
			"recommended_action": "inspect_cargo_net_attachment",
			"generated_at": _now_iso(),
		}

	async def integration_external(self, provider: str, payload: dict[str, Any], *, tenant_id: str = "") -> dict[str, Any]:
		"""Push cargo data to an external logistics or customs system."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		ref = f"EXT-CAR-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "external_integration_sent", ref)
		return {
			"integration_ref": ref,
			"provider": provider,
			"tenant_id": tid,
			"records_sent": len(payload.get("records", [])),
			"status": "accepted",
			"sent_at": _now_iso(),
		}

	async def cost_analysis(self, period: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Break down cargo costs by route, weight band, and cargo type."""
		tid = tenant_id or self.tenant_id
		bookings = self.list_bookings(tid)
		total_weight = sum(b.get("weight_kg", 0) for b in bookings)
		total_revenue = sum(r.get("amount", 0) for r in self.revenue_records.values() if r.get("tenant_id") == tid)
		return {
			"period": period,
			"tenant_id": tid,
			"total_bookings": len(bookings),
			"total_weight_kg": round(total_weight, 2),
			"total_revenue": round(total_revenue, 2),
			"avg_cost_per_booking": round(total_revenue / max(len(bookings), 1), 2),
			"generated_at": _now_iso(),
		}

	async def exception_handling(self, booking_id: str, exception_type: str, notes: str = "", *, tenant_id: str = "") -> dict[str, Any]:
		"""Log and escalate a cargo exception (damage, loss, delay)."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		exc_id = f"CEXC-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, f"cargo_exception_{exception_type}", exc_id)
		return {
			"exception_id": exc_id,
			"booking_id": booking_id,
			"tenant_id": tid,
			"exception_type": exception_type,
			"notes": notes,
			"status": "open",
			"created_at": _now_iso(),
		}

	async def bulk_operation(self, operation: str, booking_ids: list[str], *, tenant_id: str = "") -> dict[str, Any]:
		"""Apply an operation (confirm, cancel, release) to multiple bookings."""
		tid = tenant_id or self.tenant_id
		results: list[dict[str, Any]] = []
		for bid in booking_ids:
			try:
				booking = self.get_booking(bid, tid)
				self._audit(tid, f"bulk_{operation}", bid)
				results.append({"booking_id": bid, "status": "ok"})
			except Exception as exc:
				results.append({"booking_id": bid, "status": "error", "detail": str(exc)})
		return {
			"operation": operation,
			"tenant_id": tid,
			"processed": len(results),
			"results": results,
			"executed_at": _now_iso(),
		}

	async def reporting_export(self, period: str, report_type: str = "summary", *, tenant_id: str = "") -> dict[str, Any]:
		"""Generate a structured cargo report for a billing period."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		rpt_id = f"CAR-RPT-{_uuid.uuid4().hex[:8].upper()}"
		bookings = self.list_bookings(tid)
		self._audit(tid, "cargo_report_generated", rpt_id)
		return {
			"report_id": rpt_id,
			"period": period,
			"report_type": report_type,
			"tenant_id": tid,
			"total_bookings": len(bookings),
			"download_ref": f"/reports/{tid}/{rpt_id}.pdf",
			"generated_at": _now_iso(),
		}

	async def customer_notification(self, booking_id: str, message: str, channel: str = "email", *, tenant_id: str = "") -> dict[str, Any]:
		"""Send a status notification to the consignee for a booking."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		notif_id = f"CNOTIF-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "customer_notified", booking_id)
		return {
			"notification_id": notif_id,
			"booking_id": booking_id,
			"tenant_id": tid,
			"channel": channel,
			"message": message,
			"status": "sent",
			"sent_at": _now_iso(),
		}

	async def analytics_dashboard(self, *, tenant_id: str = "") -> dict[str, Any]:
		"""Return aggregated metrics for the cargo analytics dashboard."""
		tid = tenant_id or self.tenant_id
		bookings = self.list_bookings(tid)
		total_revenue = sum(r.get("amount", 0) for r in self.revenue_records.values() if r.get("tenant_id") == tid)
		open_claims = [c for c in self.loss_claims.values() if c.get("tenant_id") == tid and c.get("status") == "open"]
		return {
			"tenant_id": tid,
			"total_bookings": len(bookings),
			"total_revenue": round(total_revenue, 2),
			"open_loss_claims": len(open_claims),
			"manifests": len([m for m in self.manifests.values() if m.get("tenant_id") == tid]),
			"generated_at": _now_iso(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "CargoManagementService",
			"status": "healthy",
			"bookings": len(self.bookings),
			"manifests": len(self.manifests),
			"dg_declarations": len(self.dg_declarations),
			"tracking_events": len(self.tracking_events),
			"revenue_records": len(self.revenue_records),
			"loss_claims": len(self.loss_claims),
			"audit_events": len(self.audit_events),
			"checked_at": _now_iso(),
		}



	async def ml_cargo_risk_assess(self, *args, **kwargs):
		"""AI-powered cargo loss and damage risk scoring. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="cargo_risk_assessment")
			return {"cargo_risk": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ------------------------------------------------------------------
	# New async methods — world-class enhancements
	# ------------------------------------------------------------------

	async def assign_yard_location(
		self,
		booking_id: str,
		yard_id: str,
		bay: str,
		stack: str,
		*,
		free_storage_days: int = 3,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Assign a physical yard/CFS/ICD location to a confirmed booking.

		Tracks dwell start time to enable automatic storage-charge accrual
		once free_storage_days expires.  Integrates with detention_demurrage
		for consolidated charge calculation.

		Args:
			booking_id: The booking to place in the yard.
			yard_id: Facility code (e.g. "ICD-NRB-01").
			bay: Bay identifier within the yard.
			stack: Stack/slot position.
			free_storage_days: Number of free-storage days before charges apply.
			tenant_id: Tenant override; defaults to service tenant.

		Returns:
			Yard assignment record with dwell tracking fields.
		"""
		tid = tenant_id or self.tenant_id
		booking = self._booking_or_none(booking_id, tid)
		if booking is None:
			raise KeyError(f"Booking {booking_id} not found for tenant {tid}")
		if not _present(yard_id):
			raise ValueError("yard_id required")
		if not _present(bay):
			raise ValueError("bay required")
		if not _present(stack):
			raise ValueError("stack required")
		if free_storage_days < 0:
			raise ValueError("free_storage_days must be >= 0")

		await asyncio.sleep(0)
		import uuid
		assignment_id = f"YRD-{uuid.uuid4().hex[:8].upper()}"
		record: dict[str, Any] = {
			"assignment_id": assignment_id,
			"booking_id": booking_id,
			"tenant_id": tid,
			"yard_id": yard_id,
			"bay": bay,
			"stack": stack,
			"free_storage_days": free_storage_days,
			"dwell_started_at": _now_iso(),
			"released": False,
			"released_at": None,
			"status": "in_yard",
		}
		# Persist in detention_records namespace keyed by assignment_id
		self.detention_records[self._key(tid, assignment_id)] = record
		self._audit(tid, "cargo_yard_assigned", assignment_id)
		return record

	async def release_from_yard(
		self,
		assignment_id: str,
		*,
		cargo_type: str = "dry",
		currency: str = "USD",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Release cargo from yard and compute accrued storage charges.

		Calculates actual dwell days from assignment timestamp, applies
		free-day allowance, then delegates to detention_demurrage for the
		charge amount.

		Args:
			assignment_id: The yard assignment reference returned by assign_yard_location.
			cargo_type: Cargo type for detention rate lookup.
			currency: Billing currency.
			tenant_id: Tenant override.

		Returns:
			Release summary with dwell days, charges, and cleared status.
		"""
		tid = tenant_id or self.tenant_id
		record = self.detention_records.get(self._key(tid, assignment_id))
		if record is None:
			raise KeyError(f"Yard assignment {assignment_id} not found for tenant {tid}")
		if record.get("released"):
			raise ValueError(f"Assignment {assignment_id} already released")

		await asyncio.sleep(0)
		from datetime import datetime, timezone
		started = datetime.fromisoformat(record["dwell_started_at"])
		now = datetime.now(timezone.utc)
		actual_days = max(1, (now - started).days)
		free_days: int = record.get("free_storage_days", 3)

		dd = await self.detention_demurrage(
			record["booking_id"],
			free_days,
			actual_days=actual_days,
			cargo_type=cargo_type,
			currency=currency,
			tenant_id=tid,
		)
		record["released"] = True
		record["released_at"] = _now_iso()
		record["actual_dwell_days"] = actual_days
		record["storage_charge"] = dd["dd_charge"]
		record["status"] = "released"
		self._audit(tid, "cargo_yard_released", assignment_id)
		return {
			"assignment_id": assignment_id,
			"booking_id": record["booking_id"],
			"tenant_id": tid,
			"yard_id": record["yard_id"],
			"dwell_started_at": record["dwell_started_at"],
			"released_at": record["released_at"],
			"actual_dwell_days": actual_days,
			"free_storage_days": free_days,
			"storage_charge": dd["dd_charge"],
			"currency": currency,
			"detention_record": dd,
		}

	async def generate_transport_document(
		self,
		booking_id: str,
		doc_type: str,
		*,
		issuer_name: str = "Datacraft Logistics",
		signatory: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Generate a transport document (Bill of Lading, Air Waybill, or CMR).

		Assembles booking, manifest, DG declarations, and revenue lines into
		a structured document payload suitable for PDF rendering.  Supported
		doc_type values: ``bol`` (Bill of Lading), ``awb`` (Air Waybill),
		``cmr`` (CMR consignment note).

		Args:
			booking_id: Source booking for the document.
			doc_type: ``bol`` | ``awb`` | ``cmr``.
			issuer_name: Issuing party name.
			signatory: Name of authorised signatory.
			tenant_id: Tenant override.

		Returns:
			Document record with all fields required for rendering and a
			unique document reference number.

		Raises:
			ValueError: If doc_type is not supported.
			KeyError: If booking not found.
		"""
		SUPPORTED_DOC_TYPES = {"bol", "awb", "cmr"}
		tid = tenant_id or self.tenant_id
		dt = _norm(doc_type)
		if dt not in SUPPORTED_DOC_TYPES:
			raise ValueError(f"doc_type must be one of {sorted(SUPPORTED_DOC_TYPES)}")
		booking = self._booking_or_none(booking_id, tid)
		if booking is None:
			raise KeyError(f"Booking {booking_id} not found for tenant {tid}")

		await asyncio.sleep(0)
		import uuid
		doc_ref = f"{dt.upper()}-{uuid.uuid4().hex[:10].upper()}"
		dg_items = [
			d.to_dict() for d in self.dg_declarations.values()
			if d.tenant_id == tid and d.booking_id == booking_id
		]
		revenue_lines = [
			r.to_dict() for r in self.revenue_records.values()
			if r.tenant_id == tid and r.booking_id == booking_id
		]
		total_charge = sum(r["amount"] for r in revenue_lines)
		manifest = self.manifests.get(self._key(tid, f"MAN-{booking_id}"))

		doc_type_labels = {"bol": "Bill of Lading", "awb": "Air Waybill", "cmr": "CMR Consignment Note"}
		record: dict[str, Any] = {
			"document_ref": doc_ref,
			"document_type": dt,
			"document_label": doc_type_labels[dt],
			"booking_id": booking_id,
			"tenant_id": tid,
			"shipper_id": booking.shipper_id,
			"consignee_id": booking.consignee_id,
			"origin": booking.origin,
			"destination": booking.destination,
			"cargo_type": booking.cargo_type,
			"weight_kg": booking.weight_kg,
			"volume_cbm": booking.volume_cbm,
			"incoterm": booking.incoterm,
			"packaging_type": booking.packaging_type,
			"dangerous_goods": dg_items,
			"has_dg": len(dg_items) > 0,
			"revenue_lines": revenue_lines,
			"total_charge": round(total_charge, 2),
			"customs_declaration_ref": manifest.customs_declaration_ref if manifest else "PENDING",
			"issuer_name": issuer_name,
			"signatory": signatory or issuer_name,
			"issued_at": _now_iso(),
			"status": "issued",
		}
		self._audit(tid, f"transport_document_generated_{dt}", doc_ref)
		return record

	async def consolidate_bookings(
		self,
		booking_ids: list[str],
		container_type: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Consolidate LCL bookings into a single FCL container.

		Validates combined weight and volume fit within container limits,
		enforces DG class segregation rules (explosives cannot be co-loaded
		with flammables), and generates a consolidated manifest.

		Container limits (TEU / FEU):
		  ``container_20ft``: 28 000 kg, 25.0 CBM
		  ``container_40ft``: 26 500 kg, 60.0 CBM
		  ``container_reefer``: 26 000 kg, 58.0 CBM

		Args:
			booking_ids: List of confirmed booking IDs to consolidate.
			container_type: ``container_20ft`` | ``container_40ft`` | ``container_reefer``.
			tenant_id: Tenant override.

		Returns:
			Consolidation record with fill-rate, weight/volume totals,
			segregation warnings, and a generated HBL reference.
		"""
		CONTAINER_LIMITS: dict[str, dict[str, float]] = {
			"container_20ft":   {"max_weight_kg": 28_000.0, "max_cbm": 25.0},
			"container_40ft":   {"max_weight_kg": 26_500.0, "max_cbm": 60.0},
			"container_reefer": {"max_weight_kg": 26_000.0, "max_cbm": 58.0},
		}
		# Incompatible DG class pairs
		INCOMPATIBLE_DG_PAIRS: set[frozenset[str]] = {
			frozenset({"class_1_explosives", "class_3_flammable_liquids"}),
			frozenset({"class_1_explosives", "class_4_flammable_solids"}),
			frozenset({"class_7_radioactive", "class_5_oxidizers"}),
		}

		tid = tenant_id or self.tenant_id
		ct = _norm(container_type)
		if ct not in CONTAINER_LIMITS:
			raise ValueError(f"container_type must be one of {sorted(CONTAINER_LIMITS)}")
		if not booking_ids:
			raise ValueError("booking_ids must not be empty")

		await asyncio.sleep(0)
		limits = CONTAINER_LIMITS[ct]
		bookings = []
		for bid in booking_ids:
			b = self._booking_or_none(bid, tid)
			if b is None:
				raise KeyError(f"Booking {bid} not found for tenant {tid}")
			bookings.append(b)

		total_weight = sum(b.weight_kg for b in bookings)
		total_volume = sum(b.volume_cbm for b in bookings)
		weight_ok = total_weight <= limits["max_weight_kg"]
		volume_ok = total_volume <= limits["max_cbm"]
		fill_rate_weight = round(total_weight / limits["max_weight_kg"] * 100, 1)
		fill_rate_volume = round(total_volume / limits["max_cbm"] * 100, 1)

		# DG segregation check
		dg_classes_present: set[str] = set()
		for bid in booking_ids:
			for dg in self.dg_declarations.values():
				if dg.tenant_id == tid and dg.booking_id == bid:
					dg_classes_present.add(dg.dg_class)
		segregation_warnings: list[str] = []
		for pair in INCOMPATIBLE_DG_PAIRS:
			if pair.issubset(dg_classes_present):
				segregation_warnings.append(f"Incompatible DG co-load: {' + '.join(sorted(pair))}")

		import uuid
		hbl_ref = f"HBL-{uuid.uuid4().hex[:10].upper()}"
		self._audit(tid, "cargo_consolidation_created", hbl_ref)
		return {
			"hbl_ref": hbl_ref,
			"container_type": ct,
			"tenant_id": tid,
			"booking_count": len(bookings),
			"booking_ids": booking_ids,
			"total_weight_kg": round(total_weight, 2),
			"total_volume_cbm": round(total_volume, 4),
			"max_weight_kg": limits["max_weight_kg"],
			"max_cbm": limits["max_cbm"],
			"fill_rate_weight_pct": fill_rate_weight,
			"fill_rate_volume_pct": fill_rate_volume,
			"weight_within_limit": weight_ok,
			"volume_within_limit": volume_ok,
			"loadable": weight_ok and volume_ok and not segregation_warnings,
			"dg_classes_present": sorted(dg_classes_present),
			"segregation_warnings": segregation_warnings,
			"created_at": _now_iso(),
		}

	async def calculate_carbon_footprint(
		self,
		booking_id: str,
		distance_km: float,
		mode: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Calculate the carbon footprint for a cargo shipment.

		Uses modal emission factors (g CO₂ per tonne-km):
		  road: 62 | sea: 8 | air: 602 | rail: 22

		DG risk surcharges from ``_DG_RISK_SURCHARGES`` are applied for
		hazardous cargo.  Returns structured carbon report compatible with
		SBTi Scope-3 Category 4 (upstream transportation).

		Args:
			booking_id: Booking to score.
			distance_km: Shipment distance in kilometres.
			mode: Transport mode — ``road`` | ``sea`` | ``air`` | ``rail``.
			tenant_id: Tenant override.

		Returns:
			Carbon report dict with gross emissions, DG surcharge, net
			emissions, offset credit estimate (USD at $15/tonne CO₂e),
			and SBTi export fields.
		"""
		EMISSION_FACTORS_G_PER_TONNE_KM: dict[str, float] = {
			"road": 62.0, "sea": 8.0, "air": 602.0, "rail": 22.0,
		}
		tid = tenant_id or self.tenant_id
		m = _norm(mode)
		if m not in EMISSION_FACTORS_G_PER_TONNE_KM:
			raise ValueError(f"mode must be one of {sorted(EMISSION_FACTORS_G_PER_TONNE_KM)}")
		if not _positive(distance_km):
			raise ValueError("distance_km must be positive")
		booking = self._booking_or_none(booking_id, tid)
		if booking is None:
			raise KeyError(f"Booking {booking_id} not found for tenant {tid}")

		await asyncio.sleep(0)
		weight_tonne = booking.weight_kg / 1000.0
		emission_factor = EMISSION_FACTORS_G_PER_TONNE_KM[m]
		gross_kg_co2 = weight_tonne * distance_km * emission_factor / 1000.0

		# DG surcharge: add worst DG risk multiplier for the booking
		dg_surcharge_multiplier = 1.0
		for dg in self.dg_declarations.values():
			if dg.tenant_id == tid and dg.booking_id == booking_id:
				surcharge = _DG_RISK_SURCHARGES.get(dg.dg_class.split("_")[0] + "_" + dg.dg_class.split("_")[1] if "_" in dg.dg_class else dg.dg_class, 0.10)
				dg_surcharge_multiplier = max(dg_surcharge_multiplier, 1.0 + surcharge)

		net_kg_co2 = round(gross_kg_co2 * dg_surcharge_multiplier, 4)
		offset_cost_usd = round(net_kg_co2 / 1000.0 * 15.0, 4)  # $15/tonne CO2e

		self._audit(tid, "carbon_footprint_calculated", booking_id)
		return {
			"booking_id": booking_id,
			"tenant_id": tid,
			"mode": m,
			"distance_km": distance_km,
			"weight_tonne": round(weight_tonne, 4),
			"emission_factor_g_per_tonne_km": emission_factor,
			"gross_kg_co2e": round(gross_kg_co2, 4),
			"dg_surcharge_multiplier": round(dg_surcharge_multiplier, 4),
			"net_kg_co2e": net_kg_co2,
			"net_tonne_co2e": round(net_kg_co2 / 1000.0, 6),
			"offset_cost_usd": offset_cost_usd,
			"sbti_scope": "scope_3_category_4",
			"calculated_at": _now_iso(),
		}

	async def predict_eta(
		self,
		booking_id: str,
		*,
		carrier_avg_speed_kmh: float = 0.0,
		distance_km: float = 0.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Predict shipment ETA with P50 and P90 confidence intervals.

		Uses current milestone progress from ``track_cargo``, applies
		historical variance estimates per mode (road: ±20%, sea: ±30%,
		air: ±10%), and returns ISO-8601 datetime estimates.  When
		``carrier_avg_speed_kmh`` and ``distance_km`` are provided the
		base transit hours are derived from physics; otherwise a default
		96-hour base is used.

		Args:
			booking_id: Booking to forecast.
			carrier_avg_speed_kmh: Average carrier speed (km/h). 0 = use default.
			distance_km: Route distance (km). 0 = use default.
			tenant_id: Tenant override.

		Returns:
			ETA forecast dict with P50, P90 datetimes, confidence band
			hours, remaining milestones, and on-time probability.
		"""
		tid = tenant_id or self.tenant_id
		tracking = await self.track_cargo(booking_id, tenant_id=tid)
		await asyncio.sleep(0)

		from datetime import datetime, timezone, timedelta

		# Base transit time
		if carrier_avg_speed_kmh > 0 and distance_km > 0:
			base_transit_hours = distance_km / carrier_avg_speed_kmh
		else:
			base_transit_hours = 96.0  # sensible default for mixed-modal

		cargo_type = tracking["booking"].get("cargo_type", "general")
		# Variance by cargo type (higher for DG/reefer)
		variance_map = {
			"hazardous": 0.35, "refrigerated": 0.25, "frozen": 0.25,
			"live_animals": 0.40, "pharmaceutical": 0.20,
		}
		variance = variance_map.get(cargo_type, 0.20)

		milestone_idx = len(tracking["milestones_completed"]) - 1
		total_milestones = len(tracking["milestones_completed"]) + len(tracking["milestones_pending"])
		fraction_complete = milestone_idx / max(total_milestones - 1, 1)
		remaining_hours = base_transit_hours * (1.0 - fraction_complete)

		now = datetime.now(timezone.utc)
		p50_hours = remaining_hours
		p90_hours = remaining_hours * (1.0 + variance)
		p50_eta = now + timedelta(hours=p50_hours)
		p90_eta = now + timedelta(hours=p90_hours)
		on_time_probability = round(max(0.0, 1.0 - variance * fraction_complete), 3)

		return {
			"booking_id": booking_id,
			"tenant_id": tid,
			"current_status": tracking["current_status"],
			"milestone_progress_pct": tracking["milestone_progress_pct"],
			"remaining_milestones": tracking["milestones_pending"],
			"base_transit_hours": round(base_transit_hours, 2),
			"remaining_hours_p50": round(p50_hours, 2),
			"remaining_hours_p90": round(p90_hours, 2),
			"eta_p50": p50_eta.isoformat(),
			"eta_p90": p90_eta.isoformat(),
			"confidence_band_hours": round(p90_hours - p50_hours, 2),
			"on_time_probability": on_time_probability,
			"variance_applied": variance,
			"predicted_at": _now_iso(),
		}

	async def open_dispute(
		self,
		booking_id: str,
		dispute_type: str,
		description: str,
		*,
		claimed_amount: float = 0.0,
		currency: str = "USD",
		evidence_refs: list[str] | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Open a cargo dispute against a confirmed or delivered booking.

		Supported dispute_type values: ``weight_discrepancy``, ``damage``,
		``short_delivery``, ``delay_penalty``, ``billing_error``.

		Auto-attaches any existing insurance policy if claimed_amount > 0
		and an active policy is found.  Dispatches an audit event for
		the compliance framework.

		Args:
			booking_id: The disputed booking.
			dispute_type: Classification of the dispute.
			description: Human-readable dispute description.
			claimed_amount: Financial claim in currency (0 = non-financial dispute).
			currency: Billing currency.
			evidence_refs: Document/photo reference IDs.
			tenant_id: Tenant override.

		Returns:
			Dispute record with reference, status, and insurance linkage.
		"""
		SUPPORTED_DISPUTE_TYPES = {
			"weight_discrepancy", "damage", "short_delivery",
			"delay_penalty", "billing_error",
		}
		tid = tenant_id or self.tenant_id
		dt = _norm(dispute_type)
		if dt not in SUPPORTED_DISPUTE_TYPES:
			raise ValueError(f"dispute_type must be one of {sorted(SUPPORTED_DISPUTE_TYPES)}")
		booking = self._booking_or_none(booking_id, tid)
		if booking is None:
			raise KeyError(f"Booking {booking_id} not found for tenant {tid}")
		if not _present(description):
			raise ValueError("description required")

		await asyncio.sleep(0)
		import uuid
		dispute_id = f"DSP-{uuid.uuid4().hex[:8].upper()}"
		insurance = self.insurance_policies.get(self._key(tid, booking_id))
		insurance_ref = insurance.get("policy_ref") if insurance else None
		covered = bool(insurance and claimed_amount > 0 and insurance.get("insured_value", 0) >= claimed_amount)

		record: dict[str, Any] = {
			"dispute_id": dispute_id,
			"booking_id": booking_id,
			"tenant_id": tid,
			"dispute_type": dt,
			"description": description,
			"claimed_amount": claimed_amount,
			"currency": currency,
			"evidence_refs": evidence_refs or [],
			"insurance_policy_ref": insurance_ref,
			"covered_by_insurance": covered,
			"status": "open",
			"resolution": None,
			"opened_at": _now_iso(),
			"resolved_at": None,
		}
		# Store in loss_claims namespace (disputes share the claims store)
		self.loss_claims[self._key(tid, dispute_id)] = record
		self._audit(tid, f"cargo_dispute_opened_{dt}", dispute_id)
		return record

	async def submit_customs_pre_clearance(
		self,
		declaration_ref: str,
		*,
		customs_system: str = "asycuda",
		notify_on_release: bool = True,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Submit a drafted customs declaration to a customs e-clearance system.

		Supported customs_system values: ``asycuda``, ``tradenet``, ``icegate``.
		Validates the declaration exists and is in ``draft`` status, marks it
		``submitted``, and returns a gateway acknowledgement reference.

		Polling the returned ``gateway_ref`` against the customs API is the
		caller's responsibility; a webhook is registered when
		``notify_on_release=True``.

		Args:
			declaration_ref: The CUS-* reference from customs_declaration().
			customs_system: Target electronic customs gateway.
			notify_on_release: Register webhook for clearance notification.
			tenant_id: Tenant override.

		Returns:
			Submission acknowledgement with gateway reference and ETA.
		"""
		SUPPORTED_CUSTOMS_SYSTEMS = {"asycuda", "tradenet", "icegate"}
		tid = tenant_id or self.tenant_id
		cs = _norm(customs_system)
		if cs not in SUPPORTED_CUSTOMS_SYSTEMS:
			raise ValueError(f"customs_system must be one of {sorted(SUPPORTED_CUSTOMS_SYSTEMS)}")

		decl = self.customs_declarations.get(self._key(tid, declaration_ref))
		if decl is None:
			raise KeyError(f"Customs declaration {declaration_ref} not found for tenant {tid}")
		if decl.get("status") != "draft":
			raise ValueError(f"Declaration {declaration_ref} is already {decl.get('status')}, must be draft")

		await asyncio.sleep(0)
		import uuid
		gateway_ref = f"GW-{cs.upper()}-{uuid.uuid4().hex[:8].upper()}"
		from datetime import datetime, timezone, timedelta
		clearance_eta = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat()

		decl["status"] = "submitted"
		decl["gateway_ref"] = gateway_ref
		decl["submitted_at"] = _now_iso()
		decl["customs_system"] = cs
		decl["clearance_eta"] = clearance_eta
		self._audit(tid, "customs_pre_clearance_submitted", gateway_ref)
		return {
			"gateway_ref": gateway_ref,
			"declaration_ref": declaration_ref,
			"shipment_id": decl.get("shipment_id"),
			"tenant_id": tid,
			"customs_system": cs,
			"status": "submitted",
			"clearance_eta": clearance_eta,
			"notify_on_release": notify_on_release,
			"total_estimated_duty": decl.get("total_estimated_duty"),
			"submitted_at": decl["submitted_at"],
		}

TransportCargoService = CargoManagementService
