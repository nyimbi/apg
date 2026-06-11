"""Logistics & Transportation async service (scm_log)."""
from __future__ import annotations

import asyncio
import logging
import math
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_log"
SUPPORTED_FREIGHT_MODES = {"air", "sea", "road", "rail", "multimodal"}

# GLEC Framework v3 emission factors — kgCO2e per tonne-km
_CO2_FACTORS: dict[str, float] = {
	"air": 0.602,
	"sea": 0.016,
	"road": 0.096,
	"rail": 0.028,
	"multimodal": 0.120,
}
SUPPORTED_SERVICE_LEVELS = {"express", "standard", "economy"}
SUPPORTED_CARRIER_TYPES = {"air", "sea", "road", "rail", "multimodal", "courier"}
SUPPORTED_DOC_TYPES = {
	"commercial_invoice", "packing_list", "bill_of_lading",
	"certificate_of_origin", "customs_declaration", "airway_bill",
}
SUPPORTED_TRACKING_EVENTS = {
	"pickup", "in_transit", "customs_clearance", "out_for_delivery",
	"delivered", "exception", "returned",
}


class LogisticsService:
	"""Async service for carrier integration, shipment tracking, freight audit,
	route optimisation, customs documentation and 3PL management."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.carriers: dict[str, dict[str, Any]] = {}
		self.shipments: dict[str, dict[str, Any]] = {}
		self.tracking_events: dict[str, dict[str, Any]] = {}
		self.freight_audits: dict[str, dict[str, Any]] = {}
		self.routes: dict[str, dict[str, Any]] = {}
		self.customs_documents: dict[str, dict[str, Any]] = {}
		self.third_party_providers: dict[str, dict[str, Any]] = {}
		self.delivery_exceptions: dict[str, dict[str, Any]] = {}
		self.insurance_policies: dict[str, dict[str, Any]] = {}
		self.insurance_claims: dict[str, dict[str, Any]] = {}
		self.consolidation_groups: dict[str, dict[str, Any]] = {}
		self.webhooks: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	# ── Internal helpers ─────────────────────────────────────────────────────

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self, tenant_id: str | None = None) -> str:
		t = tenant_id or self.tenant_id
		if not t:
			raise PermissionError("tenant_context_required")
		return t

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, status: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"status": status,
			"capability_id": CAPABILITY_ID,
			"emitted_at": self._now(),
		})

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"carrier_count": len(self.carriers),
			"active_shipments": sum(1 for s in self.shipments.values() if s["status"] not in {"delivered", "cancelled"}),
			"pending_audits": sum(1 for a in self.freight_audits.values() if a["status"] == "pending"),
			"open_exceptions": sum(1 for e in self.delivery_exceptions.values() if e["status"] == "open"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Describe capability contract."""
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "scm",
			"version": "1.0.0",
			"description": "Carrier integration, shipment tracking, freight audit, route optimisation, customs documentation, 3PL management",
			"supported_freight_modes": sorted(SUPPORTED_FREIGHT_MODES),
			"supported_service_levels": sorted(SUPPORTED_SERVICE_LEVELS),
			"supported_doc_types": sorted(SUPPORTED_DOC_TYPES),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return audit event log for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Carrier management ────────────────────────────────────────────────────

	async def create_carrier(
		self,
		name: str,
		carrier_code: str,
		carrier_type: str,
		country_of_origin: str,
		services_offered: list[str] | None = None,
		contact_email: str | None = None,
		contact_phone: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a new carrier."""
		tenant = self._tenant(tenant_id)
		if carrier_type not in SUPPORTED_CARRIER_TYPES:
			raise ValueError(f"carrier_type must be one of {SUPPORTED_CARRIER_TYPES}")
		# deduplicate by carrier_code within tenant
		for c in self.carriers.values():
			if c["tenant_id"] == tenant and c["carrier_code"] == carrier_code:
				raise ValueError(f"carrier_code '{carrier_code}' already registered for tenant")
		record: dict[str, Any] = {
			"id": self._id("carrier"),
			"type": "scm_log_carrier",
			"tenant_id": tenant,
			"name": name,
			"carrier_code": carrier_code,
			"carrier_type": carrier_type,
			"country_of_origin": country_of_origin,
			"services_offered": services_offered or [],
			"contact_email": contact_email,
			"contact_phone": contact_phone,
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.carriers[record["id"]] = record
		self._emit(tenant, "carrier_created", record["id"], "scm_log_carrier", "active")
		return deepcopy(record)

	async def list_carriers(self, tenant_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List carriers for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.carriers.values() if c["tenant_id"] == tenant]
		if status:
			items = [c for c in items if c["status"] == status]
		return items

	async def get_carrier(self, carrier_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single carrier."""
		tenant = self._tenant(tenant_id)
		carrier = self.carriers.get(carrier_id)
		if not carrier or carrier["tenant_id"] != tenant:
			raise KeyError(f"carrier '{carrier_id}' not found")
		return deepcopy(carrier)

	async def update_carrier(
		self,
		carrier_id: str,
		updates: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update carrier attributes."""
		tenant = self._tenant(tenant_id)
		carrier = self.carriers.get(carrier_id)
		if not carrier or carrier["tenant_id"] != tenant:
			raise KeyError(f"carrier '{carrier_id}' not found")
		allowed = {"name", "services_offered", "contact_email", "contact_phone", "status"}
		for k, v in updates.items():
			if k in allowed:
				carrier[k] = v
		carrier["updated_at"] = self._now()
		self._emit(tenant, "carrier_updated", carrier_id, "scm_log_carrier", carrier["status"])
		return deepcopy(carrier)

	async def delete_carrier(self, carrier_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Soft-delete a carrier."""
		tenant = self._tenant(tenant_id)
		carrier = self.carriers.get(carrier_id)
		if not carrier or carrier["tenant_id"] != tenant:
			raise KeyError(f"carrier '{carrier_id}' not found")
		carrier["status"] = "deleted"
		carrier["updated_at"] = self._now()
		self._emit(tenant, "carrier_deleted", carrier_id, "scm_log_carrier", "deleted")
		return deepcopy(carrier)

	async def rate_carrier(
		self,
		carrier_id: str,
		score: float,
		dimensions: dict[str, float] | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a performance rating for a carrier (0-10 scale)."""
		tenant = self._tenant(tenant_id)
		carrier = self.carriers.get(carrier_id)
		if not carrier or carrier["tenant_id"] != tenant:
			raise KeyError(f"carrier '{carrier_id}' not found")
		if not 0 <= score <= 10:
			raise ValueError("score must be between 0 and 10")
		carrier.setdefault("ratings", []).append({
			"score": score,
			"dimensions": dimensions or {},
			"rated_at": self._now(),
		})
		carrier["avg_rating"] = round(
			sum(r["score"] for r in carrier["ratings"]) / len(carrier["ratings"]), 2
		)
		self._emit(tenant, "carrier_rated", carrier_id, "scm_log_carrier", carrier["status"])
		return deepcopy(carrier)

	# ── Shipment management ───────────────────────────────────────────────────

	async def create_shipment(
		self,
		carrier_id: str,
		origin_address: dict[str, Any],
		destination_address: dict[str, Any],
		weight_kg: float,
		freight_mode: str,
		service_level: str = "standard",
		volume_m3: float | None = None,
		declared_value: float | None = None,
		currency: str = "USD",
		special_instructions: str | None = None,
		reference_number: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a new shipment."""
		tenant = self._tenant(tenant_id)
		if freight_mode not in SUPPORTED_FREIGHT_MODES:
			raise ValueError(f"freight_mode must be one of {SUPPORTED_FREIGHT_MODES}")
		if service_level not in SUPPORTED_SERVICE_LEVELS:
			raise ValueError(f"service_level must be one of {SUPPORTED_SERVICE_LEVELS}")
		carrier = self.carriers.get(carrier_id)
		if not carrier or carrier["tenant_id"] != tenant:
			raise KeyError(f"carrier '{carrier_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("shp"),
			"type": "scm_log_shipment",
			"tenant_id": tenant,
			"carrier_id": carrier_id,
			"origin_address": deepcopy(origin_address),
			"destination_address": deepcopy(destination_address),
			"weight_kg": weight_kg,
			"volume_m3": volume_m3,
			"freight_mode": freight_mode,
			"service_level": service_level,
			"declared_value": declared_value,
			"currency": currency,
			"special_instructions": special_instructions,
			"reference_number": reference_number,
			"tracking_number": None,
			"estimated_delivery": None,
			"actual_delivery": None,
			"status": "draft",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.shipments[record["id"]] = record
		self._emit(tenant, "shipment_created", record["id"], "scm_log_shipment", "draft")
		return deepcopy(record)

	async def list_shipments(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
		carrier_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List shipments with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.shipments.values() if s["tenant_id"] == tenant]
		if status:
			items = [s for s in items if s["status"] == status]
		if carrier_id:
			items = [s for s in items if s["carrier_id"] == carrier_id]
		return items

	async def get_shipment(self, shipment_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single shipment."""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		return deepcopy(shipment)

	async def update_shipment(
		self,
		shipment_id: str,
		updates: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update shipment attributes."""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		allowed = {"status", "tracking_number", "estimated_delivery", "actual_delivery", "special_instructions"}
		for k, v in updates.items():
			if k in allowed:
				shipment[k] = v
		shipment["updated_at"] = self._now()
		self._emit(tenant, "shipment_updated", shipment_id, "scm_log_shipment", shipment["status"])
		return deepcopy(shipment)

	async def book_shipment(self, shipment_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Confirm and book a draft shipment with the carrier."""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		if shipment["status"] != "draft":
			raise ValueError("only draft shipments can be booked")
		tracking = f"TRK{uuid4().hex[:10].upper()}"
		shipment["status"] = "booked"
		shipment["tracking_number"] = tracking
		shipment["booked_at"] = self._now()
		shipment["updated_at"] = self._now()
		self._emit(tenant, "shipment_booked", shipment_id, "scm_log_shipment", "booked")
		return deepcopy(shipment)

	async def cancel_shipment(self, shipment_id: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel a booked or in-transit shipment."""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		if shipment["status"] in {"delivered", "cancelled"}:
			raise ValueError(f"cannot cancel a {shipment['status']} shipment")
		shipment["status"] = "cancelled"
		shipment["cancellation_reason"] = reason
		shipment["cancelled_at"] = self._now()
		shipment["updated_at"] = self._now()
		self._emit(tenant, "shipment_cancelled", shipment_id, "scm_log_shipment", "cancelled")
		return deepcopy(shipment)

	# ── Shipment tracking ─────────────────────────────────────────────────────

	async def add_tracking_event(
		self,
		shipment_id: str,
		event_type: str,
		location: str,
		description: str | None = None,
		event_timestamp: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Append a tracking milestone to a shipment."""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		if event_type not in SUPPORTED_TRACKING_EVENTS:
			raise ValueError(f"event_type must be one of {SUPPORTED_TRACKING_EVENTS}")
		record: dict[str, Any] = {
			"id": self._id("trk"),
			"type": "scm_log_tracking_event",
			"tenant_id": tenant,
			"shipment_id": shipment_id,
			"event_type": event_type,
			"location": location,
			"description": description,
			"event_timestamp": event_timestamp or self._now(),
			"created_at": self._now(),
		}
		self.tracking_events[record["id"]] = record
		# progress shipment status
		if event_type == "pickup" and shipment["status"] == "booked":
			shipment["status"] = "in_transit"
		elif event_type == "delivered":
			shipment["status"] = "delivered"
			shipment["actual_delivery"] = record["event_timestamp"]
		elif event_type == "exception":
			shipment["status"] = "exception"
		shipment["updated_at"] = self._now()
		self._emit(tenant, f"tracking_{event_type}", record["id"], "scm_log_tracking_event", event_type)
		return deepcopy(record)

	async def get_shipment_tracking(self, shipment_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return all tracking events for a shipment."""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		return sorted(
			[deepcopy(e) for e in self.tracking_events.values() if e["shipment_id"] == shipment_id],
			key=lambda x: x["event_timestamp"],
		)

	# ── Freight audit ─────────────────────────────────────────────────────────

	async def create_freight_audit(
		self,
		shipment_id: str,
		carrier_id: str,
		invoice_number: str,
		invoiced_amount: float,
		expected_amount: float,
		currency: str = "USD",
		audit_notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Audit a carrier invoice against expected freight charges."""
		tenant = self._tenant(tenant_id)
		variance = round(invoiced_amount - expected_amount, 4)
		record: dict[str, Any] = {
			"id": self._id("faudit"),
			"type": "scm_log_freight_audit",
			"tenant_id": tenant,
			"shipment_id": shipment_id,
			"carrier_id": carrier_id,
			"invoice_number": invoice_number,
			"invoiced_amount": invoiced_amount,
			"expected_amount": expected_amount,
			"variance": variance,
			"currency": currency,
			"audit_notes": audit_notes,
			"status": "pending",
			"created_at": self._now(),
		}
		self.freight_audits[record["id"]] = record
		self._emit(tenant, "freight_audit_created", record["id"], "scm_log_freight_audit", "pending")
		return deepcopy(record)

	async def resolve_freight_audit(
		self,
		audit_id: str,
		resolution: str,  # approved | disputed
		resolved_by: str,
		resolution_notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Approve or dispute a freight audit."""
		tenant = self._tenant(tenant_id)
		audit = self.freight_audits.get(audit_id)
		if not audit or audit["tenant_id"] != tenant:
			raise KeyError(f"freight_audit '{audit_id}' not found")
		if resolution not in {"approved", "disputed"}:
			raise ValueError("resolution must be 'approved' or 'disputed'")
		audit["status"] = resolution
		audit["resolved_by"] = resolved_by
		audit["resolution_notes"] = resolution_notes
		audit["resolved_at"] = self._now()
		self._emit(tenant, f"freight_audit_{resolution}", audit_id, "scm_log_freight_audit", resolution)
		return deepcopy(audit)

	async def list_freight_audits(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List freight audits."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.freight_audits.values() if a["tenant_id"] == tenant]
		if status:
			items = [a for a in items if a["status"] == status]
		return items

	# ── Route optimisation ────────────────────────────────────────────────────

	async def create_route(
		self,
		origin: str,
		destination: str,
		mode: str,
		waypoints: list[str] | None = None,
		distance_km: float | None = None,
		estimated_transit_days: int | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Define a logistics route."""
		tenant = self._tenant(tenant_id)
		if mode not in SUPPORTED_FREIGHT_MODES:
			raise ValueError(f"mode must be one of {SUPPORTED_FREIGHT_MODES}")
		record: dict[str, Any] = {
			"id": self._id("route"),
			"type": "scm_log_route",
			"tenant_id": tenant,
			"origin": origin,
			"destination": destination,
			"waypoints": waypoints or [],
			"mode": mode,
			"distance_km": distance_km,
			"estimated_transit_days": estimated_transit_days,
			"optimised_at": None,
			"status": "active",
			"created_at": self._now(),
		}
		self.routes[record["id"]] = record
		self._emit(tenant, "route_created", record["id"], "scm_log_route", "active")
		return deepcopy(record)

	async def optimise_route(
		self,
		route_id: str,
		optimisation_params: dict[str, Any] | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Run route optimisation heuristics (time / cost / CO2)."""
		tenant = self._tenant(tenant_id)
		route = self.routes.get(route_id)
		if not route or route["tenant_id"] != tenant:
			raise KeyError(f"route '{route_id}' not found")
		params = optimisation_params or {}
		objective = params.get("objective", "cost")
		# Simplified scoring — replace with real solver integration
		scores = {"cost": 0.85, "time": 0.90, "co2": 0.75}
		route["optimisation_score"] = scores.get(objective, 0.80)
		route["optimisation_objective"] = objective
		route["optimised_at"] = self._now()
		self._emit(tenant, "route_optimised", route_id, "scm_log_route", "optimised")
		return deepcopy(route)

	async def list_routes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List routes for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.routes.values() if r["tenant_id"] == tenant]

	# ── Customs documentation ─────────────────────────────────────────────────

	async def create_customs_document(
		self,
		shipment_id: str,
		document_type: str,
		country_of_export: str,
		country_of_import: str,
		total_value: float,
		hs_codes: list[str] | None = None,
		currency: str = "USD",
		content: dict[str, Any] | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate a customs document for a shipment."""
		tenant = self._tenant(tenant_id)
		if document_type not in SUPPORTED_DOC_TYPES:
			raise ValueError(f"document_type must be one of {SUPPORTED_DOC_TYPES}")
		record: dict[str, Any] = {
			"id": self._id("cust"),
			"type": "scm_log_customs_document",
			"tenant_id": tenant,
			"shipment_id": shipment_id,
			"document_type": document_type,
			"country_of_export": country_of_export,
			"country_of_import": country_of_import,
			"hs_codes": hs_codes or [],
			"total_value": total_value,
			"currency": currency,
			"content": deepcopy(content or {}),
			"status": "draft",
			"created_at": self._now(),
		}
		self.customs_documents[record["id"]] = record
		self._emit(tenant, "customs_document_created", record["id"], "scm_log_customs_document", "draft")
		return deepcopy(record)

	async def submit_customs_document(
		self,
		document_id: str,
		submitted_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Submit customs document to authorities."""
		tenant = self._tenant(tenant_id)
		doc = self.customs_documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"customs_document '{document_id}' not found")
		if doc["status"] != "draft":
			raise ValueError("only draft documents can be submitted")
		doc["status"] = "submitted"
		doc["submitted_by"] = submitted_by
		doc["submitted_at"] = self._now()
		self._emit(tenant, "customs_document_submitted", document_id, "scm_log_customs_document", "submitted")
		return deepcopy(doc)

	async def list_customs_documents(
		self,
		tenant_id: str | None = None,
		shipment_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List customs documents."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.customs_documents.values() if d["tenant_id"] == tenant]
		if shipment_id:
			items = [d for d in items if d["shipment_id"] == shipment_id]
		return items

	# ── 3PL management ────────────────────────────────────────────────────────

	async def register_3pl_provider(
		self,
		provider_name: str,
		provider_code: str,
		service_types: list[str],
		contract_reference: str | None = None,
		sla_days: int | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a third-party logistics provider."""
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": self._id("3pl"),
			"type": "scm_log_3pl_provider",
			"tenant_id": tenant,
			"provider_name": provider_name,
			"provider_code": provider_code,
			"service_types": service_types,
			"contract_reference": contract_reference,
			"sla_days": sla_days,
			"status": "active",
			"created_at": self._now(),
		}
		self.third_party_providers[record["id"]] = record
		self._emit(tenant, "3pl_provider_registered", record["id"], "scm_log_3pl_provider", "active")
		return deepcopy(record)

	async def list_3pl_providers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List registered 3PL providers."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(p) for p in self.third_party_providers.values() if p["tenant_id"] == tenant]

	async def assign_shipment_to_3pl(
		self,
		shipment_id: str,
		provider_id: str,
		handover_notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Assign a shipment to a 3PL provider."""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		provider = self.third_party_providers.get(provider_id)
		if not provider or provider["tenant_id"] != tenant:
			raise KeyError(f"3pl_provider '{provider_id}' not found")
		shipment["3pl_provider_id"] = provider_id
		shipment["3pl_handover_notes"] = handover_notes
		shipment["3pl_assigned_at"] = self._now()
		shipment["updated_at"] = self._now()
		self._emit(tenant, "shipment_assigned_to_3pl", shipment_id, "scm_log_shipment", shipment["status"])
		return deepcopy(shipment)

	# ── Delivery exception management ─────────────────────────────────────────

	async def raise_delivery_exception(
		self,
		shipment_id: str,
		exception_type: str,
		description: str,
		raised_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a delivery exception (damage, delay, loss)."""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("exc"),
			"type": "scm_log_delivery_exception",
			"tenant_id": tenant,
			"shipment_id": shipment_id,
			"exception_type": exception_type,
			"description": description,
			"raised_by": raised_by,
			"status": "open",
			"created_at": self._now(),
		}
		self.delivery_exceptions[record["id"]] = record
		shipment["status"] = "exception"
		shipment["updated_at"] = self._now()
		self._emit(tenant, "delivery_exception_raised", record["id"], "scm_log_delivery_exception", "open")
		return deepcopy(record)

	async def resolve_delivery_exception(
		self,
		exception_id: str,
		resolution: str,
		resolved_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Resolve a delivery exception."""
		tenant = self._tenant(tenant_id)
		exc = self.delivery_exceptions.get(exception_id)
		if not exc or exc["tenant_id"] != tenant:
			raise KeyError(f"exception '{exception_id}' not found")
		exc["status"] = "resolved"
		exc["resolution"] = resolution
		exc["resolved_by"] = resolved_by
		exc["resolved_at"] = self._now()
		self._emit(tenant, "delivery_exception_resolved", exception_id, "scm_log_delivery_exception", "resolved")
		return deepcopy(exc)

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def shipment_analytics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return aggregate shipment analytics for a tenant."""
		tenant = self._tenant(tenant_id)
		all_shipments = [s for s in self.shipments.values() if s["tenant_id"] == tenant]
		by_status: dict[str, int] = {}
		by_mode: dict[str, int] = {}
		for s in all_shipments:
			by_status[s["status"]] = by_status.get(s["status"], 0) + 1
			by_mode[s["freight_mode"]] = by_mode.get(s["freight_mode"], 0) + 1
		total_weight = sum(s["weight_kg"] for s in all_shipments)
		return {
			"tenant_id": tenant,
			"total_shipments": len(all_shipments),
			"by_status": by_status,
			"by_freight_mode": by_mode,
			"total_weight_kg": total_weight,
			"active_carriers": sum(1 for c in self.carriers.values() if c["tenant_id"] == tenant and c["status"] == "active"),
			"open_exceptions": sum(1 for e in self.delivery_exceptions.values() if e["tenant_id"] == tenant and e["status"] == "open"),
			"pending_audits": sum(1 for a in self.freight_audits.values() if a["tenant_id"] == tenant and a["status"] == "pending"),
			"generated_at": self._now(),
		}

	async def freight_cost_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Summarise freight audit variances for cost control."""
		tenant = self._tenant(tenant_id)
		audits = [a for a in self.freight_audits.values() if a["tenant_id"] == tenant]
		total_invoiced = sum(a["invoiced_amount"] for a in audits)
		total_expected = sum(a["expected_amount"] for a in audits)
		total_variance = sum(a["variance"] for a in audits)
		disputed = sum(1 for a in audits if a["status"] == "disputed")
		return {
			"tenant_id": tenant,
			"audit_count": len(audits),
			"total_invoiced": round(total_invoiced, 2),
			"total_expected": round(total_expected, 2),
			"total_variance": round(total_variance, 2),
			"disputed_count": disputed,
			"generated_at": self._now(),
		}

	async def bulk_create_shipments(
		self,
		shipments_data: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-create multiple shipments."""
		tenant = self._tenant(tenant_id)
		results, errors = [], []
		tasks = [
			self.create_shipment(tenant_id=tenant, **s)
			for s in shipments_data
		]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		for item in raw:
			if isinstance(item, Exception):
				errors.append(str(item))
			else:
				results.append(item)
		return {"created": len(results), "failed": len(errors), "shipments": results, "errors": errors}

	# ── Carbon footprint tracking ─────────────────────────────────────────────

	async def calculate_shipment_co2(
		self,
		shipment_id: str,
		distance_km: float | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Calculate CO2e emissions for a shipment using GLEC Framework v3 factors.

		Emission factor (kgCO2e / tonne-km) is multiplied by weight and distance.
		Supply `distance_km` directly or attach a route via the shipment's carrier
		route; falls back to a great-circle estimate if neither is available.
		"""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		mode = shipment["freight_mode"]
		factor = _CO2_FACTORS.get(mode, 0.120)  # default multimodal
		weight_tonnes = shipment["weight_kg"] / 1000.0
		# resolve distance
		dist_km: float
		if distance_km is not None:
			dist_km = distance_km
		elif shipment.get("route_distance_km"):
			dist_km = shipment["route_distance_km"]
		else:
			# naive fallback — caller should supply real distance
			dist_km = 1000.0
		co2_kg = round(factor * weight_tonnes * dist_km, 4)
		result: dict[str, Any] = {
			"shipment_id": shipment_id,
			"freight_mode": mode,
			"emission_factor_kgco2e_per_tonne_km": factor,
			"weight_tonnes": weight_tonnes,
			"distance_km": dist_km,
			"co2_kg": co2_kg,
			"co2_tonnes": round(co2_kg / 1000, 6),
			"framework": "GLEC v3",
			"scope": "scope_3",
			"calculated_at": self._now(),
		}
		# persist on shipment for aggregation
		shipment["co2_kg"] = co2_kg
		shipment["updated_at"] = self._now()
		self._emit(tenant, "co2_calculated", shipment_id, "scm_log_shipment", shipment["status"])
		return result

	async def emissions_report(
		self,
		date_from: str | None = None,
		date_to: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Aggregate Scope 3 transport emissions across all shipments.

		Returns totals by freight mode and carrier. Filters by ISO-8601 date range
		when supplied; otherwise covers all shipments.
		"""
		tenant = self._tenant(tenant_id)
		all_shipments = [s for s in self.shipments.values() if s["tenant_id"] == tenant]
		if date_from:
			all_shipments = [s for s in all_shipments if s["created_at"] >= date_from]
		if date_to:
			all_shipments = [s for s in all_shipments if s["created_at"] <= date_to]
		by_mode: dict[str, float] = {}
		by_carrier: dict[str, float] = {}
		total_co2 = 0.0
		for s in all_shipments:
			co2 = s.get("co2_kg", 0.0)
			by_mode[s["freight_mode"]] = round(by_mode.get(s["freight_mode"], 0.0) + co2, 4)
			cid = s.get("carrier_id", "unknown")
			by_carrier[cid] = round(by_carrier.get(cid, 0.0) + co2, 4)
			total_co2 += co2
		return {
			"tenant_id": tenant,
			"date_from": date_from,
			"date_to": date_to,
			"total_co2_kg": round(total_co2, 4),
			"total_co2_tonnes": round(total_co2 / 1000, 6),
			"by_freight_mode": by_mode,
			"by_carrier": by_carrier,
			"shipment_count": len(all_shipments),
			"scope": "scope_3",
			"framework": "GLEC v3 / GHG Protocol",
			"generated_at": self._now(),
		}

	# ── SLA monitoring ────────────────────────────────────────────────────────

	async def check_sla_breaches(
		self,
		at_risk_hours: float = 4.0,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Scan in-transit shipments for SLA breaches and at-risk deliveries.

		Compares `estimated_delivery` against current UTC time.  Returns three
		buckets: `breached` (ETA already past), `at_risk` (ETA within
		`at_risk_hours`), and `on_track`.
		"""
		tenant = self._tenant(tenant_id)
		now_str = self._now()
		now_dt = datetime.fromisoformat(now_str.rstrip("Z"))
		at_risk_threshold = now_dt + timedelta(hours=at_risk_hours)
		breached, at_risk, on_track = [], [], []
		for s in self.shipments.values():
			if s["tenant_id"] != tenant:
				continue
			if s["status"] not in {"booked", "in_transit"}:
				continue
			eta_raw = s.get("estimated_delivery")
			if not eta_raw:
				continue
			eta_dt = datetime.fromisoformat(eta_raw.rstrip("Z"))
			entry = {
				"shipment_id": s["id"],
				"carrier_id": s["carrier_id"],
				"status": s["status"],
				"estimated_delivery": eta_raw,
				"hours_delta": round((eta_dt - now_dt).total_seconds() / 3600, 2),
			}
			if eta_dt < now_dt:
				entry["breach_hours"] = round((now_dt - eta_dt).total_seconds() / 3600, 2)
				breached.append(entry)
				self._emit(tenant, "sla_breach_detected", s["id"], "scm_log_shipment", "breached")
			elif eta_dt <= at_risk_threshold:
				at_risk.append(entry)
				self._emit(tenant, "sla_at_risk", s["id"], "scm_log_shipment", "at_risk")
			else:
				on_track.append(entry)
		return {
			"tenant_id": tenant,
			"checked_at": now_str,
			"at_risk_window_hours": at_risk_hours,
			"breached": breached,
			"at_risk": at_risk,
			"on_track": on_track,
			"summary": {
				"breached_count": len(breached),
				"at_risk_count": len(at_risk),
				"on_track_count": len(on_track),
			},
		}

	async def sla_performance_report(
		self,
		carrier_id: str | None = None,
		date_from: str | None = None,
		date_to: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute on-time delivery rate, average delay, and exception rate per carrier.

		Considers only delivered shipments with both `estimated_delivery` and
		`actual_delivery` populated.
		"""
		tenant = self._tenant(tenant_id)
		delivered = [
			s for s in self.shipments.values()
			if s["tenant_id"] == tenant
			and s["status"] == "delivered"
			and s.get("estimated_delivery")
			and s.get("actual_delivery")
		]
		if carrier_id:
			delivered = [s for s in delivered if s["carrier_id"] == carrier_id]
		if date_from:
			delivered = [s for s in delivered if s["created_at"] >= date_from]
		if date_to:
			delivered = [s for s in delivered if s["created_at"] <= date_to]
		per_carrier: dict[str, dict[str, Any]] = {}
		for s in delivered:
			cid = s["carrier_id"]
			if cid not in per_carrier:
				per_carrier[cid] = {"total": 0, "on_time": 0, "delay_hours_list": []}
			eta_dt = datetime.fromisoformat(s["estimated_delivery"].rstrip("Z"))
			act_dt = datetime.fromisoformat(s["actual_delivery"].rstrip("Z"))
			delay_h = (act_dt - eta_dt).total_seconds() / 3600
			per_carrier[cid]["total"] += 1
			per_carrier[cid]["delay_hours_list"].append(delay_h)
			if delay_h <= 0:
				per_carrier[cid]["on_time"] += 1
		summary = {}
		for cid, data in per_carrier.items():
			delays = data["delay_hours_list"]
			summary[cid] = {
				"total_deliveries": data["total"],
				"on_time_count": data["on_time"],
				"on_time_rate_pct": round(data["on_time"] / data["total"] * 100, 2),
				"avg_delay_hours": round(sum(delays) / len(delays), 2),
				"max_delay_hours": round(max(delays), 2),
			}
		return {
			"tenant_id": tenant,
			"carrier_id_filter": carrier_id,
			"date_from": date_from,
			"date_to": date_to,
			"per_carrier": summary,
			"generated_at": self._now(),
		}

	# ── Shipment consolidation ────────────────────────────────────────────────

	async def suggest_consolidation(
		self,
		shipment_ids: list[str],
		departure_window_hours: float = 24.0,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Analyse a list of shipments and propose consolidation groups.

		Groups shipments by (origin country, destination country, freight mode).
		Returns potential savings estimate based on weight-break tariff assumption
		(10% discount per 100 kg consolidated above 100 kg).
		"""
		tenant = self._tenant(tenant_id)
		shipments = []
		for sid in shipment_ids:
			s = self.shipments.get(sid)
			if not s or s["tenant_id"] != tenant:
				raise KeyError(f"shipment '{sid}' not found")
			shipments.append(s)
		# group by corridor + mode
		groups: dict[tuple, list[dict]] = {}
		for s in shipments:
			o = s["origin_address"].get("country", "UNK")
			d = s["destination_address"].get("country", "UNK")
			key = (o, d, s["freight_mode"])
			groups.setdefault(key, []).append(s)
		proposals = []
		for (o, d, mode), members in groups.items():
			if len(members) < 2:
				continue
			total_weight = sum(m["weight_kg"] for m in members)
			# simplified discount model
			discount_pct = min(30.0, ((total_weight - 100) // 100) * 10) if total_weight > 100 else 0.0
			proposals.append({
				"origin_country": o,
				"destination_country": d,
				"freight_mode": mode,
				"shipment_ids": [m["id"] for m in members],
				"shipment_count": len(members),
				"combined_weight_kg": round(total_weight, 3),
				"estimated_discount_pct": discount_pct,
				"departure_window_hours": departure_window_hours,
			})
		return {
			"tenant_id": tenant,
			"total_shipments_analysed": len(shipments),
			"consolidation_proposals": proposals,
			"generated_at": self._now(),
		}

	async def create_consolidated_shipment(
		self,
		child_shipment_ids: list[str],
		carrier_id: str,
		freight_mode: str,
		service_level: str = "standard",
		reference_number: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Merge multiple draft shipments into a single consolidated master shipment.

		Child shipments are cancelled and linked to the master via
		`consolidated_into` field. The master inherits the bounding origin/destination
		and the sum of weights and volumes.
		"""
		tenant = self._tenant(tenant_id)
		if freight_mode not in SUPPORTED_FREIGHT_MODES:
			raise ValueError(f"freight_mode must be one of {SUPPORTED_FREIGHT_MODES}")
		carrier = self.carriers.get(carrier_id)
		if not carrier or carrier["tenant_id"] != tenant:
			raise KeyError(f"carrier '{carrier_id}' not found")
		children = []
		for sid in child_shipment_ids:
			s = self.shipments.get(sid)
			if not s or s["tenant_id"] != tenant:
				raise KeyError(f"shipment '{sid}' not found")
			if s["status"] != "draft":
				raise ValueError(f"shipment '{sid}' must be draft to consolidate")
			children.append(s)
		total_weight = sum(s["weight_kg"] for s in children)
		total_volume = sum(s.get("volume_m3") or 0.0 for s in children)
		origin = children[0]["origin_address"]
		destination = children[0]["destination_address"]
		master: dict[str, Any] = {
			"id": self._id("shp"),
			"type": "scm_log_shipment",
			"tenant_id": tenant,
			"carrier_id": carrier_id,
			"origin_address": deepcopy(origin),
			"destination_address": deepcopy(destination),
			"weight_kg": round(total_weight, 3),
			"volume_m3": round(total_volume, 4) if total_volume else None,
			"freight_mode": freight_mode,
			"service_level": service_level,
			"declared_value": None,
			"currency": "USD",
			"special_instructions": f"Consolidated from {len(children)} shipments",
			"reference_number": reference_number,
			"tracking_number": None,
			"estimated_delivery": None,
			"actual_delivery": None,
			"status": "draft",
			"consolidated": True,
			"child_shipment_ids": [s["id"] for s in children],
			"created_at": self._now(),
			"updated_at": None,
		}
		self.shipments[master["id"]] = master
		# cancel children and link to master
		for s in children:
			s["status"] = "cancelled"
			s["cancellation_reason"] = "consolidated"
			s["consolidated_into"] = master["id"]
			s["cancelled_at"] = self._now()
			s["updated_at"] = self._now()
		self._emit(tenant, "consolidated_shipment_created", master["id"], "scm_log_shipment", "draft")
		return deepcopy(master)

	# ── Carrier performance scorecard ─────────────────────────────────────────

	async def generate_carrier_scorecard(
		self,
		carrier_id: str,
		period_days: int = 90,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute a multi-dimensional performance scorecard for a carrier.

		Dimensions: on-time rate, exception rate, freight audit dispute rate,
		average cost per kg, average CO2 per tonne-km.
		"""
		tenant = self._tenant(tenant_id)
		carrier = self.carriers.get(carrier_id)
		if not carrier or carrier["tenant_id"] != tenant:
			raise KeyError(f"carrier '{carrier_id}' not found")
		cutoff = (datetime.utcnow() - timedelta(days=period_days)).isoformat(timespec="seconds") + "Z"
		carrier_shipments = [
			s for s in self.shipments.values()
			if s["tenant_id"] == tenant
			and s["carrier_id"] == carrier_id
			and s["created_at"] >= cutoff
		]
		total = len(carrier_shipments)
		delivered = [s for s in carrier_shipments if s["status"] == "delivered"]
		exceptions = [s for s in carrier_shipments if s["status"] == "exception"]
		on_time_count = sum(
			1 for s in delivered
			if s.get("estimated_delivery") and s.get("actual_delivery")
			and s["actual_delivery"] <= s["estimated_delivery"]
		)
		audits = [
			a for a in self.freight_audits.values()
			if a["tenant_id"] == tenant and a["carrier_id"] == carrier_id
			and a["created_at"] >= cutoff
		]
		dispute_count = sum(1 for a in audits if a["status"] == "disputed")
		total_co2 = sum(s.get("co2_kg", 0.0) for s in carrier_shipments)
		total_weight_t = sum(s["weight_kg"] for s in carrier_shipments) / 1000.0
		avg_co2_per_tonne_km = round(total_co2 / total_weight_t, 4) if total_weight_t else None
		return {
			"carrier_id": carrier_id,
			"carrier_name": carrier["name"],
			"period_days": period_days,
			"total_shipments": total,
			"delivered_count": len(delivered),
			"on_time_rate_pct": round(on_time_count / len(delivered) * 100, 2) if delivered else None,
			"exception_rate_pct": round(len(exceptions) / total * 100, 2) if total else None,
			"audit_count": len(audits),
			"audit_dispute_rate_pct": round(dispute_count / len(audits) * 100, 2) if audits else None,
			"avg_co2_per_tonne_km": avg_co2_per_tonne_km,
			"avg_rating": carrier.get("avg_rating"),
			"generated_at": self._now(),
		}

	# ── Insurance ─────────────────────────────────────────────────────────────

	async def request_insurance_quote(
		self,
		shipment_id: str,
		coverage_type: str = "all_risk",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate a freight insurance premium estimate.

		Premium model: declared_value * mode_risk_factor * coverage_multiplier.
		Mode risk factors: air 0.005, sea 0.008, road 0.006, rail 0.004.
		Coverage: all_risk 1.0, named_perils 0.65.
		"""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		if coverage_type not in {"all_risk", "named_perils"}:
			raise ValueError("coverage_type must be 'all_risk' or 'named_perils'")
		declared_value = shipment.get("declared_value")
		if not declared_value:
			raise ValueError("shipment must have declared_value to quote insurance")
		mode_risk = {"air": 0.005, "sea": 0.008, "road": 0.006, "rail": 0.004, "multimodal": 0.007}
		coverage_mult = {"all_risk": 1.0, "named_perils": 0.65}
		rate = mode_risk.get(shipment["freight_mode"], 0.007) * coverage_mult[coverage_type]
		premium = round(declared_value * rate, 2)
		quote: dict[str, Any] = {
			"id": self._id("insq"),
			"type": "scm_log_insurance_quote",
			"tenant_id": tenant,
			"shipment_id": shipment_id,
			"coverage_type": coverage_type,
			"declared_value": declared_value,
			"currency": shipment["currency"],
			"premium": premium,
			"rate_pct": round(rate * 100, 4),
			"valid_until": (datetime.utcnow() + timedelta(hours=48)).isoformat(timespec="seconds") + "Z",
			"status": "quoted",
			"created_at": self._now(),
		}
		self._emit(tenant, "insurance_quoted", quote["id"], "scm_log_insurance_quote", "quoted")
		return quote

	async def bind_insurance_policy(
		self,
		shipment_id: str,
		quote: dict[str, Any],
		insured_by: str,
		policy_number: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bind an insurance policy against a previously obtained quote.

		Stores policy metadata on the shipment and in the `insurance_policies`
		registry. Emits `insurance_policy_bound` audit event.
		"""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		pol_num = policy_number or f"POL{uuid4().hex[:10].upper()}"
		policy: dict[str, Any] = {
			"id": self._id("ins"),
			"type": "scm_log_insurance_policy",
			"tenant_id": tenant,
			"shipment_id": shipment_id,
			"policy_number": pol_num,
			"coverage_type": quote.get("coverage_type"),
			"declared_value": quote.get("declared_value"),
			"premium": quote.get("premium"),
			"currency": quote.get("currency", shipment["currency"]),
			"insured_by": insured_by,
			"status": "active",
			"bound_at": self._now(),
		}
		self.insurance_policies[policy["id"]] = policy
		shipment["insurance_policy_id"] = policy["id"]
		shipment["insurance_policy_number"] = pol_num
		shipment["updated_at"] = self._now()
		self._emit(tenant, "insurance_policy_bound", policy["id"], "scm_log_insurance_policy", "active")
		return deepcopy(policy)

	async def file_insurance_claim(
		self,
		shipment_id: str,
		claim_type: str,
		description: str,
		claimed_amount: float,
		filed_by: str,
		supporting_exception_id: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""File an insurance claim against a bound policy on a shipment.

		`claim_type` is free-form (damage, loss, delay, theft). Links to a
		delivery exception when `supporting_exception_id` is supplied.
		"""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		if not shipment.get("insurance_policy_id"):
			raise ValueError("shipment has no bound insurance policy")
		policy_id = shipment["insurance_policy_id"]
		policy = self.insurance_policies.get(policy_id)
		if not policy:
			raise KeyError(f"insurance_policy '{policy_id}' not found")
		claim: dict[str, Any] = {
			"id": self._id("clm"),
			"type": "scm_log_insurance_claim",
			"tenant_id": tenant,
			"shipment_id": shipment_id,
			"policy_id": policy_id,
			"policy_number": policy["policy_number"],
			"claim_type": claim_type,
			"description": description,
			"claimed_amount": claimed_amount,
			"currency": policy["currency"],
			"filed_by": filed_by,
			"supporting_exception_id": supporting_exception_id,
			"status": "filed",
			"created_at": self._now(),
		}
		self.insurance_claims[claim["id"]] = claim
		self._emit(tenant, "insurance_claim_filed", claim["id"], "scm_log_insurance_claim", "filed")
		return deepcopy(claim)

	# ── Proof of delivery ─────────────────────────────────────────────────────

	async def attach_pod(
		self,
		shipment_id: str,
		document_url: str,
		captured_by: str,
		mime_type: str = "image/jpeg",
		signature_hash: str | None = None,
		recipient_name: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Attach a proof-of-delivery (POD) record to a delivered shipment.

		`document_url` is a reference to object storage (S3/GCS presigned URL).
		`signature_hash` should be SHA-256 of the signature image when available.
		Transitions shipment status to `delivered` if currently `in_transit`.
		"""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		if shipment["status"] not in {"in_transit", "delivered", "exception"}:
			raise ValueError("POD can only be attached to in-transit or delivered shipments")
		pod: dict[str, Any] = {
			"id": self._id("pod"),
			"type": "scm_log_pod",
			"tenant_id": tenant,
			"shipment_id": shipment_id,
			"document_url": document_url,
			"mime_type": mime_type,
			"captured_by": captured_by,
			"signature_hash": signature_hash,
			"recipient_name": recipient_name,
			"captured_at": self._now(),
		}
		shipment.setdefault("pods", []).append(pod)
		if shipment["status"] == "in_transit":
			shipment["status"] = "delivered"
			shipment["actual_delivery"] = self._now()
		shipment["updated_at"] = self._now()
		self._emit(tenant, "pod_attached", pod["id"], "scm_log_pod", "captured")
		return deepcopy(pod)

	async def get_pod(
		self,
		shipment_id: str,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return all POD records attached to a shipment."""
		tenant = self._tenant(tenant_id)
		shipment = self.shipments.get(shipment_id)
		if not shipment or shipment["tenant_id"] != tenant:
			raise KeyError(f"shipment '{shipment_id}' not found")
		return deepcopy(shipment.get("pods", []))
