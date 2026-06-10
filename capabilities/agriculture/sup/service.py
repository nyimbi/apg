"""Agricultural Supply Chain service — agr_sup."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_sup"

# Cold chain temperature bounds (°C) per product category
_COLD_CHAIN_LIMITS: dict[str, dict[str, float]] = {
	"default":   {"min": 2.0, "max": 8.0},
	"vegetables": {"min": 4.0, "max": 10.0},
	"flowers":   {"min": 2.0, "max": 6.0},
	"dairy":     {"min": 1.0, "max": 4.0},
	"meat":      {"min": 0.0, "max": 4.0},
}


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


class SupplyChainService:
	"""Async service for agricultural supply chain: farm-to-buyer traceability,
	input procurement, cold chain management, and export documentation."""

	def __init__(self, tenant_id: str = "default") -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		self._batches: dict[str, dict[str, Any]] = {}
		self._trace_events: dict[str, list[dict[str, Any]]] = {}
		self._procurement: dict[str, dict[str, Any]] = {}
		self._cold_chain: dict[str, dict[str, Any]] = {}
		self._export_docs: dict[str, dict[str, Any]] = {}
		self._audit: list[dict[str, Any]] = []

	def _emit(self, event_type: str, entity_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		self._audit.append({
			"id": _new_id("evt"),
			"tenant_id": self.tenant_id,
			"event_type": event_type,
			"entity_type": entity_type,
			"entity_id": entity_id,
			"payload": payload,
			"occurred_at": _now(),
		})

	# ------------------------------------------------------------------ health

	async def health_check(self) -> dict[str, Any]:
		return {
			"status": "ok",
			"capability": _CAPABILITY_ID,
			"tenant_id": self.tenant_id,
			"counts": {
				"batches": len(self._batches),
				"procurement_orders": len(self._procurement),
				"cold_chain_logs": len(self._cold_chain),
				"export_documents": len(self._export_docs),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Agricultural Supply Chain",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Farm-to-buyer traceability, input procurement, cold chain management, export documentation.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ batches / traceability

	async def list_batches(self, farmer_id: str | None = None, buyer_id: str | None = None,
						status: str | None = None, product_type: str | None = None,
						limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._batches.values())
		if farmer_id:
			items = [b for b in items if b.get("farmer_id") == farmer_id]
		if buyer_id:
			items = [b for b in items if b.get("buyer_id") == buyer_id]
		if status:
			items = [b for b in items if b.get("status") == status]
		if product_type:
			items = [b for b in items if b.get("product_type") == product_type]
		return items[offset: offset + limit]

	async def get_batch(self, batch_id: str) -> dict[str, Any]:
		if batch_id not in self._batches:
			raise KeyError(f"batch_not_found:{batch_id}")
		return self._batches[batch_id]

	async def create_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			bid = _new_id("bat")
			ts = _now()
			record: dict[str, Any] = {
				"id": bid,
				"tenant_id": self.tenant_id,
				"batch_id": payload.get("batch_id") or bid,
				"product_type": payload["product_type"],
				"farm_parcel_id": payload["farm_parcel_id"],
				"farmer_id": payload["farmer_id"],
				"harvest_date": payload["harvest_date"],
				"weight_kg": float(payload["weight_kg"]),
				"quality_grade": payload.get("quality_grade"),
				"status": "farm",
				"buyer_id": payload.get("buyer_id"),
				"current_location": payload.get("current_location"),
				"notes": payload.get("notes"),
				"metadata": dict(payload.get("metadata", {})),
				"created_at": ts,
				"updated_at": ts,
			}
			self._batches[bid] = record
			self._trace_events[bid] = [{
				"event": "batch_created",
				"status": "farm",
				"location": "farm",
				"occurred_at": ts,
			}]
			self._emit("batch.created", "batch", bid, record)
			return record
		except Exception as exc:
			_log.error("create_batch failed: %s", exc)
			raise

	async def update_batch(self, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if batch_id not in self._batches:
				raise KeyError(f"batch_not_found:{batch_id}")
			record = self._batches[batch_id]
			prev_status = record.get("status")
			for field in ["status", "buyer_id", "current_location", "weight_kg", "notes", "metadata"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			if record.get("status") != prev_status:
				if batch_id not in self._trace_events:
					self._trace_events[batch_id] = []
				self._trace_events[batch_id].append({
					"event": "status_changed",
					"from": prev_status,
					"to": record["status"],
					"location": record.get("current_location"),
					"occurred_at": _now(),
				})
			self._emit("batch.updated", "batch", batch_id, payload)
			return record
		except Exception as exc:
			_log.error("update_batch failed: %s", exc)
			raise

	async def delete_batch(self, batch_id: str) -> dict[str, Any]:
		try:
			if batch_id not in self._batches:
				raise KeyError(f"batch_not_found:{batch_id}")
			self._batches.pop(batch_id)
			self._trace_events.pop(batch_id, None)
			self._emit("batch.deleted", "batch", batch_id, {"id": batch_id})
			return {"deleted": True, "id": batch_id}
		except Exception as exc:
			_log.error("delete_batch failed: %s", exc)
			raise

	async def get_batch_trace(self, batch_id: str) -> dict[str, Any]:
		"""Return full provenance chain for a batch."""
		if batch_id not in self._batches:
			raise KeyError(f"batch_not_found:{batch_id}")
		batch = self._batches[batch_id]
		events = self._trace_events.get(batch_id, [])
		cold_logs = [c for c in self._cold_chain.values() if c.get("batch_id") == batch_id]
		docs = [d for d in self._export_docs.values() if d.get("batch_id") == batch_id]
		return {
			"batch": batch,
			"trace_events": events,
			"cold_chain_entries": len(cold_logs),
			"export_documents": len(docs),
			"document_types": [d.get("document_type") for d in docs],
		}

	async def link_buyer(self, batch_id: str, buyer_id: str) -> dict[str, Any]:
		"""Assign or reassign a buyer to a batch."""
		try:
			if batch_id not in self._batches:
				raise KeyError(f"batch_not_found:{batch_id}")
			self._batches[batch_id]["buyer_id"] = buyer_id
			self._batches[batch_id]["updated_at"] = _now()
			self._emit("batch.buyer_linked", "batch", batch_id, {"buyer_id": buyer_id})
			return self._batches[batch_id]
		except Exception as exc:
			_log.error("link_buyer failed: %s", exc)
			raise

	# ------------------------------------------------------------------ procurement

	async def list_procurement(self, supplier_id: str | None = None, status: str | None = None,
							limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._procurement.values())
		if supplier_id:
			items = [p for p in items if p.get("supplier_id") == supplier_id]
		if status:
			items = [p for p in items if p.get("status") == status]
		return items[offset: offset + limit]

	async def get_procurement_order(self, order_id: str) -> dict[str, Any]:
		if order_id not in self._procurement:
			raise KeyError(f"procurement_order_not_found:{order_id}")
		return self._procurement[order_id]

	async def create_procurement_order(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			oid = _new_id("ord")
			ts = _now()
			qty = float(payload["quantity"])
			unit_price = float(payload["unit_price"])
			record: dict[str, Any] = {
				"id": oid,
				"tenant_id": self.tenant_id,
				"supplier_id": payload["supplier_id"],
				"product_name": payload["product_name"],
				"quantity": qty,
				"unit": payload["unit"],
				"unit_price": unit_price,
				"total_value": round(qty * unit_price, 2),
				"required_date": payload["required_date"],
				"status": "requested",
				"actual_delivery_date": None,
				"quantity_received": None,
				"invoice_reference": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._procurement[oid] = record
			self._emit("procurement.created", "procurement_order", oid, record)
			return record
		except Exception as exc:
			_log.error("create_procurement_order failed: %s", exc)
			raise

	async def update_procurement_order(self, order_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if order_id not in self._procurement:
				raise KeyError(f"procurement_order_not_found:{order_id}")
			record = self._procurement[order_id]
			for field in ["status", "actual_delivery_date", "quantity_received", "invoice_reference", "notes"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("procurement.updated", "procurement_order", order_id, payload)
			return record
		except Exception as exc:
			_log.error("update_procurement_order failed: %s", exc)
			raise

	async def delete_procurement_order(self, order_id: str) -> dict[str, Any]:
		try:
			if order_id not in self._procurement:
				raise KeyError(f"procurement_order_not_found:{order_id}")
			self._procurement.pop(order_id)
			self._emit("procurement.deleted", "procurement_order", order_id, {"id": order_id})
			return {"deleted": True, "id": order_id}
		except Exception as exc:
			_log.error("delete_procurement_order failed: %s", exc)
			raise

	async def get_supplier_performance(self, supplier_id: str) -> dict[str, Any]:
		"""Compute on-time delivery rate and fulfilment ratio for a supplier."""
		orders = [p for p in self._procurement.values() if p.get("supplier_id") == supplier_id]
		delivered = [o for o in orders if o.get("status") == "delivered"]
		on_time = [o for o in delivered
				if o.get("actual_delivery_date") and o.get("required_date")
				and o["actual_delivery_date"] <= o["required_date"]]
		fill_ratios = []
		for o in delivered:
			if o.get("quantity") and o.get("quantity_received"):
				fill_ratios.append(o["quantity_received"] / o["quantity"])
		return {
			"supplier_id": supplier_id,
			"total_orders": len(orders),
			"delivered_orders": len(delivered),
			"on_time_deliveries": len(on_time),
			"on_time_rate_pct": round(len(on_time) / max(len(delivered), 1) * 100, 1),
			"avg_fill_ratio": round(sum(fill_ratios) / len(fill_ratios), 3) if fill_ratios else None,
		}

	# ------------------------------------------------------------------ cold chain

	async def log_cold_chain(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Record a cold chain temperature reading for a batch."""
		try:
			log_id = _new_id("ccl")
			ts = _now()
			temp = float(payload["temperature_c"])
			batch_id = payload["batch_id"]
			# Classify status
			product_type = "default"
			if batch_id in self._batches:
				product_type = self._batches[batch_id].get("product_type", "default").lower()
			limits = _COLD_CHAIN_LIMITS.get(product_type, _COLD_CHAIN_LIMITS["default"])
			if temp < limits["min"] - 2 or temp > limits["max"] + 2:
				status = "critical"
			elif temp < limits["min"] or temp > limits["max"]:
				status = "breach"
			else:
				status = "normal"
			record: dict[str, Any] = {
				"id": log_id,
				"tenant_id": self.tenant_id,
				"batch_id": batch_id,
				"location": payload["location"],
				"temperature_c": temp,
				"humidity_pct": payload.get("humidity_pct"),
				"status": status,
				"recorded_at": payload.get("recorded_at") or ts,
				"created_at": ts,
			}
			self._cold_chain[log_id] = record
			self._emit("cold_chain.logged", "cold_chain_log", log_id, {"batch_id": batch_id, "temp": temp, "status": status})
			return record
		except Exception as exc:
			_log.error("log_cold_chain failed: %s", exc)
			raise

	async def list_cold_chain_logs(self, batch_id: str | None = None, status: str | None = None,
								limit: int = 100) -> list[dict[str, Any]]:
		items = list(self._cold_chain.values())
		if batch_id:
			items = [c for c in items if c.get("batch_id") == batch_id]
		if status:
			items = [c for c in items if c.get("status") == status]
		return sorted(items, key=lambda x: x.get("recorded_at", ""), reverse=True)[:limit]

	async def get_cold_chain_summary(self, batch_id: str) -> dict[str, Any]:
		"""Summarise cold chain integrity for a batch."""
		logs = [c for c in self._cold_chain.values() if c.get("batch_id") == batch_id]
		if not logs:
			return {"batch_id": batch_id, "logs": 0, "breaches": 0, "integrity": "no_data"}
		breaches = [l for l in logs if l.get("status") in ("breach", "critical")]
		integrity = "ok" if not breaches else ("compromised" if any(l["status"] == "critical" for l in breaches) else "warning")
		temps = [l["temperature_c"] for l in logs]
		return {
			"batch_id": batch_id,
			"log_count": len(logs),
			"breach_count": len(breaches),
			"integrity": integrity,
			"min_temp_c": min(temps),
			"max_temp_c": max(temps),
			"avg_temp_c": round(sum(temps) / len(temps), 2),
		}

	# ------------------------------------------------------------------ export documents

	async def list_export_docs(self, batch_id: str | None = None, document_type: str | None = None) -> list[dict[str, Any]]:
		items = list(self._export_docs.values())
		if batch_id:
			items = [d for d in items if d.get("batch_id") == batch_id]
		if document_type:
			items = [d for d in items if d.get("document_type") == document_type]
		return items

	async def create_export_doc(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			doc_id = _new_id("doc")
			ts = _now()
			record: dict[str, Any] = {
				"id": doc_id,
				"tenant_id": self.tenant_id,
				"batch_id": payload["batch_id"],
				"document_type": payload["document_type"],
				"issuing_authority": payload.get("issuing_authority"),
				"issue_date": payload["issue_date"],
				"expiry_date": payload.get("expiry_date"),
				"reference_number": payload["reference_number"],
				"file_url": payload.get("file_url"),
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._export_docs[doc_id] = record
			self._emit("export_doc.created", "export_document", doc_id, record)
			return record
		except Exception as exc:
			_log.error("create_export_doc failed: %s", exc)
			raise

	async def delete_export_doc(self, doc_id: str) -> dict[str, Any]:
		try:
			if doc_id not in self._export_docs:
				raise KeyError(f"export_doc_not_found:{doc_id}")
			self._export_docs.pop(doc_id)
			self._emit("export_doc.deleted", "export_document", doc_id, {"id": doc_id})
			return {"deleted": True, "id": doc_id}
		except Exception as exc:
			_log.error("delete_export_doc failed: %s", exc)
			raise

	async def check_export_readiness(self, batch_id: str) -> dict[str, Any]:
		"""Verify a batch has all required documents and cold chain integrity for export."""
		if batch_id not in self._batches:
			raise KeyError(f"batch_not_found:{batch_id}")
		required_docs = {"phytosanitary_certificate", "certificate_of_origin", "commercial_invoice"}
		docs = [d for d in self._export_docs.values() if d.get("batch_id") == batch_id]
		present = {d.get("document_type") for d in docs}
		missing = required_docs - present
		cold_summary = await self.get_cold_chain_summary(batch_id)
		ready = not missing and cold_summary.get("integrity") in ("ok", "no_data")
		return {
			"batch_id": batch_id,
			"export_ready": ready,
			"missing_documents": list(missing),
			"cold_chain_integrity": cold_summary.get("integrity"),
			"present_documents": list(present),
		}
