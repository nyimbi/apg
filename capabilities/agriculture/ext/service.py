"""Extension Services service — agr_ext."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)
_CAPABILITY_ID = "agr_ext"


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_id(prefix: str = "") -> str:
	suffix = uuid4().hex[:12]
	return f"{prefix}-{suffix}" if prefix else suffix


class ExtensionServicesService:
	"""Async service for extension: advisory delivery, demo plot management,
	training records, and knowledge base."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		if not tenant_id:
			raise ValueError("tenant_id required")
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self._advisories = WriteThruDict('advisories', tenant_id, _store)
		self._demo_plots = WriteThruDict('demo_plots', tenant_id, _store)
		self._trainings = WriteThruDict('trainings', tenant_id, _store)
		self._knowledge = WriteThruDict('knowledge', tenant_id, _store)
		self._audit = WriteThruList('audit', tenant_id, _store)

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
				"advisories": len(self._advisories),
				"demo_plots": len(self._demo_plots),
				"trainings": len(self._trainings),
				"knowledge_articles": len(self._knowledge),
			},
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": _CAPABILITY_ID,
			"name": "Extension Services",
			"domain": "agriculture",
			"version": "1.0.0",
			"description": "Agricultural advisory delivery, demo plot management, training records, knowledge base.",
		}

	async def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
		return self._audit[-limit:]

	# ------------------------------------------------------------------ advisories

	async def list_advisories(self, farmer_id: str | None = None, extension_worker_id: str | None = None,
							channel: str | None = None, follow_up_required: bool | None = None,
							limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._advisories.values())
		if farmer_id:
			items = [a for a in items if a.get("farmer_id") == farmer_id]
		if extension_worker_id:
			items = [a for a in items if a.get("extension_worker_id") == extension_worker_id]
		if channel:
			items = [a for a in items if a.get("channel") == channel]
		if follow_up_required is not None:
			items = [a for a in items if a.get("follow_up_required") == follow_up_required]
		return items[offset: offset + limit]

	async def get_advisory(self, advisory_id: str) -> dict[str, Any]:
		if advisory_id not in self._advisories:
			raise KeyError(f"advisory_not_found:{advisory_id}")
		return self._advisories[advisory_id]

	async def create_advisory(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			aid = _new_id("adv")
			ts = _now()
			record: dict[str, Any] = {
				"id": aid,
				"tenant_id": self.tenant_id,
				"farmer_id": payload["farmer_id"],
				"extension_worker_id": payload["extension_worker_id"],
				"channel": payload["channel"],
				"topic": payload["topic"],
				"message": payload["message"],
				"crop_type": payload.get("crop_type"),
				"farm_parcel_id": payload.get("farm_parcel_id"),
				"delivered_at": payload.get("delivered_at") or ts,
				"follow_up_required": bool(payload.get("follow_up_required", False)),
				"follow_up_done": False,
				"notes": payload.get("notes"),
				"created_at": ts,
			}
			self._advisories[aid] = record
			self._emit("advisory.created", "advisory", aid, record)
			return record
		except Exception as exc:
			_log.error("create_advisory failed: %s", exc)
			raise

	async def mark_follow_up_done(self, advisory_id: str, notes: str | None = None) -> dict[str, Any]:
		try:
			if advisory_id not in self._advisories:
				raise KeyError(f"advisory_not_found:{advisory_id}")
			self._advisories[advisory_id]["follow_up_done"] = True
			if notes:
				self._advisories[advisory_id]["follow_up_notes"] = notes
			self._emit("advisory.follow_up_done", "advisory", advisory_id, {"id": advisory_id})
			return self._advisories[advisory_id]
		except Exception as exc:
			_log.error("mark_follow_up_done failed: %s", exc)
			raise

	async def delete_advisory(self, advisory_id: str) -> dict[str, Any]:
		try:
			if advisory_id not in self._advisories:
				raise KeyError(f"advisory_not_found:{advisory_id}")
			self._advisories.pop(advisory_id)
			self._emit("advisory.deleted", "advisory", advisory_id, {"id": advisory_id})
			return {"deleted": True, "id": advisory_id}
		except Exception as exc:
			_log.error("delete_advisory failed: %s", exc)
			raise

	async def get_extension_worker_stats(self, worker_id: str) -> dict[str, Any]:
		"""Compute advisory reach and follow-up completion for a worker."""
		advisories = [a for a in self._advisories.values() if a.get("extension_worker_id") == worker_id]
		follow_ups = [a for a in advisories if a.get("follow_up_required")]
		completed = [a for a in follow_ups if a.get("follow_up_done")]
		farmers_reached = len({a.get("farmer_id") for a in advisories})
		channels = {}
		for a in advisories:
			ch = a.get("channel", "unknown")
			channels[ch] = channels.get(ch, 0) + 1
		return {
			"worker_id": worker_id,
			"total_advisories": len(advisories),
			"unique_farmers_reached": farmers_reached,
			"follow_up_required": len(follow_ups),
			"follow_up_completed": len(completed),
			"follow_up_rate_pct": round(len(completed) / len(follow_ups) * 100, 1) if follow_ups else 100,
			"channel_breakdown": channels,
		}

	# ------------------------------------------------------------------ demo plots

	async def list_demo_plots(self, extension_worker_id: str | None = None,
							crop_type: str | None = None) -> list[dict[str, Any]]:
		items = list(self._demo_plots.values())
		if extension_worker_id:
			items = [d for d in items if d.get("extension_worker_id") == extension_worker_id]
		if crop_type:
			items = [d for d in items if d.get("crop_type") == crop_type]
		return items

	async def get_demo_plot(self, plot_id: str) -> dict[str, Any]:
		if plot_id not in self._demo_plots:
			raise KeyError(f"demo_plot_not_found:{plot_id}")
		return self._demo_plots[plot_id]

	async def create_demo_plot(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			pid = _new_id("dmo")
			ts = _now()
			record: dict[str, Any] = {
				"id": pid,
				"tenant_id": self.tenant_id,
				"name": payload["name"],
				"farm_parcel_id": payload["farm_parcel_id"],
				"extension_worker_id": payload["extension_worker_id"],
				"crop_type": payload["crop_type"],
				"variety": payload.get("variety"),
				"demonstration_topic": payload["demonstration_topic"],
				"start_date": payload["start_date"],
				"end_date": payload.get("end_date"),
				"target_farmers": list(payload.get("target_farmers", [])),
				"farmer_visits": 0,
				"outcome": None,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._demo_plots[pid] = record
			self._emit("demo_plot.created", "demo_plot", pid, record)
			return record
		except Exception as exc:
			_log.error("create_demo_plot failed: %s", exc)
			raise

	async def update_demo_plot(self, plot_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if plot_id not in self._demo_plots:
				raise KeyError(f"demo_plot_not_found:{plot_id}")
			record = self._demo_plots[plot_id]
			for field in ["end_date", "farmer_visits", "outcome", "notes", "target_farmers"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("demo_plot.updated", "demo_plot", plot_id, payload)
			return record
		except Exception as exc:
			_log.error("update_demo_plot failed: %s", exc)
			raise

	async def delete_demo_plot(self, plot_id: str) -> dict[str, Any]:
		try:
			if plot_id not in self._demo_plots:
				raise KeyError(f"demo_plot_not_found:{plot_id}")
			self._demo_plots.pop(plot_id)
			self._emit("demo_plot.deleted", "demo_plot", plot_id, {"id": plot_id})
			return {"deleted": True, "id": plot_id}
		except Exception as exc:
			_log.error("delete_demo_plot failed: %s", exc)
			raise

	# ------------------------------------------------------------------ trainings

	async def list_trainings(self, trainer_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		items = list(self._trainings.values())
		if trainer_id:
			items = [t for t in items if t.get("trainer_id") == trainer_id]
		if status:
			items = [t for t in items if t.get("status") == status]
		return items

	async def get_training(self, training_id: str) -> dict[str, Any]:
		if training_id not in self._trainings:
			raise KeyError(f"training_not_found:{training_id}")
		return self._trainings[training_id]

	async def create_training(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			tid = _new_id("trn")
			ts = _now()
			record: dict[str, Any] = {
				"id": tid,
				"tenant_id": self.tenant_id,
				"title": payload["title"],
				"trainer_id": payload["trainer_id"],
				"topic": payload["topic"],
				"scheduled_date": payload["scheduled_date"],
				"location": payload["location"],
				"participant_ids": list(payload.get("participant_ids", [])),
				"max_participants": int(payload.get("max_participants", 50)),
				"status": "scheduled",
				"actual_attendance": 0,
				"notes": payload.get("notes"),
				"created_at": ts,
				"updated_at": ts,
			}
			self._trainings[tid] = record
			self._emit("training.created", "training", tid, record)
			return record
		except Exception as exc:
			_log.error("create_training failed: %s", exc)
			raise

	async def update_training(self, training_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if training_id not in self._trainings:
				raise KeyError(f"training_not_found:{training_id}")
			record = self._trainings[training_id]
			for field in ["status", "actual_attendance", "notes", "participant_ids", "scheduled_date"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("training.updated", "training", training_id, payload)
			return record
		except Exception as exc:
			_log.error("update_training failed: %s", exc)
			raise

	async def delete_training(self, training_id: str) -> dict[str, Any]:
		try:
			if training_id not in self._trainings:
				raise KeyError(f"training_not_found:{training_id}")
			self._trainings.pop(training_id)
			self._emit("training.deleted", "training", training_id, {"id": training_id})
			return {"deleted": True, "id": training_id}
		except Exception as exc:
			_log.error("delete_training failed: %s", exc)
			raise

	async def get_training_reach(self) -> dict[str, Any]:
		"""Compute total and unique farmers trained."""
		all_participants: set[str] = set()
		for t in self._trainings.values():
			if t.get("status") == "completed":
				all_participants.update(t.get("participant_ids", []))
		return {
			"total_trainings": len(self._trainings),
			"completed_trainings": len([t for t in self._trainings.values() if t.get("status") == "completed"]),
			"unique_participants": len(all_participants),
			"total_training_hours": len([t for t in self._trainings.values() if t.get("status") == "completed"]),
		}

	# ------------------------------------------------------------------ knowledge base

	async def list_knowledge(self, category: str | None = None, crop_type: str | None = None,
							language: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		items = list(self._knowledge.values())
		if category:
			items = [k for k in items if k.get("category") == category]
		if crop_type:
			items = [k for k in items if crop_type in k.get("crop_types", [])]
		if language:
			items = [k for k in items if k.get("language") == language]
		return items[offset: offset + limit]

	async def get_knowledge_article(self, article_id: str) -> dict[str, Any]:
		if article_id not in self._knowledge:
			raise KeyError(f"article_not_found:{article_id}")
		self._knowledge[article_id]["views"] = self._knowledge[article_id].get("views", 0) + 1
		return self._knowledge[article_id]

	async def create_knowledge_article(self, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			kid = _new_id("knw")
			ts = _now()
			record: dict[str, Any] = {
				"id": kid,
				"tenant_id": self.tenant_id,
				"title": payload["title"],
				"category": payload["category"],
				"content": payload["content"],
				"crop_types": list(payload.get("crop_types", [])),
				"author_id": payload.get("author_id"),
				"tags": list(payload.get("tags", [])),
				"language": payload.get("language", "en"),
				"views": 0,
				"created_at": ts,
				"updated_at": ts,
			}
			self._knowledge[kid] = record
			self._emit("knowledge.created", "knowledge_article", kid, record)
			return record
		except Exception as exc:
			_log.error("create_knowledge_article failed: %s", exc)
			raise

	async def update_knowledge_article(self, article_id: str, payload: dict[str, Any]) -> dict[str, Any]:
		try:
			if article_id not in self._knowledge:
				raise KeyError(f"article_not_found:{article_id}")
			record = self._knowledge[article_id]
			for field in ["title", "content", "crop_types", "tags", "language"]:
				if field in payload and payload[field] is not None:
					record[field] = payload[field]
			record["updated_at"] = _now()
			self._emit("knowledge.updated", "knowledge_article", article_id, payload)
			return record
		except Exception as exc:
			_log.error("update_knowledge_article failed: %s", exc)
			raise

	async def delete_knowledge_article(self, article_id: str) -> dict[str, Any]:
		try:
			if article_id not in self._knowledge:
				raise KeyError(f"article_not_found:{article_id}")
			self._knowledge.pop(article_id)
			self._emit("knowledge.deleted", "knowledge_article", article_id, {"id": article_id})
			return {"deleted": True, "id": article_id}
		except Exception as exc:
			_log.error("delete_knowledge_article failed: %s", exc)
			raise

	async def search_knowledge(self, query: str, language: str | None = None) -> list[dict[str, Any]]:
		"""Simple text search across knowledge base titles and content."""
		q = query.lower()
		items = list(self._knowledge.values())
		if language:
			items = [k for k in items if k.get("language") == language]
		results = [k for k in items
				if q in k.get("title", "").lower() or q in k.get("content", "").lower()
				or any(q in t.lower() for t in k.get("tags", []))]
		return results

	async def get_extension_reach_summary(self) -> dict[str, Any]:
		"""High-level extension programme summary."""
		farmers_advised = len({a.get("farmer_id") for a in self._advisories.values()})
		workers = len({a.get("extension_worker_id") for a in self._advisories.values()})
		return {
			"total_advisories": len(self._advisories),
			"unique_farmers_advised": farmers_advised,
			"extension_workers_active": workers,
			"demo_plots": len(self._demo_plots),
			"trainings": len(self._trainings),
			"knowledge_articles": len(self._knowledge),
			"pending_follow_ups": len([a for a in self._advisories.values()
									if a.get("follow_up_required") and not a.get("follow_up_done")]),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_advisories', '_demo_plots', '_trainings', '_knowledge', '_audit']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

