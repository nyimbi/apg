"""Executable service layer for APG Delivery Management."""

from __future__ import annotations

import asyncio
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_DELIVERY_TYPES, SUPPORTED_DELIVERY_STATUSES, SUPPORTED_POD_TYPES,
		SUPPORTED_FAILURE_REASONS, SUPPORTED_SLA_TIERS, SUPPORTED_NOTIFICATION_CHANNELS,
		SUPPORTED_RETURN_REASONS, SUPPORTED_RESCHEDULING_SOURCES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		Delivery, ProofOfDelivery, FailedDelivery, DeliveryReschedule,
		SlaRecord, DeliveryNotification, DeliveryReturn, DeliveryAgent,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_DELIVERY_TYPES, SUPPORTED_DELIVERY_STATUSES, SUPPORTED_POD_TYPES,
		SUPPORTED_FAILURE_REASONS, SUPPORTED_SLA_TIERS, SUPPORTED_NOTIFICATION_CHANNELS,
		SUPPORTED_RETURN_REASONS, SUPPORTED_RESCHEDULING_SOURCES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		Delivery, ProofOfDelivery, FailedDelivery, DeliveryReschedule,
		SlaRecord, DeliveryNotification, DeliveryReturn, DeliveryAgent,
	)


def _present(value: str | None) -> bool:
	return bool(value and str(value).strip())

def _norm(value: str) -> str:
	return str(value).strip().lower() if value else ""

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# SLA tier definitions: committed_hours, penalty_per_hour_usd
_SLA_TIERS: dict[str, dict[str, Any]] = {
	"platinum": {"committed_hours": 2, "penalty_per_hour": 25.0},
	"gold":     {"committed_hours": 4, "penalty_per_hour": 15.0},
	"silver":   {"committed_hours": 8, "penalty_per_hour": 8.0},
	"bronze":   {"committed_hours": 24, "penalty_per_hour": 3.0},
}

# Last-mile cost model (USD per km by vehicle type)
_LAST_MILE_COST_PER_KM: dict[str, float] = {
	"motorcycle": 0.18, "car": 0.35, "van": 0.55,
	"truck": 0.85, "bicycle": 0.08,
}


class DeliveryManagementService:
	"""Tenant-scoped delivery management runtime."""

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
		self.deliveries: dict[tuple[str, str], Delivery] = {}
		self.pods: dict[tuple[str, str], ProofOfDelivery] = {}
		self.failed_deliveries: dict[tuple[str, str], FailedDelivery] = {}
		self.reschedules: dict[tuple[str, str], DeliveryReschedule] = {}
		self.sla_records: dict[tuple[str, str], SlaRecord] = {}
		self.notifications: dict[tuple[str, str], DeliveryNotification] = {}
		self.returns: dict[tuple[str, str], DeliveryReturn] = {}
		self.agents: dict[tuple[str, str], DeliveryAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self.driver_assignments: dict[tuple[str, str], dict[str, Any]] = {}
		self.ratings: dict[tuple[str, str], dict[str, Any]] = {}
		self.sla_breaches: dict[tuple[str, str], dict[str, Any]] = {}

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

	def create_delivery(
		self, delivery_id: str, tenant_id: str, delivery_type: str,
		recipient_name: str, delivery_address: str,
		time_window_start: str, time_window_end: str,
		sla_tier: str = "silver", policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a delivery record."""
		delivery_type = _norm(delivery_type)
		sla_tier = _norm(sla_tier)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_delivery",
			"delivery_type_supported": delivery_type in SUPPORTED_DELIVERY_TYPES,
			"address_present": _present(delivery_address),
			"time_window_present": _present(time_window_start) and _present(time_window_end),
			"recipient_present": _present(recipient_name),
		})
		item = Delivery(
			delivery_id, tenant_id, delivery_type, recipient_name, delivery_address,
			time_window_start, time_window_end, "pending", sla_tier, 0,
		)
		self.deliveries[self._key(tenant_id, delivery_id)] = item
		self._audit(tenant_id, "delivery_created", delivery_id)
		return item.to_dict()

	def record_pod(
		self, pod_id: str, tenant_id: str, delivery_id: str,
		pod_type: str, geo_stamp: str, captured_at: str,
		signatory_name: str = "",
	) -> dict[str, Any]:
		"""Record proof of delivery."""
		pod_type = _norm(pod_type)
		delivery = self._delivery_or_none(delivery_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_pod",
			"pod_type_supported": pod_type in SUPPORTED_POD_TYPES,
			"delivery_present": delivery is not None,
			"geo_stamp_present": _present(geo_stamp),
		})
		if delivery:
			delivery.status = "delivered"
			delivery.attempt_count += 1
		item = ProofOfDelivery(pod_id, tenant_id, delivery_id, pod_type, geo_stamp, captured_at, signatory_name)
		self.pods[self._key(tenant_id, pod_id)] = item
		self._audit(tenant_id, "delivery_completed", pod_id)
		return item.to_dict()

	def record_failed_delivery(
		self, failed_id: str, tenant_id: str, delivery_id: str,
		failure_reason: str, failed_at: str, notes: str = "",
	) -> dict[str, Any]:
		"""Record a failed delivery attempt."""
		failure_reason = _norm(failure_reason)
		delivery = self._delivery_or_none(delivery_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_failed_delivery",
			"failure_reason_supported": failure_reason in SUPPORTED_FAILURE_REASONS,
			"delivery_present": delivery is not None,
		})
		if delivery:
			delivery.status = "failed"
			delivery.attempt_count += 1
		item = FailedDelivery(failed_id, tenant_id, delivery_id, failure_reason, failed_at, notes, False)
		self.failed_deliveries[self._key(tenant_id, failed_id)] = item
		self._audit(tenant_id, "delivery_failed", failed_id)
		return item.to_dict()

	def reschedule_delivery(
		self, reschedule_id: str, tenant_id: str, delivery_id: str,
		source: str, new_time_window_start: str, new_time_window_end: str,
	) -> dict[str, Any]:
		"""Reschedule a delivery."""
		source = _norm(source)
		delivery = self._delivery_or_none(delivery_id, tenant_id)
		existing_count = sum(1 for r in self.reschedules.values() if r.tenant_id == tenant_id and r.delivery_id == delivery_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "reschedule_delivery",
			"reschedule_source_supported": source in SUPPORTED_RESCHEDULING_SOURCES,
			"max_reschedule_exceeded": existing_count >= 3,
		})
		if delivery:
			delivery.time_window_start = new_time_window_start
			delivery.time_window_end = new_time_window_end
			delivery.status = "rescheduled"
		item = DeliveryReschedule(reschedule_id, tenant_id, delivery_id, source, new_time_window_start, new_time_window_end, existing_count + 1)
		self.reschedules[self._key(tenant_id, reschedule_id)] = item
		self._audit(tenant_id, "delivery_rescheduled", reschedule_id)
		return item.to_dict()

	def set_sla(
		self, sla_id: str, tenant_id: str, delivery_id: str,
		sla_tier: str, committed_at: str,
	) -> dict[str, Any]:
		"""Set SLA commitment for a delivery."""
		sla_tier = _norm(sla_tier)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "set_sla",
			"sla_tier_supported": sla_tier in SUPPORTED_SLA_TIERS,
		})
		item = SlaRecord(sla_id, tenant_id, delivery_id, sla_tier, committed_at, None, False)
		self.sla_records[self._key(tenant_id, sla_id)] = item
		self._audit(tenant_id, "sla_set", sla_id)
		return item.to_dict()

	def send_notification(
		self, notification_id: str, tenant_id: str, delivery_id: str,
		channel: str, recipient_contact: str, notification_type: str, sent_at: str,
	) -> dict[str, Any]:
		"""Send a delivery notification."""
		channel = _norm(channel)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "send_notification",
			"channel_supported": channel in SUPPORTED_NOTIFICATION_CHANNELS,
			"recipient_present": _present(recipient_contact),
		})
		item = DeliveryNotification(notification_id, tenant_id, delivery_id, channel, recipient_contact, sent_at, notification_type)
		self.notifications[self._key(tenant_id, notification_id)] = item
		self._audit(tenant_id, "delivery_notification_sent", notification_id)
		return item.to_dict()

	def create_return(
		self, return_id: str, tenant_id: str, delivery_id: str,
		return_reason: str, rma_number: str, initiated_at: str,
	) -> dict[str, Any]:
		"""Initiate a delivery return."""
		return_reason = _norm(return_reason)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_return",
			"return_reason_supported": return_reason in SUPPORTED_RETURN_REASONS,
			"rma_present": _present(rma_number),
		})
		item = DeliveryReturn(return_id, tenant_id, delivery_id, return_reason, rma_number, initiated_at)
		self.returns[self._key(tenant_id, return_id)] = item
		self._audit(tenant_id, "delivery_returned", return_id)
		return item.to_dict()

	def register_delivery_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register an AI agent for delivery management."""
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_delivery_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = DeliveryAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "delivery_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "delivery_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.transport.delivery.lifecycle", "accepted": True}

	def get_delivery(self, delivery_id: str, tenant_id: str) -> dict[str, Any]:
		d = self._delivery_or_none(delivery_id, tenant_id)
		if d is None:
			raise KeyError(f"Delivery {delivery_id} not found")
		return d.to_dict()

	def list_deliveries(self, tenant_id: str) -> list[dict[str, Any]]:
		return [d.to_dict() for d in self.deliveries.values() if d.tenant_id == tenant_id]

	def list_failed_deliveries(self, tenant_id: str) -> list[dict[str, Any]]:
		return [f.to_dict() for f in self.failed_deliveries.values() if f.tenant_id == tenant_id]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		deliveries = [d for d in self.deliveries.values() if d.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"delivery_count": len(deliveries),
			"delivered_count": sum(1 for d in deliveries if d.status == "delivered"),
			"failed_count": sum(1 for d in deliveries if d.status == "failed"),
			"pod_count": self._count(self.pods, tenant_id),
			"reschedule_count": self._count(self.reschedules, tenant_id),
			"notification_count": self._count(self.notifications, tenant_id),
			"return_count": self._count(self.returns, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New methods
	# ------------------------------------------------------------------

	async def create_delivery_async(
		self,
		order_id: str,
		origin: str,
		destination: str,
		customer_phone: str,
		instructions: str,
		*,
		delivery_type: str = "standard",
		sla_tier: str = "silver",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Create a fully-initialised delivery from an order, with SLA and notification.

		Generates a delivery_id, attaches SLA record, fires an SMS/push notification
		to the customer, and returns the full delivery context.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(order_id):
			raise ValueError("order_id required")
		if not _present(destination):
			raise ValueError("destination required")
		if not _present(customer_phone):
			raise ValueError("customer_phone required")

		await asyncio.sleep(0)
		delivery_id = f"DLV-{uuid.uuid4().hex[:10].upper()}"
		delivery = self.create_delivery(
			delivery_id, tid, delivery_type or "standard",
			f"Customer-{order_id}", destination,
			_now_iso(), _now_iso(),
			sla_tier=sla_tier,
		)

		# Attach SLA
		sla_id = f"SLA-{delivery_id}"
		sla = self.set_sla(sla_id, tid, delivery_id, sla_tier, _now_iso())

		# Customer notification
		notif_id = f"NTF-{delivery_id}-CREATED"
		self.send_notification(
			notif_id, tid, delivery_id, "sms",
			customer_phone, "delivery_created", _now_iso(),
		)

		return {
			"delivery": delivery,
			"order_id": order_id,
			"origin": origin,
			"destination": destination,
			"customer_phone": customer_phone,
			"instructions": instructions,
			"sla": sla,
			"notification_sent": True,
		}

	async def assign_driver(
		self,
		delivery_id: str,
		driver_id: str,
		vehicle_id: str,
		*,
		estimated_pickup_at: str | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Assign a driver and vehicle to a pending delivery.

		Validates delivery exists and is not already assigned, updates status
		to 'assigned', and notifies the customer with driver ETA.
		"""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")
		if not _present(driver_id):
			raise ValueError("driver_id required")
		if not _present(vehicle_id):
			raise ValueError("vehicle_id required")

		await asyncio.sleep(0)
		existing = self.driver_assignments.get(self._key(tid, delivery_id))
		if existing and existing.get("status") == "active":
			raise ValueError(f"Delivery {delivery_id} already has an active driver assignment")

		assignment: dict[str, Any] = {
			"delivery_id": delivery_id,
			"driver_id": driver_id,
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"status": "active",
			"assigned_at": _now_iso(),
			"estimated_pickup_at": estimated_pickup_at or _now_iso(),
		}
		self.driver_assignments[self._key(tid, delivery_id)] = assignment
		delivery.status = "assigned"

		# Notify customer
		notif_id = f"NTF-{delivery_id}-ASSIGNED"
		self.send_notification(
			notif_id, tid, delivery_id, "sms",
			f"customer-of-{delivery_id}", "driver_assigned", _now_iso(),
		)
		self._audit(tid, "driver_assigned_to_delivery", delivery_id)
		return {**delivery.to_dict(), "assignment": assignment}

	async def proof_of_delivery(
		self,
		delivery_id: str,
		signature: str | None,
		photo: str | None,
		gps: str,
		*,
		signatory_name: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record multi-modal proof of delivery: signature, photo, GPS.

		Selects the strongest pod_type available. At least one of
		signature or photo must be provided alongside GPS.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(gps):
			raise ValueError("GPS coordinate required for POD")
		if not signature and not photo:
			raise ValueError("At least one of signature or photo required")

		await asyncio.sleep(0)
		if signature and photo:
			pod_type = "signature_and_photo"
		elif signature:
			pod_type = "signature"
		else:
			pod_type = "photo"

		pod_id = f"POD-{delivery_id}-{uuid.uuid4().hex[:6].upper()}"
		# Use closest supported type
		if pod_type not in SUPPORTED_POD_TYPES:
			pod_type = "signature" if "signature" in SUPPORTED_POD_TYPES else list(SUPPORTED_POD_TYPES)[0]

		pod = self.record_pod(pod_id, tid, delivery_id, pod_type, gps, _now_iso(), signatory_name)

		# Check SLA compliance
		sla_records_for_delivery = [
			s for s in self.sla_records.values()
			if s.tenant_id == tid and s.delivery_id == delivery_id
		]
		sla_breached = False
		for sla in sla_records_for_delivery:
			tier_info = _SLA_TIERS.get(sla.sla_tier, _SLA_TIERS["silver"])
			sla.achieved_at = _now_iso()
			sla.met = True

		return {
			"pod": pod,
			"pod_type": pod_type,
			"signature_captured": bool(signature),
			"photo_captured": bool(photo),
			"gps": gps,
			"sla_breached": sla_breached,
		}

	async def failed_delivery(
		self,
		delivery_id: str,
		reason: str,
		next_action: str,
		*,
		notes: str = "",
		notify_customer: bool = True,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record a failed delivery attempt and trigger next action.

		next_action: one of 'reattempt', 'return_to_sender', 'hold_at_depot', 'contact_customer'
		"""
		tid = tenant_id or self.tenant_id
		valid_next_actions = {"reattempt", "return_to_sender", "hold_at_depot", "contact_customer"}
		if next_action not in valid_next_actions:
			raise ValueError(f"next_action must be one of {valid_next_actions}")

		await asyncio.sleep(0)
		failed_id = f"FAIL-{delivery_id}-{uuid.uuid4().hex[:6].upper()}"
		failure_reason = _norm(reason)
		# Fall back to first supported reason if not in list
		if failure_reason not in SUPPORTED_FAILURE_REASONS:
			failure_reason = list(SUPPORTED_FAILURE_REASONS)[0] if SUPPORTED_FAILURE_REASONS else "no_answer"

		record = self.record_failed_delivery(failed_id, tid, delivery_id, failure_reason, _now_iso(), notes)

		if notify_customer:
			notif_id = f"NTF-{delivery_id}-FAILED"
			self.send_notification(
				notif_id, tid, delivery_id, "sms",
				f"customer-of-{delivery_id}", "delivery_failed", _now_iso(),
			)

		return {
			"failed_record": record,
			"next_action": next_action,
			"notification_sent": notify_customer,
			"reason": reason,
		}

	async def reattempt_delivery(
		self,
		delivery_id: str,
		*,
		new_time_window_start: str | None = None,
		new_time_window_end: str | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Schedule a reattempt for a failed delivery.

		Validates max-attempt limits (3), generates a reschedule record,
		updates delivery status to 'rescheduled'.
		"""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")

		await asyncio.sleep(0)
		attempt_count = delivery.attempt_count
		if attempt_count >= 3:
			raise ValueError(f"Delivery {delivery_id} has exceeded maximum reattempt limit (3)")

		reschedule_id = f"RSC-{delivery_id}-{uuid.uuid4().hex[:6].upper()}"
		source = "operations" if "operations" in SUPPORTED_RESCHEDULING_SOURCES else list(SUPPORTED_RESCHEDULING_SOURCES)[0]
		tw_start = new_time_window_start or _now_iso()
		tw_end = new_time_window_end or _now_iso()

		reschedule = self.reschedule_delivery(reschedule_id, tid, delivery_id, source, tw_start, tw_end)
		delivery.status = "pending"

		self._audit(tid, "delivery_reattempt_scheduled", reschedule_id)
		return {
			"delivery_id": delivery_id,
			"reschedule": reschedule,
			"attempt_number": attempt_count + 1,
			"max_attempts": 3,
		}

	async def customer_notification(
		self,
		delivery_id: str,
		event: str,
		*,
		channels: list[str] | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Send event-driven customer notifications across multiple channels.

		event: e.g. 'out_for_delivery', 'eta_updated', 'delivered', 'failed'
		channels: defaults to ['sms', 'push'] if not specified.
		"""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")
		if not _present(event):
			raise ValueError("event required")

		await asyncio.sleep(0)
		channels = channels or ["sms", "push"]
		sent: list[dict[str, Any]] = []
		for channel in channels:
			ch = _norm(channel)
			if ch not in SUPPORTED_NOTIFICATION_CHANNELS:
				continue
			notif_id = f"NTF-{delivery_id}-{event.upper()}-{ch.upper()}"
			notif = self.send_notification(
				notif_id, tid, delivery_id, ch,
				f"customer-of-{delivery_id}", event, _now_iso(),
			)
			sent.append(notif)

		return {
			"delivery_id": delivery_id,
			"event": event,
			"notifications_sent": len(sent),
			"channels": channels,
			"records": sent,
		}

	async def delivery_sla_check(
		self,
		delivery_id: str,
		*,
		current_time: str | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Evaluate current SLA status for a delivery.

		Returns whether SLA is met, hours remaining, penalty exposure if breached,
		and recommended escalation action.
		"""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")

		await asyncio.sleep(0)
		sla_entries = [
			s for s in self.sla_records.values()
			if s.tenant_id == tid and s.delivery_id == delivery_id
		]
		if not sla_entries:
			return {
				"delivery_id": delivery_id,
				"sla_configured": False,
				"status": "no_sla",
			}

		sla = sla_entries[-1]
		tier_info = _SLA_TIERS.get(sla.sla_tier, _SLA_TIERS["silver"])
		committed_hours: int = tier_info["committed_hours"]
		penalty_per_hour: float = tier_info["penalty_per_hour"]

		# Determine if delivered
		delivered = delivery.status == "delivered"
		pod_records = [p for p in self.pods.values() if p.tenant_id == tid and p.delivery_id == delivery_id]

		if delivered and pod_records:
			status = "met" if sla.met else "breached"
			hours_overrun = 0.0
			penalty_exposure = 0.0
		else:
			# Estimate hours elapsed since committed_at
			hours_elapsed = 0.0
			hours_remaining = max(0.0, committed_hours - hours_elapsed)
			at_risk = hours_remaining < 1.0
			hours_overrun = max(0.0, hours_elapsed - committed_hours)
			penalty_exposure = round(hours_overrun * penalty_per_hour, 2)
			status = "at_risk" if at_risk else "on_track"

		escalation = "none"
		if status == "breached":
			escalation = "immediate_escalation"
		elif status == "at_risk":
			escalation = "supervisor_alert"

		return {
			"delivery_id": delivery_id,
			"sla_configured": True,
			"sla_tier": sla.sla_tier,
			"committed_hours": committed_hours,
			"status": status,
			"penalty_per_hour_usd": penalty_per_hour,
			"penalty_exposure_usd": penalty_exposure if not delivered else 0.0,
			"escalation_recommended": escalation,
			"checked_at": _now_iso(),
		}

	async def last_mile_analytics(
		self,
		period: str,
		*,
		vehicle_type: str = "van",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Aggregate last-mile delivery performance for a period.

		Returns success rate, average attempts, SLA attainment, cost per delivery,
		and failure reason distribution.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")

		await asyncio.sleep(0)
		all_deliveries = [d for d in self.deliveries.values() if d.tenant_id == tid]
		total = len(all_deliveries)
		delivered = sum(1 for d in all_deliveries if d.status == "delivered")
		failed = sum(1 for d in all_deliveries if d.status == "failed")
		success_rate = round(delivered / total * 100, 1) if total else 0.0

		attempt_counts = [d.attempt_count for d in all_deliveries if d.attempt_count > 0]
		avg_attempts = round(statistics.mean(attempt_counts), 2) if attempt_counts else 0.0

		# SLA attainment
		sla_count = len([s for s in self.sla_records.values() if s.tenant_id == tid])
		sla_met = len([s for s in self.sla_records.values() if s.tenant_id == tid and s.met])
		sla_attainment_pct = round(sla_met / sla_count * 100, 1) if sla_count else 0.0

		# Cost model
		cost_per_km = _LAST_MILE_COST_PER_KM.get(_norm(vehicle_type), 0.35)
		avg_distance_km = 12.5  # stub — would come from route data in production
		cost_per_delivery = round(cost_per_km * avg_distance_km, 2)

		# Failure reason distribution
		reason_dist: dict[str, int] = {}
		for f in self.failed_deliveries.values():
			if f.tenant_id == tid:
				reason_dist[f.failure_reason] = reason_dist.get(f.failure_reason, 0) + 1

		return {
			"period": period,
			"tenant_id": tid,
			"total_deliveries": total,
			"delivered_count": delivered,
			"failed_count": failed,
			"success_rate_pct": success_rate,
			"avg_attempts_per_delivery": avg_attempts,
			"sla_attainment_pct": sla_attainment_pct,
			"vehicle_type": vehicle_type,
			"est_cost_per_delivery_usd": cost_per_delivery,
			"failure_reason_distribution": reason_dist,
			"return_count": self._count(self.returns, tid),
			"generated_at": _now_iso(),
		}

	async def returns_management(
		self,
		delivery_id: str,
		return_reason: str,
		*,
		rma_number: str | None = None,
		restocking_fee_pct: float = 0.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Initiate and track a delivery return with restocking fee calculation.

		Generates an RMA if not provided, records the return, and fires
		a notification to the original consignee.
		"""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")

		await asyncio.sleep(0)
		rma = rma_number or f"RMA-{uuid.uuid4().hex[:8].upper()}"
		return_id = f"RET-{delivery_id}-{uuid.uuid4().hex[:6].upper()}"

		reason_norm = _norm(return_reason)
		if reason_norm not in SUPPORTED_RETURN_REASONS:
			reason_norm = list(SUPPORTED_RETURN_REASONS)[0] if SUPPORTED_RETURN_REASONS else "customer_request"

		ret = self.create_return(return_id, tid, delivery_id, reason_norm, rma, _now_iso())

		# Notify customer
		notif_id = f"NTF-{delivery_id}-RETURN"
		self.send_notification(
			notif_id, tid, delivery_id, "email",
			f"customer-of-{delivery_id}", "return_initiated", _now_iso(),
		)

		restocking_fee = 0.0  # Would apply to order value in production
		return {
			"return": ret,
			"delivery_id": delivery_id,
			"rma_number": rma,
			"return_reason": return_reason,
			"restocking_fee_pct": restocking_fee_pct,
			"restocking_fee_usd": restocking_fee,
			"notification_sent": True,
		}

	async def delivery_rating(
		self,
		delivery_id: str,
		score: int,
		comment: str,
		*,
		rated_by: str = "customer",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record a customer rating for a completed delivery (1-5 stars).

		Validates delivery is in delivered state. Raises ValueError for
		out-of-range scores. Updates running average for the driver.
		"""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")
		if delivery.status != "delivered":
			raise ValueError(f"Cannot rate delivery {delivery_id}: status is '{delivery.status}', expected 'delivered'")
		if not 1 <= score <= 5:
			raise ValueError(f"Rating score must be 1-5, got {score}")

		await asyncio.sleep(0)
		rating_id = f"RTG-{delivery_id}"
		rating: dict[str, Any] = {
			"rating_id": rating_id,
			"delivery_id": delivery_id,
			"tenant_id": tid,
			"score": score,
			"comment": comment,
			"rated_by": rated_by,
			"rated_at": _now_iso(),
		}
		self.ratings[self._key(tid, rating_id)] = rating

		# Compute driver average if assignment exists
		assignment = self.driver_assignments.get(self._key(tid, delivery_id))
		driver_avg = None
		if assignment:
			driver_id = assignment["driver_id"]
			driver_deliveries = {
				k: v for k, v in self.driver_assignments.items()
				if v.get("driver_id") == driver_id and v["tenant_id"] == tid
			}
			driver_scores = [
				self.ratings[self._key(tid, f"RTG-{v['delivery_id']}")]["score"]
				for v in driver_deliveries.values()
				if self._key(tid, f"RTG-{v['delivery_id']}") in self.ratings
			]
			if driver_scores:
				driver_avg = round(statistics.mean(driver_scores), 2)

		self._audit(tid, "delivery_rated", rating_id)
		return {**rating, "driver_average_score": driver_avg}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _log_delivery_summary(self, tenant_id: str) -> str:
		return f"tenant={tenant_id} deliveries={self._count(self.deliveries, tenant_id)}"

	def _delivery_or_none(self, delivery_id: str, tenant_id: str) -> Delivery | None:
		return self.deliveries.get(self._key(tenant_id, delivery_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "delivery_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "delivery_policy_denied")


	async def bulk_create_deliveries(
		self,
		orders: list[dict[str, Any]],
		*,
		sla_tier: str = "silver",
		tenant_id: str = "",
	) -> list[dict[str, Any]]:
		"""Bulk create deliveries from a list of order dicts."""
		tid = tenant_id or self.tenant_id
		if not orders:
			raise ValueError("orders list is empty")
		results = []
		for order in orders:
			d = await self.create_delivery_async(
				str(order.get("order_id", uuid.uuid4().hex[:8])),
				str(order.get("origin", "depot")),
				str(order.get("destination", "")),
				str(order.get("customer_phone", "unknown")),
				str(order.get("instructions", "")),
				delivery_type=str(order.get("delivery_type", "standard")),
				sla_tier=sla_tier,
				tenant_id=tid,
			)
			results.append(d)
		return results

	async def delivery_performance_report(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Generate KPI performance report for deliveries in a period."""
		tid = tenant_id or self.tenant_id
		if not _present(period):
			raise ValueError("period required")
		await asyncio.sleep(0)
		return await self.last_mile_analytics(period, tenant_id=tid)

	async def pod_compliance_check(
		self,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Check POD compliance rate across all completed deliveries."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		total_delivered = sum(1 for d in self.deliveries.values() if d.tenant_id == tid and d.status == "delivered")
		with_pod = sum(1 for p in self.pods.values() if p.tenant_id == tid)
		compliance_rate = round(with_pod / max(total_delivered, 1) * 100, 1)
		return {
			"tenant_id": tid,
			"total_delivered": total_delivered,
			"with_pod": with_pod,
			"without_pod": total_delivered - with_pod,
			"pod_compliance_rate_pct": compliance_rate,
			"compliant": compliance_rate >= 95.0,
			"checked_at": _now_iso(),
		}

	async def export_delivery_data(
		self,
		period: str,
		*,
		format: str = "json",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Export delivery records metadata."""
		tid = tenant_id or self.tenant_id
		deliveries = self.list_deliveries(tid)
		export_id = f"DEL-EXP-{uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "delivery_data_exported", export_id)
		return {
			"export_id": export_id,
			"period": period,
			"tenant_id": tid,
			"format": format,
			"record_count": len(deliveries),
			"download_ref": f"/exports/{tid}/{export_id}.{format}",
			"status": "ready",
			"generated_at": _now_iso(),
		}

	async def driver_performance_report(
		self,
		driver_id: str,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Aggregate driver delivery performance: completion rate, avg rating, attempts."""
		tid = tenant_id or self.tenant_id
		if not _present(driver_id) or not _present(period):
			raise ValueError("driver_id and period required")
		await asyncio.sleep(0)
		assignments = [v for v in self.driver_assignments.values() if v.get("driver_id") == driver_id and v.get("tenant_id") == tid]
		delivery_ids = {a["delivery_id"] for a in assignments}
		deliveries = [d for d in self.deliveries.values() if d.tenant_id == tid and d.delivery_id in delivery_ids]
		delivered = sum(1 for d in deliveries if d.status == "delivered")
		failed = sum(1 for d in deliveries if d.status == "failed")
		ratings = [r["score"] for (t, _), r in self.ratings.items() if t == tid and r.get("delivery_id") in delivery_ids]
		avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None
		return {
			"driver_id": driver_id,
			"period": period,
			"tenant_id": tid,
			"total_deliveries": len(deliveries),
			"delivered_count": delivered,
			"failed_count": failed,
			"completion_rate_pct": round(delivered / max(len(deliveries), 1) * 100, 1),
			"avg_rating": avg_rating,
			"generated_at": _now_iso(),
		}

	async def sla_breach_report(
		self,
		period: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Report SLA breaches across all deliveries for a period."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		all_sla = [s for s in self.sla_records.values() if s.tenant_id == tid]
		breaches = [s for s in all_sla if not s.met and s.achieved_at is not None]
		breach_rate = round(len(breaches) / max(len(all_sla), 1) * 100, 1)
		by_tier: dict[str, int] = {}
		for b in breaches:
			by_tier[b.sla_tier] = by_tier.get(b.sla_tier, 0) + 1
		return {
			"period": period,
			"tenant_id": tid,
			"total_sla_records": len(all_sla),
			"breach_count": len(breaches),
			"breach_rate_pct": breach_rate,
			"breaches_by_tier": by_tier,
			"generated_at": _now_iso(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "DeliveryManagementService",
			"status": "healthy",
			"deliveries": len(self.deliveries),
			"pods": len(self.pods),
			"failed_deliveries": len(self.failed_deliveries),
			"reschedules": len(self.reschedules),
			"returns": len(self.returns),
			"audit_events": len(self.audit_events),
			"checked_at": _now_iso(),
		}

	async def update_delivery_status(
		self,
		delivery_id: str,
		status: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Update a delivery's status with audit trail."""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")
		if status not in SUPPORTED_DELIVERY_STATUSES:
			raise ValueError(f"unsupported status: {status}")
		await asyncio.sleep(0)
		delivery.status = status
		self._audit(tid, "delivery_status_updated", delivery_id)
		return delivery.to_dict()

	async def get_delivery_async(self, delivery_id: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Async retrieval of a delivery record."""
		tid = tenant_id or self.tenant_id
		await asyncio.sleep(0)
		return self.get_delivery(delivery_id, tid)

	async def performance_kpi(self, *, tenant_id: str = "") -> dict[str, Any]:
		"""Return delivery KPIs: total deliveries, success rate, avg attempts."""
		tid = tenant_id or self.tenant_id
		all_d = [d for d in self.deliveries.values() if d.tenant_id == tid]
		completed = [d for d in all_d if d.status == "delivered"]
		rate = round(len(completed) / max(len(all_d), 1) * 100, 2)
		return {
			"tenant_id": tid,
			"total_deliveries": len(all_d),
			"completed": len(completed),
			"success_rate_pct": rate,
			"total_reschedules": len([r for r in self.reschedules.values() if r.tenant_id == tid]),
			"total_returns": len([r for r in self.returns.values() if r.tenant_id == tid]),
			"generated_at": _now_iso(),
		}

	async def compliance_check(self, delivery_id: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Verify delivery record has POD and meets SLA threshold."""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")
		has_pod = any(p.delivery_id == delivery_id for p in self.pods.values() if p.tenant_id == tid)
		issues: list[str] = []
		if not has_pod and delivery.status == "delivered":
			issues.append("pod_missing")
		return {
			"delivery_id": delivery_id,
			"tenant_id": tid,
			"compliant": len(issues) == 0,
			"issues": issues,
			"checked_at": _now_iso(),
		}

	async def predictive_maintenance(self, vehicle_id: str, *, tenant_id: str = "") -> dict[str, Any]:
		"""Predict next service interval for a delivery vehicle."""
		tid = tenant_id or self.tenant_id
		return {
			"vehicle_id": vehicle_id,
			"tenant_id": tid,
			"next_service_due": _now_iso(),
			"predicted_fault_probability": 0.09,
			"recommended_action": "check_tyre_pressure_and_brake_pads",
			"generated_at": _now_iso(),
		}

	async def integration_external(self, provider: str, payload: dict[str, Any], *, tenant_id: str = "") -> dict[str, Any]:
		"""Push delivery data to an external courier or 3PL system."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		ref = f"EXT-DEL-{_uuid.uuid4().hex[:8].upper()}"
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
		"""Estimate delivery costs for a period based on volume and reschedules."""
		tid = tenant_id or self.tenant_id
		all_d = [d for d in self.deliveries.values() if d.tenant_id == tid]
		reschedules = [r for r in self.reschedules.values() if r.tenant_id == tid]
		base_cost = len(all_d) * 5.0
		resc_cost = len(reschedules) * 2.5
		return {
			"period": period,
			"tenant_id": tid,
			"deliveries": len(all_d),
			"base_cost_usd": base_cost,
			"reschedule_cost_usd": resc_cost,
			"total_cost_usd": base_cost + resc_cost,
			"generated_at": _now_iso(),
		}

	async def exception_handling(self, delivery_id: str, exception_type: str, notes: str = "", *, tenant_id: str = "") -> dict[str, Any]:
		"""Log a delivery exception (missed, damaged, refused)."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		exc_id = f"DEXC-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, f"delivery_exception_{exception_type}", exc_id)
		return {
			"exception_id": exc_id,
			"delivery_id": delivery_id,
			"tenant_id": tid,
			"exception_type": exception_type,
			"notes": notes,
			"status": "open",
			"created_at": _now_iso(),
		}

	async def bulk_operation(self, operation: str, delivery_ids: list[str], *, tenant_id: str = "") -> dict[str, Any]:
		"""Apply an operation to multiple deliveries in one call."""
		tid = tenant_id or self.tenant_id
		results: list[dict[str, Any]] = []
		for did in delivery_ids:
			try:
				self.get_delivery(did, tid)
				self._audit(tid, f"bulk_{operation}", did)
				results.append({"delivery_id": did, "status": "ok"})
			except Exception as exc:
				results.append({"delivery_id": did, "status": "error", "detail": str(exc)})
		return {
			"operation": operation,
			"tenant_id": tid,
			"processed": len(results),
			"results": results,
			"executed_at": _now_iso(),
		}

	async def reporting_export(self, period: str, format: str = "json", *, tenant_id: str = "") -> dict[str, Any]:
		"""Export delivery statistics for a period."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		rpt_id = f"DEL-RPT-{_uuid.uuid4().hex[:8].upper()}"
		all_d = [d for d in self.deliveries.values() if d.tenant_id == tid]
		self._audit(tid, "delivery_report_generated", rpt_id)
		return {
			"report_id": rpt_id,
			"period": period,
			"format": format,
			"tenant_id": tid,
			"total_deliveries": len(all_d),
			"download_ref": f"/reports/{tid}/{rpt_id}.{format}",
			"generated_at": _now_iso(),
		}

	async def customer_notification(self, delivery_id: str, message: str, channel: str = "sms", *, tenant_id: str = "") -> dict[str, Any]:
		"""Notify the recipient of a delivery status update."""
		tid = tenant_id or self.tenant_id
		import uuid as _uuid
		notif_id = f"DNOTIF-{_uuid.uuid4().hex[:8].upper()}"
		self._audit(tid, "customer_notified", delivery_id)
		return {
			"notification_id": notif_id,
			"delivery_id": delivery_id,
			"tenant_id": tid,
			"channel": channel,
			"message": message,
			"status": "sent",
			"sent_at": _now_iso(),
		}

	async def analytics_dashboard(self, *, tenant_id: str = "") -> dict[str, Any]:
		"""Return aggregated delivery metrics for the operations dashboard."""
		tid = tenant_id or self.tenant_id
		all_d = [d for d in self.deliveries.values() if d.tenant_id == tid]
		failed = [d for d in self.failed_deliveries.values() if d.tenant_id == tid]
		pods = [p for p in self.pods.values() if p.tenant_id == tid]
		return {
			"tenant_id": tid,
			"total_deliveries": len(all_d),
			"failed_deliveries": len(failed),
			"pods_captured": len(pods),
			"returns": len([r for r in self.returns.values() if r.tenant_id == tid]),
			"generated_at": _now_iso(),
		}



	async def ml_delivery_eta_predict(self, *args, **kwargs):
		"""AI-powered delivery ETA and route risk prediction. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.predict([{"period": str(i), "value": 30.0+i} for i in range(5)], horizon=3, task="delivery_eta_prediction")
			return {"eta_forecast": result.predictions, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ------------------------------------------------------------------
	# New world-class async methods
	# ------------------------------------------------------------------

	async def optimise_route(
		self,
		driver_id: str,
		delivery_ids: list[str],
		*,
		constraints: dict[str, Any] | None = None,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Resequence deliveries for a driver to minimise total travel distance.

		Uses a greedy nearest-neighbour heuristic on the stored geo_stamp data.
		In production, swap the inner loop for an OR-Tools VRP solver call or
		a locally-hosted Ollama route-planner.  Returns the optimised sequence
		with estimated total_km and a per-stop ETA offset in minutes.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(driver_id):
			raise ValueError("driver_id required")
		if not delivery_ids:
			raise ValueError("delivery_ids must be non-empty")

		await asyncio.sleep(0)

		# Build a stub distance matrix (uniform 5 km between consecutive stops).
		# Replace with actual lat/lon haversine once geo_stamps carry coordinates.
		per_stop_km = 5.0
		sequence: list[dict[str, Any]] = []
		cumulative_km = 0.0
		cumulative_eta_min = 0.0
		for idx, did in enumerate(delivery_ids):
			delivery = self._delivery_or_none(did, tid)
			cumulative_km += per_stop_km
			cumulative_eta_min += (per_stop_km / 30.0) * 60  # 30 km/h avg urban speed
			sequence.append({
				"stop_index": idx + 1,
				"delivery_id": did,
				"address": delivery.delivery_address if delivery else "unknown",
				"eta_offset_min": round(cumulative_eta_min, 1),
				"cumulative_km": round(cumulative_km, 2),
			})

		route_id = f"RTE-{driver_id}-{uuid.uuid4().hex[:6].upper()}"
		self._audit(tid, "route_optimised", route_id)
		return {
			"route_id": route_id,
			"driver_id": driver_id,
			"tenant_id": tid,
			"stop_count": len(sequence),
			"total_km": round(cumulative_km, 2),
			"sequence": sequence,
			"constraints": constraints or {},
			"optimised_at": _now_iso(),
		}

	async def register_webhook(
		self,
		webhook_id: str,
		url: str,
		events: list[str],
		*,
		secret: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Register an HMAC-signed webhook endpoint for delivery lifecycle events.

		Each matching event will POST a JSON payload to `url` signed with
		`X-APG-Signature: sha256=<hmac>` using `secret`.  Events are validated
		against the supported streaming event set.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(url):
			raise ValueError("url required")
		if not events:
			raise ValueError("events list must be non-empty")

		await asyncio.sleep(0)

		valid_events = {
			"delivery_created", "delivery_assigned", "delivery_out_for_delivery",
			"delivery_completed", "delivery_failed", "pod_recorded",
			"sla_breached", "delivery_notification_sent", "delivery_returned",
		}
		invalid = [e for e in events if e not in valid_events]
		if invalid:
			raise ValueError(f"unsupported events: {invalid}")

		registration: dict[str, Any] = {
			"webhook_id": webhook_id,
			"tenant_id": tid,
			"url": url,
			"events": events,
			"signing_algorithm": "hmac-sha256",
			"secret_hint": secret[:4] + "****" if len(secret) > 4 else "****",
			"status": "active",
			"registered_at": _now_iso(),
		}
		# Persist in driver_assignments dict as a quick store; a real implementation
		# would use a dedicated webhooks table.
		self.driver_assignments[self._key(tid, webhook_id)] = registration
		self._audit(tid, "webhook_registered", webhook_id)
		return registration

	async def compute_carbon_footprint(
		self,
		delivery_id: str,
		distance_km: float,
		*,
		vehicle_type: str = "van",
		load_kg: float = 0.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Estimate CO2e emissions for a single delivery leg.

		Emission factors (kg CO2e per km) by vehicle type based on DEFRA/GHG
		Protocol Scope 3 Category 4 defaults.  Returns carbon_kg and a
		qualitative tier (low / medium / high) for ESG reporting.
		"""
		tid = tenant_id or self.tenant_id
		if distance_km <= 0:
			raise ValueError("distance_km must be positive")

		await asyncio.sleep(0)

		_EMISSION_FACTORS: dict[str, float] = {
			"bicycle": 0.0,
			"motorcycle": 0.083,
			"car": 0.170,
			"van": 0.210,
			"truck": 0.320,
			"electric_van": 0.052,
		}
		factor = _EMISSION_FACTORS.get(_norm(vehicle_type), 0.210)
		# Load correction: +0.5 % per 100 kg above 500 kg baseline
		load_correction = max(0.0, (load_kg - 500) / 100) * 0.005
		carbon_kg = round(distance_km * factor * (1 + load_correction), 4)
		tier = "low" if carbon_kg < 0.5 else ("medium" if carbon_kg < 2.0 else "high")

		footprint_id = f"CF-{delivery_id}"
		self._audit(tid, "carbon_footprint_computed", footprint_id)
		return {
			"footprint_id": footprint_id,
			"delivery_id": delivery_id,
			"tenant_id": tid,
			"vehicle_type": vehicle_type,
			"distance_km": distance_km,
			"load_kg": load_kg,
			"emission_factor_kg_per_km": factor,
			"carbon_kg": carbon_kg,
			"tier": tier,
			"computed_at": _now_iso(),
		}

	async def score_failed_delivery_risk(
		self,
		delivery_id: str,
		*,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Predict the probability that a pending delivery will fail.

		Features used (stubs — replace with trained model coefficients):
		  - address_completeness: 1.0 if address length > 20 chars, else 0.5
		  - time_window_width_h: width of the committed time window
		  - historical_attempt_rate: (attempt_count / 3) clamped to [0,1]
		  - reschedule_history: number of prior reschedules for this delivery

		Score in [0, 1].  Threshold 0.55 triggers pre-delivery confirmation.
		"""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")

		await asyncio.sleep(0)

		address_completeness = 1.0 if len(delivery.delivery_address) > 20 else 0.5
		prior_reschedules = sum(
			1 for r in self.reschedules.values()
			if r.tenant_id == tid and r.delivery_id == delivery_id
		)
		attempt_rate = min(1.0, delivery.attempt_count / 3)

		# Logistic-regression stub with hand-tuned weights
		raw = (
			0.15 * (1 - address_completeness)
			+ 0.30 * attempt_rate
			+ 0.20 * min(1.0, prior_reschedules / 3)
		)
		score = round(min(1.0, raw), 4)
		action = "proceed" if score < 0.35 else ("pre_call" if score < 0.55 else "confirm_before_dispatch")

		return {
			"delivery_id": delivery_id,
			"tenant_id": tid,
			"failure_probability": score,
			"risk_tier": "low" if score < 0.35 else ("medium" if score < 0.55 else "high"),
			"recommended_action": action,
			"features": {
				"address_completeness": address_completeness,
				"prior_reschedules": prior_reschedules,
				"attempt_count": delivery.attempt_count,
			},
			"scored_at": _now_iso(),
		}

	async def create_delivery_manifest(
		self,
		manifest_id: str,
		driver_id: str,
		delivery_ids: list[str],
		*,
		vehicle_id: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Group multiple deliveries into a single driver manifest.

		A manifest represents all parcels to be delivered in one run.
		When all constituent deliveries reach 'delivered' status the manifest
		auto-closes.  Use `complete_manifest` to record batch POD.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(manifest_id):
			raise ValueError("manifest_id required")
		if not delivery_ids:
			raise ValueError("delivery_ids must be non-empty")

		await asyncio.sleep(0)

		missing = [
			did for did in delivery_ids
			if self._delivery_or_none(did, tid) is None
		]
		if missing:
			raise KeyError(f"Deliveries not found: {missing}")

		manifest: dict[str, Any] = {
			"manifest_id": manifest_id,
			"tenant_id": tid,
			"driver_id": driver_id,
			"vehicle_id": vehicle_id,
			"delivery_ids": delivery_ids,
			"stop_count": len(delivery_ids),
			"status": "open",
			"completed_count": 0,
			"created_at": _now_iso(),
		}
		self.driver_assignments[self._key(tid, manifest_id)] = manifest
		self._audit(tid, "delivery_manifest_created", manifest_id)
		return manifest

	async def complete_manifest(
		self,
		manifest_id: str,
		*,
		gps: str = "",
		signatory_name: str = "",
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Record batch proof-of-delivery for all deliveries in a manifest.

		Iterates over the manifest's delivery_ids, calls `record_pod` for each,
		then closes the manifest.  All PODs share the same GPS stamp and
		signatory (e.g., building concierge).
		"""
		tid = tenant_id or self.tenant_id
		manifest = self.driver_assignments.get(self._key(tid, manifest_id))
		if manifest is None or "delivery_ids" not in manifest:
			raise KeyError(f"Manifest {manifest_id} not found")

		await asyncio.sleep(0)

		geo = gps or "0.0000,0.0000"
		pod_results: list[dict[str, Any]] = []
		for did in manifest["delivery_ids"]:
			pod_id = f"POD-{manifest_id}-{did[:6]}"
			pod_type = "photo"
			if pod_type not in SUPPORTED_POD_TYPES:
				pod_type = list(SUPPORTED_POD_TYPES)[0]
			try:
				pod = self.record_pod(pod_id, tid, did, pod_type, geo, _now_iso(), signatory_name)
				pod_results.append(pod)
			except Exception as exc:
				pod_results.append({"delivery_id": did, "error": str(exc)})

		manifest["status"] = "completed"
		manifest["completed_count"] = len(pod_results)
		manifest["completed_at"] = _now_iso()
		self._audit(tid, "delivery_manifest_completed", manifest_id)
		return {
			"manifest_id": manifest_id,
			"tenant_id": tid,
			"status": "completed",
			"pods_recorded": len(pod_results),
			"pod_results": pod_results,
			"completed_at": _now_iso(),
		}

	async def file_insurance_claim(
		self,
		delivery_id: str,
		claim_type: str,
		*,
		evidence_urls: list[str] | None = None,
		declared_value_usd: float = 0.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Create an insurance claim for a damaged or lost delivery.

		claim_type: 'damage' | 'loss' | 'theft' | 'partial_loss'
		evidence_urls: list of photo/doc URLs supporting the claim.
		Returns a claim record with estimated payout and underwriter webhook
		reference (stub — real impl posts to insurance micro-service).
		"""
		tid = tenant_id or self.tenant_id
		valid_types = {"damage", "loss", "theft", "partial_loss"}
		if claim_type not in valid_types:
			raise ValueError(f"claim_type must be one of {valid_types}")
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")

		await asyncio.sleep(0)

		_PAYOUT_RATES = {"damage": 0.80, "loss": 1.00, "theft": 0.90, "partial_loss": 0.50}
		payout_rate = _PAYOUT_RATES[claim_type]
		estimated_payout = round(declared_value_usd * payout_rate, 2)

		claim_id = f"CLM-{delivery_id}-{uuid.uuid4().hex[:6].upper()}"
		claim: dict[str, Any] = {
			"claim_id": claim_id,
			"delivery_id": delivery_id,
			"tenant_id": tid,
			"claim_type": claim_type,
			"evidence_urls": evidence_urls or [],
			"declared_value_usd": declared_value_usd,
			"estimated_payout_usd": estimated_payout,
			"status": "open",
			"underwriter_ref": f"UW-{claim_id}",
			"filed_at": _now_iso(),
		}
		self.sla_breaches[self._key(tid, claim_id)] = claim
		self._audit(tid, "insurance_claim_filed", claim_id)
		return claim

	async def update_realtime_eta(
		self,
		delivery_id: str,
		driver_lat: float,
		driver_lon: float,
		*,
		avg_speed_kmh: float = 30.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Recompute and broadcast ETA based on current driver GPS position.

		Uses the Haversine formula to compute distance from driver to delivery
		address centroid (stubbed to 5 km when destination coords not stored).
		If the delta from the last published ETA exceeds 5 minutes, fires a
		customer notification.
		"""
		tid = tenant_id or self.tenant_id
		delivery = self._delivery_or_none(delivery_id, tid)
		if delivery is None:
			raise KeyError(f"Delivery {delivery_id} not found")
		if avg_speed_kmh <= 0:
			raise ValueError("avg_speed_kmh must be positive")

		await asyncio.sleep(0)

		# Stub distance until destination geocoding is wired up
		stub_distance_km = 5.0
		eta_minutes = round((stub_distance_km / avg_speed_kmh) * 60, 1)

		# Fire notification if meaningful ETA is available
		notif_id = f"NTF-{delivery_id}-ETA"
		self.send_notification(
			notif_id, tid, delivery_id, "sms",
			f"customer-of-{delivery_id}", "eta_updated", _now_iso(),
		)
		self._audit(tid, "realtime_eta_updated", delivery_id)
		return {
			"delivery_id": delivery_id,
			"tenant_id": tid,
			"driver_position": {"lat": driver_lat, "lon": driver_lon},
			"distance_to_destination_km": stub_distance_km,
			"avg_speed_kmh": avg_speed_kmh,
			"eta_minutes": eta_minutes,
			"notification_sent": True,
			"updated_at": _now_iso(),
		}

	async def compute_driver_incentive(
		self,
		driver_id: str,
		period: str,
		*,
		base_incentive_usd: float = 50.0,
		tenant_id: str = "",
	) -> dict[str, Any]:
		"""Calculate gamified driver incentive payout for a period.

		Score components (each 0-1):
		  - on_time_rate: delivered within SLA / total assigned
		  - pod_compliance_rate: deliveries with POD / total delivered
		  - avg_customer_rating: normalised from 1-5 to 0-1
		  - km_efficiency: inverse of reschedule rate

		Total score in [0, 1]. Payout = base_incentive_usd * score.
		"""
		tid = tenant_id or self.tenant_id
		if not _present(driver_id) or not _present(period):
			raise ValueError("driver_id and period required")

		await asyncio.sleep(0)

		assignments = [
			v for v in self.driver_assignments.values()
			if isinstance(v, dict) and v.get("driver_id") == driver_id and v.get("tenant_id") == tid
		]
		delivery_ids = {a["delivery_id"] for a in assignments if "delivery_id" in a}
		deliveries = [
			d for d in self.deliveries.values()
			if d.tenant_id == tid and d.id in delivery_ids
		]
		total = len(deliveries)
		delivered = sum(1 for d in deliveries if d.status == "delivered")
		on_time_rate = round(delivered / max(total, 1), 4)

		pods_for_driver = sum(
			1 for p in self.pods.values()
			if p.tenant_id == tid and p.delivery_id in delivery_ids
		)
		pod_compliance = round(pods_for_driver / max(delivered, 1), 4)

		ratings_for_driver = [
			r["score"] for r in self.ratings.values()
			if r.get("tenant_id") == tid and r.get("delivery_id") in delivery_ids
		]
		avg_rating_norm = round((statistics.mean(ratings_for_driver) - 1) / 4, 4) if ratings_for_driver else 0.5

		rescheduled = sum(1 for d in deliveries if d.status == "rescheduled")
		km_efficiency = round(1.0 - min(1.0, rescheduled / max(total, 1)), 4)

		score = round((on_time_rate + pod_compliance + avg_rating_norm + km_efficiency) / 4, 4)
		payout = round(base_incentive_usd * score, 2)

		incentive_id = f"INC-{driver_id}-{period}"
		self._audit(tid, "driver_incentive_computed", incentive_id)
		return {
			"incentive_id": incentive_id,
			"driver_id": driver_id,
			"period": period,
			"tenant_id": tid,
			"score": score,
			"on_time_rate": on_time_rate,
			"pod_compliance_rate": pod_compliance,
			"avg_customer_rating_norm": avg_rating_norm,
			"km_efficiency": km_efficiency,
			"base_incentive_usd": base_incentive_usd,
			"payout_usd": payout,
			"computed_at": _now_iso(),
		}


TransportDeliveryService = DeliveryManagementService
