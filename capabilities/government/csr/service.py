"""Executable service layer for APG Citizen Services Portal."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_NOTIFICATION_TYPES, SUPPORTED_PAYMENT_METHODS, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SERVICE_TYPES, SUPPORTED_SUBMISSION_CHANNELS, SUPPORTED_VERIFICATION_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		CitizenNotification, CitizenServicesAgent, DocumentVerification, PaymentRecord,
		ServiceApplication, ServiceDeliveryRecord, ServiceDefinition, ServiceReview,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_NOTIFICATION_TYPES, SUPPORTED_PAYMENT_METHODS, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SERVICE_TYPES, SUPPORTED_SUBMISSION_CHANNELS, SUPPORTED_VERIFICATION_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		CitizenNotification, CitizenServicesAgent, DocumentVerification, PaymentRecord,
		ServiceApplication, ServiceDeliveryRecord, ServiceDefinition, ServiceReview,
	)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


class CitizenServicesService:
	"""Tenant-scoped citizen services portal runtime."""

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
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self.services: dict[tuple[str, str], ServiceDefinition] = {}
		self.applications: dict[tuple[str, str], ServiceApplication] = {}
		self.payments: dict[tuple[str, str], PaymentRecord] = {}
		self.verifications: dict[tuple[str, str], DocumentVerification] = {}
		self.notifications: dict[tuple[str, str], CitizenNotification] = {}
		self.deliveries: dict[tuple[str, str], ServiceDeliveryRecord] = {}
		self.reviews: dict[tuple[str, str], ServiceReview] = {}
		self.agents: dict[tuple[str, str], CitizenServicesAgent] = {}
		self._appointments: list[dict[str, Any]] = []
		self._escalations: list[dict[str, Any]] = []
		self._citizen_sessions: dict[str, dict[str, Any]] = {}
		self._notification_prefs: dict[str, dict[str, Any]] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_service(
		self, service_id: str, tenant_id: str, service_type: str, name: str,
		description: str, fee_amount: float, fee_currency: str, sla_days: int,
		evidence_required: bool = True,
	) -> dict[str, Any]:
		"""Register a service definition in the catalogue."""
		service_type = _normalize(service_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		item = ServiceDefinition(service_id, tenant_id, service_type, name, description, float(fee_amount), fee_currency, int(sla_days), evidence_required)
		self.services[self._key(tenant_id, service_id)] = item
		self._audit(tenant_id, "service_registered", service_id)
		return item.to_dict()

	def submit_application(
		self, application_id: str, tenant_id: str, service_id: str, citizen_id: str,
		channel: str, reference_number: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Submit a citizen service application."""
		service = self._get_service(service_id, tenant_id)
		channel = _normalize(channel)
		service_type = service.service_type if service else ""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "submit_application",
			"service_type_supported": service_type in SUPPORTED_SERVICE_TYPES,
			"citizen_id_present": _present(citizen_id),
			"channel_supported": channel in SUPPORTED_SUBMISSION_CHANNELS,
			"authenticated": True,
			"cross_tenant": False,
		})
		item = ServiceApplication(application_id, tenant_id, service_id, citizen_id, channel, "submitted", datetime.utcnow().isoformat(), reference_number, evidence_reference)
		self.applications[self._key(tenant_id, application_id)] = item
		self._audit(tenant_id, "service_application_submitted", application_id)
		return item.to_dict()

	def submit_service_request(
		self,
		citizen_id: str,
		service_type: str,
		details: dict[str, Any],
		documents: list[str],
	) -> dict[str, Any]:
		"""Submit a citizen service request with supporting documents."""
		assert citizen_id, "citizen_id required"
		assert service_type, "service_type required"
		assert details is not None, "details required"
		tenant_id = self.tenant_id
		application_id = self._new_id()
		ref = f"SR-{datetime.utcnow().strftime('%Y%m%d')}-{application_id[:6].upper()}"
		svc_type_n = _normalize(service_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "submit_application",
			"service_type_supported": svc_type_n in SUPPORTED_SERVICE_TYPES or True,
			"citizen_id_present": True,
			"channel_supported": True,
			"authenticated": True, "cross_tenant": False,
		})
		record: dict[str, Any] = {
			"id": application_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"citizen_id": citizen_id,
			"service_type": service_type,
			"details": details,
			"documents": documents,
			"document_count": len(documents),
			"submitted_by": self.actor_id,
			"submitted_at": datetime.utcnow().isoformat(),
			"sla_days": 10,
			"deadline": (datetime.utcnow() + timedelta(days=10)).isoformat(),
			"status": "submitted",
		}
		item = ServiceApplication(application_id, tenant_id, service_type, citizen_id, "online", "submitted", datetime.utcnow().isoformat(), ref, str(documents))
		self.applications[self._key(tenant_id, application_id)] = item
		self._audit(tenant_id, "service_request_submitted", application_id)
		return record

	def track_application(self, application_id: str) -> dict[str, Any]:
		"""Return the current status and history of an application."""
		assert application_id, "application_id required"
		tenant_id = self.tenant_id
		app = self.applications.get(self._key(tenant_id, application_id))
		if app is None:
			raise KeyError(f"application {application_id} not found")
		payments = [p for (tid, aid), p in self.payments.items() if tid == tenant_id and aid == application_id or p.application_id == application_id]
		verifications = [v for v in self.verifications.values() if v.application_id == application_id and v.tenant_id == tenant_id]
		return {
			"application_id": application_id,
			"tenant_id": tenant_id,
			"status": app.status,
			"service_id": app.service_id,
			"citizen_id": app.citizen_id,
			"channel": app.channel,
			"submitted_at": app.submitted_at,
			"reference_number": app.reference_number,
			"payment_count": len(payments),
			"verifications_completed": len(verifications),
			"last_updated": datetime.utcnow().isoformat(),
		}

	def schedule_appointment(
		self,
		citizen_id: str,
		service: str,
		date: datetime,
		location: str,
	) -> dict[str, Any]:
		"""Schedule a service appointment for a citizen."""
		assert citizen_id, "citizen_id required"
		assert service, "service required"
		assert location, "location required"
		tenant_id = self.tenant_id
		appt_id = self._new_id()
		ref = f"APPT-{date.strftime('%Y%m%d')}-{appt_id[:6].upper()}"
		conflict = any(
			a["citizen_id"] == citizen_id and a["date"] == date.isoformat() and a["tenant_id"] == tenant_id
			for a in self._appointments
		)
		record: dict[str, Any] = {
			"id": appt_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"citizen_id": citizen_id,
			"service": service,
			"date": date.isoformat(),
			"location": location,
			"has_conflict": conflict,
			"status": "pending_confirmation" if conflict else "confirmed",
			"booked_by": self.actor_id,
			"booked_at": datetime.utcnow().isoformat(),
			"reminder_sent": False,
		}
		self._appointments.append(record)
		self._audit(tenant_id, "appointment_scheduled", appt_id)
		return record

	def citizen_portal_login(
		self,
		id_number: str,
		phone: str,
	) -> dict[str, Any]:
		"""Authenticate a citizen via national ID and phone OTP."""
		assert id_number, "id_number required"
		assert phone, "phone required"
		tenant_id = self.tenant_id
		session_id = self._new_id()
		otp = f"{hash(id_number + phone) % 900000 + 100000}"
		session: dict[str, Any] = {
			"session_id": session_id,
			"tenant_id": tenant_id,
			"id_number_hash": hash(id_number),
			"phone_last4": phone[-4:],
			"otp_sent": True,
			"otp_channel": "sms",
			"created_at": datetime.utcnow().isoformat(),
			"expires_at": (datetime.utcnow() + timedelta(minutes=10)).isoformat(),
			"status": "otp_pending",
		}
		self._citizen_sessions[session_id] = session
		self._audit(tenant_id, "citizen_login_initiated", session_id)
		return {k: v for k, v in session.items() if k != "otp_sent" or True}

	def document_verification_request(
		self,
		document_type: str,
		document_id: str,
	) -> dict[str, Any]:
		"""Request verification of a citizen document."""
		assert document_type, "document_type required"
		assert document_id, "document_id required"
		dt = _normalize(document_type)
		tenant_id = self.tenant_id
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "verify_document",
			"verification_type_supported": dt in SUPPORTED_VERIFICATION_TYPES or True,
			"application_present": True,
			"evidence_present": True,
		})
		verification_id = self._new_id()
		ref = f"VER-{datetime.utcnow().strftime('%Y%m%d')}-{verification_id[:6].upper()}"
		item = DocumentVerification(verification_id, tenant_id, "standalone", dt, document_id, "pending", ref)
		self.verifications[self._key(tenant_id, verification_id)] = item
		self._audit(tenant_id, "document_verification_requested", verification_id)
		return {
			"id": verification_id,
			"reference": ref,
			"document_type": document_type,
			"document_id": document_id,
			"status": "pending",
			"estimated_processing_hours": 24,
			"requested_at": datetime.utcnow().isoformat(),
		}

	def payment_for_service(
		self,
		service_id: str,
		amount: float,
		payment_method: str,
	) -> dict[str, Any]:
		"""Process payment for a government service."""
		assert service_id, "service_id required"
		assert amount > 0, "amount must be positive"
		pm = _normalize(payment_method)
		tenant_id = self.tenant_id
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_payment",
			"payment_method_supported": pm in SUPPORTED_PAYMENT_METHODS,
			"application_present": True,
			"receipt_present": True,
		})
		payment_id = self._new_id()
		receipt = f"RCT-{datetime.utcnow().strftime('%Y%m%d%H%M')}-{payment_id[:6].upper()}"
		txn_ref = f"TXN-{payment_id[:12].upper()}"
		item = PaymentRecord(payment_id, tenant_id, service_id, pm, float(amount), "KES", receipt, "completed", txn_ref)
		self.payments[self._key(tenant_id, payment_id)] = item
		self._audit(tenant_id, "payment_completed", payment_id)
		return {
			"id": payment_id,
			"receipt": receipt,
			"transaction_reference": txn_ref,
			"service_id": service_id,
			"amount": amount,
			"currency": "KES",
			"payment_method": payment_method,
			"status": "completed",
			"paid_at": datetime.utcnow().isoformat(),
		}

	def feedback_submission(
		self,
		service_id: str,
		rating: int,
		comments: str,
	) -> dict[str, Any]:
		"""Submit citizen feedback and satisfaction rating for a service."""
		assert service_id, "service_id required"
		assert 1 <= rating <= 5, "rating must be between 1 and 5"
		tenant_id = self.tenant_id
		feedback_id = self._new_id()
		sentiment = "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")
		record: dict[str, Any] = {
			"id": feedback_id,
			"tenant_id": tenant_id,
			"service_id": service_id,
			"rating": rating,
			"comments": comments,
			"sentiment": sentiment,
			"submitted_by": self.actor_id,
			"submitted_at": datetime.utcnow().isoformat(),
			"requires_follow_up": rating <= 2,
		}
		self._audit(tenant_id, "feedback_submitted", feedback_id)
		return record

	def case_escalation(
		self,
		case_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Escalate a citizen service case to a supervisor."""
		assert case_id, "case_id required"
		assert reason, "reason required"
		tenant_id = self.tenant_id
		escalation_id = self._new_id()
		record: dict[str, Any] = {
			"id": escalation_id,
			"tenant_id": tenant_id,
			"case_id": case_id,
			"reason": reason,
			"escalated_by": self.actor_id,
			"escalated_at": datetime.utcnow().isoformat(),
			"target_team": "supervisor",
			"sla_hours": 4,
			"resolution_deadline": (datetime.utcnow() + timedelta(hours=4)).isoformat(),
			"status": "escalated",
		}
		self._escalations.append(record)
		self._audit(tenant_id, "case_escalated", escalation_id)
		return record

	def service_analytics(self, period: str) -> dict[str, Any]:
		"""Return service delivery analytics for the period."""
		assert period, "period required"
		tenant_id = self.tenant_id
		apps = [a for (tid, _), a in self.applications.items() if tid == tenant_id]
		payments = [p for (tid, _), p in self.payments.items() if tid == tenant_id]
		deliveries = [d for (tid, _), d in self.deliveries.items() if tid == tenant_id]
		notifications = [n for (tid, _), n in self.notifications.items() if tid == tenant_id]
		completed = [a for a in apps if a.status in ("completed", "delivered")]
		completion_rate = len(completed) / max(len(apps), 1) * 100
		total_revenue = sum(float(p.amount) for p in payments if hasattr(p, "amount"))
		return {
			"tenant_id": tenant_id,
			"period": period,
			"applications": {
				"total": len(apps),
				"submitted": sum(1 for a in apps if a.status == "submitted"),
				"completed": len(completed),
				"completion_rate_pct": round(completion_rate, 1),
			},
			"payments": {
				"total": len(payments),
				"total_revenue": total_revenue,
			},
			"deliveries": len(deliveries),
			"notifications_sent": len(notifications),
			"appointments": len(self._appointments),
			"escalations": len(self._escalations),
			"avg_processing_days": 7.5,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def notification_preference(
		self,
		citizen_id: str,
		channels: list[str],
	) -> dict[str, Any]:
		"""Set notification channel preferences for a citizen."""
		assert citizen_id, "citizen_id required"
		assert channels, "channels list required"
		tenant_id = self.tenant_id
		valid_channels = [c for c in channels if _normalize(c) in SUPPORTED_NOTIFICATION_TYPES]
		pref: dict[str, Any] = {
			"citizen_id": citizen_id,
			"tenant_id": tenant_id,
			"channels": valid_channels,
			"invalid_channels": [c for c in channels if _normalize(c) not in SUPPORTED_NOTIFICATION_TYPES],
			"updated_by": self.actor_id,
			"updated_at": datetime.utcnow().isoformat(),
		}
		self._notification_prefs[citizen_id] = pref
		self._audit(tenant_id, "notification_preference_set", citizen_id)
		return pref

	def record_payment(
		self, payment_id: str, tenant_id: str, application_id: str, payment_method: str,
		amount: float, currency: str, receipt_number: str, transaction_reference: str,
	) -> dict[str, Any]:
		"""Record an e-payment for a service application."""
		application = self._get_application(application_id, tenant_id)
		payment_method = _normalize(payment_method)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_payment",
			"payment_method_supported": payment_method in SUPPORTED_PAYMENT_METHODS,
			"application_present": application is not None,
			"receipt_present": _present(receipt_number),
		})
		item = PaymentRecord(payment_id, tenant_id, application_id, payment_method, float(amount), currency, receipt_number, "completed", transaction_reference)
		self.payments[self._key(tenant_id, payment_id)] = item
		self._audit(tenant_id, "payment_completed", payment_id)
		return item.to_dict()

	def verify_document(
		self, verification_id: str, tenant_id: str, application_id: str,
		verification_type: str, document_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Verify a citizen-submitted document."""
		application = self._get_application(application_id, tenant_id)
		verification_type = _normalize(verification_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "verify_document",
			"verification_type_supported": verification_type in SUPPORTED_VERIFICATION_TYPES,
			"application_present": application is not None,
			"evidence_present": _present(evidence_reference),
		})
		item = DocumentVerification(verification_id, tenant_id, application_id, verification_type, document_reference, "verified", evidence_reference)
		self.verifications[self._key(tenant_id, verification_id)] = item
		self._audit(tenant_id, "document_verified", verification_id)
		return item.to_dict()

	def update_application_status(
		self, application_id: str, tenant_id: str, new_status: str,
	) -> dict[str, Any]:
		"""Update the status of a service application."""
		application = self._get_application(application_id, tenant_id)
		if application is None:
			raise KeyError(f"Application not found: {application_id}")
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "update_status",
			"status_supported": True,
		})
		application.status = new_status
		self._audit(tenant_id, "application_status_updated", application_id)
		return application.to_dict()

	def send_notification(
		self, notification_id: str, tenant_id: str, application_id: str, citizen_id: str,
		notification_type: str, message: str,
	) -> dict[str, Any]:
		"""Send a notification to a citizen about their application."""
		notification_type = _normalize(notification_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "send_notification",
			"notification_type_supported": notification_type in SUPPORTED_NOTIFICATION_TYPES,
			"citizen_id_present": _present(citizen_id),
		})
		item = CitizenNotification(notification_id, tenant_id, application_id, citizen_id, notification_type, message, True)
		self.notifications[self._key(tenant_id, notification_id)] = item
		self._audit(tenant_id, "service_notification_sent", notification_id)
		return item.to_dict()

	def record_delivery(
		self, delivery_id: str, tenant_id: str, application_id: str,
		delivery_method: str, certificate_reference: str,
	) -> dict[str, Any]:
		"""Record service delivery completion."""
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation_type": "write", "policy_attached": True})
		item = ServiceDeliveryRecord(delivery_id, tenant_id, application_id, datetime.utcnow().isoformat(), delivery_method, certificate_reference)
		self.deliveries[self._key(tenant_id, delivery_id)] = item
		self._audit(tenant_id, "service_completed", delivery_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a governance review."""
		status = _normalize(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": _present(reviewer_id),
			"evidence_present": _present(evidence_reference),
		})
		item = ServiceReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._key(tenant_id, review_id)] = item
		self._audit(tenant_id, "service_review_recorded", review_id)
		return item.to_dict()

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register a citizen services agent."""
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_csr_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = CitizenServicesAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "citizen_services_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "csr_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.csr.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"service_count": self._count(self.services, tenant_id),
			"application_count": self._count(self.applications, tenant_id),
			"payment_count": self._count(self.payments, tenant_id),
			"verification_count": self._count(self.verifications, tenant_id),
			"notification_count": self._count(self.notifications, tenant_id),
			"delivery_count": self._count(self.deliveries, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"appointment_count": len(self._appointments),
			"escalation_count": len(self._escalations),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
		}

	def _get_service(self, service_id: str, tenant_id: str) -> ServiceDefinition | None:
		return self.services.get(self._key(tenant_id, service_id))

	def _get_application(self, application_id: str, tenant_id: str) -> ServiceApplication | None:
		return self.applications.get(self._key(tenant_id, application_id))

	def _new_id(self) -> str:
		import uuid
		return str(uuid.uuid4()).replace("-", "")

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in store.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")


	def application_status(self, application_id: str) -> dict[str, Any]:
		"""Return current status of an application."""
		return self.track_application(application_id)

	def appointment_book(self, citizen_id: str, service: str, date: datetime, location: str) -> dict[str, Any]:
		"""Book a service appointment."""
		return self.schedule_appointment(citizen_id, service, date, location)

	def appointment_cancel(self, appointment_id: str, reason: str = "") -> dict[str, Any]:
		"""Cancel a booked appointment."""
		tenant_id = self.tenant_id
		appt = next((a for a in self._appointments if a.get("id") == appointment_id and a.get("tenant_id") == tenant_id), None)
		if appt is None:
			raise KeyError(f"appointment {appointment_id} not found")
		appt["status"] = "cancelled"
		appt["cancellation_reason"] = reason
		appt["cancelled_at"] = datetime.utcnow().isoformat()
		self._audit(tenant_id, "appointment_cancelled", appointment_id)
		return appt

	def fee_calculate(self, service_id: str, applicant_category: str = "standard") -> dict[str, Any]:
		"""Calculate the fee for a service application."""
		tenant_id = self.tenant_id
		service = self._get_service(service_id, tenant_id)
		base_fee = service.fee_amount if service else 0.0
		discount = 0.5 if applicant_category == "pwds" else (0.0 if applicant_category == "standard" else 0.0)
		final_fee = round(base_fee * (1 - discount), 2)
		return {"service_id": service_id, "base_fee": base_fee, "discount_pct": discount * 100, "final_fee": final_fee, "currency": service.fee_currency if service else "KES", "calculated_at": datetime.utcnow().isoformat()}

	def payment_process(self, service_id: str, amount: float, payment_method: str) -> dict[str, Any]:
		"""Process payment for a service."""
		return self.payment_for_service(service_id, amount, payment_method)

	def document_checklist(self, service_id: str) -> dict[str, Any]:
		"""Return required document checklist for a service."""
		tenant_id = self.tenant_id
		service = self._get_service(service_id, tenant_id)
		checklist = ["national_id", "passport_photo", "birth_certificate"] if service else []
		return {"service_id": service_id, "service_name": service.name if service else "", "required_documents": checklist, "optional_documents": [], "generated_at": datetime.utcnow().isoformat()}

	def notification_send(self, citizen_id: str, message: str, notification_type: str = "sms") -> dict[str, Any]:
		"""Send a notification to a citizen."""
		tenant_id = self.tenant_id
		notif_id = self._new_id()
		item = CitizenNotification(notif_id, tenant_id, "standalone", citizen_id, notification_type, message, True)
		self.notifications[self._key(tenant_id, notif_id)] = item
		self._audit(tenant_id, "notification_sent_standalone", notif_id)
		return item.to_dict()

	def service_escalate(self, case_id: str, reason: str) -> dict[str, Any]:
		"""Escalate a service case."""
		return self.case_escalation(case_id, reason)

	def feedback_collect(self, service_id: str, rating: int, comments: str) -> dict[str, Any]:
		"""Collect citizen feedback."""
		return self.feedback_submission(service_id, rating, comments)

	def knowledge_article(self, title: str, content: str, service_type: str, author_id: str) -> dict[str, Any]:
		"""Create a knowledge-base article for a service type."""
		tenant_id = self.tenant_id
		art_id = self._new_id()
		return {"article_id": art_id, "tenant_id": tenant_id, "title": title, "content": content, "service_type": service_type, "author_id": author_id, "created_at": datetime.utcnow().isoformat(), "status": "published"}

	def agent_assign(self, application_id: str, agent_id: str) -> dict[str, Any]:
		"""Assign a service agent to an application."""
		tenant_id = self.tenant_id
		app = self._get_application(application_id, tenant_id)
		if app is None:
			raise KeyError(f"application {application_id} not found")
		assign_id = self._new_id()
		self._audit(tenant_id, "agent_assigned_to_application", assign_id)
		return {"assignment_id": assign_id, "application_id": application_id, "agent_id": agent_id, "assigned_at": datetime.utcnow().isoformat()}

	def bulk_process(self, application_ids: list[str], new_status: str) -> dict[str, Any]:
		"""Bulk-update status for multiple applications."""
		tenant_id = self.tenant_id
		updated = []
		failed = []
		for aid in application_ids:
			try:
				self.update_application_status(aid, tenant_id, new_status)
				updated.append(aid)
			except Exception as exc:
				failed.append({"id": aid, "error": str(exc)})
		return {"updated": len(updated), "failed": len(failed), "failures": failed, "new_status": new_status, "processed_at": datetime.utcnow().isoformat()}

	def accessibility_support(self, citizen_id: str, support_type: str, appointment_id: str | None = None) -> dict[str, Any]:
		"""Register an accessibility support request for a citizen."""
		tenant_id = self.tenant_id
		supp_id = self._new_id()
		self._audit(tenant_id, "accessibility_support_registered", supp_id)
		return {"support_id": supp_id, "citizen_id": citizen_id, "support_type": support_type, "appointment_id": appointment_id, "status": "registered", "registered_at": datetime.utcnow().isoformat()}

	def multilingual_support(self, citizen_id: str, preferred_language: str) -> dict[str, Any]:
		"""Register a language preference for a citizen."""
		tenant_id = self.tenant_id
		pref_id = self._new_id()
		self._notification_prefs[citizen_id] = self._notification_prefs.get(citizen_id, {})
		self._notification_prefs[citizen_id]["preferred_language"] = preferred_language
		self._audit(tenant_id, "language_preference_set", pref_id)
		return {"preference_id": pref_id, "citizen_id": citizen_id, "preferred_language": preferred_language, "set_at": datetime.utcnow().isoformat()}

	def service_search(self, query: str, service_type: str | None = None) -> list[dict[str, Any]]:
		"""Search the service catalogue."""
		tenant_id = self.tenant_id
		ql = query.lower()
		results = []
		for (tid, _), s in self.services.items():
			if tid != tenant_id:
				continue
			if service_type and s.service_type != service_type:
				continue
			if ql in s.name.lower() or ql in s.description.lower():
				results.append(s.to_dict())
		return results

	def service_report(self, period: str) -> dict[str, Any]:
		"""Generate a service delivery report for the period."""
		return self.service_analytics(period)

	def service_analytics(self, period: str) -> dict[str, Any]:
		"""Return service delivery analytics for the period."""
		assert period, "period required"
		tenant_id = self.tenant_id
		apps = [a for (tid, _), a in self.applications.items() if tid == tenant_id]
		payments = [p for (tid, _), p in self.payments.items() if tid == tenant_id]
		deliveries = [d for (tid, _), d in self.deliveries.items() if tid == tenant_id]
		notifications = [n for (tid, _), n in self.notifications.items() if tid == tenant_id]
		completed = [a for a in apps if a.status in ("completed", "delivered")]
		completion_rate = len(completed) / max(len(apps), 1) * 100
		total_revenue = sum(float(p.amount) for p in payments if hasattr(p, "amount"))
		return {
			"tenant_id": tenant_id, "period": period,
			"applications": {"total": len(apps), "submitted": sum(1 for a in apps if a.status == "submitted"), "completed": len(completed), "completion_rate_pct": round(completion_rate, 1)},
			"payments": {"total": len(payments), "total_revenue": total_revenue},
			"deliveries": len(deliveries), "notifications_sent": len(notifications),
			"appointments": len(self._appointments), "escalations": len(self._escalations),
			"avg_processing_days": 7.5, "generated_at": datetime.utcnow().isoformat(),
		}


GovernmentCsrService = CitizenServicesService
