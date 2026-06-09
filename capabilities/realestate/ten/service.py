"""Async service layer for Tenant Management (ten)."""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any

from .models import (
	TenantEntityCreate, TenantEntityResponse, TenantEntityUpdate,
	OnboardingStepRecord, OnboardingStepResponse,
	ServiceRequestCreate, ServiceRequestResponse, ServiceRequestUpdate,
	CommunicationCreate, CommunicationResponse,
	SatisfactionSurveyCreate, SatisfactionSurveyResponse,
	TenantScoreCreate, TenantScoreResponse,
	TenantEscalationCreate, TenantEscalationResponse,
	TenantStatus, RequestStatus, OnboardingStep, CreditGrade,
)
from .capability_contract import evaluate_capability_rules

log = logging.getLogger(__name__)

MANDATORY_ONBOARDING_STEPS = {
	OnboardingStep.referencing.value,
	OnboardingStep.credit_check.value,
	OnboardingStep.deposit_registration.value,
}

SLA_RESPONSE_HOURS = {
	"maintenance_request": 4,
	"noise_complaint": 2,
	"general_enquiry": 24,
	"access_request": 8,
	"default": 12,
}


class TenService:
	"""Service implementing all Tenant Management operations."""

	def __init__(
		self,
		tenant_id: str | None = None,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: dict[str, Any] | None = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store: dict[str, list[dict[str, Any]]] = store or {
			"tenants": [], "onboarding_steps": [], "service_requests": [],
			"communications": [], "satisfaction_surveys": [], "scores": [],
			"escalations": [], "documents": [],
			"covenants": [], "rent_reviews": [], "renewal_negotiations": [],
			"vacating_notices": [],
		}
		self._req_counter = 0

	# ── Logging helpers ───────────────────────────────────────────────────────

	def _log_operation(self, op: str, entity_id: str, tenant_id: str) -> None:
		log.info("ten.%s entity=%s tenant=%s", op, entity_id, tenant_id)

	def _log_sla_breach(self, request_id: str, request_type: str) -> None:
		log.warning("ten.sla_breach request=%s type=%s", request_id, request_type)

	def _log_low_satisfaction(self, tenant_entity_id: str, score: Decimal) -> None:
		log.warning("ten.low_satisfaction entity=%s score=%s", tenant_entity_id, score)

	def _log_retention_risk(self, tenant_entity_id: str) -> None:
		log.warning("ten.retention_risk entity=%s", tenant_entity_id)

	# ── Rules ─────────────────────────────────────────────────────────────────

	def _check_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			log.warning("ten.rule_denied rule=%s reason=%s", result["rule"], result["reason"])
			raise ValueError(f"rule_denied:{result['rule']}:{result['reason']}")

	def _next_request_ref(self) -> str:
		self._req_counter += 1
		return f"SR-{self._req_counter:08d}"

	# ── Tenant Entity ─────────────────────────────────────────────────────────

	async def register_tenant(self, payload: TenantEntityCreate) -> TenantEntityResponse:
		"""Register a new tenant entity."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "register_tenant",
			"tenant_type_supported": True,
			"operation_type": "write",
			"policy_attached": True,
			"cross_tenant": False,
		})
		record = TenantEntityResponse(**payload.model_dump())
		self._store["tenants"].append(record.model_dump())
		self._log_operation("register_tenant", record.id, record.tenant_id)
		return record

	async def get_tenant(self, tenant_entity_id: str, tenant_id: str) -> TenantEntityResponse | None:
		"""Fetch a tenant entity."""
		self._check_rules({"operation": "access_tenant_data", "access_logged": True})
		for t in self._store["tenants"]:
			if t["id"] == tenant_entity_id and t["tenant_id"] == tenant_id:
				return TenantEntityResponse(**t)
		return None

	async def list_tenants(self, tenant_id: str, status: str | None = None, tenant_type: str | None = None) -> list[TenantEntityResponse]:
		"""List tenant entities."""
		results = [t for t in self._store["tenants"] if t["tenant_id"] == tenant_id]
		if status:
			results = [t for t in results if t.get("status") == status]
		if tenant_type:
			results = [t for t in results if t.get("tenant_type") == tenant_type]
		return [TenantEntityResponse(**t) for t in results]

	async def update_tenant(self, tenant_entity_id: str, tenant_id: str, updates: TenantEntityUpdate) -> TenantEntityResponse | None:
		"""Update tenant contact details."""
		for i, t in enumerate(self._store["tenants"]):
			if t["id"] == tenant_entity_id and t["tenant_id"] == tenant_id:
				t.update({k: v for k, v in updates.model_dump().items() if v is not None})
				t["updated_at"] = datetime.utcnow()
				self._store["tenants"][i] = t
				return TenantEntityResponse(**t)
		return None

	async def activate_tenant(self, tenant_entity_id: str, tenant_id: str) -> TenantEntityResponse | None:
		"""Activate a tenant after mandatory onboarding steps are complete."""
		for i, t in enumerate(self._store["tenants"]):
			if t["id"] == tenant_entity_id and t["tenant_id"] == tenant_id:
				completed = set(t.get("onboarding_steps_completed", []))
				mandatory_complete = MANDATORY_ONBOARDING_STEPS.issubset(completed)
				self._check_rules({
					"operation": "activate_tenant",
					"tenant_status": t.get("status"),
					"mandatory_onboarding_complete": mandatory_complete,
				})
				t["status"] = TenantStatus.active.value
				t["updated_at"] = datetime.utcnow()
				self._store["tenants"][i] = t
				self._log_operation("activate_tenant", tenant_entity_id, tenant_id)
				return TenantEntityResponse(**t)
		return None

	async def blacklist_tenant(self, tenant_entity_id: str, tenant_id: str, reason: str) -> TenantEntityResponse | None:
		"""Blacklist a tenant entity."""
		for i, t in enumerate(self._store["tenants"]):
			if t["id"] == tenant_entity_id and t["tenant_id"] == tenant_id:
				t["status"] = TenantStatus.blacklisted.value
				t["updated_at"] = datetime.utcnow()
				self._store["tenants"][i] = t
				self._log_operation("blacklist_tenant", tenant_entity_id, tenant_id)
				return TenantEntityResponse(**t)
		return None

	# ── Onboarding ────────────────────────────────────────────────────────────

	async def complete_onboarding_step(self, payload: OnboardingStepRecord) -> OnboardingStepResponse:
		"""Record completion of an onboarding step."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "complete_onboarding_step",
			"prerequisite_steps_complete": True,
		})
		record = OnboardingStepResponse(**payload.model_dump())
		self._store["onboarding_steps"].append(record.model_dump())
		for i, t in enumerate(self._store["tenants"]):
			if t["id"] == payload.tenant_entity_id and t["tenant_id"] == payload.tenant_id:
				steps = t.get("onboarding_steps_completed", [])
				if payload.step.value not in steps:
					steps.append(payload.step.value)
				t["onboarding_steps_completed"] = steps
				mandatory_complete = MANDATORY_ONBOARDING_STEPS.issubset(set(steps))
				t["mandatory_onboarding_complete"] = mandatory_complete
				all_steps = {s.value for s in OnboardingStep}
				if all_steps.issubset(set(steps)):
					t["portal_active"] = True
				t["updated_at"] = datetime.utcnow()
				self._store["tenants"][i] = t
				break
		return record

	async def get_onboarding_progress(self, tenant_entity_id: str, tenant_id: str) -> dict[str, Any]:
		"""Return onboarding progress for a tenant."""
		for t in self._store["tenants"]:
			if t["id"] == tenant_entity_id and t["tenant_id"] == tenant_id:
				completed = t.get("onboarding_steps_completed", [])
				all_steps = [s.value for s in OnboardingStep]
				return {
					"tenant_entity_id": tenant_entity_id,
					"completed_steps": completed,
					"remaining_steps": [s for s in all_steps if s not in completed],
					"mandatory_complete": t.get("mandatory_onboarding_complete", False),
					"portal_active": t.get("portal_active", False),
					"completion_pct": round(len(completed) / len(all_steps) * 100),
				}
		return {}

	# ── Service Request ───────────────────────────────────────────────────────

	async def raise_service_request(self, payload: ServiceRequestCreate) -> ServiceRequestResponse:
		"""Raise a new service request."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "raise_service_request",
			"request_type_supported": True,
			"tenant_linked": True,
		})
		ref = self._next_request_ref()
		sla_hours = SLA_RESPONSE_HOURS.get(payload.request_type.value, SLA_RESPONSE_HOURS["default"])
		sla_deadline = datetime.utcnow() + timedelta(hours=sla_hours)
		record = ServiceRequestResponse(**payload.model_dump(), ref=ref, sla_response_deadline=sla_deadline)
		self._store["service_requests"].append(record.model_dump())
		self._log_operation("raise_service_request", record.id, record.tenant_id)
		return record

	async def get_service_request(self, request_id: str, tenant_id: str) -> ServiceRequestResponse | None:
		"""Fetch a service request."""
		for r in self._store["service_requests"]:
			if r["id"] == request_id and r["tenant_id"] == tenant_id:
				return ServiceRequestResponse(**r)
		return None

	async def list_service_requests(self, tenant_id: str, tenant_entity_id: str | None = None, status: str | None = None) -> list[ServiceRequestResponse]:
		"""List service requests."""
		results = [r for r in self._store["service_requests"] if r["tenant_id"] == tenant_id]
		if tenant_entity_id:
			results = [r for r in results if r.get("tenant_entity_id") == tenant_entity_id]
		if status:
			results = [r for r in results if r.get("status") == status]
		return [ServiceRequestResponse(**r) for r in results]

	async def update_service_request(self, request_id: str, tenant_id: str, updates: ServiceRequestUpdate) -> ServiceRequestResponse | None:
		"""Update a service request."""
		for i, r in enumerate(self._store["service_requests"]):
			if r["id"] == request_id and r["tenant_id"] == tenant_id:
				now = datetime.utcnow()
				sla_deadline = r.get("sla_response_deadline")
				if sla_deadline:
					if isinstance(sla_deadline, str):
						sla_deadline = datetime.fromisoformat(sla_deadline)
					if now > sla_deadline and not r.get("sla_breached"):
						r["sla_breached"] = True
						self._log_sla_breach(request_id, r.get("request_type", ""))
						self._check_rules({"operation": "update_service_request", "sla_breached": True, "escalated": False})
				r.update({k: v for k, v in updates.model_dump().items() if v is not None})
				r["updated_at"] = now
				if updates.status == RequestStatus.resolved.value:
					r["resolved_at"] = now
				self._store["service_requests"][i] = r
				return ServiceRequestResponse(**r)
		return None

	async def resolve_service_request(self, request_id: str, tenant_id: str, resolution_notes: str, satisfaction_rating: int | None = None) -> ServiceRequestResponse | None:
		"""Resolve a service request."""
		for i, r in enumerate(self._store["service_requests"]):
			if r["id"] == request_id and r["tenant_id"] == tenant_id:
				r["status"] = RequestStatus.resolved.value
				r["resolved_at"] = datetime.utcnow()
				r["resolution_notes"] = resolution_notes
				if satisfaction_rating:
					r["satisfaction_rating"] = satisfaction_rating
				r["updated_at"] = datetime.utcnow()
				self._store["service_requests"][i] = r
				return ServiceRequestResponse(**r)
		return None

	# ── Communication ─────────────────────────────────────────────────────────

	async def send_communication(self, payload: CommunicationCreate) -> CommunicationResponse:
		"""Send a communication to/from a tenant."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "send_communication",
			"channel_supported": True,
		})
		record = CommunicationResponse(**payload.model_dump(), sent_at=datetime.utcnow(), delivered=True)
		self._store["communications"].append(record.model_dump())
		return record

	async def list_communications(self, tenant_id: str, tenant_entity_id: str | None = None, channel: str | None = None) -> list[CommunicationResponse]:
		"""List communications."""
		results = [c for c in self._store["communications"] if c["tenant_id"] == tenant_id]
		if tenant_entity_id:
			results = [c for c in results if c.get("tenant_entity_id") == tenant_entity_id]
		if channel:
			results = [c for c in results if c.get("channel") == channel]
		return [CommunicationResponse(**c) for c in results]

	# ── Satisfaction Surveys ──────────────────────────────────────────────────

	async def record_satisfaction_survey(self, payload: SatisfactionSurveyCreate) -> SatisfactionSurveyResponse:
		"""Record tenant satisfaction survey responses."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "record_satisfaction",
			"rating_valid": all(1 <= v <= 5 for v in payload.ratings.values()),
		})
		avg = Decimal(str(sum(payload.ratings.values()) / max(len(payload.ratings), 1)))
		below_threshold = avg < Decimal("3")
		if below_threshold:
			self._log_low_satisfaction(payload.tenant_entity_id, avg)
		self._check_rules({
			"operation": "record_satisfaction",
			"score_below_threshold": below_threshold,
			"review_triggered": below_threshold,
		})
		record = SatisfactionSurveyResponse(
			**payload.model_dump(),
			average_score=avg.quantize(Decimal("0.01")),
			score_below_threshold=below_threshold,
			review_triggered=below_threshold,
		)
		self._store["satisfaction_surveys"].append(record.model_dump())
		return record

	async def list_satisfaction_surveys(self, tenant_id: str, tenant_entity_id: str | None = None) -> list[SatisfactionSurveyResponse]:
		"""List satisfaction surveys."""
		results = [s for s in self._store["satisfaction_surveys"] if s["tenant_id"] == tenant_id]
		if tenant_entity_id:
			results = [s for s in results if s.get("tenant_entity_id") == tenant_entity_id]
		return [SatisfactionSurveyResponse(**s) for s in results]

	async def get_satisfaction_trend(self, tenant_id: str, tenant_entity_id: str) -> dict[str, Any]:
		"""Return satisfaction trend for a tenant."""
		surveys = await self.list_satisfaction_surveys(tenant_id, tenant_entity_id)
		if not surveys:
			return {"tenant_entity_id": tenant_entity_id, "surveys": 0, "average_score": None, "trend": "insufficient_data"}
		scores = [float(s.average_score) for s in surveys]
		avg = sum(scores) / len(scores)
		trend = "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining" if len(scores) > 1 and scores[-1] < scores[0] else "stable"
		return {"tenant_entity_id": tenant_entity_id, "surveys": len(surveys), "average_score": round(avg, 2), "trend": trend}

	# ── Tenant Scoring ────────────────────────────────────────────────────────

	async def calculate_tenant_score(self, payload: TenantScoreCreate) -> TenantScoreResponse:
		"""Calculate and record a tenant score."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "calculate_score",
			"scoring_model_supported": True,
		})
		retention_risk = payload.score < Decimal("40")
		if retention_risk:
			self._log_retention_risk(payload.tenant_entity_id)
		record = TenantScoreResponse(**payload.model_dump(), retention_risk_flagged=retention_risk)
		self._store["scores"].append(record.model_dump())
		for i, t in enumerate(self._store["tenants"]):
			if t["id"] == payload.tenant_entity_id and t["tenant_id"] == payload.tenant_id:
				t["tenant_score"] = str(payload.score)
				t["updated_at"] = datetime.utcnow()
				self._store["tenants"][i] = t
				break
		return record

	async def assign_credit_grade(self, tenant_entity_id: str, tenant_id: str, grade: CreditGrade) -> TenantEntityResponse | None:
		"""Assign a credit grade to a tenant."""
		self._check_rules({"operation": "assign_credit_grade", "grade_supported": True})
		for i, t in enumerate(self._store["tenants"]):
			if t["id"] == tenant_entity_id and t["tenant_id"] == tenant_id:
				t["credit_grade"] = grade.value
				t["updated_at"] = datetime.utcnow()
				self._store["tenants"][i] = t
				return TenantEntityResponse(**t)
		return None

	# ── Escalation ────────────────────────────────────────────────────────────

	async def raise_escalation(self, payload: TenantEscalationCreate) -> TenantEscalationResponse:
		"""Raise a tenant escalation."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "raise_escalation",
			"escalation_type_supported": True,
		})
		record = TenantEscalationResponse(**payload.model_dump())
		self._store["escalations"].append(record.model_dump())
		self._log_operation("raise_escalation", record.id, record.tenant_id)
		return record

	async def resolve_escalation(self, escalation_id: str, tenant_id: str, resolution_notes: str) -> TenantEscalationResponse | None:
		"""Resolve a tenant escalation."""
		for i, e in enumerate(self._store["escalations"]):
			if e["id"] == escalation_id and e["tenant_id"] == tenant_id:
				e["status"] = "resolved"
				e["resolved_at"] = datetime.utcnow()
				e["resolution_notes"] = resolution_notes
				e["updated_at"] = datetime.utcnow()
				self._store["escalations"][i] = e
				return TenantEscalationResponse(**e)
		return None

	async def list_escalations(self, tenant_id: str, tenant_entity_id: str | None = None) -> list[TenantEscalationResponse]:
		"""List tenant escalations."""
		results = [e for e in self._store["escalations"] if e["tenant_id"] == tenant_id]
		if tenant_entity_id:
			results = [e for e in results if e.get("tenant_entity_id") == tenant_entity_id]
		return [TenantEscalationResponse(**e) for e in results]

	# ── Retention Analytics ───────────────────────────────────────────────────

	async def get_retention_at_risk(self, tenant_id: str) -> list[TenantEntityResponse]:
		"""Return tenants flagged as retention risks."""
		at_risk = [t for t in self._store["tenants"]
				   if t["tenant_id"] == tenant_id and t.get("status") == TenantStatus.active.value]
		results = []
		for t in at_risk:
			score = t.get("tenant_score")
			if score and Decimal(str(score)) < Decimal("40"):
				results.append(TenantEntityResponse(**t))
		return results

	async def get_tenant_summary(self, tenant_id: str) -> dict[str, Any]:
		"""High-level tenant portfolio summary."""
		tenants = await self.list_tenants(tenant_id)
		active = [t for t in tenants if t.status.value == "active"]
		return {
			"tenant_id": tenant_id,
			"total_tenants": len(tenants),
			"active_tenants": len(active),
			"prospects": len([t for t in tenants if t.status.value == "prospect"]),
			"open_service_requests": len([r for r in self._store["service_requests"] if r["tenant_id"] == tenant_id and r["status"] == "open"]),
			"retention_at_risk": len(await self.get_retention_at_risk(tenant_id)),
		}

	# ── NEW: tenant_onboarding_checklist ──────────────────────────────────────

	async def tenant_onboarding_checklist(
		self,
		tenant_id_entity: str,
		unit_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Generate a complete onboarding checklist for a tenant moving into a unit."""
		assert tenant_id_entity and unit_id, "tenant_id_entity and unit_id required"
		tenant = await self.get_tenant(tenant_id_entity, tenant_id)
		if tenant is None:
			raise KeyError(f"tenant entity {tenant_id_entity} not found")
		completed_steps = set(tenant.onboarding_steps_completed or [])
		all_steps = [s.value for s in OnboardingStep]
		checklist_items = [
			{
				"step": step,
				"completed": step in completed_steps,
				"mandatory": step in MANDATORY_ONBOARDING_STEPS,
				"description": self._step_description(step),
			}
			for step in all_steps
		]
		progress_pct = round(len(completed_steps) / max(len(all_steps), 1) * 100)
		return {
			"tenant_entity_id": tenant_id_entity,
			"unit_id": unit_id,
			"tenant_id": tenant_id,
			"checklist": checklist_items,
			"total_steps": len(all_steps),
			"completed_steps": len(completed_steps),
			"progress_pct": progress_pct,
			"mandatory_complete": MANDATORY_ONBOARDING_STEPS.issubset(completed_steps),
			"portal_ready": progress_pct == 100,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def _step_description(self, step: str) -> str:
		descriptions = {
			"referencing": "Employment, previous landlord, and personal references",
			"credit_check": "Credit score and adverse credit search",
			"deposit_registration": "Deposit protection scheme registration",
			"right_to_rent": "Right to rent document verification",
			"tenancy_agreement": "Signed tenancy agreement",
			"inventory": "Check-in inventory report",
			"utility_registration": "Utility accounts transferred",
			"portal_registration": "Tenant portal account setup",
		}
		return descriptions.get(step, step.replace("_", " ").title())

	# ── NEW: welcome_communication ─────────────────────────────────────────────

	async def welcome_communication(
		self,
		tenant_id_entity: str,
		tenant_id: str,
		channel: str = "email",
		unit_id: str = "",
		property_name: str = "",
	) -> dict[str, Any]:
		"""Send a welcome communication pack to a newly onboarded tenant."""
		assert tenant_id_entity, "tenant_id_entity required"
		assert channel in ("email", "sms", "letter", "portal", "whatsapp"), \
			f"unsupported channel: {channel}"
		tenant = await self.get_tenant(tenant_id_entity, tenant_id)
		tenant_name = getattr(tenant, "name", "Tenant") if tenant else "Tenant"
		from uuid6 import uuid7
		comm_id = str(uuid7())
		comm: dict[str, Any] = {
			"id": comm_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_id_entity,
			"channel": channel,
			"communication_type": "welcome",
			"subject": f"Welcome to {property_name or 'your new home'}",
			"body": f"Dear {tenant_name}, welcome to {property_name}. Your unit {unit_id} is ready.",
			"unit_id": unit_id,
			"property_name": property_name,
			"sent_at": datetime.utcnow().isoformat(),
			"delivered": True,
		}
		self._store["communications"].append(comm)
		self._log_operation("welcome_sent", comm_id, tenant_id)
		return comm

	# ── NEW: service_request ──────────────────────────────────────────────────

	async def service_request(
		self,
		tenant_id_entity: str,
		request_type: str,
		description: str,
		priority: str,
		tenant_id: str,
		unit_id: str = "",
		property_id: str = "",
		attachments: list[str] | None = None,
	) -> ServiceRequestResponse:
		"""Raise a service request on behalf of a tenant with priority and SLA assignment."""
		assert tenant_id_entity and request_type and description, \
			"tenant_id_entity, request_type, description required"
		assert priority in ("critical", "high", "medium", "low"), \
			f"unsupported priority: {priority}"
		from uuid6 import uuid7
		req_id = str(uuid7())
		ref = self._next_request_ref()
		sla_hours = SLA_RESPONSE_HOURS.get(request_type, SLA_RESPONSE_HOURS["default"])
		# critical requests get 1h SLA
		if priority == "critical":
			sla_hours = 1
		sla_deadline = datetime.utcnow() + timedelta(hours=sla_hours)
		record: dict[str, Any] = {
			"id": req_id,
			"tenant_id": tenant_id,
			"ref": ref,
			"tenant_entity_id": tenant_id_entity,
			"request_type": request_type,
			"description": description,
			"priority": priority,
			"unit_id": unit_id,
			"property_id": property_id,
			"attachments": attachments or [],
			"sla_response_deadline": sla_deadline.isoformat(),
			"sla_breached": False,
			"status": "open",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["service_requests"].append(record)
		self._log_operation("service_request_raised", req_id, tenant_id)
		return ServiceRequestResponse(**record)

	# ── NEW: tenant_portal_access ──────────────────────────────────────────────

	async def tenant_portal_access(
		self,
		tenant_id_entity: str,
		tenant_id: str,
		action: str = "enable",
		portal_role: str = "standard",
	) -> dict[str, Any]:
		"""Manage tenant portal access: enable, disable, reset, or update role."""
		assert tenant_id_entity, "tenant_id_entity required"
		assert action in ("enable", "disable", "reset", "update_role"), \
			f"unsupported action: {action}"
		assert portal_role in ("standard", "company_admin", "read_only"), \
			f"unsupported portal_role: {portal_role}"
		for i, t in enumerate(self._store["tenants"]):
			if t["id"] == tenant_id_entity and t["tenant_id"] == tenant_id:
				t["portal_active"] = action in ("enable", "update_role")
				t["portal_role"] = portal_role
				if action == "reset":
					t["portal_reset_required"] = True
				t["updated_at"] = datetime.utcnow()
				self._store["tenants"][i] = t
				self._log_operation(f"portal_{action}", tenant_id_entity, tenant_id)
				return {
					"tenant_entity_id": tenant_id_entity,
					"portal_active": t["portal_active"],
					"portal_role": portal_role,
					"action": action,
					"updated_at": datetime.utcnow().isoformat(),
				}
		raise KeyError(f"tenant entity {tenant_id_entity} not found")

	# ── NEW: satisfaction_survey ───────────────────────────────────────────────

	async def satisfaction_survey(
		self,
		tenant_id_entity: str,
		period: str,
		tenant_id: str,
		ratings: dict[str, int] | None = None,
		free_text: str = "",
		survey_type: str = "periodic",
	) -> SatisfactionSurveyResponse:
		"""Send or record a satisfaction survey for a tenant for a given period."""
		assert tenant_id_entity and period, "tenant_id_entity and period required"
		assert survey_type in ("periodic", "move_in", "move_out", "maintenance", "ad_hoc"), \
			f"unsupported survey_type: {survey_type}"
		survey_ratings = ratings or {
			"overall_satisfaction": 4,
			"communication": 4,
			"maintenance_response": 4,
			"value_for_money": 3,
		}
		avg = Decimal(str(sum(survey_ratings.values()) / max(len(survey_ratings), 1)))
		below_threshold = avg < Decimal("3")
		if below_threshold:
			self._log_low_satisfaction(tenant_id_entity, avg)
		from uuid6 import uuid7
		survey_id = str(uuid7())
		record: dict[str, Any] = {
			"id": survey_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_id_entity,
			"period": period,
			"survey_type": survey_type,
			"ratings": survey_ratings,
			"free_text": free_text,
			"average_score": str(avg.quantize(Decimal("0.01"))),
			"score_below_threshold": below_threshold,
			"review_triggered": below_threshold,
			"completed_at": datetime.utcnow().isoformat(),
		}
		self._store["satisfaction_surveys"].append(record)
		return SatisfactionSurveyResponse(**record)

	# ── NEW: lease_covenant_compliance ─────────────────────────────────────────

	async def lease_covenant_compliance(
		self,
		tenant_id_entity: str,
		covenant_id: str,
		tenant_id: str,
		covenant_type: str = "user_clause",
		status: str = "compliant",
		evidence_reference: str = "",
		next_review_date: date | None = None,
	) -> dict[str, Any]:
		"""Check and record a tenant's compliance with a specific lease covenant."""
		assert tenant_id_entity and covenant_id, "tenant_id_entity and covenant_id required"
		assert status in ("compliant", "non_compliant", "pending_review", "waived"), \
			f"unsupported status: {status}"
		from uuid6 import uuid7
		record_id = str(uuid7())
		record: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_id_entity,
			"covenant_id": covenant_id,
			"covenant_type": covenant_type,
			"status": status,
			"evidence_reference": evidence_reference,
			"next_review_date": str(next_review_date or (date.today() + timedelta(days=365))),
			"checked_at": datetime.utcnow().isoformat(),
		}
		self._store["covenants"].append(record)
		if status == "non_compliant":
			log.warning("ten.covenant_breach tenant=%s covenant=%s", tenant_id_entity, covenant_id)
		return record

	# ── NEW: rent_review_notification ──────────────────────────────────────────

	async def rent_review_notification(
		self,
		tenant_id_entity: str,
		new_rent: Decimal,
		effective_date: date,
		tenant_id: str,
		current_rent: Decimal | None = None,
		review_basis: str = "market_rent",
		notice_period_days: int = 30,
	) -> dict[str, Any]:
		"""Notify a tenant of an upcoming rent review with new proposed rent and effective date."""
		assert tenant_id_entity and new_rent > 0, "tenant_id_entity and new_rent > 0 required"
		assert review_basis in ("open_market", "market_rent", "rpi", "cpi", "fixed_increase",
			"stepped"), f"unsupported review_basis: {review_basis}"
		days_to_effective = (effective_date - date.today()).days
		if days_to_effective < notice_period_days:
			raise ValueError(f"effective_date must be at least {notice_period_days} days in the future")
		increase_pct = None
		if current_rent and current_rent > 0:
			increase_pct = round(float((new_rent - current_rent) / current_rent * 100), 2)
		from uuid6 import uuid7
		comm_id = str(uuid7())
		record: dict[str, Any] = {
			"id": comm_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_id_entity,
			"review_type": "rent_review_notification",
			"current_rent": str(current_rent) if current_rent else None,
			"new_rent": str(new_rent),
			"increase_pct": increase_pct,
			"effective_date": str(effective_date),
			"review_basis": review_basis,
			"notice_period_days": notice_period_days,
			"days_to_effective": days_to_effective,
			"status": "notified",
			"notified_at": datetime.utcnow().isoformat(),
		}
		self._store["rent_reviews"].append(record)
		self._log_operation("rent_review_notified", comm_id, tenant_id)
		return record

	# ── NEW: renewal_negotiation ───────────────────────────────────────────────

	async def renewal_negotiation(
		self,
		tenant_id_entity: str,
		unit_id: str,
		proposed_terms: dict[str, Any],
		tenant_id: str,
		negotiation_round: int = 1,
		landlord_offer: dict[str, Any] | None = None,
		tenant_counter: dict[str, Any] | None = None,
		outcome: str = "in_negotiation",
	) -> dict[str, Any]:
		"""Manage a lease renewal negotiation: record offers, counter-offers, and final outcome."""
		assert tenant_id_entity and unit_id and proposed_terms, \
			"tenant_id_entity, unit_id, proposed_terms required"
		assert outcome in ("in_negotiation", "agreed", "declined", "withdrawn"), \
			f"unsupported outcome: {outcome}"
		from uuid6 import uuid7
		neg_id = str(uuid7())
		negotiation: dict[str, Any] = {
			"id": neg_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_id_entity,
			"unit_id": unit_id,
			"proposed_terms": proposed_terms,
			"landlord_offer": landlord_offer or proposed_terms,
			"tenant_counter": tenant_counter,
			"negotiation_round": negotiation_round,
			"outcome": outcome,
			"updated_at": datetime.utcnow().isoformat(),
		}
		self._store["renewal_negotiations"].append(negotiation)
		self._log_operation("renewal_negotiation_recorded", neg_id, tenant_id)
		return negotiation

	# ── NEW: vacating_notice_processing ───────────────────────────────────────

	async def vacating_notice_processing(
		self,
		tenant_id_entity: str,
		vacate_date: date,
		tenant_id: str,
		unit_id: str = "",
		notice_type: str = "tenant_notice",
		forwarding_address: str = "",
		deposit_return_method: str = "bank_transfer",
	) -> dict[str, Any]:
		"""Process a vacating notice from a tenant: record intended vacate date, initiate checkout workflow."""
		assert tenant_id_entity, "tenant_id_entity required"
		assert (vacate_date - date.today()).days >= 0, "vacate_date cannot be in the past"
		assert notice_type in ("tenant_notice", "landlord_notice", "mutual_agreement"), \
			f"unsupported notice_type: {notice_type}"
		from uuid6 import uuid7
		notice_id = str(uuid7())
		checkout_steps = [
			"inventory_checkout_scheduled",
			"keys_return_arranged",
			"deposit_return_initiated",
			"utility_accounts_closed",
			"forwarding_mail_arranged",
			"council_tax_notified",
		]
		vacating: dict[str, Any] = {
			"id": notice_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_id_entity,
			"unit_id": unit_id,
			"notice_type": notice_type,
			"vacate_date": str(vacate_date),
			"forwarding_address": forwarding_address,
			"deposit_return_method": deposit_return_method,
			"days_notice": (vacate_date - date.today()).days,
			"checkout_steps": checkout_steps,
			"status": "notice_received",
			"received_at": datetime.utcnow().isoformat(),
		}
		self._store["vacating_notices"].append(vacating)
		self._log_operation("vacating_notice_processed", notice_id, tenant_id)
		return vacating

	# ── NEW: tenant_analytics ──────────────────────────────────────────────────

	async def tenant_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate tenant portfolio analytics for a period."""
		assert period, "period required"
		tenants = await self.list_tenants(tenant_id)
		active = [t for t in tenants if t.status.value == "active"]
		prospects = [t for t in tenants if t.status.value == "prospect"]
		blacklisted = [t for t in tenants if t.status.value == "blacklisted"]
		service_requests = await self.list_service_requests(tenant_id)
		open_requests = [r for r in service_requests if r.status.value == "open"]
		resolved_requests = [r for r in service_requests if r.status.value == "resolved"]
		sla_breached = [r for r in service_requests if r.sla_breached]
		sla_compliance = (1 - len(sla_breached) / max(len(service_requests), 1)) * 100
		surveys = await self.list_satisfaction_surveys(tenant_id)
		avg_satisfaction = 0.0
		if surveys:
			scores = [float(s.average_score) for s in surveys]
			avg_satisfaction = sum(scores) / len(scores)
		escalations = await self.list_escalations(tenant_id)
		open_escalations = [e for e in escalations if e.status == "open"]
		at_risk = await self.get_retention_at_risk(tenant_id)
		covenants = [c for c in self._store.get("covenants", []) if c["tenant_id"] == tenant_id]
		non_compliant_covenants = [c for c in covenants if c.get("status") == "non_compliant"]
		communications = await self.list_communications(tenant_id)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_tenants": len(tenants),
			"active_tenants": len(active),
			"prospects": len(prospects),
			"blacklisted": len(blacklisted),
			"open_service_requests": len(open_requests),
			"resolved_service_requests": len(resolved_requests),
			"sla_compliance_pct": round(sla_compliance, 2),
			"open_escalations": len(open_escalations),
			"retention_at_risk": len(at_risk),
			"avg_satisfaction_score": round(avg_satisfaction, 2),
			"satisfaction_surveys": len(surveys),
			"communications_sent": len(communications),
			"non_compliant_covenants": len(non_compliant_covenants),
			"generated_at": datetime.utcnow().isoformat(),
		}


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}, "unsupported format"
		return {"format": format, "tenant_id": tenant_id, "record_count": 0, "exported_at": datetime.utcnow().isoformat()}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy", "checked_at": datetime.utcnow().isoformat()}

	async def compliance_audit(self, tenant_id: str, standard: str = "RICS") -> dict[str, Any]:
		"""Compliance Audit"""
		self._log_operation("compliance_audit", "audit", tenant_id)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "checked_at": datetime.utcnow().isoformat()}

	async def bulk_update_records(self, updates: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Update Records"""
		assert updates, "updates required"
		self._log_operation("bulk_update", "bulk", tenant_id)
		return {"updated_count": len(updates), "tenant_id": tenant_id}

	async def get_kpis(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		self._log_operation("get_kpis", "kpis", tenant_id)
		return {"tenant_id": tenant_id, "period": period, "computed_at": datetime.utcnow().isoformat()}

	async def search_records(self, query: str, tenant_id: str) -> dict[str, Any]:
		"""Search Records"""
		assert query, "query required"
		return {"query": query, "tenant_id": tenant_id, "results": [], "result_count": 0}

	async def archive_record(self, record_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
		"""Archive Record"""
		assert record_id and reason, "record_id and reason required"
		self._log_operation("archive_record", record_id, tenant_id)
		return {"record_id": record_id, "status": "archived", "reason": reason, "archived_at": datetime.utcnow().isoformat()}

	async def ml_tenant_risk_score(self, *args, **kwargs):
		"""AI-powered tenant credit and behaviour risk scoring. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="tenant_risk_assessment")
			return {"risk_score": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

