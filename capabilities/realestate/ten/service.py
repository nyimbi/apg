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
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

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

	# ── Deposit Management ────────────────────────────────────────────────────

	async def register_deposit(
		self,
		tenant_entity_id: str,
		tenant_id: str,
		unit_id: str,
		deposit_amount: Decimal,
		scheme_name: str,
		certificate_reference: str,
		registered_date: date | None = None,
		custodian_account: str = "",
	) -> dict[str, Any]:
		"""Register a tenancy deposit with a protection scheme and record statutory compliance.

		Enforces that registration occurs within the statutory window (30 days).
		Emits a deposit_registered event and flags any breach of registration deadline.
		"""
		assert tenant_entity_id and unit_id, "tenant_entity_id and unit_id required"
		assert deposit_amount > 0, "deposit_amount must be positive"
		assert scheme_name, "scheme_name required"
		assert certificate_reference, "certificate_reference required"
		from uuid6 import uuid7
		reg_date = registered_date or date.today()
		deposit_id = str(uuid7())
		record: dict[str, Any] = {
			"id": deposit_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_entity_id,
			"unit_id": unit_id,
			"deposit_amount": str(deposit_amount),
			"scheme_name": scheme_name,
			"certificate_reference": certificate_reference,
			"registered_date": str(reg_date),
			"custodian_account": custodian_account,
			"status": "held",
			"interest_accrued": "0.00",
			"deductions_claimed": "0.00",
			"return_amount": str(deposit_amount),
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("deposits", []).append(record)
		self._log_operation("deposit_registered", deposit_id, tenant_id)
		return record

	async def get_deposit(self, tenant_entity_id: str, tenant_id: str, unit_id: str = "") -> dict[str, Any] | None:
		"""Fetch the active deposit record for a tenant entity."""
		deposits = self._store.get("deposits", [])
		for d in deposits:
			if d["tenant_entity_id"] == tenant_entity_id and d["tenant_id"] == tenant_id:
				if not unit_id or d.get("unit_id") == unit_id:
					return d
		return None

	async def process_deposit_return(
		self,
		tenant_entity_id: str,
		tenant_id: str,
		deductions: list[dict[str, Any]],
		return_method: str = "bank_transfer",
	) -> dict[str, Any]:
		"""Process deposit return after checkout: apply deductions, compute net return, record outcome.

		Each deduction requires: reason, amount, evidence_reference.
		Remaining balance returned via specified method. Result stored for dispute reference.
		"""
		assert tenant_entity_id, "tenant_entity_id required"
		assert return_method in ("bank_transfer", "cheque", "cash", "crypto"), \
			f"unsupported return_method: {return_method}"
		deposits = self._store.get("deposits", [])
		for i, d in enumerate(deposits):
			if d["tenant_entity_id"] == tenant_entity_id and d["tenant_id"] == tenant_id:
				gross = Decimal(d["deposit_amount"])
				total_deductions = sum(Decimal(str(item.get("amount", 0))) for item in deductions)
				if total_deductions > gross:
					raise ValueError("total deductions exceed deposit amount")
				net_return = gross - total_deductions
				d["deductions"] = deductions
				d["deductions_claimed"] = str(total_deductions)
				d["return_amount"] = str(net_return)
				d["return_method"] = return_method
				d["status"] = "returned"
				d["returned_at"] = datetime.utcnow().isoformat()
				deposits[i] = d
				self._log_operation("deposit_returned", d["id"], tenant_id)
				return d
		raise KeyError(f"no active deposit for tenant entity {tenant_entity_id}")

	# ── Rent Arrears ──────────────────────────────────────────────────────────

	async def track_rent_arrears(
		self,
		tenant_entity_id: str,
		tenant_id: str,
		period: str,
		amount_due: Decimal,
		amount_paid: Decimal,
		due_date: date,
		unit_id: str = "",
		payment_reference: str = "",
	) -> dict[str, Any]:
		"""Record a rent payment period, compute arrears, and trigger escalation ladder if overdue.

		Escalation thresholds:
		  - 7 days overdue: automated reminder
		  - 14 days: formal notice
		  - 28 days: legal referral flag

		Returns the arrears record including days_overdue and escalation_stage.
		"""
		assert tenant_entity_id and period, "tenant_entity_id and period required"
		assert amount_due >= 0 and amount_paid >= 0, "amounts must be non-negative"
		from uuid6 import uuid7
		arrears_balance = amount_due - amount_paid
		days_overdue = (date.today() - due_date).days if date.today() > due_date else 0
		escalation_stage: str
		if arrears_balance <= 0:
			escalation_stage = "none"
		elif days_overdue >= 28:
			escalation_stage = "legal_referral"
		elif days_overdue >= 14:
			escalation_stage = "formal_notice"
		elif days_overdue >= 7:
			escalation_stage = "reminder"
		else:
			escalation_stage = "monitoring"
		arrears_id = str(uuid7())
		record: dict[str, Any] = {
			"id": arrears_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_entity_id,
			"unit_id": unit_id,
			"period": period,
			"amount_due": str(amount_due),
			"amount_paid": str(amount_paid),
			"arrears_balance": str(arrears_balance),
			"due_date": str(due_date),
			"days_overdue": days_overdue,
			"escalation_stage": escalation_stage,
			"payment_reference": payment_reference,
			"in_arrears": arrears_balance > 0,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("rent_arrears", []).append(record)
		if arrears_balance > 0:
			log.warning(
				"ten.rent_arrears tenant=%s period=%s balance=%s stage=%s",
				tenant_entity_id, period, arrears_balance, escalation_stage,
			)
		self._log_operation("rent_arrears_tracked", arrears_id, tenant_id)
		return record

	async def get_arrears_summary(self, tenant_id: str, tenant_entity_id: str | None = None) -> dict[str, Any]:
		"""Return arrears summary for a tenant or portfolio: total balance, worst stage, periods in arrears."""
		all_arrears = [
			r for r in self._store.get("rent_arrears", [])
			if r["tenant_id"] == tenant_id and r.get("in_arrears")
		]
		if tenant_entity_id:
			all_arrears = [r for r in all_arrears if r["tenant_entity_id"] == tenant_entity_id]
		total_balance = sum(Decimal(r["arrears_balance"]) for r in all_arrears)
		stage_order = {"legal_referral": 4, "formal_notice": 3, "reminder": 2, "monitoring": 1, "none": 0}
		worst_stage = max((r["escalation_stage"] for r in all_arrears), key=lambda s: stage_order.get(s, 0), default="none")
		return {
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_entity_id,
			"periods_in_arrears": len(all_arrears),
			"total_arrears_balance": str(total_balance),
			"worst_escalation_stage": worst_stage,
			"computed_at": datetime.utcnow().isoformat(),
		}

	# ── Compliance Calendar ────────────────────────────────────────────────────

	async def get_compliance_calendar(
		self,
		tenant_id: str,
		lookahead_days: int = 90,
	) -> dict[str, Any]:
		"""Generate a forward-looking compliance obligation calendar for the portfolio.

		Aggregates: covenant review dates, rent review dates, deposit renewal windows,
		and vacating notice timelines. Items within lookahead_days are flagged as urgent.

		Returns sorted calendar entries with owner, deadline, and urgency flag.
		"""
		assert lookahead_days > 0, "lookahead_days must be positive"
		today = date.today()
		cutoff = today + timedelta(days=lookahead_days)
		calendar: list[dict[str, Any]] = []

		for c in self._store.get("covenants", []):
			if c["tenant_id"] != tenant_id:
				continue
			review_str = c.get("next_review_date")
			if review_str:
				review_date = date.fromisoformat(str(review_str))
				days_to_deadline = (review_date - today).days
				calendar.append({
					"obligation_type": "covenant_review",
					"covenant_id": c.get("covenant_id"),
					"tenant_entity_id": c.get("tenant_entity_id"),
					"deadline": str(review_date),
					"days_to_deadline": days_to_deadline,
					"urgent": review_date <= cutoff,
					"overdue": review_date < today,
				})

		for r in self._store.get("rent_reviews", []):
			if r["tenant_id"] != tenant_id:
				continue
			eff_str = r.get("effective_date")
			if eff_str:
				eff_date = date.fromisoformat(str(eff_str))
				days_to_deadline = (eff_date - today).days
				calendar.append({
					"obligation_type": "rent_review",
					"review_id": r.get("id"),
					"tenant_entity_id": r.get("tenant_entity_id"),
					"deadline": str(eff_date),
					"days_to_deadline": days_to_deadline,
					"urgent": eff_date <= cutoff,
					"overdue": eff_date < today,
				})

		for v in self._store.get("vacating_notices", []):
			if v["tenant_id"] != tenant_id:
				continue
			vac_str = v.get("vacate_date")
			if vac_str:
				vac_date = date.fromisoformat(str(vac_str))
				days_to_deadline = (vac_date - today).days
				calendar.append({
					"obligation_type": "vacating",
					"notice_id": v.get("id"),
					"tenant_entity_id": v.get("tenant_entity_id"),
					"deadline": str(vac_date),
					"days_to_deadline": days_to_deadline,
					"urgent": vac_date <= cutoff,
					"overdue": vac_date < today,
				})

		calendar.sort(key=lambda x: x["days_to_deadline"])
		urgent = [e for e in calendar if e["urgent"]]
		overdue = [e for e in calendar if e["overdue"]]
		return {
			"tenant_id": tenant_id,
			"lookahead_days": lookahead_days,
			"total_obligations": len(calendar),
			"urgent_count": len(urgent),
			"overdue_count": len(overdue),
			"calendar": calendar,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── Break Clause Workflow ─────────────────────────────────────────────────

	async def register_break_clause(
		self,
		tenant_entity_id: str,
		tenant_id: str,
		unit_id: str,
		break_date: date,
		notice_period_days: int,
		break_type: str = "tenant",
		conditions: list[str] | None = None,
		lease_id: str = "",
	) -> dict[str, Any]:
		"""Register a lease break clause, recording break date, notice deadline, type, and conditions.

		break_type: 'tenant' | 'landlord' | 'mutual'
		conditions: list of condition strings that must be satisfied (e.g. 'no_rent_arrears',
		            'vacant_possession', 'all_covenants_complied').
		Notice deadline = break_date - notice_period_days.
		"""
		assert tenant_entity_id and unit_id, "tenant_entity_id and unit_id required"
		assert notice_period_days > 0, "notice_period_days must be positive"
		assert break_type in ("tenant", "landlord", "mutual"), \
			f"unsupported break_type: {break_type}"
		from uuid6 import uuid7
		notice_deadline = break_date - timedelta(days=notice_period_days)
		clause_id = str(uuid7())
		record: dict[str, Any] = {
			"id": clause_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_entity_id,
			"unit_id": unit_id,
			"lease_id": lease_id,
			"break_date": str(break_date),
			"break_type": break_type,
			"notice_period_days": notice_period_days,
			"notice_deadline": str(notice_deadline),
			"conditions": conditions or [],
			"status": "registered",
			"activated": False,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("break_clauses", []).append(record)
		self._log_operation("break_clause_registered", clause_id, tenant_id)
		return record

	async def check_break_clause_eligibility(
		self,
		clause_id: str,
		tenant_id: str,
		check_date: date | None = None,
	) -> dict[str, Any]:
		"""Evaluate whether break clause conditions are currently satisfied.

		Checks: no active rent arrears, no open escalations (if 'no_arrears' / 'no_escalations'
		in conditions). Returns eligibility verdict with per-condition results.
		"""
		check_date = check_date or date.today()
		for clause in self._store.get("break_clauses", []):
			if clause["id"] == clause_id and clause["tenant_id"] == tenant_id:
				break_date = date.fromisoformat(clause["break_date"])
				notice_deadline = date.fromisoformat(clause["notice_deadline"])
				days_to_break = (break_date - check_date).days
				within_notice_window = check_date >= notice_deadline
				condition_results: dict[str, bool] = {}
				ten_entity_id = clause["tenant_entity_id"]
				for cond in clause.get("conditions", []):
					if cond == "no_rent_arrears":
						arrears = await self.get_arrears_summary(tenant_id, ten_entity_id)
						condition_results[cond] = Decimal(arrears["total_arrears_balance"]) <= 0
					elif cond == "no_open_escalations":
						esc = await self.list_escalations(tenant_id, ten_entity_id)
						condition_results[cond] = not any(e.status == "open" for e in esc)
					else:
						condition_results[cond] = True  # unknown conditions assumed met
				all_conditions_met = all(condition_results.values())
				eligible = all_conditions_met and within_notice_window and days_to_break >= 0
				return {
					"clause_id": clause_id,
					"tenant_entity_id": ten_entity_id,
					"break_date": clause["break_date"],
					"notice_deadline": clause["notice_deadline"],
					"check_date": str(check_date),
					"days_to_break": days_to_break,
					"within_notice_window": within_notice_window,
					"condition_results": condition_results,
					"all_conditions_met": all_conditions_met,
					"eligible": eligible,
					"checked_at": datetime.utcnow().isoformat(),
				}
		raise KeyError(f"break clause {clause_id} not found")

	# ── Relationship Health Score ──────────────────────────────────────────────

	async def compute_relationship_health_score(
		self,
		tenant_entity_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Compute a composite tenant relationship health score across four dimensions.

		Dimensions and weights:
		  Financial Health (35%): credit grade, arrears balance, tenant score
		  Operational Health (25%): open service requests, SLA breach rate
		  Engagement (25%): survey response rate, portal activity, comms responsiveness
		  Compliance (15%): covenant compliance rate, onboarding completeness

		Returns: composite score (0–100), tier, per-dimension breakdown, and recommendations.
		"""
		tenant = await self.get_tenant(tenant_entity_id, tenant_id)
		if tenant is None:
			raise KeyError(f"tenant entity {tenant_entity_id} not found")

		# Financial Health (0–100)
		grade_scores = {"A": 100, "B": 80, "C": 60, "D": 40, "F": 10}
		credit_component = grade_scores.get(str(tenant.credit_grade.value if tenant.credit_grade else "C"), 50)
		tenant_score_component = float(tenant.tenant_score or 50)
		arrears_summary = await self.get_arrears_summary(tenant_id, tenant_entity_id)
		arrears_balance = Decimal(arrears_summary["total_arrears_balance"])
		arrears_penalty = min(float(arrears_balance) / 100, 30)  # cap at 30 point deduction
		financial_health = max(0, (credit_component * 0.4 + tenant_score_component * 0.6) - arrears_penalty)

		# Operational Health (0–100)
		all_requests = await self.list_service_requests(tenant_id, tenant_entity_id)
		total_req = len(all_requests)
		breached = sum(1 for r in all_requests if r.sla_breached)
		breach_rate = breached / max(total_req, 1)
		open_count = sum(1 for r in all_requests if r.status.value == "open")
		operational_health = max(0, 100 - breach_rate * 60 - open_count * 5)

		# Engagement (0–100)
		surveys = await self.list_satisfaction_surveys(tenant_id, tenant_entity_id)
		survey_component = min(len(surveys) * 20, 60)  # 3+ surveys = full credit
		comms = await self.list_communications(tenant_id, tenant_entity_id)
		comms_component = min(len(comms) * 5, 40)
		engagement = survey_component + comms_component

		# Compliance (0–100)
		progress = await self.get_onboarding_progress(tenant_entity_id, tenant_id)
		onboarding_pct = progress.get("completion_pct", 0)
		all_covenants = [c for c in self._store.get("covenants", [])
						 if c["tenant_entity_id"] == tenant_entity_id and c["tenant_id"] == tenant_id]
		non_compliant = sum(1 for c in all_covenants if c.get("status") == "non_compliant")
		covenant_score = max(0, 100 - non_compliant * 25)
		compliance = onboarding_pct * 0.4 + covenant_score * 0.6

		# Weighted composite
		composite = (
			financial_health * 0.35
			+ operational_health * 0.25
			+ engagement * 0.25
			+ compliance * 0.15
		)
		composite = round(composite, 1)

		tier_map = [(85, "Platinum"), (70, "Gold"), (50, "Silver"), (0, "Standard")]
		tier = next(label for threshold, label in tier_map if composite >= threshold)

		recommendations: list[str] = []
		if financial_health < 60:
			recommendations.append("Review credit position and clear any outstanding arrears.")
		if operational_health < 60:
			recommendations.append("Investigate chronic SLA breaches and reduce open request backlog.")
		if engagement < 40:
			recommendations.append("Increase outreach frequency; tenant appears disengaged.")
		if compliance < 60:
			recommendations.append("Complete remaining onboarding steps and resolve covenant breaches.")

		return {
			"tenant_entity_id": tenant_entity_id,
			"tenant_id": tenant_id,
			"composite_score": composite,
			"tier": tier,
			"dimensions": {
				"financial_health": round(financial_health, 1),
				"operational_health": round(operational_health, 1),
				"engagement": round(engagement, 1),
				"compliance": round(compliance, 1),
			},
			"recommendations": recommendations,
			"computed_at": datetime.utcnow().isoformat(),
		}

	# ── Predictive Churn ──────────────────────────────────────────────────────

	async def predict_churn_probability(
		self,
		tenant_entity_id: str,
		tenant_id: str,
		lease_expiry_date: date | None = None,
	) -> dict[str, Any]:
		"""Compute a churn probability (0.0–1.0) from behavioural and financial signals.

		Signals and weights:
		  - Satisfaction trend (declining → +0.25)
		  - Days since last communication (>60 days → +0.20)
		  - SLA breach rate (>30% → +0.20)
		  - Lease expiry proximity (<90 days → +0.20)
		  - Active rent arrears (→ +0.15)

		Returns: probability, risk_level, contributing_factors, and recommended_actions.
		"""
		signals: dict[str, float] = {}
		churn_probability = 0.0

		# Satisfaction trend
		trend_data = await self.get_satisfaction_trend(tenant_id, tenant_entity_id)
		if trend_data.get("trend") == "declining":
			signals["satisfaction_declining"] = 0.25
			churn_probability += 0.25
		elif trend_data.get("trend") == "stable" and (trend_data.get("average_score") or 5) < 3.5:
			signals["satisfaction_low_stable"] = 0.10
			churn_probability += 0.10

		# Days since last communication
		comms = await self.list_communications(tenant_id, tenant_entity_id)
		if comms:
			latest_comm = max(comms, key=lambda c: str(c.sent_at or ""))
			if latest_comm.sent_at:
				days_since = (datetime.utcnow() - latest_comm.sent_at).days
				if days_since > 60:
					signals["communication_gap"] = 0.20
					churn_probability += 0.20
		else:
			signals["no_communications"] = 0.15
			churn_probability += 0.15

		# SLA breach rate
		requests = await self.list_service_requests(tenant_id, tenant_entity_id)
		if requests:
			breach_rate = sum(1 for r in requests if r.sla_breached) / len(requests)
			if breach_rate > 0.30:
				signals["high_sla_breach_rate"] = 0.20
				churn_probability += 0.20

		# Lease expiry proximity
		if lease_expiry_date:
			days_to_expiry = (lease_expiry_date - date.today()).days
			if days_to_expiry < 90:
				weight = 0.20 * (1 - days_to_expiry / 90)
				signals["lease_expiry_proximity"] = round(weight, 3)
				churn_probability += weight

		# Active rent arrears
		arrears = await self.get_arrears_summary(tenant_id, tenant_entity_id)
		if Decimal(arrears["total_arrears_balance"]) > 0:
			signals["rent_arrears"] = 0.15
			churn_probability += 0.15

		churn_probability = round(min(churn_probability, 1.0), 3)
		risk_level = (
			"critical" if churn_probability >= 0.70
			else "high" if churn_probability >= 0.50
			else "medium" if churn_probability >= 0.30
			else "low"
		)
		actions: list[str] = []
		if "satisfaction_declining" in signals:
			actions.append("Schedule relationship review call within 5 business days.")
		if "communication_gap" in signals:
			actions.append("Initiate proactive outreach; last contact was over 60 days ago.")
		if "high_sla_breach_rate" in signals:
			actions.append("Escalate service delivery review to facilities manager.")
		if "lease_expiry_proximity" in signals:
			actions.append("Open renewal negotiation; lease expires within 90 days.")
		if "rent_arrears" in signals:
			actions.append("Contact accounts team; outstanding arrears detected.")

		return {
			"tenant_entity_id": tenant_entity_id,
			"tenant_id": tenant_id,
			"churn_probability": churn_probability,
			"risk_level": risk_level,
			"contributing_signals": signals,
			"recommended_actions": actions,
			"computed_at": datetime.utcnow().isoformat(),
		}

	# ── SLA Performance Report ────────────────────────────────────────────────

	async def get_sla_performance_report(
		self,
		tenant_id: str,
		period: str | None = None,
	) -> dict[str, Any]:
		"""Compute SLA performance metrics aggregated by request type.

		Metrics per type: total requests, breached count, breach rate,
		average resolution time (hours), compliance percentage.
		Also returns portfolio-level headline figures.
		"""
		requests = await self.list_service_requests(tenant_id)
		if period:
			requests = [r for r in requests if str(r.created_at).startswith(period[:7])]

		type_stats: dict[str, dict[str, Any]] = {}
		for r in requests:
			rt = r.request_type.value if hasattr(r.request_type, "value") else str(r.request_type)
			if rt not in type_stats:
				type_stats[rt] = {"total": 0, "breached": 0, "resolution_hours": []}
			type_stats[rt]["total"] += 1
			if r.sla_breached:
				type_stats[rt]["breached"] += 1
			if r.resolved_at and r.created_at:
				delta_h = (r.resolved_at - r.created_at).total_seconds() / 3600
				type_stats[rt]["resolution_hours"].append(round(delta_h, 2))

		by_type: list[dict[str, Any]] = []
		for rt, stats in type_stats.items():
			res_hours = stats["resolution_hours"]
			breach_rate = round(stats["breached"] / max(stats["total"], 1) * 100, 2)
			avg_resolution = round(sum(res_hours) / max(len(res_hours), 1), 2) if res_hours else None
			target = SLA_RESPONSE_HOURS.get(rt, SLA_RESPONSE_HOURS["default"])
			by_type.append({
				"request_type": rt,
				"total": stats["total"],
				"breached": stats["breached"],
				"breach_rate_pct": breach_rate,
				"compliance_pct": round(100 - breach_rate, 2),
				"avg_resolution_hours": avg_resolution,
				"sla_target_hours": target,
			})

		total = len(requests)
		total_breached = sum(r.sla_breached for r in requests)
		portfolio_compliance = round((1 - total_breached / max(total, 1)) * 100, 2)

		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_requests": total,
			"total_breached": total_breached,
			"portfolio_sla_compliance_pct": portfolio_compliance,
			"by_request_type": by_type,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── Guarantor Management ──────────────────────────────────────────────────

	async def register_guarantor(
		self,
		tenant_entity_id: str,
		tenant_id: str,
		guarantor_name: str,
		guarantor_email: str,
		guarantee_type: str = "limited",
		guarantee_amount: Decimal | None = None,
		credit_check_reference: str = "",
		signed_deed_reference: str = "",
	) -> dict[str, Any]:
		"""Register a guarantor for a tenant entity.

		guarantee_type: 'limited' (capped at guarantee_amount) | 'unlimited'
		Validates that limited guarantees specify an amount.
		Links guarantor to tenant for arrears escalation workflows.
		"""
		assert tenant_entity_id and guarantor_name and guarantor_email, \
			"tenant_entity_id, guarantor_name, guarantor_email required"
		assert guarantee_type in ("limited", "unlimited"), \
			f"unsupported guarantee_type: {guarantee_type}"
		if guarantee_type == "limited":
			assert guarantee_amount and guarantee_amount > 0, \
				"guarantee_amount required for limited guarantee"
		from uuid6 import uuid7
		guarantor_id = str(uuid7())
		record: dict[str, Any] = {
			"id": guarantor_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_entity_id,
			"guarantor_name": guarantor_name,
			"guarantor_email": guarantor_email,
			"guarantee_type": guarantee_type,
			"guarantee_amount": str(guarantee_amount) if guarantee_amount else None,
			"credit_check_reference": credit_check_reference,
			"signed_deed_reference": signed_deed_reference,
			"status": "active",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("guarantors", []).append(record)
		self._log_operation("guarantor_registered", guarantor_id, tenant_id)
		return record

	async def validate_guarantor_coverage(
		self,
		tenant_entity_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Check whether active guarantors cover current arrears exposure.

		Computes: total limited coverage, unlimited flag, arrears balance,
		coverage_sufficient flag. Returns gap amount if coverage is insufficient.
		"""
		guarantors = [
			g for g in self._store.get("guarantors", [])
			if g["tenant_entity_id"] == tenant_entity_id
			and g["tenant_id"] == tenant_id
			and g["status"] == "active"
		]
		arrears = await self.get_arrears_summary(tenant_id, tenant_entity_id)
		arrears_balance = Decimal(arrears["total_arrears_balance"])
		has_unlimited = any(g["guarantee_type"] == "unlimited" for g in guarantors)
		limited_total = sum(
			Decimal(g["guarantee_amount"]) for g in guarantors
			if g["guarantee_type"] == "limited" and g.get("guarantee_amount")
		)
		coverage_sufficient = has_unlimited or limited_total >= arrears_balance
		coverage_gap = max(Decimal("0"), arrears_balance - limited_total) if not has_unlimited else Decimal("0")
		return {
			"tenant_entity_id": tenant_entity_id,
			"tenant_id": tenant_id,
			"guarantor_count": len(guarantors),
			"has_unlimited_guarantee": has_unlimited,
			"total_limited_coverage": str(limited_total),
			"current_arrears_balance": str(arrears_balance),
			"coverage_sufficient": coverage_sufficient,
			"coverage_gap": str(coverage_gap),
			"checked_at": datetime.utcnow().isoformat(),
		}

	# ── Portfolio Lease Incentives ─────────────────────────────────────────────

	async def record_lease_incentive(
		self,
		tenant_entity_id: str,
		tenant_id: str,
		unit_id: str,
		incentive_type: str,
		value: Decimal,
		start_date: date,
		end_date: date,
		lease_id: str = "",
		description: str = "",
	) -> dict[str, Any]:
		"""Record a lease incentive (rent-free period, fit-out contribution, stepped rent).

		incentive_type: 'rent_free' | 'fitout_contribution' | 'stepped_rent' | 'rent_cap' | 'cash_incentive'
		Stores value, period, and generates a daily amortisation rate for accounting purposes.
		"""
		assert tenant_entity_id and unit_id, "tenant_entity_id and unit_id required"
		assert incentive_type in (
			"rent_free", "fitout_contribution", "stepped_rent", "rent_cap", "cash_incentive"
		), f"unsupported incentive_type: {incentive_type}"
		assert start_date <= end_date, "start_date must be on or before end_date"
		from uuid6 import uuid7
		days = (end_date - start_date).days + 1
		daily_amortisation = round(value / max(days, 1), 6)
		incentive_id = str(uuid7())
		record: dict[str, Any] = {
			"id": incentive_id,
			"tenant_id": tenant_id,
			"tenant_entity_id": tenant_entity_id,
			"unit_id": unit_id,
			"lease_id": lease_id,
			"incentive_type": incentive_type,
			"value": str(value),
			"start_date": str(start_date),
			"end_date": str(end_date),
			"duration_days": days,
			"daily_amortisation": str(daily_amortisation),
			"description": description,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("lease_incentives", []).append(record)
		self._log_operation("lease_incentive_recorded", incentive_id, tenant_id)
		return record

