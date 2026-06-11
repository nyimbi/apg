"""Professional Development async service."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, date
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "hcm_pro"
SKILL_CATEGORIES = {"technical", "leadership", "communication", "analytical", "domain", "soft"}
PROFICIENCY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]
MEETING_FREQUENCIES = {"weekly", "fortnightly", "monthly", "quarterly"}
ACTIVITY_TYPES = {"course", "workshop", "conference", "book", "elearning", "coaching", "webinar"}
FEEDBACK_RATER_TYPES = {"self", "peer", "manager", "skip_level", "report"}
PDI_WEIGHTS = {"plan_completion": 0.25, "skill_gap_closure": 0.25, "certifications": 0.20, "career_milestones": 0.20, "mentoring": 0.10}


class PROService:
	"""Professional Development — dev plans, skills, mentoring, certifications, career paths."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.development_plans: dict[str, dict[str, Any]] = {}
		self.plan_templates: dict[str, dict[str, Any]] = {}
		self.skills: dict[str, dict[str, Any]] = {}
		self.skill_assessments: dict[str, dict[str, Any]] = {}
		self.skill_endorsements: dict[str, dict[str, Any]] = {}
		self.mentoring_programmes: dict[str, dict[str, Any]] = {}
		self.mentoring_sessions: dict[str, dict[str, Any]] = {}
		self.certifications: dict[str, dict[str, Any]] = {}
		self.career_paths: dict[str, dict[str, Any]] = {}
		self.learning_activities: dict[str, dict[str, Any]] = {}
		self.learning_budgets: dict[str, dict[str, Any]] = {}
		self.training_providers: dict[str, dict[str, Any]] = {}
		self.feedback_requests: dict[str, dict[str, Any]] = {}
		self.feedback_responses: dict[str, dict[str, Any]] = {}
		self.pdi_snapshots: dict[str, list[dict[str, Any]]] = {}
		self._audit_events: list[dict[str, Any]] = []

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _uid(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _emit(self, tenant_id: str, event_type: str, entity_type: str, entity_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"id": self._uid("evt"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_type": entity_type,
			"entity_id": entity_id,
			"payload": deepcopy(payload),
			"emitted_at": self._now(),
		})

	def _days_to_expiry(self, expiry_date: str | None) -> int | None:
		if not expiry_date:
			return None
		try:
			exp = date.fromisoformat(expiry_date)
			return (exp - date.today()).days
		except Exception:
			return None

	def _level_index(self, level: str) -> int:
		try:
			return PROFICIENCY_LEVELS.index(level)
		except ValueError:
			return -1

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"development_plans": len(self.development_plans),
			"plan_templates": len(self.plan_templates),
			"skills": len(self.skills),
			"skill_assessments": len(self.skill_assessments),
			"skill_endorsements": len(self.skill_endorsements),
			"mentoring_programmes": len(self.mentoring_programmes),
			"certifications": len(self.certifications),
			"career_paths": len(self.career_paths),
			"learning_activities": len(self.learning_activities),
			"training_providers": len(self.training_providers),
			"feedback_requests": len(self.feedback_requests),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "hcm",
			"version": "1.0.0",
			"description": "Professional Development — plans, skills, mentoring, certifications, career paths",
			"skill_categories": sorted(SKILL_CATEGORIES),
			"proficiency_levels": PROFICIENCY_LEVELS,
			"meeting_frequencies": sorted(MEETING_FREQUENCIES),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == t]

	# ── Development Plans ─────────────────────────────────────────────────────

	async def create_development_plan(
		self,
		tenant_id: str,
		employee_id: str,
		plan_year: int,
		objectives: list[str] | None = None,
		focus_areas: list[str] | None = None,
		target_role_id: str | None = None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		"""Create a personal development plan for an employee."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		record: dict[str, Any] = {
			"id": self._uid("dp"),
			"tenant_id": t,
			"employee_id": employee_id,
			"plan_year": plan_year,
			"objectives": objectives or [],
			"focus_areas": focus_areas or [],
			"target_role_id": target_role_id,
			"reviewed_by": reviewed_by,
			"completion_notes": None,
			"status": "draft",
			"progress_pct": 0.0,
			"created_at": self._now(),
			"updated_at": None,
		}
		self.development_plans[record["id"]] = record
		self._emit(t, "development_plan_created", "development_plan", record["id"], record)
		_log.info("development_plan created: %s employee=%s", record["id"], employee_id)
		return deepcopy(record)

	async def list_development_plans(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		status: str | None = None,
		plan_year: int | None = None,
	) -> list[dict[str, Any]]:
		"""List development plans."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.development_plans.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if status:
			items = [r for r in items if r["status"] == status]
		if plan_year:
			items = [r for r in items if r["plan_year"] == plan_year]
		return items

	async def get_development_plan(self, tenant_id: str, plan_id: str) -> dict[str, Any]:
		"""Get a development plan by ID."""
		t = self._tenant(tenant_id)
		record = self.development_plans.get(plan_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"development_plan {plan_id} not found")
		return deepcopy(record)

	async def update_development_plan(self, tenant_id: str, plan_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a development plan."""
		t = self._tenant(tenant_id)
		record = self.development_plans.get(plan_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"development_plan {plan_id} not found")
		allowed = {"objectives", "focus_areas", "target_role_id", "status", "reviewed_by", "completion_notes", "progress_pct"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		record["updated_at"] = self._now()
		self._emit(t, "development_plan_updated", "development_plan", record["id"], record)
		return deepcopy(record)

	async def activate_development_plan(self, tenant_id: str, plan_id: str, reviewed_by: str) -> dict[str, Any]:
		"""Activate a draft development plan."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(reviewed_by, "reviewed_by")
		record = self.development_plans.get(plan_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"development_plan {plan_id} not found")
		if record["status"] != "draft":
			raise PermissionError("only_draft_plans_can_be_activated")
		record["status"] = "active"
		record["reviewed_by"] = reviewed_by
		record["updated_at"] = self._now()
		self._emit(t, "development_plan_activated", "development_plan", record["id"], record)
		return deepcopy(record)

	async def update_plan_progress(self, tenant_id: str, plan_id: str, progress_pct: float) -> dict[str, Any]:
		"""Update progress percentage of a development plan."""
		t = self._tenant(tenant_id)
		if not 0.0 <= progress_pct <= 100.0:
			raise ValueError("progress_pct must be between 0 and 100")
		record = self.development_plans.get(plan_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"development_plan {plan_id} not found")
		record["progress_pct"] = progress_pct
		if progress_pct >= 100.0:
			record["status"] = "completed"
		record["updated_at"] = self._now()
		self._emit(t, "development_plan_progress_updated", "development_plan", record["id"], {"progress_pct": progress_pct})
		return deepcopy(record)

	async def delete_development_plan(self, tenant_id: str, plan_id: str) -> bool:
		"""Delete a draft development plan."""
		t = self._tenant(tenant_id)
		record = self.development_plans.get(plan_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"development_plan {plan_id} not found")
		if record["status"] != "draft":
			raise PermissionError("only_draft_plans_can_be_deleted")
		del self.development_plans[plan_id]
		self._emit(t, "development_plan_deleted", "development_plan", plan_id, {"id": plan_id})
		return True

	# ── Skills ────────────────────────────────────────────────────────────────

	async def create_skill(
		self,
		tenant_id: str,
		name: str,
		category: str,
		description: str | None = None,
	) -> dict[str, Any]:
		"""Define a skill in the skills catalogue."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		if category not in SKILL_CATEGORIES:
			raise ValueError(f"category must be one of {SKILL_CATEGORIES}")
		record: dict[str, Any] = {
			"id": self._uid("sk"),
			"tenant_id": t,
			"name": name,
			"category": category,
			"description": description,
			"proficiency_levels": PROFICIENCY_LEVELS,
			"status": "active",
			"created_at": self._now(),
		}
		self.skills[record["id"]] = record
		self._emit(t, "skill_created", "skill", record["id"], record)
		return deepcopy(record)

	async def list_skills(self, tenant_id: str, category: str | None = None) -> list[dict[str, Any]]:
		"""List skills in the catalogue."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.skills.values() if r["tenant_id"] == t]
		if category:
			items = [r for r in items if r["category"] == category]
		return items

	async def get_skill(self, tenant_id: str, skill_id: str) -> dict[str, Any]:
		"""Get a skill by ID."""
		t = self._tenant(tenant_id)
		record = self.skills.get(skill_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"skill {skill_id} not found")
		return deepcopy(record)

	async def update_skill(self, tenant_id: str, skill_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a skill definition."""
		t = self._tenant(tenant_id)
		record = self.skills.get(skill_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"skill {skill_id} not found")
		for k, v in kwargs.items():
			if k in {"name", "description", "status"} and v is not None:
				record[k] = v
		self._emit(t, "skill_updated", "skill", record["id"], record)
		return deepcopy(record)

	async def delete_skill(self, tenant_id: str, skill_id: str) -> bool:
		"""Delete a skill from the catalogue."""
		t = self._tenant(tenant_id)
		record = self.skills.get(skill_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"skill {skill_id} not found")
		in_use = [a for a in self.skill_assessments.values() if a["tenant_id"] == t and a["skill_id"] == skill_id]
		if in_use:
			raise PermissionError("skill_in_use_by_assessments")
		del self.skills[skill_id]
		self._emit(t, "skill_deleted", "skill", skill_id, {"id": skill_id})
		return True

	# ── Skill Assessments / Gap Analysis ─────────────────────────────────────

	async def assess_skill(
		self,
		tenant_id: str,
		employee_id: str,
		skill_id: str,
		current_level: str,
		target_level: str,
		assessed_by: str | None = None,
		evidence: str | None = None,
	) -> dict[str, Any]:
		"""Record a skill assessment for an employee."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		skill = self.skills.get(skill_id)
		if not skill or skill["tenant_id"] != t:
			raise KeyError(f"skill {skill_id} not found")
		if current_level not in PROFICIENCY_LEVELS:
			raise ValueError(f"current_level must be one of {PROFICIENCY_LEVELS}")
		if target_level not in PROFICIENCY_LEVELS:
			raise ValueError(f"target_level must be one of {PROFICIENCY_LEVELS}")
		gap_exists = self._level_index(current_level) < self._level_index(target_level)
		record: dict[str, Any] = {
			"id": self._uid("sa"),
			"tenant_id": t,
			"employee_id": employee_id,
			"skill_id": skill_id,
			"skill_name": skill["name"],
			"current_level": current_level,
			"target_level": target_level,
			"gap_exists": gap_exists,
			"assessed_by": assessed_by,
			"evidence": evidence,
			"status": "current",
			"assessed_at": self._now(),
		}
		self.skill_assessments[record["id"]] = record
		self._emit(t, "skill_assessed", "skill_assessment", record["id"], record)
		return deepcopy(record)

	async def list_skill_assessments(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		gap_only: bool = False,
	) -> list[dict[str, Any]]:
		"""List skill assessments."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.skill_assessments.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if gap_only:
			items = [r for r in items if r["gap_exists"]]
		return items

	async def get_skill_gap_report(self, tenant_id: str, employee_id: str) -> dict[str, Any]:
		"""Generate a skill gap report for an employee."""
		t = self._tenant(tenant_id)
		assessments = await self.list_skill_assessments(t, employee_id=employee_id)
		gaps = [a for a in assessments if a["gap_exists"]]
		return {
			"employee_id": employee_id,
			"total_skills_assessed": len(assessments),
			"skills_with_gaps": len(gaps),
			"gap_details": gaps,
			"overall_readiness": "ready" if not gaps else ("partial" if len(gaps) < len(assessments) / 2 else "needs_development"),
			"generated_at": self._now(),
		}

	# ── Mentoring Programmes ──────────────────────────────────────────────────

	async def create_mentoring_programme(
		self,
		tenant_id: str,
		mentee_employee_id: str,
		mentor_employee_id: str,
		programme_name: str,
		start_date: str,
		objectives: list[str] | None = None,
		end_date: str | None = None,
		meeting_frequency: str = "monthly",
	) -> dict[str, Any]:
		"""Create a mentoring programme pairing."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(mentee_employee_id, "mentee_employee_id")
		guard_non_empty_string(mentor_employee_id, "mentor_employee_id")
		if mentee_employee_id == mentor_employee_id:
			raise ValueError("mentee_and_mentor_cannot_be_the_same_person")
		if meeting_frequency not in MEETING_FREQUENCIES:
			raise ValueError(f"meeting_frequency must be one of {MEETING_FREQUENCIES}")
		record: dict[str, Any] = {
			"id": self._uid("mp"),
			"tenant_id": t,
			"mentee_employee_id": mentee_employee_id,
			"mentor_employee_id": mentor_employee_id,
			"programme_name": programme_name,
			"objectives": objectives or [],
			"start_date": start_date,
			"end_date": end_date,
			"meeting_frequency": meeting_frequency,
			"sessions_completed": 0,
			"status": "active",
			"completion_notes": None,
			"created_at": self._now(),
		}
		self.mentoring_programmes[record["id"]] = record
		self._emit(t, "mentoring_programme_created", "mentoring_programme", record["id"], record)
		return deepcopy(record)

	async def list_mentoring_programmes(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		role: str | None = None,  # mentee or mentor
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List mentoring programmes."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.mentoring_programmes.values() if r["tenant_id"] == t]
		if employee_id and role == "mentee":
			items = [r for r in items if r["mentee_employee_id"] == employee_id]
		elif employee_id and role == "mentor":
			items = [r for r in items if r["mentor_employee_id"] == employee_id]
		elif employee_id:
			items = [r for r in items if r["mentee_employee_id"] == employee_id or r["mentor_employee_id"] == employee_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def get_mentoring_programme(self, tenant_id: str, programme_id: str) -> dict[str, Any]:
		"""Get a mentoring programme by ID."""
		t = self._tenant(tenant_id)
		record = self.mentoring_programmes.get(programme_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"mentoring_programme {programme_id} not found")
		return deepcopy(record)

	async def update_mentoring_programme(self, tenant_id: str, programme_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a mentoring programme."""
		t = self._tenant(tenant_id)
		record = self.mentoring_programmes.get(programme_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"mentoring_programme {programme_id} not found")
		allowed = {"objectives", "status", "end_date", "completion_notes", "meeting_frequency"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		self._emit(t, "mentoring_programme_updated", "mentoring_programme", record["id"], record)
		return deepcopy(record)

	async def log_mentoring_session(
		self,
		tenant_id: str,
		programme_id: str,
		session_date: str,
		topics_covered: list[str],
		action_items: list[str] | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Log a mentoring session."""
		t = self._tenant(tenant_id)
		programme = self.mentoring_programmes.get(programme_id)
		if not programme or programme["tenant_id"] != t:
			raise KeyError(f"mentoring_programme {programme_id} not found")
		session: dict[str, Any] = {
			"id": self._uid("ms"),
			"tenant_id": t,
			"programme_id": programme_id,
			"session_date": session_date,
			"topics_covered": topics_covered,
			"action_items": action_items or [],
			"notes": notes,
			"created_at": self._now(),
		}
		self.mentoring_sessions[session["id"]] = session
		programme["sessions_completed"] += 1
		self._emit(t, "mentoring_session_logged", "mentoring_session", session["id"], session)
		return deepcopy(session)

	async def delete_mentoring_programme(self, tenant_id: str, programme_id: str) -> bool:
		"""Delete a mentoring programme."""
		t = self._tenant(tenant_id)
		record = self.mentoring_programmes.get(programme_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"mentoring_programme {programme_id} not found")
		del self.mentoring_programmes[programme_id]
		self._emit(t, "mentoring_programme_deleted", "mentoring_programme", programme_id, {"id": programme_id})
		return True

	# ── Certifications ────────────────────────────────────────────────────────

	async def add_certification(
		self,
		tenant_id: str,
		employee_id: str,
		certification_name: str,
		issuing_body: str,
		issue_date: str,
		expiry_date: str | None = None,
		credential_id: str | None = None,
		certificate_url: str | None = None,
	) -> dict[str, Any]:
		"""Record a professional certification for an employee."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		guard_non_empty_string(certification_name, "certification_name")
		days_to_exp = self._days_to_expiry(expiry_date)
		record: dict[str, Any] = {
			"id": self._uid("cert"),
			"tenant_id": t,
			"employee_id": employee_id,
			"certification_name": certification_name,
			"issuing_body": issuing_body,
			"issue_date": issue_date,
			"expiry_date": expiry_date,
			"renewal_date": None,
			"credential_id": credential_id,
			"certificate_url": certificate_url,
			"days_to_expiry": days_to_exp,
			"status": "expired" if (days_to_exp is not None and days_to_exp < 0) else "active",
			"created_at": self._now(),
		}
		self.certifications[record["id"]] = record
		self._emit(t, "certification_added", "certification", record["id"], record)
		return deepcopy(record)

	async def list_certifications(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		expiring_within_days: int | None = None,
	) -> list[dict[str, Any]]:
		"""List certifications, optionally filtering by expiry window."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.certifications.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		# Refresh days_to_expiry
		for item in items:
			item["days_to_expiry"] = self._days_to_expiry(item.get("expiry_date"))
		if expiring_within_days is not None:
			items = [r for r in items if r["days_to_expiry"] is not None and 0 <= r["days_to_expiry"] <= expiring_within_days]
		return items

	async def get_certification(self, tenant_id: str, cert_id: str) -> dict[str, Any]:
		"""Get a certification by ID."""
		t = self._tenant(tenant_id)
		record = self.certifications.get(cert_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"certification {cert_id} not found")
		result = deepcopy(record)
		result["days_to_expiry"] = self._days_to_expiry(result.get("expiry_date"))
		return result

	async def update_certification(self, tenant_id: str, cert_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a certification record."""
		t = self._tenant(tenant_id)
		record = self.certifications.get(cert_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"certification {cert_id} not found")
		allowed = {"expiry_date", "renewal_date", "certificate_url", "status"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		record["days_to_expiry"] = self._days_to_expiry(record.get("expiry_date"))
		self._emit(t, "certification_updated", "certification", record["id"], record)
		return deepcopy(record)

	async def delete_certification(self, tenant_id: str, cert_id: str) -> bool:
		"""Delete a certification record."""
		t = self._tenant(tenant_id)
		record = self.certifications.get(cert_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"certification {cert_id} not found")
		del self.certifications[cert_id]
		self._emit(t, "certification_deleted", "certification", cert_id, {"id": cert_id})
		return True

	# ── Career Paths ──────────────────────────────────────────────────────────

	async def create_career_path(
		self,
		tenant_id: str,
		employee_id: str,
		current_role: str,
		target_role: str,
		target_timeline_months: int = 24,
		milestones: list[dict[str, Any]] | None = None,
		advisor_employee_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a career path plan for an employee."""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		guard_non_empty_string(current_role, "current_role")
		guard_non_empty_string(target_role, "target_role")
		record: dict[str, Any] = {
			"id": self._uid("cp"),
			"tenant_id": t,
			"employee_id": employee_id,
			"current_role": current_role,
			"target_role": target_role,
			"target_timeline_months": target_timeline_months,
			"milestones": milestones or [],
			"milestones_completed": 0,
			"advisor_employee_id": advisor_employee_id,
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.career_paths[record["id"]] = record
		self._emit(t, "career_path_created", "career_path", record["id"], record)
		return deepcopy(record)

	async def list_career_paths(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List career paths."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.career_paths.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def get_career_path(self, tenant_id: str, path_id: str) -> dict[str, Any]:
		"""Get a career path by ID."""
		t = self._tenant(tenant_id)
		record = self.career_paths.get(path_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"career_path {path_id} not found")
		return deepcopy(record)

	async def update_career_path(self, tenant_id: str, path_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a career path."""
		t = self._tenant(tenant_id)
		record = self.career_paths.get(path_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"career_path {path_id} not found")
		allowed = {"target_role", "target_timeline_months", "milestones", "status", "advisor_employee_id"}
		for k, v in kwargs.items():
			if k in allowed and v is not None:
				record[k] = v
		if "milestones" in kwargs:
			record["milestones_completed"] = sum(1 for m in record["milestones"] if m.get("completed"))
		record["updated_at"] = self._now()
		self._emit(t, "career_path_updated", "career_path", record["id"], record)
		return deepcopy(record)

	async def complete_milestone(self, tenant_id: str, path_id: str, milestone_index: int) -> dict[str, Any]:
		"""Mark a milestone as completed on a career path."""
		t = self._tenant(tenant_id)
		record = self.career_paths.get(path_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"career_path {path_id} not found")
		milestones = record["milestones"]
		if milestone_index < 0 or milestone_index >= len(milestones):
			raise IndexError(f"milestone index {milestone_index} out of range")
		milestones[milestone_index]["completed"] = True
		milestones[milestone_index]["completed_at"] = self._now()
		record["milestones_completed"] = sum(1 for m in milestones if m.get("completed"))
		if record["milestones_completed"] == len(milestones) and milestones:
			record["status"] = "achieved"
		record["updated_at"] = self._now()
		self._emit(t, "career_milestone_completed", "career_path", record["id"], {"milestone_index": milestone_index})
		return deepcopy(record)

	async def delete_career_path(self, tenant_id: str, path_id: str) -> bool:
		"""Delete a career path."""
		t = self._tenant(tenant_id)
		record = self.career_paths.get(path_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"career_path {path_id} not found")
		del self.career_paths[path_id]
		self._emit(t, "career_path_deleted", "career_path", path_id, {"id": path_id})
		return True

	# ── Analytics & Dashboard ─────────────────────────────────────────────────

	async def professional_development_report(self, tenant_id: str, employee_id: str) -> dict[str, Any]:
		"""Full professional development report for one employee."""
		t = self._tenant(tenant_id)
		plans, gaps, certs, career, programmes = await asyncio.gather(
			self.list_development_plans(t, employee_id=employee_id),
			self.get_skill_gap_report(t, employee_id),
			self.list_certifications(t, employee_id=employee_id),
			self.list_career_paths(t, employee_id=employee_id),
			self.list_mentoring_programmes(t, employee_id=employee_id),
			return_exceptions=True,
		)
		return {
			"employee_id": employee_id,
			"development_plans": plans if not isinstance(plans, Exception) else [],
			"skill_gap_report": gaps if not isinstance(gaps, Exception) else {},
			"certifications": certs if not isinstance(certs, Exception) else [],
			"career_paths": career if not isinstance(career, Exception) else [],
			"mentoring_programmes": programmes if not isinstance(programmes, Exception) else [],
			"generated_at": self._now(),
		}

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Tenant-level professional development dashboard."""
		t = self._tenant(tenant_id)
		return {
			"tenant_id": t,
			"development_plans": {
				"total": sum(1 for r in self.development_plans.values() if r["tenant_id"] == t),
				"active": sum(1 for r in self.development_plans.values() if r["tenant_id"] == t and r["status"] == "active"),
			},
			"skill_assessments": sum(1 for r in self.skill_assessments.values() if r["tenant_id"] == t),
			"skills_with_gaps": sum(1 for r in self.skill_assessments.values() if r["tenant_id"] == t and r["gap_exists"]),
			"mentoring_programmes": {
				"total": sum(1 for r in self.mentoring_programmes.values() if r["tenant_id"] == t),
				"active": sum(1 for r in self.mentoring_programmes.values() if r["tenant_id"] == t and r["status"] == "active"),
			},
			"certifications": {
				"total": sum(1 for r in self.certifications.values() if r["tenant_id"] == t),
				"expiring_soon": sum(
					1 for r in self.certifications.values()
					if r["tenant_id"] == t and (d := self._days_to_expiry(r.get("expiry_date"))) is not None and 0 <= d <= 90
				),
			},
			"career_paths": sum(1 for r in self.career_paths.values() if r["tenant_id"] == t),
			"learning_activities": sum(1 for r in self.learning_activities.values() if r["tenant_id"] == t),
			"generated_at": self._now(),
		}

	# ── Plan Templates ────────────────────────────────────────────────────────

	async def create_plan_template(
		self,
		tenant_id: str,
		name: str,
		target_role: str | None = None,
		department: str | None = None,
		objectives: list[str] | None = None,
		focus_areas: list[str] | None = None,
		recommended_skills: list[dict[str, Any]] | None = None,
		created_by: str | None = None,
	) -> dict[str, Any]:
		"""Create a reusable development plan template for a role or department.

		Templates seed new plans with standardised objectives and focus areas,
		reducing authoring effort and ensuring institutional consistency.
		"""
		t = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		record: dict[str, Any] = {
			"id": self._uid("pt"),
			"tenant_id": t,
			"name": name,
			"target_role": target_role,
			"department": department,
			"objectives": objectives or [],
			"focus_areas": focus_areas or [],
			"recommended_skills": recommended_skills or [],
			"created_by": created_by,
			"usage_count": 0,
			"status": "active",
			"created_at": self._now(),
		}
		self.plan_templates[record["id"]] = record
		self._emit(t, "plan_template_created", "plan_template", record["id"], record)
		_log.info("plan_template created: %s name=%s", record["id"], name)
		return deepcopy(record)

	async def list_plan_templates(
		self,
		tenant_id: str,
		target_role: str | None = None,
		department: str | None = None,
	) -> list[dict[str, Any]]:
		"""List development plan templates, optionally filtered by role or department."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.plan_templates.values() if r["tenant_id"] == t]
		if target_role:
			items = [r for r in items if r.get("target_role") == target_role]
		if department:
			items = [r for r in items if r.get("department") == department]
		return items

	async def apply_plan_template(
		self,
		tenant_id: str,
		template_id: str,
		employee_id: str,
		plan_year: int,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		"""Create a development plan pre-seeded from a template.

		Increments the template's usage_count so adoption can be tracked.
		"""
		t = self._tenant(tenant_id)
		tmpl = self.plan_templates.get(template_id)
		if not tmpl or tmpl["tenant_id"] != t:
			raise KeyError(f"plan_template {template_id} not found")
		plan = await self.create_development_plan(
			t,
			employee_id=employee_id,
			plan_year=plan_year,
			objectives=list(tmpl["objectives"]),
			focus_areas=list(tmpl["focus_areas"]),
			target_role_id=tmpl.get("target_role"),
			reviewed_by=reviewed_by,
		)
		tmpl["usage_count"] += 1
		self._emit(t, "plan_template_applied", "plan_template", template_id, {"plan_id": plan["id"], "employee_id": employee_id})
		return plan

	async def clone_development_plan(
		self,
		tenant_id: str,
		source_plan_id: str,
		new_plan_year: int,
		employee_id: str | None = None,
	) -> dict[str, Any]:
		"""Clone an existing development plan into a new draft for a new year.

		Copies objectives, focus_areas, and target_role_id; resets progress and
		status to draft. Useful for annual rollover without data loss.
		"""
		t = self._tenant(tenant_id)
		source = self.development_plans.get(source_plan_id)
		if not source or source["tenant_id"] != t:
			raise KeyError(f"development_plan {source_plan_id} not found")
		cloned = await self.create_development_plan(
			t,
			employee_id=employee_id or source["employee_id"],
			plan_year=new_plan_year,
			objectives=list(source["objectives"]),
			focus_areas=list(source["focus_areas"]),
			target_role_id=source.get("target_role_id"),
		)
		self._emit(t, "development_plan_cloned", "development_plan", cloned["id"], {"source_plan_id": source_plan_id})
		return cloned

	# ── Learning Activities ───────────────────────────────────────────────────

	async def add_learning_activity(
		self,
		tenant_id: str,
		employee_id: str,
		title: str,
		activity_type: str,
		plan_id: str | None = None,
		provider_name: str | None = None,
		provider_id: str | None = None,
		hours_cpe: float = 0.0,
		cost: float = 0.0,
		currency: str = "KES",
		scheduled_date: str | None = None,
		completed_date: str | None = None,
	) -> dict[str, Any]:
		"""Record a learning activity (course, workshop, conference, book, etc.).

		Links optionally to a development plan and/or training provider. CPE hours
		are tracked for certification renewal calculations.
		"""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		guard_non_empty_string(title, "title")
		if activity_type not in ACTIVITY_TYPES:
			raise ValueError(f"activity_type must be one of {ACTIVITY_TYPES}")
		if plan_id:
			p = self.development_plans.get(plan_id)
			if not p or p["tenant_id"] != t:
				raise KeyError(f"development_plan {plan_id} not found")
		if provider_id:
			prov = self.training_providers.get(provider_id)
			if not prov or prov["tenant_id"] != t:
				raise KeyError(f"training_provider {provider_id} not found")
			provider_name = provider_name or prov["name"]
		record: dict[str, Any] = {
			"id": self._uid("la"),
			"tenant_id": t,
			"employee_id": employee_id,
			"plan_id": plan_id,
			"provider_id": provider_id,
			"provider_name": provider_name,
			"title": title,
			"activity_type": activity_type,
			"hours_cpe": hours_cpe,
			"cost": cost,
			"currency": currency,
			"scheduled_date": scheduled_date,
			"completed_date": completed_date,
			"status": "completed" if completed_date else "planned",
			"created_at": self._now(),
		}
		self.learning_activities[record["id"]] = record
		self._emit(t, "learning_activity_added", "learning_activity", record["id"], record)
		_log.info("learning_activity added: %s employee=%s type=%s", record["id"], employee_id, activity_type)
		return deepcopy(record)

	async def list_learning_activities(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		plan_id: str | None = None,
		provider_id: str | None = None,
		activity_type: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List learning activities with flexible filters."""
		t = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.learning_activities.values() if r["tenant_id"] == t]
		if employee_id:
			items = [r for r in items if r["employee_id"] == employee_id]
		if plan_id:
			items = [r for r in items if r.get("plan_id") == plan_id]
		if provider_id:
			items = [r for r in items if r.get("provider_id") == provider_id]
		if activity_type:
			items = [r for r in items if r["activity_type"] == activity_type]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def complete_learning_activity(
		self,
		tenant_id: str,
		activity_id: str,
		completed_date: str,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Mark a learning activity as completed and record the completion date."""
		t = self._tenant(tenant_id)
		record = self.learning_activities.get(activity_id)
		if not record or record["tenant_id"] != t:
			raise KeyError(f"learning_activity {activity_id} not found")
		record["status"] = "completed"
		record["completed_date"] = completed_date
		if notes:
			record["completion_notes"] = notes
		self._emit(t, "learning_activity_completed", "learning_activity", activity_id, {"completed_date": completed_date})
		return deepcopy(record)

	# ── Training Providers ────────────────────────────────────────────────────

	async def add_training_provider(
		self,
		tenant_id: str,
		name: str,
		website: str | None = None,
		specialisations: list[str] | None = None,
		contact_email: str | None = None,
	) -> dict[str, Any]:
		"""Register an external training provider in the catalogue.

		Linking activities to providers enables spend aggregation and quality
		comparison across vendors.
		"""
		t = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		record: dict[str, Any] = {
			"id": self._uid("tp"),
			"tenant_id": t,
			"name": name,
			"website": website,
			"specialisations": specialisations or [],
			"contact_email": contact_email,
			"total_activities": 0,
			"total_spend": 0.0,
			"status": "active",
			"created_at": self._now(),
		}
		self.training_providers[record["id"]] = record
		self._emit(t, "training_provider_added", "training_provider", record["id"], record)
		return deepcopy(record)

	async def list_training_providers(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all registered training providers."""
		t = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.training_providers.values() if r["tenant_id"] == t]

	async def get_provider_stats(self, tenant_id: str, provider_id: str) -> dict[str, Any]:
		"""Return spend, activity count, and CPE hours for a training provider."""
		t = self._tenant(tenant_id)
		prov = self.training_providers.get(provider_id)
		if not prov or prov["tenant_id"] != t:
			raise KeyError(f"training_provider {provider_id} not found")
		activities = [r for r in self.learning_activities.values() if r["tenant_id"] == t and r.get("provider_id") == provider_id]
		total_spend = sum(r.get("cost", 0.0) for r in activities)
		total_cpe = sum(r.get("hours_cpe", 0.0) for r in activities)
		completed = [r for r in activities if r["status"] == "completed"]
		return {
			"provider_id": provider_id,
			"provider_name": prov["name"],
			"total_activities": len(activities),
			"completed_activities": len(completed),
			"completion_rate_pct": round(100 * len(completed) / len(activities), 1) if activities else 0.0,
			"total_spend": total_spend,
			"currency": "KES",
			"total_cpe_hours": total_cpe,
			"generated_at": self._now(),
		}

	# ── Skill Endorsements ────────────────────────────────────────────────────

	async def endorse_skill(
		self,
		tenant_id: str,
		endorsee_employee_id: str,
		endorser_employee_id: str,
		skill_id: str,
		endorsed_level: str,
		evidence: str | None = None,
	) -> dict[str, Any]:
		"""Peer-endorse an employee's proficiency at a given level for a skill.

		Self-endorsement is rejected. Endorsements complement manager assessments
		by surfacing peer visibility of expertise.
		"""
		t = self._tenant(tenant_id)
		guard_non_empty_string(endorsee_employee_id, "endorsee_employee_id")
		guard_non_empty_string(endorser_employee_id, "endorser_employee_id")
		if endorsee_employee_id == endorser_employee_id:
			raise ValueError("self_endorsement_not_permitted")
		skill = self.skills.get(skill_id)
		if not skill or skill["tenant_id"] != t:
			raise KeyError(f"skill {skill_id} not found")
		if endorsed_level not in PROFICIENCY_LEVELS:
			raise ValueError(f"endorsed_level must be one of {PROFICIENCY_LEVELS}")
		record: dict[str, Any] = {
			"id": self._uid("end"),
			"tenant_id": t,
			"endorsee_employee_id": endorsee_employee_id,
			"endorser_employee_id": endorser_employee_id,
			"skill_id": skill_id,
			"skill_name": skill["name"],
			"endorsed_level": endorsed_level,
			"evidence": evidence,
			"created_at": self._now(),
		}
		self.skill_endorsements[record["id"]] = record
		self._emit(t, "skill_endorsed", "skill_endorsement", record["id"], record)
		return deepcopy(record)

	async def get_endorsement_summary(self, tenant_id: str, employee_id: str, skill_id: str) -> dict[str, Any]:
		"""Aggregate endorsements for one employee-skill pair.

		Returns endorsement_count, highest_endorsed_level, and a level frequency
		breakdown. Useful for calibration and promotion evidence.
		"""
		t = self._tenant(tenant_id)
		skill = self.skills.get(skill_id)
		if not skill or skill["tenant_id"] != t:
			raise KeyError(f"skill {skill_id} not found")
		endorsements = [
			r for r in self.skill_endorsements.values()
			if r["tenant_id"] == t and r["endorsee_employee_id"] == employee_id and r["skill_id"] == skill_id
		]
		if not endorsements:
			return {
				"employee_id": employee_id,
				"skill_id": skill_id,
				"skill_name": skill["name"],
				"endorsement_count": 0,
				"highest_endorsed_level": None,
				"level_breakdown": {},
			}
		level_freq: dict[str, int] = {}
		for e in endorsements:
			level_freq[e["endorsed_level"]] = level_freq.get(e["endorsed_level"], 0) + 1
		highest = max(endorsements, key=lambda e: self._level_index(e["endorsed_level"]))["endorsed_level"]
		return {
			"employee_id": employee_id,
			"skill_id": skill_id,
			"skill_name": skill["name"],
			"endorsement_count": len(endorsements),
			"highest_endorsed_level": highest,
			"level_breakdown": level_freq,
			"generated_at": self._now(),
		}

	# ── Team Skill Gap Report ─────────────────────────────────────────────────

	async def get_team_skill_gap_report(
		self,
		tenant_id: str,
		employee_ids: list[str],
	) -> dict[str, Any]:
		"""Aggregate skill gap analysis across a team of employees.

		Gathers individual gap reports in parallel, then computes per-skill gap
		prevalence and identifies the top gaps by frequency. Returns a heat_map
		keyed by skill category.
		"""
		t = self._tenant(tenant_id)
		if not employee_ids:
			raise ValueError("employee_ids must not be empty")
		# Fetch all per-employee gap reports concurrently
		results = await asyncio.gather(
			*[self.get_skill_gap_report(t, eid) for eid in employee_ids],
			return_exceptions=True,
		)
		skill_gap_counts: dict[str, int] = {}
		category_gap_counts: dict[str, int] = {}
		total_assessments = 0
		total_gaps = 0
		for res in results:
			if isinstance(res, Exception):
				continue
			total_assessments += res.get("total_skills_assessed", 0)
			total_gaps += res.get("skills_with_gaps", 0)
			for gap in res.get("gap_details", []):
				sk_name = gap.get("skill_name", gap.get("skill_id", "unknown"))
				skill_gap_counts[sk_name] = skill_gap_counts.get(sk_name, 0) + 1
				# Resolve category from skills store
				sk = self.skills.get(gap.get("skill_id", ""))
				if sk:
					cat = sk.get("category", "unknown")
					category_gap_counts[cat] = category_gap_counts.get(cat, 0) + 1
		top_gaps = sorted(skill_gap_counts.items(), key=lambda x: x[1], reverse=True)[:10]
		return {
			"team_size": len(employee_ids),
			"reports_processed": sum(1 for r in results if not isinstance(r, Exception)),
			"total_assessments": total_assessments,
			"total_gaps": total_gaps,
			"gap_prevalence_pct": round(100 * total_gaps / total_assessments, 1) if total_assessments else 0.0,
			"top_skill_gaps": [{"skill_name": name, "affected_employees": count} for name, count in top_gaps],
			"heat_map": category_gap_counts,
			"generated_at": self._now(),
		}

	# ── 360-Degree Feedback ───────────────────────────────────────────────────

	async def request_360_feedback(
		self,
		tenant_id: str,
		subject_employee_id: str,
		skill_id: str,
		rater_employee_ids: list[str],
		rater_types: list[str],
		due_date: str | None = None,
	) -> dict[str, Any]:
		"""Create a 360-degree feedback request for a specific skill.

		Validates rater_types against FEEDBACK_RATER_TYPES. Each rater gets an
		individual response slot tracked in feedback_responses.
		"""
		t = self._tenant(tenant_id)
		guard_non_empty_string(subject_employee_id, "subject_employee_id")
		if len(rater_employee_ids) != len(rater_types):
			raise ValueError("rater_employee_ids and rater_types must have equal length")
		invalid_types = set(rater_types) - FEEDBACK_RATER_TYPES
		if invalid_types:
			raise ValueError(f"invalid rater_types {invalid_types}; allowed: {FEEDBACK_RATER_TYPES}")
		skill = self.skills.get(skill_id)
		if not skill or skill["tenant_id"] != t:
			raise KeyError(f"skill {skill_id} not found")
		record: dict[str, Any] = {
			"id": self._uid("fr"),
			"tenant_id": t,
			"subject_employee_id": subject_employee_id,
			"skill_id": skill_id,
			"skill_name": skill["name"],
			"raters": [
				{"employee_id": eid, "rater_type": rt, "responded": False}
				for eid, rt in zip(rater_employee_ids, rater_types)
			],
			"due_date": due_date,
			"status": "open",
			"created_at": self._now(),
		}
		self.feedback_requests[record["id"]] = record
		self._emit(t, "feedback_request_created", "feedback_request", record["id"], record)
		return deepcopy(record)

	async def submit_feedback_response(
		self,
		tenant_id: str,
		request_id: str,
		rater_employee_id: str,
		observed_level: str,
		comments: str | None = None,
	) -> dict[str, Any]:
		"""Submit one rater's response to a 360 feedback request.

		Marks the rater slot as responded. When all raters have responded the
		request status moves to 'complete'.
		"""
		t = self._tenant(tenant_id)
		req = self.feedback_requests.get(request_id)
		if not req or req["tenant_id"] != t:
			raise KeyError(f"feedback_request {request_id} not found")
		if observed_level not in PROFICIENCY_LEVELS:
			raise ValueError(f"observed_level must be one of {PROFICIENCY_LEVELS}")
		rater_slot = next((r for r in req["raters"] if r["employee_id"] == rater_employee_id), None)
		if rater_slot is None:
			raise KeyError(f"rater {rater_employee_id} not in feedback request {request_id}")
		if rater_slot["responded"]:
			raise PermissionError("rater_already_responded")
		response: dict[str, Any] = {
			"id": self._uid("frs"),
			"tenant_id": t,
			"request_id": request_id,
			"rater_employee_id": rater_employee_id,
			"rater_type": rater_slot["rater_type"],
			"observed_level": observed_level,
			"comments": comments,
			"submitted_at": self._now(),
		}
		self.feedback_responses[response["id"]] = response
		rater_slot["responded"] = True
		if all(r["responded"] for r in req["raters"]):
			req["status"] = "complete"
		self._emit(t, "feedback_response_submitted", "feedback_response", response["id"], response)
		return deepcopy(response)

	async def aggregate_360_results(self, tenant_id: str, request_id: str) -> dict[str, Any]:
		"""Aggregate multi-rater feedback into a consensus score.

		Returns per-rater-type average level index, overall consensus, and a
		variance indicator (high/medium/low) based on level spread.
		"""
		t = self._tenant(tenant_id)
		req = self.feedback_requests.get(request_id)
		if not req or req["tenant_id"] != t:
			raise KeyError(f"feedback_request {request_id} not found")
		responses = [r for r in self.feedback_responses.values() if r["tenant_id"] == t and r["request_id"] == request_id]
		if not responses:
			return {"request_id": request_id, "response_count": 0, "consensus": None, "variance": None}
		indices = [self._level_index(r["observed_level"]) for r in responses]
		avg_idx = sum(indices) / len(indices)
		variance = max(indices) - min(indices)
		consensus_level = PROFICIENCY_LEVELS[round(avg_idx)] if 0 <= round(avg_idx) < len(PROFICIENCY_LEVELS) else PROFICIENCY_LEVELS[-1]
		by_type: dict[str, list[int]] = {}
		for r in responses:
			by_type.setdefault(r["rater_type"], []).append(self._level_index(r["observed_level"]))
		type_averages = {rt: PROFICIENCY_LEVELS[min(round(sum(v) / len(v)), len(PROFICIENCY_LEVELS) - 1)] for rt, v in by_type.items()}
		return {
			"request_id": request_id,
			"skill_id": req["skill_id"],
			"skill_name": req["skill_name"],
			"subject_employee_id": req["subject_employee_id"],
			"response_count": len(responses),
			"consensus_level": consensus_level,
			"variance_levels": variance,
			"consensus_quality": "high" if variance <= 1 else ("medium" if variance == 2 else "low"),
			"by_rater_type": type_averages,
			"generated_at": self._now(),
		}

	# ── Automated Nudges ──────────────────────────────────────────────────────

	async def generate_nudges(
		self,
		tenant_id: str,
		stale_plan_days: int = 60,
		cert_expiry_warning_days: int = 30,
		mentoring_inactive_days: int = 45,
	) -> list[dict[str, Any]]:
		"""Scan all entities and return a prioritised list of actionable nudges.

		Nudge categories: stale_plan, cert_expiring, mentoring_inactive, cert_expired.
		Each nudge carries a priority (high/medium/low) and a message.
		"""
		t = self._tenant(tenant_id)
		nudges: list[dict[str, Any]] = []
		today = date.today()

		# Stale active plans (progress < 10%, older than stale_plan_days)
		for p in self.development_plans.values():
			if p["tenant_id"] != t or p["status"] != "active":
				continue
			created = date.fromisoformat(p["created_at"][:10])
			age_days = (today - created).days
			if age_days >= stale_plan_days and p.get("progress_pct", 0.0) < 10.0:
				nudges.append({
					"type": "stale_plan",
					"priority": "high",
					"entity_type": "development_plan",
					"entity_id": p["id"],
					"employee_id": p["employee_id"],
					"message": f"Development plan {p['id']} has been active for {age_days} days with < 10% progress.",
				})

		# Certifications expiring soon or already expired
		for c in self.certifications.values():
			if c["tenant_id"] != t:
				continue
			days = self._days_to_expiry(c.get("expiry_date"))
			if days is None:
				continue
			if days < 0:
				nudges.append({
					"type": "cert_expired",
					"priority": "high",
					"entity_type": "certification",
					"entity_id": c["id"],
					"employee_id": c["employee_id"],
					"message": f"Certification '{c['certification_name']}' expired {abs(days)} days ago.",
				})
			elif days <= cert_expiry_warning_days:
				nudges.append({
					"type": "cert_expiring",
					"priority": "medium",
					"entity_type": "certification",
					"entity_id": c["id"],
					"employee_id": c["employee_id"],
					"message": f"Certification '{c['certification_name']}' expires in {days} days.",
				})

		# Mentoring programmes with no session in mentoring_inactive_days
		for mp in self.mentoring_programmes.values():
			if mp["tenant_id"] != t or mp["status"] != "active":
				continue
			sessions = [s for s in self.mentoring_sessions.values() if s["tenant_id"] == t and s["programme_id"] == mp["id"]]
			if sessions:
				latest_date = max(date.fromisoformat(s["session_date"][:10]) for s in sessions)
				inactive = (today - latest_date).days
			else:
				inactive = (today - date.fromisoformat(mp["created_at"][:10])).days
			if inactive >= mentoring_inactive_days:
				nudges.append({
					"type": "mentoring_inactive",
					"priority": "medium",
					"entity_type": "mentoring_programme",
					"entity_id": mp["id"],
					"employee_id": mp["mentee_employee_id"],
					"message": f"Mentoring programme '{mp['programme_name']}' has had no session in {inactive} days.",
				})

		priority_order = {"high": 0, "medium": 1, "low": 2}
		nudges.sort(key=lambda n: priority_order.get(n["priority"], 3))
		return nudges

	# ── Professional Development Index ───────────────────────────────────────

	async def compute_pdi(self, tenant_id: str, employee_id: str) -> dict[str, Any]:
		"""Compute a Professional Development Index (0–100) for one employee.

		Weighted composite of five sub-scores:
		  - plan_completion (25%): avg progress_pct of active/completed plans
		  - skill_gap_closure (25%): % assessments where gap_exists is False
		  - certifications (20%): ratio of active certs to total certs (0 certs = 50)
		  - career_milestones (20%): milestone completion rate across active paths
		  - mentoring (10%): 100 if any active mentoring programme, else 0

		Snapshots the result for trend analysis.
		"""
		t = self._tenant(tenant_id)

		plans, assessments, certs, paths, programmes = await asyncio.gather(
			self.list_development_plans(t, employee_id=employee_id),
			self.list_skill_assessments(t, employee_id=employee_id),
			self.list_certifications(t, employee_id=employee_id),
			self.list_career_paths(t, employee_id=employee_id, status="active"),
			self.list_mentoring_programmes(t, employee_id=employee_id, status="active"),
      return_exceptions=True,
		)

		# Plan completion sub-score
		if plans:
			plan_score = sum(p.get("progress_pct", 0.0) for p in plans) / len(plans)
		else:
			plan_score = 0.0

		# Skill gap closure sub-score
		if assessments:
			no_gap = sum(1 for a in assessments if not a["gap_exists"])
			gap_score = 100.0 * no_gap / len(assessments)
		else:
			gap_score = 50.0  # neutral when no data

		# Certification sub-score
		if certs:
			active_certs = sum(1 for c in certs if c.get("status") == "active")
			cert_score = 100.0 * active_certs / len(certs)
		else:
			cert_score = 50.0

		# Career milestone sub-score
		total_milestones = sum(len(p["milestones"]) for p in paths)
		completed_milestones = sum(p.get("milestones_completed", 0) for p in paths)
		if total_milestones:
			milestone_score = 100.0 * completed_milestones / total_milestones
		else:
			milestone_score = 50.0

		# Mentoring sub-score
		mentoring_score = 100.0 if programmes else 0.0

		w = PDI_WEIGHTS
		pdi = (
			plan_score * w["plan_completion"]
			+ gap_score * w["skill_gap_closure"]
			+ cert_score * w["certifications"]
			+ milestone_score * w["career_milestones"]
			+ mentoring_score * w["mentoring"]
		)
		pdi = round(pdi, 1)

		snapshot: dict[str, Any] = {
			"employee_id": employee_id,
			"pdi": pdi,
			"sub_scores": {
				"plan_completion": round(plan_score, 1),
				"skill_gap_closure": round(gap_score, 1),
				"certifications": round(cert_score, 1),
				"career_milestones": round(milestone_score, 1),
				"mentoring": round(mentoring_score, 1),
			},
			"computed_at": self._now(),
		}
		key = f"{t}:{employee_id}"
		self.pdi_snapshots.setdefault(key, []).append(snapshot)
		return deepcopy(snapshot)

	async def get_pdi_trend(self, tenant_id: str, employee_id: str, last_n: int = 8) -> dict[str, Any]:
		"""Return PDI snapshots over time, most recent first (up to last_n).

		Use compute_pdi periodically (e.g., quarterly) to build trend data.
		"""
		t = self._tenant(tenant_id)
		key = f"{t}:{employee_id}"
		snapshots = self.pdi_snapshots.get(key, [])
		ordered = list(reversed(snapshots))[:last_n]
		return {
			"employee_id": employee_id,
			"snapshots": ordered,
			"trend": "improving" if len(ordered) >= 2 and ordered[0]["pdi"] > ordered[-1]["pdi"] else (
				"declining" if len(ordered) >= 2 and ordered[0]["pdi"] < ordered[-1]["pdi"] else "stable"
			),
		}

	# ── Learning Budget ───────────────────────────────────────────────────────

	async def set_learning_budget(
		self,
		tenant_id: str,
		employee_id: str,
		fiscal_year: int,
		amount: float,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Set or replace a learning budget allocation for an employee and fiscal year.

		Budget utilisation is computed dynamically from completed learning_activities.
		"""
		t = self._tenant(tenant_id)
		guard_non_empty_string(employee_id, "employee_id")
		if amount < 0:
			raise ValueError("amount must be non-negative")
		key = f"{t}:{employee_id}:{fiscal_year}"
		record: dict[str, Any] = {
			"id": self._uid("lb"),
			"tenant_id": t,
			"employee_id": employee_id,
			"fiscal_year": fiscal_year,
			"amount": amount,
			"currency": currency,
			"created_at": self._now(),
		}
		self.learning_budgets[key] = record
		self._emit(t, "learning_budget_set", "learning_budget", key, record)
		return deepcopy(record)

	async def get_budget_utilisation(self, tenant_id: str, employee_id: str, fiscal_year: int) -> dict[str, Any]:
		"""Return budget allocation vs actual spend for an employee and fiscal year.

		Spend is summed from completed learning_activities irrespective of whether
		they are linked to a plan.
		"""
		t = self._tenant(tenant_id)
		key = f"{t}:{employee_id}:{fiscal_year}"
		budget = self.learning_budgets.get(key)
		allocated = budget["amount"] if budget else 0.0
		currency = budget["currency"] if budget else "KES"
		activities = [
			r for r in self.learning_activities.values()
			if r["tenant_id"] == t
			and r["employee_id"] == employee_id
			and r["status"] == "completed"
			and r.get("scheduled_date", "")[:4] == str(fiscal_year)
		]
		spent = sum(r.get("cost", 0.0) for r in activities)
		return {
			"employee_id": employee_id,
			"fiscal_year": fiscal_year,
			"allocated": allocated,
			"spent": round(spent, 2),
			"remaining": round(allocated - spent, 2),
			"utilisation_pct": round(100 * spent / allocated, 1) if allocated else 0.0,
			"currency": currency,
			"activity_count": len(activities),
			"generated_at": self._now(),
		}
