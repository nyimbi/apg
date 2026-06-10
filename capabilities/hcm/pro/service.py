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


class PROService:
	"""Professional Development — dev plans, skills, mentoring, certifications, career paths."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.development_plans: dict[str, dict[str, Any]] = {}
		self.skills: dict[str, dict[str, Any]] = {}
		self.skill_assessments: dict[str, dict[str, Any]] = {}
		self.mentoring_programmes: dict[str, dict[str, Any]] = {}
		self.mentoring_sessions: dict[str, dict[str, Any]] = {}
		self.certifications: dict[str, dict[str, Any]] = {}
		self.career_paths: dict[str, dict[str, Any]] = {}
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
			"skills": len(self.skills),
			"skill_assessments": len(self.skill_assessments),
			"mentoring_programmes": len(self.mentoring_programmes),
			"certifications": len(self.certifications),
			"career_paths": len(self.career_paths),
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
			"generated_at": self._now(),
		}
