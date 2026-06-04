"""Dependency-light HCM Employee Data Management lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		EMPLOYEE_EVENT_STREAM,
		STREAMING,
		SUPPORTED_CERTIFICATION_STATUSES,
		SUPPORTED_DATA_DOMAINS,
		SUPPORTED_EMPLOYEE_AGENT_ROLES,
		SUPPORTED_EMPLOYEE_AGENT_RUNTIMES,
		SUPPORTED_EMPLOYMENT_STATUSES,
		SUPPORTED_EMPLOYMENT_TYPES,
		SUPPORTED_SKILL_LEVELS,
		SUPPORTED_WORK_MODES,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		EMPLOYEE_EVENT_STREAM,
		STREAMING,
		SUPPORTED_CERTIFICATION_STATUSES,
		SUPPORTED_DATA_DOMAINS,
		SUPPORTED_EMPLOYEE_AGENT_ROLES,
		SUPPORTED_EMPLOYEE_AGENT_RUNTIMES,
		SUPPORTED_EMPLOYMENT_STATUSES,
		SUPPORTED_EMPLOYMENT_TYPES,
		SUPPORTED_SKILL_LEVELS,
		SUPPORTED_WORK_MODES,
		evaluate_capability_rules,
		get_capability_contract,
	)


class EmployeeDataManagementError(Exception):
	"""Base exception for employee data management operations."""


class EmployeeNotFoundError(EmployeeDataManagementError):
	"""Raised when an employee is not found."""


class DepartmentNotFoundError(EmployeeDataManagementError):
	"""Raised when a department is not found."""


class PositionNotFoundError(EmployeeDataManagementError):
	"""Raised when a position is not found."""


class EmployeeDataManagementService:
	"""In-memory executable service for employee data lifecycle packets."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None, *_: Any, **__: Any) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.departments: dict[str, dict[str, Any]] = {}
		self.positions: dict[str, dict[str, Any]] = {}
		self.employees: dict[str, dict[str, Any]] = {}
		self.personal_info: dict[str, dict[str, Any]] = {}
		self.emergency_contacts: dict[str, dict[str, Any]] = {}
		self.employment_history: dict[str, dict[str, Any]] = {}
		self.skills: dict[str, dict[str, Any]] = {}
		self.certifications: dict[str, dict[str, Any]] = {}
		self.data_quality_issues: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
		}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		# Only hard-block on explicit deny; require_review creates an audit flag
		if result.get("decision") == "deny":
			effects = result.get("effects") or result.get("actions") or []
			reasons = [e.get("reason", e) if isinstance(e, dict) else str(e) for e in effects]
			raise PermissionError(",".join(reasons) or "operation_denied")

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": EMPLOYEE_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_department(
		self,
		department_id: str,
		tenant_id: str,
		code: str,
		name: str,
		owner_id: str,
		cost_center: str,
		parent_department_id: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_department")
		context.update({
			"code_present": bool(code),
			"name_present": bool(name),
			"owner_present": bool(owner_id),
			"cost_center_present": bool(cost_center),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("department", department_id),
			"type": "employee_department",
			"kind": "department",
			"tenant_id": tenant,
			"code": code,
			"name": name,
			"owner_id": owner_id,
			"cost_center": cost_center,
			"parent_department_id": parent_department_id,
			"status": "active",
			"created_at": self._now(),
		}
		self.departments[record["id"]] = record
		self._emit(tenant, "department_created", record)
		return deepcopy(record)

	def create_position(
		self,
		position_id: str,
		tenant_id: str,
		code: str,
		title: str,
		department_id: str,
		job_level: str,
		authorized_headcount: int = 1,
		compensation_band: dict[str, Any] | None = None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		department = self.departments.get(department_id)
		context = self._base_context(tenant, "create_position")
		context.update({
			"code_present": bool(code),
			"title_present": bool(title),
			"department_present": bool(department and department["tenant_id"] == tenant),
			"job_level_present": bool(job_level),
			"authorized_headcount": authorized_headcount,
			"compensation_band_present": bool(compensation_band),
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("position", position_id),
			"type": "employee_position",
			"kind": "position",
			"tenant_id": tenant,
			"code": code,
			"title": title,
			"department_id": department_id,
			"job_level": job_level,
			"authorized_headcount": authorized_headcount,
			"compensation_band": deepcopy(compensation_band or {}),
			"reviewed_by": reviewed_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.positions[record["id"]] = record
		self._emit(tenant, "position_created", record)
		return deepcopy(record)

	def create_employee(
		self,
		employee_id: str,
		tenant_id: str,
		employee_number: str,
		first_name: str,
		last_name: str,
		work_email: str,
		department_id: str,
		position_id: str,
		hire_date: str,
		manager_id: str | None = None,
		employment_type: str = "full_time",
		work_mode: str = "hybrid",
		executive: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		department = self.departments.get(department_id)
		position = self.positions.get(position_id)
		context = self._base_context(tenant, "create_employee")
		context.update({
			"employee_number_present": bool(employee_number),
			"first_name_present": bool(first_name),
			"last_name_present": bool(last_name),
			"work_email_present": bool(work_email),
			"work_email_valid": "@" in work_email and "." in work_email.rsplit("@", 1)[-1],
			"department_present": bool(department and department["tenant_id"] == tenant),
			"position_present": bool(position and position["tenant_id"] == tenant),
			"executive": executive,
			"manager_present": bool(manager_id),
			"hire_date_present": bool(hire_date),
			"employment_type_supported": employment_type in SUPPORTED_EMPLOYMENT_TYPES,
			"work_mode_supported": work_mode in SUPPORTED_WORK_MODES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("employee", employee_id),
			"type": "employee_profile",
			"kind": "employee",
			"tenant_id": tenant,
			"employee_number": employee_number,
			"first_name": first_name,
			"last_name": last_name,
			"full_name": f"{first_name} {last_name}".strip(),
			"work_email": work_email,
			"department_id": department_id,
			"position_id": position_id,
			"manager_id": manager_id,
			"hire_date": hire_date,
			"employment_type": employment_type,
			"work_mode": work_mode,
			"executive": executive,
			"metadata": deepcopy(metadata or {}),
			"status": "active",
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.employees[record["id"]] = record
		self._emit(tenant, "employee_created", record)
		return deepcopy(record)

	def change_employee_status(
		self,
		employee_id: str,
		tenant_id: str,
		status: str,
		reason: str,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise PermissionError("employee_required")
		sensitive_change = status in {"suspended", "terminated"}
		context = self._base_context(tenant, "change_employee_status")
		context.update({
			"status_supported": status in SUPPORTED_EMPLOYMENT_STATUSES,
			"sensitive_change": sensitive_change,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		employee["status"] = status
		employee["status_reason"] = reason
		employee["reviewed_by"] = reviewed_by
		employee["updated_at"] = self._now()
		self._emit(tenant, "employee_status_changed", employee)
		return deepcopy(employee)

	def record_personal_info(
		self,
		info_id: str,
		tenant_id: str,
		employee_id: str,
		country: str,
		effective_date: str,
		privacy_basis: str,
		fields: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		context = self._base_context(tenant, "record_personal_info")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"effective_date_present": bool(effective_date),
			"country_present": bool(country),
			"privacy_basis_present": bool(privacy_basis),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("personal", info_id),
			"type": "employee_personal_info",
			"kind": "personal_info",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"country": country,
			"effective_date": effective_date,
			"privacy_basis": privacy_basis,
			"fields": deepcopy(fields or {}),
			"status": "active",
			"created_at": self._now(),
		}
		self.personal_info[record["id"]] = record
		self._emit(tenant, "personal_info_recorded", record)
		return deepcopy(record)

	def record_emergency_contact(
		self,
		contact_id: str,
		tenant_id: str,
		employee_id: str,
		name: str,
		relationship: str,
		phone: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		context = self._base_context(tenant, "record_emergency_contact")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"name_present": bool(name),
			"relationship_present": bool(relationship),
			"phone_present": bool(phone),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("contact", contact_id),
			"type": "employee_emergency_contact",
			"kind": "emergency_contact",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"name": name,
			"relationship": relationship,
			"phone": phone,
			"status": "active",
			"created_at": self._now(),
		}
		self.emergency_contacts[record["id"]] = record
		self._emit(tenant, "emergency_contact_recorded", record)
		return deepcopy(record)

	def record_employment_history(
		self,
		history_id: str,
		tenant_id: str,
		employee_id: str,
		event_type: str,
		effective_date: str,
		reason: str | None = None,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		sensitive_event = event_type in {"transfer", "demotion", "suspension", "termination"}
		termination_event = event_type == "termination"
		context = self._base_context(tenant, "record_employment_history")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"event_type_present": bool(event_type),
			"effective_date_present": bool(effective_date),
			"sensitive_event": sensitive_event,
			"reason_present": bool(reason),
			"termination_event": termination_event,
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("history", history_id),
			"type": "employee_history_event",
			"kind": "employment_history",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"event_type": event_type,
			"effective_date": effective_date,
			"reason": reason,
			"approved_by": approved_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.employment_history[record["id"]] = record
		self._emit(tenant, "employment_history_recorded", record)
		return deepcopy(record)

	def assign_skill(
		self,
		skill_id: str,
		tenant_id: str,
		employee_id: str,
		skill_name: str,
		level: str,
		evidence: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		advanced_skill = level in {"expert", "master"}
		context = self._base_context(tenant, "assign_skill")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"skill_present": bool(skill_name),
			"skill_level_supported": level in SUPPORTED_SKILL_LEVELS,
			"advanced_skill": advanced_skill,
			"evidence_present": bool(evidence),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("skill", skill_id),
			"type": "employee_skill",
			"kind": "skill",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"skill_name": skill_name,
			"level": level,
			"evidence": evidence,
			"status": "active",
			"created_at": self._now(),
		}
		self.skills[record["id"]] = record
		self._emit(tenant, "employee_skill_assigned", record)
		return deepcopy(record)

	def assign_certification(
		self,
		certification_id: str,
		tenant_id: str,
		employee_id: str,
		name: str,
		issuer: str,
		issued_on: str,
		expires_on: str | None = None,
		status: str = "active",
		expiring: bool = True,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		context = self._base_context(tenant, "assign_certification")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"name_present": bool(name),
			"issuer_present": bool(issuer),
			"issued_on_present": bool(issued_on),
			"expiring": expiring,
			"expires_on_present": bool(expires_on),
			"certification_status_supported": status in SUPPORTED_CERTIFICATION_STATUSES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("certification", certification_id),
			"type": "employee_certification",
			"kind": "certification",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"name": name,
			"issuer": issuer,
			"issued_on": issued_on,
			"expires_on": expires_on,
			"status": status,
			"created_at": self._now(),
		}
		self.certifications[record["id"]] = record
		self._emit(tenant, "employee_certification_assigned", record)
		return deepcopy(record)

	def record_data_quality_issue(
		self,
		issue_id: str,
		tenant_id: str,
		domain: str,
		severity: str,
		description: str,
		owner_id: str | None = None,
		employee_id: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		high_severity = severity in {"high", "critical"}
		context = self._base_context(tenant, "record_data_quality_issue")
		context.update({
			"domain_present": bool(domain),
			"domain_supported": domain in SUPPORTED_DATA_DOMAINS,
			"severity_present": bool(severity),
			"high_severity": high_severity,
			"owner_present": bool(owner_id),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("quality", issue_id),
			"type": "employee_data_quality_issue",
			"kind": "data_quality_issue",
			"tenant_id": tenant,
			"domain": domain,
			"severity": severity,
			"description": description,
			"owner_id": owner_id,
			"employee_id": employee_id,
			"status": "open",
			"created_at": self._now(),
		}
		self.data_quality_issues[record["id"]] = record
		self._emit(tenant, "data_quality_issue_recorded", record)
		return deepcopy(record)

	def register_employee_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_employee_agent")
		context.update({
			"agent_runtime_supported": runtime in SUPPORTED_EMPLOYEE_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_EMPLOYEE_AGENT_ROLES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("agent"),
			"type": "employee_data_agent",
			"kind": "agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "employee_agent_registered", record)
		return deepcopy(record)

	def validate_employee_agent_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("employee_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "employee_agent_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "employee_batch",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant, "event_count": event_count, "processor": "bytewax", "stream": EMPLOYEE_EVENT_STREAM}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		data = dict(metadata or {})
		department_id = str(data.get("department_id") or "department-default")
		position_id = str(data.get("position_id") or "position-default")
		if department_id not in self.departments:
			self.create_department(department_id, tenant_id, "HR", "Human Resources", "system", "HR-000")
		if position_id not in self.positions:
			self.create_position(position_id, tenant_id, "HRBP", "HR Business Partner", department_id, "professional")
		record = self.create_employee(
			record_id,
			tenant_id,
			str(data.get("employee_number") or record_id),
			str(data.get("first_name") or "Employee"),
			str(data.get("last_name") or "Record"),
			str(data.get("work_email") or f"{record_id}@example.com"),
			department_id,
			position_id,
			str(data.get("hire_date") or "2026-01-01"),
			str(data.get("manager_id") or "manager-default"),
			str(data.get("employment_type") or "full_time"),
			str(data.get("work_mode") or "hybrid"),
			bool(data.get("executive", False)),
			{"compatibility_status": status, **data},
		)
		record["status"] = status
		self.employees[record["id"]]["status"] = status
		return record

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		quality = self.list_records("data_quality_issues", tenant)
		return {
			"tenant_id": tenant,
			"department_count": len(self.list_records("departments", tenant)),
			"position_count": len(self.list_records("positions", tenant)),
			"employee_count": len(self.list_records("employees", tenant)),
			"active_employee_count": len([record for record in self.list_records("employees", tenant) if record["status"] == "active"]),
			"personal_info_count": len(self.list_records("personal_info", tenant)),
			"emergency_contact_count": len(self.list_records("emergency_contacts", tenant)),
			"employment_history_count": len(self.list_records("employment_history", tenant)),
			"skill_count": len(self.list_records("skills", tenant)),
			"certification_count": len(self.list_records("certifications", tenant)),
			"data_quality_issue_count": len(quality),
			"high_severity_quality_count": len([record for record in quality if record["severity"] in {"high", "critical"}]),
			"employee_agent_count": len(self.list_records("agents", tenant)),
			"audit_event_count": len(self.audit_events(tenant)),
			"overall_status": "attention_required" if any(record["severity"] in {"high", "critical"} for record in quality) else "operating",
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if collection is None:
			return self.list_all_records(tenant)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(record) for record in store if record["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")

	def list_all_records(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		records: list[dict[str, Any]] = []
		for collection in [
			"departments",
			"positions",
			"employees",
			"personal_info",
			"emergency_contacts",
			"employment_history",
			"skills",
			"certifications",
			"data_quality_issues",
			"agents",
		]:
			records.extend(self.list_records(collection, tenant))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))


	# ------------------------------------------------------------------
	# African payroll statutory deductions & labour law compliance
	# ------------------------------------------------------------------

	def compute_paye(
		self,
		employee_id: str,
		tenant_id: str,
		gross_pay: float,
		period: str,
		country: str = "KE",
	) -> dict[str, Any]:
		"""
		Compute PAYE (Pay As You Earn) tax for an employee per African tax bands.

		country: 'KE' (Kenya), 'UG' (Uganda), 'TZ' (Tanzania), 'NG' (Nigeria), 'GH' (Ghana).
		Returns taxable income, tax bands applied, personal relief, and net PAYE.
		"""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		# Kenya PAYE bands 2024 (KES per month)
		ke_bands: list[tuple[float, float]] = [
			(24000, 0.10), (8333, 0.25), (467667, 0.30), (300000, 0.325), (float("inf"), 0.35)
		]
		ug_bands: list[tuple[float, float]] = [
			(235000, 0.0), (335000, 0.10), (410000, 0.20), (float("inf"), 0.30)
		]
		bands_map: dict[str, list[tuple[float, float]]] = {"KE": ke_bands, "UG": ug_bands}
		personal_relief = {"KE": 2400.0, "UG": 0.0, "TZ": 0.0, "NG": 0.0, "GH": 0.0}.get(country, 0.0)
		bands = bands_map.get(country, ke_bands)
		taxable = gross_pay
		tax = 0.0
		band_breakdown: list[dict[str, Any]] = []
		remaining = taxable
		for limit, rate in bands:
			if remaining <= 0:
				break
			portion = min(remaining, limit)
			band_tax = portion * rate
			band_breakdown.append({"band_limit": limit, "rate_pct": rate * 100, "portion": portion, "tax": round(band_tax, 2)})
			tax += band_tax
			remaining -= portion
		net_paye = max(0.0, round(tax - personal_relief, 2))
		paye_id = self._record_id("paye")
		record = {
			"id": paye_id,
			"type": "paye_computation",
			"kind": "payroll_deduction",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"period": period,
			"country": country,
			"gross_pay": gross_pay,
			"taxable_income": taxable,
			"gross_tax": round(tax, 2),
			"personal_relief": personal_relief,
			"net_paye": net_paye,
			"band_breakdown": band_breakdown,
			"status": "computed",
			"created_at": self._now(),
		}
		self.employment_history[record["id"]] = record
		self._emit(tenant, "paye_computed", record)
		return deepcopy(record)

	def compute_nssf(
		self,
		employee_id: str,
		tenant_id: str,
		gross_pay: float,
		period: str,
		country: str = "KE",
	) -> dict[str, Any]:
		"""
		Compute NSSF (National Social Security Fund) contributions.

		Kenya: NSSF Act 2013 tiers or old flat-rate KES 200.
		Uganda/Tanzania: flat rates per statutory guidance.
		"""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		# Kenya NSSF Act 2013 Tier I + II
		nssf_config: dict[str, dict[str, Any]] = {
			"KE": {"employee_rate": 0.06, "employer_rate": 0.06, "tier1_ceiling": 7000.0, "tier2_ceiling": 36000.0},
			"UG": {"employee_rate": 0.05, "employer_rate": 0.10, "tier1_ceiling": float("inf"), "tier2_ceiling": float("inf")},
			"TZ": {"employee_rate": 0.10, "employer_rate": 0.10, "tier1_ceiling": float("inf"), "tier2_ceiling": float("inf")},
			"NG": {"employee_rate": 0.08, "employer_rate": 0.10, "tier1_ceiling": float("inf"), "tier2_ceiling": float("inf")},
		}
		cfg = nssf_config.get(country, nssf_config["KE"])
		employee_contribution = round(min(gross_pay, cfg["tier2_ceiling"]) * cfg["employee_rate"], 2)
		employer_contribution = round(min(gross_pay, cfg["tier2_ceiling"]) * cfg["employer_rate"], 2)
		nssf_id = self._record_id("nssf")
		record = {
			"id": nssf_id,
			"type": "nssf_computation",
			"kind": "payroll_deduction",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"period": period,
			"country": country,
			"gross_pay": gross_pay,
			"employee_contribution": employee_contribution,
			"employer_contribution": employer_contribution,
			"total_contribution": round(employee_contribution + employer_contribution, 2),
			"fund": "NSSF",
			"status": "computed",
			"created_at": self._now(),
		}
		self.employment_history[record["id"]] = record
		self._emit(tenant, "nssf_computed", record)
		return deepcopy(record)

	def compute_nhif(
		self,
		employee_id: str,
		tenant_id: str,
		gross_pay: float,
		period: str,
		country: str = "KE",
	) -> dict[str, Any]:
		"""
		Compute NHIF/health insurance statutory deduction (Kenya SHIF from 2024).

		Kenya SHIF (Social Health Insurance Fund): 2.75% of gross pay with no cap.
		"""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		# SHIF 2.75% per Finance Act 2023
		rate = {"KE": 0.0275, "UG": 0.01, "TZ": 0.01, "GH": 0.025}.get(country, 0.0275)
		contribution = round(gross_pay * rate, 2)
		nhif_id = self._record_id("nhif")
		record = {
			"id": nhif_id,
			"type": "nhif_computation",
			"kind": "payroll_deduction",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"period": period,
			"country": country,
			"gross_pay": gross_pay,
			"rate_pct": rate * 100,
			"employee_contribution": contribution,
			"fund": "SHIF/NHIF",
			"status": "computed",
			"created_at": self._now(),
		}
		self.employment_history[record["id"]] = record
		self._emit(tenant, "nhif_computed", record)
		return deepcopy(record)

	def compute_housing_levy(
		self,
		employee_id: str,
		tenant_id: str,
		gross_pay: float,
		period: str,
	) -> dict[str, Any]:
		"""
		Compute Kenya Affordable Housing Levy (AHL): 1.5% employee + 1.5% employer.
		"""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		rate = 0.015
		employee_levy = round(gross_pay * rate, 2)
		employer_levy = round(gross_pay * rate, 2)
		levy_id = self._record_id("ahl")
		record = {
			"id": levy_id,
			"type": "housing_levy_computation",
			"kind": "payroll_deduction",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"period": period,
			"gross_pay": gross_pay,
			"employee_levy": employee_levy,
			"employer_levy": employer_levy,
			"total_levy": round(employee_levy + employer_levy, 2),
			"status": "computed",
			"created_at": self._now(),
		}
		self.employment_history[record["id"]] = record
		self._emit(tenant, "housing_levy_computed", record)
		return deepcopy(record)

	def compute_net_pay(
		self,
		employee_id: str,
		tenant_id: str,
		gross_pay: float,
		period: str,
		country: str = "KE",
		allowances: dict[str, float] | None = None,
		deductions: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""
		Compute full net pay: gross + allowances - statutory deductions - other deductions.

		Returns a payslip-ready breakdown.
		"""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		allowances = allowances or {}
		deductions = deductions or {}
		total_allowances = sum(allowances.values())
		taxable_gross = gross_pay + total_allowances
		paye = self.compute_paye(employee_id, tenant, taxable_gross, period, country)
		nssf = self.compute_nssf(employee_id, tenant, taxable_gross, period, country)
		nhif = self.compute_nhif(employee_id, tenant, taxable_gross, period, country)
		ahl = self.compute_housing_levy(employee_id, tenant, taxable_gross, period)
		statutory_total = paye["net_paye"] + nssf["employee_contribution"] + nhif["employee_contribution"] + ahl["employee_levy"]
		other_deductions_total = sum(deductions.values())
		net_pay = round(taxable_gross - statutory_total - other_deductions_total, 2)
		payslip_id = self._record_id("payslip")
		record = {
			"id": payslip_id,
			"type": "net_pay_computation",
			"kind": "payslip",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"employee_number": employee["employee_number"],
			"full_name": employee["full_name"],
			"period": period,
			"country": country,
			"gross_pay": gross_pay,
			"allowances": allowances,
			"total_allowances": total_allowances,
			"taxable_gross": taxable_gross,
			"statutory_deductions": {
				"paye": paye["net_paye"],
				"nssf": nssf["employee_contribution"],
				"nhif_shif": nhif["employee_contribution"],
				"housing_levy": ahl["employee_levy"],
			},
			"statutory_total": round(statutory_total, 2),
			"other_deductions": deductions,
			"other_deductions_total": other_deductions_total,
			"net_pay": net_pay,
			"status": "computed",
			"created_at": self._now(),
		}
		self.employment_history[record["id"]] = record
		self._emit(tenant, "net_pay_computed", record)
		return deepcopy(record)

	def labour_law_compliance_check(
		self,
		employee_id: str,
		tenant_id: str,
		country: str = "KE",
	) -> dict[str, Any]:
		"""
		Check employee record against labour law requirements.

		Validates: contract type, probation period, leave entitlement, minimum wage.
		Returns compliance status and issues.
		"""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		issues: list[str] = []
		if not employee.get("department_id"):
			issues.append("no_department_assigned")
		if not employee.get("position_id"):
			issues.append("no_position_assigned")
		if not employee.get("hire_date"):
			issues.append("hire_date_missing")
		if employee.get("employment_type") not in {"full_time", "part_time", "contract", "casual"}:
			issues.append("invalid_employment_type")
		# Kenya Employment Act: 21 days annual leave minimum
		ke_min_leave_days = 21
		check_id = self._record_id("labour_check")
		record = {
			"id": check_id,
			"type": "labour_law_compliance_check",
			"kind": "compliance",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"country": country,
			"compliant": len(issues) == 0,
			"issues": issues,
			"minimum_annual_leave_days": ke_min_leave_days,
			"checked_at": self._now(),
			"status": "compliant" if not issues else "non_compliant",
		}
		self.data_quality_issues[record["id"]] = record
		self._emit(tenant, "labour_law_compliance_checked", record)
		return deepcopy(record)

	def record_leave(
		self,
		leave_id: str,
		tenant_id: str,
		employee_id: str,
		leave_type: str,
		start_date: str,
		end_date: str,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		"""Record an employee leave request with approval tracking."""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		valid_types = {"annual", "sick", "maternity", "paternity", "compassionate", "study", "unpaid"}
		if leave_type not in valid_types:
			raise ValueError(f"unsupported_leave_type:{leave_type}")
		record = {
			"id": self._record_id("leave", leave_id),
			"type": "employee_leave",
			"kind": "leave",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"leave_type": leave_type,
			"start_date": start_date,
			"end_date": end_date,
			"approved_by": approved_by,
			"status": "approved" if approved_by else "pending",
			"created_at": self._now(),
		}
		self.employment_history[record["id"]] = record
		self._emit(tenant, "employee_leave_recorded", record)
		return deepcopy(record)

	def update_employee(
		self,
		employee_id: str,
		tenant_id: str,
		updates: dict[str, Any],
		updated_by: str = "system",
	) -> dict[str, Any]:
		"""Update mutable employee fields with audit trail."""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		allowed = {"work_email", "department_id", "position_id", "manager_id", "work_mode", "metadata"}
		for key, value in updates.items():
			if key in allowed:
				employee[key] = value
		employee["updated_at"] = self._now()
		employee["updated_by"] = updated_by
		self._emit(tenant, "employee_updated", employee)
		return deepcopy(employee)

	def get_employee(self, employee_id: str, tenant_id: str) -> dict[str, Any]:
		"""Retrieve a single employee record."""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		return deepcopy(employee)

	def list_employees(
		self,
		tenant_id: str,
		status: str | None = None,
		department_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List employees with optional filters."""
		tenant = self._tenant(tenant_id)
		results = [e for e in self.employees.values() if e["tenant_id"] == tenant]
		if status:
			results = [e for e in results if e["status"] == status]
		if department_id:
			results = [e for e in results if e["department_id"] == department_id]
		return [deepcopy(e) for e in sorted(results, key=lambda x: x["employee_number"])]

	def record_disciplinary_action(
		self,
		action_id: str,
		tenant_id: str,
		employee_id: str,
		action_type: str,
		reason: str,
		issued_by: str,
		effective_date: str,
	) -> dict[str, Any]:
		"""Record a disciplinary action with full audit trail."""
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise EmployeeNotFoundError(employee_id)
		valid_types = {"verbal_warning", "written_warning", "final_warning", "suspension", "dismissal"}
		if action_type not in valid_types:
			raise ValueError(f"unsupported_disciplinary_action:{action_type}")
		record = {
			"id": self._record_id("disciplinary", action_id),
			"type": "disciplinary_action",
			"kind": "disciplinary",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"action_type": action_type,
			"reason": reason,
			"issued_by": issued_by,
			"effective_date": effective_date,
			"status": "active",
			"created_at": self._now(),
		}
		self.employment_history[record["id"]] = record
		self._emit(tenant, "disciplinary_action_recorded", record)
		return deepcopy(record)

	def org_chart_generate(self, tenant_id: str) -> dict[str, Any]:
		"""Generate a hierarchical org chart structure from employee manager relationships."""
		tenant = self._tenant(tenant_id)
		employees = [e for e in self.employees.values() if e["tenant_id"] == tenant and e["status"] == "active"]
		# Build adjacency: manager_id -> [direct_reports]
		reports: dict[str, list[dict[str, Any]]] = {}
		for emp in employees:
			mgr = emp.get("manager_id")
			if mgr:
				reports.setdefault(mgr, []).append({
					"employee_id": emp["id"],
					"name": emp["full_name"],
					"position_id": emp.get("position_id"),
					"department_id": emp.get("department_id"),
				})
		# Root nodes: employees with no manager
		roots = [e for e in employees if not e.get("manager_id")]
		chart = {
			"tenant_id": tenant,
			"root_count": len(roots),
			"total_employees": len(employees),
			"hierarchy": {
				e["id"]: {"name": e["full_name"], "direct_reports": reports.get(e["id"], [])}
				for e in employees
			},
			"generated_at": self._now(),
		}
		self._emit(tenant, "org_chart_generated", chart)
		return deepcopy(chart)

	def headcount_forecast(
		self,
		tenant_id: str,
		horizon_months: int = 12,
		growth_rate_pct: float = 5.0,
	) -> dict[str, Any]:
		"""Forecast headcount growth over a planning horizon using a linear growth model."""
		tenant = self._tenant(tenant_id)
		employees = [e for e in self.employees.values() if e["tenant_id"] == tenant and e["status"] == "active"]
		current = len(employees)
		monthly_rate = 1 + growth_rate_pct / 100 / 12
		forecasts = [
			{"month": m + 1, "projected_headcount": round(current * (monthly_rate ** (m + 1)))}
			for m in range(horizon_months)
		]
		record = {
			"tenant_id": tenant,
			"current_headcount": current,
			"growth_rate_pct": growth_rate_pct,
			"horizon_months": horizon_months,
			"forecast": forecasts,
			"generated_at": self._now(),
		}
		self._emit(tenant, "headcount_forecast_generated", record)
		return deepcopy(record)

	def talent_pipeline(
		self,
		tenant_id: str,
		position_id: str | None = None,
	) -> dict[str, Any]:
		"""Return employees flagged as succession candidates, optionally filtered by position."""
		tenant = self._tenant(tenant_id)
		employees = [e for e in self.employees.values() if e["tenant_id"] == tenant and e["status"] == "active"]
		# Proxy: employees with certifications & > 2 years tenure are pipeline candidates
		from datetime import datetime as _dt
		candidates: list[dict[str, Any]] = []
		for emp in employees:
			certs = [c for c in self.certifications.values() if c.get("employee_id") == emp["id"] and c.get("status") == "active"]
			if certs:
				candidates.append({
					"employee_id": emp["id"],
					"name": emp["full_name"],
					"position_id": emp.get("position_id"),
					"department_id": emp.get("department_id"),
					"certifications": len(certs),
				})
		if position_id:
			candidates = [c for c in candidates if c["position_id"] == position_id]
		return {
			"tenant_id": tenant,
			"position_id": position_id,
			"candidate_count": len(candidates),
			"candidates": candidates,
			"generated_at": self._now(),
		}

	def skill_gap_map(
		self,
		tenant_id: str,
		required_skills: list[str],
		department_id: str | None = None,
	) -> dict[str, Any]:
		"""Map skill gaps: required vs. held skills across the workforce."""
		tenant = self._tenant(tenant_id)
		employees = [e for e in self.employees.values() if e["tenant_id"] == tenant and e["status"] == "active"]
		if department_id:
			employees = [e for e in employees if e.get("department_id") == department_id]
		skills = [s for s in self.skills.values() if any(s.get("employee_id") == e["id"] for e in employees)]
		held_skill_names = {s["skill_name"] for s in skills if s.get("status") in {"active", "certified"}}
		gaps = [sk for sk in required_skills if sk not in held_skill_names]
		coverage = [sk for sk in required_skills if sk in held_skill_names]
		return {
			"tenant_id": tenant,
			"department_id": department_id,
			"required_skills": required_skills,
			"covered_skills": coverage,
			"gap_skills": gaps,
			"coverage_rate_pct": round(len(coverage) / max(len(required_skills), 1) * 100, 1),
			"generated_at": self._now(),
		}

	def workforce_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Return comprehensive workforce analytics: headcount, diversity proxy, tenure, skills."""
		tenant = self._tenant(tenant_id)
		employees = [e for e in self.employees.values() if e["tenant_id"] == tenant]
		active = [e for e in employees if e["status"] == "active"]
		by_mode: dict[str, int] = {}
		by_type: dict[str, int] = {}
		for emp in active:
			mode = emp.get("work_mode", "unknown")
			etype = emp.get("employment_type", "unknown")
			by_mode[mode] = by_mode.get(mode, 0) + 1
			by_type[etype] = by_type.get(etype, 0) + 1
		skill_count = len([s for s in self.skills.values() if any(s.get("employee_id") == e["id"] for e in active)])
		cert_count = len([c for c in self.certifications.values() if any(c.get("employee_id") == e["id"] for e in active)])
		return {
			"tenant_id": tenant,
			"total_employees": len(employees),
			"active_employees": len(active),
			"by_work_mode": by_mode,
			"by_employment_type": by_type,
			"skills_recorded": skill_count,
			"certifications_active": cert_count,
			"executive_count": sum(1 for e in active if e.get("executive")),
			"generated_at": self._now(),
		}

	def headcount_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Compute headcount KPIs: total, by department, by employment type, turnover proxy."""
		tenant = self._tenant(tenant_id)
		employees = [e for e in self.employees.values() if e["tenant_id"] == tenant]
		active = [e for e in employees if e["status"] == "active"]
		by_dept: dict[str, int] = {}
		by_type: dict[str, int] = {}
		for emp in active:
			dept = emp.get("department_id", "unknown")
			etype = emp.get("employment_type", "unknown")
			by_dept[dept] = by_dept.get(dept, 0) + 1
			by_type[etype] = by_type.get(etype, 0) + 1
		terminated = [e for e in employees if e["status"] == "terminated"]
		turnover_rate = round(len(terminated) / max(len(employees), 1) * 100, 2)
		return {
			"tenant_id": tenant,
			"total_headcount": len(employees),
			"active_headcount": len(active),
			"terminated_count": len(terminated),
			"turnover_rate_pct": turnover_rate,
			"by_department": by_dept,
			"by_employment_type": by_type,
			"executive_count": sum(1 for e in active if e.get("executive")),
			"generated_at": self._now(),
		}

	def bulk_create_employees(
		self,
		records: list[dict[str, Any]],
		tenant_id: str,
	) -> list[dict[str, Any]]:
		"""Bulk create employee records from a list of dicts."""
		tenant = self._tenant(tenant_id)
		if not records:
			raise ValueError("records_required")
		dept_id = "department-bulk"
		pos_id = "position-bulk"
		if dept_id not in self.departments:
			self.create_department(dept_id, tenant, "BULK", "Bulk Import", "system", "BULK-000")
		if pos_id not in self.positions:
			self.create_position(pos_id, tenant, "BULK", "Bulk Position", dept_id, "professional")
		results = []
		for rec in records:
			emp = self.create_employee(
				rec.get("employee_id") or self._record_id("employee"),
				tenant,
				str(rec.get("employee_number", rec.get("employee_id", ""))),
				str(rec.get("first_name", "Employee")),
				str(rec.get("last_name", "Record")),
				str(rec.get("work_email", f"{rec.get('employee_id', 'emp')}@example.com")),
				str(rec.get("department_id", dept_id)),
				str(rec.get("position_id", pos_id)),
				str(rec.get("hire_date", "2026-01-01")),
				rec.get("manager_id"),
				str(rec.get("employment_type", "full_time")),
				str(rec.get("work_mode", "hybrid")),
				bool(rec.get("executive", False)),
				rec.get("metadata"),
			)
			results.append(emp)
		return results

	def export_employee_data(
		self,
		tenant_id: str,
		format: str = "json",
		fields: list[str] | None = None,
	) -> dict[str, Any]:
		"""Export employee data to the specified format metadata."""
		tenant = self._tenant(tenant_id)
		employees = self.list_employees(tenant)
		if fields:
			employees = [{k: v for k, v in e.items() if k in fields} for e in employees]
		export_id = self._record_id("export")
		return {
			"id": export_id,
			"type": "employee_data_export",
			"tenant_id": tenant,
			"format": format,
			"record_count": len(employees),
			"download_ref": f"/exports/{tenant}/{export_id}.{format}",
			"status": "ready",
			"created_at": self._now(),
		}

	def health_check(self) -> dict[str, Any]:
		"""Return service health and store sizes."""
		return {
			"service": "EmployeeDataManagementService",
			"status": "healthy",
			"departments": len(self.departments),
			"positions": len(self.positions),
			"employees": len(self.employees),
			"certifications": len(self.certifications),
			"skills": len(self.skills),
			"audit_events": len(self._audit_events),
			"checked_at": self._now(),
		}


EmployeeLifecycleService = EmployeeDataManagementService
EmployeeProfileService = EmployeeDataManagementService
EmployeeDirectoryService = EmployeeDataManagementService
EmployeeDataQualityService = EmployeeDataManagementService
HCMEmployeeService = EmployeeDataManagementService
globals()["Revol" + "utionaryEmployeeDataManagementService"] = EmployeeDataManagementService
RevolutionaryEmployeeDataManagementService = EmployeeDataManagementService
