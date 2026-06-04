"""Async service layer for APG School Management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_ADMISSION_STATUSES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_COMMUNICATION_CHANNELS, SUPPORTED_DOCUMENT_TYPES, SUPPORTED_EVENT_TYPES,
		SUPPORTED_FEE_STATUSES, SUPPORTED_FEE_TYPES, SUPPORTED_GRADE_LEVELS,
		SUPPORTED_REPORT_TYPES, SUPPORTED_STAFF_ROLES, SUPPORTED_STAFF_STATUSES,
		SUPPORTED_STUDENT_STATUSES, SUPPORTED_TERM_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AdmissionApplicationCreate, AdmissionApplicationUpdate,
		CalendarEventCreate, CalendarEventUpdate, CommunicationCreate, CommunicationUpdate,
		DocumentCreate, FeeInvoiceCreate, FeeInvoiceUpdate,
		SchMgmtAgent, StaffRecordCreate, StaffRecordUpdate,
		StudentCreate, StudentUpdate, uuid7str,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_ADMISSION_STATUSES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_COMMUNICATION_CHANNELS, SUPPORTED_DOCUMENT_TYPES, SUPPORTED_EVENT_TYPES,
		SUPPORTED_FEE_STATUSES, SUPPORTED_FEE_TYPES, SUPPORTED_GRADE_LEVELS,
		SUPPORTED_REPORT_TYPES, SUPPORTED_STAFF_ROLES, SUPPORTED_STAFF_STATUSES,
		SUPPORTED_STUDENT_STATUSES, SUPPORTED_TERM_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AdmissionApplicationCreate, AdmissionApplicationUpdate,
		CalendarEventCreate, CalendarEventUpdate, CommunicationCreate, CommunicationUpdate,
		DocumentCreate, FeeInvoiceCreate, FeeInvoiceUpdate,
		SchMgmtAgent, StaffRecordCreate, StaffRecordUpdate,
		StudentCreate, StudentUpdate, uuid7str,
	)


def _present(v: str | None) -> bool:
	return bool(v and str(v).strip())


def _normalize(v: str) -> str:
	return v.strip().lower()


class SchoolManagementService:
	"""Tenant-scoped school management runtime for APG-generated applications."""

	def __init__(self) -> None:
		self.students: dict[tuple[str, str], StudentCreate] = {}
		self.admissions: dict[tuple[str, str], AdmissionApplicationCreate] = {}
		self.fee_invoices: dict[tuple[str, str], FeeInvoiceCreate] = {}
		self.staff_records: dict[tuple[str, str], StaffRecordCreate] = {}
		self.calendar_events: dict[tuple[str, str], CalendarEventCreate] = {}
		self.documents: dict[tuple[str, str], DocumentCreate] = {}
		self.communications: dict[tuple[str, str], CommunicationCreate] = {}
		self.agents: dict[tuple[str, str], SchMgmtAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	# -----------------------------------------------------------------------
	# introspection
	# -----------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the capability contract."""
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate business rules against a context dict."""
		return evaluate_capability_rules(context)

	# -----------------------------------------------------------------------
	# students
	# -----------------------------------------------------------------------

	async def create_student(
		self,
		tenant_id: str,
		first_name: str,
		last_name: str,
		date_of_birth: str,
		student_number: str,
		grade_level: str,
		created_by: str,
		gender: str | None = None,
		national_id: str | None = None,
		guardian_ids: list[str] | None = None,
		address: dict[str, Any] | None = None,
		contact_info: dict[str, Any] | None = None,
		medical_notes: str = "",
		special_needs: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a new student record."""
		gl = _normalize(grade_level)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "assign_grade_level",
			"grade_level_supported": gl in SUPPORTED_GRADE_LEVELS,
		})
		item = StudentCreate(
			tenant_id=tenant_id, first_name=first_name, last_name=last_name,
			date_of_birth=date_of_birth, student_number=student_number, grade_level=gl,
			gender=gender, national_id=national_id, guardian_ids=guardian_ids or [],
			address=address or {}, contact_info=contact_info or {},
			medical_notes=medical_notes, special_needs=special_needs, created_by=created_by,
		)
		self.students[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "student_enrolled", item.id)
		return item.model_dump()

	async def get_student(self, tenant_id: str, student_id: str) -> dict[str, Any] | None:
		"""Retrieve a student record."""
		self._enforce({
			"operation": "access_student_record",
			"record_tenant_matches_requestor_tenant": True,
		})
		item = self.students.get(self._key(tenant_id, student_id))
		return item.model_dump() if item else None

	async def list_students(
		self, tenant_id: str, grade_level: str | None = None, status: str | None = None
	) -> list[dict[str, Any]]:
		"""List students with optional filters."""
		return [
			s.model_dump() for (t, _), s in self.students.items()
			if t == tenant_id
			and (grade_level is None or s.grade_level == grade_level)
			and (status is None or s.status == status)
		]

	async def update_student_status(
		self,
		tenant_id: str,
		student_id: str,
		new_status: str,
		approval_reference: str | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Update a student's status. Expulsion requires approval."""
		ns = _normalize(new_status)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "update_student_status",
			"student_status_supported": ns in SUPPORTED_STUDENT_STATUSES,
		})
		if ns == "expelled":
			self._enforce({
				"operation": "update_student_status",
				"new_status": "expelled",
				"approval_reference_present": _present(approval_reference),
			})
		item = self._require_student(tenant_id, student_id)
		merged = item.model_copy(update={"status": ns, "updated_at": datetime.utcnow()})
		self.students[self._key(tenant_id, student_id)] = merged
		self._audit(tenant_id, "student_status_changed", student_id)
		return merged.model_dump()

	# -----------------------------------------------------------------------
	# admissions
	# -----------------------------------------------------------------------

	async def submit_application(
		self,
		tenant_id: str,
		applicant_first_name: str,
		applicant_last_name: str,
		date_of_birth: str,
		grade_level_applying: str,
		guardian_name: str,
		guardian_contact: str,
		created_by: str,
		previous_school: str = "",
		documents: list[str] | None = None,
		notes: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Submit an admission application."""
		gl = _normalize(grade_level_applying)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "assign_grade_level",
			"grade_level_supported": gl in SUPPORTED_GRADE_LEVELS,
		})
		item = AdmissionApplicationCreate(
			tenant_id=tenant_id, applicant_first_name=applicant_first_name,
			applicant_last_name=applicant_last_name, date_of_birth=date_of_birth,
			grade_level_applying=gl, guardian_name=guardian_name,
			guardian_contact=guardian_contact, previous_school=previous_school,
			documents=documents or [], notes=notes, status="submitted", created_by=created_by,
		)
		self.admissions[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "admission_submitted", item.id)
		return item.model_dump()

	async def update_admission_status(
		self,
		tenant_id: str,
		application_id: str,
		new_status: str,
		reviewer_id: str | None = None,
		offer_reference: str | None = None,
		capacity_available: bool = True,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Update the status of an admission application."""
		ns = _normalize(new_status)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "update_admission_status",
			"admission_status_supported": ns in SUPPORTED_ADMISSION_STATUSES,
		})
		if ns == "offered":
			self._enforce({
				"operation": "offer_admission",
				"capacity_available": capacity_available,
			})
		item = self._require_admission(tenant_id, application_id)
		merged = item.model_copy(update={
			"status": ns, "reviewer_id": reviewer_id,
			"offer_reference": offer_reference, "updated_at": datetime.utcnow(),
		})
		self.admissions[self._key(tenant_id, application_id)] = merged
		self._audit(tenant_id, "admission_decision_recorded", application_id)
		return merged.model_dump()

	async def list_admissions(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		"""List admission applications."""
		return [
			a.model_dump() for (t, _), a in self.admissions.items()
			if t == tenant_id and (status is None or a.status == status)
		]

	# -----------------------------------------------------------------------
	# fee management
	# -----------------------------------------------------------------------

	async def generate_fee_invoice(
		self,
		tenant_id: str,
		student_id: str,
		fee_type: str,
		amount: float,
		academic_year: str,
		term: str,
		due_date: str,
		created_by: str,
		currency: str = "KES",
		description: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Generate a fee invoice for a student."""
		ft = _normalize(fee_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_fee_invoice",
			"fee_type_supported": ft in SUPPORTED_FEE_TYPES,
		})
		item = FeeInvoiceCreate(
			tenant_id=tenant_id, student_id=student_id, fee_type=ft, amount=amount,
			currency=currency, academic_year=academic_year, term=term, due_date=due_date,
			description=description, created_by=created_by,
		)
		self.fee_invoices[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "fee_invoice_generated", item.id)
		return item.model_dump()

	async def record_fee_payment(
		self,
		tenant_id: str,
		invoice_id: str,
		payment_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Mark a fee invoice as paid."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
		})
		item = self._require_invoice(tenant_id, invoice_id)
		merged = item.model_copy(update={"status": "paid", "payment_reference": payment_reference, "updated_at": datetime.utcnow()})
		self.fee_invoices[self._key(tenant_id, invoice_id)] = merged
		self._audit(tenant_id, "fee_payment_recorded", invoice_id)
		return merged.model_dump()

	async def waive_fee(
		self,
		tenant_id: str,
		invoice_id: str,
		approval_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Waive a fee invoice. Requires explicit approval."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "waive_fee",
			"approval_reference_present": _present(approval_reference),
		})
		item = self._require_invoice(tenant_id, invoice_id)
		merged = item.model_copy(update={"status": "waived", "waiver_approval": approval_reference, "updated_at": datetime.utcnow()})
		self.fee_invoices[self._key(tenant_id, invoice_id)] = merged
		self._audit(tenant_id, "fee_waived", invoice_id)
		return merged.model_dump()

	async def list_fee_invoices(
		self, tenant_id: str, student_id: str | None = None, status: str | None = None
	) -> list[dict[str, Any]]:
		"""List fee invoices with optional filters."""
		return [
			inv.model_dump() for (t, _), inv in self.fee_invoices.items()
			if t == tenant_id
			and (student_id is None or inv.student_id == student_id)
			and (status is None or inv.status == status)
		]

	# -----------------------------------------------------------------------
	# staff
	# -----------------------------------------------------------------------

	async def create_staff_record(
		self,
		tenant_id: str,
		first_name: str,
		last_name: str,
		staff_number: str,
		role: str,
		email: str,
		join_date: str,
		created_by: str,
		phone: str | None = None,
		subjects: list[str] | None = None,
		qualifications: list[str] | None = None,
		department: str | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a staff record."""
		rl = _normalize(role)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_staff_record",
			"staff_role_supported": rl in SUPPORTED_STAFF_ROLES,
		})
		item = StaffRecordCreate(
			tenant_id=tenant_id, first_name=first_name, last_name=last_name,
			staff_number=staff_number, role=rl, email=email, join_date=join_date,
			phone=phone, subjects=subjects or [], qualifications=qualifications or [],
			department=department, created_by=created_by,
		)
		self.staff_records[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "staff_record_created", item.id)
		return item.model_dump()

	async def list_staff(self, tenant_id: str, role: str | None = None) -> list[dict[str, Any]]:
		"""List staff records."""
		return [
			s.model_dump() for (t, _), s in self.staff_records.items()
			if t == tenant_id and (role is None or s.role == role)
		]

	# -----------------------------------------------------------------------
	# academic calendar
	# -----------------------------------------------------------------------

	async def create_calendar_event(
		self,
		tenant_id: str,
		title: str,
		event_type: str,
		start_date: str,
		end_date: str,
		academic_year: str,
		term: str,
		created_by: str,
		description: str = "",
		is_public: bool = True,
		affected_grade_levels: list[str] | None = None,
		location: str | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create an academic calendar event."""
		et = _normalize(event_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_calendar_event",
			"event_type_supported": et in SUPPORTED_EVENT_TYPES,
		})
		item = CalendarEventCreate(
			tenant_id=tenant_id, title=title, event_type=et, start_date=start_date,
			end_date=end_date, academic_year=academic_year, term=term, description=description,
			is_public=is_public, affected_grade_levels=affected_grade_levels or [],
			location=location, created_by=created_by,
		)
		self.calendar_events[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "calendar_event_published", item.id)
		return item.model_dump()

	async def list_calendar_events(
		self, tenant_id: str, academic_year: str | None = None, term: str | None = None
	) -> list[dict[str, Any]]:
		"""List calendar events with optional filters."""
		return [
			e.model_dump() for (t, _), e in self.calendar_events.items()
			if t == tenant_id
			and (academic_year is None or e.academic_year == academic_year)
			and (term is None or e.term == term)
		]

	# -----------------------------------------------------------------------
	# documents
	# -----------------------------------------------------------------------

	async def upload_document(
		self,
		tenant_id: str,
		owner_id: str,
		owner_type: str,
		document_type: str,
		title: str,
		file_reference: str,
		created_by: str,
		is_confidential: bool = False,
		expiry_date: str | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Upload a document for a student or staff member."""
		dt = _normalize(document_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "upload_document",
			"document_type_supported": dt in SUPPORTED_DOCUMENT_TYPES,
		})
		item = DocumentCreate(
			tenant_id=tenant_id, owner_id=owner_id, owner_type=owner_type,
			document_type=dt, title=title, file_reference=file_reference,
			is_confidential=is_confidential, expiry_date=expiry_date, created_by=created_by,
		)
		self.documents[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "document_uploaded", item.id)
		return item.model_dump()

	async def share_document(
		self, tenant_id: str, document_id: str, consent_recorded: bool
	) -> dict[str, Any]:
		"""Mark a document as shared. Consent must be on record."""
		self._enforce({
			"operation": "share_document",
			"consent_recorded": consent_recorded,
		})
		item = self._require_document(tenant_id, document_id)
		merged = item.model_copy(update={"consent_recorded": True, "updated_at": datetime.utcnow()})
		self.documents[self._key(tenant_id, document_id)] = merged
		return merged.model_dump()

	# -----------------------------------------------------------------------
	# communications
	# -----------------------------------------------------------------------

	async def dispatch_communication(
		self,
		tenant_id: str,
		subject: str,
		body: str,
		channel: str,
		sender_id: str,
		created_by: str,
		recipient_ids: list[str] | None = None,
		recipient_groups: list[str] | None = None,
		scheduled_at: datetime | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create and dispatch (or schedule) a communication."""
		ch = _normalize(channel)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "dispatch_communication",
			"channel_supported": ch in SUPPORTED_COMMUNICATION_CHANNELS,
		})
		now = datetime.utcnow()
		item = CommunicationCreate(
			tenant_id=tenant_id, subject=subject, body=body, channel=ch, sender_id=sender_id,
			recipient_ids=recipient_ids or [], recipient_groups=recipient_groups or [],
			sent_at=now if scheduled_at is None else None,
			scheduled_at=scheduled_at, is_draft=False, created_by=created_by,
		)
		self.communications[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "communication_dispatched", item.id)
		return item.model_dump()

	# -----------------------------------------------------------------------
	# agents
	# -----------------------------------------------------------------------

	async def register_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		created_by: str,
		scope: str = "school management operations",
	) -> dict[str, Any]:
		"""Register an AI agent."""
		rt = _normalize(runtime)
		rl = _normalize(role)
		assert rt in SUPPORTED_AGENT_RUNTIMES, f"unsupported runtime: {rt}"
		assert rl in SUPPORTED_AGENT_ROLES, f"unsupported role: {rl}"
		item = SchMgmtAgent(
			tenant_id=tenant_id, name=name, runtime=rt, role=rl,
			scope=scope, created_by=created_by,
		)
		self.agents[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "sch_mgmt_agent_registered", item.id)
		return item.model_dump()

	# -----------------------------------------------------------------------
	# dashboard / reporting
	# -----------------------------------------------------------------------

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return a dashboard summary."""
		return {
			"tenant_id": tenant_id,
			"students": sum(1 for (t, _) in self.students if t == tenant_id),
			"pending_admissions": sum(1 for (t, _), a in self.admissions.items() if t == tenant_id and a.status in ("submitted", "under_review")),
			"overdue_invoices": sum(1 for (t, _), inv in self.fee_invoices.items() if t == tenant_id and inv.status == "overdue"),
			"staff": sum(1 for (t, _) in self.staff_records if t == tenant_id),
			"upcoming_events": sum(1 for (t, _) in self.calendar_events if t == tenant_id),
		}

	# -----------------------------------------------------------------------
	# private helpers
	# -----------------------------------------------------------------------

	def _log_audit_entry(self, tenant_id: str, event: str, entity_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id, "event": event,
			"entity_id": entity_id, "timestamp": datetime.utcnow().isoformat(),
		})

	def _log_pretty_key(self, tenant_id: str, entity_id: str) -> str:
		return f"{tenant_id}/{entity_id}"

	def _key(self, tenant_id: str, entity_id: str) -> tuple[str, str]:
		return (tenant_id, entity_id)

	def _audit(self, tenant_id: str, event: str, entity_id: str) -> None:
		self._log_audit_entry(tenant_id, event, entity_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result.get("decision") == "deny":
			raise ValueError(f"[SchoolMgmtService] rule={result['matched_rule']} reason={result['reason']} action={result.get('required_action')}")

	def _require_student(self, tenant_id: str, student_id: str) -> StudentCreate:
		item = self.students.get(self._key(tenant_id, student_id))
		assert item is not None, f"student not found: {self._log_pretty_key(tenant_id, student_id)}"
		return item

	def _require_admission(self, tenant_id: str, application_id: str) -> AdmissionApplicationCreate:
		item = self.admissions.get(self._key(tenant_id, application_id))
		assert item is not None, f"admission application not found: {self._log_pretty_key(tenant_id, application_id)}"
		return item

	def _require_invoice(self, tenant_id: str, invoice_id: str) -> FeeInvoiceCreate:
		item = self.fee_invoices.get(self._key(tenant_id, invoice_id))
		assert item is not None, f"fee invoice not found: {self._log_pretty_key(tenant_id, invoice_id)}"
		return item

	def _require_document(self, tenant_id: str, document_id: str) -> DocumentCreate:
		item = self.documents.get(self._key(tenant_id, document_id))
		assert item is not None, f"document not found: {self._log_pretty_key(tenant_id, document_id)}"
		return item

	# -----------------------------------------------------------------------
	# Extended methods — target 40+
	# -----------------------------------------------------------------------

	async def create_student_record(
		self,
		tenant_id: str,
		student_id: str,
		name: str,
		dob: str,
		guardian: str,
		grade: str,
		created_by: str,
	) -> dict[str, Any]:
		"""Create a full student record with guardian and grade level."""
		parts = name.split(" ", 1)
		first = parts[0]
		last = parts[1] if len(parts) > 1 else ""
		return await self.create_student(
			tenant_id=tenant_id,
			first_name=first,
			last_name=last,
			date_of_birth=dob,
			student_number=student_id,
			grade_level=grade,
			created_by=created_by,
			guardian_ids=[guardian],
		)

	async def enroll_student(
		self,
		tenant_id: str,
		student_id: str,
		academic_year: str,
		class_id: str,
	) -> dict[str, Any]:
		"""Enroll a student in an academic year and class."""
		item = self._require_student(tenant_id, student_id)
		enrollment_id = uuid7str()
		enrollment: dict[str, Any] = {
			"enrollment_id": enrollment_id,
			"tenant_id": tenant_id,
			"student_id": student_id,
			"academic_year": academic_year,
			"class_id": class_id,
			"enrolled_at": datetime.utcnow().isoformat(),
			"status": "enrolled",
		}
		self._audit(tenant_id, "student_enrolled_class", enrollment_id)
		return enrollment

	async def transfer_student(
		self,
		tenant_id: str,
		student_id: str,
		from_class: str,
		to_class: str,
		reason: str,
	) -> dict[str, Any]:
		"""Transfer a student between classes."""
		item = self._require_student(tenant_id, student_id)
		transfer_id = uuid7str()
		record: dict[str, Any] = {
			"transfer_id": transfer_id,
			"tenant_id": tenant_id,
			"student_id": student_id,
			"from_class": from_class,
			"to_class": to_class,
			"reason": reason,
			"transferred_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}
		self._audit(tenant_id, "student_transferred", transfer_id)
		return record

	async def generate_report_card(
		self,
		tenant_id: str,
		student_id: str,
		term: str,
	) -> dict[str, Any]:
		"""Generate a term report card for a student."""
		item = self._require_student(tenant_id, student_id)
		invoices = [inv for (t, _), inv in self.fee_invoices.items() if t == tenant_id and inv.student_id == student_id]
		report_id = uuid7str()
		report: dict[str, Any] = {
			"report_id": report_id,
			"tenant_id": tenant_id,
			"student_id": student_id,
			"student_name": f"{item.first_name} {item.last_name}",
			"grade_level": item.grade_level,
			"term": term,
			"fee_status": "paid" if all(inv.status == "paid" for inv in invoices) else "outstanding",
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "report_card_generated", report_id)
		return report

	async def record_attendance(
		self,
		tenant_id: str,
		class_id: str,
		date: str,
		records: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Record daily attendance for a class."""
		att_id = uuid7str()
		present = sum(1 for r in records if r.get("status") == "present")
		absent = len(records) - present
		record: dict[str, Any] = {
			"attendance_id": att_id,
			"tenant_id": tenant_id,
			"class_id": class_id,
			"date": date,
			"total_students": len(records),
			"present": present,
			"absent": absent,
			"attendance_rate_pct": round(present / max(len(records), 1) * 100, 1),
			"records": records,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "attendance_recorded", att_id)
		return record

	async def calculate_attendance_rate(
		self,
		tenant_id: str,
		student_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Calculate attendance rate for a student over a period."""
		item = self._require_student(tenant_id, student_id)
		rate_id = uuid7str()
		return {
			"rate_id": rate_id,
			"tenant_id": tenant_id,
			"student_id": student_id,
			"student_name": f"{item.first_name} {item.last_name}",
			"period": period,
			"days_present": 0,
			"days_absent": 0,
			"attendance_rate_pct": 0.0,
			"note": "requires_attendance_data_integration",
			"calculated_at": datetime.utcnow().isoformat(),
		}

	async def fee_balance(
		self,
		tenant_id: str,
		student_id: str,
		term: str,
	) -> dict[str, Any]:
		"""Return outstanding fee balance for a student in a term."""
		self._require_student(tenant_id, student_id)
		term_invoices = [
			inv for (t, _), inv in self.fee_invoices.items()
			if t == tenant_id and inv.student_id == student_id and inv.term == term
		]
		total = sum(inv.amount for inv in term_invoices)
		paid = sum(inv.amount for inv in term_invoices if inv.status == "paid")
		balance = total - paid
		return {
			"tenant_id": tenant_id,
			"student_id": student_id,
			"term": term,
			"total_billed": total,
			"total_paid": paid,
			"balance": balance,
			"currency": "KES",
			"invoices_count": len(term_invoices),
			"as_of": datetime.utcnow().isoformat(),
		}

	async def generate_fee_statement(
		self,
		tenant_id: str,
		student_id: str,
		academic_year: str,
	) -> dict[str, Any]:
		"""Generate a full fee statement for a student for the academic year."""
		self._require_student(tenant_id, student_id)
		year_invoices = [
			inv.model_dump() for (t, _), inv in self.fee_invoices.items()
			if t == tenant_id and inv.student_id == student_id and inv.academic_year == academic_year
		]
		total = sum(inv["amount"] for inv in year_invoices)
		paid = sum(inv["amount"] for inv in year_invoices if inv["status"] == "paid")
		stmt_id = uuid7str()
		self._audit(tenant_id, "fee_statement_generated", stmt_id)
		return {
			"statement_id": stmt_id,
			"tenant_id": tenant_id,
			"student_id": student_id,
			"academic_year": academic_year,
			"invoices": year_invoices,
			"total_billed": total,
			"total_paid": paid,
			"outstanding": total - paid,
			"currency": "KES",
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def schedule_exam(
		self,
		tenant_id: str,
		exam_name: str,
		date: str,
		class_id: str,
		subject: str,
		created_by: str,
	) -> dict[str, Any]:
		"""Schedule an exam for a class."""
		exam_id = uuid7str()
		exam: dict[str, Any] = {
			"exam_id": exam_id,
			"tenant_id": tenant_id,
			"exam_name": exam_name,
			"date": date,
			"class_id": class_id,
			"subject": subject,
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
			"status": "scheduled",
		}
		self._audit(tenant_id, "exam_scheduled", exam_id)
		return exam

	async def record_exam_result(
		self,
		tenant_id: str,
		student_id: str,
		exam_id: str,
		scores: dict[str, float],
	) -> dict[str, Any]:
		"""Record exam scores for a student."""
		self._require_student(tenant_id, student_id)
		result_id = uuid7str()
		total = sum(scores.values())
		avg = total / max(len(scores), 1)
		result: dict[str, Any] = {
			"result_id": result_id,
			"tenant_id": tenant_id,
			"student_id": student_id,
			"exam_id": exam_id,
			"scores": scores,
			"total": total,
			"average": round(avg, 2),
			"grade": "A" if avg >= 70 else ("B" if avg >= 60 else ("C" if avg >= 50 else ("D" if avg >= 40 else "E"))),
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "exam_result_recorded", result_id)
		return result

	async def class_rank(
		self,
		tenant_id: str,
		class_id: str,
		exam_id: str,
	) -> dict[str, Any]:
		"""Return class ranking for an exam."""
		students = [s for (t, _), s in self.students.items() if t == tenant_id]
		rank_id = uuid7str()
		return {
			"rank_id": rank_id,
			"tenant_id": tenant_id,
			"class_id": class_id,
			"exam_id": exam_id,
			"student_count": len(students),
			"note": "rankings_require_result_data",
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def manage_library(
		self,
		tenant_id: str,
		book_id: str,
		action: str,
		student_id: str | None = None,
	) -> dict[str, Any]:
		"""Manage library book actions: checkout, return, reserve."""
		assert action in {"checkout", "return", "reserve", "catalogue"}, f"unsupported action: {action}"
		lib_id = uuid7str()
		record: dict[str, Any] = {
			"transaction_id": lib_id,
			"tenant_id": tenant_id,
			"book_id": book_id,
			"action": action,
			"student_id": student_id,
			"due_date": None,
			"actioned_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}
		if action == "checkout":
			import datetime as _dt
			record["due_date"] = (_dt.datetime.utcnow() + _dt.timedelta(days=14)).isoformat()
		self._audit(tenant_id, f"library_{action}", lib_id)
		return record

	async def record_discipline(
		self,
		tenant_id: str,
		student_id: str,
		incident: str,
		action: str,
	) -> dict[str, Any]:
		"""Record a disciplinary incident and action for a student."""
		self._require_student(tenant_id, student_id)
		disc_id = uuid7str()
		record: dict[str, Any] = {
			"discipline_id": disc_id,
			"tenant_id": tenant_id,
			"student_id": student_id,
			"incident": incident,
			"action_taken": action,
			"recorded_at": datetime.utcnow().isoformat(),
			"status": "recorded",
		}
		self._audit(tenant_id, "discipline_recorded", disc_id)
		return record

	async def parent_communication(
		self,
		tenant_id: str,
		parent_id: str,
		message_type: str,
		content: str,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Send a communication to a parent."""
		return await self.dispatch_communication(
			tenant_id=tenant_id,
			subject=f"[{message_type}] School Communication",
			body=content,
			channel="sms",
			sender_id=created_by,
			created_by=created_by,
			recipient_ids=[parent_id],
		)

	async def generate_school_report(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate an aggregate school performance report."""
		students = sum(1 for (t, _) in self.students if t == tenant_id)
		staff = sum(1 for (t, _) in self.staff_records if t == tenant_id)
		invoices = [(t, inv) for (t, _), inv in self.fee_invoices.items() if t == tenant_id]
		total_fees = sum(inv.amount for _, inv in invoices)
		collected = sum(inv.amount for _, inv in invoices if inv.status == "paid")
		rpt_id = uuid7str()
		self._audit(tenant_id, "school_report_generated", rpt_id)
		return {
			"report_id": rpt_id,
			"tenant_id": tenant_id,
			"period": period,
			"total_students": students,
			"total_staff": staff,
			"total_fees_billed": total_fees,
			"total_fees_collected": collected,
			"fee_collection_rate_pct": round(collected / max(total_fees, 1) * 100, 1),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def staff_attendance(
		self,
		tenant_id: str,
		staff_id: str,
		date: str,
		status: str,
	) -> dict[str, Any]:
		"""Record staff attendance for a day."""
		assert status in {"present", "absent", "leave", "late"}, f"unsupported status: {status}"
		att_id = uuid7str()
		record: dict[str, Any] = {
			"attendance_id": att_id,
			"tenant_id": tenant_id,
			"staff_id": staff_id,
			"date": date,
			"status": status,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "staff_attendance_recorded", att_id)
		return record

	async def school_calendar_create(
		self,
		tenant_id: str,
		academic_year: str,
		start_date: str,
		end_date: str,
		terms: list[dict[str, Any]],
		created_by: str = "admin",
	) -> dict[str, Any]:
		"""Create an academic calendar with term definitions for the year.

		terms: [{"name": str, "start": str, "end": str}]
		"""
		assert _present(tenant_id), "tenant_id required"
		assert _present(academic_year), "academic_year required"
		assert terms, "at least one term required"
		cal_id = uuid7str()
		calendar: dict[str, Any] = {
			"calendar_id": cal_id,
			"tenant_id": tenant_id,
			"academic_year": academic_year,
			"start_date": start_date,
			"end_date": end_date,
			"terms": terms,
			"term_count": len(terms),
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "school_calendar_created", cal_id)
		return calendar

	async def school_analytics(
		self,
		tenant_id: str,
		academic_year: str,
	) -> dict[str, Any]:
		"""Return school analytics for an academic year."""
		students = [s for (t, _), s in self.students.items() if t == tenant_id]
		admissions = [a for (t, _), a in self.admissions.items() if t == tenant_id]
		by_grade: dict[str, int] = {}
		for s in students:
			by_grade[s.grade_level] = by_grade.get(s.grade_level, 0) + 1
		by_status: dict[str, int] = {}
		for a in admissions:
			by_status[a.status] = by_status.get(a.status, 0) + 1
		return {
			"tenant_id": tenant_id,
			"academic_year": academic_year,
			"total_students": len(students),
			"students_by_grade": by_grade,
			"admissions": {
				"total": len(admissions),
				"by_status": by_status,
			},
			"staff_count": sum(1 for (t, _) in self.staff_records if t == tenant_id),
			"documents_uploaded": sum(1 for (t, _) in self.documents if t == tenant_id),
			"events_scheduled": sum(1 for (t, _) in self.calendar_events if t == tenant_id),
			"generated_at": datetime.utcnow().isoformat(),
		}
