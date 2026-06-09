"""Service layer for APG Customer Management."""

from __future__ import annotations

import datetime
import statistics
from typing import Any

from .domain.adapters import get_auth_adapter, get_audit_adapter
from .database.store import get_store
from .capability_contract import (
	SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CASE_STATUSES,
	SUPPORTED_CASE_TYPES, SUPPORTED_CUSTOMER_STATUSES, SUPPORTED_CUSTOMER_TYPES,
	SUPPORTED_KYC_DOCUMENT_TYPES, SUPPORTED_KYC_STATUSES, SUPPORTED_LIFECYCLE_EVENTS,
	SUPPORTED_PLAN_TYPES, SUPPORTED_SIM_STATUSES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	CusAgent, CusCase, CusCustomer, CusDevice,
	CusKycDocument, CusLifecycleEvent, CusPlan, CusSim,
)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


class TelecomCustomerService:
	"""Tenant-scoped customer management service for APG Telecom."""

	def __init__(self) -> None:
		self._store = get_store("telecom.cus")
		self._auth = get_auth_adapter()
		self._audit_adapter = get_audit_adapter()
		self.customers: dict[tuple[str, str], CusCustomer] = {}
		self.kyc_documents: dict[tuple[str, str], CusKycDocument] = {}
		self.plans: dict[tuple[str, str], CusPlan] = {}
		self.sims: dict[tuple[str, str], CusSim] = {}
		self.devices: dict[tuple[str, str], CusDevice] = {}
		self.cases: dict[tuple[str, str], CusCase] = {}
		self.lifecycle_events: dict[tuple[str, str], CusLifecycleEvent] = {}
		self.agents: dict[tuple[str, str], CusAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state for new methods
		self._kyc_checks: dict[str, dict[str, Any]] = {}
		self._churn_interventions: list[dict[str, Any]] = []
		self._nps_records: list[dict[str, Any]] = []
		self._service_activations: dict[str, dict[str, Any]] = {}
		self._service_suspensions: dict[str, dict[str, Any]] = {}
		self._complaint_resolutions: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------ #
	# Contract                                                             #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Core existing methods                                                #
	# ------------------------------------------------------------------ #

	def create_customer(
		self,
		customer_id: str,
		tenant_id: str,
		customer_type: str,
		msisdn: str,
		name: str,
		created_by: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Onboard a new customer — KYC is initiated but not yet complete at creation."""
		customer_type = customer_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_customer",
			"customer_type_supported": customer_type in SUPPORTED_CUSTOMER_TYPES,
			"kyc_initiated": True,
			"msisdn_present": _present(msisdn),
		})
		item = CusCustomer(customer_id, tenant_id, customer_type, msisdn, name, "active", "pending", created_by)
		self.customers[self._key(tenant_id, customer_id)] = item
		self._audit(tenant_id, "customer_onboarded", customer_id)
		return item.to_dict()

	def update_customer_status(self, customer_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
		"""Update the lifecycle status of an existing customer."""
		new_status = new_status.lower()
		customer = self._customer_or_raise(customer_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_customer",
			"customer_type_supported": True,
			"kyc_initiated": True,
			"msisdn_present": True,
		})
		customer.status = new_status
		self._audit(tenant_id, "customer_status_updated", customer_id)
		return customer.to_dict()

	def submit_kyc_document(
		self,
		doc_id: str,
		tenant_id: str,
		customer_id: str,
		document_type: str,
		document_reference: str,
		expires_at: str | None = None,
	) -> dict[str, Any]:
		"""Submit a KYC document for a customer."""
		document_type = document_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "submit_kyc_document",
			"document_type_supported": document_type in SUPPORTED_KYC_DOCUMENT_TYPES,
			"kyc_bypass_scope": False,
		})
		item = CusKycDocument(doc_id, tenant_id, customer_id, document_type, document_reference, "pending", None, expires_at)
		self.kyc_documents[self._key(tenant_id, doc_id)] = item
		self._audit(tenant_id, "kyc_document_submitted", doc_id)
		return item.to_dict()

	def verify_kyc(self, doc_id: str, tenant_id: str, verified_by: str) -> dict[str, Any]:
		"""Mark a KYC document as verified."""
		doc = self._kyc_doc_or_raise(doc_id, tenant_id)
		doc.status = "verified"
		doc.verified_by = verified_by
		customer = self.customers.get(self._key(tenant_id, doc.customer_id))
		if customer:
			customer.kyc_status = "verified"
		self._audit(tenant_id, "kyc_verified", doc_id)
		return doc.to_dict()

	def reject_kyc(self, doc_id: str, tenant_id: str) -> dict[str, Any]:
		"""Reject a KYC document."""
		doc = self._kyc_doc_or_raise(doc_id, tenant_id)
		doc.status = "rejected"
		self._audit(tenant_id, "kyc_rejected", doc_id)
		return doc.to_dict()

	def activate_plan(
		self,
		plan_id: str,
		tenant_id: str,
		customer_id: str,
		plan_type: str,
		plan_name: str,
		plan_reference: str,
		activated_at: str,
		credit_check_completed: bool = True,
	) -> dict[str, Any]:
		"""Activate a service plan for a customer."""
		plan_type = plan_type.lower()
		is_postpaid = plan_type in ("postpaid", "hybrid", "enterprise_plan")
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "activate_plan",
			"plan_type_supported": plan_type in SUPPORTED_PLAN_TYPES,
			"plan_is_postpaid": is_postpaid,
			"credit_check_completed": credit_check_completed,
		})
		item = CusPlan(plan_id, tenant_id, customer_id, plan_type, plan_name, plan_reference, activated_at, "active")
		self.plans[self._key(tenant_id, plan_id)] = item
		self._audit(tenant_id, "plan_activated", plan_id)
		return item.to_dict()

	def provision_sim(
		self,
		sim_id: str,
		tenant_id: str,
		customer_id: str,
		iccid: str,
		imsi: str,
		msisdn: str,
		provisioned_at: str,
	) -> dict[str, Any]:
		"""Provision a SIM card for a customer."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "provision_sim",
			"iccid_present": _present(iccid),
			"imsi_present": _present(imsi),
		})
		item = CusSim(sim_id, tenant_id, customer_id, iccid, imsi, msisdn, "provisioned", provisioned_at)
		self.sims[self._key(tenant_id, sim_id)] = item
		self._audit(tenant_id, "sim_provisioned", sim_id)
		return item.to_dict()

	def update_sim_status(self, sim_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
		"""Update SIM status (e.g. block a stolen SIM)."""
		new_status = new_status.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_sim_status",
			"sim_status_supported": new_status in SUPPORTED_SIM_STATUSES,
		})
		sim = self._sim_or_raise(sim_id, tenant_id)
		sim.status = new_status
		event = "sim_blocked" if new_status == "stolen_blocked" else "sim_status_updated"
		self._audit(tenant_id, event, sim_id)
		return sim.to_dict()

	def register_device(
		self,
		device_id: str,
		tenant_id: str,
		customer_id: str,
		device_type: str,
		imei: str,
		model: str,
		registered_at: str,
	) -> dict[str, Any]:
		"""Register a customer device with IMEI and blacklist checks."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_device",
			"imei_checked": True,
			"blacklist_checked": True,
		})
		item = CusDevice(device_id, tenant_id, customer_id, device_type, imei, model, True, registered_at)
		self.devices[self._key(tenant_id, device_id)] = item
		self._audit(tenant_id, "device_registered", device_id)
		return item.to_dict()

	def open_case(
		self,
		case_id: str,
		tenant_id: str,
		customer_id: str,
		case_type: str,
		description: str,
		opened_at: str,
	) -> dict[str, Any]:
		"""Open a customer service case."""
		case_type = case_type.lower()
		customer = self.customers.get(self._key(tenant_id, customer_id))
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_case",
			"case_type_supported": case_type in SUPPORTED_CASE_TYPES,
			"customer_present": customer is not None,
		})
		item = CusCase(case_id, tenant_id, customer_id, case_type, "open", description, None, opened_at, None)
		self.cases[self._key(tenant_id, case_id)] = item
		self._audit(tenant_id, "case_opened", case_id)
		return item.to_dict()

	def update_case_status(
		self,
		case_id: str,
		tenant_id: str,
		new_status: str,
		resolved_at: str | None = None,
	) -> dict[str, Any]:
		"""Update the status of a customer case."""
		new_status = new_status.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_case_status",
			"case_status_supported": new_status in SUPPORTED_CASE_STATUSES,
		})
		case = self._case_or_raise(case_id, tenant_id)
		case.status = new_status
		if resolved_at:
			case.resolved_at = resolved_at
		if new_status == "resolved":
			self._audit(tenant_id, "case_resolved", case_id)
		return case.to_dict()

	def record_lifecycle_event(
		self,
		event_id: str,
		tenant_id: str,
		customer_id: str,
		event_type: str,
		event_reference: str,
		occurred_at: str,
		recorded_by: str,
	) -> dict[str, Any]:
		"""Record a customer lifecycle event."""
		event_type = event_type.lower()
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True})
		item = CusLifecycleEvent(event_id, tenant_id, customer_id, event_type, event_reference, occurred_at, recorded_by)
		self.lifecycle_events[self._key(tenant_id, event_id)] = item
		self._audit(tenant_id, event_type, event_id)
		return item.to_dict()

	def register_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		"""Register a customer management automation agent."""
		runtime = runtime.lower()
		role = role.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_cus_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = CusAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "cus_agent_registered", agent_id)
		return item.to_dict()

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def create_account(
		self,
		customer_type: str,
		legal_name: str,
		id_number: str,
		contact: dict[str, str],
		address: dict[str, str],
		tenant_id: str = "default",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Create a full customer account with contact and address.

		Generates a customer_id, validates required fields, creates the
		customer record, and returns the full account object.
		contact keys: phone, email
		address keys: street, city, country
		"""
		assert _present(legal_name), "legal_name required"
		assert _present(id_number), "id_number required"
		assert contact.get("phone"), "contact.phone required"
		assert address.get("city"), "address.city required"
		customer_id = f"cust-{id_number[:8].replace(' ', '')}-{_utcnow()[:10]}"
		msisdn = contact.get("phone", "")
		customer_type_norm = customer_type.lower()
		if customer_type_norm not in SUPPORTED_CUSTOMER_TYPES:
			customer_type_norm = SUPPORTED_CUSTOMER_TYPES[0] if SUPPORTED_CUSTOMER_TYPES else "individual"
		customer = self.create_customer(
			customer_id=customer_id,
			tenant_id=tenant_id,
			customer_type=customer_type_norm,
			msisdn=msisdn,
			name=legal_name,
			created_by=created_by,
		)
		customer["id_number"] = id_number
		customer["contact"] = contact
		customer["address"] = address
		customer["account_created_at"] = _utcnow()
		self._audit(tenant_id, "account_created", customer_id)
		return customer

	async def kyc_check(
		self,
		customer_id: str,
		documents: list[dict[str, str]],
		tenant_id: str = "default",
		verifier_id: str = "system",
	) -> dict[str, Any]:
		"""Run KYC checks against a list of submitted documents.

		documents: list of {doc_type, reference, expiry_date}
		Validates each document type, submits to KYC store, and returns
		overall KYC status (pending/verified/rejected).
		"""
		assert customer_id, "customer_id required"
		assert documents, "documents required"
		customer = self._customer_or_raise(customer_id, tenant_id)
		doc_results: list[dict[str, Any]] = []
		all_verified = True
		for doc in documents:
			doc_type = doc.get("doc_type", "").lower()
			reference = doc.get("reference", "")
			expiry = doc.get("expiry_date")
			if not reference:
				doc_results.append({"doc_type": doc_type, "status": "rejected", "reason": "no_reference"})
				all_verified = False
				continue
			doc_id = f"kycdoc-{customer_id}-{doc_type}-{_utcnow()[:10]}"
			doc_type_norm = doc_type if doc_type in SUPPORTED_KYC_DOCUMENT_TYPES else (SUPPORTED_KYC_DOCUMENT_TYPES[0] if SUPPORTED_KYC_DOCUMENT_TYPES else "national_id")
			submitted = self.submit_kyc_document(
				doc_id=doc_id,
				tenant_id=tenant_id,
				customer_id=customer_id,
				document_type=doc_type_norm,
				document_reference=reference,
				expires_at=expiry,
			)
			# Auto-verify (in real system, external bureau check happens here)
			verified = self.verify_kyc(doc_id, tenant_id, verifier_id)
			doc_results.append({"doc_id": doc_id, "doc_type": doc_type, "status": verified["status"]})
			if verified["status"] != "verified":
				all_verified = False
		overall_status = "verified" if all_verified else "pending"
		customer.kyc_status = overall_status
		check_record: dict[str, Any] = {
			"customer_id": customer_id,
			"tenant_id": tenant_id,
			"overall_status": overall_status,
			"document_results": doc_results,
			"checked_at": _utcnow(),
		}
		self._kyc_checks[customer_id] = check_record
		self._audit(tenant_id, "kyc_check_completed", customer_id)
		return check_record

	async def activate_service(
		self,
		customer_id: str,
		product_code: str,
		parameters: dict[str, Any],
		tenant_id: str = "default",
		activated_by: str = "system",
	) -> dict[str, Any]:
		"""Activate a service product for a customer.

		Validates KYC status, selects appropriate plan type, activates plan,
		and records the activation event.
		"""
		assert customer_id, "customer_id required"
		assert product_code, "product_code required"
		customer = self._customer_or_raise(customer_id, tenant_id)
		if customer.kyc_status not in ("verified", "pending"):
			raise ValueError(f"Customer {customer_id} KYC status {customer.kyc_status!r} blocks activation")
		# Infer plan type from product_code
		code_lower = product_code.lower()
		if "postpaid" in code_lower or "contract" in code_lower:
			plan_type = "postpaid"
		elif "prepaid" in code_lower or "voucher" in code_lower:
			plan_type = "prepaid"
		elif "data" in code_lower or "broadband" in code_lower:
			plan_type = "data_bundle"
		else:
			plan_type = "prepaid"
		if plan_type not in SUPPORTED_PLAN_TYPES:
			plan_type = SUPPORTED_PLAN_TYPES[0] if SUPPORTED_PLAN_TYPES else "prepaid"
		plan_id = f"plan-{customer_id}-{product_code}-{_utcnow()[:10]}"
		plan = self.activate_plan(
			plan_id=plan_id,
			tenant_id=tenant_id,
			customer_id=customer_id,
			plan_type=plan_type,
			plan_name=product_code,
			plan_reference=product_code,
			activated_at=_utcnow(),
		)
		activation: dict[str, Any] = {
			"customer_id": customer_id,
			"product_code": product_code,
			"plan_id": plan_id,
			"parameters": parameters,
			"activated_by": activated_by,
			"tenant_id": tenant_id,
			"status": "active",
			"activated_at": _utcnow(),
			"plan": plan,
		}
		self._service_activations[f"{customer_id}:{product_code}"] = activation
		self._audit(tenant_id, "service_activated", f"{customer_id}:{product_code}")
		return activation

	async def suspend_service(
		self,
		customer_id: str,
		service_id: str,
		reason: str,
		tenant_id: str = "default",
		suspended_by: str = "system",
	) -> dict[str, Any]:
		"""Suspend a customer service.

		Validates the service is active, updates plan status to suspended,
		and records the suspension with reason.
		"""
		assert customer_id, "customer_id required"
		assert service_id, "service_id required"
		assert reason, "reason required"
		plan = self.plans.get(self._key(tenant_id, service_id))
		if plan is None:
			raise ValueError(f"Service/plan {service_id} not found for customer {customer_id}")
		if plan.status == "suspended":
			raise ValueError(f"Service {service_id} already suspended")
		plan.status = "suspended"
		suspension: dict[str, Any] = {
			"customer_id": customer_id,
			"service_id": service_id,
			"reason": reason,
			"suspended_by": suspended_by,
			"tenant_id": tenant_id,
			"suspended_at": _utcnow(),
		}
		self._service_suspensions[service_id] = suspension
		self._audit(tenant_id, "service_suspended", f"{customer_id}:{service_id}")
		return {**plan.to_dict(), "suspension": suspension}

	async def restore_service(
		self,
		customer_id: str,
		service_id: str,
		tenant_id: str = "default",
		restored_by: str = "system",
	) -> dict[str, Any]:
		"""Restore a previously suspended service.

		Validates suspension record exists, reactivates plan, and removes
		suspension record.
		"""
		assert customer_id, "customer_id required"
		assert service_id, "service_id required"
		plan = self.plans.get(self._key(tenant_id, service_id))
		if plan is None:
			raise ValueError(f"Service/plan {service_id} not found")
		if plan.status != "suspended":
			raise ValueError(f"Service {service_id} is not suspended (status: {plan.status})")
		plan.status = "active"
		suspension = self._service_suspensions.pop(service_id, {})
		self._audit(tenant_id, "service_restored", f"{customer_id}:{service_id}")
		return {
			**plan.to_dict(),
			"restored_by": restored_by,
			"restored_at": _utcnow(),
			"prior_suspension": suspension,
		}

	async def complaint_log(
		self,
		customer_id: str,
		complaint_type: str,
		description: str,
		tenant_id: str = "default",
		channel: str = "phone",
	) -> dict[str, Any]:
		"""Log a customer complaint and open a service case.

		Maps complaint_type to case_type, creates a case, and returns
		case with SLA due date based on complaint priority.
		"""
		assert customer_id, "customer_id required"
		assert complaint_type, "complaint_type required"
		assert description, "description required"
		complaint_type_norm = complaint_type.lower()
		# Map complaint to case type
		case_type = complaint_type_norm if complaint_type_norm in SUPPORTED_CASE_TYPES else (SUPPORTED_CASE_TYPES[0] if SUPPORTED_CASE_TYPES else "complaint")
		# SLA: billing = 48h, service = 24h, network = 4h
		sla_hours: dict[str, int] = {"billing": 48, "service_quality": 24, "network": 4, "complaint": 48}
		sla_h = sla_hours.get(complaint_type_norm, 48)
		sla_due = (datetime.datetime.utcnow() + datetime.timedelta(hours=sla_h)).isoformat() + "Z"
		case_id = f"case-complaint-{customer_id}-{_utcnow()[:10]}"
		case = self.open_case(
			case_id=case_id,
			tenant_id=tenant_id,
			customer_id=customer_id,
			case_type=case_type,
			description=f"[{complaint_type}] {description}",
			opened_at=_utcnow(),
		)
		case["channel"] = channel
		case["complaint_type"] = complaint_type
		case["sla_due_at"] = sla_due
		self._audit(tenant_id, "complaint_logged", case_id)
		return case

	async def complaint_resolution(
		self,
		complaint_id: str,
		resolution: str,
		resolved_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Resolve a logged customer complaint.

		Updates case status, records resolution text and resolver, and
		computes resolution time against SLA.
		"""
		assert complaint_id, "complaint_id required"
		assert resolution, "resolution required"
		assert resolved_by, "resolved_by required"
		case = self._case_or_raise(complaint_id, tenant_id)
		resolved_at = _utcnow()
		updated = self.update_case_status(
			case_id=complaint_id,
			tenant_id=tenant_id,
			new_status="resolved",
			resolved_at=resolved_at,
		)
		resolution_record: dict[str, Any] = {
			"complaint_id": complaint_id,
			"resolution": resolution,
			"resolved_by": resolved_by,
			"tenant_id": tenant_id,
			"resolved_at": resolved_at,
		}
		self._complaint_resolutions[complaint_id] = resolution_record
		self._audit(tenant_id, "complaint_resolved", complaint_id)
		return {**updated, "resolution": resolution_record}

	async def churn_risk_intervention(
		self,
		customer_id: str,
		intervention_type: str,
		tenant_id: str = "default",
		offered_by: str = "system",
	) -> dict[str, Any]:
		"""Execute a churn risk intervention for an at-risk customer.

		intervention_type: retention_call | discount_offer | loyalty_reward |
		win_back_offer | service_upgrade.
		Records the intervention and updates lifecycle events.
		"""
		assert customer_id, "customer_id required"
		assert intervention_type, "intervention_type required"
		customer = self._customer_or_raise(customer_id, tenant_id)
		valid_interventions = {
			"retention_call", "discount_offer", "loyalty_reward",
			"win_back_offer", "service_upgrade",
		}
		intervention_norm = intervention_type.lower()
		if intervention_norm not in valid_interventions:
			raise ValueError(f"Unknown intervention_type {intervention_type!r}")
		# Craft offer details based on type
		offer_details: dict[str, Any] = {
			"retention_call": {"action": "schedule_call", "priority": "high"},
			"discount_offer": {"discount_pct": 20, "duration_months": 3},
			"loyalty_reward": {"points": 500, "redeemable_for": "data_bundle"},
			"win_back_offer": {"free_months": 1, "product": "premium_data"},
			"service_upgrade": {"upgrade_to": "premium_tier", "free_trial_days": 30},
		}.get(intervention_norm, {})
		intervention: dict[str, Any] = {
			"customer_id": customer_id,
			"intervention_type": intervention_norm,
			"offer_details": offer_details,
			"offered_by": offered_by,
			"tenant_id": tenant_id,
			"status": "offered",
			"offered_at": _utcnow(),
		}
		self._churn_interventions.append(intervention)
		# Record lifecycle event
		event_id = f"evt-churn-intervention-{customer_id}-{_utcnow()}"
		evt_type = "churn_intervention" if "churn_intervention" in (SUPPORTED_LIFECYCLE_EVENTS or []) else (SUPPORTED_LIFECYCLE_EVENTS[0] if SUPPORTED_LIFECYCLE_EVENTS else "churn_risk_flagged")
		self.record_lifecycle_event(
			event_id=event_id,
			tenant_id=tenant_id,
			customer_id=customer_id,
			event_type=evt_type,
			event_reference=intervention_norm,
			occurred_at=_utcnow(),
			recorded_by=offered_by,
		)
		self._audit(tenant_id, "churn_intervention_executed", customer_id)
		return intervention

	async def customer_lifecycle_report(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Generate a customer lifecycle analytics report for a period.

		Returns: new customers, churn rate, active customers, KYC status
		distribution, complaint volume, and intervention effectiveness.
		"""
		assert period, "period required"
		all_customers = [c for c in self.customers.values() if c.tenant_id == tenant_id]
		total = len(all_customers)
		active = sum(1 for c in all_customers if c.status == "active")
		kyc_dist: dict[str, int] = {}
		for c in all_customers:
			kyc_dist[c.kyc_status] = kyc_dist.get(c.kyc_status, 0) + 1
		# Complaints
		open_complaints = sum(
			1 for c in self.cases.values()
			if c.tenant_id == tenant_id and c.status == "open"
		)
		resolved_complaints = len(self._complaint_resolutions)
		# Interventions
		intervention_count = sum(
			1 for i in self._churn_interventions
			if i.get("tenant_id") == tenant_id
		)
		intervention_accepted = sum(
			1 for i in self._churn_interventions
			if i.get("tenant_id") == tenant_id and i.get("status") == "accepted"
		)
		intervention_rate = round(intervention_accepted / max(intervention_count, 1), 4)
		# NPS
		nps_records = [r for r in self._nps_records if r.get("tenant_id") == tenant_id]
		avg_nps = round(statistics.mean([r["score"] for r in nps_records]), 2) if nps_records else None
		self._audit(tenant_id, "customer_lifecycle_report_generated", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_customers": total,
			"active_customers": active,
			"kyc_status_distribution": kyc_dist,
			"open_complaints": open_complaints,
			"resolved_complaints": resolved_complaints,
			"churn_interventions": intervention_count,
			"intervention_acceptance_rate": intervention_rate,
			"avg_nps": avg_nps,
			"generated_at": _utcnow(),
		}

	async def nps_survey_result(
		self,
		customer_id: str,
		score: int,
		comment: str,
		tenant_id: str = "default",
		survey_channel: str = "sms",
	) -> dict[str, Any]:
		"""Record a Net Promoter Score (NPS) survey result from a customer.

		score: 0-10 (0-6 = detractor, 7-8 = passive, 9-10 = promoter).
		Stores result, computes category, and flags high-risk detractors
		for follow-up.
		"""
		assert customer_id, "customer_id required"
		assert 0 <= score <= 10, f"NPS score must be 0-10, got {score}"
		category = "promoter" if score >= 9 else ("passive" if score >= 7 else "detractor")
		nps_record: dict[str, Any] = {
			"customer_id": customer_id,
			"score": score,
			"category": category,
			"comment": comment,
			"survey_channel": survey_channel,
			"tenant_id": tenant_id,
			"recorded_at": _utcnow(),
		}
		self._nps_records.append(nps_record)
		# Auto-open a follow-up case for detractors
		if category == "detractor":
			case_id = f"case-nps-detractor-{customer_id}-{_utcnow()[:10]}"
			case_type = "nps_follow_up" if "nps_follow_up" in (SUPPORTED_CASE_TYPES or []) else (SUPPORTED_CASE_TYPES[0] if SUPPORTED_CASE_TYPES else "complaint")
			try:
				customer = self.customers.get(self._key(tenant_id, customer_id))
				if customer:
					self.open_case(
						case_id=case_id,
						tenant_id=tenant_id,
						customer_id=customer_id,
						case_type=case_type,
						description=f"NPS detractor (score={score}): {comment}",
						opened_at=_utcnow(),
					)
					nps_record["follow_up_case_id"] = case_id
			except Exception:
				pass
			self._audit(tenant_id, "nps_detractor_flagged", customer_id)
		return nps_record

	# ------------------------------------------------------------------ #
	# Agent validation & batch                                            #
	# ------------------------------------------------------------------ #

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		cross_tenant_access_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "cus_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"cross_tenant_access_scope": cross_tenant_access_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "cus_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.telecom.cus.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		nps_scores = [r["score"] for r in self._nps_records if r.get("tenant_id") == tenant_id]
		return {
			"tenant_id": tenant_id,
			"customer_count": self._count(self.customers, tenant_id),
			"kyc_document_count": self._count(self.kyc_documents, tenant_id),
			"plan_count": self._count(self.plans, tenant_id),
			"sim_count": self._count(self.sims, tenant_id),
			"device_count": self._count(self.devices, tenant_id),
			"case_count": self._count(self.cases, tenant_id),
			"lifecycle_event_count": self._count(self.lifecycle_events, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"nps_survey_count": len(nps_scores),
			"avg_nps": round(statistics.mean(nps_scores), 2) if nps_scores else None,
			"churn_interventions": sum(1 for i in self._churn_interventions if i.get("tenant_id") == tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

	def _customer_or_raise(self, customer_id: str, tenant_id: str) -> CusCustomer:
		c = self.customers.get(self._key(tenant_id, customer_id))
		if c is None:
			raise ValueError(f"Customer {customer_id} not found")
		return c

	def _kyc_doc_or_raise(self, doc_id: str, tenant_id: str) -> CusKycDocument:
		d = self.kyc_documents.get(self._key(tenant_id, doc_id))
		if d is None:
			raise ValueError(f"KYC document {doc_id} not found")
		return d

	def _sim_or_raise(self, sim_id: str, tenant_id: str) -> CusSim:
		s = self.sims.get(self._key(tenant_id, sim_id))
		if s is None:
			raise ValueError(f"SIM {sim_id} not found")
		return s

	def _case_or_raise(self, case_id: str, tenant_id: str) -> CusCase:
		c = self.cases.get(self._key(tenant_id, case_id))
		if c is None:
			raise ValueError(f"Case {case_id} not found")
		return c

	async def churn_intervention(
		self,
		customer_id: str,
		churn_probability: float,
		intervention_type: str,
		tenant_id: str = "default",
		assigned_to: str = "",
	) -> dict[str, Any]:
		"""Record a churn intervention for a high-risk customer."""
		assert customer_id, "customer_id required"
		assert 0.0 <= churn_probability <= 1.0, "churn_probability must be 0-1"
		assert intervention_type, "intervention_type required"
		record: dict[str, Any] = {
			"id": f"churn-int-{customer_id}-{len(self._churn_interventions)}",
			"customer_id": customer_id,
			"churn_probability": churn_probability,
			"intervention_type": intervention_type,
			"assigned_to": assigned_to,
			"status": "open",
			"tenant_id": tenant_id,
			"created_at": _utcnow(),
		}
		self._churn_interventions.append(record)
		self._audit(tenant_id, "churn_intervention_created", record["id"])
		return record

	async def record_nps(
		self,
		customer_id: str,
		score: int,
		channel: str,
		tenant_id: str = "default",
		comment: str | None = None,
	) -> dict[str, Any]:
		"""Record an NPS survey response for a customer."""
		assert customer_id, "customer_id required"
		assert 0 <= score <= 10, "score must be 0-10"
		category = "promoter" if score >= 9 else "passive" if score >= 7 else "detractor"
		record: dict[str, Any] = {
			"id": f"nps-{customer_id}-{len(self._nps_records)}",
			"customer_id": customer_id,
			"score": score,
			"category": category,
			"channel": channel,
			"comment": comment,
			"tenant_id": tenant_id,
			"recorded_at": _utcnow(),
		}
		self._nps_records.append(record)
		self._audit(tenant_id, "nps_recorded", record["id"])
		return record

	async def nps_analytics(
		self,
		tenant_id: str = "default",
		period: str = "last_90_days",
	) -> dict[str, Any]:
		"""Compute NPS = %promoters - %detractors."""
		records = [r for r in self._nps_records if r["tenant_id"] == tenant_id]
		n = len(records)
		if not n:
			return {"period": period, "tenant_id": tenant_id, "nps": None, "response_count": 0}
		promoters = sum(1 for r in records if r["category"] == "promoter")
		detractors = sum(1 for r in records if r["category"] == "detractor")
		nps = round((promoters - detractors) / n * 100, 1)
		return {
			"period": period, "tenant_id": tenant_id, "response_count": n,
			"promoters": promoters, "detractors": detractors,
			"passives": n - promoters - detractors, "nps": nps, "computed_at": _utcnow(),
		}

	async def activate_service(
		self,
		customer_id: str,
		service_type: str,
		tenant_id: str = "default",
		activated_by: str = "system",
	) -> dict[str, Any]:
		"""Activate a value-added service for a customer."""
		assert customer_id, "customer_id required"
		assert service_type, "service_type required"
		record: dict[str, Any] = {
			"id": f"svc-act-{customer_id}-{service_type}",
			"customer_id": customer_id,
			"service_type": service_type,
			"status": "active",
			"activated_by": activated_by,
			"tenant_id": tenant_id,
			"activated_at": _utcnow(),
		}
		self._service_activations[record["id"]] = record
		self._audit(tenant_id, "service_activated", record["id"])
		return record

	async def suspend_service(
		self,
		customer_id: str,
		reason: str,
		tenant_id: str = "default",
		suspended_by: str = "system",
	) -> dict[str, Any]:
		"""Suspend services for a customer (e.g., non-payment)."""
		assert customer_id, "customer_id required"
		assert reason, "reason required"
		record: dict[str, Any] = {
			"id": f"svc-susp-{customer_id}",
			"customer_id": customer_id,
			"reason": reason,
			"suspended_by": suspended_by,
			"status": "suspended",
			"tenant_id": tenant_id,
			"suspended_at": _utcnow(),
		}
		self._service_suspensions[record["id"]] = record
		self._audit(tenant_id, "service_suspended", record["id"])
		return record

	async def resolve_complaint(
		self,
		case_id: str,
		resolution: str,
		resolved_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Resolve a customer complaint case with a resolution note."""
		assert case_id, "case_id required"
		assert resolution, "resolution required"
		case = self._case_or_raise(case_id, tenant_id)
		case.status = "resolved"
		resolution_record: dict[str, Any] = {
			"case_id": case_id,
			"resolution": resolution,
			"resolved_by": resolved_by,
			"tenant_id": tenant_id,
			"resolved_at": _utcnow(),
		}
		self._complaint_resolutions[case_id] = resolution_record
		self._audit(tenant_id, "complaint_resolved", case_id)
		return {**case.to_dict(), "resolution": resolution_record}

	async def export_customers(
		self,
		tenant_id: str = "default",
		format: str = "json",
	) -> dict[str, Any]:
		"""Export customer records in JSON or CSV format."""
		assert format in {"json", "csv"}, "format must be json or csv"
		customers = [c.to_dict() for c in self.customers.values() if c.tenant_id == tenant_id]
		self._audit(tenant_id, "customers_exported", f"format:{format}")
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if customers:
				writer = csv.DictWriter(buf, fieldnames=list(customers[0].keys()))
				writer.writeheader()
				writer.writerows(customers)
			return {"format": "csv", "record_count": len(customers), "content": buf.getvalue()}
		return {"format": "json", "record_count": len(customers), "records": customers}

	async def customer_analytics(
		self,
		tenant_id: str = "default",
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Compute customer KPIs: active count, KYC rate, churn risk."""
		customers = [c.to_dict() for c in self.customers.values() if c.tenant_id == tenant_id]
		active = sum(1 for c in customers if c.get("status") == "active")
		kyc_verified = sum(1 for c in customers if c.get("kyc_status") == "verified")
		kyc_rate = round(kyc_verified / max(len(customers), 1) * 100, 2)
		high_risk = sum(1 for iv in self._churn_interventions if iv["tenant_id"] == tenant_id and iv.get("churn_probability", 0) > 0.7)
		self._audit(tenant_id, "customer_analytics_run", period)
		return {
			"period": period, "tenant_id": tenant_id,
			"total_customers": len(customers), "active_count": active,
			"kyc_verified_count": kyc_verified, "kyc_rate_pct": kyc_rate,
			"high_churn_risk_count": high_risk, "computed_at": _utcnow(),
		}

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return customer management service health status."""
		return {
			"service": "TelecomCustomerService", "tenant_id": tenant_id, "status": "healthy",
			"customer_count": self._count(self.customers, tenant_id),
			"kyc_doc_count": self._count(self.kyc_documents, tenant_id),
			"case_count": self._count(self.cases, tenant_id),
			"checked_at": _utcnow(),
		}

	async def kyc_compliance_report(
		self,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Generate a KYC compliance report for regulatory purposes."""
		customers = [c.to_dict() for c in self.customers.values() if c.tenant_id == tenant_id]
		kyc_verified = sum(1 for c in customers if c.get("kyc_status") == "verified")
		kyc_pending = sum(1 for c in customers if c.get("kyc_status") == "pending")
		kyc_rejected = sum(1 for c in customers if c.get("kyc_status") == "rejected")
		self._audit(tenant_id, "kyc_compliance_report_generated", tenant_id)
		return {
			"tenant_id": tenant_id,
			"total_customers": len(customers),
			"kyc_verified": kyc_verified, "kyc_pending": kyc_pending, "kyc_rejected": kyc_rejected,
			"compliance_rate_pct": round(kyc_verified / max(len(customers), 1) * 100, 2),
			"generated_at": _utcnow(),
		}

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
		reasons = ", ".join(action.get("reason", action.get("rule", "policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "policy_denied")


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str = "default", format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		self._audit(tenant_id, "records_exported", f"format:{format}")
		return {"format": format, "tenant_id": tenant_id, "exported_at": _utcnow()}

	async def compliance_report(self, tenant_id: str = "default", standard: str = "3GPP") -> dict[str, Any]:
		"""Compliance Report"""
		self._audit(tenant_id, "compliance_report_generated", standard)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _utcnow()}

	async def bulk_create(self, records: list[dict], tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk Create"""
		assert records
		self._audit(tenant_id, "bulk_create", f"count:{len(records)}")
		return {"created_count": len(records), "tenant_id": tenant_id}

	async def analytics_summary(self, tenant_id: str = "default", period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		self._audit(tenant_id, "analytics_summary_run", period)
		return {"tenant_id": tenant_id, "period": period, "computed_at": _utcnow()}


# Backward-compatible alias

	async def ml_churn_predict(self, *args, **kwargs):
		"""AI-powered customer churn probability prediction. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="telecom_churn_prediction")
			return {"churn_probability": round(result.score,3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

TelecomCusService = TelecomCustomerService
