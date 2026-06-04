"""Deterministic domain rules for Pharmacy Management.

Every business rule from the capability contract is implemented here as a
callable Python function. Rules are the single source of truth for governance
decisions. The service layer calls assert_* functions; the rule engine calls
evaluate_capability_rules from capability_contract.py.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


# ── Exception ─────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ── Tenant / Access Control ───────────────────────────────────────────────────

def assert_tenant_context(tenant_id: str | None) -> None:
	"""All operations require a non-empty tenant_id."""
	if not tenant_id or not tenant_id.strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant resource access is unconditionally denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resources of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_write_policy(operation_type: str, policy_attached: bool) -> None:
	"""Write operations require an attached policy."""
	if operation_type == "write" and not policy_attached:
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_write_policy",
		)


# ── Formulary Rules ───────────────────────────────────────────────────────────

def assert_drug_type_supported(drug_type: str, supported: list[str]) -> None:
	"""Drug type must be in the supported formulary drug types."""
	if drug_type not in supported:
		raise RuleViolation(
			"drug_type_not_supported",
			f"drug_type '{drug_type}' is not supported; valid: {supported}",
			"select_supported_drug_type",
		)


def assert_drug_schedule_supported(schedule: str, supported: list[str]) -> None:
	"""Drug schedule must be a recognised DEA/formulary schedule."""
	if schedule not in supported:
		raise RuleViolation(
			"drug_schedule_not_supported",
			f"drug_schedule '{schedule}' is not supported; valid: {supported}",
			"select_supported_drug_schedule",
		)


def assert_dosage_form_supported(dosage_form: str, supported: list[str]) -> None:
	"""Dosage form must be in the supported set."""
	if dosage_form not in supported:
		raise RuleViolation(
			"dosage_form_not_supported",
			f"dosage_form '{dosage_form}' is not supported; valid: {supported}",
			"select_supported_dosage_form",
		)


def assert_lasa_alert_type_supported(alert_type: str, supported: list[str]) -> None:
	"""LASA alert type must be one of: look_alike, sound_alike, look_and_sound_alike."""
	if alert_type not in supported:
		raise RuleViolation(
			"lasa_alert_type_not_supported",
			f"lasa_alert_type '{alert_type}' is not supported; valid: {supported}",
			"select_supported_lasa_alert_type",
		)


# ── Dispensing Rules ──────────────────────────────────────────────────────────

def assert_pharmacist_verified(pharmacist_verified: bool) -> None:
	"""A pharmacist must verify a prescription before it can be dispensed."""
	if not pharmacist_verified:
		raise RuleViolation(
			"pharmacist_verification_required",
			"pharmacist verification is required before dispensing",
			"obtain_pharmacist_verification",
		)


def assert_no_contraindicated_interaction(interaction_severity: str | None) -> None:
	"""Dispensing is blocked when a contraindicated drug interaction is present."""
	if interaction_severity == "contraindicated":
		raise RuleViolation(
			"contraindicated_dispense_denied",
			"contraindicated drug interaction blocks dispensing",
			"select_alternative_drug",
		)


def assert_drug_not_recalled(inventory_status: str) -> None:
	"""Recalled drugs must never be dispensed."""
	if inventory_status == "recalled":
		raise RuleViolation(
			"recalled_drug_dispense_denied",
			"recalled drug cannot be dispensed",
			"quarantine_recalled_drug",
		)


def assert_drug_not_expired(inventory_status: str) -> None:
	"""Expired drugs must not be dispensed."""
	if inventory_status == "expired":
		raise RuleViolation(
			"expired_drug_dispense_denied",
			"expired drug cannot be dispensed",
			"remove_expired_drug_from_inventory",
		)


def assert_drug_in_stock(inventory_status: str) -> None:
	"""Out-of-stock drugs cannot be dispensed."""
	if inventory_status == "out_of_stock":
		raise RuleViolation(
			"drug_out_of_stock",
			"drug is out of stock and cannot be dispensed",
			"place_inventory_order",
		)


def assert_prior_auth_approved(formulary_status: str, prior_auth_approved: bool) -> None:
	"""Drugs requiring prior authorisation must have an approved PA before dispensing."""
	if formulary_status == "prior_auth_required" and not prior_auth_approved:
		raise RuleViolation(
			"prior_authorization_required",
			"prior authorization is required for this drug but has not been approved",
			"obtain_prior_authorization",
		)


def assert_step_therapy_completed(formulary_status: str, step_therapy_completed: bool) -> None:
	"""Step therapy protocol must be completed before dispensing a step-therapy drug."""
	if formulary_status == "step_therapy" and not step_therapy_completed:
		raise RuleViolation(
			"step_therapy_not_completed",
			"step therapy protocol has not been completed for this drug",
			"complete_step_therapy_protocol",
		)


def assert_formulary_override_present(formulary_status: str, override_present: bool) -> None:
	"""Non-formulary drugs require an explicit formulary override."""
	if formulary_status == "non_formulary" and not override_present:
		raise RuleViolation(
			"non_formulary_requires_override",
			"non-formulary drug requires a formulary override",
			"obtain_formulary_override",
		)


def assert_lot_not_expired(expiry_date: datetime, reference: datetime | None = None) -> None:
	"""The specific inventory lot being dispensed must not be expired."""
	ref = reference or datetime.utcnow()
	if expiry_date <= ref:
		raise RuleViolation(
			"lot_expired",
			f"lot expired on {expiry_date.isoformat()} — cannot dispense",
			"select_non_expired_lot",
		)


def assert_return_reason_present(return_reason: str | None) -> None:
	"""A return reason is mandatory when processing a medication return."""
	if not return_reason or not return_reason.strip():
		raise RuleViolation(
			"return_reason_required",
			"return_reason is required when processing a medication return",
			"specify_return_reason",
		)


# ── Controlled Substance Rules ────────────────────────────────────────────────

def assert_dual_witness_for_waste(action: str, witness_id: str | None) -> None:
	"""Wasting a controlled substance requires a second pharmacist witness."""
	if action == "waste" and not witness_id:
		raise RuleViolation(
			"dual_witness_required_for_waste",
			"dual witness is required for controlled substance waste",
			"obtain_witness_signature",
		)


def assert_controlled_substance_action_supported(action: str, supported: list[str]) -> None:
	"""Controlled substance action must be one of the authorised types."""
	if action not in supported:
		raise RuleViolation(
			"controlled_substance_action_not_supported",
			f"action '{action}' is not a supported controlled substance action; valid: {supported}",
			"select_supported_action",
		)


def assert_cii_no_refills(schedule: str, refills: int) -> None:
	"""Schedule II substances do not permit refills under DEA regulations."""
	if schedule == "schedule_ii" and refills > 0:
		raise RuleViolation(
			"schedule_ii_no_refills",
			"Schedule II controlled substances may not be refilled; a new prescription is required",
			"issue_new_prescription",
		)


def assert_register_fields_present(register_entry: dict[str, Any]) -> None:
	"""Narcotics register entry must contain all legally required fields."""
	required = {
		"drug_name", "strength", "quantity",
		"patient_name", "patient_id", "prescriber_name", "prescriber_dea",
	}
	missing = required - set(register_entry.keys())
	if missing:
		raise RuleViolation(
			"register_fields_missing",
			f"narcotics register entry is missing required fields: {sorted(missing)}",
			"populate_all_register_fields",
		)


# ── Inventory Rules ───────────────────────────────────────────────────────────

def assert_inventory_status_supported(status: str, supported: list[str]) -> None:
	"""Inventory status must be one of the supported values."""
	if status not in supported:
		raise RuleViolation(
			"inventory_status_not_supported",
			f"inventory status '{status}' is not supported; valid: {supported}",
			"select_supported_inventory_status",
		)


def assert_quantity_positive(quantity: float, field_name: str = "quantity") -> None:
	"""Any dispensable quantity must be strictly positive."""
	if quantity <= 0:
		raise RuleViolation(
			"quantity_must_be_positive",
			f"{field_name} must be greater than zero, got {quantity}",
			"provide_positive_quantity",
		)


def assert_quantity_non_negative(quantity: float, field_name: str = "quantity") -> None:
	"""Stock quantities must be non-negative."""
	if quantity < 0:
		raise RuleViolation(
			"quantity_cannot_be_negative",
			f"{field_name} cannot be negative, got {quantity}",
			"provide_non_negative_quantity",
		)


# ── Cold Chain Rules ──────────────────────────────────────────────────────────

def assert_temperature_range_valid(min_acceptable: float, max_acceptable: float) -> None:
	"""Cold chain min temperature must be strictly less than max."""
	if min_acceptable >= max_acceptable:
		raise RuleViolation(
			"invalid_temperature_range",
			f"min_acceptable_c ({min_acceptable}) must be less than max_acceptable_c ({max_acceptable})",
			"correct_temperature_range",
		)


def assert_refrigerated_drug_has_cold_chain(
	requires_refrigeration: bool,
	cold_chain_verified: bool,
) -> None:
	"""Drugs requiring refrigeration must have a verified cold chain before dispensing."""
	if requires_refrigeration and not cold_chain_verified:
		raise RuleViolation(
			"cold_chain_verification_required",
			"refrigerated drug requires cold chain verification before dispensing",
			"verify_cold_chain_integrity",
		)


# ── Agent / Automation Rules ──────────────────────────────────────────────────

def assert_agent_privileged_action_approved(
	agent_action: bool,
	privileged_scope: bool,
	human_approval_recorded: bool,
) -> None:
	"""Privileged agent actions (dispense, waste, etc.) require recorded human approval."""
	if agent_action and privileged_scope and not human_approval_recorded:
		raise RuleViolation(
			"privileged_agent_action_requires_human_approval",
			"this agent action falls within a privileged scope and requires human approval",
			"record_human_approval",
		)


# ── Calculation Helpers ───────────────────────────────────────────────────────

def calculate_narcotics_balance(
	balance_before: float,
	action: str,
	quantity: float,
	waste_amount: float = 0.0,
) -> float:
	"""Compute expected post-action narcotics register balance.

	Additive actions: receive.
	Subtractive actions: dispense, waste, destroy, transfer.
	Neutral actions: count, audit — no balance change.
	"""
	additive = {"receive"}
	subtractive = {"dispense", "waste", "destroy", "transfer"}
	effective = quantity + waste_amount if action == "waste" else quantity
	if action in additive:
		result = balance_before + effective
	elif action in subtractive:
		result = balance_before - effective
	else:
		result = balance_before
	return round(result, 6)


def calculate_days_remaining(expiry_date: datetime, reference: datetime | None = None) -> int:
	"""Return whole days until expiry_date; negative if already expired."""
	ref = reference or datetime.utcnow()
	return (expiry_date - ref).days


def classify_expiry_alert(days_remaining: int) -> str:
	"""Map days-remaining to an alert level string."""
	if days_remaining <= 0:
		return "expired"
	if days_remaining < 7:
		return "critical"
	if days_remaining < 30:
		return "warning"
	if days_remaining < 90:
		return "notice"
	return "ok"


def classify_cold_chain_status(
	recorded_temp: float,
	min_acceptable: float,
	max_acceptable: float,
	excursion_minutes: int | None = None,
) -> str:
	"""Classify a temperature reading as compliant, excursion, or critical."""
	if min_acceptable <= recorded_temp <= max_acceptable:
		return "compliant"
	if excursion_minutes is not None and excursion_minutes > 60:
		return "critical"
	return "excursion"


def counselling_completion_score(checklist: dict[str, bool]) -> float:
	"""Return fraction of standard counselling items completed (0.0–1.0)."""
	fields = [
		"indication_explained", "dosage_explained", "administration_explained",
		"side_effects_explained", "interactions_explained", "storage_explained",
		"missed_dose_explained", "patient_questions_addressed", "patient_understood",
	]
	if not fields:
		return 0.0
	completed = sum(1 for f in fields if checklist.get(f, False))
	return round(completed / len(fields), 4)
