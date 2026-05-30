"""Deterministic generated-app runtime for MFAU."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract


ACTIVE_METHOD_TYPES = {"totp", "webauthn", "push", "email_otp", "sms_otp", "backup_codes", "hardware_key", "biometric"}
CHANNEL_METHOD_TYPES = {"email_otp", "sms_otp", "push"}
DEVICE_BOUND_METHOD_TYPES = {"webauthn", "push", "hardware_key", "biometric"}
PHISHING_RESISTANT_TYPES = {"webauthn", "hardware_key"}


@dataclass
class MfauRecord:
	"""Serializable MFAU runtime record."""

	id: str
	tenant_id: str
	kind: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


class MfauGuardrailError(ValueError):
	"""Raised when an MFAU rule denies or requires review for an operation."""

	def __init__(self, result: dict[str, Any]):
		self.result = result
		super().__init__(f"{result['decision']}:{','.join(result['matched_rules'])}")


class MfauService:
	"""In-memory MFAU runtime for generated APG applications."""

	def __init__(self, tenant_id: str = "default", configuration_overrides: dict[str, Any] | None = None):
		self.contract = get_capability_contract(tenant_id, configuration_overrides)
		self.configuration = self.contract["configuration"]
		self._profiles: dict[str, dict[str, Any]] = {}
		self._methods: dict[str, dict[str, Any]] = {}
		self._devices: dict[str, dict[str, Any]] = {}
		self._risk_assessments: dict[str, dict[str, Any]] = {}
		self._challenges: dict[str, dict[str, Any]] = {}
		self._recoveries: dict[str, dict[str, Any]] = {}
		self._backup_codes: dict[str, dict[str, Any]] = {}
		self._policies: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def describe(self) -> dict[str, Any]:
		return {
			"capability": self.contract["capability"],
			"display_name": self.contract["display_name"],
			"routes": self.contract["ui"]["routes"],
			"adapters": self.configuration["adapters"],
			"rule_count": len(self.contract["rule_engine"]["rules"]),
		}

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		enriched = {"tenant_context_present": True, **context}
		return evaluate_capability_rules(enriched)

	def create_user_profile(
		self,
		profile_id: str,
		tenant_id: str,
		user_id: str,
		policy_id: str,
		primary_channel: str,
		status: str = "active",
	) -> dict[str, Any]:
		self._raise_if_denied({
			"operation": "register_profile",
			"tenant_context_present": bool(tenant_id),
			"user_present": bool(user_id),
			"policy_present": bool(policy_id),
			"profile_status_allowed": status in {"active", "locked", "disabled", "pending"},
		})
		record = MfauRecord(
			id=profile_id,
			tenant_id=tenant_id,
			kind="user_profile",
			status=status,
			metadata={
				"user_id": user_id,
				"policy_id": policy_id,
				"primary_channel": primary_channel,
				"failed_attempts": 0,
				"locked": status == "locked",
			},
		).as_dict()
		self._profiles[profile_id] = record
		self._audit(tenant_id, "profile_registered", profile_id, user_id=user_id, audit_event_recorded=True)
		return record

	def enroll_method(
		self,
		method_id: str,
		tenant_id: str,
		user_id: str,
		method_type: str,
		channel_verified: bool = True,
		biometric_consent_recorded: bool = True,
		template_encrypted: bool = True,
		secret_encrypted: bool = True,
		device_id: str = "",
		phishing_resistant: bool = False,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		profile = self._profile_for_user(tenant_id, user_id)
		active_count = len([method for method in self._methods.values() if method["tenant_id"] == tenant_id and method["metadata"]["user_id"] == user_id and method["status"] == "active"])
		method_type_allowed = method_type in self.configuration["methods"]["enabled"]
		device_bound_method = method_type in DEVICE_BOUND_METHOD_TYPES
		context = {
			"operation": "enroll_method",
			"tenant_context_present": bool(tenant_id),
			"profile_present": profile is not None,
			"method_type_present": bool(method_type),
			"method_type_allowed": method_type_allowed,
			"channel_method": method_type in CHANNEL_METHOD_TYPES,
			"verified_channel": channel_verified,
			"method_type": method_type,
			"biometric_consent_recorded": biometric_consent_recorded,
			"template_encrypted": template_encrypted,
			"device_bound_method": device_bound_method,
			"device_binding_present": not device_bound_method or bool(device_id),
			"active_method_count": active_count,
			"review_recorded": review_recorded,
			"secret_encrypted": secret_encrypted,
		}
		self._raise_if_review_required(context)
		record = MfauRecord(
			id=method_id,
			tenant_id=tenant_id,
			kind="mfa_method",
			status="active",
			metadata={
				"user_id": user_id,
				"profile_id": profile["id"] if profile else "",
				"method_type": method_type,
				"device_id": device_id,
				"channel_verified": channel_verified,
				"phishing_resistant": phishing_resistant or method_type in PHISHING_RESISTANT_TYPES,
				"biometric_consent_recorded": biometric_consent_recorded if method_type == "biometric" else None,
				"template_encrypted": template_encrypted if method_type == "biometric" else None,
			},
		).as_dict()
		self._methods[method_id] = record
		self._audit(tenant_id, "method_enrolled", method_id, user_id=user_id, audit_event_recorded=True)
		return record

	def bind_device(self, device_id: str, tenant_id: str, user_id: str, trust_score: float, reviewed: bool = False) -> dict[str, Any]:
		self._raise_if_review_required({
			"tenant_context_present": bool(tenant_id),
			"external_risk_signal": False,
			"review_recorded": reviewed,
		})
		record = MfauRecord(
			id=device_id,
			tenant_id=tenant_id,
			kind="trusted_device",
			status="trusted" if trust_score >= self.configuration["risk"]["low_trust_device_threshold"] else "low_trust",
			metadata={"user_id": user_id, "trust_score": self._clamp_score(trust_score), "reviewed": reviewed},
		).as_dict()
		record["trust_score"] = record["metadata"]["trust_score"]
		self._devices[device_id] = record
		self._audit(tenant_id, "device_bound", device_id, user_id=user_id, audit_event_recorded=True)
		return record

	def assess_risk(
		self,
		assessment_id: str,
		tenant_id: str,
		user_id: str,
		risk_score: float,
		device_trust_score: float,
		external_signal: bool = False,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._raise_if_review_required({
			"tenant_context_present": bool(tenant_id),
			"external_risk_signal": external_signal,
			"review_recorded": review_recorded,
		})
		score = self._clamp_score(risk_score)
		record = MfauRecord(
			id=assessment_id,
			tenant_id=tenant_id,
			kind="risk_assessment",
			status="critical" if score >= self.configuration["risk"]["critical_risk_threshold"] else "high" if score > self.configuration["risk"]["high_risk_threshold"] else "normal",
			metadata={
				"user_id": user_id,
				"risk_score": score,
				"device_trust_score": self._clamp_score(device_trust_score),
				"external_signal": external_signal,
				"review_recorded": review_recorded,
			},
		).as_dict()
		record["risk_score"] = record["metadata"]["risk_score"]
		record["device_trust_score"] = record["metadata"]["device_trust_score"]
		self._risk_assessments[assessment_id] = record
		self._audit(tenant_id, "risk_assessed", assessment_id, user_id=user_id, audit_event_recorded=True)
		return record

	def create_challenge(
		self,
		challenge_id: str,
		tenant_id: str,
		user_id: str,
		method_id: str,
		assessment_id: str,
		action_risk: str = "normal",
		step_up_completed: bool = True,
		phishing_resistant_factor_present: bool = True,
		token_unexpired: bool = True,
		token_reused: bool = False,
		verification_evidence: bool = True,
		failed_attempts: int = 0,
		user_locked: bool = False,
		device_review_recorded: bool = False,
		risk_override_approved: bool = False,
	) -> dict[str, Any]:
		profile = self._profile_for_user(tenant_id, user_id)
		method = self._methods.get(method_id)
		risk = self._risk_assessments.get(assessment_id)
		context = {
			"operation": "create_challenge",
			"tenant_context_present": bool(tenant_id),
			"profile_present": profile is not None,
			"active_method_present": bool(method and method["tenant_id"] == tenant_id and method["status"] == "active"),
			"risk_score": risk["metadata"]["risk_score"] if risk else 1.0,
			"risk_override_approved": risk_override_approved,
			"step_up_completed": step_up_completed,
			"action_risk": action_risk,
			"phishing_resistant_factor_present": phishing_resistant_factor_present,
			"device_trust_score": risk["metadata"]["device_trust_score"] if risk else 0.0,
			"device_review_recorded": device_review_recorded,
			"profile_locked": user_locked or bool(profile and profile["metadata"].get("locked")),
		}
		self._raise_if_review_required(context)
		verify_context = {
			"operation": "verify_challenge",
			"tenant_context_present": bool(tenant_id),
			"method_present": bool(method),
			"verification_evidence_present": verification_evidence,
			"challenge_expired": not token_unexpired,
			"challenge_already_used": token_reused,
			"failed_attempt_count": failed_attempts,
		}
		self._raise_if_denied(verify_context)
		record = MfauRecord(
			id=challenge_id,
			tenant_id=tenant_id,
			kind="mfa_challenge",
			status="issued",
			metadata={
				"user_id": user_id,
				"method_id": method_id,
				"assessment_id": assessment_id,
				"action_risk": action_risk,
				"risk_score": context["risk_score"],
				"device_trust_score": context["device_trust_score"],
				"single_use": True,
				"used": False,
			},
		).as_dict()
		self._challenges[challenge_id] = record
		self._audit(tenant_id, "challenge_issued", challenge_id, user_id=user_id, audit_event_recorded=True)
		return record

	def complete_challenge(self, challenge_id: str, tenant_id: str, verification_evidence: bool = True) -> dict[str, Any]:
		challenge = self._get_tenant_record(self._challenges, challenge_id, tenant_id)
		self._raise_if_denied({
			"operation": "verify_challenge",
			"tenant_context_present": bool(tenant_id),
			"method_present": bool(challenge["metadata"].get("method_id")),
			"verification_evidence_present": verification_evidence,
			"challenge_expired": False,
			"challenge_already_used": bool(challenge["metadata"].get("used")),
			"failed_attempt_count": 0,
		})
		challenge["status"] = "completed"
		challenge["metadata"]["used"] = True
		challenge["metadata"]["completed_at"] = datetime.now(timezone.utc).isoformat()
		self._audit(tenant_id, "challenge_completed", challenge_id, user_id=challenge["metadata"]["user_id"], audit_event_recorded=True)
		return challenge

	def recover_account(
		self,
		recovery_id: str,
		tenant_id: str,
		user_id: str,
		verified_recovery_channel: bool = True,
		audit_event_recorded: bool = True,
		admin_recovery: bool = False,
		admin_approval_recorded: bool = True,
		recovery_evidence_present: bool = True,
	) -> dict[str, Any]:
		profile = self._profile_for_user(tenant_id, user_id)
		self._raise_if_denied({
			"operation": "recover_account",
			"tenant_context_present": bool(tenant_id),
			"profile_present": profile is not None,
			"verified_recovery_channel": verified_recovery_channel,
			"recovery_evidence_present": recovery_evidence_present,
			"admin_assisted": admin_recovery,
			"admin_approval_recorded": admin_approval_recorded,
			"audit_event_recorded": audit_event_recorded,
		})
		record = MfauRecord(
			id=recovery_id,
			tenant_id=tenant_id,
			kind="account_recovery",
			status="approved",
			metadata={
				"user_id": user_id,
				"profile_id": profile["id"] if profile else "",
				"admin_recovery": admin_recovery,
				"admin_approval_recorded": admin_approval_recorded,
			},
		).as_dict()
		self._recoveries[recovery_id] = record
		self._audit(tenant_id, "account_recovered", recovery_id, user_id=user_id, audit_event_recorded=audit_event_recorded)
		return record

	def generate_backup_codes(self, code_set_id: str, tenant_id: str, user_id: str, code_count: int = 10) -> dict[str, Any]:
		profile = self._profile_for_user(tenant_id, user_id)
		self._raise_if_denied({
			"operation": "recover_account",
			"tenant_context_present": bool(tenant_id),
			"profile_present": profile is not None,
			"verified_recovery_channel": True,
			"recovery_evidence_present": True,
			"admin_assisted": False,
			"admin_approval_recorded": True,
			"audit_event_recorded": True,
		})
		codes = [self._backup_code_value(code_set_id, user_id, index) for index in range(code_count)]
		record = MfauRecord(
			id=code_set_id,
			tenant_id=tenant_id,
			kind="backup_code_set",
			status="active",
			metadata={"user_id": user_id, "codes": codes, "used_codes": [], "remaining": len(codes)},
		).as_dict()
		self._backup_codes[code_set_id] = record
		self._audit(tenant_id, "backup_codes_generated", code_set_id, user_id=user_id, audit_event_recorded=True)
		return record

	def use_backup_code(self, code_set_id: str, tenant_id: str, user_id: str, code_value: str) -> dict[str, Any]:
		code_set = self._get_tenant_record(self._backup_codes, code_set_id, tenant_id)
		used = code_value in code_set["metadata"]["used_codes"]
		remaining = code_set["metadata"]["remaining"]
		self._raise_if_denied({
			"operation": "use_backup_code",
			"tenant_context_present": bool(tenant_id),
			"backup_codes_remaining": remaining,
			"backup_code_already_used": used,
		})
		if code_value not in code_set["metadata"]["codes"]:
			raise MfauGuardrailError({"decision": "deny", "matched_rules": ["backup_code_invalid"], "actions": [{"reason": "backup_code_invalid"}]})
		code_set["metadata"]["used_codes"].append(code_value)
		code_set["metadata"]["remaining"] -= 1
		code_set["metadata"]["user_id"] = user_id
		self._audit(tenant_id, "backup_code_used", code_set_id, user_id=user_id, audit_event_recorded=True)
		return code_set

	def disable_method(self, method_id: str, tenant_id: str, alternative_method_present: bool = True) -> dict[str, Any]:
		method = self._get_tenant_record(self._methods, method_id, tenant_id)
		self._raise_if_denied({
			"operation": "disable_method",
			"tenant_context_present": bool(tenant_id),
			"alternative_method_present": alternative_method_present,
		})
		method["status"] = "disabled"
		self._audit(tenant_id, "method_disabled", method_id, user_id=method["metadata"]["user_id"], audit_event_recorded=True)
		return method

	def rotate_method(self, method_id: str, tenant_id: str, recent_verification_present: bool = True) -> dict[str, Any]:
		method = self._get_tenant_record(self._methods, method_id, tenant_id)
		self._raise_if_denied({
			"operation": "rotate_method",
			"tenant_context_present": bool(tenant_id),
			"recent_verification_present": recent_verification_present,
		})
		method["metadata"]["rotated_at"] = datetime.now(timezone.utc).isoformat()
		self._audit(tenant_id, "method_rotated", method_id, user_id=method["metadata"]["user_id"], audit_event_recorded=True)
		return method

	def create_policy(self, policy_id: str, tenant_id: str, name: str, audit_event_recorded: bool = True) -> dict[str, Any]:
		self._raise_if_denied({
			"operation": "change_policy",
			"tenant_context_present": bool(tenant_id),
			"audit_event_recorded": audit_event_recorded,
		})
		record = MfauRecord(
			id=policy_id,
			tenant_id=tenant_id,
			kind="mfa_policy",
			status="active",
			metadata={"name": name, "audit_event_recorded": audit_event_recorded},
		).as_dict()
		self._policies[policy_id] = record
		self._audit(tenant_id, "policy_changed", policy_id, audit_event_recorded=audit_event_recorded)
		return record

	def list_profiles(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._profiles, tenant_id)

	def list_methods(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._methods, tenant_id)

	def list_devices(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._devices, tenant_id)

	def list_risk_assessments(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._risk_assessments, tenant_id)

	def list_challenges(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._challenges, tenant_id)

	def list_recoveries(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._recoveries, tenant_id)

	def list_backup_code_sets(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._backup_codes, tenant_id)

	def list_policies(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._policies, tenant_id)

	def list_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [event for event in self._audit_events if event["tenant_id"] == tenant_id]

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		methods = self.list_methods(tenant_id)
		challenges = self.list_challenges(tenant_id)
		risk = self.list_risk_assessments(tenant_id)
		return {
			"profile_count": len(self.list_profiles(tenant_id)),
			"active_method_count": len([method for method in methods if method["status"] == "active"]),
			"trusted_device_count": len([device for device in self.list_devices(tenant_id) if device["status"] == "trusted"]),
			"challenge_count": len(challenges),
			"completed_challenge_count": len([challenge for challenge in challenges if challenge["status"] == "completed"]),
			"high_risk_assessment_count": len([assessment for assessment in risk if assessment["status"] in {"high", "critical"}]),
			"recovery_count": len(self.list_recoveries(tenant_id)),
			"backup_code_set_count": len(self.list_backup_code_sets(tenant_id)),
			"policy_count": len(self.list_policies(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def package(self, tenant_id: str) -> dict[str, Any]:
		return {
			"contract": self.contract,
			"summary": self.dashboard_summary(tenant_id),
			"profiles": self.list_profiles(tenant_id),
			"methods": self.list_methods(tenant_id),
			"devices": self.list_devices(tenant_id),
			"risk_assessments": self.list_risk_assessments(tenant_id),
			"challenges": self.list_challenges(tenant_id),
			"recoveries": self.list_recoveries(tenant_id),
			"backup_codes": self.list_backup_code_sets(tenant_id),
			"policies": self.list_policies(tenant_id),
			"audit_events": self.list_audit_events(tenant_id),
		}

	def _profile_for_user(self, tenant_id: str, user_id: str) -> dict[str, Any] | None:
		for profile in self._profiles.values():
			if profile["tenant_id"] == tenant_id and profile["metadata"]["user_id"] == user_id:
				return profile
		return None

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [record for record in records.values() if record["tenant_id"] == tenant_id]

	def _get_tenant_record(self, records: dict[str, dict[str, Any]], record_id: str, tenant_id: str) -> dict[str, Any]:
		record = records[record_id]
		result = evaluate_capability_rules({"tenant_context_present": bool(tenant_id), "cross_tenant_access": record["tenant_id"] != tenant_id})
		if result["decision"] == "deny":
			raise MfauGuardrailError(result)
		return record

	def _raise_if_denied(self, context: dict[str, Any]) -> dict[str, Any]:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise MfauGuardrailError(result)
		return result

	def _raise_if_review_required(self, context: dict[str, Any]) -> dict[str, Any]:
		result = self._raise_if_denied(context)
		if result["decision"] == "require_review":
			raise MfauGuardrailError(result)
		return result

	def _audit(self, tenant_id: str, event_type: str, subject_id: str, audit_event_recorded: bool = True, **metadata: Any) -> None:
		self._raise_if_denied({
			"tenant_context_present": bool(tenant_id),
			"state_change_requested": True,
			"audit_event_recorded": audit_event_recorded,
		})
		self._audit_events.append({
			"id": f"audit_{len(self._audit_events) + 1}",
			"tenant_id": tenant_id,
			"event_type": event_type,
			"subject_id": subject_id,
			"metadata": metadata,
			"created_at": datetime.now(timezone.utc).isoformat(),
		})

	@staticmethod
	def _clamp_score(value: float | int) -> float:
		return max(0.0, min(1.0, float(value)))

	@staticmethod
	def _backup_code_value(code_set_id: str, user_id: str, index: int) -> str:
		digest = sha256(f"{code_set_id}:{user_id}:{index}".encode("utf-8")).hexdigest()
		return f"{digest[:4]}-{digest[4:8]}"
