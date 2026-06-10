#!/usr/bin/env python3

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
"""
APG Key Management Service
Core service implementation with APG integration patterns

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import hashlib
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid_extensions import uuid7str

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from .models import (
	KeySpec, Key, KeyOperation, SecurityThreat, AuditEvent, KeyUsageStats,
	KeyAlgorithm, KeyUsage, KeyState, SecurityLevel, ComplianceFramework,
	HSMConfiguration, CloudKeyStore, create_key_spec_async
)


KEY_CLASSES = {"data", "root", "tenant", "signing", "wrapping"}
KEY_STATUSES = {"active", "disabled", "compromised", "destroyed"}
DECISIONS = {"allow", "deny", "require_review"}


def _utc_now() -> str:
	return datetime.utcnow().isoformat() + "Z"


def _stable_id(prefix: str, *parts: object) -> str:
	payload = "|".join(str(part) for part in parts)
	return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _normalize_key_class(value: str) -> str:
	normalized = str(value or "data").strip().lower()
	if normalized not in KEY_CLASSES:
		raise ValueError(f"unsupported_key_class:{value}")
	return normalized


def _normalize_status(value: str) -> str:
	normalized = str(value or "active").strip().lower()
	if normalized not in KEY_STATUSES:
		raise ValueError(f"unsupported_key_status:{value}")
	return normalized


def _required_actions(result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in result.get("actions", [])
		if action.get("required_action")
	]


@dataclass(slots=True)
class ManagedKeyRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	algorithm: str
	key_class: str
	policy_ref: str
	hsm_attested: bool
	status: str = "active"
	rotation_age_days: int = 0
	created_at: str = field(default_factory=_utc_now)
	last_rotated_at: str = ""
	compromised_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class KeyOperationRecord:
	id: str
	tenant_id: str
	key_id: str
	operation: str
	decision: str
	status: str
	matched_rules: list[str]
	required_actions: list[str]
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class ExportApprovalRecord:
	id: str
	tenant_id: str
	key_id: str
	requested_by: str
	reason: str
	status: str = "pending"
	decision: str = ""
	reviewer: str = ""
	notes: str = ""
	policy_decision: str = "require_review"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=lambda: ["export_approval_review_required"])
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class RotationExceptionRecord:
	id: str
	tenant_id: str
	key_id: str
	requested_by: str
	reason: str
	status: str = "pending"
	decision: str = ""
	reviewer: str = ""
	notes: str = ""
	policy_decision: str = "require_review"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=lambda: ["rotation_exception_review_required"])
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class KeyRotationRecord:
	id: str
	tenant_id: str
	key_id: str
	requested_by: str
	reason: str
	status: str = "scheduled"
	actor: str = ""
	evidence: str = ""
	policy_decision: str = "require_review"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=lambda: ["key_rotation_review_required"])
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=_utc_now)
	completed_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class KeymAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "info"
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class KeymAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool
	human_approval_required: bool
	policy_ref: str | None = None
	status: str = "active"
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	registered_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class KeyLifecycleBatchRecord:
	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	status: str = "accepted"
	processor: str = "bytewax"
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=_utc_now)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


class KeymService:
	"""Dependency-light KEYM service for generated APG applications."""

	def __init__(self) -> None:
		from .capability_contract import (
			PRIVILEGED_KEYM_AGENT_ROLES,
			SUPPORTED_KEYM_AGENT_ROLES,
			SUPPORTED_KEYM_AGENT_RUNTIMES,
			evaluate_capability_rules,
			get_capability_contract,
		)

		self._evaluate_rules = evaluate_capability_rules
		self._get_contract = get_capability_contract
		self._agent_runtimes = set(SUPPORTED_KEYM_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_KEYM_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_KEYM_AGENT_ROLES)
		self.keys: dict[str, ManagedKeyRecord] = {}
		self.operations: dict[str, KeyOperationRecord] = {}
		self.export_approvals: dict[str, ExportApprovalRecord] = {}
		self.rotation_exceptions: dict[str, RotationExceptionRecord] = {}
		self.rotations: dict[str, KeyRotationRecord] = {}
		self.audit_events: dict[str, KeymAuditEventRecord] = {}
		self.key_agents: dict[str, KeymAgentRecord] = {}
		self.key_lifecycle_batches: dict[str, KeyLifecycleBatchRecord] = {}

	def describe(self, tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
		return self._get_contract(tenant_id, overrides)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return self._evaluate_rules(dict(context))

	def create_managed_key(
		self,
		tenant_id: str,
		key_id: str,
		name: str,
		owner: str,
		algorithm: str = "AES-256",
		key_class: str = "data",
		policy_ref: str = "",
		hsm_attested: bool = False,
		rotation_age_days: int = 0,
		status: str = "active",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(key_id or "").strip():
			raise ValueError("managed_key_id_required")
		if not str(name or "").strip():
			raise ValueError("managed_key_name_required")
		if not str(owner or "").strip():
			raise ValueError("managed_key_owner_required")
		if not str(algorithm or "").strip():
			raise ValueError("managed_key_algorithm_required")
		key_class_value = _normalize_key_class(key_class)
		status_value = _normalize_status(status)
		context = {
			"tenant_context_present": True,
			"operation": "create_key",
			"policy_attached": bool(str(policy_ref or "").strip()),
			"key_class": key_class_value,
			"hsm_attested": bool(hsm_attested),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		record_id = _stable_id("keym_key", tenant_id, key_id)
		if record_id in self.keys:
			raise ValueError(f"managed_key_already_exists:{key_id}")
		record = ManagedKeyRecord(
			id=record_id,
			tenant_id=tenant_id,
			name=str(name).strip(),
			owner=str(owner).strip(),
			algorithm=str(algorithm).strip(),
			key_class=key_class_value,
			policy_ref=str(policy_ref).strip(),
			hsm_attested=bool(hsm_attested),
			status=status_value,
			rotation_age_days=max(0, int(rotation_age_days)),
		)
		self.keys[record.id] = record
		self._record_event(tenant_id, "managed_key_created", record.id, f"Managed key created: {record.name}", owner)
		return record.to_dict()

	def evaluate_key_operation(
		self,
		tenant_id: str,
		operation_id: str,
		key_id: str,
		operation: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(operation_id or "").strip():
			raise ValueError("key_operation_id_required")
		if not str(operation or "").strip():
			raise ValueError("key_operation_required")
		key = self._get_key(tenant_id, key_id)
		operation_name = str(operation).strip().lower()
		context = {
			"tenant_context_present": True,
			"operation": operation_name,
			"policy_attached": bool(key.policy_ref),
			"key_class": key.key_class,
			"hsm_attested": key.hsm_attested,
			"dual_control_approved": self._export_approved(tenant_id, key.id),
			"rotation_age_days": key.rotation_age_days,
			"rotation_exception_recorded": self._rotation_exception_approved(tenant_id, key.id),
			"key_status": key.status,
			"operation_is_cryptographic": operation_name in {"use_key", "encrypt", "decrypt", "sign", "verify", "export_key"},
		}
		result = self.evaluate(context)
		status = {
			"allow": "allowed",
			"deny": "denied",
			"require_review": "review_required",
		}[result["decision"]]
		record = KeyOperationRecord(
			id=_stable_id("keym_operation", tenant_id, operation_id),
			tenant_id=tenant_id,
			key_id=key.id,
			operation=operation_name,
			decision=result["decision"],
			status=status,
			matched_rules=list(result["matched_rules"]),
			required_actions=_required_actions(result),
			policy_decision=result["decision"],
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result),
		)
		self.operations[record.id] = record
		severity = "high" if status == "denied" else "medium" if status == "review_required" else "info"
		self._record_event(
			tenant_id,
			f"key_operation_{status}",
			record.id,
			f"Key operation {status}: {operation_name}",
			key.owner,
			severity,
			policy_result=result,
		)
		return record.to_dict()

	def request_export_approval(
		self,
		tenant_id: str,
		approval_id: str,
		key_id: str,
		requested_by: str,
		reason: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		key = self._get_key(tenant_id, key_id)
		if not str(approval_id or "").strip():
			raise ValueError("export_approval_id_required")
		if not str(requested_by or "").strip():
			raise ValueError("export_approval_requester_required")
		if not str(reason or "").strip():
			raise ValueError("export_approval_reason_required")
		record_id = _stable_id("keym_export_approval", tenant_id, approval_id)
		if record_id in self.export_approvals:
			raise ValueError(f"export_approval_already_exists:{approval_id}")
		policy_result = _review_result("export_approval_review_required", "review_key_export")
		record = ExportApprovalRecord(
			id=record_id,
			tenant_id=tenant_id,
			key_id=key.id,
			requested_by=str(requested_by).strip(),
			reason=str(reason).strip(),
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
		)
		self.export_approvals[record.id] = record
		self._record_event(
			tenant_id,
			"export_approval_requested",
			record.id,
			f"Export approval requested: {key.name}",
			requested_by,
			"medium",
			policy_result=policy_result,
		)
		return record.to_dict()

	def decide_export_approval(
		self,
		tenant_id: str,
		approval_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		record = self._get_export_approval(tenant_id, approval_id)
		result = self._decide_review_record(
			record,
			operation="decide_export_approval",
			reviewer=reviewer,
			decision=decision,
			notes=notes,
			self_review_reason="independent_export_reviewer_required",
		)
		self._record_event(
			tenant_id,
			"export_approval_decided",
			record.id,
			f"Export approval {decision}: {record.key_id}",
			reviewer,
			"medium",
			policy_result=result,
		)
		return record.to_dict()

	def request_rotation_exception(
		self,
		tenant_id: str,
		exception_id: str,
		key_id: str,
		requested_by: str,
		reason: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		key = self._get_key(tenant_id, key_id)
		if key.rotation_age_days <= 90:
			raise ValueError("rotation_exception_not_required")
		if not str(exception_id or "").strip():
			raise ValueError("rotation_exception_id_required")
		if not str(requested_by or "").strip():
			raise ValueError("rotation_exception_requester_required")
		if not str(reason or "").strip():
			raise ValueError("rotation_exception_reason_required")
		record_id = _stable_id("keym_rotation_exception", tenant_id, exception_id)
		if record_id in self.rotation_exceptions:
			raise ValueError(f"rotation_exception_already_exists:{exception_id}")
		policy_result = _review_result("rotation_exception_review_required", "review_rotation_exception")
		record = RotationExceptionRecord(
			id=record_id,
			tenant_id=tenant_id,
			key_id=key.id,
			requested_by=str(requested_by).strip(),
			reason=str(reason).strip(),
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
		)
		self.rotation_exceptions[record.id] = record
		self._record_event(
			tenant_id,
			"rotation_exception_requested",
			record.id,
			f"Rotation exception requested: {key.name}",
			requested_by,
			"medium",
			policy_result=policy_result,
		)
		return record.to_dict()

	def decide_rotation_exception(
		self,
		tenant_id: str,
		exception_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		record = self._get_rotation_exception(tenant_id, exception_id)
		result = self._decide_review_record(
			record,
			operation="decide_rotation_exception",
			reviewer=reviewer,
			decision=decision,
			notes=notes,
			self_review_reason="independent_rotation_exception_reviewer_required",
		)
		if record.status == "approved":
			for operation_record in self.operations.values():
				if operation_record.tenant_id == tenant_id and operation_record.key_id == record.key_id and operation_record.status == "review_required":
					operation_record.status = "allowed"
					operation_record.decision = "allow"
					operation_record.policy_decision = "allow"
					operation_record.required_actions = []
					operation_record.review_evidence = self._review_evidence(result, review_recorded=True)
		self._record_event(
			tenant_id,
			"rotation_exception_decided",
			record.id,
			f"Rotation exception {decision}: {record.key_id}",
			reviewer,
			"medium",
			policy_result=result,
		)
		return record.to_dict()

	def schedule_rotation(
		self,
		tenant_id: str,
		rotation_id: str,
		key_id: str,
		requested_by: str,
		reason: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		key = self._get_key(tenant_id, key_id)
		if key.status == "destroyed":
			raise PermissionError("key_destroyed")
		if not str(rotation_id or "").strip():
			raise ValueError("key_rotation_id_required")
		if not str(requested_by or "").strip():
			raise ValueError("key_rotation_requester_required")
		if not str(reason or "").strip():
			raise ValueError("key_rotation_reason_required")
		record_id = _stable_id("keym_rotation", tenant_id, rotation_id)
		if record_id in self.rotations:
			raise ValueError(f"key_rotation_already_exists:{rotation_id}")
		policy_result = _review_result("key_rotation_review_required", "complete_key_rotation_with_evidence")
		record = KeyRotationRecord(
			id=record_id,
			tenant_id=tenant_id,
			key_id=key.id,
			requested_by=str(requested_by).strip(),
			reason=str(reason).strip(),
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
		)
		self.rotations[record.id] = record
		self._record_event(
			tenant_id,
			"key_rotation_scheduled",
			record.id,
			f"Key rotation scheduled: {key.name}",
			requested_by,
			"medium",
			policy_result=policy_result,
		)
		return record.to_dict()

	def complete_rotation(self, tenant_id: str, rotation_id: str, actor: str, evidence: str) -> dict[str, Any]:
		record = self._get_rotation(tenant_id, rotation_id)
		key = self._get_key(tenant_id, record.key_id)
		if key.status == "destroyed":
			raise PermissionError("key_destroyed")
		if record.status == "completed":
			raise ValueError("key_rotation_already_completed")
		if not str(actor or "").strip():
			raise ValueError("key_rotation_actor_required")
		result = self.evaluate({
			"operation": "complete_rotation",
			"key_rotation_evidence_attached": bool(str(evidence or "").strip()),
		})
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		record.status = "completed"
		record.actor = str(actor).strip()
		record.evidence = str(evidence).strip()
		record.completed_at = _utc_now()
		record.policy_decision = result["decision"]
		record.matched_rules = list(result["matched_rules"])
		record.review_reasons = self._reasons(result)
		record.review_evidence = self._review_evidence(result, review_recorded=True)
		key.status = "active"
		key.rotation_age_days = 0
		key.last_rotated_at = record.completed_at
		self._record_event(
			tenant_id,
			"key_rotation_completed",
			record.id,
			f"Key rotation completed: {key.name}",
			actor,
			"medium",
			policy_result=result,
		)
		return record.to_dict()

	def mark_key_compromised(self, tenant_id: str, key_id: str, actor: str, evidence: str) -> dict[str, Any]:
		key = self._get_key(tenant_id, key_id)
		if not str(actor or "").strip():
			raise ValueError("compromise_actor_required")
		if not str(evidence or "").strip():
			raise ValueError("compromise_evidence_required")
		key.status = "compromised"
		key.compromised_at = _utc_now()
		self._record_event(tenant_id, "managed_key_compromised", key.id, evidence, actor, "high")
		return key.to_dict()

	def register_key_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
		policy_ref: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(agent_id or "").strip():
			raise ValueError("key_agent_id_required")
		if not str(name or "").strip():
			raise ValueError("key_agent_name_required")
		if not str(owner or "").strip():
			raise ValueError("key_agent_owner_required")
		if not str(purpose or "").strip():
			raise ValueError("key_agent_purpose_required")
		normalized_runtime = self._normalize_agent_token(runtime)
		normalized_role = self._normalize_agent_token(role)
		result = self.evaluate({
			"operation": "register_key_agent",
			"key_agent_runtime_supported": normalized_runtime in self._agent_runtimes,
			"key_agent_role_supported": normalized_role in self._agent_roles,
			"key_agent_scope_attached": bool(str(scope or "").strip()),
			"key_agent_privileged_role": normalized_role in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		if not contribution_disclosed:
			raise PermissionError("key_agent_contribution_disclosure_required")
		record_id = _stable_id("keym_agent", tenant_id, agent_id)
		if record_id in self.key_agents:
			raise ValueError(f"key_agent_already_exists:{agent_id}")
		record = KeymAgentRecord(
			id=record_id,
			tenant_id=tenant_id,
			name=str(name).strip(),
			runtime=normalized_runtime,
			role=normalized_role,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=True,
			human_approval_required=bool(human_approval_required),
			policy_ref=str(policy_ref).strip() if policy_ref else None,
			status="pending_review" if result["decision"] == "require_review" else "active",
			policy_decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result, review_recorded=bool(human_approval_required)),
		)
		self.key_agents[record.id] = record
		self._record_event(
			tenant_id,
			"key_agent_registered",
			record.id,
			f"Key agent registered: {record.name}",
			owner,
			"medium",
			policy_result=result,
		)
		return record.to_dict()

	def validate_key_lifecycle_batch(self, tenant_id: str, event_stream: str, mutation_count: int) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if mutation_count < 1:
			raise ValueError("key_lifecycle_batch_empty")
		stream = self._normalize_agent_token(event_stream)
		result = self.evaluate({
			"operation": "validate_key_lifecycle_batch",
			"event_stream": stream,
		})
		record = KeyLifecycleBatchRecord(
			id=_stable_id("keym_batch", tenant_id, stream, len(self.key_lifecycle_batches)),
			tenant_id=tenant_id,
			event_stream=stream,
			mutation_count=int(mutation_count),
			status="denied" if result["decision"] == "deny" else "accepted",
			policy_decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result),
		)
		self.key_lifecycle_batches[record.id] = record
		self._record_event(
			tenant_id,
			"key_lifecycle_batch_validated",
			record.id,
			f"Key lifecycle batch {record.status}: {stream}",
			"system",
			"medium" if record.status == "denied" else "info",
			policy_result=result,
		)
		if result["decision"] == "deny":
			raise PermissionError(self._first_reason(result))
		payload = record.to_dict()
		payload.update({
			"tenant_id": tenant_id,
			"event_stream": stream,
			"mutation_count": int(mutation_count),
			"accepted": True,
			"required_processor": "bytewax",
			"rule_result": result,
		})
		return payload

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.create_managed_key(
			tenant_id=tenant_id,
			key_id=record_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or metadata.get("created_by") or "system"),
			algorithm=str(metadata.get("algorithm") or "AES-256"),
			key_class=str(metadata.get("key_class") or "data"),
			policy_ref=str(metadata.get("policy_ref") or "policy://default"),
			hsm_attested=bool(metadata.get("hsm_attested", False)),
			rotation_age_days=int(metadata.get("rotation_age_days", 0) or 0),
			status=status,
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_keys(tenant_id)

	def list_keys(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.keys, tenant_id)

	def list_operations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.operations, tenant_id)

	def list_export_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.export_approvals, tenant_id)

	def list_rotation_exceptions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.rotation_exceptions, tenant_id)

	def list_rotations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.rotations, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def list_key_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.key_agents, tenant_id)

	def list_key_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.key_lifecycle_batches, tenant_id)

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = (
			self.list_operations(tenant_id)
			+ self.list_export_approvals(tenant_id)
			+ self.list_rotation_exceptions(tenant_id)
			+ self.list_rotations(tenant_id)
			+ self.list_key_agents(tenant_id)
			+ self.list_key_lifecycle_batches(tenant_id)
		)
		return [
			item
			for item in items
			if item.get("status") in {"pending", "pending_review", "review_required", "scheduled"}
		]

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		operations = self.list_operations(tenant_id)
		return {
			"tenant_id": tenant_id,
			"key_count": len(self.list_keys(tenant_id)),
			"operation_count": len(operations),
			"key_agent_count": len(self.list_key_agents(tenant_id)),
			"pending_key_agent_review_count": sum(1 for item in self.list_key_agents(tenant_id) if item["status"] == "pending_review"),
			"key_lifecycle_batch_count": len(self.list_key_lifecycle_batches(tenant_id)),
			"denied_key_lifecycle_batch_count": sum(1 for item in self.list_key_lifecycle_batches(tenant_id) if item["status"] == "denied"),
			"denied_operation_count": sum(1 for item in operations if item["status"] == "denied"),
			"review_required_count": sum(1 for item in operations if item["status"] == "review_required"),
			"pending_export_approval_count": sum(1 for item in self.list_export_approvals(tenant_id) if item["status"] == "pending"),
			"pending_rotation_exception_count": sum(1 for item in self.list_rotation_exceptions(tenant_id) if item["status"] == "pending"),
			"scheduled_rotation_count": sum(1 for item in self.list_rotations(tenant_id) if item["status"] == "scheduled"),
			"compromised_key_count": sum(1 for item in self.list_keys(tenant_id) if item["status"] == "compromised"),
			"pending_review_count": len(self.list_pending_reviews(tenant_id)),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			raise PermissionError("tenant_context_required")

	def _get_key(self, tenant_id: str, key_id: str) -> ManagedKeyRecord:
		record = self.keys.get(_stable_id("keym_key", tenant_id, key_id)) or self.keys.get(key_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"managed_key_not_found:{key_id}")
		return record

	def _get_export_approval(self, tenant_id: str, approval_id: str) -> ExportApprovalRecord:
		record = self.export_approvals.get(_stable_id("keym_export_approval", tenant_id, approval_id)) or self.export_approvals.get(approval_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"export_approval_not_found:{approval_id}")
		return record

	def _get_rotation_exception(self, tenant_id: str, exception_id: str) -> RotationExceptionRecord:
		record = self.rotation_exceptions.get(_stable_id("keym_rotation_exception", tenant_id, exception_id)) or self.rotation_exceptions.get(exception_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"rotation_exception_not_found:{exception_id}")
		return record

	def _get_rotation(self, tenant_id: str, rotation_id: str) -> KeyRotationRecord:
		record = self.rotations.get(_stable_id("keym_rotation", tenant_id, rotation_id)) or self.rotations.get(rotation_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(f"key_rotation_not_found:{rotation_id}")
		return record

	def _export_approved(self, tenant_id: str, key_id: str) -> bool:
		return any(item.tenant_id == tenant_id and item.key_id == key_id and item.status == "approved" for item in self.export_approvals.values())

	def _rotation_exception_approved(self, tenant_id: str, key_id: str) -> bool:
		return any(item.tenant_id == tenant_id and item.key_id == key_id and item.status == "approved" for item in self.rotation_exceptions.values())

	def _decide_review_record(self, record: Any, operation: str, reviewer: str, decision: str, notes: str, self_review_reason: str) -> dict[str, Any]:
		if record.status != "pending":
			raise ValueError("review_already_decided")
		decision_value = str(decision or "").strip().lower()
		if decision_value not in {"approved", "rejected"}:
			raise ValueError("review_decision_invalid")
		reviewer_value = str(reviewer or "").strip()
		notes_value = str(notes or "").strip()
		if not reviewer_value:
			raise ValueError("reviewer_required")
		if not notes_value:
			raise ValueError("review_notes_required")
		requester_value = str(record.requested_by or "").strip()
		result = self.evaluate({
			"operation": operation,
			"reviewer_same_as_requester": reviewer_value.casefold() == requester_value.casefold(),
			"review_notes_attached": bool(notes_value),
		})
		if result["decision"] == "deny":
			reason = self._first_reason(result)
			raise PermissionError(self_review_reason if reason == "independent_reviewer_required" else reason)
		record.status = decision_value
		record.decision = decision_value
		record.reviewer = reviewer_value
		record.notes = notes_value
		record.policy_decision = result["decision"]
		record.matched_rules = list(result["matched_rules"])
		record.review_reasons = self._reasons(result)
		record.review_evidence = self._review_evidence(result, review_recorded=True)
		return result

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "info",
		policy_result: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		policy_result = policy_result or _allow_result()
		record = KeymAuditEventRecord(
			id=_stable_id("keym_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _first_reason(self, result: dict[str, Any]) -> str:
		for action in result.get("actions", []):
			if action.get("reason"):
				return str(action["reason"])
		return "key_operation_denied"

	def _reasons(self, result: dict[str, Any]) -> list[str]:
		return [
			str(action["reason"])
			for action in result.get("actions", [])
			if action.get("reason")
		]

	def _review_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": [
				str(action.get("required_action"))
				for action in result.get("actions", [])
				if action.get("required_action")
			],
			"reasons": self._reasons(result),
			"review_recorded": bool(review_recorded),
		}

	def _normalize_agent_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])


def _allow_result() -> dict[str, Any]:
	return {"decision": "allow", "matched_rules": [], "actions": []}


def _review_result(reason: str, required_action: str) -> dict[str, Any]:
	return {
		"decision": "require_review",
		"matched_rules": [],
		"actions": [{"reason": reason, "required_action": required_action}],
	}


class KeyManagementService:
	"""
	AI-powered key management service with APG integration
	Provides secure key lifecycle management, cryptographic operations, 
	and intelligent automation within the APG ecosystem
	"""
	
	def __init__(self):
		self.is_initialized = False
		self.config: Dict[str, Any] = {}
		self.keys: Dict[str, Key] = {}  # In-memory key store (use secure storage in production)
		self.usage_stats: Dict[str, KeyUsageStats] = {}
		self.threats: Dict[str, SecurityThreat] = {}
		self.audit_events: List[AuditEvent] = []
		self.hsm_configs: Dict[str, HSMConfiguration] = {}
		self.cloud_stores: Dict[str, CloudKeyStore] = {}
		
		# APG integration clients (initialized during setup)
		self.auth_client = None
		self.audit_client = None
		self.security_client = None
		self.config_client = None
		self.ai_client = None
		
		# Blockchain audit logger (initialized during setup)
		self.blockchain_audit_logger: Optional['BlockchainAuditLogger'] = None
		
		# IoT and edge computing manager (initialized during setup)
		self.iot_device_manager: Optional['IoTDeviceManager'] = None
	
	async def _log_audit_event(self, event_type: str, resource_id: str, action: str, 
							  user_id: str | None = None, details: Dict[str, Any] | None = None,
							  result: str = "success", ip_address: str = "", user_agent: str = "") -> None:
		"""Log audit event for APG compliance with blockchain immutability"""
		# Traditional audit event for backward compatibility
		event = AuditEvent(
			tenant_id=self.config.get('tenant_id', 'default'),
			event_type=event_type,
			resource_type='key',
			resource_id=resource_id,
			user_id=user_id,
			action=action,
			outcome=result,
			details=details or {}
		)
		self.audit_events.append(event)
		
		# Integration with APG audit capability
		if self.audit_client:
			await self.audit_client.log_event(event.model_dump())
		
		# Blockchain audit logging for immutable trail
		if self.blockchain_audit_logger:
			try:
				from .blockchain_audit import AuditEvent as BlockchainAuditEvent, AuditEventType
				
				# Map event types to blockchain audit types
				blockchain_event_type = self._map_to_blockchain_event_type(event_type)
				
				blockchain_event = BlockchainAuditEvent(
					event_type=blockchain_event_type,
					tenant_id=self.config.get('tenant_id', 'default'),
					user_id=user_id or "system",
					resource_id=resource_id,
					resource_type='cryptographic_key',
					action=action,
					result=result,
					ip_address=ip_address,
					user_agent=user_agent,
					context=details or {},
					metadata={
						'service': 'keym',
						'version': '1.0.0',
						'original_event_type': event_type
					}
				)
				
				await self.blockchain_audit_logger.log_audit_event(blockchain_event)
				
			except Exception as e:
				# Don't fail the operation if blockchain audit fails
				await self._log_key_operation("BLOCKCHAIN_AUDIT", "audit_event", False, 
											 f"Blockchain audit logging failed: {str(e)}")
	
	def _map_to_blockchain_event_type(self, event_type: str) -> 'AuditEventType':
		"""Map traditional event types to blockchain audit event types"""
		from .blockchain_audit import AuditEventType
		
		mapping = {
			'key_created': AuditEventType.KEY_CREATED,
			'key_accessed': AuditEventType.KEY_ACCESSED,
			'key_rotated': AuditEventType.KEY_ROTATED,
			'key_deleted': AuditEventType.KEY_DELETED,
			'encrypt': AuditEventType.ENCRYPTION_OPERATION,
			'decrypt': AuditEventType.DECRYPTION_OPERATION,
			'hsm_operation': AuditEventType.HSM_OPERATION,
			'policy_change': AuditEventType.POLICY_CHANGE,
			'user_access': AuditEventType.USER_ACCESS,
			'admin_action': AuditEventType.ADMIN_ACTION,
			'compliance_check': AuditEventType.COMPLIANCE_CHECK,
			'security_incident': AuditEventType.SECURITY_INCIDENT
		}
		
		return mapping.get(event_type, AuditEventType.KEY_ACCESSED)
	
	async def _log_key_operation(self, operation_type: str, key_id: str, success: bool = True, 
								details: str | None = None) -> None:
		"""Log key operations for monitoring"""
		print(f"[KEYM] {operation_type} key {key_id}: {'SUCCESS' if success else 'FAILED'} - {details or ''}")
	
	async def _validate_tenant_access(self, tenant_id: str, user_id: str | None = None) -> bool:
		"""Validate tenant access through APG auth integration"""
		assert tenant_id, "Tenant ID required"
		
		# Integration with APG auth/rbac capability
		if self.auth_client and user_id:
			return await self.auth_client.validate_tenant_access(user_id, tenant_id)
		
		return True  # Default allow for development
	
	async def _check_key_permissions(self, key_id: str, user_id: str | None, operation: str) -> bool:
		"""Check key-level permissions through APG RBAC"""
		assert key_id, "Key ID required"
		assert operation, "Operation required"
		
		# Integration with APG auth/rbac capability
		if self.auth_client and user_id:
			permission = f"keym.{operation}"
			return await self.auth_client.check_permission(user_id, permission, key_id)
		
		return True  # Default allow for development
	
	async def _detect_security_threats(self, operation: KeyOperation) -> List[SecurityThreat]:
		"""AI-powered security threat detection"""
		threats = []
		
		# Anomaly detection through APG integration
		if self.security_client:
			anomalies = await self.security_client.detect_anomalies({
				'operation_type': operation.operation_type,
				'user_id': operation.user_id,
				'source_ip': operation.request_ip,
				'timestamp': operation.requested_at
			})
			
			for anomaly in anomalies:
				threat = SecurityThreat(
					tenant_id=self.config.get('tenant_id', 'default'),
					threat_type=anomaly.get('type', 'unknown'),
					severity=anomaly.get('severity', 'low'),
					confidence=anomaly.get('confidence', 0.5),
					source_ip=operation.request_ip,
					user_id=operation.user_id,
					detection_method='ml_anomaly_detection'
				)
				threats.append(threat)
				self.threats[threat.threat_id] = threat
		
		return threats
	
	async def _generate_symmetric_key(self, algorithm: KeyAlgorithm, key_size: int) -> bytes:
		"""Generate symmetric cryptographic key"""
		assert algorithm in [KeyAlgorithm.AES_128, KeyAlgorithm.AES_256, KeyAlgorithm.CHACHA20_POLY1305]
		assert key_size > 0
		
		key_bytes = key_size // 8
		return secrets.token_bytes(key_bytes)
	
	async def _generate_asymmetric_key_pair(self, algorithm: KeyAlgorithm, key_size: int) -> Tuple[bytes, bytes]:
		"""Generate asymmetric key pair"""
		assert algorithm in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_4096, KeyAlgorithm.ECDSA_P256, KeyAlgorithm.ECDSA_P384]
		assert key_size > 0
		
		if algorithm in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_4096]:
			# Generate RSA key pair
			private_key = rsa.generate_private_key(
				public_exponent=65537,
				key_size=key_size,
				backend=default_backend()
			)
			
			private_pem = private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
			
			public_pem = private_key.public_key().public_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PublicFormat.SubjectPublicKeyInfo
			)
			
			return private_pem, public_pem
			
		elif algorithm in [KeyAlgorithm.ECDSA_P256, KeyAlgorithm.ECDSA_P384]:
			# Generate ECDSA key pair
			curve = ec.SECP256R1() if algorithm == KeyAlgorithm.ECDSA_P256 else ec.SECP384R1()
			private_key = ec.generate_private_key(curve, default_backend())
			
			private_pem = private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
			
			public_pem = private_key.public_key().public_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PublicFormat.SubjectPublicKeyInfo
			)
			
			return private_pem, public_pem
		
		raise ValueError(f"Unsupported asymmetric algorithm: {algorithm}")
	
	async def _encrypt_key_material(self, key_material: bytes, tenant_id: str) -> bytes:
		"""Encrypt key material for secure storage"""
		# Use tenant-specific encryption key (integration with APG security)
		tenant_key = self.config.get(f'tenant_key_{tenant_id}', b'default_key_32_bytes_for_dev_only')[:32]
		
		# Generate random IV
		iv = secrets.token_bytes(16)
		
		# AES-GCM encryption
		cipher = Cipher(algorithms.AES(tenant_key), modes.GCM(iv), backend=default_backend())
		encryptor = cipher.encryptor()
		ciphertext = encryptor.update(key_material) + encryptor.finalize()
		
		# Return IV + tag + ciphertext
		return iv + encryptor.tag + ciphertext
	
	async def _decrypt_key_material(self, encrypted_material: bytes, tenant_id: str) -> bytes:
		"""Decrypt key material from secure storage"""
		# Use tenant-specific encryption key
		tenant_key = self.config.get(f'tenant_key_{tenant_id}', b'default_key_32_bytes_for_dev_only')[:32]
		
		# Extract IV, tag, and ciphertext
		iv = encrypted_material[:16]
		tag = encrypted_material[16:32] 
		ciphertext = encrypted_material[32:]
		
		# AES-GCM decryption
		cipher = Cipher(algorithms.AES(tenant_key), modes.GCM(iv, tag), backend=default_backend())
		decryptor = cipher.decryptor()
		return decryptor.update(ciphertext) + decryptor.finalize()
	
	async def initialize(self, config: Dict[str, Any]) -> None:
		"""Initialize key management service with APG configuration"""
		assert isinstance(config, dict), "Configuration must be a dictionary"
		
		self.config = config
		
		# Initialize APG client connections
		await self._initialize_apg_clients()
		
		# Initialize blockchain audit system
		await self._initialize_blockchain_audit()
		
		# Initialize IoT and edge computing system
		await self._initialize_iot_edge_computing()
		
		# Load HSM configurations
		await self._load_hsm_configurations()
		
		# Load cloud key store configurations
		await self._load_cloud_configurations()
		
		self.is_initialized = True
		await self._log_key_operation("INITIALIZE", "service", True, "Key management service initialized")
	
	async def _initialize_apg_clients(self) -> None:
		"""Initialize APG capability client connections"""
		# These would be actual APG client implementations in production
		self.auth_client = None  # APG auth/rbac client
		self.audit_client = None  # APG audit logging client
		self.security_client = None  # APG security framework client
		self.config_client = None  # APG configuration client
		self.ai_client = None  # APG AI orchestration client
	
	async def _initialize_blockchain_audit(self) -> None:
		"""Initialize blockchain-based immutable audit system"""
		try:
			from .blockchain_audit import create_blockchain_audit_system
			
			# Get blockchain configuration
			blockchain_config = self.config.get('blockchain_audit', {
				'type': 'private',
				'block_size': 100,
				'block_interval': 300,
				'difficulty': 4
			})
			
			# Initialize blockchain audit logger
			self.blockchain_audit_logger = await create_blockchain_audit_system(
				self, blockchain_config
			)
			
			await self._log_key_operation("BLOCKCHAIN_AUDIT", "system", True, 
										 "Blockchain audit system initialized")
			
		except Exception as e:
			await self._log_key_operation("BLOCKCHAIN_AUDIT", "system", False, 
										 f"Failed to initialize blockchain audit: {str(e)}")
			# Continue without blockchain audit - fallback to traditional audit
			self.blockchain_audit_logger = None
	
	async def _initialize_iot_edge_computing(self) -> None:
		"""Initialize IoT and edge computing management system"""
		try:
			from .edge_iot_integration import create_iot_device_manager
			
			# Get IoT configuration
			iot_config = self.config.get('iot_edge', {
				'mqtt': {
					'broker': 'localhost',
					'port': 1883,
					'username': '',
					'password': '',
					'use_tls': False
				},
				'edge_nodes': {
					'auto_discovery': True,
					'heartbeat_interval': 60,
					'offline_threshold': 300
				},
				'device_management': {
					'auto_key_rotation': True,
					'default_rotation_interval': 86400,
					'batch_operations': True
				}
			})
			
			# Initialize IoT device manager
			self.iot_device_manager = await create_iot_device_manager(self, iot_config)
			
			await self._log_key_operation("IOT_EDGE", "system", True, 
										 "IoT and edge computing system initialized")
			
		except Exception as e:
			await self._log_key_operation("IOT_EDGE", "system", False, 
										 f"Failed to initialize IoT/edge system: {str(e)}")
			# Continue without IoT/edge - service can still function
			self.iot_device_manager = None
	
	async def _load_hsm_configurations(self) -> None:
		"""Load Hardware Security Module configurations"""
		# Load from APG configuration management
		try:
			# Simulate loading HSM configurations from APG config store
			hsm_configs = {
				'thales_hsm_01': {
					'vendor': 'Thales',
					'model': 'Luna SA',
					'ip_address': '10.0.1.100',
					'port': 1792,
					'partition': 'main',
					'auth_method': 'password',
					'enabled': True
				},
				'safenet_hsm_01': {
					'vendor': 'SafeNet',
					'model': 'ProtectServer',
					'ip_address': '10.0.1.101',
					'port': 1792,
					'partition': 'primary',
					'auth_method': 'certificate',
					'enabled': True
				},
				'aws_cloudhsm_01': {
					'vendor': 'AWS',
					'model': 'CloudHSM',
					'cluster_id': 'cluster-abc123',
					'region': 'us-west-2',
					'auth_method': 'iam',
					'enabled': False  # Disabled by default
				}
			}
			
			# Store configurations for HSM integration manager
			self.hsm_configs = hsm_configs
			print(f"[SERVICE] Loaded {len(hsm_configs)} HSM configurations")
			
		except Exception as e:
			print(f"[SERVICE] Failed to load HSM configurations: {e}")
			self.hsm_configs = {}
	
	async def _load_cloud_configurations(self) -> None:
		"""Load cloud key store configurations"""
		# Load from APG configuration management
		try:
			# Simulate loading cloud configurations from APG config store
			cloud_configs = {
				'aws_kms': {
					'provider': 'aws',
					'region': 'us-west-2',
					'service_name': 'AWS KMS',
					'endpoint': 'kms.us-west-2.amazonaws.com',
					'auth_method': 'iam',
					'enabled': True,
					'encryption_context_required': True,
					'key_spec': 'SYMMETRIC_DEFAULT'
				},
				'azure_keyvault': {
					'provider': 'azure',
					'region': 'westus2',
					'service_name': 'Azure Key Vault',
					'vault_name': 'apg-keym-vault',
					'auth_method': 'managed_identity',
					'enabled': True,
					'key_type': 'RSA-HSM',
					'key_size': 2048
				},
				'gcp_kms': {
					'provider': 'gcp',
					'region': 'us-central1',
					'service_name': 'Google Cloud KMS',
					'project_id': 'apg-keym-project',
					'auth_method': 'service_account',
					'enabled': True,
					'protection_level': 'HSM',
					'key_ring': 'apg-keyring'
				}
			}
			
			# Store configurations for cloud federation manager
			self.cloud_configs = cloud_configs
			print(f"[SERVICE] Loaded {len(cloud_configs)} cloud configurations")
			
		except Exception as e:
			print(f"[SERVICE] Failed to load cloud configurations: {e}")
			self.cloud_configs = {}
	
	async def create_key(self, spec: KeySpec, user_id: str | None = None) -> Key:
		"""Create new cryptographic key"""
		assert self.is_initialized, "Service not initialized"
		assert spec, "Key specification required"
		
		# Validate tenant access
		await self._validate_tenant_access(spec.tenant_id, user_id)
		
		# Check permissions
		await self._check_key_permissions(spec.id, user_id, "create_key")
		
		try:
			# Generate cryptographic material
			if spec.algorithm in [KeyAlgorithm.AES_128, KeyAlgorithm.AES_256, KeyAlgorithm.CHACHA20_POLY1305]:
				# Symmetric key
				key_material = await self._generate_symmetric_key(spec.algorithm, spec.key_size)
				public_key = None
			else:
				# Asymmetric key pair
				key_material, public_key = await self._generate_asymmetric_key_pair(spec.algorithm, spec.key_size)
			
			# Encrypt key material for storage
			encrypted_material = await self._encrypt_key_material(key_material, spec.tenant_id)
			
			# Calculate key checksum
			checksum = hashlib.sha256(key_material).hexdigest()
			
			# Create key object
			key = Key(
				spec=spec,
				key_material=encrypted_material,
				public_key=public_key,
				key_checksum=checksum
			)
			
			# Store key
			self.keys[spec.id] = key
			
			# Initialize usage statistics
			self.usage_stats[spec.id] = KeyUsageStats(
				key_id=spec.id,
				tenant_id=spec.tenant_id,
				first_used=datetime.utcnow()
			)
			
			# Update key state
			key.spec.state = KeyState.ACTIVE
			key.spec.updated_at = datetime.utcnow()
			
			# Schedule automatic rotation if enabled
			if spec.policy.auto_rotate:
				key.next_rotation = datetime.utcnow() + timedelta(days=spec.policy.rotation_interval_days)
			
			# Log audit event
			await self._log_audit_event("key_created", spec.id, "create_key", user_id, {
				"algorithm": spec.algorithm.value,
				"key_size": spec.key_size,
				"usage": [usage.value for usage in spec.usage]
			})
			
			await self._log_key_operation("CREATE", spec.id, True, f"Algorithm: {spec.algorithm}, Size: {spec.key_size}")
			
			return key
			
		except Exception as e:
			await self._log_key_operation("CREATE", spec.id, False, str(e))
			raise RuntimeError(f"Key creation failed: {e}")
	
	async def retrieve_key(self, key_id: str, user_id: str | None = None, include_material: bool = False) -> Key | None:
		"""Retrieve key by ID"""
		assert self.is_initialized, "Service not initialized" 
		assert key_id, "Key ID required"
		
		# Check permissions
		await self._check_key_permissions(key_id, user_id, "read_key")
		
		key = self.keys.get(key_id)
		if not key:
			return None
		
		# Validate tenant access
		await self._validate_tenant_access(key.spec.tenant_id, user_id)
		
		# Create copy without sensitive material unless specifically requested
		if not include_material:
			key_copy = key.model_copy(deep=True)
			key_copy.key_material = None
			key_copy.hsm_key_id = None
			return key_copy
		
		# Log audit event for material access
		await self._log_audit_event("key_accessed", key_id, "retrieve_key_material", user_id)
		
		return key
	
	async def rotate_key(self, key_id: str, user_id: str | None = None) -> Key:
		"""Rotate cryptographic key"""
		assert self.is_initialized, "Service not initialized"
		assert key_id, "Key ID required"
		
		# Check permissions
		await self._check_key_permissions(key_id, user_id, "rotate_key")
		
		key = self.keys.get(key_id)
		if not key:
			raise ValueError("Key not found")
		
		# Validate tenant access
		await self._validate_tenant_access(key.spec.tenant_id, user_id)
		
		try:
			# Store previous version
			key.previous_versions.append(f"{key_id}_v{len(key.previous_versions)}")
			
			# Generate new cryptographic material
			if key.spec.algorithm in [KeyAlgorithm.AES_128, KeyAlgorithm.AES_256, KeyAlgorithm.CHACHA20_POLY1305]:
				new_material = await self._generate_symmetric_key(key.spec.algorithm, key.spec.key_size)
				new_public_key = None
			else:
				new_material, new_public_key = await self._generate_asymmetric_key_pair(key.spec.algorithm, key.spec.key_size)
			
			# Encrypt new material
			encrypted_material = await self._encrypt_key_material(new_material, key.spec.tenant_id)
			
			# Update key
			key.key_material = encrypted_material
			key.public_key = new_public_key
			key.key_checksum = hashlib.sha256(new_material).hexdigest()
			key.spec.updated_at = datetime.utcnow()
			
			# Schedule next rotation
			if key.spec.policy.auto_rotate:
				key.next_rotation = datetime.utcnow() + timedelta(days=key.spec.policy.rotation_interval_days)
			
			# Log audit event
			await self._log_audit_event("key_rotated", key_id, "rotate_key", user_id, {
				"previous_versions": len(key.previous_versions),
				"next_rotation": key.next_rotation.isoformat() if key.next_rotation else None
			})
			
			await self._log_key_operation("ROTATE", key_id, True, f"Version: {len(key.previous_versions)}")
			
			return key
			
		except Exception as e:
			await self._log_key_operation("ROTATE", key_id, False, str(e))
			raise RuntimeError(f"Key rotation failed: {e}")
	
	async def delete_key(self, key_id: str, user_id: str | None = None, secure_delete: bool = True) -> bool:
		"""Delete cryptographic key"""
		assert self.is_initialized, "Service not initialized"
		assert key_id, "Key ID required"
		
		# Check permissions
		await self._check_key_permissions(key_id, user_id, "delete_key")
		
		key = self.keys.get(key_id)
		if not key:
			return False
		
		# Validate tenant access
		await self._validate_tenant_access(key.spec.tenant_id, user_id)
		
		try:
			if secure_delete:
				# Secure deletion - overwrite key material
				if key.key_material:
					# Overwrite with random data multiple times
					for _ in range(3):
						secrets.token_bytes(len(key.key_material))
				
				# Remove from storage
				del self.keys[key_id]
				if key_id in self.usage_stats:
					del self.usage_stats[key_id]
				
				key.spec.state = KeyState.DESTROYED
			else:
				# Soft deletion - mark as deactivated
				key.spec.state = KeyState.DEACTIVATED
				key.spec.updated_at = datetime.utcnow()
			
			# Log audit event
			await self._log_audit_event("key_deleted", key_id, "delete_key", user_id, {
				"secure_delete": secure_delete,
				"state": key.spec.state.value
			})
			
			await self._log_key_operation("DELETE", key_id, True, f"Secure: {secure_delete}")
			
			return True
			
		except Exception as e:
			await self._log_key_operation("DELETE", key_id, False, str(e))
			raise RuntimeError(f"Key deletion failed: {e}")
	
	async def encrypt_data(self, key_id: str, data: bytes, user_id: str | None = None, 
						  parameters: Dict[str, Any] | None = None) -> bytes:
		"""Encrypt data using specified key"""
		assert self.is_initialized, "Service not initialized"
		assert key_id, "Key ID required"
		assert data, "Data required"
		assert isinstance(data, bytes), "Data must be bytes"
		
		# Check permissions
		await self._check_key_permissions(key_id, user_id, "encrypt_decrypt")
		
		key = self.keys.get(key_id)
		if not key:
			raise ValueError("Key not found")
		
		# Validate key can be used for encryption
		if KeyUsage.ENCRYPT not in key.spec.usage:
			raise ValueError("Key not authorized for encryption")
		
		# Validate tenant access
		await self._validate_tenant_access(key.spec.tenant_id, user_id)
		
		# Create operation record
		operation = KeyOperation(
			key_id=key_id,
			operation_type="encrypt",
			data=data,
			parameters=parameters or {},
			user_id=user_id or "system"
		)
		
		# Detect security threats
		threats = await self._detect_security_threats(operation)
		if threats:
			raise RuntimeError(f"Security threat detected: {threats[0].threat_type}")
		
		try:
			# Decrypt key material for use
			key_material = await self._decrypt_key_material(key.key_material, key.spec.tenant_id)
			
			if key.spec.algorithm in [KeyAlgorithm.AES_128, KeyAlgorithm.AES_256]:
				# AES-GCM encryption
				iv = secrets.token_bytes(16)
				cipher = Cipher(algorithms.AES(key_material), modes.GCM(iv), backend=default_backend())
				encryptor = cipher.encryptor()
				ciphertext = encryptor.update(data) + encryptor.finalize()
				
				# Return IV + tag + ciphertext
				result = iv + encryptor.tag + ciphertext
			else:
				raise ValueError(f"Encryption not supported for algorithm: {key.spec.algorithm}")
			
			# Update usage statistics
			stats = self.usage_stats.get(key_id)
			if stats:
				stats.total_operations += 1
				stats.encrypt_operations += 1
				stats.last_used = datetime.utcnow()
			
			# Log audit event
			await self._log_audit_event("data_encrypted", key_id, "encrypt_data", user_id, {
				"data_size": len(data),
				"algorithm": key.spec.algorithm.value
			})
			
			await self._log_key_operation("ENCRYPT", key_id, True, f"Data size: {len(data)} bytes")
			
			return result
			
		except Exception as e:
			await self._log_key_operation("ENCRYPT", key_id, False, str(e))
			raise RuntimeError(f"Encryption failed: {e}")
	
	async def decrypt_data(self, key_id: str, encrypted_data: bytes, user_id: str | None = None,
						  parameters: Dict[str, Any] | None = None) -> bytes:
		"""Decrypt data using specified key"""
		assert self.is_initialized, "Service not initialized"
		assert key_id, "Key ID required"
		assert encrypted_data, "Encrypted data required"
		assert isinstance(encrypted_data, bytes), "Encrypted data must be bytes"
		
		# Check permissions
		await self._check_key_permissions(key_id, user_id, "encrypt_decrypt")
		
		key = self.keys.get(key_id)
		if not key:
			raise ValueError("Key not found")
		
		# Validate key can be used for decryption
		if KeyUsage.DECRYPT not in key.spec.usage:
			raise ValueError("Key not authorized for decryption")
		
		# Validate tenant access
		await self._validate_tenant_access(key.spec.tenant_id, user_id)
		
		try:
			# Decrypt key material for use
			key_material = await self._decrypt_key_material(key.key_material, key.spec.tenant_id)
			
			if key.spec.algorithm in [KeyAlgorithm.AES_128, KeyAlgorithm.AES_256]:
				# AES-GCM decryption
				iv = encrypted_data[:16]
				tag = encrypted_data[16:32]
				ciphertext = encrypted_data[32:]
				
				cipher = Cipher(algorithms.AES(key_material), modes.GCM(iv, tag), backend=default_backend())
				decryptor = cipher.decryptor()
				result = decryptor.update(ciphertext) + decryptor.finalize()
			else:
				raise ValueError(f"Decryption not supported for algorithm: {key.spec.algorithm}")
			
			# Update usage statistics
			stats = self.usage_stats.get(key_id)
			if stats:
				stats.total_operations += 1
				stats.decrypt_operations += 1
				stats.last_used = datetime.utcnow()
			
			# Log audit event
			await self._log_audit_event("data_decrypted", key_id, "decrypt_data", user_id, {
				"data_size": len(result),
				"algorithm": key.spec.algorithm.value
			})
			
			await self._log_key_operation("DECRYPT", key_id, True, f"Data size: {len(result)} bytes")
			
			return result
			
		except Exception as e:
			await self._log_key_operation("DECRYPT", key_id, False, str(e))
			raise RuntimeError(f"Decryption failed: {e}")
	
	async def list_keys(self, tenant_id: str, user_id: str | None = None, 
					   filters: Dict[str, Any] | None = None) -> List[Key]:
		"""List keys for tenant with optional filters"""
		assert self.is_initialized, "Service not initialized"
		assert tenant_id, "Tenant ID required"
		
		# Validate tenant access
		await self._validate_tenant_access(tenant_id, user_id)
		
		# Filter keys by tenant and user permissions
		result = []
		for key in self.keys.values():
			if key.spec.tenant_id == tenant_id:
				# Check read permissions
				if await self._check_key_permissions(key.spec.id, user_id, "read_key"):
					# Apply filters if provided
					if self._matches_filters(key, filters or {}):
						# Return copy without sensitive material
						key_copy = key.model_copy(deep=True)
						key_copy.key_material = None
						key_copy.hsm_key_id = None
						result.append(key_copy)
		
		return result
	
	def _matches_filters(self, key: Key, filters: Dict[str, Any]) -> bool:
		"""Check if key matches filter criteria"""
		for filter_key, filter_value in filters.items():
			if filter_key == "algorithm" and key.spec.algorithm != filter_value:
				return False
			elif filter_key == "state" and key.spec.state != filter_value:
				return False
			elif filter_key == "usage" and not set(filter_value).issubset(set(key.spec.usage)):
				return False
		return True
	
	async def get_key_usage_stats(self, key_id: str, user_id: str | None = None) -> KeyUsageStats | None:
		"""Get usage statistics for key"""
		assert self.is_initialized, "Service not initialized"
		assert key_id, "Key ID required"
		
		# Check permissions
		await self._check_key_permissions(key_id, user_id, "read_key")
		
		key = self.keys.get(key_id)
		if not key:
			return None
		
		# Validate tenant access
		await self._validate_tenant_access(key.spec.tenant_id, user_id)
		
		return self.usage_stats.get(key_id)
	
	async def get_audit_events(self, tenant_id: str, user_id: str | None = None,
							  filters: Dict[str, Any] | None = None) -> List[AuditEvent]:
		"""Get audit events for tenant"""
		assert self.is_initialized, "Service not initialized"
		assert tenant_id, "Tenant ID required"
		
		# Validate tenant access
		await self._validate_tenant_access(tenant_id, user_id)
		
		# Check audit permissions
		if not await self._check_key_permissions("*", user_id, "view_audit_logs"):
			raise PermissionError("Insufficient permissions for audit logs")
		
		# Filter events by tenant
		result = [event for event in self.audit_events if event.tenant_id == tenant_id]
		
		# Apply additional filters
		if filters:
			result = [event for event in result if self._matches_audit_filters(event, filters)]
		
		return result
	
	def _matches_audit_filters(self, event: AuditEvent, filters: Dict[str, Any]) -> bool:
		"""Check if audit event matches filter criteria"""
		for filter_key, filter_value in filters.items():
			if filter_key == "event_type" and event.event_type != filter_value:
				return False
			elif filter_key == "user_id" and event.user_id != filter_value:
				return False
			elif filter_key == "resource_id" and event.resource_id != filter_value:
				return False
		return True
	
	async def get_service_health(self) -> Dict[str, Any]:
		"""Get service health status"""
		return {
			"status": "healthy" if self.is_initialized else "unhealthy",
			"initialized": self.is_initialized,
			"total_keys": len(self.keys),
			"active_keys": len([k for k in self.keys.values() if k.spec.state == KeyState.ACTIVE]),
			"total_operations": sum(stats.total_operations for stats in self.usage_stats.values()),
			"threats_detected": len(self.threats),
			"audit_events": len(self.audit_events),
			"blockchain_audit_enabled": self.blockchain_audit_logger is not None,
			"iot_edge_enabled": self.iot_device_manager is not None,
			"iot_devices": len(self.iot_device_manager.devices) if self.iot_device_manager else 0,
			"edge_nodes": len(self.iot_device_manager.edge_nodes) if self.iot_device_manager else 0,
			"timestamp": datetime.utcnow()
		}
	
	# Blockchain audit query methods
	async def get_blockchain_audit_trail(self, resource_id: str = None, user_id: str = None,
										start_date: datetime = None, end_date: datetime = None,
										user_requesting: str | None = None) -> List[Dict[str, Any]]:
		"""Get blockchain audit trail with filtering"""
		assert self.is_initialized, "Service not initialized"
		
		# Check permissions for audit access
		if user_requesting:
			await self._check_key_permissions("audit_trail", user_requesting, "read_audit")
		
		if not self.blockchain_audit_logger:
			return []
		
		try:
			from .blockchain_audit import AuditEvent as BlockchainAuditEvent
			
			events = await self.blockchain_audit_logger.get_audit_trail(
				resource_id=resource_id,
				user_id=user_id,
				start_date=start_date,
				end_date=end_date
			)
			
			# Convert to dictionaries for API response
			return [event.to_dict() for event in events]
			
		except Exception as e:
			await self._log_key_operation("BLOCKCHAIN_AUDIT", "query", False, f"Query failed: {str(e)}")
			return []
	
	async def verify_audit_event_integrity(self, event_id: str, user_requesting: str | None = None) -> Dict[str, Any]:
		"""Verify integrity of specific audit event using blockchain proof"""
		assert self.is_initialized, "Service not initialized"
		assert event_id, "Event ID required"
		
		# Check permissions for audit verification
		if user_requesting:
			await self._check_key_permissions("audit_verification", user_requesting, "verify_audit")
		
		if not self.blockchain_audit_logger:
			return {
				'valid': False,
				'error': 'Blockchain audit not enabled'
			}
		
		try:
			verification_result = await self.blockchain_audit_logger.verify_event_integrity(event_id)
			
			# Log the verification attempt
			await self._log_audit_event(
				event_type="compliance_check",
				resource_id=event_id,
				action="verify_audit_event",
				user_id=user_requesting,
				details={
					'verification_result': verification_result['valid'],
					'event_id': event_id
				}
			)
			
			return verification_result
			
		except Exception as e:
			await self._log_key_operation("BLOCKCHAIN_AUDIT", "verify", False, f"Verification failed: {str(e)}")
			return {
				'valid': False,
				'error': str(e)
			}
	
	async def verify_blockchain_integrity(self, user_requesting: str | None = None) -> Dict[str, Any]:
		"""Verify entire blockchain integrity"""
		assert self.is_initialized, "Service not initialized"
		
		# Check permissions for blockchain verification
		if user_requesting:
			await self._check_key_permissions("blockchain_integrity", user_requesting, "verify_blockchain")
		
		if not self.blockchain_audit_logger:
			return {
				'valid': False,
				'error': 'Blockchain audit not enabled'
			}
		
		try:
			integrity_results = await self.blockchain_audit_logger.verify_blockchain_integrity()
			
			# Log the integrity check
			await self._log_audit_event(
				event_type="compliance_check",
				resource_id="blockchain",
				action="verify_integrity",
				user_id=user_requesting,
				details={
					'integrity_valid': integrity_results['valid'],
					'total_blocks': integrity_results['total_blocks'],
					'total_events': integrity_results['total_events'],
					'issues_found': len(integrity_results.get('invalid_blocks', [])) + 
								   len(integrity_results.get('chain_breaks', []))
				}
			)
			
			return integrity_results
			
		except Exception as e:
			await self._log_key_operation("BLOCKCHAIN_AUDIT", "integrity_check", False, f"Check failed: {str(e)}")
			return {
				'valid': False,
				'error': str(e)
			}
	
	async def get_merkle_proof(self, event_id: str, user_requesting: str | None = None) -> Optional[Dict[str, Any]]:
		"""Get Merkle proof for specific audit event"""
		assert self.is_initialized, "Service not initialized"
		assert event_id, "Event ID required"
		
		# Check permissions for proof generation
		if user_requesting:
			await self._check_key_permissions("audit_proof", user_requesting, "generate_proof")
		
		if not self.blockchain_audit_logger:
			return None
		
		try:
			proof = await self.blockchain_audit_logger.get_merkle_proof(event_id)
			
			# Log the proof generation
			await self._log_audit_event(
				event_type="compliance_check",
				resource_id=event_id,
				action="generate_merkle_proof",
				user_id=user_requesting,
				details={
					'proof_generated': proof is not None,
					'event_id': event_id
				}
			)
			
			return proof
			
		except Exception as e:
			await self._log_key_operation("BLOCKCHAIN_AUDIT", "proof", False, f"Proof generation failed: {str(e)}")
			return None
	
	# IoT and Edge Computing API methods
	async def register_iot_device(self, device_spec: Dict[str, Any], user_requesting: str | None = None) -> Dict[str, Any]:
		"""Register new IoT device"""
		assert self.is_initialized, "Service not initialized"
		assert device_spec, "Device specification required"
		
		# Check permissions
		if user_requesting:
			await self._check_key_permissions("iot_device_register", user_requesting, "register_device")
		
		if not self.iot_device_manager:
			raise RuntimeError("IoT device management not enabled")
		
		try:
			from .edge_iot_integration import IoTDevice
			
			device = await self.iot_device_manager.register_device(device_spec, user_requesting)
			
			# Log registration
			await self._log_audit_event(
				event_type="device_registered",
				resource_id=device.device_id,
				action="register_iot_device",
				user_id=user_requesting,
				details={
					'device_type': device.device_type.value,
					'manufacturer': device.manufacturer,
					'model': device.model,
					'security_level': device.security_level.value,
					'edge_location': device.edge_location.value
				}
			)
			
			return device.to_dict()
			
		except Exception as e:
			await self._log_key_operation("IOT_DEVICE", "register", False, f"Registration failed: {str(e)}")
			raise
	
	async def register_edge_node(self, node_spec: Dict[str, Any], user_requesting: str | None = None) -> Dict[str, Any]:
		"""Register edge computing node"""
		assert self.is_initialized, "Service not initialized"
		assert node_spec, "Node specification required"
		
		# Check permissions
		if user_requesting:
			await self._check_key_permissions("edge_node_register", user_requesting, "register_node")
		
		if not self.iot_device_manager:
			raise RuntimeError("IoT device management not enabled")
		
		try:
			node = await self.iot_device_manager.register_edge_node(node_spec, user_requesting)
			
			# Log registration
			await self._log_audit_event(
				event_type="edge_node_registered",
				resource_id=node.node_id,
				action="register_edge_node",
				user_id=user_requesting,
				details={
					'location': node.location.value,
					'cpu_cores': node.cpu_cores,
					'memory_gb': node.memory_gb,
					'max_capacity': node.max_device_capacity
				}
			)
			
			return {
				'node_id': node.node_id,
				'node_name': node.node_name,
				'location': node.location.value,
				'status': node.status,
				'cpu_cores': node.cpu_cores,
				'memory_gb': node.memory_gb,
				'managed_devices': len(node.managed_devices),
				'max_capacity': node.max_device_capacity,
				'created_at': node.created_at.isoformat()
			}
			
		except Exception as e:
			await self._log_key_operation("EDGE_NODE", "register", False, f"Registration failed: {str(e)}")
			raise
	
	async def assign_device_to_edge_node(self, device_id: str, node_id: str, user_requesting: str | None = None):
		"""Assign IoT device to edge node"""
		assert self.is_initialized, "Service not initialized"
		assert device_id, "Device ID required"
		assert node_id, "Node ID required"
		
		# Check permissions
		if user_requesting:
			await self._check_key_permissions("device_assignment", user_requesting, "assign_device")
		
		if not self.iot_device_manager:
			raise RuntimeError("IoT device management not enabled")
		
		try:
			await self.iot_device_manager.assign_device_to_edge_node(device_id, node_id, user_requesting)
			
			# Log assignment
			await self._log_audit_event(
				event_type="device_assigned",
				resource_id=device_id,
				action="assign_to_edge_node",
				user_id=user_requesting,
				details={
					'edge_node_id': node_id
				}
			)
			
		except Exception as e:
			await self._log_key_operation("DEVICE_ASSIGNMENT", "assign", False, f"Assignment failed: {str(e)}")
			raise
	
	async def rotate_iot_device_keys(self, device_id: str, user_requesting: str | None = None) -> Dict[str, str]:
		"""Rotate keys for IoT device"""
		assert self.is_initialized, "Service not initialized"
		assert device_id, "Device ID required"
		
		# Check permissions
		if user_requesting:
			await self._check_key_permissions(device_id, user_requesting, "rotate_keys")
		
		if not self.iot_device_manager:
			raise RuntimeError("IoT device management not enabled")
		
		try:
			new_keys = await self.iot_device_manager.rotate_device_keys(device_id, user_requesting)
			
			# Log key rotation
			await self._log_audit_event(
				event_type="key_rotated",
				resource_id=device_id,
				action="rotate_iot_device_keys",
				user_id=user_requesting,
				details={
					'new_key_count': len(new_keys),
					'algorithms': list(new_keys.keys())
				}
			)
			
			return new_keys
			
		except Exception as e:
			await self._log_key_operation("IOT_KEYS", "rotate", False, f"Rotation failed: {str(e)}")
			raise
	
	async def get_iot_device_status(self, device_id: str, user_requesting: str | None = None) -> Dict[str, Any]:
		"""Get IoT device status"""
		assert self.is_initialized, "Service not initialized"
		assert device_id, "Device ID required"
		
		# Check permissions
		if user_requesting:
			await self._check_key_permissions(device_id, user_requesting, "read_status")
		
		if not self.iot_device_manager:
			raise RuntimeError("IoT device management not enabled")
		
		try:
			status = await self.iot_device_manager.get_device_status(device_id)
			
			# Log status access
			await self._log_audit_event(
				event_type="device_accessed",
				resource_id=device_id,
				action="get_device_status",
				user_id=user_requesting
			)
			
			return status
			
		except Exception as e:
			await self._log_key_operation("IOT_STATUS", "get", False, f"Status retrieval failed: {str(e)}")
			raise
	
	async def get_devices_by_location(self, location: str, user_requesting: str | None = None) -> List[Dict[str, Any]]:
		"""Get all devices at specific edge location"""
		assert self.is_initialized, "Service not initialized"
		assert location, "Location required"
		
		# Check permissions
		if user_requesting:
			await self._check_key_permissions("device_list", user_requesting, "list_devices")
		
		if not self.iot_device_manager:
			return []
		
		try:
			from .edge_iot_integration import EdgeLocation
			
			edge_location = EdgeLocation(location)
			devices = await self.iot_device_manager.get_devices_by_location(edge_location)
			
			# Convert to dictionaries
			return [device.to_dict() for device in devices]
			
		except Exception as e:
			await self._log_key_operation("IOT_DEVICES", "list", False, f"Listing failed: {str(e)}")
			return []
	
	async def get_edge_node_devices(self, node_id: str, user_requesting: str | None = None) -> List[Dict[str, Any]]:
		"""Get all devices managed by edge node"""
		assert self.is_initialized, "Service not initialized"
		assert node_id, "Node ID required"
		
		# Check permissions
		if user_requesting:
			await self._check_key_permissions(node_id, user_requesting, "read_node_devices")
		
		if not self.iot_device_manager:
			return []
		
		try:
			devices = await self.iot_device_manager.get_edge_node_devices(node_id)
			
			# Convert to dictionaries
			return [device.to_dict() for device in devices]
			
		except Exception as e:
			await self._log_key_operation("EDGE_DEVICES", "list", False, f"Listing failed: {str(e)}")
			return []
	
	async def get_iot_security_summary(self, user_requesting: str | None = None) -> Dict[str, Any]:
		"""Get IoT security summary across all devices"""
		assert self.is_initialized, "Service not initialized"
		
		# Check permissions
		if user_requesting:
			await self._check_key_permissions("security_summary", user_requesting, "read_security_summary")
		
		if not self.iot_device_manager:
			return {
				'iot_enabled': False,
				'total_devices': 0
			}
		
		try:
			summary = await self.iot_device_manager.get_security_summary()
			summary['iot_enabled'] = True
			
			# Log summary access
			await self._log_audit_event(
				event_type="compliance_check",
				resource_id="iot_security",
				action="get_security_summary",
				user_id=user_requesting,
				details=summary
			)
			
			return summary
			
		except Exception as e:
			await self._log_key_operation("IOT_SECURITY", "summary", False, f"Summary failed: {str(e)}")
			return {
				'iot_enabled': True,
				'error': str(e)
			}


# Factory function for creating service instances
async def create_key_management_service(config: Dict[str, Any] | None = None) -> KeyManagementService:
	"""Create and initialize key management service"""
	service = KeyManagementService()
	await service.initialize(config or {})
	return service


# Export service classes and factory
__all__ = [
	"ExportApprovalRecord",
	"KeyManagementService",
	"KeyOperationRecord",
	"KeyRotationRecord",
	"KeymAuditEventRecord",
	"KeymService",
	"ManagedKeyRecord",
	"RotationExceptionRecord",
	"create_key_management_service",
]
