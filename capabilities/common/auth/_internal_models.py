"""
Dependency-light domain models for APG Authentication and RBAC.

These are frozen dataclasses with to_dict() helpers — no external dependencies.
The Pydantic v2 models for REST API surfaces live in models.py.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuthIdentity:
	"""Tenant identity with authentication posture and privacy budget."""
	id: str
	tenant_id: str
	email: str
	display_name: str
	status: str = "active"
	tenant_memberships: tuple[str, ...] = field(default_factory=tuple)
	mfa_enabled: bool = False
	behavioral_trust_score: float = 1.0
	biometric_enrolled: bool = False
	quantum_key_registered: bool = False
	privacy_budget: float = 1.0
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":                     self.id,
			"tenant_id":              self.tenant_id,
			"email":                  self.email,
			"display_name":           self.display_name,
			"status":                 self.status,
			"tenant_memberships":     list(self.tenant_memberships),
			"mfa_enabled":            self.mfa_enabled,
			"behavioral_trust_score": self.behavioral_trust_score,
			"biometric_enrolled":     self.biometric_enrolled,
			"quantum_key_registered": self.quantum_key_registered,
			"privacy_budget":         self.privacy_budget,
			"metadata":               dict(self.metadata),
		}


@dataclass(frozen=True)
class AuthRole:
	"""Tenant role definition with permission and approval posture."""
	id: str
	tenant_id: str
	name: str
	permissions: tuple[str, ...] = field(default_factory=tuple)
	tier: str = "standard"
	approval_recorded: bool = False
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":                self.id,
			"tenant_id":         self.tenant_id,
			"name":              self.name,
			"permissions":       list(self.permissions),
			"tier":              self.tier,
			"approval_recorded": self.approval_recorded,
			"status":            self.status,
		}


@dataclass(frozen=True)
class AuthRoleAssignmentApproval:
	"""Independent approval evidence for privileged role assignment."""
	id: str
	tenant_id: str
	user_id: str
	role_id: str
	requested_by: str
	justification: str
	decision: str = "pending"
	reviewer: str | None = None
	notes: str | None = None
	status: str = "pending"
	policy_decision: str = "require_review"
	matched_rules: tuple[str, ...] = field(default_factory=tuple)
	review_reasons: tuple[str, ...] = field(default_factory=tuple)
	review_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":               self.id,
			"tenant_id":        self.tenant_id,
			"user_id":          self.user_id,
			"role_id":          self.role_id,
			"requested_by":     self.requested_by,
			"justification":    self.justification,
			"decision":         self.decision,
			"reviewer":         self.reviewer,
			"notes":            self.notes,
			"status":           self.status,
			"policy_decision":  self.policy_decision,
			"matched_rules":    list(self.matched_rules),
			"review_reasons":   list(self.review_reasons),
			"review_evidence":  dict(self.review_evidence),
		}


@dataclass(frozen=True)
class AuthRoleAssignment:
	"""Assignment of a role to a tenant identity."""
	id: str
	tenant_id: str
	user_id: str
	role_id: str
	assigned_by: str
	approval_id: str | None = None
	approval_recorded: bool = False
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":                self.id,
			"tenant_id":         self.tenant_id,
			"user_id":           self.user_id,
			"role_id":           self.role_id,
			"assigned_by":       self.assigned_by,
			"approval_id":       self.approval_id,
			"approval_recorded": self.approval_recorded,
			"status":            self.status,
		}


@dataclass(frozen=True)
class AuthSession:
	"""Authentication session with risk, federation, and MFA evidence."""
	id: str
	tenant_id: str
	user_id: str
	device_id: str
	auth_source: str = "local"
	risk_level: str = "low"
	mfa_verified: bool = False
	step_up_completed: bool = False
	issuer_trusted: bool = True
	status: str = "active"
	trust_score: float = 1.0
	required_actions: tuple[str, ...] = field(default_factory=tuple)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":                self.id,
			"tenant_id":         self.tenant_id,
			"user_id":           self.user_id,
			"device_id":         self.device_id,
			"auth_source":       self.auth_source,
			"risk_level":        self.risk_level,
			"mfa_verified":      self.mfa_verified,
			"step_up_completed": self.step_up_completed,
			"issuer_trusted":    self.issuer_trusted,
			"status":            self.status,
			"trust_score":       self.trust_score,
			"required_actions":  list(self.required_actions),
		}


@dataclass(frozen=True)
class AuthAccessDecision:
	"""Deterministic authorization decision emitted by AUTH."""
	id: str
	tenant_id: str
	user_id: str
	permission: str
	decision: str
	matched_rules: tuple[str, ...] = field(default_factory=tuple)
	reasons: tuple[str, ...] = field(default_factory=tuple)
	session_id: str | None = None
	role_ids: tuple[str, ...] = field(default_factory=tuple)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":            self.id,
			"tenant_id":     self.tenant_id,
			"user_id":       self.user_id,
			"permission":    self.permission,
			"decision":      self.decision,
			"matched_rules": list(self.matched_rules),
			"reasons":       list(self.reasons),
			"session_id":    self.session_id,
			"role_ids":      list(self.role_ids),
		}


@dataclass(frozen=True)
class AuthPrivacyQuery:
	"""Privacy-preserving analytics request and budget decision."""
	id: str
	tenant_id: str
	user_id: str
	query_type: str
	epsilon_cost: float
	status: str
	remaining_budget: float
	approval_recorded: bool = False
	reasons: tuple[str, ...] = field(default_factory=tuple)
	approval_id: str | None = None
	policy_decision: str = "allow"
	matched_rules: tuple[str, ...] = field(default_factory=tuple)
	review_reasons: tuple[str, ...] = field(default_factory=tuple)
	review_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":                self.id,
			"tenant_id":         self.tenant_id,
			"user_id":           self.user_id,
			"query_type":        self.query_type,
			"epsilon_cost":      self.epsilon_cost,
			"status":            self.status,
			"remaining_budget":  self.remaining_budget,
			"approval_recorded": self.approval_recorded,
			"reasons":           list(self.reasons),
			"approval_id":       self.approval_id,
			"policy_decision":   self.policy_decision,
			"matched_rules":     list(self.matched_rules),
			"review_reasons":    list(self.review_reasons),
			"review_evidence":   dict(self.review_evidence),
		}


@dataclass(frozen=True)
class AuthPrivacyBudgetApproval:
	"""Independent approval evidence for privacy-budget exhaustion."""
	id: str
	tenant_id: str
	user_id: str
	query_type: str
	epsilon_cost: float
	requested_by: str
	justification: str
	decision: str = "pending"
	reviewer: str | None = None
	notes: str | None = None
	status: str = "pending"
	policy_decision: str = "require_review"
	matched_rules: tuple[str, ...] = field(default_factory=tuple)
	review_reasons: tuple[str, ...] = field(default_factory=tuple)
	review_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":               self.id,
			"tenant_id":        self.tenant_id,
			"user_id":          self.user_id,
			"query_type":       self.query_type,
			"epsilon_cost":     self.epsilon_cost,
			"requested_by":     self.requested_by,
			"justification":    self.justification,
			"decision":         self.decision,
			"reviewer":         self.reviewer,
			"notes":            self.notes,
			"status":           self.status,
			"policy_decision":  self.policy_decision,
			"matched_rules":    list(self.matched_rules),
			"review_reasons":   list(self.review_reasons),
			"review_evidence":  dict(self.review_evidence),
		}


@dataclass(frozen=True)
class AuthAuditEvent:
	"""Governance event emitted by identity, session, role, and privacy actions."""
	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = field(default_factory=tuple)
	metadata: dict[str, Any] = field(default_factory=dict)
	policy_decision: str = "allow"
	matched_rules: tuple[str, ...] = field(default_factory=tuple)
	review_reasons: tuple[str, ...] = field(default_factory=tuple)
	review_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":               self.id,
			"tenant_id":        self.tenant_id,
			"subject_id":       self.subject_id,
			"event_type":       self.event_type,
			"actor":            self.actor,
			"decision":         self.decision,
			"reasons":          list(self.reasons),
			"metadata":         dict(self.metadata),
			"policy_decision":  self.policy_decision,
			"matched_rules":    list(self.matched_rules),
			"review_reasons":   list(self.review_reasons),
			"review_evidence":  dict(self.review_evidence),
		}


@dataclass(frozen=True)
class AuthSecurityAgent:
	"""Governed AI security agent registration for AUTH workflows."""
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	owner: str = ""
	purpose: str = ""
	human_approval_required: bool = True
	policy_ref: str | None = None
	status: str = "active"
	policy_decision: str = "allow"
	matched_rules: tuple[str, ...] = field(default_factory=tuple)
	review_reasons: tuple[str, ...] = field(default_factory=tuple)
	review_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":                      self.id,
			"tenant_id":               self.tenant_id,
			"name":                    self.name,
			"runtime":                 self.runtime,
			"role":                    self.role,
			"scope":                   self.scope,
			"registered":              self.registered,
			"contribution_disclosed":  self.contribution_disclosed,
			"owner":                   self.owner,
			"purpose":                 self.purpose,
			"human_approval_required": self.human_approval_required,
			"policy_ref":              self.policy_ref,
			"status":                  self.status,
			"policy_decision":         self.policy_decision,
			"matched_rules":           list(self.matched_rules),
			"review_reasons":          list(self.review_reasons),
			"review_evidence":         dict(self.review_evidence),
		}


@dataclass(frozen=True)
class AuthBatchMutationEvidence:
	"""Bytewax batch AUTH mutation validation evidence."""
	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	status: str = "accepted"
	processor: str = "bytewax"
	policy_decision: str = "allow"
	matched_rules: tuple[str, ...] = field(default_factory=tuple)
	review_reasons: tuple[str, ...] = field(default_factory=tuple)
	review_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id":               self.id,
			"tenant_id":        self.tenant_id,
			"event_stream":     self.event_stream,
			"mutation_count":   self.mutation_count,
			"status":           self.status,
			"processor":        self.processor,
			"policy_decision":  self.policy_decision,
			"matched_rules":    list(self.matched_rules),
			"review_reasons":   list(self.review_reasons),
			"review_evidence":  dict(self.review_evidence),
		}
