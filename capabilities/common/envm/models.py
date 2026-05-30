"""Domain models for APG Environment Management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvironmentDefinition:
	"""Tenant environment with owner, stage, region, policy, and source state."""

	id: str
	tenant_id: str
	name: str
	stage: str
	region: str
	owner: str
	configuration_source: str
	rbac_policy: str
	secret_scope_policy: str
	fingerprint: str
	status: str = "active"
	production_locked: bool = False

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"stage": self.stage,
			"region": self.region,
			"owner": self.owner,
			"configuration_source": self.configuration_source,
			"rbac_policy": self.rbac_policy,
			"secret_scope_policy": self.secret_scope_policy,
			"fingerprint": self.fingerprint,
			"status": self.status,
			"production_locked": self.production_locked,
		}


@dataclass(frozen=True)
class PromotionPath:
	"""Approved promotion path between managed environments."""

	id: str
	tenant_id: str
	source_environment_id: str
	target_environment_id: str
	deployment_link: str
	rollback_environment_id: str
	approval_recorded: bool
	status: str

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"source_environment_id": self.source_environment_id,
			"target_environment_id": self.target_environment_id,
			"deployment_link": self.deployment_link,
			"rollback_environment_id": self.rollback_environment_id,
			"approval_recorded": self.approval_recorded,
			"status": self.status,
		}


@dataclass(frozen=True)
class PromotionRun:
	"""Promotion execution record linked to a governed path."""

	id: str
	tenant_id: str
	promotion_path_id: str
	requested_by: str
	artifact_ref: str
	status: str
	approval_recorded: bool

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"promotion_path_id": self.promotion_path_id,
			"requested_by": self.requested_by,
			"artifact_ref": self.artifact_ref,
			"status": self.status,
			"approval_recorded": self.approval_recorded,
		}


@dataclass(frozen=True)
class DriftReport:
	"""Configuration drift report against declared environment state."""

	id: str
	tenant_id: str
	environment_id: str
	declared_version: str
	observed_version: str
	drift_percent: float
	changed_items: int
	total_items: int
	status: str
	drift_review_recorded: bool
	remediation_action: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"environment_id": self.environment_id,
			"declared_version": self.declared_version,
			"observed_version": self.observed_version,
			"drift_percent": self.drift_percent,
			"changed_items": self.changed_items,
			"total_items": self.total_items,
			"status": self.status,
			"drift_review_recorded": self.drift_review_recorded,
			"remediation_action": self.remediation_action,
		}


@dataclass(frozen=True)
class SecretScope:
	"""Secret access scope for one managed environment."""

	id: str
	tenant_id: str
	environment_id: str
	name: str
	policy_ref: str
	secret_refs: tuple[str, ...]
	access_roles: tuple[str, ...]
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"environment_id": self.environment_id,
			"name": self.name,
			"policy_ref": self.policy_ref,
			"secret_refs": list(self.secret_refs),
			"access_roles": list(self.access_roles),
			"status": self.status,
		}


@dataclass(frozen=True)
class EnvmAgent:
	"""Registered AI agent allowed to assist environment operations."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"status": self.status,
		}


@dataclass(frozen=True)
class EnvmAuditEvent:
	"""Governance event emitted by environment-management operations."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
		}
