"""Domain models for APG Blockchain Ledger Services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LedgerNetwork:
	"""Tenant-owned distributed ledger network and consensus policy."""

	id: str
	tenant_id: str
	name: str
	owner: str
	consensus_profile: str
	network_policy: str
	participants: tuple[str, ...] = ()
	fork_monitoring_enabled: bool = True
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"consensus_profile": self.consensus_profile,
			"network_policy": self.network_policy,
			"participants": list(self.participants),
			"fork_monitoring_enabled": self.fork_monitoring_enabled,
			"status": self.status,
		}


@dataclass(frozen=True)
class KeyCustodyBinding:
	"""Managed key-custody binding required before ledger mutation."""

	id: str
	tenant_id: str
	ledger_id: str
	key_id: str
	custodian: str
	rotation_policy: str = "90d"
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ledger_id": self.ledger_id,
			"key_id": self.key_id,
			"custodian": self.custodian,
			"rotation_policy": self.rotation_policy,
			"status": self.status,
		}


@dataclass(frozen=True)
class LedgerTransaction:
	"""Signed ledger transaction with deterministic hash and review state."""

	id: str
	tenant_id: str
	ledger_id: str
	from_account: str
	to_account: str
	amount: float
	asset: str
	signature: str
	key_custody_id: str
	compliance_tags: tuple[str, ...] = ()
	submitted_by: str = "ledger-operator"
	review_id: str | None = None
	reviewer: str | None = None
	review_notes: str | None = None
	status: str = "committed"
	review_status: str = "approved"
	transaction_hash: str = ""
	block_hash: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ledger_id": self.ledger_id,
			"from_account": self.from_account,
			"to_account": self.to_account,
			"amount": self.amount,
			"asset": self.asset,
			"signature": self.signature,
			"key_custody_id": self.key_custody_id,
			"compliance_tags": list(self.compliance_tags),
			"submitted_by": self.submitted_by,
			"review_id": self.review_id,
			"reviewer": self.reviewer,
			"review_notes": self.review_notes,
			"status": self.status,
			"review_status": self.review_status,
			"transaction_hash": self.transaction_hash,
			"block_hash": self.block_hash,
		}


@dataclass(frozen=True)
class TransactionReviewApproval:
	"""Independent review evidence for a high-value ledger transaction."""

	id: str
	tenant_id: str
	transaction_id: str
	requested_by: str
	justification: str
	decision: str = "pending"
	reviewer: str | None = None
	notes: str | None = None
	status: str = "pending"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"transaction_id": self.transaction_id,
			"requested_by": self.requested_by,
			"justification": self.justification,
			"decision": self.decision,
			"reviewer": self.reviewer,
			"notes": self.notes,
			"status": self.status,
		}


@dataclass(frozen=True)
class SmartContractArtifact:
	"""Reviewed smart contract artifact and deployment governance record."""

	id: str
	tenant_id: str
	ledger_id: str
	name: str
	version: str
	artifact_hash: str
	reviewed_by: str
	rollback_plan: str
	approval_id: str | None = None
	status: str = "deployed"
	deployment_hash: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ledger_id": self.ledger_id,
			"name": self.name,
			"version": self.version,
			"artifact_hash": self.artifact_hash,
			"reviewed_by": self.reviewed_by,
			"rollback_plan": self.rollback_plan,
			"approval_id": self.approval_id,
			"status": self.status,
			"deployment_hash": self.deployment_hash,
		}


@dataclass(frozen=True)
class ContractDeploymentApproval:
	"""Independent review evidence for smart contract deployment."""

	id: str
	tenant_id: str
	ledger_id: str
	name: str
	version: str
	artifact_hash: str
	requested_by: str
	rollback_plan: str
	decision: str = "pending"
	reviewer: str | None = None
	notes: str | None = None
	status: str = "pending"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ledger_id": self.ledger_id,
			"name": self.name,
			"version": self.version,
			"artifact_hash": self.artifact_hash,
			"requested_by": self.requested_by,
			"rollback_plan": self.rollback_plan,
			"decision": self.decision,
			"reviewer": self.reviewer,
			"notes": self.notes,
			"status": self.status,
		}


@dataclass(frozen=True)
class LedgerAuditEvent:
	"""Governance event emitted by ledger, transaction, and contract actions."""

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
