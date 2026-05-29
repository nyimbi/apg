"""Service layer for APG Blockchain Ledger Services."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .ledger_engine import LedgerEngine
from .models import (
	KeyCustodyBinding,
	LedgerAuditEvent,
	LedgerNetwork,
	LedgerTransaction,
	SmartContractArtifact,
)


class BclgService:
	"""Tenant ledger registry, custody policy, transactions, contracts, and audit."""

	def __init__(self) -> None:
		self._ledgers: dict[str, LedgerNetwork] = {}
		self._custody_bindings: dict[str, KeyCustodyBinding] = {}
		self._transactions: dict[str, LedgerTransaction] = {}
		self._contracts: dict[str, SmartContractArtifact] = {}
		self._audit_events: dict[str, LedgerAuditEvent] = {}
		self._ledger_heads: dict[str, str] = {}
		self._engine = LedgerEngine()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_ledger(
		self,
		ledger_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		consensus_profile: str,
		network_policy: str,
		participants: list[str] | tuple[str, ...] | None = None,
		fork_monitoring_enabled: bool = True,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_ledger",
			"ledger_owner_assigned": bool(owner),
		})
		self._raise_if_denied(result)
		if not consensus_profile:
			raise PermissionError("consensus_profile_required")
		if not network_policy:
			raise PermissionError("network_policy_required")
		ledger = LedgerNetwork(
			id=ledger_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			consensus_profile=consensus_profile,
			network_policy=network_policy,
			participants=tuple(participants or ()),
			fork_monitoring_enabled=fork_monitoring_enabled,
		)
		self._ledgers[ledger_id] = ledger
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=ledger_id,
			event_type="ledger_registered",
			actor=owner,
			decision=result["decision"],
		)
		return ledger.to_dict()

	def list_ledgers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		ledgers = list(self._ledgers.values())
		if tenant_id is not None:
			ledgers = [item for item in ledgers if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(ledgers, key=lambda item: item.id)]

	def bind_key_custody(
		self,
		binding_id: str,
		tenant_id: str,
		ledger_id: str,
		key_id: str,
		custodian: str,
		rotation_policy: str = "90d",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_ledger(ledger_id, tenant_id)
		if not key_id:
			raise PermissionError("key_id_required")
		if not custodian:
			raise PermissionError("key_custodian_required")
		binding = KeyCustodyBinding(
			id=binding_id,
			tenant_id=tenant_id,
			ledger_id=ledger_id,
			key_id=key_id,
			custodian=custodian,
			rotation_policy=rotation_policy,
		)
		self._custody_bindings[binding_id] = binding
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=binding_id,
			event_type="key_custody_bound",
			actor=custodian,
			decision="allow",
			metadata={"ledger_id": ledger_id, "key_id": key_id},
		)
		return binding.to_dict()

	def list_key_custody(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		bindings = list(self._custody_bindings.values())
		if tenant_id is not None:
			bindings = [item for item in bindings if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(bindings, key=lambda item: item.id)]

	def submit_transaction(
		self,
		transaction_id: str,
		tenant_id: str,
		ledger_id: str,
		from_account: str,
		to_account: str,
		amount: float,
		signature: str,
		key_custody_id: str,
		asset: str = "TOKEN",
		compliance_tags: list[str] | tuple[str, ...] | None = None,
		transaction_review_recorded: bool = False,
		actor: str = "ledger-operator",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_ledger(ledger_id, tenant_id)
		custody = self._custody_bindings.get(key_custody_id)
		custody_bound = bool(custody and custody.tenant_id == tenant_id and custody.ledger_id == ledger_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_transaction",
			"signature_present": bool(signature),
			"key_custody_bound": custody_bound,
			"transaction_value": float(amount),
			"transaction_review_recorded": bool(transaction_review_recorded),
		})
		self._raise_if_denied(result)
		if amount <= 0:
			raise PermissionError("positive_transaction_amount_required")
		review_status = "required" if result["decision"] == "require_review" else "approved"
		status = "pending_review" if result["decision"] == "require_review" else "committed"
		transaction_payload = {
			"id": transaction_id,
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"from_account": from_account,
			"to_account": to_account,
			"amount": float(amount),
			"asset": asset,
			"signature": signature,
			"key_custody_id": key_custody_id,
			"compliance_tags": list(compliance_tags or ()),
		}
		transaction_hash = self._engine.transaction_hash(transaction_payload)
		block_hash = None
		if status == "committed":
			block_hash = self._commit_block(ledger_id, [transaction_hash])
		transaction = LedgerTransaction(
			id=transaction_id,
			tenant_id=tenant_id,
			ledger_id=ledger_id,
			from_account=from_account,
			to_account=to_account,
			amount=float(amount),
			asset=asset,
			signature=signature,
			key_custody_id=key_custody_id,
			compliance_tags=tuple(compliance_tags or ()),
			status=status,
			review_status=review_status,
			transaction_hash=transaction_hash,
			block_hash=block_hash,
		)
		self._transactions[transaction_id] = transaction
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=transaction_id,
			event_type="transaction_submitted",
			actor=actor,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"ledger_id": ledger_id, "amount": float(amount), "asset": asset},
		)
		return transaction.to_dict()

	def approve_transaction(
		self,
		transaction_id: str,
		reviewer: str,
	) -> dict[str, Any]:
		transaction = self._transactions.get(transaction_id)
		if transaction is None:
			raise KeyError(f"unknown ledger transaction: {transaction_id}")
		if transaction.status != "pending_review":
			return transaction.to_dict()
		block_hash = self._commit_block(transaction.ledger_id, [transaction.transaction_hash])
		approved = LedgerTransaction(
			id=transaction.id,
			tenant_id=transaction.tenant_id,
			ledger_id=transaction.ledger_id,
			from_account=transaction.from_account,
			to_account=transaction.to_account,
			amount=transaction.amount,
			asset=transaction.asset,
			signature=transaction.signature,
			key_custody_id=transaction.key_custody_id,
			compliance_tags=transaction.compliance_tags,
			status="committed",
			review_status="approved",
			transaction_hash=transaction.transaction_hash,
			block_hash=block_hash,
		)
		self._transactions[transaction_id] = approved
		self._record_audit(
			tenant_id=approved.tenant_id,
			subject_id=transaction_id,
			event_type="transaction_review_approved",
			actor=reviewer,
			decision="allow",
			metadata={"block_hash": block_hash},
		)
		return approved.to_dict()

	def list_transactions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		transactions = list(self._transactions.values())
		if tenant_id is not None:
			transactions = [item for item in transactions if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(transactions, key=lambda item: item.id)]

	def deploy_contract(
		self,
		contract_id: str,
		tenant_id: str,
		ledger_id: str,
		name: str,
		version: str,
		artifact_hash: str,
		reviewed_by: str,
		rollback_plan: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_ledger(ledger_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy_contract",
			"contract_review_recorded": bool(reviewed_by),
			"key_custody_bound": bool(self._custody_for_ledger(ledger_id, tenant_id)),
		})
		self._raise_if_denied(result)
		if not artifact_hash:
			raise PermissionError("contract_artifact_hash_required")
		if not rollback_plan:
			raise PermissionError("contract_rollback_plan_required")
		payload = {
			"id": contract_id,
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"name": name,
			"version": version,
			"artifact_hash": artifact_hash,
			"reviewed_by": reviewed_by,
			"rollback_plan": rollback_plan,
		}
		contract = SmartContractArtifact(
			id=contract_id,
			tenant_id=tenant_id,
			ledger_id=ledger_id,
			name=name,
			version=version,
			artifact_hash=artifact_hash,
			reviewed_by=reviewed_by,
			rollback_plan=rollback_plan,
			deployment_hash=self._engine.contract_deployment_hash(payload),
		)
		self._contracts[contract_id] = contract
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=contract_id,
			event_type="contract_deployed",
			actor=reviewed_by,
			decision=result["decision"],
			metadata={"ledger_id": ledger_id, "version": version},
		)
		return contract.to_dict()

	def list_contracts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		contracts = list(self._contracts.values())
		if tenant_id is not None:
			contracts = [item for item in contracts if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(contracts, key=lambda item: item.id)]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._audit_events.values())
		if tenant_id is not None:
			events = [item for item in events if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(events, key=lambda item: item.id)]

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing ledger transactions as BCLG records."""
		return self.list_transactions(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that records an auditable ledger note."""
		self._require_tenant(tenant_id)
		metadata = dict(metadata or {})
		event = self._record_audit(
			tenant_id=tenant_id,
			subject_id=record_id,
			event_type=str(metadata.get("event_type") or "ledger_note"),
			actor=str(metadata.get("actor") or "system"),
			decision=status,
			metadata=metadata,
		)
		return event.to_dict()

	def ledger_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		transactions = self.list_transactions(tenant_id)
		contracts = self.list_contracts(tenant_id)
		audit_events = self.list_audit_events(tenant_id)
		return {
			"ledger_count": len(self.list_ledgers(tenant_id)),
			"key_custody_count": len(self.list_key_custody(tenant_id)),
			"transaction_count": len(transactions),
			"pending_review_count": len([item for item in transactions if item["status"] == "pending_review"]),
			"committed_transaction_count": len([item for item in transactions if item["status"] == "committed"]),
			"contract_count": len(contracts),
			"deployed_contract_count": len([item for item in contracts if item["status"] == "deployed"]),
			"audit_event_count": len(audit_events),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_ledger(self, ledger_id: str, tenant_id: str) -> LedgerNetwork:
		ledger = self._ledgers.get(ledger_id)
		if ledger is None or ledger.tenant_id != tenant_id:
			raise KeyError(f"unknown ledger network: {ledger_id}")
		return ledger

	def _custody_for_ledger(self, ledger_id: str, tenant_id: str) -> list[KeyCustodyBinding]:
		return [
			item for item in self._custody_bindings.values()
			if item.ledger_id == ledger_id and item.tenant_id == tenant_id and item.status == "active"
		]

	def _commit_block(self, ledger_id: str, transaction_hashes: list[str]) -> str:
		previous = self._ledger_heads.get(ledger_id)
		block_hash = self._engine.block_hash(ledger_id, transaction_hashes, previous)
		self._ledger_heads[ledger_id] = block_hash
		return block_hash

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> LedgerAuditEvent:
		event_id = f"audit:{len(self._audit_events) + 1:06d}"
		event = LedgerAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)
		self._audit_events[event_id] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "ledger_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "ledger_policy_blocked")
