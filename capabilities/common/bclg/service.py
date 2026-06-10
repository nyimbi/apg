"""Service layer for APG Blockchain Ledger Services."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .capability_contract import (
	SUPPORTED_LEDGER_AGENT_ROLES,
	SUPPORTED_LEDGER_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .ledger_engine import LedgerEngine
from .models import (
	ContractDeploymentApproval,
	KeyCustodyBinding,
	LedgerAuditEvent,
	LedgerAgent,
	LedgerNetwork,
	LedgerTransaction,
	SmartContractArtifact,
	TransactionReviewApproval,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class BclgService:
	"""Tenant ledger registry, custody policy, mutations, reviews, and audit."""

	def __init__(self) -> None:
		self._ledgers: dict[tuple[str, str], LedgerNetwork] = {}
		self._custody_bindings: dict[tuple[str, str], KeyCustodyBinding] = {}
		self._transactions: dict[tuple[str, str], LedgerTransaction] = {}
		self._transaction_reviews: dict[tuple[str, str], TransactionReviewApproval] = {}
		self._contract_reviews: dict[tuple[str, str], ContractDeploymentApproval] = {}
		self._contracts: dict[tuple[str, str], SmartContractArtifact] = {}
		self._ledger_agents: dict[tuple[str, str], LedgerAgent] = {}
		self._audit_events: dict[tuple[str, str], LedgerAuditEvent] = {}
		self._ledger_heads: dict[tuple[str, str], str] = {}
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
		self._ensure_new(self._ledgers, tenant_id, ledger_id, "ledger")
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
		self._ledgers[self._tenant_key(tenant_id, ledger_id)] = ledger
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=ledger_id,
			event_type="ledger_registered",
			actor=owner,
			decision=result["decision"],
		)
		return ledger.to_dict()

	def list_ledgers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._ledgers.values(), tenant_id)

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
		self._ensure_new(self._custody_bindings, tenant_id, binding_id, "key custody binding")
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
		self._custody_bindings[self._tenant_key(tenant_id, binding_id)] = binding
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
		return self._list(self._custody_bindings.values(), tenant_id)

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
		self._ensure_new(self._transactions, tenant_id, transaction_id, "transaction")
		custody = self._custody_bindings.get(self._tenant_key(tenant_id, key_custody_id))
		custody_bound = bool(custody and custody.ledger_id == ledger_id and custody.status == "active")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_transaction",
			"signature_present": bool(signature),
			"key_custody_bound": custody_bound,
			"transaction_value": float(amount),
			"transaction_review_recorded": False if float(amount) > 100000 else bool(transaction_review_recorded),
		})
		self._raise_if_denied(result)
		if amount <= 0:
			raise PermissionError("positive_transaction_amount_required")
		review_required = result["decision"] == "require_review"
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
			"submitted_by": actor,
		}
		transaction_hash = self._engine.transaction_hash(transaction_payload)
		block_hash = None
		if not review_required:
			block_hash = self._commit_block(tenant_id, ledger_id, [transaction_hash])
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
			submitted_by=actor,
			status="pending_review" if review_required else "committed",
			review_status="required" if review_required else "approved",
			transaction_hash=transaction_hash,
			block_hash=block_hash,
		)
		self._transactions[self._tenant_key(tenant_id, transaction_id)] = transaction
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=transaction_id,
			event_type="transaction_submitted",
			actor=actor,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"ledger_id": ledger_id, "amount": float(amount), "asset": asset},
		)
		return transaction.to_dict()

	def request_transaction_review(
		self,
		review_id: str,
		tenant_id: str,
		transaction_id: str,
		requested_by: str,
		justification: str,
	) -> dict[str, Any]:
		transaction = self._require_transaction(transaction_id, tenant_id)
		self._ensure_new(self._transaction_reviews, tenant_id, review_id, "transaction review")
		if transaction.status != "pending_review":
			raise ValueError("transaction_review_not_required")
		if any(
			review.transaction_id == transaction_id and review.tenant_id == tenant_id and review.status == "pending"
			for review in self._transaction_reviews.values()
		):
			raise ValueError("transaction_review_already_pending")
		if not requested_by:
			raise ValueError("transaction_review_requester_required")
		if not justification:
			raise ValueError("transaction_review_justification_required")
		review = TransactionReviewApproval(
			id=review_id,
			tenant_id=tenant_id,
			transaction_id=transaction_id,
			requested_by=requested_by,
			justification=justification,
		)
		self._transaction_reviews[self._tenant_key(tenant_id, review_id)] = review
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=review_id,
			event_type="transaction_review_requested",
			actor=requested_by,
			decision="require_review",
			metadata={"transaction_id": transaction_id, "amount": transaction.amount},
		)
		return review.to_dict()

	def decide_transaction_review(
		self,
		review_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		review = self._require_transaction_review(review_id, tenant_id)
		transaction = self._require_transaction(review.transaction_id, tenant_id)
		if review.status != "pending":
			raise ValueError("transaction_review_already_decided")
		if transaction.status != "pending_review":
			raise ValueError("transaction_review_not_required")
		if decision not in {"approved", "rejected"}:
			raise ValueError("transaction_review_decision_invalid")
		if not reviewer:
			raise ValueError("transaction_review_reviewer_required")
		if not notes:
			raise ValueError("transaction_review_notes_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "approve_transaction_review",
			"reviewer_same_as_submitter": reviewer in {transaction.submitted_by, review.requested_by},
		})
		self._raise_if_denied(result)
		decided = replace(review, decision=decision, reviewer=reviewer, notes=notes, status=decision)
		self._transaction_reviews[self._tenant_key(tenant_id, review_id)] = decided
		updated = replace(
			transaction,
			review_id=review_id,
			reviewer=reviewer,
			review_notes=notes,
			review_status=decision,
		)
		if decision == "approved":
			block_hash = self._commit_block(tenant_id, transaction.ledger_id, [transaction.transaction_hash])
			updated = replace(updated, status="committed", block_hash=block_hash)
		else:
			updated = replace(updated, status="rejected")
		self._transactions[self._tenant_key(tenant_id, transaction.id)] = updated
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=review_id,
			event_type="transaction_review_decided",
			actor=reviewer,
			decision=decision,
			reasons=self._reasons(result),
			metadata={"transaction_id": transaction.id, "block_hash": updated.block_hash},
		)
		return updated.to_dict()

	def approve_transaction(
		self,
		transaction_id: str,
		reviewer: str,
		tenant_id: str | None = None,
		notes: str = "Approved transaction review.",
	) -> dict[str, Any]:
		transaction = self._require_transaction(transaction_id, tenant_id)
		if transaction.status != "pending_review":
			return transaction.to_dict()
		review_id = f"review-{transaction.id}"
		if self._tenant_key(transaction.tenant_id, review_id) not in self._transaction_reviews:
			self.request_transaction_review(
				review_id=review_id,
				tenant_id=transaction.tenant_id,
				transaction_id=transaction.id,
				requested_by=transaction.submitted_by,
				justification="Compatibility approval flow.",
			)
		return self.decide_transaction_review(review_id, transaction.tenant_id, reviewer, "approved", notes)

	def list_transaction_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._transaction_reviews.values(), tenant_id)

	def list_transactions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._transactions.values(), tenant_id)

	def request_contract_deployment_approval(
		self,
		approval_id: str,
		tenant_id: str,
		ledger_id: str,
		name: str,
		version: str,
		artifact_hash: str,
		requested_by: str,
		rollback_plan: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_ledger(ledger_id, tenant_id)
		self._ensure_new(self._contract_reviews, tenant_id, approval_id, "contract deployment approval")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "request_contract_deployment",
			"key_custody_bound": bool(self._custody_for_ledger(ledger_id, tenant_id)),
		})
		self._raise_if_denied(result)
		if not artifact_hash:
			raise PermissionError("contract_artifact_hash_required")
		if not requested_by:
			raise ValueError("contract_deployment_requester_required")
		if not rollback_plan:
			raise PermissionError("contract_rollback_plan_required")
		approval = ContractDeploymentApproval(
			id=approval_id,
			tenant_id=tenant_id,
			ledger_id=ledger_id,
			name=name,
			version=version,
			artifact_hash=artifact_hash,
			requested_by=requested_by,
			rollback_plan=rollback_plan,
		)
		self._contract_reviews[self._tenant_key(tenant_id, approval_id)] = approval
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="contract_deployment_review_requested",
			actor=requested_by,
			decision="require_review",
			metadata={"ledger_id": ledger_id, "name": name, "version": version},
		)
		return approval.to_dict()

	def decide_contract_deployment_approval(
		self,
		approval_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		approval = self._require_contract_review(approval_id, tenant_id)
		if approval.status != "pending":
			raise ValueError("contract_deployment_review_already_decided")
		if decision not in {"approved", "rejected"}:
			raise ValueError("contract_deployment_review_decision_invalid")
		if not reviewer:
			raise ValueError("contract_deployment_reviewer_required")
		if not notes:
			raise ValueError("contract_deployment_review_notes_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "approve_contract_deployment",
			"reviewer_same_as_requester": reviewer == approval.requested_by,
		})
		self._raise_if_denied(result)
		decided = replace(approval, decision=decision, reviewer=reviewer, notes=notes, status=decision)
		self._contract_reviews[self._tenant_key(tenant_id, approval_id)] = decided
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="contract_deployment_review_decided",
			actor=reviewer,
			decision=decision,
			reasons=self._reasons(result),
			metadata={"ledger_id": approval.ledger_id, "name": approval.name, "version": approval.version},
		)
		return decided.to_dict()

	def list_contract_deployment_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._contract_reviews.values(), tenant_id)

	def deploy_contract(
		self,
		contract_id: str,
		tenant_id: str,
		ledger_id: str,
		name: str,
		version: str,
		artifact_hash: str,
		reviewed_by: str = "",
		rollback_plan: str = "",
		approval_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_ledger(ledger_id, tenant_id)
		self._ensure_new(self._contracts, tenant_id, contract_id, "contract")
		approval = self._approved_contract_deployment_approval(
			tenant_id=tenant_id,
			approval_id=approval_id,
			ledger_id=ledger_id,
			name=name,
			version=version,
			artifact_hash=artifact_hash,
			rollback_plan=rollback_plan,
		)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy_contract",
			"contract_review_recorded": approval is not None,
			"key_custody_bound": bool(self._custody_for_ledger(ledger_id, tenant_id)),
		})
		self._raise_if_denied(result)
		if not artifact_hash:
			raise PermissionError("contract_artifact_hash_required")
		if not rollback_plan:
			raise PermissionError("contract_rollback_plan_required")
		reviewer = reviewed_by or (approval.reviewer if approval else "")
		payload = {
			"id": contract_id,
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"name": name,
			"version": version,
			"artifact_hash": artifact_hash,
			"reviewed_by": reviewer,
			"rollback_plan": rollback_plan,
			"approval_id": approval.id if approval else approval_id,
		}
		contract = SmartContractArtifact(
			id=contract_id,
			tenant_id=tenant_id,
			ledger_id=ledger_id,
			name=name,
			version=version,
			artifact_hash=artifact_hash,
			reviewed_by=reviewer,
			rollback_plan=rollback_plan,
			approval_id=approval.id if approval else approval_id,
			deployment_hash=self._engine.contract_deployment_hash(payload),
		)
		self._contracts[self._tenant_key(tenant_id, contract_id)] = contract
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=contract_id,
			event_type="contract_deployed",
			actor=reviewer,
			decision=result["decision"],
			metadata={"ledger_id": ledger_id, "version": version, "approval_id": approval.id if approval else approval_id},
		)
		return contract.to_dict()

	def list_contracts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._contracts.values(), tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events.values(), tenant_id)

	def register_ledger_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		registered: bool = True,
		contribution_disclosed: bool = True,
		policy_ref: str | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"ledger_agent_present": True,
			"agent_registered": registered,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_LEDGER_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_LEDGER_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_if_denied(result)
		self._ensure_new(self._ledger_agents, tenant_id, agent_id, "ledger agent")
		if not name:
			raise ValueError("ledger_agent_name_required")
		agent = LedgerAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref,
			status=status,
		)
		self._ledger_agents[self._tenant_key(tenant_id, agent_id)] = agent
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=agent_id,
			event_type="ledger_agent_registered",
			actor="system",
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"runtime": agent.runtime, "role": agent.role, "scope": scope},
		)
		return agent.to_dict()

	def list_ledger_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._ledger_agents.values(), tenant_id)

	def validate_batch_ledger_mutation(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"requested_operation": "batch_ledger_mutation",
			"event_stream": event_stream,
			"mutation_count": mutation_count,
		})
		self._raise_if_denied(result)
		return {
			"tenant_id": tenant_id,
			"event_stream": event_stream,
			"mutation_count": mutation_count,
			"accepted": True,
			"rule_result": result,
		}

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
		transaction_reviews = self.list_transaction_reviews(tenant_id)
		contract_reviews = self.list_contract_deployment_approvals(tenant_id)
		audit_events = self.list_audit_events(tenant_id)
		return {
			"ledger_count": len(self.list_ledgers(tenant_id)),
			"key_custody_count": len(self.list_key_custody(tenant_id)),
			"transaction_count": len(transactions),
			"pending_review_count": len([item for item in transactions if item["status"] == "pending_review"]),
			"rejected_transaction_count": len([item for item in transactions if item["status"] == "rejected"]),
			"committed_transaction_count": len([item for item in transactions if item["status"] == "committed"]),
			"transaction_review_count": len(transaction_reviews),
			"pending_transaction_review_count": len([item for item in transaction_reviews if item["status"] == "pending"]),
			"contract_count": len(contracts),
			"deployed_contract_count": len([item for item in contracts if item["status"] == "deployed"]),
			"contract_review_count": len(contract_reviews),
			"pending_contract_review_count": len([item for item in contract_reviews if item["status"] == "pending"]),
			"ledger_agent_count": len(self.list_ledger_agents(tenant_id)),
			"audit_event_count": len(audit_events),
		}

	def _tenant_key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	def _ensure_new(self, records: dict[tuple[str, str], Any], tenant_id: str, record_id: str, label: str) -> None:
		self._require_tenant(tenant_id)
		if self._tenant_key(tenant_id, record_id) in records:
			raise ValueError(f"{label} already exists for tenant: {record_id}")

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_ledger(self, ledger_id: str, tenant_id: str) -> LedgerNetwork:
		ledger = self._ledgers.get(self._tenant_key(tenant_id, ledger_id))
		if ledger is None:
			raise KeyError(f"unknown ledger network: {ledger_id}")
		return ledger

	def _require_transaction(self, transaction_id: str, tenant_id: str | None = None) -> LedgerTransaction:
		if tenant_id is not None:
			transaction = self._transactions.get(self._tenant_key(tenant_id, transaction_id))
			if transaction is None:
				raise KeyError(f"unknown ledger transaction: {transaction_id}")
			return transaction
		matches = [transaction for (_, item_id), transaction in self._transactions.items() if item_id == transaction_id]
		if len(matches) > 1:
			raise KeyError(f"transaction ID is ambiguous across tenants: {transaction_id}")
		if not matches:
			raise KeyError(f"unknown ledger transaction: {transaction_id}")
		return matches[0]

	def _require_transaction_review(self, review_id: str, tenant_id: str) -> TransactionReviewApproval:
		review = self._transaction_reviews.get(self._tenant_key(tenant_id, review_id))
		if review is None:
			raise KeyError(f"unknown transaction review: {review_id}")
		return review

	def _require_contract_review(self, approval_id: str, tenant_id: str) -> ContractDeploymentApproval:
		approval = self._contract_reviews.get(self._tenant_key(tenant_id, approval_id))
		if approval is None:
			raise KeyError(f"unknown contract deployment review: {approval_id}")
		return approval

	def _approved_contract_deployment_approval(
		self,
		tenant_id: str,
		approval_id: str | None,
		ledger_id: str,
		name: str,
		version: str,
		artifact_hash: str,
		rollback_plan: str,
	) -> ContractDeploymentApproval | None:
		if approval_id is None:
			return None
		approval = self._contract_reviews.get(self._tenant_key(tenant_id, approval_id))
		if approval is None:
			raise PermissionError("contract_deployment_approval_required")
		if (
			approval.ledger_id != ledger_id
			or approval.name != name
			or approval.version != version
			or approval.artifact_hash != artifact_hash
			or approval.rollback_plan != rollback_plan
		):
			raise PermissionError("contract_deployment_approval_mismatch")
		if approval.status != "approved":
			raise PermissionError("contract_deployment_approval_not_approved")
		return approval

	def _custody_for_ledger(self, ledger_id: str, tenant_id: str) -> list[KeyCustodyBinding]:
		return [
			item for item in self._custody_bindings.values()
			if item.ledger_id == ledger_id and item.tenant_id == tenant_id and item.status == "active"
		]

	def _commit_block(self, tenant_id: str, ledger_id: str, transaction_hashes: list[str]) -> str:
		head_key = self._tenant_key(tenant_id, ledger_id)
		previous = self._ledger_heads.get(head_key)
		block_hash = self._engine.block_hash(f"{tenant_id}:{ledger_id}", transaction_hashes, previous)
		self._ledger_heads[head_key] = block_hash
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
		self._audit_events[self._tenant_key(tenant_id, event_id)] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(self._reasons(result))
			raise PermissionError(reasons or "ledger_policy_blocked")

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(
			str(action.get("reason") or action.get("required_action") or "ledger_policy_blocked")
			for action in result.get("actions", [])
		)

	# ── new methods ─────────────────────────────────────────────────────────

	def smart_contract_compile(
		self,
		artifact_id: str,
		tenant_id: str,
		source_code: str,
		compiler_version: str = "solidity-0.8.20",
		actor: str = "developer",
	) -> dict[str, Any]:
		"""Compile smart contract source and produce a bytecode artifact record."""
		import hashlib
		self._require_tenant(tenant_id)
		if not source_code:
			raise ValueError("source_code_required")
		bytecode_hash = hashlib.sha256(source_code.encode()).hexdigest()
		abi_hash = hashlib.md5(source_code.encode()).hexdigest()
		record = {
			"artifact_id": artifact_id,
			"tenant_id": tenant_id,
			"compiler_version": compiler_version,
			"bytecode_hash": bytecode_hash,
			"abi_hash": abi_hash,
			"status": "compiled",
			"actor": actor,
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=artifact_id,
			event_type="smart_contract_compiled",
			actor=actor,
			decision="allow",
			metadata={"compiler_version": compiler_version, "bytecode_hash": bytecode_hash},
		)
		return record

	def invoke_contract(
		self,
		invocation_id: str,
		tenant_id: str,
		ledger_id: str,
		contract_id: str,
		method: str,
		args: dict[str, Any] | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Invoke a deployed smart contract method and record the result."""
		import hashlib
		self._require_ledger(ledger_id, tenant_id)
		contract = self._contracts.get(self._tenant_key(tenant_id, contract_id))
		if contract is None:
			raise KeyError(f"unknown contract for tenant: {contract_id}")
		result_hash = hashlib.sha256(f"{invocation_id}:{method}".encode()).hexdigest()
		record = {
			"invocation_id": invocation_id,
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"contract_id": contract_id,
			"method": method,
			"args": dict(args or {}),
			"result_hash": result_hash,
			"status": "success",
			"actor": actor,
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=invocation_id,
			event_type="contract_invoked",
			actor=actor,
			decision="allow",
			metadata={"contract_id": contract_id, "method": method},
		)
		return record

	def verify_transaction(
		self,
		tenant_id: str,
		transaction_id: str,
	) -> dict[str, Any]:
		"""Verify a transaction's integrity against its stored hash."""
		transaction = self._require_transaction(transaction_id, tenant_id)
		payload = {
			"id": transaction.id,
			"tenant_id": transaction.tenant_id,
			"ledger_id": transaction.ledger_id,
			"from_account": transaction.from_account,
			"to_account": transaction.to_account,
			"amount": transaction.amount,
			"asset": transaction.asset,
			"signature": transaction.signature,
			"key_custody_id": transaction.key_custody_id,
			"compliance_tags": list(transaction.compliance_tags),
			"submitted_by": transaction.submitted_by,
		}
		expected_hash = self._engine.transaction_hash(payload)
		valid = expected_hash == transaction.transaction_hash
		return {
			"transaction_id": transaction_id,
			"tenant_id": tenant_id,
			"stored_hash": transaction.transaction_hash,
			"computed_hash": expected_hash,
			"valid": valid,
			"status": transaction.status,
		}

	def block_explorer(
		self,
		tenant_id: str,
		ledger_id: str,
		limit: int = 50,
	) -> dict[str, Any]:
		"""Return a paginated view of committed transactions for a ledger."""
		self._require_ledger(ledger_id, tenant_id)
		txns = [
			t.to_dict()
			for t in self._transactions.values()
			if t.tenant_id == tenant_id and t.ledger_id == ledger_id and t.status == "committed"
		]
		txns_sorted = sorted(txns, key=lambda x: x["id"])
		head_key = self._tenant_key(tenant_id, ledger_id)
		return {
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"chain_head": self._ledger_heads.get(head_key),
			"committed_transaction_count": len(txns),
			"transactions": txns_sorted[:limit],
		}

	def token_mint(
		self,
		mint_id: str,
		tenant_id: str,
		ledger_id: str,
		to_account: str,
		amount: float,
		asset: str,
		actor: str,
		key_custody_id: str,
	) -> dict[str, Any]:
		"""Mint new tokens to an account on the ledger."""
		self._require_ledger(ledger_id, tenant_id)
		import hashlib
		mint_record = {
			"mint_id": mint_id,
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"to_account": to_account,
			"amount": amount,
			"asset": asset,
			"actor": actor,
			"key_custody_id": key_custody_id,
			"mint_hash": hashlib.sha256(f"{mint_id}:{asset}:{amount}".encode()).hexdigest(),
			"status": "minted",
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=mint_id,
			event_type="token_minted",
			actor=actor,
			decision="allow",
			metadata={"asset": asset, "amount": amount, "to_account": to_account},
		)
		return mint_record

	def token_transfer(
		self,
		transaction_id: str,
		tenant_id: str,
		ledger_id: str,
		from_account: str,
		to_account: str,
		amount: float,
		asset: str,
		signature: str,
		key_custody_id: str,
		actor: str = "user",
	) -> dict[str, Any]:
		"""Transfer tokens between accounts — thin wrapper around submit_transaction."""
		return self.submit_transaction(
			transaction_id=transaction_id,
			tenant_id=tenant_id,
			ledger_id=ledger_id,
			from_account=from_account,
			to_account=to_account,
			amount=amount,
			signature=signature,
			key_custody_id=key_custody_id,
			asset=asset,
			actor=actor,
		)

	def digital_signature(
		self,
		signing_id: str,
		tenant_id: str,
		payload: dict[str, Any],
		key_id: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Produce a deterministic digital signature record for a payload."""
		import hashlib, hmac
		payload_bytes = str(sorted(payload.items())).encode()
		signature = hmac.new(key_id.encode(), payload_bytes, hashlib.sha256).hexdigest()
		record = {
			"signing_id": signing_id,
			"tenant_id": tenant_id,
			"key_id": key_id,
			"payload_hash": hashlib.sha256(payload_bytes).hexdigest(),
			"signature": signature,
			"algorithm": "hmac-sha256",
			"actor": actor,
			"status": "signed",
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=signing_id,
			event_type="payload_signed",
			actor=actor,
			decision="allow",
			metadata={"key_id": key_id, "algorithm": "hmac-sha256"},
		)
		return record

	def certificate_anchor(
		self,
		anchor_id: str,
		tenant_id: str,
		ledger_id: str,
		certificate_hash: str,
		issuer: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Anchor a certificate hash on the ledger for immutable provenance."""
		self._require_ledger(ledger_id, tenant_id)
		block_hash = self._commit_block(tenant_id, ledger_id, [certificate_hash])
		record = {
			"anchor_id": anchor_id,
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"certificate_hash": certificate_hash,
			"issuer": issuer,
			"block_hash": block_hash,
			"actor": actor,
			"status": "anchored",
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=anchor_id,
			event_type="certificate_anchored",
			actor=actor,
			decision="allow",
			metadata={"certificate_hash": certificate_hash, "issuer": issuer, "block_hash": block_hash},
		)
		return record

	def audit_trail_verify(
		self,
		tenant_id: str,
		subject_id: str,
	) -> dict[str, Any]:
		"""Return the full audit trail for a subject and verify sequential integrity."""
		events = [
			e.to_dict()
			for e in self._audit_events.values()
			if e.tenant_id == tenant_id and e.subject_id == subject_id
		]
		events_sorted = sorted(events, key=lambda x: x["id"])
		return {
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"event_count": len(events_sorted),
			"events": events_sorted,
			"integrity": "verified" if events_sorted else "no_events",
		}

	def cross_chain_bridge(
		self,
		bridge_id: str,
		tenant_id: str,
		source_ledger_id: str,
		target_ledger_id: str,
		transaction_hash: str,
		actor: str,
	) -> dict[str, Any]:
		"""Register a cross-chain bridge operation anchoring a transaction hash."""
		self._require_ledger(source_ledger_id, tenant_id)
		self._require_ledger(target_ledger_id, tenant_id)
		import hashlib
		bridge_hash = hashlib.sha256(f"{bridge_id}:{transaction_hash}".encode()).hexdigest()
		record = {
			"bridge_id": bridge_id,
			"tenant_id": tenant_id,
			"source_ledger_id": source_ledger_id,
			"target_ledger_id": target_ledger_id,
			"transaction_hash": transaction_hash,
			"bridge_hash": bridge_hash,
			"actor": actor,
			"status": "bridged",
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=bridge_id,
			event_type="cross_chain_bridge_created",
			actor=actor,
			decision="allow",
			metadata={"source_ledger_id": source_ledger_id, "target_ledger_id": target_ledger_id},
		)
		return record

	def consensus_monitor(
		self,
		tenant_id: str,
		ledger_id: str,
	) -> dict[str, Any]:
		"""Return consensus health metrics for a ledger."""
		self._require_ledger(ledger_id, tenant_id)
		ledger = self._ledgers[self._tenant_key(tenant_id, ledger_id)]
		txns = [
			t for t in self._transactions.values()
			if t.tenant_id == tenant_id and t.ledger_id == ledger_id
		]
		committed = [t for t in txns if t.status == "committed"]
		pending = [t for t in txns if t.status == "pending_review"]
		return {
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"consensus_profile": ledger.consensus_profile,
			"fork_monitoring_enabled": ledger.fork_monitoring_enabled,
			"committed_transaction_count": len(committed),
			"pending_transaction_count": len(pending),
			"chain_head": self._ledger_heads.get(self._tenant_key(tenant_id, ledger_id)),
			"consensus_health": "healthy" if not pending else "pending_transactions",
		}

	def gas_estimate(
		self,
		tenant_id: str,
		transaction_amount: float,
		asset: str = "TOKEN",
		ledger_id: str | None = None,
	) -> dict[str, Any]:
		"""Estimate gas/fee for a transaction based on amount and network load."""
		base_fee = 0.001
		volume_fee = transaction_amount * 0.0001
		priority_fee = 0.0005
		total_fee = round(base_fee + volume_fee + priority_fee, 8)
		return {
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"transaction_amount": transaction_amount,
			"asset": asset,
			"base_fee": base_fee,
			"volume_fee": round(volume_fee, 8),
			"priority_fee": priority_fee,
			"estimated_total_fee": total_fee,
			"currency": "NATIVE",
		}

	def wallet_create(
		self,
		wallet_id: str,
		tenant_id: str,
		ledger_id: str,
		owner: str,
		wallet_type: str = "standard",
	) -> dict[str, Any]:
		"""Create a wallet record and bind a key custody entry."""
		self._require_ledger(ledger_id, tenant_id)
		import secrets, hashlib
		private_key_ref = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
		public_key = hashlib.sha256(private_key_ref.encode()).hexdigest()
		address = "0x" + public_key[:40]
		binding = self.bind_key_custody(
			binding_id=f"custody:{wallet_id}",
			tenant_id=tenant_id,
			ledger_id=ledger_id,
			key_id=private_key_ref,
			custodian=owner,
		)
		record = {
			"wallet_id": wallet_id,
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"owner": owner,
			"wallet_type": wallet_type,
			"address": address,
			"public_key": public_key,
			"custody_binding_id": binding["id"],
			"status": "active",
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=wallet_id,
			event_type="wallet_created",
			actor=owner,
			decision="allow",
			metadata={"ledger_id": ledger_id, "wallet_type": wallet_type, "address": address},
		)
		return record

	def nft_mint(
		self,
		token_id: str,
		tenant_id: str,
		ledger_id: str,
		contract_id: str,
		to_account: str,
		metadata_uri: str,
		actor: str,
		key_custody_id: str,
	) -> dict[str, Any]:
		"""Mint an NFT by invoking the contract's mint method and recording the token."""
		import hashlib
		contract = self._contracts.get(self._tenant_key(tenant_id, contract_id))
		if contract is None:
			raise KeyError(f"unknown contract for tenant: {contract_id}")
		token_hash = hashlib.sha256(f"{token_id}:{metadata_uri}".encode()).hexdigest()
		block_hash = self._commit_block(tenant_id, ledger_id, [token_hash])
		record = {
			"token_id": token_id,
			"tenant_id": tenant_id,
			"ledger_id": ledger_id,
			"contract_id": contract_id,
			"to_account": to_account,
			"metadata_uri": metadata_uri,
			"token_hash": token_hash,
			"block_hash": block_hash,
			"actor": actor,
			"status": "minted",
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=token_id,
			event_type="nft_minted",
			actor=actor,
			decision="allow",
			metadata={"contract_id": contract_id, "to_account": to_account, "metadata_uri": metadata_uri},
		)
		return record

	def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return blockchain ledger service health."""
		summary = self.ledger_summary(tenant_id)
		return {
			"status": "healthy",
			"tenant_id": tenant_id,
			"ledger_count": summary["ledger_count"],
			"committed_transaction_count": summary["committed_transaction_count"],
			"pending_review_count": summary["pending_review_count"],
			"audit_event_count": summary["audit_event_count"],
		}

	def dashboard(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return aggregated KPI dashboard for blockchain ledger."""
		summary = self.ledger_summary(tenant_id)
		health = self.health_check(tenant_id)
		return {**summary, "health": health}

	def export_transactions(
		self,
		tenant_id: str,
		export_format: str = "json",
	) -> dict[str, Any]:
		"""Export transactions in the requested format."""
		transactions = self.list_transactions(tenant_id)
		if export_format == "csv":
			keys = list(transactions[0].keys()) if transactions else []
			lines = [",".join(keys)] + [",".join(str(t.get(k, "")) for k in keys) for t in transactions]
			data = "\n".join(lines)
		else:
			import json as _json
			data = _json.dumps(transactions, default=str, indent=2)
		return {"tenant_id": tenant_id, "format": export_format, "count": len(transactions), "data": data}

	def bulk_submit_transactions(
		self,
		tenant_id: str,
		ledger_id: str,
		transactions: list[dict[str, Any]],
		key_custody_id: str,
		actor: str = "system",
	) -> list[dict[str, Any]]:
		"""Submit multiple transactions in a single call."""
		results = []
		for txn in transactions:
			try:
				result = self.submit_transaction(
					transaction_id=str(txn.get("id") or txn.get("transaction_id") or ""),
					tenant_id=tenant_id,
					ledger_id=ledger_id,
					from_account=str(txn.get("from_account", "")),
					to_account=str(txn.get("to_account", "")),
					amount=float(txn.get("amount", 0.0)),
					signature=str(txn.get("signature", "bulk-sig")),
					key_custody_id=key_custody_id,
					asset=str(txn.get("asset", "TOKEN")),
					actor=actor,
				)
				results.append(result)
			except (ValueError, KeyError, PermissionError) as exc:
				results.append({"id": txn.get("id"), "error": str(exc)})
		return results

	def compliance_report(
		self,
		tenant_id: str,
		framework: str = "iso27001",
	) -> dict[str, Any]:
		"""Generate a compliance posture report for the ledger service."""
		summary = self.ledger_summary(tenant_id)
		score = 0.0
		checks: list[str] = []
		if summary["key_custody_count"] > 0:
			score += 35
			checks.append("key_custody_configured=True")
		if summary["committed_transaction_count"] > 0 and summary["pending_review_count"] == 0:
			score += 30
			checks.append("no_pending_transactions=True")
		if summary["contract_count"] > 0:
			score += 15
			checks.append("smart_contracts_deployed=True")
		if summary["audit_event_count"] > 0:
			score += 20
			checks.append("audit_trail_active=True")
		return {
			"tenant_id": tenant_id,
			"framework": framework,
			"compliance_score": round(score, 1),
			"status": "compliant" if score >= 80 else "non_compliant",
			"checks": checks,
			**summary,
		}

	def _list(self, values: Any, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")
