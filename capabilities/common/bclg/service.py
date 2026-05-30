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

	def _list(self, values: Any, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")
