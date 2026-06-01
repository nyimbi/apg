"""Executable service layer for APG Decentralized Finance."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_ACTION_STATUSES, SUPPORTED_ACTION_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_GOVERNANCE_VOTES, SUPPORTED_POSITION_TYPES, SUPPORTED_PROTOCOL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_REWARD_TYPES, SUPPORTED_RISK_TIERS, evaluate_capability_rules, get_capability_contract
	from .defi_runtime import non_negative_int, normalize_code, positive_int, present
	from .models import DeFiAction, DeFiAgent, DeFiProtocol, DeFiReview, GovernanceProposal, LiquidityPosition, RewardAccrual, RiskAssessment, YieldStrategy
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_ACTION_STATUSES, SUPPORTED_ACTION_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_GOVERNANCE_VOTES, SUPPORTED_POSITION_TYPES, SUPPORTED_PROTOCOL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_REWARD_TYPES, SUPPORTED_RISK_TIERS, evaluate_capability_rules, get_capability_contract  # type: ignore
	from defi_runtime import non_negative_int, normalize_code, positive_int, present  # type: ignore
	from models import DeFiAction, DeFiAgent, DeFiProtocol, DeFiReview, GovernanceProposal, LiquidityPosition, RewardAccrual, RiskAssessment, YieldStrategy  # type: ignore


class DecentralizedFinanceService:
	"""Dependency-light DeFi runtime for generated APG applications."""

	def __init__(self) -> None:
		self.protocols: dict[str, DeFiProtocol] = {}
		self.positions: dict[str, LiquidityPosition] = {}
		self.actions: dict[str, DeFiAction] = {}
		self.strategies: dict[str, YieldStrategy] = {}
		self.rewards: dict[str, RewardAccrual] = {}
		self.governance: dict[str, GovernanceProposal] = {}
		self.risk_assessments: dict[str, RiskAssessment] = {}
		self.reviews: dict[str, DeFiReview] = {}
		self.agents: dict[str, DeFiAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_protocol(self, protocol_id: str, tenant_id: str, protocol_type: str, network_reference: str, protocol_reference: str, owner_id: str, evidence_reference: str, risk_tier: str, policy_attached: bool = True) -> dict[str, Any]:
		protocol_type = normalize_code(protocol_type)
		risk_tier = normalize_code(risk_tier)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_protocol", "protocol_type_supported": protocol_type in SUPPORTED_PROTOCOL_TYPES, "network_present": present(network_reference), "protocol_reference_present": present(protocol_reference), "owner_present": present(owner_id), "evidence_present": present(evidence_reference), "risk_tier_supported": risk_tier in SUPPORTED_RISK_TIERS})
		item = DeFiProtocol(protocol_id, tenant_id, protocol_type, network_reference, protocol_reference, owner_id, evidence_reference, risk_tier)
		self.protocols[protocol_id] = item
		self._audit(tenant_id, "defi_protocol_registered", protocol_id)
		return item.to_dict()

	def open_position(self, position_id: str, tenant_id: str, protocol_id: str, account_reference: str, asset_pair_reference: str, position_type: str, amount_minor: int, collateral_minor: int, health_factor_bps: int, evidence_reference: str) -> dict[str, Any]:
		protocol = self._tenant_protocol_or_none(protocol_id, tenant_id)
		position_type = normalize_code(position_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_position", "protocol_present": protocol is not None, "account_present": present(account_reference), "asset_pair_present": present(asset_pair_reference), "position_type_supported": position_type in SUPPORTED_POSITION_TYPES, "amount_valid": positive_int(amount_minor), "collateral_valid": non_negative_int(collateral_minor), "health_factor_valid": positive_int(health_factor_bps), "evidence_present": present(evidence_reference)})
		item = LiquidityPosition(position_id, tenant_id, protocol_id, account_reference, asset_pair_reference, position_type, int(amount_minor), int(collateral_minor), int(health_factor_bps), evidence_reference)
		self.positions[position_id] = item
		self._audit(tenant_id, "defi_position_opened", position_id)
		return item.to_dict()

	def record_action(self, action_id: str, tenant_id: str, protocol_id: str, position_id: str, action_type: str, amount_minor: int, requester_id: str, approval_reference: str, evidence_reference: str, status: str = "requested") -> dict[str, Any]:
		protocol = self._tenant_protocol_or_none(protocol_id, tenant_id)
		position = self._tenant_position_or_none(position_id, tenant_id)
		action_type = normalize_code(action_type)
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_action", "protocol_present": protocol is not None, "position_present": position is not None, "position_protocol_match": position is not None and position.protocol_id == protocol_id, "action_type_supported": action_type in SUPPORTED_ACTION_TYPES, "amount_valid": positive_int(amount_minor), "requester_present": present(requester_id), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference), "status_supported": status in SUPPORTED_ACTION_STATUSES})
		item = DeFiAction(action_id, tenant_id, protocol_id, position_id, action_type, int(amount_minor), requester_id, approval_reference, evidence_reference, status)
		self.actions[action_id] = item
		self._audit(tenant_id, "defi_action_recorded", action_id)
		return item.to_dict()

	def register_yield_strategy(self, strategy_id: str, tenant_id: str, protocol_id: str, strategy_reference: str, target_apy_bps: int, max_risk_tier: str, owner_id: str, evidence_reference: str) -> dict[str, Any]:
		protocol = self._tenant_protocol_or_none(protocol_id, tenant_id)
		max_risk_tier = normalize_code(max_risk_tier)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_yield_strategy", "protocol_present": protocol is not None, "strategy_reference_present": present(strategy_reference), "target_apy_valid": non_negative_int(target_apy_bps), "max_risk_supported": max_risk_tier in SUPPORTED_RISK_TIERS, "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = YieldStrategy(strategy_id, tenant_id, protocol_id, strategy_reference, int(target_apy_bps), max_risk_tier, owner_id, evidence_reference)
		self.strategies[strategy_id] = item
		self._audit(tenant_id, "defi_yield_strategy_registered", strategy_id)
		return item.to_dict()

	def record_reward(self, reward_id: str, tenant_id: str, position_id: str, reward_type: str, asset_reference: str, amount_minor: int, evidence_reference: str) -> dict[str, Any]:
		position = self._tenant_position_or_none(position_id, tenant_id)
		reward_type = normalize_code(reward_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_reward", "position_present": position is not None, "reward_type_supported": reward_type in SUPPORTED_REWARD_TYPES, "asset_present": present(asset_reference), "amount_valid": positive_int(amount_minor), "evidence_present": present(evidence_reference)})
		item = RewardAccrual(reward_id, tenant_id, position_id, reward_type, asset_reference, int(amount_minor), evidence_reference)
		self.rewards[reward_id] = item
		self._audit(tenant_id, "defi_reward_recorded", reward_id)
		return item.to_dict()

	def record_governance_vote(self, proposal_id: str, tenant_id: str, protocol_id: str, proposal_reference: str, vote_choice: str, voter_id: str, evidence_reference: str) -> dict[str, Any]:
		protocol = self._tenant_protocol_or_none(protocol_id, tenant_id)
		vote_choice = normalize_code(vote_choice)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_governance_vote", "protocol_present": protocol is not None, "proposal_present": present(proposal_reference), "vote_supported": vote_choice in SUPPORTED_GOVERNANCE_VOTES, "voter_present": present(voter_id), "evidence_present": present(evidence_reference)})
		item = GovernanceProposal(proposal_id, tenant_id, protocol_id, proposal_reference, vote_choice, voter_id, evidence_reference)
		self.governance[proposal_id] = item
		self._audit(tenant_id, "defi_governance_vote_recorded", proposal_id)
		return item.to_dict()

	def record_risk_assessment(self, assessment_id: str, tenant_id: str, reference_id: str, risk_tier: str, reviewer_id: str, evidence_reference: str) -> dict[str, Any]:
		risk_tier = normalize_code(risk_tier)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_risk_assessment", "reference_present": present(reference_id), "risk_tier_supported": risk_tier in SUPPORTED_RISK_TIERS, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = RiskAssessment(assessment_id, tenant_id, reference_id, risk_tier, reviewer_id, evidence_reference)
		self.risk_assessments[assessment_id] = item
		self._audit(tenant_id, "defi_risk_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = DeFiReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "defi_review_recorded", review_id)
		return item.to_dict()

	def register_defi_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_defi_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = DeFiAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "defi_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "defi_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "defi_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.defi.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "protocol_count": self._count(self.protocols, tenant_id), "position_count": self._count(self.positions, tenant_id), "critical_protocol_count": sum(1 for item in self.protocols.values() if item.tenant_id == tenant_id and item.risk_tier == "critical"), "action_count": self._count(self.actions, tenant_id), "open_action_count": sum(1 for item in self.actions.values() if item.tenant_id == tenant_id and item.status in {"requested", "approved", "submitted"}), "yield_strategy_count": self._count(self.strategies, tenant_id), "reward_count": self._count(self.rewards, tenant_id), "governance_vote_count": self._count(self.governance, tenant_id), "risk_assessment_count": self._count(self.risk_assessments, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_protocol_or_none(self, item_id: str, tenant_id: str) -> DeFiProtocol | None:
		item = self.protocols.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_position_or_none(self, item_id: str, tenant_id: str) -> LiquidityPosition | None:
		item = self.positions.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "defi_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "defi_policy_denied")


FintechDeFiService = DecentralizedFinanceService
