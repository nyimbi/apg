"""Process-local API helpers for APG Decentralized Finance."""

from __future__ import annotations

try:
	from .service import DecentralizedFinanceService
except ImportError:  # pragma: no cover
	from service import DecentralizedFinanceService  # type: ignore


_SERVICE = DecentralizedFinanceService()


def service() -> DecentralizedFinanceService:
	return _SERVICE


def register_protocol(payload: dict):
	return _SERVICE.register_protocol(payload["protocol_id"], payload.get("tenant_id", "default"), payload["protocol_type"], payload["network_reference"], payload["protocol_reference"], payload["owner_id"], payload["evidence_reference"], payload["risk_tier"], payload.get("policy_attached", True))


def open_position(payload: dict):
	return _SERVICE.open_position(payload["position_id"], payload.get("tenant_id", "default"), payload["protocol_id"], payload["account_reference"], payload["asset_pair_reference"], payload["position_type"], payload["amount_minor"], payload.get("collateral_minor", 0), payload["health_factor_bps"], payload["evidence_reference"])


def record_action(payload: dict):
	return _SERVICE.record_action(payload["action_id"], payload.get("tenant_id", "default"), payload["protocol_id"], payload["position_id"], payload["action_type"], payload["amount_minor"], payload["requester_id"], payload["approval_reference"], payload["evidence_reference"], payload.get("status", "requested"))


def register_yield_strategy(payload: dict):
	return _SERVICE.register_yield_strategy(payload["strategy_id"], payload.get("tenant_id", "default"), payload["protocol_id"], payload["strategy_reference"], payload["target_apy_bps"], payload["max_risk_tier"], payload["owner_id"], payload["evidence_reference"])


def record_reward(payload: dict):
	return _SERVICE.record_reward(payload["reward_id"], payload.get("tenant_id", "default"), payload["position_id"], payload["reward_type"], payload["asset_reference"], payload["amount_minor"], payload["evidence_reference"])


def record_governance_vote(payload: dict):
	return _SERVICE.record_governance_vote(payload["proposal_id"], payload.get("tenant_id", "default"), payload["protocol_id"], payload["proposal_reference"], payload["vote_choice"], payload["voter_id"], payload["evidence_reference"])


def record_risk_assessment(payload: dict):
	return _SERVICE.record_risk_assessment(payload["assessment_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["risk_tier"], payload["reviewer_id"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_defi_agent(payload: dict):
	return _SERVICE.register_defi_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "defi operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
