"""Dependency-light API helpers for APG Robo Advisory."""

from __future__ import annotations

from typing import Any

try:
	from .service import RoboAdvisoryService
except ImportError:  # pragma: no cover
	from service import RoboAdvisoryService  # type: ignore


_SERVICE = RoboAdvisoryService()


def service() -> RoboAdvisoryService:
	return _SERVICE


def create_investor_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_investor_profile(payload["profile_id"], payload["tenant_id"], payload["client_id"], payload["kyc_reference"], payload["suitability_reference"], payload["risk_profile"], payload.get("policy_attached", True))


def define_goal_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.define_goal_plan(payload["goal_id"], payload["tenant_id"], payload["profile_id"], payload["goal_type"], payload["target_amount_minor"], payload["currency"], payload["horizon_date"])


def publish_model_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.publish_model_portfolio(payload["model_id"], payload["tenant_id"], payload["name"], payload["risk_profile"], dict(payload["target_allocation"]), payload["policy_reference"])


def generate_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.generate_recommendation(payload["recommendation_id"], payload["tenant_id"], payload["profile_id"], payload["goal_id"], payload["model_id"], payload["analysis_reference"])


def approve_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.approve_recommendation(payload["recommendation_id"], payload["tenant_id"], payload["reviewer_id"])


def configure_automation_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.configure_automation_plan(payload["plan_id"], payload["tenant_id"], payload["recommendation_id"], payload["funding_source_reference"], payload["cadence"])


def record_drift(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_drift(payload["drift_id"], payload["tenant_id"], payload["profile_id"], payload["drift_bps"], payload["analysis_reference"])


def record_tax_loss_candidate(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_tax_loss_candidate(payload["candidate_id"], payload["tenant_id"], payload["profile_id"], payload["instrument_id"], payload["loss_minor"], payload["tax_lot_reference"])


def record_review(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_review(payload["review_id"], payload["tenant_id"], payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_robo_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_robo_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "robo advisory review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
