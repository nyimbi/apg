"""Process-local API helpers for APG Portfolio Analytics (pan)."""

from __future__ import annotations

try:
	from .service import PortfolioAnalyticsService
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from service import PortfolioAnalyticsService  # type: ignore

_SERVICE = PortfolioAnalyticsService()


def service() -> PortfolioAnalyticsService:
	return _SERVICE


def create_portfolio(payload: dict):
	return _SERVICE.create_portfolio(
		payload["portfolio_id"], payload.get("tenant_id", "default"),
		payload["name"], payload.get("status", "proposed"),
		payload.get("classification", "internal"), payload["owner_id"],
		payload["approval_reference"], payload["evidence_reference"],
		payload.get("policy_attached", True),
	)


def score_alignment(payload: dict):
	return _SERVICE.score_alignment(
		payload["score_id"], payload.get("tenant_id", "default"),
		payload["portfolio_id"], payload["dimension"],
		payload["scoring_method"], float(payload["score"]),
		payload.get("rationale", ""), payload["evidence_reference"],
	)


def analyse_risk_return(payload: dict):
	return _SERVICE.analyse_risk_return(
		payload["analysis_id"], payload.get("tenant_id", "default"),
		payload["portfolio_id"], payload["risk_category"],
		payload["return_metric"], float(payload["risk_score"]),
		float(payload["return_value"]), payload.get("analysis_period", ""),
		payload["evidence_reference"],
	)


def generate_heat_map(payload: dict):
	return _SERVICE.generate_heat_map(
		payload["heat_map_id"], payload.get("tenant_id", "default"),
		payload["portfolio_id"], payload["dimension"],
		payload.get("snapshot_period", ""), payload.get("heat_map_data", "{}"),
		payload.get("generated_by", "system"),
	)


def snapshot_performance(payload: dict):
	return _SERVICE.snapshot_performance(
		payload["snapshot_id"], payload.get("tenant_id", "default"),
		payload["portfolio_id"], payload["period"],
		payload.get("metrics", "{}"), payload["benchmark_type"],
		float(payload["benchmark_value"]), float(payload["actual_value"]),
	)


def run_scenario(payload: dict):
	return _SERVICE.run_scenario(
		payload["scenario_id"], payload.get("tenant_id", "default"),
		payload["portfolio_id"], payload["scenario_name"],
		payload.get("assumptions", "{}"), payload.get("projected_outcome", "{}"),
		payload["analyst_id"], payload["evidence_reference"],
	)


def generate_report(payload: dict):
	return _SERVICE.generate_report(
		payload["report_id"], payload.get("tenant_id", "default"),
		payload["portfolio_id"], payload["dashboard_type"],
		payload.get("format", "dashboard"), payload.get("generated_by", "system"),
		payload.get("report_data", "{}"),
	)


def register_agent(payload: dict):
	return _SERVICE.register_agent(
		payload["agent_id"], payload.get("tenant_id", "default"),
		payload["name"], payload["runtime"], payload["role"],
		payload.get("scope", "portfolio analytics operations"),
	)


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(
		payload.get("tenant_id", "default"),
		payload.get("privileged_scope", False),
		payload.get("human_approval_recorded", False),
	)


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(
		payload.get("tenant_id", "default"),
		payload["item_count"],
		payload.get("event_stream", "bytewax"),
	)


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
