"""View models for generated Portfolio Analytics screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import PortfolioAnalyticsService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import PortfolioAnalyticsService  # type: ignore


def dashboard_model(service: PortfolioAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the portfolio analytics dashboard."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Portfolio Analytics",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def portfolio_list_model(service: PortfolioAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the portfolio list screen."""
	return {
		"tenant_id": tenant_id,
		"portfolios": _tenant_items(service.portfolios, tenant_id),
	}


def alignment_scorecard_model(service: PortfolioAnalyticsService, tenant_id: str = "default", portfolio_id: str | None = None) -> dict[str, Any]:
	"""View model for strategic alignment scorecard."""
	return {
		"tenant_id": tenant_id,
		"portfolio_id": portfolio_id,
		"alignment_scores": [
			v.to_dict() for v in sorted(service.alignment_scores.values(), key=lambda x: x.id)
			if v.tenant_id == tenant_id and (portfolio_id is None or v.portfolio_id == portfolio_id)
		],
	}


def risk_return_matrix_model(service: PortfolioAnalyticsService, tenant_id: str = "default", portfolio_id: str | None = None) -> dict[str, Any]:
	"""View model for the risk-return matrix."""
	return {
		"tenant_id": tenant_id,
		"portfolio_id": portfolio_id,
		"risk_return_analyses": [
			v.to_dict() for v in sorted(service.risk_return_analyses.values(), key=lambda x: x.id)
			if v.tenant_id == tenant_id and (portfolio_id is None or v.portfolio_id == portfolio_id)
		],
	}


def capacity_heat_map_model(service: PortfolioAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for capacity heat maps."""
	return {
		"tenant_id": tenant_id,
		"heat_maps": _tenant_items(service.heat_maps, tenant_id),
	}


def performance_scoreboard_model(service: PortfolioAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for performance snapshots."""
	return {
		"tenant_id": tenant_id,
		"performance_snapshots": _tenant_items(service.performance_snapshots, tenant_id),
	}


def scenario_analysis_model(service: PortfolioAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for scenario analyses."""
	return {
		"tenant_id": tenant_id,
		"scenarios": _tenant_items(service.scenarios, tenant_id),
	}


def agent_workbench_model(service: PortfolioAnalyticsService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the analytics agent workbench."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [v.to_dict() for v in service.agents.values() if v.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for v in sorted(items.values(), key=lambda x: x.id) if v.tenant_id == tenant_id]
