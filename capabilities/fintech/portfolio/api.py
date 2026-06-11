"""Dependency-light API helpers for APG Portfolio Management."""

from __future__ import annotations

import asyncio
from typing import Any

try:
	from .service import PortfolioManagementService
except ImportError:  # pragma: no cover
	from service import PortfolioManagementService  # type: ignore


_SERVICE = PortfolioManagementService()


def service() -> PortfolioManagementService:
	return _SERVICE


def _run(coro):
	"""Execute an async coroutine from synchronous context."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				future = pool.submit(asyncio.run, coro)
				return future.result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def create_portfolio_book(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])
	return _run(svc.create_portfolio(
		name=payload["name"],
		client_id=payload["owner_id"],
		strategy=payload.get("strategy", ""),
		benchmark=payload.get("benchmark", ""),
		portfolio_type=payload["portfolio_type"],
		base_currency=payload["base_currency"],
		policy_reference=payload["policy_reference"],
		portfolio_id=payload.get("portfolio_id"),
	))


def record_holding(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])
	# pre-populate portfolio so the service can find it
	import asyncio as _asyncio

	async def _op():
		# create a placeholder portfolio entry so add_holding can resolve it
		p = await svc.create_portfolio("placeholder", payload["tenant_id"], "", "",
			portfolio_id=payload["portfolio_id"])
		return await svc.add_holding(
			portfolio_id=payload["portfolio_id"],
			asset_id=payload["instrument_id"],
			quantity=float(payload["quantity"]),
			cost_basis=payload["cost_minor"] / 100,
			currency=payload["currency"],
			holding_id=payload.get("holding_id"),
		)
	return _run(_op())


def activate_allocation_policy(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])

	async def _op():
		await svc.create_portfolio("placeholder", payload["tenant_id"], "", "",
			portfolio_id=payload["portfolio_id"])
		return await svc.activate_allocation_policy(
			allocation_id=payload["allocation_id"],
			portfolio_id=payload["portfolio_id"],
			target_allocation=dict(payload["target_allocation"]),
			policy_reference=payload["policy_reference"],
		)
	return _run(_op())


def record_valuation(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])

	async def _op():
		await svc.create_portfolio("placeholder", payload["tenant_id"], "", "",
			portfolio_id=payload["portfolio_id"])
		return await svc.portfolio_valuation(
			portfolio_id=payload["portfolio_id"],
			as_of_date=payload["valuation_date"],
			source_reference=payload["source_reference"],
		)
	return _run(_op())


def assign_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])

	async def _op():
		await svc.create_portfolio("placeholder", payload["tenant_id"], "", "",
			portfolio_id=payload["portfolio_id"])
		return await svc.assign_benchmark(
			benchmark_id=payload["benchmark_id"],
			portfolio_id=payload["portfolio_id"],
			index_id=payload["index_id"],
			policy_reference=payload["policy_reference"],
		)
	return _run(_op())


def record_risk_exposure(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])

	async def _op():
		await svc.create_portfolio("placeholder", payload["tenant_id"], "", "",
			portfolio_id=payload["portfolio_id"])
		return await svc.record_risk_exposure(
			exposure_id=payload["exposure_id"],
			portfolio_id=payload["portfolio_id"],
			metric=payload["metric"],
			value=payload["value"],
			as_of_date=payload["as_of_date"],
			source_reference=payload["source_reference"],
			limit_reference=payload.get("limit_reference", ""),
		)
	return _run(_op())


def record_attribution(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])

	async def _op():
		await svc.create_portfolio("placeholder", payload["tenant_id"], "", "",
			portfolio_id=payload["portfolio_id"])
		return await svc.performance_attribution(
			portfolio_id=payload["portfolio_id"],
			period=payload["period"],
			benchmark_id=payload.get("benchmark_id", ""),
		)
	return _run(_op())


def record_cash_movement(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])

	async def _op():
		await svc.create_portfolio("placeholder", payload["tenant_id"], "", "",
			portfolio_id=payload["portfolio_id"])
		return await svc.record_cash_movement(
			movement_id=payload["movement_id"],
			portfolio_id=payload["portfolio_id"],
			amount_minor=int(payload["amount_minor"]),
			currency=payload["currency"],
			reference=payload["reference"],
		)
	return _run(_op())


def record_corporate_action(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])
	return _run(svc.record_corporate_action(
		action_id=payload["action_id"],
		instrument_id=payload["instrument_id"],
		action_type=payload["action_type"],
		effective_date=payload["effective_date"],
		evidence_reference=payload["evidence_reference"],
		ratio=payload.get("ratio"),
	))


def record_compliance_breach(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])

	async def _op():
		await svc.create_portfolio("placeholder", payload["tenant_id"], "", "",
			portfolio_id=payload["portfolio_id"])
		return await svc.record_compliance_breach(
			breach_id=payload["breach_id"],
			portfolio_id=payload["portfolio_id"],
			severity=payload["severity"],
			evidence_reference=payload["evidence_reference"],
		)
	return _run(_op())


def record_review(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])
	return _run(svc.record_review(
		review_id=payload["review_id"],
		reference_id=payload["reference_id"],
		reviewer_id=payload["reviewer_id"],
		status=payload["status"],
		evidence_reference=payload["evidence_reference"],
	))


def register_portfolio_agent(payload: dict[str, Any]) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=payload["tenant_id"])
	return _run(svc.register_portfolio_agent(
		agent_id=payload["agent_id"],
		name=payload["name"],
		runtime=payload["runtime"],
		role=payload["role"],
		scope=payload.get("scope", "portfolio management review"),
	))


def dashboard(tenant_id: str) -> dict[str, Any]:
	svc = PortfolioManagementService(tenant_id=tenant_id)
	return _run(svc.dashboard_summary())
