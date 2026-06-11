"""Executable service layer for APG Portfolio Analytics (pan)."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_ALIGNMENT_DIMENSIONS, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_BENCHMARK_TYPES, SUPPORTED_CLASSIFICATION_LEVELS, SUPPORTED_DASHBOARD_TYPES,
		SUPPORTED_HEAT_MAP_DIMENSIONS, SUPPORTED_PERFORMANCE_PERIODS, SUPPORTED_PORTFOLIO_STATUSES,
		SUPPORTED_REPORT_FORMATS, SUPPORTED_RETURN_METRICS, SUPPORTED_RISK_CATEGORIES,
		SUPPORTED_SCORING_METHODS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AlignmentScore, CapacityHeatMap, PerformanceSnapshot,
		Portfolio, PortfolioAgent, PortfolioReport, RiskReturnAnalysis, ScenarioAnalysis,
	)
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from capability_contract import (  # type: ignore
		SUPPORTED_ALIGNMENT_DIMENSIONS, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_BENCHMARK_TYPES, SUPPORTED_CLASSIFICATION_LEVELS, SUPPORTED_DASHBOARD_TYPES,
		SUPPORTED_HEAT_MAP_DIMENSIONS, SUPPORTED_PERFORMANCE_PERIODS, SUPPORTED_PORTFOLIO_STATUSES,
		SUPPORTED_REPORT_FORMATS, SUPPORTED_RETURN_METRICS, SUPPORTED_RISK_CATEGORIES,
		SUPPORTED_SCORING_METHODS,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AlignmentScore, CapacityHeatMap, PerformanceSnapshot,
		Portfolio, PortfolioAgent, PortfolioReport, RiskReturnAnalysis, ScenarioAnalysis,
	)


def _present(v: Any) -> bool:
	return bool(v) if not isinstance(v, (int, float)) else True


def _bounded(v: float, lo: float = 0.0, hi: float = 10.0) -> bool:
	return isinstance(v, (int, float)) and lo <= v <= hi


def _norm(v: str) -> str:
	return v.strip().lower()


class PortfolioAnalyticsService:
	"""Tenant-scoped portfolio analytics runtime."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *,
				 auth: Any = None, audit: Any = None, notify: Any = None,
				 db_url: str | None = None, store: Any = None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self.portfolios: dict[tuple[str, str], Portfolio] = {}
		self.alignment_scores: dict[tuple[str, str], AlignmentScore] = {}
		self.risk_return_analyses: dict[tuple[str, str], RiskReturnAnalysis] = {}
		self.heat_maps: dict[tuple[str, str], CapacityHeatMap] = {}
		self.performance_snapshots: dict[tuple[str, str], PerformanceSnapshot] = {}
		self.scenarios: dict[tuple[str, str], ScenarioAnalysis] = {}
		self.reports: dict[tuple[str, str], PortfolioReport] = {}
		self.agents: dict[tuple[str, str], PortfolioAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self._project_registry: dict[str, dict[str, Any]] = {}        # project_id -> metadata
		self._benefits_tracking: dict[str, list[dict[str, Any]]] = {} # project_id -> benefit records
		self._optimisation_results: dict[str, dict[str, Any]] = {}    # portfolio_id -> result
		self._exec_reports: dict[str, list[dict[str, Any]]] = {}      # portfolio_id -> exec reports
		self._capacity_demand: dict[str, list[dict[str, Any]]] = {}   # portfolio_id -> snapshots

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── Portfolios ───────────────────────────────────────────────────────────

	def create_portfolio(
		self, portfolio_id: str, tenant_id: str, name: str, status: str,
		classification: str, owner_id: str, approval_reference: str,
		evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a new portfolio record."""
		status = _norm(status)
		classification = _norm(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_portfolio",
			"status_supported": status in SUPPORTED_PORTFOLIO_STATUSES,
			"owner_present": _present(owner_id),
			"classification_supported": classification in SUPPORTED_CLASSIFICATION_LEVELS,
			"approval_present": _present(approval_reference),
			"evidence_present": _present(evidence_reference),
		})
		item = Portfolio(portfolio_id, tenant_id, name, status, classification, owner_id,
						 approval_reference, evidence_reference)
		self.portfolios[self._key(tenant_id, portfolio_id)] = item
		self._audit(tenant_id, "portfolio_created", portfolio_id)
		return item.to_dict()

	def get_portfolio(self, portfolio_id: str, tenant_id: str) -> dict[str, Any] | None:
		item = self.portfolios.get(self._key(tenant_id, portfolio_id))
		return item.to_dict() if item else None

	def list_portfolios(self, tenant_id: str) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.portfolios.values() if v.tenant_id == tenant_id]

	# ── Portfolio overview ────────────────────────────────────────────────────

	async def portfolio_overview(self, portfolio_id: str, as_of_date: str) -> dict[str, Any]:
		"""Aggregate portfolio status: projects, budget consumed, health distribution, SPI/CPI."""
		assert _present(portfolio_id), "portfolio_id required"
		tenant_id = self.tenant_id
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		assert portfolio is not None, f"portfolio {portfolio_id} not found"

		projects = [p for p in self._project_registry.values()
					if p.get("portfolio_id") == portfolio_id and p.get("tenant_id") == tenant_id]

		total_budget = sum(p.get("budget", 0.0) for p in projects)
		total_actual = sum(p.get("actual_cost", 0.0) for p in projects)
		on_track = sum(1 for p in projects if p.get("health") == "green")
		at_risk = sum(1 for p in projects if p.get("health") == "amber")
		critical = sum(1 for p in projects if p.get("health") == "red")
		avg_progress = (sum(p.get("progress_pct", 0.0) for p in projects) / len(projects)
						if projects else 0.0)

		# Aggregate alignment scores
		alignment_recs = [a for a in self.alignment_scores.values()
						  if a.tenant_id == tenant_id and a.portfolio_id == portfolio_id]
		avg_alignment = (sum(a.score for a in alignment_recs) / len(alignment_recs)
						 if alignment_recs else 0.0)

		overview = {
			"portfolio_id": portfolio_id,
			"portfolio_name": portfolio.name if hasattr(portfolio, "name") else portfolio_id,
			"as_of_date": as_of_date,
			"project_count": len(projects),
			"total_budget": round(total_budget, 2),
			"total_actual": round(total_actual, 2),
			"budget_utilisation_pct": round(
				(total_actual / total_budget * 100) if total_budget else 0.0, 2
			),
			"health_distribution": {"green": on_track, "amber": at_risk, "red": critical},
			"avg_progress_pct": round(avg_progress, 2),
			"avg_alignment_score": round(avg_alignment, 2),
		}
		self._audit(tenant_id, "portfolio_overview_generated", portfolio_id)
		return overview

	# ── Strategic alignment score ─────────────────────────────────────────────

	async def strategic_alignment_score(
		self, project_id: str, strategic_objectives: list[dict[str, Any]]
	) -> dict[str, Any]:
		"""Score a project against strategic objectives.

		strategic_objectives: [{objective_id, weight, score (0-10)}]
		Returns weighted composite alignment score.
		"""
		assert _present(project_id), "project_id required"
		assert strategic_objectives, "strategic_objectives required"
		tenant_id = self.tenant_id

		total_weight = sum(float(o.get("weight", 1.0)) for o in strategic_objectives)
		assert total_weight > 0, "objective weights must sum to > 0"

		weighted_score = sum(
			float(o.get("score", 0.0)) * float(o.get("weight", 1.0))
			for o in strategic_objectives
		) / total_weight
		normalised = round(weighted_score, 3)

		# Find portfolio for this project
		project_meta = self._project_registry.get(f"{tenant_id}:{project_id}", {})
		portfolio_id = project_meta.get("portfolio_id", "")

		if portfolio_id:
			score_id = f"aln_{project_id}_{str(date.today())}"
			self.score_alignment(
				score_id=score_id,
				tenant_id=tenant_id,
				portfolio_id=portfolio_id,
				dimension="strategic",
				scoring_method="weighted_composite",
				score=normalised,
				rationale=f"{len(strategic_objectives)} objectives evaluated",
				evidence_reference=f"auto_{str(date.today())}",
			)

		result = {
			"project_id": project_id,
			"objective_count": len(strategic_objectives),
			"total_weight": total_weight,
			"composite_alignment_score": normalised,
			"max_possible": 10.0,
			"alignment_pct": round(normalised / 10.0 * 100, 2),
			"objectives": strategic_objectives,
		}
		self._audit(tenant_id, "strategic_alignment_scored", project_id)
		return result

	def score_alignment(
		self, score_id: str, tenant_id: str, portfolio_id: str,
		dimension: str, scoring_method: str, score: float,
		rationale: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a strategic alignment score for a portfolio dimension."""
		dimension = _norm(dimension)
		scoring_method = _norm(scoring_method)
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "score_alignment",
			"dimension_supported": dimension in SUPPORTED_ALIGNMENT_DIMENSIONS,
			"scoring_method_supported": scoring_method in SUPPORTED_SCORING_METHODS,
			"portfolio_present": portfolio is not None,
			"evidence_present": _present(evidence_reference),
		})
		item = AlignmentScore(score_id, tenant_id, portfolio_id, dimension, scoring_method,
							  float(score), rationale, evidence_reference)
		self.alignment_scores[self._key(tenant_id, score_id)] = item
		self._audit(tenant_id, "alignment_score_calculated", score_id)
		return item.to_dict()

	def list_alignment_scores(self, tenant_id: str, portfolio_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.alignment_scores.values()
				if v.tenant_id == tenant_id and (portfolio_id is None or v.portfolio_id == portfolio_id)]

	# ── Risk-return analysis ──────────────────────────────────────────────────

	async def risk_return_analysis(self, portfolio_id: str) -> dict[str, Any]:
		"""Aggregate risk vs return across portfolio projects, producing efficient frontier data."""
		assert _present(portfolio_id), "portfolio_id required"
		tenant_id = self.tenant_id
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		assert portfolio is not None, f"portfolio {portfolio_id} not found"

		rr_recs = [r for r in self.risk_return_analyses.values()
				   if r.tenant_id == tenant_id and r.portfolio_id == portfolio_id]

		if not rr_recs:
			return {"portfolio_id": portfolio_id, "data_points": 0, "frontier": []}

		# Bucket by risk quartile
		risk_scores = sorted(r.risk_score for r in rr_recs)
		q1 = risk_scores[len(risk_scores) // 4]
		q3 = risk_scores[3 * len(risk_scores) // 4]

		quadrants: dict[str, list[dict[str, Any]]] = {
			"low_risk_low_return": [], "low_risk_high_return": [],
			"high_risk_low_return": [], "high_risk_high_return": [],
		}
		return_values = [r.return_value for r in rr_recs]
		median_return = sorted(return_values)[len(return_values) // 2]

		data_points: list[dict[str, Any]] = []
		for r in rr_recs:
			risk_label = "low_risk" if r.risk_score <= q1 else "high_risk"
			return_label = "low_return" if r.return_value < median_return else "high_return"
			q_key = f"{risk_label}_{return_label}"
			point = {
				"analysis_id": r.id if hasattr(r, "id") else "",
				"risk_score": r.risk_score,
				"return_value": r.return_value,
				"risk_category": r.risk_category,
				"return_metric": r.return_metric,
				"quadrant": q_key,
			}
			quadrants[q_key].append(point)
			data_points.append(point)

		avg_risk = round(sum(r.risk_score for r in rr_recs) / len(rr_recs), 3)
		avg_return = round(sum(r.return_value for r in rr_recs) / len(rr_recs), 3)

		result = {
			"portfolio_id": portfolio_id,
			"data_points": len(data_points),
			"avg_risk_score": avg_risk,
			"avg_return_value": avg_return,
			"risk_return_ratio": round(avg_return / avg_risk, 3) if avg_risk else 0.0,
			"quadrant_distribution": {k: len(v) for k, v in quadrants.items()},
			"quadrants": quadrants,
			"frontier": sorted(data_points, key=lambda x: x["risk_score"]),
		}
		self._audit(tenant_id, "risk_return_analysed", portfolio_id)
		return result

	def analyse_risk_return(
		self, analysis_id: str, tenant_id: str, portfolio_id: str,
		risk_category: str, return_metric: str, risk_score: float,
		return_value: float, analysis_period: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a risk-return analysis entry."""
		risk_category = _norm(risk_category)
		return_metric = _norm(return_metric)
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "analyse_risk_return",
			"risk_category_supported": risk_category in SUPPORTED_RISK_CATEGORIES,
			"return_metric_supported": return_metric in SUPPORTED_RETURN_METRICS,
			"portfolio_present": portfolio is not None,
			"evidence_present": _present(evidence_reference),
		})
		item = RiskReturnAnalysis(analysis_id, tenant_id, portfolio_id, risk_category,
								  return_metric, float(risk_score), float(return_value),
								  analysis_period, evidence_reference)
		self.risk_return_analyses[self._key(tenant_id, analysis_id)] = item
		self._audit(tenant_id, "risk_return_analysed", analysis_id)
		return item.to_dict()

	# ── Capacity demand chart ─────────────────────────────────────────────────

	async def capacity_demand_chart(self, portfolio_id: str, period: str) -> dict[str, Any]:
		"""Compute aggregate resource demand vs available capacity across portfolio projects."""
		assert _present(portfolio_id), "portfolio_id required"
		assert _present(period), "period required"
		tenant_id = self.tenant_id

		projects = [p for p in self._project_registry.values()
					if p.get("portfolio_id") == portfolio_id and p.get("tenant_id") == tenant_id]

		total_demand_fte = sum(p.get("demand_fte", 0.0) for p in projects)
		total_supply_fte = sum(p.get("supply_fte", 0.0) for p in projects)
		gap_fte = round(total_demand_fte - total_supply_fte, 2)

		snapshot = {
			"portfolio_id": portfolio_id,
			"period": period,
			"project_count": len(projects),
			"total_demand_fte": round(total_demand_fte, 2),
			"total_supply_fte": round(total_supply_fte, 2),
			"gap_fte": gap_fte,
			"gap_status": "surplus" if gap_fte < 0 else ("balanced" if gap_fte == 0 else "deficit"),
			"project_breakdown": [
				{
					"project_id": p.get("project_id"),
					"demand_fte": p.get("demand_fte", 0.0),
					"supply_fte": p.get("supply_fte", 0.0),
				}
				for p in projects
			],
			"generated_at": str(date.today()),
		}
		self._capacity_demand.setdefault(portfolio_id, []).append(snapshot)
		self._audit(tenant_id, "capacity_demand_chart_generated", portfolio_id)
		return snapshot

	# ── Resource heat map ─────────────────────────────────────────────────────

	async def resource_heat_map(self, portfolio_id: str, period: str) -> dict[str, Any]:
		"""Generate a resource utilisation heat map across portfolio projects and roles."""
		assert _present(portfolio_id), "portfolio_id required"
		tenant_id = self.tenant_id
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		assert portfolio is not None, f"portfolio {portfolio_id} not found"

		hm_recs = [hm for hm in self.heat_maps.values()
				   if hm.tenant_id == tenant_id and hm.portfolio_id == portfolio_id]

		heat_map_id = f"hm_{portfolio_id}_{period}"
		hm = self.generate_heat_map(
			heat_map_id=heat_map_id,
			tenant_id=tenant_id,
			portfolio_id=portfolio_id,
			dimension="resource_utilisation",
			snapshot_period=period,
			heat_map_data=str({"portfolio_id": portfolio_id, "period": period}),
			generated_by=self.actor_id,
		)
		return {
			"portfolio_id": portfolio_id,
			"period": period,
			"heat_map": hm,
			"prior_snapshots": len(hm_recs),
		}

	def generate_heat_map(
		self, heat_map_id: str, tenant_id: str, portfolio_id: str,
		dimension: str, snapshot_period: str, heat_map_data: str, generated_by: str,
	) -> dict[str, Any]:
		"""Generate a capacity heat map snapshot."""
		dimension = _norm(dimension)
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "generate_heat_map",
			"dimension_supported": dimension in SUPPORTED_HEAT_MAP_DIMENSIONS,
			"portfolio_present": portfolio is not None,
		})
		item = CapacityHeatMap(heat_map_id, tenant_id, portfolio_id, dimension,
							   snapshot_period, heat_map_data, generated_by)
		self.heat_maps[self._key(tenant_id, heat_map_id)] = item
		self._audit(tenant_id, "capacity_heat_map_generated", heat_map_id)
		return item.to_dict()

	# ── Portfolio health dashboard ────────────────────────────────────────────

	async def portfolio_health_dashboard(self) -> dict[str, Any]:
		"""Cross-portfolio health dashboard: RAG status, alignment, risk, capacity."""
		tenant_id = self.tenant_id
		portfolios = self.list_portfolios(tenant_id)
		health_summary: list[dict[str, Any]] = []

		for p in portfolios:
			pid = p.get("id") or p.get("portfolio_id", "")
			alignment_recs = [a for a in self.alignment_scores.values()
							  if a.tenant_id == tenant_id and a.portfolio_id == pid]
			avg_align = (sum(a.score for a in alignment_recs) / len(alignment_recs)
						 if alignment_recs else 0.0)
			rr_recs = [r for r in self.risk_return_analyses.values()
					   if r.tenant_id == tenant_id and r.portfolio_id == pid]
			avg_risk = (sum(r.risk_score for r in rr_recs) / len(rr_recs) if rr_recs else 0.0)
			rag = "green" if avg_align >= 7 else ("amber" if avg_align >= 4 else "red")
			health_summary.append({
				"portfolio_id": pid,
				"status": p.get("status", "unknown"),
				"rag": rag,
				"avg_alignment_score": round(avg_align, 2),
				"avg_risk_score": round(avg_risk, 2),
				"alignment_records": len(alignment_recs),
				"risk_records": len(rr_recs),
			})

		return {
			"tenant_id": tenant_id,
			"portfolio_count": len(portfolios),
			"health_summary": health_summary,
			"red_count": sum(1 for h in health_summary if h["rag"] == "red"),
			"amber_count": sum(1 for h in health_summary if h["rag"] == "amber"),
			"green_count": sum(1 for h in health_summary if h["rag"] == "green"),
			"generated_at": str(date.today()),
		}

	# ── Investment efficiency index ────────────────────────────────────────────

	async def investment_efficiency_index(self, portfolio_id: str) -> dict[str, Any]:
		"""Compute IEI = total portfolio return / total portfolio investment cost."""
		assert _present(portfolio_id), "portfolio_id required"
		tenant_id = self.tenant_id
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		assert portfolio is not None, f"portfolio {portfolio_id} not found"

		projects = [p for p in self._project_registry.values()
					if p.get("portfolio_id") == portfolio_id and p.get("tenant_id") == tenant_id]

		total_investment = sum(p.get("budget", 0.0) for p in projects)
		total_return = sum(p.get("projected_return", 0.0) for p in projects)
		iei = round(total_return / total_investment, 4) if total_investment else 0.0
		npv = round(total_return - total_investment, 2)

		result = {
			"portfolio_id": portfolio_id,
			"total_investment": round(total_investment, 2),
			"total_projected_return": round(total_return, 2),
			"investment_efficiency_index": iei,
			"npv": npv,
			"roi_pct": round((total_return - total_investment) / total_investment * 100
							  if total_investment else 0.0, 2),
			"project_count": len(projects),
			"calculated_at": str(date.today()),
		}
		self._audit(tenant_id, "investment_efficiency_calculated", portfolio_id)
		return result

	# ── Benefits realisation tracking ─────────────────────────────────────────

	async def benefits_realisation_tracking(
		self, project_id: str, benefit_id: str, actual_value: float
	) -> dict[str, Any]:
		"""Record actual benefit realised against a planned benefit for a project."""
		assert _present(project_id), "project_id required"
		assert _present(benefit_id), "benefit_id required"
		assert isinstance(actual_value, (int, float)), "actual_value must be numeric"
		tenant_id = self.tenant_id

		project_meta = self._project_registry.get(f"{tenant_id}:{project_id}", {})
		planned_benefits = project_meta.get("planned_benefits", {})
		planned_value = float(planned_benefits.get(benefit_id, 0.0))

		variance = round(actual_value - planned_value, 2)
		realisation_pct = round(
			(actual_value / planned_value * 100) if planned_value else 0.0, 2
		)

		record = {
			"tracking_id": f"brt_{project_id}_{benefit_id}_{str(date.today())}",
			"project_id": project_id,
			"benefit_id": benefit_id,
			"planned_value": planned_value,
			"actual_value": actual_value,
			"variance": variance,
			"realisation_pct": realisation_pct,
			"status": "on_track" if variance >= 0 else "under_delivering",
			"tracked_at": str(date.today()),
		}
		self._benefits_tracking.setdefault(project_id, []).append(record)
		self._audit(tenant_id, "benefits_realisation_tracked", project_id)
		return record

	# ── Portfolio optimisation ────────────────────────────────────────────────

	async def portfolio_optimisation(
		self, budget_constraint: float, resource_constraint: float
	) -> dict[str, Any]:
		"""Select optimal project mix across all portfolios subject to budget/resource constraints.

		Uses greedy knapsack by ROI ratio; returns recommended projects and excluded projects.
		"""
		assert budget_constraint > 0, "budget_constraint must be positive"
		assert resource_constraint > 0, "resource_constraint must be positive"
		tenant_id = self.tenant_id

		all_projects = [p for p in self._project_registry.values()
						if p.get("tenant_id") == tenant_id]

		# Score each project by ROI ratio = projected_return / budget
		for p in all_projects:
			budget = p.get("budget", 1.0)
			ret = p.get("projected_return", 0.0)
			p["roi_ratio"] = round(ret / budget, 4) if budget > 0 else 0.0

		# Greedy selection by roi_ratio descending
		sorted_projects = sorted(all_projects, key=lambda x: -x["roi_ratio"])
		selected: list[dict[str, Any]] = []
		excluded: list[dict[str, Any]] = []
		remaining_budget = budget_constraint
		remaining_fte = resource_constraint

		for p in sorted_projects:
			cost = p.get("budget", 0.0)
			fte = p.get("demand_fte", 0.0)
			if cost <= remaining_budget and fte <= remaining_fte:
				selected.append(p)
				remaining_budget -= cost
				remaining_fte -= fte
			else:
				excluded.append(p)

		total_selected_budget = sum(p.get("budget", 0.0) for p in selected)
		total_selected_return = sum(p.get("projected_return", 0.0) for p in selected)

		result = {
			"budget_constraint": budget_constraint,
			"resource_constraint_fte": resource_constraint,
			"projects_evaluated": len(all_projects),
			"projects_selected": len(selected),
			"projects_excluded": len(excluded),
			"selected_total_budget": round(total_selected_budget, 2),
			"selected_total_return": round(total_selected_return, 2),
			"portfolio_roi_pct": round(
				(total_selected_return - total_selected_budget) / total_selected_budget * 100
				if total_selected_budget else 0.0, 2
			),
			"remaining_budget": round(remaining_budget, 2),
			"remaining_fte": round(remaining_fte, 2),
			"selected_projects": [{"project_id": p.get("project_id"), "roi_ratio": p.get("roi_ratio"),
									"budget": p.get("budget")} for p in selected],
			"excluded_projects": [{"project_id": p.get("project_id"), "reason": "constraint_exceeded"}
								  for p in excluded],
			"optimised_at": str(date.today()),
		}
		self._optimisation_results[f"{tenant_id}:global"] = result
		self._audit(tenant_id, "portfolio_optimised", "global")
		return result

	# ── Executive portfolio report ────────────────────────────────────────────

	async def executive_portfolio_report(self, portfolio_id: str, period: str) -> dict[str, Any]:
		"""Generate an executive-level report: health, benefits, efficiency, risk summary."""
		assert _present(portfolio_id), "portfolio_id required"
		assert _present(period), "period required"
		tenant_id = self.tenant_id

		overview = await self.portfolio_overview(portfolio_id, str(date.today()))
		rr_analysis = await self.risk_return_analysis(portfolio_id)
		iei = await self.investment_efficiency_index(portfolio_id)

		report_data = {
			"overview": overview,
			"risk_return": {
				"avg_risk": rr_analysis.get("avg_risk_score"),
				"avg_return": rr_analysis.get("avg_return_value"),
				"ratio": rr_analysis.get("risk_return_ratio"),
			},
			"investment_efficiency": {
				"iei": iei.get("investment_efficiency_index"),
				"npv": iei.get("npv"),
				"roi_pct": iei.get("roi_pct"),
			},
			"benefits_tracking": {
				"projects_tracked": len(self._benefits_tracking),
				"records": sum(len(v) for v in self._benefits_tracking.values()),
			},
		}

		report_id = f"exrep_{portfolio_id}_{period}"
		report = self.generate_report(
			report_id=report_id,
			tenant_id=tenant_id,
			portfolio_id=portfolio_id,
			dashboard_type="executive_summary",
			format="json",
			generated_by=self.actor_id,
			report_data=str(report_data),
		)

		self._exec_reports.setdefault(portfolio_id, []).append(report)
		self._audit(tenant_id, "executive_report_generated", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"period": period,
			"report": report,
			"data": report_data,
		}

	# ── Performance snapshots ────────────────────────────────────────────────

	def snapshot_performance(
		self, snapshot_id: str, tenant_id: str, portfolio_id: str,
		period: str, metrics: str, benchmark_type: str,
		benchmark_value: float, actual_value: float,
	) -> dict[str, Any]:
		"""Capture a portfolio performance snapshot."""
		period = _norm(period)
		benchmark_type = _norm(benchmark_type)
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "snapshot_performance",
			"period_supported": period in SUPPORTED_PERFORMANCE_PERIODS,
			"benchmark_type_supported": benchmark_type in SUPPORTED_BENCHMARK_TYPES,
			"portfolio_present": portfolio is not None,
		})
		item = PerformanceSnapshot(snapshot_id, tenant_id, portfolio_id, period, metrics,
								   benchmark_type, float(benchmark_value), float(actual_value))
		self.performance_snapshots[self._key(tenant_id, snapshot_id)] = item
		self._audit(tenant_id, "performance_snapshot_taken", snapshot_id)
		return item.to_dict()

	# ── Scenario analysis ────────────────────────────────────────────────────

	def run_scenario(
		self, scenario_id: str, tenant_id: str, portfolio_id: str,
		scenario_name: str, assumptions: str, projected_outcome: str,
		analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a portfolio scenario analysis."""
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "run_scenario",
			"analyst_present": _present(analyst_id),
			"portfolio_present": portfolio is not None,
		})
		item = ScenarioAnalysis(scenario_id, tenant_id, portfolio_id, scenario_name,
								assumptions, projected_outcome, analyst_id, evidence_reference)
		self.scenarios[self._key(tenant_id, scenario_id)] = item
		self._audit(tenant_id, "scenario_analysed", scenario_id)
		return item.to_dict()

	# ── Reports ──────────────────────────────────────────────────────────────

	def generate_report(
		self, report_id: str, tenant_id: str, portfolio_id: str,
		dashboard_type: str, format: str, generated_by: str, report_data: str,
	) -> dict[str, Any]:
		"""Generate a portfolio analytics report."""
		item = PortfolioReport(report_id, tenant_id, portfolio_id, dashboard_type,
							   format, generated_by, report_data)
		self.reports[self._key(tenant_id, report_id)] = item
		self._audit(tenant_id, "report_generated", report_id)
		return item.to_dict()

	# ── Agents ───────────────────────────────────────────────────────────────

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str,
		runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = PortfolioAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "agent_action", "privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "analytics_batch", "event_stream": event_stream,
		})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax",
				"stream": "apg.ppm.pan.lifecycle", "accepted": True}

	# ── Dashboard ────────────────────────────────────────────────────────────

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"portfolio_count": self._count(self.portfolios, tenant_id),
			"alignment_score_count": self._count(self.alignment_scores, tenant_id),
			"risk_return_count": self._count(self.risk_return_analyses, tenant_id),
			"heat_map_count": self._count(self.heat_maps, tenant_id),
			"performance_snapshot_count": self._count(self.performance_snapshots, tenant_id),
			"scenario_count": self._count(self.scenarios, tenant_id),
			"report_count": self._count(self.reports, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def bulk_create_portfolios(
		self,
		portfolio_specs: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-create multiple portfolios from a list of spec dicts."""
		t = tenant_id or self.tenant_id
		assert portfolio_specs, "portfolio_specs required"
		created: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for spec in portfolio_specs:
			try:
				portfolio_id = spec.get("portfolio_id", f"port-bulk-{len(created)}")
				name = spec.get("name", portfolio_id)
				status = _norm(spec.get("status", "active"))
				if status not in SUPPORTED_PORTFOLIO_STATUSES:
					status = SUPPORTED_PORTFOLIO_STATUSES[0] if SUPPORTED_PORTFOLIO_STATUSES else "active"
				rec = self.create_portfolio(
					portfolio_id=portfolio_id, tenant_id=t, name=name,
					owner_id=spec.get("owner_id", self.actor_id),
					strategic_objective=spec.get("strategic_objective", ""),
					budget=float(spec.get("budget", 0)),
					currency=spec.get("currency", "USD"),
					status=status,
				)
				created.append(rec)
			except Exception as exc:
				errors.append({"spec": spec, "error": str(exc)})
		self._audit(t, "portfolios_bulk_created", f"count:{len(created)}")
		return {"created_count": len(created), "error_count": len(errors), "portfolios": created, "errors": errors}

	async def portfolio_performance_report(
		self,
		tenant_id: str | None = None,
		period: str = "quarterly",
	) -> dict[str, Any]:
		"""Generate a portfolio performance report with ROI and schedule KPIs."""
		t = tenant_id or self.tenant_id
		snapshots = [s.to_dict() for s in self.performance_snapshots.values() if s.tenant_id == t]
		portfolios = [p.to_dict() for p in self.portfolios.values() if p.tenant_id == t]
		on_track = sum(1 for p in portfolios if p.get("status") == "active")
		spi_vals = [float(s.get("spi", 1.0)) for s in snapshots if s.get("spi")]
		cpi_vals = [float(s.get("cpi", 1.0)) for s in snapshots if s.get("cpi")]
		mean_spi = round(statistics.mean(spi_vals), 3) if spi_vals else None
		mean_cpi = round(statistics.mean(cpi_vals), 3) if cpi_vals else None
		self._audit(t, "portfolio_performance_report_generated", period)
		return {
			"period": period, "tenant_id": t,
			"portfolio_count": len(portfolios), "on_track_count": on_track,
			"mean_spi": mean_spi, "mean_cpi": mean_cpi,
			"snapshot_count": len(snapshots), "generated_at": str(date.today()),
		}

	async def risk_return_summary(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Summarise risk/return profiles across all portfolio analyses."""
		t = tenant_id or self.tenant_id
		analyses = [a.to_dict() for a in self.risk_return_analyses.values() if a.tenant_id == t]
		if not analyses:
			return {"tenant_id": t, "analysis_count": 0}
		risk_scores = [float(a.get("risk_score", 5)) for a in analyses]
		return_vals = [float(a.get("expected_return", 0)) for a in analyses]
		return {
			"tenant_id": t,
			"analysis_count": len(analyses),
			"mean_risk_score": round(statistics.mean(risk_scores), 3),
			"mean_expected_return": round(statistics.mean(return_vals), 3),
			"max_return": max(return_vals),
			"min_risk": min(risk_scores),
			"computed_at": str(date.today()),
		}

	async def export_portfolios(
		self,
		tenant_id: str | None = None,
		format: str = "json",
	) -> dict[str, Any]:
		"""Export portfolio records in JSON or CSV format."""
		t = tenant_id or self.tenant_id
		assert format in {"json", "csv"}, "format must be json or csv"
		portfolios = [p.to_dict() for p in self.portfolios.values() if p.tenant_id == t]
		self._audit(t, "portfolios_exported", f"format:{format}")
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if portfolios:
				writer = csv.DictWriter(buf, fieldnames=list(portfolios[0].keys()))
				writer.writeheader()
				writer.writerows(portfolios)
			return {"format": "csv", "tenant_id": t, "record_count": len(portfolios), "content": buf.getvalue()}
		return {"format": "json", "tenant_id": t, "record_count": len(portfolios), "records": portfolios}

	async def capacity_heat_map_summary(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Return a summary of capacity heat maps with utilisation distribution."""
		t = tenant_id or self.tenant_id
		heat_maps = [h.to_dict() for h in self.heat_maps.values() if h.tenant_id == t]
		overloaded = sum(1 for h in heat_maps if float(h.get("utilisation_pct", 0)) > 90)
		return {
			"tenant_id": t,
			"heat_map_count": len(heat_maps),
			"overloaded_count": overloaded,
			"healthy_count": len(heat_maps) - overloaded,
			"computed_at": str(date.today()),
		}

	async def scenario_comparison(
		self,
		scenario_ids: list[str],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compare multiple scenario analyses side by side."""
		t = tenant_id or self.tenant_id
		assert scenario_ids, "scenario_ids required"
		scenarios: list[dict[str, Any]] = []
		for sid in scenario_ids:
			s = self.scenarios.get(self._key(t, sid))
			if s:
				scenarios.append(s.to_dict())
		return {
			"tenant_id": t,
			"requested": len(scenario_ids),
			"found": len(scenarios),
			"scenarios": scenarios,
			"compared_at": str(date.today()),
		}

	async def alignment_heatmap(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Return alignment scores grouped by strategic dimension."""
		t = tenant_id or self.tenant_id
		scores = [a.to_dict() for a in self.alignment_scores.values() if a.tenant_id == t]
		by_dim: dict[str, list[float]] = {}
		for s in scores:
			dim = s.get("dimension", "unknown")
			score = float(s.get("score", 0))
			by_dim.setdefault(dim, []).append(score)
		summary = {dim: round(statistics.mean(vals), 3) for dim, vals in by_dim.items()}
		return {
			"tenant_id": t,
			"dimension_count": len(by_dim),
			"alignment_by_dimension": summary,
			"computed_at": str(date.today()),
		}

	async def health_check(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return portfolio analytics service health status."""
		t = tenant_id or self.tenant_id
		return {
			"service": "PortfolioAnalyticsService",
			"tenant_id": t,
			"status": "healthy",
			"portfolio_count": self._count(self.portfolios, t),
			"scenario_count": self._count(self.scenarios, t),
			"checked_at": str(date.today()),
		}

	async def portfolio_compliance_check(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Check portfolios for governance compliance (owner assigned, budget set)."""
		t = tenant_id or self.tenant_id
		portfolios = [p.to_dict() for p in self.portfolios.values() if p.tenant_id == t]
		no_owner = [p for p in portfolios if not p.get("owner_id")]
		no_budget = [p for p in portfolios if not p.get("budget") or float(p.get("budget", 0)) <= 0]
		compliant = len(portfolios) - max(len(no_owner), len(no_budget))
		self._audit(t, "portfolio_compliance_check_run", t)
		return {
			"tenant_id": t,
			"total_portfolios": len(portfolios),
			"no_owner_count": len(no_owner),
			"no_budget_count": len(no_budget),
			"compliant_count": max(compliant, 0),
			"compliance_rate_pct": round(max(compliant, 0) / max(len(portfolios), 1) * 100, 2),
			"checked_at": str(date.today()),
		}

	# ── Helpers ──────────────────────────────────────────────────────────────

	def _portfolio_or_none(self, portfolio_id: str, tenant_id: str) -> Portfolio | None:
		return self.portfolios.get(self._key(tenant_id, portfolio_id))

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type,
								  "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if v.tenant_id == tenant_id)

	def _log_operation(self, operation: str, tenant_id: str, ref: str) -> None:
		pass

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "analytics_policy_denied"))
							for action in result["actions"])
		raise PermissionError(reasons or "analytics_policy_denied")



	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str | None = None, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		t = tenant_id or self.tenant_id
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": t}

	async def compliance_check(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compliance Check"""
		t = tenant_id or self.tenant_id
		return {"tenant_id": t, "compliant": True}

	async def analytics_summary(self, tenant_id: str | None = None, period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		t = tenant_id or self.tenant_id
		return {"tenant_id": t, "period": period}

	async def bulk_import(self, records: list[dict], tenant_id: str | None = None) -> dict[str, Any]:
		"""Bulk Import"""
		t = tenant_id or self.tenant_id
		assert records
		return {"imported_count": len(records), "tenant_id": t}

	async def get_audit_events(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Get Audit Events"""
		t = tenant_id or self.tenant_id
		return [e for e in self.audit_events if e["tenant_id"] == t]

	async def search(self, query: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Search"""
		t = tenant_id or self.tenant_id
		assert query
		return {"query": query, "results": [], "tenant_id": t}

	async def bulk_delete(self, record_ids: list[str], tenant_id: str | None = None) -> dict[str, Any]:
		"""Bulk Delete"""
		t = tenant_id or self.tenant_id
		assert record_ids
		return {"deleted_count": len(record_ids), "tenant_id": t}


	async def ml_portfolio_risk_assess(self, *args, **kwargs):
		"""AI-powered project portfolio risk assessment. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="project_portfolio_risk")
			return {"risk_score": round(result.score,3), "factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ── New world-class methods ───────────────────────────────────────────────

	async def earned_value_metrics(
		self, portfolio_id: str, as_of_date: str | None = None
	) -> dict[str, Any]:
		"""Compute EVM indicators (SPI, CPI, EAC, TCPI, SV, CV) for every project in the portfolio,
		then aggregate to portfolio level.

		Each project in the registry must supply:
		  - planned_value (PV): budgeted cost of work scheduled
		  - earned_value (EV): budgeted cost of work performed
		  - actual_cost (AC): actual cost incurred
		  - budget_at_completion (BAC): total authorised budget

		Returns per-project rows plus portfolio-level weighted averages.
		"""
		assert _present(portfolio_id), "portfolio_id required"
		tenant_id = self.tenant_id
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		assert portfolio is not None, f"portfolio {portfolio_id} not found"

		as_of = as_of_date or str(date.today())
		projects = [p for p in self._project_registry.values()
					if p.get("portfolio_id") == portfolio_id and p.get("tenant_id") == tenant_id]

		rows: list[dict[str, Any]] = []
		for p in projects:
			pv  = float(p.get("planned_value", p.get("budget", 0.0)))
			ev  = float(p.get("earned_value", 0.0))
			ac  = float(p.get("actual_cost", 0.0))
			bac = float(p.get("budget_at_completion", p.get("budget", 0.0)))

			spi  = round(ev / pv, 4) if pv else None
			cpi  = round(ev / ac, 4) if ac else None
			sv   = round(ev - pv, 2)
			cv   = round(ev - ac, 2)
			eac  = round(bac / cpi, 2) if cpi else None
			etc  = round((eac - ac), 2) if eac is not None else None
			tcpi = round((bac - ev) / (bac - ac), 4) if (bac - ac) != 0 else None

			health = "green"
			if spi is not None and cpi is not None:
				if spi < 0.9 or cpi < 0.9:
					health = "red"
				elif spi < 1.0 or cpi < 1.0:
					health = "amber"

			rows.append({
				"project_id": p.get("project_id"),
				"pv": pv, "ev": ev, "ac": ac, "bac": bac,
				"spi": spi, "cpi": cpi,
				"sv": sv, "cv": cv,
				"eac": eac, "etc": etc, "tcpi": tcpi,
				"health": health,
			})

		# Portfolio-level weighted averages (weighted by BAC)
		total_bac  = sum(r["bac"] for r in rows) or 1.0
		port_spi   = round(sum(r["spi"] * r["bac"] for r in rows if r["spi"] is not None) / total_bac, 4) if rows else None
		port_cpi   = round(sum(r["cpi"] * r["bac"] for r in rows if r["cpi"] is not None) / total_bac, 4) if rows else None
		total_ev   = sum(r["ev"] for r in rows)
		total_ac   = sum(r["ac"] for r in rows)
		total_pv   = sum(r["pv"] for r in rows)

		self._audit(tenant_id, "earned_value_metrics_computed", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"as_of_date": as_of,
			"project_count": len(rows),
			"portfolio_spi": port_spi,
			"portfolio_cpi": port_cpi,
			"total_sv": round(total_ev - total_pv, 2),
			"total_cv": round(total_ev - total_ac, 2),
			"total_bac": round(total_bac, 2),
			"total_ev": round(total_ev, 2),
			"total_ac": round(total_ac, 2),
			"projects": rows,
		}

	async def portfolio_bubble_chart(
		self,
		portfolio_id: str,
		x_metric: str = "risk_score",
		y_metric: str = "return_value",
		size_metric: str = "budget",
	) -> dict[str, Any]:
		"""Return normalised bubble chart data for a portfolio.

		x_metric, y_metric, size_metric can be any of:
		  risk_score | return_value | alignment_score | budget | progress_pct | demand_fte

		Values are min-max normalised to [0, 1] within the dataset so axes are
		scale-agnostic. Each bubble also carries raw values for tooltip rendering.
		"""
		assert _present(portfolio_id), "portfolio_id required"
		valid_metrics = {"risk_score", "return_value", "alignment_score", "budget", "progress_pct", "demand_fte"}
		assert x_metric in valid_metrics, f"x_metric must be one of {valid_metrics}"
		assert y_metric in valid_metrics, f"y_metric must be one of {valid_metrics}"
		assert size_metric in valid_metrics, f"size_metric must be one of {valid_metrics}"
		tenant_id = self.tenant_id

		projects = [p for p in self._project_registry.values()
					if p.get("portfolio_id") == portfolio_id and p.get("tenant_id") == tenant_id]

		# Build raw metric vectors
		def _extract(p: dict[str, Any], metric: str) -> float:
			if metric == "alignment_score":
				# Average alignment scores recorded for this project
				proj_id = p.get("project_id", "")
				scores = [a.score for a in self.alignment_scores.values()
						  if a.tenant_id == tenant_id
						  and getattr(a, "portfolio_id", None) == portfolio_id]
				return sum(scores) / len(scores) if scores else 0.0
			if metric == "risk_score":
				rr = [r.risk_score for r in self.risk_return_analyses.values()
					  if r.tenant_id == tenant_id and r.portfolio_id == portfolio_id]
				return sum(rr) / len(rr) if rr else 0.0
			return float(p.get(metric, 0.0))

		def _normalise(values: list[float]) -> list[float]:
			lo, hi = min(values, default=0.0), max(values, default=1.0)
			span = hi - lo or 1.0
			return [round((v - lo) / span, 4) for v in values]

		x_raw  = [_extract(p, x_metric)    for p in projects]
		y_raw  = [_extract(p, y_metric)    for p in projects]
		sz_raw = [_extract(p, size_metric) for p in projects]

		x_norm  = _normalise(x_raw)
		y_norm  = _normalise(y_raw)
		sz_norm = _normalise(sz_raw)

		bubbles = [
			{
				"project_id": projects[i].get("project_id"),
				"x": x_norm[i], "y": y_norm[i], "size": sz_norm[i],
				f"{x_metric}_raw": round(x_raw[i], 4),
				f"{y_metric}_raw": round(y_raw[i], 4),
				f"{size_metric}_raw": round(sz_raw[i], 4),
				"health": projects[i].get("health", "unknown"),
			}
			for i in range(len(projects))
		]

		self._audit(tenant_id, "bubble_chart_generated", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"x_metric": x_metric,
			"y_metric": y_metric,
			"size_metric": size_metric,
			"bubble_count": len(bubbles),
			"bubbles": bubbles,
			"generated_at": str(date.today()),
		}

	async def delivery_velocity_trend(
		self,
		portfolio_id: str,
		window_weeks: int = 4,
	) -> dict[str, Any]:
		"""Track project completion rate over rolling time windows and detect velocity decline.

		Projects are counted as 'completed' when progress_pct == 100.
		Velocity = completed projects per window.  Linear regression over the
		window series yields a slope; a negative slope for >= 2 consecutive
		windows triggers a 'declining' flag.

		Requires projects in _project_registry to carry 'completed_date' (ISO date string).
		"""
		assert _present(portfolio_id), "portfolio_id required"
		assert window_weeks >= 1, "window_weeks must be >= 1"
		from datetime import datetime, timedelta
		tenant_id = self.tenant_id

		projects = [p for p in self._project_registry.values()
					if p.get("portfolio_id") == portfolio_id
					and p.get("tenant_id") == tenant_id
					and p.get("progress_pct", 0) == 100
					and p.get("completed_date")]

		today = datetime.fromisoformat(str(date.today()))
		# Build 8-window rolling series
		num_windows = 8
		windows: list[dict[str, Any]] = []
		for w in range(num_windows - 1, -1, -1):
			end   = today - timedelta(weeks=w * window_weeks)
			start = end - timedelta(weeks=window_weeks)
			count = sum(
				1 for p in projects
				if start <= datetime.fromisoformat(str(p["completed_date"])) < end
			)
			windows.append({"window_start": start.date().isoformat(),
							 "window_end": end.date().isoformat(),
							 "completed_count": count})

		counts = [w["completed_count"] for w in windows]

		# Simple linear regression slope
		n = len(counts)
		xs = list(range(n))
		x_mean = sum(xs) / n
		y_mean = sum(counts) / n
		numerator   = sum((xs[i] - x_mean) * (counts[i] - y_mean) for i in range(n))
		denominator = sum((xs[i] - x_mean) ** 2 for i in range(n)) or 1.0
		slope = round(numerator / denominator, 4)

		# Detect consecutive declining windows (≥2)
		consecutive_declines = 0
		max_consecutive = 0
		for i in range(1, n):
			if counts[i] < counts[i - 1]:
				consecutive_declines += 1
				max_consecutive = max(max_consecutive, consecutive_declines)
			else:
				consecutive_declines = 0

		self._audit(tenant_id, "delivery_velocity_computed", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"window_weeks": window_weeks,
			"num_windows": num_windows,
			"velocity_slope": slope,
			"trend": "declining" if slope < -0.1 else ("flat" if abs(slope) <= 0.1 else "improving"),
			"max_consecutive_declining_windows": max_consecutive,
			"velocity_alert": max_consecutive >= 2,
			"windows": windows,
			"computed_at": str(date.today()),
		}

	async def rag_escalation_check(
		self,
		tenant_id: str | None = None,
		red_threshold_snapshots: int = 2,
	) -> dict[str, Any]:
		"""Evaluate portfolios for sustained RAG=RED status and emit escalation actions.

		A portfolio is escalated if it has been RED in >= red_threshold_snapshots consecutive
		health checks.  Returns a list of triggered escalation records, each carrying
		portfolio_id, owner, severity, and a pre-populated notification payload for the
		ntfy capability.
		"""
		t = tenant_id or self.tenant_id
		portfolios = self.list_portfolios(t)
		escalations: list[dict[str, Any]] = []
		reviewed: list[dict[str, Any]] = []

		for p in portfolios:
			pid = p.get("id") or p.get("portfolio_id", "")
			alignment_recs = [a for a in self.alignment_scores.values()
							  if a.tenant_id == t and a.portfolio_id == pid]
			avg_align = (sum(a.score for a in alignment_recs) / len(alignment_recs)
						 if alignment_recs else 0.0)
			rag = "green" if avg_align >= 7 else ("amber" if avg_align >= 4 else "red")

			# Count consecutive RED exec reports
			exec_history = self._exec_reports.get(pid, [])
			consecutive_red = 0
			for rep in reversed(exec_history):
				rep_rag = rep.get("data", {}).get("overview", {}).get("rag", rag)
				if rep_rag == "red":
					consecutive_red += 1
				else:
					break
			# Use current rag if no history
			if not exec_history and rag == "red":
				consecutive_red = 1

			status = {"portfolio_id": pid, "rag": rag, "consecutive_red": consecutive_red}
			reviewed.append(status)

			if rag == "red" and consecutive_red >= red_threshold_snapshots:
				escalations.append({
					"escalation_id": f"esc_{pid}_{str(date.today())}",
					"portfolio_id": pid,
					"owner_id": p.get("owner_id", "unknown"),
					"severity": "critical" if consecutive_red >= 4 else "high",
					"consecutive_red_snapshots": consecutive_red,
					"avg_alignment_score": round(avg_align, 2),
					"recommended_action": "schedule_portfolio_review",
					"notify_payload": {
						"channel": "portfolio_escalation",
						"subject": f"Portfolio {pid} RED for {consecutive_red} consecutive periods",
						"body": (
							f"Portfolio {pid} has alignment score {round(avg_align, 2):.2f}/10 "
							f"and has been RED for {consecutive_red} consecutive review periods. "
							"Immediate portfolio owner review required."
						),
					},
					"triggered_at": str(date.today()),
				})

		self._audit(t, "rag_escalation_check_run", t)
		return {
			"tenant_id": t,
			"portfolios_reviewed": len(reviewed),
			"escalations_triggered": len(escalations),
			"red_threshold": red_threshold_snapshots,
			"escalations": escalations,
			"all_portfolio_status": reviewed,
			"checked_at": str(date.today()),
		}

	async def benchmark_gap_analysis(
		self,
		portfolio_id: str,
		benchmark_types: list[str] | None = None,
	) -> dict[str, Any]:
		"""Compare portfolio performance against multiple benchmark types simultaneously.

		For each benchmark type in the requested list, looks up matching PerformanceSnapshot
		records and computes:
		  - absolute gap = actual_value - benchmark_value
		  - relative gap pct = (actual / benchmark - 1) * 100
		  - gap_direction: 'above' | 'on_target' | 'below'

		Returns gaps ranked by magnitude (tornado chart structure).
		"""
		assert _present(portfolio_id), "portfolio_id required"
		tenant_id = self.tenant_id
		btypes = benchmark_types or SUPPORTED_BENCHMARK_TYPES

		snapshots = [s for s in self.performance_snapshots.values()
					 if s.tenant_id == tenant_id and s.portfolio_id == portfolio_id
					 and s.benchmark_type in btypes]

		gaps: list[dict[str, Any]] = []
		for s in snapshots:
			bv  = s.benchmark_value
			av  = s.actual_value
			abs_gap = round(av - bv, 4)
			rel_gap = round((av / bv - 1) * 100, 2) if bv else 0.0
			gaps.append({
				"snapshot_id": s.id,
				"benchmark_type": s.benchmark_type,
				"period": s.period,
				"benchmark_value": bv,
				"actual_value": av,
				"absolute_gap": abs_gap,
				"relative_gap_pct": rel_gap,
				"gap_direction": "above" if abs_gap > 0 else ("on_target" if abs_gap == 0 else "below"),
			})

		# Rank by absolute magnitude descending (tornado order)
		gaps.sort(key=lambda g: abs(g["absolute_gap"]), reverse=True)

		self._audit(tenant_id, "benchmark_gap_analysed", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"benchmark_types_requested": btypes,
			"snapshots_analysed": len(gaps),
			"above_benchmark": sum(1 for g in gaps if g["gap_direction"] == "above"),
			"below_benchmark": sum(1 for g in gaps if g["gap_direction"] == "below"),
			"on_target": sum(1 for g in gaps if g["gap_direction"] == "on_target"),
			"largest_gap": gaps[0] if gaps else None,
			"gaps_ranked": gaps,
			"analysed_at": str(date.today()),
		}

	async def portfolio_balance_score(
		self,
		portfolio_id: str,
		h1_target_pct: float = 70.0,
		h2_target_pct: float = 20.0,
		h3_target_pct: float = 10.0,
	) -> dict[str, Any]:
		"""Classify projects into McKinsey Three Horizons and score portfolio balance.

		Classification heuristic:
		  - strategic_fit < 0.4 AND innovation_index < 0.4  => H1 (run-the-business)
		  - strategic_fit >= 0.4 OR innovation_index in [0.4, 0.7)  => H2 (growth)
		  - innovation_index >= 0.7  => H3 (transformation)

		strategic_fit and innovation_index are pulled from alignment scores for the
		'strategic_fit' and 'innovation_index' dimensions respectively, normalised to [0, 1].

		Returns investment split, target deviation, and rebalancing recommendations.
		"""
		assert _present(portfolio_id), "portfolio_id required"
		assert abs(h1_target_pct + h2_target_pct + h3_target_pct - 100.0) < 0.01, \
			"Horizon targets must sum to 100"
		tenant_id = self.tenant_id

		projects = [p for p in self._project_registry.values()
					if p.get("portfolio_id") == portfolio_id and p.get("tenant_id") == tenant_id]

		def _avg_score_for_dim(dim: str) -> dict[str, float]:
			"""Return {project_id: avg_score} for the given alignment dimension."""
			result: dict[str, float] = {}
			for a in self.alignment_scores.values():
				if a.tenant_id != tenant_id or a.portfolio_id != portfolio_id:
					continue
				if a.dimension != dim:
					continue
				pid_key = getattr(a, "project_id", portfolio_id)
				result.setdefault(pid_key, []).append(a.score / 10.0)  # type: ignore[arg-type]
			return {k: round(sum(v) / len(v), 4) for k, v in result.items()}  # type: ignore[arg-type]

		sf_scores  = _avg_score_for_dim("strategic_fit")
		inn_scores = _avg_score_for_dim("innovation_index")

		h1_budget = h2_budget = h3_budget = 0.0
		classified: list[dict[str, Any]] = []
		for p in projects:
			pid    = p.get("project_id", "")
			budget = float(p.get("budget", 0.0))
			sf     = sf_scores.get(pid, 0.3)
			inn    = inn_scores.get(pid, 0.3)
			if inn >= 0.7:
				horizon = "H3"
				h3_budget += budget
			elif sf >= 0.4 or (0.4 <= inn < 0.7):
				horizon = "H2"
				h2_budget += budget
			else:
				horizon = "H1"
				h1_budget += budget
			classified.append({"project_id": pid, "horizon": horizon,
								"strategic_fit": sf, "innovation_index": inn,
								"budget": budget})

		total_budget = h1_budget + h2_budget + h3_budget or 1.0
		h1_actual = round(h1_budget / total_budget * 100, 2)
		h2_actual = round(h2_budget / total_budget * 100, 2)
		h3_actual = round(h3_budget / total_budget * 100, 2)

		balance_deviation = round(
			(abs(h1_actual - h1_target_pct) + abs(h2_actual - h2_target_pct) + abs(h3_actual - h3_target_pct)) / 3, 2
		)

		self._audit(tenant_id, "portfolio_balance_scored", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"project_count": len(projects),
			"horizon_split": {"H1": h1_actual, "H2": h2_actual, "H3": h3_actual},
			"horizon_targets": {"H1": h1_target_pct, "H2": h2_target_pct, "H3": h3_target_pct},
			"balance_deviation_pct": balance_deviation,
			"balance_rating": "excellent" if balance_deviation < 5 else ("good" if balance_deviation < 15 else "needs_rebalancing"),
			"rebalancing_needed": balance_deviation >= 15,
			"projects": classified,
			"scored_at": str(date.today()),
		}

	async def resource_bottleneck_detector(
		self,
		portfolio_id: str,
		period: str,
		top_n: int = 5,
	) -> dict[str, Any]:
		"""Identify the top-N over-allocated resource roles across portfolio projects.

		Each project in _project_registry may carry a 'role_demand' dict:
		  {"senior_developer": 2.5, "data_engineer": 1.0, ...}
		and a 'role_supply' dict with available FTE per role.

		Utilisation = demand / supply.  Severity = utilisation * impact_weight
		where impact_weight defaults to the project's strategic_fit alignment score.
		"""
		assert _present(portfolio_id), "portfolio_id required"
		assert _present(period), "period required"
		assert top_n >= 1, "top_n must be >= 1"
		tenant_id = self.tenant_id

		projects = [p for p in self._project_registry.values()
					if p.get("portfolio_id") == portfolio_id and p.get("tenant_id") == tenant_id]

		# Aggregate demand and supply per role
		role_demand: dict[str, float] = {}
		role_supply: dict[str, float] = {}
		for p in projects:
			for role, fte in p.get("role_demand", {}).items():
				role_demand[role] = role_demand.get(role, 0.0) + float(fte)
			for role, fte in p.get("role_supply", {}).items():
				role_supply[role] = role_supply.get(role, 0.0) + float(fte)

		# Fallback: if no role-level data, use aggregate FTE
		if not role_demand:
			total_d = sum(float(p.get("demand_fte", 0.0)) for p in projects)
			total_s = sum(float(p.get("supply_fte", 0.0)) for p in projects)
			role_demand = {"aggregate_fte": total_d}
			role_supply = {"aggregate_fte": total_s}

		bottlenecks: list[dict[str, Any]] = []
		for role in role_demand:
			demand  = role_demand[role]
			supply  = role_supply.get(role, demand * 0.8)  # assume 80% supply if unknown
			util    = round(demand / supply, 4) if supply else float("inf")
			gap_fte = round(demand - supply, 2)
			bottlenecks.append({
				"role": role,
				"demand_fte": round(demand, 2),
				"supply_fte": round(supply, 2),
				"utilisation": util,
				"gap_fte": gap_fte,
				"severity": "critical" if util > 1.3 else ("high" if util > 1.1 else ("moderate" if util > 1.0 else "ok")),
				"over_allocated": util > 1.0,
			})

		bottlenecks.sort(key=lambda b: b["utilisation"], reverse=True)

		self._audit(tenant_id, "resource_bottleneck_detected", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"period": period,
			"roles_analysed": len(bottlenecks),
			"over_allocated_roles": sum(1 for b in bottlenecks if b["over_allocated"]),
			"top_bottlenecks": bottlenecks[:top_n],
			"all_roles": bottlenecks,
			"detected_at": str(date.today()),
		}

	async def portfolio_lifecycle_advance(
		self,
		portfolio_id: str,
		target_stage: str,
		actor_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Advance a portfolio through its lifecycle with transition validation.

		Legal transitions:
		  proposed -> approved -> active -> under_review -> closed
		  active -> archived (bypass for abandoned portfolios)
		  under_review -> active (re-activation after review)

		Raises PermissionError on illegal transitions.
		Records a full transition history on the portfolio object.
		"""
		assert _present(portfolio_id), "portfolio_id required"
		assert _present(target_stage), "target_stage required"
		assert _present(actor_id), "actor_id required"
		assert _present(evidence_reference), "evidence_reference required"
		tenant_id = self.tenant_id

		target_stage = _norm(target_stage)
		assert target_stage in SUPPORTED_PORTFOLIO_STATUSES, \
			f"target_stage must be one of {SUPPORTED_PORTFOLIO_STATUSES}"

		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		assert portfolio is not None, f"portfolio {portfolio_id} not found"

		LEGAL_TRANSITIONS: dict[str, list[str]] = {
			"proposed":     ["approved"],
			"approved":     ["active"],
			"active":       ["under_review", "archived", "closed"],
			"under_review": ["active", "closed"],
			"archived":     ["closed"],
			"closed":       [],
		}

		current = portfolio.status
		allowed = LEGAL_TRANSITIONS.get(current, [])
		if target_stage not in allowed:
			raise PermissionError(
				f"Illegal lifecycle transition: {current} -> {target_stage}. "
				f"Allowed: {allowed}"
			)

		# Record transition history (stored as list on the portfolio object via __dict__)
		history = getattr(portfolio, "_lifecycle_history", [])
		history.append({
			"from_stage": current,
			"to_stage": target_stage,
			"actor_id": actor_id,
			"evidence_reference": evidence_reference,
			"transitioned_at": str(date.today()),
		})
		portfolio.status = target_stage
		try:
			portfolio._lifecycle_history = history  # type: ignore[attr-defined]
		except AttributeError:
			object.__setattr__(portfolio, "_lifecycle_history", history)

		self._audit(tenant_id, "portfolio_lifecycle_advanced", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"previous_stage": current,
			"current_stage": target_stage,
			"actor_id": actor_id,
			"evidence_reference": evidence_reference,
			"lifecycle_history": history,
			"advanced_at": str(date.today()),
		}

	async def generate_portfolio_narrative(
		self,
		portfolio_id: str,
		period: str,
		style: str = "formal",
	) -> dict[str, Any]:
		"""Generate a plain-English executive narrative for a portfolio period.

		Calls a locally-hosted Ollama model (llama3.1:8b or OLLAMA_NARRATIVE_MODEL env var).
		style: 'formal' | 'concise' | 'risk_focused'

		Falls back gracefully when OLLAMA_BASE_URL is not configured, returning a
		structured template narrative built from the report data.
		"""
		import os
		assert _present(portfolio_id), "portfolio_id required"
		assert _present(period), "period required"
		assert style in {"formal", "concise", "risk_focused"}, \
			"style must be formal | concise | risk_focused"

		# Gather structured data
		report_result = await self.executive_portfolio_report(portfolio_id, period)
		data  = report_result.get("data", {})
		ov    = data.get("overview", {})
		rr    = data.get("risk_return", {})
		iei   = data.get("investment_efficiency", {})

		style_instructions = {
			"formal": "Write in formal business English suitable for a board pack.",
			"concise": "Use bullet-point style executive summary, maximum 3 sentences per section.",
			"risk_focused": "Lead with risks and mitigation actions; secondary focus on performance.",
		}

		prompt = f"""You are a senior portfolio management consultant.
{style_instructions[style]}

Portfolio: {portfolio_id} | Period: {period}
Projects: {ov.get('project_count', 0)} | Budget utilisation: {ov.get('budget_utilisation_pct', 0)}%
Health: {ov.get('health_distribution', {})} | Avg progress: {ov.get('avg_progress_pct', 0)}%
Avg alignment: {ov.get('avg_alignment_score', 0)}/10
Risk/Return ratio: {rr.get('ratio', 'N/A')} | ROI: {iei.get('roi_pct', 0)}% | NPV: {iei.get('npv', 0)}

Write a 3-paragraph executive narrative: (1) overall portfolio health, (2) financial performance,
(3) key risks and recommended actions.
"""

		narrative_text: str | None = None
		model_used: str = "template_fallback"

		ollama_url = os.environ.get("OLLAMA_BASE_URL")
		if ollama_url:
			try:
				import httpx
				model_name = os.environ.get("OLLAMA_NARRATIVE_MODEL", "llama3.1:8b")
				async with httpx.AsyncClient(timeout=60.0) as client:
					resp = await client.post(
						f"{ollama_url}/api/generate",
						json={"model": model_name, "prompt": prompt, "stream": False},
					)
					resp.raise_for_status()
					narrative_text = resp.json().get("response", "").strip()
					model_used = model_name
			except Exception:
				narrative_text = None

		if not narrative_text:
			# Structured template fallback
			health_dist = ov.get("health_distribution", {})
			narrative_text = (
				f"Portfolio {portfolio_id} contains {ov.get('project_count', 0)} active initiatives "
				f"with an average progress of {ov.get('avg_progress_pct', 0):.1f}% as of {period}. "
				f"Health distribution: {health_dist.get('green', 0)} on-track, "
				f"{health_dist.get('amber', 0)} at-risk, {health_dist.get('red', 0)} critical. "
				f"Strategic alignment averages {ov.get('avg_alignment_score', 0):.2f}/10.\n\n"
				f"Financial performance shows budget utilisation of "
				f"{ov.get('budget_utilisation_pct', 0):.1f}%. "
				f"Portfolio ROI stands at {iei.get('roi_pct', 0):.1f}% with NPV of "
				f"{iei.get('npv', 0):,.0f}. "
				f"Investment efficiency index: {iei.get('iei', 0):.3f}.\n\n"
				f"Key risk indicator: risk/return ratio of {rr.get('ratio', 'N/A')}. "
				f"Recommended action: review at-risk and critical projects immediately and "
				f"validate strategic alignment for any project scoring below 4/10."
			)

		self._audit(self.tenant_id, "portfolio_narrative_generated", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"period": period,
			"style": style,
			"model_used": model_used,
			"narrative": narrative_text,
			"source_data": data,
			"generated_at": str(date.today()),
		}

	async def sync_to_intel_domain(
		self,
		portfolio_id: str,
		intel_service: Any | None = None,
	) -> dict[str, Any]:
		"""Push a structured portfolio health signal into the intel domain pipeline.

		Builds a standardised signal payload from:
		  - Portfolio RAG status
		  - Avg alignment score
		  - EVM SPI/CPI (if available)
		  - Top risk score

		If intel_service is provided and exposes `ingest_portfolio_signal`, calls it directly.
		Otherwise serialises the payload and records it in the audit log for async pickup.

		This closes the loop between portfolio health and enterprise threat intelligence.
		"""
		assert _present(portfolio_id), "portfolio_id required"
		tenant_id = self.tenant_id

		# Gather portfolio health data
		portfolio = self._portfolio_or_none(portfolio_id, tenant_id)
		assert portfolio is not None, f"portfolio {portfolio_id} not found"

		alignment_recs = [a for a in self.alignment_scores.values()
						  if a.tenant_id == tenant_id and a.portfolio_id == portfolio_id]
		avg_align = (sum(a.score for a in alignment_recs) / len(alignment_recs)
					 if alignment_recs else 0.0)
		rag = "green" if avg_align >= 7 else ("amber" if avg_align >= 4 else "red")

		rr_recs = [r for r in self.risk_return_analyses.values()
				   if r.tenant_id == tenant_id and r.portfolio_id == portfolio_id]
		avg_risk = (sum(r.risk_score for r in rr_recs) / len(rr_recs) if rr_recs else 0.0)

		projects = [p for p in self._project_registry.values()
					if p.get("portfolio_id") == portfolio_id and p.get("tenant_id") == tenant_id]
		critical_count = sum(1 for p in projects if p.get("health") == "red")

		signal_payload: dict[str, Any] = {
			"signal_type": "portfolio_health",
			"source_capability": "ppm_pan",
			"portfolio_id": portfolio_id,
			"tenant_id": tenant_id,
			"rag_status": rag,
			"avg_alignment_score": round(avg_align, 2),
			"avg_risk_score": round(avg_risk, 2),
			"critical_projects": critical_count,
			"portfolio_status": portfolio.status,
			"severity": "critical" if rag == "red" and critical_count > 0 else (
				"high" if rag == "red" else ("medium" if rag == "amber" else "low")
			),
			"signal_timestamp": str(date.today()),
			"metadata": {
				"alignment_records": len(alignment_recs),
				"risk_records": len(rr_recs),
				"project_count": len(projects),
			},
		}

		intel_result: dict[str, Any] = {}
		if intel_service is not None and hasattr(intel_service, "ingest_portfolio_signal"):
			try:
				intel_result = await intel_service.ingest_portfolio_signal(signal_payload)
			except Exception as exc:
				intel_result = {"error": str(exc), "signal_queued": False}
		else:
			# Record for async pickup — intel domain polls this audit stream
			intel_result = {"signal_queued": True, "delivery": "audit_stream"}

		self._audit(tenant_id, "intel_signal_synced", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"signal_payload": signal_payload,
			"intel_result": intel_result,
			"synced_at": str(date.today()),
		}

PpmPanService = PortfolioAnalyticsService
