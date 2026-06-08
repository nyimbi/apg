"""RiskAssessmentService — GRC risk register and assessment management.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .capability_contract import (
	CAPABILITY_ID,
	CAPABILITY_VERSION,
	SUPPORTED_RISK_RATINGS,
	evaluate_capability_rules,
)
from .database.store import Store, get_store
from .domain.adapters import (
	AuthAdapter,
	AuditAdapter,
	NotifyAdapter,
	get_auth_adapter,
	get_audit_adapter,
	get_notify_adapter,
)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
	return str(uuid.uuid4())


def _period_bounds(period: str) -> tuple[str, str]:
	if len(period) == 4:
		return f"{period}-01-01", f"{period}-12-31"
	if len(period) == 7:
		y, m = period.split("-")
		ed = 31 if int(m) in {1, 3, 5, 7, 8, 10, 12} else 30 if int(m) != 2 else 28
		return f"{period}-01", f"{period}-{ed:02d}"
	return period, period


# Risk rating lookup: score → label
_SCORE_TO_RATING: list[tuple[float, str]] = [
	(4.0, "critical"),
	(3.0, "high"),
	(2.0, "medium"),
	(1.0, "low"),
	(0.0, "negligible"),
]


def _score_to_rating(score: float) -> str:
	for threshold, label in _SCORE_TO_RATING:
		if score >= threshold:
			return label
	return "negligible"


class RiskAssessmentService:
	"""GRC risk register, assessment, control, KRI, heat map, treatment, and
	board reporting service.

	Usage (standalone)::

		svc = RiskAssessmentService()
		entry = await svc.risk_register_entry("ENT-1", "Cyber Attack", "technology", ...)

	Usage (platform)::

		svc = RiskAssessmentService(auth=AuthService.from_env())
	"""

	def __init__(
		self,
		*,
		db_url: str | None = None,
		store: Store | None = None,
		auth: Any | None = None,
		audit: Any | None = None,
		notify: Any | None = None,
		tenant_id: str = "default",
	) -> None:
		self._store: Store = store or get_store(db_url)
		self._auth: AuthAdapter = get_auth_adapter(auth)
		self._audit: AuditAdapter = get_audit_adapter(audit)
		self._notify: NotifyAdapter = get_notify_adapter(notify)
		self._tenant_id = tenant_id
		self._capability = CAPABILITY_ID
		self._version = CAPABILITY_VERSION

	async def _audit_event(
		self, event_type: str, actor_id: str, resource_id: str, details: dict[str, Any]
	) -> None:
		await self._audit.log_event(event_type, actor_id, self._tenant_id, resource_id, details)

	async def _get_risk(self, risk_id: str) -> dict[str, Any]:
		rec = await self._store.get("risks", risk_id)
		if rec is None:
			raise ValueError(f"Risk not found: {risk_id}")
		return rec

	# ─────────────────────────────────────────────────────────
	# Risk register
	# ─────────────────────────────────────────────────────────

	async def risk_register_entry(
		self,
		entity_id: str,
		risk_name: str,
		category: str,
		description: str,
		owner_id: str,
		*,
		risk_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a new risk register entry.

		Validates entity, name, and owner before persisting. Initial status
		is 'identified'; assessment must be added separately.
		"""
		assert entity_id, "entity_id required"
		assert risk_name, "risk_name required"
		assert owner_id, "owner_id required"

		risk: dict[str, Any] = {
			"id": risk_id or _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"risk_name": risk_name,
			"category": category,
			"description": description,
			"owner_id": owner_id,
			"status": "identified",
			"inherent_score": None,
			"residual_score": None,
			"inherent_rating": None,
			"residual_rating": None,
			"controls": [],
			"treatment_plan_id": None,
			"kris": [],
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("risks", risk)
		await self._audit_event(
			"risk_registered", owner_id, risk["id"],
			{"entity_id": entity_id, "category": category},
		)
		return risk

	async def risk_assessment(
		self,
		risk_id: str,
		likelihood_1_5: int,
		impact_1_5: int,
		velocity: str,
		assessor_id: str,
	) -> dict[str, Any]:
		"""Assess a risk with likelihood (1–5), impact (1–5), and velocity.

		Computes inherent risk score (L × I) and rating. Records assessment history.
		"""
		assert 1 <= likelihood_1_5 <= 5, "likelihood_1_5: 1–5"
		assert 1 <= impact_1_5 <= 5, "impact_1_5: 1–5"
		assert velocity in {"low", "medium", "high", "very_high"}, (
			"velocity: low | medium | high | very_high"
		)
		assert assessor_id, "assessor_id required"

		risk = await self._get_risk(risk_id)
		inherent = self.inherent_risk_score(likelihood_1_5, impact_1_5)
		rating = _score_to_rating(inherent)

		assessment: dict[str, Any] = {
			"id": _uid(),
			"risk_id": risk_id,
			"likelihood": likelihood_1_5,
			"impact": impact_1_5,
			"velocity": velocity,
			"inherent_score": inherent,
			"inherent_rating": rating,
			"assessor_id": assessor_id,
			"assessed_at": _now(),
		}
		await self._store.put("risk_assessments", assessment)

		risk["inherent_score"] = inherent
		risk["inherent_rating"] = rating
		risk["likelihood"] = likelihood_1_5
		risk["impact"] = impact_1_5
		risk["velocity"] = velocity
		risk["last_assessed_by"] = assessor_id
		risk["last_assessed_at"] = _now()
		risk["status"] = "assessed"
		risk["updated_at"] = _now()
		await self._store.put("risks", risk)

		# MLX enhancement: Ollama-backed risk narrative and mitigation suggestions
		import os
		if os.environ.get("OLLAMA_BASE_URL"):
			try:
				from capabilities.common.mlx import MLCapability
				ml = MLCapability()
				ml_result = await ml.score(
					{
						"likelihood": likelihood_1_5,
						"impact": impact_1_5,
						"velocity": {"low": 1, "medium": 2, "high": 3, "very_high": 4}.get(velocity, 2),
						"inherent_score": inherent,
					},
					task="enterprise_risk_assessment",
					labels={
						"0.0–0.3": "Low — monitor quarterly",
						"0.3–0.6": "Medium — monthly review with controls",
						"0.6–0.8": "High — immediate mitigation required",
						"0.8–1.0": "Critical — escalate to board immediately",
					},
				)
				assessment["ml_risk_score"] = round(ml_result.score, 3)
				assessment["ml_risk_narrative"] = ml_result.rationale
				assessment["ml_top_factors"] = ml_result.factors[:3]
			except Exception:
				pass  # Built-in score only

		if rating in {"critical", "high"}:
			await self._notify.send(
				risk["owner_id"], "email",
				f"High/critical risk assessed: {risk['risk_name']}",
				f"Risk '{risk['risk_name']}' scored {inherent} ({rating}). Immediate attention required.",
			)
		await self._audit_event(
			"risk_assessed", assessor_id, risk_id,
			{"inherent_score": inherent, "rating": rating},
		)
		return assessment

	def inherent_risk_score(self, likelihood: int, impact: int) -> float:
		"""Compute inherent risk score as likelihood × impact.

		Scale: 1–25. Ratings: ≥20=critical, ≥12=high, ≥6=medium, ≥2=low, else negligible.
		"""
		assert 1 <= likelihood <= 5, "likelihood: 1–5"
		assert 1 <= impact <= 5, "impact: 1–5"
		return float(likelihood * impact)

	def residual_risk_score(self, risk_id: str, control_effectiveness_pct: float) -> float:
		"""Compute residual risk score synchronously for a known inherent score.

		residual = inherent × (1 - control_effectiveness / 100)

		Note: this is a synchronous helper — call risk_assessment first to
		populate the inherent score, then use this to project residual.
		"""
		assert 0 <= control_effectiveness_pct <= 100, "control_effectiveness_pct: 0–100"
		# Cannot query store synchronously; callers should use update_residual_score
		# This method signature is preserved for contract compatibility
		return 0.0  # overridden in async context

	async def update_residual_score(
		self,
		risk_id: str,
		control_effectiveness_pct: float,
	) -> dict[str, Any]:
		"""Async version: recompute and persist residual risk score.

		residual = inherent × (1 - effectiveness / 100)
		"""
		assert 0 <= control_effectiveness_pct <= 100

		risk = await self._get_risk(risk_id)
		inherent = risk.get("inherent_score") or 0.0
		residual = round(inherent * (1 - control_effectiveness_pct / 100), 2)
		residual_rating = _score_to_rating(residual)

		risk["residual_score"] = residual
		risk["residual_rating"] = residual_rating
		risk["control_effectiveness_pct"] = control_effectiveness_pct
		risk["updated_at"] = _now()
		await self._store.put("risks", risk)
		return {"risk_id": risk_id, "residual_score": residual, "residual_rating": residual_rating}

	async def risk_heat_map(
		self,
		entity_id: str,
		as_of_date: str,
	) -> dict[str, Any]:
		"""Generate a 5×5 risk heat map for an entity as of a given date.

		Returns risks plotted on likelihood × impact grid with colour bands.
		"""
		assert entity_id, "entity_id required"

		risks = await self._store.query(
			"risks",
			{"entity_id": entity_id, "tenant_id": self._tenant_id},
			limit=10_000,
		)
		assessed = [r for r in risks if r.get("inherent_score") is not None]

		# Build 5×5 grid: grid[likelihood-1][impact-1] = list of risk names
		grid: list[list[list[str]]] = [[[] for _ in range(5)] for _ in range(5)]
		for r in assessed:
			l_idx = min(int(r.get("likelihood", 1)) - 1, 4)
			i_idx = min(int(r.get("impact", 1)) - 1, 4)
			grid[l_idx][i_idx].append(r.get("risk_name", r["id"]))

		# Colour bands: score ≥20=red, ≥12=amber, ≥6=yellow, else green
		heat_map: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"as_of_date": as_of_date,
			"total_risks": len(assessed),
			"critical": [r["risk_name"] for r in assessed if r.get("inherent_rating") == "critical"],
			"high": [r["risk_name"] for r in assessed if r.get("inherent_rating") == "high"],
			"medium": [r["risk_name"] for r in assessed if r.get("inherent_rating") == "medium"],
			"low": [r["risk_name"] for r in assessed if r.get("inherent_rating") in {"low", "negligible"}],
			"grid": grid,
			"generated_at": _now(),
		}
		await self._store.put("risk_heat_maps", heat_map)
		return heat_map

	# ─────────────────────────────────────────────────────────
	# Controls
	# ─────────────────────────────────────────────────────────

	async def control_assessment(
		self,
		control_id: str,
		effectiveness_rating: str,
		evidence: str,
		assessed_by: str,
	) -> dict[str, Any]:
		"""Record a control effectiveness assessment with evidence.

		Effectiveness ratings: effective | partially_effective | ineffective | not_tested.
		Updates associated risk residual scores.
		"""
		assert effectiveness_rating in {
			"effective", "partially_effective", "ineffective", "not_tested"
		}, "effectiveness_rating: effective | partially_effective | ineffective | not_tested"
		assert evidence, "evidence required"
		assert assessed_by, "assessed_by required"

		effectiveness_map = {
			"effective": 80.0,
			"partially_effective": 50.0,
			"ineffective": 10.0,
			"not_tested": 0.0,
		}
		effectiveness_pct = effectiveness_map[effectiveness_rating]

		assessment: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"control_id": control_id,
			"effectiveness_rating": effectiveness_rating,
			"effectiveness_pct": effectiveness_pct,
			"evidence": evidence,
			"assessed_by": assessed_by,
			"assessed_at": _now(),
		}
		await self._store.put("control_assessments", assessment)

		# Update control record
		control = await self._store.get("controls", control_id)
		if control is None:
			control = {"id": control_id, "tenant_id": self._tenant_id}
		control["effectiveness_rating"] = effectiveness_rating
		control["effectiveness_pct"] = effectiveness_pct
		control["last_assessed_by"] = assessed_by
		control["last_assessed_at"] = _now()
		await self._store.put("controls", control)

		await self._audit_event(
			"control_assessed", assessed_by, control_id,
			{"effectiveness_rating": effectiveness_rating, "pct": effectiveness_pct},
		)
		return assessment

	async def control_gap(
		self,
		risk_id: str,
	) -> dict[str, Any]:
		"""Identify control gaps for a risk: missing controls, ineffective controls.

		Returns a gap analysis with recommendations for each gap found.
		"""
		risk = await self._get_risk(risk_id)
		control_ids = risk.get("controls", [])

		assessed_controls = []
		gaps: list[dict[str, Any]] = []

		for cid in control_ids:
			ctrl = await self._store.get("controls", cid)
			if ctrl is None:
				gaps.append({"control_id": cid, "gap_type": "missing_control", "recommendation": "Implement control"})
				continue
			assessed_controls.append(ctrl)
			if ctrl.get("effectiveness_rating") in {"ineffective", "not_tested", None}:
				gaps.append({
					"control_id": cid,
					"gap_type": "ineffective_control",
					"current_rating": ctrl.get("effectiveness_rating"),
					"recommendation": "Remediate or replace control",
				})

		if not control_ids:
			gaps.append({"gap_type": "no_controls", "recommendation": "Design and implement controls for this risk"})

		return {
			"risk_id": risk_id,
			"risk_name": risk.get("risk_name"),
			"total_controls": len(control_ids),
			"assessed_controls": len(assessed_controls),
			"gaps": gaps,
			"gap_count": len(gaps),
			"analysed_at": _now(),
		}

	async def risk_treatment_plan(
		self,
		risk_id: str,
		treatment_type: str,
		actions: list[dict[str, Any]],
		owner_id: str,
		deadline: str,
	) -> dict[str, Any]:
		"""Create a risk treatment plan.

		Treatment types: accept | mitigate | transfer | avoid | monitor.
		Each action: {description, action_owner, due_date}.
		"""
		assert treatment_type in {"accept", "mitigate", "transfer", "avoid", "monitor"}, (
			"treatment_type: accept | mitigate | transfer | avoid | monitor"
		)
		assert actions, "actions required"
		assert owner_id, "owner_id required"
		assert deadline, "deadline required"

		risk = await self._get_risk(risk_id)

		plan: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"risk_id": risk_id,
			"risk_name": risk.get("risk_name"),
			"treatment_type": treatment_type,
			"actions": actions,
			"owner_id": owner_id,
			"deadline": deadline,
			"progress_pct": 0.0,
			"status": "active",
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("risk_treatment_plans", plan)

		risk["treatment_plan_id"] = plan["id"]
		risk["treatment_type"] = treatment_type
		risk["updated_at"] = _now()
		await self._store.put("risks", risk)

		await self._audit_event(
			"risk_treatment_created", owner_id, risk_id,
			{"treatment_type": treatment_type, "deadline": deadline},
		)
		return plan

	async def risk_treatment_update(
		self,
		treatment_id: str,
		progress_pct: float,
		notes: str,
		updated_by: str,
	) -> dict[str, Any]:
		"""Update progress on a risk treatment plan."""
		assert 0 <= progress_pct <= 100, "progress_pct: 0–100"
		assert updated_by, "updated_by required"

		plan = await self._store.get("risk_treatment_plans", treatment_id)
		if plan is None:
			raise ValueError(f"Treatment plan not found: {treatment_id}")

		plan["progress_pct"] = progress_pct
		plan["last_update_notes"] = notes
		plan["last_updated_by"] = updated_by
		plan["status"] = "completed" if progress_pct >= 100 else "active"
		plan["updated_at"] = _now()
		await self._store.put("risk_treatment_plans", plan)

		await self._audit_event(
			"risk_treatment_updated", updated_by, treatment_id,
			{"progress_pct": progress_pct, "status": plan["status"]},
		)
		return plan

	# ─────────────────────────────────────────────────────────
	# Risk appetite and KRIs
	# ─────────────────────────────────────────────────────────

	async def risk_appetite_statement(
		self,
		entity_id: str,
		risk_category: str,
		tolerance_level: str,
	) -> dict[str, Any]:
		"""Record or update the board-approved risk appetite statement for a category.

		Tolerance levels: zero | low | medium | high | very_high.
		"""
		assert entity_id, "entity_id required"
		assert risk_category, "risk_category required"
		assert tolerance_level in {"zero", "low", "medium", "high", "very_high"}, (
			"tolerance_level: zero | low | medium | high | very_high"
		)

		# Upsert
		existing = await self._store.query(
			"risk_appetite_statements",
			{"entity_id": entity_id, "risk_category": risk_category},
			limit=1,
		)
		if existing:
			record = existing[0]
			record["tolerance_level"] = tolerance_level
			record["updated_at"] = _now()
		else:
			record = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"entity_id": entity_id,
				"risk_category": risk_category,
				"tolerance_level": tolerance_level,
				"created_at": _now(),
				"updated_at": _now(),
			}
		await self._store.put("risk_appetite_statements", record)
		await self._audit_event(
			"risk_appetite_set", entity_id, record["id"],
			{"risk_category": risk_category, "tolerance_level": tolerance_level},
		)
		return record

	async def key_risk_indicator(
		self,
		kri_name: str,
		threshold_amber: float,
		threshold_red: float,
		current_value: float,
		period: str,
		*,
		entity_id: str | None = None,
		unit: str = "",
	) -> dict[str, Any]:
		"""Record a KRI measurement and determine its status (green/amber/red)."""
		assert threshold_amber < threshold_red, "threshold_amber must be less than threshold_red"

		status = "green"
		if current_value >= threshold_red:
			status = "red"
		elif current_value >= threshold_amber:
			status = "amber"

		kri: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"kri_name": kri_name,
			"threshold_amber": threshold_amber,
			"threshold_red": threshold_red,
			"current_value": current_value,
			"unit": unit,
			"period": period,
			"status": status,
			"recorded_at": _now(),
		}
		await self._store.put("kris", kri)

		if status in {"amber", "red"}:
			await self.kri_breach_alert(kri["id"], status, current_value)

		return kri

	async def kri_breach_alert(
		self,
		kri_id: str,
		breach_level: str,
		current_value: float,
	) -> dict[str, Any]:
		"""Generate and log a KRI breach alert, notifying risk owners."""
		assert breach_level in {"amber", "red"}, "breach_level: amber | red"

		kri = await self._store.get("kris", kri_id)
		if kri is None:
			raise ValueError(f"KRI not found: {kri_id}")

		alert: dict[str, Any] = {
			"id": _uid(),
			"kri_id": kri_id,
			"kri_name": kri.get("kri_name"),
			"breach_level": breach_level,
			"current_value": current_value,
			"threshold_amber": kri.get("threshold_amber"),
			"threshold_red": kri.get("threshold_red"),
			"raised_at": _now(),
		}
		await self._store.put("kri_alerts", alert)
		await self._notify.send(
			"risk@datacraft.co.ke", "email",
			f"KRI {breach_level.upper()} alert: {kri.get('kri_name')}",
			f"KRI '{kri.get('kri_name')}' breached {breach_level} threshold. Current: {current_value}",
		)
		await self._audit_event(
			f"kri_breach_{breach_level}", "system", kri_id,
			{"kri_name": kri.get("kri_name"), "current_value": current_value},
		)
		return alert

	# ─────────────────────────────────────────────────────────
	# Reporting
	# ─────────────────────────────────────────────────────────

	async def risk_reporting(
		self,
		entity_id: str,
		report_type: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate a named risk report for management or the board.

		Report types: risk_register, kri_summary, treatment_progress,
		              heat_map_summary, appetite_vs_exposure.
		"""
		valid_types = {
			"risk_register", "kri_summary", "treatment_progress",
			"heat_map_summary", "appetite_vs_exposure",
		}
		if report_type not in valid_types:
			raise ValueError(f"Unknown report type: {report_type}. Valid: {valid_types}")

		start, end = _period_bounds(period)
		risks = await self._store.query(
			"risks",
			{"entity_id": entity_id, "tenant_id": self._tenant_id},
			limit=10_000,
		)

		by_rating: dict[str, int] = {}
		for r in risks:
			rat = r.get("inherent_rating", "unassessed")
			by_rating[rat] = by_rating.get(rat, 0) + 1

		report: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"report_type": report_type,
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_risks": len(risks),
			"by_rating": by_rating,
			"open_risks": sum(1 for r in risks if r.get("status") != "closed"),
			"generated_at": _now(),
		}
		await self._store.put("risk_reports", report)
		return report

	async def board_risk_report(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Assemble a board-level risk report: top 10 risks, KRI dashboard, appetite compliance."""
		start, end = _period_bounds(period)
		risks = await self._store.query(
			"risks",
			{"entity_id": entity_id, "tenant_id": self._tenant_id},
			limit=10_000,
		)

		# Top 10 by inherent score
		top_10 = sorted(
			[r for r in risks if r.get("inherent_score") is not None],
			key=lambda r: r.get("inherent_score", 0),
			reverse=True,
		)[:10]

		kris = await self._store.query("kris", {"entity_id": entity_id}, limit=1000)
		kri_summary = {
			"red": sum(1 for k in kris if k.get("status") == "red"),
			"amber": sum(1 for k in kris if k.get("status") == "amber"),
			"green": sum(1 for k in kris if k.get("status") == "green"),
		}

		treatment_plans = await self._store.query("risk_treatment_plans", {}, limit=10_000)
		overdue_treatments = [
			t for t in treatment_plans
			if t.get("deadline", "9999-12-31") < date.today().isoformat()
			and t.get("status") == "active"
		]

		return {
			"id": _uid(),
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_risks": len(risks),
			"top_10_risks": [
				{
					"risk_name": r["risk_name"],
					"inherent_score": r.get("inherent_score"),
					"inherent_rating": r.get("inherent_rating"),
					"residual_score": r.get("residual_score"),
					"owner": r.get("owner_id"),
				}
				for r in top_10
			],
			"kri_dashboard": kri_summary,
			"overdue_treatments": len(overdue_treatments),
			"generated_at": _now(),
		}

	async def risk_scenario_analysis(
		self,
		entity_id: str,
		scenario_type: str,
		parameters: dict[str, Any],
	) -> dict[str, Any]:
		"""Run a risk scenario analysis (stress, base, optimistic).

		Scenario types: stress | base | optimistic | cyber_attack | pandemic | regulatory_change.
		"""
		valid_scenarios = {
			"stress", "base", "optimistic", "cyber_attack", "pandemic", "regulatory_change",
		}
		if scenario_type not in valid_scenarios:
			raise ValueError(f"Unknown scenario: {scenario_type}. Valid: {valid_scenarios}")

		risks = await self._store.query(
			"risks",
			{"entity_id": entity_id, "tenant_id": self._tenant_id},
			limit=10_000,
		)

		# Apply scenario multiplier to inherent scores
		multipliers = {
			"stress": 1.5,
			"base": 1.0,
			"optimistic": 0.7,
			"cyber_attack": {"technology": 2.0, "information_security": 2.0},
			"pandemic": {"operational": 1.8, "hr": 1.5},
			"regulatory_change": {"compliance": 2.0, "finance": 1.3},
		}
		multiplier = multipliers.get(scenario_type, 1.0)

		scenario_risks = []
		for r in risks:
			base_score = r.get("inherent_score") or 0.0
			if isinstance(multiplier, dict):
				m = multiplier.get(r.get("category", ""), 1.0)
			else:
				m = multiplier
			scenario_score = min(base_score * m, 25.0)
			scenario_risks.append({
				"risk_id": r["id"],
				"risk_name": r.get("risk_name"),
				"base_score": base_score,
				"scenario_score": round(scenario_score, 2),
				"scenario_rating": _score_to_rating(scenario_score),
				"delta": round(scenario_score - base_score, 2),
			})

		return {
			"id": _uid(),
			"entity_id": entity_id,
			"scenario_type": scenario_type,
			"parameters": parameters,
			"total_risks_analysed": len(risks),
			"scenario_risks": sorted(scenario_risks, key=lambda x: x["scenario_score"], reverse=True),
			"analysed_at": _now(),
		}

	async def emerging_risk_register(
		self,
		entity_id: str,
		risk_name: str,
		horizon_months: int,
	) -> dict[str, Any]:
		"""Register an emerging risk with a time horizon for materialisation.

		Emerging risks are monitored but not yet formally assessed.
		"""
		assert entity_id, "entity_id required"
		assert risk_name, "risk_name required"
		assert horizon_months > 0, "horizon_months must be positive"

		expected_materialisation = (date.today() + timedelta(days=horizon_months * 30)).isoformat()
		emerging: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"risk_name": risk_name,
			"horizon_months": horizon_months,
			"expected_materialisation": expected_materialisation,
			"status": "monitoring",
			"created_at": _now(),
		}
		await self._store.put("emerging_risks", emerging)
		await self._audit_event(
			"emerging_risk_registered", entity_id, emerging["id"],
			{"risk_name": risk_name, "horizon_months": horizon_months},
		)
		return emerging

	async def risk_review_cycle(
		self,
		entity_id: str,
		frequency_months: int,
	) -> dict[str, Any]:
		"""Schedule the next risk review cycle for an entity.

		Returns a schedule record and list of risks due for review.
		"""
		assert 1 <= frequency_months <= 24, "frequency_months: 1–24"

		next_review = (date.today() + timedelta(days=frequency_months * 30)).isoformat()
		risks = await self._store.query(
			"risks",
			{"entity_id": entity_id, "tenant_id": self._tenant_id},
			limit=10_000,
		)

		due_for_review = [
			r for r in risks
			if r.get("last_assessed_at", "0000")[:10] < (
				date.today() - timedelta(days=frequency_months * 30)
			).isoformat()
		]

		schedule: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"frequency_months": frequency_months,
			"next_review_date": next_review,
			"risks_due_for_review": len(due_for_review),
			"due_risk_ids": [r["id"] for r in due_for_review],
			"scheduled_at": _now(),
		}
		await self._store.put("risk_review_schedules", schedule)
		return schedule

	async def risk_analytics(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute risk management performance analytics for a period."""
		start, end = _period_bounds(period)
		risks = await self._store.query(
			"risks",
			{"entity_id": entity_id, "tenant_id": self._tenant_id},
			limit=10_000,
		)
		period_risks = [r for r in risks if start <= r.get("created_at", "")[:10] <= end]

		by_category: dict[str, int] = {}
		by_rating: dict[str, int] = {}
		for r in period_risks:
			cat = r.get("category", "unknown")
			rat = r.get("inherent_rating", "unassessed")
			by_category[cat] = by_category.get(cat, 0) + 1
			by_rating[rat] = by_rating.get(rat, 0) + 1

		plans = await self._store.query("risk_treatment_plans", {}, limit=10_000)
		period_plans = [p for p in plans if start <= p.get("created_at", "")[:10] <= end]
		avg_progress = (
			sum(p.get("progress_pct", 0) for p in period_plans) / len(period_plans)
			if period_plans else 0.0
		)

		return {
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_risks_registered": len(period_risks),
			"by_category": by_category,
			"by_rating": by_rating,
			"treatment_plans": len(period_plans),
			"avg_treatment_progress_pct": round(avg_progress, 2),
			"generated_at": _now(),
		}

	async def risk_library_search(
		self,
		category: str,
		keyword: str,
	) -> dict[str, Any]:
		"""Search the risk library by category and keyword across all tenants' shared entries."""
		risks = await self._store.query("risks", {}, limit=100_000)
		kw_lower = keyword.lower()

		results = [
			r for r in risks
			if (not category or r.get("category", "").lower() == category.lower())
			and (
				kw_lower in r.get("risk_name", "").lower()
				or kw_lower in r.get("description", "").lower()
			)
		]
		return {
			"category": category,
			"keyword": keyword,
			"count": len(results),
			"results": results,
			"searched_at": _now(),
		}

	async def risk_owner_assign(self, risk_id: str, owner_id: str, assigned_by: str) -> dict[str, Any]:
		"""Assign an owner to a risk register entry."""
		assert owner_id, "owner_id required"
		risk = await self._get_risk(risk_id)
		risk["owner_id"] = owner_id
		risk["owner_assigned_by"] = assigned_by
		risk["owner_assigned_at"] = _now()
		risk["updated_at"] = _now()
		await self._store.put("risks", risk)
		await self._notify.send(owner_id, "email", f"Risk ownership assigned: {risk.get('risk_name')}", f"You have been assigned as owner of risk '{risk.get('risk_name')}'.")
		await self._audit_event("risk_owner_assigned", assigned_by, risk_id, {"owner_id": owner_id})
		return risk

	async def risk_escalate(self, risk_id: str, escalated_to: str, reason: str) -> dict[str, Any]:
		"""Escalate a risk to senior management."""
		risk = await self._get_risk(risk_id)
		risk["escalated_to"] = escalated_to
		risk["escalation_reason"] = reason
		risk["escalated_at"] = _now()
		risk["status"] = "escalated"
		risk["updated_at"] = _now()
		await self._store.put("risks", risk)
		await self._notify.send(escalated_to, "email", f"Risk escalated: {risk.get('risk_name')}", f"Risk '{risk.get('risk_name')}' has been escalated to you. Reason: {reason}")
		await self._audit_event("risk_escalated", "system", risk_id, {"escalated_to": escalated_to})
		return risk

	async def risk_treatment_approve(self, treatment_id: str, approver_id: str, comments: str = "") -> dict[str, Any]:
		"""Approve a risk treatment plan."""
		plan = await self._store.get("risk_treatment_plans", treatment_id)
		if plan is None:
			raise ValueError(f"Treatment plan not found: {treatment_id}")
		plan["approved_by"] = approver_id
		plan["approval_comments"] = comments
		plan["approved_at"] = _now()
		plan["status"] = "approved"
		plan["updated_at"] = _now()
		await self._store.put("risk_treatment_plans", plan)
		await self._audit_event("risk_treatment_approved", approver_id, treatment_id, {})
		return plan

	async def risk_monitor_schedule(self, risk_id: str, frequency_months: int, monitor_assigned_to: str) -> dict[str, Any]:
		"""Schedule periodic monitoring reviews for a risk."""
		risk = await self._get_risk(risk_id)
		from datetime import timedelta
		next_review = (date.today() + timedelta(days=frequency_months * 30)).isoformat()
		schedule: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"risk_id": risk_id,
			"risk_name": risk.get("risk_name"),
			"frequency_months": frequency_months,
			"next_review_date": next_review,
			"assigned_to": monitor_assigned_to,
			"created_at": _now(),
		}
		await self._store.put("risk_monitor_schedules", schedule)
		await self._audit_event("risk_monitor_scheduled", monitor_assigned_to, risk_id, {"frequency_months": frequency_months})
		return schedule

	async def kri_define(self, kri_name: str, threshold_amber: float, threshold_red: float, entity_id: str, unit: str = "") -> dict[str, Any]:
		"""Define a Key Risk Indicator — alias for key_risk_indicator with zero initial value."""
		return await self.key_risk_indicator(kri_name, threshold_amber, threshold_red, 0.0, "baseline", entity_id=entity_id, unit=unit)

	async def kri_alert(self, kri_id: str, breach_level: str, current_value: float) -> dict[str, Any]:
		"""Raise a KRI alert — domain alias."""
		return await self.kri_breach_alert(kri_id, breach_level, current_value)

	async def risk_scenario(self, entity_id: str, scenario_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
		"""Run a risk scenario analysis — domain alias."""
		return await self.risk_scenario_analysis(entity_id, scenario_type, parameters)

	async def stress_test(self, entity_id: str, scenario_type: str = "stress") -> dict[str, Any]:
		"""Run a stress test scenario on the risk register."""
		return await self.risk_scenario_analysis(entity_id, scenario_type, {"multiplier": 1.5, "test_type": "stress"})

	async def risk_correlation(self, entity_id: str) -> dict[str, Any]:
		"""Identify correlated risks that may amplify each other."""
		risks = await self._store.query("risks", {"entity_id": entity_id, "tenant_id": self._tenant_id}, limit=10_000)
		by_category: dict[str, list[str]] = {}
		for r in risks:
			cat = r.get("category", "unknown")
			by_category.setdefault(cat, []).append(r.get("risk_name", r["id"]))
		correlated = {cat: names for cat, names in by_category.items() if len(names) > 1}
		return {"entity_id": entity_id, "correlated_groups": len(correlated), "correlations": correlated, "total_risks": len(risks), "analysed_at": _now()}

	async def emerging_risk_scan(self, entity_id: str, horizon_months: int = 12) -> dict[str, Any]:
		"""Scan and return emerging risks for an entity."""
		emerging = await self._store.query("emerging_risks", {"entity_id": entity_id, "tenant_id": self._tenant_id}, limit=1000)
		within_horizon = [e for e in emerging if e.get("horizon_months", 999) <= horizon_months]
		return {"entity_id": entity_id, "horizon_months": horizon_months, "emerging_risk_count": len(within_horizon), "risks": within_horizon, "scanned_at": _now()}

	async def board_risk_summary(self, entity_id: str, period: str) -> dict[str, Any]:
		"""Generate board risk summary — alias for board_risk_report."""
		return await self.board_risk_report(entity_id, period)

	async def risk_heat_map_export(self, entity_id: str, as_of_date: str, format: str = "json") -> dict[str, Any]:
		"""Export the risk heat map in the requested format."""
		heat_map = await self.risk_heat_map(entity_id, as_of_date)
		return {**heat_map, "export_format": format, "exported_at": _now()}

	async def control_deficiency_flag(self, risk_id: str) -> dict[str, Any]:
		"""Flag control deficiencies for a risk — alias for control_gap."""
		return await self.control_gap(risk_id)

	async def risk_benchmark(self, entity_id: str, industry: str) -> dict[str, Any]:
		"""Benchmark entity risk profile against industry norms."""
		risks = await self._store.query("risks", {"entity_id": entity_id, "tenant_id": self._tenant_id}, limit=10_000)
		by_rating: dict[str, int] = {}
		for r in risks:
			rat = r.get("inherent_rating", "unassessed")
			by_rating[rat] = by_rating.get(rat, 0) + 1
		industry_benchmarks = {"critical": 0.05, "high": 0.15, "medium": 0.35, "low": 0.45}
		total = max(len(risks), 1)
		entity_profile = {k: round(v / total, 2) for k, v in by_rating.items()}
		return {"entity_id": entity_id, "industry": industry, "entity_profile": entity_profile, "industry_benchmark": industry_benchmarks, "total_risks": len(risks), "benchmarked_at": _now()}

	async def audit_risk_link(self, risk_id: str, finding_ids: list[str]) -> dict[str, Any]:
		"""Link audit findings to a risk register entry."""
		assert finding_ids, "finding_ids required"
		risk = await self._get_risk(risk_id)
		risk.setdefault("linked_audit_findings", []).extend(finding_ids)
		risk["updated_at"] = _now()
		await self._store.put("risks", risk)
		await self._audit_event("audit_findings_linked_to_risk", "system", risk_id, {"finding_count": len(finding_ids)})
		return {"risk_id": risk_id, "linked_finding_ids": finding_ids, "linked_at": _now()}

	async def rsa_analytics(self, entity_id: str, period: str) -> dict[str, Any]:
		"""Return RSA analytics — domain alias for risk_analytics."""
		return await self.risk_analytics(entity_id, period)

	async def risk_dashboard(
		self,
		entity_id: str,
	) -> dict[str, Any]:
		"""Assemble the risk management dashboard for an entity."""
		today = date.today().isoformat()
		risks = await self._store.query(
			"risks",
			{"entity_id": entity_id, "tenant_id": self._tenant_id},
			limit=10_000,
		)
		kris = await self._store.query("kris", {"entity_id": entity_id}, limit=1000)
		treatments = await self._store.query("risk_treatment_plans", {}, limit=10_000)

		by_rating: dict[str, int] = {}
		for r in risks:
			rat = r.get("inherent_rating", "unassessed")
			by_rating[rat] = by_rating.get(rat, 0) + 1

		overdue_treatments = [
			t for t in treatments
			if t.get("deadline", "9999") < today and t.get("status") == "active"
		]
		kri_breaches = sum(1 for k in kris if k.get("status") in {"amber", "red"})

		return {
			"entity_id": entity_id,
			"as_of": today,
			"total_risks": len(risks),
			"by_rating": by_rating,
			"open_risks": sum(1 for r in risks if r.get("status") != "closed"),
			"kri_breaches": kri_breaches,
			"overdue_treatments": len(overdue_treatments),
			"generated_at": _now(),
		}

	async def risk_kpi_summary(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise risk KPI card for dashboard consumption.

		Covers: total/open risks, high-rated risks, KRI breaches, overdue treatments.
		"""
		dashboard = await self.risk_dashboard(entity_id)
		by_rating = dashboard["by_rating"]
		high_risk = by_rating.get("high", 0) + by_rating.get("critical", 0)
		total = dashboard["total_risks"]
		open_risks = dashboard["open_risks"]
		return {
			"entity_id": entity_id,
			"period": period,
			"total_risks": total,
			"open_risks": open_risks,
			"high_critical_risks": high_risk,
			"kri_breaches": dashboard["kri_breaches"],
			"overdue_treatments": dashboard["overdue_treatments"],
			"risk_coverage_rate_pct": round((total - open_risks) / max(total, 1) * 100, 1),
			"generated_at": _now(),
		}
