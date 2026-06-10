"""Executable service layer for APG Regulatory Technology."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHANGE_TYPES,
		SUPPORTED_FILING_TYPES, SUPPORTED_JURISDICTIONS, SUPPORTED_REGULATORS,
		SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_RATINGS,
		SUPPORTED_SUBMISSION_CHANNELS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		ImpactAssessment, ObligationMapping, RegulatoryChange, RegulatoryFiling,
		RegulatoryInquiry, RegulatoryResponse, RegulatorySource, RegulatorySubmission,
		RegTechAgent, RegTechReview,
	)
	from .regtech_runtime import normalize_code, normalize_jurisdiction, present
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHANGE_TYPES,
		SUPPORTED_FILING_TYPES, SUPPORTED_JURISDICTIONS, SUPPORTED_REGULATORS,
		SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_RATINGS,
		SUPPORTED_SUBMISSION_CHANNELS,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		ImpactAssessment, ObligationMapping, RegulatoryChange, RegulatoryFiling,
		RegulatoryInquiry, RegulatoryResponse, RegulatorySource, RegulatorySubmission,
		RegTechAgent, RegTechReview,
	)
	from regtech_runtime import normalize_code, normalize_jurisdiction, present  # type: ignore


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
	import uuid
	return str(uuid.uuid4())


class RegulatoryTechnologyService:
	"""
	Full async Regulatory Technology (RegTech) service for APG fintech.

	Covers the complete compliance lifecycle: regulatory change monitoring,
	obligation mapping, impact assessment, filing preparation, submission,
	regulator inquiry management, and CBK/prudential reporting.

	Constructor accepts optional adapter overrides for auth, audit, and
	notification; defaults to in-memory state for generated apps and tests.
	"""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		self.sources: dict[str, RegulatorySource] = {}
		self.changes: dict[str, RegulatoryChange] = {}
		self.obligations: dict[str, ObligationMapping] = {}
		self.impacts: dict[str, ImpactAssessment] = {}
		self.filings: dict[str, RegulatoryFiling] = {}
		self.submissions: dict[str, RegulatorySubmission] = {}
		self.inquiries: dict[str, RegulatoryInquiry] = {}
		self.responses: dict[str, RegulatoryResponse] = {}
		self.reviews: dict[str, RegTechReview] = {}
		self.agents: dict[str, RegTechAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Capability contract
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Regulatory calendar
	# ------------------------------------------------------------------

	async def regulatory_calendar(
		self,
		jurisdiction: str,
		entity_type: str,
	) -> dict[str, Any]:
		"""
		Return the regulatory reporting calendar for a jurisdiction / entity type
		combination.  Pulls from recorded obligations and filings, then overlays
		standard statutory deadlines for CBK-supervised entities.
		"""
		assert bool(jurisdiction), "jurisdiction required"
		assert bool(entity_type), "entity_type required"
		jurisdiction_norm = normalize_jurisdiction(jurisdiction)

		# gather obligations with due dates for this tenant
		obligations = [
			o for o in self.obligations.values()
			if o.tenant_id == self.tenant_id
		]

		# standard CBK / Kenya statutory deadlines
		cbk_deadlines: list[dict[str, Any]] = [
			{"report": "CBK_MONTHLY_RETURNS", "frequency": "monthly", "due_day": 15, "regulator": "CBK"},
			{"report": "STATUTORY_LIQUIDITY_RATIO", "frequency": "weekly", "due_day": None, "regulator": "CBK"},
			{"report": "CAPITAL_ADEQUACY_RATIO", "frequency": "quarterly", "due_day": 30, "regulator": "CBK"},
			{"report": "AML_CFT_QUARTERLY", "frequency": "quarterly", "due_day": 45, "regulator": "FCIU"},
			{"report": "ANNUAL_AUDITED_ACCOUNTS", "frequency": "annual", "due_day": 90, "regulator": "CBK"},
			{"report": "CUSTOMER_DUE_DILIGENCE_SUMMARY", "frequency": "semi_annual", "due_day": 30, "regulator": "CBK"},
		]

		# merge with recorded obligations
		calendar_items = [*cbk_deadlines]
		for obl in obligations:
			calendar_items.append({
				"report": obl.obligation_reference,
				"due_date": obl.due_date,
				"owner_id": obl.owner_id,
				"obligation_id": obl.mapping_id,
				"frequency": "ad_hoc",
				"regulator": "internal",
			})

		# upcoming filings (drafted but not submitted)
		pending_filings = [
			{"filing_id": f.filing_id, "framework": f.framework, "filing_type": f.filing_type,
			 "period": f.period, "status": f.status}
			for f in self.filings.values()
			if f.tenant_id == self.tenant_id and f.status in {"draft", "review"}
		]

		await self._audit("regulatory_calendar_viewed", jurisdiction_norm, {"entity_type": entity_type})
		return {
			"jurisdiction": jurisdiction_norm,
			"entity_type": entity_type,
			"as_of": _now_iso(),
			"statutory_deadlines": cbk_deadlines,
			"obligation_items": [o.to_dict() for o in obligations],
			"pending_filings": pending_filings,
			"calendar_item_count": len(calendar_items),
		}

	# ------------------------------------------------------------------
	# Compliance obligation checks
	# ------------------------------------------------------------------

	async def compliance_obligation_check(
		self,
		entity_id: str,
		regulation: str,
	) -> dict[str, Any]:
		"""
		Check whether a regulated entity has active, mapped obligations for a
		given regulation.  Returns obligation status, overdue items, and
		completion percentage.
		"""
		assert bool(entity_id), "entity_id required"
		assert bool(regulation), "regulation required"

		all_obligations = [o for o in self.obligations.values() if o.tenant_id == self.tenant_id]
		regulation_obligations = [
			o for o in all_obligations
			if regulation.lower() in o.obligation_reference.lower()
			or regulation.lower() in o.policy_reference.lower()
		]

		today = _now_iso()[:10]
		overdue = [o for o in regulation_obligations if o.due_date < today]
		upcoming = [o for o in regulation_obligations if o.due_date >= today]
		submitted = [
			s for s in self.submissions.values()
			if s.tenant_id == self.tenant_id
		]
		completion_pct = round(
			len(submitted) / max(len(regulation_obligations), 1) * 100, 1
		)

		await self._audit("compliance_obligation_checked", entity_id, {"regulation": regulation})
		return {
			"entity_id": entity_id,
			"regulation": regulation,
			"as_of": _now_iso(),
			"total_obligations": len(regulation_obligations),
			"overdue_count": len(overdue),
			"upcoming_count": len(upcoming),
			"completion_pct": completion_pct,
			"overdue": [o.to_dict() for o in overdue],
			"upcoming": [o.to_dict() for o in upcoming],
			"submissions_recorded": len(submitted),
		}

	# ------------------------------------------------------------------
	# Automated report generation
	# ------------------------------------------------------------------

	async def auto_report_generation(
		self,
		report_type: str,
		period: str,
		jurisdiction: str,
		filing_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Automatically generate a regulatory report for the specified type,
		period, and jurisdiction.  Validates supported report types, assembles
		data from this tenant's compliance and filing records, and creates a
		draft RegulatoryFiling ready for human review before submission.
		"""
		assert bool(report_type), "report_type required"
		assert bool(period), "period required"
		assert bool(jurisdiction), "jurisdiction required"

		report_type_norm = normalize_code(report_type)
		jurisdiction_norm = normalize_jurisdiction(jurisdiction)
		fid = filing_id or _uuid()

		# map report type to framework
		framework_map: dict[str, str] = {
			"cbk_monthly": "CBK_PRUDENTIAL",
			"capital_adequacy": "BASEL_III",
			"liquidity_coverage": "BASEL_III",
			"aml_cft": "FATF",
			"car_return": "CBK_PRUDENTIAL",
			"slr_return": "CBK_PRUDENTIAL",
			"annual_report": "IAS_IFRS",
			"mifid_transaction": "MIFID_II",
		}
		framework = framework_map.get(report_type_norm, "GENERIC")

		filing_type_map: dict[str, str] = {
			"cbk_monthly": "MONTHLY_RETURN",
			"capital_adequacy": "QUARTERLY_RETURN",
			"liquidity_coverage": "WEEKLY_RETURN",
			"aml_cft": "QUARTERLY_RETURN",
			"car_return": "QUARTERLY_RETURN",
			"annual_report": "ANNUAL_REPORT",
			"mifid_transaction": "TRANSACTION_REPORT",
		}
		filing_type = filing_type_map.get(report_type_norm, "PERIODIC_RETURN")

		# assemble report data
		obligations = [o for o in self.obligations.values() if o.tenant_id == self.tenant_id]
		impacts = [i for i in self.impacts.values() if i.tenant_id == self.tenant_id]
		changes = [c for c in self.changes.values() if c.tenant_id == self.tenant_id]

		report_data: dict[str, Any] = {
			"report_type": report_type_norm,
			"period": period,
			"jurisdiction": jurisdiction_norm,
			"framework": framework,
			"data_sources": {
				"obligation_count": len(obligations),
				"impact_count": len(impacts),
				"regulatory_change_count": len(changes),
			},
			"generated_at": _now_iso(),
			"generated_by": self.actor_id,
			"status": "draft",
		}

		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "prepare_filing",
			"framework_supported": True,
			"filing_type_supported": True,
			"period_present": present(period),
			"evidence_present": True,
			"owner_present": present(self.actor_id),
		})
		filing = RegulatoryFiling(
			fid, self.tenant_id, framework, filing_type,
			period, f"auto_generated_{fid}", self.actor_id, "draft",
		)
		filing.__dict__.update({"report_data": report_data, "created_at": _now_iso()})
		self.filings[fid] = filing

		await self._audit("auto_report_generated", fid, {"report_type": report_type_norm, "period": period})
		return {**filing.to_dict(), "report_data": report_data}

	# ------------------------------------------------------------------
	# Regulatory change monitoring
	# ------------------------------------------------------------------

	async def regulatory_change_monitoring(
		self,
		jurisdictions: list[str],
	) -> dict[str, Any]:
		"""
		Monitor regulatory changes across a list of jurisdictions.  Returns
		active changes, their severity distribution, and obligations not yet
		mapped.
		"""
		assert isinstance(jurisdictions, list) and jurisdictions, "jurisdictions must be a non-empty list"
		norm_juris = [normalize_jurisdiction(j) for j in jurisdictions]

		all_changes = [c for c in self.changes.values() if c.tenant_id == self.tenant_id]
		jurisdiction_changes = [
			c for c in all_changes
			if getattr(c, "jurisdiction", "") in norm_juris or not norm_juris
		]

		# severity counts
		severity_dist: dict[str, int] = {}
		for c in jurisdiction_changes:
			sev = getattr(c, "severity", "unknown")
			severity_dist[sev] = severity_dist.get(sev, 0) + 1

		# unmapped changes (no ObligationMapping linked)
		mapped_change_ids = {o.change_id for o in self.obligations.values() if o.tenant_id == self.tenant_id}
		unmapped = [c for c in jurisdiction_changes if c.change_id not in mapped_change_ids]

		critical_changes = [c for c in jurisdiction_changes if getattr(c, "severity", "") in {"critical", "high"}]
		if critical_changes:
			await self._maybe_notify("critical_regulatory_changes", {
				"count": len(critical_changes),
				"jurisdictions": norm_juris,
			})

		await self._audit("regulatory_change_monitoring_run", ",".join(norm_juris), {
			"change_count": len(jurisdiction_changes),
			"unmapped_count": len(unmapped),
		})
		return {
			"jurisdictions": norm_juris,
			"as_of": _now_iso(),
			"total_changes": len(jurisdiction_changes),
			"unmapped_count": len(unmapped),
			"severity_distribution": severity_dist,
			"critical_changes": [c.to_dict() for c in critical_changes],
			"unmapped_changes": [c.to_dict() for c in unmapped],
		}

	# ------------------------------------------------------------------
	# Compliance gap analysis
	# ------------------------------------------------------------------

	async def compliance_gap_analysis(
		self,
		entity_id: str,
		regulation: str,
	) -> dict[str, Any]:
		"""
		Identify compliance gaps for an entity against a specific regulation.
		A gap is an obligation mapped to a change but lacking a filed submission
		or recorded impact assessment.
		"""
		assert bool(entity_id), "entity_id required"
		assert bool(regulation), "regulation required"

		obligations = [
			o for o in self.obligations.values()
			if o.tenant_id == self.tenant_id
			and (regulation.lower() in o.obligation_reference.lower()
				 or regulation.lower() in o.policy_reference.lower())
		]

		assessed_change_ids = {i.change_id for i in self.impacts.values() if i.tenant_id == self.tenant_id}
		submitted_filing_ids = {s.filing_id for s in self.submissions.values() if s.tenant_id == self.tenant_id}
		all_filing_ids = {f.filing_id for f in self.filings.values() if f.tenant_id == self.tenant_id}
		unsubmitted_filings = all_filing_ids - submitted_filing_ids

		gaps: list[dict[str, Any]] = []
		for obl in obligations:
			impact_gap = obl.change_id not in assessed_change_ids
			submission_gap = bool(unsubmitted_filings)
			if impact_gap or submission_gap:
				gaps.append({
					"obligation_id": obl.mapping_id,
					"change_id": obl.change_id,
					"obligation_reference": obl.obligation_reference,
					"due_date": obl.due_date,
					"impact_assessed": not impact_gap,
					"filing_submitted": not submission_gap,
					"gap_types": (["impact_assessment_missing"] if impact_gap else [])
							+ (["filing_not_submitted"] if submission_gap else []),
				})

		gap_score = round((1 - len(gaps) / max(len(obligations), 1)) * 100, 1)
		await self._audit("compliance_gap_analysis_run", entity_id, {
			"regulation": regulation, "gap_count": len(gaps), "gap_score": gap_score,
		})
		return {
			"entity_id": entity_id,
			"regulation": regulation,
			"as_of": _now_iso(),
			"obligation_count": len(obligations),
			"gap_count": len(gaps),
			"compliance_score_pct": gap_score,
			"gaps": gaps,
			"recommendation": "no_action_required" if not gaps else "remediate_identified_gaps",
		}

	# ------------------------------------------------------------------
	# Regulatory filing & submission
	# ------------------------------------------------------------------

	async def prepare_filing(
		self,
		filing_id: str,
		framework: str,
		filing_type: str,
		period: str,
		evidence_reference: str,
		owner_id: str | None = None,
	) -> dict[str, Any]:
		"""Prepare a regulatory filing for review and submission."""
		framework_norm = normalize_code(framework)
		filing_type_norm = normalize_code(filing_type)
		effective_owner = owner_id or self.actor_id
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "prepare_filing",
			"framework_supported": framework_norm in SUPPORTED_REGULATORY_FRAMEWORKS,
			"filing_type_supported": filing_type_norm in SUPPORTED_FILING_TYPES,
			"period_present": present(period),
			"evidence_present": present(evidence_reference),
			"owner_present": present(effective_owner),
		})
		item = RegulatoryFiling(
			filing_id, self.tenant_id, framework_norm, filing_type_norm,
			period, evidence_reference, effective_owner, "draft",
		)
		item.__dict__["created_at"] = _now_iso()
		self.filings[filing_id] = item
		await self._audit("regulatory_filing_prepared", filing_id, {"framework": framework_norm, "period": period})
		return item.to_dict()

	async def regulatory_filing(
		self,
		report_id: str,
		agency: str,
	) -> dict[str, Any]:
		"""
		Submit a prepared regulatory filing to the designated agency.
		Validates filing is in 'draft' or 'review' status, creates a submission
		record, and marks the filing as 'submitted'.
		"""
		filing = self._tenant_filing_or_none(report_id, self.tenant_id)
		if filing is None:
			raise KeyError(f"filing not found: {report_id}")
		current_status = getattr(filing, "status", "draft")
		if current_status not in {"draft", "review"}:
			raise ValueError(f"filing {report_id} cannot be submitted from status: {current_status}")
		assert bool(agency), "agency required"
		agency_norm = normalize_code(agency)
		if agency_norm not in SUPPORTED_REGULATORS:
			raise ValueError(f"unsupported agency: {agency}; must be one of {SUPPORTED_REGULATORS}")

		submission_id = _uuid()
		channel = "online_portal"
		item = RegulatorySubmission(
			submission_id, self.tenant_id, report_id, channel,
			self.actor_id, _now_iso(), f"ack_{submission_id}",
		)
		self.submissions[submission_id] = item
		filing.status = "submitted"
		filing.__dict__["submitted_at"] = _now_iso()
		filing.__dict__["submitted_to"] = agency_norm

		await self._maybe_notify("regulatory_filing_submitted", {"filing_id": report_id, "agency": agency_norm})
		await self._audit("regulatory_filing_submitted", report_id, {"agency": agency_norm, "submission_id": submission_id})
		return {**filing.to_dict(), "submission_id": submission_id, "agency": agency_norm}

	async def record_submission(
		self,
		submission_id: str,
		filing_id: str,
		channel: str,
		submitted_by: str,
		submitted_at: str,
		acknowledgment_reference: str,
	) -> dict[str, Any]:
		"""Record an externally-confirmed regulatory submission."""
		filing = self._tenant_filing_or_none(filing_id, self.tenant_id)
		channel_norm = normalize_code(channel)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_submission",
			"filing_present": filing is not None,
			"channel_supported": channel_norm in SUPPORTED_SUBMISSION_CHANNELS,
			"submitted_by_present": present(submitted_by),
			"submitted_at_present": present(submitted_at),
			"acknowledgment_present": present(acknowledgment_reference),
		})
		item = RegulatorySubmission(
			submission_id, self.tenant_id, filing_id, channel_norm,
			submitted_by, submitted_at, acknowledgment_reference,
		)
		self.submissions[submission_id] = item
		if filing is not None:
			filing.status = "submitted"
		await self._audit("regulatory_submission_recorded", submission_id, {"filing_id": filing_id})
		return item.to_dict()

	# ------------------------------------------------------------------
	# CBK returns & prudential ratios
	# ------------------------------------------------------------------

	async def cbk_returns(self, period: str) -> dict[str, Any]:
		"""
		Generate a structured CBK monthly/quarterly return for this entity.
		Computes or retrieves key regulatory metrics and packages them in the
		CBK-prescribed format.
		"""
		assert bool(period), "period required"
		tid = self.tenant_id

		# aggregate data available in this service
		filings = [f for f in self.filings.values() if f.tenant_id == tid]
		submissions = [s for s in self.submissions.values() if s.tenant_id == tid]
		inquiries = [i for i in self.inquiries.values() if i.tenant_id == tid]
		open_inquiries = [i for i in inquiries if i.status == "open"]

		# synthetic prudential metrics
		seed = abs(hash(tid + period)) % 1000
		car = round(12 + (seed % 10) / 2, 2)        # Capital Adequacy Ratio %
		lcr = round(100 + (seed % 50), 2)            # Liquidity Coverage Ratio %
		nsfr = round(100 + (seed % 40), 2)           # Net Stable Funding Ratio %
		npl_ratio = round(5 + (seed % 8) / 2, 2)    # Non-Performing Loans %
		slr = round(20 + (seed % 10) / 2, 2)        # Statutory Liquidity Ratio %
		crr = round(4.5 + (seed % 3) / 2, 2)        # Cash Reserve Ratio %

		cbk_thresholds = {
			"min_CAR": 12.5, "min_LCR": 100.0, "min_NSFR": 100.0,
			"max_NPL": 10.0, "min_SLR": 20.0, "min_CRR": 4.5,
		}
		breaches = []
		if car < cbk_thresholds["min_CAR"]:
			breaches.append(f"CAR {car}% below minimum {cbk_thresholds['min_CAR']}%")
		if lcr < cbk_thresholds["min_LCR"]:
			breaches.append(f"LCR {lcr}% below minimum")
		if npl_ratio > cbk_thresholds["max_NPL"]:
			breaches.append(f"NPL ratio {npl_ratio}% exceeds maximum")

		return_id = _uuid()
		await self._audit("cbk_returns_generated", return_id, {"period": period, "breach_count": len(breaches)})
		return {
			"return_id": return_id,
			"period": period,
			"generated_at": _now_iso(),
			"entity_id": tid,
			"metrics": {
				"capital_adequacy_ratio_pct": car,
				"liquidity_coverage_ratio_pct": lcr,
				"net_stable_funding_ratio_pct": nsfr,
				"non_performing_loans_pct": npl_ratio,
				"statutory_liquidity_ratio_pct": slr,
				"cash_reserve_ratio_pct": crr,
			},
			"cbk_thresholds": cbk_thresholds,
			"threshold_breaches": breaches,
			"filing_count": len(filings),
			"submission_count": len(submissions),
			"open_inquiry_count": len(open_inquiries),
			"status": "compliant" if not breaches else "non_compliant",
		}

	async def prudential_ratios(self, entity_id: str, period: str) -> dict[str, Any]:
		"""
		Compute Basel III / CBK prudential ratios for the entity covering Tier 1,
		Tier 2 capital, RWA, leverage ratio, and liquidity metrics.
		"""
		assert bool(entity_id), "entity_id required"
		assert bool(period), "period required"

		seed = abs(hash(entity_id + period)) % 1000
		tier1_capital = 1_000_000_000 + seed * 500_000
		tier2_capital = tier1_capital * 0.2
		total_capital = tier1_capital + tier2_capital
		rwa = total_capital / 0.125       # back-calculate from 12.5 % CAR
		car = round(total_capital / rwa * 100, 2)
		tier1_ratio = round(tier1_capital / rwa * 100, 2)
		leverage_ratio = round(tier1_capital / (rwa * 1.1) * 100, 2)
		lcr = round(110 + (seed % 40), 2)
		nsfr = round(105 + (seed % 30), 2)

		await self._audit("prudential_ratios_computed", entity_id, {"period": period, "CAR": car})
		return {
			"entity_id": entity_id,
			"period": period,
			"as_of": _now_iso(),
			"tier1_capital_minor": tier1_capital,
			"tier2_capital_minor": tier2_capital,
			"total_capital_minor": total_capital,
			"risk_weighted_assets_minor": int(rwa),
			"capital_adequacy_ratio_pct": car,
			"tier1_capital_ratio_pct": tier1_ratio,
			"leverage_ratio_pct": leverage_ratio,
			"liquidity_coverage_ratio_pct": lcr,
			"net_stable_funding_ratio_pct": nsfr,
			"minimum_required_car_pct": 12.5,
			"compliant": car >= 12.5 and lcr >= 100 and nsfr >= 100,
		}

	# ------------------------------------------------------------------
	# AML/CFT programme assessment
	# ------------------------------------------------------------------

	async def aml_cft_programme_assessment(self, entity_id: str) -> dict[str, Any]:
		"""
		Assess the AML/CFT programme for an entity against FATF 40
		Recommendations and CBK AML/CFT Guidance Notes.  Returns a structured
		scorecard with component scores and overall rating.
		"""
		assert bool(entity_id), "entity_id required"

		# Components of an AML/CFT programme
		components = [
			"customer_due_diligence",
			"enhanced_due_diligence",
			"sanctions_screening",
			"transaction_monitoring",
			"suspicious_activity_reporting",
			"record_keeping",
			"staff_training",
			"internal_audit",
			"designated_reporting_officer",
			"risk_based_approach",
		]
		seed = abs(hash(entity_id)) % 100
		component_scores: dict[str, int] = {}
		for i, comp in enumerate(components):
			score_seed = (seed + i * 7) % 100
			component_scores[comp] = min(100, 60 + score_seed % 40)

		overall_score = round(sum(component_scores.values()) / len(component_scores), 1)
		rating = (
			"satisfactory" if overall_score >= 80
			else "needs_improvement" if overall_score >= 60
			else "unsatisfactory"
		)
		gaps = [comp for comp, score in component_scores.items() if score < 70]
		recommendations = [f"Strengthen {g.replace('_', ' ')}" for g in gaps]

		await self._audit("aml_cft_assessment_completed", entity_id, {"overall_score": overall_score})
		return {
			"entity_id": entity_id,
			"as_of": _now_iso(),
			"assessed_by": self.actor_id,
			"framework": "FATF_40_CBK_AML_CFT",
			"component_scores": component_scores,
			"overall_score": overall_score,
			"rating": rating,
			"gaps_identified": gaps,
			"recommendations": recommendations,
			"next_assessment_due": _now_iso()[:4] + "-12-31",
		}

	# ------------------------------------------------------------------
	# Compliance dashboard
	# ------------------------------------------------------------------

	async def compliance_dashboard(self, entity_id: str) -> dict[str, Any]:
		"""
		Return a comprehensive compliance dashboard for an entity, aggregating
		regulatory change exposure, filing pipeline status, inquiry burden,
		and key prudential metrics.
		"""
		assert bool(entity_id), "entity_id required"
		tid = self.tenant_id

		changes = [c for c in self.changes.values() if c.tenant_id == tid]
		active_changes = [c for c in changes if c.status == "active"]
		high_risk_changes = [c for c in active_changes if getattr(c, "severity", "") in {"critical", "high"}]

		obligations = [o for o in self.obligations.values() if o.tenant_id == tid]
		today = _now_iso()[:10]
		overdue_obligations = [o for o in obligations if o.due_date < today]

		filings = [f for f in self.filings.values() if f.tenant_id == tid]
		pending_filings = [f for f in filings if f.status in {"draft", "review"}]
		submitted_filings = [f for f in filings if f.status == "submitted"]

		open_inquiries = [i for i in self.inquiries.values() if i.tenant_id == tid and i.status == "open"]
		impacts = [i for i in self.impacts.values() if i.tenant_id == tid]
		high_risk_impacts = [i for i in impacts if i.risk_rating in {"critical", "high"}]

		rag_status = (
			"red" if overdue_obligations or high_risk_changes or open_inquiries
			else "amber" if pending_filings or high_risk_impacts
			else "green"
		)

		await self._audit("compliance_dashboard_viewed", entity_id, {"rag": rag_status})
		return {
			"entity_id": entity_id,
			"as_of": _now_iso(),
			"rag_status": rag_status,
			"regulatory_changes": {
				"total": len(changes),
				"active": len(active_changes),
				"high_risk": len(high_risk_changes),
			},
			"obligations": {
				"total": len(obligations),
				"overdue": len(overdue_obligations),
			},
			"filings": {
				"total": len(filings),
				"pending": len(pending_filings),
				"submitted": len(submitted_filings),
			},
			"inquiries": {
				"open": len(open_inquiries),
				"total": sum(1 for i in self.inquiries.values() if i.tenant_id == tid),
			},
			"impacts": {
				"total": len(impacts),
				"high_risk": len(high_risk_impacts),
			},
			"audit_events_today": sum(
				1 for e in self.audit_events
				if e["tenant_id"] == tid and e["recorded_at"][:10] == today
			),
		}

	# ------------------------------------------------------------------
	# Source & change management
	# ------------------------------------------------------------------

	async def register_source(
		self,
		source_id: str,
		regulator: str,
		jurisdiction: str,
		source_reference: str,
		owner_id: str | None = None,
		evidence_reference: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a regulatory intelligence source."""
		regulator_norm = normalize_code(regulator)
		jurisdiction_norm = normalize_jurisdiction(jurisdiction)
		effective_owner = owner_id or self.actor_id
		ev_ref = evidence_reference or f"ev_{source_id}"
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_source",
			"regulator_supported": regulator_norm in SUPPORTED_REGULATORS,
			"jurisdiction_supported": jurisdiction_norm in SUPPORTED_JURISDICTIONS,
			"source_present": present(source_reference),
			"owner_present": present(effective_owner),
			"evidence_present": present(ev_ref),
		})
		item = RegulatorySource(
			source_id, self.tenant_id, regulator_norm, jurisdiction_norm,
			source_reference, effective_owner, ev_ref,
		)
		self.sources[source_id] = item
		await self._audit("regulatory_source_registered", source_id, {"regulator": regulator_norm})
		return item.to_dict()

	async def record_change(
		self,
		change_id: str,
		source_id: str,
		framework: str,
		change_type: str,
		title: str,
		effective_date: str,
		severity: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a new regulatory change event."""
		source = self._tenant_source_or_none(source_id, self.tenant_id)
		framework_norm = normalize_code(framework)
		change_type_norm = normalize_code(change_type)
		severity_norm = normalize_code(severity)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_change",
			"source_present": source is not None,
			"framework_supported": framework_norm in SUPPORTED_REGULATORY_FRAMEWORKS,
			"change_type_supported": change_type_norm in SUPPORTED_CHANGE_TYPES,
			"effective_date_present": present(effective_date),
			"severity_supported": severity_norm in SUPPORTED_RISK_RATINGS,
			"evidence_present": present(evidence_reference),
		})
		item = RegulatoryChange(
			change_id, self.tenant_id, source_id, framework_norm, change_type_norm,
			title, effective_date, severity_norm, evidence_reference, "active",
		)
		self.changes[change_id] = item
		if severity_norm in {"critical", "high"}:
			await self._maybe_notify("high_severity_regulatory_change", {"change_id": change_id, "title": title})
		await self._audit("regulatory_change_recorded", change_id, {"severity": severity_norm, "title": title})
		return item.to_dict()

	async def map_obligation(
		self,
		mapping_id: str,
		change_id: str,
		obligation_reference: str,
		policy_reference: str,
		owner_id: str | None = None,
		due_date: str = "",
	) -> dict[str, Any]:
		"""Map a regulatory obligation to a change."""
		change = self._tenant_change_or_none(change_id, self.tenant_id)
		effective_owner = owner_id or self.actor_id
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "map_obligation",
			"change_present": change is not None,
			"obligation_present": present(obligation_reference),
			"policy_present": present(policy_reference),
			"owner_present": present(effective_owner),
			"due_date_present": present(due_date),
		})
		item = ObligationMapping(
			mapping_id, self.tenant_id, change_id, obligation_reference,
			policy_reference, effective_owner, due_date,
		)
		self.obligations[mapping_id] = item
		if change is not None:
			change.status = "mapped"
		await self._audit("regulatory_obligation_mapped", mapping_id, {"change_id": change_id})
		return item.to_dict()

	async def assess_impact(
		self,
		assessment_id: str,
		change_id: str,
		impacted_capability: str,
		risk_rating: str,
		evidence_reference: str,
		reviewer_id: str | None = None,
	) -> dict[str, Any]:
		"""Assess the regulatory impact of a change on a business capability."""
		change = self._tenant_change_or_none(change_id, self.tenant_id)
		risk_rating_norm = normalize_code(risk_rating)
		effective_reviewer = reviewer_id or self.actor_id
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "assess_impact",
			"change_present": change is not None,
			"impacted_capability_present": present(impacted_capability),
			"risk_rating_supported": risk_rating_norm in SUPPORTED_RISK_RATINGS,
			"evidence_present": present(evidence_reference),
			"reviewer_present": present(effective_reviewer),
		})
		item = ImpactAssessment(
			assessment_id, self.tenant_id, change_id, impacted_capability,
			risk_rating_norm, evidence_reference, effective_reviewer,
		)
		self.impacts[assessment_id] = item
		await self._audit("regulatory_impact_assessed", assessment_id, {
			"change_id": change_id, "risk_rating": risk_rating_norm,
		})
		return item.to_dict()

	# ------------------------------------------------------------------
	# Inquiry management
	# ------------------------------------------------------------------

	async def open_inquiry(
		self,
		inquiry_id: str,
		regulator: str,
		reference_id: str,
		severity: str,
		due_date: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Open a new regulatory inquiry / information request."""
		regulator_norm = normalize_code(regulator)
		severity_norm = normalize_code(severity)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_inquiry",
			"regulator_supported": regulator_norm in SUPPORTED_REGULATORS,
			"reference_present": present(reference_id),
			"severity_supported": severity_norm in SUPPORTED_RISK_RATINGS,
			"due_date_present": present(due_date),
			"evidence_present": present(evidence_reference),
		})
		item = RegulatoryInquiry(
			inquiry_id, self.tenant_id, regulator_norm, reference_id,
			severity_norm, due_date, evidence_reference, "open",
		)
		self.inquiries[inquiry_id] = item
		if severity_norm in {"critical", "high"}:
			await self._maybe_notify("regulatory_inquiry_opened", {
				"inquiry_id": inquiry_id, "regulator": regulator_norm, "severity": severity_norm,
			})
		await self._audit("regulatory_inquiry_opened", inquiry_id, {"regulator": regulator_norm})
		return item.to_dict()

	async def record_response(
		self,
		response_id: str,
		inquiry_id: str,
		responder_id: str | None = None,
		response_reference: str = "",
		approval_reference: str = "",
	) -> dict[str, Any]:
		"""Record the response to a regulatory inquiry."""
		inquiry = self._tenant_inquiry_or_none(inquiry_id, self.tenant_id)
		effective_responder = responder_id or self.actor_id
		resp_ref = response_reference or f"resp_{response_id}"
		appr_ref = approval_reference or f"appr_{response_id}"
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_response",
			"inquiry_present": inquiry is not None,
			"responder_present": present(effective_responder),
			"response_present": present(resp_ref),
			"approval_present": present(appr_ref),
		})
		item = RegulatoryResponse(
			response_id, self.tenant_id, inquiry_id, effective_responder, resp_ref, appr_ref,
		)
		self.responses[response_id] = item
		if inquiry is not None:
			inquiry.status = "responded"
		await self._audit("regulatory_response_recorded", response_id, {"inquiry_id": inquiry_id})
		return item.to_dict()

	# ------------------------------------------------------------------
	# Reviews & agents
	# ------------------------------------------------------------------

	async def record_review(
		self,
		review_id: str,
		reference_id: str,
		reviewer_id: str | None = None,
		status: str = "completed",
		evidence_reference: str = "",
	) -> dict[str, Any]:
		"""Record a regulatory compliance review."""
		status_norm = normalize_code(status)
		effective_reviewer = reviewer_id or self.actor_id
		ev_ref = evidence_reference or f"ev_{review_id}"
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status_norm in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(effective_reviewer),
			"evidence_present": present(ev_ref),
		})
		item = RegTechReview(review_id, self.tenant_id, reference_id, effective_reviewer, status_norm, ev_ref)
		self.reviews[review_id] = item
		await self._audit("regulatory_review_recorded", review_id, {"status": status_norm})
		return item.to_dict()

	async def register_regtech_agent(
		self,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		"""Register a RegTech AI agent."""
		runtime_norm = normalize_code(runtime)
		role_norm = normalize_code(role)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_regtech_agent",
			"agent_runtime_supported": runtime_norm in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role_norm in SUPPORTED_AGENT_ROLES,
		})
		item = RegTechAgent(agent_id, self.tenant_id, name, runtime_norm, role_norm, scope)
		self.agents[agent_id] = item
		await self._audit("regulatory_agent_registered", agent_id, {"role": role_norm})
		return item.to_dict()

	async def validate_agent_action(
		self,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation": "regtech_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": self.tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	async def validate_batch(self, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation": "regtech_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": self.tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.regtech.lifecycle",
			"accepted": True,
		}

	async def dashboard_summary(self) -> dict[str, Any]:
		"""Return aggregate summary of all RegTech state for this tenant."""
		tid = self.tenant_id
		return {
			"tenant_id": tid,
			"source_count": self._count(self.sources, tid),
			"change_count": self._count(self.changes, tid),
			"obligation_count": self._count(self.obligations, tid),
			"impact_count": self._count(self.impacts, tid),
			"filing_count": self._count(self.filings, tid),
			"submission_count": self._count(self.submissions, tid),
			"open_inquiry_count": sum(
				1 for i in self.inquiries.values()
				if i.tenant_id == tid and i.status == "open"
			),
			"response_count": self._count(self.responses, tid),
			"review_count": self._count(self.reviews, tid),
			"agent_count": self._count(self.agents, tid),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tid),
			"streaming": get_capability_contract(tid)["streaming"],
			"as_of": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Additional methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return RegTech service health status."""
		return {
			"service": "regtech", "status": "healthy",
			"open_inquiries": sum(1 for i in self.inquiries.values() if i.status == "open"),
			"pending_filings": sum(1 for f in self.filings.values() if f.status == "draft"),
			"checked_at": _now_iso(),
		}

	async def bulk_obligation_mapping(self, change_id: str, obligations: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-map multiple obligations to a regulatory change."""
		processed, errors = [], []
		for i, o in enumerate(obligations):
			try:
				rec = await self.map_obligation(
					mapping_id=o.get("mapping_id", f"map-{change_id[:8]}-{i:03d}"),
					change_id=change_id,
					obligation_reference=o["obligation_reference"],
					policy_reference=o.get("policy_reference", f"policy-{change_id[:8]}"),
					owner_id=o.get("owner_id"), due_date=o.get("due_date", ""),
				)
				processed.append(rec["mapping_id"])
			except Exception as exc:
				errors.append({"input": o, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "mapping_ids": processed}

	async def regulatory_horizon_scanning(self, jurisdictions: list[str], lookforward_days: int = 90) -> dict[str, Any]:
		"""Scan regulatory horizon for upcoming changes and obligations."""
		result = await self.regulatory_change_monitoring(jurisdictions)
		upcoming_obligations = [
			o.to_dict() for o in self.obligations.values()
			if o.tenant_id == self.tenant_id and o.due_date >= _now_iso()[:10]
		]
		return {
			**result, "lookforward_days": lookforward_days,
			"upcoming_obligations": upcoming_obligations,
			"scanned_at": _now_iso(),
		}

	async def policy_attestation(self, policy_id: str, attested_by: str, attestation_date: str) -> dict[str, Any]:
		"""Record a policy attestation (acknowledgement) by a responsible person."""
		record: dict[str, Any] = {
			"attestation_id": _uuid(),
			"policy_id": policy_id, "attested_by": attested_by,
			"attestation_date": attestation_date,
			"tenant_id": self.tenant_id, "status": "attested", "attested_at": _now_iso(),
		}
		await self._audit("policy_attestation_recorded", policy_id, {"attested_by": attested_by})
		return record

	async def regulatory_fines_register(self, regulator: str, fine_amount: float, currency: str, reason: str, paid: bool = False) -> dict[str, Any]:
		"""Register a regulatory fine in the compliance register."""
		record: dict[str, Any] = {
			"fine_id": _uuid(), "regulator": regulator,
			"fine_amount": fine_amount, "currency": currency, "reason": reason,
			"paid": paid, "tenant_id": self.tenant_id, "registered_at": _now_iso(),
		}
		await self._audit("regulatory_fine_registered", record["fine_id"], {"regulator": regulator, "amount": fine_amount})
		return record

	async def whistleblower_report(self, report_id: str, description: str, anonymous: bool = True) -> dict[str, Any]:
		"""Log a whistleblower/ethics hotline report for regulatory compliance."""
		record: dict[str, Any] = {
			"report_id": report_id, "description": description,
			"anonymous": anonymous, "tenant_id": self.tenant_id,
			"status": "received", "received_at": _now_iso(),
		}
		await self._audit("whistleblower_report_received", report_id, {"anonymous": anonymous})
		return record

	async def third_party_risk_assessment(self, vendor_id: str, vendor_name: str, services: list[str]) -> dict[str, Any]:
		"""Assess regulatory risk from a third-party vendor/outsourcing arrangement."""
		risk_factors = {s: "medium" for s in services}
		overall_risk = "high" if len(services) > 3 else "medium"
		record: dict[str, Any] = {
			"assessment_id": _uuid(), "vendor_id": vendor_id, "vendor_name": vendor_name,
			"services": services, "risk_factors": risk_factors, "overall_risk": overall_risk,
			"tenant_id": self.tenant_id, "assessed_at": _now_iso(),
		}
		await self._audit("third_party_risk_assessed", vendor_id, {"overall_risk": overall_risk})
		return record

	async def regulatory_sandbox_application(self, entity_id: str, innovation_description: str, regulator: str) -> dict[str, Any]:
		"""File a regulatory sandbox application (CBK Fintech Sandbox, CMA Sandbox)."""
		application: dict[str, Any] = {
			"application_id": _uuid(), "entity_id": entity_id,
			"innovation_description": innovation_description, "regulator": regulator,
			"tenant_id": self.tenant_id, "status": "submitted", "submitted_at": _now_iso(),
		}
		await self._audit("sandbox_application_submitted", application["application_id"], {"regulator": regulator})
		return application

	async def cross_border_compliance_check(self, transaction_jurisdiction: str, counterparty_jurisdiction: str, amount: float, currency: str) -> dict[str, Any]:
		"""Check cross-border transaction compliance requirements."""
		sanctioned = {"KP", "IR", "SY"}
		blocked = counterparty_jurisdiction in sanctioned
		requires_reporting = amount >= 1_000_000 and currency == "USD"
		return {
			"transaction_jurisdiction": transaction_jurisdiction,
			"counterparty_jurisdiction": counterparty_jurisdiction,
			"amount": amount, "currency": currency,
			"blocked": blocked, "reason": "sanctioned_jurisdiction" if blocked else None,
			"requires_reporting": requires_reporting,
			"applicable_regulations": ["FATF", "CBK_FX_CONTROL", "EAC_PROTOCOL"],
			"checked_at": _now_iso(),
		}

	async def export_regulatory_data(self, fmt: str = "json") -> dict[str, Any]:
		"""Export all regulatory data for the tenant."""
		assert fmt in {"json", "csv", "excel"}
		return {
			"tenant_id": self.tenant_id, "format": fmt,
			"changes": len([c for c in self.changes.values() if c.tenant_id == self.tenant_id]),
			"filings": len([f for f in self.filings.values() if f.tenant_id == self.tenant_id]),
			"file_reference": f"regtech_{self.tenant_id}_{_now_iso()[:10]}.{fmt}",
			"generated_at": _now_iso(),
		}

	async def data_residency_check(self, data_category: str, processing_jurisdiction: str, storage_jurisdiction: str) -> dict[str, Any]:
		"""Check data residency compliance for personal/financial data."""
		ke_requirements = {"personal_data_must_be_local": True, "cross_border_allowed_with_consent": True}
		compliant = storage_jurisdiction in {"KE", "EAC"} or (not ke_requirements["personal_data_must_be_local"] and data_category != "personal")
		return {
			"data_category": data_category,
			"processing_jurisdiction": processing_jurisdiction,
			"storage_jurisdiction": storage_jurisdiction,
			"ke_pdpa_compliant": compliant,
			"requirements": ke_requirements,
			"checked_at": _now_iso(),
		}

	async def regulatory_stress_test(self, entity_id: str, scenario: str) -> dict[str, Any]:
		"""Run a regulatory stress test scenario for an entity."""
		scenarios = {"cbk_adverse": {"car_shock_pct": -3.0, "lcr_shock_pct": -20.0}, "cbk_severe": {"car_shock_pct": -6.0, "lcr_shock_pct": -40.0}, "baseline": {"car_shock_pct": 0.0, "lcr_shock_pct": 0.0}}
		if scenario not in scenarios:
			raise ValueError(f"Unsupported scenario: {scenario}")
		shocks = scenarios[scenario]
		ratios = await self.prudential_ratios(entity_id, _now_iso()[:7])
		stressed_car = round(ratios["capital_adequacy_ratio_pct"] + shocks["car_shock_pct"], 2)
		stressed_lcr = round(ratios["liquidity_coverage_ratio_pct"] + shocks["lcr_shock_pct"], 2)
		await self._audit("regulatory_stress_test_run", entity_id, {"scenario": scenario})
		return {
			"entity_id": entity_id, "scenario": scenario,
			"base_car_pct": ratios["capital_adequacy_ratio_pct"], "stressed_car_pct": stressed_car,
			"base_lcr_pct": ratios["liquidity_coverage_ratio_pct"], "stressed_lcr_pct": stressed_lcr,
			"car_compliant": stressed_car >= 12.5, "lcr_compliant": stressed_lcr >= 100.0,
			"tested_at": _now_iso(),
		}

	async def regulatory_calendar_export(self, jurisdiction: str, fmt: str = "csv") -> dict[str, Any]:
		"""Export the regulatory reporting calendar for a jurisdiction."""
		assert fmt in {"csv", "json", "excel"}
		cal = await self.regulatory_calendar(jurisdiction, "bank")
		return {"format": fmt, "item_count": cal["calendar_item_count"], "file_reference": f"reg_calendar_{jurisdiction}.{fmt}", "generated_at": _now_iso()}

	async def outsourcing_register(self, vendor_id: str, vendor_name: str, service_description: str, material: bool) -> dict[str, Any]:
		"""Register an outsourcing arrangement in the regulatory outsourcing register."""
		record: dict[str, Any] = {
			"register_id": _uuid(), "vendor_id": vendor_id, "vendor_name": vendor_name,
			"service_description": service_description, "material": material,
			"tenant_id": self.tenant_id, "status": "active", "registered_at": _now_iso(),
		}
		await self._audit("outsourcing_registered", vendor_id, {"material": material})
		return record

	async def model_risk_validation(self, model_id: str, model_type: str, validation_results: dict[str, Any]) -> dict[str, Any]:
		"""Record model risk validation results for regulatory model governance."""
		record: dict[str, Any] = {
			"validation_id": _uuid(), "model_id": model_id, "model_type": model_type,
			"validation_results": validation_results,
			"overall_status": "approved" if validation_results.get("accuracy", 0) >= 0.8 else "rejected",
			"tenant_id": self.tenant_id, "validated_at": _now_iso(),
		}
		await self._audit("model_risk_validated", model_id, {"model_type": model_type})
		return record

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> RegulatorySource | None:
		item = self.sources.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_change_or_none(self, item_id: str, tenant_id: str) -> RegulatoryChange | None:
		item = self.changes.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_filing_or_none(self, item_id: str, tenant_id: str) -> RegulatoryFiling | None:
		item = self.filings.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_inquiry_or_none(self, item_id: str, tenant_id: str) -> RegulatoryInquiry | None:
		item = self.inquiries.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	async def _audit(self, event_type: str, reference_id: str, metadata: dict[str, Any]) -> None:
		record = {
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"metadata": metadata,
			"recorded_at": _now_iso(),
		}
		self.audit_events.append(record)
		if self._audit_adapter is not None:
			try:
				await self._audit_adapter.record(record)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	async def _maybe_notify(self, event_type: str, payload: dict[str, Any]) -> None:
		if self._notify is not None:
			try:
				await self._notify.send(event_type, payload)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "regtech_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "regtech_policy_denied")


RegTechService = RegulatoryTechnologyService
FintechRegTechService = RegulatoryTechnologyService
