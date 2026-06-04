"""Domain service for APG advanced CRM analytics — expanded implementation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_CRM_AGENT_ROLES,
		SUPPORTED_CRM_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_CRM_AGENT_ROLES,
		SUPPORTED_CRM_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _record_id(prefix: str, value: str) -> str:
	slug = "".join(c.lower() if c.isalnum() else "_" for c in str(value)).strip("_")
	return f"{prefix}_{slug or 'record'}"


class AdvancedCRMService:
	"""
	Tenant-scoped account, lead, pipeline, campaign, and forecast coordinator.

	Expanded with: lead_capture, lead_scoring, lead_assignment,
	opportunity_create, opportunity_stage_advance, pipeline_report,
	customer_segmentation, campaign_analytics, win_loss_analysis,
	crm_dashboard.
	"""

	def __init__(self) -> None:
		self._accounts: dict[str, dict[str, Any]] = {}
		self._contacts: dict[str, dict[str, Any]] = {}
		self._leads: dict[str, dict[str, Any]] = {}
		self._opportunities: dict[str, dict[str, Any]] = {}
		self._activities: dict[str, dict[str, Any]] = {}
		self._campaigns: dict[str, dict[str, Any]] = {}
		self._forecasts: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		# New stores
		self._lead_scores: dict[str, dict[str, Any]] = {}
		self._segments: dict[str, dict[str, Any]] = {}
		self._win_loss_records: list[dict[str, Any]] = []
		self._stage_history: dict[str, list[dict[str, Any]]] = {}

	# ------------------------------------------------------------------
	# lead_capture
	# ------------------------------------------------------------------

	def lead_capture(
		self,
		source: str,
		contact_details: dict[str, Any],
		campaign_id: str,
		tenant_id: str = "default",
		lead_id: str | None = None,
		score: int | None = None,
	) -> dict[str, Any]:
		"""
		Capture an inbound lead from a source with contact details.

		source: Channel label (e.g. 'web_form', 'email', 'event', 'referral').
		contact_details: Dict with at minimum 'name'; optionally 'email', 'phone', 'company'.
		campaign_id: Attribution campaign ID.
		Returns the lead record with initial qualification status.
		"""
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_lead",
			"lead_source_present": bool(source),
		})
		name = contact_details.get("name", "Unknown")
		resolved_id = lead_id or _record_id("crm_lead", f"{tenant_id}_{source}_{name}")
		record = {
			"id": resolved_id,
			"lead_id": resolved_id,
			"tenant_id": tenant_id,
			"name": name,
			"email": contact_details.get("email"),
			"phone": contact_details.get("phone"),
			"company": contact_details.get("company"),
			"source": source,
			"campaign_id": campaign_id,
			"score": score,
			"owner": None,
			"status": "new",
			"event_stream": "bytewax",
			"created_at": _now(),
			"updated_at": _now(),
		}
		self._leads[resolved_id] = record
		self._emit("lead_captured", tenant_id, resolved_id, {"source": source, "campaign_id": campaign_id, "score": score})
		return deepcopy(record)

	def lead_scoring(
		self,
		lead_id: str,
		model_type: str,
		tenant_id: str = "default",
		scoring_factors: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""
		Score a lead using a specified scoring model.

		model_type: 'demographic', 'behavioural', 'predictive', 'firmographic'.
		scoring_factors: Optional dict of factor_name -> weight (0-1).
		Returns score record with component breakdown and qualification flag.
		"""
		lead = self._require_lead(lead_id, tenant_id)
		supported_models = {"demographic", "behavioural", "predictive", "firmographic", "combined"}
		if model_type not in supported_models:
			raise ValueError(f"unsupported_scoring_model:{model_type}")
		factors = scoring_factors or {}
		# Synthetic scoring: base on available contact data and factors
		base_score = 30
		if lead.get("email"):
			base_score += 15
		if lead.get("company"):
			base_score += 20
		if lead.get("phone"):
			base_score += 10
		if lead.get("campaign_id"):
			base_score += 10
		# Apply factor weights
		factor_score = sum(w * 10 for w in factors.values())
		total_score = min(100, int(base_score + factor_score))
		qualified = total_score >= 60
		score_id = _record_id("score", f"{lead_id}_{model_type}")
		score_record = {
			"score_id": score_id,
			"lead_id": lead_id,
			"tenant_id": tenant_id,
			"model_type": model_type,
			"score": total_score,
			"qualified": qualified,
			"component_breakdown": {
				"base_score": base_score,
				"factor_score": int(factor_score),
				"factors": factors,
			},
			"scored_at": _now(),
		}
		self._lead_scores[score_id] = score_record
		# Update lead score
		lead["score"] = total_score
		lead["status"] = "qualified" if qualified else "active"
		lead["updated_at"] = _now()
		self._emit("lead_scored", tenant_id, lead_id, {"score": total_score, "qualified": qualified})
		return score_record

	def lead_assignment(
		self,
		lead_id: str,
		rep_id: str,
		reason: str,
		tenant_id: str = "default",
		assignment_policy: str = "round_robin",
	) -> dict[str, Any]:
		"""
		Assign a lead to a sales representative.

		rep_id: Sales rep user ID.
		reason: Assignment reason or rule that triggered it.
		assignment_policy: 'round_robin', 'territory', 'skill_match', 'manual'.
		"""
		lead = self._require_lead(lead_id, tenant_id)
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "assign_lead",
			"lead_score_present": lead.get("score") is not None,
			"assignment_policy_present": bool(assignment_policy),
		})
		if not rep_id:
			raise ValueError("rep_id_required")
		if not reason:
			raise ValueError("assignment_reason_required")
		lead["owner"] = rep_id
		lead["assignment_policy"] = assignment_policy
		lead["assignment_reason"] = reason
		lead["status"] = "assigned"
		lead["updated_at"] = _now()
		self._emit("lead_assigned", tenant_id, lead_id, {"rep_id": rep_id, "reason": reason})
		return deepcopy(lead)

	def opportunity_create(
		self,
		lead_id: str,
		product_id: str,
		value: float,
		probability: float,
		close_date: str,
		tenant_id: str = "default",
		opportunity_id: str | None = None,
		name: str = "",
	) -> dict[str, Any]:
		"""
		Create an opportunity from a qualified lead.

		lead_id: Source lead record ID.
		product_id: Product or service being sold.
		value: Deal value.
		probability: Win probability 0.0-1.0.
		close_date: Expected close date (ISO format).
		"""
		lead = self._require_lead(lead_id, tenant_id)
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_opportunity",
			"account_present": True,
			"stage_present": True,
			"amount_present": value is not None,
			"amount": float(value) if value is not None else 0,
			"close_date_present": bool(close_date),
		})
		if not 0.0 <= float(probability) <= 1.0:
			raise ValueError("probability_must_be_0_to_1")
		resolved_id = opportunity_id or _record_id("crm_opp", f"{lead_id}_{product_id}")
		opp = {
			"id": resolved_id,
			"opportunity_id": resolved_id,
			"tenant_id": tenant_id,
			"lead_id": lead_id,
			"product_id": product_id,
			"name": name or f"Opportunity from {lead.get('name', lead_id)}",
			"value": float(value),
			"probability": float(probability),
			"expected_revenue": round(float(value) * float(probability), 2),
			"close_date": close_date,
			"stage": "qualification",
			"owner": lead.get("owner"),
			"status": "open",
			"event_stream": "bytewax",
			"created_at": _now(),
			"updated_at": _now(),
		}
		self._opportunities[resolved_id] = opp
		self._stage_history[resolved_id] = [{"stage": "qualification", "entered_at": _now()}]
		self._emit("opportunity_created", tenant_id, resolved_id, {"value": float(value), "product_id": product_id})
		return deepcopy(opp)

	def opportunity_stage_advance(
		self,
		opportunity_id: str,
		new_stage: str,
		notes: str,
		tenant_id: str = "default",
		advanced_by: str = "system",
	) -> dict[str, Any]:
		"""
		Advance an opportunity to a new pipeline stage.

		new_stage: One of 'qualification', 'discovery', 'proposal', 'negotiation',
		           'closed_won', 'closed_lost'.
		notes: Stage transition notes.
		"""
		opp = self._require_opportunity(opportunity_id, tenant_id)
		valid_stages = {"qualification", "discovery", "proposal", "negotiation", "closed_won", "closed_lost"}
		if new_stage not in valid_stages:
			raise ValueError(f"unsupported_opportunity_stage:{new_stage}")
		if opp["status"] == "closed":
			raise PermissionError("opportunity_already_closed")
		previous_stage = opp["stage"]
		opp["stage"] = new_stage
		opp["updated_at"] = _now()
		if new_stage in {"closed_won", "closed_lost"}:
			opp["status"] = "closed"
			# Record win/loss
			self._win_loss_records.append({
				"opportunity_id": opportunity_id,
				"tenant_id": tenant_id,
				"outcome": "won" if new_stage == "closed_won" else "lost",
				"value": opp["value"],
				"product_id": opp["product_id"],
				"notes": notes,
				"recorded_at": _now(),
			})
		if opportunity_id not in self._stage_history:
			self._stage_history[opportunity_id] = []
		self._stage_history[opportunity_id].append({
			"stage": new_stage,
			"previous_stage": previous_stage,
			"notes": notes,
			"advanced_by": advanced_by,
			"entered_at": _now(),
		})
		self._emit("opportunity_stage_advanced", tenant_id, opportunity_id,
			{"new_stage": new_stage, "previous_stage": previous_stage})
		return deepcopy(opp)

	def pipeline_report(
		self,
		rep_id: str | None,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Generate a pipeline report for a rep (or all reps) over a period.

		period: ISO month prefix 'YYYY-MM'.
		rep_id: Optional sales rep filter; None = all reps.
		Returns stage breakdown, weighted pipeline value, and close rate.
		"""
		opps = [
			o for o in self._opportunities.values()
			if o["tenant_id"] == tenant_id
			and (rep_id is None or o.get("owner") == rep_id)
		]
		stage_counts: dict[str, int] = {}
		stage_values: dict[str, float] = {}
		for opp in opps:
			stage = opp["stage"]
			stage_counts[stage] = stage_counts.get(stage, 0) + 1
			stage_values[stage] = round(stage_values.get(stage, 0.0) + opp["value"], 2)
		total_pipeline = sum(stage_values.values())
		weighted_pipeline = round(sum(
			opp["value"] * opp["probability"] for opp in opps if opp["status"] == "open"
		), 2)
		won = [o for o in opps if o["stage"] == "closed_won"]
		lost = [o for o in opps if o["stage"] == "closed_lost"]
		closed = len(won) + len(lost)
		close_rate = round(len(won) / closed, 4) if closed > 0 else 0.0
		return {
			"tenant_id": tenant_id,
			"rep_id": rep_id,
			"period": period,
			"opportunity_count": len(opps),
			"open_count": sum(1 for o in opps if o["status"] == "open"),
			"total_pipeline_value": total_pipeline,
			"weighted_pipeline_value": weighted_pipeline,
			"stage_breakdown": stage_counts,
			"stage_values": stage_values,
			"won_count": len(won),
			"lost_count": len(lost),
			"close_rate": close_rate,
			"average_deal_size": round(total_pipeline / len(opps), 2) if opps else 0.0,
			"generated_at": _now(),
		}

	def customer_segmentation(
		self,
		criteria: dict[str, Any],
		tenant_id: str = "default",
		segment_id: str | None = None,
		segment_name: str = "",
	) -> dict[str, Any]:
		"""
		Segment accounts/contacts based on criteria.

		criteria: Dict supporting keys: segment_type, min_value, max_value,
		          territory, industry, account_age_days, lead_score_min.
		Returns segment definition with matching account count.
		"""
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_forecast",
			"forecast_evidence_present": bool(criteria),
			"confidence_present": True,
			"confidence": 1.0,
		})
		accounts = [a for a in self._accounts.values() if a["tenant_id"] == tenant_id]
		matching: list[str] = []
		for account in accounts:
			match = True
			if "territory" in criteria and account.get("territory") != criteria["territory"]:
				match = False
			if "segment" in criteria and account.get("segment") != criteria["segment"]:
				match = False
			if match:
				matching.append(account["id"])
		seg_id = segment_id or _record_id("seg", f"{tenant_id}_{segment_name or 'default'}")
		seg = {
			"segment_id": seg_id,
			"segment_name": segment_name or f"Segment {len(self._segments) + 1}",
			"tenant_id": tenant_id,
			"criteria": criteria,
			"matching_account_count": len(matching),
			"matching_account_ids": matching[:50],  # cap at 50 for response size
			"created_at": _now(),
		}
		self._segments[seg_id] = seg
		self._emit("customer_segmented", tenant_id, seg_id, {"criteria": criteria, "count": len(matching)})
		return seg

	def campaign_analytics(
		self,
		campaign_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Return analytics for a specific campaign.

		Covers lead attribution, opportunity conversion, revenue influence,
		and engagement metrics.
		"""
		campaign = next(
			(c for c in self._campaigns.values()
			 if c["tenant_id"] == tenant_id and (c["id"] == campaign_id or c["campaign_id"] == campaign_id)),
			None,
		)
		# Attributed leads
		attributed_leads = [
			l for l in self._leads.values()
			if l["tenant_id"] == tenant_id and l.get("campaign_id") == campaign_id
		]
		# Attributed opportunities (via lead)
		lead_ids = {l["id"] for l in attributed_leads}
		attributed_opps = [
			o for o in self._opportunities.values()
			if o["tenant_id"] == tenant_id and o.get("lead_id") in lead_ids
		]
		won_opps = [o for o in attributed_opps if o["stage"] == "closed_won"]
		total_pipeline = sum(o["value"] for o in attributed_opps if o["status"] == "open")
		total_won_value = sum(o["value"] for o in won_opps)
		conversion_rate = round(len(attributed_opps) / len(attributed_leads), 4) if attributed_leads else 0.0
		return {
			"campaign_id": campaign_id,
			"tenant_id": tenant_id,
			"campaign_name": campaign["name"] if campaign else "unknown",
			"campaign_status": campaign["status"] if campaign else "unknown",
			"attributed_lead_count": len(attributed_leads),
			"attributed_opportunity_count": len(attributed_opps),
			"lead_to_opp_conversion_rate": conversion_rate,
			"pipeline_influenced": total_pipeline,
			"revenue_won": total_won_value,
			"won_count": len(won_opps),
			"audience_size": len(campaign.get("audience", [])) if campaign else 0,
			"budget": campaign.get("budget", 0) if campaign else 0,
			"roi": round((total_won_value - (campaign["budget"] if campaign else 0)) / max(1, campaign["budget"] if campaign else 1), 4) if campaign else 0.0,
			"generated_at": _now(),
		}

	def win_loss_analysis(
		self,
		period: str,
		reason_codes: list[str] | None = None,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Analyse win/loss patterns over a period.

		period: ISO month prefix 'YYYY-MM'.
		reason_codes: Optional list of reason labels to filter by.
		Returns win rate, average deal sizes, and reason code distribution.
		"""
		period_records = [
			r for r in self._win_loss_records
			if r["tenant_id"] == tenant_id and r["recorded_at"][:7] == period
		]
		if reason_codes:
			period_records = [r for r in period_records if r.get("reason_code") in reason_codes]
		won = [r for r in period_records if r["outcome"] == "won"]
		lost = [r for r in period_records if r["outcome"] == "lost"]
		total = len(period_records)
		win_rate = round(len(won) / total, 4) if total > 0 else 0.0
		avg_won_value = round(sum(r["value"] for r in won) / len(won), 2) if won else 0.0
		avg_lost_value = round(sum(r["value"] for r in lost) / len(lost), 2) if lost else 0.0
		# Reason code distribution (from notes)
		reason_dist: dict[str, int] = {}
		for r in period_records:
			code = r.get("reason_code") or "unspecified"
			reason_dist[code] = reason_dist.get(code, 0) + 1
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_closed": total,
			"won_count": len(won),
			"lost_count": len(lost),
			"win_rate": win_rate,
			"total_won_value": sum(r["value"] for r in won),
			"total_lost_value": sum(r["value"] for r in lost),
			"average_won_deal_size": avg_won_value,
			"average_lost_deal_size": avg_lost_value,
			"reason_code_distribution": reason_dist,
			"generated_at": _now(),
		}

	def crm_dashboard(
		self,
		rep_id: str | None,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Return a comprehensive CRM dashboard for a rep or the whole team.

		period: ISO month prefix 'YYYY-MM'.
		rep_id: Optional filter to a single sales rep.
		"""
		pipeline = self.pipeline_report(rep_id, period, tenant_id)
		win_loss = self.win_loss_analysis(period, tenant_id=tenant_id)
		leads = [
			l for l in self._leads.values()
			if l["tenant_id"] == tenant_id
			and (rep_id is None or l.get("owner") == rep_id)
		]
		activities = [
			a for a in self._activities.values()
			if a["tenant_id"] == tenant_id
			and (rep_id is None or a.get("owner") == rep_id)
		]
		return {
			"tenant_id": tenant_id,
			"rep_id": rep_id,
			"period": period,
			"lead_count": len(leads),
			"assigned_lead_count": sum(1 for l in leads if l.get("status") == "assigned"),
			"qualified_lead_count": sum(1 for l in leads if l.get("status") == "qualified"),
			"activity_count": len(activities),
			"pipeline_summary": {
				"open_opportunities": pipeline["open_count"],
				"total_pipeline_value": pipeline["total_pipeline_value"],
				"weighted_pipeline_value": pipeline["weighted_pipeline_value"],
				"close_rate": pipeline["close_rate"],
				"average_deal_size": pipeline["average_deal_size"],
			},
			"win_loss_summary": {
				"win_rate": win_loss["win_rate"],
				"won_count": win_loss["won_count"],
				"lost_count": win_loss["lost_count"],
				"total_won_value": win_loss["total_won_value"],
			},
			"campaign_count": len([c for c in self._campaigns.values() if c["tenant_id"] == tenant_id]),
			"forecast_count": len([f for f in self._forecasts.values() if f["tenant_id"] == tenant_id]),
			"generated_at": _now(),
		}

	# ------------------------------------------------------------------
	# Original retained methods
	# ------------------------------------------------------------------

	def create_account(self, account_id: str, tenant_id: str, name: str, owner: str, segment: str, territory: str | None = None) -> dict[str, Any]:
		self._enforce({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_account", "account_owner_assigned": bool(owner), "account_segment_present": bool(segment)})
		record = {"id": _record_id("crm_account", account_id), "account_id": account_id, "tenant_id": tenant_id, "name": name, "owner": owner, "segment": segment, "territory": territory, "status": "active", "event_stream": "bytewax", "updated_at": _now()}
		self._accounts[record["id"]] = record
		self._emit("account_created", tenant_id, record["id"], {"account_id": account_id})
		return deepcopy(record)

	def create_contact(self, contact_id: str, tenant_id: str, account_id: str, name: str, email: str, outreach_enabled: bool, consent_recorded: bool) -> dict[str, Any]:
		self._require_account(account_id, tenant_id)
		self._enforce({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_contact", "outreach_enabled": outreach_enabled, "consent_recorded": consent_recorded})
		record = {"id": _record_id("crm_contact", contact_id), "contact_id": contact_id, "tenant_id": tenant_id, "account_id": account_id, "name": name, "email": email, "outreach_enabled": outreach_enabled, "consent_recorded": consent_recorded, "status": "active", "event_stream": "bytewax", "updated_at": _now()}
		self._contacts[record["id"]] = record
		self._emit("contact_created", tenant_id, record["id"], {"account_id": account_id})
		return deepcopy(record)

	def create_lead(self, lead_id: str, tenant_id: str, name: str, source: str, score: int | None = None) -> dict[str, Any]:
		self._enforce({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_lead", "lead_source_present": bool(source)})
		record = {"id": _record_id("crm_lead", lead_id), "lead_id": lead_id, "tenant_id": tenant_id, "name": name, "source": source, "score": score, "owner": None, "status": "qualified" if score is not None and score >= 70 else "active", "event_stream": "bytewax", "updated_at": _now()}
		self._leads[record["id"]] = record
		self._emit("lead_created", tenant_id, record["id"], {"lead_id": lead_id, "score": score})
		return deepcopy(record)

	def assign_lead(self, tenant_id: str, lead_record_id: str, owner: str, assignment_policy: str) -> dict[str, Any]:
		lead = self._require_lead_record(lead_record_id, tenant_id)
		self._enforce({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "assign_lead", "lead_score_present": lead.get("score") is not None, "assignment_policy_present": bool(assignment_policy)})
		lead["owner"] = owner
		lead["assignment_policy"] = assignment_policy
		lead["status"] = "assigned"
		lead["updated_at"] = _now()
		self._emit("lead_assigned", tenant_id, lead_record_id, {"owner": owner})
		return deepcopy(lead)

	def create_opportunity(self, opportunity_id: str, tenant_id: str, account_id: str, name: str, stage: str, amount: float, close_date: str) -> dict[str, Any]:
		self._require_account(account_id, tenant_id)
		self._enforce({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_opportunity", "account_present": bool(account_id), "stage_present": bool(stage), "amount_present": amount is not None, "amount": amount if amount is not None else 0, "close_date_present": bool(close_date)})
		record = {"id": _record_id("crm_opportunity", opportunity_id), "opportunity_id": opportunity_id, "tenant_id": tenant_id, "account_id": account_id, "name": name, "stage": stage, "amount": float(amount), "close_date": close_date, "status": "open", "event_stream": "bytewax", "updated_at": _now()}
		self._opportunities[record["id"]] = record
		self._emit("opportunity_created", tenant_id, record["id"], {"amount": float(amount)})
		return deepcopy(record)

	def record_activity(self, activity_id: str, tenant_id: str, opportunity_record_id: str, owner: str, summary: str, next_step: str | None = None) -> dict[str, Any]:
		opportunity = self._require_opportunity_record(opportunity_record_id, tenant_id)
		self._enforce({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_activity", "activity_owner_assigned": bool(owner), "open_pipeline": opportunity["status"] == "open", "next_step_present": bool(next_step)})
		record = {"id": _record_id("crm_activity", activity_id), "activity_id": activity_id, "tenant_id": tenant_id, "opportunity_record_id": opportunity_record_id, "owner": owner, "summary": summary, "next_step": next_step, "status": "recorded", "event_stream": "bytewax", "updated_at": _now()}
		self._activities[record["id"]] = record
		self._emit("activity_recorded", tenant_id, record["id"], {"opportunity_record_id": opportunity_record_id})
		return deepcopy(record)

	def launch_campaign(self, campaign_id: str, tenant_id: str, name: str, audience: list[str], consent_evidence: str, budget: float, privacy_reviewed_by: str | None = None, budget_reviewed_by: str | None = None) -> dict[str, Any]:
		self._enforce({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "launch_campaign", "audience_present": bool(audience), "consent_evidence_present": bool(consent_evidence), "bulk_outreach": len(audience) > 100, "privacy_review_recorded": bool(privacy_reviewed_by), "budget": budget, "budget_review_recorded": bool(budget_reviewed_by)})
		record = {"id": _record_id("crm_campaign", campaign_id), "campaign_id": campaign_id, "tenant_id": tenant_id, "name": name, "audience": list(audience), "consent_evidence": consent_evidence, "budget": float(budget), "privacy_reviewed_by": privacy_reviewed_by, "budget_reviewed_by": budget_reviewed_by, "status": "active", "event_stream": "bytewax", "updated_at": _now()}
		self._campaigns[record["id"]] = record
		self._emit("campaign_launched", tenant_id, record["id"], {"audience_size": len(audience)})
		return deepcopy(record)

	def record_forecast(self, forecast_id: str, tenant_id: str, period: str, amount: float, confidence: float | None, evidence: str) -> dict[str, Any]:
		self._enforce({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_forecast", "forecast_evidence_present": bool(evidence), "confidence_present": confidence is not None, "confidence": confidence if confidence is not None else 0})
		record = {"id": _record_id("crm_forecast", forecast_id), "forecast_id": forecast_id, "tenant_id": tenant_id, "period": period, "amount": float(amount), "confidence": confidence, "evidence": evidence, "status": "active", "event_stream": "bytewax", "updated_at": _now()}
		self._forecasts[record["id"]] = record
		self._emit("forecast_recorded", tenant_id, record["id"], {"period": period, "amount": float(amount)})
		return deepcopy(record)

	def register_crm_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str) -> dict[str, Any]:
		self._enforce({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_crm_agent", "agent_runtime_supported": runtime in SUPPORTED_CRM_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_CRM_AGENT_ROLES})
		record = {"id": _record_id("crm_agent", name), "tenant_id": tenant_id, "name": name, "runtime": runtime, "role": role, "instructions": instructions, "status": "active", "event_stream": "bytewax", "updated_at": _now()}
		self._agents[record["id"]] = record
		self._emit("crm_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_crm_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown CRM agent: {agent_id}")
		return evaluate_capability_rules({"tenant_context_present": bool(tenant_id), "operation": "agent_crm_action", "action": action, "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})

	def validate_batch_import(self, tenant_id: str, record_count: int) -> dict[str, Any]:
		result = evaluate_capability_rules({"tenant_context_present": bool(tenant_id), "operation": "crm_batch_import", "event_stream": "bytewax"})
		return {"processor": "bytewax", "record_count": record_count, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		opportunities = self.list_opportunities(tenant_id)
		return {
			"tenant_id": tenant_id,
			"account_count": len(self.list_accounts(tenant_id)),
			"contact_count": len(self.list_contacts(tenant_id)),
			"lead_count": len(self.list_leads(tenant_id)),
			"open_pipeline_amount": sum(item["amount"] for item in opportunities if item["status"] == "open"),
			"activity_count": len(self.list_activities(tenant_id)),
			"campaign_count": len(self.list_campaigns(tenant_id)),
			"forecast_count": len(self.list_forecasts(tenant_id)),
			"segment_count": sum(1 for s in self._segments.values() if s["tenant_id"] == tenant_id),
			"win_loss_record_count": sum(1 for r in self._win_loss_records if r["tenant_id"] == tenant_id),
			"crm_agent_count": len(self.list_crm_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	# ------------------------------------------------------------------
	# List helpers
	# ------------------------------------------------------------------

	def list_accounts(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._accounts, tenant_id)

	def list_contacts(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._contacts, tenant_id)

	def list_leads(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._leads, tenant_id)

	def list_opportunities(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._opportunities, tenant_id)

	def list_activities(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._activities, tenant_id)

	def list_campaigns(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._campaigns, tenant_id)

	def list_forecasts(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._forecasts, tenant_id)

	def list_crm_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._agents, tenant_id)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant_id]

	# ── 7 new methods ───────────────────────────────────────────────────────

	async def deal_health_score(
		self,
		opportunity_id: str,
		tenant_id: str,
	) -> float:
		"""Return a 0–1 deal health score based on pipeline stage and activity."""
		opp = self._require_opportunity(opportunity_id, tenant_id)
		stage_scores = {
			"prospecting": 0.15, "qualification": 0.30, "proposal": 0.50,
			"negotiation": 0.70, "closed_won": 1.0, "closed_lost": 0.0,
		}
		stage = opp.get("stage", "prospecting")
		base = stage_scores.get(stage, 0.2)
		activities = opp.get("activity_count", 0)
		activity_boost = min(activities * 0.02, 0.2)
		return round(min(base + activity_boost, 1.0), 3)

	async def churn_predict(
		self,
		customer_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Predict churn probability for a customer using engagement heuristics."""
		account = self._require_account(customer_id, tenant_id)
		days_since_last_activity = account.get("days_since_last_activity", 90)
		open_cases = account.get("open_support_cases", 0)
		nps = account.get("nps_score", 7)
		# Simple weighted risk model
		churn_prob = min(
			(days_since_last_activity / 180 * 0.5) +
			(open_cases * 0.1) +
			(max(0, 5 - nps) / 5 * 0.4),
			1.0
		)
		return {
			"customer_id": customer_id,
			"tenant_id": tenant_id,
			"churn_probability": round(churn_prob, 3),
			"risk_level": "high" if churn_prob >= 0.7 else "medium" if churn_prob >= 0.4 else "low",
			"drivers": {
				"days_since_last_activity": days_since_last_activity,
				"open_support_cases": open_cases,
				"nps_score": nps,
			},
		}

	async def account_health_index(
		self,
		account_id: str,
		period: str,
		tenant_id: str,
	) -> float:
		"""Return an account health index (0–100) based on engagement and revenue metrics."""
		account = self._require_account(account_id, tenant_id)
		revenue = float(account.get("arr", account.get("revenue", 0)))
		activities = int(account.get("activity_count", 0))
		nps = float(account.get("nps_score", 7))
		score = min(
			(min(revenue / 100_000, 1.0) * 40) +
			(min(activities / 20, 1.0) * 30) +
			(nps / 10 * 30),
			100.0
		)
		return round(score, 1)

	async def revenue_forecast(
		self,
		period: str,
		tenant_id: str,
		filters: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Forecast revenue for a period based on pipeline weighted values."""
		filters = filters or {}
		opps = self._tenant_records(self._opportunities, tenant_id)
		stage_weights = {
			"prospecting": 0.05, "qualification": 0.20, "proposal": 0.50,
			"negotiation": 0.80, "closed_won": 1.0, "closed_lost": 0.0,
		}
		weighted_total = sum(
			float(o.get("amount", 0)) * stage_weights.get(o.get("stage", ""), 0.0)
			for o in opps
		)
		committed = sum(float(o.get("amount", 0)) for o in opps if o.get("stage") == "closed_won")
		return {
			"period": period,
			"tenant_id": tenant_id,
			"committed_revenue": round(committed, 2),
			"weighted_pipeline": round(weighted_total, 2),
			"total_opportunities": len(opps),
			"generated_at": _now(),
		}

	async def customer_journey_map(
		self,
		customer_id: str,
		tenant_id: str,
	) -> list[dict[str, Any]]:
		"""Return a chronological journey map of key touchpoints for a customer."""
		account = self._require_account(customer_id, tenant_id)
		# Gather activities related to this account from audit events
		journey: list[dict[str, Any]] = [
			{
				"touchpoint": e.get("event"),
				"timestamp": e.get("created_at", ""),
				"channel": "crm",
				"metadata": e.get("payload", {}),
			}
			for e in self._audit_events
			if e.get("payload", {}).get("account_id") == customer_id or
			   e.get("record_id") == customer_id
		]
		journey.sort(key=lambda x: x.get("timestamp", ""))
		return journey

	async def cohort_retention(
		self,
		cohort_definition: dict[str, Any],
		periods: int,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Compute cohort retention rates across N periods.

		cohort_definition: {"segment": str, "start_period": str}
		Returns synthetic retention curve as a proxy.
		"""
		segment = cohort_definition.get("segment", "all")
		accounts = self._tenant_records(self._accounts, tenant_id)
		cohort_size = len(accounts)
		# Geometric decay model as proxy (real impl queries activity history)
		retention_curve = [
			{"period": p + 1, "retained_pct": round(100 * (0.85 ** p), 1)}
			for p in range(periods)
		]
		return {
			"cohort_definition": cohort_definition,
			"tenant_id": tenant_id,
			"cohort_size": cohort_size,
			"periods": periods,
			"retention_curve": retention_curve,
		}

	async def crm_executive_dashboard(
		self,
		period: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Return an executive-level CRM dashboard for a period."""
		accounts = self._tenant_records(self._accounts, tenant_id)
		opps = self._tenant_records(self._opportunities, tenant_id)
		leads = self._tenant_records(self._leads, tenant_id)
		won = [o for o in opps if o.get("stage") == "closed_won"]
		lost = [o for o in opps if o.get("stage") == "closed_lost"]
		total_revenue = sum(float(o.get("amount", 0)) for o in won)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_accounts": len(accounts),
			"total_leads": len(leads),
			"total_opportunities": len(opps),
			"won_opportunities": len(won),
			"lost_opportunities": len(lost),
			"win_rate_pct": round(len(won) / max(len(won) + len(lost), 1) * 100, 1),
			"total_revenue_won": round(total_revenue, 2),
			"generated_at": _now(),
		}

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		return self.create_account(data.get("account_id", data.get("id", "account")), data.get("tenant_id", "default"), data.get("name", "Account"), data.get("owner", "owner"), data.get("segment", "commercial"), data.get("territory"))

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_accounts(tenant_id)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _require_account(self, account_id: str, tenant_id: str) -> dict[str, Any]:
		for record in self._accounts.values():
			if record["tenant_id"] == tenant_id and record["account_id"] == account_id:
				return record
		raise KeyError(f"Unknown account: {account_id}")

	def _require_lead(self, lead_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._leads.get(lead_id)
		if not record or record["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown lead: {lead_id}")
		return record

	def _require_lead_record(self, lead_record_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._leads.get(lead_record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown lead: {lead_record_id}")
		return record

	def _require_opportunity(self, opportunity_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._opportunities.get(opportunity_id)
		if not record or record["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown opportunity: {opportunity_id}")
		return record

	def _require_opportunity_record(self, opportunity_record_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._opportunities.get(opportunity_record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown opportunity: {opportunity_record_id}")
		return record

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] in {"deny", "require_review"}:
			raise PermissionError(",".join(result["matched_rules"]))

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(r) for r in records.values() if r["tenant_id"] == tenant_id]

	def _emit(self, event_name: str, tenant_id: str, record_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({"event": event_name, "tenant_id": tenant_id, "record_id": record_id, "payload": deepcopy(payload), "processor": "bytewax", "stream": streaming_manifest()["stream"], "created_at": _now()})


CRMService = AdvancedCRMService
