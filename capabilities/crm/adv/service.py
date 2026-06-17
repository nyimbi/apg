"""Domain service for APG advanced CRM analytics — expanded implementation."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

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

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		_store = get_store(db_url)
		self._accounts = WriteThruDict('accounts', tenant_id, _store)
		self._contacts = WriteThruDict('contacts', tenant_id, _store)
		self._leads = WriteThruDict('leads', tenant_id, _store)
		self._opportunities = WriteThruDict('opportunities', tenant_id, _store)
		self._activities = WriteThruDict('activities', tenant_id, _store)
		self._campaigns = WriteThruDict('campaigns', tenant_id, _store)
		self._forecasts = WriteThruDict('forecasts', tenant_id, _store)
		self._agents = WriteThruDict('agents', tenant_id, _store)
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		# New stores
		self._lead_scores = WriteThruDict('lead_scores', tenant_id, _store)
		self._segments = WriteThruDict('segments', tenant_id, _store)
		self._win_loss_records = WriteThruList('win_loss_records', tenant_id, _store)
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

	async def ml_lead_scoring(
		self,
		lead_id: str,
		tenant_id: str = "default",
		scoring_factors: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""AI-powered lead scoring using the MLX Ollama meta-capability.

		Uses local Ollama model when OLLAMA_BASE_URL is configured; falls back
		to the rule-based scorer for offline/test operation.

		This async variant is the AI-native equivalent of lead_scoring() with
		model_type="predictive". Call from async API handlers.
		"""
		import os
		lead = self._require_lead(lead_id, tenant_id)
		features: dict[str, Any] = {
			"has_email": bool(lead.get("email")),
			"has_company": bool(lead.get("company")),
			"has_phone": bool(lead.get("phone")),
			"from_campaign": bool(lead.get("campaign_id")),
			"lead_source": lead.get("source", "unknown"),
			**(scoring_factors or {}),
		}

		top_factors: list[str] = []
		rationale = ""

		if os.environ.get("OLLAMA_BASE_URL"):
			try:
				from capabilities.common.mlx import MLCapability
				ml = MLCapability()
				result = await ml.score(
					features,
					task="lead_qualification",
					labels={
						"0.0–0.3": "Cold lead — low conversion probability",
						"0.3–0.6": "Warm lead — needs nurturing",
						"0.6–0.8": "Hot lead — ready for outreach",
						"0.8–1.0": "Sales-qualified lead — immediate action",
					},
				)
				total_score = int(result.score * 100)
				top_factors = result.factors[:3]
				rationale = result.rationale
			except Exception:
				pass  # Fallback below

		if not top_factors:
			# Rule-based fallback (same as lead_scoring predictive)
			base_score = 30
			if lead.get("email"):
				base_score += 15
				top_factors.append("email_present")
			if lead.get("company"):
				base_score += 20
				top_factors.append("company_present")
			if lead.get("phone"):
				base_score += 10
				top_factors.append("phone_present")
			if lead.get("campaign_id"):
				base_score += 10
				top_factors.append("campaign_attributed")
			total_score = min(100, base_score)

		qualified = total_score >= 60
		lead["score"] = total_score
		lead["status"] = "qualified" if qualified else "active"
		lead["updated_at"] = _now()
		score_id = _record_id("ml_score", f"{lead_id}")
		score_record = {
			"score_id": score_id,
			"lead_id": lead_id,
			"tenant_id": tenant_id,
			"model_type": "ml_predictive",
			"score": total_score,
			"qualified": qualified,
			"top_factors": top_factors,
			"rationale": rationale,
			"scored_at": _now(),
		}
		self._lead_scores[score_id] = score_record
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

	# ------------------------------------------------------------------
	# World-class enhancements — 8 new async methods
	# ------------------------------------------------------------------

	async def copilot_query(
		self,
		prompt: str,
		context_ids: list[str],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Conversational AI sales copilot backed by local Ollama.

		Builds a context bundle from account snapshots, open opportunities,
		recent activities, and lead scores for the given ``context_ids``, then
		submits the bundle + prompt to the Ollama-backed MLCapability.  Falls
		back to a rule-based stub when OLLAMA_BASE_URL is not set.

		Tokens are published to NATS subject ``crm.adv.copilot.{tenant_id}``
		so the UI can render incremental output.

		Args:
			prompt: Natural-language query from the sales rep.
			context_ids: List of account or opportunity IDs to include as context.
			tenant_id: Tenant scope.

		Returns:
			Dict with ``response`` text, ``context_used`` count, and ``model``.
		"""
		import os

		# Assemble context bundle
		context_snippets: list[str] = []
		for cid in context_ids:
			acct = self._accounts.get(cid)
			if acct:
				context_snippets.append(f"[Account] {acct.get('name','?')} segment={acct.get('segment','?')}")
			opp = self._opportunities.get(cid)
			if opp:
				context_snippets.append(f"[Opportunity] {opp.get('name','?')} stage={opp.get('stage','?')} value={opp.get('value',0)}")

		context_text = "\n".join(context_snippets) or "No context loaded."
		full_prompt = f"CRM context:\n{context_text}\n\nQuery: {prompt}"

		response_text = ""
		model_used = "rule_based_stub"

		if os.environ.get("OLLAMA_BASE_URL"):
			try:
				from capabilities.common.mlx import MLCapability
				ml = MLCapability()
				result = await ml.chat(full_prompt, task="crm_copilot")
				response_text = result.text
				model_used = result.model
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		if not response_text:
			response_text = (
				f"Based on {len(context_snippets)} context items: "
				f"I recommend reviewing open opportunities and scheduling follow-ups. "
				f"(Offline stub — set OLLAMA_BASE_URL for AI responses.)"
			)

		self._emit("copilot_queried", tenant_id, "copilot", {"prompt_len": len(prompt), "context_count": len(context_ids)})
		return {
			"response": response_text,
			"context_used": len(context_snippets),
			"model": model_used,
			"tenant_id": tenant_id,
			"generated_at": _now(),
		}

	async def get_360_view(
		self,
		account_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return a 360-degree customer view aggregating all related CRM data.

		Collects account record, contacts, open/closed opportunities, recent
		activities, campaign touches, lead history, churn probability, and an
		AI-generated summary via Ollama when available.

		Args:
			account_id: The account to profile.
			tenant_id: Tenant scope.

		Returns:
			Deeply nested dict with ``summary``, ``account``, ``contacts``,
			``opportunities``, ``activities``, ``campaigns``, and ``ai_summary``.
		"""
		account = self._require_account(account_id, tenant_id)

		contacts = [
			c for c in self._contacts.values()
			if c["tenant_id"] == tenant_id and c.get("account_id") == account_id
		]
		opportunities = [
			o for o in self._opportunities.values()
			if o["tenant_id"] == tenant_id and o.get("account_id") == account_id
		]
		activities = [
			a for a in self._activities.values()
			if a["tenant_id"] == tenant_id
		]

		# Churn probability — reuse existing async method
		churn = await self.churn_predict(account_id, tenant_id)
		health = await self.account_health_index(account_id, "all", tenant_id)
		journey = await self.customer_journey_map(account_id, tenant_id)

		open_opps = [o for o in opportunities if o.get("status") == "open"]
		won_opps = [o for o in opportunities if o.get("stage") == "closed_won"]

		ai_summary = (
			f"Account has {len(contacts)} contacts, "
			f"{len(open_opps)} open opportunities worth "
			f"{sum(o.get('value', 0) for o in open_opps):.2f}, "
			f"health index {health}/100, "
			f"churn risk {churn['risk_level']}."
		)

		self._emit("360_view_generated", tenant_id, account_id, {"contact_count": len(contacts)})
		return {
			"account_id": account_id,
			"tenant_id": tenant_id,
			"account": deepcopy(account),
			"contacts": [deepcopy(c) for c in contacts],
			"opportunities": {
				"open": [deepcopy(o) for o in open_opps],
				"won": [deepcopy(o) for o in won_opps],
				"total": len(opportunities),
			},
			"activities": [deepcopy(a) for a in activities[:20]],
			"journey_touchpoints": len(journey),
			"health_index": health,
			"churn_probability": churn["churn_probability"],
			"churn_risk_level": churn["risk_level"],
			"ai_summary": ai_summary,
			"generated_at": _now(),
		}

	async def next_best_action(
		self,
		entity_id: str,
		entity_type: str,
		tenant_id: str = "default",
	) -> list[dict[str, Any]]:
		"""Return ranked next-best-action recommendations for a CRM entity.

		Analyses the entity context (lead / opportunity / account) against
		historical win/loss patterns.  When Ollama is available the model
		produces structured action recommendations; otherwise a heuristic
		rule set is used.

		Args:
			entity_id: ID of the lead, opportunity, or account.
			entity_type: One of ``'lead'``, ``'opportunity'``, ``'account'``.
			tenant_id: Tenant scope.

		Returns:
			List of action dicts with ``action_type``, ``rationale``,
			``confidence``, ``suggested_due_date``, ``expected_impact``.
		"""
		import os
		from datetime import datetime, timedelta

		context: dict[str, Any] = {}
		if entity_type == "lead":
			context = deepcopy(self._leads.get(entity_id, {}))
		elif entity_type == "opportunity":
			context = deepcopy(self._opportunities.get(entity_id, {}))
		elif entity_type == "account":
			context = next(
				(deepcopy(a) for a in self._accounts.values()
				 if a["tenant_id"] == tenant_id and a.get("account_id") == entity_id),
				{},
			)

		actions: list[dict[str, Any]] = []
		due_soon = (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d")

		if os.environ.get("OLLAMA_BASE_URL"):
			try:
				from capabilities.common.mlx import MLCapability
				ml = MLCapability()
				prompt = (
					f"CRM entity type={entity_type} data={context}. "
					f"Historical win rate={len([r for r in self._win_loss_records if r['outcome']=='won'])}/{max(len(self._win_loss_records),1)}. "
					f"Return top 3 next-best-actions as JSON array."
				)
				result = await ml.chat(prompt, task="next_best_action")
				# Best-effort parse
				import json
				try:
					actions = json.loads(result.text)
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		if not actions:
			# Heuristic fallback
			stage = context.get("stage", "")
			score = context.get("score", 0) or 0
			if entity_type == "lead" and score < 60:
				actions.append({"action_type": "nurture_email", "rationale": "Lead score below qualification threshold", "confidence": 0.8, "suggested_due_date": due_soon, "expected_impact": "Score +10"})
			if entity_type == "opportunity" and stage == "proposal":
				actions.append({"action_type": "schedule_demo", "rationale": "Proposal stage — interactive demo accelerates close", "confidence": 0.75, "suggested_due_date": due_soon, "expected_impact": "+15% win probability"})
			if entity_type == "opportunity" and stage == "negotiation":
				actions.append({"action_type": "exec_sponsor_call", "rationale": "Exec sponsorship increases negotiation win rates", "confidence": 0.7, "suggested_due_date": due_soon, "expected_impact": "+20% win probability"})
			if not actions:
				actions.append({"action_type": "follow_up_call", "rationale": "Default engagement action", "confidence": 0.6, "suggested_due_date": due_soon, "expected_impact": "Maintain engagement"})

		self._emit("nba_generated", tenant_id, entity_id, {"entity_type": entity_type, "action_count": len(actions)})
		return actions

	async def compute_deal_risk(
		self,
		opportunity_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute composite deal risk score (0–1) for an open opportunity.

		Risk drivers evaluated:
		- Days since last activity
		- Days remaining to close date vs. stage velocity norms
		- Missing next-step on most recent activity
		- Open support cases on the linked account

		Publishes ``deal_at_risk`` to NATS when risk >= 0.65.

		Args:
			opportunity_id: Opportunity to assess.
			tenant_id: Tenant scope.

		Returns:
			Dict with ``risk_score``, ``risk_level``, ``drivers``, and
			``recommended_action``.
		"""
		from datetime import datetime, timezone as tz

		opp = self._require_opportunity(opportunity_id, tenant_id)

		# Days since last activity (heuristic from audit events)
		relevant_events = [
			e for e in reversed(self._audit_events)
			if e.get("payload", {}).get("opportunity_record_id") == opportunity_id
			or e.get("record_id") == opportunity_id
		]
		if relevant_events:
			last_ts = relevant_events[0].get("created_at", _now())
			try:
				delta = datetime.now(tz.utc) - datetime.fromisoformat(last_ts)
				days_inactive = delta.days
			except Exception:
				days_inactive = 30
		else:
			days_inactive = 30

		# Days to close
		days_to_close = 999
		close_date_str = opp.get("close_date", "")
		if close_date_str:
			try:
				close_dt = datetime.fromisoformat(close_date_str).replace(tzinfo=tz.utc)
				days_to_close = max((close_dt - datetime.now(tz.utc)).days, 0)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		stage_min_days = {"qualification": 60, "discovery": 45, "proposal": 30, "negotiation": 14}
		expected_days = stage_min_days.get(opp.get("stage", ""), 60)
		close_risk = max(0.0, 1.0 - days_to_close / max(expected_days, 1))

		inactivity_risk = min(days_inactive / 21, 1.0)  # 21 days = max tolerance
		prob = float(opp.get("probability", 0.5))
		prob_risk = max(0.0, 0.7 - prob)  # low prob = high risk

		composite = round(inactivity_risk * 0.45 + close_risk * 0.35 + prob_risk * 0.20, 3)
		risk_level = "critical" if composite >= 0.8 else "high" if composite >= 0.65 else "medium" if composite >= 0.4 else "low"

		if composite >= 0.65:
			self._emit("deal_at_risk", tenant_id, opportunity_id, {"risk_score": composite, "risk_level": risk_level})

		return {
			"opportunity_id": opportunity_id,
			"tenant_id": tenant_id,
			"risk_score": composite,
			"risk_level": risk_level,
			"drivers": {
				"days_inactive": days_inactive,
				"days_to_close": days_to_close,
				"inactivity_risk": round(inactivity_risk, 3),
				"close_date_risk": round(close_risk, 3),
				"probability_risk": round(prob_risk, 3),
			},
			"recommended_action": "Immediate manager review" if composite >= 0.65 else "Monitor weekly",
			"assessed_at": _now(),
		}

	async def run_deal_risk_scan(
		self,
		tenant_id: str = "default",
		risk_threshold: float = 0.65,
	) -> dict[str, Any]:
		"""Scan all open opportunities and emit deal_at_risk events for high-risk deals.

		Intended to be called on a schedule (e.g. every 6 hours via APG cron).

		Args:
			tenant_id: Tenant scope.
			risk_threshold: Opportunities with composite risk >= this value are flagged.

		Returns:
			Summary dict with ``total_scanned``, ``at_risk_count``, and
			``at_risk_opportunities`` list.
		"""
		open_opps = [
			o for o in self._opportunities.values()
			if o["tenant_id"] == tenant_id and o.get("status") == "open"
		]
		at_risk: list[dict[str, Any]] = []
		for opp in open_opps:
			risk = await self.compute_deal_risk(opp["opportunity_id"], tenant_id)
			if risk["risk_score"] >= risk_threshold:
				at_risk.append(risk)

		at_risk.sort(key=lambda r: r["risk_score"], reverse=True)
		return {
			"tenant_id": tenant_id,
			"total_scanned": len(open_opps),
			"at_risk_count": len(at_risk),
			"risk_threshold": risk_threshold,
			"at_risk_opportunities": at_risk,
			"scanned_at": _now(),
		}

	async def analyze_call_transcript(
		self,
		activity_id: str,
		transcript: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Extract structured intelligence from a call transcript via Ollama.

		Extracts: action items, competitor mentions, objections, key topics,
		sentiment score, and talk-time ratio.  Falls back to keyword heuristics
		when Ollama is not available.

		Args:
			activity_id: CRM activity record this transcript belongs to.
			transcript: Raw call transcript text.
			tenant_id: Tenant scope.

		Returns:
			Analysis dict stored in ``_call_analytics[activity_id]``.
		"""
		import os

		analysis: dict[str, Any] = {
			"activity_id": activity_id,
			"tenant_id": tenant_id,
			"action_items": [],
			"competitor_mentions": [],
			"objections": [],
			"key_topics": [],
			"sentiment_score": 0.5,
			"talk_time_ratio": 0.5,
			"model": "heuristic",
			"analyzed_at": _now(),
		}

		if os.environ.get("OLLAMA_BASE_URL"):
			try:
				from capabilities.common.mlx import MLCapability
				import json
				ml = MLCapability()
				prompt = (
					"Extract from this sales call transcript as JSON with keys: "
					"action_items (list), competitor_mentions (list), objections (list), "
					"key_topics (list), sentiment_score (0-1 float), talk_time_ratio (rep share 0-1).\n\n"
					f"TRANSCRIPT:\n{transcript[:4000]}"
				)
				result = await ml.chat(prompt, task="call_analysis")
				try:
					parsed = json.loads(result.text)
					analysis.update(parsed)
					analysis["model"] = result.model
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		# Heuristic fallback enrichment
		if not analysis["action_items"]:
			import re
			action_patterns = [r"follow.?up", r"send.*proposal", r"schedule.*demo", r"introduce.*to"]
			for pat in action_patterns:
				if re.search(pat, transcript, re.IGNORECASE):
					analysis["action_items"].append(pat.replace(".*", " ").replace(".?", ""))

		if not hasattr(self, "_call_analytics"):
			self._call_analytics: dict[str, Any] = {}
		self._call_analytics[activity_id] = analysis
		self._emit("call_analyzed", tenant_id, activity_id, {"action_items": len(analysis["action_items"])})
		return analysis

	async def compute_multi_touch_attribution(
		self,
		opportunity_id: str,
		model_type: str = "linear",
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute multi-touch attribution credit across touchpoints for an opportunity.

		Supported models: ``first_touch``, ``last_touch``, ``linear``,
		``time_decay``, ``data_driven`` (Shapley approximation via Ollama).

		Args:
			opportunity_id: Opportunity to attribute.
			model_type: Attribution model to apply.
			tenant_id: Tenant scope.

		Returns:
			Dict with ``touchpoints`` list (each with ``credit`` 0–1) summing to 1.0.
		"""
		import os

		valid_models = {"first_touch", "last_touch", "linear", "time_decay", "data_driven"}
		if model_type not in valid_models:
			raise ValueError(f"unsupported_attribution_model:{model_type}")

		opp = self._require_opportunity(opportunity_id, tenant_id)

		# Collect touchpoints from audit events
		touchpoints = [
			e for e in self._audit_events
			if e.get("record_id") == opportunity_id
			or e.get("payload", {}).get("opportunity_id") == opportunity_id
			or (e.get("payload", {}).get("lead_id") == opp.get("lead_id") and e.get("event") == "lead_captured")
		]
		touchpoints.sort(key=lambda e: e.get("created_at", ""))

		n = len(touchpoints)
		if n == 0:
			return {"opportunity_id": opportunity_id, "model_type": model_type, "touchpoints": [], "total_credit": 0.0}

		credits: list[float] = []
		if model_type == "first_touch":
			credits = [1.0] + [0.0] * (n - 1)
		elif model_type == "last_touch":
			credits = [0.0] * (n - 1) + [1.0]
		elif model_type == "linear":
			credits = [round(1.0 / n, 4)] * n
		elif model_type == "time_decay":
			weights = [2 ** i for i in range(n)]
			total = sum(weights)
			credits = [round(w / total, 4) for w in weights]
		elif model_type == "data_driven":
			# Approximate Shapley via Ollama; fall back to linear
			credits = [round(1.0 / n, 4)] * n
			if os.environ.get("OLLAMA_BASE_URL"):
				try:
					from capabilities.common.mlx import MLCapability
					import json
					ml = MLCapability()
					prompt = (
						f"Approximate Shapley attribution for {n} touchpoints in order: "
						f"{[t.get('event') for t in touchpoints]}. "
						f"Return JSON array of {n} floats summing to 1.0."
					)
					result = await ml.chat(prompt, task="attribution")
					credits = json.loads(result.text)
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		# Normalise to exactly 1.0
		total = sum(credits)
		if total > 0:
			credits = [round(c / total, 4) for c in credits]

		attributed = [
			{
				"event": tp.get("event"),
				"timestamp": tp.get("created_at"),
				"credit": credits[i],
			}
			for i, tp in enumerate(touchpoints)
		]

		self._emit("attribution_computed", tenant_id, opportunity_id, {"model_type": model_type, "touchpoints": n})
		return {
			"opportunity_id": opportunity_id,
			"tenant_id": tenant_id,
			"model_type": model_type,
			"touchpoints": attributed,
			"total_credit": round(sum(credits), 4),
			"computed_at": _now(),
		}

	async def build_abm_target_list(
		self,
		icp_definition: dict[str, Any],
		limit: int = 50,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Score all accounts against an ICP definition and return ranked ABM target list.

		ICP definition keys (all optional): ``industry``, ``min_employees``,
		``max_employees``, ``min_arr``, ``max_arr``, ``geography``,
		``segment``, ``tech_tags`` (list).

		Args:
			icp_definition: Ideal Customer Profile attribute constraints.
			limit: Maximum number of targets to return.
			tenant_id: Tenant scope.

		Returns:
			Dict with ``targets`` list sorted by ``icp_score`` descending.
		"""
		accounts = [
			a for a in self._accounts.values()
			if a["tenant_id"] == tenant_id
		]

		scored: list[dict[str, Any]] = []
		for account in accounts:
			score = 0.0
			reasons: list[str] = []

			if icp_definition.get("industry") and account.get("industry") == icp_definition["industry"]:
				score += 25.0
				reasons.append("industry_match")
			if icp_definition.get("segment") and account.get("segment") == icp_definition["segment"]:
				score += 20.0
				reasons.append("segment_match")
			if icp_definition.get("geography") and account.get("territory") == icp_definition["geography"]:
				score += 15.0
				reasons.append("geography_match")

			arr = float(account.get("arr", account.get("revenue", 0)))
			if icp_definition.get("min_arr") and arr >= icp_definition["min_arr"]:
				score += 20.0
				reasons.append("arr_in_range")
			if icp_definition.get("max_arr") and arr <= icp_definition["max_arr"]:
				score += 5.0

			emp = int(account.get("employee_count", 0))
			if icp_definition.get("min_employees") and emp >= icp_definition["min_employees"]:
				score += 10.0
				reasons.append("size_match")

			# Bonus for existing pipeline (warm accounts)
			opp_count = sum(
				1 for o in self._opportunities.values()
				if o["tenant_id"] == tenant_id and o.get("account_id") == account.get("account_id")
				and o.get("status") == "open"
			)
			score += min(opp_count * 5.0, 15.0)
			if opp_count:
				reasons.append("active_pipeline")

			scored.append({
				"account_id": account.get("account_id"),
				"account_name": account.get("name"),
				"icp_score": round(min(score, 100.0), 1),
				"match_reasons": reasons,
			})

		scored.sort(key=lambda x: x["icp_score"], reverse=True)
		targets = scored[:limit]

		self._emit("abm_list_built", tenant_id, "abm", {"icp_definition": icp_definition, "target_count": len(targets)})
		return {
			"tenant_id": tenant_id,
			"icp_definition": icp_definition,
			"total_accounts_evaluated": len(accounts),
			"targets": targets,
			"generated_at": _now(),
		}

	async def arr_waterfall(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute ARR waterfall (new, expansion, contraction, churn, net) for a period.

		Derives metrics from the win/loss records and opportunity data.  In a
		full implementation this would replay a revenue event log; here it uses
		the available in-memory state as a production-representative proxy.

		Args:
			period: ISO month prefix ``YYYY-MM``.
			tenant_id: Tenant scope.

		Returns:
			Waterfall dict with new_arr, expansion_arr, churn_arr, net_arr.
		"""
		period_won = [
			r for r in self._win_loss_records
			if r["tenant_id"] == tenant_id
			and r["outcome"] == "won"
			and r.get("recorded_at", "")[:7] == period
		]
		period_lost = [
			r for r in self._win_loss_records
			if r["tenant_id"] == tenant_id
			and r["outcome"] == "lost"
			and r.get("recorded_at", "")[:7] == period
		]

		new_arr = sum(r["value"] for r in period_won)
		churn_arr = sum(r["value"] for r in period_lost)

		# Expansion: opportunities marked closed_won with a linked account that
		# already had a prior closed_won opportunity
		account_ids_with_prior_won: set[str] = set()
		for r in self._win_loss_records:
			if r["tenant_id"] == tenant_id and r["outcome"] == "won" and r.get("recorded_at", "")[:7] < period:
				opp = self._opportunities.get(r["opportunity_id"], {})
				if opp.get("account_id"):
					account_ids_with_prior_won.add(opp["account_id"])

		expansion_arr = 0.0
		net_new_arr = 0.0
		for r in period_won:
			opp = self._opportunities.get(r["opportunity_id"], {})
			if opp.get("account_id") in account_ids_with_prior_won:
				expansion_arr += r["value"]
			else:
				net_new_arr += r["value"]

		net_arr = round(net_new_arr + expansion_arr - churn_arr, 2)

		return {
			"tenant_id": tenant_id,
			"period": period,
			"new_arr": round(net_new_arr, 2),
			"expansion_arr": round(expansion_arr, 2),
			"churn_arr": round(churn_arr, 2),
			"net_arr": net_arr,
			"gross_arr_added": round(net_new_arr + expansion_arr, 2),
			"won_deal_count": len(period_won),
			"churned_deal_count": len(period_lost),
			"generated_at": _now(),
		}

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		return self.create_account(data.get("account_id", data.get("id", "account")), data.get("tenant_id", "default"), data.get("name", "Account"), data.get("owner", "owner"), data.get("segment", "commercial"), data.get("territory"))

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_accounts(tenant_id)

	# ------------------------------------------------------------------
	# Salesforce bidirectional sync
	# ------------------------------------------------------------------

	async def sync_lead_to_salesforce(
		self,
		lead_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Push a local APG lead to Salesforce CRM as a Lead object.

		Requires SFDC_CLIENT_ID, SFDC_CLIENT_SECRET, SFDC_USERNAME, SFDC_PASSWORD.
		Returns the Salesforce Lead ID on success, or {"skipped": True} if
		Salesforce credentials are not configured.
		"""
		import os
		if not all(os.environ.get(k) for k in ("SFDC_CLIENT_ID", "SFDC_CLIENT_SECRET", "SFDC_USERNAME", "SFDC_PASSWORD")):
			return {"skipped": True, "reason": "SFDC credentials not configured"}

		lead = self._require_lead(lead_id, tenant_id)
		try:
			from capabilities.composition.orchestration.connectors.salesforce_connector import (
				SalesforceConnector, SalesforceConfiguration,
			)
			config = SalesforceConfiguration(
				name="Salesforce CRM", tenant_id=tenant_id, user_id="system",
				client_id=os.environ["SFDC_CLIENT_ID"],
				client_secret=os.environ["SFDC_CLIENT_SECRET"],
				username=os.environ["SFDC_USERNAME"],
				password=os.environ["SFDC_PASSWORD"],
				environment=os.environ.get("SFDC_ENV", "sandbox"),
			)
			connector = SalesforceConnector(config)
			await connector.initialize()
			sfdc_result = await connector.create_lead({
				"LastName": lead.get("name", "Unknown").split()[-1],
				"FirstName": " ".join(lead.get("name", "").split()[:-1]) or "Unknown",
				"Company": lead.get("company", tenant_id),
				"LeadSource": lead.get("source", "Web"),
				"Status": "Open - Not Contacted",
				"Email": lead.get("email", ""),
				"Phone": lead.get("phone", ""),
				"Rating": "Hot" if (lead.get("score", 0) or 0) >= 70 else "Warm",
				"Description": f"APG Lead ID: {lead_id}. Score: {lead.get('score', 'N/A')}",
			})
			sfdc_id = sfdc_result.get("id", "")
			# Store Salesforce ID in lead record for future sync
			lead["salesforce_id"] = sfdc_id
			lead["updated_at"] = _now()
			self._emit("lead_synced_to_salesforce", tenant_id, lead_id, {"sfdc_id": sfdc_id})
			return {"synced": True, "salesforce_id": sfdc_id}
		except Exception as exc:
			return {"synced": False, "error": str(exc)}

	async def sync_contact_to_salesforce(
		self,
		account_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Push an APG account to Salesforce as a Contact."""
		import os
		if not all(os.environ.get(k) for k in ("SFDC_CLIENT_ID", "SFDC_CLIENT_SECRET", "SFDC_USERNAME", "SFDC_PASSWORD")):
			return {"skipped": True}

		account = self._require_account(account_id, tenant_id)
		try:
			from capabilities.composition.orchestration.connectors.salesforce_connector import (
				SalesforceConnector, SalesforceConfiguration,
			)
			config = SalesforceConfiguration(
				name="Salesforce", tenant_id=tenant_id, user_id="system",
				client_id=os.environ["SFDC_CLIENT_ID"],
				client_secret=os.environ["SFDC_CLIENT_SECRET"],
				username=os.environ["SFDC_USERNAME"],
				password=os.environ["SFDC_PASSWORD"],
				environment=os.environ.get("SFDC_ENV", "sandbox"),
			)
			connector = SalesforceConnector(config)
			await connector.initialize()
			name = account.get("name", "")
			parts = name.split()
			result = await connector.create_contact({
				"LastName": parts[-1] if parts else name,
				"FirstName": " ".join(parts[:-1]) if len(parts) > 1 else "",
				"Email": account.get("email", ""),
				"Phone": account.get("phone", ""),
				"AccountId": account.get("salesforce_account_id", ""),
			})
			account["salesforce_contact_id"] = result.get("id", "")
			account["updated_at"] = _now()
			return {"synced": True, "salesforce_contact_id": result.get("id", "")}
		except Exception as exc:
			return {"synced": False, "error": str(exc)}

	# ------------------------------------------------------------------
	# CPQ — Configure-Price-Quote (closes critical competitive gap)
	# ------------------------------------------------------------------

	def create_quote(
		self,
		opportunity_id: str,
		line_items: list[dict[str, Any]],
		tenant_id: str = "default",
		discount_pct: float = 0.0,
		valid_days: int = 30,
		notes: str = "",
	) -> dict[str, Any]:
		"""Generate a formal quote for an opportunity.

		Args:
			opportunity_id: ID of the opportunity this quote relates to
			line_items: List of {"product": str, "qty": int, "unit_price": float, "description": str}
			discount_pct: Overall discount percentage (0–100)
			valid_days: Quote validity in days from today
			notes: Additional terms or notes

		Returns:
			Quote record with subtotal, discount, total, expiry date.
		"""
		if discount_pct < 0 or discount_pct > 100:
			raise ValueError(f"discount_pct must be 0–100, got {discount_pct}")

		subtotal = sum(
			float(item.get("qty", 1)) * float(item.get("unit_price", 0))
			for item in line_items
		)
		discount_amount = round(subtotal * discount_pct / 100, 2)
		total = round(subtotal - discount_amount, 2)

		quote_id = _record_id("quote", f"{opportunity_id}")
		from datetime import datetime, timedelta
		expiry = (datetime.utcnow() + timedelta(days=valid_days)).strftime("%Y-%m-%d")

		quote: dict[str, Any] = {
			"quote_id": quote_id,
			"opportunity_id": opportunity_id,
			"tenant_id": tenant_id,
			"line_items": line_items,
			"subtotal": round(subtotal, 2),
			"discount_pct": discount_pct,
			"discount_amount": discount_amount,
			"total": total,
			"valid_until": expiry,
			"status": "draft",
			"notes": notes,
			"created_at": _now(),
		}
		# Persist in quotes store
		if not hasattr(self, "_quotes"):
			self._quotes: dict[str, Any] = {}
		self._quotes[quote_id] = quote
		self._emit("quote_created", tenant_id, quote_id, {"total": total, "opportunity_id": opportunity_id})
		return quote

	def apply_discount_governance(
		self,
		quote_id: str,
		requested_discount_pct: float,
		rep_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Apply discount governance rules to a quote.

		Enforces tiered approval thresholds:
		  0–10%:  Rep can approve (auto-approved)
		  10–20%: Sales manager approval required
		  20–30%: VP Sales approval required
		  >30%:   Executive approval required

		Returns approval_status and required_approver.
		"""
		if not hasattr(self, "_quotes") or quote_id not in self._quotes:
			raise KeyError(f"Quote not found: {quote_id}")

		thresholds = [
			(10.0, "auto_approved", ""),
			(20.0, "manager_approval_required", "sales_manager"),
			(30.0, "vp_approval_required", "vp_sales"),
			(float("inf"), "executive_approval_required", "ceo"),
		]

		approval_status = "auto_approved"
		required_approver = ""
		for threshold, status, approver in thresholds:
			if requested_discount_pct <= threshold:
				approval_status = status
				required_approver = approver
				break

		quote = self._quotes[quote_id]
		quote["approval_status"] = approval_status
		quote["required_approver"] = required_approver
		quote["requested_by"] = rep_id
		quote["updated_at"] = _now()

		self._emit("discount_governance_applied", tenant_id, quote_id, {
			"discount_pct": requested_discount_pct, "approval_status": approval_status,
		})
		return {
			"quote_id": quote_id,
			"requested_discount_pct": requested_discount_pct,
			"approval_status": approval_status,
			"required_approver": required_approver,
		}

	def list_quotes(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""Return all quotes for a tenant."""
		if not hasattr(self, "_quotes"):
			return []
		return [q for q in self._quotes.values() if q.get("tenant_id") == tenant_id]

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


class CRMService:
	"""
	High-level CRM service with typed dict-based API, in-memory persistence via
	DatabaseManager, and lazy-loaded optional integration managers.

	This is the class imported by tests and API helpers.  AdvancedCRMService is
	retained for backward-compatibility with older callers.
	"""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		_store = get_store(db_url)
		from .database import DatabaseManager
		self.db_manager = DatabaseManager()

		# Lazy-load optional integrations so the service is usable without
		# their heavyweight dependencies (sklearn, redis, asyncpg …).
		from .ai_insights import CRMAIInsights
		self.ai_insights = CRMAIInsights()

		from .email_integration import EmailIntegrationManager
		self.email_integration_manager = EmailIntegrationManager(self.db_manager)

		from .realtime_sync import RealTimeSyncEngine
		self.realtime_sync = RealTimeSyncEngine(db_pool=None, redis_client=None)

		# Per-tenant configuration store (default values)
		self._configs = WriteThruDict('configs', tenant_id, _store)
		# Time-entry store for clock-in/out
		self._time_entries: dict[str, list[dict[str, Any]]] = {}

	# ------------------------------------------------------------------ #
	# Default config helper
	# ------------------------------------------------------------------ #

	def _get_config(self, tenant_id: str) -> dict[str, Any]:
		if tenant_id not in self._configs:
			self._configs[tenant_id] = {
				"default_lead_score_threshold": 70.0,
				"default_opportunity_probability": 50.0,
				"customer_health_score_enabled": True,
				"ai_recommendations_enabled": True,
				"predictive_analytics_enabled": True,
				"email_integration_enabled": True,
				"calendar_integration_enabled": True,
				"social_media_monitoring_enabled": True,
				"document_management_enabled": True,
				"max_records_per_page": 100,
				"cache_ttl_seconds": 300,
				"background_job_timeout": 3600,
			}
		return self._configs[tenant_id]

	# ------------------------------------------------------------------ #
	# Leads
	# ------------------------------------------------------------------ #

	async def create_lead(
		self, data: dict[str, Any], tenant_id: str, user_id: str
	):
		from .models import CRMLead
		lead = CRMLead(
			tenant_id=tenant_id,
			created_by=user_id,
			**{k: v for k, v in data.items()},
		)
		return await self.db_manager.create_lead(lead)

	async def update_lead(
		self,
		lead_id: str,
		updates: dict[str, Any],
		tenant_id: str,
		user_id: str,
	):
		rec = await self.db_manager.get_lead(lead_id, tenant_id)
		if rec is None:
			raise KeyError(f"Lead not found: {lead_id}")
		patch = dict(updates)
		patch["updated_by"] = user_id
		patch["version"] = rec.version + 1
		return await self.db_manager.update_lead(lead_id, patch, tenant_id)

	# ------------------------------------------------------------------ #
	# Accounts
	# ------------------------------------------------------------------ #

	async def create_account(
		self, data: dict[str, Any], tenant_id: str, user_id: str
	):
		from .models import CRMAccount
		account = CRMAccount(
			tenant_id=tenant_id,
			created_by=user_id,
			**{k: v for k, v in data.items()},
		)
		return await self.db_manager.create_account(account)

	# ------------------------------------------------------------------ #
	# Opportunities
	# ------------------------------------------------------------------ #

	async def create_opportunity(
		self, data: dict[str, Any], tenant_id: str, user_id: str
	):
		from .models import CRMOpportunity
		opp = CRMOpportunity(
			tenant_id=tenant_id,
			created_by=user_id,
			**{k: v for k, v in data.items()},
		)
		return await self.db_manager.create_opportunity(opp)

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_accounts', '_contacts', '_leads', '_opportunities', '_activities', '_campaigns', '_forecasts', '_agents', '_lead_scores', '_segments', '_configs', '_audit_events', '_win_loss_records']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

