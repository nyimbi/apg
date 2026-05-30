"""Domain service for APG advanced CRM analytics."""

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
	from capability_contract import (
		SUPPORTED_CRM_AGENT_ROLES,
		SUPPORTED_CRM_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)


class AdvancedCRMService:
	"""Tenant-scoped account, lead, pipeline, campaign, and forecast coordinator."""

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

	def create_account(self, account_id: str, tenant_id: str, name: str, owner: str, segment: str, territory: str | None = None) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_account",
			"account_owner_assigned": bool(owner),
			"account_segment_present": bool(segment),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("crm_account", account_id),
			"account_id": account_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"segment": segment,
			"territory": territory,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._accounts[record["id"]] = record
		self._emit("account_created", tenant_id, record["id"], {"account_id": account_id})
		return deepcopy(record)

	def create_contact(self, contact_id: str, tenant_id: str, account_id: str, name: str, email: str, outreach_enabled: bool, consent_recorded: bool) -> dict[str, Any]:
		self._require_account(account_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_contact",
			"outreach_enabled": outreach_enabled,
			"consent_recorded": consent_recorded,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("crm_contact", contact_id),
			"contact_id": contact_id,
			"tenant_id": tenant_id,
			"account_id": account_id,
			"name": name,
			"email": email,
			"outreach_enabled": outreach_enabled,
			"consent_recorded": consent_recorded,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._contacts[record["id"]] = record
		self._emit("contact_created", tenant_id, record["id"], {"account_id": account_id})
		return deepcopy(record)

	def create_lead(self, lead_id: str, tenant_id: str, name: str, source: str, score: int | None = None) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_lead",
			"lead_source_present": bool(source),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("crm_lead", lead_id),
			"lead_id": lead_id,
			"tenant_id": tenant_id,
			"name": name,
			"source": source,
			"score": score,
			"owner": None,
			"status": "qualified" if score is not None and score >= 70 else "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._leads[record["id"]] = record
		self._emit("lead_created", tenant_id, record["id"], {"lead_id": lead_id, "score": score})
		return deepcopy(record)

	def assign_lead(self, tenant_id: str, lead_record_id: str, owner: str, assignment_policy: str) -> dict[str, Any]:
		lead = self._require_lead_record(lead_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "assign_lead",
			"lead_score_present": lead.get("score") is not None,
			"assignment_policy_present": bool(assignment_policy),
		}
		self._enforce(context)
		lead["owner"] = owner
		lead["assignment_policy"] = assignment_policy
		lead["status"] = "assigned"
		lead["updated_at"] = self._now()
		self._emit("lead_assigned", tenant_id, lead_record_id, {"owner": owner})
		return deepcopy(lead)

	def create_opportunity(
		self,
		opportunity_id: str,
		tenant_id: str,
		account_id: str,
		name: str,
		stage: str,
		amount: float,
		close_date: str,
	) -> dict[str, Any]:
		self._require_account(account_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_opportunity",
			"account_present": bool(account_id),
			"stage_present": bool(stage),
			"amount_present": amount is not None,
			"amount": amount if amount is not None else 0,
			"close_date_present": bool(close_date),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("crm_opportunity", opportunity_id),
			"opportunity_id": opportunity_id,
			"tenant_id": tenant_id,
			"account_id": account_id,
			"name": name,
			"stage": stage,
			"amount": float(amount),
			"close_date": close_date,
			"status": "open",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._opportunities[record["id"]] = record
		self._emit("opportunity_created", tenant_id, record["id"], {"amount": float(amount)})
		return deepcopy(record)

	def record_activity(self, activity_id: str, tenant_id: str, opportunity_record_id: str, owner: str, summary: str, next_step: str | None = None) -> dict[str, Any]:
		opportunity = self._require_opportunity_record(opportunity_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_activity",
			"activity_owner_assigned": bool(owner),
			"open_pipeline": opportunity["status"] == "open",
			"next_step_present": bool(next_step),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("crm_activity", activity_id),
			"activity_id": activity_id,
			"tenant_id": tenant_id,
			"opportunity_record_id": opportunity_record_id,
			"owner": owner,
			"summary": summary,
			"next_step": next_step,
			"status": "recorded",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._activities[record["id"]] = record
		self._emit("activity_recorded", tenant_id, record["id"], {"opportunity_record_id": opportunity_record_id})
		return deepcopy(record)

	def launch_campaign(self, campaign_id: str, tenant_id: str, name: str, audience: list[str], consent_evidence: str, budget: float, privacy_reviewed_by: str | None = None, budget_reviewed_by: str | None = None) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "launch_campaign",
			"audience_present": bool(audience),
			"consent_evidence_present": bool(consent_evidence),
			"bulk_outreach": len(audience) > 100,
			"privacy_review_recorded": bool(privacy_reviewed_by),
			"budget": budget,
			"budget_review_recorded": bool(budget_reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("crm_campaign", campaign_id),
			"campaign_id": campaign_id,
			"tenant_id": tenant_id,
			"name": name,
			"audience": list(audience),
			"consent_evidence": consent_evidence,
			"budget": float(budget),
			"privacy_reviewed_by": privacy_reviewed_by,
			"budget_reviewed_by": budget_reviewed_by,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._campaigns[record["id"]] = record
		self._emit("campaign_launched", tenant_id, record["id"], {"audience_size": len(audience)})
		return deepcopy(record)

	def record_forecast(self, forecast_id: str, tenant_id: str, period: str, amount: float, confidence: float | None, evidence: str) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_forecast",
			"forecast_evidence_present": bool(evidence),
			"confidence_present": confidence is not None,
			"confidence": confidence if confidence is not None else 0,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("crm_forecast", forecast_id),
			"forecast_id": forecast_id,
			"tenant_id": tenant_id,
			"period": period,
			"amount": float(amount),
			"confidence": confidence,
			"evidence": evidence,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._forecasts[record["id"]] = record
		self._emit("forecast_recorded", tenant_id, record["id"], {"period": period, "amount": float(amount)})
		return deepcopy(record)

	def register_crm_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_crm_agent",
			"agent_runtime_supported": runtime in SUPPORTED_CRM_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_CRM_AGENT_ROLES,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("crm_agent", name),
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"instructions": instructions,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._agents[record["id"]] = record
		self._emit("crm_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_crm_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown CRM agent: {agent_id}")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "agent_crm_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		}
		return evaluate_capability_rules(context)

	def validate_batch_import(self, tenant_id: str, record_count: int) -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation": "crm_batch_import", "event_stream": "bytewax"}
		result = evaluate_capability_rules(context)
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
			"crm_agent_count": len(self.list_crm_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

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
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		return self.create_account(
			data.get("account_id", data.get("id", "account")),
			data.get("tenant_id", "default"),
			data.get("name", "Account"),
			data.get("owner", "owner"),
			data.get("segment", "commercial"),
			data.get("territory"),
		)

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_accounts(tenant_id)

	def _require_account(self, account_id: str, tenant_id: str) -> dict[str, Any]:
		for record in self._accounts.values():
			if record["tenant_id"] == tenant_id and record["account_id"] == account_id:
				return record
		raise KeyError(f"Unknown account: {account_id}")

	def _require_lead_record(self, lead_record_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._leads.get(lead_record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown lead: {lead_record_id}")
		return record

	def _require_opportunity_record(self, opportunity_record_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._opportunities.get(opportunity_record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown opportunity: {opportunity_record_id}")
		return record

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review":
			raise PermissionError(",".join(result["matched_rules"]))

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(record) for record in records.values() if record["tenant_id"] == tenant_id]

	def _emit(self, event_name: str, tenant_id: str, record_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"event": event_name,
			"tenant_id": tenant_id,
			"record_id": record_id,
			"payload": deepcopy(payload),
			"processor": "bytewax",
			"stream": streaming_manifest()["stream"],
			"created_at": self._now(),
		})

	def _record_id(self, prefix: str, value: str) -> str:
		slug = "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")
		return f"{prefix}_{slug or 'record'}"

	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()


CRMService = AdvancedCRMService
