"""View models for APG advanced CRM analytics screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_CRM_AGENT_ROLES, SUPPORTED_CRM_AGENT_RUNTIMES, get_capability_contract
	from .service import AdvancedCRMService
except ImportError:
	from capability_contract import SUPPORTED_CRM_AGENT_ROLES, SUPPORTED_CRM_AGENT_RUNTIMES, get_capability_contract
	from service import AdvancedCRMService


def navigation_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"capability": contract["capability"], "routes": contract["ui"]["routes"], "theme": contract["theme"], "api_prefix": contract["ui"]["api_prefix"]}


def dashboard_model(service: AdvancedCRMService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "dashboard", "title": "Advanced CRM Analytics", "summary": service.dashboard_summary(tenant_id), "sections": ["pipeline_health", "lead_quality", "account_segments", "forecast_confidence"]}


def account_model(service: AdvancedCRMService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "accounts", "records": service.list_accounts(tenant_id), "columns": ["account_id", "name", "owner", "segment", "territory", "status"], "actions": ["create_account", "map_relationships"]}


def contact_model(service: AdvancedCRMService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "contacts", "records": service.list_contacts(tenant_id), "columns": ["contact_id", "account_id", "name", "email", "consent_recorded", "status"], "actions": ["create_contact", "record_consent"]}


def lead_model(service: AdvancedCRMService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "leads", "records": service.list_leads(tenant_id), "columns": ["lead_id", "name", "source", "score", "owner", "status"], "actions": ["create_lead", "score", "assign"]}


def pipeline_model(service: AdvancedCRMService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "pipeline", "records": service.list_opportunities(tenant_id), "columns": ["opportunity_id", "account_id", "stage", "amount", "close_date", "status"], "actions": ["create_opportunity", "record_activity", "close"]}


def activity_model(service: AdvancedCRMService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "activities", "records": service.list_activities(tenant_id), "columns": ["activity_id", "opportunity_record_id", "owner", "summary", "next_step"], "actions": ["record_activity", "schedule_next_step"]}


def campaign_model(service: AdvancedCRMService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "campaigns", "records": service.list_campaigns(tenant_id), "columns": ["campaign_id", "name", "budget", "privacy_reviewed_by", "status"], "actions": ["launch_campaign", "record_privacy_review"]}


def forecast_model(service: AdvancedCRMService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "forecasts", "records": service.list_forecasts(tenant_id), "columns": ["forecast_id", "period", "amount", "confidence", "status"], "actions": ["record_forecast", "review_confidence"]}


def agent_workbench_model(service: AdvancedCRMService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "agents", "records": service.list_crm_agents(tenant_id), "supported_runtimes": SUPPORTED_CRM_AGENT_RUNTIMES, "supported_roles": SUPPORTED_CRM_AGENT_ROLES, "actions": ["register_agent", "validate_action", "record_human_approval"]}
