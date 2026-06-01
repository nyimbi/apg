"""Executable service layer for APG Threat Intelligence."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_ACTOR_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CAMPAIGN_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_INDICATOR_TYPES, SUPPORTED_MITIGATION_TYPES, SUPPORTED_REPORT_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import ThreatActor, ThreatAgent, ThreatAssessment, ThreatAuthority, ThreatCampaign, ThreatIndicator, ThreatMitigation, ThreatReport, ThreatReview, ThreatSource, ThreatWorkspace
	from .threat_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_ACTOR_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CAMPAIGN_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_INDICATOR_TYPES, SUPPORTED_MITIGATION_TYPES, SUPPORTED_REPORT_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import ThreatActor, ThreatAgent, ThreatAssessment, ThreatAuthority, ThreatCampaign, ThreatIndicator, ThreatMitigation, ThreatReport, ThreatReview, ThreatSource, ThreatWorkspace  # type: ignore
	from threat_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


class ThreatIntelligenceService:
	"""Tenant-scoped threat-intelligence runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], ThreatAuthority] = {}
		self.workspaces: dict[tuple[str, str], ThreatWorkspace] = {}
		self.sources: dict[tuple[str, str], ThreatSource] = {}
		self.indicators: dict[tuple[str, str], ThreatIndicator] = {}
		self.actors: dict[tuple[str, str], ThreatActor] = {}
		self.campaigns: dict[tuple[str, str], ThreatCampaign] = {}
		self.assessments: dict[tuple[str, str], ThreatAssessment] = {}
		self.reports: dict[tuple[str, str], ThreatReport] = {}
		self.mitigations: dict[tuple[str, str], ThreatMitigation] = {}
		self.reviews: dict[tuple[str, str], ThreatReview] = {}
		self.agents: dict[tuple[str, str], ThreatAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = ThreatAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "threat_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(self, workspace_id: str, tenant_id: str, workspace_type: str, name: str, classification: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_workspace", "workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES, "workspace_name_present": present(name), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = ThreatWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "threat_workspace_recorded", workspace_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, workspace_id: str, source_type: str, source_reference: str, custodian_id: str, lineage_reference: str, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "workspace_present": workspace is not None, "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "source_reference_present": present(source_reference), "custodian_present": present(custodian_id), "lineage_present": present(lineage_reference), "evidence_present": present(evidence_reference)})
		item = ThreatSource(source_id, tenant_id, workspace_id, source_type, source_reference, custodian_id, lineage_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "threat_source_registered", source_id)
		return item.to_dict()

	def record_indicator(self, indicator_id: str, tenant_id: str, source_id: str, indicator_type: str, indicator_reference: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		indicator_type = normalize_code(indicator_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_indicator", "source_present": source is not None, "indicator_type_supported": indicator_type in SUPPORTED_INDICATOR_TYPES, "indicator_reference_present": present(indicator_reference), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = ThreatIndicator(indicator_id, tenant_id, source_id, indicator_type, indicator_reference, float(confidence_score), evidence_reference)
		self.indicators[self._tenant_key(tenant_id, indicator_id)] = item
		self._audit(tenant_id, "threat_indicator_recorded", indicator_id)
		return item.to_dict()

	def record_actor(self, actor_id: str, tenant_id: str, workspace_id: str, actor_type: str, actor_reference: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		actor_type = normalize_code(actor_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_actor", "workspace_present": workspace is not None, "actor_type_supported": actor_type in SUPPORTED_ACTOR_TYPES, "actor_reference_present": present(actor_reference), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = ThreatActor(actor_id, tenant_id, workspace_id, actor_type, actor_reference, float(confidence_score), evidence_reference)
		self.actors[self._tenant_key(tenant_id, actor_id)] = item
		self._audit(tenant_id, "threat_actor_recorded", actor_id)
		return item.to_dict()

	def record_campaign(self, campaign_id: str, tenant_id: str, actor_id: str, campaign_type: str, campaign_reference: str, risk_level: str, evidence_reference: str) -> dict[str, Any]:
		actor = self._tenant_actor_or_none(actor_id, tenant_id)
		campaign_type = normalize_code(campaign_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_campaign", "actor_present": actor is not None, "campaign_type_supported": campaign_type in SUPPORTED_CAMPAIGN_TYPES, "campaign_reference_present": present(campaign_reference), "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "evidence_present": present(evidence_reference)})
		item = ThreatCampaign(campaign_id, tenant_id, actor_id, campaign_type, campaign_reference, risk_level, evidence_reference)
		self.campaigns[self._tenant_key(tenant_id, campaign_id)] = item
		self._audit(tenant_id, "threat_campaign_recorded", campaign_id)
		return item.to_dict()

	def record_assessment(self, assessment_id: str, tenant_id: str, campaign_id: str, assessment_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		campaign = self._tenant_campaign_or_none(campaign_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_assessment", "campaign_present": campaign is not None, "assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = ThreatAssessment(assessment_id, tenant_id, campaign_id, assessment_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.assessments[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "threat_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_report(self, report_id: str, tenant_id: str, assessment_id: str, report_type: str, report_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_assessment_or_none(assessment_id, tenant_id)
		report_type = normalize_code(report_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_report", "assessment_present": assessment is not None, "report_type_supported": report_type in SUPPORTED_REPORT_TYPES, "report_reference_present": present(report_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = ThreatReport(report_id, tenant_id, assessment_id, report_type, report_reference, approval_reference, evidence_reference)
		self.reports[self._tenant_key(tenant_id, report_id)] = item
		self._audit(tenant_id, "threat_report_recorded", report_id)
		return item.to_dict()

	def record_mitigation(self, mitigation_id: str, tenant_id: str, assessment_id: str, mitigation_type: str, action_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_assessment_or_none(assessment_id, tenant_id)
		mitigation_type = normalize_code(mitigation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_mitigation", "assessment_present": assessment is not None, "mitigation_type_supported": mitigation_type in SUPPORTED_MITIGATION_TYPES, "action_present": present(action_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = ThreatMitigation(mitigation_id, tenant_id, assessment_id, mitigation_type, action_reference, approval_reference, evidence_reference)
		self.mitigations[self._tenant_key(tenant_id, mitigation_id)] = item
		self._audit(tenant_id, "threat_mitigation_recorded", mitigation_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = ThreatReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "threat_review_recorded", reference_id)
		return item.to_dict()

	def register_threat_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_threat_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES, "agent_name_present": present(name), "agent_scope_present": present(scope)})
		item = ThreatAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "threat_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, unsupported_attribution_scope: bool = False, fabricated_indicator_scope: bool = False, source_tampering_scope: bool = False, privacy_bypass_scope: bool = False, autonomous_mitigation_scope: bool = False, unapproved_publication_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "threat_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "unsupported_attribution_scope": unsupported_attribution_scope, "fabricated_indicator_scope": fabricated_indicator_scope, "source_tampering_scope": source_tampering_scope, "privacy_bypass_scope": privacy_bypass_scope, "autonomous_mitigation_scope": autonomous_mitigation_scope, "unapproved_publication_scope": unapproved_publication_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "threat_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.threats.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "workspace_count": self._count(self.workspaces, tenant_id), "source_count": self._count(self.sources, tenant_id), "indicator_count": self._count(self.indicators, tenant_id), "actor_count": self._count(self.actors, tenant_id), "campaign_count": self._count(self.campaigns, tenant_id), "assessment_count": self._count(self.assessments, tenant_id), "report_count": self._count(self.reports, tenant_id), "mitigation_count": self._count(self.mitigations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> ThreatAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> ThreatWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> ThreatSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_actor_or_none(self, item_id: str, tenant_id: str) -> ThreatActor | None:
		return self.actors.get(self._tenant_key(tenant_id, item_id))

	def _tenant_campaign_or_none(self, item_id: str, tenant_id: str) -> ThreatCampaign | None:
		return self.campaigns.get(self._tenant_key(tenant_id, item_id))

	def _tenant_assessment_or_none(self, item_id: str, tenant_id: str) -> ThreatAssessment | None:
		return self.assessments.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "threat_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "threat_policy_denied")


IntelThreatsService = ThreatIntelligenceService

