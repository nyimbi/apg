"""Executable service layer for APG Social Media Intelligence."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_INFLUENCE_TYPES, SUPPORTED_NETWORK_TYPES, SUPPORTED_PLATFORM_TYPES, SUPPORTED_POST_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SIGNAL_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_TOPIC_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import InfluenceAssessment, NetworkAssessment, SOCINTAgent, SOCINTDissemination, SOCINTReferral, SOCINTReview, SocialAuthority, SocialPost, SocialSignal, SocialSource, SocialTopic
	from .socint_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_INFLUENCE_TYPES, SUPPORTED_NETWORK_TYPES, SUPPORTED_PLATFORM_TYPES, SUPPORTED_POST_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SIGNAL_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_TOPIC_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import InfluenceAssessment, NetworkAssessment, SOCINTAgent, SOCINTDissemination, SOCINTReferral, SOCINTReview, SocialAuthority, SocialPost, SocialSignal, SocialSource, SocialTopic  # type: ignore
	from socint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


class SocialMediaIntelligenceService:
	"""Tenant-scoped SOCINT coordination runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], SocialAuthority] = {}
		self.topics: dict[tuple[str, str], SocialTopic] = {}
		self.sources: dict[tuple[str, str], SocialSource] = {}
		self.posts: dict[tuple[str, str], SocialPost] = {}
		self.signals: dict[tuple[str, str], SocialSignal] = {}
		self.influence: dict[tuple[str, str], InfluenceAssessment] = {}
		self.networks: dict[tuple[str, str], NetworkAssessment] = {}
		self.referrals: dict[tuple[str, str], SOCINTReferral] = {}
		self.disseminations: dict[tuple[str, str], SOCINTDissemination] = {}
		self.reviews: dict[tuple[str, str], SOCINTReview] = {}
		self.agents: dict[tuple[str, str], SOCINTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = SocialAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "socint_authority_recorded", authority_id)
		return item.to_dict()

	def record_topic(self, topic_id: str, tenant_id: str, topic_type: str, name: str, priority: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		topic_type = normalize_code(topic_type)
		priority = normalize_code(priority)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_topic", "topic_type_supported": topic_type in SUPPORTED_TOPIC_TYPES, "topic_name_present": present(name), "priority_supported": priority in SUPPORTED_RISK_LEVELS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = SocialTopic(topic_id, tenant_id, topic_type, name, priority, authority_id, evidence_reference)
		self.topics[self._tenant_key(tenant_id, topic_id)] = item
		self._audit(tenant_id, "socint_topic_recorded", topic_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, source_type: str, platform_type: str, source_reference: str, owner_id: str, authority_id: str, terms_review_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		platform_type = normalize_code(platform_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "platform_type_supported": platform_type in SUPPORTED_PLATFORM_TYPES, "source_reference_present": present(source_reference), "owner_present": present(owner_id), "authority_present": authority is not None, "terms_review_present": present(terms_review_reference), "evidence_present": present(evidence_reference)})
		item = SocialSource(source_id, tenant_id, source_type, platform_type, source_reference, owner_id, authority_id, terms_review_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "socint_source_registered", source_id)
		return item.to_dict()

	def record_post(self, post_id: str, tenant_id: str, topic_id: str, source_id: str, post_type: str, post_reference: str, content_fingerprint: str, observed_at: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		topic = self._tenant_topic_or_none(topic_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		post_type = normalize_code(post_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_post", "topic_present": topic is not None, "source_present": source is not None, "topic_source_authority_match": topic is not None and source is not None and topic.authority_id == source.authority_id, "post_type_supported": post_type in SUPPORTED_POST_TYPES, "post_reference_present": present(post_reference), "fingerprint_present": present(content_fingerprint), "observed_at_present": present(observed_at), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = SocialPost(post_id, tenant_id, topic_id, source_id, post_type, post_reference, content_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.posts[self._tenant_key(tenant_id, post_id)] = item
		self._audit(tenant_id, "socint_post_recorded", post_id)
		return item.to_dict()

	def record_signal(self, signal_id: str, tenant_id: str, post_id: str, signal_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		post = self._tenant_post_or_none(post_id, tenant_id)
		signal_type = normalize_code(signal_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_signal", "post_present": post is not None, "signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = SocialSignal(signal_id, tenant_id, post_id, signal_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.signals[self._tenant_key(tenant_id, signal_id)] = item
		self._audit(tenant_id, "socint_signal_recorded", signal_id)
		return item.to_dict()

	def record_influence(self, assessment_id: str, tenant_id: str, signal_id: str, influence_type: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		influence_type = normalize_code(influence_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_influence", "signal_present": signal is not None, "influence_type_supported": influence_type in SUPPORTED_INFLUENCE_TYPES, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = InfluenceAssessment(assessment_id, tenant_id, signal_id, influence_type, float(confidence_score), analyst_id, evidence_reference)
		self.influence[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "socint_influence_recorded", assessment_id)
		return item.to_dict()

	def record_network(self, assessment_id: str, tenant_id: str, signal_id: str, network_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		network_type = normalize_code(network_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_network", "signal_present": signal is not None, "network_type_supported": network_type in SUPPORTED_NETWORK_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = NetworkAssessment(assessment_id, tenant_id, signal_id, network_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.networks[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "socint_network_recorded", assessment_id)
		return item.to_dict()

	def record_referral(self, referral_id: str, tenant_id: str, assessment_id: str, referral_type: str, recipient: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._assessment_or_none(assessment_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_referral", "assessment_present": assessment is not None, "referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES, "recipient_present": present(recipient), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = SOCINTReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "socint_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, assessment_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._assessment_or_none(assessment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "assessment_present": assessment is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = SOCINTDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "socint_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = SOCINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "socint_review_recorded", reference_id)
		return item.to_dict()

	def register_socint_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_socint_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = SOCINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "socint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, platform_abuse_scope: bool = False, harassment_scope: bool = False, doxxing_scope: bool = False, evasion_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "socint_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "platform_abuse_scope": platform_abuse_scope, "harassment_scope": harassment_scope, "doxxing_scope": doxxing_scope, "evasion_scope": evasion_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "socint_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.socint.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "topic_count": self._count(self.topics, tenant_id), "source_count": self._count(self.sources, tenant_id), "post_count": self._count(self.posts, tenant_id), "signal_count": self._count(self.signals, tenant_id), "influence_count": self._count(self.influence, tenant_id), "network_count": self._count(self.networks, tenant_id), "referral_count": self._count(self.referrals, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> SocialAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_topic_or_none(self, item_id: str, tenant_id: str) -> SocialTopic | None:
		return self.topics.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> SocialSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_post_or_none(self, item_id: str, tenant_id: str) -> SocialPost | None:
		return self.posts.get(self._tenant_key(tenant_id, item_id))

	def _tenant_signal_or_none(self, item_id: str, tenant_id: str) -> SocialSignal | None:
		return self.signals.get(self._tenant_key(tenant_id, item_id))

	def _assessment_or_none(self, item_id: str, tenant_id: str) -> InfluenceAssessment | NetworkAssessment | None:
		return self.influence.get(self._tenant_key(tenant_id, item_id)) or self.networks.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "socint_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "socint_policy_denied")


IntelSOCINTService = SocialMediaIntelligenceService
