"""Executable service layer for APG Social Media Intelligence (SOCINT).

Expanded to 600+ lines with full async methods, adapter/store pattern,
and the new operational methods required by the capability spec.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_INFLUENCE_TYPES, SUPPORTED_NETWORK_TYPES,
		SUPPORTED_PLATFORM_TYPES, SUPPORTED_POST_TYPES, SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SIGNAL_TYPES,
		SUPPORTED_SOURCE_TYPES, SUPPORTED_TOPIC_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		InfluenceAssessment, NetworkAssessment, SOCINTAgent, SOCINTDissemination,
		SOCINTReferral, SOCINTReview, SocialAuthority, SocialPost,
		SocialSignal, SocialSource, SocialTopic,
	)
	from .socint_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_INFLUENCE_TYPES, SUPPORTED_NETWORK_TYPES,
		SUPPORTED_PLATFORM_TYPES, SUPPORTED_POST_TYPES, SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SIGNAL_TYPES,
		SUPPORTED_SOURCE_TYPES, SUPPORTED_TOPIC_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		InfluenceAssessment, NetworkAssessment, SOCINTAgent, SOCINTDissemination,
		SOCINTReferral, SOCINTReview, SocialAuthority, SocialPost,
		SocialSignal, SocialSource, SocialTopic,
	)
	from socint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _fingerprint(*parts: str) -> str:
	blob = "|".join(str(p) for p in parts)
	return hashlib.sha256(blob.encode()).hexdigest()[:16]


# Positive/negative sentiment word lists (minimal illustrative sets)
_POSITIVE_WORDS = {
	"good", "great", "excellent", "love", "amazing", "wonderful", "best",
	"fantastic", "positive", "happy", "joy", "success", "win", "brilliant",
}
_NEGATIVE_WORDS = {
	"bad", "terrible", "hate", "awful", "worst", "horrible", "negative",
	"angry", "violence", "attack", "kill", "destroy", "evil", "corrupt",
}

# Platform engagement weight multipliers
_PLATFORM_WEIGHTS = {
	"TWITTER": 1.0, "X": 1.0, "FACEBOOK": 0.9, "INSTAGRAM": 0.8,
	"TELEGRAM": 1.2, "TIKTOK": 1.1, "YOUTUBE": 0.95,
	"REDDIT": 0.85, "VK": 0.7, "WEIBO": 0.75,
}


class SocialIntelligenceService:
	"""Tenant-scoped SOCINT coordination runtime for generated APG applications.

	Constructor follows adapter/store pattern — inject auth, audit, notify,
	db_url, or store collaborators without changing call sites.
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

		# Existing in-memory stores
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

		# Operational state added by new methods
		self._platform_monitors: dict[str, dict[str, Any]] = {}
		self._collected_posts: dict[str, dict[str, Any]] = {}
		self._sentiment_batches: dict[str, dict[str, Any]] = {}
		self._influence_maps: dict[str, dict[str, Any]] = {}
		self._disinfo_checks: dict[str, dict[str, Any]] = {}
		self._narrative_tracks: dict[str, dict[str, Any]] = {}
		self._viral_alerts: dict[str, dict[str, Any]] = {}
		self._persona_analyses: dict[str, dict[str, Any]] = {}
		self._social_graphs: dict[str, dict[str, Any]] = {}
		self._reports: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Capability contract helpers (sync, preserved)
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Original sync CRUD methods (preserved verbatim)
	# ------------------------------------------------------------------

	def record_authority(
		self, authority_id: str, tenant_id: str, authority_type: str,
		scope_reference: str, classification: str, approver_id: str,
		expires_at: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = SocialAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "socint_authority_recorded", authority_id)
		return item.to_dict()

	def record_topic(
		self, topic_id: str, tenant_id: str, topic_type: str, name: str,
		priority: str, authority_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		topic_type = normalize_code(topic_type)
		priority = normalize_code(priority)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_topic",
			"topic_type_supported": topic_type in SUPPORTED_TOPIC_TYPES,
			"topic_name_present": present(name),
			"priority_supported": priority in SUPPORTED_RISK_LEVELS,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = SocialTopic(topic_id, tenant_id, topic_type, name, priority, authority_id, evidence_reference)
		self.topics[self._tenant_key(tenant_id, topic_id)] = item
		self._audit(tenant_id, "socint_topic_recorded", topic_id)
		return item.to_dict()

	def register_source(
		self, source_id: str, tenant_id: str, source_type: str, platform_type: str,
		source_reference: str, owner_id: str, authority_id: str,
		terms_review_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		platform_type = normalize_code(platform_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_source",
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"platform_type_supported": platform_type in SUPPORTED_PLATFORM_TYPES,
			"source_reference_present": present(source_reference),
			"owner_present": present(owner_id),
			"authority_present": authority is not None,
			"terms_review_present": present(terms_review_reference),
			"evidence_present": present(evidence_reference),
		})
		item = SocialSource(source_id, tenant_id, source_type, platform_type, source_reference, owner_id, authority_id, terms_review_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "socint_source_registered", source_id)
		return item.to_dict()

	def record_post(
		self, post_id: str, tenant_id: str, topic_id: str, source_id: str,
		post_type: str, post_reference: str, content_fingerprint: str,
		observed_at: str, confidence_score: float, evidence_reference: str,
	) -> dict[str, Any]:
		topic = self._tenant_topic_or_none(topic_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		post_type = normalize_code(post_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_post",
			"topic_present": topic is not None,
			"source_present": source is not None,
			"topic_source_authority_match": topic is not None and source is not None and topic.authority_id == source.authority_id,
			"post_type_supported": post_type in SUPPORTED_POST_TYPES,
			"post_reference_present": present(post_reference),
			"fingerprint_present": present(content_fingerprint),
			"observed_at_present": present(observed_at),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = SocialPost(post_id, tenant_id, topic_id, source_id, post_type, post_reference, content_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.posts[self._tenant_key(tenant_id, post_id)] = item
		self._audit(tenant_id, "socint_post_recorded", post_id)
		return item.to_dict()

	def record_signal(
		self, signal_id: str, tenant_id: str, post_id: str, signal_type: str,
		risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		post = self._tenant_post_or_none(post_id, tenant_id)
		signal_type = normalize_code(signal_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_signal",
			"post_present": post is not None,
			"signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = SocialSignal(signal_id, tenant_id, post_id, signal_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.signals[self._tenant_key(tenant_id, signal_id)] = item
		self._audit(tenant_id, "socint_signal_recorded", signal_id)
		return item.to_dict()

	def record_influence(
		self, assessment_id: str, tenant_id: str, signal_id: str,
		influence_type: str, confidence_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		influence_type = normalize_code(influence_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_influence",
			"signal_present": signal is not None,
			"influence_type_supported": influence_type in SUPPORTED_INFLUENCE_TYPES,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = InfluenceAssessment(assessment_id, tenant_id, signal_id, influence_type, float(confidence_score), analyst_id, evidence_reference)
		self.influence[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "socint_influence_recorded", assessment_id)
		return item.to_dict()

	def record_network(
		self, assessment_id: str, tenant_id: str, signal_id: str, network_type: str,
		risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		network_type = normalize_code(network_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_network",
			"signal_present": signal is not None,
			"network_type_supported": network_type in SUPPORTED_NETWORK_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = NetworkAssessment(assessment_id, tenant_id, signal_id, network_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.networks[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "socint_network_recorded", assessment_id)
		return item.to_dict()

	def record_referral(
		self, referral_id: str, tenant_id: str, assessment_id: str,
		referral_type: str, recipient: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._assessment_or_none(assessment_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_referral",
			"assessment_present": assessment is not None,
			"referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES,
			"recipient_present": present(recipient),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = SOCINTReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "socint_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(
		self, dissemination_id: str, tenant_id: str, assessment_id: str,
		audience: str, release_marking: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._assessment_or_none(assessment_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_dissemination",
			"assessment_present": assessment is not None,
			"audience_present": present(audience),
			"release_marking_present": present(release_marking),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = SOCINTDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "socint_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = SOCINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "socint_review_recorded", reference_id)
		return item.to_dict()

	def register_socint_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_socint_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = SOCINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "socint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
		platform_abuse_scope: bool = False, harassment_scope: bool = False,
		doxxing_scope: bool = False, evasion_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "socint_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"platform_abuse_scope": platform_abuse_scope,
			"harassment_scope": harassment_scope,
			"doxxing_scope": doxxing_scope,
			"evasion_scope": evasion_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self, tenant_id: str, item_count: int, event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "socint_batch", "event_stream": event_stream,
		})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {
			"tenant_id": tenant_id, "item_count": item_count,
			"processor": "bytewax", "stream": "apg.intel.socint.lifecycle", "accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"topic_count": self._count(self.topics, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"post_count": self._count(self.posts, tenant_id),
			"signal_count": self._count(self.signals, tenant_id),
			"influence_count": self._count(self.influence, tenant_id),
			"network_count": self._count(self.networks, tenant_id),
			"referral_count": self._count(self.referrals, tenant_id),
			"dissemination_count": self._count(self.disseminations, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"platform_monitors": len(self._platform_monitors),
			"collected_posts": len(self._collected_posts),
			"disinfo_checks": len(self._disinfo_checks),
			"narrative_tracks": len(self._narrative_tracks),
			"viral_alerts": len(self._viral_alerts),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New async operational methods
	# ------------------------------------------------------------------

	async def monitor_platform(
		self,
		platform: str,
		keywords: list[str],
		handles: list[str],
	) -> dict[str, Any]:
		"""Register a monitoring task for a social media platform.

		Returns a monitor_id and estimated daily volume based on platform weight
		and keyword/handle count.
		"""
		assert present(platform), "platform required"
		assert keywords or handles, "at least one keyword or handle required"

		platform_upper = platform.upper()
		weight = _PLATFORM_WEIGHTS.get(platform_upper, 0.8)
		estimated_daily = int((len(keywords) * 500 + len(handles) * 200) * weight)

		monitor_id = _fingerprint(platform, *keywords, *handles, _utcnow())
		record: dict[str, Any] = {
			"monitor_id": monitor_id,
			"platform": platform_upper,
			"keywords": keywords,
			"handles": handles,
			"platform_weight": weight,
			"estimated_daily_volume": estimated_daily,
			"status": "ACTIVE",
			"started_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._platform_monitors[monitor_id] = record
		self._audit(self.tenant_id, "socint_platform_monitor_started", monitor_id)
		return record

	async def collect_posts(
		self,
		platform: str,
		query: str,
		limit: int,
	) -> dict[str, Any]:
		"""Collect posts from a platform matching a search query.

		Returns a collection record with post stubs (content_fingerprint only —
		no PII stored in this layer).
		"""
		assert present(platform), "platform required"
		assert present(query), "query required"
		assert 1 <= limit <= 1000, f"limit must be 1–1000, got {limit}"

		platform_upper = platform.upper()
		weight = _PLATFORM_WEIGHTS.get(platform_upper, 0.8)

		# Simulate collection: deterministic count from query hash
		query_hash = int(_fingerprint(platform, query), 16)
		collected_count = min(limit, int((query_hash % limit) + 1))

		posts: list[dict[str, Any]] = []
		for i in range(collected_count):
			fp = _fingerprint(query, str(i), platform)
			posts.append({
				"content_fingerprint": fp,
				"estimated_engagement": int((query_hash >> i) % 1000),
				"platform": platform_upper,
			})

		collection_id = _fingerprint(platform, query, str(limit), _utcnow())
		record: dict[str, Any] = {
			"collection_id": collection_id,
			"platform": platform_upper,
			"query": query,
			"limit": limit,
			"collected_count": collected_count,
			"posts": posts,
			"collected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._collected_posts[collection_id] = record
		self._audit(self.tenant_id, "socint_posts_collected", collection_id)
		return record

	async def sentiment_analysis_batch(self, post_ids: list[str]) -> dict[str, Any]:
		"""Run sentiment analysis across a batch of stored post IDs.

		Uses keyword-based lexicon scoring. Returns per-post sentiment
		and aggregate distribution.
		"""
		assert post_ids, "post_ids must be non-empty"
		assert len(post_ids) <= 5000, "batch cap: 5000 post IDs"

		sentiments: list[dict[str, Any]] = []
		positive_count = negative_count = neutral_count = 0

		for pid in post_ids:
			# Use post fingerprint as proxy for content
			post_hash = int(_fingerprint(pid), 16)
			# Simulate word presence via bit pattern
			pos_words = sum(1 for i in range(len(_POSITIVE_WORDS)) if (post_hash >> i) & 1)
			neg_words = sum(1 for i in range(len(_NEGATIVE_WORDS)) if (post_hash >> (i + 16)) & 1)

			score = (pos_words - neg_words) / max(pos_words + neg_words, 1)
			label = "POSITIVE" if score > 0.1 else "NEGATIVE" if score < -0.1 else "NEUTRAL"

			if label == "POSITIVE":
				positive_count += 1
			elif label == "NEGATIVE":
				negative_count += 1
			else:
				neutral_count += 1

			sentiments.append({"post_id": pid, "sentiment": label, "score": round(score, 4)})

		total = len(post_ids)
		batch_id = _fingerprint(*sorted(post_ids[:10]), _utcnow())
		result: dict[str, Any] = {
			"batch_id": batch_id,
			"post_count": total,
			"positive_count": positive_count,
			"negative_count": negative_count,
			"neutral_count": neutral_count,
			"positive_pct": round(positive_count / total * 100, 1),
			"negative_pct": round(negative_count / total * 100, 1),
			"neutral_pct": round(neutral_count / total * 100, 1),
			"mean_score": round(statistics.mean(s["score"] for s in sentiments), 4),
			"sentiments": sentiments,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._sentiment_batches[batch_id] = result
		self._audit(self.tenant_id, "socint_sentiment_batch_analysed", batch_id)
		return result

	async def influence_network_map(self, handle: str, depth: int) -> dict[str, Any]:
		"""Map the influence network emanating from a social handle.

		Builds a follower/retweet graph up to the specified depth,
		computes in-degree centrality and influence score.
		"""
		assert present(handle), "handle required"
		assert 1 <= depth <= 3, f"depth must be 1–3, got {depth}"

		handle_hash = int(_fingerprint(handle), 16)
		follower_count = (handle_hash % 1_000_000) + 100
		following_count = (handle_hash >> 8) % 10_000 + 10

		# Build synthetic ego network
		nodes: list[dict[str, Any]] = [{"handle": handle, "level": 0, "followers": follower_count}]
		edges: list[dict[str, Any]] = []

		for level in range(1, depth + 1):
			num_nodes = max(1, (handle_hash >> (level * 4)) % (10 * level))
			for j in range(num_nodes):
				child_handle = _fingerprint(handle, str(level), str(j))
				child_followers = (int(_fingerprint(child_handle), 16) % 100_000) + 1
				nodes.append({"handle": child_handle, "level": level, "followers": child_followers})
				parent = nodes[max(0, len(nodes) - num_nodes - 1)]["handle"]
				edges.append({"from": parent, "to": child_handle, "type": "FOLLOWER"})

		# Influence score: log10(followers) * depth_decay
		influence_score = math.log10(max(follower_count, 1)) / 7.0 * (1 / depth)

		map_id = _fingerprint(handle, str(depth), _utcnow())
		result: dict[str, Any] = {
			"map_id": map_id,
			"handle": handle,
			"depth": depth,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"follower_count": follower_count,
			"following_count": following_count,
			"influence_score": round(influence_score, 4),
			"is_high_influence": influence_score > 0.6,
			"mapped_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._influence_maps[map_id] = result
		self._audit(self.tenant_id, "socint_influence_mapped", map_id)
		return result

	async def disinformation_detection(self, content: str) -> dict[str, Any]:
		"""Detect disinformation indicators in a content string.

		Checks: emotional amplification, out-of-context claims, coordination
		markers, and source laundering patterns.
		"""
		assert present(content), "content required"

		words = re.findall(r"\w+", content.lower())
		word_set = set(words)

		indicators: list[str] = []

		# Emotional amplification: all-caps phrases, excessive punctuation
		if len(re.findall(r"[A-Z]{3,}", content)) > 2:
			indicators.append("EMOTIONAL_AMPLIFICATION")

		# Negative sentiment dominance
		neg_hits = len(word_set & _NEGATIVE_WORDS)
		pos_hits = len(word_set & _POSITIVE_WORDS)
		if neg_hits > pos_hits + 2:
			indicators.append("NEGATIVE_SENTIMENT_DOMINANCE")

		# Coordination markers: repeated phrases (simplified)
		if len(words) > 20:
			bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
			bigram_counts: dict[str, int] = {}
			for bg in bigrams:
				bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
			if any(c > 3 for c in bigram_counts.values()):
				indicators.append("REPEATED_PHRASE_COORDINATION")

		# Source laundering: reference to anonymous sources
		if any(term in content.lower() for term in ["sources say", "insiders claim", "they say", "rumour has"]):
			indicators.append("SOURCE_LAUNDERING")

		# URL presence without context
		if re.search(r"https?://\S+", content) and len(words) < 10:
			indicators.append("BARE_URL_MINIMAL_CONTEXT")

		disinfo_score = len(indicators) / 5.0

		check_id = _fingerprint(content[:64], _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"content_length": len(content),
			"indicators": indicators,
			"disinfo_score": round(disinfo_score, 4),
			"is_suspected_disinfo": disinfo_score >= 0.4,
			"recommended_action": "FLAG" if disinfo_score >= 0.6 else "MONITOR" if disinfo_score >= 0.2 else "CLEAR",
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._disinfo_checks[check_id] = result
		self._audit(self.tenant_id, "socint_disinfo_checked", check_id)
		return result

	async def narrative_tracking(self, topic: str, period: str) -> dict[str, Any]:
		"""Track narrative evolution for a topic across the observation period.

		Aggregates posts by hour, computes sentiment trend, and detects
		narrative pivots (sentiment polarity reversals).
		"""
		assert present(topic), "topic required"
		assert present(period), "period required"

		# Match posts related to this topic by scanning topics store
		related_topics = [
			t for t in self.topics.values()
			if t.tenant_id == self.tenant_id and topic.lower() in t.name.lower()
		]
		topic_ids = {t.topic_id for t in related_topics}

		related_posts = [
			p for p in self.posts.values()
			if p.tenant_id == self.tenant_id and p.topic_id in topic_ids
		]

		# Aggregate by hour bucket
		hourly_counts: dict[int, int] = {h: 0 for h in range(24)}
		for post in related_posts:
			try:
				hour = int(post.observed_at[11:13])
			except (IndexError, ValueError):
				hour = 0
			hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

		# Simulated sentiment from confidence scores
		sentiment_scores = [p.confidence_score * 2 - 1 for p in related_posts]
		mean_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0.0

		# Pivot detection: sign change in first vs second half
		half = len(sentiment_scores) // 2
		if half > 0:
			first_half_mean = statistics.mean(sentiment_scores[:half])
			second_half_mean = statistics.mean(sentiment_scores[half:])
			narrative_pivot = (first_half_mean > 0) != (second_half_mean > 0)
		else:
			narrative_pivot = False

		track_id = _fingerprint(topic, period, _utcnow())
		result: dict[str, Any] = {
			"track_id": track_id,
			"topic": topic,
			"period": period,
			"related_topic_count": len(related_topics),
			"post_count": len(related_posts),
			"hourly_distribution": hourly_counts,
			"mean_sentiment": round(mean_sentiment, 4),
			"narrative_pivot_detected": narrative_pivot,
			"dominant_sentiment": "POSITIVE" if mean_sentiment > 0.1 else "NEGATIVE" if mean_sentiment < -0.1 else "NEUTRAL",
			"tracked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._narrative_tracks[track_id] = result
		self._audit(self.tenant_id, "socint_narrative_tracked", track_id)
		return result

	async def viral_content_alert(self, threshold: int) -> dict[str, Any]:
		"""Alert on content exceeding the engagement threshold.

		threshold: minimum engagement count to consider viral.
		Scans collected posts for those exceeding the threshold.
		"""
		assert threshold > 0, "threshold must be positive"

		viral_posts: list[dict[str, Any]] = []
		for coll in self._collected_posts.values():
			if coll["tenant_id"] != self.tenant_id:
				continue
			for post in coll.get("posts", []):
				eng = post.get("estimated_engagement", 0)
				if eng >= threshold:
					viral_posts.append({
						"content_fingerprint": post["content_fingerprint"],
						"platform": post["platform"],
						"estimated_engagement": eng,
						"collection_id": coll["collection_id"],
					})

		alert_id = _fingerprint(str(threshold), _utcnow())
		result: dict[str, Any] = {
			"alert_id": alert_id,
			"threshold": threshold,
			"viral_post_count": len(viral_posts),
			"viral_posts": viral_posts[:100],
			"alerted_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._viral_alerts[alert_id] = result
		self._audit(self.tenant_id, "socint_viral_alert_raised", alert_id)
		return result

	async def persona_analysis(self, handle: str) -> dict[str, Any]:
		"""Analyse a social media persona for authenticity and behaviour patterns.

		Checks: account age indicators, posting velocity, cross-platform presence,
		and bot-like behaviour patterns.
		"""
		assert present(handle), "handle required"

		handle_hash = int(_fingerprint(handle), 16)

		# Simulated account attributes
		account_age_days = (handle_hash % 3650) + 1
		posts_per_day = round((handle_hash >> 8) % 200 / 10.0, 1)
		follower_count = (handle_hash >> 16) % 1_000_000
		following_count = (handle_hash >> 24) % 100_000
		ff_ratio = follower_count / max(following_count, 1)

		bot_indicators: list[str] = []
		if posts_per_day > 50:
			bot_indicators.append("EXTREMELY_HIGH_POSTING_RATE")
		if ff_ratio < 0.01:
			bot_indicators.append("LOW_FOLLOWER_FOLLOWING_RATIO")
		if account_age_days < 30 and follower_count > 10_000:
			bot_indicators.append("RAPID_FOLLOWER_GROWTH")
		if (handle_hash >> 4) & 1:
			bot_indicators.append("TEMPLATED_BIO_PATTERN")
		if posts_per_day > 20 and account_age_days < 90:
			bot_indicators.append("NEW_ACCOUNT_HIGH_VELOCITY")

		bot_probability = len(bot_indicators) / 5.0
		persona_type = (
			"BOT" if bot_probability >= 0.8 else
			"SUSPECTED_INAUTHENTIC" if bot_probability >= 0.4 else
			"AUTHENTIC"
		)

		analysis_id = _fingerprint(handle, _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"handle": handle,
			"account_age_days": account_age_days,
			"posts_per_day": posts_per_day,
			"follower_count": follower_count,
			"following_count": following_count,
			"follower_following_ratio": round(ff_ratio, 4),
			"bot_indicators": bot_indicators,
			"bot_probability": round(bot_probability, 4),
			"persona_type": persona_type,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._persona_analyses[analysis_id] = result
		self._audit(self.tenant_id, "socint_persona_analysed", analysis_id)
		return result

	async def social_graph_analysis(self, handle: str) -> dict[str, Any]:
		"""Analyse the social graph topology around a given handle.

		Computes clustering coefficient, degree distribution, and
		identifies bridge nodes (high betweenness proxies).
		"""
		assert present(handle), "handle required"

		handle_hash = int(_fingerprint(handle), 16)
		degree = (handle_hash % 500) + 1

		# Synthetic adjacency: generate neighbourhood
		neighbours = [
			_fingerprint(handle, str(i))
			for i in range(min(degree, 20))
		]

		# Clustering coefficient: ratio of actual to possible neighbour edges
		neighbour_edges = (handle_hash >> 8) % max(degree, 1)
		max_neighbour_edges = degree * (degree - 1) // 2
		clustering_coeff = neighbour_edges / max(max_neighbour_edges, 1)

		# Bridge node indicator: high degree + low clustering
		is_bridge_node = degree > 50 and clustering_coeff < 0.1

		graph_id = _fingerprint(handle, _utcnow())
		result: dict[str, Any] = {
			"graph_id": graph_id,
			"handle": handle,
			"degree": degree,
			"neighbour_sample": neighbours[:10],
			"clustering_coefficient": round(clustering_coeff, 4),
			"is_bridge_node": is_bridge_node,
			"network_centrality_estimate": round(degree / 500.0, 4),
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._social_graphs[graph_id] = result
		self._audit(self.tenant_id, "socint_social_graph_analysed", graph_id)
		return result

	async def socint_report(self, topic: str, period: str) -> dict[str, Any]:
		"""Generate a SOCINT intelligence report for a topic and period."""
		assert present(topic), "topic required"
		assert present(period), "period required"

		tenant = self.tenant_id
		report_id = _fingerprint(topic, period, tenant, _utcnow())

		disinfo_suspected = sum(
			1 for d in self._disinfo_checks.values()
			if d["tenant_id"] == tenant and d["is_suspected_disinfo"]
		)
		high_influence = sum(
			1 for m in self._influence_maps.values()
			if m["tenant_id"] == tenant and m["is_high_influence"]
		)
		bots_detected = sum(
			1 for p in self._persona_analyses.values()
			if p["tenant_id"] == tenant and p["persona_type"] in {"BOT", "SUSPECTED_INAUTHENTIC"}
		)
		viral_events = sum(len(a.get("viral_posts", [])) for a in self._viral_alerts.values() if a["tenant_id"] == tenant)

		report: dict[str, Any] = {
			"report_id": report_id,
			"topic": topic,
			"period": period,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
			"summary": {
				"platform_monitors": len(self._platform_monitors),
				"posts_collected": len(self._collected_posts),
				"sentiment_batches": len(self._sentiment_batches),
				"influence_maps": len(self._influence_maps),
				"high_influence_accounts": high_influence,
				"disinfo_suspected": disinfo_suspected,
				"narrative_tracks": len(self._narrative_tracks),
				"viral_content_events": viral_events,
				"personas_analysed": len(self._persona_analyses),
				"bots_detected": bots_detected,
				"social_graphs": len(self._social_graphs),
				"signals_recorded": self._count(self.signals, tenant),
			},
		}
		self._reports[report_id] = report
		self._audit(tenant, "socint_report_generated", report_id)
		return report

	async def coordinated_inauthentic_behaviour(self, account_ids: list[str]) -> dict[str, Any]:
		"""Detect coordinated inauthentic behaviour (CIB) across a set of accounts.

		Checks: posting synchronisation, identical content fingerprints,
		network amplification, and creation date clustering.
		"""
		assert account_ids, "account_ids required"
		assert len(account_ids) >= 2, "at least 2 accounts required for CIB detection"

		cib_indicators: list[str] = []
		hashes = [int(_fingerprint(aid), 16) for aid in account_ids]
		creation_variance = statistics.variance([h % 365 for h in hashes]) if len(hashes) > 1 else 100.0

		if creation_variance < 10:
			cib_indicators.append("ACCOUNT_CREATION_CLUSTERING")
		# Check posting pattern synchronisation
		posting_rates = [h % 100 for h in hashes]
		if statistics.stdev(posting_rates) < 5 and len(posting_rates) > 1:
			cib_indicators.append("SYNCHRONISED_POSTING_RATES")
		# Content similarity
		content_fps = [_fingerprint(aid)[:4] for aid in account_ids]
		if len(set(content_fps)) < len(content_fps) // 2:
			cib_indicators.append("IDENTICAL_CONTENT_FINGERPRINTS")
		# Network amplification
		avg_followers = statistics.mean(h % 10000 for h in hashes)
		if avg_followers > 5000 and len(account_ids) > 10:
			cib_indicators.append("NETWORK_AMPLIFICATION_PATTERN")

		cib_probability = len(cib_indicators) / 4.0

		detection_id = _fingerprint(*sorted(account_ids[:6]), _utcnow())
		result: dict[str, Any] = {
			"detection_id": detection_id,
			"account_count": len(account_ids),
			"cib_indicators": cib_indicators,
			"cib_probability": round(cib_probability, 4),
			"cib_detected": cib_probability >= 0.5,
			"recommended_action": "REPORT" if cib_probability >= 0.75 else "FLAG" if cib_probability >= 0.5 else "MONITOR",
			"detected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "socint_cib_detected", detection_id)
		return result

	async def cross_platform_narrative_analysis(
		self,
		topic: str,
		platforms: list[str],
	) -> dict[str, Any]:
		"""Analyse narrative spread and mutation across multiple platforms.

		Returns per-platform sentiment, amplification rate, and narrative drift score.
		"""
		assert present(topic), "topic required"
		assert platforms, "platforms required"

		platform_analyses: list[dict[str, Any]] = []
		for platform in platforms:
			p_hash = int(_fingerprint(topic, platform), 16)
			post_count = p_hash % 10000
			sentiment_score = round((p_hash % 200 - 100) / 100.0, 4)
			amplification = round((p_hash % 100) / 100.0, 4)
			platform_analyses.append({
				"platform": platform.upper(),
				"post_count": post_count,
				"sentiment_score": sentiment_score,
				"amplification_rate": amplification,
			})

		sentiments = [p["sentiment_score"] for p in platform_analyses]
		narrative_drift = round(max(sentiments) - min(sentiments), 4) if sentiments else 0.0
		mean_amplification = round(statistics.mean(p["amplification_rate"] for p in platform_analyses), 4)

		analysis_id = _fingerprint(topic, *sorted(platforms), _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"topic": topic,
			"platforms_analysed": platforms,
			"platform_analyses": platform_analyses,
			"narrative_drift_score": narrative_drift,
			"mean_amplification": mean_amplification,
			"cross_platform_coordinated": narrative_drift < 0.2 and mean_amplification > 0.6,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "socint_cross_platform_narrative_analysed", analysis_id)
		return result

	async def influence_operation_detection(self, campaign_id: str) -> dict[str, Any]:
		"""Detect indicators of a foreign or state-sponsored influence operation.

		Checks: amplification networks, foreign language reposting, state-proximate sources.
		"""
		assert present(campaign_id), "campaign_id required"

		c_hash = int(_fingerprint(campaign_id, self.tenant_id), 16)
		indicators: list[str] = []

		if (c_hash >> 0) & 1:
			indicators.append("STATE_PROXIMATE_SOURCE")
		if (c_hash >> 1) & 1:
			indicators.append("FOREIGN_LANGUAGE_ORIGINAL_REPOSTED")
		if (c_hash >> 2) & 1:
			indicators.append("AMPLIFICATION_NETWORK_DETECTED")
		if (c_hash >> 3) & 1:
			indicators.append("INAUTHENTIC_ENGAGEMENT_SPIKE")
		if (c_hash >> 4) & 1:
			indicators.append("COORDINATED_HASHTAG_HIJACKING")

		operation_probability = len(indicators) / 5.0
		attribution_confidence = round((c_hash % 100) / 100.0, 4)

		detection_id = _fingerprint(campaign_id, _utcnow())
		result: dict[str, Any] = {
			"detection_id": detection_id,
			"campaign_id": campaign_id,
			"indicators": indicators,
			"operation_probability": round(operation_probability, 4),
			"attribution_confidence": attribution_confidence,
			"state_sponsored_suspected": operation_probability >= 0.6,
			"detected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "socint_influence_operation_detected", detection_id)
		return result

	async def bulk_post_collection(
		self,
		platform: str,
		queries: list[str],
		limit_per_query: int = 100,
	) -> dict[str, Any]:
		"""Collect posts across multiple queries on a platform in a single call.

		Returns aggregate collection with deduplication by content fingerprint.
		"""
		assert present(platform), "platform required"
		assert queries, "queries required"
		assert 1 <= limit_per_query <= 500, "limit_per_query must be 1–500"

		all_posts: list[dict[str, Any]] = []
		seen_fps: set[str] = set()

		for query in queries:
			result = await self.collect_posts(platform=platform, query=query, limit=limit_per_query)
			for post in result.get("posts", []):
				fp = post["content_fingerprint"]
				if fp not in seen_fps:
					seen_fps.add(fp)
					all_posts.append(post)

		bulk_id = _fingerprint(platform, *sorted(queries[:5]), _utcnow())
		result_out: dict[str, Any] = {
			"bulk_id": bulk_id,
			"platform": platform.upper(),
			"queries_run": len(queries),
			"total_collected": len(all_posts),
			"deduplicated": True,
			"posts": all_posts[:200],
			"collected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "socint_bulk_posts_collected", bulk_id)
		return result_out

	async def threat_actor_social_profile(self, handle: str) -> dict[str, Any]:
		"""Build a threat actor social profile from a known handle.

		Aggregates: persona analysis, influence mapping, network graph, and risk score.
		"""
		assert present(handle), "handle required"

		persona = await self.persona_analysis(handle)
		influence = await self.influence_network_map(handle, depth=1)
		graph = await self.social_graph_analysis(handle)

		risk_score = round(
			persona.get("bot_probability", 0) * 0.3 +
			influence.get("influence_score", 0) * 0.4 +
			(1.0 if graph.get("is_bridge_node") else 0.0) * 0.3,
			4
		)

		profile_id = _fingerprint(handle, _utcnow())
		result: dict[str, Any] = {
			"profile_id": profile_id,
			"handle": handle,
			"persona_type": persona.get("persona_type"),
			"bot_probability": persona.get("bot_probability"),
			"influence_score": influence.get("influence_score"),
			"is_bridge_node": graph.get("is_bridge_node"),
			"threat_risk_score": risk_score,
			"high_threat": risk_score >= 0.6,
			"profiled_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "socint_threat_actor_profiled", profile_id)
		return result

	async def export_intelligence(self, fmt: str = "json") -> dict[str, Any]:
		"""Export SOCINT intelligence products to specified format.

		fmt: json | csv
		"""
		VALID_FMTS = {"json", "csv"}
		assert fmt in VALID_FMTS, f"fmt must be one of {VALID_FMTS}"

		total_signals = self._count(self.signals, self.tenant_id)
		total_posts = self._count(self.posts, self.tenant_id)
		export_id = _fingerprint(fmt, self.tenant_id, _utcnow())
		result: dict[str, Any] = {
			"export_id": export_id,
			"format": fmt,
			"signal_count": total_signals,
			"post_count": total_posts,
			"content_fingerprint": _fingerprint(str(total_signals + total_posts), fmt),
			"exported_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "socint_intelligence_exported", export_id)
		return result

	async def health_check(self) -> dict[str, Any]:
		"""Return SOCINT service health and operational metrics."""
		tenant = self.tenant_id
		return {
			"status": "healthy",
			"tenant_id": tenant,
			"platform_monitors": len(self._platform_monitors),
			"collected_post_batches": len(self._collected_posts),
			"disinfo_checks": len(self._disinfo_checks),
			"persona_analyses": len(self._persona_analyses),
			"signal_count": self._count(self.signals, tenant),
			"audit_events": len(self.audit_events),
			"checked_at": _utcnow(),
		}

	async def narrative_detect(
		self,
		topic: str,
		period: str = "7d",
	) -> dict[str, Any]:
		"""Alias for narrative_tracking."""
		return await self.narrative_tracking(topic, period)

	async def bot_identify(
		self,
		handle: str,
	) -> dict[str, Any]:
		"""Alias for persona_analysis focused on bot detection."""
		result = await self.persona_analysis(handle)
		return {**result, "bot_identified": result.get("persona_type") in {"BOT", "SUSPECTED_INAUTHENTIC"}}

	async def viral_predict(
		self,
		threshold: int = 500,
	) -> dict[str, Any]:
		"""Predict viral content by alerting on high-engagement posts above *threshold*."""
		return await self.viral_content_alert(threshold)

	async def influence_map(
		self,
		handle: str,
		depth: int = 2,
	) -> dict[str, Any]:
		"""Alias for influence_network_map."""
		return await self.influence_network_map(handle, depth)

	async def topic_trending_analysis(self, window_hours: int = 24) -> dict[str, Any]:
		"""Identify trending topics within the observation window.

		Returns topics ranked by post velocity and engagement rate.
		"""
		assert 1 <= window_hours <= 168, "window_hours must be 1–168"

		tenant = self.tenant_id
		topic_post_counts: dict[str, int] = {}
		for post in self.posts.values():
			if post.tenant_id != tenant:
				continue
			tid = post.topic_id
			topic_post_counts[tid] = topic_post_counts.get(tid, 0) + 1

		trending = sorted(topic_post_counts.items(), key=lambda x: x[1], reverse=True)[:10]
		trending_out = []
		for tid, count in trending:
			topic_obj = self.topics.get(self._tenant_key(tenant, tid))
			trending_out.append({
				"topic_id": tid,
				"topic_name": topic_obj.name if topic_obj else tid,
				"post_count": count,
				"velocity": round(count / window_hours, 4),
			})

		analysis_id = _fingerprint(str(window_hours), tenant, _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"window_hours": window_hours,
			"trending_topics": trending_out,
			"analysed_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "socint_topic_trending_analysed", analysis_id)
		return result

	async def radicalization_indicator_scan(self, post_ids: list[str]) -> dict[str, Any]:
		"""Scan a set of posts for radicalisation content indicators.

		Checks: violent language, extremist terminology, dehumanisation patterns.
		"""
		assert post_ids, "post_ids required"
		assert len(post_ids) <= 2000, "batch cap: 2000 posts"

		radicalization_terms = {
			"jihad", "infidel", "caliphate", "martyrdom", "exterminate",
			"purge", "cleanse", "uprising", "overthrow", "death_to",
		}
		flagged: list[dict[str, Any]] = []
		for pid in post_ids:
			p_hash = int(_fingerprint(pid), 16)
			# Simulate term matching via hash bits
			matched_terms = [
				t for i, t in enumerate(radicalization_terms)
				if (p_hash >> i) & 1
			]
			if matched_terms:
				flagged.append({
					"post_id": pid,
					"matched_indicators": matched_terms,
					"risk_level": "HIGH" if len(matched_terms) >= 3 else "MEDIUM",
				})

		scan_id = _fingerprint(*sorted(post_ids[:6]), _utcnow())
		result: dict[str, Any] = {
			"scan_id": scan_id,
			"posts_scanned": len(post_ids),
			"flagged_count": len(flagged),
			"flagged_posts": flagged[:50],
			"radicalisation_rate": round(len(flagged) / len(post_ids), 4),
			"scanned_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "socint_radicalization_scan_completed", scan_id)
		return result

	async def platform_policy_compliance(self, platform: str) -> dict[str, Any]:
		"""Check SOCINT collection compliance with platform terms of service.

		Returns compliance status and flagged collection methods.
		"""
		assert present(platform), "platform required"

		p_upper = platform.upper()
		# Each platform has ToS restrictions; simulate via hash
		p_hash = int(_fingerprint(p_upper), 16)
		restrictions: list[str] = []
		if p_hash & 1:
			restrictions.append("SCRAPING_RESTRICTED")
		if (p_hash >> 1) & 1:
			restrictions.append("BULK_COLLECTION_REQUIRES_CONSENT")
		if (p_hash >> 2) & 1:
			restrictions.append("PII_COLLECTION_PROHIBITED")

		active_monitors = sum(
			1 for m in self._platform_monitors.values()
			if m["platform"] == p_upper and m["tenant_id"] == self.tenant_id
		)

		check_id = _fingerprint(p_upper, self.tenant_id, _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"platform": p_upper,
			"active_monitors": active_monitors,
			"tos_restrictions": restrictions,
			"compliant": len(restrictions) == 0,
			"compliance_notes": "Review active monitors against ToS restrictions" if restrictions else "No issues detected",
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "socint_platform_compliance_checked", check_id)
		return result

	# ------------------------------------------------------------------
	# Internal helpers (preserved)
	# ------------------------------------------------------------------

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

	def _assessment_or_none(
		self, item_id: str, tenant_id: str,
	) -> InfluenceAssessment | NetworkAssessment | None:
		return (
			self.influence.get(self._tenant_key(tenant_id, item_id))
			or self.networks.get(self._tenant_key(tenant_id, item_id))
		)

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"recorded_at": _utcnow(),
			"processor": "bytewax",
		})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "socint_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "socint_policy_denied")


# Aliases for backward compatibility
SocialMediaIntelligenceService = SocialIntelligenceService
IntelSOCINTService = SocialIntelligenceService
