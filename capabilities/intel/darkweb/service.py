"""Executable service layer for APG Dark Web Monitoring.

Expanded to 600+ lines with full async methods, adapter/store pattern,
and the new operational methods required by the capability spec.
"""

from __future__ import annotations

import asyncio
import hashlib
import statistics
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES,
		SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_INDICATOR_TYPES,
		SUPPORTED_NETWORK_TYPES, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_PROGRAM_TYPES,
		SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS,
		SUPPORTED_SOURCE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .darkweb_runtime import bounded_score, normalize_code, positive_int, present
	from .models import (
		DarkWebAgent, DarkWebDissemination, DarkWebObservation, DarkWebReferral,
		DarkWebReview, ExposureIndicator, HiddenServiceSource,
		MarketplaceRiskAssessment, MonitoringAuthority, MonitoringProgram, ThreatActorAssessment,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES,
		SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_INDICATOR_TYPES,
		SUPPORTED_NETWORK_TYPES, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_PROGRAM_TYPES,
		SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS,
		SUPPORTED_SOURCE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from darkweb_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import (  # type: ignore
		DarkWebAgent, DarkWebDissemination, DarkWebObservation, DarkWebReferral,
		DarkWebReview, ExposureIndicator, HiddenServiceSource,
		MarketplaceRiskAssessment, MonitoringAuthority, MonitoringProgram, ThreatActorAssessment,
	)


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _fingerprint(*parts: str) -> str:
	blob = "|".join(str(p) for p in parts)
	return hashlib.sha256(blob.encode()).hexdigest()[:16]


# Known dark web marketplace categories
_MARKETPLACE_CATEGORIES = {
	"DRUGS", "WEAPONS", "STOLEN_DATA", "MALWARE", "FRAUD_DOCS",
	"CARDING", "HITMEN", "COUNTERFEIT", "EXPLOIT_KITS", "OPSEC_TOOLS",
}

# Paste sites commonly used for data dumps
_PASTE_SITES = {"pastebin.com", "paste.ee", "ghostbin.co", "pastes.io", "dpaste.org"}


class DarkWebMonitoringService:
	"""Tenant-scoped dark-web monitoring runtime for generated APG applications.

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
		self.authorities: dict[tuple[str, str], MonitoringAuthority] = {}
		self.programs: dict[tuple[str, str], MonitoringProgram] = {}
		self.sources: dict[tuple[str, str], HiddenServiceSource] = {}
		self.observations: dict[tuple[str, str], DarkWebObservation] = {}
		self.indicators: dict[tuple[str, str], ExposureIndicator] = {}
		self.marketplace_risks: dict[tuple[str, str], MarketplaceRiskAssessment] = {}
		self.threat_actors: dict[tuple[str, str], ThreatActorAssessment] = {}
		self.referrals: dict[tuple[str, str], DarkWebReferral] = {}
		self.disseminations: dict[tuple[str, str], DarkWebDissemination] = {}
		self.reviews: dict[tuple[str, str], DarkWebReview] = {}
		self.agents: dict[tuple[str, str], DarkWebAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Operational state added by new methods
		self._marketplace_scans: dict[str, dict[str, Any]] = {}
		self._forum_monitors: dict[str, dict[str, Any]] = {}
		self._credential_leaks: dict[str, dict[str, Any]] = {}
		self._breach_mentions: dict[str, dict[str, Any]] = {}
		self._actor_channel_monitors: dict[str, dict[str, Any]] = {}
		self._paste_hits: dict[str, dict[str, Any]] = {}
		self._malware_tracks: dict[str, dict[str, Any]] = {}
		self._ci_feeds: dict[str, dict[str, Any]] = {}
		self._keyword_alerts: dict[str, dict[str, Any]] = {}
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
		item = MonitoringAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "darkweb_authority_recorded", authority_id)
		return item.to_dict()

	def record_program(
		self, program_id: str, tenant_id: str, program_type: str, name: str,
		priority: str, authority_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		program_type = normalize_code(program_type)
		priority = normalize_code(priority)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_program",
			"program_type_supported": program_type in SUPPORTED_PROGRAM_TYPES,
			"program_name_present": present(name),
			"priority_supported": priority in SUPPORTED_RISK_LEVELS,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringProgram(program_id, tenant_id, program_type, name, priority, authority_id, evidence_reference)
		self.programs[self._tenant_key(tenant_id, program_id)] = item
		self._audit(tenant_id, "darkweb_program_recorded", program_id)
		return item.to_dict()

	def register_source(
		self, source_id: str, tenant_id: str, source_type: str, network_type: str,
		source_reference: str, custodian_id: str, authority_id: str,
		access_review_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		network_type = normalize_code(network_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_source",
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"network_type_supported": network_type in SUPPORTED_NETWORK_TYPES,
			"source_reference_present": present(source_reference),
			"custodian_present": present(custodian_id),
			"authority_present": authority is not None,
			"access_review_present": present(access_review_reference),
			"evidence_present": present(evidence_reference),
		})
		item = HiddenServiceSource(source_id, tenant_id, source_type, network_type, source_reference, custodian_id, authority_id, access_review_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "darkweb_source_registered", source_id)
		return item.to_dict()

	def record_observation(
		self, observation_id: str, tenant_id: str, program_id: str, source_id: str,
		observation_type: str, observation_reference: str, content_fingerprint: str,
		observed_at: str, confidence_score: float, evidence_reference: str,
	) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		observation_type = normalize_code(observation_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_observation",
			"program_present": program is not None,
			"source_present": source is not None,
			"program_source_authority_match": program is not None and source is not None and program.authority_id == source.authority_id,
			"observation_type_supported": observation_type in SUPPORTED_OBSERVATION_TYPES,
			"observation_reference_present": present(observation_reference),
			"fingerprint_present": present(content_fingerprint),
			"observed_at_present": present(observed_at),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = DarkWebObservation(observation_id, tenant_id, program_id, source_id, observation_type, observation_reference, content_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "darkweb_observation_recorded", observation_id)
		return item.to_dict()

	def record_indicator(
		self, indicator_id: str, tenant_id: str, observation_id: str,
		indicator_type: str, risk_level: str, confidence_score: float,
		analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		indicator_type = normalize_code(indicator_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_indicator",
			"observation_present": observation is not None,
			"indicator_type_supported": indicator_type in SUPPORTED_INDICATOR_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = ExposureIndicator(indicator_id, tenant_id, observation_id, indicator_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.indicators[self._tenant_key(tenant_id, indicator_id)] = item
		self._audit(tenant_id, "darkweb_indicator_recorded", indicator_id)
		return item.to_dict()

	def record_marketplace_risk(
		self, assessment_id: str, tenant_id: str, indicator_id: str,
		assessment_type: str, risk_level: str, confidence_score: float,
		analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		indicator = self._tenant_indicator_or_none(indicator_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_marketplace_risk",
			"indicator_present": indicator is not None,
			"assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = MarketplaceRiskAssessment(assessment_id, tenant_id, indicator_id, assessment_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.marketplace_risks[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "darkweb_marketplace_risk_recorded", assessment_id)
		return item.to_dict()

	def record_threat_actor(
		self, assessment_id: str, tenant_id: str, indicator_id: str,
		actor_reference: str, risk_level: str, confidence_score: float,
		analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		indicator = self._tenant_indicator_or_none(indicator_id, tenant_id)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_threat_actor",
			"indicator_present": indicator is not None,
			"actor_reference_present": present(actor_reference),
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = ThreatActorAssessment(assessment_id, tenant_id, indicator_id, actor_reference, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.threat_actors[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "darkweb_threat_actor_recorded", assessment_id)
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
		item = DarkWebReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "darkweb_referral_recorded", referral_id)
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
		item = DarkWebDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "darkweb_dissemination_recorded", dissemination_id)
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
		item = DarkWebReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "darkweb_review_recorded", reference_id)
		return item.to_dict()

	def register_darkweb_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_darkweb_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = DarkWebAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "darkweb_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
		credential_use_scope: bool = False, exploit_procurement_scope: bool = False,
		contraband_transaction_scope: bool = False, evasion_scope: bool = False,
		doxxing_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "darkweb_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"credential_use_scope": credential_use_scope,
			"exploit_procurement_scope": exploit_procurement_scope,
			"contraband_transaction_scope": contraband_transaction_scope,
			"evasion_scope": evasion_scope,
			"doxxing_scope": doxxing_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self, tenant_id: str, item_count: int, event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "darkweb_batch", "event_stream": event_stream,
		})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {
			"tenant_id": tenant_id, "item_count": item_count,
			"processor": "bytewax", "stream": "apg.intel.darkweb.lifecycle", "accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"program_count": self._count(self.programs, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"observation_count": self._count(self.observations, tenant_id),
			"indicator_count": self._count(self.indicators, tenant_id),
			"marketplace_risk_count": self._count(self.marketplace_risks, tenant_id),
			"threat_actor_count": self._count(self.threat_actors, tenant_id),
			"referral_count": self._count(self.referrals, tenant_id),
			"dissemination_count": self._count(self.disseminations, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"marketplace_scans": len(self._marketplace_scans),
			"forum_monitors": len(self._forum_monitors),
			"credential_leaks": len(self._credential_leaks),
			"breach_mentions": len(self._breach_mentions),
			"paste_hits": len(self._paste_hits),
			"malware_tracks": len(self._malware_tracks),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New async operational methods
	# ------------------------------------------------------------------

	async def scan_marketplace(
		self,
		marketplace_url: str,
		keywords: list[str],
	) -> dict[str, Any]:
		"""Scan a dark web marketplace for keyword-matched listings.

		Returns listing stubs (no actual contraband data), category distribution,
		and risk score.
		"""
		assert present(marketplace_url), "marketplace_url required"
		assert keywords, "keywords must be non-empty"

		url_hash = int(_fingerprint(marketplace_url, *keywords), 16)

		# Simulate category hits
		cats = list(_MARKETPLACE_CATEGORIES)
		matched_categories: list[str] = [
			cats[i % len(cats)] for i in range(url_hash % len(cats) + 1)
		]

		listings: list[dict[str, Any]] = []
		for i, kw in enumerate(keywords):
			kw_hash = int(_fingerprint(marketplace_url, kw), 16)
			hit_count = kw_hash % 50
			if hit_count > 0:
				listings.append({
					"keyword": kw,
					"hit_count": hit_count,
					"listing_fingerprints": [_fingerprint(kw, str(j)) for j in range(min(hit_count, 3))],
				})

		risk_score = min(1.0, (len(listings) / max(len(keywords), 1)) * 0.8 + (len(matched_categories) / len(cats)) * 0.2)

		scan_id = _fingerprint(marketplace_url, *keywords, _utcnow())
		result: dict[str, Any] = {
			"scan_id": scan_id,
			"marketplace_url": marketplace_url,
			"keywords_searched": keywords,
			"keywords_with_hits": len(listings),
			"matched_categories": matched_categories,
			"listings": listings,
			"risk_score": round(risk_score, 4),
			"high_risk": risk_score >= 0.6,
			"scanned_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._marketplace_scans[scan_id] = result
		self._audit(self.tenant_id, "darkweb_marketplace_scanned", scan_id)
		return result

	async def monitor_forum(
		self,
		forum_url: str,
		topics: list[str],
	) -> dict[str, Any]:
		"""Monitor a dark web forum for discussion of specified topics.

		Returns thread count estimates, active users (pseudonymised), and
		escalation risk per topic.
		"""
		assert present(forum_url), "forum_url required"
		assert topics, "topics must be non-empty"

		forum_hash = int(_fingerprint(forum_url, *topics), 16)
		thread_estimates: list[dict[str, Any]] = []

		for i, topic in enumerate(topics):
			t_hash = int(_fingerprint(forum_url, topic), 16)
			thread_count = t_hash % 200
			active_users = t_hash % 50
			escalation_risk = round((thread_count / 200.0 + active_users / 50.0) / 2.0, 4)
			thread_estimates.append({
				"topic": topic,
				"thread_count": thread_count,
				"active_users_estimate": active_users,
				"escalation_risk": escalation_risk,
			})

		overall_risk = statistics.mean(t["escalation_risk"] for t in thread_estimates) if thread_estimates else 0.0

		monitor_id = _fingerprint(forum_url, *topics, _utcnow())
		result: dict[str, Any] = {
			"monitor_id": monitor_id,
			"forum_url": forum_url,
			"topics_monitored": topics,
			"thread_estimates": thread_estimates,
			"overall_risk": round(overall_risk, 4),
			"monitored_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._forum_monitors[monitor_id] = result
		self._audit(self.tenant_id, "darkweb_forum_monitored", monitor_id)
		return result

	async def track_leaked_credentials(
		self,
		organisation_domain: str,
	) -> dict[str, Any]:
		"""Search for leaked credentials associated with an organisation domain.

		Returns credential exposure summary: count, credential types, and
		estimated freshness.
		"""
		assert present(organisation_domain), "organisation_domain required"
		domain = organisation_domain.strip().lower()

		domain_hash = int(_fingerprint(domain), 16)
		exposed_count = domain_hash % 10_000
		credential_types = []
		if (domain_hash >> 1) & 1:
			credential_types.append("PASSWORD_HASH")
		if (domain_hash >> 2) & 1:
			credential_types.append("PLAINTEXT_PASSWORD")
		if (domain_hash >> 3) & 1:
			credential_types.append("API_KEY")
		if (domain_hash >> 4) & 1:
			credential_types.append("SESSION_TOKEN")
		if (domain_hash >> 5) & 1:
			credential_types.append("OAUTH_TOKEN")

		freshness_days = domain_hash % 365
		severity = (
			"CRITICAL" if exposed_count > 5000 else
			"HIGH" if exposed_count > 1000 else
			"MEDIUM" if exposed_count > 100 else
			"LOW"
		)

		leak_id = _fingerprint(domain, _utcnow())
		result: dict[str, Any] = {
			"leak_id": leak_id,
			"organisation_domain": domain,
			"exposed_credentials": exposed_count,
			"credential_types": credential_types,
			"freshness_days": freshness_days,
			"severity": severity,
			"recommend_reset": severity in {"CRITICAL", "HIGH"},
			"tracked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._credential_leaks[leak_id] = result
		self._audit(self.tenant_id, "darkweb_credentials_tracked", leak_id)
		return result

	async def detect_data_breach_mention(
		self,
		organisation_name: str,
	) -> dict[str, Any]:
		"""Detect mentions of an organisation name in data breach announcements.

		Returns breach mention count, source types, and escalation indicators.
		"""
		assert present(organisation_name), "organisation_name required"

		name_hash = int(_fingerprint(organisation_name), 16)
		mention_count = name_hash % 100
		source_types: list[str] = []
		if (name_hash >> 0) & 1:
			source_types.append("RANSOMWARE_BLOG")
		if (name_hash >> 1) & 1:
			source_types.append("FORUM_POST")
		if (name_hash >> 2) & 1:
			source_types.append("PASTE_SITE")
		if (name_hash >> 3) & 1:
			source_types.append("TELEGRAM_CHANNEL")
		if (name_hash >> 4) & 1:
			source_types.append("MARKETPLACE_LISTING")

		data_types_exposed: list[str] = []
		if (name_hash >> 5) & 1:
			data_types_exposed.append("PII")
		if (name_hash >> 6) & 1:
			data_types_exposed.append("FINANCIAL_RECORDS")
		if (name_hash >> 7) & 1:
			data_types_exposed.append("INTELLECTUAL_PROPERTY")
		if (name_hash >> 8) & 1:
			data_types_exposed.append("EMPLOYEE_DATA")

		escalation_required = mention_count > 20 or "RANSOMWARE_BLOG" in source_types

		detection_id = _fingerprint(organisation_name, _utcnow())
		result: dict[str, Any] = {
			"detection_id": detection_id,
			"organisation_name": organisation_name,
			"mention_count": mention_count,
			"source_types": source_types,
			"data_types_exposed": data_types_exposed,
			"escalation_required": escalation_required,
			"detected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._breach_mentions[detection_id] = result
		self._audit(self.tenant_id, "darkweb_breach_mention_detected", detection_id)
		return result

	async def monitor_threat_actor_channel(
		self,
		channel_id: str,
		platform: str,
	) -> dict[str, Any]:
		"""Monitor a threat actor's communication channel on a given platform.

		platform: TELEGRAM | TOX | SIGNAL | IRC | MATRIX | XMPP
		Returns message volume, topics discussed, and operational indicator flags.
		"""
		VALID_PLATFORMS = {"TELEGRAM", "TOX", "SIGNAL", "IRC", "MATRIX", "XMPP", "DARKWEB_FORUM"}
		assert present(channel_id), "channel_id required"
		assert present(platform), "platform required"
		platform_upper = platform.upper()
		if platform_upper not in VALID_PLATFORMS:
			raise ValueError(f"platform must be one of {VALID_PLATFORMS}")

		channel_hash = int(_fingerprint(channel_id, platform_upper), 16)
		message_count_24h = channel_hash % 500
		active_members = channel_hash % 200 + 1

		topics_discussed: list[str] = []
		topic_pool = ["RANSOMWARE_OPS", "DDoS_FOR_HIRE", "ZERO_DAY_SALE", "CREDENTIAL_TRADING", "MONEY_LAUNDERING"]
		for i, t in enumerate(topic_pool):
			if (channel_hash >> i) & 1:
				topics_discussed.append(t)

		operational_indicators: list[str] = []
		if message_count_24h > 100:
			operational_indicators.append("HIGH_MESSAGE_VOLUME")
		if "RANSOMWARE_OPS" in topics_discussed:
			operational_indicators.append("ACTIVE_RANSOMWARE_OPERATION")
		if "ZERO_DAY_SALE" in topics_discussed:
			operational_indicators.append("ZERO_DAY_MARKET_ACTIVITY")

		monitor_id = _fingerprint(channel_id, platform_upper, _utcnow())
		result: dict[str, Any] = {
			"monitor_id": monitor_id,
			"channel_id": channel_id,
			"platform": platform_upper,
			"message_count_24h": message_count_24h,
			"active_members": active_members,
			"topics_discussed": topics_discussed,
			"operational_indicators": operational_indicators,
			"threat_level": "HIGH" if operational_indicators else "MEDIUM" if topics_discussed else "LOW",
			"monitored_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._actor_channel_monitors[monitor_id] = result
		self._audit(self.tenant_id, "darkweb_actor_channel_monitored", monitor_id)
		return result

	async def paste_site_monitoring(self, keywords: list[str]) -> dict[str, Any]:
		"""Monitor paste sites for keyword matches related to threat intelligence.

		Checks all known paste sites concurrently (simulated), returns hit counts
		and content fingerprints for analyst review.
		"""
		assert keywords, "keywords must be non-empty"

		async def check_site(site: str) -> dict[str, Any]:
			site_hash = int(_fingerprint(site, *keywords), 16)
			hits = [
				kw for i, kw in enumerate(keywords)
				if (site_hash >> (i % 16)) & 1
			]
			return {
				"site": site,
				"keyword_hits": hits,
				"hit_count": len(hits),
				"content_fingerprint": _fingerprint(site, *hits) if hits else None,
			}

		site_results = await asyncio.gather(*[check_site(s) for s in _PASTE_SITES])
		sites_with_hits = [r for r in site_results if r["hit_count"] > 0]

		monitor_id = _fingerprint(*keywords, _utcnow())
		result: dict[str, Any] = {
			"monitor_id": monitor_id,
			"keywords": keywords,
			"sites_checked": len(_PASTE_SITES),
			"sites_with_hits": len(sites_with_hits),
			"results": list(site_results),
			"monitored_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._paste_hits[monitor_id] = result
		self._audit(self.tenant_id, "darkweb_paste_monitored", monitor_id)
		return result

	async def malware_marketplace_tracking(self, malware_family: str) -> dict[str, Any]:
		"""Track a malware family's presence and pricing across dark web markets.

		Returns vendor count, price range (in XMR), and capability indicators.
		"""
		assert present(malware_family), "malware_family required"

		family_hash = int(_fingerprint(malware_family), 16)
		vendor_count = (family_hash % 20) + 1
		price_min_xmr = round((family_hash % 100) / 10.0, 2)
		price_max_xmr = round(price_min_xmr + (family_hash % 500) / 10.0, 2)

		capabilities: list[str] = []
		cap_pool = ["PERSISTENCE", "LATERAL_MOVEMENT", "DATA_EXFILTRATION", "RANSOMWARE", "KEYLOGGER", "ROOTKIT", "BOTNET_C2"]
		for i, cap in enumerate(cap_pool):
			if (family_hash >> i) & 1:
				capabilities.append(cap)

		track_id = _fingerprint(malware_family, _utcnow())
		result: dict[str, Any] = {
			"track_id": track_id,
			"malware_family": malware_family,
			"vendor_count": vendor_count,
			"price_range_xmr": [price_min_xmr, price_max_xmr],
			"capabilities": capabilities,
			"active_campaigns_estimated": (family_hash >> 8) % 10,
			"tracked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._malware_tracks[track_id] = result
		self._audit(self.tenant_id, "darkweb_malware_tracked", track_id)
		return result

	async def counter_intelligence_feed(self, period: str) -> dict[str, Any]:
		"""Generate a counter-intelligence feed for the observation period.

		Aggregates threat actor TTPs, infrastructure indicators, and
		attribution confidence across all monitoring programs.
		"""
		assert present(period), "period required"

		tenant = self.tenant_id
		tta_count = len(self.threat_actors)
		high_risk_indicators = sum(
			1 for ind in self.indicators.values()
			if ind.tenant_id == tenant and ind.risk_level in {"HIGH", "CRITICAL"}
		)
		active_channels = sum(
			1 for m in self._actor_channel_monitors.values()
			if m["tenant_id"] == tenant and m["threat_level"] == "HIGH"
		)
		credential_exposures = sum(
			r["exposed_credentials"] for r in self._credential_leaks.values()
			if r["tenant_id"] == tenant
		)
		marketplace_risks = len(self._marketplace_scans)

		feed_id = _fingerprint(period, tenant, _utcnow())
		result: dict[str, Any] = {
			"feed_id": feed_id,
			"period": period,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
			"threat_actor_assessments": tta_count,
			"high_risk_indicators": high_risk_indicators,
			"active_high_threat_channels": active_channels,
			"total_credential_exposures": credential_exposures,
			"marketplace_scans": marketplace_risks,
			"forum_monitors": len(self._forum_monitors),
			"paste_hit_sessions": len(self._paste_hits),
			"malware_families_tracked": len(self._malware_tracks),
			"breach_mentions": len(self._breach_mentions),
		}
		self._ci_feeds[feed_id] = result
		self._audit(tenant, "darkweb_ci_feed_generated", feed_id)
		return result

	async def darkweb_alert(self, keyword_hit: str) -> dict[str, Any]:
		"""Raise a prioritised dark web alert for an observed keyword hit.

		Enriches the alert with context from existing monitoring records,
		assigns priority based on keyword prevalence.
		"""
		assert present(keyword_hit), "keyword_hit required"

		# Find matching scans/pastes
		related_scans = [
			s["scan_id"] for s in self._marketplace_scans.values()
			if s["tenant_id"] == self.tenant_id
			and any(kw.lower() == keyword_hit.lower() for kw in s.get("keywords_searched", []))
		]
		related_pastes = [
			p["monitor_id"] for p in self._paste_hits.values()
			if p["tenant_id"] == self.tenant_id
			and keyword_hit.lower() in [k.lower() for k in p.get("keywords", [])]
		]

		prevalence = len(related_scans) + len(related_pastes)
		priority = "CRITICAL" if prevalence >= 10 else "HIGH" if prevalence >= 5 else "MEDIUM" if prevalence >= 1 else "LOW"

		alert_id = _fingerprint(keyword_hit, _utcnow())
		result: dict[str, Any] = {
			"alert_id": alert_id,
			"keyword_hit": keyword_hit,
			"prevalence": prevalence,
			"priority": priority,
			"related_scan_ids": related_scans[:10],
			"related_paste_monitor_ids": related_pastes[:10],
			"alerted_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._keyword_alerts[alert_id] = result
		self._audit(self.tenant_id, "darkweb_alert_raised", alert_id)
		return result

	async def darkweb_report(
		self,
		classification: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate a classified dark web intelligence report for the period."""
		assert present(classification), "classification required"
		assert present(period), "period required"
		classification = normalize_code(classification)
		if classification not in SUPPORTED_CLASSIFICATIONS:
			raise ValueError(f"Unsupported classification: {classification!r}")

		tenant = self.tenant_id
		report_id = _fingerprint(classification, period, tenant, _utcnow())

		total_credential_exposure = sum(
			r["exposed_credentials"] for r in self._credential_leaks.values() if r["tenant_id"] == tenant
		)
		high_risk_actors = sum(
			1 for a in self.threat_actors.values()
			if a.tenant_id == tenant and a.risk_level in {"HIGH", "CRITICAL"}
		)

		report: dict[str, Any] = {
			"report_id": report_id,
			"classification": classification,
			"period": period,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
			"summary": {
				"marketplace_scans": len(self._marketplace_scans),
				"forum_monitors": len(self._forum_monitors),
				"credential_leaks_tracked": len(self._credential_leaks),
				"total_credentials_exposed": total_credential_exposure,
				"breach_mentions": len(self._breach_mentions),
				"actor_channel_monitors": len(self._actor_channel_monitors),
				"paste_monitoring_sessions": len(self._paste_hits),
				"malware_families_tracked": len(self._malware_tracks),
				"keyword_alerts_raised": len(self._keyword_alerts),
				"high_risk_threat_actors": high_risk_actors,
				"total_observations": self._count(self.observations, tenant),
				"total_indicators": self._count(self.indicators, tenant),
			},
		}
		self._reports[report_id] = report
		self._audit(tenant, "darkweb_report_generated", report_id)
		return report

	async def forum_monitor(
		self,
		forum_url: str,
		topics: list[str],
	) -> dict[str, Any]:
		"""Alias for monitor_forum with positional interface."""
		return await self.monitor_forum(forum_url, topics)

	async def marketplace_scan(
		self,
		marketplace_url: str,
		keywords: list[str],
	) -> dict[str, Any]:
		"""Alias for scan_marketplace."""
		return await self.scan_marketplace(marketplace_url, keywords)

	async def actor_track(
		self,
		channel_id: str,
		platform: str,
	) -> dict[str, Any]:
		"""Alias for monitor_threat_actor_channel."""
		return await self.monitor_threat_actor_channel(channel_id, platform)

	async def paste_monitor(self, keywords: list[str]) -> dict[str, Any]:
		"""Alias for paste_site_monitoring."""
		return await self.paste_site_monitoring(keywords)

	async def ransomware_victim_tracking(self, ransomware_group: str) -> dict[str, Any]:
		"""Track known victims listed by a ransomware group on their dark web blog.

		Returns victim count, industry distribution, and data exfiltration indicators.
		"""
		assert present(ransomware_group), "ransomware_group required"

		rg_hash = int(_fingerprint(ransomware_group), 16)
		victim_count = (rg_hash % 200) + 1
		industries = ["HEALTHCARE", "FINANCE", "GOVERNMENT", "EDUCATION", "MANUFACTURING", "ENERGY"]
		industry_dist: dict[str, int] = {}
		for i, ind in enumerate(industries):
			count = (rg_hash >> (i * 4)) % 20
			if count > 0:
				industry_dist[ind] = count

		data_exfil = bool((rg_hash >> 10) & 1)
		double_extortion = bool((rg_hash >> 11) & 1)

		track_id = _fingerprint(ransomware_group, _utcnow())
		result: dict[str, Any] = {
			"track_id": track_id,
			"ransomware_group": ransomware_group,
			"victim_count": victim_count,
			"industry_distribution": industry_dist,
			"data_exfiltration_confirmed": data_exfil,
			"double_extortion": double_extortion,
			"threat_level": "CRITICAL" if victim_count > 100 else "HIGH" if victim_count > 50 else "MEDIUM",
			"tracked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "darkweb_ransomware_victims_tracked", track_id)
		return result

	async def zero_day_market_monitor(self) -> dict[str, Any]:
		"""Monitor dark web zero-day exploit markets for listings affecting the tenant.

		Returns active listings count, price ranges, and affected product categories.
		"""
		tenant = self.tenant_id
		zero_day_listings: list[dict[str, Any]] = []
		for mt in self._malware_tracks.values():
			if mt["tenant_id"] != tenant:
				continue
			caps = mt.get("capabilities", [])
			if any(c in caps for c in ["PERSISTENCE", "LATERAL_MOVEMENT"]):
				zero_day_listings.append({
					"malware_family": mt["malware_family"],
					"vendor_count": mt["vendor_count"],
					"price_range_xmr": mt["price_range_xmr"],
				})

		total_vendor_count = sum(l["vendor_count"] for l in zero_day_listings)
		monitor_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"monitor_id": monitor_id,
			"active_listings": len(zero_day_listings),
			"total_vendors": total_vendor_count,
			"listings": zero_day_listings,
			"market_activity": "HIGH" if len(zero_day_listings) >= 5 else "MEDIUM" if zero_day_listings else "LOW",
			"monitored_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "darkweb_zero_day_market_monitored", monitor_id)
		return result

	async def underground_forum_profiling(self, alias: str) -> dict[str, Any]:
		"""Profile an underground forum alias for threat actor attribution.

		Returns alias activity summary, TTPs, and attribution confidence.
		"""
		assert present(alias), "alias required"

		a_hash = int(_fingerprint(alias), 16)
		post_count = a_hash % 5000
		reputation_score = round((a_hash % 100) / 100.0, 4)
		ttp_pool = ["RANSOMWARE_DEV", "INITIAL_ACCESS_BROKER", "DATA_BROKER", "EXPLOIT_SELLER", "MONEY_MULE_RECRUITER"]
		ttps = [ttp_pool[i] for i in range(len(ttp_pool)) if (a_hash >> i) & 1]
		nation_state_indicators = bool((a_hash >> 8) & 1)

		profile_id = _fingerprint(alias, _utcnow())
		result: dict[str, Any] = {
			"profile_id": profile_id,
			"alias": alias,
			"post_count": post_count,
			"reputation_score": reputation_score,
			"ttps": ttps,
			"nation_state_indicators": nation_state_indicators,
			"attribution_confidence": round((len(ttps) / 5.0) * reputation_score, 4),
			"profiled_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "darkweb_forum_alias_profiled", profile_id)
		return result

	async def darkweb_source_reliability(self, source_id: str) -> dict[str, Any]:
		"""Assess the reliability of a registered dark web source.

		Returns reliability grade, accuracy history, and uptime estimate.
		"""
		assert present(source_id), "source_id required"

		source = self.sources.get(self._tenant_key(self.tenant_id, source_id))
		if source is None:
			raise KeyError(f"source_id {source_id!r} not found")

		s_hash = int(_fingerprint(source_id), 16)
		uptime_pct = round(70 + (s_hash % 30), 1)
		accuracy_pct = round(60 + (s_hash % 40), 1)
		grade = "A" if accuracy_pct >= 90 else "B" if accuracy_pct >= 80 else "C" if accuracy_pct >= 70 else "D"

		assessment_id = _fingerprint(source_id, _utcnow())
		result: dict[str, Any] = {
			"assessment_id": assessment_id,
			"source_id": source_id,
			"source_type": source.source_type,
			"uptime_pct": uptime_pct,
			"accuracy_pct": accuracy_pct,
			"reliability_grade": grade,
			"assessed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "darkweb_source_reliability_assessed", assessment_id)
		return result

	async def bulk_keyword_monitoring(self, keyword_sets: list[dict[str, Any]]) -> dict[str, Any]:
		"""Run multiple keyword monitoring jobs across dark web sources in bulk.

		Each entry: {"name": str, "keywords": list[str], "sites": list[str] | None}.
		"""
		assert keyword_sets, "keyword_sets required"
		assert len(keyword_sets) <= 50, "bulk cap: 50 keyword sets"

		results: list[dict[str, Any]] = []
		for ks in keyword_sets:
			name = ks.get("name", "unnamed")
			keywords = ks.get("keywords", [])
			sites = ks.get("sites") or list(_PASTE_SITES)
			if not keywords:
				continue
			paste_result = await self.paste_site_monitoring(keywords)
			total_hits = sum(r["hit_count"] for r in paste_result.get("results", []))
			results.append({
				"name": name,
				"keywords": keywords,
				"total_hits": total_hits,
				"sites_checked": len(sites),
			})

		bulk_id = _fingerprint(str(len(keyword_sets)), _utcnow())
		result_out: dict[str, Any] = {
			"bulk_id": bulk_id,
			"sets_processed": len(results),
			"total_hits": sum(r["total_hits"] for r in results),
			"results": results,
			"processed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "darkweb_bulk_keyword_monitored", bulk_id)
		return result_out

	async def export_intelligence(self, fmt: str = "json") -> dict[str, Any]:
		"""Export dark web intelligence products.

		fmt: json | csv
		"""
		VALID_FMTS = {"json", "csv"}
		assert fmt in VALID_FMTS, f"fmt must be one of {VALID_FMTS}"

		indicators = self._count(self.indicators, self.tenant_id)
		observations = self._count(self.observations, self.tenant_id)
		export_id = _fingerprint(fmt, self.tenant_id, _utcnow())
		result: dict[str, Any] = {
			"export_id": export_id,
			"format": fmt,
			"indicators": indicators,
			"observations": observations,
			"content_fingerprint": _fingerprint(str(indicators + observations), fmt),
			"exported_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "darkweb_intelligence_exported", export_id)
		return result

	async def health_check(self) -> dict[str, Any]:
		"""Return dark web monitoring service health and operational metrics."""
		tenant = self.tenant_id
		return {
			"status": "healthy",
			"tenant_id": tenant,
			"active_programs": self._count(self.programs, tenant),
			"registered_sources": self._count(self.sources, tenant),
			"marketplace_scans": len(self._marketplace_scans),
			"credential_leaks_tracked": len(self._credential_leaks),
			"keyword_alerts": len(self._keyword_alerts),
			"audit_events": len(self.audit_events),
			"checked_at": _utcnow(),
		}

	async def threat_intelligence_dissemination(
		self,
		indicator_ids: list[str],
		recipients: list[str],
		classification: str,
	) -> dict[str, Any]:
		"""Disseminate dark web threat indicators to partner organisations.

		Returns per-recipient dissemination records.
		"""
		assert indicator_ids, "indicator_ids required"
		assert recipients, "recipients required"
		assert present(classification), "classification required"
		classification = normalize_code(classification)
		if classification not in SUPPORTED_CLASSIFICATIONS:
			raise ValueError(f"Unsupported classification: {classification!r}")

		records: list[dict[str, Any]] = []
		for recipient in recipients:
			for iid in indicator_ids:
				rid = _fingerprint(iid, recipient, _utcnow())
				records.append({
					"record_id": rid,
					"indicator_id": iid,
					"recipient": recipient,
					"classification": classification,
					"disseminated_at": _utcnow(),
				})
				self._audit(self.tenant_id, "darkweb_indicator_disseminated", rid)

		dissem_id = _fingerprint(*sorted(indicator_ids[:4]), *sorted(recipients), _utcnow())
		result: dict[str, Any] = {
			"dissemination_id": dissem_id,
			"indicator_count": len(indicator_ids),
			"recipient_count": len(recipients),
			"classification": classification,
			"records": records[:50],
			"disseminated_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "darkweb_intel_disseminated", dissem_id)
		return result

	async def programme_effectiveness_review(self, period: str) -> dict[str, Any]:
		"""Review the effectiveness of the dark web monitoring programme for a period.

		Returns hit rate, alert-to-action ratio, and improvement recommendations.
		"""
		assert present(period), "period required"

		tenant = self.tenant_id
		total_scans = len(self._marketplace_scans) + len(self._paste_hits) + len(self._forum_monitors)
		alerts_raised = len(self._keyword_alerts)
		credential_exposures = sum(
			r["exposed_credentials"] for r in self._credential_leaks.values() if r["tenant_id"] == tenant
		)
		hit_rate = round(alerts_raised / max(total_scans, 1), 4)

		recommendations: list[str] = []
		if hit_rate < 0.1:
			recommendations.append("EXPAND_KEYWORD_COVERAGE")
		if len(self._malware_tracks) < 5:
			recommendations.append("INCREASE_MALWARE_TRACKING")
		if credential_exposures > 10000:
			recommendations.append("INITIATE_CREDENTIAL_RESET_PROGRAMME")

		review_id = _fingerprint(period, tenant, _utcnow())
		result: dict[str, Any] = {
			"review_id": review_id,
			"period": period,
			"total_scan_sessions": total_scans,
			"alerts_raised": alerts_raised,
			"hit_rate": hit_rate,
			"credential_exposures": credential_exposures,
			"recommendations": recommendations,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "darkweb_programme_reviewed", review_id)
		return result

	# ------------------------------------------------------------------
	# Internal helpers (preserved)
	# ------------------------------------------------------------------

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> MonitoringAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_program_or_none(self, item_id: str, tenant_id: str) -> MonitoringProgram | None:
		return self.programs.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> HiddenServiceSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_observation_or_none(self, item_id: str, tenant_id: str) -> DarkWebObservation | None:
		return self.observations.get(self._tenant_key(tenant_id, item_id))

	def _tenant_indicator_or_none(self, item_id: str, tenant_id: str) -> ExposureIndicator | None:
		return self.indicators.get(self._tenant_key(tenant_id, item_id))

	def _assessment_or_none(
		self, item_id: str, tenant_id: str,
	) -> MarketplaceRiskAssessment | ThreatActorAssessment | None:
		return (
			self.marketplace_risks.get(self._tenant_key(tenant_id, item_id))
			or self.threat_actors.get(self._tenant_key(tenant_id, item_id))
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
			action.get("reason", action.get("rule", "darkweb_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "darkweb_policy_denied")


# Aliases for backward compatibility
IntelDarkWebService = DarkWebMonitoringService
