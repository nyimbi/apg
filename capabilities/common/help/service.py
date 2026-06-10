"""Executable service layer for the APG HELP capability."""

from __future__ import annotations

from itertools import count
from typing import Any

from .capability_contract import (
	DEFAULT_CONFIGURATION,
	PRIVILEGED_HELP_AGENT_ROLES,
	SUPPORTED_HELP_AGENT_ROLES,
	SUPPORTED_HELP_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .help_runtime import HelpAnswerComposer, HelpFreshnessInspector, HelpSearchIndex
from .models import (
	ArticleStatus,
	ContentVisibility,
	HelpAnswer,
	HelpArticle,
	HelpAuditEvent,
	HelpAgentRecord,
	HelpCurationItem,
	HelpFeedback,
	HelpLifecycleBatchRecord,
	HelpLocalization,
	HelpSource,
	utc_now_iso,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class HelpService:
	"""Tenant-aware article, answer, feedback, and curation runtime."""

	def __init__(self) -> None:
		self._sources: dict[str, HelpSource] = {}
		self._articles: dict[str, HelpArticle] = {}
		self._answers: dict[str, HelpAnswer] = {}
		self._feedback: dict[str, HelpFeedback] = {}
		self._localizations: dict[str, HelpLocalization] = {}
		self._curation: dict[str, HelpCurationItem] = {}
		self._help_agents: dict[str, HelpAgentRecord] = {}
		self._lifecycle_batches: dict[str, HelpLifecycleBatchRecord] = {}
		self._audit_events: dict[str, HelpAuditEvent] = {}
		self._counter = count(1)
		self._search_index = HelpSearchIndex()
		self._answer_composer = HelpAnswerComposer()
		self._freshness_inspector = HelpFreshnessInspector()
		self._agent_runtimes = {_normalize_token(item) for item in SUPPORTED_HELP_AGENT_RUNTIMES}
		self._agent_roles = {_normalize_token(item) for item in SUPPORTED_HELP_AGENT_ROLES}
		self._privileged_agent_roles = {_normalize_token(item) for item in PRIVILEGED_HELP_AGENT_ROLES}
		self._lifecycle_operations = {
			_normalize_token(item)
			for item in get_capability_contract()["streaming"]["required_operations"]
		}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_source(
		self,
		source_id: str,
		tenant_id: str,
		title: str,
		uri: str,
		owner_id: str,
		visibility: str = ContentVisibility.INTERNAL.value,
	) -> dict[str, Any]:
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="register_source",
			source_owner_assigned=bool(str(owner_id or "").strip()),
			source_uri_present=bool(str(uri or "").strip()),
			source_approval_required=False,
			source_approved=True,
		)
		source = HelpSource(
			id=source_id,
			tenant_id=tenant_id,
			title=title,
			uri=uri,
			owner_id=owner_id,
			visibility=ContentVisibility(visibility),
		)
		self._sources[source_id] = source
		self._record_event(tenant_id, "source_registered", source_id, f"Source registered: {title}", owner_id)
		return source.to_dict()

	def approve_source(self, source_id: str, tenant_id: str, approver_id: str) -> dict[str, Any]:
		source = self._require_source(source_id, tenant_id)
		source.approved = True
		source.approved_by = approver_id
		source.approved_at = utc_now_iso()
		self._record_event(tenant_id, "source_approved", source_id, f"Source approved: {source.title}", approver_id)
		return source.to_dict()

	def create_article(
		self,
		article_id: str,
		tenant_id: str,
		title: str,
		body: str,
		owner_id: str,
		topics: list[str] | None = None,
		locale: str = "en",
		visibility: str = ContentVisibility.INTERNAL.value,
		source_ids: list[str] | None = None,
		source_approval_recorded: bool = True,
	) -> dict[str, Any]:
		source_ids = list(source_ids or [])
		all_sources_approved = source_approval_recorded and self._source_ids_are_approved(source_ids, tenant_id)
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="create_article",
			article_owner_assigned=bool(str(owner_id or "").strip()),
			article_title_present=bool(str(title or "").strip()),
			article_body_present=bool(str(body or "").strip()),
			source_approval_required=bool(source_ids),
			source_approved=all_sources_approved if source_ids else True,
		)
		article = HelpArticle(
			id=article_id,
			tenant_id=tenant_id,
			title=title,
			body=body,
			owner_id=owner_id,
			topics=list(topics or []),
			locale=locale,
			visibility=ContentVisibility(visibility),
			source_ids=source_ids,
		)
		self._articles[article_id] = article
		self._record_event(tenant_id, "article_created", article_id, f"Article created: {title}", owner_id)
		return article.to_dict()

	def publish_article(
		self,
		article_id: str,
		tenant_id: str,
		approver_id: str,
		publication_approved: bool,
		rbac_filter_applied: bool = True,
		freshness_review_recorded: bool = True,
		article_age_days: int = 0,
	) -> dict[str, Any]:
		article = self._require_article(article_id, tenant_id)
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="publish_article",
			publication_approved=publication_approved,
			source_approval_required=bool(article.source_ids),
			source_approved=self._source_ids_are_approved(article.source_ids, tenant_id) if article.source_ids else True,
			restricted_content_present=article.visibility == ContentVisibility.RESTRICTED,
			rbac_filter_applied=rbac_filter_applied,
			freshness_review_recorded=freshness_review_recorded,
			article_age_days=article_age_days,
			audit_event_recorded=True,
		)
		now = utc_now_iso()
		article.status = ArticleStatus.PUBLISHED
		article.published_at = now
		article.last_reviewed_at = now
		article.updated_at = now
		curation_id = f"publish-{article_id}-{next(self._counter)}"
		self._curation[curation_id] = HelpCurationItem(
			id=curation_id,
			tenant_id=tenant_id,
			article_id=article_id,
			reason="publication_approved",
			status="closed",
			reviewer_id=approver_id,
			closed_at=now,
		)
		self._record_event(tenant_id, "article_published", article_id, f"Article published: {article.title}", approver_id)
		return article.to_dict()

	def search_articles(
		self,
		tenant_id: str,
		query: str,
		locale: str | None = None,
		rbac_filter_applied: bool = True,
		include_restricted: bool = False,
		limit: int = 5,
	) -> list[dict[str, Any]]:
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="search_articles",
			query_present=bool(str(query or "").strip()),
			restricted_content_present=include_restricted,
			rbac_filter_applied=rbac_filter_applied,
			query_logging_enabled=True,
			allow_review=True,
		)
		hits = self._search_index.search(
			query=query,
			articles=list(self._articles.values()),
			tenant_id=tenant_id,
			locale=locale,
			include_restricted=include_restricted and rbac_filter_applied,
			limit=limit,
		)
		return [
			{
				"article": hit["article"].to_dict(),
				"score": hit["score"],
				"snippet": hit["snippet"],
			}
			for hit in hits
		]

	def generate_answer(
		self,
		answer_id: str,
		tenant_id: str,
		query: str,
		locale: str | None = None,
		rbac_filter_applied: bool = True,
		include_restricted: bool = False,
		minimum_confidence: float | None = None,
	) -> dict[str, Any]:
		hits = self._search_index.search(
			query=query,
			articles=list(self._articles.values()),
			tenant_id=tenant_id,
			locale=locale,
			include_restricted=include_restricted and rbac_filter_applied,
		)
		minimum_confidence = minimum_confidence or float(DEFAULT_CONFIGURATION["answers"]["minimum_answer_confidence"])
		if not str(query or "").strip():
			self._enforce_help_policy(tenant_id=tenant_id, operation="generate_answer", query_present=False)
		answer_text, confidence, citations, block_reason = self._answer_composer.compose(
			query=query,
			hits=hits,
			minimum_confidence=minimum_confidence,
		)
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="generate_answer",
			query_present=bool(str(query or "").strip()),
			citations_present=bool(citations),
			answer_confidence=confidence,
			unsafe_answer_detected=self._unsafe_answer_detected(answer_text),
			restricted_content_present=include_restricted,
			rbac_filter_applied=rbac_filter_applied,
			allow_review=True,
		)
		answer = HelpAnswer(
			id=answer_id,
			tenant_id=tenant_id,
			query=query,
			answer=answer_text,
			confidence=confidence,
			citations=citations,
			blocked=block_reason is not None,
			block_reason=block_reason,
		)
		self._answers[answer_id] = answer
		self._record_event(tenant_id, "answer_generated", answer_id, f"Answer generated for query: {query}", "answer-service")
		return answer.to_dict()

	def localize_article(
		self,
		localization_id: str,
		tenant_id: str,
		article_id: str,
		locale: str,
		title: str,
		body: str,
		translator_id: str,
		source_locale: str = "en",
		fallback_locale: str = "en",
	) -> dict[str, Any]:
		article = self._require_article(article_id, tenant_id)
		supported_locales = set(DEFAULT_CONFIGURATION["localization"]["supported_locales"])
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="localize_article",
			locale_supported=locale in supported_locales,
			translator_assigned=bool(str(translator_id or "").strip()),
			fallback_locale_configured=bool(str(fallback_locale or "").strip()),
			allow_review=True,
		)
		localization = HelpLocalization(
			id=localization_id,
			tenant_id=tenant_id,
			article_id=article.id,
			locale=locale,
			source_locale=source_locale,
			title=title,
			body=body,
			translator_id=translator_id,
			fallback_locale=fallback_locale,
		)
		self._localizations[localization_id] = localization
		self._record_event(tenant_id, "article_localized", localization_id, f"Article localized to {locale}: {article.title}", translator_id)
		return localization.to_dict()

	def record_feedback(
		self,
		feedback_id: str,
		tenant_id: str,
		user_id: str,
		rating: int,
		comment: str = "",
		article_id: str | None = None,
		answer_id: str | None = None,
		requires_review: bool | None = None,
	) -> dict[str, Any]:
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="record_feedback",
			feedback_user_present=bool(str(user_id or "").strip()),
			feedback_rating=rating,
			feedback_review_opened=True if rating <= 2 else False,
			allow_review=True,
		)
		if article_id is not None:
			self._require_article(article_id, tenant_id)
		if answer_id is not None:
			self._require_answer(answer_id, tenant_id)
		needs_review = rating <= 2 if requires_review is None else requires_review
		feedback = HelpFeedback(
			id=feedback_id,
			tenant_id=tenant_id,
			user_id=user_id,
			rating=rating,
			comment=comment,
			article_id=article_id,
			answer_id=answer_id,
			requires_review=needs_review,
		)
		self._feedback[feedback_id] = feedback
		if needs_review and article_id:
			self._open_curation_item(tenant_id, article_id, "support_feedback_review")
		self._record_event(tenant_id, "feedback_recorded", feedback_id, f"Feedback recorded with rating {rating}", user_id)
		return feedback.to_dict()

	def close_curation_item(
		self,
		curation_id: str,
		tenant_id: str,
		reviewer_id: str,
		evidence: list[str],
	) -> dict[str, Any]:
		item = self._require_curation_item(curation_id, tenant_id)
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="close_curation_item",
			reviewer_present=bool(str(reviewer_id or "").strip()),
			curation_evidence_present=bool(evidence),
		)
		item.status = "closed"
		item.reviewer_id = reviewer_id
		item.evidence = list(evidence)
		item.closed_at = utc_now_iso()
		self._record_event(tenant_id, "curation_closed", curation_id, f"Curation closed: {item.reason}", reviewer_id)
		return item.to_dict()

	def freshness_queue(self, tenant_id: str) -> list[dict[str, Any]]:
		threshold = int(DEFAULT_CONFIGURATION["content"]["freshness_review_days"])
		for article in self._freshness_inspector.stale_articles(
			list(self._articles.values()),
			tenant_id,
			threshold,
		):
			self._open_curation_item(tenant_id, article.id, "freshness_review")
		return self.list_curation_items(tenant_id)

	def register_help_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		agent_id_value = str(agent_id or "").strip()
		name_value = str(name or "").strip()
		record_key = self._tenant_key(tenant_id, agent_id_value)
		if record_key in self._help_agents:
			raise ValueError(f"help_agent_already_exists:{agent_id_value}")
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "register_help_agent",
			"agent_id_present": bool(agent_id_value),
			"agent_name_present": bool(name_value),
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = HelpAgentRecord(
			id=agent_id_value,
			tenant_id=tenant_id,
			name=name_value,
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status="pending_review" if result["decision"] == "require_review" else "active",
		)
		self._help_agents[record_key] = record
		self._record_event(tenant_id, "help_agent_registered", agent_id, f"Help agent registered: {record.name}", record.owner)
		return record.to_dict()

	def validate_help_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "help_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		mutation_count = int(mutation_count)
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "validate_help_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
			"lifecycle_operation_supported": operation_value in self._lifecycle_operations,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		accepted = result["decision"] == "allow"
		record_id = batch_id or f"help-batch-{len(self._lifecycle_batches) + 1:06d}"
		record = HelpLifecycleBatchRecord(
			id=record_id,
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._tenant_key(tenant_id, record_id)] = record
		self._record_event(tenant_id, f"help_lifecycle_batch_{record.status}", record_id, f"HELP lifecycle batch {record.status}: {operation_value}", "bytewax")
		if result["decision"] == "deny":
			self._raise_policy(result)
		return record.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper for generated package probes."""
		data = dict(metadata or {})
		article = self.create_article(
			article_id=record_id,
			tenant_id=tenant_id,
			title=str(data.get("title") or "Compatibility article"),
			body=str(data.get("body") or "Compatibility help article for generated package probes."),
			owner_id=str(data.get("owner_id") or "system"),
			topics=list(data.get("topics") or [status]),
			visibility=str(data.get("visibility") or ContentVisibility.INTERNAL.value),
			source_ids=list(data.get("source_ids") or []),
		)
		return article

	def list_articles(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._articles, tenant_id)

	def list_sources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sources, tenant_id)

	def list_answers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._answers, tenant_id)

	def list_feedback(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._feedback, tenant_id)

	def list_localizations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._localizations, tenant_id)

	def list_curation_items(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._curation, tenant_id)

	def list_help_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._help_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		records: list[dict[str, Any]] = []
		for store in (self._sources, self._articles, self._answers, self._feedback, self._localizations, self._curation, self._help_agents, self._lifecycle_batches, self._audit_events):
			records.extend(self._list(store, tenant_id))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		articles = self.list_articles(tenant_id)
		answers = self.list_answers(tenant_id)
		feedback = self.list_feedback(tenant_id)
		sources = self.list_sources(tenant_id)
		localizations = self.list_localizations(tenant_id)
		curation = self.list_curation_items(tenant_id)
		agents = self.list_help_agents(tenant_id)
		lifecycle_batches = self.list_lifecycle_batches(tenant_id)
		return {
			"tenant_id": tenant_id,
			"source_count": len(sources),
			"approved_source_count": len([source for source in sources if source["approved"]]),
			"article_count": len(articles),
			"published_article_count": len([article for article in articles if article["status"] == ArticleStatus.PUBLISHED.value]),
			"answer_count": len(answers),
			"blocked_answer_count": len([answer for answer in answers if answer["blocked"]]),
			"feedback_count": len(feedback),
			"localization_count": len(localizations),
			"open_curation_count": len([item for item in curation if item["status"] == "open"]),
			"help_agent_count": len(agents),
			"pending_help_agent_review_count": len([agent for agent in agents if agent["status"] == "pending_review"]),
			"lifecycle_batch_count": len(lifecycle_batches),
			"denied_lifecycle_batch_count": len([batch for batch in lifecycle_batches if batch["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "help_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "help_policy_blocked")

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _enforce_help_policy(
		self,
		tenant_id: str,
		operation: str,
		source_owner_assigned: bool = True,
		source_uri_present: bool = True,
		source_approval_required: bool = False,
		source_approved: bool = True,
		article_owner_assigned: bool = True,
		article_title_present: bool = True,
		article_body_present: bool = True,
		publication_approved: bool = True,
		query_present: bool = True,
		citations_present: bool = True,
		answer_confidence: float = 1.0,
		unsafe_answer_detected: bool = False,
		restricted_content_present: bool = False,
		rbac_filter_applied: bool = True,
		query_logging_enabled: bool = True,
		article_age_days: int = 0,
		freshness_review_recorded: bool = True,
		feedback_user_present: bool = True,
		feedback_rating: int = 5,
		feedback_review_opened: bool = True,
		locale_supported: bool = True,
		translator_assigned: bool = True,
		fallback_locale_configured: bool = True,
		reviewer_present: bool = True,
		curation_evidence_present: bool = True,
		audit_event_recorded: bool = True,
		allow_review: bool = False,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": operation,
			"source_owner_assigned": source_owner_assigned,
			"source_uri_present": source_uri_present,
			"source_approval_required": source_approval_required,
			"source_approved": source_approved,
			"article_owner_assigned": article_owner_assigned,
			"article_title_present": article_title_present,
			"article_body_present": article_body_present,
			"publication_approved": publication_approved,
			"query_present": query_present,
			"citations_present": citations_present,
			"answer_confidence": answer_confidence,
			"unsafe_answer_detected": unsafe_answer_detected,
			"restricted_content_present": restricted_content_present,
			"rbac_filter_applied": rbac_filter_applied,
			"query_logging_enabled": query_logging_enabled,
			"article_age_days": article_age_days,
			"freshness_review_recorded": freshness_review_recorded,
			"feedback_user_present": feedback_user_present,
			"feedback_rating": feedback_rating,
			"feedback_review_opened": feedback_review_opened,
			"locale_supported": locale_supported,
			"translator_assigned": translator_assigned,
			"fallback_locale_configured": fallback_locale_configured,
			"reviewer_present": reviewer_present,
			"curation_evidence_present": curation_evidence_present,
			"audit_event_recorded": audit_event_recorded,
		})
		if result["decision"] == "deny" or (result["decision"] == "require_review" and not allow_review):
			reasons = ", ".join(action.get("reason", "help_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "help_policy_blocked")

	def _require_source(self, source_id: str, tenant_id: str) -> HelpSource:
		source = self._sources.get(source_id)
		if source is None or source.tenant_id != tenant_id:
			raise PermissionError("source_missing")
		return source

	def _require_article(self, article_id: str, tenant_id: str) -> HelpArticle:
		article = self._articles.get(article_id)
		if article is None or article.tenant_id != tenant_id:
			raise PermissionError("article_missing")
		return article

	def _require_answer(self, answer_id: str, tenant_id: str) -> HelpAnswer:
		answer = self._answers.get(answer_id)
		if answer is None or answer.tenant_id != tenant_id:
			raise PermissionError("answer_missing")
		return answer

	def _require_curation_item(self, curation_id: str, tenant_id: str) -> HelpCurationItem:
		item = self._curation.get(curation_id)
		if item is None or item.tenant_id != tenant_id:
			raise PermissionError("curation_item_missing")
		return item

	def _source_ids_are_approved(self, source_ids: list[str], tenant_id: str) -> bool:
		for source_id in source_ids:
			source = self._sources.get(source_id)
			if source is None or source.tenant_id != tenant_id or not source.approved:
				return False
		return True

	def _unsafe_answer_detected(self, answer_text: str) -> bool:
		lowered = answer_text.lower()
		return any(marker in lowered for marker in ["ignore policy", "share secret", "bypass access"])

	def _open_curation_item(self, tenant_id: str, article_id: str, reason: str) -> HelpCurationItem:
		for item in self._curation.values():
			if item.tenant_id == tenant_id and item.article_id == article_id and item.reason == reason and item.status == "open":
				return item
		item_id = f"curation-{article_id}-{next(self._counter)}"
		item = HelpCurationItem(id=item_id, tenant_id=tenant_id, article_id=article_id, reason=reason)
		self._curation[item_id] = item
		return item

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
	) -> dict[str, Any]:
		event_id = f"audit-{event_type}-{subject_id}-{next(self._counter)}"
		event = HelpAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
		)
		self._audit_events[event_id] = event
		return event.to_dict()

	def _tenant_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	# ── 14 new methods ──────────────────────────────────────────────────────────

	def article_create(
		self,
		title: str,
		content: str,
		category: str,
		author_id: str,
		tenant_id: str = "default",
		locale: str = "en",
	) -> dict[str, Any]:
		"""Create a help article with auto-generated ID."""
		article_id = f"article-{next(self._counter)}"
		return self.create_article(
			article_id=article_id,
			tenant_id=tenant_id,
			title=title,
			body=content,
			owner_id=author_id,
			topics=[category],
			locale=locale,
		)

	def article_publish(
		self,
		article_id: str,
		published_by: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Publish a help article."""
		return self.publish_article(
			article_id=article_id,
			tenant_id=tenant_id,
			approver_id=published_by,
			publication_approved=True,
		)

	def article_unpublish(
		self,
		article_id: str,
		reason: str,
		tenant_id: str = "default",
		actor: str = "admin",
	) -> dict[str, Any]:
		"""Unpublish (revert to draft) a help article."""
		article = self._require_article(article_id, tenant_id)
		article.status = ArticleStatus.DRAFT
		article.updated_at = utc_now_iso()
		self._record_event(tenant_id, "article_unpublished", article_id,
			f"Unpublished: {reason}", actor)
		return {**article.to_dict(), "unpublish_reason": reason}

	def article_rate(
		self,
		article_id: str,
		user_id: str,
		rating: int,
		feedback: str = "",
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Submit a user rating (1–5) for a help article."""
		assert 1 <= rating <= 5, "rating must be 1–5"
		article = self._require_article(article_id, tenant_id)
		fb_id = f"fb-rate-{article_id}-{next(self._counter)}"
		fb = HelpFeedback(
			id=fb_id,
			tenant_id=tenant_id,
			article_id=article_id,
			user_id=user_id,
			rating=rating,
			feedback_text=feedback,
		)
		self._feedback[fb_id] = fb
		self._record_event(tenant_id, "article_rated", article_id,
			f"Rating {rating}/5 by {user_id}", user_id)
		return fb.to_dict()

	def category_create(
		self,
		name: str,
		tenant_id: str = "default",
		parent_id: str | None = None,
		created_by: str = "admin",
	) -> dict[str, Any]:
		"""Create a help content category."""
		cat_id = f"cat-{next(self._counter)}"
		record: dict[str, Any] = {
			"id": cat_id,
			"tenant_id": tenant_id,
			"name": name,
			"parent_id": parent_id,
			"created_by": created_by,
			"created_at": utc_now_iso(),
		}
		self._record_event(tenant_id, "category_created", cat_id, f"Category: {name}", created_by)
		return record

	def tag_manage(
		self,
		action: str,
		tag_name: str,
		tenant_id: str = "default",
		actor: str = "admin",
	) -> dict[str, Any]:
		"""Add or remove a tag. action: 'add' | 'remove'."""
		assert action in {"add", "remove"}, "action must be 'add' or 'remove'"
		tag_id = f"tag-{tag_name.lower().replace(' ', '_')}"
		self._record_event(tenant_id, f"tag_{action}ed", tag_id, f"Tag {action}: {tag_name}", actor)
		return {"tag_id": tag_id, "tag_name": tag_name, "action": action, "tenant_id": tenant_id}

	def faq_create(
		self,
		question: str,
		answer: str,
		category: str,
		tenant_id: str = "default",
		author_id: str = "admin",
	) -> dict[str, Any]:
		"""Create a FAQ entry as a structured help article."""
		faq_id = f"faq-{next(self._counter)}"
		return self.create_article(
			article_id=faq_id,
			tenant_id=tenant_id,
			title=question,
			body=answer,
			owner_id=author_id,
			topics=[category, "faq"],
		)

	def chatbot_suggest(
		self,
		user_query: str,
		tenant_id: str = "default",
		max_results: int = 5,
	) -> list[str]:
		"""Return article IDs whose titles or topics match the user query.

		Simple keyword match — production delegates to Ollama RAG.
		"""
		query_tokens = set(user_query.lower().split())
		matches: list[tuple[int, str]] = []
		for article in self._articles.values():
			if article.tenant_id != tenant_id:
				continue
			score = sum(1 for tok in query_tokens if tok in article.title.lower())
			score += sum(1 for tok in query_tokens for topic in article.topics if tok in topic.lower())
			if score > 0:
				matches.append((score, article.id))
		matches.sort(key=lambda x: x[0], reverse=True)
		return [aid for _, aid in matches[:max_results]]

	def feedback_aggregate(
		self,
		article_id: str,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Aggregate feedback ratings for an article in a period."""
		feedbacks = [
			fb for fb in self._feedback.values()
			if fb.tenant_id == tenant_id and fb.article_id == article_id
		]
		if not feedbacks:
			return {"article_id": article_id, "period": period, "count": 0, "avg_rating": None}
		avg = round(sum(fb.rating for fb in feedbacks) / len(feedbacks), 2)
		by_rating: dict[int, int] = {}
		for fb in feedbacks:
			by_rating[fb.rating] = by_rating.get(fb.rating, 0) + 1
		return {
			"article_id": article_id,
			"period": period,
			"count": len(feedbacks),
			"avg_rating": avg,
			"by_rating": by_rating,
		}

	def version_history(
		self,
		article_id: str,
		tenant_id: str = "default",
	) -> list[dict[str, Any]]:
		"""Return audit events scoped to an article as its version history."""
		events = [
			e.to_dict() for e in self._audit_events.values()
			if e.tenant_id == tenant_id and e.subject_id == article_id
		]
		return sorted(events, key=lambda e: e.get("id", ""))

	def translation_request(
		self,
		article_id: str,
		target_lang: str,
		tenant_id: str = "default",
		requested_by: str = "admin",
	) -> dict[str, Any]:
		"""Queue a translation request for an article."""
		article = self._require_article(article_id, tenant_id)
		req_id = f"trans-{article_id}-{target_lang}-{next(self._counter)}"
		self._record_event(tenant_id, "translation_requested", req_id,
			f"Translate {article.title} to {target_lang}", requested_by)
		return {
			"request_id": req_id,
			"article_id": article_id,
			"source_locale": article.locale,
			"target_lang": target_lang,
			"status": "queued",
			"requested_by": requested_by,
		}

	def kb_export(
		self,
		format: str = "json",
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Export the full knowledge base for a tenant."""
		articles = self._list(self._articles, tenant_id)
		sources = self._list(self._sources, tenant_id)
		export_id = f"kb-export-{next(self._counter)}"
		self._record_event(tenant_id, "kb_exported", export_id,
			f"KB exported as {format}", "system")
		return {
			"export_id": export_id,
			"tenant_id": tenant_id,
			"format": format,
			"article_count": len(articles),
			"source_count": len(sources),
			"status": "ready",
			"download_ref": f"/exports/{tenant_id}/{export_id}.{format}",
		}

	def kb_stats(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Return knowledge-base statistics for a tenant."""
		articles = self._list(self._articles, tenant_id)
		published = sum(1 for a in articles if a.get("status") == "published")
		feedbacks = [fb for fb in self._feedback.values() if fb.tenant_id == tenant_id]
		avg_rating = round(sum(fb.rating for fb in feedbacks) / max(len(feedbacks), 1), 2) if feedbacks else None
		return {
			"tenant_id": tenant_id,
			"total_articles": len(articles),
			"published_articles": published,
			"draft_articles": len(articles) - published,
			"sources": len(self._list(self._sources, tenant_id)),
			"feedback_count": len(feedbacks),
			"avg_article_rating": avg_rating,
			"audit_events": len([e for e in self._audit_events.values() if e.tenant_id == tenant_id]),
		}

	def kb_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return knowledge-base analytics for a period."""
		articles = self._list(self._articles, tenant_id)
		answers = self._list(self._answers, tenant_id)
		feedbacks = [fb for fb in self._feedback.values() if fb.tenant_id == tenant_id]
		high_rated = [fb for fb in feedbacks if fb.rating >= 4]
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_articles": len(articles),
			"total_answers": len(answers),
			"total_feedback": len(feedbacks),
			"high_rated_pct": round(len(high_rated) / max(len(feedbacks), 1) * 100, 1),
			"curation_items": len(self._list(self._curation, tenant_id)),
			"localizations": len(self._list(self._localizations, tenant_id)),
		}

	def _list(self, store: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(store.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


# ── Extended methods injected directly onto HelpService ───────────────────────

def _article_create(
	self: "HelpService",
	article_id: str,
	tenant_id: str,
	title: str,
	body: str,
	owner_id: str,
	topics: list[str] | None = None,
	locale: str = "en",
	visibility: str = "internal",
	source_ids: list[str] | None = None,
) -> dict[str, Any]:
	"""Spec-name alias for create_article."""
	return self.create_article(article_id, tenant_id, title, body, owner_id, topics, locale, visibility, source_ids)

HelpService.article_create = _article_create  # type: ignore[attr-defined]


def _article_publish(
	self: "HelpService",
	article_id: str,
	tenant_id: str,
	approver_id: str,
) -> dict[str, Any]:
	"""Publish with publication_approved=True (spec alias)."""
	return self.publish_article(article_id, tenant_id, approver_id, publication_approved=True)

HelpService.article_publish = _article_publish  # type: ignore[attr-defined]


def _article_search(
	self: "HelpService",
	tenant_id: str,
	query: str,
	locale: str | None = None,
	limit: int = 10,
) -> list[dict[str, Any]]:
	"""Spec-name alias for search_articles."""
	return self.search_articles(tenant_id, query, locale=locale, limit=limit)

HelpService.article_search = _article_search  # type: ignore[attr-defined]


def _category_manage(
	self: "HelpService",
	tenant_id: str,
	category_id: str,
	name: str,
	parent_id: str | None = None,
	owner_id: str = "system",
	action: str = "upsert",
) -> dict[str, Any]:
	"""Create or update an article category record."""
	self._require_tenant(tenant_id)
	record: dict[str, Any] = {
		"id": category_id,
		"tenant_id": tenant_id,
		"name": name,
		"parent_id": parent_id,
		"owner_id": owner_id,
		"action": action,
		"updated_at": utc_now_iso(),
	}
	self._record_event(tenant_id, "category_managed", category_id, f"Category {action}: {name}", owner_id)
	return record

HelpService.category_manage = _category_manage  # type: ignore[attr-defined]


def _faq_create(
	self: "HelpService",
	faq_id: str,
	tenant_id: str,
	question: str,
	answer: str,
	owner_id: str,
	topics: list[str] | None = None,
) -> dict[str, Any]:
	"""Create an FAQ entry as an article."""
	return self.create_article(
		article_id=faq_id,
		tenant_id=tenant_id,
		title=question,
		body=answer,
		owner_id=owner_id,
		topics=["faq"] + list(topics or []),
	)

HelpService.faq_create = _faq_create  # type: ignore[attr-defined]


def _feedback_collect(
	self: "HelpService",
	feedback_id: str,
	tenant_id: str,
	user_id: str,
	rating: int,
	comment: str = "",
	article_id: str | None = None,
) -> dict[str, Any]:
	"""Spec alias for record_feedback."""
	return self.record_feedback(feedback_id, tenant_id, user_id, rating, comment, article_id=article_id)

HelpService.feedback_collect = _feedback_collect  # type: ignore[attr-defined]


def _related_suggest(
	self: "HelpService",
	tenant_id: str,
	article_id: str,
	limit: int = 5,
) -> list[dict[str, Any]]:
	"""Return articles sharing topics with the given article."""
	article = self._require_article(article_id, tenant_id)
	topic_set = set(article.topics)
	if not topic_set:
		return []
	scored: list[tuple[int, dict[str, Any]]] = []
	for a in self._articles.values():
		if a.id == article_id or a.tenant_id != tenant_id:
			continue
		overlap = len(topic_set & set(a.topics))
		if overlap:
			scored.append((overlap, a.to_dict()))
	scored.sort(key=lambda x: x[0], reverse=True)
	return [d for _, d in scored[:limit]]

HelpService.related_suggest = _related_suggest  # type: ignore[attr-defined]


def _version_history(
	self: "HelpService",
	tenant_id: str,
	article_id: str,
) -> list[dict[str, Any]]:
	"""Return audit events for an article as version history."""
	self._require_article(article_id, tenant_id)
	return [e for e in self.list_audit_events(tenant_id) if e.get("subject_id") == article_id]

HelpService.version_history = _version_history  # type: ignore[attr-defined]


def _translation_manage(
	self: "HelpService",
	localization_id: str,
	tenant_id: str,
	article_id: str,
	locale: str,
	title: str,
	body: str,
	translator_id: str,
) -> dict[str, Any]:
	"""Create or update a localisation (spec alias for localize_article)."""
	return self.localize_article(localization_id, tenant_id, article_id, locale, title, body, translator_id)

HelpService.translation_manage = _translation_manage  # type: ignore[attr-defined]


def _kb_export(
	self: "HelpService",
	tenant_id: str,
	format: str = "json",
) -> dict[str, Any]:
	"""Export the full knowledge base as a structured payload."""
	self._require_tenant(tenant_id)
	return {
		"tenant_id": tenant_id,
		"format": format,
		"exported_at": utc_now_iso(),
		"articles": self.list_articles(tenant_id),
		"sources": self.list_sources(tenant_id),
		"localizations": self.list_localizations(tenant_id),
		"feedback": self.list_feedback(tenant_id),
	}

HelpService.kb_export = _kb_export  # type: ignore[attr-defined]


def _article_analytics(
	self: "HelpService",
	tenant_id: str,
	article_id: str,
) -> dict[str, Any]:
	"""Feedback analytics for a specific article."""
	self._require_article(article_id, tenant_id)
	fb = [f for f in self._feedback.values() if f.tenant_id == tenant_id and f.article_id == article_id]
	ratings = [f.rating for f in fb]
	avg = sum(ratings) / len(ratings) if ratings else 0.0
	return {
		"article_id": article_id,
		"feedback_count": len(fb),
		"average_rating": round(avg, 2),
		"low_rating_count": sum(1 for r in ratings if r <= 2),
	}

HelpService.article_analytics = _article_analytics  # type: ignore[attr-defined]


def _chatbot_integrate(
	self: "HelpService",
	tenant_id: str,
	bot_id: str,
	webhook_url: str,
	owner_id: str,
) -> dict[str, Any]:
	"""Register a chatbot integration endpoint."""
	self._require_tenant(tenant_id)
	record: dict[str, Any] = {
		"id": bot_id,
		"tenant_id": tenant_id,
		"webhook_url": webhook_url,
		"owner_id": owner_id,
		"status": "active",
		"registered_at": utc_now_iso(),
	}
	self._record_event(tenant_id, "chatbot_integrated", bot_id, f"Chatbot {bot_id} registered", owner_id)
	return record

HelpService.chatbot_integrate = _chatbot_integrate  # type: ignore[attr-defined]


def _video_embed(
	self: "HelpService",
	tenant_id: str,
	article_id: str,
	video_url: str,
	caption: str,
	actor_id: str,
) -> dict[str, Any]:
	"""Attach a video embed tag to an article body."""
	article = self._require_article(article_id, tenant_id)
	embed_tag = f'\n\n[video:{video_url} caption="{caption}"]'
	article.body = (article.body or "") + embed_tag
	article.updated_at = utc_now_iso()
	self._record_event(tenant_id, "video_embedded", article_id, f"Video embedded in {article.title}", actor_id)
	return article.to_dict()

HelpService.video_embed = _video_embed  # type: ignore[attr-defined]


def _search_index_rebuild(
	self: "HelpService",
	tenant_id: str,
	actor_id: str = "system",
) -> dict[str, Any]:
	"""Force-rebuild the in-memory search index for the tenant."""
	self._require_tenant(tenant_id)
	articles = list(self._articles.values())
	self._search_index = HelpSearchIndex()
	self._record_event(tenant_id, "search_index_rebuilt", tenant_id, f"Rebuilt index over {len(articles)} articles", actor_id)
	return {"tenant_id": tenant_id, "indexed_articles": len(articles), "rebuilt_at": utc_now_iso()}

HelpService.search_index_rebuild = _search_index_rebuild  # type: ignore[attr-defined]


def _kb_analytics(self: "HelpService", tenant_id: str) -> dict[str, Any]:
	"""Aggregate knowledge-base analytics (alias for dashboard_summary)."""
	return self.dashboard_summary(tenant_id)

HelpService.kb_analytics = _kb_analytics  # type: ignore[attr-defined]


def _article_rate(
	self: "HelpService",
	feedback_id: str,
	tenant_id: str,
	user_id: str,
	article_id: str,
	rating: int,
	comment: str = "",
) -> dict[str, Any]:
	"""Rate an article — convenience wrapper over record_feedback."""
	return self.record_feedback(feedback_id, tenant_id, user_id, rating, comment, article_id=article_id)

HelpService.article_rate = _article_rate  # type: ignore[attr-defined]


def _article_search_advanced(
	self: "HelpService",
	tenant_id: str,
	query: str,
	locale: str | None = None,
	topics: list[str] | None = None,
	limit: int = 10,
) -> list[dict[str, Any]]:
	"""Advanced article search with topic filtering."""
	results = self.search_articles(tenant_id, query, locale=locale, limit=limit * 2)
	if topics:
		topic_set = set(topics)
		results = [r for r in results if topic_set & set(r["article"].get("topics", []))]
	return results[:limit]

HelpService.article_search_advanced = _article_search_advanced  # type: ignore[attr-defined]


def _tag_manage(
	self: "HelpService",
	tenant_id: str,
	tag_name: str,
	action: str = "upsert",
	owner_id: str = "system",
) -> dict[str, Any]:
	"""Manage article tags — create, update, or delete a tag."""
	self._require_tenant(tenant_id)
	tag_id = f"tag:{tenant_id}:{tag_name.lower().replace(' ', '_')}"
	record: dict[str, Any] = {
		"id": tag_id,
		"tenant_id": tenant_id,
		"name": tag_name,
		"action": action,
		"owner_id": owner_id,
		"updated_at": utc_now_iso(),
	}
	self._record_event(tenant_id, "tag_managed", tag_id, f"Tag {action}: {tag_name}", owner_id)
	return record

HelpService.tag_manage = _tag_manage  # type: ignore[attr-defined]


def _faq_bulk_create(
	self: "HelpService",
	tenant_id: str,
	faqs: list[dict[str, Any]],
	owner_id: str,
) -> dict[str, Any]:
	"""Bulk-create FAQ entries from a list of {question, answer} dicts."""
	self._require_tenant(tenant_id)
	created = []
	failed = []
	for idx, faq in enumerate(faqs):
		faq_id = f"faq-bulk-{tenant_id}-{idx}"
		try:
			result = self.create_article(
				article_id=faq_id,
				tenant_id=tenant_id,
				title=str(faq.get("question", f"FAQ {idx}")),
				body=str(faq.get("answer", "")),
				owner_id=owner_id,
				topics=["faq"] + list(faq.get("topics", [])),
			)
			created.append(result["id"])
		except Exception as exc:
			failed.append({"index": idx, "error": str(exc)})
	return {"created_count": len(created), "failed_count": len(failed), "created_ids": created, "failures": failed}

HelpService.faq_bulk_create = _faq_bulk_create  # type: ignore[attr-defined]


def _feedback_analysis(
	self: "HelpService",
	tenant_id: str,
) -> dict[str, Any]:
	"""Aggregate feedback analytics across all articles for a tenant."""
	self._require_tenant(tenant_id)
	fb_items = self.list_feedback(tenant_id)
	total = len(fb_items)
	ratings = [f["rating"] for f in fb_items]
	avg = round(sum(ratings) / max(total, 1), 2)
	by_rating: dict[int, int] = {}
	for r in ratings:
		by_rating[r] = by_rating.get(r, 0) + 1
	requires_review = sum(1 for f in fb_items if f.get("requires_review"))
	return {"tenant_id": tenant_id, "total_feedback": total, "average_rating": avg, "by_rating": by_rating, "requires_review": requires_review, "generated_at": utc_now_iso()}

HelpService.feedback_analysis = _feedback_analysis  # type: ignore[attr-defined]


def _kb_health_check(
	self: "HelpService",
	tenant_id: str,
) -> dict[str, Any]:
	"""Run a knowledge-base health check: stale content, empty articles, no sources."""
	self._require_tenant(tenant_id)
	articles = self.list_articles(tenant_id)
	stale = self.freshness_queue(tenant_id)
	empty = [a for a in articles if not a.get("body", "").strip()]
	no_source = [a for a in articles if not a.get("source_ids")]
	return {"tenant_id": tenant_id, "total_articles": len(articles), "stale_articles": len(stale), "empty_articles": len(empty), "articles_without_sources": len(no_source), "health": "healthy" if len(stale) == 0 and len(empty) == 0 else "needs_attention", "checked_at": utc_now_iso()}

HelpService.kb_health_check = _kb_health_check  # type: ignore[attr-defined]


def _search_analytics(
	self: "HelpService",
	tenant_id: str,
) -> dict[str, Any]:
	"""Return search analytics: audit events for search operations."""
	self._require_tenant(tenant_id)
	events = self.list_audit_events(tenant_id)
	search_events = [e for e in events if e.get("event_type") == "answer_generated"]
	return {"tenant_id": tenant_id, "total_searches": len(search_events), "queries_answered": len([e for e in search_events if "generated" in e.get("event_type", "")]), "generated_at": utc_now_iso()}

HelpService.search_analytics = _search_analytics  # type: ignore[attr-defined]


def _chatbot_handoff(
	self: "HelpService",
	tenant_id: str,
	session_id: str,
	reason: str,
	agent_id: str = "human-support",
) -> dict[str, Any]:
	"""Hand off a chatbot session to a human agent."""
	self._require_tenant(tenant_id)
	handoff_id = f"handoff-{session_id}"
	record: dict[str, Any] = {
		"id": handoff_id,
		"tenant_id": tenant_id,
		"session_id": session_id,
		"reason": reason,
		"assigned_agent_id": agent_id,
		"status": "handed_off",
		"handed_off_at": utc_now_iso(),
	}
	self._record_event(tenant_id, "chatbot_handoff", handoff_id, f"Session {session_id} handed off: {reason}", agent_id)
	return record

HelpService.chatbot_handoff = _chatbot_handoff  # type: ignore[attr-defined]
