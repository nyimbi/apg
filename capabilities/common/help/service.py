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

	def _list(self, store: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(store.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
