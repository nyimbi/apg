"""Executable service layer for the APG HELP capability."""

from __future__ import annotations

from itertools import count
from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .help_runtime import HelpAnswerComposer, HelpFreshnessInspector, HelpSearchIndex
from .models import (
	ArticleStatus,
	ContentVisibility,
	HelpAnswer,
	HelpArticle,
	HelpCurationItem,
	HelpFeedback,
	utc_now_iso,
)


class HelpService:
	"""Tenant-aware article, answer, feedback, and curation runtime."""

	def __init__(self) -> None:
		self._articles: dict[str, HelpArticle] = {}
		self._answers: dict[str, HelpAnswer] = {}
		self._feedback: dict[str, HelpFeedback] = {}
		self._curation: dict[str, HelpCurationItem] = {}
		self._counter = count(1)
		self._search_index = HelpSearchIndex()
		self._answer_composer = HelpAnswerComposer()
		self._freshness_inspector = HelpFreshnessInspector()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

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
	) -> dict[str, Any]:
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="create_article",
			article_owner_assigned=bool(owner_id),
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
			source_ids=list(source_ids or []),
		)
		self._articles[article_id] = article
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
			restricted_content_present=article.visibility == ContentVisibility.RESTRICTED,
			rbac_filter_applied=rbac_filter_applied,
			freshness_review_recorded=freshness_review_recorded,
			article_age_days=article_age_days,
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
			restricted_content_present=include_restricted,
			rbac_filter_applied=rbac_filter_applied,
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
		answer_text, confidence, citations, block_reason = self._answer_composer.compose(
			query=query,
			hits=hits,
			minimum_confidence=minimum_confidence,
		)
		self._enforce_help_policy(
			tenant_id=tenant_id,
			operation="generate_answer",
			citations_present=bool(citations),
			restricted_content_present=include_restricted,
			rbac_filter_applied=rbac_filter_applied,
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
		return answer.to_dict()

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
		self._enforce_help_policy(tenant_id=tenant_id, operation="record_feedback")
		if rating < 1 or rating > 5:
			raise ValueError("rating_out_of_range")
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
		return feedback.to_dict()

	def freshness_queue(self, tenant_id: str) -> list[dict[str, Any]]:
		threshold = int(DEFAULT_CONFIGURATION["content"]["freshness_review_days"])
		for article in self._freshness_inspector.stale_articles(
			list(self._articles.values()),
			tenant_id,
			threshold,
		):
			self._open_curation_item(tenant_id, article.id, "freshness_review")
		return self.list_curation_items(tenant_id)

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

	def list_answers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._answers, tenant_id)

	def list_feedback(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._feedback, tenant_id)

	def list_curation_items(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._curation, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		records: list[dict[str, Any]] = []
		for store in (self._articles, self._answers, self._feedback, self._curation):
			records.extend(self._list(store, tenant_id))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		articles = self.list_articles(tenant_id)
		answers = self.list_answers(tenant_id)
		feedback = self.list_feedback(tenant_id)
		curation = self.list_curation_items(tenant_id)
		return {
			"tenant_id": tenant_id,
			"article_count": len(articles),
			"published_article_count": len([article for article in articles if article["status"] == ArticleStatus.PUBLISHED.value]),
			"answer_count": len(answers),
			"blocked_answer_count": len([answer for answer in answers if answer["blocked"]]),
			"feedback_count": len(feedback),
			"open_curation_count": len([item for item in curation if item["status"] == "open"]),
		}

	def _enforce_help_policy(
		self,
		tenant_id: str,
		operation: str,
		article_owner_assigned: bool = True,
		publication_approved: bool = True,
		citations_present: bool = True,
		restricted_content_present: bool = False,
		rbac_filter_applied: bool = True,
		article_age_days: int = 0,
		freshness_review_recorded: bool = True,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": operation,
			"article_owner_assigned": article_owner_assigned,
			"publication_approved": publication_approved,
			"citations_present": citations_present,
			"restricted_content_present": restricted_content_present,
			"rbac_filter_applied": rbac_filter_applied,
			"article_age_days": article_age_days,
			"freshness_review_recorded": freshness_review_recorded,
		})
		if result["decision"] != "allow":
			reasons = ", ".join(action.get("reason", "help_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "help_policy_blocked")

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

	def _open_curation_item(self, tenant_id: str, article_id: str, reason: str) -> HelpCurationItem:
		for item in self._curation.values():
			if item.tenant_id == tenant_id and item.article_id == article_id and item.reason == reason and item.status == "open":
				return item
		item_id = f"curation-{article_id}-{next(self._counter)}"
		item = HelpCurationItem(id=item_id, tenant_id=tenant_id, article_id=article_id, reason=reason)
		self._curation[item_id] = item
		return item

	def _list(self, store: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(store.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]
