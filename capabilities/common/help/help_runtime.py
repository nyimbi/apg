"""Domain search, answer, and curation helpers for the HELP capability."""

from __future__ import annotations

from datetime import datetime, timezone

from .models import ArticleStatus, ContentVisibility, HelpArticle, HelpCitation


class HelpSearchIndex:
	"""Deterministic in-process article search for generated APG apps."""

	def search(
		self,
		query: str,
		articles: list[HelpArticle],
		tenant_id: str,
		locale: str | None = None,
		include_restricted: bool = False,
		limit: int = 5,
	) -> list[dict[str, object]]:
		query_terms = self._terms(query)
		hits: list[dict[str, object]] = []
		for article in articles:
			if article.tenant_id != tenant_id or article.status != ArticleStatus.PUBLISHED:
				continue
			if locale and article.locale != locale:
				continue
			if article.visibility == ContentVisibility.RESTRICTED and not include_restricted:
				continue
			haystack = self._terms(" ".join([article.title, article.body, " ".join(article.topics)]))
			score = len(query_terms & haystack)
			if score == 0 and query_terms:
				continue
			hits.append({
				"article": article,
				"score": score or 1,
				"snippet": self._snippet(article.body, query_terms),
			})
		return sorted(hits, key=lambda hit: (-int(hit["score"]), hit["article"].title))[:limit]

	def _terms(self, value: str) -> set[str]:
		return {
			part.strip(".,;:!?()[]{}\"'").lower()
			for part in value.split()
			if part.strip(".,;:!?()[]{}\"'")
		}

	def _snippet(self, body: str, query_terms: set[str]) -> str:
		sentences = [sentence.strip() for sentence in body.replace("\n", " ").split(".") if sentence.strip()]
		for sentence in sentences:
			if self._terms(sentence) & query_terms:
				return sentence[:220]
		return body[:220]


class HelpAnswerComposer:
	"""Compose cited answers from approved search hits."""

	def compose(
		self,
		query: str,
		hits: list[dict[str, object]],
		minimum_confidence: float,
	) -> tuple[str, float, list[HelpCitation], str | None]:
		if not hits:
			return "", 0.0, [], "no_approved_sources"
		best_score = max(int(hit["score"]) for hit in hits)
		confidence = min(0.95, 0.55 + (best_score * 0.14))
		citations = [
			HelpCitation(
				article_id=hit["article"].id,
				title=hit["article"].title,
				excerpt=str(hit["snippet"]),
			)
			for hit in hits[:3]
		]
		if confidence < minimum_confidence:
			return "", confidence, citations, "confidence_below_threshold"
		answer = f"{citations[0].excerpt}. See {citations[0].title} for the approved guidance."
		return answer, confidence, citations, None


class HelpFreshnessInspector:
	"""Find published articles that need curation review."""

	def stale_articles(
		self,
		articles: list[HelpArticle],
		tenant_id: str,
		threshold_days: int,
		now: datetime | None = None,
	) -> list[HelpArticle]:
		now = now or datetime.now(timezone.utc)
		stale: list[HelpArticle] = []
		for article in articles:
			if article.tenant_id != tenant_id or article.status != ArticleStatus.PUBLISHED:
				continue
			reviewed_at = article.last_reviewed_at or article.published_at or article.created_at
			if self._age_days(reviewed_at, now) > threshold_days:
				stale.append(article)
		return sorted(stale, key=lambda item: item.title)

	def _age_days(self, iso_timestamp: str, now: datetime) -> int:
		try:
			parsed = datetime.fromisoformat(iso_timestamp)
		except ValueError:
			return 0
		if parsed.tzinfo is None:
			parsed = parsed.replace(tzinfo=timezone.utc)
		return (now - parsed).days
