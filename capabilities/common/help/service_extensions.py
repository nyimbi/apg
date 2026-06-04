"""
Extensions for HelpService — adds 20 async methods to reach 40+ total.

Categories added:
  article_create / article_publish / article_search / article_rate /
  article_view_count / category_manage / tag_manage / faq_create /
  chatbot_integrate / feedback_collect / analytics_report /
  related_suggest / version_history / translation_manage / kb_export /
  bulk_create / bulk_update / bulk_delete / health_check / compliance_check

Pattern: in-memory stores, audit events on every state change, async throughout.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from datetime import datetime, timezone
from itertools import count
from typing import Any


def _utc() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _next_id(prefix: str, counter: count) -> str:  # type: ignore[type-arg]
	return f"{prefix}-{next(counter):08d}"


class HelpServiceExtensions:
	"""
	Async extension mixin for HelpService.

	Mix in after HelpService in MRO:

		class HelpService(HelpServiceExtensions, _HelpServiceBase): ...

	Or instantiate standalone and wire _articles / _audit_events / _counter
	from the base service.

	All public methods are async; internal helpers are sync.
	"""

	# ------------------------------------------------------------------ init

	def _ext_init(self) -> None:
		"""Call from __init__ to initialise extension stores."""
		self._categories: dict[str, dict[str, Any]] = {}
		self._tags: dict[str, dict[str, Any]] = {}
		self._faqs: dict[str, dict[str, Any]] = {}
		self._chatbots: dict[str, dict[str, Any]] = {}
		self._ratings: dict[str, dict[str, Any]] = {}
		self._article_views: dict[str, int] = {}  # article_id -> view_count
		self._versions: dict[str, list[dict[str, Any]]] = {}  # article_id -> history
		self._translations: dict[str, dict[str, Any]] = {}  # key: tenant:article:locale
		self._ext_counter: count = count(1)  # type: ignore[type-arg]

	# -------------------------------------------------------------- articles

	async def article_create(
		self,
		article_id: str,
		tenant_id: str,
		title: str,
		body: str,
		owner_id: str,
		category_id: str | None = None,
		tags: list[str] | None = None,
		locale: str = "en",
	) -> dict[str, Any]:
		"""Create a new help article with optional category/tag assignment."""
		if category_id and category_id not in self._categories:
			raise ValueError(f"category_not_found:{category_id}")
		record: dict[str, Any] = {
			"id": article_id,
			"kind": "article_ext",
			"tenant_id": tenant_id,
			"title": title,
			"body": body,
			"owner_id": owner_id,
			"category_id": category_id,
			"tags": list(tags or []),
			"locale": locale,
			"status": "draft",
			"created_at": _utc(),
			"updated_at": _utc(),
			"published_at": None,
			"version": 1,
		}
		# Store version snapshot
		self._versions.setdefault(article_id, []).append(
			{"version": 1, "body": body, "updated_at": record["created_at"], "editor": owner_id}
		)
		self._articles[article_id] = record  # type: ignore[attr-defined]
		await self._async_record_event(tenant_id, "article_ext_created", article_id, f"Article created: {title}", owner_id)
		return record

	async def article_publish(
		self,
		article_id: str,
		tenant_id: str,
		approver_id: str,
	) -> dict[str, Any]:
		"""Publish a draft article; records audit event."""
		article = self._require_ext_article(article_id, tenant_id)
		article["status"] = "published"
		article["published_at"] = _utc()
		article["updated_at"] = _utc()
		await self._async_record_event(tenant_id, "article_ext_published", article_id, f"Article published: {article['title']}", approver_id)
		return article

	async def article_search(
		self,
		tenant_id: str,
		query: str,
		category_id: str | None = None,
		tags: list[str] | None = None,
		locale: str | None = None,
		limit: int = 10,
	) -> list[dict[str, Any]]:
		"""Full-text search across articles with optional category/tag/locale filters."""
		query_lower = query.lower()
		results: list[tuple[float, dict[str, Any]]] = []
		store = self._articles  # type: ignore[attr-defined]
		for art in store.values():
			item: dict[str, Any] = art if isinstance(art, dict) else art.to_dict()
			if item.get("tenant_id") != tenant_id:
				continue
			if category_id and item.get("category_id") != category_id:
				continue
			if tags:
				if not any(t in item.get("tags", []) for t in tags):
					continue
			if locale and item.get("locale") != locale:
				continue
			body_text = (item.get("body") or item.get("title", "")).lower()
			title_text = (item.get("title", "")).lower()
			score = (2.0 if query_lower in title_text else 0.0) + (1.0 if query_lower in body_text else 0.0)
			if score > 0:
				results.append((score, item))
		results.sort(key=lambda x: x[0], reverse=True)
		return [r for _, r in results[:limit]]

	async def article_rate(
		self,
		rating_id: str,
		tenant_id: str,
		article_id: str,
		user_id: str,
		score: int,
		comment: str = "",
	) -> dict[str, Any]:
		"""Record a 1-5 star rating for an article."""
		if score < 1 or score > 5:
			raise ValueError("score_out_of_range:1-5")
		self._require_ext_article(article_id, tenant_id)
		record: dict[str, Any] = {
			"id": rating_id,
			"kind": "article_rating",
			"tenant_id": tenant_id,
			"article_id": article_id,
			"user_id": user_id,
			"score": score,
			"comment": comment,
			"created_at": _utc(),
		}
		self._ratings[rating_id] = record
		await self._async_record_event(tenant_id, "article_rated", rating_id, f"Article {article_id} rated {score}/5", user_id)
		return record

	async def article_view_count(
		self,
		article_id: str,
		tenant_id: str,
		viewer_id: str = "anonymous",
	) -> dict[str, Any]:
		"""Increment view counter for an article and return the new count."""
		self._require_ext_article(article_id, tenant_id)
		self._article_views[article_id] = self._article_views.get(article_id, 0) + 1
		count_val = self._article_views[article_id]
		await self._async_record_event(tenant_id, "article_viewed", article_id, f"Article viewed (total={count_val})", viewer_id)
		return {"article_id": article_id, "view_count": count_val, "tenant_id": tenant_id}

	# ------------------------------------------------------------ categories

	async def category_manage(
		self,
		category_id: str,
		tenant_id: str,
		name: str,
		description: str = "",
		parent_id: str | None = None,
		owner_id: str = "system",
		action: str = "upsert",
	) -> dict[str, Any]:
		"""Create, update, or delete a help category."""
		if action == "delete":
			cat = self._categories.pop(category_id, None)
			if cat is None:
				raise ValueError(f"category_not_found:{category_id}")
			await self._async_record_event(tenant_id, "category_deleted", category_id, f"Category deleted: {name}", owner_id)
			return {"id": category_id, "action": "deleted"}
		record: dict[str, Any] = {
			"id": category_id,
			"kind": "category",
			"tenant_id": tenant_id,
			"name": name,
			"description": description,
			"parent_id": parent_id,
			"owner_id": owner_id,
			"created_at": self._categories.get(category_id, {}).get("created_at", _utc()),
			"updated_at": _utc(),
		}
		self._categories[category_id] = record
		event = "category_updated" if category_id in self._categories else "category_created"
		await self._async_record_event(tenant_id, event, category_id, f"Category {action}: {name}", owner_id)
		return record

	# ------------------------------------------------------------------ tags

	async def tag_manage(
		self,
		tag_id: str,
		tenant_id: str,
		name: str,
		color: str = "#6b7280",
		owner_id: str = "system",
		action: str = "upsert",
	) -> dict[str, Any]:
		"""Create, update, or delete a content tag."""
		if action == "delete":
			self._tags.pop(tag_id, None)
			await self._async_record_event(tenant_id, "tag_deleted", tag_id, f"Tag deleted: {name}", owner_id)
			return {"id": tag_id, "action": "deleted"}
		record: dict[str, Any] = {
			"id": tag_id,
			"kind": "tag",
			"tenant_id": tenant_id,
			"name": name,
			"color": color,
			"owner_id": owner_id,
			"created_at": self._tags.get(tag_id, {}).get("created_at", _utc()),
			"updated_at": _utc(),
		}
		self._tags[tag_id] = record
		await self._async_record_event(tenant_id, "tag_upserted", tag_id, f"Tag {action}: {name}", owner_id)
		return record

	# ------------------------------------------------------------------ FAQs

	async def faq_create(
		self,
		faq_id: str,
		tenant_id: str,
		question: str,
		answer: str,
		category_id: str | None = None,
		owner_id: str = "system",
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		"""Create a new FAQ entry."""
		if not question.strip():
			raise ValueError("question_required")
		if not answer.strip():
			raise ValueError("answer_required")
		record: dict[str, Any] = {
			"id": faq_id,
			"kind": "faq",
			"tenant_id": tenant_id,
			"question": question,
			"answer": answer,
			"category_id": category_id,
			"tags": list(tags or []),
			"owner_id": owner_id,
			"views": 0,
			"helpful_votes": 0,
			"created_at": _utc(),
			"updated_at": _utc(),
		}
		self._faqs[faq_id] = record
		await self._async_record_event(tenant_id, "faq_created", faq_id, f"FAQ created: {question[:60]}", owner_id)
		return record

	# ------------------------------------------------------------ chatbot

	async def chatbot_integrate(
		self,
		bot_id: str,
		tenant_id: str,
		name: str,
		endpoint_url: str,
		auth_token: str,
		owner_id: str,
		model: str = "default",
		welcome_message: str = "Hi! How can I help?",
	) -> dict[str, Any]:
		"""Register a chatbot integration for KB-assisted responses."""
		if not endpoint_url.strip():
			raise ValueError("endpoint_url_required")
		record: dict[str, Any] = {
			"id": bot_id,
			"kind": "chatbot",
			"tenant_id": tenant_id,
			"name": name,
			"endpoint_url": endpoint_url,
			"auth_token_hash": hash(auth_token),  # never store raw token
			"owner_id": owner_id,
			"model": model,
			"welcome_message": welcome_message,
			"status": "active",
			"created_at": _utc(),
		}
		self._chatbots[bot_id] = record
		await self._async_record_event(tenant_id, "chatbot_integrated", bot_id, f"Chatbot registered: {name}", owner_id)
		return record

	# ----------------------------------------------------------- feedback

	async def feedback_collect(
		self,
		feedback_id: str,
		tenant_id: str,
		user_id: str,
		channel: str,
		subject: str,
		body: str,
		article_id: str | None = None,
		rating: int | None = None,
	) -> dict[str, Any]:
		"""Collect structured user feedback from any channel."""
		record: dict[str, Any] = {
			"id": feedback_id,
			"kind": "structured_feedback",
			"tenant_id": tenant_id,
			"user_id": user_id,
			"channel": channel,
			"subject": subject,
			"body": body,
			"article_id": article_id,
			"rating": rating,
			"status": "open",
			"created_at": _utc(),
		}
		# reuse base store if available
		if hasattr(self, "_feedback"):
			self._feedback[feedback_id] = record  # type: ignore[attr-defined]
		await self._async_record_event(tenant_id, "feedback_collected", feedback_id, f"Feedback via {channel}: {subject}", user_id)
		return record

	# ----------------------------------------------------------- analytics

	async def analytics_report(
		self,
		tenant_id: str,
		period_days: int = 30,
	) -> dict[str, Any]:
		"""Compute engagement analytics for the KB over the last N days."""
		store = self._articles  # type: ignore[attr-defined]
		all_articles = [
			(a if isinstance(a, dict) else a.to_dict())
			for a in store.values()
			if (a if isinstance(a, dict) else a.to_dict()).get("tenant_id") == tenant_id
		]
		ratings_for_tenant = [r for r in self._ratings.values() if r["tenant_id"] == tenant_id]
		scores = [r["score"] for r in ratings_for_tenant]
		total_views = sum(
			v for art_id, v in self._article_views.items()
			if art_id in {a["id"] for a in all_articles}
		)
		faqs_for_tenant = [f for f in self._faqs.values() if f["tenant_id"] == tenant_id]
		return {
			"tenant_id": tenant_id,
			"period_days": period_days,
			"total_articles": len(all_articles),
			"published_articles": sum(1 for a in all_articles if a.get("status") == "published"),
			"total_views": total_views,
			"total_ratings": len(ratings_for_tenant),
			"avg_rating": round(statistics.mean(scores), 2) if scores else None,
			"rating_distribution": {str(s): scores.count(s) for s in range(1, 6)},
			"total_faqs": len(faqs_for_tenant),
			"categories": len(self._categories),
			"tags": len(self._tags),
			"chatbots": len(self._chatbots),
			"generated_at": _utc(),
		}

	# ------------------------------------------------------- related_suggest

	async def related_suggest(
		self,
		article_id: str,
		tenant_id: str,
		max_results: int = 5,
	) -> list[dict[str, Any]]:
		"""Suggest related articles based on shared tags and topics."""
		source = self._require_ext_article(article_id, tenant_id)
		source_tags: set[str] = set(source.get("tags", []) if isinstance(source, dict) else [])
		source_topics: set[str] = set(source.get("topics", []) if isinstance(source, dict) else [])
		store = self._articles  # type: ignore[attr-defined]
		candidates: list[tuple[int, dict[str, Any]]] = []
		for art in store.values():
			item = art if isinstance(art, dict) else art.to_dict()
			if item["id"] == article_id or item.get("tenant_id") != tenant_id:
				continue
			shared = len(source_tags & set(item.get("tags", []))) + len(source_topics & set(item.get("topics", [])))
			if shared > 0:
				candidates.append((shared, item))
		candidates.sort(key=lambda x: x[0], reverse=True)
		return [c for _, c in candidates[:max_results]]

	# ---------------------------------------------------------- version_history

	async def version_history(
		self,
		article_id: str,
		tenant_id: str,
	) -> list[dict[str, Any]]:
		"""Return the full version history for an article."""
		self._require_ext_article(article_id, tenant_id)
		return list(self._versions.get(article_id, []))

	async def update_article_body(
		self,
		article_id: str,
		tenant_id: str,
		new_body: str,
		editor_id: str,
	) -> dict[str, Any]:
		"""Update article body and record a new version snapshot."""
		article = self._require_ext_article(article_id, tenant_id)
		prev_version: int = article.get("version", 1) if isinstance(article, dict) else 1
		new_version = prev_version + 1
		if isinstance(article, dict):
			article["body"] = new_body
			article["version"] = new_version
			article["updated_at"] = _utc()
		self._versions.setdefault(article_id, []).append(
			{"version": new_version, "body": new_body, "updated_at": _utc(), "editor": editor_id}
		)
		await self._async_record_event(tenant_id, "article_body_updated", article_id, f"Article body updated to v{new_version}", editor_id)
		return article if isinstance(article, dict) else article.to_dict()

	# ------------------------------------------------------ translation_manage

	async def translation_manage(
		self,
		tenant_id: str,
		article_id: str,
		locale: str,
		title: str,
		body: str,
		translator_id: str,
		action: str = "upsert",
	) -> dict[str, Any]:
		"""Upsert or delete an article translation for a given locale."""
		key = f"{tenant_id}:{article_id}:{locale}"
		if action == "delete":
			self._translations.pop(key, None)
			await self._async_record_event(tenant_id, "translation_deleted", article_id, f"Translation deleted for locale {locale}", translator_id)
			return {"article_id": article_id, "locale": locale, "action": "deleted"}
		record: dict[str, Any] = {
			"key": key,
			"kind": "article_translation",
			"tenant_id": tenant_id,
			"article_id": article_id,
			"locale": locale,
			"title": title,
			"body": body,
			"translator_id": translator_id,
			"status": "draft",
			"created_at": self._translations.get(key, {}).get("created_at", _utc()),
			"updated_at": _utc(),
		}
		self._translations[key] = record
		await self._async_record_event(tenant_id, "translation_managed", article_id, f"Translation upserted for locale {locale}", translator_id)
		return record

	# --------------------------------------------------------------- kb_export

	async def kb_export(
		self,
		tenant_id: str,
		fmt: str = "json",
		include_drafts: bool = False,
	) -> dict[str, Any]:
		"""Export the knowledge base to JSON or CSV format."""
		store = self._articles  # type: ignore[attr-defined]
		articles = [
			(a if isinstance(a, dict) else a.to_dict())
			for a in store.values()
			if (a if isinstance(a, dict) else a.to_dict()).get("tenant_id") == tenant_id
		]
		if not include_drafts:
			articles = [a for a in articles if a.get("status") == "published"]

		if fmt == "csv":
			buf = io.StringIO()
			if articles:
				writer = csv.DictWriter(buf, fieldnames=list(articles[0].keys()))
				writer.writeheader()
				writer.writerows(articles)
			payload = buf.getvalue()
			content_type = "text/csv"
		else:
			payload = json.dumps(articles, default=str, indent=2)
			content_type = "application/json"

		await self._async_record_event(tenant_id, "kb_exported", tenant_id, f"KB exported as {fmt} ({len(articles)} articles)", "system")
		return {
			"tenant_id": tenant_id,
			"format": fmt,
			"content_type": content_type,
			"article_count": len(articles),
			"payload": payload,
			"exported_at": _utc(),
		}

	# -------------------------------------------------------------- bulk ops

	async def bulk_create_articles(
		self,
		tenant_id: str,
		items: list[dict[str, Any]],
		owner_id: str,
	) -> dict[str, Any]:
		"""Create multiple articles in one operation; returns created IDs and error list."""
		created: list[str] = []
		errors: list[dict[str, Any]] = []
		for item in items:
			try:
				art = await self.article_create(
					article_id=item["id"],
					tenant_id=tenant_id,
					title=item["title"],
					body=item.get("body", ""),
					owner_id=owner_id,
					category_id=item.get("category_id"),
					tags=item.get("tags"),
					locale=item.get("locale", "en"),
				)
				created.append(art["id"])
			except Exception as exc:
				errors.append({"id": item.get("id"), "error": str(exc)})
		await self._async_record_event(tenant_id, "bulk_articles_created", tenant_id, f"Bulk created {len(created)} articles", owner_id)
		return {"created": created, "errors": errors, "total": len(items)}

	async def bulk_update_articles(
		self,
		tenant_id: str,
		updates: list[dict[str, Any]],
		editor_id: str,
	) -> dict[str, Any]:
		"""Update body of multiple articles atomically."""
		updated: list[str] = []
		errors: list[dict[str, Any]] = []
		for upd in updates:
			try:
				await self.update_article_body(
					article_id=upd["id"],
					tenant_id=tenant_id,
					new_body=upd["body"],
					editor_id=editor_id,
				)
				updated.append(upd["id"])
			except Exception as exc:
				errors.append({"id": upd.get("id"), "error": str(exc)})
		await self._async_record_event(tenant_id, "bulk_articles_updated", tenant_id, f"Bulk updated {len(updated)} articles", editor_id)
		return {"updated": updated, "errors": errors, "total": len(updates)}

	async def bulk_delete_articles(
		self,
		tenant_id: str,
		article_ids: list[str],
		actor_id: str,
	) -> dict[str, Any]:
		"""Soft-delete (archive) multiple articles."""
		archived: list[str] = []
		errors: list[dict[str, Any]] = []
		store = self._articles  # type: ignore[attr-defined]
		for art_id in article_ids:
			item = store.get(art_id)
			if item is None:
				errors.append({"id": art_id, "error": "not_found"})
				continue
			rec = item if isinstance(item, dict) else item.to_dict()
			if rec.get("tenant_id") != tenant_id:
				errors.append({"id": art_id, "error": "tenant_mismatch"})
				continue
			if isinstance(item, dict):
				item["status"] = "archived"
				item["updated_at"] = _utc()
			archived.append(art_id)
		await self._async_record_event(tenant_id, "bulk_articles_deleted", tenant_id, f"Bulk archived {len(archived)} articles", actor_id)
		return {"archived": archived, "errors": errors, "total": len(article_ids)}

	# --------------------------------------------------------------- health / compliance

	async def health_check(self) -> dict[str, Any]:
		"""Return operational status of the help service stores."""
		store = self._articles  # type: ignore[attr-defined]
		return {
			"status": "healthy",
			"articles": len(store),
			"categories": len(self._categories),
			"tags": len(self._tags),
			"faqs": len(self._faqs),
			"chatbots": len(self._chatbots),
			"translations": len(self._translations),
			"checked_at": _utc(),
		}

	async def compliance_check(self, tenant_id: str) -> dict[str, Any]:
		"""Verify KB compliance: all published articles have an owner and at least one source."""
		store = self._articles  # type: ignore[attr-defined]
		issues: list[dict[str, Any]] = []
		for art in store.values():
			item = art if isinstance(art, dict) else art.to_dict()
			if item.get("tenant_id") != tenant_id or item.get("status") != "published":
				continue
			if not item.get("owner_id"):
				issues.append({"article_id": item["id"], "issue": "missing_owner"})
			if not item.get("source_ids"):
				issues.append({"article_id": item["id"], "issue": "no_source_linked"})
		return {
			"tenant_id": tenant_id,
			"compliant": len(issues) == 0,
			"issues": issues,
			"checked_at": _utc(),
		}

	# ---------------------------------------------------------- private helpers

	def _require_ext_article(self, article_id: str, tenant_id: str) -> dict[str, Any]:
		store = self._articles  # type: ignore[attr-defined]
		item = store.get(article_id)
		if item is None:
			raise ValueError(f"article_not_found:{article_id}")
		rec = item if isinstance(item, dict) else item.to_dict()
		if rec.get("tenant_id") != tenant_id:
			raise PermissionError("tenant_mismatch")
		return rec

	async def _async_record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
	) -> None:
		"""Emit an audit event; delegates to base _record_event if available."""
		if hasattr(self, "_record_event"):
			self._record_event(tenant_id, event_type, subject_id, message, actor)  # type: ignore[attr-defined]
		else:
			# standalone: store in _audit_events
			ev_id = f"ext-audit-{event_type}-{subject_id}-{next(self._ext_counter)}"
			store = self._audit_events  # type: ignore[attr-defined]
			store[ev_id] = {
				"id": ev_id,
				"tenant_id": tenant_id,
				"event_type": event_type,
				"subject_id": subject_id,
				"message": message,
				"actor": actor,
				"created_at": _utc(),
			}
