"""Executable service layer for APG Internationalization."""

from __future__ import annotations

from itertools import count
from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .localization_runtime import CoverageCalculator, LocaleFallbackResolver, TranslationMemoryMatcher
from .models import (
	CoverageReport,
	GlossaryTerm,
	LocaleDefinition,
	PublishBatch,
	TranslationEntry,
	TranslationSource,
	TranslationStatus,
	utc_now_iso,
)


class I18nService:
	"""Tenant-aware locale, glossary, translation, coverage, and publishing runtime."""

	def __init__(self) -> None:
		self._locales: dict[str, LocaleDefinition] = {}
		self._glossary_terms: dict[str, GlossaryTerm] = {}
		self._translations: dict[str, TranslationEntry] = {}
		self._coverage_reports: dict[str, CoverageReport] = {}
		self._publish_batches: dict[str, PublishBatch] = {}
		self._counter = count(1)
		self._fallback_resolver = LocaleFallbackResolver()
		self._memory_matcher = TranslationMemoryMatcher()
		self._coverage_calculator = CoverageCalculator()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_locale(
		self,
		locale_id: str,
		tenant_id: str,
		locale_code: str,
		display_name: str,
		owner_id: str,
		fallback_locale: str | None = None,
		regional_format: dict[str, str] | None = None,
		timezone: str = "UTC",
	) -> dict[str, Any]:
		self._enforce_i18n_policy(
			tenant_id=tenant_id,
			operation="create_locale",
			locale_owner_assigned=bool(owner_id),
		)
		locale = LocaleDefinition(
			id=locale_id,
			tenant_id=tenant_id,
			locale_code=locale_code,
			display_name=display_name,
			owner_id=owner_id,
			fallback_locale=fallback_locale or str(DEFAULT_CONFIGURATION["locales"]["default_locale"]),
			regional_format=dict(regional_format or {}),
			timezone=timezone,
		)
		self._locales[locale_id] = locale
		return locale.to_dict()

	def add_glossary_term(
		self,
		term_id: str,
		tenant_id: str,
		source_term: str,
		localized_terms: dict[str, str] | None = None,
		description: str = "",
		owner_id: str = "",
	) -> dict[str, Any]:
		self._enforce_i18n_policy(tenant_id=tenant_id, operation="add_glossary_term")
		term = GlossaryTerm(
			id=term_id,
			tenant_id=tenant_id,
			source_term=source_term,
			localized_terms=dict(localized_terms or {}),
			description=description,
			owner_id=owner_id,
		)
		self._glossary_terms[term_id] = term
		return term.to_dict()

	def upsert_translation(
		self,
		translation_id: str,
		tenant_id: str,
		key: str,
		locale_code: str,
		source_text: str,
		translated_text: str,
		machine_translation_used: bool = False,
		translation_review_recorded: bool = True,
		reviewer_id: str | None = None,
		restricted: bool = False,
		rbac_filter_applied: bool = True,
	) -> dict[str, Any]:
		self._require_locale(tenant_id, locale_code)
		self._enforce_i18n_policy(
			tenant_id=tenant_id,
			operation="upsert_translation",
			machine_translation_used=machine_translation_used,
			translation_review_recorded=translation_review_recorded,
			restricted_content_present=restricted,
			rbac_filter_applied=rbac_filter_applied,
		)
		existing = self._translations.get(translation_id)
		version = 1 if existing is None else existing.version + 1
		source = TranslationSource.MACHINE if machine_translation_used else TranslationSource.HUMAN
		status = TranslationStatus.REVIEWED if translation_review_recorded else TranslationStatus.DRAFT
		entry = TranslationEntry(
			id=translation_id,
			tenant_id=tenant_id,
			key=key,
			locale_code=locale_code,
			source_text=source_text,
			translated_text=translated_text,
			status=status,
			source=source,
			reviewer_id=reviewer_id,
			restricted=restricted,
			version=version,
		)
		self._translations[translation_id] = entry
		return entry.to_dict()

	def reuse_translation_memory(
		self,
		translation_id: str,
		tenant_id: str,
		key: str,
		locale_code: str,
		source_text: str,
		reviewer_id: str,
	) -> dict[str, Any]:
		self._require_locale(tenant_id, locale_code)
		match = self._memory_matcher.match(source_text, locale_code, list(self._translations.values()))
		if match is None or match.tenant_id != tenant_id:
			raise PermissionError("translation_memory_miss")
		entry = TranslationEntry(
			id=translation_id,
			tenant_id=tenant_id,
			key=key,
			locale_code=locale_code,
			source_text=source_text,
			translated_text=match.translated_text,
			status=TranslationStatus.REVIEWED,
			source=TranslationSource.MEMORY,
			reviewer_id=reviewer_id,
		)
		self._translations[translation_id] = entry
		return entry.to_dict()

	def publish_translations(
		self,
		batch_id: str,
		tenant_id: str,
		locale_code: str,
		translation_ids: list[str],
		approver_id: str,
		approval_recorded: bool,
		coverage_review_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_locale(tenant_id, locale_code)
		entries = [self._require_translation(translation_id, tenant_id) for translation_id in translation_ids]
		coverage = self._coverage_calculator.coverage(locale_code, [entry.key for entry in entries], entries)
		self._enforce_i18n_policy(
			tenant_id=tenant_id,
			operation="publish_translations",
			approval_recorded=approval_recorded,
			coverage_percent=float(coverage["coverage_percent"]),
			coverage_review_recorded=coverage_review_recorded,
		)
		now = utc_now_iso()
		for entry in entries:
			if entry.locale_code != locale_code:
				raise PermissionError("translation_locale_mismatch")
			if entry.status == TranslationStatus.DRAFT:
				raise PermissionError("translation_not_reviewed")
			entry.status = TranslationStatus.PUBLISHED
			entry.published_at = now
			entry.updated_at = now
		batch = PublishBatch(
			id=batch_id,
			tenant_id=tenant_id,
			locale_code=locale_code,
			translation_ids=list(translation_ids),
			approver_id=approver_id,
		)
		self._publish_batches[batch_id] = batch
		return batch.to_dict()

	def resolve_text(
		self,
		tenant_id: str,
		key: str,
		locale_code: str,
		default_locale: str | None = None,
	) -> dict[str, Any]:
		locales = [locale for locale in self._locales.values() if locale.tenant_id == tenant_id]
		chain = self._fallback_resolver.chain(
			locale_code,
			locales,
			default_locale or str(DEFAULT_CONFIGURATION["locales"]["default_locale"]),
		)
		for candidate in chain:
			for entry in self._translations.values():
				if (
					entry.tenant_id == tenant_id
					and entry.key == key
					and entry.locale_code == candidate
					and entry.status == TranslationStatus.PUBLISHED
				):
					return {"key": key, "locale_code": candidate, "fallback_chain": chain, "text": entry.translated_text}
		raise PermissionError("translation_missing")

	def coverage_report(
		self,
		report_id: str,
		tenant_id: str,
		locale_code: str,
		required_keys: list[str],
		coverage_review_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_locale(tenant_id, locale_code)
		metrics = self._coverage_calculator.coverage(
			locale_code,
			required_keys,
			[entry for entry in self._translations.values() if entry.tenant_id == tenant_id],
		)
		coverage_percent = float(metrics["coverage_percent"])
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "coverage_report",
			"coverage_percent": coverage_percent,
			"coverage_review_recorded": coverage_review_recorded,
		})
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "i18n_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "i18n_policy_blocked")
		report = CoverageReport(
			id=report_id,
			tenant_id=tenant_id,
			locale_code=locale_code,
			total_key_count=int(metrics["total_key_count"]),
			published_key_count=int(metrics["published_key_count"]),
			missing_keys=list(metrics["missing_keys"]),
			coverage_percent=coverage_percent,
			requires_review=result["decision"] == "require_review",
		)
		self._coverage_reports[report_id] = report
		return report.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper for generated package probes."""
		data = dict(metadata or {})
		return self.create_locale(
			locale_id=record_id,
			tenant_id=tenant_id,
			locale_code=str(data.get("locale_code") or "en-US"),
			display_name=str(data.get("display_name") or "Compatibility Locale"),
			owner_id=str(data.get("owner_id") or "system"),
			fallback_locale=str(data.get("fallback_locale") or "en-US"),
			regional_format=dict(data.get("regional_format") or {"date": "yyyy-MM-dd", "number": "1,234.56", "status": status}),
		)

	def list_locales(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._locales, tenant_id)

	def list_glossary_terms(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._glossary_terms, tenant_id)

	def list_translations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._translations, tenant_id)

	def list_coverage_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._coverage_reports, tenant_id)

	def list_publish_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._publish_batches, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		records: list[dict[str, Any]] = []
		for store in (self._locales, self._glossary_terms, self._translations, self._coverage_reports, self._publish_batches):
			records.extend(self._list(store, tenant_id))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		translations = self.list_translations(tenant_id)
		coverage = self.list_coverage_reports(tenant_id)
		return {
			"tenant_id": tenant_id,
			"locale_count": len(self.list_locales(tenant_id)),
			"glossary_term_count": len(self.list_glossary_terms(tenant_id)),
			"translation_count": len(translations),
			"published_translation_count": len([item for item in translations if item["status"] == TranslationStatus.PUBLISHED.value]),
			"coverage_report_count": len(coverage),
			"coverage_review_count": len([item for item in coverage if item["requires_review"]]),
			"publish_batch_count": len(self.list_publish_batches(tenant_id)),
		}

	def _enforce_i18n_policy(
		self,
		tenant_id: str,
		operation: str,
		locale_owner_assigned: bool = True,
		machine_translation_used: bool = False,
		translation_review_recorded: bool = True,
		approval_recorded: bool = True,
		restricted_content_present: bool = False,
		rbac_filter_applied: bool = True,
		coverage_percent: float = 100.0,
		coverage_review_recorded: bool = True,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": operation,
			"locale_owner_assigned": locale_owner_assigned,
			"machine_translation_used": machine_translation_used,
			"translation_review_recorded": translation_review_recorded,
			"approval_recorded": approval_recorded,
			"restricted_content_present": restricted_content_present,
			"rbac_filter_applied": rbac_filter_applied,
			"coverage_percent": coverage_percent,
			"coverage_review_recorded": coverage_review_recorded,
		})
		if result["decision"] != "allow":
			reasons = ", ".join(action.get("reason", "i18n_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "i18n_policy_blocked")

	def _require_locale(self, tenant_id: str, locale_code: str) -> LocaleDefinition:
		for locale in self._locales.values():
			if locale.tenant_id == tenant_id and locale.locale_code == locale_code:
				return locale
		raise PermissionError("locale_missing")

	def _require_translation(self, translation_id: str, tenant_id: str) -> TranslationEntry:
		translation = self._translations.get(translation_id)
		if translation is None or translation.tenant_id != tenant_id:
			raise PermissionError("translation_missing")
		return translation

	def _list(self, store: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(store.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]
