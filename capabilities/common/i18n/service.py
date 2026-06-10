"""Executable service layer for APG Internationalization."""

from __future__ import annotations

import json
import re
from itertools import count
from typing import Any

from .capability_contract import (
	DEFAULT_CONFIGURATION,
	SUPPORTED_I18N_AGENT_ROLES,
	SUPPORTED_I18N_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
	language_code_supported,
)
from .localization_runtime import CoverageCalculator, LocaleFallbackResolver, TranslationMemoryMatcher
from .models import (
	CoverageReport,
	GlossaryTerm,
	I18nAgent,
	I18nAuditEvent,
	LocaleDefinition,
	PublishBatch,
	TranslationEntry,
	TranslationSource,
	TranslationStatus,
	utc_now_iso,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class I18nService:
	"""Tenant-aware locale, glossary, translation, coverage, and publishing runtime."""

	def __init__(self) -> None:
		self._locales: dict[str, LocaleDefinition] = {}
		self._glossary_terms: dict[str, GlossaryTerm] = {}
		self._translations: dict[str, TranslationEntry] = {}
		self._coverage_reports: dict[str, CoverageReport] = {}
		self._publish_batches: dict[str, PublishBatch] = {}
		self._agents: dict[str, I18nAgent] = {}
		self._audit_events: dict[str, I18nAuditEvent] = {}
		# extra in-memory stores for new methods
		self._translation_versions: dict[str, list[dict[str, Any]]] = {}
		self._machine_translation_jobs: dict[str, dict[str, Any]] = {}
		self._plural_rules: dict[str, dict[str, Any]] = {}
		self._font_hints: dict[str, dict[str, Any]] = {}
		self._locale_analytics: dict[str, list[dict[str, Any]]] = {}
		self._export_jobs: dict[str, dict[str, Any]] = {}
		self._review_assignments: dict[str, dict[str, Any]] = {}
		self._counter = count(1)
		self._fallback_resolver = LocaleFallbackResolver()
		self._memory_matcher = TranslationMemoryMatcher()
		self._coverage_calculator = CoverageCalculator()

	# ------------------------------------------------------------------ #
	# Original 21 methods                                                  #
	# ------------------------------------------------------------------ #

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
		effective_fallback = fallback_locale or str(DEFAULT_CONFIGURATION["locales"]["default_locale"])
		effective_regional_format = dict(regional_format or {"date": "yyyy-MM-dd", "number": "1,234.56"})
		self._enforce_i18n_policy(
			tenant_id=tenant_id,
			operation="create_locale",
			locale_owner_assigned=bool(owner_id),
			supported_language_code=language_code_supported(locale_code),
			fallback_locale_present=bool(effective_fallback),
			regional_format_present=bool(effective_regional_format),
		)
		locale = LocaleDefinition(
			id=locale_id,
			tenant_id=tenant_id,
			locale_code=locale_code,
			display_name=display_name,
			owner_id=owner_id,
			fallback_locale=effective_fallback,
			regional_format=effective_regional_format,
			timezone=timezone,
		)
		self._locales[_state_key(tenant_id, locale_id)] = locale
		self._record_audit(tenant_id, locale_id, "locale_created", owner_id, "allow", metadata={"locale_code": locale_code})
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
		result = self._enforce_i18n_policy(
			tenant_id=tenant_id,
			operation="add_glossary_term",
			glossary_owner_present=bool(owner_id),
		)
		term = GlossaryTerm(
			id=term_id,
			tenant_id=tenant_id,
			source_term=source_term,
			localized_terms=dict(localized_terms or {}),
			description=description,
			owner_id=owner_id,
		)
		self._glossary_terms[_state_key(tenant_id, term_id)] = term
		self._record_audit(tenant_id, term_id, "glossary_term_added", owner_id, result["decision"], metadata={"source_term": source_term})
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
			translation_key_present=bool(key),
			translated_text_present=bool(translated_text),
			machine_translation_used=machine_translation_used,
			translation_review_recorded=translation_review_recorded,
			restricted_content_present=restricted,
			rbac_filter_applied=rbac_filter_applied,
		)
		existing = self._translations.get(_state_key(tenant_id, translation_id))
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
		self._translations[_state_key(tenant_id, translation_id)] = entry
		# Snapshot version history
		history = self._translation_versions.setdefault(_state_key(tenant_id, translation_id), [])
		history.append({"version": version, "translated_text": translated_text, "source": source.value, "updated_at": utc_now_iso()})
		self._record_audit(tenant_id, translation_id, "translation_upserted", reviewer_id or "translator", "allow", metadata={"key": key, "locale_code": locale_code, "source": source.value})
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
		self._translations[_state_key(tenant_id, translation_id)] = entry
		self._record_audit(tenant_id, translation_id, "translation_memory_reused", reviewer_id, "allow", metadata={"matched_translation": match.id, "locale_code": locale_code})
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
		missing_keys = [
			key for key in {entry.key for entry in entries}
			if not any(entry.key == key and entry.locale_code == locale_code for entry in entries)
		]
		result = self._enforce_i18n_policy(
			tenant_id=tenant_id,
			operation="publish_translations",
			approval_recorded=approval_recorded,
			approver_present=bool(approver_id),
			coverage_percent=float(coverage["coverage_percent"]),
			coverage_review_recorded=coverage_review_recorded,
			missing_key_count=len(missing_keys),
			missing_key_review_recorded=not missing_keys,
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
		self._publish_batches[_state_key(tenant_id, batch_id)] = batch
		self._record_audit(tenant_id, batch_id, "translations_published", approver_id, result["decision"], metadata={"locale_code": locale_code, "translation_count": len(translation_ids)})
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
		self._coverage_reports[_state_key(tenant_id, report_id)] = report
		self._record_audit(tenant_id, report_id, "coverage_reported", "coverage-dashboard", result["decision"], reasons=tuple(action.get("reason", "") for action in result["actions"]), metadata=metrics)
		return report.to_dict()

	def register_i18n_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"i18n_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_I18N_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_I18N_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		if result["decision"] == "deny":
			raise PermissionError(_reasons(result) or "i18n_policy_blocked")
		agent = I18nAgent(
			id=agent_id or f"i18n-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._record_audit(tenant_id, agent.id, "i18n_agent_registered", name, result["decision"], metadata=agent.to_dict())
		return agent.to_dict()

	def validate_batch_i18n_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": True,
			"requested_operation": "batch_i18n_mutation",
			"event_stream": event_stream,
		})

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

	def list_i18n_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		records: list[dict[str, Any]] = []
		for store in (self._locales, self._glossary_terms, self._translations, self._coverage_reports, self._publish_batches, self._agents, self._audit_events):
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
			"i18n_agent_count": len(self.list_i18n_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id or "default")["streaming"],
		}

	# ------------------------------------------------------------------ #
	# New methods (16 new, reaching 37 total public methods)               #
	# ------------------------------------------------------------------ #

	async def locale_create(
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
		"""Async alias for create_locale; preferred for new callers."""
		return self.create_locale(
			locale_id=locale_id,
			tenant_id=tenant_id,
			locale_code=locale_code,
			display_name=display_name,
			owner_id=owner_id,
			fallback_locale=fallback_locale,
			regional_format=regional_format,
			timezone=timezone,
		)

	async def translation_import(
		self,
		tenant_id: str,
		locale_code: str,
		entries: list[dict[str, Any]],
		importer_id: str,
		overwrite_existing: bool = False,
	) -> dict[str, Any]:
		"""Bulk-import translation entries from an external payload (e.g. PO/JSON export).

		Each entry must contain: translation_id, key, source_text, translated_text.
		Returns a summary of imported, skipped, and failed items.
		"""
		self._require_locale(tenant_id, locale_code)
		imported, skipped, failed = 0, 0, []
		for item in entries:
			tid = str(item.get("translation_id") or "")
			key = str(item.get("key") or "")
			if not tid or not key:
				failed.append({"item": item, "reason": "missing_id_or_key"})
				continue
			key_exists = _state_key(tenant_id, tid) in self._translations
			if key_exists and not overwrite_existing:
				skipped += 1
				continue
			try:
				self.upsert_translation(
					translation_id=tid,
					tenant_id=tenant_id,
					key=key,
					locale_code=locale_code,
					source_text=str(item.get("source_text") or ""),
					translated_text=str(item.get("translated_text") or ""),
					machine_translation_used=bool(item.get("machine_translation_used", False)),
					translation_review_recorded=bool(item.get("translation_review_recorded", True)),
					reviewer_id=str(item.get("reviewer_id") or importer_id),
				)
				imported += 1
			except Exception as exc:
				failed.append({"item": item, "reason": str(exc)})
		self._record_audit(tenant_id, f"import:{locale_code}", "translation_import_completed", importer_id, "allow", metadata={"imported": imported, "skipped": skipped, "failed_count": len(failed)})
		return {"locale_code": locale_code, "imported": imported, "skipped": skipped, "failed": failed}

	async def machine_translate(
		self,
		translation_id: str,
		tenant_id: str,
		key: str,
		locale_code: str,
		source_text: str,
		engine: str = "ollama",
		model: str = "qwen3",
		reviewer_id: str | None = None,
	) -> dict[str, Any]:
		"""Submit a machine-translation job and store result as a DRAFT entry.

		In production this would call an Ollama-served model; here we record the
		job metadata and create a draft translation that requires human review.
		"""
		self._require_locale(tenant_id, locale_code)
		job_id = f"mt-job:{next(self._counter):06d}"
		job = {
			"id": job_id,
			"tenant_id": tenant_id,
			"translation_id": translation_id,
			"key": key,
			"locale_code": locale_code,
			"source_text": source_text,
			"engine": engine,
			"model": model,
			"status": "submitted",
			"submitted_at": utc_now_iso(),
		}
		self._machine_translation_jobs[_state_key(tenant_id, job_id)] = job
		# Produce a placeholder translated text pending real MT output
		placeholder = f"[MT:{engine}/{model}] {source_text}"
		entry = self.upsert_translation(
			translation_id=translation_id,
			tenant_id=tenant_id,
			key=key,
			locale_code=locale_code,
			source_text=source_text,
			translated_text=placeholder,
			machine_translation_used=True,
			translation_review_recorded=False,
			reviewer_id=reviewer_id,
		)
		job["status"] = "completed"
		job["entry_id"] = translation_id
		return {"job": job, "entry": entry}

	async def plural_rules(
		self,
		tenant_id: str,
		locale_code: str,
		rules: dict[str, str] | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Register or retrieve CLDR-style plural rules for a locale.

		rules dict maps category names (zero, one, two, few, many, other) to
		CLDR rule strings.  If rules is None the stored rules are returned.
		"""
		self._require_locale(tenant_id, locale_code)
		store_key = _state_key(tenant_id, locale_code)
		if rules is not None:
			allowed_categories = {"zero", "one", "two", "few", "many", "other"}
			invalid = set(rules.keys()) - allowed_categories
			if invalid:
				raise ValueError(f"invalid_plural_categories:{','.join(sorted(invalid))}")
			self._plural_rules[store_key] = {
				"tenant_id": tenant_id,
				"locale_code": locale_code,
				"rules": dict(rules),
				"updated_at": utc_now_iso(),
				"actor": actor,
			}
			self._record_audit(tenant_id, locale_code, "plural_rules_updated", actor, "allow", metadata={"categories": list(rules.keys())})
		record = self._plural_rules.get(store_key, {
			"tenant_id": tenant_id,
			"locale_code": locale_code,
			"rules": {"other": "n != 1"},
			"updated_at": utc_now_iso(),
			"actor": "default",
		})
		return record

	async def date_localise(
		self,
		tenant_id: str,
		locale_code: str,
		iso_datetime: str,
		format_name: str = "medium",
	) -> dict[str, Any]:
		"""Format an ISO-8601 datetime string using the locale's regional_format.

		format_name: short | medium | long | full
		Returns the formatted string alongside locale metadata.
		"""
		locale = self._require_locale(tenant_id, locale_code)
		date_fmt = locale.regional_format.get("date", "yyyy-MM-dd")
		# Trivial format substitution — real impl would use Babel / arrow
		formatted = _apply_date_format(iso_datetime, date_fmt, format_name)
		self._record_audit(tenant_id, locale_code, "date_localised", "system", "allow", metadata={"iso_datetime": iso_datetime, "format_name": format_name})
		return {
			"tenant_id": tenant_id,
			"locale_code": locale_code,
			"iso_datetime": iso_datetime,
			"format_name": format_name,
			"formatted": formatted,
			"date_format_pattern": date_fmt,
		}

	async def number_localise(
		self,
		tenant_id: str,
		locale_code: str,
		value: int | float,
		decimal_places: int = 2,
	) -> dict[str, Any]:
		"""Format a numeric value according to the locale's regional_format."""
		locale = self._require_locale(tenant_id, locale_code)
		number_fmt = locale.regional_format.get("number", "1,234.56")
		formatted = _apply_number_format(value, number_fmt, decimal_places)
		return {
			"tenant_id": tenant_id,
			"locale_code": locale_code,
			"value": value,
			"decimal_places": decimal_places,
			"formatted": formatted,
			"number_format_pattern": number_fmt,
		}

	async def currency_localise(
		self,
		tenant_id: str,
		locale_code: str,
		amount: int | float,
		currency_code: str,
		decimal_places: int = 2,
	) -> dict[str, Any]:
		"""Format a monetary amount using locale conventions and ISO 4217 currency code."""
		locale = self._require_locale(tenant_id, locale_code)
		currency_fmt = locale.regional_format.get("currency", locale.regional_format.get("number", "1,234.56"))
		formatted_number = _apply_number_format(amount, currency_fmt, decimal_places)
		formatted = f"{currency_code.upper()} {formatted_number}"
		return {
			"tenant_id": tenant_id,
			"locale_code": locale_code,
			"amount": amount,
			"currency_code": currency_code.upper(),
			"formatted": formatted,
		}

	async def rtl_check(
		self,
		tenant_id: str,
		locale_code: str,
	) -> dict[str, Any]:
		"""Return whether the locale uses a right-to-left script."""
		locale = self._require_locale(tenant_id, locale_code)
		rtl_codes = {"ar", "he", "fa", "ur", "dv", "ha", "ps", "sd", "ug", "yi", "arc", "ckb"}
		lang_part = locale_code.split("-")[0].lower()
		is_rtl = lang_part in rtl_codes
		return {
			"tenant_id": tenant_id,
			"locale_code": locale_code,
			"is_rtl": is_rtl,
			"direction": "rtl" if is_rtl else "ltr",
			"locale_display_name": locale.display_name,
		}

	async def font_detect(
		self,
		tenant_id: str,
		locale_code: str,
		fallback_font: str = "sans-serif",
	) -> dict[str, Any]:
		"""Return recommended font stack for the locale's script.

		Heuristic only — production callers should cross-reference a font registry.
		"""
		self._require_locale(tenant_id, locale_code)
		store_key = _state_key(tenant_id, locale_code)
		cached = self._font_hints.get(store_key)
		if cached:
			return cached
		script_font_map: dict[str, list[str]] = {
			"ar": ["Noto Naskh Arabic", "Amiri", fallback_font],
			"he": ["Noto Serif Hebrew", "Frank Ruhl Libre", fallback_font],
			"zh": ["Noto Sans CJK SC", "PingFang SC", fallback_font],
			"ja": ["Noto Sans CJK JP", "Hiragino Sans", fallback_font],
			"ko": ["Noto Sans CJK KR", "Apple SD Gothic Neo", fallback_font],
			"th": ["Noto Sans Thai", "Leelawadee", fallback_font],
			"hi": ["Noto Sans Devanagari", "Mangal", fallback_font],
			"bn": ["Noto Sans Bengali", "Vrinda", fallback_font],
		}
		lang_part = locale_code.split("-")[0].lower()
		fonts = script_font_map.get(lang_part, ["Noto Sans", fallback_font])
		hint = {
			"tenant_id": tenant_id,
			"locale_code": locale_code,
			"recommended_fonts": fonts,
			"font_stack": ", ".join(f'"{f}"' if " " in f else f for f in fonts),
			"detected_at": utc_now_iso(),
		}
		self._font_hints[store_key] = hint
		return hint

	async def translation_export(
		self,
		tenant_id: str,
		locale_code: str,
		format_: str = "json",
		status_filter: str | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Export all translations for a locale to a serialisable structure.

		format_: json | po | csv
		status_filter: published | reviewed | draft | None (all)
		"""
		self._require_locale(tenant_id, locale_code)
		entries = [
			e for e in self._translations.values()
			if e.tenant_id == tenant_id and e.locale_code == locale_code
			and (status_filter is None or e.status.value == status_filter)
		]
		rows = [
			{"key": e.key, "source_text": e.source_text, "translated_text": e.translated_text, "status": e.status.value}
			for e in sorted(entries, key=lambda e: e.key)
		]
		if format_ == "json":
			payload = json.dumps({row["key"]: row["translated_text"] for row in rows}, ensure_ascii=False, indent=2)
		elif format_ == "csv":
			header = "key,source_text,translated_text,status\n"
			lines = "\n".join(f'{row["key"]},{row["source_text"]},{row["translated_text"]},{row["status"]}' for row in rows)
			payload = header + lines
		else:
			# Minimal PO format
			lines = [f'# APG i18n export — {locale_code}', ""]
			for row in rows:
				lines += [f'msgid "{row["source_text"]}"', f'msgstr "{row["translated_text"]}"', ""]
			payload = "\n".join(lines)
		job_id = f"export:{next(self._counter):06d}"
		export_job = {
			"id": job_id,
			"tenant_id": tenant_id,
			"locale_code": locale_code,
			"format": format_,
			"status_filter": status_filter,
			"entry_count": len(rows),
			"payload": payload,
			"created_at": utc_now_iso(),
		}
		self._export_jobs[_state_key(tenant_id, job_id)] = export_job
		self._record_audit(tenant_id, job_id, "translation_export_created", actor, "allow", metadata={"locale_code": locale_code, "format": format_, "entry_count": len(rows)})
		return export_job

	async def translation_review(
		self,
		tenant_id: str,
		translation_id: str,
		reviewer_id: str,
		approved: bool,
		notes: str = "",
	) -> dict[str, Any]:
		"""Mark a translation as reviewed/approved or rejected by a human reviewer."""
		entry = self._require_translation(translation_id, tenant_id)
		if approved:
			entry.status = TranslationStatus.REVIEWED
			entry.reviewer_id = reviewer_id
		else:
			entry.status = TranslationStatus.DRAFT
		entry.updated_at = utc_now_iso()
		assignment = {
			"translation_id": translation_id,
			"reviewer_id": reviewer_id,
			"approved": approved,
			"notes": notes,
			"reviewed_at": utc_now_iso(),
		}
		self._review_assignments[_state_key(tenant_id, translation_id)] = assignment
		self._record_audit(tenant_id, translation_id, "translation_reviewed", reviewer_id, "allow", metadata={"approved": approved, "notes": notes})
		return {"entry": entry.to_dict(), "review": assignment}

	async def missing_keys_report(
		self,
		tenant_id: str,
		locale_code: str,
		reference_locale: str = "en-US",
	) -> dict[str, Any]:
		"""Compare locale against reference_locale and return keys present in reference but absent in locale."""
		self._require_locale(tenant_id, locale_code)
		ref_keys = {
			e.key
			for e in self._translations.values()
			if e.tenant_id == tenant_id and e.locale_code == reference_locale
		}
		locale_keys = {
			e.key
			for e in self._translations.values()
			if e.tenant_id == tenant_id and e.locale_code == locale_code
		}
		missing = sorted(ref_keys - locale_keys)
		extra = sorted(locale_keys - ref_keys)
		return {
			"tenant_id": tenant_id,
			"locale_code": locale_code,
			"reference_locale": reference_locale,
			"missing_key_count": len(missing),
			"extra_key_count": len(extra),
			"missing_keys": missing,
			"extra_keys": extra,
			"generated_at": utc_now_iso(),
		}

	async def locale_fallback(
		self,
		tenant_id: str,
		locale_code: str,
		key: str,
	) -> dict[str, Any]:
		"""Resolve fallback chain for a key and return the first matching translation."""
		try:
			result = self.resolve_text(tenant_id, key, locale_code)
			return {**result, "resolved": True}
		except PermissionError:
			return {
				"key": key,
				"locale_code": locale_code,
				"resolved": False,
				"text": None,
				"fallback_chain": [],
			}

	async def locale_clone(
		self,
		tenant_id: str,
		source_locale_code: str,
		new_locale_id: str,
		new_locale_code: str,
		new_display_name: str,
		owner_id: str,
		clone_translations: bool = True,
	) -> dict[str, Any]:
		"""Clone a locale definition and optionally all its translation entries.

		The cloned translations start in DRAFT status so they can be reviewed
		before publication.
		"""
		source_locale = self._require_locale(tenant_id, source_locale_code)
		new_locale = self.create_locale(
			locale_id=new_locale_id,
			tenant_id=tenant_id,
			locale_code=new_locale_code,
			display_name=new_display_name,
			owner_id=owner_id,
			fallback_locale=source_locale.fallback_locale,
			regional_format=dict(source_locale.regional_format),
			timezone=source_locale.timezone,
		)
		cloned_count = 0
		if clone_translations:
			source_entries = [
				e for e in self._translations.values()
				if e.tenant_id == tenant_id and e.locale_code == source_locale_code
			]
			for e in source_entries:
				new_tid = f"{new_locale_id}:{e.key}"
				self.upsert_translation(
					translation_id=new_tid,
					tenant_id=tenant_id,
					key=e.key,
					locale_code=new_locale_code,
					source_text=e.source_text,
					translated_text=e.translated_text,
					machine_translation_used=False,
					translation_review_recorded=False,
					reviewer_id=owner_id,
				)
				cloned_count += 1
		self._record_audit(tenant_id, new_locale_id, "locale_cloned", owner_id, "allow", metadata={"source_locale_code": source_locale_code, "cloned_translations": cloned_count})
		return {"locale": new_locale, "cloned_translation_count": cloned_count}

	async def locale_analytics(
		self,
		tenant_id: str,
		locale_code: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return per-locale analytics: translation counts, coverage, last activity."""
		target_locales = [
			lc for lc in self._locales.values()
			if lc.tenant_id == tenant_id and (locale_code is None or lc.locale_code == locale_code)
		]
		results: list[dict[str, Any]] = []
		for lc in target_locales:
			entries = [e for e in self._translations.values() if e.tenant_id == tenant_id and e.locale_code == lc.locale_code]
			published = [e for e in entries if e.status == TranslationStatus.PUBLISHED]
			draft = [e for e in entries if e.status == TranslationStatus.DRAFT]
			reviewed = [e for e in entries if e.status == TranslationStatus.REVIEWED]
			coverage_pct = round(len(published) / max(len(entries), 1) * 100, 2)
			last_activity = max((e.updated_at for e in entries), default=None)
			analytic = {
				"locale_code": lc.locale_code,
				"display_name": lc.display_name,
				"total_entries": len(entries),
				"published": len(published),
				"reviewed": len(reviewed),
				"draft": len(draft),
				"coverage_percent": coverage_pct,
				"last_activity": last_activity,
				"generated_at": utc_now_iso(),
			}
			results.append(analytic)
			store_list = self._locale_analytics.setdefault(_state_key(tenant_id, lc.locale_code), [])
			store_list.append(analytic)
		return results

	async def translation_version(
		self,
		tenant_id: str,
		translation_id: str,
	) -> list[dict[str, Any]]:
		"""Return the full version history of a translation entry."""
		self._require_translation(translation_id, tenant_id)
		return list(self._translation_versions.get(_state_key(tenant_id, translation_id), []))

	async def glossary_lookup(
		self,
		tenant_id: str,
		source_term: str,
		locale_code: str | None = None,
	) -> list[dict[str, Any]]:
		"""Find glossary terms matching source_term; optionally filter by locale."""
		matches = [
			t.to_dict()
			for t in self._glossary_terms.values()
			if t.tenant_id == tenant_id and source_term.lower() in t.source_term.lower()
			and (locale_code is None or locale_code in t.localized_terms)
		]
		return sorted(matches, key=lambda m: m["id"])

	async def translation_search(
		self,
		tenant_id: str,
		query: str,
		locale_code: str | None = None,
		status_filter: str | None = None,
	) -> list[dict[str, Any]]:
		"""Full-text search across source_text and translated_text for a tenant."""
		q = query.lower()
		results = [
			e.to_dict()
			for e in self._translations.values()
			if e.tenant_id == tenant_id
			and (q in e.source_text.lower() or q in e.translated_text.lower())
			and (locale_code is None or e.locale_code == locale_code)
			and (status_filter is None or e.status.value == status_filter)
		]
		return sorted(results, key=lambda r: r["id"])

	async def batch_approve_translations(
		self,
		tenant_id: str,
		translation_ids: list[str],
		reviewer_id: str,
	) -> dict[str, Any]:
		"""Approve a list of translation entries in one call, setting them to REVIEWED."""
		approved, failed = [], []
		for tid in translation_ids:
			try:
				entry = self._require_translation(tid, tenant_id)
				entry.status = TranslationStatus.REVIEWED
				entry.reviewer_id = reviewer_id
				entry.updated_at = utc_now_iso()
				approved.append(tid)
			except Exception as exc:
				failed.append({"id": tid, "reason": str(exc)})
		self._record_audit(
			tenant_id, f"batch_approve:{reviewer_id}", "batch_translations_approved",
			reviewer_id, "allow",
			metadata={"approved": len(approved), "failed": len(failed)},
		)
		return {"approved": approved, "failed": failed, "reviewer_id": reviewer_id}

	async def locale_timezone_list(
		self,
		tenant_id: str,
	) -> list[dict[str, Any]]:
		"""Return a list of all distinct timezones used across tenant locales."""
		seen: dict[str, str] = {}
		for lc in self._locales.values():
			if lc.tenant_id == tenant_id and lc.timezone not in seen:
				seen[lc.timezone] = lc.locale_code
		return [{"timezone": tz, "example_locale": lc} for tz, lc in sorted(seen.items())]

	# ------------------------------------------------------------------ #
	# Private helpers                                                      #
	# ------------------------------------------------------------------ #

	def _enforce_i18n_policy(
		self,
		tenant_id: str,
		operation: str,
		locale_owner_assigned: bool = True,
		supported_language_code: bool = True,
		fallback_locale_present: bool = True,
		regional_format_present: bool = True,
		machine_translation_used: bool = False,
		translation_review_recorded: bool = True,
		approval_recorded: bool = True,
		approver_present: bool = True,
		restricted_content_present: bool = False,
		rbac_filter_applied: bool = True,
		coverage_percent: float = 100.0,
		coverage_review_recorded: bool = True,
		glossary_owner_present: bool = True,
		translation_key_present: bool = True,
		translated_text_present: bool = True,
		missing_key_count: int = 0,
		missing_key_review_recorded: bool = True,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": operation,
			"locale_owner_assigned": locale_owner_assigned,
			"language_code_supported": supported_language_code,
			"fallback_locale_present": fallback_locale_present,
			"regional_format_present": regional_format_present,
			"glossary_owner_present": glossary_owner_present,
			"translation_key_present": translation_key_present,
			"translated_text_present": translated_text_present,
			"machine_translation_used": machine_translation_used,
			"translation_review_recorded": translation_review_recorded,
			"approval_recorded": approval_recorded,
			"approver_present": approver_present,
			"restricted_content_present": restricted_content_present,
			"rbac_filter_applied": rbac_filter_applied,
			"coverage_percent": coverage_percent,
			"coverage_review_recorded": coverage_review_recorded,
			"missing_key_count": missing_key_count,
			"missing_key_review_recorded": missing_key_review_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(_reasons(result) or "i18n_policy_blocked")
		return result

	def _require_locale(self, tenant_id: str, locale_code: str) -> LocaleDefinition:
		for locale in self._locales.values():
			if locale.tenant_id == tenant_id and locale.locale_code == locale_code:
				return locale
		raise PermissionError("locale_missing")

	def _require_translation(self, translation_id: str, tenant_id: str) -> TranslationEntry:
		translation = self._translations.get(_state_key(tenant_id, translation_id))
		if translation is None or translation.tenant_id != tenant_id:
			raise PermissionError("translation_missing")
		return translation

	def _list(self, store: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(store.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> I18nAuditEvent:
		event_id = f"audit:{len(self._audit_events) + 1:06d}"
		event = I18nAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)
		self._audit_events[event_id] = event
		return event


# ------------------------------------------------------------------ #
# Module-level helpers                                                 #
# ------------------------------------------------------------------ #

def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"


def _reasons(result: dict[str, Any]) -> str:
	return ", ".join(action.get("reason", "i18n_policy_blocked") for action in result["actions"])


def _apply_date_format(iso_datetime: str, date_fmt: str, format_name: str) -> str:
	"""Minimal date formatter — returns the ISO date portion reformatted."""
	date_part = iso_datetime[:10]  # e.g. 2026-06-04
	parts = date_part.split("-")
	if len(parts) != 3:
		return iso_datetime
	year, month, day = parts
	if format_name == "short":
		sep = "/" if "/" in date_fmt else "-"
		return f"{month}{sep}{day}{sep}{year[-2:]}"
	if format_name == "full":
		return f"{day} {_month_name(int(month))} {year}"
	# medium / long
	return f"{day} {_month_abbr(int(month))} {year}"


def _apply_number_format(value: int | float, pattern: str, decimal_places: int) -> str:
	"""Minimal number formatter that respects the locale pattern's separator conventions."""
	use_comma_decimal = "," in pattern and pattern.index(",") > pattern.index(".")  if "." in pattern and "," in pattern else False
	rounded = round(float(value), decimal_places)
	formatted = f"{rounded:,.{decimal_places}f}"
	if use_comma_decimal:
		# European style: swap . and ,
		formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
	return formatted


def _month_name(n: int) -> str:
	names = ["January", "February", "March", "April", "May", "June",
	         "July", "August", "September", "October", "November", "December"]
	return names[max(0, min(n - 1, 11))]


def _month_abbr(n: int) -> str:
	return _month_name(n)[:3]
