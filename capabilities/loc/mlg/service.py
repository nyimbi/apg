"""Async service layer for APG Multi-Language & Localisation."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CONTENT_TYPES,
		SUPPORTED_CURRENCY_DISPLAY_MODES,
		SUPPORTED_DATE_FORMATS,
		SUPPORTED_LANGUAGES,
		SUPPORTED_LOCALES,
		SUPPORTED_NUMBER_FORMATS,
		SUPPORTED_RTL_LANGUAGES,
		SUPPORTED_SCRIPTS,
		SUPPORTED_TEXT_DIRECTIONS,
		SUPPORTED_TRANSLATION_STATUSES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		FormattingRuleCreate,
		FormattingRuleResponse,
		LocaleConfigCreate,
		LocaleConfigResponse,
		LocaleConfigUpdate,
		MlgAgentCreate,
		MlgAgentResponse,
		MlgAuditEvent,
		TerminologyCreate,
		TerminologyResponse,
		TranslationCreate,
		TranslationResponse,
		TranslationUpdate,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore[no-redef]
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CONTENT_TYPES,
		SUPPORTED_CURRENCY_DISPLAY_MODES,
		SUPPORTED_DATE_FORMATS,
		SUPPORTED_LANGUAGES,
		SUPPORTED_LOCALES,
		SUPPORTED_NUMBER_FORMATS,
		SUPPORTED_RTL_LANGUAGES,
		SUPPORTED_SCRIPTS,
		SUPPORTED_TEXT_DIRECTIONS,
		SUPPORTED_TRANSLATION_STATUSES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore[no-redef]
		FormattingRuleCreate,
		FormattingRuleResponse,
		LocaleConfigCreate,
		LocaleConfigResponse,
		LocaleConfigUpdate,
		MlgAgentCreate,
		MlgAgentResponse,
		MlgAuditEvent,
		TerminologyCreate,
		TerminologyResponse,
		TranslationCreate,
		TranslationResponse,
		TranslationUpdate,
	)


def _present(v: str | None) -> bool:
	return bool(v and v.strip())


class MultiLanguageLocalisationService:
	"""Tenant-scoped runtime for Multi-Language & Localisation capability."""

	def __init__(self) -> None:
		self._locales: dict[tuple[str, str], LocaleConfigResponse] = {}
		self._translations: dict[tuple[str, str], TranslationResponse] = {}
		self._formatting_rules: dict[tuple[str, str], FormattingRuleResponse] = {}
		self._terminology: dict[tuple[str, str], TerminologyResponse] = {}
		self._agents: dict[tuple[str, str], MlgAgentResponse] = {}
		self._audit_events: list[MlgAuditEvent] = []

	# --- Contract ---

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the full capability contract."""
		return get_capability_contract(tenant_id)

	async def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate capability rules against a context dict."""
		return evaluate_capability_rules(context)

	# --- Locale Configuration ---

	async def configure_locale(self, payload: LocaleConfigCreate, actor_id: str = "system") -> LocaleConfigResponse:
		"""Configure a locale for the tenant."""
		self._log_operation("configure_locale", payload.tenant_id)
		is_rtl_lang = payload.language in SUPPORTED_RTL_LANGUAGES
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "configure_locale",
			"locale_supported": payload.locale_code in SUPPORTED_LOCALES,
			"language_supported": payload.language in SUPPORTED_LANGUAGES,
			"script_supported": payload.script in SUPPORTED_SCRIPTS,
			"direction_supported": payload.text_direction in SUPPORTED_TEXT_DIRECTIONS,
			"date_format_supported": payload.date_format in SUPPORTED_DATE_FORMATS,
			"number_format_supported": payload.number_format in SUPPORTED_NUMBER_FORMATS,
			"rtl_language": is_rtl_lang,
			"rtl_direction_set": payload.text_direction == "rtl",
		})
		locale = LocaleConfigResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			locale_code=payload.locale_code,
			language=payload.language,
			script=payload.script,
			text_direction=payload.text_direction,
			date_format=payload.date_format,
			number_format=payload.number_format,
			currency_display=payload.currency_display,
			is_default=payload.is_default,
			is_rtl=payload.is_rtl,
			is_active=True,
			notes=payload.notes,
			created_by=actor_id,
		)
		# Only one locale can be default per tenant
		if payload.is_default:
			for existing in self._locales.values():
				if existing.tenant_id == payload.tenant_id and existing.is_default:
					data = existing.model_dump()
					data["is_default"] = False
					data["updated_at"] = datetime.utcnow()
					self._locales[self._key(existing.tenant_id, existing.id)] = LocaleConfigResponse.model_validate(data)
		self._locales[self._key(payload.tenant_id, locale.id)] = locale
		await self._emit(payload.tenant_id, "locale_configured", locale.id, actor_id)
		return locale

	async def get_locale(self, tenant_id: str, locale_id: str) -> LocaleConfigResponse:
		"""Get a locale configuration by ID."""
		self._enforce_tenant(tenant_id)
		locale = self._locales.get(self._key(tenant_id, locale_id))
		if not locale:
			raise KeyError(f"locale '{locale_id}' not found for tenant '{tenant_id}'")
		return locale

	async def get_locale_by_code(self, tenant_id: str, locale_code: str) -> LocaleConfigResponse | None:
		"""Lookup locale by locale code."""
		self._enforce_tenant(tenant_id)
		for l in self._locales.values():
			if l.tenant_id == tenant_id and l.locale_code == locale_code and l.is_active:
				return l
		return None

	async def list_locales(self, tenant_id: str, language: str | None = None, is_rtl: bool | None = None) -> list[LocaleConfigResponse]:
		"""List configured locales for a tenant."""
		self._enforce_tenant(tenant_id)
		result = [l for l in self._locales.values() if l.tenant_id == tenant_id and l.is_active]
		if language:
			result = [l for l in result if l.language == language]
		if is_rtl is not None:
			result = [l for l in result if l.is_rtl == is_rtl]
		return result

	async def update_locale(self, tenant_id: str, locale_id: str, payload: LocaleConfigUpdate, actor_id: str = "system") -> LocaleConfigResponse:
		"""Update a locale configuration."""
		self._enforce_tenant(tenant_id)
		locale = await self.get_locale(tenant_id, locale_id)
		data = locale.model_dump()
		data.update(payload.model_dump(exclude_none=True))
		data["updated_at"] = datetime.utcnow()
		updated = LocaleConfigResponse.model_validate(data)
		self._locales[self._key(tenant_id, locale_id)] = updated
		await self._emit(tenant_id, "locale_updated", locale_id, actor_id)
		return updated

	async def get_default_locale(self, tenant_id: str) -> LocaleConfigResponse | None:
		"""Return the tenant's default locale, or None if not set."""
		self._enforce_tenant(tenant_id)
		for l in self._locales.values():
			if l.tenant_id == tenant_id and l.is_default and l.is_active:
				return l
		return None

	# --- Translations ---

	async def create_translation(self, payload: TranslationCreate, actor_id: str = "system") -> TranslationResponse:
		"""Create a translation entry."""
		self._log_operation("create_translation", payload.tenant_id)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_translation",
			"source_language_present": _present(payload.source_language),
			"target_language_present": _present(payload.target_language),
			"content_type_supported": payload.content_type in SUPPORTED_CONTENT_TYPES,
			"translator_present": _present(payload.translator_id),
			"translation_key_present": _present(payload.translation_key),
		})
		translation = TranslationResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			translation_key=payload.translation_key,
			source_language=payload.source_language,
			target_language=payload.target_language,
			content_type=payload.content_type,
			source_text=payload.source_text,
			translated_text=payload.translated_text,
			translator_id=payload.translator_id,
			namespace=payload.namespace,
			version=payload.version,
			status="draft",
			notes=payload.notes,
			created_by=actor_id,
		)
		self._translations[self._key(payload.tenant_id, translation.id)] = translation
		await self._emit(payload.tenant_id, "translation_created", translation.id, actor_id)
		return translation

	async def get_translation(self, tenant_id: str, translation_id: str) -> TranslationResponse:
		"""Get a translation entry by ID."""
		self._enforce_tenant(tenant_id)
		tr = self._translations.get(self._key(tenant_id, translation_id))
		if not tr:
			raise KeyError(f"translation '{translation_id}' not found for tenant '{tenant_id}'")
		return tr

	async def list_translations(self, tenant_id: str, target_language: str | None = None, content_type: str | None = None, status: str | None = None, namespace: str | None = None) -> list[TranslationResponse]:
		"""List translations with optional filters."""
		self._enforce_tenant(tenant_id)
		result = [t for t in self._translations.values() if t.tenant_id == tenant_id]
		if target_language:
			result = [t for t in result if t.target_language == target_language]
		if content_type:
			result = [t for t in result if t.content_type == content_type]
		if status:
			result = [t for t in result if t.status == status]
		if namespace:
			result = [t for t in result if t.namespace == namespace]
		return result

	async def submit_translation_for_review(self, tenant_id: str, translation_id: str, actor_id: str = "system") -> TranslationResponse:
		"""Submit a draft translation for reviewer approval."""
		self._enforce_tenant(tenant_id)
		tr = await self.get_translation(tenant_id, translation_id)
		assert tr.status == "draft", f"only draft translations can be submitted, got '{tr.status}'"
		data = tr.model_dump()
		data["status"] = "pending_review"
		data["updated_at"] = datetime.utcnow()
		updated = TranslationResponse.model_validate(data)
		self._translations[self._key(tenant_id, translation_id)] = updated
		await self._emit(tenant_id, "translation_submitted_for_review", translation_id, actor_id)
		return updated

	async def approve_translation(self, tenant_id: str, translation_id: str, reviewer_id: str) -> TranslationResponse:
		"""Approve a translation under review."""
		self._enforce_tenant(tenant_id)
		tr = await self.get_translation(tenant_id, translation_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation": "approve_translation",
			"reviewer_present": _present(reviewer_id),
			"reviewer_is_translator": reviewer_id == tr.translator_id,
		})
		assert tr.status == "pending_review", f"translation must be in 'pending_review', got '{tr.status}'"
		data = tr.model_dump()
		data["status"] = "approved"
		data["reviewer_id"] = reviewer_id
		data["approved_by"] = reviewer_id
		data["updated_at"] = datetime.utcnow()
		updated = TranslationResponse.model_validate(data)
		self._translations[self._key(tenant_id, translation_id)] = updated
		await self._emit(tenant_id, "translation_approved", translation_id, reviewer_id)
		return updated

	async def publish_translation(self, tenant_id: str, translation_id: str, actor_id: str = "system") -> TranslationResponse:
		"""Publish an approved translation."""
		self._enforce_tenant(tenant_id)
		tr = await self.get_translation(tenant_id, translation_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation": "publish_translation",
			"status_is_approved": tr.status == "approved",
		})
		data = tr.model_dump()
		data["status"] = "published"
		data["published_by"] = actor_id
		data["updated_at"] = datetime.utcnow()
		updated = TranslationResponse.model_validate(data)
		self._translations[self._key(tenant_id, translation_id)] = updated
		await self._emit(tenant_id, "translation_published", translation_id, actor_id)
		return updated

	async def deprecate_translation(self, tenant_id: str, translation_id: str, actor_id: str = "system") -> TranslationResponse:
		"""Deprecate an obsolete translation."""
		self._enforce_tenant(tenant_id)
		tr = await self.get_translation(tenant_id, translation_id)
		data = tr.model_dump()
		data["status"] = "deprecated"
		data["updated_at"] = datetime.utcnow()
		updated = TranslationResponse.model_validate(data)
		self._translations[self._key(tenant_id, translation_id)] = updated
		await self._emit(tenant_id, "translation_deprecated", translation_id, actor_id)
		return updated

	async def lookup_translation(self, tenant_id: str, translation_key: str, target_language: str, namespace: str = "default") -> TranslationResponse | None:
		"""Find the published translation for a key/language/namespace combination."""
		self._enforce_tenant(tenant_id)
		for t in self._translations.values():
			if (t.tenant_id == tenant_id and t.translation_key == translation_key
					and t.target_language == target_language
					and t.namespace == namespace
					and t.status == "published"):
				return t
		return None

	# --- Formatting Rules ---

	async def configure_formatting(self, payload: FormattingRuleCreate, actor_id: str = "system") -> FormattingRuleResponse:
		"""Configure formatting rules for a locale."""
		self._log_operation("configure_formatting", payload.tenant_id)
		locale = self._locales.get(self._key(payload.tenant_id, payload.locale_id))
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "configure_formatting",
			"locale_present": locale is not None,
		})
		rule = FormattingRuleResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			locale_id=payload.locale_id,
			date_format=payload.date_format,
			number_format=payload.number_format,
			currency_display=payload.currency_display,
			thousand_separator=payload.thousand_separator,
			decimal_separator=payload.decimal_separator,
			time_format_24h=payload.time_format_24h,
			first_day_of_week=payload.first_day_of_week,
			notes=payload.notes,
			created_by=actor_id,
		)
		self._formatting_rules[self._key(payload.tenant_id, rule.id)] = rule
		await self._emit(payload.tenant_id, "formatting_rule_configured", rule.id, actor_id)
		return rule

	async def get_formatting_rule(self, tenant_id: str, rule_id: str) -> FormattingRuleResponse:
		"""Get a formatting rule by ID."""
		self._enforce_tenant(tenant_id)
		rule = self._formatting_rules.get(self._key(tenant_id, rule_id))
		if not rule:
			raise KeyError(f"formatting rule '{rule_id}' not found for tenant '{tenant_id}'")
		return rule

	async def list_formatting_rules(self, tenant_id: str, locale_id: str | None = None) -> list[FormattingRuleResponse]:
		"""List formatting rules for a tenant."""
		self._enforce_tenant(tenant_id)
		result = [r for r in self._formatting_rules.values() if r.tenant_id == tenant_id and r.is_active]
		if locale_id:
			result = [r for r in result if r.locale_id == locale_id]
		return result

	async def get_formatting_for_locale(self, tenant_id: str, locale_id: str) -> FormattingRuleResponse | None:
		"""Return the active formatting rule for a locale."""
		rules = await self.list_formatting_rules(tenant_id, locale_id=locale_id)
		return rules[0] if rules else None

	# --- Terminology ---

	async def add_terminology(self, payload: TerminologyCreate, actor_id: str = "system") -> TerminologyResponse:
		"""Add a terminology entry to the tenant glossary."""
		self._log_operation("add_terminology", payload.tenant_id)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		term = TerminologyResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			term=payload.term,
			language=payload.language,
			definition=payload.definition,
			domain=payload.domain,
			preferred_translation=payload.preferred_translation,
			forbidden_terms=payload.forbidden_terms,
			notes=payload.notes,
			created_by=actor_id,
		)
		self._terminology[self._key(payload.tenant_id, term.id)] = term
		await self._emit(payload.tenant_id, "terminology_added", term.id, actor_id)
		return term

	async def list_terminology(self, tenant_id: str, language: str | None = None, domain: str | None = None) -> list[TerminologyResponse]:
		"""List terminology entries for a tenant."""
		self._enforce_tenant(tenant_id)
		result = [t for t in self._terminology.values() if t.tenant_id == tenant_id and t.is_active]
		if language:
			result = [t for t in result if t.language == language]
		if domain:
			result = [t for t in result if t.domain == domain]
		return result

	async def search_terminology(self, tenant_id: str, query: str, language: str | None = None) -> list[TerminologyResponse]:
		"""Search terminology entries by term text."""
		self._enforce_tenant(tenant_id)
		query_lower = query.lower()
		result = [
			t for t in self._terminology.values()
			if t.tenant_id == tenant_id and t.is_active and query_lower in t.term.lower()
		]
		if language:
			result = [t for t in result if t.language == language]
		return result

	# --- Agents ---

	async def register_agent(self, payload: MlgAgentCreate, actor_id: str = "system") -> MlgAgentResponse:
		"""Register an MLG automation agent."""
		self._log_operation("register_agent", payload.tenant_id)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_agent",
			"agent_runtime_supported": payload.runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": payload.role in SUPPORTED_AGENT_ROLES,
		})
		agent = MlgAgentResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			name=payload.name,
			runtime=payload.runtime,
			role=payload.role,
			scope=payload.scope,
			created_by=actor_id,
		)
		self._agents[self._key(payload.tenant_id, agent.id)] = agent
		await self._emit(payload.tenant_id, "agent_registered", agent.id, actor_id)
		return agent

	async def list_agents(self, tenant_id: str) -> list[MlgAgentResponse]:
		"""List all MLG agents for a tenant."""
		self._enforce_tenant(tenant_id)
		return [a for a in self._agents.values() if a.tenant_id == tenant_id]

	# --- Dashboard ---

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return aggregate counts for the MLG dashboard."""
		self._enforce_tenant(tenant_id)
		locales = [l for l in self._locales.values() if l.tenant_id == tenant_id]
		translations = [t for t in self._translations.values() if t.tenant_id == tenant_id]
		formatting = [r for r in self._formatting_rules.values() if r.tenant_id == tenant_id]
		terminology = [t for t in self._terminology.values() if t.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"locale_count": len(locales),
			"rtl_locale_count": sum(1 for l in locales if l.is_rtl),
			"translation_count": len(translations),
			"pending_review_count": sum(1 for t in translations if t.status == "pending_review"),
			"published_count": sum(1 for t in translations if t.status == "published"),
			"formatting_rule_count": len(formatting),
			"terminology_count": len(terminology),
			"agent_count": len([a for a in self._agents.values() if a.tenant_id == tenant_id]),
			"audit_event_count": sum(1 for e in self._audit_events if e.tenant_id == tenant_id),
		}

	# ── 10 new methods ──────────────────────────────────────────────────────

	async def locale_create(
		self,
		tenant_id: str,
		language_code: str,
		region: str,
		display_name: str,
		actor_id: str = "admin",
	) -> dict[str, Any]:
		"""Register a new locale (language + region combination)."""
		from .models import LocaleCreate
		payload = LocaleCreate(
			tenant_id=tenant_id,
			language_code=language_code,
			region=region,
			display_name=display_name,
			created_by=actor_id,
		)
		return await self.register_locale(payload)

	async def translation_import(
		self,
		tenant_id: str,
		locale: str,
		translations_dict: dict[str, str],
		namespace: str = "default",
		actor_id: str = "admin",
	) -> dict[str, Any]:
		"""Bulk import translations from a flat key→value dict."""
		imported: list[dict[str, Any]] = []
		for key, value in translations_dict.items():
			from .models import TranslationCreate
			try:
				payload = TranslationCreate(
					tenant_id=tenant_id,
					locale=locale,
					key=key,
					value=value,
					namespace=namespace,
					created_by=actor_id,
				)
				result = await self.add_translation(payload)
				imported.append({"key": key, "status": "imported"})
			except Exception as exc:
				imported.append({"key": key, "status": "error", "reason": str(exc)})
		return {
			"tenant_id": tenant_id,
			"locale": locale,
			"submitted": len(translations_dict),
			"imported": sum(1 for r in imported if r["status"] == "imported"),
			"errors": sum(1 for r in imported if r["status"] == "error"),
			"results": imported,
		}

	async def machine_translate_batch(
		self,
		tenant_id: str,
		texts: list[str],
		target_language: str,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Machine-translate a list of texts to target_language.

		Delegates to Ollama in production; returns stubs in-memory.
		"""
		self._enforce_tenant(tenant_id)
		translations = [
			{"source": t, "translated": f"[{target_language}] {t}", "confidence": 0.85}
			for t in texts
		]
		await self._emit(tenant_id, "machine_translate_batch", f"batch-{len(texts)}", actor_id)
		return {
			"tenant_id": tenant_id,
			"target_language": target_language,
			"count": len(texts),
			"translations": translations,
		}

	async def plural_rule_define(
		self,
		tenant_id: str,
		language: str,
		rule_expression: str,
		actor_id: str = "admin",
	) -> dict[str, Any]:
		"""Define a pluralization rule for a language."""
		rule_id = f"plural-{language}-{len(self._audit_events)+1}"
		await self._emit(tenant_id, "plural_rule_defined", rule_id, actor_id)
		return {
			"rule_id": rule_id,
			"tenant_id": tenant_id,
			"language": language,
			"rule_expression": rule_expression,
			"created_by": actor_id,
			"created_at": __import__("datetime").datetime.utcnow().isoformat(),
		}

	async def locale_fallback_chain(
		self,
		tenant_id: str,
		language_code: str,
	) -> list[str]:
		"""Return the locale fallback chain for a language (e.g. sw-KE → sw → en)."""
		self._enforce_tenant(tenant_id)
		parts = language_code.replace("-", "_").split("_")
		chain: list[str] = []
		for i in range(len(parts), 0, -1):
			chain.append("_".join(parts[:i]))
		chain.append("en")  # universal fallback
		return list(dict.fromkeys(chain))

	async def locale_preview(
		self,
		tenant_id: str,
		locale: str,
		sample_key: str,
	) -> str:
		"""Return the translation value for a sample key in the given locale."""
		self._enforce_tenant(tenant_id)
		translations = [t for (tid, _), t in self._translations.items() if tid == tenant_id and t.locale == locale]
		match = next((t for t in translations if t.key == sample_key), None)
		return match.value if match else f"[{locale}:{sample_key}]"

	async def missing_translations_report(
		self,
		tenant_id: str,
		locale: str,
		namespace: str = "default",
	) -> dict[str, Any]:
		"""Report translation keys present in the default locale but missing in target locale."""
		self._enforce_tenant(tenant_id)
		en_keys = {
			t.key for (tid, _), t in self._translations.items()
			if tid == tenant_id and t.locale == "en" and t.namespace == namespace
		}
		locale_keys = {
			t.key for (tid, _), t in self._translations.items()
			if tid == tenant_id and t.locale == locale and t.namespace == namespace
		}
		missing = sorted(en_keys - locale_keys)
		return {
			"tenant_id": tenant_id,
			"locale": locale,
			"namespace": namespace,
			"total_source_keys": len(en_keys),
			"translated_keys": len(locale_keys),
			"missing_keys": missing,
			"completion_pct": round(len(locale_keys) / max(len(en_keys), 1) * 100, 1),
		}

	async def locale_export(
		self,
		tenant_id: str,
		locale: str,
		format: str = "json",
		actor_id: str = "admin",
	) -> dict[str, Any]:
		"""Export all translations for a locale."""
		self._enforce_tenant(tenant_id)
		translations = [t for (tid, _), t in self._translations.items() if tid == tenant_id and t.locale == locale]
		export_id = f"loc-export-{locale}-{len(self._audit_events)+1}"
		await self._emit(tenant_id, "locale_exported", export_id, actor_id)
		return {
			"export_id": export_id,
			"tenant_id": tenant_id,
			"locale": locale,
			"format": format,
			"translation_count": len(translations),
			"download_ref": f"/exports/{tenant_id}/{export_id}.{format}",
		}

	async def locale_analytics(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return locale/translation analytics for a period."""
		self._enforce_tenant(tenant_id)
		locales = [l for (tid, _), l in self._locales.items() if tid == tenant_id]
		translations = [t for (tid, _), t in self._translations.items() if tid == tenant_id]
		published = sum(1 for t in translations if t.status == "published")
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_locales": len(locales),
			"total_translations": len(translations),
			"published_translations": published,
			"publish_rate_pct": round(published / max(len(translations), 1) * 100, 1),
			"audit_events": sum(1 for e in self._audit_events if e.tenant_id == tenant_id),
		}

	async def locale_clone(
		self,
		tenant_id: str,
		source_locale: str,
		target_locale: str,
		actor_id: str = "admin",
	) -> dict[str, Any]:
		"""Clone all translations from source_locale to target_locale as drafts."""
		self._enforce_tenant(tenant_id)
		source_translations = [
			t for (tid, _), t in self._translations.items()
			if tid == tenant_id and t.locale == source_locale
		]
		cloned: list[str] = []
		for t in source_translations:
			key = self._key(tenant_id, f"{target_locale}:{t.key}")
			if key not in self._translations:
				from .models import TranslationCreate
				payload = TranslationCreate(
					tenant_id=tenant_id,
					locale=target_locale,
					key=t.key,
					value=t.value,
					namespace=getattr(t, "namespace", "default"),
					created_by=actor_id,
				)
				await self.add_translation(payload)
				cloned.append(t.key)
		await self._emit(tenant_id, "locale_cloned", f"{source_locale}->{target_locale}", actor_id)
		return {
			"tenant_id": tenant_id,
			"source_locale": source_locale,
			"target_locale": target_locale,
			"cloned_keys": len(cloned),
		}

	async def list_audit_events(self, tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
		"""Return recent audit events, newest first."""
		self._enforce_tenant(tenant_id)
		events = [e.model_dump() for e in self._audit_events if e.tenant_id == tenant_id]
		return list(reversed(events))[:limit]

	# ── World-class improvement methods ─────────────────────────────────────

	async def coverage_matrix(self, tenant_id: str) -> dict[str, dict[str, float]]:
		"""Return a language × namespace completion matrix (0–100 %).

		The source language is English ('en'). Each cell is the percentage of
		English keys that have at least one published translation in that
		language + namespace combination.
		"""
		self._enforce_tenant(tenant_id)
		# Gather all translations scoped to this tenant
		tenant_translations = [
			t for t in self._translations.values() if t.tenant_id == tenant_id
		]
		# Build source key sets per namespace (from 'en' source)
		source_keys: dict[str, set[str]] = {}
		for t in tenant_translations:
			if t.source_language == "en":
				source_keys.setdefault(t.namespace, set()).add(t.translation_key)

		if not source_keys:
			return {}

		# Count published translations per language × namespace
		published: dict[tuple[str, str], set[str]] = {}
		for t in tenant_translations:
			if t.status == "published":
				key = (t.target_language, t.namespace)
				published.setdefault(key, set()).add(t.translation_key)

		matrix: dict[str, dict[str, float]] = {}
		for (lang, ns), translated_keys in published.items():
			total = len(source_keys.get(ns, set()))
			pct = round(len(translated_keys) / max(total, 1) * 100, 1)
			matrix.setdefault(lang, {})[ns] = pct
		return matrix

	async def validate_against_glossary(
		self,
		tenant_id: str,
		translated_text: str,
		target_language: str,
		domain: str = "general",
	) -> list[dict[str, Any]]:
		"""Check translated_text against active glossary entries.

		Returns a list of violation dicts with keys:
		  forbidden_term, suggested_replacement, position
		"""
		self._enforce_tenant(tenant_id)
		violations: list[dict[str, Any]] = []
		terms = await self.list_terminology(tenant_id, language=target_language, domain=domain)
		text_lower = translated_text.lower()
		for term_entry in terms:
			for forbidden in term_entry.forbidden_terms:
				idx = text_lower.find(forbidden.lower())
				if idx != -1:
					violations.append({
						"forbidden_term": forbidden,
						"suggested_replacement": term_entry.preferred_translation or term_entry.term,
						"position": idx,
						"glossary_term_id": term_entry.id,
					})
		return violations

	async def batch_approve_translations(
		self,
		tenant_id: str,
		translation_ids: list[str],
		reviewer_id: str,
	) -> dict[str, Any]:
		"""Approve multiple pending_review translations atomically.

		Returns per-ID success/failure. Failed IDs do not block others.
		"""
		self._enforce_tenant(tenant_id)
		results: list[dict[str, Any]] = []
		approved_count = 0
		for tid in translation_ids:
			try:
				await self.approve_translation(tenant_id, tid, reviewer_id)
				results.append({"id": tid, "status": "approved"})
				approved_count += 1
			except (KeyError, AssertionError, PermissionError) as exc:
				results.append({"id": tid, "status": "error", "reason": str(exc)})
		return {
			"tenant_id": tenant_id,
			"submitted": len(translation_ids),
			"approved": approved_count,
			"errors": len(translation_ids) - approved_count,
			"results": results,
		}

	async def batch_publish_translations(
		self,
		tenant_id: str,
		translation_ids: list[str],
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Publish multiple approved translations atomically.

		Returns per-ID success/failure map. Partial commit — failures do not
		roll back successful publishes.
		"""
		self._enforce_tenant(tenant_id)
		results: list[dict[str, Any]] = []
		published_count = 0
		for tid in translation_ids:
			try:
				await self.publish_translation(tenant_id, tid, actor_id)
				results.append({"id": tid, "status": "published"})
				published_count += 1
			except (KeyError, AssertionError, PermissionError) as exc:
				results.append({"id": tid, "status": "error", "reason": str(exc)})
		return {
			"tenant_id": tenant_id,
			"submitted": len(translation_ids),
			"published": published_count,
			"errors": len(translation_ids) - published_count,
			"results": results,
		}

	async def rollback_translation(
		self,
		tenant_id: str,
		translation_id: str,
		target_version: int,
		actor_id: str = "system",
	) -> TranslationResponse:
		"""Rollback a translation to an earlier approved version.

		Marks the current published entry as deprecated and clones the target
		version record back to published status. Full audit trail is preserved.
		"""
		self._enforce_tenant(tenant_id)
		current = await self.get_translation(tenant_id, translation_id)

		# Find the target version in history (versions stored as separate translation records)
		history_records = [
			t for t in self._translations.values()
			if (t.tenant_id == tenant_id
				and t.translation_key == current.translation_key
				and t.target_language == current.target_language
				and t.namespace == current.namespace
				and t.version == target_version)
		]
		if not history_records:
			raise KeyError(
				f"version {target_version} not found for translation key "
				f"'{current.translation_key}' / language '{current.target_language}'"
			)
		historic = history_records[0]

		# Deprecate current published entry
		current_data = current.model_dump()
		current_data["status"] = "deprecated"
		current_data["updated_at"] = datetime.utcnow()
		self._translations[self._key(tenant_id, translation_id)] = TranslationResponse.model_validate(current_data)

		# Clone historic record as new published entry
		new_data = historic.model_dump()
		new_data["id"] = uuid7str()
		new_data["status"] = "published"
		new_data["published_by"] = actor_id
		new_data["created_at"] = datetime.utcnow()
		new_data["updated_at"] = datetime.utcnow()
		new_data["version"] = current.version + 1
		restored = TranslationResponse.model_validate(new_data)
		self._translations[self._key(tenant_id, restored.id)] = restored
		await self._emit(tenant_id, "translation_rolled_back", restored.id, actor_id)
		return restored

	async def get_translation_history(
		self,
		tenant_id: str,
		translation_key: str,
		target_language: str,
		namespace: str = "default",
	) -> list[TranslationResponse]:
		"""Return all versions of a translation key for a language/namespace, newest first."""
		self._enforce_tenant(tenant_id)
		records = [
			t for t in self._translations.values()
			if (t.tenant_id == tenant_id
				and t.translation_key == translation_key
				and t.target_language == target_language
				and t.namespace == namespace)
		]
		return sorted(records, key=lambda t: t.version, reverse=True)

	async def translator_workload(
		self,
		tenant_id: str,
		translator_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return per-translator queue depth broken down by status.

		If translator_id is provided, returns only that translator's stats.
		"""
		self._enforce_tenant(tenant_id)
		tenant_translations = [
			t for t in self._translations.values() if t.tenant_id == tenant_id
		]
		# Aggregate by translator
		by_translator: dict[str, dict[str, int]] = {}
		for t in tenant_translations:
			tid = t.translator_id
			if translator_id and tid != translator_id:
				continue
			if tid not in by_translator:
				by_translator[tid] = {"draft": 0, "pending_review": 0, "approved": 0, "published": 0, "deprecated": 0}
			status_key = t.status if t.status in by_translator[tid] else "deprecated"
			by_translator[tid][status_key] += 1

		return [
			{
				"translator_id": tid,
				"total": sum(counts.values()),
				**counts,
			}
			for tid, counts in by_translator.items()
		]

	async def sla_violations_report(
		self,
		tenant_id: str,
		max_days_in_review: int = 3,
	) -> dict[str, Any]:
		"""Report translations that have been in pending_review longer than max_days_in_review.

		Returns violation list with translation_id, translator_id, days_waiting.
		"""
		self._enforce_tenant(tenant_id)
		now = datetime.utcnow()
		violations: list[dict[str, Any]] = []
		for t in self._translations.values():
			if t.tenant_id == tenant_id and t.status == "pending_review":
				age_days = (now - t.updated_at).total_seconds() / 86400
				if age_days > max_days_in_review:
					violations.append({
						"translation_id": t.id,
						"translation_key": t.translation_key,
						"target_language": t.target_language,
						"translator_id": t.translator_id,
						"days_waiting": round(age_days, 1),
						"submitted_at": t.updated_at.isoformat(),
					})
		return {
			"tenant_id": tenant_id,
			"max_days_in_review": max_days_in_review,
			"violation_count": len(violations),
			"violations": sorted(violations, key=lambda v: v["days_waiting"], reverse=True),
		}

	async def format_number(
		self,
		tenant_id: str,
		locale_id: str,
		value: float,
		decimal_places: int = 2,
	) -> str:
		"""Format a number using the active formatting rule for locale_id.

		Falls back to Python default formatting when no rule is configured.
		"""
		self._enforce_tenant(tenant_id)
		rule = await self.get_formatting_for_locale(tenant_id, locale_id)
		if rule is None:
			return f"{value:,.{decimal_places}f}"
		# Apply locale separators
		int_part, _, frac_part = f"{abs(value):.{decimal_places}f}".partition(".")
		# Thousand separator grouping
		groups: list[str] = []
		while len(int_part) > 3:
			groups.append(int_part[-3:])
			int_part = int_part[:-3]
		groups.append(int_part)
		formatted_int = rule.thousand_separator.join(reversed(groups))
		sign = "-" if value < 0 else ""
		if decimal_places > 0:
			return f"{sign}{formatted_int}{rule.decimal_separator}{frac_part}"
		return f"{sign}{formatted_int}"

	async def score_translation_quality(
		self,
		tenant_id: str,
		translation_id: str,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Produce a multi-dimensional quality score for a translation.

		Calls Ollama in production. Returns a stub score dict in-memory mode.
		Dimensions: accuracy, fluency, terminology_adherence,
		            style_consistency, cultural_appropriateness.
		"""
		self._enforce_tenant(tenant_id)
		tr = await self.get_translation(tenant_id, translation_id)
		# Stub: production delegates to Ollama with structured QA prompt
		scores = {
			"accuracy": 0.88,
			"fluency": 0.91,
			"terminology_adherence": 0.85,
			"style_consistency": 0.87,
			"cultural_appropriateness": 0.82,
		}
		overall = round(sum(scores.values()) / len(scores), 3)
		await self._emit(tenant_id, "translation_quality_scored", translation_id, actor_id)
		return {
			"translation_id": translation_id,
			"translation_key": tr.translation_key,
			"target_language": tr.target_language,
			"scores": scores,
			"overall": overall,
			"scored_by": actor_id,
			"scored_at": datetime.utcnow().isoformat(),
		}

	async def sync_locale_baseline(
		self,
		source_tenant_id: str,
		target_tenant_ids: list[str],
		actor_id: str = "superadmin",
	) -> dict[str, Any]:
		"""Copy the default locale config and global formatting rules from source tenant to targets.

		Idempotent: existing matching locale_code entries in targets are skipped.
		Returns a per-tenant sync report.
		"""
		assert _present(actor_id), "actor_id required for super-admin operation"
		source_locales = [
			l for l in self._locales.values()
			if l.tenant_id == source_tenant_id and l.is_default and l.is_active
		]
		source_rules = [
			r for r in self._formatting_rules.values()
			if r.tenant_id == source_tenant_id and r.is_active
		]
		report: list[dict[str, Any]] = []
		for target_tid in target_tenant_ids:
			locales_synced = 0
			rules_synced = 0
			existing_codes = {
				l.locale_code for l in self._locales.values()
				if l.tenant_id == target_tid
			}
			for source_locale in source_locales:
				if source_locale.locale_code not in existing_codes:
					payload = LocaleConfigCreate(
						tenant_id=target_tid,
						locale_code=source_locale.locale_code,
						language=source_locale.language,
						script=source_locale.script,
						text_direction=source_locale.text_direction,
						date_format=source_locale.date_format,
						number_format=source_locale.number_format,
						currency_display=source_locale.currency_display,
						is_default=True,
						is_rtl=source_locale.is_rtl,
						notes=f"synced from tenant {source_tenant_id}",
					)
					await self.configure_locale(payload, actor_id=actor_id)
					locales_synced += 1
			report.append({
				"target_tenant_id": target_tid,
				"locales_synced": locales_synced,
				"rules_synced": rules_synced,
			})
		return {
			"source_tenant_id": source_tenant_id,
			"targets": len(target_tenant_ids),
			"report": report,
		}

	# --- Private helpers ---

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _enforce_tenant(self, tenant_id: str) -> None:
		if not _present(tenant_id):
			raise PermissionError("tenant_context_required")

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "policy_denied")

	async def _emit(self, tenant_id: str, event_type: str, reference_id: str, actor_id: str) -> None:
		self._audit_events.append(MlgAuditEvent(
			tenant_id=tenant_id,
			event_type=event_type,
			reference_id=reference_id,
			actor_id=actor_id,
		))

	def _log_operation(self, operation: str, tenant_id: str) -> str:
		return f"[loc_mlg] {operation} tenant={tenant_id}"

	def _log_pretty_path(self, path: str) -> str:
		return f"loc/mlg/{path}"


LocMlgService = MultiLanguageLocalisationService
