"""
Extensions for I18nService — adds 20 async methods to reach 40+ total.

Categories added:
  locale_create / translation_key_add / translation_value_set /
  plural_form / date_format / number_format / currency_format /
  rtl_support / font_support / translation_export / translation_import /
  machine_translate / review_translation / locale_analytics /
  missing_keys_report / bulk_create / bulk_update / bulk_delete /
  health_check / compliance_check

Pattern: in-memory stores, async throughout, audit events on every state change.
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


# RTL locales as per Unicode CLDR
_RTL_LOCALES: frozenset[str] = frozenset({
	"ar", "arc", "dv", "fa", "ha", "he", "khw", "ks", "ku", "ps",
	"ur", "yi", "ar-*", "he-*", "fa-*",
})

# Machine-translation stub vocabulary (deterministic for tests)
_MT_STUB: dict[str, str] = {
	"en:fr": "Traduit automatiquement: ",
	"en:de": "Automatisch übersetzt: ",
	"en:es": "Traducido automáticamente: ",
	"en:sw": "Imetafsiriwa kiotomatiki: ",
}


class I18nServiceExtensions:
	"""
	Async extension mixin for I18nService.

	All public methods are async; helpers are sync.
	"""

	def _ext_init(self) -> None:
		"""Call from __init__ to initialise extension stores."""
		self._translation_keys: dict[str, dict[str, Any]] = {}   # key: tenant:key_name
		self._plural_forms: dict[str, dict[str, Any]] = {}        # key: tenant:locale:key
		self._date_formats: dict[str, dict[str, Any]] = {}        # key: tenant:locale
		self._number_formats: dict[str, dict[str, Any]] = {}
		self._currency_formats: dict[str, dict[str, Any]] = {}
		self._font_support: dict[str, dict[str, Any]] = {}        # key: tenant:locale
		self._ext_counter: count = count(1)  # type: ignore[type-arg]

	# ----------------------------------------------------------- locale_create

	async def locale_create(
		self,
		locale_id: str,
		tenant_id: str,
		locale_code: str,
		display_name: str,
		owner_id: str,
		fallback_locale: str = "en",
		timezone: str = "UTC",
	) -> dict[str, Any]:
		"""Create a new locale definition (async wrapper over I18nService.create_locale)."""
		if hasattr(self, "create_locale"):
			result = self.create_locale(  # type: ignore[attr-defined]
				locale_id=locale_id,
				tenant_id=tenant_id,
				locale_code=locale_code,
				display_name=display_name,
				owner_id=owner_id,
				fallback_locale=fallback_locale,
				timezone=timezone,
			)
			await self._emit_audit(tenant_id, "locale_ext_created", locale_id, f"Locale created: {locale_code}", owner_id)
			return result
		# standalone path
		record: dict[str, Any] = {
			"id": locale_id,
			"kind": "locale",
			"tenant_id": tenant_id,
			"locale_code": locale_code,
			"display_name": display_name,
			"owner_id": owner_id,
			"fallback_locale": fallback_locale,
			"timezone": timezone,
			"enabled": True,
			"created_at": _utc(),
		}
		self._locales[f"{tenant_id}:{locale_id}"] = record  # type: ignore[attr-defined]
		await self._emit_audit(tenant_id, "locale_ext_created", locale_id, f"Locale created: {locale_code}", owner_id)
		return record

	# ------------------------------------------------------ translation_key_add

	async def translation_key_add(
		self,
		tenant_id: str,
		key_name: str,
		description: str = "",
		source_locale: str = "en",
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Register a new i18n translation key."""
		tk_key = f"{tenant_id}:{key_name}"
		if tk_key in self._translation_keys:
			raise ValueError(f"translation_key_exists:{key_name}")
		record: dict[str, Any] = {
			"key": tk_key,
			"kind": "translation_key",
			"tenant_id": tenant_id,
			"key_name": key_name,
			"description": description,
			"source_locale": source_locale,
			"owner_id": owner_id,
			"locales_covered": [],
			"created_at": _utc(),
		}
		self._translation_keys[tk_key] = record
		await self._emit_audit(tenant_id, "translation_key_added", key_name, f"Key added: {key_name}", owner_id)
		return record

	# -------------------------------------------------- translation_value_set

	async def translation_value_set(
		self,
		tenant_id: str,
		key_name: str,
		locale: str,
		value: str,
		translator_id: str = "system",
		status: str = "draft",
	) -> dict[str, Any]:
		"""Set (or overwrite) the translated value for a key/locale pair."""
		tk_key = f"{tenant_id}:{key_name}"
		if tk_key not in self._translation_keys:
			await self.translation_key_add(tenant_id=tenant_id, key_name=key_name)
		entry_key = f"{tk_key}:{locale}"
		existing: dict[str, Any] | None = None
		if hasattr(self, "_translations"):
			existing = self._translations.get(entry_key)  # type: ignore[attr-defined]
		record: dict[str, Any] = {
			"entry_key": entry_key,
			"kind": "translation_value",
			"tenant_id": tenant_id,
			"key_name": key_name,
			"locale": locale,
			"value": value,
			"translator_id": translator_id,
			"status": status,
			"version": (existing.get("version", 0) + 1) if existing else 1,
			"created_at": existing.get("created_at", _utc()) if existing else _utc(),
			"updated_at": _utc(),
		}
		if hasattr(self, "_translations"):
			self._translations[entry_key] = record  # type: ignore[attr-defined]
		# Track locale coverage on the key record
		key_rec = self._translation_keys.get(tk_key, {})
		locales_covered: list[str] = key_rec.get("locales_covered", [])
		if locale not in locales_covered:
			locales_covered.append(locale)
			key_rec["locales_covered"] = locales_covered
		await self._emit_audit(tenant_id, "translation_value_set", key_name, f"Value set for {key_name}[{locale}]", translator_id)
		return record

	# ------------------------------------------------------- plural_form

	async def plural_form(
		self,
		tenant_id: str,
		key_name: str,
		locale: str,
		forms: dict[str, str],
		translator_id: str = "system",
	) -> dict[str, Any]:
		"""
		Set plural forms for a key.  `forms` maps CLDR plural categories to strings:
		{"one": "1 item", "other": "{n} items"}.
		"""
		valid_categories = {"zero", "one", "two", "few", "many", "other"}
		invalid = set(forms.keys()) - valid_categories
		if invalid:
			raise ValueError(f"invalid_plural_categories:{invalid}")
		pf_key = f"{tenant_id}:{locale}:{key_name}"
		record: dict[str, Any] = {
			"key": pf_key,
			"kind": "plural_form",
			"tenant_id": tenant_id,
			"key_name": key_name,
			"locale": locale,
			"forms": dict(forms),
			"translator_id": translator_id,
			"updated_at": _utc(),
		}
		self._plural_forms[pf_key] = record
		await self._emit_audit(tenant_id, "plural_form_set", key_name, f"Plural forms set for {key_name}[{locale}]", translator_id)
		return record

	# ------------------------------------------------------- format methods

	async def date_format(
		self,
		tenant_id: str,
		locale: str,
		short: str = "yyyy-MM-dd",
		medium: str = "MMM d, yyyy",
		long_: str = "MMMM d, yyyy",
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Configure date format patterns for a locale."""
		df_key = f"{tenant_id}:{locale}"
		record: dict[str, Any] = {
			"key": df_key,
			"kind": "date_format",
			"tenant_id": tenant_id,
			"locale": locale,
			"short": short,
			"medium": medium,
			"long": long_,
			"owner_id": owner_id,
			"updated_at": _utc(),
		}
		self._date_formats[df_key] = record
		await self._emit_audit(tenant_id, "date_format_set", locale, f"Date format configured for {locale}", owner_id)
		return record

	async def number_format(
		self,
		tenant_id: str,
		locale: str,
		decimal_separator: str = ".",
		thousands_separator: str = ",",
		decimal_places: int = 2,
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Configure number format for a locale."""
		nf_key = f"{tenant_id}:{locale}"
		record: dict[str, Any] = {
			"key": nf_key,
			"kind": "number_format",
			"tenant_id": tenant_id,
			"locale": locale,
			"decimal_separator": decimal_separator,
			"thousands_separator": thousands_separator,
			"decimal_places": decimal_places,
			"owner_id": owner_id,
			"updated_at": _utc(),
		}
		self._number_formats[nf_key] = record
		await self._emit_audit(tenant_id, "number_format_set", locale, f"Number format configured for {locale}", owner_id)
		return record

	async def currency_format(
		self,
		tenant_id: str,
		locale: str,
		currency_code: str = "USD",
		symbol: str = "$",
		symbol_position: str = "prefix",
		decimal_places: int = 2,
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Configure currency format for a locale."""
		if symbol_position not in {"prefix", "suffix"}:
			raise ValueError("symbol_position must be prefix or suffix")
		cf_key = f"{tenant_id}:{locale}"
		record: dict[str, Any] = {
			"key": cf_key,
			"kind": "currency_format",
			"tenant_id": tenant_id,
			"locale": locale,
			"currency_code": currency_code,
			"symbol": symbol,
			"symbol_position": symbol_position,
			"decimal_places": decimal_places,
			"owner_id": owner_id,
			"updated_at": _utc(),
		}
		self._currency_formats[cf_key] = record
		await self._emit_audit(tenant_id, "currency_format_set", locale, f"Currency format configured for {locale} ({currency_code})", owner_id)
		return record

	# -------------------------------------------- rtl / font support

	async def rtl_support(
		self,
		tenant_id: str,
		locale: str,
		enabled: bool | None = None,
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Query or override RTL support for a locale."""
		base_code = locale.split("-")[0].lower()
		auto_rtl = base_code in _RTL_LOCALES
		is_rtl = enabled if enabled is not None else auto_rtl
		result: dict[str, Any] = {
			"tenant_id": tenant_id,
			"locale": locale,
			"rtl": is_rtl,
			"auto_detected": enabled is None,
			"checked_at": _utc(),
		}
		await self._emit_audit(tenant_id, "rtl_support_checked", locale, f"RTL for {locale}: {is_rtl}", owner_id)
		return result

	async def font_support(
		self,
		tenant_id: str,
		locale: str,
		font_family: str,
		fallback_fonts: list[str] | None = None,
		unicode_ranges: list[str] | None = None,
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Register font support configuration for a locale."""
		fs_key = f"{tenant_id}:{locale}"
		record: dict[str, Any] = {
			"key": fs_key,
			"kind": "font_support",
			"tenant_id": tenant_id,
			"locale": locale,
			"font_family": font_family,
			"fallback_fonts": list(fallback_fonts or []),
			"unicode_ranges": list(unicode_ranges or []),
			"owner_id": owner_id,
			"updated_at": _utc(),
		}
		self._font_support[fs_key] = record
		await self._emit_audit(tenant_id, "font_support_registered", locale, f"Font {font_family} registered for {locale}", owner_id)
		return record

	# ----------------------------------------- translation_export / import

	async def translation_export(
		self,
		tenant_id: str,
		locale: str | None = None,
		fmt: str = "json",
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Export translations as JSON or CSV."""
		if not hasattr(self, "_translations"):
			return {"tenant_id": tenant_id, "format": fmt, "payload": "", "count": 0, "exported_at": _utc()}
		trans_store: dict[str, Any] = self._translations  # type: ignore[attr-defined]
		rows = [
			v for v in trans_store.values()
			if isinstance(v, dict) and v.get("tenant_id") == tenant_id
			and (locale is None or v.get("locale") == locale)
		]
		if fmt == "csv":
			buf = io.StringIO()
			if rows:
				writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
				writer.writeheader()
				writer.writerows(rows)
			payload = buf.getvalue()
			content_type = "text/csv"
		else:
			payload = json.dumps(rows, default=str, indent=2)
			content_type = "application/json"
		await self._emit_audit(tenant_id, "translations_exported", tenant_id, f"Translations exported as {fmt} ({len(rows)} entries)", actor_id)
		return {
			"tenant_id": tenant_id,
			"locale_filter": locale,
			"format": fmt,
			"content_type": content_type,
			"count": len(rows),
			"payload": payload,
			"exported_at": _utc(),
		}

	async def translation_import(
		self,
		tenant_id: str,
		locale: str,
		entries: list[dict[str, Any]],
		translator_id: str = "import",
		overwrite: bool = True,
	) -> dict[str, Any]:
		"""Import a batch of translation entries."""
		imported: list[str] = []
		skipped: list[str] = []
		errors: list[dict[str, Any]] = []
		for entry in entries:
			key_name = entry.get("key_name") or entry.get("key")
			value = entry.get("value")
			if not key_name or value is None:
				errors.append({"entry": entry, "error": "missing_key_name_or_value"})
				continue
			entry_key = f"{tenant_id}:{key_name}:{locale}"
			existing = None
			if hasattr(self, "_translations"):
				existing = self._translations.get(entry_key)  # type: ignore[attr-defined]
			if existing and not overwrite:
				skipped.append(str(key_name))
				continue
			await self.translation_value_set(
				tenant_id=tenant_id,
				key_name=str(key_name),
				locale=locale,
				value=str(value),
				translator_id=translator_id,
			)
			imported.append(str(key_name))
		await self._emit_audit(tenant_id, "translations_imported", tenant_id, f"Imported {len(imported)} translations for {locale}", translator_id)
		return {
			"tenant_id": tenant_id,
			"locale": locale,
			"imported": imported,
			"skipped": skipped,
			"errors": errors,
			"total": len(entries),
		}

	# ----------------------------------------------- machine_translate

	async def machine_translate(
		self,
		tenant_id: str,
		key_name: str,
		source_locale: str,
		target_locale: str,
		source_value: str,
		actor_id: str = "mt-engine",
	) -> dict[str, Any]:
		"""Produce a machine-translated value and store it as a draft translation."""
		pair_key = f"{source_locale}:{target_locale}"
		prefix = _MT_STUB.get(pair_key, f"[MT {target_locale}] ")
		translated_value = prefix + source_value
		result = await self.translation_value_set(
			tenant_id=tenant_id,
			key_name=key_name,
			locale=target_locale,
			value=translated_value,
			translator_id=actor_id,
			status="machine_draft",
		)
		await self._emit_audit(tenant_id, "machine_translated", key_name, f"MT: {source_locale}->{target_locale} for {key_name}", actor_id)
		return {**result, "source_locale": source_locale, "source_value": source_value, "mt_engine": "stub"}

	# --------------------------------------------- review_translation

	async def review_translation(
		self,
		tenant_id: str,
		key_name: str,
		locale: str,
		reviewer_id: str,
		approved: bool,
		notes: str = "",
	) -> dict[str, Any]:
		"""Approve or reject a translation; updates its status."""
		entry_key = f"{tenant_id}:{key_name}:{locale}"
		record: dict[str, Any] | None = None
		if hasattr(self, "_translations"):
			record = self._translations.get(entry_key)  # type: ignore[attr-defined]
		if record is None:
			raise ValueError(f"translation_not_found:{entry_key}")
		record["status"] = "reviewed" if approved else "rejected"
		record["reviewer_id"] = reviewer_id
		record["review_notes"] = notes
		record["reviewed_at"] = _utc()
		event = "translation_approved" if approved else "translation_rejected"
		await self._emit_audit(tenant_id, event, key_name, f"Translation {event} for {key_name}[{locale}]", reviewer_id)
		return dict(record)

	# ---------------------------------------- locale_analytics / missing_keys

	async def locale_analytics(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Compute per-locale coverage and quality stats."""
		if not hasattr(self, "_translations"):
			return {"tenant_id": tenant_id, "locales": {}, "generated_at": _utc()}
		trans_store: dict[str, Any] = self._translations  # type: ignore[attr-defined]
		rows = [
			v for v in trans_store.values()
			if isinstance(v, dict) and v.get("tenant_id") == tenant_id
		]
		by_locale: dict[str, dict[str, Any]] = {}
		for row in rows:
			loc = row.get("locale", "unknown")
			if loc not in by_locale:
				by_locale[loc] = {"total": 0, "reviewed": 0, "machine_draft": 0, "rejected": 0, "draft": 0}
			by_locale[loc]["total"] += 1
			status = row.get("status", "draft")
			if status in by_locale[loc]:
				by_locale[loc][status] += 1
		total_keys = len(self._translation_keys)
		for loc, stats in by_locale.items():
			stats["coverage_pct"] = round(stats["total"] / total_keys * 100, 1) if total_keys > 0 else 0.0
		return {
			"tenant_id": tenant_id,
			"total_keys": total_keys,
			"locales": by_locale,
			"generated_at": _utc(),
		}

	async def missing_keys_report(
		self,
		tenant_id: str,
		target_locale: str,
	) -> dict[str, Any]:
		"""List translation keys not yet translated into target_locale."""
		all_keys = {
			rec["key_name"]
			for key, rec in self._translation_keys.items()
			if key.startswith(f"{tenant_id}:")
		}
		translated_keys: set[str] = set()
		if hasattr(self, "_translations"):
			trans_store: dict[str, Any] = self._translations  # type: ignore[attr-defined]
			translated_keys = {
				v["key_name"]
				for v in trans_store.values()
				if isinstance(v, dict)
				and v.get("tenant_id") == tenant_id
				and v.get("locale") == target_locale
				and v.get("status") not in ("rejected",)
			}
		missing = sorted(all_keys - translated_keys)
		return {
			"tenant_id": tenant_id,
			"target_locale": target_locale,
			"total_keys": len(all_keys),
			"missing_count": len(missing),
			"missing_keys": missing,
			"coverage_pct": round((len(all_keys) - len(missing)) / len(all_keys) * 100, 1) if all_keys else 100.0,
			"generated_at": _utc(),
		}

	# ---------------------------------------------------------------- bulk ops

	async def bulk_add_translation_keys(
		self,
		tenant_id: str,
		keys: list[dict[str, Any]],
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Add multiple translation keys in one call."""
		added: list[str] = []
		errors: list[dict[str, Any]] = []
		for k in keys:
			try:
				result = await self.translation_key_add(
					tenant_id=tenant_id,
					key_name=k["key_name"],
					description=k.get("description", ""),
					source_locale=k.get("source_locale", "en"),
					owner_id=owner_id,
				)
				added.append(result["key_name"])
			except Exception as exc:
				errors.append({"key_name": k.get("key_name"), "error": str(exc)})
		return {"added": added, "errors": errors, "total": len(keys)}

	async def bulk_set_translation_values(
		self,
		tenant_id: str,
		locale: str,
		values: list[dict[str, Any]],
		translator_id: str = "system",
	) -> dict[str, Any]:
		"""Set multiple translation values for a locale at once."""
		set_keys: list[str] = []
		errors: list[dict[str, Any]] = []
		for v in values:
			try:
				result = await self.translation_value_set(
					tenant_id=tenant_id,
					key_name=v["key_name"],
					locale=locale,
					value=v["value"],
					translator_id=translator_id,
					status=v.get("status", "draft"),
				)
				set_keys.append(result["key_name"])
			except Exception as exc:
				errors.append({"key_name": v.get("key_name"), "error": str(exc)})
		return {"set": set_keys, "errors": errors, "total": len(values)}

	async def bulk_delete_translation_keys(
		self,
		tenant_id: str,
		key_names: list[str],
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Remove multiple translation keys and all their values."""
		deleted: list[str] = []
		errors: list[dict[str, Any]] = []
		for key_name in key_names:
			tk_key = f"{tenant_id}:{key_name}"
			if tk_key not in self._translation_keys:
				errors.append({"key_name": key_name, "error": "not_found"})
				continue
			del self._translation_keys[tk_key]
			# Remove all locale values for this key
			if hasattr(self, "_translations"):
				to_remove = [k for k in self._translations if k.startswith(f"{tk_key}:")]  # type: ignore[attr-defined]
				for k in to_remove:
					del self._translations[k]  # type: ignore[attr-defined]
			deleted.append(key_name)
		await self._emit_audit(tenant_id, "bulk_keys_deleted", tenant_id, f"Bulk deleted {len(deleted)} translation keys", actor_id)
		return {"deleted": deleted, "errors": errors, "total": len(key_names)}

	# --------------------------------------------------------------- health / compliance

	async def health_check(self) -> dict[str, Any]:
		"""Return operational status of the i18n service stores."""
		return {
			"status": "healthy",
			"translation_keys": len(self._translation_keys),
			"plural_forms": len(self._plural_forms),
			"date_formats": len(self._date_formats),
			"number_formats": len(self._number_formats),
			"currency_formats": len(self._currency_formats),
			"font_support_entries": len(self._font_support),
			"checked_at": _utc(),
		}

	async def compliance_check(self, tenant_id: str) -> dict[str, Any]:
		"""Verify all tenant locales have at least date, number, and currency formats configured."""
		issues: list[dict[str, Any]] = []
		locales_store = getattr(self, "_locales", {})
		for lkey, loc in locales_store.items():
			item = loc if isinstance(loc, dict) else loc.to_dict()
			if item.get("tenant_id") != tenant_id:
				continue
			locale = item.get("locale_code", "")
			fkey = f"{tenant_id}:{locale}"
			if fkey not in self._date_formats:
				issues.append({"locale": locale, "issue": "missing_date_format"})
			if fkey not in self._number_formats:
				issues.append({"locale": locale, "issue": "missing_number_format"})
			if fkey not in self._currency_formats:
				issues.append({"locale": locale, "issue": "missing_currency_format"})
		return {
			"tenant_id": tenant_id,
			"compliant": len(issues) == 0,
			"issues": issues,
			"checked_at": _utc(),
		}

	# ---------------------------------------------------------------- private

	async def _emit_audit(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
	) -> None:
		if hasattr(self, "_record_audit"):
			try:
				self._record_audit(  # type: ignore[attr-defined]
					tenant_id=tenant_id,
					subject_id=subject_id,
					event_type=event_type,
					actor=actor,
					decision="allow",
					metadata={"message": message},
				)
				return
			except TypeError:
				pass
		if not hasattr(self, "_ext_audit_store"):
			self._ext_audit_store: dict[str, dict[str, Any]] = {}
		ev_id = f"ext-{event_type}-{subject_id}-{next(self._ext_counter)}"
		self._ext_audit_store[ev_id] = {
			"id": ev_id,
			"tenant_id": tenant_id,
			"event_type": event_type,
			"subject_id": subject_id,
			"message": message,
			"actor": actor,
			"created_at": _utc(),
		}
