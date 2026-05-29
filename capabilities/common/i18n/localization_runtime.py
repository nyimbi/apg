"""Domain localization helpers for the I18N capability runtime."""

from __future__ import annotations

from .models import LocaleDefinition, TranslationEntry, TranslationStatus


class LocaleFallbackResolver:
	"""Resolve locale fallback chains without external dependencies."""

	def chain(
		self,
		locale_code: str,
		locales: list[LocaleDefinition],
		default_locale: str,
	) -> list[str]:
		by_code = {locale.locale_code: locale for locale in locales}
		chain = [locale_code]
		seen = {locale_code}
		current = by_code.get(locale_code)
		while current and current.fallback_locale and current.fallback_locale not in seen:
			chain.append(current.fallback_locale)
			seen.add(current.fallback_locale)
			current = by_code.get(current.fallback_locale)
		if default_locale not in seen:
			chain.append(default_locale)
		return chain


class TranslationMemoryMatcher:
	"""Find reusable reviewed or published translations."""

	def match(
		self,
		source_text: str,
		locale_code: str,
		translations: list[TranslationEntry],
	) -> TranslationEntry | None:
		for translation in sorted(translations, key=lambda item: item.updated_at, reverse=True):
			if translation.locale_code != locale_code:
				continue
			if translation.source_text != source_text:
				continue
			if translation.status in {TranslationStatus.REVIEWED, TranslationStatus.PUBLISHED}:
				return translation
		return None


class CoverageCalculator:
	"""Compute published translation coverage for a locale."""

	def coverage(
		self,
		locale_code: str,
		required_keys: list[str],
		translations: list[TranslationEntry],
	) -> dict[str, object]:
		required = list(dict.fromkeys(required_keys))
		published_keys = {
			translation.key
			for translation in translations
			if translation.locale_code == locale_code
			and translation.status == TranslationStatus.PUBLISHED
		}
		missing = [key for key in required if key not in published_keys]
		total = len(required)
		published = total - len(missing)
		percent = 100.0 if total == 0 else round((published / total) * 100, 2)
		return {
			"total_key_count": total,
			"published_key_count": published,
			"missing_keys": missing,
			"coverage_percent": percent,
		}
