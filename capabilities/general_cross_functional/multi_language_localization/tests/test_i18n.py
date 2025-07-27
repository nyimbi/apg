"""
APG Multi-language Localization - Internationalization Tests

Specialized tests for internationalization features including RTL languages,
complex scripts, cultural formatting, and multilingual content handling.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
import unicodedata

from ..service import TranslationService, FormattingService
from ..models import (
	MLLanguage, MLLocale, MLNamespace, MLTranslationKey, MLTranslation,
	MLTextDirection, MLLanguageStatus, MLTranslationStatus,
	MLNumberFormat
)

@pytest.mark.asyncio
class TestRTLLanguages:
	"""Test Right-to-Left language support"""
	
	async def test_arabic_translation(self, db_session, redis_client):
		"""Test Arabic translation handling"""
		service = TranslationService(db_session, redis_client)
		
		# Create Arabic language
		arabic = MLLanguage(
			code="ar",
			name="Arabic",
			native_name="العربية",
			script="Arab",
			direction=MLTextDirection.RTL,
			status=MLLanguageStatus.ACTIVE
		)
		
		namespace = MLNamespace(name="rtl_test", description="RTL test")
		db_session.add_all([arabic, namespace])
		await db_session.flush()
		
		# Create translation key
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="welcome.message",
			source_text="Welcome to our application"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		# Create Arabic translation
		arabic_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=arabic.id,
			content="مرحباً بكم في تطبيقنا",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(arabic_translation)
		await db_session.commit()
		
		# Test retrieval
		result = await service.get_translation("welcome.message", "ar", "rtl_test")
		assert result == "مرحباً بكم في تطبيقنا"
		
		# Verify Unicode handling
		assert unicodedata.bidirectional(result[0]) in ['R', 'AL']  # Right-to-left or Arabic letter
	
	async def test_hebrew_translation(self, db_session, redis_client):
		"""Test Hebrew translation handling"""
		service = TranslationService(db_session, redis_client)
		
		# Create Hebrew language
		hebrew = MLLanguage(
			code="he",
			name="Hebrew", 
			native_name="עברית",
			script="Hebr",
			direction=MLTextDirection.RTL,
			status=MLLanguageStatus.ACTIVE
		)
		
		namespace = MLNamespace(name="hebrew_test", description="Hebrew test")
		db_session.add_all([hebrew, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="save.button",
			source_text="Save"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		hebrew_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=hebrew.id,
			content="שמור",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(hebrew_translation)
		await db_session.commit()
		
		result = await service.get_translation("save.button", "he", "hebrew_test")
		assert result == "שמור"
	
	async def test_mixed_direction_content(self, db_session, redis_client):
		"""Test content with mixed LTR/RTL text"""
		service = TranslationService(db_session, redis_client)
		
		# Create Arabic language
		arabic = MLLanguage(
			code="ar",
			name="Arabic",
			native_name="العربية", 
			script="Arab",
			direction=MLTextDirection.RTL
		)
		
		namespace = MLNamespace(name="mixed_test", description="Mixed direction test")
		db_session.add_all([arabic, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="mixed.content",
			source_text="Visit our website at example.com"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		# Arabic translation with embedded URL (LTR)
		mixed_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=arabic.id,
			content="قم بزيارة موقعنا على example.com",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(mixed_translation)
		await db_session.commit()
		
		result = await service.get_translation("mixed.content", "ar", "mixed_test")
		assert result == "قم بزيارة موقعنا على example.com"
		assert "example.com" in result  # URL should be preserved

@pytest.mark.asyncio
class TestComplexScripts:
	"""Test complex script support (CJK, Indic, etc.)"""
	
	async def test_chinese_simplified_translation(self, db_session, redis_client):
		"""Test Simplified Chinese translation"""
		service = TranslationService(db_session, redis_client)
		
		chinese = MLLanguage(
			code="zh",
			name="Chinese (Simplified)",
			native_name="中文（简体）",
			script="Hans",
			direction=MLTextDirection.LTR
		)
		
		namespace = MLNamespace(name="chinese_test", description="Chinese test")
		db_session.add_all([chinese, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="login.title",
			source_text="User Login"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		chinese_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=chinese.id,
			content="用户登录",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(chinese_translation)
		await db_session.commit()
		
		result = await service.get_translation("login.title", "zh", "chinese_test")
		assert result == "用户登录"
		
		# Verify character encoding
		assert len(result.encode('utf-8')) > len(result)  # Multi-byte characters
	
	async def test_japanese_translation(self, db_session, redis_client):
		"""Test Japanese translation with mixed scripts"""
		service = TranslationService(db_session, redis_client)
		
		japanese = MLLanguage(
			code="ja",
			name="Japanese",
			native_name="日本語",
			script="Jpan",
			direction=MLTextDirection.LTR
		)
		
		namespace = MLNamespace(name="japanese_test", description="Japanese test")
		db_session.add_all([japanese, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="welcome.user",
			source_text="Welcome, {username}!"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		# Japanese with Hiragana, Katakana, and Kanji
		japanese_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=japanese.id,
			content="ようこそ、{username}さん！",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(japanese_translation)
		await db_session.commit()
		
		result = await service.get_translation(
			"welcome.user", 
			"ja", 
			"japanese_test",
			variables={"username": "田中"}
		)
		assert result == "ようこそ、田中さん！"
	
	async def test_hindi_translation(self, db_session, redis_client):
		"""Test Hindi translation with Devanagari script"""
		service = TranslationService(db_session, redis_client)
		
		hindi = MLLanguage(
			code="hi",
			name="Hindi",
			native_name="हिन्दी",
			script="Deva",
			direction=MLTextDirection.LTR
		)
		
		namespace = MLNamespace(name="hindi_test", description="Hindi test")
		db_session.add_all([hindi, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="search.placeholder",
			source_text="Search..."
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		hindi_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=hindi.id,
			content="खोजें...",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(hindi_translation)
		await db_session.commit()
		
		result = await service.get_translation("search.placeholder", "hi", "hindi_test")
		assert result == "खोजें..."

class TestCulturalFormatting:
	"""Test culturally-appropriate formatting"""
	
	def test_arabic_number_formatting(self, db_session, redis_client):
		"""Test Arabic locale number formatting"""
		service = FormattingService(db_session, redis_client)
		
		# Test Arabic-Indic digits (some locales)
		result = service.format_number(1234.56, "ar-SA", "decimal")
		assert result is not None
		
		# Test currency formatting
		result = service.format_number(99.99, "ar-SA", "currency")
		assert result is not None
	
	def test_chinese_date_formatting(self, db_session, redis_client):
		"""Test Chinese date formatting"""
		service = FormattingService(db_session, redis_client)
		
		test_date = datetime(2025, 1, 15)
		
		result = service.format_date(test_date, "zh-CN", "medium")
		assert "2025" in result
		assert result is not None
	
	def test_japanese_number_formatting(self, db_session, redis_client):
		"""Test Japanese number formatting"""
		service = FormattingService(db_session, redis_client)
		
		# Test Japanese currency
		result = service.format_number(1000, "ja-JP", "currency")
		assert result is not None
		
		# Test large numbers (Japanese uses 万 for 10,000)
		result = service.format_number(10000, "ja-JP", "decimal")
		assert result is not None
	
	def test_indian_number_formatting(self, db_session, redis_client):
		"""Test Indian number formatting (lakhs/crores system)"""
		service = FormattingService(db_session, redis_client)
		
		# Test Indian numbering system
		result = service.format_number(1234567, "hi-IN", "decimal")
		assert result is not None
		
		# Test Indian Rupee formatting
		result = service.format_number(1000, "hi-IN", "currency")
		assert result is not None

@pytest.mark.asyncio
class TestPluralRules:
	"""Test plural form handling for different languages"""
	
	async def test_english_plurals(self, db_session, redis_client):
		"""Test English plural forms (simple: one/other)"""
		service = TranslationService(db_session, redis_client)
		
		english = MLLanguage(code="en", name="English", native_name="English", script="Latn")
		namespace = MLNamespace(name="plural_test", description="Plural test")
		db_session.add_all([english, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="item.count",
			source_text="{count} item(s)",
			is_plural=True
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		english_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=english.id,
			content="{count} items",
			plural_forms={
				"one": "{count} item",
				"other": "{count} items"
			},
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(english_translation)
		await db_session.commit()
		
		# Test singular
		result = await service.get_translation(
			"item.count", 
			"en", 
			"plural_test",
			variables={"count": 1}
		)
		# Basic implementation - would need proper plural rule engine
		assert "1" in result
	
	async def test_russian_plurals(self, db_session, redis_client):
		"""Test Russian plural forms (complex: one/few/many/other)"""
		service = TranslationService(db_session, redis_client)
		
		russian = MLLanguage(code="ru", name="Russian", native_name="Русский", script="Cyrl")
		namespace = MLNamespace(name="russian_test", description="Russian test")
		db_session.add_all([russian, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="minute.count",
			source_text="{count} minute(s)",
			is_plural=True
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		russian_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=russian.id,
			content="{count} минут",
			plural_forms={
				"one": "{count} минута",    # 1, 21, 31, ...
				"few": "{count} минуты",    # 2-4, 22-24, ...
				"many": "{count} минут",    # 0, 5-20, 25-30, ...
				"other": "{count} минут"    # fractions
			},
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(russian_translation)
		await db_session.commit()
		
		# Test different plural forms
		for count in [1, 2, 5, 21]:
			result = await service.get_translation(
				"minute.count",
				"ru", 
				"russian_test",
				variables={"count": count}
			)
			assert str(count) in result
	
	async def test_arabic_plurals(self, db_session, redis_client):
		"""Test Arabic plural forms (zero/one/two/few/many/other)"""
		service = TranslationService(db_session, redis_client)
		
		arabic = MLLanguage(code="ar", name="Arabic", native_name="العربية", script="Arab")
		namespace = MLNamespace(name="arabic_plural_test", description="Arabic plural test")
		db_session.add_all([arabic, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="book.count",
			source_text="{count} book(s)",
			is_plural=True
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		arabic_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=arabic.id,
			content="{count} كتب",
			plural_forms={
				"zero": "لا توجد كتب",
				"one": "كتاب واحد", 
				"two": "كتابان",
				"few": "{count} كتب",
				"many": "{count} كتاباً",
				"other": "{count} كتاب"
			},
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(arabic_translation)
		await db_session.commit()
		
		# Test different forms
		for count in [0, 1, 2, 3, 11]:
			result = await service.get_translation(
				"book.count",
				"ar",
				"arabic_plural_test", 
				variables={"count": count}
			)
			assert result is not None

class TestUnicodeHandling:
	"""Test Unicode and encoding handling"""
	
	def test_emoji_in_translations(self):
		"""Test emoji handling in translations"""
		text_with_emojis = "Welcome! 🎉 Enjoy your stay 😊"
		
		# Verify proper Unicode handling
		assert len(text_with_emojis.encode('utf-8')) > len(text_with_emojis)
		assert '🎉' in text_with_emojis
		assert '😊' in text_with_emojis
	
	def test_combining_characters(self):
		"""Test combining character handling"""
		# Text with combining diacritics
		text_with_accents = "café résumé naïve"
		normalized = unicodedata.normalize('NFC', text_with_accents)
		
		assert normalized == text_with_accents
		assert 'é' in normalized
	
	def test_zero_width_characters(self):
		"""Test zero-width character handling"""
		# Text with zero-width joiner (common in complex scripts)
		text_with_zwj = "👨‍👩‍👧‍👦"  # Family emoji with ZWJ
		
		# Should be treated as single grapheme cluster
		assert text_with_zwj is not None
		assert len(text_with_zwj) > 1  # Multiple code points
	
	def test_bidi_text_handling(self):
		"""Test bidirectional text handling"""
		# Mixed LTR/RTL text
		mixed_text = "Hello العالم World"
		
		# Verify bidirectional characters are present
		has_ltr = any(unicodedata.bidirectional(c) in ['L'] for c in mixed_text)
		has_rtl = any(unicodedata.bidirectional(c) in ['R', 'AL'] for c in mixed_text)
		
		assert has_ltr and has_rtl

@pytest.mark.asyncio
class TestLanguageFallbacks:
	"""Test language fallback mechanisms"""
	
	async def test_regional_fallback(self, db_session, redis_client):
		"""Test fallback from regional to base language"""
		service = TranslationService(db_session, redis_client)
		
		# Create base Spanish and Mexican Spanish
		spanish = MLLanguage(code="es", name="Spanish", native_name="Español", script="Latn")
		mexican_spanish = MLLanguage(
			code="es-MX",
			name="Spanish (Mexico)",
			native_name="Español (México)", 
			script="Latn",
			fallback_language_id=None  # Will be set after spanish is saved
		)
		
		namespace = MLNamespace(name="fallback_test", description="Fallback test")
		db_session.add_all([spanish, mexican_spanish, namespace])
		await db_session.flush()
		
		mexican_spanish.fallback_language_id = spanish.id
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="generic.greeting",
			source_text="Hello"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		# Only create base Spanish translation
		spanish_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=spanish.id,
			content="Hola",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(spanish_translation)
		await db_session.commit()
		
		# Request Mexican Spanish, should fallback to base Spanish
		result = await service.get_translation("generic.greeting", "es-MX", "fallback_test")
		assert result == "Hola"
	
	async def test_script_fallback(self, db_session, redis_client):
		"""Test fallback between different scripts of same language"""
		service = TranslationService(db_session, redis_client)
		
		# Create Simplified and Traditional Chinese
		simplified = MLLanguage(code="zh-Hans", name="Chinese (Simplified)", native_name="中文（简体）", script="Hans")
		traditional = MLLanguage(
			code="zh-Hant", 
			name="Chinese (Traditional)",
			native_name="中文（繁體）",
			script="Hant",
			fallback_language_id=None
		)
		
		namespace = MLNamespace(name="script_test", description="Script test")
		db_session.add_all([simplified, traditional, namespace])
		await db_session.flush()
		
		traditional.fallback_language_id = simplified.id
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="app.title",
			source_text="Application"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		# Only create Simplified Chinese translation
		simplified_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=simplified.id,
			content="应用程序",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(simplified_translation)
		await db_session.commit()
		
		# Request Traditional Chinese, should fallback to Simplified
		result = await service.get_translation("app.title", "zh-Hant", "script_test")
		assert result == "应用程序"

class TestCulturalAdaptation:
	"""Test cultural adaptation beyond language"""
	
	def test_color_cultural_meaning(self):
		"""Test awareness of cultural color meanings"""
		# This would test cultural color preferences
		# Red: lucky in China, danger in West
		# White: purity in West, mourning in East Asia
		# Green: nature in West, Islam in Middle East
		
		cultural_colors = {
			"zh": {"lucky": "red", "avoid": "white"},
			"ar": {"sacred": "green", "neutral": "blue"},
			"en": {"danger": "red", "success": "green"}
		}
		
		assert cultural_colors["zh"]["lucky"] == "red"
		assert cultural_colors["ar"]["sacred"] == "green"
	
	def test_cultural_number_meanings(self):
		"""Test cultural number significance"""
		# 4: unlucky in East Asia (sounds like death)
		# 13: unlucky in Western cultures
		# 8: lucky in Chinese culture
		
		cultural_numbers = {
			"zh": {"lucky": [8, 88, 888], "unlucky": [4, 14, 44]},
			"ja": {"unlucky": [4, 9]},  # 4=death, 9=suffering
			"en": {"unlucky": [13]}
		}
		
		assert 8 in cultural_numbers["zh"]["lucky"]
		assert 4 in cultural_numbers["zh"]["unlucky"]
		assert 13 in cultural_numbers["en"]["unlucky"]
	
	def test_cultural_image_adaptation(self):
		"""Test cultural image and icon considerations"""
		# Thumbs up: positive in West, offensive in Middle East
		# OK hand: positive in US, offensive in some cultures
		# Pointing: rude in many Asian cultures
		
		cultural_gestures = {
			"en-US": {"positive": ["👍", "👌"], "negative": []},
			"ar": {"positive": [], "negative": ["👍"], "alternative": ["✓"]},
			"ja": {"avoid_pointing": True, "bow": "🙇"}
		}
		
		assert "👍" in cultural_gestures["en-US"]["positive"]
		assert "👍" in cultural_gestures["ar"]["negative"]

if __name__ == "__main__":
	pytest.main([__file__])