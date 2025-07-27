"""
APG Multi-language Localization - Service Tests

Integration and unit tests for localization services including translation,
language management, formatting, and content management.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from ..service import (
	TranslationService, LanguageManagementService, FormattingService,
	ContentManagementService, UserPreferenceService
)
from ..models import (
	MLLanguage, MLLocale, MLNamespace, MLTranslationKey, MLTranslation,
	MLTextDirection, MLLanguageStatus, MLTranslationStatus, MLTranslationType,
	MLContentType, MLLanguageCreate, MLLocaleCreate, MLTranslationKeyCreate
)
from . import SAMPLE_LANGUAGES, SAMPLE_TRANSLATION_KEYS

@pytest.mark.asyncio
class TestTranslationService:
	"""Test TranslationService functionality"""
	
	async def test_get_translation_basic(self, db_session, redis_client):
		"""Test basic translation retrieval"""
		service = TranslationService(db_session, redis_client)
		
		# Setup test data
		language = MLLanguage(code="es", name="Spanish", native_name="Español", script="Latn")
		namespace = MLNamespace(name="test", description="Test namespace")
		db_session.add_all([language, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="test.save",
			source_text="Save"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=language.id,
			content="Guardar",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(translation)
		await db_session.commit()
		
		# Test retrieval
		result = await service.get_translation("test.save", "es", "test")
		assert result == "Guardar"
	
	async def test_get_translation_with_variables(self, db_session, redis_client):
		"""Test translation with variable substitution"""
		service = TranslationService(db_session, redis_client)
		
		# Setup test data
		language = MLLanguage(code="en", name="English", native_name="English", script="Latn")
		namespace = MLNamespace(name="test", description="Test namespace")
		db_session.add_all([language, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="welcome.message",
			source_text="Welcome, {username}!"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=language.id,
			content="Welcome, {username}!",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(translation)
		await db_session.commit()
		
		# Test with variable substitution
		result = await service.get_translation(
			"welcome.message", 
			"en", 
			"test",
			variables={"username": "John"}
		)
		assert result == "Welcome, John!"
	
	async def test_get_translation_fallback(self, db_session, redis_client):
		"""Test fallback language functionality"""
		service = TranslationService(db_session, redis_client)
		
		# Setup languages with fallback
		english = MLLanguage(
			code="en", 
			name="English", 
			native_name="English", 
			script="Latn",
			is_default=True
		)
		spanish = MLLanguage(
			code="es",
			name="Spanish", 
			native_name="Español",
			script="Latn",
			fallback_language_id=None  # Will be set after english is saved
		)
		namespace = MLNamespace(name="test", description="Test namespace")
		
		db_session.add_all([english, spanish, namespace])
		await db_session.flush()
		
		# Set fallback relationship
		spanish.fallback_language_id = english.id
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="missing.translation",
			source_text="Missing Translation"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		# Only create English translation (Spanish missing)
		english_translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=english.id,
			content="Missing Translation",
			status=MLTranslationStatus.PUBLISHED
		)
		db_session.add(english_translation)
		await db_session.commit()
		
		# Test fallback to English when Spanish is missing
		result = await service.get_translation("missing.translation", "es", "test", fallback=True)
		assert result == "Missing Translation"
	
	async def test_get_translations_batch(self, db_session, redis_client):
		"""Test batch translation retrieval"""
		service = TranslationService(db_session, redis_client)
		
		# Setup test data
		language = MLLanguage(code="fr", name="French", native_name="Français", script="Latn")
		namespace = MLNamespace(name="test", description="Test namespace")
		db_session.add_all([language, namespace])
		await db_session.flush()
		
		# Create multiple translation keys and translations
		keys_data = [
			("save", "Save", "Enregistrer"),
			("cancel", "Cancel", "Annuler"),
			("delete", "Delete", "Supprimer")
		]
		
		for key_name, source, translation_text in keys_data:
			translation_key = MLTranslationKey(
				namespace_id=namespace.id,
				key=f"test.{key_name}",
				source_text=source
			)
			db_session.add(translation_key)
			await db_session.flush()
			
			translation = MLTranslation(
				translation_key_id=translation_key.id,
				language_id=language.id,
				content=translation_text,
				status=MLTranslationStatus.PUBLISHED
			)
			db_session.add(translation)
		
		await db_session.commit()
		
		# Test batch retrieval
		keys = ["test.save", "test.cancel", "test.delete", "test.nonexistent"]
		results = await service.get_translations(keys, "fr", "test")
		
		assert results["test.save"] == "Enregistrer"
		assert results["test.cancel"] == "Annuler"
		assert results["test.delete"] == "Supprimer"
		assert results["test.nonexistent"] is None
	
	async def test_set_translation(self, db_session, redis_client):
		"""Test creating/updating translations"""
		service = TranslationService(db_session, redis_client)
		
		# Setup test data
		language = MLLanguage(code="de", name="German", native_name="Deutsch", script="Latn")
		namespace = MLNamespace(name="test", description="Test namespace")
		db_session.add_all([language, namespace])
		await db_session.commit()
		
		# Test creating new translation
		translation_id = await service.set_translation(
			key="test.hello",
			language_code="de",
			content="Hallo",
			namespace="test",
			translation_type=MLTranslationType.HUMAN,
			translator_id="user_123",
			auto_approve=True
		)
		
		assert translation_id is not None
		
		# Verify translation was created
		result = await service.get_translation("test.hello", "de", "test")
		assert result == "Hallo"
	
	@patch('langdetect.detect')
	async def test_detect_language(self, mock_detect, db_session, redis_client):
		"""Test language detection"""
		service = TranslationService(db_session, redis_client)
		
		# Setup supported language
		language = MLLanguage(code="fr", name="French", native_name="Français", script="Latn")
		db_session.add(language)
		await db_session.commit()
		
		# Mock detection
		mock_detect.return_value = "fr"
		
		result = await service.detect_language("Bonjour le monde")
		assert result == "fr"
		
		# Test unsupported language detection
		mock_detect.return_value = "xx"  # Unsupported
		result = await service.detect_language("Unknown language")
		assert result is None
	
	async def test_translation_stats(self, db_session, redis_client):
		"""Test translation statistics generation"""
		service = TranslationService(db_session, redis_client)
		
		# Setup test data
		language = MLLanguage(code="it", name="Italian", native_name="Italiano", script="Latn")
		namespace = MLNamespace(name="stats_test", description="Stats test")
		db_session.add_all([language, namespace])
		await db_session.flush()
		
		# Create translation keys with some translations
		for i in range(5):
			translation_key = MLTranslationKey(
				namespace_id=namespace.id,
				key=f"stats.test.{i}",
				source_text=f"Test text {i}"
			)
			db_session.add(translation_key)
			await db_session.flush()
			
			# Only translate some keys
			if i < 3:
				translation = MLTranslation(
					translation_key_id=translation_key.id,
					language_id=language.id,
					content=f"Testo {i}",
					status=MLTranslationStatus.PUBLISHED,
					quality_score=8.5
				)
				db_session.add(translation)
		
		await db_session.commit()
		
		# Test stats generation
		stats = await service.get_translation_stats("stats_test")
		
		assert stats.total_keys >= 5
		assert stats.translated_keys >= 3
		assert stats.completion_percentage > 0
		assert stats.languages_supported >= 1

@pytest.mark.asyncio
class TestLanguageManagementService:
	"""Test LanguageManagementService functionality"""
	
	async def test_create_language(self, db_session, redis_client):
		"""Test language creation"""
		service = LanguageManagementService(db_session, redis_client)
		
		language_data = MLLanguageCreate(
			code="pt",
			name="Portuguese",
			native_name="Português",
			script="Latn",
			direction=MLTextDirection.LTR,
			priority=50
		)
		
		result = await service.create_language(language_data)
		
		assert result.code == "pt"
		assert result.name == "Portuguese"
		assert result.native_name == "Português"
		assert result.priority == 50
	
	async def test_create_duplicate_language(self, db_session, redis_client):
		"""Test creating duplicate language throws error"""
		service = LanguageManagementService(db_session, redis_client)
		
		# Create first language
		existing_language = MLLanguage(
			code="nl",
			name="Dutch",
			native_name="Nederlands",
			script="Latn"
		)
		db_session.add(existing_language)
		await db_session.commit()
		
		# Try to create duplicate
		language_data = MLLanguageCreate(
			code="nl",  # Same code
			name="Dutch (Belgium)",
			native_name="Nederlands (België)",
			script="Latn"
		)
		
		with pytest.raises(ValueError, match="already exists"):
			await service.create_language(language_data)
	
	async def test_get_supported_languages(self, db_session, redis_client):
		"""Test retrieving supported languages"""
		service = LanguageManagementService(db_session, redis_client)
		
		# Create test languages
		languages_data = [
			("en", "English", "English", MLLanguageStatus.ACTIVE),
			("es", "Spanish", "Español", MLLanguageStatus.ACTIVE),
			("fr", "French", "Français", MLLanguageStatus.DEPRECATED)
		]
		
		for code, name, native_name, status in languages_data:
			language = MLLanguage(
				code=code,
				name=name,
				native_name=native_name,
				script="Latn",
				status=status,
				priority=50
			)
			db_session.add(language)
		
		await db_session.commit()
		
		# Test getting only active languages
		active_languages = await service.get_supported_languages(include_inactive=False)
		active_codes = [lang.code for lang in active_languages]
		assert "en" in active_codes
		assert "es" in active_codes
		assert "fr" not in active_codes  # Deprecated
		
		# Test getting all languages
		all_languages = await service.get_supported_languages(include_inactive=True)
		all_codes = [lang.code for lang in all_languages]
		assert "en" in all_codes
		assert "es" in all_codes
		assert "fr" in all_codes
	
	async def test_create_locale(self, db_session, redis_client):
		"""Test locale creation"""
		service = LanguageManagementService(db_session, redis_client)
		
		# Create language first
		language = MLLanguage(
			code="en",
			name="English",
			native_name="English",
			script="Latn"
		)
		db_session.add(language)
		await db_session.commit()
		
		# Create locale
		locale_data = MLLocaleCreate(
			language_id=language.id,
			region_code="GB",
			currency_code="GBP",
			date_format="dd/MM/yyyy",
			time_format="HH:mm"
		)
		
		result = await service.create_locale(locale_data)
		
		assert result["locale_code"] == "en-GB"
		assert result["currency_code"] == "GBP"
		assert result["region_code"] == "GB"
	
	async def test_get_locales(self, db_session, redis_client):
		"""Test retrieving locales"""
		service = LanguageManagementService(db_session, redis_client)
		
		# Create language and locales
		language = MLLanguage(
			code="en",
			name="English", 
			native_name="English",
			script="Latn"
		)
		db_session.add(language)
		await db_session.flush()
		
		locales_data = [
			("US", "en-US", "USD"),
			("GB", "en-GB", "GBP"),
			("CA", "en-CA", "CAD")
		]
		
		for region, locale_code, currency in locales_data:
			locale = MLLocale(
				language_id=language.id,
				region_code=region,
				locale_code=locale_code,
				currency_code=currency,
				number_format={"decimal_separator": ".", "group_separator": ","}
			)
			db_session.add(locale)
		
		await db_session.commit()
		
		# Test getting all locales
		all_locales = await service.get_locales()
		assert len(all_locales) == 3
		
		# Test filtering by language
		en_locales = await service.get_locales("en")
		assert len(en_locales) == 3
		assert all(locale["locale_code"].startswith("en-") for locale in en_locales)

@pytest.mark.asyncio
class TestFormattingService:
	"""Test FormattingService functionality"""
	
	def test_format_number_decimal(self, db_session, redis_client):
		"""Test decimal number formatting"""
		service = FormattingService(db_session, redis_client)
		
		# Test US locale
		result = service.format_number(1234.56, "en-US", "decimal")
		assert "1,234.56" in result or "1234.56" in result  # Depending on Babel version
		
		# Test German locale
		result = service.format_number(1234.56, "de-DE", "decimal")
		assert "1.234,56" in result or "1234,56" in result
	
	def test_format_number_currency(self, db_session, redis_client):
		"""Test currency formatting"""
		service = FormattingService(db_session, redis_client)
		
		result = service.format_number(99.99, "en-US", "currency")
		assert "$" in result
		assert "99.99" in result
	
	def test_format_number_percent(self, db_session, redis_client):
		"""Test percentage formatting"""
		service = FormattingService(db_session, redis_client)
		
		result = service.format_number(0.75, "en-US", "percent")
		assert "%" in result
		assert "75" in result
	
	def test_format_date(self, db_session, redis_client):
		"""Test date formatting"""
		service = FormattingService(db_session, redis_client)
		
		test_date = datetime(2025, 1, 15)
		
		# Test US format
		result = service.format_date(test_date, "en-US", "medium")
		assert "2025" in result
		assert "Jan" in result or "January" in result
		
		# Test German format
		result = service.format_date(test_date, "de-DE", "medium")
		assert "2025" in result
	
	def test_format_time(self, db_session, redis_client):
		"""Test time formatting"""
		service = FormattingService(db_session, redis_client)
		
		test_time = datetime(2025, 1, 15, 14, 30, 0)
		
		result = service.format_time(test_time, "en-US", "medium")
		assert "14:30" in result or "2:30" in result  # 24h or 12h format
	
	def test_format_fallback(self, db_session, redis_client):
		"""Test fallback formatting for unknown locales"""
		service = FormattingService(db_session, redis_client)
		
		# Test with invalid locale - should fallback gracefully
		result = service.format_number(1234.56, "xx-XX", "decimal")
		assert "1,234.56" in result or "1234.56" in result
		
		test_date = datetime(2025, 1, 15)
		result = service.format_date(test_date, "xx-XX", "medium")
		assert "2025-01-15" in result

@pytest.mark.asyncio
class TestContentManagementService:
	"""Test ContentManagementService functionality"""
	
	async def test_create_translation_key(self, db_session, redis_client):
		"""Test translation key creation"""
		service = ContentManagementService(db_session, redis_client)
		
		# Create namespace
		namespace = MLNamespace(name="content_test", description="Content test")
		db_session.add(namespace)
		await db_session.commit()
		
		# Create translation key
		key_data = MLTranslationKeyCreate(
			namespace_id=namespace.id,
			key="content.test.title",
			source_text="Test Title with {variable}",
			context="Page title with dynamic content",
			content_type=MLContentType.CONTENT,
			max_length=100,
			translation_priority=80
		)
		
		key_id = await service.create_translation_key(key_data)
		assert key_id is not None
	
	async def test_duplicate_translation_key(self, db_session, redis_client):
		"""Test creating duplicate translation key throws error"""
		service = ContentManagementService(db_session, redis_client)
		
		# Setup namespace and existing key
		namespace = MLNamespace(name="duplicate_test", description="Duplicate test")
		db_session.add(namespace)
		await db_session.flush()
		
		existing_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="duplicate.key",
			source_text="Existing key"
		)
		db_session.add(existing_key)
		await db_session.commit()
		
		# Try to create duplicate
		key_data = MLTranslationKeyCreate(
			namespace_id=namespace.id,
			key="duplicate.key",  # Same key
			source_text="New key with same name"
		)
		
		with pytest.raises(ValueError, match="already exists"):
			await service.create_translation_key(key_data)
	
	async def test_extract_content_keys_html(self, db_session, redis_client):
		"""Test extracting translatable content from HTML"""
		service = ContentManagementService(db_session, redis_client)
		
		html_content = """
		<html>
			<head><title>Test Page</title></head>
			<body>
				<h1>Welcome to our site</h1>
				<p>This is a test paragraph with <strong>bold text</strong>.</p>
				<button>Click me</button>
				<script>console.log('this should be ignored');</script>
			</body>
		</html>
		"""
		
		extracted = await service.extract_content_keys(html_content, "html")
		
		assert "Test Page" in extracted
		assert "Welcome to our site" in extracted
		assert "Click me" in extracted
		assert "console.log" not in extracted  # Script content should be ignored
	
	async def test_extract_content_keys_json(self, db_session, redis_client):
		"""Test extracting translatable content from JSON"""
		service = ContentManagementService(db_session, redis_client)
		
		json_content = """
		{
			"title": "Application Title",
			"messages": {
				"welcome": "Welcome to the app",
				"error": "An error occurred"
			},
			"buttons": ["Save", "Cancel", "Delete"],
			"config": {
				"max_items": 100,
				"debug": true
			}
		}
		"""
		
		extracted = await service.extract_content_keys(json_content, "json")
		
		assert "Application Title" in extracted
		assert "Welcome to the app" in extracted
		assert "An error occurred" in extracted
		assert "Save" in extracted
		assert "Cancel" in extracted
		assert "Delete" in extracted
		assert 100 not in [str(item) for item in extracted]  # Numbers should be excluded
	
	async def test_get_translation_keys_filtering(self, db_session, redis_client):
		"""Test translation key retrieval with filtering"""
		service = ContentManagementService(db_session, redis_client)
		
		# Setup test data
		namespace1 = MLNamespace(name="ui", description="UI namespace")
		namespace2 = MLNamespace(name="content", description="Content namespace")
		db_session.add_all([namespace1, namespace2])
		await db_session.flush()
		
		# Create keys in different namespaces and content types
		keys_data = [
			(namespace1.id, "ui.button.save", "Save", MLContentType.UI_TEXT),
			(namespace1.id, "ui.error.required", "Required field", MLContentType.ERROR_MESSAGE),
			(namespace2.id, "content.article.title", "Article Title", MLContentType.CONTENT),
			(namespace2.id, "content.email.welcome", "Welcome Email", MLContentType.EMAIL_TEMPLATE)
		]
		
		for namespace_id, key, source_text, content_type in keys_data:
			translation_key = MLTranslationKey(
				namespace_id=namespace_id,
				key=key,
				source_text=source_text,
				content_type=content_type
			)
			db_session.add(translation_key)
		
		await db_session.commit()
		
		# Test filtering by namespace
		ui_keys = await service.get_translation_keys(namespace="ui")
		assert len(ui_keys) == 2
		assert all("ui." in key["key"] for key in ui_keys)
		
		# Test filtering by content type
		error_keys = await service.get_translation_keys(content_type=MLContentType.ERROR_MESSAGE)
		assert len(error_keys) == 1
		assert error_keys[0]["key"] == "ui.error.required"

@pytest.mark.asyncio 
class TestUserPreferenceService:
	"""Test UserPreferenceService functionality"""
	
	async def test_get_user_preferences_new_user(self, db_session, redis_client):
		"""Test getting preferences for new user"""
		service = UserPreferenceService(db_session, redis_client)
		
		result = await service.get_user_preferences("new_user_123")
		assert result is None
	
	async def test_set_and_get_user_preferences(self, db_session, redis_client):
		"""Test setting and retrieving user preferences"""
		service = UserPreferenceService(db_session, redis_client)
		
		# Create test language and locale
		language = MLLanguage(code="ja", name="Japanese", native_name="日本語", script="Jpan")
		db_session.add(language)
		await db_session.flush()
		
		locale = MLLocale(
			language_id=language.id,
			region_code="JP",
			locale_code="ja-JP",
			currency_code="JPY",
			number_format={"decimal_separator": ".", "group_separator": ","}
		)
		db_session.add(locale)
		await db_session.commit()
		
		# Set preferences
		preferences = {
			"primary_language_id": language.id,
			"preferred_locale_id": locale.id,
			"timezone": "Asia/Tokyo",
			"font_size_adjustment": 1.2,
			"high_contrast": True,
			"auto_translate_enabled": False
		}
		
		success = await service.set_user_preferences("user_456", preferences)
		assert success is True
		
		# Get preferences
		result = await service.get_user_preferences("user_456")
		assert result is not None
		assert result["primary_language_id"] == language.id
		assert result["preferred_locale_id"] == locale.id
		assert result["timezone"] == "Asia/Tokyo"
		assert result["font_size_adjustment"] == 1.2
		assert result["high_contrast"] is True
		assert result["auto_translate_enabled"] is False
	
	async def test_get_user_language(self, db_session, redis_client):
		"""Test getting user's primary language code"""
		service = UserPreferenceService(db_session, redis_client)
		
		# Create test data
		language = MLLanguage(code="ko", name="Korean", native_name="한국어", script="Kore")
		db_session.add(language)
		await db_session.flush()
		
		locale = MLLocale(
			language_id=language.id,
			region_code="KR", 
			locale_code="ko-KR",
			currency_code="KRW",
			number_format={"decimal_separator": ".", "group_separator": ","}
		)
		db_session.add(locale)
		await db_session.commit()
		
		# Set user preferences
		await service.set_user_preferences("user_789", {
			"primary_language_id": language.id,
			"preferred_locale_id": locale.id
		})
		
		# Test getting language code
		language_code = await service.get_user_language("user_789")
		assert language_code == "ko"
	
	async def test_get_user_locale(self, db_session, redis_client):
		"""Test getting user's preferred locale code"""
		service = UserPreferenceService(db_session, redis_client)
		
		# Create test data  
		language = MLLanguage(code="th", name="Thai", native_name="ไทย", script="Thai")
		db_session.add(language)
		await db_session.flush()
		
		locale = MLLocale(
			language_id=language.id,
			region_code="TH",
			locale_code="th-TH", 
			currency_code="THB",
			number_format={"decimal_separator": ".", "group_separator": ","}
		)
		db_session.add(locale)
		await db_session.commit()
		
		# Set user preferences
		await service.set_user_preferences("user_101", {
			"primary_language_id": language.id,
			"preferred_locale_id": locale.id
		})
		
		# Test getting locale code
		locale_code = await service.get_user_locale("user_101")
		assert locale_code == "th-TH"

@pytest.mark.integration
class TestServiceIntegration:
	"""Integration tests across multiple services"""
	
	async def test_end_to_end_translation_workflow(self, db_session, redis_client):
		"""Test complete translation workflow from key creation to retrieval"""
		translation_service = TranslationService(db_session, redis_client)
		content_service = ContentManagementService(db_session, redis_client)
		language_service = LanguageManagementService(db_session, redis_client)
		
		# Create language
		language_data = MLLanguageCreate(
			code="sv",
			name="Swedish",
			native_name="Svenska",
			script="Latn"
		)
		language_response = await language_service.create_language(language_data)
		
		# Create namespace
		namespace = MLNamespace(name="integration_test", description="Integration test")
		db_session.add(namespace)
		await db_session.commit()
		
		# Create translation key
		key_data = MLTranslationKeyCreate(
			namespace_id=namespace.id,
			key="integration.test.message",
			source_text="Hello, {name}!",
			context="Greeting message with name variable"
		)
		key_id = await content_service.create_translation_key(key_data)
		
		# Create translation
		translation_id = await translation_service.set_translation(
			key="integration.test.message",
			language_code="sv",
			content="Hej, {name}!",
			namespace="integration_test",
			translation_type=MLTranslationType.HUMAN,
			auto_approve=True
		)
		
		# Retrieve translation with variables
		result = await translation_service.get_translation(
			"integration.test.message",
			"sv",
			"integration_test",
			variables={"name": "Erik"}
		)
		
		assert result == "Hej, Erik!"

if __name__ == "__main__":
	pytest.main([__file__])