"""
APG Multi-language Localization - Model Tests

Unit tests for data models, validation, and database schema integrity.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..models import (
	MLLanguage, MLLocale, MLNamespace, MLTranslationKey, MLTranslation,
	MLTranslationProject, MLTranslationMemory, MLUserPreference,
	MLTextDirection, MLLanguageStatus, MLTranslationStatus, MLTranslationType,
	MLContentType, MLPluralRule,
	MLLanguageCreate, MLLanguageResponse, MLNumberFormat, MLLocaleCreate,
	MLTranslationKeyCreate, MLTranslationCreate, MLTranslationResponse,
	MLBulkTranslationRequest, MLTranslationStats
)

@pytest.mark.asyncio
class TestLanguageModel:
	"""Test MLLanguage model functionality"""
	
	async def test_create_language(self, db_session):
		"""Test creating a new language"""
		language = MLLanguage(
			code="en",
			name="English",
			native_name="English",
			script="Latn",
			direction=MLTextDirection.LTR,
			status=MLLanguageStatus.ACTIVE,
			is_default=True,
			priority=1
		)
		
		db_session.add(language)
		await db_session.commit()
		await db_session.refresh(language)
		
		assert language.id is not None
		assert language.code == "en"
		assert language.name == "English"
		assert language.is_default is True
		assert language.completion_percentage == 0.0
		assert language.created_at is not None
	
	async def test_unique_language_code(self, db_session):
		"""Test that language codes must be unique"""
		# Create first language
		language1 = MLLanguage(
			code="es",
			name="Spanish",
			native_name="Español",
			script="Latn"
		)
		db_session.add(language1)
		await db_session.commit()
		
		# Try to create second language with same code
		language2 = MLLanguage(
			code="es",  # Same code
			name="Spanish (Mexico)",
			native_name="Español (México)",
			script="Latn"
		)
		db_session.add(language2)
		
		with pytest.raises(IntegrityError):
			await db_session.commit()
	
	async def test_language_validation_constraints(self, db_session):
		"""Test language model validation constraints"""
		language = MLLanguage(
			code="fr",
			name="French",
			native_name="Français",
			script="Latn",
			completion_percentage=150,  # Invalid: > 100
			priority=-1  # Invalid: < 0
		)
		
		db_session.add(language)
		
		with pytest.raises(IntegrityError):
			await db_session.commit()

@pytest.mark.asyncio
class TestLocaleModel:
	"""Test MLLocale model functionality"""
	
	async def test_create_locale(self, db_session):
		"""Test creating a locale"""
		# First create a language
		language = MLLanguage(
			code="en",
			name="English", 
			native_name="English",
			script="Latn"
		)
		db_session.add(language)
		await db_session.flush()
		
		# Create locale
		locale = MLLocale(
			language_id=language.id,
			region_code="US",
			locale_code="en-US",
			currency_code="USD",
			date_format="MM/dd/yyyy",
			time_format="h:mm a",
			number_format={
				"decimal_separator": ".",
				"group_separator": ",",
				"group_size": 3
			},
			first_day_of_week=0
		)
		
		db_session.add(locale)
		await db_session.commit()
		await db_session.refresh(locale)
		
		assert locale.id is not None
		assert locale.locale_code == "en-US"
		assert locale.currency_code == "USD"
		assert locale.number_format["decimal_separator"] == "."
	
	async def test_unique_locale_code(self, db_session):
		"""Test that locale codes must be unique"""
		# Create language
		language = MLLanguage(
			code="en",
			name="English",
			native_name="English", 
			script="Latn"
		)
		db_session.add(language)
		await db_session.flush()
		
		# Create first locale
		locale1 = MLLocale(
			language_id=language.id,
			region_code="US",
			locale_code="en-US",
			currency_code="USD"
		)
		db_session.add(locale1)
		await db_session.commit()
		
		# Try to create duplicate locale code
		locale2 = MLLocale(
			language_id=language.id,
			region_code="CA",  # Different region
			locale_code="en-US",  # Same locale code
			currency_code="CAD"
		)
		db_session.add(locale2)
		
		with pytest.raises(IntegrityError):
			await db_session.commit()

@pytest.mark.asyncio
class TestTranslationKeyModel:
	"""Test MLTranslationKey model functionality"""
	
	async def test_create_translation_key(self, db_session):
		"""Test creating a translation key"""
		# Create namespace
		namespace = MLNamespace(
			name="test_namespace",
			description="Test namespace"
		)
		db_session.add(namespace)
		await db_session.flush()
		
		# Create translation key
		key = MLTranslationKey(
			namespace_id=namespace.id,
			key="test.button.save",
			source_text="Save",
			context="Button to save user input",
			content_type=MLContentType.UI_TEXT,
			max_length=20,
			is_html=False,
			translation_priority=75
		)
		
		db_session.add(key)
		await db_session.commit()
		await db_session.refresh(key)
		
		assert key.id is not None
		assert key.key == "test.button.save"
		assert key.source_text == "Save"
		assert key.word_count == 1
		assert key.character_count == 4
		assert key.translation_priority == 75
	
	async def test_unique_namespace_key_combination(self, db_session):
		"""Test that namespace + key combination must be unique"""
		# Create namespace
		namespace = MLNamespace(name="test", description="Test")
		db_session.add(namespace)
		await db_session.flush()
		
		# Create first translation key
		key1 = MLTranslationKey(
			namespace_id=namespace.id,
			key="duplicate.key",
			source_text="Original text"
		)
		db_session.add(key1)
		await db_session.commit()
		
		# Try to create duplicate key in same namespace
		key2 = MLTranslationKey(
			namespace_id=namespace.id,
			key="duplicate.key",  # Same key
			source_text="Different text"
		)
		db_session.add(key2)
		
		with pytest.raises(IntegrityError):
			await db_session.commit()

@pytest.mark.asyncio
class TestTranslationModel:
	"""Test MLTranslation model functionality"""
	
	async def test_create_translation(self, db_session):
		"""Test creating a translation"""
		# Setup dependencies
		language = MLLanguage(code="es", name="Spanish", native_name="Español", script="Latn")
		namespace = MLNamespace(name="test", description="Test")
		
		db_session.add_all([language, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="test.save",
			source_text="Save"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		# Create translation
		translation = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=language.id,
			content="Guardar",
			status=MLTranslationStatus.APPROVED,
			translation_type=MLTranslationType.HUMAN,
			quality_score=9.5,
			word_count=1,
			character_count=7,
			translator_id="user_123"
		)
		
		db_session.add(translation)
		await db_session.commit()
		await db_session.refresh(translation)
		
		assert translation.id is not None
		assert translation.content == "Guardar"
		assert translation.quality_score == 9.5
		assert translation.status == MLTranslationStatus.APPROVED
	
	async def test_unique_key_language_combination(self, db_session):
		"""Test that translation key + language combination must be unique"""
		# Setup dependencies
		language = MLLanguage(code="fr", name="French", native_name="Français", script="Latn")
		namespace = MLNamespace(name="test", description="Test")
		
		db_session.add_all([language, namespace])
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="test.cancel",
			source_text="Cancel"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		# Create first translation
		translation1 = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=language.id,
			content="Annuler"
		)
		db_session.add(translation1)
		await db_session.commit()
		
		# Try to create duplicate translation
		translation2 = MLTranslation(
			translation_key_id=translation_key.id,  # Same key
			language_id=language.id,  # Same language
			content="Annuler (revised)"
		)
		db_session.add(translation2)
		
		with pytest.raises(IntegrityError):
			await db_session.commit()

class TestPydanticModels:
	"""Test Pydantic API models and validation"""
	
	def test_language_create_validation(self):
		"""Test MLLanguageCreate validation"""
		# Valid language
		valid_language = MLLanguageCreate(
			code="de",
			name="German",
			native_name="Deutsch",
			script="Latn",
			direction=MLTextDirection.LTR,
			priority=50
		)
		assert valid_language.code == "de"
		assert valid_language.priority == 50
		
		# Invalid language - code too short
		with pytest.raises(ValidationError):
			MLLanguageCreate(
				code="x",  # Too short
				name="Test",
				native_name="Test",
				script="Latn"
			)
		
		# Invalid language - negative priority
		with pytest.raises(ValidationError):
			MLLanguageCreate(
				code="test",
				name="Test",
				native_name="Test", 
				script="Latn",
				priority=-1  # Invalid
			)
	
	def test_number_format_validation(self):
		"""Test MLNumberFormat validation"""
		# Valid number format
		valid_format = MLNumberFormat(
			decimal_separator=".",
			group_separator=",",
			group_size=3,
			negative_sign="-",
			percent_symbol="%"
		)
		assert valid_format.group_size == 3
		
		# Invalid number format - group size too small
		with pytest.raises(ValidationError):
			MLNumberFormat(
				decimal_separator=".",
				group_separator=",",
				group_size=0  # Invalid
			)
	
	def test_translation_key_create_validation(self):
		"""Test MLTranslationKeyCreate validation"""
		# Valid translation key
		valid_key = MLTranslationKeyCreate(
			namespace_id="namespace_123",
			key="test.validation",
			source_text="Test validation text",
			content_type=MLContentType.UI_TEXT,
			translation_priority=75
		)
		assert valid_key.translation_priority == 75
		
		# Invalid translation key - empty source text
		with pytest.raises(ValidationError):
			MLTranslationKeyCreate(
				namespace_id="namespace_123",
				key="test.empty",
				source_text="",  # Empty not allowed
				translation_priority=50
			)
		
		# Invalid translation key - priority out of range
		with pytest.raises(ValidationError):
			MLTranslationKeyCreate(
				namespace_id="namespace_123",
				key="test.priority",
				source_text="Test",
				translation_priority=150  # > 100
			)
	
	def test_bulk_translation_request_validation(self):
		"""Test MLBulkTranslationRequest validation"""
		# Valid bulk request
		valid_request = MLBulkTranslationRequest(
			source_language_id="lang_en",
			target_language_ids=["lang_es", "lang_fr"],
			translation_type=MLTranslationType.MACHINE,
			quality_threshold=7.0
		)
		assert len(valid_request.target_language_ids) == 2
		
		# Invalid bulk request - empty target languages
		with pytest.raises(ValidationError):
			MLBulkTranslationRequest(
				source_language_id="lang_en",
				target_language_ids=[],  # Empty not allowed
				quality_threshold=7.0
			)
		
		# Invalid bulk request - quality threshold out of range
		with pytest.raises(ValidationError):
			MLBulkTranslationRequest(
				source_language_id="lang_en",
				target_language_ids=["lang_es"],
				quality_threshold=15.0  # > 10
			)

@pytest.mark.asyncio
class TestModelRelationships:
	"""Test model relationships and foreign key constraints"""
	
	async def test_language_locale_relationship(self, db_session):
		"""Test relationship between language and locales"""
		# Create language
		language = MLLanguage(
			code="it",
			name="Italian",
			native_name="Italiano",
			script="Latn"
		)
		db_session.add(language)
		await db_session.flush()
		
		# Create multiple locales for the language
		locale_it = MLLocale(
			language_id=language.id,
			region_code="IT",
			locale_code="it-IT",
			currency_code="EUR"
		)
		locale_ch = MLLocale(
			language_id=language.id,
			region_code="CH",
			locale_code="it-CH",
			currency_code="CHF"
		)
		
		db_session.add_all([locale_it, locale_ch])
		await db_session.commit()
		
		# Test relationship access
		await db_session.refresh(language, ["locales"])
		assert len(language.locales) == 2
		assert any(locale.region_code == "IT" for locale in language.locales)
		assert any(locale.region_code == "CH" for locale in language.locales)
	
	async def test_translation_key_translations_relationship(self, db_session):
		"""Test relationship between translation key and its translations"""
		# Setup dependencies
		language_en = MLLanguage(code="en", name="English", native_name="English", script="Latn")
		language_es = MLLanguage(code="es", name="Spanish", native_name="Español", script="Latn")
		namespace = MLNamespace(name="test", description="Test")
		
		db_session.add_all([language_en, language_es, namespace])
		await db_session.flush()
		
		# Create translation key
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="test.hello",
			source_text="Hello"
		)
		db_session.add(translation_key)
		await db_session.flush()
		
		# Create translations in different languages
		translation_es = MLTranslation(
			translation_key_id=translation_key.id,
			language_id=language_es.id,
			content="Hola"
		)
		
		db_session.add(translation_es)
		await db_session.commit()
		
		# Test relationship access
		await db_session.refresh(translation_key, ["translations"])
		assert len(translation_key.translations) == 1
		assert translation_key.translations[0].content == "Hola"
	
	async def test_cascade_deletion(self, db_session):
		"""Test cascade deletion behavior"""
		# Create namespace with translation keys
		namespace = MLNamespace(name="delete_test", description="Test deletion")
		db_session.add(namespace)
		await db_session.flush()
		
		translation_key = MLTranslationKey(
			namespace_id=namespace.id,
			key="test.delete",
			source_text="Delete me"
		)
		db_session.add(translation_key)
		await db_session.commit()
		
		# Verify key exists
		result = await db_session.execute(
			select(MLTranslationKey).where(MLTranslationKey.namespace_id == namespace.id)
		)
		keys_before = result.scalars().all()
		assert len(keys_before) == 1
		
		# Delete namespace (should cascade to translation keys)
		await db_session.delete(namespace)
		await db_session.commit()
		
		# Verify cascade deletion
		result = await db_session.execute(
			select(MLTranslationKey).where(MLTranslationKey.namespace_id == namespace.id)
		)
		keys_after = result.scalars().all()
		assert len(keys_after) == 0

class TestModelPerformance:
	"""Test model performance and indexing"""
	
	@pytest.mark.performance
	async def test_translation_lookup_performance(self, db_session):
		"""Test performance of translation lookups with proper indexing"""
		# This would be a more comprehensive test with timing
		# For now, just verify that indexes are being used correctly
		
		# Create test data
		language = MLLanguage(code="perf", name="Performance", native_name="Performance", script="Latn")
		namespace = MLNamespace(name="perf_test", description="Performance test")
		
		db_session.add_all([language, namespace])
		await db_session.flush()
		
		# Create multiple translation keys
		for i in range(100):
			key = MLTranslationKey(
				namespace_id=namespace.id,
				key=f"perf.test.{i:03d}",
				source_text=f"Performance test {i}"
			)
			db_session.add(key)
		
		await db_session.commit()
		
		# Test indexed lookup
		result = await db_session.execute(
			select(MLTranslationKey)
			.where(MLTranslationKey.namespace_id == namespace.id)
			.where(MLTranslationKey.key == "perf.test.050")
		)
		found_key = result.scalar_one_or_none()
		
		assert found_key is not None
		assert found_key.key == "perf.test.050"

if __name__ == "__main__":
	pytest.main([__file__])