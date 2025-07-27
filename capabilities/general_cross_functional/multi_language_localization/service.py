"""
APG Multi-language Localization - Service Layer Implementation

This module provides comprehensive service layer for internationalization and localization,
including translation management, language configuration, and cultural formatting services.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

import aioredis
from babel import Locale, numbers, dates
from babel.core import UnknownLocaleError
from langdetect import detect, LangDetectError
from sqlalchemy import and_, desc, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
	MLLanguage, MLLocale, MLNamespace, MLTranslationKey, MLTranslation,
	MLTranslationProject, MLTranslationMemory, MLUserPreference,
	MLTextDirection, MLLanguageStatus, MLTranslationStatus, MLTranslationType,
	MLContentType, MLPluralRule,
	MLLanguageCreate, MLLanguageResponse, MLNumberFormat, MLLocaleCreate,
	MLTranslationKeyCreate, MLTranslationCreate, MLTranslationResponse,
	MLBulkTranslationRequest, MLTranslationStats
)

# =====================
# Translation Service
# =====================

class TranslationService:
	"""Core translation management service"""
	
	def __init__(self, db_session: AsyncSession, redis_client: aioredis.Redis):
		self.db = db_session
		self.redis = redis_client
		self.cache_ttl = 3600  # 1 hour cache TTL
		self.translation_providers = {}  # Will be initialized with external providers
	
	async def get_translation(
		self, 
		key: str, 
		language_code: str, 
		namespace: str = "default",
		fallback: bool = True,
		variables: Optional[Dict[str, Any]] = None
	) -> Optional[str]:
		"""
		Get translation for a specific key and language
		
		Args:
			key: Translation key
			language_code: Target language code (ISO 639-1)
			namespace: Translation namespace
			fallback: Whether to use fallback language if translation not found
			variables: Variables to substitute in translation
		
		Returns:
			Translated string or None if not found
		"""
		cache_key = f"translation:{namespace}:{key}:{language_code}"
		
		# Try cache first
		cached_translation = await self.redis.get(cache_key)
		if cached_translation:
			translation = cached_translation.decode('utf-8')
			return self._substitute_variables(translation, variables) if variables else translation
		
		# Get language
		language = await self._get_language_by_code(language_code)
		if not language:
			return None
		
		# Get namespace
		namespace_obj = await self._get_or_create_namespace(namespace)
		
		# Get translation key
		stmt = select(MLTranslationKey).where(
			and_(
				MLTranslationKey.namespace_id == namespace_obj.id,
				MLTranslationKey.key == key
			)
		)
		result = await self.db.execute(stmt)
		translation_key = result.scalar_one_or_none()
		
		if not translation_key:
			return None
		
		# Get translation
		translation = await self._get_translation_for_key_and_language(
			translation_key.id, language.id, fallback
		)
		
		if translation:
			# Cache the translation
			await self.redis.setex(cache_key, self.cache_ttl, translation.content)
			
			# Update usage statistics
			await self._update_usage_stats(translation_key.id, translation.id)
			
			# Substitute variables if provided
			content = translation.content
			if variables:
				content = self._substitute_variables(content, variables)
			
			return content
		
		return None
	
	async def get_translations(
		self,
		keys: List[str],
		language_code: str,
		namespace: str = "default",
		fallback: bool = True
	) -> Dict[str, Optional[str]]:
		"""Get multiple translations at once"""
		translations = {}
		
		# Batch cache lookup
		cache_keys = [f"translation:{namespace}:{key}:{language_code}" for key in keys]
		cached_values = await self.redis.mget(cache_keys)
		
		uncached_keys = []
		for i, (key, cached_value) in enumerate(zip(keys, cached_values)):
			if cached_value:
				translations[key] = cached_value.decode('utf-8')
			else:
				uncached_keys.append(key)
		
		# Get uncached translations from database
		if uncached_keys:
			language = await self._get_language_by_code(language_code)
			namespace_obj = await self._get_or_create_namespace(namespace)
			
			if language and namespace_obj:
				db_translations = await self._get_translations_batch(
					uncached_keys, namespace_obj.id, language.id, fallback
				)
				
				# Cache and add to results
				cache_pipeline = self.redis.pipeline()
				for key, translation in db_translations.items():
					if translation:
						translations[key] = translation
						cache_key = f"translation:{namespace}:{key}:{language_code}"
						cache_pipeline.setex(cache_key, self.cache_ttl, translation)
				
				await cache_pipeline.execute()
		
		# Ensure all requested keys are in the result
		for key in keys:
			if key not in translations:
				translations[key] = None
		
		return translations
	
	async def set_translation(
		self,
		key: str,
		language_code: str,
		content: str,
		namespace: str = "default",
		translation_type: MLTranslationType = MLTranslationType.HUMAN,
		translator_id: Optional[str] = None,
		context: Optional[str] = None,
		auto_approve: bool = False
	) -> str:
		"""Create or update a translation"""
		
		# Get or create language
		language = await self._get_or_create_language(language_code)
		
		# Get or create namespace
		namespace_obj = await self._get_or_create_namespace(namespace)
		
		# Get or create translation key
		translation_key = await self._get_or_create_translation_key(
			key, namespace_obj.id, content, context
		)
		
		# Get existing translation or create new one
		existing_translation = await self._get_translation_for_key_and_language(
			translation_key.id, language.id, fallback=False
		)
		
		if existing_translation:
			# Update existing translation
			existing_translation.content = content
			existing_translation.translation_type = translation_type
			existing_translation.translator_id = translator_id
			existing_translation.status = MLTranslationStatus.APPROVED if auto_approve else MLTranslationStatus.PENDING_REVIEW
			existing_translation.updated_at = datetime.now(timezone.utc)
			existing_translation.translated_at = datetime.now(timezone.utc)
			
			if auto_approve:
				existing_translation.published_at = datetime.now(timezone.utc)
			
			translation_id = existing_translation.id
		else:
			# Create new translation
			new_translation = MLTranslation(
				translation_key_id=translation_key.id,
				language_id=language.id,
				content=content,
				translation_type=translation_type,
				translator_id=translator_id,
				status=MLTranslationStatus.APPROVED if auto_approve else MLTranslationStatus.PENDING_REVIEW,
				word_count=len(content.split()),
				character_count=len(content),
				translated_at=datetime.now(timezone.utc)
			)
			
			if auto_approve:
				new_translation.published_at = datetime.now(timezone.utc)
			
			self.db.add(new_translation)
			await self.db.flush()
			translation_id = new_translation.id
		
		await self.db.commit()
		
		# Clear cache
		cache_key = f"translation:{namespace}:{key}:{language_code}"
		await self.redis.delete(cache_key)
		
		# Add to translation memory if approved
		if auto_approve and translation_key.source_text != content:
			await self._add_to_translation_memory(
				translation_key.source_text,
				content,
				language.id,
				language.id  # Assuming source is in same language for now
			)
		
		return translation_id
	
	async def bulk_translate(
		self,
		request: MLBulkTranslationRequest,
		translator_id: Optional[str] = None
	) -> Dict[str, Any]:
		"""Perform bulk translation using machine translation"""
		
		source_language = await self._get_language_by_id(request.source_language_id)
		if not source_language:
			raise ValueError(f"Source language not found: {request.source_language_id}")
		
		target_languages = []
		for lang_id in request.target_language_ids:
			lang = await self._get_language_by_id(lang_id)
			if lang:
				target_languages.append(lang)
		
		if not target_languages:
			raise ValueError("No valid target languages found")
		
		# Get translation keys to process
		translation_keys = await self._get_translation_keys_for_bulk(
			request.namespace_ids, source_language.id
		)
		
		results = {
			"total_keys": len(translation_keys),
			"processed": 0,
			"success": 0,
			"failed": 0,
			"languages": {},
			"errors": []
		}
		
		for target_language in target_languages:
			lang_results = {
				"processed": 0,
				"success": 0,
				"failed": 0,
				"translations": []
			}
			
			for translation_key in translation_keys:
				try:
					# Check if translation already exists
					existing = await self._get_translation_for_key_and_language(
						translation_key.id, target_language.id, fallback=False
					)
					
					if existing and existing.status in [MLTranslationStatus.APPROVED, MLTranslationStatus.PUBLISHED]:
						continue  # Skip if already translated
					
					# Perform machine translation
					translated_content = await self._machine_translate(
						translation_key.source_text,
						source_language.code,
						target_language.code
					)
					
					if translated_content:
						# Create translation
						translation_id = await self.set_translation(
							translation_key.key,
							target_language.code,
							translated_content,
							translation_key.namespace.name,
							request.translation_type,
							translator_id,
							auto_approve=request.auto_publish
						)
						
						lang_results["success"] += 1
						lang_results["translations"].append({
							"key": translation_key.key,
							"translation_id": translation_id,
							"content": translated_content
						})
					else:
						lang_results["failed"] += 1
						results["errors"].append(f"Translation failed for key: {translation_key.key}")
				
				except Exception as e:
					lang_results["failed"] += 1
					results["errors"].append(f"Error translating {translation_key.key}: {str(e)}")
				
				lang_results["processed"] += 1
			
			results["languages"][target_language.code] = lang_results
			results["processed"] += lang_results["processed"]
			results["success"] += lang_results["success"]
			results["failed"] += lang_results["failed"]
		
		return results
	
	async def detect_language(self, text: str) -> Optional[str]:
		"""Detect language of given text"""
		try:
			detected = detect(text)
			# Validate that we support this language
			language = await self._get_language_by_code(detected)
			return detected if language else None
		except LangDetectError:
			return None
	
	async def get_translation_stats(self, namespace: str = None) -> MLTranslationStats:
		"""Get translation statistics"""
		
		base_query = select(MLTranslationKey)
		if namespace:
			namespace_obj = await self._get_namespace_by_name(namespace)
			if namespace_obj:
				base_query = base_query.where(MLTranslationKey.namespace_id == namespace_obj.id)
		
		# Total keys
		total_keys_result = await self.db.execute(select(func.count(MLTranslationKey.id)))
		total_keys = total_keys_result.scalar() or 0
		
		# Translated keys
		translated_keys_query = select(func.count(func.distinct(MLTranslation.translation_key_id))).where(
			MLTranslation.status.in_([MLTranslationStatus.APPROVED, MLTranslationStatus.PUBLISHED])
		)
		if namespace:
			translated_keys_query = translated_keys_query.join(MLTranslationKey).where(
				MLTranslationKey.namespace_id == namespace_obj.id
			)
		
		translated_keys_result = await self.db.execute(translated_keys_query)
		translated_keys = translated_keys_result.scalar() or 0
		
		# Languages supported
		languages_result = await self.db.execute(
			select(func.count(MLLanguage.id)).where(MLLanguage.status == MLLanguageStatus.ACTIVE)
		)
		languages_supported = languages_result.scalar() or 0
		
		# Total translations
		total_translations_result = await self.db.execute(select(func.count(MLTranslation.id)))
		total_translations = total_translations_result.scalar() or 0
		
		# Quality average
		quality_result = await self.db.execute(
			select(func.avg(MLTranslation.quality_score)).where(
				MLTranslation.quality_score.isnot(None)
			)
		)
		quality_average = quality_result.scalar()
		
		# Pending review
		pending_result = await self.db.execute(
			select(func.count(MLTranslation.id)).where(
				MLTranslation.status == MLTranslationStatus.PENDING_REVIEW
			)
		)
		pending_review = pending_result.scalar() or 0
		
		# Last updated
		last_updated_result = await self.db.execute(
			select(func.max(MLTranslation.updated_at))
		)
		last_updated = last_updated_result.scalar()
		
		completion_percentage = (translated_keys / total_keys * 100) if total_keys > 0 else 0
		
		return MLTranslationStats(
			total_keys=total_keys,
			translated_keys=translated_keys,
			completion_percentage=completion_percentage,
			languages_supported=languages_supported,
			total_translations=total_translations,
			quality_average=quality_average,
			pending_review=pending_review,
			last_updated=last_updated
		)
	
	# Private helper methods
	
	async def _get_language_by_code(self, code: str) -> Optional[MLLanguage]:
		"""Get language by ISO code"""
		stmt = select(MLLanguage).where(MLLanguage.code == code.lower())
		result = await self.db.execute(stmt)
		return result.scalar_one_or_none()
	
	async def _get_language_by_id(self, language_id: str) -> Optional[MLLanguage]:
		"""Get language by ID"""
		stmt = select(MLLanguage).where(MLLanguage.id == language_id)
		result = await self.db.execute(stmt)
		return result.scalar_one_or_none()
	
	async def _get_or_create_language(self, code: str) -> MLLanguage:
		"""Get existing language or create a basic one"""
		language = await self._get_language_by_code(code)
		if not language:
			# Create basic language entry
			language = MLLanguage(
				code=code.lower(),
				name=code.upper(),
				native_name=code.upper(),
				script="Latn",  # Default to Latin script
				status=MLLanguageStatus.ACTIVE
			)
			self.db.add(language)
			await self.db.flush()
		return language
	
	async def _get_or_create_namespace(self, name: str) -> MLNamespace:
		"""Get or create namespace"""
		stmt = select(MLNamespace).where(MLNamespace.name == name)
		result = await self.db.execute(stmt)
		namespace = result.scalar_one_or_none()
		
		if not namespace:
			namespace = MLNamespace(
				name=name,
				description=f"Auto-created namespace for {name}"
			)
			self.db.add(namespace)
			await self.db.flush()
		
		return namespace
	
	async def _get_namespace_by_name(self, name: str) -> Optional[MLNamespace]:
		"""Get namespace by name"""
		stmt = select(MLNamespace).where(MLNamespace.name == name)
		result = await self.db.execute(stmt)
		return result.scalar_one_or_none()
	
	async def _get_or_create_translation_key(
		self,
		key: str,
		namespace_id: str,
		source_text: str,
		context: Optional[str] = None
	) -> MLTranslationKey:
		"""Get or create translation key"""
		stmt = select(MLTranslationKey).where(
			and_(
				MLTranslationKey.namespace_id == namespace_id,
				MLTranslationKey.key == key
			)
		)
		result = await self.db.execute(stmt)
		translation_key = result.scalar_one_or_none()
		
		if not translation_key:
			translation_key = MLTranslationKey(
				namespace_id=namespace_id,
				key=key,
				source_text=source_text,
				context=context,
				word_count=len(source_text.split()),
				character_count=len(source_text)
			)
			self.db.add(translation_key)
			await self.db.flush()
		
		return translation_key
	
	async def _get_translation_for_key_and_language(
		self,
		translation_key_id: str,
		language_id: str,
		fallback: bool = True
	) -> Optional[MLTranslation]:
		"""Get translation for specific key and language with fallback support"""
		
		# Try direct translation first
		stmt = select(MLTranslation).where(
			and_(
				MLTranslation.translation_key_id == translation_key_id,
				MLTranslation.language_id == language_id,
				MLTranslation.status.in_([MLTranslationStatus.APPROVED, MLTranslationStatus.PUBLISHED])
			)
		)
		result = await self.db.execute(stmt)
		translation = result.scalar_one_or_none()
		
		if translation or not fallback:
			return translation
		
		# Try fallback language
		language = await self._get_language_by_id(language_id)
		if language and language.fallback_language_id:
			return await self._get_translation_for_key_and_language(
				translation_key_id, language.fallback_language_id, fallback=False
			)
		
		# Try default language as last resort
		default_language = await self._get_default_language()
		if default_language and default_language.id != language_id:
			return await self._get_translation_for_key_and_language(
				translation_key_id, default_language.id, fallback=False
			)
		
		return None
	
	async def _get_default_language(self) -> Optional[MLLanguage]:
		"""Get the default language"""
		stmt = select(MLLanguage).where(MLLanguage.is_default == True)
		result = await self.db.execute(stmt)
		return result.scalar_one_or_none()
	
	async def _get_translations_batch(
		self,
		keys: List[str],
		namespace_id: str,
		language_id: str,
		fallback: bool = True
	) -> Dict[str, Optional[str]]:
		"""Get multiple translations in a single query"""
		
		stmt = select(MLTranslationKey, MLTranslation).join(
			MLTranslation, 
			and_(
				MLTranslation.translation_key_id == MLTranslationKey.id,
				MLTranslation.language_id == language_id,
				MLTranslation.status.in_([MLTranslationStatus.APPROVED, MLTranslationStatus.PUBLISHED])
			),
			isouter=True
		).where(
			and_(
				MLTranslationKey.namespace_id == namespace_id,
				MLTranslationKey.key.in_(keys)
			)
		)
		
		result = await self.db.execute(stmt)
		results = {}
		
		for translation_key, translation in result.all():
			if translation:
				results[translation_key.key] = translation.content
			else:
				results[translation_key.key] = None
		
		return results
	
	async def _substitute_variables(self, text: str, variables: Dict[str, Any]) -> str:
		"""Substitute variables in translated text"""
		for key, value in variables.items():
			placeholder = f"{{{key}}}"
			text = text.replace(placeholder, str(value))
		return text
	
	async def _update_usage_stats(self, translation_key_id: str, translation_id: str):
		"""Update usage statistics"""
		# Update translation key usage
		await self.db.execute(
			update(MLTranslationKey)
			.where(MLTranslationKey.id == translation_key_id)
			.values(
				usage_count=MLTranslationKey.usage_count + 1,
				last_used_at=datetime.now(timezone.utc)
			)
		)
		
		# This could be done asynchronously to avoid blocking the main request
		await self.db.commit()
	
	async def _machine_translate(
		self,
		text: str,
		source_lang: str,
		target_lang: str
	) -> Optional[str]:
		"""Perform machine translation using external providers"""
		# This would integrate with actual translation providers like:
		# - Google Translate API
		# - Amazon Translate
		# - Microsoft Translator
		# - Azure Cognitive Services
		
		# For now, return a placeholder
		return f"[MT:{source_lang}->{target_lang}] {text}"
	
	async def _add_to_translation_memory(
		self,
		source_text: str,
		target_text: str,
		source_language_id: str,
		target_language_id: str
	):
		"""Add translation pair to translation memory"""
		source_hash = hashlib.sha256(source_text.encode()).hexdigest()
		
		# Check if already exists
		stmt = select(MLTranslationMemory).where(
			and_(
				MLTranslationMemory.source_hash == source_hash,
				MLTranslationMemory.source_language_id == source_language_id,
				MLTranslationMemory.target_language_id == target_language_id
			)
		)
		result = await self.db.execute(stmt)
		existing = result.scalar_one_or_none()
		
		if existing:
			# Update usage count
			existing.usage_count += 1
			existing.last_used_at = datetime.now(timezone.utc)
		else:
			# Create new entry
			tm_entry = MLTranslationMemory(
				source_language_id=source_language_id,
				target_language_id=target_language_id,
				source_text=source_text,
				target_text=target_text,
				source_hash=source_hash,
				usage_count=1,
				last_used_at=datetime.now(timezone.utc)
			)
			self.db.add(tm_entry)
		
		await self.db.commit()
	
	async def _get_translation_keys_for_bulk(
		self,
		namespace_ids: Optional[List[str]],
		source_language_id: str
	) -> List[MLTranslationKey]:
		"""Get translation keys for bulk processing"""
		query = select(MLTranslationKey).options(selectinload(MLTranslationKey.namespace))
		
		if namespace_ids:
			query = query.where(MLTranslationKey.namespace_id.in_(namespace_ids))
		
		query = query.where(MLTranslationKey.is_deprecated == False)
		
		result = await self.db.execute(query)
		return result.scalars().all()

# =====================
# Language Management Service
# =====================

class LanguageManagementService:
	"""Service for managing languages and locales"""
	
	def __init__(self, db_session: AsyncSession, redis_client: aioredis.Redis):
		self.db = db_session
		self.redis = redis_client
		self.cache_ttl = 7200  # 2 hours cache TTL
	
	async def create_language(self, language_data: MLLanguageCreate) -> MLLanguageResponse:
		"""Create a new language"""
		
		# Check if language already exists
		existing = await self._get_language_by_code(language_data.code)
		if existing:
			raise ValueError(f"Language with code '{language_data.code}' already exists")
		
		# Create language
		language = MLLanguage(
			**language_data.model_dump(),
			completion_percentage=0.0,
			total_translators=0
		)
		
		self.db.add(language)
		await self.db.commit()
		await self.db.refresh(language)
		
		# Clear language cache
		await self.redis.delete("languages:all")
		
		return MLLanguageResponse.model_validate(language.__dict__)
	
	async def get_supported_languages(self, include_inactive: bool = False) -> List[MLLanguageResponse]:
		"""Get list of supported languages"""
		
		cache_key = f"languages:all:{'with_inactive' if include_inactive else 'active_only'}"
		cached = await self.redis.get(cache_key)
		
		if cached:
			data = json.loads(cached.decode('utf-8'))
			return [MLLanguageResponse(**lang) for lang in data]
		
		# Query from database
		query = select(MLLanguage).order_by(MLLanguage.priority, MLLanguage.name)
		
		if not include_inactive:
			query = query.where(MLLanguage.status == MLLanguageStatus.ACTIVE)
		
		result = await self.db.execute(query)
		languages = result.scalars().all()
		
		# Convert to response models
		response_data = [MLLanguageResponse.model_validate(lang.__dict__) for lang in languages]
		
		# Cache the results
		cache_data = [lang.model_dump(mode='json') for lang in response_data]
		await self.redis.setex(cache_key, self.cache_ttl, json.dumps(cache_data, default=str))
		
		return response_data
	
	async def create_locale(self, locale_data: MLLocaleCreate) -> dict:
		"""Create a new locale"""
		
		# Validate language exists
		language = await self._get_language_by_id(locale_data.language_id)
		if not language:
			raise ValueError(f"Language not found: {locale_data.language_id}")
		
		# Generate locale code
		locale_code = f"{language.code}-{locale_data.region_code.upper()}"
		
		# Check if locale already exists
		existing = await self._get_locale_by_code(locale_code)
		if existing:
			raise ValueError(f"Locale '{locale_code}' already exists")
		
		# Create locale
		locale = MLLocale(
			**locale_data.model_dump(exclude={'language_id'}),
			language_id=locale_data.language_id,
			locale_code=locale_code,
			number_format=locale_data.number_format.model_dump()
		)
		
		self.db.add(locale)
		await self.db.commit()
		await self.db.refresh(locale)
		
		# Clear locale cache
		await self.redis.delete("locales:all")
		
		return {
			"id": locale.id,
			"locale_code": locale.locale_code,
			"language_code": language.code,
			"region_code": locale.region_code,
			"currency_code": locale.currency_code
		}
	
	async def get_locales(self, language_code: str = None) -> List[dict]:
		"""Get available locales"""
		
		cache_key = f"locales:{'all' if not language_code else language_code}"
		cached = await self.redis.get(cache_key)
		
		if cached:
			return json.loads(cached.decode('utf-8'))
		
		# Query from database
		query = select(MLLocale).join(MLLanguage).where(MLLocale.is_active == True)
		
		if language_code:
			query = query.where(MLLanguage.code == language_code.lower())
		
		query = query.order_by(MLLocale.locale_code)
		
		result = await self.db.execute(query)
		locales = result.scalars().all()
		
		# Convert to response format
		response_data = []
		for locale in locales:
			response_data.append({
				"id": locale.id,
				"locale_code": locale.locale_code,
				"language_id": locale.language_id,
				"region_code": locale.region_code,
				"currency_code": locale.currency_code,
				"date_format": locale.date_format,
				"time_format": locale.time_format,
				"number_format": locale.number_format,
				"first_day_of_week": locale.first_day_of_week,
				"measurement_system": locale.measurement_system
			})
		
		# Cache the results
		await self.redis.setex(cache_key, self.cache_ttl, json.dumps(response_data, default=str))
		
		return response_data
	
	async def _get_language_by_code(self, code: str) -> Optional[MLLanguage]:
		"""Get language by code"""
		stmt = select(MLLanguage).where(MLLanguage.code == code.lower())
		result = await self.db.execute(stmt)
		return result.scalar_one_or_none()
	
	async def _get_language_by_id(self, language_id: str) -> Optional[MLLanguage]:
		"""Get language by ID"""
		stmt = select(MLLanguage).where(MLLanguage.id == language_id)
		result = await self.db.execute(stmt)
		return result.scalar_one_or_none()
	
	async def _get_locale_by_code(self, locale_code: str) -> Optional[MLLocale]:
		"""Get locale by code"""
		stmt = select(MLLocale).where(MLLocale.locale_code == locale_code)
		result = await self.db.execute(stmt)
		return result.scalar_one_or_none()

# =====================
# Formatting Service
# =====================

class FormattingService:
	"""Service for cultural and regional formatting"""
	
	def __init__(self, db_session: AsyncSession, redis_client: aioredis.Redis):
		self.db = db_session
		self.redis = redis_client
	
	def format_number(
		self, 
		number: Union[int, float, Decimal], 
		locale_code: str,
		format_type: str = "decimal"
	) -> str:
		"""Format number according to locale"""
		try:
			locale = Locale.parse(locale_code)
			
			if format_type == "currency":
				# Would need currency code from locale
				return numbers.format_currency(number, "USD", locale=locale)
			elif format_type == "percent":
				return numbers.format_percent(number, locale=locale)
			else:
				return numbers.format_decimal(number, locale=locale)
		
		except (UnknownLocaleError, ValueError):
			# Fallback to basic formatting
			if format_type == "percent":
				return f"{float(number) * 100:.2f}%"
			elif format_type == "currency":
				return f"${number:.2f}"
			else:
				return f"{number:,.2f}" if isinstance(number, (float, Decimal)) else f"{number:,}"
	
	def format_date(
		self,
		date_obj: datetime,
		locale_code: str,
		format_type: str = "medium"
	) -> str:
		"""Format date according to locale"""
		try:
			locale = Locale.parse(locale_code)
			return dates.format_date(date_obj, format=format_type, locale=locale)
		except (UnknownLocaleError, ValueError):
			# Fallback to ISO format
			return date_obj.strftime("%Y-%m-%d")
	
	def format_time(
		self,
		time_obj: datetime,
		locale_code: str,
		format_type: str = "medium"
	) -> str:
		"""Format time according to locale"""
		try:
			locale = Locale.parse(locale_code)
			return dates.format_time(time_obj, format=format_type, locale=locale)
		except (UnknownLocaleError, ValueError):
			# Fallback to ISO format
			return time_obj.strftime("%H:%M:%S")
	
	def format_datetime(
		self,
		datetime_obj: datetime,
		locale_code: str,
		format_type: str = "medium"
	) -> str:
		"""Format datetime according to locale"""
		try:
			locale = Locale.parse(locale_code)
			return dates.format_datetime(datetime_obj, format=format_type, locale=locale)
		except (UnknownLocaleError, ValueError):
			# Fallback to ISO format
			return datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

# =====================
# Content Management Service
# =====================

class ContentManagementService:
	"""Service for managing translatable content and keys"""
	
	def __init__(self, db_session: AsyncSession, redis_client: aioredis.Redis):
		self.db = db_session
		self.redis = redis_client
	
	async def extract_content_keys(self, content: str, content_type: str = "html") -> List[str]:
		"""Extract translatable content keys from markup or code"""
		
		keys = []
		
		if content_type == "html":
			# Extract text content from HTML
			import re
			from html.parser import HTMLParser
			
			class TextExtractor(HTMLParser):
				def __init__(self):
					super().__init__()
					self.texts = []
					self.in_script = False
					self.in_style = False
				
				def handle_starttag(self, tag, attrs):
					if tag.lower() in ['script', 'style']:
						self.in_script = True
				
				def handle_endtag(self, tag):
					if tag.lower() in ['script', 'style']:
						self.in_script = False
				
				def handle_data(self, data):
					if not self.in_script and not self.in_style:
						text = data.strip()
						if text and len(text) > 2:  # Ignore very short strings
							self.texts.append(text)
			
			extractor = TextExtractor()
			extractor.feed(content)
			keys = extractor.texts
		
		elif content_type == "json":
			# Extract string values from JSON
			try:
				import json
				data = json.loads(content)
				keys = self._extract_strings_from_dict(data)
			except json.JSONDecodeError:
				pass
		
		elif content_type == "yaml":
			# Extract string values from YAML
			try:
				import yaml
				data = yaml.safe_load(content)
				keys = self._extract_strings_from_dict(data)
			except yaml.YAMLError:
				pass
		
		# Filter and clean keys
		cleaned_keys = []
		for key in keys:
			# Skip URLs, emails, technical strings
			if not re.match(r'^(https?://|mailto:|[A-Z_][A-Z0-9_]*$)', key):
				cleaned_keys.append(key[:500])  # Limit length
		
		return cleaned_keys
	
	async def create_translation_key(self, key_data: MLTranslationKeyCreate) -> str:
		"""Create a new translation key"""
		
		# Get namespace
		namespace = await self._get_namespace_by_id(key_data.namespace_id)
		if not namespace:
			raise ValueError(f"Namespace not found: {key_data.namespace_id}")
		
		# Check if key already exists
		existing = await self._get_translation_key(key_data.namespace_id, key_data.key)
		if existing:
			raise ValueError(f"Translation key already exists: {key_data.key}")
		
		# Extract variables from source text
		variables = self._extract_variables(key_data.source_text)
		
		# Create translation key
		translation_key = MLTranslationKey(
			**key_data.model_dump(exclude={'variables'}),
			variables=variables,
			word_count=len(key_data.source_text.split()),
			character_count=len(key_data.source_text)
		)
		
		self.db.add(translation_key)
		await self.db.commit()
		await self.db.refresh(translation_key)
		
		# Update namespace statistics
		await self._update_namespace_stats(key_data.namespace_id)
		
		return translation_key.id
	
	async def get_translation_keys(
		self,
		namespace: str = None,
		content_type: str = None,
		limit: int = 100,
		offset: int = 0
	) -> List[dict]:
		"""Get translation keys with filtering and pagination"""
		
		query = select(MLTranslationKey).join(MLNamespace)
		
		if namespace:
			query = query.where(MLNamespace.name == namespace)
		
		if content_type:
			query = query.where(MLTranslationKey.content_type == content_type)
		
		query = query.where(MLTranslationKey.is_deprecated == False)
		query = query.order_by(MLTranslationKey.translation_priority.desc(), MLTranslationKey.created_at)
		query = query.limit(limit).offset(offset)
		
		result = await self.db.execute(query)
		keys = result.scalars().all()
		
		return [
			{
				"id": key.id,
				"key": key.key,
				"source_text": key.source_text,
				"context": key.context,
				"content_type": key.content_type,
				"max_length": key.max_length,
				"is_html": key.is_html,
				"is_plural": key.is_plural,
				"variables": key.variables,
				"priority": key.translation_priority,
				"translation_count": key.translation_count,
				"usage_count": key.usage_count,
				"namespace": key.namespace.name if key.namespace else None
			}
			for key in keys
		]
	
	def _extract_strings_from_dict(self, data: dict, keys: List[str] = None) -> List[str]:
		"""Recursively extract string values from dictionary"""
		if keys is None:
			keys = []
		
		if isinstance(data, dict):
			for value in data.values():
				if isinstance(value, str) and len(value.strip()) > 2:
					keys.append(value.strip())
				elif isinstance(value, (dict, list)):
					self._extract_strings_from_dict(value, keys)
		elif isinstance(data, list):
			for item in data:
				if isinstance(item, str) and len(item.strip()) > 2:
					keys.append(item.strip())
				elif isinstance(item, (dict, list)):
					self._extract_strings_from_dict(item, keys)
		
		return keys
	
	def _extract_variables(self, text: str) -> Dict[str, str]:
		"""Extract variable placeholders from text"""
		variables = {}
		
		# Match {variable_name} pattern
		matches = re.findall(r'\{([^}]+)\}', text)
		for match in matches:
			variables[match] = "string"  # Default type
		
		# Match {{variable_name}} pattern (some template engines)
		matches = re.findall(r'\{\{([^}]+)\}\}', text)
		for match in matches:
			variables[match.strip()] = "string"
		
		# Match %{variable_name} pattern
		matches = re.findall(r'%\{([^}]+)\}', text)
		for match in matches:
			variables[match] = "string"
		
		return variables
	
	async def _get_namespace_by_id(self, namespace_id: str) -> Optional[MLNamespace]:
		"""Get namespace by ID"""
		stmt = select(MLNamespace).where(MLNamespace.id == namespace_id)
		result = await self.db.execute(stmt)
		return result.scalar_one_or_none()
	
	async def _get_translation_key(self, namespace_id: str, key: str) -> Optional[MLTranslationKey]:
		"""Get translation key by namespace and key"""
		stmt = select(MLTranslationKey).where(
			and_(
				MLTranslationKey.namespace_id == namespace_id,
				MLTranslationKey.key == key
			)
		)
		result = await self.db.execute(stmt)
		return result.scalar_one_or_none()
	
	async def _update_namespace_stats(self, namespace_id: str):
		"""Update namespace statistics"""
		# Count total keys
		total_result = await self.db.execute(
			select(func.count(MLTranslationKey.id)).where(
				MLTranslationKey.namespace_id == namespace_id
			)
		)
		total_keys = total_result.scalar() or 0
		
		# Count translated keys (keys with at least one approved translation)
		translated_result = await self.db.execute(
			select(func.count(func.distinct(MLTranslation.translation_key_id)))
			.select_from(MLTranslation.join(MLTranslationKey))
			.where(
				and_(
					MLTranslationKey.namespace_id == namespace_id,
					MLTranslation.status.in_([MLTranslationStatus.APPROVED, MLTranslationStatus.PUBLISHED])
				)
			)
		)
		translated_keys = translated_result.scalar() or 0
		
		# Update namespace
		await self.db.execute(
			update(MLNamespace)
			.where(MLNamespace.id == namespace_id)
			.values(
				total_keys=total_keys,
				translated_keys=translated_keys
			)
		)
		
		await self.db.commit()

# =====================
# User Preference Service
# =====================

class UserPreferenceService:
	"""Service for managing user localization preferences"""
	
	def __init__(self, db_session: AsyncSession, redis_client: aioredis.Redis):
		self.db = db_session
		self.redis = redis_client
		self.cache_ttl = 3600  # 1 hour cache TTL
	
	async def get_user_preferences(self, user_id: str) -> Optional[dict]:
		"""Get user localization preferences"""
		
		cache_key = f"user_prefs:{user_id}"
		cached = await self.redis.get(cache_key)
		
		if cached:
			return json.loads(cached.decode('utf-8'))
		
		# Query from database
		stmt = select(MLUserPreference).where(MLUserPreference.user_id == user_id)
		result = await self.db.execute(stmt)
		prefs = result.scalar_one_or_none()
		
		if not prefs:
			return None
		
		# Convert to response format
		response_data = {
			"user_id": prefs.user_id,
			"primary_language_id": prefs.primary_language_id,
			"secondary_language_ids": prefs.secondary_language_ids,
			"preferred_locale_id": prefs.preferred_locale_id,
			"timezone": prefs.timezone,
			"date_format_preference": prefs.date_format_preference,
			"time_format_preference": prefs.time_format_preference,
			"number_format_preference": prefs.number_format_preference,
			"font_size_adjustment": prefs.font_size_adjustment,
			"high_contrast": prefs.high_contrast,
			"screen_reader_optimized": prefs.screen_reader_optimized,
			"auto_translate_enabled": prefs.auto_translate_enabled,
			"machine_translation_threshold": prefs.machine_translation_threshold
		}
		
		# Cache the results
		await self.redis.setex(cache_key, self.cache_ttl, json.dumps(response_data, default=str))
		
		return response_data
	
	async def set_user_preferences(self, user_id: str, preferences: dict) -> bool:
		"""Set user localization preferences"""
		
		# Get existing preferences or create new
		stmt = select(MLUserPreference).where(MLUserPreference.user_id == user_id)
		result = await self.db.execute(stmt)
		prefs = result.scalar_one_or_none()
		
		if prefs:
			# Update existing
			for key, value in preferences.items():
				if hasattr(prefs, key):
					setattr(prefs, key, value)
		else:
			# Create new preferences
			prefs = MLUserPreference(user_id=user_id, **preferences)
			self.db.add(prefs)
		
		await self.db.commit()
		
		# Clear cache
		cache_key = f"user_prefs:{user_id}"
		await self.redis.delete(cache_key)
		
		return True
	
	async def get_user_language(self, user_id: str) -> Optional[str]:
		"""Get user's primary language code"""
		prefs = await self.get_user_preferences(user_id)
		if prefs and prefs.get("primary_language_id"):
			# Get language code from language ID
			stmt = select(MLLanguage.code).where(MLLanguage.id == prefs["primary_language_id"])
			result = await self.db.execute(stmt)
			return result.scalar_one_or_none()
		return None
	
	async def get_user_locale(self, user_id: str) -> Optional[str]:
		"""Get user's preferred locale code"""
		prefs = await self.get_user_preferences(user_id)
		if prefs and prefs.get("preferred_locale_id"):
			# Get locale code from locale ID
			stmt = select(MLLocale.locale_code).where(MLLocale.id == prefs["preferred_locale_id"])
			result = await self.db.execute(stmt)
			return result.scalar_one_or_none()
		return None