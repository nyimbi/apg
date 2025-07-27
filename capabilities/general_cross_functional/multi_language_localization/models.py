"""
APG Multi-language Localization - Data Models and Database Schema

This module defines the comprehensive data models for internationalization and localization
services, supporting multiple languages, regions, and cultural preferences.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, validator, root_validator
from pydantic.types import Json
from sqlalchemy import (
	Column, String, Text, Integer, Boolean, DateTime, Float, 
	ForeignKey, Index, UniqueConstraint, CheckConstraint,
	JSON, DECIMAL, Table
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from uuid_extensions import uuid7str

Base = declarative_base()

# =====================
# Enums and Constants
# =====================

class MLTextDirection(str, Enum):
	"""Text direction for different writing systems"""
	LTR = "ltr"  # Left-to-right (English, Spanish, French, etc.)
	RTL = "rtl"  # Right-to-left (Arabic, Hebrew, etc.)
	TTB = "ttb"  # Top-to-bottom (Traditional Chinese, Japanese, etc.)

class MLLanguageStatus(str, Enum):
	"""Language availability status"""
	ACTIVE = "active"           # Fully supported and available
	BETA = "beta"              # In testing phase
	DEPRECATED = "deprecated"   # Being phased out
	MAINTENANCE = "maintenance" # Temporarily unavailable

class MLTranslationStatus(str, Enum):
	"""Translation completion and quality status"""
	DRAFT = "draft"             # Initial translation, not reviewed
	PENDING_REVIEW = "pending_review"  # Awaiting human review
	APPROVED = "approved"       # Reviewed and approved
	REJECTED = "rejected"       # Review failed, needs revision
	PUBLISHED = "published"     # Live and available to users
	ARCHIVED = "archived"       # No longer in active use

class MLTranslationType(str, Enum):
	"""Method used for translation"""
	HUMAN = "human"            # Professional human translation
	MACHINE = "machine"        # Automated machine translation
	HYBRID = "hybrid"          # Machine translation with human review
	CROWD = "crowd"            # Community-based translation

class MLContentType(str, Enum):
	"""Type of content being translated"""
	UI_TEXT = "ui_text"        # User interface strings
	CONTENT = "content"        # Article/document content
	METADATA = "metadata"      # Titles, descriptions, labels
	ERROR_MESSAGE = "error_message"  # Error and system messages
	EMAIL_TEMPLATE = "email_template"  # Email content
	NOTIFICATION = "notification"  # Push/SMS notifications

class MLPluralRule(str, Enum):
	"""Plural rule categories for different languages"""
	ZERO = "zero"       # Used for zero in some languages
	ONE = "one"         # Singular form
	TWO = "two"         # Dual form (some languages)
	FEW = "few"         # Small numbers (2-4 in some languages)
	MANY = "many"       # Large numbers or default plural
	OTHER = "other"     # Fallback/default form

# =====================
# SQLAlchemy Models
# =====================

class MLLanguage(Base):
	"""Core language configuration and metadata"""
	__tablename__ = 'ml_languages'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	code = Column(String(10), nullable=False, unique=True, index=True)  # ISO 639-1/639-3
	name = Column(String(100), nullable=False)  # English name
	native_name = Column(String(100), nullable=False)  # Native script name
	script = Column(String(10), nullable=False)  # ISO 15924 script code
	direction = Column(String(10), nullable=False, default=MLTextDirection.LTR)
	status = Column(String(20), nullable=False, default=MLLanguageStatus.ACTIVE)
	is_default = Column(Boolean, nullable=False, default=False)
	fallback_language_id = Column(String(36), ForeignKey('ml_languages.id'), nullable=True)
	
	# Metadata
	priority = Column(Integer, nullable=False, default=100)  # Display priority
	completion_percentage = Column(Float, nullable=False, default=0.0)
	total_translators = Column(Integer, nullable=False, default=0)
	quality_score = Column(Float, nullable=True)  # Average quality score
	
	# Timestamps
	created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
	updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
	last_activity_at = Column(DateTime(timezone=True), nullable=True)
	
	# Relationships
	fallback_language = relationship("MLLanguage", remote_side=[id])
	locales = relationship("MLLocale", back_populates="language", cascade="all, delete-orphan")
	translations = relationship("MLTranslation", back_populates="language", cascade="all, delete-orphan")
	
	# Constraints and Indexes
	__table_args__ = (
		Index('idx_ml_languages_code_status', 'code', 'status'),
		Index('idx_ml_languages_priority', 'priority'),
		CheckConstraint('completion_percentage >= 0 AND completion_percentage <= 100'),
		CheckConstraint('priority >= 0'),
		CheckConstraint('total_translators >= 0'),
		CheckConstraint('quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 10)'),
	)

class MLLocale(Base):
	"""Locale configuration combining language and region"""
	__tablename__ = 'ml_locales'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	language_id = Column(String(36), ForeignKey('ml_languages.id'), nullable=False)
	region_code = Column(String(10), nullable=False)  # ISO 3166-1 alpha-2
	locale_code = Column(String(20), nullable=False, unique=True, index=True)  # e.g., 'en-US'
	currency_code = Column(String(10), nullable=False)  # ISO 4217
	
	# Formatting configuration
	date_format = Column(String(50), nullable=False, default='MM/dd/yyyy')
	time_format = Column(String(50), nullable=False, default='HH:mm:ss')
	datetime_format = Column(String(100), nullable=False, default='MM/dd/yyyy HH:mm:ss')
	number_format = Column(JSONB, nullable=False)  # Decimal separator, grouping, etc.
	
	# Cultural settings
	first_day_of_week = Column(Integer, nullable=False, default=0)  # 0=Sunday, 1=Monday
	measurement_system = Column(String(20), nullable=False, default='metric')  # metric/imperial
	paper_size = Column(String(10), nullable=False, default='A4')  # A4/Letter/Legal
	
	# Status and metadata
	is_active = Column(Boolean, nullable=False, default=True)
	usage_count = Column(Integer, nullable=False, default=0)
	
	# Timestamps
	created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
	updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
	
	# Relationships
	language = relationship("MLLanguage", back_populates="locales")
	
	# Constraints and Indexes
	__table_args__ = (
		Index('idx_ml_locales_language_region', 'language_id', 'region_code'),
		Index('idx_ml_locales_locale_code', 'locale_code'),
		Index('idx_ml_locales_active', 'is_active'),
		UniqueConstraint('language_id', 'region_code', name='uq_ml_locales_lang_region'),
		CheckConstraint('first_day_of_week >= 0 AND first_day_of_week <= 6'),
		CheckConstraint('usage_count >= 0'),
	)

class MLNamespace(Base):
	"""Namespace for organizing translation keys by capability/module"""
	__tablename__ = 'ml_namespaces'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	name = Column(String(100), nullable=False, unique=True, index=True)
	description = Column(Text, nullable=True)
	capability_id = Column(String(100), nullable=True)  # APG capability identifier
	
	# Configuration
	default_translation_type = Column(String(20), nullable=False, default=MLTranslationType.MACHINE)
	require_review = Column(Boolean, nullable=False, default=True)
	auto_publish = Column(Boolean, nullable=False, default=False)
	
	# Statistics
	total_keys = Column(Integer, nullable=False, default=0)
	translated_keys = Column(Integer, nullable=False, default=0)
	
	# Timestamps
	created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
	updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
	
	# Relationships
	translation_keys = relationship("MLTranslationKey", back_populates="namespace", cascade="all, delete-orphan")
	
	# Constraints and Indexes
	__table_args__ = (
		Index('idx_ml_namespaces_capability', 'capability_id'),
		Index('idx_ml_namespaces_name', 'name'),
		CheckConstraint('total_keys >= 0'),
		CheckConstraint('translated_keys >= 0'),
		CheckConstraint('translated_keys <= total_keys'),
	)

class MLTranslationKey(Base):
	"""Translation key and metadata for translatable content"""
	__tablename__ = 'ml_translation_keys'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	namespace_id = Column(String(36), ForeignKey('ml_namespaces.id'), nullable=False)
	key = Column(String(255), nullable=False, index=True)  # Hierarchical key like 'user.profile.edit_button'
	
	# Content metadata
	source_text = Column(Text, nullable=False)  # Original text in default language
	context = Column(Text, nullable=True)  # Additional context for translators
	description = Column(Text, nullable=True)  # Detailed description
	content_type = Column(String(30), nullable=False, default=MLContentType.UI_TEXT)
	
	# Constraints and formatting
	max_length = Column(Integer, nullable=True)  # Character limit for translations
	is_html = Column(Boolean, nullable=False, default=False)
	is_plural = Column(Boolean, nullable=False, default=False)
	variables = Column(JSONB, nullable=True)  # Variable placeholders like {user_name}
	
	# Quality and workflow
	translation_priority = Column(Integer, nullable=False, default=50)  # 1-100 priority
	requires_context = Column(Boolean, nullable=False, default=False)
	requires_review = Column(Boolean, nullable=False, default=True)
	is_deprecated = Column(Boolean, nullable=False, default=False)
	
	# Statistics
	translation_count = Column(Integer, nullable=False, default=0)
	usage_count = Column(Integer, nullable=False, default=0)
	last_used_at = Column(DateTime(timezone=True), nullable=True)
	
	# Timestamps
	created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
	updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
	
	# Relationships
	namespace = relationship("MLNamespace", back_populates="translation_keys")
	translations = relationship("MLTranslation", back_populates="translation_key", cascade="all, delete-orphan")
	
	# Constraints and Indexes
	__table_args__ = (
		Index('idx_ml_translation_keys_namespace_key', 'namespace_id', 'key'),
		Index('idx_ml_translation_keys_content_type', 'content_type'),
		Index('idx_ml_translation_keys_priority', 'translation_priority'),
		Index('idx_ml_translation_keys_deprecated', 'is_deprecated'),
		UniqueConstraint('namespace_id', 'key', name='uq_ml_translation_keys_namespace_key'),
		CheckConstraint('translation_priority >= 1 AND translation_priority <= 100'),
		CheckConstraint('max_length IS NULL OR max_length > 0'),
		CheckConstraint('translation_count >= 0'),
		CheckConstraint('usage_count >= 0'),
	)

class MLTranslation(Base):
	"""Translation content for specific language and key"""
	__tablename__ = 'ml_translations'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	translation_key_id = Column(String(36), ForeignKey('ml_translation_keys.id'), nullable=False)
	language_id = Column(String(36), ForeignKey('ml_languages.id'), nullable=False)
	
	# Translation content
	content = Column(Text, nullable=False)
	plural_forms = Column(JSONB, nullable=True)  # Plural variations for languages that need them
	
	# Quality and workflow
	status = Column(String(20), nullable=False, default=MLTranslationStatus.DRAFT)
	translation_type = Column(String(20), nullable=False, default=MLTranslationType.MACHINE)
	quality_score = Column(Float, nullable=True)  # 0-10 quality rating
	confidence_score = Column(Float, nullable=True)  # MT confidence
	
	# Attribution
	translator_id = Column(String(36), nullable=True)  # User who created translation
	reviewer_id = Column(String(36), nullable=True)   # User who reviewed translation
	editor_id = Column(String(36), nullable=True)     # Last user to edit
	
	# Metadata
	word_count = Column(Integer, nullable=False, default=0)
	character_count = Column(Integer, nullable=False, default=0)
	translation_time = Column(Integer, nullable=True)  # Time taken in seconds
	review_notes = Column(Text, nullable=True)
	
	# Timestamps
	created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
	updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
	translated_at = Column(DateTime(timezone=True), nullable=True)
	reviewed_at = Column(DateTime(timezone=True), nullable=True)
	published_at = Column(DateTime(timezone=True), nullable=True)
	
	# Relationships
	translation_key = relationship("MLTranslationKey", back_populates="translations")
	language = relationship("MLLanguage", back_populates="translations")
	
	# Constraints and Indexes
	__table_args__ = (
		Index('idx_ml_translations_key_lang', 'translation_key_id', 'language_id'),
		Index('idx_ml_translations_status', 'status'),
		Index('idx_ml_translations_type', 'translation_type'),
		Index('idx_ml_translations_quality', 'quality_score'),
		Index('idx_ml_translations_translator', 'translator_id'),
		Index('idx_ml_translations_published', 'published_at'),
		UniqueConstraint('translation_key_id', 'language_id', name='uq_ml_translations_key_lang'),
		CheckConstraint('quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 10)'),
		CheckConstraint('confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)'),
		CheckConstraint('word_count >= 0'),
		CheckConstraint('character_count >= 0'),
		CheckConstraint('translation_time IS NULL OR translation_time >= 0'),
	)

class MLTranslationProject(Base):
	"""Translation project for organizing work"""
	__tablename__ = 'ml_translation_projects'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Project scope
	source_language_id = Column(String(36), ForeignKey('ml_languages.id'), nullable=False)
	target_language_ids = Column(JSONB, nullable=False)  # List of target language IDs
	namespace_ids = Column(JSONB, nullable=True)  # Specific namespaces (null = all)
	
	# Project settings
	translation_type = Column(String(20), nullable=False, default=MLTranslationType.HYBRID)
	quality_threshold = Column(Float, nullable=False, default=7.0)
	deadline = Column(DateTime(timezone=True), nullable=True)
	budget = Column(DECIMAL(10, 2), nullable=True)
	
	# Status tracking
	status = Column(String(20), nullable=False, default='active')  # active, completed, cancelled
	progress_percentage = Column(Float, nullable=False, default=0.0)
	total_words = Column(Integer, nullable=False, default=0)
	completed_words = Column(Integer, nullable=False, default=0)
	
	# Assignment
	project_manager_id = Column(String(36), nullable=True)
	translator_ids = Column(JSONB, nullable=True)  # Assigned translators
	reviewer_ids = Column(JSONB, nullable=True)    # Assigned reviewers
	
	# Timestamps
	created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
	updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
	started_at = Column(DateTime(timezone=True), nullable=True)
	completed_at = Column(DateTime(timezone=True), nullable=True)
	
	# Relationships
	source_language = relationship("MLLanguage", foreign_keys=[source_language_id])
	
	# Constraints and Indexes
	__table_args__ = (
		Index('idx_ml_projects_status', 'status'),
		Index('idx_ml_projects_deadline', 'deadline'),
		Index('idx_ml_projects_manager', 'project_manager_id'),
		Index('idx_ml_projects_progress', 'progress_percentage'),
		CheckConstraint('progress_percentage >= 0 AND progress_percentage <= 100'),
		CheckConstraint('quality_threshold >= 0 AND quality_threshold <= 10'),
		CheckConstraint('total_words >= 0'),
		CheckConstraint('completed_words >= 0'),
		CheckConstraint('completed_words <= total_words'),
		CheckConstraint('budget IS NULL OR budget >= 0'),
	)

class MLTranslationMemory(Base):
	"""Translation memory for reusing previous translations"""
	__tablename__ = 'ml_translation_memory'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	source_language_id = Column(String(36), ForeignKey('ml_languages.id'), nullable=False)
	target_language_id = Column(String(36), ForeignKey('ml_languages.id'), nullable=False)
	
	# Content
	source_text = Column(Text, nullable=False)
	target_text = Column(Text, nullable=False)
	source_hash = Column(String(64), nullable=False, index=True)  # For fast matching
	
	# Metadata
	domain = Column(String(100), nullable=True)  # Technical domain/industry
	context = Column(Text, nullable=True)
	quality_score = Column(Float, nullable=False, default=5.0)
	usage_count = Column(Integer, nullable=False, default=0)
	
	# Attribution
	created_by = Column(String(36), nullable=True)
	approved_by = Column(String(36), nullable=True)
	
	# Timestamps
	created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
	updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
	last_used_at = Column(DateTime(timezone=True), nullable=True)
	
	# Relationships
	source_language = relationship("MLLanguage", foreign_keys=[source_language_id])
	target_language = relationship("MLLanguage", foreign_keys=[target_language_id])
	
	# Constraints and Indexes
	__table_args__ = (
		Index('idx_ml_tm_source_target', 'source_language_id', 'target_language_id'),
		Index('idx_ml_tm_hash', 'source_hash'),
		Index('idx_ml_tm_quality', 'quality_score'),
		Index('idx_ml_tm_domain', 'domain'),
		Index('idx_ml_tm_usage', 'usage_count'),
		CheckConstraint('quality_score >= 0 AND quality_score <= 10'),
		CheckConstraint('usage_count >= 0'),
		CheckConstraint('source_language_id != target_language_id'),
	)

class MLUserPreference(Base):
	"""User language and localization preferences"""
	__tablename__ = 'ml_user_preferences'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	user_id = Column(String(36), nullable=False, unique=True, index=True)
	
	# Language preferences
	primary_language_id = Column(String(36), ForeignKey('ml_languages.id'), nullable=False)
	secondary_language_ids = Column(JSONB, nullable=True)  # Fallback languages
	preferred_locale_id = Column(String(36), ForeignKey('ml_locales.id'), nullable=False)
	
	# UI preferences
	timezone = Column(String(50), nullable=False, default='UTC')
	date_format_preference = Column(String(50), nullable=True)  # Override locale default
	time_format_preference = Column(String(50), nullable=True)  # Override locale default
	number_format_preference = Column(JSONB, nullable=True)     # Override locale default
	
	# Accessibility
	font_size_adjustment = Column(Float, nullable=False, default=1.0)  # Multiplier
	high_contrast = Column(Boolean, nullable=False, default=False)
	screen_reader_optimized = Column(Boolean, nullable=False, default=False)
	
	# Auto-translation settings
	auto_translate_enabled = Column(Boolean, nullable=False, default=True)
	machine_translation_threshold = Column(Float, nullable=False, default=7.0)
	
	# Timestamps
	created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
	updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
	last_accessed_at = Column(DateTime(timezone=True), nullable=True)
	
	# Relationships
	primary_language = relationship("MLLanguage", foreign_keys=[primary_language_id])
	preferred_locale = relationship("MLLocale", foreign_keys=[preferred_locale_id])
	
	# Constraints and Indexes
	__table_args__ = (
		Index('idx_ml_user_prefs_user', 'user_id'),
		Index('idx_ml_user_prefs_language', 'primary_language_id'),
		Index('idx_ml_user_prefs_locale', 'preferred_locale_id'),
		CheckConstraint('font_size_adjustment > 0 AND font_size_adjustment <= 3'),
		CheckConstraint('machine_translation_threshold >= 0 AND machine_translation_threshold <= 10'),
	)

# =====================
# Pydantic API Models
# =====================

class MLLanguageCreate(BaseModel):
	"""Pydantic model for creating a new language"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	code: str = Field(..., min_length=2, max_length=10, description="ISO 639 language code")
	name: str = Field(..., min_length=1, max_length=100, description="Language name in English")
	native_name: str = Field(..., min_length=1, max_length=100, description="Language name in native script")
	script: str = Field(..., min_length=4, max_length=10, description="ISO 15924 script code")
	direction: MLTextDirection = Field(default=MLTextDirection.LTR, description="Text direction")
	status: MLLanguageStatus = Field(default=MLLanguageStatus.ACTIVE, description="Language status")
	fallback_language_id: Optional[str] = Field(None, description="Fallback language ID")
	priority: int = Field(default=100, ge=0, description="Display priority")

class MLLanguageResponse(BaseModel):
	"""Pydantic model for language API responses"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	code: str
	name: str
	native_name: str
	script: str
	direction: MLTextDirection
	status: MLLanguageStatus
	is_default: bool
	fallback_language_id: Optional[str]
	priority: int
	completion_percentage: float
	total_translators: int
	quality_score: Optional[float]
	created_at: datetime
	updated_at: datetime

class MLNumberFormat(BaseModel):
	"""Number formatting configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	decimal_separator: str = Field(default=".", description="Decimal point character")
	group_separator: str = Field(default=",", description="Thousands separator")
	group_size: int = Field(default=3, ge=1, description="Grouping size")
	negative_sign: str = Field(default="-", description="Negative number sign")
	negative_format: str = Field(default="-#", description="Negative number format pattern")
	percent_symbol: str = Field(default="%", description="Percentage symbol")
	per_mille_symbol: str = Field(default="‰", description="Per mille symbol")

class MLLocaleCreate(BaseModel):
	"""Pydantic model for creating a new locale"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	language_id: str = Field(..., description="Language ID")
	region_code: str = Field(..., min_length=2, max_length=10, description="ISO 3166 region code")
	currency_code: str = Field(..., min_length=3, max_length=10, description="ISO 4217 currency code")
	date_format: str = Field(default="MM/dd/yyyy", description="Date format pattern")
	time_format: str = Field(default="HH:mm:ss", description="Time format pattern")
	number_format: MLNumberFormat = Field(default_factory=MLNumberFormat, description="Number formatting")
	first_day_of_week: int = Field(default=0, ge=0, le=6, description="First day of week (0=Sunday)")
	measurement_system: str = Field(default="metric", description="Measurement system")
	paper_size: str = Field(default="A4", description="Default paper size")

class MLTranslationKeyCreate(BaseModel):
	"""Pydantic model for creating translation keys"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	namespace_id: str = Field(..., description="Namespace ID")
	key: str = Field(..., min_length=1, max_length=255, description="Translation key")
	source_text: str = Field(..., min_length=1, description="Source text to translate")
	context: Optional[str] = Field(None, description="Context for translators")
	description: Optional[str] = Field(None, description="Detailed description")
	content_type: MLContentType = Field(default=MLContentType.UI_TEXT, description="Content type")
	max_length: Optional[int] = Field(None, gt=0, description="Maximum translation length")
	is_html: bool = Field(default=False, description="Contains HTML markup")
	is_plural: bool = Field(default=False, description="Requires plural forms")
	variables: Optional[Dict[str, str]] = Field(None, description="Variable placeholders")
	translation_priority: int = Field(default=50, ge=1, le=100, description="Translation priority")

class MLTranslationCreate(BaseModel):
	"""Pydantic model for creating translations"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	translation_key_id: str = Field(..., description="Translation key ID")
	language_id: str = Field(..., description="Target language ID")
	content: str = Field(..., min_length=1, description="Translated content")
	plural_forms: Optional[Dict[MLPluralRule, str]] = Field(None, description="Plural form variations")
	translation_type: MLTranslationType = Field(default=MLTranslationType.MACHINE, description="Translation method")
	quality_score: Optional[float] = Field(None, ge=0, le=10, description="Quality rating")
	translator_id: Optional[str] = Field(None, description="Translator user ID")

class MLTranslationResponse(BaseModel):
	"""Pydantic model for translation API responses"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	translation_key_id: str
	language_id: str
	content: str
	plural_forms: Optional[Dict[str, str]]
	status: MLTranslationStatus
	translation_type: MLTranslationType
	quality_score: Optional[float]
	confidence_score: Optional[float]
	word_count: int
	character_count: int
	created_at: datetime
	updated_at: datetime
	published_at: Optional[datetime]

class MLBulkTranslationRequest(BaseModel):
	"""Pydantic model for bulk translation requests"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	source_language_id: str = Field(..., description="Source language ID")
	target_language_ids: List[str] = Field(..., min_length=1, description="Target language IDs")
	namespace_ids: Optional[List[str]] = Field(None, description="Specific namespaces")
	translation_type: MLTranslationType = Field(default=MLTranslationType.MACHINE, description="Translation method")
	quality_threshold: float = Field(default=7.0, ge=0, le=10, description="Minimum quality score")
	auto_publish: bool = Field(default=False, description="Auto-publish approved translations")

class MLTranslationStats(BaseModel):
	"""Translation statistics and metrics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	total_keys: int = Field(..., ge=0, description="Total translation keys")
	translated_keys: int = Field(..., ge=0, description="Translated keys")
	completion_percentage: float = Field(..., ge=0, le=100, description="Completion percentage")
	languages_supported: int = Field(..., ge=0, description="Number of supported languages")
	total_translations: int = Field(..., ge=0, description="Total number of translations")
	quality_average: Optional[float] = Field(None, ge=0, le=10, description="Average quality score")
	pending_review: int = Field(..., ge=0, description="Translations pending review")
	last_updated: Optional[datetime] = Field(None, description="Last translation update")

# =====================
# Helper Functions
# =====================

async def get_translation_coverage(namespace_id: str, language_id: str) -> float:
	"""Calculate translation coverage percentage for a namespace and language"""
	# Implementation would query database to calculate coverage
	pass

async def get_language_pair_quality(source_lang_id: str, target_lang_id: str) -> Optional[float]:
	"""Get average quality score for a language pair"""
	# Implementation would query translation quality metrics
	pass

def validate_locale_code(language_code: str, region_code: str) -> str:
	"""Generate and validate locale code from language and region"""
	return f"{language_code}-{region_code.upper()}"

def extract_variables_from_text(text: str) -> List[str]:
	"""Extract variable placeholders from text (e.g., {user_name})"""
	import re
	return re.findall(r'\{([^}]+)\}', text)

# Model validation and constraints would be added here
# Additional utility functions for localization operations