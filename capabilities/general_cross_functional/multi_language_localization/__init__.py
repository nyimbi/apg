"""
APG Multi-language Localization Capability

Comprehensive internationalization (i18n) and localization (l10n) services for the APG platform,
enabling applications and capabilities to support multiple languages, regions, and cultural preferences.

This capability provides:
- Translation management and workflows
- Multi-language content support
- Regional formatting and cultural adaptation
- AI-powered translation services
- Quality assurance and review processes
- Developer-friendly localization APIs

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

from .models import (
	# Enums
	MLTextDirection,
	MLLanguageStatus,
	MLTranslationStatus,
	MLTranslationType,
	MLContentType,
	MLPluralRule,
	
	# SQLAlchemy Models
	MLLanguage,
	MLLocale,
	MLNamespace,
	MLTranslationKey,
	MLTranslation,
	MLTranslationProject,
	MLTranslationMemory,
	MLUserPreference,
	
	# Pydantic Models
	MLLanguageCreate,
	MLLanguageResponse,
	MLNumberFormat,
	MLLocaleCreate,
	MLTranslationKeyCreate,
	MLTranslationCreate,
	MLTranslationResponse,
	MLBulkTranslationRequest,
	MLTranslationStats,
)

__version__ = "1.0.0"
__capability_id__ = "multi_language_localization"
__capability_name__ = "Multi-language Localization"
__capability_description__ = "Comprehensive internationalization and localization services"

# Capability metadata for APG platform integration
CAPABILITY_METADATA = {
	"id": __capability_id__,
	"name": __capability_name__,
	"version": __version__,
	"description": __capability_description__,
	"category": "general_cross_functional",
	"dependencies": [
		"api_service_mesh",
		"event_streaming_bus",
		"integration_api_management"
	],
	"provides": [
		"translation_services",
		"localization_apis",
		"cultural_formatting",
		"language_management"
	],
	"endpoints": {
		"base_url": "/api/v1/localization",
		"health": "/health",
		"metrics": "/metrics",
		"docs": "/docs"
	},
	"database_prefix": "ml_",
	"supported_languages": [
		"en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
		"ar", "he", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da",
		"no", "fi", "cs", "hu", "ro", "bg", "hr", "sk", "sl", "et",
		"lv", "lt", "mt", "ga", "cy", "eu", "ca", "gl", "oc", "br"
	]
}

# Export all public components
__all__ = [
	# Version and metadata
	"__version__",
	"__capability_id__",
	"CAPABILITY_METADATA",
	
	# Enums
	"MLTextDirection",
	"MLLanguageStatus", 
	"MLTranslationStatus",
	"MLTranslationType",
	"MLContentType",
	"MLPluralRule",
	
	# SQLAlchemy Models
	"MLLanguage",
	"MLLocale",
	"MLNamespace", 
	"MLTranslationKey",
	"MLTranslation",
	"MLTranslationProject",
	"MLTranslationMemory",
	"MLUserPreference",
	
	# Pydantic Models
	"MLLanguageCreate",
	"MLLanguageResponse",
	"MLNumberFormat",
	"MLLocaleCreate",
	"MLTranslationKeyCreate",
	"MLTranslationCreate", 
	"MLTranslationResponse",
	"MLBulkTranslationRequest",
	"MLTranslationStats",
]